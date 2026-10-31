"""Real Temporal dev server + real worker/activities (incl. real CPU CAD/material rendering) + fake RunPod.

Run:  pytest tests/e2e -s        (downloads the Temporal dev server on first run)
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

pytest.importorskip("matplotlib")

from app.services.floorplan import export_artifacts  # noqa: E402
from temporal_app.activities.cpu import CPU_ACTIVITIES  # noqa: E402
from temporal_app.activities.gpu import GpuBridge  # noqa: E402
from temporal_app.config import RunPodSettings  # noqa: E402
from temporal_app.models import RoomScanInput  # noqa: E402
from temporal_app.runpod_client import RunPodClient  # noqa: E402
from temporal_app.workflows.room_scan import RoomScanWorkflow  # noqa: E402
from tests.e2e.fake_runpod import FakeRunPod  # noqa: E402

CPU_Q, GPU_Q = "e2e-cpu", "e2e-gpu"
SEARCH_ATTRS = ["TaskId", "BatchId", "UserId", "CurrentStage", "ScanMode"]


@pytest.fixture(scope="module")
def floorplan(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    layout = {"vertices": {"v0": [0, 0], "v1": [4.2, 0], "v2": [4.2, 3.6], "v3": [0, 3.6]},
              "walls": [{"wall_index": i, "joints": [f"v{i}", f"v{(i + 1) % 4}"], "height_meters": 2.45}
                        for i in range(4)], "room": {"height_meters": 2.45}, "portals": []}
    export_artifacts(layout, str(d))
    return str(d / "floorplan.json")


@pytest.fixture
async def stack(floorplan, tmp_path):
    os.environ["TEMPORAL_WORK_ROOT"] = str(tmp_path / "work")
    import temporal_app.activities.cpu as cpu
    cpu.WORK_ROOT = str(tmp_path / "work")
    fake = FakeRunPod(floorplan)
    args = []
    for a in SEARCH_ATTRS:
        args += ["--search-attribute", f"{a}=Keyword"]
    env = None
    for attempt in range(3):  # dev-server start is occasionally flaky when launched back-to-back
        try:
            env = await WorkflowEnvironment.start_local(dev_server_extra_args=args)
            break
        except RuntimeError:
            if attempt == 2:
                raise
            await asyncio.sleep(2)
    rp = RunPodClient(RunPodSettings(api_key="k", endpoint_id="e", base_url=fake.url, poll_interval_s=0.2))
    bridge = GpuBridge(env.client, rp)
    async with Worker(env.client, task_queue=CPU_Q, workflows=[RoomScanWorkflow], activities=CPU_ACTIVITIES), \
            Worker(env.client, task_queue=GPU_Q, activities=bridge.activities):
        yield env.client, fake, tmp_path
    await env.shutdown()
    fake.close()


def _inp(task, zip_url="https://example.com/scan.zip"):
    return RoomScanInput(task_id=task, zip_url=zip_url, batch_id="B1", user_id="U1", gpu_task_queue=GPU_Q)


async def test_full_flow_real_cad_render(stack):
    client, fake, tmp = stack
    tid = f"T{uuid.uuid4().hex[:6]}"
    h = await client.start_workflow(RoomScanWorkflow.run, _inp(tid), id=f"scan-{tid}", task_queue=CPU_Q)
    res = await h.result()
    st = await h.query(RoomScanWorkflow.get_status)
    print("\nstages:", [e["stage"] for e in st.stage_log])
    print("files:", sorted(res["floorplan_files"]))
    print("room_metrics:", json.dumps(res["room_metrics"])[:200])
    assert res["status"] == "success"
    assert {"floorplan_json", "floorplan_png", "cf_png", "neda_png", "plywood_png", "metrics_json"} <= set(res["floorplan_files"])
    assert res["room_metrics"] and abs(res["room_metrics"]["area_m2"] - 15.12) < 0.2
    # stage events from the (fake) GPU job were relayed once each, in order
    names = [e["stage"] for e in st.stage_log]
    assert len(names) == len(set(names)), f"duplicate stage reports: {names}"
    for s in ("vio", "depth_completion", "tsdf_integration", "materials", "done"):
        assert s in names
    assert fake.submissions[0]["mode"] == "reconstruct_only" and fake.submissions[0]["job_id"] == f"t-{tid}"
    # cleanup removed local workdir
    assert not os.path.exists(os.path.join(str(tmp / "work"), f"scan-{tid}"))
    # search attributes searchable
    found = [w.id async for w in client.list_workflows(f"TaskId = '{tid}'")]
    assert found == [f"scan-{tid}"]
    # production start policy: a succeeded scan id cannot be started again (no duplicate GPU spend)
    from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy
    from temporalio.exceptions import WorkflowAlreadyStartedError
    with pytest.raises(WorkflowAlreadyStartedError):
        await client.start_workflow(
            RoomScanWorkflow.run, _inp(tid), id=f"scan-{tid}", task_queue=CPU_Q,
            id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING)
    assert len(fake.submissions) == 1


async def test_bad_input_fails_without_retry(stack):
    client, fake, _ = stack
    tid = f"T{uuid.uuid4().hex[:6]}"
    h = await client.start_workflow(RoomScanWorkflow.run, _inp(tid, "https://example.com/bad.zip"),
                                    id=f"scan-{tid}", task_queue=CPU_Q)
    with pytest.raises(WorkflowFailureError) as ei:
        await h.result()
    assert "invalid_input" in repr(ei.value.cause) or "invalid_input" in str(getattr(ei.value.cause, "cause", ""))
    assert len(fake.submissions) == 1  # non-retryable: exactly one RunPod job


async def test_cancel_cancels_runpod_job(stack):
    client, fake, _ = stack
    tid = f"T{uuid.uuid4().hex[:6]}"
    h = await client.start_workflow(RoomScanWorkflow.run, _inp(tid, "https://example.com/hang.zip"),
                                    id=f"scan-{tid}", task_queue=CPU_Q)
    for _ in range(100):
        if fake.submissions:
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(1.0)
    await h.signal(RoomScanWorkflow.cancel_scan)
    res = await h.result()
    assert res["status"] == "cancelled"
    assert len(fake.cancelled) == 1
