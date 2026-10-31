"""Workflow tests with mocked activities (time-skipping Temporal test server)."""
from __future__ import annotations

import asyncio
import uuid

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from temporal_app.models import ResolvedInput, RoomScanInput, StageEvent
from temporal_app.workflows.room_scan import RoomScanWorkflow

CPU_Q, GPU_Q = "t-cpu", "t-gpu"


class Calls:
    def __init__(self):
        self.names: list[str] = []
        self.gpu_mode = "ok"  # ok | bad_input | flaky | slow
        self.gpu_attempts = 0
        self.submits = 0
        self.parallel_args: list = []


def build_activities(calls: Calls):
    @activity.defn(name="resolve_input")
    async def resolve_input(inp: RoomScanInput) -> ResolvedInput:
        calls.names.append("resolve_input")
        return ResolvedInput(task_id=inp.task_id, job_id=f"t-{inp.task_id}", zip_url=inp.zip_url, batch_id=inp.batch_id)

    @activity.defn(name="runpod_submit")
    async def runpod_submit(r: ResolvedInput, enable_parallel: bool | None = None) -> str:
        calls.names.append("runpod_submit")
        calls.submits += 1
        calls.parallel_args.append(enable_parallel)
        return f"job-{calls.submits}"

    @activity.defn(name="runpod_poll")
    async def runpod_poll(job: str) -> dict:
        calls.names.append("runpod_poll")
        calls.gpu_attempts += 1
        if calls.gpu_mode == "bad_input":
            raise ApplicationError("bad session", type="invalid_input", non_retryable=True)
        if calls.gpu_mode == "flaky" and calls.gpu_attempts == 1:
            raise ApplicationError("oom", type="oom", non_retryable=True)
        if calls.gpu_mode == "always_oom":
            raise ApplicationError("oom", type="oom", non_retryable=True)
        if calls.gpu_mode == "slow":
            while True:
                activity.heartbeat()
                await asyncio.sleep(0.2)
        return {"status": "success", "job_id": job, "mode": "multisensor_lidar",
                "pipeline_mode": "reconstruct_only", "session_prefix": "b/j",
                "floorplan_files": {"floorplan_json": {"presigned_url": "x"}}, "timings": {"vio_sec": 1.0},
                "warnings": [{"stage": "portals", "message": "w"}]}

    @activity.defn(name="render_materials")
    async def render_materials(wf_id: str, gpu: dict, params: dict) -> dict:
        calls.names.append("render_materials")
        return {"floorplan_files": {"cf_png": {"presigned_url": "y"}}, "room_metrics": {"area_m2": 15.1}, "warnings": []}

    @activity.defn(name="build_result")
    async def build_result(gpu: dict, mat: dict) -> dict:
        calls.names.append("build_result")
        return {"status": "success", "room_metrics": mat["room_metrics"],
                "floorplan_files": {**gpu["floorplan_files"], **mat["floorplan_files"]}}

    @activity.defn(name="notify_backend")
    async def notify_backend(wf_id: str, payload: dict) -> bool:
        calls.names.append("notify_backend")
        return True

    @activity.defn(name="cleanup")
    async def cleanup(wf_id: str) -> None:
        calls.names.append("cleanup")

    return [resolve_input, render_materials, build_result, notify_backend, cleanup], [runpod_submit, runpod_poll]


async def run_env(calls: Calls, coro_factory):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        cpu_acts, gpu_acts = build_activities(calls)
        async with Worker(env.client, task_queue=CPU_Q, workflows=[RoomScanWorkflow], activities=cpu_acts), \
                Worker(env.client, task_queue=GPU_Q, activities=gpu_acts):
            return await coro_factory(env.client)


def _inp(task="T1"):
    return RoomScanInput(task_id=task, zip_url="https://x/y.zip", gpu_task_queue=GPU_Q, use_search_attributes=False)


async def _wait_gpu_started(calls: Calls):
    for _ in range(200):
        if "runpod_poll" in calls.names:
            return
        await asyncio.sleep(0.05)


async def test_happy_path():
    calls = Calls()

    async def go(client):
        h = await client.start_workflow(RoomScanWorkflow.run, _inp(), id=f"w-{uuid.uuid4()}", task_queue=CPU_Q)
        return await h.result(), await h.query(RoomScanWorkflow.get_status)

    res, st = await run_env(calls, go)
    assert res["status"] == "success"
    assert set(res["floorplan_files"]) == {"floorplan_json", "cf_png"}
    assert calls.names == ["resolve_input", "runpod_submit", "runpod_poll", "render_materials", "build_result",
                           "notify_backend", "cleanup"]
    assert st.state == "succeeded" and st.warnings[0]["stage"] == "portals"


async def test_non_retryable_gpu_failure_runs_cleanup_and_no_retry():
    calls = Calls()
    calls.gpu_mode = "bad_input"

    async def go(client):
        h = await client.start_workflow(RoomScanWorkflow.run, _inp(), id=f"w-{uuid.uuid4()}", task_queue=CPU_Q)
        with pytest.raises(WorkflowFailureError):
            await h.result()
        return await h.query(RoomScanWorkflow.get_status)

    st = await run_env(calls, go)
    assert calls.gpu_attempts == 1 and calls.submits == 1  # bad input: no resubmission
    assert calls.names[-1] == "cleanup" and "render_materials" not in calls.names
    assert st.state == "failed"


async def test_retryable_gpu_failure_retries_then_succeeds():
    calls = Calls()
    calls.gpu_mode = "flaky"

    async def go(client):
        h = await client.start_workflow(RoomScanWorkflow.run, _inp(), id=f"w-{uuid.uuid4()}", task_queue=CPU_Q)
        return await h.result()

    res = await run_env(calls, go)
    assert res["status"] == "success" and calls.gpu_attempts == 2
    # job-level retryable failure -> exactly one resubmission, in sequential (enable_parallel=False) mode
    assert calls.submits == 2 and calls.parallel_args == [None, False]


async def test_resubmission_is_bounded():
    calls = Calls()
    calls.gpu_mode = "always_oom"

    async def go(client):
        h = await client.start_workflow(RoomScanWorkflow.run, _inp(), id=f"w-{uuid.uuid4()}", task_queue=CPU_Q)
        with pytest.raises(WorkflowFailureError):
            await h.result()

    await run_env(calls, go)
    assert calls.submits == 2  # never more than one resubmission (GPU cost guard)


async def test_cancel_signal_stops_gpu_and_cleans_up():
    calls = Calls()
    calls.gpu_mode = "slow"

    async def go(client):
        h = await client.start_workflow(RoomScanWorkflow.run, _inp(), id=f"w-{uuid.uuid4()}", task_queue=CPU_Q)
        await _wait_gpu_started(calls)
        await h.signal(RoomScanWorkflow.cancel_scan)
        return await h.result()

    res = await run_env(calls, go)
    assert res["status"] == "cancelled"
    assert calls.names[-1] == "cleanup" and "render_materials" not in calls.names


async def test_report_stage_signal_updates_status():
    calls = Calls()
    calls.gpu_mode = "slow"

    async def go(client):
        h = await client.start_workflow(RoomScanWorkflow.run, _inp(), id=f"w-{uuid.uuid4()}", task_queue=CPU_Q)
        await _wait_gpu_started(calls)
        await h.signal(RoomScanWorkflow.report_stage, StageEvent(stage="depth_completion", pct=35))
        st = await h.query(RoomScanWorkflow.get_status)
        await h.signal(RoomScanWorkflow.cancel_scan)
        await h.result()
        return st

    st = await run_env(calls, go)
    assert st.current_stage == "depth_completion" and st.pct == 35
