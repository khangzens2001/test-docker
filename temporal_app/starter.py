"""CLI: start / inspect / cancel scans.

  python -m temporal_app.starter start --task-id T1 --zip-url https://... [--wait]
  python -m temporal_app.starter status --task-id T1
  python -m temporal_app.starter cancel --task-id T1
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from temporalio.common import SearchAttributePair, TypedSearchAttributes, WorkflowIDConflictPolicy, WorkflowIDReusePolicy

from temporal_app.client import connect
from temporal_app.config import temporal_settings
from temporal_app.models import RoomScanInput
from temporal_app.workflows.room_scan import SA_BATCH, SA_TASK, SA_USER, RoomScanWorkflow


def workflow_id(task_id: str) -> str:
    return f"scan-{task_id}"


async def start(args) -> None:
    cfg = temporal_settings()
    client = await connect(cfg)
    inp = RoomScanInput(task_id=args.task_id, zip_url=args.zip_url, batch_id=args.batch_id, user_id=args.user_id,
                        gpu_task_queue=cfg.gpu_task_queue)
    handle = await client.start_workflow(
        RoomScanWorkflow.run, inp, id=workflow_id(args.task_id), task_queue=cfg.cpu_task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,  # idempotent start
        search_attributes=TypedSearchAttributes([
            SearchAttributePair(SA_TASK, args.task_id), SearchAttributePair(SA_BATCH, args.batch_id),
            SearchAttributePair(SA_USER, args.user_id or "-")]),
    )
    print(json.dumps({"workflow_id": handle.id, "run_id": handle.result_run_id}))
    if args.wait:
        print(json.dumps(await handle.result(), indent=2, default=str))


async def status(args) -> None:
    client = await connect(temporal_settings())
    h = client.get_workflow_handle(workflow_id(args.task_id))
    print(json.dumps(asdict(await h.query(RoomScanWorkflow.get_status)), indent=2, default=str))


async def cancel(args) -> None:
    client = await connect(temporal_settings())
    await client.get_workflow_handle(workflow_id(args.task_id)).signal(RoomScanWorkflow.cancel_scan)
    print("cancel signalled")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--task-id", required=True)
    s.add_argument("--zip-url", required=True)
    s.add_argument("--batch-id", default="batch")
    s.add_argument("--user-id", default="")
    s.add_argument("--wait", action="store_true")
    for n in ("status", "cancel"):
        x = sub.add_parser(n)
        x.add_argument("--task-id", required=True)
    a = p.parse_args()
    asyncio.run({"start": start, "status": status, "cancel": cancel}[a.cmd](a))


if __name__ == "__main__":
    main()
