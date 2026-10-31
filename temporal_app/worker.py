"""Worker entrypoint.  `python -m temporal_app.worker`  (WORKER_ROLES=cpu,gpu)."""
from __future__ import annotations

import asyncio
import logging
import signal
from datetime import timedelta

from temporalio.worker import Worker

from temporal_app.activities.cpu import CPU_ACTIVITIES
from temporal_app.activities.gpu import GpuBridge
from temporal_app.client import connect
from temporal_app.config import temporal_settings
from temporal_app.observability import build_interceptors, build_runtime, setup_logging
from temporal_app.workflows.room_scan import RoomScanWorkflow

log = logging.getLogger("temporal_app.worker")


async def main() -> None:
    setup_logging()
    cfg = temporal_settings()
    roles = {r.strip() for r in cfg.worker_roles.split(",") if r.strip()}
    runtime = build_runtime(cfg)
    client = await connect(cfg, runtime)
    interceptors = build_interceptors(cfg)

    workers: list[Worker] = []
    if "cpu" in roles:
        workers.append(Worker(
            client, task_queue=cfg.cpu_task_queue, workflows=[RoomScanWorkflow], activities=CPU_ACTIVITIES,
            max_concurrent_activities=cfg.cpu_max_concurrent_activities, interceptors=interceptors,
            graceful_shutdown_timeout=timedelta(seconds=60)))
    if "gpu" in roles:
        bridge = GpuBridge(client)
        workers.append(Worker(
            client, task_queue=cfg.gpu_task_queue, activities=bridge.activities,
            max_concurrent_activities=cfg.gpu_max_concurrent_activities,
            max_activities_per_second=cfg.gpu_activities_per_second,
            interceptors=interceptors,
            graceful_shutdown_timeout=timedelta(seconds=30)))
    if not workers:
        raise SystemExit(f"No valid WORKER_ROLES in {cfg.worker_roles!r}")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    log.info("starting workers roles=%s address=%s ns=%s", sorted(roles), cfg.address, cfg.namespace)
    tasks = [asyncio.create_task(w.run()) for w in workers]
    await stop.wait()
    log.info("shutdown requested; draining")
    for w in workers:
        await w.shutdown()
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
