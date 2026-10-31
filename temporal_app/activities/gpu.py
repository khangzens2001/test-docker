"""RunPod bridge: one Temporal activity == one RunPod serverless job (submit, poll, heartbeat, cancel)."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from temporalio import activity
from temporalio.client import Client

from temporal_app.config import runpod_settings
from temporal_app.errors import app_error
from temporal_app.models import ResolvedInput, StageEvent
from temporal_app.runpod_client import TERMINAL_FAIL, TERMINAL_OK, RunPodClient, RunPodError


def _classify_terminal(status: str, error: Optional[str]) -> str:
    """Map a non-successful RunPod terminal state to an error type (decides resubmission)."""
    msg = (error or "").lower()
    if "out of memory" in msg:
        return "oom"
    if status == "TIMED_OUT":
        return "RunPodTimedOut"
    if status == "CANCELLED":
        return "RunPodCancelled"
    if any(k in msg for k in ("worker", "unavailable", "connection reset", "preempt", "evict")):
        return "RunPodWorkerLost"
    return "PipelineFailed"  # deterministic pipeline/model error: do not pay for the GPU twice


class GpuBridge:
    """Holds a Temporal client (to signal stage events back to the workflow) and a RunPod client."""

    def __init__(self, client: Client, runpod: Optional[RunPodClient] = None):
        self._client = client
        self._runpod = runpod

    def _rp(self) -> RunPodClient:
        if self._runpod is None:
            try:
                self._runpod = RunPodClient(runpod_settings())
            except RunPodError as exc:
                raise app_error(str(exc), "RunPodConfigError", retryable=False) from exc
        return self._runpod

    @activity.defn(name="runpod_submit")
    async def runpod_submit(self, resolved: ResolvedInput, enable_parallel: Optional[bool] = None) -> str:
        """Submit exactly one RunPod job; the returned id is persisted in workflow history."""
        rp = self._rp()
        payload = {
            "zip_url": resolved.zip_url,
            "batch_id": resolved.batch_id,
            "job_id": resolved.job_id,
            "mode": "reconstruct_only",
            "enable_parallel": enable_parallel if enable_parallel is not None else resolved.enable_parallel,
            **resolved.gpu_params,
        }
        try:
            job = await asyncio.to_thread(rp.submit, payload)
        except RunPodError as exc:
            raise app_error(str(exc), "RunPodSubmitError", exc.retryable) from exc
        activity.logger.info("Submitted RunPod job %s", job)
        return job

    @activity.defn(name="runpod_poll")
    async def runpod_poll(self, runpod_job_id: str) -> dict[str, Any]:
        """Poll one RunPod job to completion. Safe to retry: it only ever re-polls the same job.

        Terminal job outcomes are raised non-retryable *at activity level*; the workflow decides
        whether to resubmit based on the error type.
        """
        info = activity.info()
        rp = self._rp()
        t0 = time.time()
        reported: set[str] = set()
        last_stage: Optional[str] = None
        activity.heartbeat({"runpod_job_id": runpod_job_id, "stage": "queued", "pct": 0})
        try:
            while True:
                try:
                    st = await asyncio.to_thread(rp.status, runpod_job_id)
                except RunPodError as exc:
                    if exc.retryable:  # transient poll failure: keep polling
                        activity.heartbeat({"runpod_job_id": runpod_job_id, "stage": last_stage or "queued", "pct": -1})
                        await asyncio.sleep(rp.s.poll_interval_s)
                        continue
                    raise app_error(str(exc), "RunPodStatusError", False) from exc

                for ev in st.stage_events or []:
                    name = str(ev.get("stage"))
                    if name not in reported:  # /status may replay the whole event list on every poll
                        reported.add(name)
                        last_stage = name
                        await self._report(info.workflow_id, StageEvent(
                            stage=name, pct=int(ev.get("pct", 0)), stage_index=int(ev.get("stage_index", -1))))
                activity.heartbeat({"runpod_job_id": runpod_job_id, "stage": last_stage or st.status.lower(),
                                    "pct": (st.stage_events[-1].get("pct") if st.stage_events else 0),
                                    "elapsed_s": round(time.time() - t0, 1)})

                if st.status in TERMINAL_OK:
                    out = st.output if isinstance(st.output, dict) else {}
                    if out.get("status") == "error":
                        raise app_error(str(out.get("error")), str(out.get("error_type") or "internal_error"),
                                        False)
                    return out
                if st.status in TERMINAL_FAIL:
                    raise app_error(f"RunPod job {runpod_job_id} ended {st.status}: {st.error}",
                                    _classify_terminal(st.status, st.error), False)
                await asyncio.sleep(rp.s.poll_interval_s)
        except asyncio.CancelledError:
            await asyncio.to_thread(rp.cancel, runpod_job_id)
            raise

    async def _report(self, workflow_id: str, ev: StageEvent) -> None:
        try:
            await self._client.get_workflow_handle(workflow_id).signal("report_stage", ev)
        except Exception as exc:  # observability must never fail the job
            activity.logger.warning("report_stage signal failed: %s", exc)

    @property
    def activities(self) -> list:
        return [self.runpod_submit, self.runpod_poll]
