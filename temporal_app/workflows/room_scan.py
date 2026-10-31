"""RoomScanWorkflow: deterministic orchestration only; all I/O lives in activities."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy, SearchAttributeKey
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError

with workflow.unsafe.imports_passed_through():
    from temporal_app.models import (
        NON_RETRYABLE_ERROR_TYPES,
        RESUBMIT_ERROR_TYPES,
        ResolvedInput,
        RoomScanInput,
        ScanStatus,
        StageEvent,
    )

SA_TASK = SearchAttributeKey.for_keyword("TaskId")
SA_BATCH = SearchAttributeKey.for_keyword("BatchId")
SA_USER = SearchAttributeKey.for_keyword("UserId")
SA_STAGE = SearchAttributeKey.for_keyword("CurrentStage")
SA_MODE = SearchAttributeKey.for_keyword("ScanMode")

CPU_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2), backoff_coefficient=2.0, maximum_interval=timedelta(seconds=60),
    maximum_attempts=5, non_retryable_error_types=NON_RETRYABLE_ERROR_TYPES,
)
GPU_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10), backoff_coefficient=2.0, maximum_interval=timedelta(minutes=2),
    maximum_attempts=2, non_retryable_error_types=NON_RETRYABLE_ERROR_TYPES,
)


# Polling is idempotent (same job id); only transient infrastructure errors are retried here.
POLL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5), maximum_interval=timedelta(seconds=60), maximum_attempts=5,
    non_retryable_error_types=NON_RETRYABLE_ERROR_TYPES + RESUBMIT_ERROR_TYPES,
)


class _Cancelled(Exception):
    pass


@workflow.defn(name="RoomScanWorkflow")
class RoomScanWorkflow:
    def __init__(self) -> None:
        self._status = ScanStatus(workflow_id="")
        self._cancelled = False
        self._handle: Optional[workflow.ActivityHandle] = None
        self._use_sa = True
        self._params_override: dict[str, Any] = {}

    # ---- signals / queries -------------------------------------------------
    @workflow.signal
    def cancel_scan(self) -> None:
        self._cancelled = True
        if self._handle is not None:
            self._handle.cancel()

    @workflow.signal
    def override_params(self, params: dict[str, Any]) -> None:
        """Materials params may be changed until the materials stage starts."""
        self._params_override.update(params or {})

    @workflow.signal
    def report_stage(self, ev: StageEvent) -> None:
        self._set_stage(ev.stage, ev.pct, source=ev.source)

    @workflow.query
    def get_status(self) -> ScanStatus:
        return self._status

    # ---- helpers -----------------------------------------------------------
    def _set_stage(self, stage: str, pct: int, source: str = "workflow") -> None:
        s = self._status
        s.current_stage, s.pct = stage, max(0, pct)
        s.stage_log.append({"stage": stage, "pct": pct, "source": source,
                            "at": workflow.now().isoformat()})
        s.stage_log = s.stage_log[-60:]
        if self._use_sa:
            workflow.upsert_search_attributes([SA_STAGE.value_set(stage)])

    async def _act(self, name: str, args: list, result_type: type, *, queue: Optional[str] = None,
                   start_to_close: timedelta, retry: RetryPolicy,
                   heartbeat: Optional[timedelta] = None) -> Any:
        if self._cancelled:
            raise _Cancelled()
        self._handle = workflow.start_activity(
            name, args=args, result_type=result_type, task_queue=queue,
            start_to_close_timeout=start_to_close, heartbeat_timeout=heartbeat, retry_policy=retry,
            cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
        )
        try:
            return await self._handle
        except ActivityError as exc:
            if self._cancelled and isinstance(exc.cause, CancelledError):
                raise _Cancelled() from exc
            raise
        finally:
            self._handle = None

    async def _gpu_phase(self, inp: RoomScanInput, resolved: ResolvedInput) -> dict[str, Any]:
        """submit (once, id stored in history) -> poll (safe to retry). One resubmission on job-level failure."""
        max_submissions = 2
        for n in range(max_submissions):
            job_id: str = await self._act(
                "runpod_submit", [resolved, None if n == 0 else False], str, queue=inp.gpu_task_queue,
                start_to_close=timedelta(minutes=2), retry=CPU_RETRY)
            self._status.runpod_job_id = job_id
            try:
                return await self._act(
                    "runpod_poll", [job_id], dict, queue=inp.gpu_task_queue,
                    start_to_close=timedelta(seconds=inp.gpu_timeout_s), heartbeat=timedelta(seconds=90),
                    retry=POLL_RETRY)
            except ActivityError as exc:
                cause = exc.cause
                resubmit = (isinstance(cause, ApplicationError) and cause.type in RESUBMIT_ERROR_TYPES
                            and n + 1 < max_submissions and not self._cancelled)
                if not resubmit:
                    raise
                self._status.warnings.append({"stage": "gpu", "message": f"resubmitting after {cause.type}"})
        raise RuntimeError("unreachable")

    # ---- main --------------------------------------------------------------
    @workflow.run
    async def run(self, inp: RoomScanInput) -> dict[str, Any]:
        wf_id = workflow.info().workflow_id
        self._status = ScanStatus(workflow_id=wf_id, state="running")
        self._use_sa = inp.use_search_attributes
        if self._use_sa:
            workflow.upsert_search_attributes([
                SA_TASK.value_set(inp.task_id), SA_BATCH.value_set(inp.batch_id),
                SA_USER.value_set(inp.user_id or "-"),
            ])
        try:
            self._set_stage("resolve_input", 1)
            resolved: ResolvedInput = await self._act(
                "resolve_input", [inp], ResolvedInput,
                start_to_close=timedelta(minutes=2), retry=CPU_RETRY)

            self._set_stage("gpu_submitted", 2)
            gpu: dict[str, Any] = await self._gpu_phase(inp, resolved)
            self._status.timings.update(gpu.get("timings") or {})
            self._status.warnings.extend(gpu.get("warnings") or [])
            if self._use_sa:
                workflow.upsert_search_attributes([SA_MODE.value_set(str(gpu.get("mode")))])

            materials: Optional[dict[str, Any]] = None
            if gpu.get("mode") == "multisensor_lidar" and gpu.get("pipeline_mode") == "reconstruct_only":
                self._set_stage("materials", 95)
                params = {**inp.params, **self._params_override}
                materials = await self._act(
                    "render_materials", [wf_id, gpu, params], dict,
                    start_to_close=timedelta(minutes=10), heartbeat=timedelta(seconds=90),
                    retry=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=5),
                                      non_retryable_error_types=NON_RETRYABLE_ERROR_TYPES))
                self._status.warnings.extend(materials.get("warnings") or [])

            self._set_stage("build_result", 98)
            result: dict[str, Any] = await self._act(
                "build_result", [gpu, materials], dict,
                start_to_close=timedelta(seconds=30), retry=CPU_RETRY)

            if inp.notify:
                self._set_stage("notify_backend", 99)
                try:
                    await self._act("notify_backend", [wf_id, result], bool,
                                    start_to_close=timedelta(seconds=30), retry=CPU_RETRY)
                except ActivityError as exc:  # BE can still read the workflow result
                    self._status.warnings.append({"stage": "notify_backend", "message": str(exc.cause)})

            self._set_stage("done", 100)
            self._status.state = "succeeded"
            result["warnings"] = self._status.warnings
            return result
        except _Cancelled:
            self._status.state = "cancelled"
            return {"status": "cancelled", "job_id": wf_id}
        except asyncio.CancelledError:
            self._status.state = "cancelled"
            raise
        except BaseException as exc:
            self._status.state = "failed"
            self._status.error = str(getattr(exc, "cause", exc))
            raise
        finally:
            # runs for success, failure and external cancellation
            await workflow.execute_activity("cleanup", wf_id, start_to_close_timeout=timedelta(minutes=1),
                                            retry_policy=RetryPolicy(maximum_attempts=3))
