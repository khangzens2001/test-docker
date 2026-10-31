"""CPU-side activities. Heavy imports (pipeline/bundle) are deferred to call time."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import shutil
import tempfile
import time
from typing import Any
from urllib.parse import urlparse

import requests
from temporalio import activity

from temporal_app.config import callback_settings
from temporal_app.errors import app_error
from temporal_app.models import ResolvedInput, RoomScanInput

WORK_ROOT = os.getenv("TEMPORAL_WORK_ROOT", os.path.join(tempfile.gettempdir(), "lidar_temporal"))
ALLOWED_SCHEMES = {"http", "https"}
MAX_ZIP_BYTES = int(os.getenv("MAX_ZIP_BYTES", str(300 * 1024 * 1024)))


def _safe(part: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in part)[:64] or "x"


def workdir_for(workflow_id: str) -> str:
    return os.path.join(WORK_ROOT, _safe(workflow_id))


def _stage_metric(stage: str, seconds: float) -> None:
    try:
        meter = activity.metric_meter()
        meter.create_histogram_float("lidar_stage_duration_seconds", "Stage wall time", "s").with_additional_attributes(
            {"stage": stage}
        ).record(seconds)
    except Exception:
        pass


@activity.defn
async def resolve_input(inp: RoomScanInput) -> ResolvedInput:
    """Validate request, derive deterministic job id (idempotent across retries)."""
    t0 = time.time()
    if not inp.task_id:
        raise app_error("task_id is required", "invalid_input")
    parsed = urlparse(inp.zip_url or "")
    is_local = bool(inp.zip_url) and os.path.isfile(inp.zip_url)  # dev only
    if not is_local and (parsed.scheme not in ALLOWED_SCHEMES or not parsed.netloc):
        raise app_error(f"zip_url must be http(s): {inp.zip_url!r}", "invalid_input")
    # Cheap pre-check of size without downloading (best effort; many hosts reject HEAD).
    if parsed.scheme in ALLOWED_SCHEMES and "drive.google.com" not in parsed.netloc:
        try:
            r = await asyncio.to_thread(requests.head, inp.zip_url, allow_redirects=True, timeout=15)
            size = int(r.headers.get("Content-Length", "0") or 0)
            if size > MAX_ZIP_BYTES:
                raise app_error(f"ZIP too large: {size} bytes", "ZipUnsafe")
        except requests.RequestException:
            activity.logger.info("HEAD precheck skipped for %s", inp.zip_url)
    job_id = _safe(f"t-{inp.task_id}")[:40]
    out = ResolvedInput(
        task_id=inp.task_id,
        job_id=job_id,
        zip_url=inp.zip_url,
        batch_id=inp.batch_id,
        params=inp.params,
        gpu_params=inp.gpu_params,
        enable_parallel=inp.enable_parallel,
    )
    _stage_metric("resolve_input", time.time() - t0)
    return out


_pool = None


def _render_pool():
    global _pool
    if _pool is None:
        import concurrent.futures
        import multiprocessing

        _pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=int(os.getenv("RENDER_PROCESSES", "2")), mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _render_materials_sync(workdir: str, gpu_result: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    from app.services.artifact_upload import upload_session_artifacts
    from app.services.estimation import DEFAULT_RECALCULATE_PARAMS
    from app.services.bundle import export_session_bundle
    from app.services.r2_storage import download_to_file, upload_to_r2

    ff = gpu_result.get("floorplan_files") or {}
    fp = (ff.get("floorplan_json") or {}).get("presigned_url")
    if not fp:
        raise app_error("GPU result has no floorplan_json checkpoint", "invalid_input", retryable=False)
    session_dir = os.path.join(workdir, "session")
    shutil.rmtree(session_dir, ignore_errors=True)
    os.makedirs(session_dir, exist_ok=True)
    fp_path = download_to_file(fp, os.path.join(session_dir, "floorplan.json"))
    with open(fp_path, "r", encoding="utf-8") as f:
        floorplan = json.load(f)

    active = dict(DEFAULT_RECALCULATE_PARAMS)
    active.update(params or {})
    summary = export_session_bundle(
        floorplan, session_dir, active,
        render_cad=True, render_materials="all", allow_partial=True,
        session_id=gpu_result.get("job_id", ""),
    )
    prefix = gpu_result["session_prefix"]
    _, files = upload_session_artifacts(session_dir, prefix, upload_to_r2)
    metrics = None
    mp = os.path.join(session_dir, "metrics.json")
    if os.path.isfile(mp):
        with open(mp, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    warnings = []
    if isinstance(summary, dict):
        raw = summary.get("warning") or summary.get("warnings") or []
        for w in ([raw] if isinstance(raw, str) else raw):
            warnings.append({"stage": "materials", "message": str(w)})
    return {"floorplan_files": files, "room_metrics": metrics, "warnings": warnings}


@activity.defn
async def render_materials(workflow_id: str, gpu_result: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Stage 10 on a CPU worker: CAD sheets, wall elevations, tatami/neda/plywood/tiling/CF."""
    t0 = time.time()
    workdir = workdir_for(workflow_id)
    os.makedirs(workdir, exist_ok=True)
    activity.heartbeat({"stage": "materials", "pct": 0})
    try:
        # Separate *process*: matplotlib/CAD rendering must not hold the GIL of the process that
        # polls RunPod and sends heartbeats (that starvation caused duplicate GPU submissions).
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(_render_pool(), _render_materials_sync, workdir, gpu_result, params)
        while True:
            done, _ = await asyncio.wait({fut}, timeout=20)
            if done:
                res = fut.result()
                break
            activity.heartbeat({"stage": "materials", "pct": 50})
    except Exception as exc:
        from app.core.exceptions import classify_pipeline_error

        if hasattr(exc, "non_retryable"):  # already an ApplicationError
            raise
        et, retryable = classify_pipeline_error(exc)
        raise app_error(str(exc), et, retryable) from exc
    _stage_metric("materials", time.time() - t0)
    return res


STAGE_ORDER = [
    ("ingestion_sec", "ingestion"), ("vio_sec", "vio"), ("keyframe_selection_sec", "keyframe_selection"),
    ("em_refinement_sec", "em_refinement"), ("depth_completion_sec", "depth_completion"),
    ("tsdf_integration_sec", "tsdf_integration"), ("vggt_sec", "vggt_prior"),
    ("plane_segmentation_sec", "plane_segmentation"), ("export_artifacts_sec", "export_artifacts"),
    ("materials_sec", "materials"),
]


@activity.defn
async def record_stage_timings(workflow_id: str, timings: dict[str, Any]) -> list[dict[str, Any]]:
    """Make per-stage wall time visible as a single Event History entry and as Prometheus metrics."""
    total = float(timings.get("total_sec") or sum(float(timings.get(k) or 0) for k, _ in STAGE_ORDER)) or 1.0
    rows = []
    for key, name in STAGE_ORDER:
        if key in timings:
            sec = float(timings[key])
            rows.append({"stage": name, "seconds": round(sec, 2), "pct_of_total": round(100 * sec / total, 1)})
            _stage_metric(name, sec)
    rows.sort(key=lambda r: -r["seconds"])
    activity.logger.info("stage timings for %s: %s", workflow_id, rows)
    return rows


@activity.defn
async def build_result(gpu_result: dict[str, Any], materials: dict[str, Any] | None) -> dict[str, Any]:
    """Merge GPU + CPU outputs into the contract of new_update.md section 5."""
    materials = materials or {}
    files = dict(gpu_result.get("floorplan_files") or {})
    files.update(materials.get("floorplan_files") or {})
    warnings = list(gpu_result.get("warnings") or []) + list(materials.get("warnings") or [])
    result = {
        "status": "success",
        "job_id": gpu_result.get("job_id"),
        "mode": gpu_result.get("mode"),
        "download_url": gpu_result.get("download_url"),
        "clean_ply": gpu_result.get("clean_ply"),
        "room_metrics": materials.get("room_metrics") or gpu_result.get("room_metrics"),
        "floorplan_files": files,
        "timings": gpu_result.get("timings") or {},
        "warnings": warnings,
    }
    if gpu_result.get("alignment") is not None:
        result["alignment"] = gpu_result["alignment"]
    return result


@activity.defn
async def notify_backend(workflow_id: str, payload: dict[str, Any]) -> bool:
    """Signed webhook to the backend (optional; BE may instead read the workflow result)."""
    cfg = callback_settings()
    if not cfg.url:
        return False
    body = json.dumps({"workflow_id": workflow_id, **payload}, default=str).encode()
    headers = {"Content-Type": "application/json", "X-Workflow-Id": workflow_id}
    if cfg.secret:
        headers["X-Signature"] = hmac.new(cfg.secret.encode(), body, hashlib.sha256).hexdigest()
    try:
        r = await asyncio.to_thread(requests.post, cfg.url, data=body, headers=headers, timeout=cfg.timeout_s)
    except requests.RequestException as exc:
        raise app_error(f"callback failed: {exc}", "transient_io") from exc
    if r.status_code >= 500 or r.status_code == 429:
        raise app_error(f"callback HTTP {r.status_code}", "transient_io")
    if r.status_code >= 400:
        raise app_error(f"callback rejected HTTP {r.status_code}", "invalid_input", retryable=False)
    return True


@activity.defn
async def cleanup(workflow_id: str) -> None:
    shutil.rmtree(workdir_for(workflow_id), ignore_errors=True)


CPU_ACTIVITIES = [resolve_input, render_materials, record_stage_timings, build_result, notify_backend, cleanup]
