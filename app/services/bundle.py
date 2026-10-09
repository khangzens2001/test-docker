"""Shared session exporter for CAD artifacts and materials estimation."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

logger = logging.getLogger(__name__)

from app.schemas.session import ArtifactTypeEnum
from app.services.cad.artifacts import artifact_url, resolve_artifact_file
from app.services.cad.contract import layout_to_metrics_mm, write_metrics_json
from app.services.cad.floorplan_render import render_floorplan
from app.services.cad.quality import build_quality_report, write_quality_report_json
from app.services.cad.wall_elevations import render_wall_elevations
from app.services.estimation import estimate_materials_for_session
from app.services.floorplan import export_artifacts
from app.services.materials import (
    render_cf,
    render_combined,
    render_neda,
    render_plywood,
    render_tiling,
)


def get_available_cpu_count() -> int:
    """Return available CPU cores considering cgroups/taskset affinity when available."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def get_render_worker_count(cores: int | None = None) -> int:
    """Determine adaptive concurrency for rendering workers.

    Rules:
    - Environment variable RENDER_WORKERS or MAX_RENDER_WORKERS takes precedence.
    - If available cores <= 2 (constrained environments): 1 worker (sequential fallback).
    - If 3-5 cores: 2 workers (leaving cores free for OS/Celery/FastAPI).
    - If >= 6 cores: min(3, cores - 2) workers (leaving cores free for OS/Celery/FastAPI).
    """
    env_override = os.environ.get("RENDER_WORKERS") or os.environ.get("MAX_RENDER_WORKERS")
    if env_override is not None:
        try:
            return max(1, int(env_override))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid RENDER_WORKERS value '%s', falling back to auto-detection",
                env_override,
            )

    if cores is None:
        cores = get_available_cpu_count()

    if cores < 3:
        return 1
    if cores < 6:
        return 2
    return max(1, min(3, cores - 2))


def _init_matplotlib_agg() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception as exc:
        logger.debug("Failed to set matplotlib Agg backend: %s", exc)


def _safe_run_worker(
    fn,
    *args,
    name: str,
    allow_partial: bool,
    warnings: list[str],
    **kwargs,
) -> Any:
    if not allow_partial:
        return fn(*args, **kwargs)
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning("Renderer %s failed: %s", name, exc)
        warnings.append(f"{name}: {exc}")
        return None


def _should_render(render_materials: set[str] | str, key: str) -> bool:
    if render_materials == "all":
        return True
    if isinstance(render_materials, (set, list, tuple)):
        return key in render_materials
    if isinstance(render_materials, str):
        return key == render_materials
    return False


def _run_batch_cad(
    metrics_mm: dict[str, Any],
    staging_dir: str,
    wall_elev_path: str,
    allow_partial: bool,
) -> list[str]:
    """Batch 1: FloorPlan A3 & A4 (PDF & PNG) + Walls elevation PNG."""
    _init_matplotlib_agg()
    warnings: list[str] = []
    try:
        _safe_run_worker(
            render_floorplan,
            metrics_mm,
            os.path.join(staging_dir, "FloorPlan_A3.png"),
            sheet_size="A3",
            name="render_floorplan_a3_png",
            allow_partial=allow_partial,
            warnings=warnings,
        )
        _safe_run_worker(
            render_floorplan,
            metrics_mm,
            os.path.join(staging_dir, "FloorPlan_A3.pdf"),
            sheet_size="A3",
            name="render_floorplan_a3_pdf",
            allow_partial=allow_partial,
            warnings=warnings,
        )
        _safe_run_worker(
            render_floorplan,
            metrics_mm,
            os.path.join(staging_dir, "FloorPlan_A4.png"),
            sheet_size="A4",
            name="render_floorplan_a4_png",
            allow_partial=allow_partial,
            warnings=warnings,
        )
        _safe_run_worker(
            render_floorplan,
            metrics_mm,
            os.path.join(staging_dir, "FloorPlan_A4.pdf"),
            sheet_size="A4",
            name="render_floorplan_a4_pdf",
            allow_partial=allow_partial,
            warnings=warnings,
        )

        def _render_wall_elev():
            render_wall_elevations(metrics_mm, wall_elev_path)
            legacy_walls_path = os.path.join(staging_dir, "Walls.png")
            if os.path.abspath(wall_elev_path) != os.path.abspath(legacy_walls_path):
                shutil.copyfile(wall_elev_path, legacy_walls_path)

        _safe_run_worker(
            _render_wall_elev,
            name="render_wall_elevations",
            allow_partial=allow_partial,
            warnings=warnings,
        )
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
    return warnings


def _run_batch_neda_plywood(
    metrics_mm: dict[str, Any],
    materials: dict[str, Any],
    staging_dir: str,
    allow_partial: bool,
    render_materials: set[str] | str,
) -> list[str]:
    """Batch 2: Neda & Plywood (PDF, PNG, Chart)."""
    _init_matplotlib_agg()
    warnings: list[str] = []
    try:
        if _should_render(render_materials, "neda") and materials.get("neda"):
            _safe_run_worker(
                render_neda,
                metrics_mm,
                materials["neda"],
                os.path.join(staging_dir, "Neda.png"),
                name="render_neda_png",
                allow_partial=allow_partial,
                warnings=warnings,
            )
            _safe_run_worker(
                render_neda,
                metrics_mm,
                materials["neda"],
                os.path.join(staging_dir, "Neda.pdf"),
                name="render_neda_pdf",
                allow_partial=allow_partial,
                warnings=warnings,
            )
            _safe_run_worker(
                render_neda,
                metrics_mm,
                materials["neda"],
                os.path.join(staging_dir, "Neda_chart.png"),
                chart_only=True,
                name="render_neda_chart",
                allow_partial=allow_partial,
                warnings=warnings,
            )

        if _should_render(render_materials, "plywood") and materials.get("plywood"):
            _safe_run_worker(
                render_plywood,
                metrics_mm,
                materials["plywood"],
                os.path.join(staging_dir, "Plywood.png"),
                name="render_plywood_png",
                allow_partial=allow_partial,
                warnings=warnings,
            )
            _safe_run_worker(
                render_plywood,
                metrics_mm,
                materials["plywood"],
                os.path.join(staging_dir, "Plywood.pdf"),
                name="render_plywood_pdf",
                allow_partial=allow_partial,
                warnings=warnings,
            )
            _safe_run_worker(
                render_plywood,
                metrics_mm,
                materials["plywood"],
                os.path.join(staging_dir, "Plywood_chart.png"),
                chart_only=True,
                name="render_plywood_chart",
                allow_partial=allow_partial,
                warnings=warnings,
            )
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
    return warnings


def _run_batch_tiling_cf(
    metrics_mm: dict[str, Any],
    materials: dict[str, Any],
    staging_dir: str,
    allow_partial: bool,
    render_materials: set[str] | str,
) -> list[str]:
    """Batch 3: Tiling & CF (PDF, PNG, Chart)."""
    _init_matplotlib_agg()
    warnings: list[str] = []
    try:
        if (
            _should_render(render_materials, "tiling")
            and materials.get("tiling")
            and not materials["tiling"].get("skipped")
        ):
            _safe_run_worker(
                render_tiling,
                metrics_mm,
                materials["tiling"],
                os.path.join(staging_dir, "Tiling.png"),
                name="render_tiling_png",
                allow_partial=allow_partial,
                warnings=warnings,
            )
            _safe_run_worker(
                render_tiling,
                metrics_mm,
                materials["tiling"],
                os.path.join(staging_dir, "Tiling.pdf"),
                name="render_tiling_pdf",
                allow_partial=allow_partial,
                warnings=warnings,
            )
            _safe_run_worker(
                render_tiling,
                metrics_mm,
                materials["tiling"],
                os.path.join(staging_dir, "Tiling_chart.png"),
                chart_only=True,
                name="render_tiling_chart",
                allow_partial=allow_partial,
                warnings=warnings,
            )

        if _should_render(render_materials, "cf") and materials.get("cf"):
            _safe_run_worker(
                render_cf,
                metrics_mm,
                materials["cf"],
                os.path.join(staging_dir, "CF.png"),
                name="render_cf_png",
                allow_partial=allow_partial,
                warnings=warnings,
            )
            _safe_run_worker(
                render_cf,
                metrics_mm,
                materials["cf"],
                os.path.join(staging_dir, "CF.pdf"),
                name="render_cf_pdf",
                allow_partial=allow_partial,
                warnings=warnings,
            )
            _safe_run_worker(
                render_cf,
                metrics_mm,
                materials["cf"],
                os.path.join(staging_dir, "CF_chart.png"),
                chart_only=True,
                name="render_cf_chart",
                allow_partial=allow_partial,
                warnings=warnings,
            )
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
    return warnings


def export_session_bundle(
    layout: dict,
    session_dir: str,
    params: dict,
    *,
    render_cad: bool,
    render_materials: set[str] | str,
    allow_partial: bool,
    session_id: str = "",
    estimate_keys: set[str] | str = "all",
    prior_results: dict | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Export session CAD artifacts and material estimations to disk and return summary dict.

    Parameters:
    - layout: Floorplan layout dictionary.
    - session_dir: Output filesystem directory for artifacts.
    - params: Estimation parameter overrides.
    - render_cad: If True, generate CAD sheets (A3/A4 PNG & PDF), wall elevations, and quality report.
    - render_materials: Collection of material keys to render or "all".
    - allow_partial: If False, any estimation error raises immediately without writing files.
    - session_id: Optional session identifier.
    - estimate_keys: Specific material keys to compute, or 'all'.
    - prior_results: Optional prior results to reuse when selectively calculating.
    """
    _init_matplotlib_agg()
    os.makedirs(session_dir, exist_ok=True)
    import uuid

    staging_dir = os.path.join(
        session_dir, f".bundle_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(staging_dir, exist_ok=True)

    try:
        project_name = session_id[:8].upper() if session_id else "ROOM SCAN"

        metrics_mm = layout_to_metrics_mm(layout, project_name=project_name)

        # 3. Material Estimation (JSON)
        # If allow_partial=False, any estimator exception propagates out immediately
        materials = estimate_materials_for_session(
            layout, params, allow_partial=allow_partial, keys=estimate_keys
        )
        if prior_results:
            for k in ("wallpaper", "tatami", "neda", "tiling", "plywood", "cf"):
                if materials.get(k) is None and prior_results.get(k) is not None:
                    materials[k] = prior_results[k]

        # 4. CAD & Quality Report Writing (if render_cad)
        render_warnings: list[str] = []

        def safe_run(fn, *args, name: str, **kwargs):
            if not allow_partial:
                return fn(*args, **kwargs)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.warning("Renderer %s failed: %s", name, exc)
                render_warnings.append(f"{name}: {exc}")
                return None

        if render_cad:
            safe_run(export_artifacts, layout, staging_dir, name="export_artifacts")
            safe_run(
                write_metrics_json,
                os.path.join(staging_dir, "metrics.json"),
                metrics_mm,
                name="write_metrics_json",
            )

            cloud_path = os.path.join(session_dir, "reconstructed.ply")
            cloud_size = (
                os.path.getsize(cloud_path) if os.path.exists(cloud_path) else None
            )
            try:
                quality_report = build_quality_report(
                    layout,
                    session_dir=session_dir,
                    ply_size_bytes=cloud_size,
                )
                write_quality_report_json(
                    os.path.join(staging_dir, "quality_report.json"), quality_report
                )
            except Exception as exc:
                if not allow_partial:
                    raise
                logger.warning("Quality report generation failed: %s", exc)
                render_warnings.append(f"quality_report: {exc}")
                quality_report = None

            wall_elev_path, _, _ = resolve_artifact_file(
                staging_dir, session_id, ArtifactTypeEnum.cad_wall_elevations
            )
        else:
            qr_path = os.path.join(session_dir, "quality_report.json")
            if os.path.exists(qr_path):
                with open(qr_path, "r", encoding="utf-8") as f:
                    quality_report = json.load(f)
            else:
                quality_report = None
            wall_elev_path = ""

        # 5. Floor Material Drawings & CAD Sheet Rendering
        tiling = materials.get("tiling")
        if isinstance(tiling, dict) and tiling.get("skipped"):
            materials["tiling"] = None
            warn = tiling.get("reason") or "skipped"
            render_warnings.append(f"tiling: {warn}")

        cad_tasks = []
        if render_cad:
            cad_tasks.append(
                (_run_batch_cad, (metrics_mm, staging_dir, wall_elev_path, allow_partial))
            )

        neda_ply_needed = (
            _should_render(render_materials, "neda") and bool(materials.get("neda"))
        ) or (
            _should_render(render_materials, "plywood") and bool(materials.get("plywood"))
        )
        neda_ply_tasks = []
        if neda_ply_needed:
            neda_ply_tasks.append(
                (
                    _run_batch_neda_plywood,
                    (metrics_mm, materials, staging_dir, allow_partial, render_materials),
                )
            )

        tiling_cf_needed = (
            _should_render(render_materials, "tiling")
            and bool(materials.get("tiling"))
            and not materials["tiling"].get("skipped")
        ) or (
            _should_render(render_materials, "cf") and bool(materials.get("cf"))
        )
        tiling_cf_tasks = []
        if tiling_cf_needed:
            tiling_cf_tasks.append(
                (
                    _run_batch_tiling_cf,
                    (metrics_mm, materials, staging_dir, allow_partial, render_materials),
                )
            )

        all_batch_tasks = cad_tasks + neda_ply_tasks + tiling_cf_tasks

        def _execute_sequentially(task_list: list) -> list[str]:
            seq_warnings: list[str] = []
            for fn, args in task_list:
                w = fn(*args)
                if w:
                    seq_warnings.extend(w)
            return seq_warnings

        worker_count = workers if workers is not None else get_render_worker_count()

        if worker_count <= 1 or len(all_batch_tasks) <= 1:
            seq_warns = _execute_sequentially(all_batch_tasks)
            render_warnings.extend(seq_warns)
        else:
            effective_workers = min(worker_count, len(all_batch_tasks))
            parallel_succeeded = False
            batch_warnings: list[str] = []
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=effective_workers
                ) as executor:
                    futures = [executor.submit(fn, *args) for fn, args in all_batch_tasks]
                    try:
                        for f in futures:
                            w = f.result()
                            if w:
                                batch_warnings.extend(w)
                    except Exception:
                        for f in futures:
                            f.cancel()
                        raise
                render_warnings.extend(batch_warnings)
                parallel_succeeded = True
            except Exception as exc:
                logger.warning(
                    "Parallel sheet rendering encountered an error (%s); falling back to sequential execution",
                    exc,
                )
                parallel_succeeded = False

            if not parallel_succeeded:
                seq_warns = _execute_sequentially(all_batch_tasks)
                render_warnings.extend(seq_warns)

        combined_payload = {}
        if _should_render(render_materials, "combined") or render_materials == "all":
            def _find_chart(name: str) -> str | None:
                for d in (staging_dir, session_dir):
                    p = os.path.join(d, name)
                    if os.path.exists(p):
                        return p
                return None

            neda_chart = _find_chart("Neda_chart.png")
            ply_chart = _find_chart("Plywood_chart.png")
            cf_chart = _find_chart("CF_chart.png")
            combined_path = os.path.join(staging_dir, "Combined.png")
            if neda_chart and ply_chart and cf_chart:
                out = safe_run(
                    render_combined,
                    neda_chart,
                    ply_chart,
                    cf_chart,
                    combined_path,
                    name="render_combined",
                )
                if not out:
                    combined_payload = {"skipped": True, "reason": "missing_chart"}
            else:
                combined_payload = {"skipped": True, "reason": "missing_chart"}
        elif prior_results and "combined" in prior_results:
            combined_payload = prior_results.get("combined")
        results_combined = combined_payload

        # Promote staging files into session_dir atomically
        for item in os.listdir(staging_dir):
            s = os.path.join(staging_dir, item)
            d = os.path.join(session_dir, item)
            if os.path.isfile(s):
                os.replace(s, d)
            elif os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
                shutil.move(s, d)

        # 6. Artifact Map & Results Assembly
        artifacts: dict[str, str] = {}
        for art_type in ArtifactTypeEnum:
            try:
                art_path, _, _ = resolve_artifact_file(session_dir, session_id, art_type)
                if os.path.isfile(art_path):
                    artifacts[art_type.value] = artifact_url(session_id, art_type)
            except KeyError:
                continue

        results: dict[str, Any] = {
            "wallpaper": materials.get("wallpaper"),
            "tatami": materials.get("tatami"),
            "neda": materials.get("neda"),
            "tiling": materials.get("tiling"),
            "plywood": materials.get("plywood"),
            "cf": materials.get("cf"),
            "combined": results_combined,
            "metrics_mm": metrics_mm,
            "quality_report": quality_report,
            "artifacts": artifacts,
        }
        if "warning" in materials:
            results["warning"] = materials["warning"]

        if render_warnings:
            warn_str = "; ".join(render_warnings)
            if "warning" in results and results["warning"]:
                results["warning"] = f"{results['warning']}; {warn_str}"
            else:
                results["warning"] = warn_str

        return results
    finally:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
