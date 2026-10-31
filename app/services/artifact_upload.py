"""Shared multi-sensor artifact discovery + R2 upload (used by handler.py and the FastAPI backend)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

CLEAN_PLY_CANDIDATES = (
    "reconstructed_visual.ply",
    "reconstructed_whiteflat.ply",
    "whiteflat.ply",
    "reconstructed.ply",
)

FILE_CANDIDATES: list[tuple[str, list[str]]] = [
    ("floorplan_png", ["FloorPlan_A3.png", "FloorPlan_A4.png", "floorplan.png", "ISO_A3_Floorplan.png", "ISO_A4_Floorplan.png"]),
    ("floorplan_pdf", ["FloorPlan_A3.pdf", "FloorPlan_A4.pdf", "floorplan.pdf", "ISO_A3_Floorplan.pdf", "ISO_A4_Floorplan.pdf"]),
    ("floorplan_a4_png", ["FloorPlan_A4.png", "ISO_A4_Floorplan.png"]),
    ("floorplan_a4_pdf", ["FloorPlan_A4.pdf", "ISO_A4_Floorplan.pdf"]),
    ("floorplan_json", ["floorplan.json"]),
    ("wall_elevation_png", ["Walls.png", "walls.png", "input_Walls.png"]),
    ("debug_topdown_png", ["debug_topdown.png"]),
    ("room_model_glb", ["room_model_texture.glb", "room_model.glb", "reconstructed.glb"]),
    ("reconstructed_poisson_glb", ["reconstructed_poisson.glb"]),
    ("dollhouse_ply", ["reconstructed_dollhouse.ply"]),
    ("whiteflat_ply", ["reconstructed_whiteflat.ply", "whiteflat.ply"]),
    ("metrics_json", ["metrics.json"]),
    ("quality_report_json", ["quality_report.json"]),
    ("cf_png", ["CF.png"]),
    ("cf_pdf", ["CF.pdf"]),
    ("cf_chart_png", ["CF_chart.png"]),
    ("neda_png", ["Neda.png"]),
    ("neda_pdf", ["Neda.pdf"]),
    ("neda_chart_png", ["Neda_chart.png"]),
    ("plywood_png", ["Plywood.png"]),
    ("plywood_pdf", ["Plywood.pdf"]),
    ("plywood_chart_png", ["Plywood_chart.png"]),
    ("tiling_png", ["Tiling.png"]),
    ("tiling_pdf", ["Tiling.pdf"]),
    ("tiling_chart_png", ["Tiling_chart.png"]),
    ("combined_png", ["Combined.png"]),
]


def upload_session_artifacts(
    session_dir: str,
    object_prefix: str,
    upload_fn: Callable[[str, str], dict],
    max_workers: int = 6,
) -> tuple[dict, dict]:
    """Upload the clean PLY and every known artifact found in session_dir.

    Returns (clean_ply_r2, floorplan_files). `upload_fn(local_path, key)` returns the R2 descriptor.
    """
    clean_ply_path = next(
        (p for p in (os.path.join(session_dir, n) for n in CLEAN_PLY_CANDIDATES) if os.path.isfile(p)),
        None,
    )

    upload_tasks: list[tuple[str, str, str]] = []
    if clean_ply_path:
        upload_tasks.append(
            ("__clean_ply__", clean_ply_path, f"ply_clean/{object_prefix}/project_point_cloud_clean.ply")
        )

    for file_key, cands in FILE_CANDIDATES:
        for c in cands:
            p = os.path.join(session_dir, c)
            if os.path.isfile(p):
                target_name = "walls.png" if "wall" in file_key else c
                upload_tasks.append((file_key, p, f"ply_clean/{object_prefix}/{target_name}"))
                break

    clean_ply_r2: dict = {}
    floorplan_files: dict = {}
    if upload_tasks:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(upload_tasks))) as pool:
            futures = [(k, pool.submit(upload_fn, p, r2_k)) for k, p, r2_k in upload_tasks]
            for k, fut in futures:
                if k == "__clean_ply__":
                    clean_ply_r2 = fut.result()
                else:
                    floorplan_files[k] = fut.result()
    return clean_ply_r2, floorplan_files
