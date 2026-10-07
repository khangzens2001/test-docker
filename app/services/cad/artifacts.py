"""CAD and session artifact resolution and URL helpers."""

from __future__ import annotations

import os
from typing import Any


def artifact_url(session_id: str, artifact_type: str | Any) -> str:
    """Return canonical API URL for a session artifact."""
    type_str = getattr(artifact_type, "value", str(artifact_type))
    return f"/api/v1/sessions/{session_id}/artifacts/{type_str}"


# Map each artifact type to (disk_filename_template, media_type, download_filename_template).
# Template variable `{prefix}` will be formatted with `session_id[:8]`.
ARTIFACT_MAP: dict[str, tuple[str, str, str]] = {
    # 3D pipeline / geometry
    "point_cloud": ("reconstructed.ply", "model/ply", "{prefix}_reconstructed.ply"),
    "whiteflat_point_cloud": (
        "reconstructed_whiteflat.ply",
        "model/ply",
        "{prefix}_reconstructed_whiteflat.ply",
    ),
    "mesh": ("reconstructed.glb", "model/gltf-binary", "{prefix}_reconstructed.glb"),
    "floorplan_json": ("floorplan.json", "application/json", "{prefix}_floorplan.json"),
    "floorplan_svg": ("floorplan.svg", "image/svg+xml", "{prefix}_floorplan.svg"),
    "floorplan_dxf": ("floorplan.dxf", "application/dxf", "{prefix}_floorplan.dxf"),
    "visual_point_cloud": (
        "reconstructed_visual.ply",
        "model/ply",
        "{prefix}_reconstructed_visual.ply",
    ),
    "dollhouse_point_cloud": (
        "reconstructed_dollhouse.ply",
        "model/ply",
        "{prefix}_reconstructed_dollhouse.ply",
    ),
    "room_model": ("room_model.glb", "model/gltf-binary", "{prefix}_room_model.glb"),
    "room_model_texture": (
        "room_model_texture.glb",
        "model/gltf-binary",
        "{prefix}_room_model_texture.glb",
    ),
    # CAD renderers & QA
    "cad_drawing_a3_png": ("FloorPlan_A3.png", "image/png", "{prefix}_FloorPlan_A3.png"),
    "cad_drawing_a3_pdf": ("FloorPlan_A3.pdf", "application/pdf", "{prefix}_FloorPlan_A3.pdf"),
    "cad_drawing_a4_png": ("FloorPlan_A4.png", "image/png", "{prefix}_FloorPlan_A4.png"),
    "cad_drawing_a4_pdf": ("FloorPlan_A4.pdf", "application/pdf", "{prefix}_FloorPlan_A4.pdf"),
    "cad_wall_elevations": ("{prefix}_Walls.png", "image/png", "{prefix}_Walls.png"),
    "cad_quality_report": (
        "quality_report.json",
        "application/json",
        "{prefix}_quality_report.json",
    ),
    "metrics_mm": ("metrics.json", "application/json", "{prefix}_metrics.json"),
    # Floor materials
    "floor_neda_png": ("Neda.png", "image/png", "{prefix}_Neda.png"),
    "floor_neda_pdf": ("Neda.pdf", "application/pdf", "{prefix}_Neda.pdf"),
    "floor_tiling_png": ("Tiling.png", "image/png", "{prefix}_Tiling.png"),
    "floor_tiling_pdf": ("Tiling.pdf", "application/pdf", "{prefix}_Tiling.pdf"),
    "floor_plywood_png": ("Plywood.png", "image/png", "{prefix}_Plywood.png"),
    "floor_plywood_pdf": ("Plywood.pdf", "application/pdf", "{prefix}_Plywood.pdf"),
    "floor_cf_png": ("CF.png", "image/png", "{prefix}_CF.png"),
    "floor_cf_pdf": ("CF.pdf", "application/pdf", "{prefix}_CF.pdf"),
    "floor_combined_png": ("Combined.png", "image/png", "{prefix}_Combined.png"),
}


def resolve_artifact_file(
    session_dir: str, session_id: str, artifact_type: str | Any
) -> tuple[str, str, str]:
    """Return (abs_path, media_type, download_filename). Raise KeyError if enum unknown."""
    type_str = getattr(artifact_type, "value", str(artifact_type))
    if type_str not in ARTIFACT_MAP:
        raise KeyError(f"Unknown artifact type: {artifact_type}")

    disk_tmpl, media_type, dl_tmpl = ARTIFACT_MAP[type_str]
    prefix = str(session_id)[:8]
    disk_name = disk_tmpl.format(prefix=prefix)
    download_filename = dl_tmpl.format(prefix=prefix)
    abs_path = os.path.join(session_dir, disk_name)
    return abs_path, media_type, download_filename
