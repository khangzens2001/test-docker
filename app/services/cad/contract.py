# Adapted from 3D-Estimate backend/floorplan_generator/core/metrics.py and main.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app.services.floorplan import enrich_layout


CAD_DIR_ORTHO_DEG = 15.0


def shape_name_for_n(n: int) -> str:
    """Return canonical shape name based on wall count n."""
    if n == 4:
        return "RECTANGULAR"
    if n == 6:
        return "L-SHAPED"
    if n == 8:
        return "T-SHAPED/U-SHAPED"
    return f"COMPLEX ({n}-gon)"


def wall_direction_label(dx: float, dy: float) -> str:
    """Classify wall direction as Hor, Ver, or Obl based on heading angle."""
    h = math.degrees(math.atan2(dy, dx)) % 180.0
    if h < 0.0:
        h += 180.0
    d0 = min(h, 180.0 - h)
    d90 = abs(h - 90.0)
    if d0 <= CAD_DIR_ORTHO_DEG:
        return "Hor"
    if d90 <= CAD_DIR_ORTHO_DEG:
        return "Ver"
    return "Obl"


def layout_to_metrics_mm(layout: dict, project_name: str = "") -> dict:
    """Convert layout in metres to canonical millimetre metrics dictionary.

    Recomputes room geometry via floorplan.enrich_layout to ensure stale fields
    are discarded.
    """
    if not isinstance(layout, dict):
        layout = {}

    enriched = enrich_layout(layout)
    vertices = enriched.get("vertices") or {}
    walls = enriched.get("walls") or []
    room = enriched.get("room") or {}

    room_h_m = float(room.get("height_meters", 0.0) or 0.0)
    if room_h_m == 0.0 and isinstance(layout.get("room"), dict):
        room_h_m = float(layout["room"].get("height_meters", 0.0) or 0.0)

    vertices_mm: list[list[float]] = []
    walls_mm: list[dict[str, Any]] = []

    for i, w in enumerate(walls):
        joints = w.get("joints") or []
        start_key = joints[0] if len(joints) > 0 else None
        end_key = joints[1] if len(joints) > 1 else None
        p = vertices.get(start_key) if start_key is not None and start_key in vertices else None
        q = vertices.get(end_key) if end_key is not None and end_key in vertices else None

        if p is not None:
            vertices_mm.append([float(round(p[0] * 1000.0, 1)), float(round(p[1] * 1000.0, 1))])

        dx = (q[0] - p[0]) if (p is not None and q is not None) else 0.0
        dy = (q[1] - p[1]) if (p is not None and q is not None) else 0.0
        direction = wall_direction_label(dx, dy)


        length_m = float(w.get("length_meters", 0.0) or 0.0)
        if length_m == 0.0 and p is not None and q is not None:
            length_m = float(math.hypot(dx, dy))

        length_mm = float(round(length_m * 1000.0, 1))

        wall_h_m = float(w.get("height_meters", 0.0) or 0.0)
        if wall_h_m == 0.0:
            wall_h_m = room_h_m

        area_m2 = float(round(length_m * wall_h_m, 4))

        walls_mm.append({
            "id": f"W{i}",
            "direction": direction,
            "length_mm": length_mm,
            "area_m2": area_m2,
        })

    if not vertices_mm and vertices:
        for v in vertices.values():
            vertices_mm.append([float(round(v[0] * 1000.0, 1)), float(round(v[1] * 1000.0, 1))])

    wall_count = len(walls_mm)
    shape_name = room.get("shape_name")
    if not shape_name:
        if wall_count == 6 and any(w.get("direction") == "Obl" for w in walls_mm):
            shape_name = "COMPLEX (6-gon)"
        else:
            shape_name = shape_name_for_n(wall_count)

    resolved_project_name = project_name or str(layout.get("project_name", "") or "")

    area_m2 = float(room.get("area_m2", 0.0) or 0.0)
    finish_verts = room.get("finishable_vertices")
    finish_area = room.get("finishable_area_m2")
    if isinstance(finish_verts, list) and len(finish_verts) >= 3 and finish_area is not None:
        area_m2 = float(finish_area)
        vertices_mm = [
            [float(round(float(p[0]) * 1000.0, 1)), float(round(float(p[1]) * 1000.0, 1))]
            for p in finish_verts
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]

    return {
        "project_name": resolved_project_name,
        "height_mm": float(round(room_h_m * 1000.0, 1)),
        "area_m2": area_m2,
        "perimeter_m": float(room.get("perimeter_m", 0.0) or 0.0),
        "bbox_width_mm": float(round(float(room.get("width_m", 0.0) or 0.0) * 1000.0, 1)),
        "bbox_depth_mm": float(round(float(room.get("depth_m", 0.0) or 0.0) * 1000.0, 1)),
        "shape_name": shape_name,
        "wall_count": wall_count,
        "vertices_mm": vertices_mm,
        "walls": walls_mm,
    }


def write_metrics_json(
    path: str | Path | dict, metrics_mm: dict | str | Path | None = None
) -> None:
    """Serialize metrics_mm dictionary to JSON file."""
    if isinstance(path, dict) and isinstance(metrics_mm, (str, Path)):
        path, metrics_mm = metrics_mm, path
    if not isinstance(metrics_mm, dict):
        raise TypeError("metrics_mm must be a dictionary")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(metrics_mm, f, indent=2, ensure_ascii=False)

