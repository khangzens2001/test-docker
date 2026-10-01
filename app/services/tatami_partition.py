from __future__ import annotations

import math
from typing import Any

from app.services.tatami import (
    TATAMI_STANDARDS,
    _vertex_aabb,
    _xy,
    calculate_tatami_layout,
    derive_edge_tapers,
)

__all__ = ["partition_l_tatami"]


def _extract_polygon_vertices(layout: dict) -> list[tuple[float, float]] | None:
    vertices = layout.get("vertices")
    walls = layout.get("walls")
    if not isinstance(vertices, dict) or not isinstance(walls, list) or len(walls) < 3:
        return None

    edges: list[tuple[str, str]] = []
    for w in walls:
        if not isinstance(w, dict):
            return None
        joints = w.get("joints")
        if not isinstance(joints, (list, tuple)) or len(joints) < 2:
            return None
        edges.append((str(joints[0]), str(joints[1])))

    # Chain edges into an ordered cycle of vertex names
    used = [False] * len(edges)
    first_u, first_v = edges[0]
    used[0] = True
    ordered_names = [first_u, first_v]
    curr = first_v

    for _ in range(1, len(edges)):
        found = False
        for idx, (u, v) in enumerate(edges):
            if not used[idx]:
                if u == curr:
                    used[idx] = True
                    curr = v
                    ordered_names.append(v)
                    found = True
                    break
                elif v == curr:
                    used[idx] = True
                    curr = u
                    ordered_names.append(u)
                    found = True
                    break
        if not found:
            return None

    if ordered_names[-1] != ordered_names[0]:
        return None
    ordered_names.pop()

    if len(ordered_names) != len(walls):
        return None

    pts: list[tuple[float, float]] = []
    for name in ordered_names:
        pt = _xy(vertices, name)
        if pt is None:
            return None
        pts.append(pt)
    return pts


def _is_orthogonal_and_ccw(
    pts: list[tuple[float, float]],
) -> tuple[bool, list[tuple[float, float]], float]:
    n = len(pts)
    # Check orthogonality
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        is_h = dy <= 0.001 and dx > 0.001
        is_v = dx <= 0.001 and dy > 0.001
        if not (is_h or is_v):
            return False, pts, 0.0

    # Signed area
    signed_area = 0.5 * sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )
    if abs(signed_area) < 1e-6:
        return False, pts, 0.0

    if signed_area < 0:
        pts = list(reversed(pts))
        signed_area = -signed_area

    return True, pts, signed_area


def _find_reflex_vertex(pts: list[tuple[float, float]]) -> int | None:
    n = len(pts)
    reflex_indices = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        pi = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = (pi[0] - p_prev[0], pi[1] - p_prev[1])
        v2 = (p_next[0] - pi[0], p_next[1] - pi[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < -1e-6:
            reflex_indices.append(i)
    if len(reflex_indices) != 1:
        return None
    return reflex_indices[0]


def _generate_candidate_cuts(
    pts: list[tuple[float, float]],
    r_idx: int,
    total_area: float,
) -> list[dict[str, Any]]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pr = pts[r_idx]
    xr, yr = pr

    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]

    missing_corners = []
    for cx, cy in corners:
        if not any(math.hypot(p[0] - cx, p[1] - cy) <= 0.001 for p in pts):
            missing_corners.append((cx, cy))

    if len(missing_corners) != 1:
        return []

    cx, cy = missing_corners[0]
    missing_area = abs(cx - xr) * abs(cy - yr)
    aabb_area = (x_max - x_min) * (y_max - y_min)
    if abs((aabb_area - missing_area) - total_area) > 1e-4:
        return []

    cuts = []

    # Cut A: Horizontal cut at y = yr
    if abs(cy - y_max) <= 0.001:  # Missing corner is on top
        r1 = {
            "origin_m": [x_min, y_min],
            "width_m": x_max - x_min,
            "depth_m": yr - y_min,
            "area": (x_max - x_min) * (yr - y_min),
            "cut_edge": "north",
        }
        rx_min = x_min if abs(cx - x_max) <= 0.001 else xr
        rx_max = xr if abs(cx - x_max) <= 0.001 else x_max
        r2 = {
            "origin_m": [rx_min, yr],
            "width_m": rx_max - rx_min,
            "depth_m": y_max - yr,
            "area": (rx_max - rx_min) * (y_max - yr),
            "cut_edge": "south",
        }
        cut_len = rx_max - rx_min
    else:  # Missing corner is on bottom
        r1 = {
            "origin_m": [x_min, yr],
            "width_m": x_max - x_min,
            "depth_m": y_max - yr,
            "area": (x_max - x_min) * (y_max - yr),
            "cut_edge": "south",
        }
        rx_min = x_min if abs(cx - x_max) <= 0.001 else xr
        rx_max = xr if abs(cx - x_max) <= 0.001 else x_max
        r2 = {
            "origin_m": [rx_min, y_min],
            "width_m": rx_max - rx_min,
            "depth_m": yr - y_min,
            "area": (rx_max - rx_min) * (yr - y_min),
            "cut_edge": "north",
        }
        cut_len = rx_max - rx_min

    if abs((r1["area"] + r2["area"]) - total_area) <= 1e-6:
        cuts.append({
            "direction": "horizontal",
            "rectangles": [r1, r2],
            "shared_cut_length": cut_len,
        })

    # Cut B: Vertical cut at x = xr
    if abs(cx - x_max) <= 0.001:  # Missing corner is on right
        r1 = {
            "origin_m": [x_min, y_min],
            "width_m": xr - x_min,
            "depth_m": y_max - y_min,
            "area": (xr - x_min) * (y_max - y_min),
            "cut_edge": "east",
        }
        ry_min = y_min if abs(cy - y_max) <= 0.001 else yr
        ry_max = yr if abs(cy - y_max) <= 0.001 else y_max
        r2 = {
            "origin_m": [xr, ry_min],
            "width_m": x_max - xr,
            "depth_m": ry_max - ry_min,
            "area": (x_max - xr) * (ry_max - ry_min),
            "cut_edge": "west",
        }
        cut_len = ry_max - ry_min
    else:  # Missing corner is on left
        r1 = {
            "origin_m": [xr, y_min],
            "width_m": x_max - xr,
            "depth_m": y_max - y_min,
            "area": (x_max - xr) * (y_max - y_min),
            "cut_edge": "west",
        }
        ry_min = y_min if abs(cy - y_max) <= 0.001 else yr
        ry_max = yr if abs(cy - y_max) <= 0.001 else y_max
        r2 = {
            "origin_m": [x_min, ry_min],
            "width_m": xr - x_min,
            "depth_m": ry_max - ry_min,
            "area": (xr - x_min) * (ry_max - ry_min),
            "cut_edge": "east",
        }
        cut_len = ry_max - ry_min

    if abs((r1["area"] + r2["area"]) - total_area) <= 1e-6:
        cuts.append({
            "direction": "vertical",
            "rectangles": [r1, r2],
            "shared_cut_length": cut_len,
        })

    return cuts


def _select_best_cut(cuts: list[dict[str, Any]], unit: float) -> dict[str, Any] | None:
    if not cuts:
        return None
    stub_threshold = unit / 2.0

    def is_stub_partition(rect: dict[str, Any]) -> bool:
        return min(rect["width_m"], rect["depth_m"]) < stub_threshold

    # If exactly one candidate cut separates a single stub leg from the main room,
    # choose it so the stub partition is dropped and the main room preserved.
    single_stub_cuts = []
    for c in cuts:
        r1, r2 = c["rectangles"]
        s1 = is_stub_partition(r1)
        s2 = is_stub_partition(r2)
        if (s1 and not s2) or (s2 and not s1):
            single_stub_cuts.append(c)

    if len(single_stub_cuts) == 1:
        return single_stub_cuts[0]

    # Otherwise choose cut that maximizes min(area(RA), area(RB)).
    # Tie-breaker: longer shared cut edge.
    def cut_score(c: dict[str, Any]) -> tuple[float, float]:
        r1, r2 = c["rectangles"]
        min_area = min(r1["area"], r2["area"])
        return (min_area, c["shared_cut_length"])

    return max(cuts, key=cut_score)


def partition_l_tatami(
    layout: dict,
    *,
    standard: str = "Edoma",
    layout_type: str = "Shugikyo",
) -> dict:
    if standard not in TATAMI_STANDARDS:
        raise ValueError(f"Unknown tatami standard: {standard}")

    walls = layout.get("walls") if isinstance(layout.get("walls"), list) else []
    wall_count = len(walls)
    unit = TATAMI_STANDARDS[standard]["width_m"]

    # 1. Shape Dispatch
    if wall_count == 4:
        room = layout.get("room") if isinstance(layout.get("room"), dict) else {}
        try:
            w = float(room.get("width_m") or 0.0)
            d = float(room.get("depth_m") or 0.0)
        except (TypeError, ValueError):
            w, d = 0.0, 0.0
        if not (math.isfinite(w) and w > 0 and math.isfinite(d) and d > 0):
            aabb = _vertex_aabb(layout)
            if aabb is not None:
                w = aabb[1] - aabb[0]
                d = aabb[3] - aabb[2]
        tapers = derive_edge_tapers(layout)
        out = calculate_tatami_layout(
            room_width_m=w,
            room_depth_m=d,
            standard=standard,
            layout_type=layout_type,
            edge_tapers=tapers,
            frozen_edges=frozenset(),
        )
        out["shape_name"] = "RECTANGULAR"
        return out

    if wall_count != 6:
        room = layout.get("room") if isinstance(layout.get("room"), dict) else {}
        shape_name = room.get("shape_name", f"{wall_count}-gon")
        return {
            "skipped": True,
            "reason": "unsupported_shape",
            "shape_name": shape_name,
        }

    # 2. Orthogonality & CCW Normalization for 6-gons
    pts = _extract_polygon_vertices(layout)
    if pts is None or len(pts) != 6:
        return {
            "skipped": True,
            "reason": "l_partition_failed",
            "shape_name": "L-SHAPED",
        }

    ok, pts, total_area = _is_orthogonal_and_ccw(pts)
    if not ok:
        return {
            "skipped": True,
            "reason": "l_partition_failed",
            "shape_name": "L-SHAPED",
        }

    # 3. Identify Concave / Reflex Vertex
    r_idx = _find_reflex_vertex(pts)
    if r_idx is None:
        return {
            "skipped": True,
            "reason": "l_partition_failed",
            "shape_name": "L-SHAPED",
        }

    # 4. Candidate Axis Cuts
    cuts = _generate_candidate_cuts(pts, r_idx, total_area)
    chosen_cut = _select_best_cut(cuts, unit)
    if chosen_cut is None:
        return {
            "skipped": True,
            "reason": "l_partition_failed",
            "shape_name": "L-SHAPED",
        }

    # 5. Stub Detection & Filtering
    stub_threshold = unit / 2.0
    kept_rects = []
    warnings = []
    for rect in chosen_cut["rectangles"]:
        if min(rect["width_m"], rect["depth_m"]) < stub_threshold:
            warnings.append("tatami_stub_partition_dropped")
        else:
            kept_rects.append(rect)

    if not kept_rects:
        return {
            "skipped": True,
            "reason": "l_partition_failed",
            "shape_name": "L-SHAPED",
        }

    # 6. Tatami Layout on Kept Rectangles
    total_full = 0
    total_half = 0
    combined_mats = []
    combined_shavings = []
    partitions = []
    mat_counter = 1

    for rect in kept_rects:
        x_min, y_min = rect["origin_m"]
        w = rect["width_m"]
        d = rect["depth_m"]
        cut_edge = rect["cut_edge"]

        part_layout = calculate_tatami_layout(
            room_width_m=w,
            room_depth_m=d,
            standard=standard,
            layout_type=layout_type,
            edge_tapers=None,
            frozen_edges=frozenset({cut_edge}),
        )

        total_full += part_layout.get("full_mats_count", 0)
        total_half += part_layout.get("half_mats_count", 0)

        part_mats = part_layout.get("mats", [])
        id_map = {}
        for m in part_mats:
            old_id = m.get("mat_id")
            new_id = f"mat_{mat_counter}"
            mat_counter += 1
            id_map[old_id] = new_id

            translated_mat = dict(m)
            translated_mat["mat_id"] = new_id
            translated_mat["x_m"] = round(float(m["x_m"]) + x_min, 4)
            translated_mat["y_m"] = round(float(m["y_m"]) + y_min, 4)
            combined_mats.append(translated_mat)

        for detail in part_layout.get("custom_shaving_details", []):
            translated_detail = dict(detail)
            old_id = translated_detail.get("mat_id")
            if old_id in id_map:
                translated_detail["mat_id"] = id_map[old_id]
            combined_shavings.append(translated_detail)

        partitions.append({
            "width_m": round(w, 4),
            "depth_m": round(d, 4),
            "origin_m": [round(x_min, 4), round(y_min, 4)],
            "cut_edge": cut_edge,
            "mats": len(part_mats),
        })

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    room_w = max(xs) - min(xs)
    room_d = max(ys) - min(ys)

    out_result = {
        "layout_type": layout_type,
        "regional_standard": standard,
        "full_mats_count": total_full,
        "half_mats_count": total_half,
        "unit_m": round(unit, 4),
        "room_width_m": round(room_w, 4),
        "room_depth_m": round(room_d, 4),
        "has_shaving_warning": any(d.get("warning") for d in combined_shavings),
        "shape_name": "L-SHAPED",
        "partitions": partitions,
        "mats": combined_mats,
        "custom_shaving_details": combined_shavings,
    }
    if warnings:
        out_result["warnings"] = warnings
    return out_result
