from __future__ import annotations

import copy
import json
import math
import os
from typing import Any

import ezdxf
import numpy as np

LENGTH_EPS_M = 1e-9
VERTEX_DECIMALS = 3
AZIMUTH_DECIMALS = 1
PX_PER_METER = 100.0
SVG_PAD_PX = 20.0
DIM_OFFSET_WALL_M = 0.35
DIM_OFFSET_OVERALL_M = 0.70
DIM_EXTENSION_GAP_M = 0.05
LABEL_INSET_M = 0.15
SCALE_BAR_M = 1.0
SCALE_BAR_GAP_M = 0.40
H_NOTE_GAP_M = 0.30
DXF_TEXT_HEIGHT_M = 0.12
DEFAULT_WALL_THICKNESS_M = 0.15
SVG_FONT_PX = 12.0
EMPTY_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
)


def _xy_of(vertices: dict, name: Any) -> np.ndarray | None:
    if not isinstance(vertices, dict) or name not in vertices:
        return None
    try:
        arr = np.asarray(vertices[name], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 2:
        return None
    if not np.isfinite(arr[0]) or not np.isfinite(arr[1]):
        return None
    return np.array([float(arr[0]), float(arr[1])], dtype=float)


def _copy_xy(vertices: dict) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if not isinstance(vertices, dict):
        return out
    for key in vertices:
        xy = _xy_of(vertices, key)
        if xy is None:
            continue
        out[str(key)] = [float(xy[0]), float(xy[1])]
    return out


def _gravity_vector(layout: dict | None) -> list[float]:
    if not isinstance(layout, dict):
        return [0.0, 0.0, 0.0]
    try:
        arr = np.asarray(layout.get("gravity_vector"), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0]
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return [0.0, 0.0, 0.0]
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _sorted_walls(walls: list) -> list:
    indexed = list(enumerate(walls))

    def key(item: tuple[int, dict]) -> tuple[int, int]:
        i, w = item
        try:
            return (int(w.get("wall_index", i)), i)
        except (TypeError, ValueError):
            return (i, i)

    indexed.sort(key=key)
    return [w for _, w in indexed]


def _wall_length_azimuth(vertices: dict, wall: dict) -> tuple[float, float]:
    joints = wall.get("joints") or []
    if not isinstance(joints, (list, tuple)) or len(joints) < 2:
        return 0.0, 0.0
    p = _xy_of(vertices, joints[0])
    q = _xy_of(vertices, joints[1])
    if p is None or q is None:
        return 0.0, 0.0
    dxy = q - p
    length = float(np.hypot(dxy[0], dxy[1]))
    if length < LENGTH_EPS_M:
        return 0.0, 0.0
    deg = math.degrees(math.atan2(float(dxy[1]), float(dxy[0]))) % 360.0
    return length, deg


def _shoelace(pts: list[np.ndarray]) -> float:
    if len(pts) < 3:
        return 0.0
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _try_chain(vertices: dict, walls: list) -> list[np.ndarray] | None:
    segs: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for w in _sorted_walls(walls):
        joints = w.get("joints") or []
        if not isinstance(joints, (list, tuple)) or len(joints) < 2:
            continue
        a, b = str(joints[0]), str(joints[1])
        p = _xy_of(vertices, a)
        q = _xy_of(vertices, b)
        if p is None or q is None:
            continue
        if float(np.hypot(q[0] - p[0], q[1] - p[1])) < LENGTH_EPS_M:
            continue
        segs.append((p, q, a, b))
    if len(segs) < 3:
        return None
    for start_idx in range(len(segs)):
        ordered = [segs[start_idx]]
        used = {start_idx}
        while len(used) < len(segs):
            if len(ordered) >= 3 and np.allclose(ordered[-1][1], ordered[0][0], atol=1e-6):
                break
            tail = ordered[-1][3]
            tail_xy = ordered[-1][1]
            found = None
            flipped = False
            for i, (p, q, a, b) in enumerate(segs):
                if i in used:
                    continue
                if a == tail or np.allclose(p, tail_xy, atol=1e-6):
                    found = i
                    flipped = False
                    break
                if b == tail or np.allclose(q, tail_xy, atol=1e-6):
                    found = i
                    flipped = True
                    break
            if found is None:
                break
            p, q, a, b = segs[found]
            ordered.append((q, p, b, a) if flipped else (p, q, a, b))
            used.add(found)
        if len(ordered) >= 3 and np.allclose(ordered[-1][1], ordered[0][0], atol=1e-6):
            unique: list[np.ndarray] = []
            for p in (seg[0] for seg in ordered):
                if not unique or not np.allclose(unique[-1], p, atol=1e-6):
                    unique.append(p)
            if len(unique) >= 3:
                return unique
    return None


def _atan2_ring(vertices: dict) -> list[np.ndarray]:
    pts = [_xy_of(vertices, k) for k in vertices]
    pts = [p for p in pts if p is not None]
    if len(pts) < 3:
        return []
    c = np.mean(np.stack(pts, axis=0), axis=0)
    pts.sort(key=lambda p: math.atan2(float(p[1] - c[1]), float(p[0] - c[0])))
    return pts


def _polygon_pts(vertices: dict, walls: list) -> list[np.ndarray]:
    chained = _try_chain(vertices, walls)
    if chained is not None:
        return chained
    return _atan2_ring(vertices)


def _copy_floorplan_frame(layout: dict | None) -> dict | None:
    if not isinstance(layout, dict):
        return None
    raw = layout.get("floorplan_frame")
    if not isinstance(raw, dict):
        return None
    try:
        rot = np.asarray(raw.get("rotation_2x2"), dtype=float)
        trans = np.asarray(raw.get("translation_xy"), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if rot.shape != (2, 2) or trans.size != 2:
        return None
    if not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans)):
        return None
    src = raw.get("from", "gravity_work_xy")
    dst = raw.get("to", "canonical_floorplan")
    return {
        "from": str(src) if src is not None else "gravity_work_xy",
        "to": str(dst) if dst is not None else "canonical_floorplan",
        "rotation_2x2": [
            [float(rot[0, 0]), float(rot[0, 1])],
            [float(rot[1, 0]), float(rot[1, 1])],
        ],
        "translation_xy": [float(trans[0]), float(trans[1])],
    }


def enrich_layout(layout: dict | None) -> dict:
    if not isinstance(layout, dict):
        layout = {}
    vertices_in = layout.get("vertices") if isinstance(layout.get("vertices"), dict) else {}
    walls_in = layout.get("walls") if isinstance(layout.get("walls"), list) else []
    vertices = _copy_xy(vertices_in)
    walls_out: list[dict] = []
    heights: list[float] = []
    raw_lengths: list[float] = []
    for i, wall in enumerate(walls_in):
        if not isinstance(wall, dict):
            continue
        length, az = _wall_length_azimuth(vertices_in, wall)
        raw_lengths.append(length)
        length_m = float(round(length, VERTEX_DECIMALS))
        az_m = float(round(az, AZIMUTH_DECIMALS)) % 360.0
        if length_m == 0.0:
            az_m = 0.0
        try:
            idx = int(wall.get("wall_index", i))
        except (TypeError, ValueError):
            idx = i
        joints = wall.get("joints") or []
        joint_names = [str(j) for j in joints[:2]] if isinstance(joints, (list, tuple)) else []
        try:
            thickness = float(wall.get("thickness_meters", DEFAULT_WALL_THICKNESS_M))
        except (TypeError, ValueError):
            thickness = DEFAULT_WALL_THICKNESS_M
        if not math.isfinite(thickness) or thickness < 0.0:
            thickness = DEFAULT_WALL_THICKNESS_M
        try:
            height = float(wall.get("height_meters", 0.0))
        except (TypeError, ValueError):
            height = 0.0
        if math.isfinite(height):
            heights.append(height)
        else:
            height = 0.0
        walls_out.append(
            {
                "wall_index": idx,
                "joints": joint_names,
                "thickness_meters": float(round(thickness, VERTEX_DECIMALS)),
                "height_meters": float(round(height, VERTEX_DECIMALS)),
                "length_meters": length_m,
                "azimuth_deg": az_m,
            }
        )
    xs = [xy[0] for xy in vertices.values()]
    ys = [xy[1] for xy in vertices.values()]
    width = (max(xs) - min(xs)) if xs else 0.0
    depth = (max(ys) - min(ys)) if ys else 0.0
    poly = _polygon_pts(vertices, walls_out)
    area = _shoelace(poly)
    perimeter = sum(raw_lengths)
    room_h = max(heights) if heights else 0.0
    portals_in = layout.get("portals") if isinstance(layout.get("portals"), list) else []
    out = {
        "vertices": vertices,
        "walls": walls_out,
        "portals": copy.deepcopy(portals_in),
        "floor": copy.deepcopy(layout.get("floor")),
        "ceiling": copy.deepcopy(layout.get("ceiling")),
        "gravity_vector": _gravity_vector(layout),
        "units": "m",
        "room": {
            "area_m2": float(round(area, VERTEX_DECIMALS)),
            "perimeter_m": float(round(perimeter, VERTEX_DECIMALS)),
            "height_meters": float(round(room_h, VERTEX_DECIMALS)),
            "width_m": float(round(width, VERTEX_DECIMALS)),
            "depth_m": float(round(depth, VERTEX_DECIMALS)),
        },
    }
    n_walls = len(walls_out)
    if n_walls == 4:
        shape_name = "RECTANGULAR"
    elif n_walls == 6:
        obl = False
        for w in walls_out:
            joints = w.get("joints") or []
            if len(joints) >= 2:
                p = _xy_of(vertices, joints[0])
                q = _xy_of(vertices, joints[1])
                if p is not None and q is not None:
                    dx = float(q[0] - p[0])
                    dy = float(q[1] - p[1])
                    h = math.degrees(math.atan2(dy, dx)) % 180.0
                    if h < 0.0:
                        h += 180.0
                    d0 = min(h, 180.0 - h)
                    d90 = abs(h - 90.0)
                    if d0 > 15.0 and d90 > 15.0:
                        obl = True
                        break
            else:
                az = float(w.get("azimuth_deg", 0.0)) % 180.0
                if az < 0.0:
                    az += 180.0
                d0 = min(az, 180.0 - az)
                d90 = abs(az - 90.0)
                if d0 > 15.0 and d90 > 15.0:
                    obl = True
                    break
        shape_name = "COMPLEX (6-gon)" if obl else "L-SHAPED"
    elif n_walls == 8:
        shape_name = "T-SHAPED/U-SHAPED"
    elif n_walls >= 3:
        shape_name = f"COMPLEX ({n_walls}-gon)"
    else:
        shape_name = None
    shape_name = shape_name or layout.get("room", {}).get("shape_name")
    if shape_name:
        out["room"]["shape_name"] = shape_name
    if isinstance(layout.get("bbox_fill_ratio"), (int, float)):
        out["room"]["bbox_fill_ratio"] = float(layout["bbox_fill_ratio"])
    elif isinstance(layout.get("room", {}).get("bbox_fill_ratio"), (int, float)):
        out["room"]["bbox_fill_ratio"] = float(layout["room"]["bbox_fill_ratio"])

    frame = _copy_floorplan_frame(layout)
    if frame is not None:
        out["floorplan_frame"] = frame
    if "level_frame" in layout and isinstance(layout["level_frame"], dict):
        out["level_frame"] = copy.deepcopy(layout["level_frame"])
    if isinstance(layout.get("diagnostics"), dict):
        out["diagnostics"] = copy.deepcopy(layout["diagnostics"])
    if isinstance(layout.get("rooms"), list):
        out["rooms"] = copy.deepcopy(layout["rooms"])
    keepouts = layout.get("keepouts")
    if isinstance(keepouts, list) and keepouts:
        out["keepouts"] = copy.deepcopy(keepouts)
        from app.services.occupancy_layout import finishable_floor_polygon

        ring = np.asarray(poly, dtype=float)
        finish = finishable_floor_polygon(
            ring, [k.get("polygon") for k in keepouts if isinstance(k, dict) and k.get("polygon")]
        )
        finish_area = float(round(float(finish["area_m2"]), VERTEX_DECIMALS))
        out["room"]["finishable_area_m2"] = finish_area
        out["room"]["finishable_vertices"] = finish["vertices"]
        out["room"]["keepout_area_m2"] = float(round(max(0.0, area - finish_area), VERTEX_DECIMALS))

    clear_interior_dims = layout.get("clear_interior_dimensions") if isinstance(layout, dict) else None
    if not isinstance(clear_interior_dims, dict):
        if width > 0.0 and depth > 0.0:
            h = float(room_h)
            short = float(round(min(width, depth), 4))
            long = float(round(max(width, depth), 4))
            clear_interior_dims = {
                "height_m": h,
                "short_side_m": short,
                "long_side_m": long,
                "height_cm": float(round(h * 100.0, 1)),
                "short_side_cm": float(round(short * 100.0, 1)),
                "long_side_cm": float(round(long * 100.0, 1)),
            }
    if clear_interior_dims is not None:
        out["clear_interior_dimensions"] = copy.deepcopy(clear_interior_dims)
        out["room"]["clear_interior_height_m"] = float(round(clear_interior_dims["height_m"], 4))
        out["room"]["clear_interior_short_side_m"] = float(round(clear_interior_dims["short_side_m"], 4))
        out["room"]["clear_interior_long_side_m"] = float(round(clear_interior_dims["long_side_m"], 4))
        out["room"]["clear_interior_height_cm"] = float(round(clear_interior_dims["height_cm"], 1))
        out["room"]["clear_interior_short_side_cm"] = float(round(clear_interior_dims["short_side_cm"], 1))
        out["room"]["clear_interior_long_side_cm"] = float(round(clear_interior_dims["long_side_cm"], 1))

    return out


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _write_json(enriched: dict, output_path: str) -> dict:
    ready = _json_ready(enriched)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ready, f, indent=2, ensure_ascii=False, allow_nan=False)
    return ready


def export_json(layout: dict | None, output_path: str) -> dict:
    return _write_json(enrich_layout(layout), output_path)


def _is_empty(layout: dict) -> bool:
    vertices = layout.get("vertices") if isinstance(layout.get("vertices"), dict) else {}
    walls = layout.get("walls") if isinstance(layout.get("walls"), list) else []
    if not vertices:
        return True
    for wall in walls:
        if not isinstance(wall, dict):
            continue
        joints = wall.get("joints") or []
        if not isinstance(joints, (list, tuple)) or len(joints) < 2:
            continue
        p = _xy_of(vertices, joints[0])
        q = _xy_of(vertices, joints[1])
        if p is None or q is None:
            continue
        if float(np.hypot(q[0] - p[0], q[1] - p[1])) >= LENGTH_EPS_M:
            return False
    return True


def _mm(meters: float) -> str:
    return str(int(round(float(meters) * 1000.0)))


def _readable_rotation(deg: float) -> float:
    a = deg % 360.0
    if 90.0 < a <= 270.0:
        a = (a + 180.0) % 360.0
    return a


def _centroid(pts: list[np.ndarray]) -> np.ndarray | None:
    if len(pts) < 2:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def _outward(p: np.ndarray, q: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    e = q - p
    nlen = float(np.hypot(e[0], e[1]))
    hat = e / nlen
    n = np.array([-hat[1], hat[0]], dtype=float)
    mid = 0.5 * (p + q)
    if float(np.dot(mid - centroid, n)) < 0.0:
        n = -n
    return n


def _annotation_items(layout: dict) -> tuple[list[dict], list[dict]]:
    """dim_lines [{x0,y0,x1,y1}], texts [{x,y,s,rot}]. Work-frame metres."""
    vertices = layout["vertices"]
    walls = layout["walls"]
    room = layout["room"]
    segs: list[tuple[np.ndarray, np.ndarray, dict]] = []
    for wall in walls:
        joints = wall.get("joints") or []
        if len(joints) < 2:
            continue
        p = _xy_of(vertices, joints[0])
        q = _xy_of(vertices, joints[1])
        if p is None or q is None:
            continue
        if float(wall.get("length_meters", 0.0)) < LENGTH_EPS_M:
            continue
        segs.append((p, q, wall))
    if not vertices:
        return [], []
    poly = _polygon_pts(vertices, walls)
    if len(poly) >= 3:
        c = _centroid(poly)
    else:
        valid = [_xy_of(vertices, k) for k in vertices]
        valid = [p for p in valid if p is not None]
        c = _centroid(valid) if len(valid) >= 2 else (valid[0] if valid else None)
    dim_lines: list[dict] = []
    texts: list[dict] = []

    def add_line(a: np.ndarray, b: np.ndarray) -> None:
        dim_lines.append(
            {"x0": float(a[0]), "y0": float(a[1]), "x1": float(b[0]), "y1": float(b[1])}
        )

    def add_text(x: float, y: float, s: str, rot: float = 0.0) -> None:
        texts.append({"x": float(x), "y": float(y), "s": s, "rot": float(rot)})

    if c is not None:
        for p, q, wall in segs:
            n = _outward(p, q, c)
            gap = DIM_EXTENSION_GAP_M
            off = DIM_OFFSET_WALL_M
            a0 = p + n * gap
            a1 = p + n * off
            b0 = q + n * gap
            b1 = q + n * off
            add_line(a0, a1)
            add_line(b0, b1)
            add_line(a1, b1)
            mid = 0.5 * (a1 + b1)
            e = q - p
            rot = _readable_rotation(math.degrees(math.atan2(float(e[1]), float(e[0]))))
            add_text(float(mid[0]), float(mid[1]), _mm(wall["length_meters"]), rot)
            inner = 0.5 * (p + q) - n * LABEL_INSET_M
            add_text(float(inner[0]), float(inner[1]), f"W{wall['wall_index']}", 0.0)

    xs = [v[0] for v in vertices.values()]
    ys = [v[1] for v in vertices.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width_m = float(room["width_m"])
    depth_m = float(room["depth_m"])
    if width_m >= LENGTH_EPS_M:
        y_dim = y_min - DIM_OFFSET_OVERALL_M
        add_line(np.array([x_min, y_min - DIM_EXTENSION_GAP_M]), np.array([x_min, y_dim]))
        add_line(np.array([x_max, y_min - DIM_EXTENSION_GAP_M]), np.array([x_max, y_dim]))
        add_line(np.array([x_min, y_dim]), np.array([x_max, y_dim]))
        add_text(0.5 * (x_min + x_max), y_dim, _mm(width_m), 0.0)
    if depth_m >= LENGTH_EPS_M:
        x_dim = x_max + DIM_OFFSET_OVERALL_M
        add_line(np.array([x_max + DIM_EXTENSION_GAP_M, y_min]), np.array([x_dim, y_min]))
        add_line(np.array([x_max + DIM_EXTENSION_GAP_M, y_max]), np.array([x_dim, y_max]))
        add_line(np.array([x_dim, y_min]), np.array([x_dim, y_max]))
        add_text(x_dim, 0.5 * (y_min + y_max), _mm(depth_m), 90.0)

    area = float(room["area_m2"])
    if c is not None and area > 0.0 and len(poly) >= 3:
        add_text(float(c[0]), float(c[1]), f"{area:.2f} m²", 0.0)

    bar_y = y_min - DIM_OFFSET_OVERALL_M - SCALE_BAR_GAP_M
    bar_x0 = x_min
    bar_x1 = x_min + SCALE_BAR_M
    add_line(np.array([bar_x0, bar_y]), np.array([bar_x1, bar_y]))
    add_text(bar_x0, bar_y, "0", 0.0)
    add_text(bar_x1, bar_y, "1000", 0.0)
    height_m = float(room["height_meters"])
    if height_m > 0.0:
        add_text(bar_x1 + H_NOTE_GAP_M, bar_y, f"H={_mm(height_m)}", 0.0)
    return dim_lines, texts


def _write_svg(layout: dict, output_path: str) -> None:
    if _is_empty(layout):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(EMPTY_SVG)
        return
    dim_lines, texts = _annotation_items(layout)
    xs: list[float] = []
    ys: list[float] = []
    for v in layout["vertices"].values():
        xs.append(float(v[0]))
        ys.append(float(v[1]))
    for ln in dim_lines:
        xs.extend([ln["x0"], ln["x1"]])
        ys.extend([ln["y0"], ln["y1"]])
    for t in texts:
        xs.append(t["x"])
        ys.append(t["y"])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    scale = PX_PER_METER
    pad = SVG_PAD_PX
    width = (x_max - x_min) * scale + 2 * pad
    height = (y_max - y_min) * scale + 2 * pad

    def sx(x: float) -> float:
        return scale * (x - x_min) + pad

    def sy(y: float) -> float:
        return scale * (y - y_min) + pad

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'  <g transform="translate(0, {height}) scale(1, -1)">',
    ]
    for wall in layout["walls"]:
        joints = wall.get("joints") or []
        if len(joints) < 2:
            continue
        v0 = layout["vertices"].get(joints[0])
        v1 = layout["vertices"].get(joints[1])
        if v0 is None or v1 is None:
            continue
        stroke_width = float(wall.get("thickness_meters", DEFAULT_WALL_THICKNESS_M)) * scale
        parts.append(
            f'    <line x1="{sx(v0[0])}" y1="{sy(v0[1])}" x2="{sx(v1[0])}" y2="{sy(v1[1])}" '
            f'stroke="black" stroke-width="{stroke_width}" stroke-linecap="round" '
            f'vector-effect="non-scaling-stroke" />'
        )
    for ln in dim_lines:
        parts.append(
            f'    <line x1="{sx(ln["x0"])}" y1="{sy(ln["y0"])}" x2="{sx(ln["x1"])}" y2="{sy(ln["y1"])}" '
            f'stroke="black" stroke-width="1" vector-effect="non-scaling-stroke" />'
        )
    for t in texts:
        tx, ty = sx(t["x"]), sy(t["y"])
        rot = t["rot"]
        label = t["s"]
        parts.append(
            f'    <text text-anchor="middle" dominant-baseline="middle" '
            f'font-family="sans-serif" font-size="{SVG_FONT_PX}" '
            f'transform="translate({tx}, {ty}) scale(1, -1) rotate({-rot})">{label}</text>'
        )
    parts.append("  </g>")
    parts.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def export_svg(layout: dict | None, output_path: str) -> None:
    _write_svg(enrich_layout(layout), output_path)


def _dxf_new():
    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 6
    doc.header["$MEASUREMENT"] = 1
    doc.header["$LUNITS"] = 2
    for name, color in [("WALL", 1), ("DOOR", 5), ("WINDOW", 4), ("DIM", 7), ("TEXT", 7)]:
        if name not in doc.layers:
            doc.layers.add(name, color=color)
    return doc


def _write_dxf(layout: dict, output_path: str) -> None:
    doc = _dxf_new()
    if _is_empty(layout):
        doc.saveas(output_path)
        return
    msp = doc.modelspace()
    for wall in layout["walls"]:
        joints = wall.get("joints") or []
        if len(joints) < 2:
            continue
        v0 = layout["vertices"].get(joints[0])
        v1 = layout["vertices"].get(joints[1])
        if v0 is None or v1 is None:
            continue
        v0_2d = np.array(v0[:2], dtype=float)
        v1_2d = np.array(v1[:2], dtype=float)
        dir_vec = v1_2d - v0_2d
        length = float(np.linalg.norm(dir_vec))
        if length < LENGTH_EPS_M:
            continue
        normal = np.array([-dir_vec[1], dir_vec[0]]) / length
        thickness = float(wall.get("thickness_meters", DEFAULT_WALL_THICKNESS_M))
        p1 = v0_2d + 0.5 * thickness * normal
        p2 = v1_2d + 0.5 * thickness * normal
        p3 = v1_2d - 0.5 * thickness * normal
        p4 = v0_2d - 0.5 * thickness * normal
        poly = msp.add_lwpolyline(
            [
                (float(p1[0]), float(p1[1])),
                (float(p4[0]), float(p4[1])),
                (float(p3[0]), float(p3[1])),
                (float(p2[0]), float(p2[1])),
            ],
            dxfattribs={"layer": "WALL"},
        )
        poly.closed = True
    dim_lines, texts = _annotation_items(layout)
    for ln in dim_lines:
        msp.add_line(
            (ln["x0"], ln["y0"]),
            (ln["x1"], ln["y1"]),
            dxfattribs={"layer": "DIM"},
        )
    for t in texts:
        s = t["s"].replace("m²", "m2")
        entity = msp.add_text(
            s,
            dxfattribs={
                "layer": "TEXT",
                "height": DXF_TEXT_HEIGHT_M,
                "rotation": t["rot"],
            },
        )
        entity.set_placement(
            (t["x"], t["y"]), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER
        )
    doc.saveas(output_path)


def export_dxf(layout: dict | None, output_path: str) -> None:
    _write_dxf(enrich_layout(layout), output_path)


def export_artifacts(layout: dict | None, session_dir: str) -> dict:
    enriched = enrich_layout(layout)
    tmp_json = os.path.join(session_dir, "floorplan.tmp.json")
    tmp_svg = os.path.join(session_dir, "floorplan.tmp.svg")
    tmp_dxf = os.path.join(session_dir, "floorplan.tmp.dxf")
    final_json = os.path.join(session_dir, "floorplan.json")
    final_svg = os.path.join(session_dir, "floorplan.svg")
    final_dxf = os.path.join(session_dir, "floorplan.dxf")
    tmps = (tmp_json, tmp_svg, tmp_dxf)
    try:
        ready = _write_json(enriched, tmp_json)
        _write_svg(enriched, tmp_svg)
        _write_dxf(enriched, tmp_dxf)
        for tmp in tmps:
            if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                raise ValueError("Failed to write floorplan artifacts.")
        os.replace(tmp_json, final_json)
        os.replace(tmp_svg, final_svg)
        os.replace(tmp_dxf, final_dxf)
        return ready
    finally:
        for tmp in tmps:
            try:
                if os.path.isfile(tmp):
                    os.unlink(tmp)
            except OSError:
                pass


def _resolve_wall_index(raw_wid: Any, n: int, walls: list[dict]) -> int | None:
    """Normalize and map a wall ID to its 0-based wall index in [0, n - 1]."""
    if raw_wid is None:
        return None

    if isinstance(raw_wid, int):
        if 0 <= raw_wid < n:
            return raw_wid
        for i, w in enumerate(walls[:n]):
            if w.get("wall_index") == raw_wid:
                return i
        return None

    s = str(raw_wid).strip()
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < n:
            return idx
        for i, w in enumerate(walls[:n]):
            if str(w.get("wall_index")) == s:
                return i
        return None

    if (s.startswith("W") or s.startswith("w")) and s[1:].isdigit():
        idx = int(s[1:])
        if 0 <= idx < n:
            return idx
        for i, w in enumerate(walls[:n]):
            if str(w.get("wall_index")) == s[1:]:
                return i
        return None

    for i, w in enumerate(walls[:n]):
        if str(w.get("wall_index")) == s or str(w.get("id")) == s:
            return i

    return None


def apply_wall_overrides_to_metrics(
    layout: dict,
    overrides: list[dict],
    update_mode: str = "auto_close",
    height_mm: float | None = None,
) -> dict:
    from app.core.exceptions import MaterialEstimationError

    if update_mode not in ("auto_close", "scale", "proportional"):
        raise MaterialEstimationError(detail=f"Unknown update_mode: {update_mode}")

    valid_items = [
        item
        for item in (overrides or [])
        if isinstance(item, dict) and item.get("wall_id") is not None and item.get("length_mm") is not None
    ]

    if not valid_items and height_mm is not None:
        rebuilt = copy.deepcopy(layout)
        height_m = float(height_mm) / 1000.0
        if "walls" in rebuilt and isinstance(rebuilt["walls"], list):
            for wall in rebuilt["walls"]:
                if isinstance(wall, dict):
                    wall["height_meters"] = height_m
        if "room" in rebuilt and isinstance(rebuilt["room"], dict):
            rebuilt["room"]["height_meters"] = height_m
        return enrich_layout(rebuilt)

    if update_mode in ("scale", "proportional") and not valid_items and height_mm is None:
        raise MaterialEstimationError(detail=f"{update_mode.capitalize()} mode needs a reference wall.")

    if not valid_items and height_mm is None:
        return enrich_layout(copy.deepcopy(layout))

    vertices_in = layout.get("vertices") if isinstance(layout.get("vertices"), dict) else {}
    walls_in = layout.get("walls") if isinstance(layout.get("walls"), list) else []
    names = []
    pts = []
    for wall in walls_in:
        joints = wall.get("joints") or []
        if len(joints) < 2:
            continue
        p = _xy_of(vertices_in, joints[0])
        q = _xy_of(vertices_in, joints[1])
        if p is None or q is None:
            continue
        names.append(joints[0])
        pts.append(p)
    if len(pts) < 3:
        raise MaterialEstimationError(detail="PLY metrics do not contain enough vertices")
    n = len(pts)

    override_by_index: dict[int, float] = {}
    override_by_id: dict[str, float] = {}

    for item in valid_items:
        raw_wid = item.get("wall_id")
        idx = _resolve_wall_index(raw_wid, n, walls_in)
        if idx is None:
            raise MaterialEstimationError(detail=f"Unknown wall id: {raw_wid}")

        try:
            length_m = float(item["length_mm"]) / 1000.0
        except (ValueError, TypeError):
            raise MaterialEstimationError(detail=f"{raw_wid} length must be a valid number")

        if length_m <= 0:
            raise MaterialEstimationError(detail=f"{raw_wid} length must be greater than 0")

        override_by_index[idx] = length_m
        override_by_id[f"W{idx}"] = length_m

    orig_lengths = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        orig_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if orig_len <= 1e-6:
            raise MaterialEstimationError(detail=f"Wall {i} has near-zero length")
        orig_lengths.append(orig_len)

    if update_mode == "proportional":
        scale_ratios = [
            override_by_index[idx] / orig_lengths[idx]
            for idx in override_by_index
        ]
        s = float(np.median(scale_ratios))

        pts_arr = np.asarray(pts, dtype=float)
        centroid = np.mean(pts_arr, axis=0)
        scaled_pts = centroid + s * (pts_arr - centroid)

        v_keys = [names[k] if (names and len(names) == n and len(set(names)) == n) else f"v{k}" for k in range(n)]
        new_vertices = {
            v_keys[k]: [
                round(float(scaled_pts[k][0]), VERTEX_DECIMALS),
                round(float(scaled_pts[k][1]), VERTEX_DECIMALS),
            ]
            for k in range(n)
        }

        height_m = float(height_mm) / 1000.0 if height_mm is not None else None
        new_walls = []
        for i, wall in enumerate(walls_in):
            item = dict(wall)
            item["joints"] = [v_keys[i], v_keys[(i + 1) % n]]
            if height_m is not None:
                item["height_meters"] = height_m
            new_walls.append(item)

        rebuilt = copy.deepcopy(layout)
        rebuilt["vertices"] = new_vertices
        rebuilt["walls"] = new_walls
        if height_m is not None and "room" in rebuilt and isinstance(rebuilt["room"], dict):
            rebuilt["room"] = dict(rebuilt["room"])
            rebuilt["room"]["height_meters"] = height_m

        return enrich_layout(rebuilt)

    axes, signs, lengths, ids = [], [], [], []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
        if abs(dx) >= abs(dy):
            if abs(dy) > 0.001:
                raise MaterialEstimationError(detail="Vertices must be orthogonal")
            axes.append("x")
            signs.append(1.0 if dx >= 0 else -1.0)
        else:
            if abs(dx) > 0.001:
                raise MaterialEstimationError(detail="Vertices must be orthogonal")
            axes.append("y")
            signs.append(1.0 if dy >= 0 else -1.0)
        wid = f"W{i}"
        ids.append(wid)
        orig_len = orig_lengths[i]
        lengths.append(override_by_id.get(wid, orig_len))

    if update_mode == "scale":
        ratios = [
            override_by_id[wid] / orig_lengths[ids.index(wid)]
            for wid in ids
            if wid in override_by_id
        ]
        first_ratio = ratios[0]
        if any(abs(r - first_ratio) > 0.01 for r in ratios):
            raise MaterialEstimationError(
                detail="Scale mode needs edited walls to share the same scale ratio."
            )
        s = first_ratio
        lengths = [orig_len * s for orig_len in orig_lengths]

    edited = set(override_by_id)
    for axis in ("x", "y"):
        idx = [i for i, a in enumerate(axes) if a == axis]
        delta = sum(signs[i] * lengths[i] for i in idx)
        if abs(delta) <= 0.001:
            continue
        candidates = [i for i in idx if ids[i] not in edited]
        adjusted = False
        for i in sorted(candidates, key=lambda j: lengths[j], reverse=True):
            new_len = lengths[i] - delta / signs[i]
            if new_len > 0.001:
                lengths[i] = new_len
                adjusted = True
                break
        if not adjusted:
            raise MaterialEstimationError(
                detail=f"Edited {axis} walls cannot close the polygon. Closing error is {delta*1000:.1f}mm."
            )
    new_pts = [np.array(pts[0], dtype=float)]
    for i in range(n):
        cur = new_pts[-1]
        nxt = (
            np.array([cur[0] + signs[i] * lengths[i], cur[1]])
            if axes[i] == "x"
            else np.array([cur[0], cur[1] + signs[i] * lengths[i]])
        )
        if i < n - 1:
            new_pts.append(nxt)
        else:
            err = float(np.hypot(nxt[0] - new_pts[0][0], nxt[1] - new_pts[0][1]))
            if err > 0.001:
                raise MaterialEstimationError(
                    detail=f"Wall lengths cannot close the polygon. Closing error: {err*1000:.1f}mm."
                )
    v_keys = [names[k] if (names and len(names) == n and len(set(names)) == n) else f"v{k}" for k in range(n)]
    new_vertices = {
        v_keys[i]: [
            round(float(new_pts[i][0]), VERTEX_DECIMALS),
            round(float(new_pts[i][1]), VERTEX_DECIMALS),
        ]
        for i in range(n)
    }
    height_m = float(height_mm) / 1000.0 if height_mm is not None else None
    new_walls = []
    for i, wall in enumerate(walls_in):
        item = dict(wall)
        item["joints"] = [v_keys[i], v_keys[(i + 1) % n]]
        if height_m is not None:
            item["height_meters"] = height_m
        new_walls.append(item)
    rebuilt = copy.deepcopy(layout)
    rebuilt["vertices"] = new_vertices
    rebuilt["walls"] = new_walls
    if height_m is not None and "room" in rebuilt and isinstance(rebuilt["room"], dict):
        rebuilt["room"] = dict(rebuilt["room"])
        rebuilt["room"]["height_meters"] = height_m
    return enrich_layout(rebuilt)


