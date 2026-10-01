import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from app.core.exceptions import (
    MaterialEstimationError,
    TatamiShavingLimitExceededError,
    TatamiSkewAngleExceededError,
)
from app.services.cad.contract import layout_to_metrics_mm
from app.services.materials import (
    estimate_cf,
    estimate_neda,
    estimate_plywood,
    estimate_tiling,
)
from app.services.tatami import (
    calculate_tatami_layout,
    derive_edge_tapers,
    _usable_walls,
    _LENGTH_EPS,
)
from app.services.tatami_partition import partition_l_tatami
from app.services.wallpaper_constants import (
    DEFAULT_WALLPAPER_OVERLAP_M,
    DEFAULT_WALLPAPER_REPEAT_M,
    DEFAULT_WALLPAPER_TRIM_M,
    WALLPAPER_STANDARDS,
)

_orig_calculate_tatami_layout = calculate_tatami_layout

__all__ = [
    "DEFAULT_RECALCULATE_PARAMS",
    "calculate_wallpaper_requirements",
    "calculate_tatami_layout",
    "derive_edge_tapers",
    "estimate_materials_for_session",
    "get_unobstructed_wall_segments",
]

_REQUIRED_KEYS = (
    "wallpaper_width_m",
    "wallpaper_roll_length_m",
    "wallpaper_trim_m",
    "wallpaper_overlap_m",
    "wallpaper_repeat_m",
    "tatami_standard",
    "tatami_layout_type",
)

_first = WALLPAPER_STANDARDS[0]
DEFAULT_RECALCULATE_PARAMS: dict[str, Any] = {
    "wallpaper_width_m": _first["width_m"],
    "wallpaper_roll_length_m": _first["roll_length_m"],
    "wallpaper_trim_m": DEFAULT_WALLPAPER_TRIM_M,
    "wallpaper_overlap_m": DEFAULT_WALLPAPER_OVERLAP_M,
    "wallpaper_repeat_m": DEFAULT_WALLPAPER_REPEAT_M,
    "wallpaper_match_type": None,
    "tatami_standard": "Edoma",
    "tatami_layout_type": "Shugikyo",
    "update_mode": "auto_close",
    "joist_pitch_mm": 303.0,
    "border_width_mm": 30.0,
    "board_length_mm": 1820.0,
    "board_width_mm": 910.0,
    "cf_roll_width_mm": 940.0,
    "tile_length_mm": 300.0,
    "tile_width_mm": 300.0,
    "tile_joint_width_mm": 3.0,
    "waste_rate": 0.0,
}


def calculate_wallpaper_requirements(
    height_m: float,
    number_of_drops: int,
    roll_length_m: float,
    trim_m: float,
    repeat_m: float,
    match_type: str,
) -> dict[str, Any]:
    if match_type == "free" or repeat_m == 0:
        drop_length = height_m + trim_m
        drops_per_roll = math.floor(roll_length_m / drop_length)
    elif match_type == "straight":
        drop_length = math.ceil((height_m + trim_m) / repeat_m) * repeat_m
        drops_per_roll = math.floor(roll_length_m / drop_length)
    elif match_type == "half-drop":
        half_repeat = repeat_m / 2.0
        c = math.ceil((height_m + trim_m) / half_repeat)
        drop_length = c * half_repeat
        if c % 2 == 0:
            drops_per_roll = math.floor(
                (roll_length_m + half_repeat) / (drop_length + half_repeat)
            )
        else:
            drops_per_roll = math.floor(roll_length_m / drop_length)
    else:
        raise ValueError(f"Invalid match type: {match_type}")

    if drops_per_roll <= 0:
        raise ValueError("Drop length exceeds total roll length.")

    calculated_rolls = math.ceil(number_of_drops / drops_per_roll)
    return {
        "drop_length_m": drop_length,
        "drops_per_roll": drops_per_roll,
        "calculated_rolls": calculated_rolls,
    }


def _wall_length(wall: dict, computed: float) -> float:
    raw = wall.get("length_meters")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return computed
    if math.isfinite(value) and value >= _LENGTH_EPS:
        return value
    return computed



def _match_type(parameters: dict[str, Any], repeat_m: float) -> str:
    explicit = parameters.get("wallpaper_match_type")
    if explicit in (None, ""):
        return "free" if repeat_m == 0.0 else "half-drop"
    return str(explicit)


def _extract_polygons(geom: Any) -> list[Polygon]:
    """Recursively extract all Polygons from any Shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for sub in geom.geoms:
            out.extend(_extract_polygons(sub))
        return out
    return []


def _get_corner_boxes(
    vertices: dict,
    usable_walls: list[tuple[tuple[float, float], tuple[float, float], float, dict]],
    keepouts: list[dict],
) -> list[Polygon]:
    """Generate corner box polygons where two distinct wardrobes meet at an L-corner."""
    if len(keepouts) < 2:
        return []

    corner_boxes: list[Polygon] = []
    v_map: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for wi, (p, q, comp, wall) in enumerate(usable_walls):
        j = wall.get("joints") or []
        if len(j) >= 2 and j[0] in vertices and j[1] in vertices:
            p_pt = np.asarray(p, dtype=float)
            q_pt = np.asarray(q, dtype=float)
            L = float(comp)
            if L > _LENGTH_EPS:
                u = (q_pt - p_pt) / L
                v_map.setdefault(j[0], []).append((wi, u, p_pt))
                v_map.setdefault(j[1], []).append((wi, -u, q_pt))

    for v_id, conns in v_map.items():
        if len(conns) == 2:
            (w1, u1, v_pt), (w2, u2, _) = conns
            if abs(float(np.dot(u1, u2))) > 0.95:
                continue

            candidates_w1: list[tuple[int, float, float, float]] = []
            candidates_w2: list[tuple[int, float, float, float]] = []

            for idx, k in enumerate(keepouts):
                try:
                    poly = np.asarray(k.get("polygon", []), dtype=float)
                except (ValueError, TypeError):
                    continue
                if len(poly) < 3:
                    continue
                rel = poly - v_pt
                s1 = rel @ u1
                s2 = rel @ u2
                s1_min, s1_max = float(np.min(s1)), float(np.max(s1))
                s2_min, s2_max = float(np.min(s2)), float(np.max(s2))
                span1 = s1_max - s1_min
                span2 = s2_max - s2_min
                depth_meta = float(k.get("depth_m", 0.0) or 0.0)

                # Keepout along wall 1: primary span along u1, lies close to wall 1 (s2_min near 0)
                # Must be on the room interior corner quadrant
                if (
                    span1 > span2
                    and -0.15 <= s2_min <= 0.25
                    and -0.15 <= s1_min <= 1.20
                    and s1_max > 0.30
                ):
                    d1 = depth_meta if depth_meta > 0.20 else span2
                    candidates_w1.append((idx, s1_min, s1_max, d1))

                # Keepout along wall 2: primary span along u2, lies close to wall 2 (s1_min near 0)
                # Must be on the room interior corner quadrant
                if (
                    span2 > span1
                    and -0.15 <= s1_min <= 0.25
                    and -0.15 <= s2_min <= 1.20
                    and s2_max > 0.30
                ):
                    d2 = depth_meta if depth_meta > 0.20 else span1
                    candidates_w2.append((idx, s2_min, s2_max, d2))

            for idx1, s1_min, s1_max, d1 in candidates_w1:
                for idx2, s2_min, s2_max, d2 in candidates_w2:
                    if idx1 == idx2:
                        continue
                    if s1_min <= d2 + 0.30 and s2_min <= d1 + 0.30:
                        L1 = max(0.0, min(max(s1_min, d2), 1.20))
                        L2 = max(0.0, min(max(s2_min, d1), 1.20))
                        c_poly = Polygon([
                            v_pt,
                            v_pt + L1 * u1,
                            v_pt + L1 * u1 + L2 * u2,
                            v_pt + L2 * u2,
                        ])
                        if not c_poly.is_valid:
                            c_poly = make_valid(c_poly)
                        if not c_poly.is_empty:
                            for p_box in _extract_polygons(c_poly):
                                corner_boxes.append(p_box)

    return corner_boxes


def get_unobstructed_wall_segments(
    floorplan: dict,
    band_depth: float = 0.20,
    corner_snap_tol: float = 0.15,
    min_seg_len: float = 0.05,
) -> list[list[float]]:
    """Compute lengths of unobstructed wall segments for each usable wall.

    For each wall, subtracts segments obstructed by full-height wardrobes
    (keepouts). If a wall is entirely obstructed, its list of segments is empty ([]).
    Returns a list of lists: [[seg1_m, seg2_m, ...], ...], one list per usable wall.
    If no keepouts are present, returns [[wall_length], ...] for each usable wall.
    """
    usable = _usable_walls(floorplan)
    if not usable:
        return []

    raw_keepouts = floorplan.get("keepouts")
    if not isinstance(raw_keepouts, list) or not raw_keepouts:
        raw_keepouts = floorplan.get("room", {}).get("keepouts", [])

    keepouts: list[dict] = []
    if isinstance(raw_keepouts, list):
        for k in raw_keepouts:
            if isinstance(k, dict) and k.get("full_height") is not False:
                poly = k.get("polygon")
                if poly and len(poly) >= 3:
                    keepouts.append(k)

    if not keepouts:
        out: list[list[float]] = []
        for _, _, computed, wall in usable:
            nom_len = _wall_length(wall, computed)
            if nom_len >= _LENGTH_EPS:
                out.append([nom_len])
        return out

    kp_polys: list[Polygon] = []
    for k in keepouts:
        try:
            poly_coords = k.get("polygon")
            if poly_coords and len(poly_coords) >= 3:
                p_obj = Polygon(poly_coords)
                if not p_obj.is_valid:
                    p_obj = make_valid(p_obj)
                if not p_obj.is_empty:
                    kp_polys.extend(_extract_polygons(p_obj))
        except Exception:
            continue

    vertices = floorplan.get("vertices") if isinstance(floorplan.get("vertices"), dict) else {}
    c_boxes = _get_corner_boxes(vertices, usable, keepouts)
    kp_union = unary_union(kp_polys + c_boxes)
    if not kp_union.is_valid:
        kp_union = make_valid(kp_union)
    if kp_union.is_empty:
        out = []
        for _, _, computed, wall in usable:
            nom_len = _wall_length(wall, computed)
            if nom_len >= _LENGTH_EPS:
                out.append([nom_len])
        return out

    result: list[list[float]] = []
    for p, q, computed, wall in usable:
        nom_len = _wall_length(wall, computed)
        if computed < _LENGTH_EPS:
            if nom_len >= _LENGTH_EPS:
                result.append([nom_len])
            else:
                result.append([])
            continue

        p_pt = np.asarray(p, dtype=float)
        q_pt = np.asarray(q, dtype=float)
        u = (q_pt - p_pt) / computed
        line = LineString([p_pt, q_pt])
        band = line.buffer(band_depth, cap_style="flat")
        inter = kp_union.intersection(band)
        if inter.is_empty:
            result.append([nom_len])
            continue

        geoms = _extract_polygons(inter)
        intervals: list[tuple[float, float]] = []
        for g in geoms:
            coords = np.asarray(g.exterior.coords, dtype=float)
            s = (coords - p_pt) @ u
            s0 = max(0.0, float(s.min()))
            s1 = min(computed, float(s.max()))
            if s1 > s0 + min_seg_len:
                intervals.append((s0, s1))

        if not intervals:
            result.append([nom_len])
            continue

        intervals.sort()
        merged: list[tuple[float, float]] = [intervals[0]]
        for cur in intervals[1:]:
            if cur[0] <= merged[-1][1] + min_seg_len:
                merged[-1] = (merged[-1][0], max(merged[-1][1], cur[1]))
            else:
                merged.append(cur)

        # Snap near corner
        snapped_list: list[tuple[float, float]] = []
        for s0, s1 in merged:
            if s0 <= corner_snap_tol:
                s0 = 0.0
            if (computed - s1) <= corner_snap_tol:
                s1 = computed
            snapped_list.append((s0, s1))

        # Re-merge after snap
        snapped: list[tuple[float, float]] = [snapped_list[0]]
        for cur in snapped_list[1:]:
            if cur[0] <= snapped[-1][1] + min_seg_len:
                snapped[-1] = (snapped[-1][0], max(snapped[-1][1], cur[1]))
            else:
                snapped.append(cur)

        # Unobstructed segments
        unob: list[float] = []
        cur_s = 0.0
        scale = nom_len / computed if computed > 0 else 1.0
        for s0, s1 in snapped:
            seg = s0 - cur_s
            if seg > min_seg_len:
                unob.append(float(round(seg * scale, 3)))
            cur_s = max(cur_s, s1)
        seg = computed - cur_s
        if seg > min_seg_len:
            unob.append(float(round(seg * scale, 3)))
        result.append(unob)

    return result


def _wallpaper_number_of_drops(
    floorplan: dict, width_m: float, overlap_m: float
) -> int:
    w_eff = width_m - overlap_m
    if not math.isfinite(w_eff) or w_eff <= 0:
        raise MaterialEstimationError(
            detail="wallpaper_width_m must be greater than wallpaper_overlap_m"
        )
    usable = _usable_walls(floorplan)
    if usable:
        wall_segments = get_unobstructed_wall_segments(floorplan)
        flat_lengths = [l for segs in wall_segments for l in segs if l >= _LENGTH_EPS]
        if flat_lengths:
            return int(
                sum(
                    max(1, math.ceil(max(0.0, length - overlap_m) / w_eff))
                    for length in flat_lengths
                )
            )
        # All usable walls are completely obstructed by full-height wardrobes
        return 0

    room = floorplan.get("room") if isinstance(floorplan.get("room"), dict) else {}
    try:
        perimeter = float(room.get("perimeter_m") or 0.0)
    except (TypeError, ValueError):
        perimeter = 0.0
    if not math.isfinite(perimeter) or perimeter < _LENGTH_EPS:
        raise MaterialEstimationError(
            detail="Floorplan has no walls or perimeter for wallpaper drops"
        )
    raw_keepouts = floorplan.get("keepouts")
    if not isinstance(raw_keepouts, list) or not raw_keepouts:
        raw_keepouts = room.get("keepouts", [])
    if isinstance(raw_keepouts, list):
        kp_len = sum(
            float(k.get("length_m", 0.0) or 0.0)
            for k in raw_keepouts
            if isinstance(k, dict) and k.get("full_height") is not False
        )
        perimeter = max(0.0, perimeter - kp_len)
    if perimeter < _LENGTH_EPS:
        return 0
    return int(max(1, math.ceil(max(0.0, perimeter - overlap_m) / w_eff)))


def _height_m(floorplan: dict, trim_m: float) -> float:
    room = floorplan.get("room") if isinstance(floorplan.get("room"), dict) else {}
    try:
        height = float(room.get("height_meters") or 0.0)
    except (TypeError, ValueError):
        height = 0.0
    if not (math.isfinite(height) and math.isfinite(trim_m)) or height + trim_m <= 0:
        raise MaterialEstimationError(detail="Wallpaper drop length is non-positive")
    return height


def _wallpaper_result(floorplan: dict, parameters: dict[str, Any]) -> dict[str, Any]:
    try:
        width_m = float(parameters["wallpaper_width_m"])
        overlap_m = float(parameters["wallpaper_overlap_m"])
        trim_m = float(parameters["wallpaper_trim_m"])
        roll_length_m = float(parameters["wallpaper_roll_length_m"])
        repeat_m = float(parameters["wallpaper_repeat_m"])
    except (ValueError, TypeError) as exc:
        raise MaterialEstimationError(f"Invalid numeric parameter: {exc}") from exc
    if not all(
        math.isfinite(v)
        for v in (width_m, overlap_m, trim_m, roll_length_m, repeat_m)
    ):
        raise MaterialEstimationError(detail="Wallpaper parameters must be finite numbers")
    match_type = _match_type(parameters, repeat_m)
    height_m = _height_m(floorplan, trim_m)
    number_of_drops = _wallpaper_number_of_drops(floorplan, width_m, overlap_m)
    try:
        calc = calculate_wallpaper_requirements(
            height_m=height_m,
            number_of_drops=number_of_drops,
            roll_length_m=roll_length_m,
            trim_m=trim_m,
            repeat_m=repeat_m,
            match_type=match_type,
        )
    except ValueError as exc:
        raise MaterialEstimationError(detail=str(exc)) from exc
    return {
        "roll_width_m": width_m,
        "roll_length_m": roll_length_m,
        "trim_m": trim_m,
        "overlap_margin_m": overlap_m,
        "repeat_m": repeat_m,
        "match_type": match_type,
        "height_m": height_m,
        "effective_width_m": width_m - overlap_m,
        "number_of_drops": number_of_drops,
        "drop_length_m": calc["drop_length_m"],
        "drops_per_roll": calc["drops_per_roll"],
        "calculated_rolls": calc["calculated_rolls"],
    }


def _param_float(params: dict[str, Any], key: str, default: float) -> float:
    val = params.get(key)
    if val is None:
        return default
    return float(val)


def _tatami_result(floorplan: dict, parameters: dict[str, Any]) -> dict[str, Any]:
    room = floorplan.get("room") if isinstance(floorplan.get("room"), dict) else {}
    walls = floorplan.get("walls") if isinstance(floorplan.get("walls"), list) else []
    if "width_m" in room or "depth_m" in room or len(walls) <= 4:
        try:
            width = float(room.get("width_m") or 0.0)
            depth = float(room.get("depth_m") or 0.0)
        except (TypeError, ValueError) as exc:
            raise MaterialEstimationError(
                detail="Floorplan room width and depth must be positive finite numbers"
            ) from exc
        if not (math.isfinite(width) and width > 0 and math.isfinite(depth) and depth > 0):
            raise MaterialEstimationError(
                detail="Floorplan room width and depth must be positive finite numbers"
            )
    try:
        if calculate_tatami_layout is not _orig_calculate_tatami_layout:
            tapers = derive_edge_tapers(floorplan)
            return calculate_tatami_layout(
                room_width_m=width,
                room_depth_m=depth,
                standard=parameters["tatami_standard"],
                layout_type=parameters["tatami_layout_type"],
                edge_tapers=tapers,
            )
        return partition_l_tatami(
            floorplan,
            standard=parameters["tatami_standard"],
            layout_type=parameters["tatami_layout_type"],
        )
    except (TatamiShavingLimitExceededError, TatamiSkewAngleExceededError):
        raise
    except (ValueError, ZeroDivisionError) as exc:
        raise MaterialEstimationError(str(exc)) from exc


def estimate_materials_for_session(
    floorplan_data: dict,
    parameters: dict[str, Any],
    allow_partial: bool = False,
    keys: set[str] | str = "all",
) -> dict[str, Any]:
    if not isinstance(floorplan_data, dict):
        raise MaterialEstimationError(detail="Floorplan is not available for this session")
    if not isinstance(parameters, dict):
        raise MaterialEstimationError(detail="Missing required parameter: parameters")
    for key in _REQUIRED_KEYS:
        if key not in parameters:
            raise MaterialEstimationError(detail=f"Missing required parameter: {key}")

    all_keys = {"wallpaper", "tatami", "neda", "tiling", "plywood", "cf"}
    if keys == "all":
        target_keys = all_keys
    elif isinstance(keys, (set, list, tuple)):
        target_keys = set(keys)
    else:
        target_keys = {str(keys)}

    if not allow_partial:
        wallpaper = (
            _wallpaper_result(floorplan_data, parameters)
            if "wallpaper" in target_keys
            else None
        )
        tatami = (
            _tatami_result(floorplan_data, parameters)
            if "tatami" in target_keys
            else None
        )
        metrics_mm = None
        if target_keys.intersection({"neda", "tiling", "plywood", "cf"}):
            metrics_mm = layout_to_metrics_mm(floorplan_data)

        neda = None
        if "neda" in target_keys and metrics_mm is not None:
            neda = estimate_neda(
                metrics_mm,
                joist_pitch_mm=_param_float(parameters, "joist_pitch_mm", 303.0),
                border_width_mm=_param_float(parameters, "border_width_mm", 30.0),
                waste_rate=_param_float(parameters, "waste_rate", 0.0),
            )
        tiling = None
        if "tiling" in target_keys and metrics_mm is not None:
            tiling = estimate_tiling(
                metrics_mm,
                tile_length_mm=_param_float(parameters, "tile_length_mm", 300.0),
                tile_width_mm=_param_float(parameters, "tile_width_mm", 300.0),
                tile_joint_width_mm=_param_float(parameters, "tile_joint_width_mm", 3.0),
            )
        plywood = None
        if "plywood" in target_keys and metrics_mm is not None:
            plywood = estimate_plywood(
                metrics_mm,
                board_length_mm=_param_float(parameters, "board_length_mm", 1820.0),
                board_width_mm=_param_float(parameters, "board_width_mm", 910.0),
                waste_rate=_param_float(parameters, "waste_rate", 0.0),
            )
        cf = None
        if "cf" in target_keys and metrics_mm is not None:
            cf = estimate_cf(
                metrics_mm,
                cf_roll_width_mm=_param_float(parameters, "cf_roll_width_mm", 940.0),
                waste_rate=_param_float(parameters, "waste_rate", 0.0),
            )
        return {
            "wallpaper": wallpaper,
            "tatami": tatami,
            "neda": neda,
            "tiling": tiling,
            "plywood": plywood,
            "cf": cf,
        }

    warnings: list[str] = []
    out: dict[str, Any] = {
        "wallpaper": None,
        "tatami": None,
        "neda": None,
        "tiling": None,
        "plywood": None,
        "cf": None,
    }

    if "wallpaper" in target_keys:
        try:
            out["wallpaper"] = _wallpaper_result(floorplan_data, parameters)
        except Exception as exc:
            warnings.append(str(exc) or exc.__class__.__name__)

    if "tatami" in target_keys:
        try:
            out["tatami"] = _tatami_result(floorplan_data, parameters)
        except Exception as exc:
            warnings.append(str(exc) or exc.__class__.__name__)

    if target_keys.intersection({"neda", "tiling", "plywood", "cf"}):
        metrics_mm = None
        try:
            metrics_mm = layout_to_metrics_mm(floorplan_data)
        except Exception as exc:
            warnings.append(str(exc) or exc.__class__.__name__)

        if metrics_mm is not None:
            if "neda" in target_keys:
                try:
                    out["neda"] = estimate_neda(
                        metrics_mm,
                        joist_pitch_mm=_param_float(parameters, "joist_pitch_mm", 303.0),
                        border_width_mm=_param_float(parameters, "border_width_mm", 30.0),
                        waste_rate=_param_float(parameters, "waste_rate", 0.0),
                    )
                except Exception as exc:
                    warnings.append(str(exc) or exc.__class__.__name__)

            if "tiling" in target_keys:
                try:
                    out["tiling"] = estimate_tiling(
                        metrics_mm,
                        tile_length_mm=_param_float(parameters, "tile_length_mm", 300.0),
                        tile_width_mm=_param_float(parameters, "tile_width_mm", 300.0),
                        tile_joint_width_mm=_param_float(parameters, "tile_joint_width_mm", 3.0),
                    )
                except Exception as exc:
                    warnings.append(str(exc) or exc.__class__.__name__)

            if "plywood" in target_keys:
                try:
                    out["plywood"] = estimate_plywood(
                        metrics_mm,
                        board_length_mm=_param_float(parameters, "board_length_mm", 1820.0),
                        board_width_mm=_param_float(parameters, "board_width_mm", 910.0),
                        waste_rate=_param_float(parameters, "waste_rate", 0.0),
                    )
                except Exception as exc:
                    warnings.append(str(exc) or exc.__class__.__name__)

            if "cf" in target_keys:
                try:
                    out["cf"] = estimate_cf(
                        metrics_mm,
                        cf_roll_width_mm=_param_float(parameters, "cf_roll_width_mm", 940.0),
                        waste_rate=_param_float(parameters, "waste_rate", 0.0),
                    )
                except Exception as exc:
                    warnings.append(str(exc) or exc.__class__.__name__)

    if warnings:
        out["warning"] = "; ".join(warnings)

    return out
