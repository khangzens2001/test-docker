from __future__ import annotations

import math
from typing import Any

from app.core.exceptions import (
    TatamiShavingLimitExceededError,
    TatamiSkewAngleExceededError,
)
from app.services.tatami_constants import (
    HARD_LIMIT_MM,
    SKEW_LIMIT_DEG,
    SOFT_LIMIT_MM,
    TATAMI_STANDARDS,
)

__all__ = [
    "HARD_LIMIT_MM",
    "SKEW_LIMIT_DEG",
    "SOFT_LIMIT_MM",
    "TATAMI_STANDARDS",
    "calculate_tatami_layout",
    "derive_edge_tapers",
    "has_four_way_plus",
    "_tile_shugikyo_ring",
]

_VALID_LAYOUTS = {"Shugikyo", "Fushugikyo", "Ryukyu"}
_EDGES = ("north", "south", "west", "east")


def has_four_way_plus(mats: list[dict], nx: int, ny: int) -> bool:
    """True if any internal grid vertex is shared by 4 distinct mat IDs."""
    owner: dict[tuple[int, int], object] = {}
    for idx, mat in enumerate(mats):
        mid = mat.get("mat_id", idx)
        for cell in mat["cells"]:
            i, j = int(cell[0]), int(cell[1])
            owner[(i, j)] = mid
    for p in range(1, nx):
        for q in range(1, ny):
            ids = [
                owner.get((p - 1, q - 1)),
                owner.get((p, q - 1)),
                owner.get((p - 1, q)),
                owner.get((p, q)),
            ]
            if any(v is None for v in ids):
                continue
            if len(set(ids)) == 4:
                return True
    return False


def _snap_grid(
    room_width_m: float,
    room_depth_m: float,
    unit: float,
    grid: tuple[int, int] | None,
) -> tuple[int, int]:
    if grid is None:
        nx = max(1, int(round(room_width_m / unit)))
        ny = max(1, int(round(room_depth_m / unit)))
        return nx, ny
    if not isinstance(grid, tuple) or len(grid) != 2:
        raise ValueError("grid must be a tuple (nx, ny)")
    nx, ny = grid
    if type(nx) is not int or type(ny) is not int or nx < 1 or ny < 1:
        raise ValueError("grid nx and ny must be ints >= 1")
    return nx, ny


def _tile_ryukyu(nx: int, ny: int) -> list[dict]:
    mats: list[dict] = []
    for j in range(ny):
        for i in range(nx):
            deg = 0 if (i + j) % 2 == 0 else 90
            mats.append(
                {
                    "kind": "full",
                    "cells": [(i, j)],
                    "orientation": "horizontal" if deg == 0 else "vertical",
                    "orientation_deg": deg,
                }
            )
    return mats


def _sort_and_id(mats: list[dict]) -> list[dict]:
    def key(m: dict) -> tuple[int, int]:
        cells = [tuple(c) for c in m["cells"]]
        js = [c[1] for c in cells]
        is_ = [c[0] for c in cells]
        return (min(js), min(is_))

    ordered = sorted(mats, key=key)
    out = []
    for k, m in enumerate(ordered, start=1):
        rec = dict(m)
        rec["mat_id"] = f"mat_{k}"
        rec["cells"] = [[int(c[0]), int(c[1])] for c in rec["cells"]]
        if "orientation_deg" not in rec:
            rec["orientation_deg"] = 0
            rec["orientation"] = "horizontal"
        out.append(rec)
    return out


SHAVE_WARN = (
    "Shaving offset exceeds 15mm safety threshold. Structural core compromise risk."
)
GAP_WARN = (
    "Gap exceeds 15mm safety threshold. Itadatami (wooden border) recommended."
)


def _column_row_geometry(
    room_width_m: float,
    room_depth_m: float,
    nx: int,
    ny: int,
    unit: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    residual_x_m = nx * unit - room_width_m
    residual_y_m = ny * unit - room_depth_m
    if nx == 1:
        col_w = [room_width_m]
    else:
        col_w = [unit] * nx
        col_w[0] = unit - residual_x_m / 2.0
        col_w[-1] = unit - residual_x_m / 2.0
    if ny == 1:
        row_h = [room_depth_m]
    else:
        row_h = [unit] * ny
        row_h[0] = unit - residual_y_m / 2.0
        row_h[-1] = unit - residual_y_m / 2.0
    if any(w <= 0 for w in col_w) or any(h <= 0 for h in row_h):
        raise ValueError("Computed non-positive column or row dimension")
    x_orig = [0.0]
    for w in col_w[:-1]:
        x_orig.append(x_orig[-1] + w)
    y_orig = [0.0]
    for h in row_h[:-1]:
        y_orig.append(y_orig[-1] + h)
    return col_w, row_h, x_orig, y_orig


def _delta_on_edge(
    spec: dict[str, float], t: float, L: float
) -> float:
    start = spec["shave_start_mm"]
    end = spec["shave_end_mm"]
    if L < 1e-9:
        return start
    return start + (t / L) * (end - start)


def _warning_for(start: float, end: float) -> str | None:
    peak = max(abs(start), abs(end))
    if peak <= SOFT_LIMIT_MM:
        return None
    if peak <= HARD_LIMIT_MM:
        if start >= 0 and end >= 0:
            return SHAVE_WARN
        if start <= 0 and end <= 0:
            return GAP_WARN
        return SHAVE_WARN
    return None


def _compute_mat_shaving(
    mats: list[dict],
    nx: int,
    ny: int,
    tapers: dict[str, dict[str, float]],
    room_width_m: float,
    room_depth_m: float,
    col_w: list[float],
    row_h: list[float],
    x_orig: list[float],
    y_orig: list[float],
) -> list[dict]:
    details: list[dict] = []
    lengths = {
        "south": room_width_m,
        "north": room_width_m,
        "west": room_depth_m,
        "east": room_depth_m,
    }
    for m in mats:
        cells = [(int(c[0]), int(c[1])) for c in m["cells"]]
        is_ = [c[0] for c in cells]
        js = [c[1] for c in cells]
        i0, i1 = min(is_), max(is_)
        j0, j1 = min(js), max(js)
        x0 = x_orig[i0]
        x1 = x_orig[i1] + col_w[i1]
        y0 = y_orig[j0]
        y1 = y_orig[j1] + row_h[j1]
        touches = []
        if any(i == 0 for i, _j in cells):
            touches.append(("west", y0, y1))
        if any(i == nx - 1 for i, _j in cells):
            touches.append(("east", y0, y1))
        if any(j == 0 for _i, j in cells):
            touches.append(("south", x0, x1))
        if any(j == ny - 1 for _i, j in cells):
            touches.append(("north", x0, x1))
        for edge, a, b in touches:
            spec = tapers[edge]
            L = lengths[edge]
            da = _delta_on_edge(spec, a, L)
            db = _delta_on_edge(spec, b, L)
            if max(abs(da), abs(db)) < 0.05:
                continue
            da_r = round(da, 1)
            db_r = round(db, 1)
            details.append(
                {
                    "mat_id": m["mat_id"],
                    "edge": edge,
                    "shave_start_mm": da_r,
                    "shave_end_mm": db_r,
                    "warning": _warning_for(da_r, db_r),
                }
            )
    return details


def _placements_to_result(
    layout_type: str,
    standard: str,
    unit: float,
    nx: int,
    ny: int,
    room_width_m: float,
    room_depth_m: float,
    mats: list[dict],
    custom_shaving_details: list[dict],
    col_w: list[float],
    row_h: list[float],
    x_orig: list[float],
    y_orig: list[float],
) -> dict:
    full = sum(1 for m in mats if m["kind"] == "full")
    half = sum(1 for m in mats if m["kind"] == "half")
    placed = []
    for m in mats:
        cells = [(int(c[0]), int(c[1])) for c in m["cells"]]
        is_ = [c[0] for c in cells]
        js = [c[1] for c in cells]
        i0, i1 = min(is_), max(is_)
        j0, j1 = min(js), max(js)
        x = x_orig[i0]
        y = y_orig[j0]
        width = sum(col_w[i] for i in range(i0, i1 + 1))
        depth = sum(row_h[j] for j in range(j0, j1 + 1))
        rec = dict(m)
        rec["cells"] = [[i, j] for i, j in cells]
        rec["x_m"] = round(x, 4)
        rec["y_m"] = round(y, 4)
        rec["width_m"] = round(width, 4)
        rec["depth_m"] = round(depth, 4)
        placed.append(rec)
    return {
        "layout_type": layout_type,
        "regional_standard": standard,
        "full_mats_count": full,
        "half_mats_count": half,
        "unit_m": round(unit, 4),
        "grid": [nx, ny],
        "room_width_m": round(room_width_m, 4),
        "room_depth_m": round(room_depth_m, 4),
        "has_shaving_warning": any(d.get("warning") for d in custom_shaving_details),
        "mats": placed,
        "custom_shaving_details": custom_shaving_details,
    }


def _tile_corridor(nx: int, ny: int) -> list[dict]:
    mats: list[dict] = []
    if nx == 1:
        j = 0
        while j + 1 < ny:
            mats.append({"kind": "full", "cells": [(0, j), (0, j + 1)]})
            j += 2
        if j < ny:
            mats.append({"kind": "half", "cells": [(0, ny - 1)]})
        return mats
    i = 0
    while i + 1 < nx:
        mats.append({"kind": "full", "cells": [(i, 0), (i + 1, 0)]})
        i += 2
    if i < nx:
        mats.append({"kind": "half", "cells": [(nx - 1, 0)]})
    return mats


def _full(a: tuple[int, int], b: tuple[int, int]) -> dict:
    return {"kind": "full", "cells": [a, b]}


def _half(a: tuple[int, int]) -> dict:
    return {"kind": "half", "cells": [a]}


_SHUGIKYO_REGISTRY: dict[tuple[int, int], list[dict]] = {
    (3, 3): [
        _full((0, 0), (1, 0)),
        _full((2, 0), (2, 1)),
        _full((1, 2), (2, 2)),
        _full((0, 1), (0, 2)),
        _half((1, 1)),
    ],
    (3, 4): [
        _full((0, 0), (1, 0)),
        _full((2, 0), (2, 1)),
        _full((2, 2), (2, 3)),
        _full((0, 3), (1, 3)),
        _full((0, 1), (0, 2)),
        _full((1, 1), (1, 2)),
    ],
    (4, 3): [
        _full((0, 0), (1, 0)),
        _full((2, 0), (3, 0)),
        _full((3, 1), (3, 2)),
        _full((1, 2), (2, 2)),
        _full((0, 1), (0, 2)),
        _full((1, 1), (2, 1)),
    ],
    (4, 4): [
        _full((0, 0), (1, 0)),
        _full((2, 0), (3, 0)),
        _full((3, 1), (3, 2)),
        _full((2, 3), (3, 3)),
        _full((0, 3), (1, 3)),
        _full((0, 1), (0, 2)),
        _full((1, 1), (2, 1)),
        _full((1, 2), (2, 2)),
    ],
    (4, 5): [
        _full((0, 0), (1, 0)),
        _full((2, 0), (3, 0)),
        _full((3, 1), (3, 2)),
        _full((3, 3), (3, 4)),
        _full((1, 4), (2, 4)),
        _full((0, 3), (0, 4)),
        _full((0, 1), (0, 2)),
        _full((1, 1), (2, 1)),
        _full((1, 2), (2, 2)),
        _full((1, 3), (2, 3)),
    ],
    (4, 6): [
        _full((0, 0), (1, 0)),
        _full((2, 0), (3, 0)),
        _full((3, 1), (3, 2)),
        _full((3, 3), (3, 4)),
        _full((2, 5), (3, 5)),
        _full((0, 5), (1, 5)),
        _full((0, 3), (0, 4)),
        _full((0, 1), (0, 2)),
        _full((1, 1), (2, 1)),
        _full((1, 2), (2, 2)),
        _full((1, 3), (2, 3)),
        _full((1, 4), (2, 4)),
    ],
}


def _assign_orientations(mats: list[dict]) -> list[dict]:
    out = []
    for m in mats:
        rec = dict(m)
        cells = [(int(c[0]), int(c[1])) for c in rec["cells"]]
        rec["cells"] = cells
        if rec["kind"] == "half" or len(cells) == 1:
            rec["orientation"] = "horizontal"
            rec["orientation_deg"] = 0
        else:
            if cells[0][1] == cells[1][1]:
                rec["orientation"] = "horizontal"
                rec["orientation_deg"] = 0
            else:
                rec["orientation"] = "vertical"
                rec["orientation_deg"] = 90
        out.append(rec)
    return out


def _majority_hanjo_cell(nx: int, ny: int) -> tuple[int, int]:
    k, m = nx // 2, ny // 2
    if (k + m) % 2 == 0:
        return k, m
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    best: tuple | None = None
    winner = (0, 0)
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 != 0:
                continue
            dist = ((i - cx) ** 2 + (j - cy) ** 2) ** 0.5
            border = min(i, nx - 1 - i, j, ny - 1 - j)
            key = (dist, -border, j, i)
            if best is None or key < best:
                best = key
                winner = (i, j)
    return winner


def _tile_shugikyo_ring(nx: int, ny: int) -> list[dict]:
    """Pure §7.4 fallback. Does not read `_SHUGIKYO_REGISTRY`."""
    if nx < 1 or ny < 1:
        raise ValueError("grid nx and ny must be ints >= 1")
    if min(nx, ny) == 1:
        return _assign_orientations(_tile_corridor(nx, ny))

    occupied: dict[tuple[int, int], int] = {}
    mats: list[dict] = []

    def is_free(i: int, j: int, i0: int, j0: int, w: int, h: int) -> bool:
        return (
            i0 <= i < i0 + w
            and j0 <= j < j0 + h
            and (i, j) not in occupied
        )

    def place(cells: list[tuple[int, int]], kind: str) -> None:
        mats.append({"kind": kind, "cells": cells})
        idx = len(mats) - 1
        for c in cells:
            occupied[c] = idx

    if (nx * ny) % 2 == 1:
        hi, hj = _majority_hanjo_cell(nx, ny)
        place([(hi, hj)], "half")

    def fill_rect(i0: int, j0: int, w: int, h: int) -> None:
        if w < 1 or h < 1:
            return

        def free(i: int, j: int) -> bool:
            return is_free(i, j, i0, j0, w, h)

        if w == 1:
            j = j0
            while j < j0 + h:
                if not free(i0, j):
                    j += 1
                    continue
                if j + 1 < j0 + h and free(i0, j + 1):
                    place([(i0, j), (i0, j + 1)], "full")
                    j += 2
                else:
                    j += 1
            return
        if h == 1:
            i = i0
            while i < i0 + w:
                if not free(i, j0):
                    i += 1
                    continue
                if i + 1 < i0 + w and free(i + 1, j0):
                    place([(i, j0), (i + 1, j0)], "full")
                    i += 2
                else:
                    i += 1
            return
        if w == 2 and h == 2:
            if free(i0, j0) and free(i0 + 1, j0):
                place([(i0, j0), (i0 + 1, j0)], "full")
            if free(i0, j0 + 1) and free(i0 + 1, j0 + 1):
                place([(i0, j0 + 1), (i0 + 1, j0 + 1)], "full")
            return
        if w == 2 and h > 2:
            for j in range(j0, j0 + h):
                if free(i0, j) and free(i0 + 1, j):
                    place([(i0, j), (i0 + 1, j)], "full")
            return
        if w > 2 and h == 2:
            for i in range(i0, i0 + w):
                if free(i, j0) and free(i, j0 + 1):
                    place([(i, j0), (i, j0 + 1)], "full")
            return

        i = i0
        while i < i0 + w:
            if not free(i, j0):
                i += 1
                continue
            if i + 1 < i0 + w and free(i + 1, j0):
                place([(i, j0), (i + 1, j0)], "full")
                i += 2
                continue
            if i == i0 + w - 1 and free(i, j0 + 1):
                place([(i, j0), (i, j0 + 1)], "full")
            i += 1

        j = j0
        while j < j0 + h:
            ei = i0 + w - 1
            if not free(ei, j):
                j += 1
                continue
            if j + 1 < j0 + h and free(ei, j + 1):
                place([(ei, j), (ei, j + 1)], "full")
                j += 2
                continue
            if j == j0 + h - 1 and free(ei - 1, j):
                place([(ei, j), (ei - 1, j)], "full")
            j += 1

        i = i0 + w - 1
        nj = j0 + h - 1
        while i >= i0:
            if not free(i, nj):
                i -= 1
                continue
            if i - 1 >= i0 and free(i - 1, nj):
                place([(i, nj), (i - 1, nj)], "full")
                i -= 2
                continue
            if i == i0 and free(i, nj - 1):
                place([(i, nj), (i, nj - 1)], "full")
            i -= 1

        j = j0 + h - 1
        while j >= j0:
            if not free(i0, j):
                j -= 1
                continue
            if j - 1 >= j0 and free(i0, j - 1):
                place([(i0, j), (i0, j - 1)], "full")
                j -= 2
                continue
            if free(i0 + 1, j):
                place([(i0, j), (i0 + 1, j)], "full")
            j -= 1

        fill_rect(i0 + 1, j0 + 1, w - 2, h - 2)

    fill_rect(0, 0, nx, ny)

    covered = list(occupied.keys())
    if len(covered) != nx * ny or len(set(covered)) != nx * ny:
        raise ValueError("Tiling incomplete: uncovered cells remaining")
    oriented = _assign_orientations(mats)
    if has_four_way_plus(oriented, nx, ny):
        if (nx, ny) in _SHUGIKYO_REGISTRY:
            return oriented
        hi, hj = _majority_hanjo_cell(nx, ny) if (nx * ny) % 2 == 1 else (None, None)
        hanjo = (hi, hj) if hi is not None and hj is not None else None
        sol = _solve_zero_plus(nx, ny, hanjo)
        if sol is not None:
            return sol
        raise ValueError("Shugikyo layout invalid: four-way '+' intersection detected")
    return oriented


def _solve_zero_plus(nx: int, ny: int, hanjo: tuple[int, int] | None) -> list[dict] | None:
    cells = [(i, j) for j in range(ny) for i in range(nx)]
    dominoes = []
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                dominoes.append([(i, j), (i + 1, j)])
            if j + 1 < ny:
                dominoes.append([(i, j), (i, j + 1)])
    rem_cells = [c for c in cells if c != hanjo]

    def solve(free_cells: list[tuple[int, int]], current_mats: list[list[tuple[int, int]]]) -> list[dict] | None:
        if not free_cells:
            mats = []
            if hanjo is not None:
                mats.append({"kind": "half", "cells": [hanjo]})
            for m in current_mats:
                mats.append({"kind": "full", "cells": m})
            oriented = _assign_orientations(mats)
            if not has_four_way_plus(oriented, nx, ny):
                return oriented
            return None
        first = free_cells[0]
        for d in dominoes:
            if first in d and all(c in free_cells for c in d):
                next_free = [c for c in free_cells if c not in d]
                res = solve(next_free, current_mats + [d])
                if res is not None:
                    return res
        return None

    return solve(rem_cells, [])



def _tile_shugikyo(nx: int, ny: int) -> list[dict]:
    if min(nx, ny) == 1:
        return _assign_orientations(_tile_corridor(nx, ny))
    key = (nx, ny)
    if key in _SHUGIKYO_REGISTRY:
        cloned = [
            {"kind": m["kind"], "cells": list(m["cells"])}
            for m in _SHUGIKYO_REGISTRY[key]
        ]
        return _assign_orientations(cloned)
    return _tile_shugikyo_ring(nx, ny)


def _tile_fushugikyo(nx: int, ny: int) -> list[dict]:
    mats: list[dict] = []
    if ny % 2 == 0:
        for i in range(nx):
            for k in range(0, ny, 2):
                mats.append({"kind": "full", "cells": [(i, k), (i, k + 1)]})
        return _assign_orientations(mats)
    if nx % 2 == 0:
        for j in range(ny):
            for k in range(0, nx, 2):
                mats.append({"kind": "full", "cells": [(k, j), (k + 1, j)]})
        return _assign_orientations(mats)
    mats.append({"kind": "half", "cells": [(0, 0)]})
    j = 1
    while j + 1 < ny:
        mats.append({"kind": "full", "cells": [(0, j), (0, j + 1)]})
        j += 2
    for row in range(ny):
        for k in range(1, nx, 2):
            mats.append({"kind": "full", "cells": [(k, row), (k + 1, row)]})
    return _assign_orientations(mats)


def _normalize_tapers(
    edge_tapers: dict[str, dict] | list[dict] | None,
) -> dict[str, dict[str, float]]:
    out = {e: {"shave_start_mm": 0.0, "shave_end_mm": 0.0} for e in _EDGES}
    if edge_tapers is None:
        return out
    if isinstance(edge_tapers, list):
        seen: set[str] = set()
        for item in edge_tapers:
            edge = item.get("edge")
            if edge not in _EDGES:
                raise ValueError(f"Unknown edge name: {edge}")
            if edge in seen:
                raise ValueError(f"Duplicate edge taper for edge '{edge}'")
            seen.add(edge)
            start = float(item.get("shave_start_mm", 0.0))
            end = float(item.get("shave_end_mm", 0.0))
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("taper values must be finite")
            out[edge] = {"shave_start_mm": start, "shave_end_mm": end}
        return out
    if isinstance(edge_tapers, dict):
        for edge, spec in edge_tapers.items():
            if edge not in _EDGES:
                raise ValueError(f"Unknown edge name: {edge}")
            spec = spec or {}
            start = float(spec.get("shave_start_mm", 0.0))
            end = float(spec.get("shave_end_mm", 0.0))
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("taper values must be finite")
            out[edge] = {"shave_start_mm": start, "shave_end_mm": end}
        return out
    raise ValueError("edge_tapers must be a dict, list, or None")


def _merge_residual(
    tapers: dict[str, dict[str, float]],
    nx: int,
    ny: int,
    unit: float,
    room_width_m: float,
    room_depth_m: float,
    frozen_edges: frozenset[str] | set[str] = frozenset(),
) -> tuple[dict[str, dict[str, float]], float, float]:
    residual_x_m = nx * unit - room_width_m
    residual_y_m = ny * unit - room_depth_m
    merged = {e: dict(v) for e, v in tapers.items()}

    active_x = [e for e in ("west", "east") if e not in frozen_edges]
    if active_x:
        share_x_mm = (residual_x_m * 1000.0) / len(active_x)
        for e in active_x:
            merged[e]["shave_start_mm"] += share_x_mm
            merged[e]["shave_end_mm"] += share_x_mm

    active_y = [e for e in ("south", "north") if e not in frozen_edges]
    if active_y:
        share_y_mm = (residual_y_m * 1000.0) / len(active_y)
        for e in active_y:
            merged[e]["shave_start_mm"] += share_y_mm
            merged[e]["shave_end_mm"] += share_y_mm

    return merged, residual_x_m, residual_y_m


def _guard_limits(
    tapers: dict[str, dict[str, float]],
    room_width_m: float,
    room_depth_m: float,
) -> None:
    lengths = {
        "south": room_width_m,
        "north": room_width_m,
        "west": room_depth_m,
        "east": room_depth_m,
    }
    for edge, spec in tapers.items():
        start = spec["shave_start_mm"]
        end = spec["shave_end_mm"]
        L = lengths[edge]
        if L >= 1e-9:
            alpha = math.degrees(math.atan(abs(end - start) / (1000.0 * L)))
            if alpha >= SKEW_LIMIT_DEG:
                raise TatamiSkewAngleExceededError(
                    f"Tatami wall skew {alpha:.1f}° exceeds 15° limit on edge {edge}."
                )
    for edge, spec in tapers.items():
        start = spec["shave_start_mm"]
        end = spec["shave_end_mm"]
        peak = max(abs(start), abs(end))
        if peak > HARD_LIMIT_MM:
            if max(start, end) < 0:
                raise TatamiShavingLimitExceededError(
                    f"Tatami gap {peak:.1f} mm exceeds 30 mm hard limit on edge {edge}."
                )
            raise TatamiShavingLimitExceededError(
                f"Tatami shaving {peak:.1f} mm exceeds 30 mm hard limit on edge {edge}."
            )


def calculate_tatami_layout(
    room_width_m: float,
    room_depth_m: float,
    standard: str = "Edoma",
    layout_type: str = "Shugikyo",
    edge_tapers: dict[str, dict] | list[dict] | None = None,
    grid: tuple[int, int] | None = None,
    frozen_edges: frozenset[str] | set[str] = frozenset(),
) -> dict:
    if not math.isfinite(room_width_m) or not math.isfinite(room_depth_m):
        raise ValueError("room dimensions must be finite")
    if room_width_m <= 0 or room_depth_m <= 0:
        raise ValueError("room dimensions must be > 0")
    if standard not in TATAMI_STANDARDS:
        raise ValueError(f"Unknown tatami standard: {standard}")
    if layout_type not in _VALID_LAYOUTS:
        raise ValueError(f"Unknown layout_type: {layout_type}")
    if standard == "Ryukyu" and layout_type != "Ryukyu":
        raise ValueError("Ryukyu standard requires layout_type='Ryukyu'")
    unit = TATAMI_STANDARDS[standard]["width_m"]
    nx, ny = _snap_grid(room_width_m, room_depth_m, unit, grid)
    tapers = _normalize_tapers(edge_tapers)
    tapers, residual_x_m, residual_y_m = _merge_residual(
        tapers, nx, ny, unit, room_width_m, room_depth_m, frozen_edges=frozen_edges
    )
    _guard_limits(tapers, room_width_m, room_depth_m)
    if layout_type == "Ryukyu":
        raw = _tile_ryukyu(nx, ny)
    elif layout_type == "Shugikyo":
        raw = _tile_shugikyo(nx, ny)
    elif layout_type == "Fushugikyo":
        raw = _tile_fushugikyo(nx, ny)
    else:
        raise ValueError(f"Unknown layout_type: {layout_type}")
    mats = _sort_and_id(raw)
    col_w, row_h, x_orig, y_orig = _column_row_geometry(
        room_width_m, room_depth_m, nx, ny, unit
    )
    details = _compute_mat_shaving(
        mats, nx, ny, tapers, room_width_m, room_depth_m, col_w, row_h, x_orig, y_orig
    )
    return _placements_to_result(
        layout_type=layout_type,
        standard=standard,
        unit=unit,
        nx=nx,
        ny=ny,
        room_width_m=room_width_m,
        room_depth_m=room_depth_m,
        mats=mats,
        custom_shaving_details=details,
        col_w=col_w,
        row_h=row_h,
        x_orig=x_orig,
        y_orig=y_orig,
    )


_LENGTH_EPS = 1e-9
_SIDES = ("south", "north", "west", "east")


def _xy(vertices: dict, name: Any) -> tuple[float, float] | None:
    if not isinstance(vertices, dict) or name not in vertices:
        return None
    try:
        pair = vertices[name]
        x, y = float(pair[0]), float(pair[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _usable_walls(
    floorplan: dict,
) -> list[tuple[tuple[float, float], tuple[float, float], float, dict]]:
    vertices = floorplan.get("vertices") if isinstance(floorplan.get("vertices"), dict) else {}
    walls_in = floorplan.get("walls") if isinstance(floorplan.get("walls"), list) else []
    out: list[tuple[tuple[float, float], tuple[float, float], float, dict]] = []
    for wall in walls_in:
        if not isinstance(wall, dict):
            continue
        joints = wall.get("joints") or []
        if not isinstance(joints, (list, tuple)) or len(joints) < 2:
            continue
        p = _xy(vertices, joints[0])
        q = _xy(vertices, joints[1])
        if p is None or q is None:
            continue
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        if length < _LENGTH_EPS:
            continue
        out.append((p, q, length, wall))
    return out


def _vertex_aabb(floorplan: dict) -> tuple[float, float, float, float] | None:
    vertices = floorplan.get("vertices") if isinstance(floorplan.get("vertices"), dict) else {}
    pts = [_xy(vertices, name) for name in vertices]
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if not all(math.isfinite(v) for v in (x_min, x_max, y_min, y_max)):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, x_max, y_min, y_max


def _side_of(
    mid: tuple[float, float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    dists = (
        ("south", abs(mid[1] - y_min)),
        ("north", abs(y_max - mid[1])),
        ("west", abs(mid[0] - x_min)),
        ("east", abs(x_max - mid[0])),
    )
    return min(enumerate(dists), key=lambda item: (item[1][1], item[0]))[1][0]


def _shave_mm(
    side: str,
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> float:
    if side == "south":
        raw = (y - y_min) * 1000.0
    elif side == "north":
        raw = (y_max - y) * 1000.0
    elif side == "west":
        raw = (x - x_min) * 1000.0
    else:
        raw = (x_max - x) * 1000.0
    return round(max(0.0, raw), 1)


def derive_edge_tapers(floorplan_data: dict) -> dict | None:
    usable = _usable_walls(floorplan_data)
    if len(usable) < 4:
        return None
    aabb = _vertex_aabb(floorplan_data)
    if aabb is None:
        return None
    x_min, x_max, y_min, y_max = aabb
    buckets: dict[str, list[tuple[float, float]]] = {side: [] for side in _SIDES}
    for p, q, _, _ in usable:
        mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
        side = _side_of(mid, x_min, x_max, y_min, y_max)
        buckets[side].extend((p, q))
    tapers: dict[str, dict[str, float]] = {}
    for side in _SIDES:
        pts = buckets[side]
        if not pts:
            tapers[side] = {"shave_start_mm": 0.0, "shave_end_mm": 0.0}
            continue
        if side in ("south", "north"):
            start = min(pts, key=lambda t: t[0])
            end = max(pts, key=lambda t: t[0])
        else:
            start = min(pts, key=lambda t: t[1])
            end = max(pts, key=lambda t: t[1])
        tapers[side] = {
            "shave_start_mm": _shave_mm(side, start[0], start[1], x_min, x_max, y_min, y_max),
            "shave_end_mm": _shave_mm(side, end[0], end[1], x_min, x_max, y_min, y_max),
        }
    return tapers

