"""Subfloor plywood estimation and CAD rendering service.

Ported from 3D-Estimate backend/api/japanese_material_calculator.py with
polygon-accurate clipping and 2D Guillotine Bin Packing.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union



# Color palette matching 3D-Estimate architectural theme
WOOD_FULL_A = "#C49A6C"  # Medium oak
WOOD_FULL_B = "#D4A574"  # Light oak
WOOD_CUT = "#B07040"  # Darker walnut for cut pieces
BOARD_EDGE = "#8B7355"  # Edge color
CUT_EDGE = "#704214"  # Edge for cut pieces
FLOOR_FILL = "#F8F4EC"  # Warm floor background
WALL_COLOR = "#1A1A1A"  # Room boundaries
HEADER_COLOR = "#1E293B"  # Primary header text
TEXT_COLOR = "#1E293B"  # Body text
DIM_COLOR = "#2563EB"  # Dimension line blue
SUCCESS_COLOR = "#16A34A"  # Green
WARNING_COLOR = "#D97706"  # Amber


def format_dim(val: float | int | None) -> str:
    """Format dimension to 2 decimal places universally."""
    if val is None:
        return ""
    return f"{float(val):.2f}"


def pack_plywood_boards(
    pieces: list[dict],
    bin_w: float,
    bin_h: float,
    allow_rotation: bool = True,
) -> int:
    """Pack rectangular plywood pieces into standard boards using Guillotine Bin Packing.

    Ported from 3D-Estimate backend/api/japanese_material_calculator.py.
    """
    if not pieces:
        return 0

    sorted_pieces = sorted(
        [(float(p["width_mm"]), float(p["height_mm"])) for p in pieces],
        key=lambda x: x[0] * x[1],
        reverse=True,
    )
    bins: list[list[tuple[float, float, float, float]]] = []

    for p_w, p_h in sorted_pieces:
        placed = False
        for free_rects in bins:
            for r_idx, rect in enumerate(free_rects):
                rx, ry, rw, rh = rect

                orientations = [(p_w, p_h)]
                if allow_rotation and abs(p_w - p_h) > 1e-4:
                    orientations.append((p_h, p_w))

                for fit_w, fit_h in orientations:
                    if fit_w <= rw + 1e-4 and fit_h <= rh + 1e-4:
                        free_rects.pop(r_idx)
                        if rw - fit_w > 1e-4:
                            free_rects.append((rx + fit_w, ry, rw - fit_w, fit_h))
                        if rh - fit_h > 1e-4:
                            free_rects.append((rx, ry + fit_h, rw, rh - fit_h))
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

        if not placed:
            new_bin: list[tuple[float, float, float, float]] = []
            p_w_fit, p_h_fit = p_w, p_h
            if allow_rotation:
                if (p_w > bin_w + 1e-4 or p_h > bin_h + 1e-4) and (
                    p_h <= bin_w + 1e-4 and p_w <= bin_h + 1e-4
                ):
                    p_w_fit, p_h_fit = p_h, p_w

            if bin_w - p_w_fit > 1e-4:
                new_bin.append((p_w_fit, 0.0, bin_w - p_w_fit, p_h_fit))
            if bin_h - p_h_fit > 1e-4:
                new_bin.append((0.0, p_h_fit, bin_w, bin_h - p_h_fit))
            bins.append(new_bin)

    return len(bins)


def _tile_and_clip_orientation(
    room_poly: ShapelyPolygon,
    *,
    orientation: str,
    x_unit_mm: float,
    y_unit_mm: float,
    board_length_mm: float,
    board_width_mm: float,
    add_extra_board: int = 0,
    waste_rate: float = 0.0,
) -> dict[str, Any]:
    """Tile bounding box with board grid, clip each cell to room polygon, and pack."""
    min_x, min_y, max_x, max_y = room_poly.bounds
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    room_area_m2 = round(room_poly.area / 1e6, 4)

    cols = math.ceil(round(bbox_w, 2) / x_unit_mm) if bbox_w > 0 else 0
    rows = math.ceil(round(bbox_h, 2) / y_unit_mm) if bbox_h > 0 else 0

    pieces: list[dict[str, Any]] = []
    has_aabb_overcount = False
    cell_area = x_unit_mm * y_unit_mm

    for r in range(rows):
        y0 = min_y + r * y_unit_mm
        y1 = y0 + y_unit_mm
        for c in range(cols):
            x0 = min_x + c * x_unit_mm
            x1 = x0 + x_unit_mm
            cell_box = shapely_box(x0, y0, x1, y1)
            inter = room_poly.intersection(cell_box)

            if inter.is_empty:
                continue

            if hasattr(inter, "geoms"):
                polys = [g for g in inter.geoms if isinstance(g, ShapelyPolygon) and g.area > 1e-4]
                if not polys:
                    continue
                clipped = unary_union(polys)
            elif isinstance(inter, ShapelyPolygon) and inter.area > 1e-4:
                clipped = inter
            else:
                continue

            clip_area = clipped.area
            if clip_area <= 1e-4:
                continue

            c_min_x, c_min_y, c_max_x, c_max_y = clipped.bounds
            w = round(c_max_x - c_min_x, 2)
            h = round(c_max_y - c_min_y, 2)
            if w <= 1e-4 or h <= 1e-4:
                continue

            # Full vs cut by area ratio >= 0.995 against full sheet area
            is_full = (clip_area / cell_area) >= 0.995
            piece_type = "full" if is_full else "cut"

            # Check if non-rectangular clip causes AABB packing overcount
            aabb_area = w * h
            if aabb_area > 0 and (clip_area / aabb_area) < 0.995:
                has_aabb_overcount = True

            # Extract polygon coordinates for CAD rendering
            poly_coords: list[list[list[float]]] = []
            if isinstance(clipped, ShapelyPolygon):
                poly_coords.append([[float(px), float(py)] for px, py in clipped.exterior.coords])
            elif isinstance(clipped, MultiPolygon):
                for g in clipped.geoms:
                    if isinstance(g, ShapelyPolygon) and g.area > 1e-4:
                        poly_coords.append([[float(px), float(py)] for px, py in g.exterior.coords])

            pieces.append({
                "piece_index": len(pieces) + 1,
                "row": r + 1,
                "col": c + 1,
                "width_mm": w,
                "height_mm": h,
                "area_m2": round((w * h) / 1e6, 4),
                "polygon_area_m2": round(clip_area / 1e6, 4),
                "is_full": is_full,
                "type": piece_type,
                "bounds": [round(c_min_x, 2), round(c_min_y, 2), round(c_max_x, 2), round(c_max_y, 2)],
                "polygon_coords": poly_coords,
            })

    total_boards_optimized = pack_plywood_boards(
        pieces=pieces,
        bin_w=board_length_mm,
        bin_h=board_width_mm,
        allow_rotation=True,
    )
    total_boards_optimized_purchase = (
        math.ceil(total_boards_optimized * (1.0 + float(waste_rate))) + add_extra_board
    )
    total_boards_raw = len(pieces)
    total_boards_purchase = (
        math.ceil(total_boards_raw * (1.0 + float(waste_rate))) + add_extra_board
    )

    board_area_m2 = round((board_length_mm * board_width_mm) / 1e6, 4)
    purchased_area_m2 = round(total_boards_optimized_purchase * board_area_m2, 4)
    waste_area_m2 = round(max(0.0, purchased_area_m2 - room_area_m2), 4)
    waste_percent = (
        round((waste_area_m2 / purchased_area_m2) * 100.0, 2) if purchased_area_m2 > 0 else 0.0
    )

    full_count = sum(1 for p in pieces if p["is_full"])
    cut_count = len(pieces) - full_count

    orientation_label_vi = (
        "Đặt tấm bình thường: cạnh 910 theo chiều dài phòng, cạnh 1820 theo chiều rộng phòng"
        if orientation == "normal"
        else "Xoay tấm: cạnh 1820 theo chiều dài phòng, cạnh 910 theo chiều rộng phòng"
    )

    return {
        "orientation": orientation,
        "orientation_label": "Đặt bình thường" if orientation == "normal" else "Xoay 90 độ",
        "orientation_label_vi": orientation_label_vi,
        "board_size_mm": {
            "length_mm": board_length_mm,
            "width_mm": board_width_mm,
        },
        "x_unit_mm": x_unit_mm,
        "y_unit_mm": y_unit_mm,
        "count_x": cols,
        "count_y": rows,
        "full_boards": full_count,
        "cut_boards": cut_count,
        "total_pieces": len(pieces),
        "total_boards_raw": total_boards_raw,
        "total_boards_purchase": total_boards_purchase,
        "total_boards_optimized": total_boards_optimized,
        "total_boards_optimized_purchase": total_boards_optimized_purchase,
        "waste_rate": float(waste_rate),
        "add_extra_board": add_extra_board,
        "covered_length_mm": round(cols * x_unit_mm, 2),
        "covered_width_mm": round(rows * y_unit_mm, 2),
        "waste_length_mm": round(max(0.0, cols * x_unit_mm - bbox_w), 2),
        "waste_width_mm": round(max(0.0, rows * y_unit_mm - bbox_h), 2),
        "room_area_m2": room_area_m2,
        "covered_area_m2": purchased_area_m2,
        "waste_area_m2": waste_area_m2,
        "waste_percent": waste_percent,
        "aabb_pack_overcount": has_aabb_overcount,
        "pieces": pieces,
        "cut_detail": {
            "pieces": pieces,
        },
        "formulas": {
            "count_x": f"ceil(bbox_width / {x_unit_mm:.0f})",
            "count_y": f"ceil(bbox_depth / {y_unit_mm:.0f})",
            "total_boards_raw": "count_x * count_y (clipped to polygon)",
            "total_boards_optimized": "Guillotine Bin Packing 2D (tận dụng mảnh thừa)",
        },
    }


def estimate_plywood(
    metrics_mm: dict[str, Any],
    *,
    board_length_mm: float = 1820.0,
    board_width_mm: float = 910.0,
    add_extra_board: int = 0,
    waste_rate: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Estimate subfloor plywood boards for room polygon metrics.

    Tries normal (910 along X) and rotated (1820 along X) orientations,
    clips every cell to the room polygon, and packs remaining pieces with
    Guillotine Bin Packing. Selects minimum purchase count, then minimum waste area.
    """
    raw_pts = metrics_mm.get("vertices_mm") or []
    if len(raw_pts) < 3:
        return {
            "orientation": "normal",
            "orientation_label": "Normal",
            "board_size_mm": {"length_mm": board_length_mm, "width_mm": board_width_mm},
            "total_boards_optimized_purchase": 0,
            "total_boards_raw": 0,
            "total_boards_purchase": 0,
            "total_boards_optimized": 0,
            "waste_area_m2": 0.0,
            "waste_rate": float(waste_rate),
            "room_area_m2": 0.0,
            "covered_area_m2": 0.0,
            "full_boards": 0,
            "cut_boards": 0,
            "total_pieces": 0,
            "aabb_pack_overcount": False,
            "pieces": [],
            "options": [],
            "best_option": {},
        }

    pts = np.asarray(raw_pts, dtype=float)
    room_poly = ShapelyPolygon(pts)
    if not room_poly.is_valid:
        room_poly = room_poly.buffer(0)

    if room_poly.is_empty or room_poly.area <= 1e-4:
        return {
            "orientation": "normal",
            "orientation_label": "Normal",
            "board_size_mm": {"length_mm": board_length_mm, "width_mm": board_width_mm},
            "total_boards_optimized_purchase": 0,
            "total_boards_raw": 0,
            "total_boards_purchase": 0,
            "total_boards_optimized": 0,
            "waste_area_m2": 0.0,
            "waste_rate": float(waste_rate),
            "room_area_m2": 0.0,
            "covered_area_m2": 0.0,
            "full_boards": 0,
            "cut_boards": 0,
            "total_pieces": 0,
            "aabb_pack_overcount": False,
            "pieces": [],
            "options": [],
            "best_option": {},
        }

    # Orientation 1: "normal" (cạnh 910 theo X, 1820 theo Y)
    opt_normal = _tile_and_clip_orientation(
        room_poly,
        orientation="normal",
        x_unit_mm=board_width_mm,
        y_unit_mm=board_length_mm,
        board_length_mm=board_length_mm,
        board_width_mm=board_width_mm,
        add_extra_board=add_extra_board,
        waste_rate=waste_rate,
    )

    # Orientation 2: "rotated" (cạnh 1820 theo X, 910 theo Y)
    opt_rotated = _tile_and_clip_orientation(
        room_poly,
        orientation="rotated",
        x_unit_mm=board_length_mm,
        y_unit_mm=board_width_mm,
        board_length_mm=board_length_mm,
        board_width_mm=board_width_mm,
        add_extra_board=add_extra_board,
        waste_rate=waste_rate,
    )

    options = [opt_normal, opt_rotated]

    # Choose smaller total_boards_optimized_purchase, then waste area
    best_option = min(
        options,
        key=lambda opt: (
            opt["total_boards_optimized_purchase"],
            opt["waste_area_m2"],
            opt["total_boards_raw"],
        ),
    )

    result = dict(best_option)
    result["material"] = f"Plywood ({board_width_mm:.0f}x{board_length_mm:.0f})"
    result["options"] = options
    result["best_option"] = best_option
    result["waste_rate"] = float(waste_rate)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────


def _draw_plywood_plan(
    ax_plan: Any,
    corners_mm: np.ndarray,
    pieces: list[dict[str, Any]],
    bb_width: float,
    bb_depth: float,
    board_length_mm: float,
    board_width_mm: float,
) -> None:
    """Draw floor plan with plywood placement and technical annotations."""
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    from matplotlib.path import Path as MPath
    poly_path = MPath(corners_mm, closed=True)
    ax_plan.add_patch(
        patches.PathPatch(poly_path, facecolor=FLOOR_FILL, edgecolor="none", zorder=1)
    )

    for p in pieces:
        is_full = p.get("is_full", False)
        row = p.get("row", 1)
        col = p.get("col", 1)

        if is_full:
            fill_color = WOOD_FULL_A if (row + col) % 2 == 0 else WOOD_FULL_B
            edge_color = BOARD_EDGE
            alpha = 0.88
        else:
            fill_color = WOOD_CUT
            edge_color = CUT_EDGE
            alpha = 0.92

        poly_coords_list = p.get("polygon_coords") or []
        for poly_coords in poly_coords_list:
            if len(poly_coords) >= 3:
                ax_plan.add_patch(
                    patches.Polygon(
                        poly_coords,
                        closed=True,
                        facecolor=fill_color,
                        edgecolor=edge_color,
                        linewidth=0.8,
                        zorder=2,
                        alpha=alpha,
                    )
                )

        # Label dimensions at centroid/center
        bounds = p.get("bounds", [0, 0, 0, 0])
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        w = p.get("width_mm", 0.0)
        h = p.get("height_mm", 0.0)

        if w >= 250.0 and h >= 200.0:
            label_txt = f"{w:.0f}×{h:.0f}"
            if not is_full:
                label_txt = f"CUT\n{label_txt}"
            ax_plan.text(
                cx,
                cy,
                label_txt,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=TEXT_COLOR if is_full else "white",
                linespacing=1.1,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white" if is_full else WOOD_CUT,
                    edgecolor=edge_color,
                    linewidth=0.5,
                    alpha=0.9,
                ),
                zorder=4,
            )

    # Room walls outline
    n_corners = len(corners_mm)
    for i in range(n_corners):
        c1 = corners_mm[i]
        c2 = corners_mm[(i + 1) % n_corners]
        ax_plan.plot(
            [c1[0], c2[0]],
            [c1[1], c2[1]],
            color=WALL_COLOR,
            linewidth=3.5,
            solid_capstyle="butt",
            zorder=5,
        )

    for c in corners_mm:
        ax_plan.plot(c[0], c[1], "o", color="#333333", markersize=5, zorder=6)

    min_x, min_y = corners_mm.min(axis=0)
    max_x, max_y = corners_mm.max(axis=0)
    margin = max(300.0, float(max(bb_width, bb_depth)) * 0.1)

    # Dimension annotation (width at bottom)
    ax_plan.annotate(
        "",
        xy=(min_x, min_y - margin * 0.5),
        xytext=(max_x, min_y - margin * 0.5),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.5),
    )
    ax_plan.text(
        (min_x + max_x) / 2.0,
        min_y - margin * 0.7,
        f"{bb_width:.2f} mm",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=DIM_COLOR,
    )

    # Dimension annotation (depth at right)
    ax_plan.annotate(
        "",
        xy=(max_x + margin * 0.5, min_y),
        xytext=(max_x + margin * 0.5, max_y),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.5),
    )
    ax_plan.text(
        max_x + margin * 0.75,
        (min_y + max_y) / 2.0,
        f"{bb_depth:.2f} mm",
        ha="left",
        va="center",
        rotation=90,
        fontsize=11,
        fontweight="bold",
        color=DIM_COLOR,
    )

    ax_plan.set_aspect("equal")
    ax_plan.grid(True, alpha=0.15, linewidth=0.4, color="#888")
    ax_plan.set_xticklabels([])
    ax_plan.set_yticklabels([])
    ax_plan.tick_params(axis="both", length=0)
    ax_plan.invert_yaxis()
    ax_plan.set_xlim(min_x - margin * 1.2, max_x + margin * 1.2)
    ax_plan.set_ylim(max_y + margin * 1.2, min_y - margin * 1.2)

    legend_elems = [
        patches.Patch(facecolor=FLOOR_FILL, edgecolor=HEADER_COLOR, label="Floor"),
        mlines.Line2D([], [], color=WALL_COLOR, linewidth=3, label="Wall"),
        patches.Patch(
            facecolor=WOOD_FULL_A,
            edgecolor=BOARD_EDGE,
            label=f"Full Board ({board_width_mm:.0f}×{board_length_mm:.0f})",
        ),
        patches.Patch(facecolor=WOOD_CUT, edgecolor=CUT_EDGE, label="Cut Board Piece"),
        mlines.Line2D(
            [],
            [],
            color=DIM_COLOR,
            linewidth=1.5,
            marker="s",
            markersize=5,
            label="BBox Dimension",
        ),
    ]
    leg = ax_plan.legend(
        handles=legend_elems,
        loc="lower left",
        fontsize=10,
        ncol=3,
        framealpha=0.9,
        edgecolor=HEADER_COLOR,
        title="LEGEND",
    )
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_fontsize(11)
    leg.get_frame().set_linewidth(1.0)


def _draw_info_panel(
    ax_info: Any,
    metrics_mm: dict[str, Any],
    plywood: dict[str, Any],
    board_length_mm: float,
    board_width_mm: float,
) -> None:
    """Render technical information panel on the right side."""
    import matplotlib.patches as patches
    ax_info.axis("off")
    ax_info.add_patch(
        patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax_info.transAxes,
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=1.2,
        )
    )

    y = 0.96
    pad_line = 0.038

    def _header(text: str) -> None:
        nonlocal y
        ax_info.text(
            0.05,
            y,
            text,
            fontsize=13,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )
        y -= 0.012
        ax_info.plot(
            [0.05, 0.95],
            [y, y],
            color="#CBD5E1",
            linewidth=1.0,
            transform=ax_info.transAxes,
        )
        y -= pad_line

    def _row(label: str, value: str, val_color: str = TEXT_COLOR, bold: bool = False) -> None:
        nonlocal y
        ax_info.text(0.05, y, label, fontsize=10.5, color="#64748B", transform=ax_info.transAxes)
        weight = "bold" if bold else "normal"
        ax_info.text(
            0.95,
            y,
            value,
            fontsize=10.5,
            color=val_color,
            fontweight=weight,
            ha="right",
            transform=ax_info.transAxes,
        )
        y -= pad_line

    _header("MATERIAL SPECIFICATION")
    _row("Board Standard", "Saburoku-ban (3×6)")
    _row("Dimensions", f"{board_length_mm:.0f} × {board_width_mm:.0f} mm")
    board_area = (board_length_mm * board_width_mm) / 1e6
    _row("Board Area", f"{board_area:.4f} m²")
    y -= 0.015

    _header("ROOM GEOMETRY")
    bb_w = float(metrics_mm.get("bbox_width_mm", 0.0))
    bb_d = float(metrics_mm.get("bbox_depth_mm", 0.0))
    _row("Bounding Width", f"{bb_w:.2f} mm")
    _row("Bounding Depth", f"{bb_d:.2f} mm")
    _row("Floor Area", f"{float(plywood.get('room_area_m2', 0.0)):.2f} m²")
    y -= 0.015

    _header("LAYOUT & OPTIMIZATION")
    orientation = str(plywood.get("orientation", "normal")).upper()
    _row("Selected Orientation", orientation)
    _row("Full Boards", str(plywood.get("full_boards", 0)))
    _row("Cut Pieces", str(plywood.get("cut_boards", 0)))
    _row("Total Pieces", str(plywood.get("total_pieces", 0)))
    _row("Raw Purchase", f"{plywood.get('total_boards_purchase', 0)} pcs")
    _row(
        "Optimized Purchase",
        f"{plywood.get('total_boards_optimized_purchase', 0)} pcs",
        SUCCESS_COLOR,
        bold=True,
    )
    _row("Waste Area", f"{float(plywood.get('waste_area_m2', 0.0)):.2f} m²")
    _row("Waste Percentage", f"{float(plywood.get('waste_percent', 0.0)):.1f} %")
    y -= 0.015

    if plywood.get("aabb_pack_overcount"):
        _header("WARNINGS")
        ax_info.text(
            0.05,
            y,
            "Non-orthogonal / diagonal room cut detected.\nAABB packing may slightly overcount.",
            fontsize=9.5,
            color=WARNING_COLOR,
            fontweight="bold",
            transform=ax_info.transAxes,
        )
        y -= pad_line * 2


def render_plywood(
    metrics_mm: dict[str, Any],
    plywood: dict[str, Any],
    output_path: str | Path,
    *,
    chart_only: bool = False,
) -> None:
    """Render subfloor plywood layout diagram to PNG or PDF (full technical sheet or chart-only)."""
    raw_pts = metrics_mm.get("vertices_mm") or []
    corners_mm = np.asarray(raw_pts, dtype=float)

    project_name = str(metrics_mm.get("project_name") or "UNTITLED")
    board_size = plywood.get("board_size_mm") or {}
    board_length_mm = float(board_size.get("length_mm", 1820.0))
    board_width_mm = float(board_size.get("width_mm", 910.0))
    orientation = str(plywood.get("orientation", "normal"))
    pieces = plywood.get("pieces") or []
    floor_area_m2 = float(plywood.get("room_area_m2", 0.0) or metrics_mm.get("area_m2", 0.0) or 0.0)
    opt_purchase = plywood.get("total_boards_optimized_purchase", 0)
    waste_area_m2 = float(plywood.get("waste_area_m2", 0.0))

    if len(corners_mm) >= 3:
        min_x, min_y = corners_mm.min(axis=0)
        max_x, max_y = corners_mm.max(axis=0)
        bb_width = max_x - min_x
        bb_depth = max_y - min_y
    else:
        bb_width = float(metrics_mm.get("bbox_width_mm", 0.0))
        bb_depth = float(metrics_mm.get("bbox_depth_mm", 0.0))

    from app.services.cad.fonts import configure_matplotlib_agg

    configure_matplotlib_agg()
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if chart_only:
        fig, ax_plan = plt.subplots(figsize=(14, 14), facecolor="white", dpi=150)
        try:
            if len(corners_mm) >= 3:
                _draw_plywood_plan(
                    ax_plan,
                    corners_mm,
                    pieces,
                    bb_width,
                    bb_depth,
                    board_length_mm,
                    board_width_mm,
                )
            else:
                ax_plan.text(0.5, 0.5, "No geometry available", ha="center", va="center")
                ax_plan.axis("off")

            fig.savefig(
                str(target_path),
                bbox_inches="tight",
                dpi=150,
                facecolor="white",
            )
        finally:
            plt.close(fig)
        return

    # Full technical sheet layout (Title, Plan, Info panel, Footer)
    fig = plt.figure(figsize=(22, 16), facecolor="white", dpi=150)
    try:
        gs = gridspec.GridSpec(
            3,
            2,
            figure=fig,
            height_ratios=[1, 14, 0.6],
            width_ratios=[2.8, 1],
            wspace=0.02,
            hspace=0.02,
            left=0.02,
            right=0.98,
            top=0.98,
            bottom=0.03,
        )

        # Row 0: Title Bar
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis("off")
        ax_title.add_patch(
            patches.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax_title.transAxes,
                facecolor="white",
                edgecolor=HEADER_COLOR,
                linewidth=1.5,
            )
        )
        title_main = (
            f"SUBFLOOR PLYWOOD (合板 / 捨て貼り)  ·  {orientation.upper()}  ·  {project_name.upper()}"
        )
        ax_title.text(
            0.5,
            0.62,
            title_main,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_title.transAxes,
        )
        title_sub = (
            f"Room: {bb_width:.2f} × {bb_depth:.2f} mm   |   "
            f"Board: {board_width_mm:.0f} × {board_length_mm:.0f} mm   |   "
            f"Floor Area: {floor_area_m2:.2f} m²   |   "
            f"Optimized Purchase: {opt_purchase} boards   |   "
            f"Waste: {waste_area_m2:.2f} m²"
        )
        ax_title.text(
            0.5,
            0.20,
            title_sub,
            ha="center",
            va="center",
            fontsize=12,
            color="#475467",
            transform=ax_title.transAxes,
        )

        # Row 1, Col 0: Floor Plan
        ax_plan = fig.add_subplot(gs[1, 0])
        if len(corners_mm) >= 3:
            _draw_plywood_plan(
                ax_plan,
                corners_mm,
                pieces,
                bb_width,
                bb_depth,
                board_length_mm,
                board_width_mm,
            )
        else:
            ax_plan.text(0.5, 0.5, "No geometry available", ha="center", va="center")
            ax_plan.axis("off")

        # Row 1, Col 1: Info Panel
        ax_info = fig.add_subplot(gs[1, 1])
        _draw_info_panel(ax_info, metrics_mm, plywood, board_length_mm, board_width_mm)

        # Row 2: Footer
        ax_footer = fig.add_subplot(gs[2, :])
        ax_footer.axis("off")
        ax_footer.text(
            0.02,
            0.5,
            "LiDAR Room Scan · Subfloor Plywood Parity Engine · Standard 1820×910mm (Saburoku-ban)",
            fontsize=10,
            color="#94A3B8",
            va="center",
            transform=ax_footer.transAxes,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        ax_footer.text(
            0.98,
            0.5,
            f"Generated {timestamp}",
            fontsize=10,
            color="#94A3B8",
            va="center",
            ha="right",
            transform=ax_footer.transAxes,
        )

        fig.savefig(
            str(target_path),
            bbox_inches="tight",
            dpi=150,
            facecolor="white",
        )
    finally:
        plt.close(fig)
