"""Cushion Floor (CF) roll material layout estimation and CAD rendering service.

Calculates CF roll layouts along room length and width, clipping strips to room
polygons and correctly summing multi-component segments for notched / non-convex rooms.
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



# CF Color palette matching 3D-Estimate architectural theme
CF_COLOR = "#66C285"
CF_EDGE = "#2E8B57"
CF_CUT_COLOR = "#F39C12"
CF_CUT_EDGE = "#E67E22"
CF_CUT_HATCH = "///"

FLOOR_FILL = "#F8F4EC"
WALL_COLOR = "#1A1A1A"
DIM_COLOR = "#0066CC"
HEADER_COLOR = "#1A1A2E"
TEXT_COLOR = "#333333"


def _extract_polygons(geom: Any) -> list[ShapelyPolygon]:
    """Extract valid 2D polygon components from shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, ShapelyPolygon):
        return [geom] if geom.area > 1e-4 else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if isinstance(g, ShapelyPolygon) and g.area > 1e-4]
    if hasattr(geom, "geoms"):
        result: list[ShapelyPolygon] = []
        for g in geom.geoms:
            result.extend(_extract_polygons(g))
        return result
    return []


def _poly_coords(poly: ShapelyPolygon) -> list[list[float]]:
    """Extract 2D exterior coordinates from shapely Polygon."""
    return [[float(x), float(y)] for x, y in poly.exterior.coords]


def _calculate_cf_direction(
    room_poly: ShapelyPolygon,
    *,
    direction: str,
    roll_width_mm: float,
    waste_rate: float = 0.0,
    round_purchase_to_m: float | None = None,
) -> dict[str, Any]:
    """Calculate CF roll layout for a specific direction with polygon clipping."""
    min_x, min_y, max_x, max_y = room_poly.bounds
    bb_width = max_x - min_x
    bb_depth = max_y - min_y
    room_area_m2 = round(room_poly.area / 1e6, 4)

    if direction == "along_length":
        bb_across = bb_width
        num_strips = math.ceil(round(bb_across, 4) / roll_width_mm) if bb_across > 0 else 0
        last_strip_width_mm = (
            bb_across - roll_width_mm * (num_strips - 1) if num_strips > 0 else 0.0
        )
        direction_label = "Along Length (Vertical)"
        direction_label_vi = "Trải dọc theo chiều dài phòng"
        orientation_label = "VERTICAL"
    else:
        bb_across = bb_depth
        num_strips = math.ceil(round(bb_across, 4) / roll_width_mm) if bb_across > 0 else 0
        last_strip_width_mm = (
            bb_across - roll_width_mm * (num_strips - 1) if num_strips > 0 else 0.0
        )
        direction_label = "Along Width (Horizontal)"
        direction_label_vi = "Trải ngang theo chiều rộng phòng"
        orientation_label = "HORIZONTAL"

    strips: list[dict[str, Any]] = []

    for i in range(num_strips):
        strip_w = roll_width_mm if i < num_strips - 1 else last_strip_width_mm
        is_cut = strip_w < (roll_width_mm - 0.5)

        if direction == "along_length":
            x0 = min_x + i * roll_width_mm
            strip_box = shapely_box(x0, min_y, x0 + strip_w, max_y)
            rect_x = x0
            rect_y = min_y
            rect_w = strip_w
            rect_h = bb_depth
            axis = "y"
        else:
            y0 = min_y + i * roll_width_mm
            strip_box = shapely_box(min_x, y0, max_x, y0 + strip_w)
            rect_x = min_x
            rect_y = y0
            rect_w = bb_width
            rect_h = strip_w
            axis = "x"

        inter = room_poly.intersection(strip_box)
        polys = _extract_polygons(inter)

        components: list[dict[str, Any]] = []
        total_strip_len_mm = 0.0

        for p in polys:
            p_min_x, p_min_y, p_max_x, p_max_y = p.bounds
            comp_len = (p_max_y - p_min_y) if axis == "y" else (p_max_x - p_min_x)
            if comp_len > 1e-3:
                total_strip_len_mm += comp_len
                components.append({
                    "bounds": [
                        round(p_min_x, 2),
                        round(p_min_y, 2),
                        round(p_max_x, 2),
                        round(p_max_y, 2),
                    ],
                    "length_mm": round(comp_len, 2),
                    "area_m2": round(p.area / 1e6, 4),
                    "polygon_coords": _poly_coords(p),
                })

        dim_str = (
            f"{strip_w:.0f}×{total_strip_len_mm:.0f}"
            if direction == "along_length"
            else f"{total_strip_len_mm:.0f}×{strip_w:.0f}"
        )

        if components:
            largest_comp = max(components, key=lambda c: c["length_mm"])
            cb = largest_comp["bounds"]
            text_x = (cb[0] + cb[2]) / 2.0
            text_y = (cb[1] + cb[3]) / 2.0
        else:
            text_x = rect_x + rect_w / 2.0
            text_y = rect_y + rect_h / 2.0

        strips.append({
            "strip_no": i + 1,
            "strip_index": i,
            "label": f"CF Strip {i + 1}",
            "direction": direction,
            "x": round(rect_x, 2),
            "y": round(rect_y, 2),
            "rect_width": round(rect_w, 2),
            "rect_height": round(rect_h, 2),
            "text_x": round(text_x, 2),
            "text_y": round(text_y, 2),
            "width": round(strip_w, 2),
            "length": round(total_strip_len_mm, 2),
            "width_mm": round(strip_w, 2),
            "length_mm": round(total_strip_len_mm, 2),
            "purchase_width_mm": roll_width_mm,
            "dim": dim_str,
            "is_cut": is_cut,
            "components": components,
        })

    total_linear_length_m = round(sum(s["length_mm"] for s in strips) / 1000.0, 4)
    total_linear_length_with_waste_m = round(total_linear_length_m * (1.0 + waste_rate), 4)

    if round_purchase_to_m is not None and round_purchase_to_m > 0:
        total_linear_length_purchase_m = round(
            math.ceil(total_linear_length_with_waste_m / round_purchase_to_m)
            * round_purchase_to_m,
            4,
        )
    else:
        total_linear_length_purchase_m = total_linear_length_with_waste_m

    material_area_m2_without_waste = round(
        sum(roll_width_mm * s["length_mm"] for s in strips) / 1e6, 4
    )
    material_area_m2_with_waste = round(
        material_area_m2_without_waste * (1.0 + waste_rate), 4
    )
    waste_area_m2_without_waste_rate = round(
        max(0.0, material_area_m2_without_waste - room_area_m2), 4
    )

    return {
        "direction": direction,
        "direction_label": direction_label,
        "direction_label_vi": direction_label_vi,
        "orientation_label": orientation_label,
        "roll_width_mm": roll_width_mm,
        "number_of_strips": num_strips,
        "last_strip_width_mm": round(last_strip_width_mm, 2),
        "total_linear_length_m": total_linear_length_m,
        "waste_rate": waste_rate,
        "total_linear_length_with_waste_m": total_linear_length_with_waste_m,
        "round_purchase_to_m": round_purchase_to_m,
        "total_linear_length_purchase_m": total_linear_length_purchase_m,
        "room_area_m2": room_area_m2,
        "material_area_m2_without_waste": material_area_m2_without_waste,
        "material_area_m2_with_waste": material_area_m2_with_waste,
        "waste_area_m2_without_waste_rate": waste_area_m2_without_waste_rate,
        "strips": strips,
    }


def estimate_cf(
    metrics_mm: dict[str, Any],
    *,
    cf_roll_width_mm: float = 940.0,
    waste_rate: float = 0.0,
    round_purchase_to_m: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Estimate Cushion Floor (CF) roll material requirements.

    Evaluates roll layout along both length and width directions, clips strips
    to the room polygon (summing multi-component lengths for notched profiles),
    and selects the direction with minimum total purchased linear length.
    """
    raw_pts = metrics_mm.get("vertices_mm") or []
    if len(raw_pts) < 3:
        empty_res = {
            "direction": "along_length",
            "direction_label": "Along Length (Vertical)",
            "direction_label_vi": "Trải dọc theo chiều dài phòng",
            "orientation_label": "VERTICAL",
            "roll_width_mm": cf_roll_width_mm,
            "number_of_strips": 0,
            "last_strip_width_mm": 0.0,
            "total_linear_length_m": 0.0,
            "waste_rate": waste_rate,
            "total_linear_length_with_waste_m": 0.0,
            "round_purchase_to_m": round_purchase_to_m,
            "total_linear_length_purchase_m": 0.0,
            "room_area_m2": 0.0,
            "material_area_m2_without_waste": 0.0,
            "material_area_m2_with_waste": 0.0,
            "waste_area_m2_without_waste_rate": 0.0,
            "strips": [],
            "options": [],
            "best_option": {},
        }
        return empty_res

    pts = np.asarray(raw_pts, dtype=float)
    room_poly = ShapelyPolygon(pts)
    if not room_poly.is_valid:
        room_poly = room_poly.buffer(0)

    if room_poly.is_empty or room_poly.area <= 1e-4:
        empty_res = {
            "direction": "along_length",
            "direction_label": "Along Length (Vertical)",
            "direction_label_vi": "Trải dọc theo chiều dài phòng",
            "orientation_label": "VERTICAL",
            "roll_width_mm": cf_roll_width_mm,
            "number_of_strips": 0,
            "last_strip_width_mm": 0.0,
            "total_linear_length_m": 0.0,
            "waste_rate": waste_rate,
            "total_linear_length_with_waste_m": 0.0,
            "round_purchase_to_m": round_purchase_to_m,
            "total_linear_length_purchase_m": 0.0,
            "room_area_m2": 0.0,
            "material_area_m2_without_waste": 0.0,
            "material_area_m2_with_waste": 0.0,
            "waste_area_m2_without_waste_rate": 0.0,
            "strips": [],
            "options": [],
            "best_option": {},
        }
        return empty_res

    opt_length = _calculate_cf_direction(
        room_poly,
        direction="along_length",
        roll_width_mm=cf_roll_width_mm,
        waste_rate=waste_rate,
        round_purchase_to_m=round_purchase_to_m,
    )

    opt_width = _calculate_cf_direction(
        room_poly,
        direction="along_width",
        roll_width_mm=cf_roll_width_mm,
        waste_rate=waste_rate,
        round_purchase_to_m=round_purchase_to_m,
    )

    options = [opt_length, opt_width]
    best_option = min(
        options,
        key=lambda x: (
            x["total_linear_length_purchase_m"],
            x["number_of_strips"],
            x["waste_area_m2_without_waste_rate"],
        ),
    )

    result = dict(best_option)
    result["options"] = options
    result["best_option"] = best_option
    return result


def _draw_cf_plan(
    ax_plan: Any,
    corners_mm: np.ndarray,
    strips: list[dict[str, Any]],
    roll_width_mm: float,
    bb_width: float,
    bb_depth: float,
) -> None:
    """Draw floor plan with clipped CF strips and dimensions."""
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    from matplotlib.path import Path as MPath
    poly_path = MPath(corners_mm, closed=True)
    clip_patch = patches.PathPatch(
        poly_path, transform=ax_plan.transData, fill=False, edgecolor="none"
    )
    ax_plan.add_patch(clip_patch)

    ax_plan.add_patch(
        patches.PathPatch(poly_path, facecolor=FLOOR_FILL, edgecolor="none", zorder=1)
    )

    for i, strip in enumerate(strips):
        f_color = CF_CUT_COLOR if strip["is_cut"] else CF_COLOR
        e_color = CF_CUT_EDGE if strip["is_cut"] else CF_EDGE
        h_pattern = CF_CUT_HATCH if strip["is_cut"] else ""

        rect = patches.Rectangle(
            (strip["x"], strip["y"]),
            strip["rect_width"],
            strip["rect_height"],
            linewidth=1.5,
            edgecolor=e_color,
            facecolor=f_color,
            hatch=h_pattern,
            alpha=0.85,
            zorder=3,
        )
        rect.set_clip_path(clip_patch)
        ax_plan.add_patch(rect)

        label = f"CF {i + 1}"
        if strip["is_cut"]:
            label += " (Cut)"

        ax_plan.text(
            strip["text_x"],
            strip["text_y"],
            f"{label}\n{strip['dim']} mm",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=TEXT_COLOR,
            linespacing=1.2,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=e_color,
                linewidth=0.8,
                alpha=0.9,
            ),
            zorder=8,
        )

    # Walls outline
    n_corners = len(corners_mm)
    for i in range(n_corners):
        c1 = corners_mm[i]
        c2 = corners_mm[(i + 1) % n_corners]
        ax_plan.plot(
            [c1[0], c2[0]],
            [c1[1], c2[1]],
            color=WALL_COLOR,
            linewidth=5.0,
            solid_capstyle="butt",
            zorder=5,
        )

    for c in corners_mm:
        ax_plan.plot(c[0], c[1], "o", color="#333333", markersize=6, zorder=6)

    min_x, min_y = corners_mm.min(axis=0)
    max_x, max_y = corners_mm.max(axis=0)
    margin = max(400.0, float(max(bb_width, bb_depth)) * 0.1)

    # Dimension annotations (width at bottom, depth at right)
    ax_plan.annotate(
        "",
        xy=(min_x, min_y - margin * 0.6),
        xytext=(max_x, min_y - margin * 0.6),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.5),
    )
    ax_plan.text(
        (min_x + max_x) / 2.0,
        min_y - margin * 0.8,
        f"{bb_width:.2f} mm",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=DIM_COLOR,
    )

    ax_plan.annotate(
        "",
        xy=(max_x + margin * 0.6, min_y),
        xytext=(max_x + margin * 0.6, max_y),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.5),
    )
    ax_plan.text(
        max_x + margin * 0.85,
        (min_y + max_y) / 2.0,
        f"{bb_depth:.2f} mm",
        ha="left",
        va="center",
        rotation=90,
        fontsize=12,
        fontweight="bold",
        color=DIM_COLOR,
    )

    ax_plan.set_aspect("equal")
    ax_plan.grid(True, alpha=0.15, linewidth=0.4, color="#888")
    ax_plan.set_xticklabels([])
    ax_plan.set_yticklabels([])
    ax_plan.tick_params(axis="both", length=0)
    ax_plan.invert_yaxis()
    ax_plan.set_xlim(min_x - margin * 1.5, max_x + margin * 1.5)
    ax_plan.set_ylim(max_y + margin * 1.5, min_y - margin * 1.5)

    legend_elems = [
        patches.Patch(facecolor=FLOOR_FILL, edgecolor=HEADER_COLOR, label="Floor Plan"),
        mlines.Line2D([], [], color=WALL_COLOR, linewidth=4, label="Wall Outline"),
        patches.Patch(
            facecolor=CF_COLOR,
            edgecolor=CF_EDGE,
            label=f"Full CF Strip ({roll_width_mm:.0f}mm)",
        ),
        patches.Patch(
            facecolor=CF_CUT_COLOR,
            edgecolor=CF_CUT_EDGE,
            hatch=CF_CUT_HATCH,
            label="Cut CF Piece",
        ),
        mlines.Line2D([], [], color=DIM_COLOR, linewidth=1.5, label="BBox dimension"),
    ]
    leg = ax_plan.legend(
        handles=legend_elems,
        loc="lower left",
        fontsize=11,
        ncol=3,
        framealpha=0.9,
        edgecolor=HEADER_COLOR,
        title="LEGEND",
    )
    if leg.get_title():
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontsize(12)
    leg.get_frame().set_linewidth(1.2)


def render_cf(
    metrics_mm: dict[str, Any],
    cf: dict[str, Any],
    output_path: str | Path,
    *,
    chart_only: bool = False,
) -> None:
    """Render CF roll material layout estimation diagram (technical sheet or chart-only)."""
    raw_pts = metrics_mm.get("vertices_mm") or []
    corners_mm = np.asarray(raw_pts, dtype=float)

    project_name = str(metrics_mm.get("project_name") or "UNTITLED")
    direction = cf.get("direction", "along_length")
    orient_label = cf.get("orientation_label", "VERTICAL")
    roll_width = float(cf.get("roll_width_mm", 940.0))
    strips = cf.get("strips") or []
    num_strips = int(cf.get("number_of_strips", len(strips)))
    floor_area_m2 = float(cf.get("room_area_m2", 0.0))
    last_width = float(cf.get("last_strip_width_mm", roll_width))
    total_linear_length_m = float(cf.get("total_linear_length_m", 0.0))
    total_linear_length_purchase_m = float(
        cf.get("total_linear_length_purchase_m", total_linear_length_m)
    )

    if len(corners_mm) >= 3:
        min_x, min_y = corners_mm.min(axis=0)
        max_x, max_y = corners_mm.max(axis=0)
        bb_width = max_x - min_x
        bb_depth = max_y - min_y
    else:
        bb_width = float(metrics_mm.get("bbox_width_mm", 0.0))
        bb_depth = float(metrics_mm.get("bbox_depth_mm", 0.0))

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    from app.services.cad.fonts import configure_matplotlib_agg

    configure_matplotlib_agg()
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    if chart_only:
        fig, ax_plan = plt.subplots(figsize=(14, 14), facecolor="white", dpi=150)
        try:
            if len(corners_mm) >= 3:
                _draw_cf_plan(
                    ax_plan, corners_mm, strips, roll_width, bb_width, bb_depth
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

    # Full technical sheet layout
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
            f"CF FINISH LAYER ESTIMATION  ·  {orient_label.upper()}  ·  {project_name.upper()}"
        )
        ax_title.text(
            0.5,
            0.62,
            title_main,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_title.transAxes,
        )
        title_sub = (
            f"Room: {bb_width:.2f} × {bb_depth:.2f} mm   |   "
            f"Roll Width: {roll_width:.0f} mm   |   "
            f"Strips: {num_strips}   |   "
            f"Floor Area: {floor_area_m2:.2f} m²"
        )
        ax_title.text(
            0.5,
            0.20,
            title_sub,
            ha="center",
            va="center",
            fontsize=13,
            color="#475467",
            transform=ax_title.transAxes,
        )

        # Row 1, Col 0: Plan
        ax_plan = fig.add_subplot(gs[1, 0])
        for spine in ax_plan.spines.values():
            spine.set_edgecolor(HEADER_COLOR)
            spine.set_linewidth(1.5)

        if len(corners_mm) >= 3:
            _draw_cf_plan(
                ax_plan, corners_mm, strips, roll_width, bb_width, bb_depth
            )
        else:
            ax_plan.text(0.5, 0.5, "No geometry available", ha="center", va="center")
            ax_plan.axis("off")

        # Row 1, Col 1: Right Info Panel
        ax_info = fig.add_subplot(gs[1, 1])
        ax_info.axis("off")
        ax_info.add_patch(
            patches.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax_info.transAxes,
                facecolor="#FAFAFA",
                edgecolor=HEADER_COLOR,
                linewidth=1.5,
                zorder=0,
            )
        )

        ax_info.text(
            0.5,
            0.97,
            "CF ESTIMATION",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )
        ax_info.plot(
            [0.05, 0.95],
            [0.94, 0.94],
            color=HEADER_COLOR,
            linewidth=1.2,
            transform=ax_info.transAxes,
            clip_on=False,
        )

        dir_label = cf.get("direction_label", direction)
        params = [
            ("Direction", dir_label),
            ("Roll Width", f"{roll_width:.0f} mm"),
            ("Room Width", f"{bb_width:.2f} mm"),
            ("Room Depth", f"{bb_depth:.2f} mm"),
            ("Floor Area", f"{floor_area_m2:.2f} m²"),
        ]

        y_cursor = 0.90
        for label, value in params:
            ax_info.text(
                0.08,
                y_cursor,
                label,
                ha="left",
                va="center",
                fontsize=11,
                color="#475467",
                transform=ax_info.transAxes,
            )
            ax_info.text(
                0.92,
                y_cursor,
                value,
                ha="right",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=HEADER_COLOR,
                transform=ax_info.transAxes,
            )
            y_cursor -= 0.028

        y_cursor -= 0.005
        ax_info.plot(
            [0.08, 0.92],
            [y_cursor, y_cursor],
            color=HEADER_COLOR,
            linewidth=0.8,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        y_cursor -= 0.025

        ax_info.text(
            0.08,
            y_cursor,
            "STRIP BREAKDOWN",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )
        y_cursor -= 0.035

        full_count = 0
        cut_count = 0
        for s_info in strips:
            color = CF_CUT_EDGE if s_info["is_cut"] else CF_EDGE
            label_str = f"{s_info['label']}"
            if s_info["is_cut"]:
                label_str += " (Cut)"
                cut_count += 1
            else:
                full_count += 1

            dim_str = f"{s_info['width']:.0f} × {s_info['length']:.0f} mm"
            ax_info.text(
                0.08,
                y_cursor,
                label_str,
                ha="left",
                va="center",
                fontsize=10,
                color=color,
                transform=ax_info.transAxes,
            )
            ax_info.text(
                0.92,
                y_cursor,
                dim_str,
                ha="right",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
                transform=ax_info.transAxes,
            )
            y_cursor -= 0.028

        y_cursor += 0.008
        ax_info.plot(
            [0.08, 0.92],
            [y_cursor, y_cursor],
            color=HEADER_COLOR,
            linewidth=1.0,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        y_cursor -= 0.022

        ax_info.text(
            0.08,
            y_cursor,
            "Summary",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )
        y_cursor -= 0.035

        total_cf_area_m2 = float(cf.get("material_area_m2_without_waste", 0.0))
        waste_area_m2 = float(cf.get("waste_area_m2_without_waste_rate", 0.0))
        waste_pct = (
            (waste_area_m2 / total_cf_area_m2 * 100.0) if total_cf_area_m2 > 0 else 0.0
        )

        summary_items = [
            ("Total Strips", f"{num_strips}"),
            ("Full Strips", f"{full_count}"),
            ("Cut Strips", f"{cut_count}"),
            ("Roll Width", f"{roll_width:.0f} mm"),
            ("Last Strip", f"{last_width:.0f} mm"),
            ("CF Area", f"{total_cf_area_m2:.2f} m²"),
            ("Waste", f"{waste_pct:.1f}%"),
        ]

        for label, value in summary_items:
            ax_info.text(
                0.08,
                y_cursor,
                label,
                ha="left",
                va="center",
                fontsize=11,
                color="#475467",
                transform=ax_info.transAxes,
            )
            ax_info.text(
                0.92,
                y_cursor,
                value,
                ha="right",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=HEADER_COLOR,
                transform=ax_info.transAxes,
            )
            y_cursor -= 0.028

        y_cursor -= 0.005
        ax_info.plot(
            [0.08, 0.92],
            [y_cursor, y_cursor],
            color=HEADER_COLOR,
            linewidth=1.0,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        y_cursor -= 0.022

        ax_info.text(
            0.08,
            y_cursor,
            "Total CF Roll",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )
        ax_info.text(
            0.92,
            y_cursor,
            f"{total_linear_length_purchase_m:.1f} m",
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )

        # Row 2: Footer Bar
        ax_footer = fig.add_subplot(gs[2, :])
        ax_footer.axis("off")
        ax_footer.add_patch(
            patches.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax_footer.transAxes,
                facecolor="#F1F5F9",
                edgecolor=HEADER_COLOR,
                linewidth=1.0,
            )
        )
        footer_text = (
            f"DATE: {datetime.now().strftime('%d-%m-%Y')}   |   "
            f"SCALE: NTS   |   "
            f"PROJECT: {project_name.upper()}   |   "
            f"LAYER: CF FINISH   |   "
            f"ORIENTATION: {orient_label.upper()}"
        )
        ax_footer.text(
            0.5,
            0.5,
            footer_text,
            ha="center",
            va="center",
            fontsize=13,
            color=HEADER_COLOR,
            transform=ax_footer.transAxes,
        )

        fig.savefig(str(target_path), bbox_inches="tight", dpi=150, facecolor="white")
    finally:
        plt.close(fig)
