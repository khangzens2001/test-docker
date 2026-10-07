"""Neda (根太) floor batten estimation and CAD rendering.

Calculates wood batten positions, counts, and lengths using Shapely polygon
clipping with 303 mm center-to-center pitch and border battens along room walls.
Ported from 3D-Estimate backend/api/estimate_pattern.py.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box
from shapely.ops import orient



# Colors matching 3D-Estimate technical styling
BORDER_FILL = "#A0522D"
BORDER_EDGE = "#8B4513"
BORDER_FILL_ALT = "#966F33"
WOOD_FILL = "#D4A574"
WOOD_EDGE = "#B8895A"
WOOD_FILL_ALT = "#C9975E"
WALL_COLOR = "#1A1A1A"
FLOOR_FILL = "#F8F4EC"
HEADER_COLOR = "#1E293B"
TEXT_COLOR = "#1E293B"
DIM_COLOR = "#2563EB"
SPACING_COLOR = "#0D9488"


def format_dim(val: float | int | None) -> str:
    """Format dimension to 2 decimal places universally."""
    if val is None:
        return ""
    return f"{float(val):.2f}"


def _extract_2d_segments(
    geom: Any, orientation: str, batten_width_mm: float
) -> tuple[list[dict[str, Any]], float, int, float]:
    """Extract 2D polygon segments along orientation axis with bounds and centroids."""
    segments: list[dict[str, Any]] = []
    total_len = 0.0
    count = 0
    min_len = float("inf")

    polys: list[ShapelyPolygon] = []
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]

    for poly in polys:
        bounds = poly.bounds
        if orientation == "vertical":
            length = bounds[3] - bounds[1]
        else:
            length = bounds[2] - bounds[0]

        if length > 1.0 and poly.area > 1.0:
            poly_oriented = orient(poly, sign=1.0)
            coords = list(poly_oriented.exterior.coords)
            vertices = [[float(x), float(y)] for x, y in coords[:-1]]
            cx = float(poly.centroid.x)
            cy = float(poly.centroid.y)
            segments.append({
                "rect": [float(b) for b in bounds],
                "length": float(length),
                "vertices": vertices,
                "centroid_mm": [cx, cy],
                "area": float(poly.area),
            })
            total_len += float(length)
            count += 1
            min_len = min(min_len, float(length))

    return segments, total_len, count, min_len


def _estimate_battens_single(
    pts: np.ndarray,
    orientation: str,
    batten_width_mm: float,
    batten_spacing_mm: float,
    waste_rate: float = 0.0,
) -> dict[str, Any]:
    """Run batten estimation for a single orientation."""
    min_x, min_z = pts.min(axis=0)
    max_x, max_z = pts.max(axis=0)

    room_poly = ShapelyPolygon(pts)
    if not room_poly.is_valid:
        room_poly = room_poly.buffer(0)
    room_area_m2 = round(room_poly.area / 1e6, 4)

    half_w = batten_width_mm / 2.0

    # Step 1: Create border battens by inward buffer
    inner_poly = room_poly.buffer(-batten_width_mm, join_style=2)
    if inner_poly is None or inner_poly.is_empty or not inner_poly.is_valid:
        inner_poly = ShapelyPolygon()

    border_battens: list[dict[str, Any]] = []
    border_total_length = 0.0
    border_count = 0

    n_pts = len(pts)
    for i in range(n_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % n_pts]
        edge_vec = p2 - p1
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1.0:
            continue

        edge_dir = edge_vec / edge_len
        normal = np.array([-edge_dir[1], edge_dir[0]])
        ext = 0.1

        best_intersection = None
        for sign in [1, -1]:
            n = normal * sign
            corners_rect = [
                p1 - edge_dir * ext + n * ext,
                p2 + edge_dir * ext + n * ext,
                p2 + edge_dir * ext - n * (batten_width_mm + ext),
                p1 - edge_dir * ext - n * (batten_width_mm + ext),
            ]
            wall_rect_poly = ShapelyPolygon(corners_rect)
            if not wall_rect_poly.is_valid:
                wall_rect_poly = wall_rect_poly.buffer(0)

            inter = room_poly.intersection(wall_rect_poly)
            if best_intersection is None or (
                not inter.is_empty and inter.area > best_intersection.area
            ):
                best_intersection = inter

        intersection = best_intersection
        if intersection is not None and not intersection.is_empty and intersection.area > 1.0:
            segments, s_len, s_count, _ = _extract_2d_segments(
                intersection, orientation, batten_width_mm
            )
            if segments:
                if orientation == "vertical":
                    pos = float((p1[0] + p2[0]) / 2.0)
                else:
                    pos = float((p1[1] + p2[1]) / 2.0)

                border_battens.append({
                    "position": pos,
                    "segments": segments,
                    "is_border": True,
                    "wall_index": i,
                })
                border_total_length += s_len
                border_count += s_count

    # Step 2: Create inner battens clipped to inner_poly
    inner_battens: list[dict[str, Any]] = []
    inner_total_length = 0.0
    inner_count = 0

    if not inner_poly.is_empty:
        pitch = batten_spacing_mm

        if orientation == "vertical":
            first_center = min_x + half_w + pitch
            last_border_center = max_x - half_w

            x_center = first_center
            while x_center < last_border_center - 1.0:
                x_left = x_center - half_w
                batten_rect = shapely_box(
                    x_left, min_z - 50.0,
                    x_left + batten_width_mm, max_z + 50.0
                )
                intersection = inner_poly.intersection(batten_rect)
                if not intersection.is_empty and intersection.area > 1.0:
                    segments, s_len, s_count, _ = _extract_2d_segments(
                        intersection, "vertical", batten_width_mm
                    )
                    if segments:
                        inner_battens.append({
                            "position": float(x_left),
                            "segments": segments,
                            "is_border": False,
                        })
                        inner_total_length += s_len
                        inner_count += s_count
                x_center += pitch
        else:
            first_center = min_z + half_w + pitch
            last_border_center = max_z - half_w

            z_center = first_center
            while z_center < last_border_center - 1.0:
                z_top = z_center - half_w
                batten_rect = shapely_box(
                    min_x - 50.0, z_top,
                    max_x + 50.0, z_top + batten_width_mm
                )
                intersection = inner_poly.intersection(batten_rect)
                if not intersection.is_empty and intersection.area > 1.0:
                    segments, s_len, s_count, _ = _extract_2d_segments(
                        intersection, "horizontal", batten_width_mm
                    )
                    if segments:
                        inner_battens.append({
                            "position": float(z_top),
                            "segments": segments,
                            "is_border": False,
                        })
                        inner_total_length += s_len
                        inner_count += s_count
                z_center += pitch

    all_battens = border_battens + inner_battens
    total_length_mm = border_total_length + inner_total_length
    total_count = border_count + inner_count

    # Cut grouping by rounded 5 mm length
    all_segments: list[dict[str, Any]] = []
    for b in all_battens:
        all_segments.extend(b["segments"])

    group_counts: dict[int, int] = defaultdict(int)
    for s in all_segments:
        key = int(round(s["length"] / 5.0) * 5)
        group_counts[key] += 1

    sorted_keys = sorted(group_counts.keys(), reverse=True)
    key_to_id = {k: f"B{i + 1}" for i, k in enumerate(sorted_keys)}

    for b in all_battens:
        for s in b["segments"]:
            key = int(round(s["length"] / 5.0) * 5)
            s["piece_id"] = key_to_id[key]

    cut_groups = [
        {"id": key_to_id[k], "length": k, "count": group_counts[k]}
        for k in sorted_keys
    ]

    return {
        "orientation": orientation,
        "pitch_mm": float(batten_spacing_mm),
        "border_width_mm": float(batten_width_mm),
        "border_count": border_count,
        "inner_count": inner_count,
        "total_battens": total_count,
        "total_length_m": round((total_length_mm / 1000.0) * (1.0 + float(waste_rate)), 3),
        "total_area_m2": round((total_length_mm * batten_width_mm) / 1e6, 3),
        "room_area_m2": room_area_m2,
        "waste_rate": float(waste_rate),
        "cut_groups": cut_groups,
        "battens": all_battens,
    }


def estimate_neda(
    metrics_mm: dict[str, Any],
    *,
    joist_pitch_mm: float = 303.0,
    border_width_mm: float = 30.0,
    waste_rate: float = 0.0,
) -> dict[str, Any]:
    """Estimate Neda floor batten layout for room metrics.

    Tries vertical and horizontal orientations; selects candidate with lower
    total_length_m (tie-breaker: fewer pieces, then "vertical").
    """
    raw_pts = metrics_mm.get("vertices_mm") or []
    pts = np.asarray(raw_pts, dtype=float)

    if len(pts) < 3:
        return {
            "orientation": "vertical",
            "pitch_mm": float(joist_pitch_mm),
            "border_width_mm": float(border_width_mm),
            "border_count": 0,
            "inner_count": 0,
            "total_battens": 0,
            "total_length_m": 0.0,
            "total_area_m2": 0.0,
            "room_area_m2": 0.0,
            "waste_rate": float(waste_rate),
            "cut_groups": [],
            "battens": [],
        }

    res_v = _estimate_battens_single(
        pts, "vertical", border_width_mm, joist_pitch_mm, waste_rate=waste_rate
    )
    res_h = _estimate_battens_single(
        pts, "horizontal", border_width_mm, joist_pitch_mm, waste_rate=waste_rate
    )

    candidates = [res_v, res_h]
    # Tie break: lower total_length_m, fewer total_battens, then "vertical"
    candidates.sort(
        key=lambda x: (
            x["total_length_m"],
            x["total_battens"],
            0 if x["orientation"] == "vertical" else 1,
        )
    )
    return candidates[0]


def _draw_floor_plan(
    ax_plan: Any,
    corners_mm: np.ndarray,
    battens: list[dict[str, Any]],
    batten_width_mm: float,
    joist_pitch_mm: float,
    orientation: str,
    bb_width: float,
    bb_depth: float,
) -> None:
    """Draw floor polygon, walls, clipped battens, dimensions, and legend on ax_plan."""
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    from matplotlib.path import Path as MPath

    for spine in ax_plan.spines.values():
        spine.set_edgecolor(HEADER_COLOR)
        spine.set_linewidth(1.5)

    poly_pts = [tuple(c) for c in corners_mm] + [tuple(corners_mm[0])]
    poly_path = MPath(poly_pts)
    clip_patch = patches.PathPatch(
        poly_path, transform=ax_plan.transData, fill=False, edgecolor="none"
    )
    ax_plan.add_patch(clip_patch)
    ax_plan.add_patch(
        patches.PathPatch(poly_path, facecolor=FLOOR_FILL, edgecolor="none", zorder=1)
    )

    for idx, batten in enumerate(battens):
        is_border = batten.get("is_border", False)
        if is_border:
            fill_color = BORDER_FILL if idx % 2 == 0 else BORDER_FILL_ALT
            edge_color = BORDER_EDGE
        else:
            fill_color = WOOD_FILL if idx % 2 == 0 else WOOD_FILL_ALT
            edge_color = WOOD_EDGE

        for seg in batten.get("segments", []):
            poly_coords = seg.get("vertices", [])
            if len(poly_coords) >= 3:
                ax_plan.add_patch(
                    patches.Polygon(
                        poly_coords,
                        closed=True,
                        facecolor=fill_color,
                        edgecolor=edge_color,
                        linewidth=0.5,
                        zorder=3,
                        alpha=0.85,
                    )
                )

            cx, cy = seg.get("centroid_mm", [0.0, 0.0])
            length = float(seg.get("length", 0.0))
            if length > 80.0:
                if is_border:
                    text_str = f"Border\n{format_dim(length)}"
                else:
                    piece_id = seg.get("piece_id", "B")
                    text_str = f"{piece_id}\n{format_dim(length)}"

                ax_plan.text(
                    cx,
                    cy,
                    text_str,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=TEXT_COLOR,
                    linespacing=1.2,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor=edge_color,
                        linewidth=0.6,
                        alpha=0.9,
                    ),
                    zorder=8,
                )

    n_corners = len(corners_mm)
    for i in range(n_corners):
        c1 = corners_mm[i]
        c2 = corners_mm[(i + 1) % n_corners]
        ax_plan.plot(
            [c1[0], c2[0]],
            [c1[1], c2[1]],
            color=WALL_COLOR,
            linewidth=4,
            solid_capstyle="butt",
            zorder=5,
        )

    for c in corners_mm:
        ax_plan.plot(c[0], c[1], "o", color="#333333", markersize=5, zorder=6)

    min_x, min_z = corners_mm.min(axis=0)
    max_x, max_z = corners_mm.max(axis=0)
    margin = max(300.0, float(max(bb_width, bb_depth)) * 0.1)

    # Bounding box width annotation (bottom)
    ax_plan.annotate(
        "",
        xy=(min_x, min_z - margin * 0.5),
        xytext=(max_x, min_z - margin * 0.5),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.5),
    )
    ax_plan.text(
        (min_x + max_x) / 2.0,
        min_z - margin * 0.7,
        f"{bb_width:.2f} mm",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=DIM_COLOR,
    )

    # Bounding box depth annotation (right)
    ax_plan.annotate(
        "",
        xy=(max_x + margin * 0.5, min_z),
        xytext=(max_x + margin * 0.5, max_z),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.5),
    )
    ax_plan.text(
        max_x + margin * 0.75,
        (min_z + max_z) / 2.0,
        f"{bb_depth:.2f} mm",
        ha="left",
        va="center",
        rotation=90,
        fontsize=11,
        fontweight="bold",
        color=DIM_COLOR,
    )

    # Spacing annotation between first two inner battens
    inner_battens_list = [b for b in battens if not b.get("is_border", False)]
    if len(inner_battens_list) >= 2:
        half_w = batten_width_mm / 2.0
        b1_center = inner_battens_list[0]["position"] + half_w
        b2_center = inner_battens_list[1]["position"] + half_w
        actual_spacing = abs(b2_center - b1_center)

        if orientation == "vertical":
            arrow_y = min_z + (max_z - min_z) * 0.15
            ax_plan.annotate(
                "",
                xy=(b1_center, arrow_y),
                xytext=(b2_center, arrow_y),
                arrowprops=dict(arrowstyle="<->", color=SPACING_COLOR, linewidth=1.2),
                zorder=10,
            )
            ax_plan.text(
                (b1_center + b2_center) / 2.0,
                arrow_y,
                f"{actual_spacing:.2f} mm\n(c-c)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=SPACING_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=SPACING_COLOR,
                    linewidth=0.6,
                ),
                zorder=11,
            )
        else:
            arrow_x = min_x + (max_x - min_x) * 0.15
            ax_plan.annotate(
                "",
                xy=(arrow_x, b1_center),
                xytext=(arrow_x, b2_center),
                arrowprops=dict(arrowstyle="<->", color=SPACING_COLOR, linewidth=1.2),
                zorder=10,
            )
            ax_plan.text(
                arrow_x,
                (b1_center + b2_center) / 2.0,
                f"{actual_spacing:.2f} mm\n(c-c)",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=SPACING_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=SPACING_COLOR,
                    linewidth=0.6,
                ),
                zorder=11,
            )

    ax_plan.set_aspect("equal")
    ax_plan.grid(True, alpha=0.15, linewidth=0.4, color="#888")
    ax_plan.set_xticklabels([])
    ax_plan.set_yticklabels([])
    ax_plan.tick_params(axis="both", length=0)
    ax_plan.invert_yaxis()
    ax_plan.set_xlim(min_x - margin * 1.2, max_x + margin * 1.2)
    ax_plan.set_ylim(max_z + margin * 1.2, min_z - margin * 1.2)

    legend_elems = [
        patches.Patch(facecolor=FLOOR_FILL, edgecolor=HEADER_COLOR, label="Floor"),
        mlines.Line2D([], [], color=WALL_COLOR, linewidth=3, label="Wall"),
        patches.Patch(
            facecolor=BORDER_FILL,
            edgecolor=BORDER_EDGE,
            label=f"Border Batten ({batten_width_mm:.0f}mm)",
        ),
        patches.Patch(
            facecolor=WOOD_FILL,
            edgecolor=WOOD_EDGE,
            label=f"Inner Batten ({batten_width_mm:.0f}mm)",
        ),
        mlines.Line2D(
            [],
            [],
            color=SPACING_COLOR,
            linewidth=1.2,
            linestyle="-",
            marker="s",
            markersize=4,
            label=f"Pitch ({joist_pitch_mm:.0f}mm c-c)",
        ),
        mlines.Line2D(
            [],
            [],
            color=DIM_COLOR,
            linewidth=1.5,
            marker="s",
            markersize=5,
            label="BBox dimension",
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


def render_neda(
    metrics_mm: dict[str, Any],
    neda: dict[str, Any],
    output_path: str | Path,
    *,
    chart_only: bool = False,
) -> None:
    """Render Neda layout diagram to PNG or PDF (full sheet or chart-only)."""
    raw_pts = metrics_mm.get("vertices_mm") or []
    corners_mm = np.asarray(raw_pts, dtype=float)

    project_name = str(metrics_mm.get("project_name") or "UNTITLED")
    floor_area_m2 = float(metrics_mm.get("area_m2", 0.0) or 0.0)

    batten_width_mm = float(neda.get("border_width_mm", 30.0))
    joist_pitch_mm = float(neda.get("pitch_mm", 303.0))
    orientation = str(neda.get("orientation", "vertical"))
    battens = neda.get("battens") or []

    if len(corners_mm) >= 3:
        min_x, min_z = corners_mm.min(axis=0)
        max_x, max_z = corners_mm.max(axis=0)
        bb_width = max_x - min_x
        bb_depth = max_z - min_z
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
                _draw_floor_plan(
                    ax_plan,
                    corners_mm,
                    battens,
                    batten_width_mm,
                    joist_pitch_mm,
                    orientation,
                    bb_width,
                    bb_depth,
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
            f"ESTIMATE PATTERN  ·  {orientation.upper()}  ·  {project_name.upper()}"
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
            f"Batten Width: {batten_width_mm:.2f} mm   |   "
            f"Spacing: {joist_pitch_mm:.2f} mm   |   "
            f"Floor Area: {floor_area_m2:.2f} m²"
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
            _draw_floor_plan(
                ax_plan,
                corners_mm,
                battens,
                batten_width_mm,
                joist_pitch_mm,
                orientation,
                bb_width,
                bb_depth,
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
            "ESTIMATION",
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

        params = [
            ("Orientation", orientation.upper()),
            ("Batten Width", f"{batten_width_mm:.2f} mm"),
            ("Spacing", f"{joist_pitch_mm:.2f} mm"),
            ("Room Width", f"{bb_width:.2f} mm"),
            ("Room Depth", f"{bb_depth:.2f} mm"),
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
            y_cursor -= 0.026

        y_cursor -= 0.006
        ax_info.plot(
            [0.05, 0.95],
            [y_cursor, y_cursor],
            color=HEADER_COLOR,
            linewidth=1.2,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        y_cursor -= 0.015

        ax_info.text(
            0.5,
            y_cursor,
            "JAPANESE ESTIMATION (根太)",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )
        y_cursor -= 0.035

        results = [
            ("Total Battens", f"{neda.get('total_battens', 0)} pcs", SPACING_COLOR),
            ("Border Battens", f"{neda.get('border_count', 0)} pcs", "#8B4513"),
            ("Inner Battens", f"{neda.get('inner_count', 0)} pcs", "#475467"),
            ("Total Length", f"{neda.get('total_length_m', 0.0):.2f} m", DIM_COLOR),
            ("Total Wood Area", f"{neda.get('total_area_m2', 0.0):.2f} m²", DIM_COLOR),
            ("Floor Area", f"{floor_area_m2:.2f} m²", "#475467"),
        ]
        for label, value, color in results:
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
                color=color,
                transform=ax_info.transAxes,
            )
            y_cursor -= 0.030

        y_cursor -= 0.008
        ax_info.plot(
            [0.05, 0.95],
            [y_cursor, y_cursor],
            color=HEADER_COLOR,
            linewidth=1.2,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        y_cursor -= 0.015

        ax_info.text(
            0.5,
            y_cursor,
            "PATTERN SPECIFICATION",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )
        y_cursor -= 0.035

        total_pieces = 0
        for grp in neda.get("cut_groups", []):
            label = f"{grp['id']}  —  {batten_width_mm:.2f} × {format_dim(grp['length'])} mm"
            qty = f"× {grp['count']} pcs"
            total_pieces += grp["count"]

            fs = 9 if len(label) > 28 else 11
            ax_info.text(
                0.08,
                y_cursor,
                label,
                ha="left",
                va="center",
                fontsize=fs,
                color="#475467",
                transform=ax_info.transAxes,
            )
            ax_info.text(
                0.92,
                y_cursor,
                qty,
                ha="right",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=SPACING_COLOR,
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
            "Total To Buy",
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
            f"{total_pieces} pcs",
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )

        # Detail View inset
        detail_y = 0.22
        ax_info.text(
            0.5,
            detail_y,
            "DETAIL VIEW",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=HEADER_COLOR,
            transform=ax_info.transAxes,
        )

        detail_ax = ax_info.inset_axes(
            [0.06, detail_y - 0.16, 0.88, 0.14],
            transform=ax_info.transAxes,
        )
        detail_ax.set_xlim(-0.5, 10.5)
        detail_ax.set_ylim(-0.8, 5.5)
        detail_ax.set_aspect("equal")
        detail_ax.axis("off")

        bw_s = 2.0
        sp_s = 4.0
        b1x = 0.5
        b2x = b1x + bw_s + sp_s
        b1_center = b1x + bw_s / 2.0
        b2_center = b2x + bw_s / 2.0

        detail_ax.add_patch(
            patches.Rectangle(
                (b1x, 0.5),
                bw_s,
                4.0,
                facecolor=WOOD_FILL,
                edgecolor=WOOD_EDGE,
                linewidth=1.5,
            )
        )
        detail_ax.add_patch(
            patches.Rectangle(
                (b2x, 0.5),
                bw_s,
                4.0,
                facecolor=WOOD_FILL_ALT,
                edgecolor=WOOD_EDGE,
                linewidth=1.5,
            )
        )

        detail_ax.plot(
            [b1_center, b1_center],
            [-0.3, 4.7],
            color=SPACING_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        detail_ax.plot(
            [b2_center, b2_center],
            [-0.3, 4.7],
            color=SPACING_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

        detail_ax.annotate(
            "",
            xy=(b1_center, -0.1),
            xytext=(b2_center, -0.1),
            arrowprops=dict(arrowstyle="<->", color=SPACING_COLOR, linewidth=1.5),
        )
        detail_ax.text(
            (b1_center + b2_center) / 2.0,
            -0.6,
            f"{joist_pitch_mm:.2f} mm (C-C)",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=SPACING_COLOR,
        )
        detail_ax.annotate(
            "",
            xy=(b1x, 5.0),
            xytext=(b1x + bw_s, 5.0),
            arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.2),
        )
        detail_ax.text(
            b1x + bw_s / 2.0,
            5.3,
            f"{batten_width_mm:.2f} mm",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=DIM_COLOR,
        )

        ax_info.plot(
            [0.05, 0.95],
            [0.05, 0.05],
            color=HEADER_COLOR,
            linewidth=1.2,
            transform=ax_info.transAxes,
            clip_on=False,
        )
        notes = (
            "1. Battens are strictly clipped inside the layout bounds.\n"
            f"2. Batten width: {batten_width_mm:.2f} mm, Spacing (C-C): {joist_pitch_mm:.2f} mm.\n"
            f"3. Orientation: {orientation.upper()}."
        )
        ax_info.text(
            0.08,
            0.03,
            notes,
            ha="left",
            va="top",
            fontsize=9,
            color="#475467",
            linespacing=1.4,
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
            f"ORIENTATION: {orientation.upper()}"
        )
        ax_footer.text(
            0.5,
            0.5,
            footer_text,
            ha="center",
            va="center",
            fontsize=12,
            color=HEADER_COLOR,
            transform=ax_footer.transAxes,
        )

        fig.savefig(
            str(target_path),
            bbox_inches="tight",
            dpi=150,
            facecolor=fig.get_facecolor(),
        )
    finally:
        plt.close(fig)
