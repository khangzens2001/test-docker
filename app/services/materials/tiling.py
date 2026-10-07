# Ported from 3D-Estimate backend/api/estimate_tiling.py and backend/api/pipeline.py
"""Floor tiling estimation and CAD rendering service.

Provides bounded center-aligned search optimizer for floor tiling and
high-resolution technical rendering to PNG and PDF.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np



from app.services.materials.tiling_optimizer import TilingOptimizer

# Technical drawing color palette
WOOD_FULL_A = "#C49A6C"       # Medium oak
WOOD_FULL_B = "#D4A574"       # Light oak
WOOD_GRAIN_A = "#A07850"      # Grain lines for A
WOOD_GRAIN_B = "#B8895C"      # Grain lines for B
WOOD_CUT = "#B07040"          # Darker walnut for cut tiles
WOOD_CUT_GRAIN = "#8D5A30"    # Grain for cut tiles
TILE_EDGE = "#8B7355"         # Grout/edge color
FLOOR_FILL = "#F8F4EC"
WALL_COLOR = "#1A1A1A"
DIM_COLOR = "#0066CC"
HEADER_COLOR = "#1A1A2E"
ACCENT_COLOR = "#CC0000"
WARNING_COLOR = "#D97706"     # Amber for warnings
SUCCESS_COLOR = "#16A34A"     # Green for success indicators


def format_dim(val: float | int | None) -> str:
    """Format dimension to 2 decimal places universally."""
    if val is None:
        return ""
    return f"{float(val):.2f}"


def estimate_tiling(
    metrics_mm: dict[str, Any],
    *,
    tile_length_mm: float = 300.0,
    tile_width_mm: float = 300.0,
    tile_joint_width_mm: float = 3.0,
    deadline_s: float = 60.0,
) -> dict[str, Any]:
    """Estimate floor tiling layout using bounded center-aligned optimizer.

    Parameters:
    - metrics_mm: canonical millimetre metrics dictionary containing vertices_mm
    - tile_length_mm: tile length in mm (default 300.0)
    - tile_width_mm: tile width in mm (default 300.0)
    - tile_joint_width_mm: grout joint width in mm (default 3.0)
    - deadline_s: wall-clock deadline in seconds (default 60.0)

    Returns:
    - Dict with tiling results or {"skipped": True, "reason": "timeout"}
    """
    raw_pts = metrics_mm.get("vertices_mm") or []
    if len(raw_pts) < 3:
        return {
            "full_tiles": 0,
            "cut_tiles": 0,
            "total_buy_tiles": 0,
            "floor_area_m2": 0.0,
            "room_area_m2": 0.0,
            "gross_waste_percent": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "orientation": "landscape",
            "num_small_pieces": 0,
            "num_forbidden_pieces": 0,
            "warnings": ["No room geometry available"],
            "total_stock_tiles": 0,
            "total_waste_percent": 0.0,
            "tiles": [],
            "cutting_plan": {"stock_tiles": [], "total_stock_tiles": 0, "total_waste_m2": 0.0, "total_waste_percent": 0.0},
        }

    start_time = time.monotonic()
    deadline_time = start_time + deadline_s if deadline_s > 0 else 0.0

    if deadline_s <= 0.0 or time.monotonic() >= deadline_time:
        return {"skipped": True, "reason": "timeout"}

    optimizer = TilingOptimizer(
        tile_length_mm=tile_length_mm,
        tile_width_mm=tile_width_mm,
        grout_gap_mm=tile_joint_width_mm,
        avoid_piece_side_under_mm=100.0,
        forbidden_piece_side_under_mm=50.0,
        search_mode="balanced",
    )

    result = optimizer.optimize(raw_pts, deadline_time=deadline_time)
    if result.get("skipped"):
        return result

    stats = result["stats"]
    cutting_plan = result["cutting_plan"]
    warnings = result.get("warnings", [])
    floor_area_m2 = stats.get("floor_area_m2", round(float(metrics_mm.get("area_m2", 0.0)), 4))

    return {
        "full_tiles": stats["full_tiles"],
        "cut_tiles": stats["cut_tiles"],
        "total_buy_tiles": stats["total_buy_tiles"],
        "floor_area_m2": floor_area_m2,
        "room_area_m2": floor_area_m2,
        "gross_waste_percent": stats["gross_waste_percent"],
        "offset_x": stats["best_offset_x_mm"],
        "offset_y": stats["best_offset_y_mm"],
        "orientation": stats["best_orientation"],
        "num_small_pieces": stats["num_small_pieces"],
        "num_forbidden_pieces": stats["num_forbidden_pieces"],
        "warnings": warnings,
        "total_stock_tiles": cutting_plan["total_stock_tiles"],
        "total_waste_percent": cutting_plan["total_waste_percent"],
        "tiles": result.get("tiles", []),
        "cutting_plan": cutting_plan,
        "piece_dimensions": result.get("piece_dimensions", []),
        "cut_tile_groups": stats.get("cut_tile_groups", []),
        "search_candidates_evaluated": stats.get("search_candidates_evaluated", 0),
        "total_tiles": stats.get("total_tiles", 0),
        "total_tile_area_m2": stats.get("total_tile_area_m2", 0.0),
        "total_buy_area_m2": stats.get("total_buy_area_m2", 0.0),
        "gross_waste_area_m2": stats.get("gross_waste_area_m2", 0.0),
        "waste_area_m2": stats.get("waste_area_m2", 0.0),
        "coverage_percent": stats.get("coverage_percent", 0.0),
        "tile_length_mm": tile_length_mm,
        "tile_width_mm": tile_width_mm,
        "tile_joint_width_mm": tile_joint_width_mm,
        "stats": stats,
    }


def _cut_group_letter(idx: int) -> str:
    """Generate cut group letter, skipping 'F' (index 5) reserved for full tiles."""
    adjusted = idx + 1 if idx >= 5 else idx
    if adjusted < 26:
        return chr(65 + adjusted)
    return f"G{idx + 1}"


def _draw_wood_grain_tile(ax, rect, facecolor, edgecolor, grain_color, clip_patch, zorder=3):
    """Draw a tile rectangle with procedural grain lines clipped to the room boundary."""
    import matplotlib.patches as mpatches

    rx, ry, rw, rh = rect
    rect_patch = mpatches.Rectangle(
        (rx, ry), rw, rh,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=0.5, zorder=zorder, alpha=0.9
    )
    rect_patch.set_clip_path(clip_patch)
    ax.add_patch(rect_patch)

    rng = np.random.RandomState(int(abs(rx * 100 + ry * 10)) % (2**32 - 1))
    num_lines = rng.randint(4, 7)
    if rw >= rh:
        for i in range(num_lines):
            y_pos = ry + rh * (i + 1.0) / (num_lines + 1)
            y_wobble = rh * 0.05 * rng.randn()
            line, = ax.plot([rx, rx + rw], [y_pos - y_wobble, y_pos + y_wobble],
                            color=grain_color, linewidth=0.5, zorder=zorder, alpha=0.6)
            line.set_clip_path(clip_patch)
    else:
        for i in range(num_lines):
            x_pos = rx + rw * (i + 1.0) / (num_lines + 1)
            x_wobble = rw * 0.05 * rng.randn()
            line, = ax.plot([x_pos - x_wobble, x_pos + x_wobble], [ry, ry + rh],
                            color=grain_color, linewidth=0.5, zorder=zorder, alpha=0.6)
            line.set_clip_path(clip_patch)


def _annotate_cut_tile_edges(ax, tile, eff_length, eff_width, clip_patch):
    """Draw dimension lines along cut edges for a cut tile piece."""
    if tile.get("type") != "cut":
        return
    shape_type = tile.get("shape_type", "Rect")
    edges = tile.get("edges", [])
    cut_w = tile.get("cut_length_mm", 0)
    cut_z = tile.get("cut_width_mm", 0)
    if min(cut_w, cut_z) <= 50.0:
        return

    for edge in edges:
        length = edge.get("length_mm", 0.0)
        if length <= 50.0:
            continue
        p1 = np.array(edge["from"])
        p2 = np.array(edge["to"])
        orientation = edge.get("orientation", "H")
        if shape_type == "Rect":
            is_cut = False
            if orientation == "H" and length < eff_length - 5:
                is_cut = True
            elif orientation == "V" and length < eff_width - 5:
                is_cut = True
            if not is_cut:
                continue

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        edge_len = math.hypot(dx, dy)
        if edge_len < 1e-3:
            continue

        nx = dy / edge_len
        ny = -dx / edge_len
        offset_dist = 60.0
        op1 = p1 + offset_dist * np.array([nx, ny])
        op2 = p2 + offset_dist * np.array([nx, ny])
        ext1_start = p1 + (offset_dist * 0.15) * np.array([nx, ny])
        ext1_end = p1 + (offset_dist * 1.15) * np.array([nx, ny])
        ext2_start = p2 + (offset_dist * 0.15) * np.array([nx, ny])
        ext2_end = p2 + (offset_dist * 1.15) * np.array([nx, ny])

        ax.plot([ext1_start[0], ext1_end[0]], [ext1_start[1], ext1_end[1]],
                color=DIM_COLOR, linewidth=0.6, alpha=0.7, zorder=6)
        ax.plot([ext2_start[0], ext2_end[0]], [ext2_start[1], ext2_end[1]],
                color=DIM_COLOR, linewidth=0.6, alpha=0.7, zorder=6)
        ax.annotate(
            "", xy=op2, xytext=op1,
            arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=0.8, shrinkA=0, shrinkB=0),
            zorder=6,
        )
        cx, cy = (op1 + op2) / 2.0
        tx = cx + 25.0 * nx
        ty = cy + 25.0 * ny
        angle_deg = math.degrees(math.atan2(dy, dx))
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
        ax.text(
            tx, ty, format_dim(length), ha="center", va="center",
            fontsize=7, fontweight="bold", color=DIM_COLOR,
            rotation=angle_deg,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8),
            zorder=7,
        )


def _draw_tiling_plan(
    ax_plan,
    corners_mm: np.ndarray,
    tiles: list[dict],
    stats: dict,
    eff_tile_length: float,
    eff_tile_width: float,
    tile_length_mm: float,
    bb_width: float,
    bb_depth: float,
) -> None:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MPath

    min_x, min_z = corners_mm.min(axis=0)
    max_x, max_z = corners_mm.max(axis=0)

    for spine in ax_plan.spines.values():
        spine.set_edgecolor(HEADER_COLOR)
        spine.set_linewidth(1.5)

    # Floor polygon
    rotated_pts = [tuple(c) for c in corners_mm]
    poly_pts = rotated_pts + [rotated_pts[0]]
    poly_path = MPath(poly_pts)
    clip_patch = mpatches.PathPatch(poly_path, transform=ax_plan.transData, fill=False, edgecolor="none")
    ax_plan.add_patch(clip_patch)
    ax_plan.add_patch(mpatches.PathPatch(poly_path, facecolor=FLOOR_FILL, edgecolor="none", zorder=1))

    # Tiles
    for idx, tile in enumerate(tiles):
        rx, ry, rw, rh = tile["rect"]
        if tile.get("type") == "full":
            facecolor = WOOD_FULL_A if idx % 2 == 0 else WOOD_FULL_B
            graincolor = WOOD_GRAIN_A if idx % 2 == 0 else WOOD_GRAIN_B
        else:
            facecolor = WOOD_CUT
            graincolor = WOOD_CUT_GRAIN
        _draw_wood_grain_tile(ax_plan, (rx, ry, rw, rh), facecolor, TILE_EDGE, graincolor, clip_patch, zorder=3)

    # Batten positions if available
    if stats.get("batten_positions_mm"):
        is_horiz = stats.get("tiling_direction", "horizontal") == "horizontal"
        for pos in stats["batten_positions_mm"]:
            if is_horiz:
                ax_plan.axvline(x=pos, linestyle="--", color="#8B4513", alpha=0.4, linewidth=1.5, zorder=4)
            else:
                ax_plan.axhline(y=pos, linestyle="--", color="#8B4513", alpha=0.4, linewidth=1.5, zorder=4)

    # Group cut tiles
    key_to_group_letter = {}
    group_counters = {}
    for idx, grp in enumerate(stats.get("cut_tile_groups", [])):
        key = (round(grp["cut_length_mm"] / 5) * 5, round(grp["cut_width_mm"] / 5) * 5, grp.get("shape_type", "Rect"))
        letter = _cut_group_letter(idx)
        key_to_group_letter[key] = letter
        group_counters[letter] = 0

    tile_to_label = {}
    for tile_idx, tile in enumerate(tiles):
        if tile.get("type") == "cut":
            cl = tile.get("cut_length_mm", 0)
            cw = tile.get("cut_width_mm", 0)
            shape_type = tile.get("shape_type", "Rect")
            key = (round(cl / 5) * 5, round(cw / 5) * 5, shape_type)
            letter = key_to_group_letter.get(key, "?")
            group_counters[letter] = group_counters.get(letter, 0) + 1
            tile_to_label[tile_idx] = f"{letter}{group_counters[letter]}"

    margin = max(400.0, bb_width * 0.1)
    plot_width_px = 4700
    plot_height_px = 4200
    data_range_x = bb_width + 3 * margin
    data_range_y = bb_depth + 3 * margin

    f_counter = 1
    for tile_idx, tile in enumerate(tiles):
        rx, ry, rw, rh = tile["rect"]
        cx, cy = rx + rw / 2.0, ry + rh / 2.0

        tile_px_w = (rw / data_range_x) * plot_width_px
        tile_px_h = (rh / data_range_y) * plot_height_px
        min_dim = min(tile_px_w, tile_px_h)
        if min_dim < 30:
            continue

        label_fontsize = 6 if min_dim < 60 else 8
        show_dimensions = min_dim >= 60

        if tile.get("type") == "full":
            label_text = f"F{f_counter}"
            f_counter += 1
            t = ax_plan.text(
                cx, cy, label_text, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", linewidth=0.8, alpha=0.9),
                zorder=8,
            )
            t.set_clip_path(clip_patch)
        else:
            cl = tile.get("cut_length_mm", 0)
            cw = tile.get("cut_width_mm", 0)
            seq_label = tile_to_label.get(tile_idx, "?")
            dim_text = f"{format_dim(cl)}×{format_dim(cw)}"
            final_text = f"{seq_label}\n{dim_text}" if show_dimensions else seq_label
            t = ax_plan.text(
                cx, cy, final_text, ha="center", va="center",
                fontsize=label_fontsize, fontweight="bold", color=ACCENT_COLOR,
                linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=ACCENT_COLOR, linewidth=0.8, alpha=0.9),
                zorder=8,
            )
            t.set_clip_path(clip_patch)

    for tile in tiles:
        if tile.get("type") == "cut":
            _annotate_cut_tile_edges(ax_plan, tile, eff_tile_length, eff_tile_width, clip_patch)

    n_pts = len(corners_mm)
    for i in range(n_pts):
        c1 = corners_mm[i]
        c2 = corners_mm[(i + 1) % n_pts]
        ax_plan.plot([c1[0], c2[0]], [c1[1], c2[1]], color=WALL_COLOR, linewidth=5, solid_capstyle="butt", zorder=5)

    for c in corners_mm:
        ax_plan.plot(c[0], c[1], "o", color="#333333", markersize=6, zorder=6)

    # Dimension arrows
    ax_plan.annotate(
        "", xy=(min_x, min_z - margin * 0.6),
        xytext=(max_x, min_z - margin * 0.6),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.5)
    )
    ax_plan.text(
        (min_x + max_x) / 2, min_z - margin * 0.8,
        f"{bb_width:.2f} mm", ha="center", va="top",
        fontsize=12, fontweight="bold", color=DIM_COLOR,
    )

    ax_plan.annotate(
        "", xy=(max_x + margin * 0.6, min_z),
        xytext=(max_x + margin * 0.6, max_z),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=1.5)
    )
    ax_plan.text(
        max_x + margin * 0.85, (min_z + max_z) / 2,
        f"{bb_depth:.2f} mm", ha="left", va="center", rotation=90,
        fontsize=12, fontweight="bold", color=DIM_COLOR,
    )

    first_full = next((t for t in tiles if t.get("type") == "full"), None)
    if first_full:
        fx, fy, fw, fh = first_full["rect"]
        ax_plan.annotate(
            "", xy=(fx, fy + fh + margin * 0.15),
            xytext=(fx + fw, fy + fh + margin * 0.15),
            arrowprops=dict(arrowstyle="<->", color=ACCENT_COLOR, linewidth=1.0),
            zorder=10
        )
        ax_plan.text(
            fx + fw / 2, fy + fh + margin * 0.25,
            f"{tile_length_mm:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=ACCENT_COLOR,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=ACCENT_COLOR, linewidth=0.5),
            zorder=11
        )

    ax_plan.set_aspect("equal")
    ax_plan.grid(True, alpha=0.15, linewidth=0.4, color="#888")
    ax_plan.set_xticklabels([])
    ax_plan.set_yticklabels([])
    ax_plan.tick_params(axis="both", length=0)
    ax_plan.invert_yaxis()
    ax_plan.set_xlim(min_x - margin * 1.5, max_x + margin * 1.5)
    ax_plan.set_ylim(max_z + margin * 1.5, min_z - margin * 1.5)

    # Legend
    orient_label = stats.get("best_orientation", stats.get("orientation", "landscape")).capitalize()
    legend_elems = [
        mpatches.Patch(facecolor=FLOOR_FILL, edgecolor=HEADER_COLOR, label="Floor Plan"),
        mlines.Line2D([], [], color=WALL_COLOR, linewidth=4, label="Wall Outline"),
        mpatches.Patch(facecolor=WOOD_FULL_A, edgecolor=TILE_EDGE,
                       label=f"Full Tile ({eff_tile_length:.2f}×{eff_tile_width:.2f}mm)"),
        mpatches.Patch(facecolor=WOOD_CUT, edgecolor=TILE_EDGE, label="Cut Tile Piece"),
        mlines.Line2D([], [], color=DIM_COLOR, linewidth=1.0, label="Cut Edge Dim (mm)"),
    ]
    if stats.get("batten_positions_mm"):
        legend_elems.append(
            mlines.Line2D([], [], color="#8B4513", linestyle="--", linewidth=1.5, alpha=0.6, label="Batten Ref")
        )
    leg = ax_plan.legend(
        handles=legend_elems, loc="lower left",
        ncol=3, fontsize=10, framealpha=0.9, edgecolor=HEADER_COLOR,
        title=f"LEGEND  ·  {orient_label}"
    )
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_fontsize(11)
    leg.get_frame().set_linewidth(1.0)


def render_tiling(
    metrics_mm: dict[str, Any],
    tiling: dict[str, Any],
    output_path: str | Path,
    *,
    chart_only: bool = False,
) -> None:
    """Render floor tiling diagram to PNG or PDF (full sheet or chart-only)."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    from app.services.cad.fonts import configure_matplotlib_agg

    configure_matplotlib_agg()
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if tiling.get("skipped"):
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white", dpi=150)
        try:
            ax.text(0.5, 0.5, f"Tiling estimation skipped ({tiling.get('reason', 'timeout')})",
                    ha="center", va="center", fontsize=14, color=WARNING_COLOR)
            ax.axis("off")
            fig.savefig(str(target_path), bbox_inches="tight", dpi=150, facecolor="white")
        finally:
            plt.close(fig)
        return

    raw_pts = metrics_mm.get("vertices_mm") or []
    corners_mm = np.asarray(raw_pts, dtype=float)

    project_name = str(metrics_mm.get("project_name") or "UNTITLED")
    floor_area_m2 = float(tiling.get("floor_area_m2") or metrics_mm.get("area_m2", 0.0))

    tile_length_mm = float(tiling.get("tile_length_mm", 300.0))
    tile_width_mm = float(tiling.get("tile_width_mm", 300.0))
    grout_gap_mm = float(tiling.get("tile_joint_width_mm", 3.0))

    stats = tiling.get("stats", tiling)
    eff_tile_length = float(stats.get("effective_tile_length_mm", tile_length_mm))
    eff_tile_width = float(stats.get("effective_tile_width_mm", tile_width_mm))

    tiles = tiling.get("tiles", [])
    warnings = tiling.get("warnings", [])

    if len(corners_mm) >= 3:
        min_x, min_z = corners_mm.min(axis=0)
        max_x, max_z = corners_mm.max(axis=0)
        bb_width = max_x - min_x
        bb_depth = max_z - min_z
    else:
        bb_width = float(metrics_mm.get("bbox_width_mm", 0.0))
        bb_depth = float(metrics_mm.get("bbox_depth_mm", 0.0))

    if chart_only:
        fig, ax_plan = plt.subplots(figsize=(14, 14), facecolor="white", dpi=150)
        try:
            if len(corners_mm) >= 3:
                _draw_tiling_plan(
                    ax_plan, corners_mm, tiles, stats,
                    eff_tile_length, eff_tile_width, tile_length_mm,
                    bb_width, bb_depth
                )
            else:
                ax_plan.text(0.5, 0.5, "No geometry available", ha="center", va="center")
                ax_plan.axis("off")

            fig.savefig(str(target_path), bbox_inches="tight", dpi=150, facecolor="white")
        finally:
            plt.close(fig)
        return

    # Full technical sheet layout
    fig = plt.figure(figsize=(22, 16), facecolor="white", dpi=150)
    try:
        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[1, 14, 0.6],
            width_ratios=[2.5, 1],
            wspace=0.02, hspace=0.02,
            left=0.02, right=0.98, top=0.98, bottom=0.03,
        )

        # Row 0: Title Bar
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis("off")
        ax_title.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, transform=ax_title.transAxes,
            facecolor="white", edgecolor=HEADER_COLOR, linewidth=1.5
        ))
        title_main = f"FLOOR TILING ESTIMATION  ·  {project_name.upper()}"
        ax_title.text(0.5, 0.62, title_main, ha="center", va="center",
                      fontsize=20, fontweight="bold", color=HEADER_COLOR,
                      transform=ax_title.transAxes)
        title_sub = (
            f"Room: {bb_width:.2f} × {bb_depth:.2f} mm   |   "
            f"Tile: {tile_length_mm:.2f} × {tile_width_mm:.2f} mm   |   "
            f"Grout: {grout_gap_mm:.2f} mm   |   "
            f"Floor Area: {floor_area_m2:.2f} m²"
        )
        ax_title.text(0.5, 0.20, title_sub, ha="center", va="center",
                      fontsize=12, color="#475467", transform=ax_title.transAxes)

        # Row 1, Col 0: Plan
        ax_plan = fig.add_subplot(gs[1, 0])
        if len(corners_mm) >= 3:
            _draw_tiling_plan(
                ax_plan, corners_mm, tiles, stats,
                eff_tile_length, eff_tile_width, tile_length_mm,
                bb_width, bb_depth
            )
        else:
            ax_plan.text(0.5, 0.5, "No geometry available", ha="center", va="center")
            ax_plan.axis("off")

        # Row 1, Col 1: Right Info Panel
        ax_info = fig.add_subplot(gs[1, 1])
        ax_info.axis("off")
        ax_info.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, transform=ax_info.transAxes,
            facecolor="#FAFAFA", edgecolor=HEADER_COLOR, linewidth=1.5, zorder=0
        ))

        ax_info.text(0.5, 0.97, "TILING ESTIMATION", ha="center", va="top",
                     fontsize=18, fontweight="bold", color=HEADER_COLOR,
                     transform=ax_info.transAxes)
        ax_info.plot([0.05, 0.95], [0.94, 0.94], color=HEADER_COLOR,
                     linewidth=1.2, transform=ax_info.transAxes, clip_on=False)

        params = [
            ("Tile Size", f"{tile_length_mm:.2f} × {tile_width_mm:.2f} mm"),
            ("Grout Gap", f"{grout_gap_mm:.2f} mm"),
            ("Room Width", f"{bb_width:.2f} mm"),
            ("Room Depth", f"{bb_depth:.2f} mm"),
            ("Floor Area", f"{floor_area_m2:.2f} m²"),
        ]
        if stats.get("batten_aligned"):
            params.append(("Batten-Aligned", "Yes"))
            if "batten_spacing_mm" in stats:
                params.append(("Batten Spacing", f"{stats['batten_spacing_mm']:.1f} mm"))
            if "tiling_direction" in stats:
                params.append(("Tiling Direction", stats["tiling_direction"].title()))

        y_cursor = 0.90
        for label, value in params:
            ax_info.text(0.08, y_cursor, label, ha="left", va="center",
                         fontsize=11, color="#475467", transform=ax_info.transAxes)
            ax_info.text(0.92, y_cursor, value, ha="right", va="center",
                         fontsize=11, fontweight="bold", color=HEADER_COLOR,
                         transform=ax_info.transAxes)
            y_cursor -= 0.026

        y_cursor -= 0.006
        ax_info.plot([0.05, 0.95], [y_cursor, y_cursor], color=HEADER_COLOR,
                     linewidth=1.2, transform=ax_info.transAxes, clip_on=False)
        y_cursor -= 0.015

        ax_info.text(0.5, y_cursor, "CALCULATION RESULTS", ha="center", va="top",
                     fontsize=13, fontweight="bold", color=HEADER_COLOR,
                     transform=ax_info.transAxes)
        y_cursor -= 0.035

        results = [
            ("Tiles to Buy (Est)", f"{stats.get('total_buy_tiles', stats.get('total_tiles', 0))} pcs", ACCENT_COLOR),
            ("Full Tiles (Pcs)", f"{stats.get('full_tiles', 0)} pcs", SUCCESS_COLOR),
            ("Cut Pieces (Pcs)", f"{stats.get('cut_tiles', 0)} pcs", TILE_EDGE),
            ("Cut Groups", f"{stats.get('num_cut_groups', 0)}", TILE_EDGE),
            ("Net Placed Area", f"{stats.get('total_tile_area_m2', 0.0):.2f} m²", DIM_COLOR),
            ("Gross Waste", f"{stats.get('gross_waste_area_m2', 0.0):.2f} m² ({stats.get('gross_waste_percent', 0.0):.1f}%)", ACCENT_COLOR),
            ("Coverage", f"{stats.get('coverage_percent', 100.0):.1f}%", "#475467"),
        ]
        if stats.get("num_small_pieces", 0) > 0:
            results.append(("Small Pieces", f"{stats['num_small_pieces']} !", WARNING_COLOR))
        if stats.get("num_forbidden_pieces", 0) > 0:
            results.append(("Forbidden Pieces", f"{stats['num_forbidden_pieces']} !", ACCENT_COLOR))

        for label, value, color in results:
            ax_info.text(0.08, y_cursor, label, ha="left", va="center",
                         fontsize=11, color="#475467", transform=ax_info.transAxes)
            ax_info.text(0.92, y_cursor, value, ha="right", va="center",
                         fontsize=12, fontweight="bold", color=color,
                         transform=ax_info.transAxes)
            y_cursor -= 0.030

        y_cursor -= 0.008
        ax_info.plot([0.05, 0.95], [y_cursor, y_cursor], color=HEADER_COLOR,
                     linewidth=1.2, transform=ax_info.transAxes, clip_on=False)
        y_cursor -= 0.015

        ax_info.text(0.5, y_cursor, "TILE SPECIFICATION", ha="center", va="top",
                     fontsize=14, fontweight="bold", color=HEADER_COLOR,
                     transform=ax_info.transAxes)
        y_cursor -= 0.035

        full_label = f"Full  —  {format_dim(eff_tile_length)} × {format_dim(eff_tile_width)} mm"
        full_qty = f"× {stats.get('full_tiles', 0)} pcs"
        ax_info.text(0.08, y_cursor, full_label, ha="left", va="center",
                     fontsize=11, color="#475467", transform=ax_info.transAxes)
        ax_info.text(0.92, y_cursor, full_qty, ha="right", va="center",
                     fontsize=12, fontweight="bold", color=SUCCESS_COLOR,
                     transform=ax_info.transAxes)
        y_cursor -= 0.028

        total_cut_pcs = 0
        cut_groups = stats.get("cut_tile_groups", [])
        for i, grp in enumerate(cut_groups):
            grp_id = _cut_group_letter(i)
            shape_type = grp.get("shape_type", "Rect")
            edges_str = grp.get("edges_str", "")

            if shape_type != "Rect" and edges_str:
                formatted_edges = "/".join(format_dim(float(x)) for x in edges_str.split("/"))
                dims = f"{shape_type}: {formatted_edges} mm"
            else:
                dims = f"{format_dim(grp['cut_length_mm'])} × {format_dim(grp['cut_width_mm'])} mm"

            qty_str = f"× {grp['count']} pcs"
            total_cut_pcs += grp["count"]

            label = f"{grp_id}  —  {dims}"
            fs = 8 if len(label) > 36 else (9 if len(label) > 28 else 11)
            ax_info.text(0.08, y_cursor, label, ha="left", va="center",
                         fontsize=fs, color="#475467", transform=ax_info.transAxes)
            ax_info.text(0.92, y_cursor, qty_str, ha="right", va="center",
                         fontsize=12, fontweight="bold", color=ACCENT_COLOR,
                         transform=ax_info.transAxes)
            y_cursor -= 0.028

        y_cursor += 0.008
        ax_info.plot([0.08, 0.92], [y_cursor, y_cursor], color=HEADER_COLOR,
                     linewidth=1.0, transform=ax_info.transAxes, clip_on=False)
        y_cursor -= 0.022
        total_all = stats.get("full_tiles", 0) + total_cut_pcs
        ax_info.text(0.08, y_cursor, "Total", ha="left", va="center",
                     fontsize=12, fontweight="bold", color=HEADER_COLOR, transform=ax_info.transAxes)
        ax_info.text(0.92, y_cursor, f"{total_all} pcs", ha="right", va="center",
                     fontsize=13, fontweight="bold", color=HEADER_COLOR, transform=ax_info.transAxes)
        y_cursor -= 0.028

        if warnings and y_cursor > 0.02:
            y_cursor -= 0.008
            ax_info.plot([0.05, 0.95], [y_cursor, y_cursor], color=WARNING_COLOR,
                         linewidth=1.0, transform=ax_info.transAxes, clip_on=False)
            y_cursor -= 0.010
            for warn_text in warnings[:3]:
                display_text = warn_text[:60] + "..." if len(warn_text) > 60 else warn_text
                ax_info.text(0.08, y_cursor, display_text, ha="left", va="center",
                             fontsize=8, color=WARNING_COLOR, style="italic", transform=ax_info.transAxes)
                y_cursor -= 0.014

        # Row 2: Footer Bar
        ax_footer = fig.add_subplot(gs[2, :])
        ax_footer.axis("off")
        ax_footer.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, transform=ax_footer.transAxes,
            facecolor="#F1F5F9", edgecolor=HEADER_COLOR, linewidth=1.0
        ))
        footer_text = (
            f"DATE: {datetime.now().strftime('%d-%m-%Y')}   |   "
            f"SCALE: NTS   |   "
            f"PROJECT: {project_name.upper()}   |   "
            f"TILE: {tile_length_mm:.2f}×{tile_width_mm:.2f} mm"
        )
        ax_footer.text(0.5, 0.5, footer_text, ha="center", va="center",
                       fontsize=12, color=HEADER_COLOR, transform=ax_footer.transAxes)

        fig.savefig(str(target_path), bbox_inches="tight", dpi=150, facecolor="white")
    finally:
        plt.close(fig)
