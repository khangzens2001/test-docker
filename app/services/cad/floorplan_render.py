"""CAD floorplan renderer for ISO A3 and A4 sheets (PNG & PDF at 300 DPI).

Ported from 3D-Estimate backend/floorplan_generator/render/floorplan.py
with locked differences for headless execution, exact ISO dimensions, and memory safety.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.services.cad.fonts import configure_matplotlib_agg
from app.services.cad.style import SHEET_SIZES, STYLE, WALL_THICKNESS_MM

# Configure headless Agg backend BEFORE pyplot import
configure_matplotlib_agg()

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

logger = logging.getLogger(__name__)


class BaseRenderer:
    def __init__(self, metrics: dict[str, Any], project_name: str = ""):
        self.metrics = metrics or {}
        self.project_name = project_name or str(self.metrics.get("project_name", "") or "")


class FloorPlanRenderer(BaseRenderer):
    """Professional CAD-style technical floor plan renderer."""

    COMPASS_ICON_PATH: Path = (
        Path(__file__).resolve().parents[2] / "static" / "icons" / "compass.png"
    )

    @staticmethod
    def _centroid(pts: list[list[float]] | np.ndarray) -> np.ndarray:
        """Polygon centroid via signed-area formula, mean-point fallback."""
        arr = np.asarray(pts, dtype=float)
        if len(arr) == 0:
            return np.zeros(2)
        a, cx, cy = 0.0, 0.0, 0.0
        for i in range(len(arr)):
            x1, y1 = arr[i]
            x2, y2 = arr[(i + 1) % len(arr)]
            cross = x1 * y2 - x2 * y1
            a += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        if abs(a) < 1e-9:
            return arr.mean(axis=0)
        a *= 0.5
        return np.array([cx / (6.0 * a), cy / (6.0 * a)])

    @staticmethod
    def _format_project_name(project_name: str, max_chars: int = 44) -> str:
        name = " ".join(str(project_name or "").split()) or "UNTITLED"
        name = name.upper()
        if len(name) <= max_chars:
            return name
        return name[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _draw_compass(host_ax: Any, pos: list[float]) -> None:
        """Draw compass asset, with inline vector fallback if asset is missing."""
        cax = host_ax.inset_axes(pos, transform=host_ax.transAxes)
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.set_aspect("equal")
        cax.axis("off")

        try:
            icon_path = FloorPlanRenderer.COMPASS_ICON_PATH
            if icon_path.exists():
                img = mpimg.imread(str(icon_path))
                cax.imshow(img, extent=[0, 1, 0, 1], origin="upper", aspect="equal")
                return
        except Exception as exc:
            logger.debug("Compass icon fallback triggered: %s", exc)

        cax.add_patch(
            mpatches.FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=(1.0, 1.0, 1.0, 0.96),
                edgecolor="#CBD5E1",
                linewidth=0.8,
            )
        )
        cax.add_patch(
            mpatches.Circle(
                (0.5, 0.54), 0.28, facecolor="#F8FAFC", edgecolor="#1F2937", linewidth=0.8
            )
        )
        cax.add_patch(mpatches.Circle((0.5, 0.54), 0.04, facecolor="#1F2937", edgecolor="none"))

        arms = {
            "N": ((0.5, 0.54), (0.5, 0.86), "#CC0000", 12, 1.6),
            "E": ((0.5, 0.54), (0.82, 0.54), "#475467", 9, 1.0),
            "S": ((0.5, 0.54), (0.5, 0.22), "#475467", 9, 1.0),
            "W": ((0.5, 0.54), (0.18, 0.54), "#475467", 9, 1.0),
        }
        lbl_pos = {
            "N": (0.5, 0.93),
            "E": (0.91, 0.54),
            "S": (0.5, 0.12),
            "W": (0.09, 0.54),
        }
        for txt, (s, e, col, sc, lw) in arms.items():
            cax.add_patch(
                mpatches.FancyArrowPatch(
                    s, e, arrowstyle="-|>", mutation_scale=sc, linewidth=lw, color=col
                )
            )
            cax.text(
                lbl_pos[txt][0],
                lbl_pos[txt][1],
                txt,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=col,
                fontfamily="monospace",
            )

    def render(self, output_path: str = "floorplan.pdf", sheet_size: str = "A3") -> None:
        """Render floorplan to output_path at 300 DPI for sheet_size."""
        s = STYLE
        sheet_w, sheet_h = SHEET_SIZES.get(sheet_size, SHEET_SIZES["A3"])
        fig_w, fig_h = sheet_w / 25.4, sheet_h / 25.4

        bb_w = float(self.metrics.get("bbox_width_mm") or 0.0)
        bb_h = float(self.metrics.get("bbox_depth_mm") or 0.0)

        raw_verts = self.metrics.get("vertices_mm", self.metrics.get("vertices", []))
        if raw_verts and len(raw_verts) > 0:
            verts = np.array(raw_verts, dtype=float)
        else:
            verts = np.array([[0.0, 0.0], [bb_w, 0.0], [bb_w, bb_h], [0.0, bb_h]], dtype=float)

        n_verts = len(verts)
        xs, ys = verts[:, 0], verts[:, 1]
        cx_off, cy_off = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
        corners = verts - np.array([cx_off, cy_off])

        if bb_w <= 0.0 and len(verts) > 0:
            bb_w = float(xs.max() - xs.min())
        if bb_h <= 0.0 and len(verts) > 0:
            bb_h = float(ys.max() - ys.min())

        centroid = self._centroid(corners)
        min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
        min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
        margin = max(400.0, bb_w * 0.12)

        proj = self.project_name or self.metrics.get("project_name", "") or "UNTITLED"
        title_project = self._format_project_name(proj)
        perimeter_m = float(self.metrics.get("perimeter_m", 0.0) or 0.0)
        area_m2 = float(self.metrics.get("area_m2", 0.0) or 0.0)
        height_mm = float(self.metrics.get("height_mm", 0.0) or 0.0)
        shape_name = str(self.metrics.get("shape_name", "") or "")

        walls = list(self.metrics.get("walls", self.metrics.get("wall_metrics", [])))
        twa = sum(float(w.get("area_m2", 0.0) or 0.0) for w in walls)

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=300)
        try:
            gs = gridspec.GridSpec(
                2,
                2,
                figure=fig,
                height_ratios=[1, 15],
                width_ratios=[3.5, 1.2],
                wspace=0.03,
                hspace=0.03,
                left=0.02,
                right=0.98,
                top=0.98,
                bottom=0.02,
            )

            # ══════════════════ TITLE BAR ═══════════════════════
            ax_t = fig.add_subplot(gs[0, :])
            ax_t.axis("off")
            ax_t.add_patch(
                mpatches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax_t.transAxes,
                    facecolor="none",
                    edgecolor=s["border"],
                    linewidth=1.5,
                )
            )
            ax_t.text(
                0.5,
                0.67,
                f"TECHNICAL FLOOR PLAN — {title_project}",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="heavy",
                color=s["border"],
                transform=ax_t.transAxes,
            )
            ax_t.text(
                0.5,
                0.25,
                f"ROOM: {bb_w:.0f}×{bb_h:.0f} mm   |   "
                f"HEIGHT: {height_mm:.0f} mm   |   "
                f"FLOOR AREA: {area_m2:.2f} m²",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#333",
                fontfamily="monospace",
                transform=ax_t.transAxes,
            )

            # ══════════════════ PLAN AREA ═══════════════════════
            ax = fig.add_subplot(gs[1, 0])
            for sp in ax.spines.values():
                sp.set_edgecolor(s["border"])
                sp.set_linewidth(1.5)

            # Floor fill
            poly_pts = [tuple(c) for c in corners] + [tuple(corners[0])]
            poly_path = MPath(poly_pts)
            ax.add_patch(
                mpatches.PathPatch(
                    poly_path, facecolor=s["floor_fill"], edgecolor="none", zorder=1
                )
            )

            # Thick walls with hatch
            wt = max(WALL_THICKNESS_MM, bb_w * 0.01)
            for i in range(n_verts):
                c1, c2 = corners[i], corners[(i + 1) % n_verts]
                d = c2 - c1
                length = float(np.linalg.norm(d))
                if length < 1e-9:
                    continue
                dn = d / length
                perp = np.array([-dn[1], dn[0]])
                if np.dot(perp, centroid - (c1 + c2) / 2.0) < 0:
                    perp = -perp
                hatch_perp = perp * wt

                ax.plot(
                    [c1[0], c2[0]],
                    [c1[1], c2[1]],
                    color=s["wall_stroke"],
                    linewidth=5,
                    solid_capstyle="butt",
                    zorder=3,
                )
                pts = [c1, c2, c2 + hatch_perp, c1 + hatch_perp]
                ax.add_patch(
                    mpatches.Polygon(
                        pts,
                        closed=True,
                        facecolor=s["wall_fill"],
                        edgecolor=s["wall_stroke"],
                        linewidth=0.5,
                        hatch=s["hatch_pattern"],
                        zorder=2,
                        alpha=0.4,
                    )
                )

            # Corner dots
            for c in corners:
                ax.plot(c[0], c[1], "o", color="#333", markersize=5, zorder=4)

            # Interior wall-ID badges with leader lines
            for i, w in enumerate(walls):
                if i >= n_verts:
                    break
                c1, c2 = corners[i], corners[(i + 1) % n_verts]
                mid = (c1 + c2) / 2.0
                d = c2 - c1
                ln = float(np.linalg.norm(d))
                if ln < 1e-9:
                    continue
                tangent = d / ln
                perp = np.array([-tangent[1], tangent[0]])
                if np.dot(perp, centroid - mid) < 0:
                    perp = -perp
                offset = max(120.0, min(max(bb_w, bb_h) * 0.08, ln * 0.3))
                label_pos = mid + perp * offset

                ax.plot(
                    [mid[0], label_pos[0]],
                    [mid[1], label_pos[1]],
                    color=s["text_mid"],
                    linewidth=0.9,
                    alpha=0.75,
                    zorder=6,
                )
                ax.scatter(
                    [mid[0]],
                    [mid[1]],
                    s=18,
                    facecolor=s["wall_stroke"],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=7,
                )
                wall_id = str(w.get("id", f"W{i}"))
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    wall_id,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=s["badge_border"],
                    fontfamily="monospace",
                    bbox=dict(
                        boxstyle="round,pad=0.35,rounding_size=0.3",
                        facecolor=s["badge_bg"],
                        edgecolor=s["badge_border"],
                        linewidth=1.1,
                    ),
                    zorder=8,
                )

            # Exterior dimension labels (red)
            for i, w in enumerate(walls):
                if i >= n_verts:
                    break
                c1, c2 = corners[i], corners[(i + 1) % n_verts]
                mid = (c1 + c2) / 2.0
                edge_mm = float(w.get("length_mm", 0.0) or 0.0)
                if edge_mm < 80.0:
                    continue
                d = c2 - c1
                dn = d / (np.linalg.norm(d) + 1e-9)
                perp = np.array([-dn[1], dn[0]])
                if np.dot(perp, centroid - mid) > 0:
                    perp = -perp
                off = max(150.0, bb_w * 0.06)
                lbl = mid + perp * off

                ax.annotate(
                    f"{edge_mm:.0f}",
                    xy=mid,
                    xytext=lbl,
                    fontsize=9,
                    fontweight="bold",
                    color=s["dim_wall_color"],
                    ha="center",
                    va="center",
                    fontfamily="monospace",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor=s["dim_wall_color"],
                        linewidth=0.8,
                    ),
                    arrowprops=dict(
                        arrowstyle="-", color=s["dim_wall_color"], linewidth=0.8
                    ),
                    zorder=9,
                )

            # Bounding-box dimensions (blue arrows)
            ax.annotate(
                "",
                xy=(min_x, min_y - margin * 0.55),
                xytext=(max_x, min_y - margin * 0.55),
                arrowprops=dict(
                    arrowstyle="<->", color=s["dim_bbox_color"], linewidth=1.5
                ),
            )
            ax.text(
                (min_x + max_x) / 2.0,
                min_y - margin * 0.72,
                f"{bb_w:.0f} mm",
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color=s["dim_bbox_color"],
                fontfamily="monospace",
            )

            ax.annotate(
                "",
                xy=(max_x + margin * 0.55, min_y),
                xytext=(max_x + margin * 0.55, max_y),
                arrowprops=dict(
                    arrowstyle="<->", color=s["dim_bbox_color"], linewidth=1.5
                ),
            )
            ax.text(
                max_x + margin * 0.72,
                (min_y + max_y) / 2.0,
                f"{bb_h:.0f} mm",
                ha="left",
                va="center",
                rotation=90,
                fontsize=11,
                fontweight="bold",
                color=s["dim_bbox_color"],
                fontfamily="monospace",
            )

            # Center label
            center_proj = proj.upper()
            ax.text(
                centroid[0],
                centroid[1],
                f"{center_proj}\n"
                f"AREA: {area_m2:.2f} m²\n"
                f"HEIGHT: {height_mm:.0f} mm",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=s["text_dark"],
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=10,
            )

            # Axes config
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.12, linewidth=0.4, color="#888888")
            ax.set_xlabel("X (mm)", fontsize=10, color="#555555")
            ax.set_ylabel("Y (mm)", fontsize=10, color="#555555")
            ax.set_xlim(min_x - margin * 1.4, max_x + margin * 1.4)
            ax.set_ylim(min_y - margin * 1.4, max_y + margin * 1.4)

            # Legend
            legend_h = [
                mpatches.Patch(
                    facecolor=s["floor_fill"], edgecolor=s["border"], label="Floor"
                ),
                mlines.Line2D([], [], color=s["wall_stroke"], linewidth=4, label="Wall"),
                mlines.Line2D(
                    [],
                    [],
                    color=s["badge_border"],
                    linewidth=0,
                    marker="o",
                    markersize=7,
                    markerfacecolor=s["badge_bg"],
                    markeredgecolor=s["badge_border"],
                    label="Wall ID",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color=s["dim_wall_color"],
                    linewidth=1.2,
                    label="Dimension (mm)",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color=s["dim_bbox_color"],
                    linewidth=1.2,
                    label="BBox (mm)",
                ),
            ]
            leg = ax.legend(
                handles=legend_h,
                loc="lower left",
                fontsize=8,
                framealpha=1.0,
                edgecolor=s["border"],
                title="LEGEND",
            )
            leg.get_title().set_fontweight("bold")
            leg.get_frame().set_linewidth(1.0)

            # ══════════════════ INFO PANEL ══════════════════════
            ax_i = fig.add_subplot(gs[1, 1])
            ax_i.axis("off")
            ax_i.add_patch(
                mpatches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax_i.transAxes,
                    facecolor=s["info_bg"],
                    edgecolor=s["border"],
                    linewidth=1.5,
                )
            )

            # Section 1: Summary Card
            ax_i.text(
                0.5,
                0.97,
                "WALL METRICS",
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                color=s["text_dark"],
                transform=ax_i.transAxes,
            )
            ax_i.plot(
                [0, 1],
                [0.94, 0.94],
                color=s["border"],
                linewidth=1.2,
                transform=ax_i.transAxes,
                clip_on=False,
            )

            m_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
            m_labels = ["Height", "Floor Area", "Wall Area", "Perimeter"]
            m_vals = [
                f"{height_mm / 1000.0:.2f} m",
                f"{area_m2:.2f} m²",
                f"{twa:.2f} m²",
                f"{perimeter_m:.2f} m",
            ]

            y0, dy = 0.88, 0.05
            for k in range(4):
                yy = y0 - k * dy
                ax_i.add_patch(
                    mpatches.Rectangle(
                        (0.05, yy - 0.015),
                        0.015,
                        0.03,
                        facecolor=m_colors[k],
                        edgecolor="none",
                        transform=ax_i.transAxes,
                    )
                )
                ax_i.text(
                    0.08,
                    yy,
                    m_labels[k],
                    ha="left",
                    va="center",
                    fontsize=10,
                    color=s["text_dark"],
                    transform=ax_i.transAxes,
                )
                ax_i.text(
                    0.48,
                    yy,
                    ":",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=s["text_dark"],
                    transform=ax_i.transAxes,
                )
                ax_i.text(
                    0.53,
                    yy,
                    m_vals[k],
                    ha="left",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=s["text_dark"],
                    fontfamily="monospace",
                    transform=ax_i.transAxes,
                )

            # Shape badge
            if shape_name:
                ax_i.text(
                    0.5,
                    0.65,
                    f" {shape_name} ",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    fontfamily="monospace",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=s["badge_border"],
                        edgecolor="none",
                    ),
                    transform=ax_i.transAxes,
                )

            ax_i.plot(
                [0, 1],
                [0.60, 0.60],
                color=s["border"],
                linewidth=1.2,
                transform=ax_i.transAxes,
                clip_on=False,
            )

            # Section 2: Wall Table
            rows = []
            for idx, w in enumerate(walls):
                if idx >= 10:
                    rows.append(["…", "…", "…", "…", "…"])
                    break
                l_m = float(w.get("length_mm", 0.0) or 0.0) / 1000.0
                a = float(w.get("area_m2", 0.0) or 0.0)
                pct = (l_m / perimeter_m) * 100 if perimeter_m > 0 else 0
                w_dir = str(w.get("direction", w.get("dir", "Hor")))
                rows.append([str(w.get("id", f"W{idx}")), w_dir, f"{l_m:.2f}", f"{a:.2f}", f"{pct:.1f}%"])

            rows.append(["TOTAL", "-", f"{perimeter_m:.2f}", f"{twa:.2f}", "100%"])

            cols = ["ID", "Dir", "Len (m)", "Area (m²)", "%"]
            tbl = ax_i.table(
                cellText=rows,
                colLabels=cols,
                loc="center",
                bbox=[0.02, 0.35, 0.96, 0.23],
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.set_zorder(10)

            for (r, _c), cell in tbl.get_celld().items():
                cell.set_text_props(fontfamily="monospace")
                cell.set_edgecolor(s["border"])
                cell.set_linewidth(0.5)
                if r == 0:
                    cell.set_text_props(fontweight="bold")
                    cell.set_facecolor(s["table_header_bg"])
                elif r == len(rows):
                    cell.set_text_props(fontweight="bold")
                    cell.set_facecolor("#E5E7EB")
                elif r % 2 == 1:
                    cell.set_facecolor("#F9FAFB")
                else:
                    cell.set_facecolor("white")

            ax_i.plot(
                [0, 1],
                [0.33, 0.33],
                color=s["border"],
                linewidth=1.2,
                transform=ax_i.transAxes,
                clip_on=False,
            )

            # Section 3: Wall Length Bar Chart
            bar_ax = ax_i.inset_axes([0.15, 0.17, 0.8, 0.14], transform=ax_i.transAxes)
            bar_ax.set_facecolor("none")
            for spine in bar_ax.spines.values():
                spine.set_visible(False)
            bar_ax.tick_params(
                axis="both", which="both", length=0, labelsize=7, colors=s["text_mid"]
            )

            walls_to_plot = walls[:10]
            if len(walls_to_plot) > 0:
                y_pos = np.arange(len(walls_to_plot))
                lengths = [float(w.get("length_mm", 0.0) or 0.0) / 1000.0 for w in walls_to_plot]
                colors = [
                    "#3B82F6"
                    if str(w.get("direction", w.get("dir", "Hor"))) == "Hor"
                    else "#F59E0B"
                    for w in walls_to_plot
                ]
                labels = [str(w.get("id", f"W{idx}")) for idx, w in enumerate(walls_to_plot)]

                bar_ax.barh(y_pos, lengths, align="center", color=colors, alpha=0.8, height=0.6)
                bar_ax.set_yticks(y_pos)
                bar_ax.set_yticklabels(labels, fontfamily="monospace", fontweight="bold")
                bar_ax.invert_yaxis()

            bar_ax.set_xlabel("Length (m)", fontsize=7, color=s["text_mid"], labelpad=1)
            bar_ax.xaxis.grid(True, linestyle="--", alpha=0.3, color=s["text_mid"])
            bar_ax.set_axisbelow(True)

            ax_i.plot(
                [0, 1],
                [0.14, 0.14],
                color=s["border"],
                linewidth=1.2,
                transform=ax_i.transAxes,
                clip_on=False,
            )

            # Section 4: Notes & Compass
            ax_i.text(
                0.05,
                0.12,
                "Note:",
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                color=s["text_dark"],
                transform=ax_i.transAxes,
            )
            ax_i.text(
                0.05,
                0.095,
                "1. Dims in mm.\n2. Scan auto-aligned.",
                ha="left",
                va="top",
                fontsize=8,
                color=s["text_dark"],
                transform=ax_i.transAxes,
            )

            # Compass
            self._draw_compass(ax_i, pos=[0.70, 0.015, 0.25, 0.13])

            # Section 5: Sheet Info Box
            now = datetime.now().strftime("%d-%m-%Y")
            info = f"SHEET: {sheet_size} ({sheet_w:.0f}×{sheet_h:.0f} mm)\nDATE: {now}"
            ax_i.add_patch(
                mpatches.Rectangle(
                    (0.05, 0.02),
                    0.6,
                    0.045,
                    transform=ax_i.transAxes,
                    facecolor="white",
                    edgecolor=s["border"],
                    linewidth=1.0,
                )
            )
            ax_i.text(
                0.08,
                0.05,
                info,
                ha="left",
                va="top",
                fontsize=7,
                color=s["text_dark"],
                fontfamily="monospace",
                transform=ax_i.transAxes,
            )

            # Save at 300 DPI without bbox_inches="tight" to preserve ISO sheet dimensions
            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(target_path), dpi=300, facecolor="white", pad_inches=0)
        finally:
            plt.close(fig)


def render_floorplan(
    metrics_mm: dict[str, Any], output_path: str, sheet_size: str = "A3"
) -> None:
    """Render canonical metrics_mm layout to CAD floorplan sheet (PNG or PDF)."""
    renderer = FloorPlanRenderer(metrics_mm)
    renderer.render(output_path=output_path, sheet_size=sheet_size)
