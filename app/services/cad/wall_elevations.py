"""CAD wall elevations renderer (PNG).

Ported from 3D-Estimate backend/floorplan_generator/render/floorplan.py
with locked differences for headless execution, memory safety, and canonical contract support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services.cad.fonts import configure_matplotlib_agg
from app.services.cad.style import STYLE

# Configure headless Agg backend BEFORE pyplot import
configure_matplotlib_agg()

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class WallElevationsRenderer:
    """Professional CAD-style technical wall elevations renderer."""

    def __init__(self, metrics: dict[str, Any] | Any, project_name: str = ""):
        self.metrics = metrics or {}
        if isinstance(self.metrics, dict):
            self.project_name = project_name or str(self.metrics.get("project_name", "") or "")
        else:
            self.project_name = project_name or str(getattr(self.metrics, "project_name", "") or "")

    def render(self, output_path: str = "elevations.png") -> None:
        """Render all wall elevation views and summary table to PNG at 200 DPI."""
        s = STYLE
        if isinstance(self.metrics, dict):
            raw_walls = self.metrics.get("walls", self.metrics.get("wall_metrics", []))
            wall_metrics = list(raw_walls or [])
            wall_count = int(self.metrics.get("wall_count", len(wall_metrics)) or len(wall_metrics))
            height_mm = float(self.metrics.get("height_mm", 0.0) or 0.0)
            perimeter_m = float(self.metrics.get("perimeter_m", 0.0) or 0.0)
        else:
            wall_metrics = list(
                getattr(self.metrics, "wall_metrics", getattr(self.metrics, "walls", [])) or []
            )
            wall_count = int(
                getattr(self.metrics, "wall_count", len(wall_metrics)) or len(wall_metrics)
            )
            height_mm = float(getattr(self.metrics, "height_mm", 0.0) or 0.0)
            perimeter_m = float(getattr(self.metrics, "perimeter_m", 0.0) or 0.0)

        n = max(wall_count, len(wall_metrics))
        cols = 3
        rows = max(1, (n + cols - 1) // cols)

        fig = plt.figure(figsize=(cols * 6 + 5, max(rows * 6, 10)), facecolor=s["bg_white"])
        try:
            gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[cols * 6, 5], wspace=0.05)

            # --- Left Side: Wall Grid ---
            gs_walls = gridspec.GridSpecFromSubplotSpec(
                rows, cols, subplot_spec=gs[0], wspace=0.3, hspace=0.3
            )

            all_portals = []
            if isinstance(self.metrics, dict):
                all_portals = list(self.metrics.get("portals", []) or [])
            else:
                all_portals = list(getattr(self.metrics, "portals", []) or [])

            for i, w in enumerate(wall_metrics):
                ax = fig.add_subplot(gs_walls[i // cols, i % cols])
                if isinstance(w, dict):
                    w_id = str(w.get("id", f"W{i}"))
                    w_dir = str(w.get("dir", w.get("direction", "Hor")))
                    length = float(w.get("length_mm", 0.0) or 0.0)
                    area_m2 = float(w.get("area_m2", 0.0) or 0.0)
                else:
                    w_id = str(getattr(w, "id", f"W{i}"))
                    w_dir = str(getattr(w, "dir", getattr(w, "direction", "Hor")))
                    length = float(getattr(w, "length_mm", 0.0) or 0.0)
                    area_m2 = float(getattr(w, "area_m2", 0.0) or 0.0)

                ax.add_patch(
                    mpatches.Rectangle(
                        (0, 0),
                        length,
                        height_mm,
                        linewidth=2,
                        edgecolor="black",
                        facecolor=s["wall_fill"],
                    )
                )

                # Render portals on wall i
                wall_portals = []
                if isinstance(w, dict):
                    wall_portals.extend(w.get("portals", []) or [])
                else:
                    wall_portals.extend(getattr(w, "portals", []) or [])

                for p in all_portals:
                    p_w_idx = p.get("wall_index")
                    p_w_id = str(p.get("wall_id", "")).lower()
                    if (p_w_idx is not None and int(p_w_idx) == i) or (p_w_id and p_w_id == w_id.lower()):
                        if p not in wall_portals:
                            wall_portals.append(p)

                wall_openings_m2 = 0.0
                for p in wall_portals:
                    if "offset_start_mm" in p:
                        u0_mm = float(p["offset_start_mm"])
                    elif "offset_along_wall_m" in p:
                        u0_mm = float(p["offset_along_wall_m"]) * 1000.0
                    elif "offset_along_wall_mm" in p:
                        u0_mm = float(p["offset_along_wall_mm"])
                    else:
                        u0_mm = 0.0

                    if "width_mm" in p:
                        w_mm = float(p["width_mm"])
                    elif "width_m" in p:
                        w_mm = float(p["width_m"]) * 1000.0
                    else:
                        w_mm = 900.0

                    if "height_mm" in p:
                        h_mm = float(p["height_mm"])
                    elif "height_m" in p:
                        h_mm = float(p["height_m"]) * 1000.0
                    else:
                        h_mm = 2100.0

                    if "sill_height_mm" in p:
                        sill_mm = float(p["sill_height_mm"])
                    elif "sill_height_m" in p:
                        sill_mm = float(p["sill_height_m"]) * 1000.0
                    else:
                        sill_mm = 0.0

                    p_type = str(p.get("type", p.get("kind", "door"))).lower()
                    wall_openings_m2 += (w_mm * h_mm) / 1e6

                    if p_type == "door" or sill_mm < 350.0:
                        sill_mm = 0.0
                        ax.add_patch(
                            mpatches.Rectangle(
                                (u0_mm, sill_mm),
                                w_mm,
                                h_mm,
                                linewidth=1.8,
                                edgecolor="#D97706",
                                facecolor="#FEF3C7",
                                zorder=3,
                            )
                        )
                        ax.text(
                            u0_mm + w_mm / 2.0,
                            sill_mm + h_mm / 2.0,
                            f"DOOR\n{w_mm/1000.0:.2f}×{h_mm/1000.0:.2f}m",
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="#92400E",
                            zorder=4,
                        )
                    else:
                        ax.add_patch(
                            mpatches.Rectangle(
                                (u0_mm, sill_mm),
                                w_mm,
                                h_mm,
                                linewidth=1.8,
                                edgecolor="#0284C7",
                                facecolor="#E0F2FE",
                                zorder=3,
                            )
                        )
                        ax.text(
                            u0_mm + w_mm / 2.0,
                            sill_mm + h_mm / 2.0,
                            f"WINDOW\n{w_mm/1000.0:.2f}×{h_mm/1000.0:.2f}m",
                            ha="center",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="#0369A1",
                            zorder=4,
                        )
                        if sill_mm > 0:
                            ax.text(
                                u0_mm + w_mm / 2.0,
                                max(0.0, sill_mm - height_mm * 0.04),
                                f"Sill: {sill_mm/1000.0:.2f}m",
                                ha="center",
                                va="top",
                                fontsize=6,
                                color="#0284C7",
                                zorder=4,
                            )

                xlim_min = -length * 0.2 if length > 0 else -100.0
                xlim_max = length * 1.2 if length > 0 else 100.0
                ylim_min = -height_mm * 0.1 if height_mm > 0 else -100.0
                ylim_max = height_mm * 1.1 if height_mm > 0 else 100.0
                ax.set_xlim(xlim_min, xlim_max)
                ax.set_ylim(ylim_min, ylim_max)
                ax.set_aspect("equal")
                ax.set_title(
                    f"{w_id} ({w_dir}) — {length:.0f}×{height_mm:.0f} mm",
                    fontweight="bold",
                )
                net_area_m2 = max(0.0, area_m2 - wall_openings_m2)
                area_text = (
                    f"Gross: {area_m2:.2f} m²\nNet: {net_area_m2:.2f} m²"
                    if wall_openings_m2 > 0
                    else f"Area: {area_m2:.2f} m²"
                )
                ax.text(
                    length / 2.0,
                    height_mm / 2.0,
                    area_text,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        edgecolor=s["border"],
                        boxstyle="round,pad=0.5",
                    ),
                    zorder=5,
                )

            # --- Right Side: Info Panel ---
            ax_i = fig.add_subplot(gs[1])
            ax_i.axis("off")
            ax_i.add_patch(
                mpatches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax_i.transAxes,
                    facecolor="white",
                    edgecolor=s["border"],
                    linewidth=1.0,
                )
            )

            # 1. Title
            ax_i.text(
                0.5,
                0.95,
                "WALL ELEVATIONS METRICS",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=s["border"],
                transform=ax_i.transAxes,
            )
            ax_i.plot(
                [0, 1],
                [0.91, 0.91],
                color=s["border"],
                linewidth=1.2,
                transform=ax_i.transAxes,
                clip_on=False,
            )

            # 2. Summary Metrics
            metrics_y = 0.85
            spacing = 0.07

            def draw_metric(y: float, color: str, label: str, val_str: str) -> None:
                ax_i.add_patch(
                    mpatches.Rectangle(
                        (0.05, y - 0.02),
                        0.01,
                        0.04,
                        facecolor=color,
                        edgecolor="none",
                        transform=ax_i.transAxes,
                    )
                )
                ax_i.text(
                    0.08,
                    y,
                    label,
                    ha="left",
                    va="center",
                    fontsize=10,
                    transform=ax_i.transAxes,
                )
                ax_i.text(
                    0.48,
                    y,
                    ":",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax_i.transAxes,
                )
                ax_i.text(
                    0.52,
                    y,
                    val_str,
                    ha="left",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    fontfamily="monospace",
                    transform=ax_i.transAxes,
                )

            twa = sum(
                float(
                    w.get("area_m2", 0.0)
                    if isinstance(w, dict)
                    else getattr(w, "area_m2", 0.0) or 0.0
                )
                for w in wall_metrics
            )
            draw_metric(metrics_y, "#3B82F6", "Height", f"{height_mm / 1000.0:.2f} m")
            draw_metric(metrics_y - spacing, "#F59E0B", "Wall Area", f"{twa:.2f} m²")
            draw_metric(metrics_y - spacing * 2, "#EF4444", "Total Length", f"{perimeter_m:.2f} m")
            draw_metric(metrics_y - spacing * 3, "#10B981", "Walls Count", f"{len(wall_metrics)}")

            ax_i.plot(
                [0, 1],
                [0.58, 0.58],
                color=s["border"],
                linewidth=1.2,
                transform=ax_i.transAxes,
                clip_on=False,
            )

            # 3. Table
            rows_data: list[list[str]] = []
            for idx, w in enumerate(wall_metrics):
                if idx >= 15:
                    rows_data.append(["…", "…", "…", "…", "…"])
                    break
                if isinstance(w, dict):
                    w_id = str(w.get("id", f"W{idx}"))
                    w_dir = str(w.get("dir", w.get("direction", "Hor")))
                    l_m = float(w.get("length_mm", 0.0) or 0.0) / 1000.0
                    a = float(w.get("area_m2", 0.0) or 0.0)
                else:
                    w_id = str(getattr(w, "id", f"W{idx}"))
                    w_dir = str(getattr(w, "dir", getattr(w, "direction", "Hor")))
                    l_m = float(getattr(w, "length_mm", 0.0) or 0.0) / 1000.0
                    a = float(getattr(w, "area_m2", 0.0) or 0.0)

                pct = (a / twa) * 100.0 if twa > 0.0 else 0.0
                rows_data.append([w_id, w_dir, f"{l_m:.2f}", f"{a:.2f}", f"{pct:.1f}%"])

            rows_data.append(["TOTAL", "-", f"{perimeter_m:.2f}", f"{twa:.2f}", "100%"])

            cols_headers = ["ID", "Dir", "Len (m)", "Area (m²)", "%"]
            tbl = ax_i.table(
                cellText=rows_data,
                colLabels=cols_headers,
                loc="center",
                bbox=[0.02, 0.35, 0.96, 0.20],
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
                elif r == len(rows_data):
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

            # 4. Bar chart
            bar_ax = ax_i.inset_axes([0.15, 0.12, 0.8, 0.17], transform=ax_i.transAxes)
            bar_ax.set_facecolor("none")
            for spine in bar_ax.spines.values():
                spine.set_visible(False)
            bar_ax.tick_params(
                axis="both", which="both", length=0, labelsize=7, colors=s["text_mid"]
            )

            walls_to_plot = wall_metrics[:10]
            if walls_to_plot:
                y_pos = np.arange(len(walls_to_plot))
                areas = [
                    float(
                        w.get("area_m2", 0.0)
                        if isinstance(w, dict)
                        else getattr(w, "area_m2", 0.0) or 0.0
                    )
                    for w in walls_to_plot
                ]
                colors = [
                    (
                        "#3B82F6"
                        if (
                            w.get("dir", w.get("direction", "Hor"))
                            if isinstance(w, dict)
                            else getattr(w, "dir", getattr(w, "direction", "Hor"))
                        )
                        == "Hor"
                        else "#F59E0B"
                    )
                    for w in walls_to_plot
                ]
                labels = [
                    str(
                        w.get("id", f"W{idx}")
                        if isinstance(w, dict)
                        else getattr(w, "id", f"W{idx}")
                    )
                    for idx, w in enumerate(walls_to_plot)
                ]

                bar_ax.barh(y_pos, areas, align="center", color=colors, alpha=0.8, height=0.6)
                bar_ax.set_yticks(y_pos)
                bar_ax.set_yticklabels(labels, fontfamily="monospace")
                bar_ax.invert_yaxis()
                bar_ax.set_xlabel("Area (m²)", fontsize=7, color=s["text_mid"])
                bar_ax.xaxis.grid(True, linestyle="--", alpha=0.5)

            ax_i.plot(
                [0, 1],
                [0.08, 0.08],
                color=s["border"],
                linewidth=1.2,
                transform=ax_i.transAxes,
                clip_on=False,
            )

            # 5. Legend/Note
            ax_i.text(
                0.05,
                0.05,
                "Note:",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                transform=ax_i.transAxes,
            )
            ax_i.text(
                0.05,
                0.03,
                "Hor = Horizontal (Blue)\nVer = Vertical (Orange)",
                ha="left",
                va="top",
                fontsize=8,
                color=s["text_mid"],
                transform=ax_i.transAxes,
            )

            # Main Title
            proj_title = self.project_name or "UNTITLED"
            fig.suptitle(
                f"WALL ELEVATIONS — {proj_title.upper()}",
                fontsize=22,
                fontweight="bold",
                y=0.96,
            )
            plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)

            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                str(target_path),
                dpi=200,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
        finally:
            plt.close(fig)


def render_wall_elevations(metrics_mm: dict[str, Any], output_path: str) -> None:
    """Render canonical metrics_mm layout to wall elevations PNG."""
    renderer = WallElevationsRenderer(metrics_mm)
    renderer.render(output_path=output_path)
