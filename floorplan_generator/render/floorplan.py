import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.path import Path as MPath
from datetime import datetime
from pathlib import Path

from floorplan_generator.config import STYLE, SHEET_SIZES, WALL_THICKNESS_MM


class BaseRenderer:
    def __init__(self, metrics, project_name="PROJECT"):
        self.metrics = metrics
        self.project_name = project_name


class FloorPlanRenderer(BaseRenderer):
    """Professional CAD-style technical floor plan renderer."""

    @classmethod
    def _find_compass_icon(cls) -> Path:
        candidates = [
            Path(__file__).resolve().parents[2] / "icon" / "compass.png",
            Path(__file__).resolve().parents[1] / "icon" / "compass.png",
            Path("/app/vggt-space/icon/compass.png"),
            Path("/app/icon/compass.png"),
            Path("icon/compass.png"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]

    COMPASS_ICON_PATH = _find_compass_icon.__func__(None)

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _centroid(pts):
        """Polygon centroid via the signed-area formula, mean-point fallback."""
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
        return np.array([cx / (6 * a), cy / (6 * a)])

    @staticmethod
    def _format_project_name(project_name, max_chars=44):
        name = " ".join(str(project_name or "").split()) or "UNTITLED"
        name = name.upper()
        if len(name) <= max_chars:
            return name
        return name[:max_chars - 3].rstrip() + "..."

    # ── main render ─────────────────────────────────────────

    def render(self, sheet_size="A3", output_path="floorplan.pdf"):
        print(f"[4/5] Rendering floor plan (sheet: {sheet_size})...")

        S = STYLE  # shorthand
        sheet_w, sheet_h = SHEET_SIZES.get(sheet_size, SHEET_SIZES["A3"])
        fig_w, fig_h = sheet_w / 25.4, sheet_h / 25.4

        # ── polygon data ────────────────────────────────────
        verts = np.array(self.metrics.vertices, dtype=float)
        N = len(verts)
        xs, ys = verts[:, 0], verts[:, 1]
        cx_off, cy_off = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        corners = verts - np.array([cx_off, cy_off])

        bb_w, bb_h = self.metrics.bbox_dims
        centroid = self._centroid(corners)
        min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
        min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
        margin = max(400, bb_w * 0.12)

        proj = self.project_name or "UNTITLED"
        title_project = self._format_project_name(proj)
        perimeter_m = self.metrics.perimeter_m
        area_m2 = self.metrics.area_m2
        height_mm = self.metrics.height_mm
        twa = sum(w["area_m2"] for w in self.metrics.wall_metrics)

        # ── figure & grid ───────────────────────────────────
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=300)
        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            height_ratios=[1, 15], width_ratios=[3.5, 1.2],
            wspace=0.03, hspace=0.03,
            left=0.02, right=0.98, top=0.98, bottom=0.02,
        )

        # ══════════════════ TITLE BAR ═══════════════════════
        ax_t = fig.add_subplot(gs[0, :])
        ax_t.axis("off")
        ax_t.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, transform=ax_t.transAxes,
            facecolor="none", edgecolor=S["border"], linewidth=1.5))
        ax_t.text(0.5, 0.67,
                  f"TECHNICAL FLOOR PLAN — {title_project}",
                  ha="center", va="center", fontsize=17,
                  fontweight="heavy", color=S["border"],
                  fontfamily="DejaVu Sans", transform=ax_t.transAxes)
        ax_t.text(0.5, 0.25,
                  f"ROOM: {bb_w:.0f}×{bb_h:.0f} mm   |   "
                  f"HEIGHT: {height_mm:.0f} mm   |   "
                  f"FLOOR AREA: {area_m2:.2f} m²",
                  ha="center", va="center", fontsize=11,
                  fontweight="bold", color="#333", fontfamily="monospace",
                  transform=ax_t.transAxes)

        # ══════════════════ PLAN AREA ═══════════════════════
        ax = fig.add_subplot(gs[1, 0])
        for sp in ax.spines.values():
            sp.set_edgecolor(S["border"])
            sp.set_linewidth(1.5)

        # floor fill
        poly_pts = [tuple(c) for c in corners] + [tuple(corners[0])]
        poly_path = MPath(poly_pts)
        ax.add_patch(mpatches.PathPatch(
            poly_path, facecolor=S["floor_fill"], edgecolor="none", zorder=1))

        # ── thick walls with hatch ──────────────────────────
        wt = max(WALL_THICKNESS_MM, bb_w * 0.01)
        for i in range(N):
            c1, c2 = corners[i], corners[(i + 1) % N]
            d = c2 - c1
            length = np.linalg.norm(d)
            if length < 1e-9:
                continue
            dn = d / length
            perp = np.array([-dn[1], dn[0]])
            # point perp inward (toward centroid)
            if np.dot(perp, centroid - (c1 + c2) / 2) < 0:
                perp = -perp
            hatch_perp = perp * wt

            # thick wall line
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                    color=S["wall_stroke"], linewidth=5,
                    solid_capstyle="butt", zorder=3)
            # hatch polygon
            pts = [c1, c2, c2 + hatch_perp, c1 + hatch_perp]
            ax.add_patch(mpatches.Polygon(
                pts, closed=True,
                facecolor=S["wall_fill"], edgecolor=S["wall_stroke"],
                linewidth=0.5, hatch=S["hatch_pattern"],
                zorder=2, alpha=0.4))

        # ── corner dots ─────────────────────────────────────
        for c in corners:
            ax.plot(c[0], c[1], "o", color="#333", markersize=5, zorder=4)

        # ── interior wall-ID badges with leader lines ───────
        for i, w in enumerate(self.metrics.wall_metrics):
            c1, c2 = corners[i], corners[(i + 1) % N]
            mid = (c1 + c2) / 2
            d = c2 - c1
            ln = np.linalg.norm(d)
            if ln < 1e-9:
                continue
            tangent = d / ln
            perp = np.array([-tangent[1], tangent[0]])
            if np.dot(perp, centroid - mid) < 0:
                perp = -perp
            offset = max(120, min(max(bb_w, bb_h) * 0.08, ln * 0.3))
            label_pos = mid + perp * offset

            # leader line + dot
            ax.plot([mid[0], label_pos[0]], [mid[1], label_pos[1]],
                    color=S["text_mid"], linewidth=0.9, alpha=0.75, zorder=6)
            ax.scatter([mid[0]], [mid[1]], s=18,
                       facecolor=S["wall_stroke"], edgecolor="white",
                       linewidth=0.4, zorder=7)
            # badge
            ax.text(label_pos[0], label_pos[1], w["id"],
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=S["badge_border"], fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.3",
                              facecolor=S["badge_bg"],
                              edgecolor=S["badge_border"], linewidth=1.1),
                    zorder=8)

        # ── exterior dimension labels (red) ─────────────────
        for i, w in enumerate(self.metrics.wall_metrics):
            c1, c2 = corners[i], corners[(i + 1) % N]
            mid = (c1 + c2) / 2
            edge_mm = w["length_mm"]
            if edge_mm < 80:
                continue
            d = c2 - c1
            dn = d / (np.linalg.norm(d) + 1e-9)
            perp = np.array([-dn[1], dn[0]])
            if np.dot(perp, centroid - mid) > 0:
                perp = -perp  # point outward
            off = max(150, bb_w * 0.06)
            lbl = mid + perp * off

            ax.annotate(
                f"{edge_mm:.0f}",
                xy=mid, xytext=lbl,
                fontsize=9, fontweight="bold", color=S["dim_wall_color"],
                ha="center", va="center", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor=S["dim_wall_color"],
                          linewidth=0.8),
                arrowprops=dict(arrowstyle="-", color=S["dim_wall_color"],
                                linewidth=0.8),
                zorder=9)

        # ── bounding-box dimensions (blue arrows) ──────────
        ax.annotate("", xy=(min_x, min_y - margin * 0.55),
                    xytext=(max_x, min_y - margin * 0.55),
                    arrowprops=dict(arrowstyle="<->",
                                   color=S["dim_bbox_color"], linewidth=1.5))
        ax.text((min_x + max_x) / 2, min_y - margin * 0.72,
                f"{bb_w:.0f} mm", ha="center", va="top",
                fontsize=11, fontweight="bold", color=S["dim_bbox_color"],
                fontfamily="monospace")

        ax.annotate("", xy=(max_x + margin * 0.55, min_y),
                    xytext=(max_x + margin * 0.55, max_y),
                    arrowprops=dict(arrowstyle="<->",
                                   color=S["dim_bbox_color"], linewidth=1.5))
        ax.text(max_x + margin * 0.72, (min_y + max_y) / 2,
                f"{bb_h:.0f} mm", ha="left", va="center", rotation=90,
                fontsize=11, fontweight="bold", color=S["dim_bbox_color"],
                fontfamily="monospace")

        # ── center label ────────────────────────────────────
        ax.text(centroid[0], centroid[1],
                f"{proj.upper()}\n"
                f"AREA: {area_m2:.2f} m²\n"
                f"HEIGHT: {height_mm:.0f} mm",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=S["text_dark"],
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=10)

        # ── axes config ─────────────────────────────────────
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.12, linewidth=0.4, color="#888")
        ax.set_xlabel("X (mm)", fontsize=10, color="#555")
        ax.set_ylabel("Y (mm)", fontsize=10, color="#555")
        ax.set_xlim(min_x - margin * 1.4, max_x + margin * 1.4)
        ax.set_ylim(min_y - margin * 1.4, max_y + margin * 1.4)

        # ── legend ──────────────────────────────────────────
        legend_h = [
            mpatches.Patch(facecolor=S["floor_fill"], edgecolor=S["border"],
                           label="Floor"),
            mlines.Line2D([], [], color=S["wall_stroke"], linewidth=4,
                          label="Wall"),
            mlines.Line2D([], [], color=S["badge_border"], linewidth=0,
                          marker="o", markersize=7,
                          markerfacecolor=S["badge_bg"],
                          markeredgecolor=S["badge_border"],
                          label="Wall ID"),
            mlines.Line2D([], [], color=S["dim_wall_color"], linewidth=1.2,
                          label="Dimension (mm)"),
            mlines.Line2D([], [], color=S["dim_bbox_color"], linewidth=1.2,
                          label="BBox (mm)"),
        ]
        leg = ax.legend(handles=legend_h, loc="lower left", fontsize=8,
                        framealpha=1.0, edgecolor=S["border"], title="LEGEND")
        leg.get_title().set_fontweight("bold")
        leg.get_frame().set_linewidth(1.0)

        # ══════════════════ INFO PANEL ══════════════════════
        ax_i = fig.add_subplot(gs[1, 1])
        ax_i.axis("off")
        ax_i.add_patch(mpatches.Rectangle(
            (0, 0), 1, 1, transform=ax_i.transAxes,
            facecolor=S["info_bg"], edgecolor=S["border"], linewidth=1.5))

        # ── Section 1: Summary Card ──
        ax_i.text(0.5, 0.97, "WALL METRICS", ha="center", va="top",
                  fontsize=13, fontweight="bold", color=S["text_dark"],
                  transform=ax_i.transAxes)
        ax_i.plot([0, 1], [0.94, 0.94], color=S["border"], linewidth=1.2,
                  transform=ax_i.transAxes, clip_on=False)

        # 2x2 grid for metrics
        m_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
        m_labels = ["Height", "Floor Area", "Wall Area", "Perimeter"]
        m_vals = [f"{height_mm / 1000:.2f} m", f"{area_m2:.2f} m²",
                  f"{twa:.2f} m²", f"{perimeter_m:.2f} m"]
        
        y0, dy = 0.88, 0.05
        for k in range(4):
            yy = y0 - k * dy
            # accent bar
            ax_i.add_patch(mpatches.Rectangle((0.05, yy - 0.015), 0.015, 0.03, 
                                              facecolor=m_colors[k], edgecolor="none", 
                                              transform=ax_i.transAxes))
            ax_i.text(0.08, yy, m_labels[k], ha="left", va="center",
                      fontsize=10, color=S["text_dark"], transform=ax_i.transAxes)
            ax_i.text(0.48, yy, ":", ha="center", va="center",
                      fontsize=10, color=S["text_dark"], transform=ax_i.transAxes)
            ax_i.text(0.53, yy, m_vals[k], ha="left", va="center",
                      fontsize=11, fontweight="bold", color=S["text_dark"],
                      fontfamily="monospace", transform=ax_i.transAxes)
                      
        # Shape badge
        ax_i.text(0.5, 0.65, f" {self.metrics.shape_name} ", ha="center", va="center",
                  fontsize=10, fontweight="bold", color="white", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=S["badge_border"], edgecolor="none"),
                  transform=ax_i.transAxes)
                  
        ax_i.plot([0, 1], [0.60, 0.60], color=S["border"], linewidth=1.2,
                  transform=ax_i.transAxes, clip_on=False)

        # ── Section 2: Wall Table ──
        rows = []
        for idx, w in enumerate(self.metrics.wall_metrics):
            if idx >= 10:
                rows.append(["…", "…", "…", "…", "…"])
                break
            l_m = w["length_mm"] / 1000.0
            a = w["area_m2"]
            pct = (l_m / perimeter_m) * 100 if perimeter_m > 0 else 0
            rows.append([w["id"], w["dir"], f"{l_m:.2f}", f"{a:.2f}", f"{pct:.1f}%"])
            
        rows.append(["TOTAL", "-", f"{perimeter_m:.2f}", f"{twa:.2f}", "100%"])
            
        cols = ["ID", "Dir", "Len (m)", "Area (m²)", "%"]
        tbl = ax_i.table(cellText=rows, colLabels=cols, loc="center",
                         bbox=[0.02, 0.35, 0.96, 0.23], cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.set_zorder(10)
        
        for (r, c_), cell in tbl.get_celld().items():
            cell.set_text_props(fontfamily="monospace")
            cell.set_edgecolor(S["border"])
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor(S["table_header_bg"])
            elif r == len(rows): # Total row
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#E5E7EB")
            elif r % 2 == 1:
                cell.set_facecolor("#F9FAFB")
            else:
                cell.set_facecolor("white")

        ax_i.plot([0, 1], [0.33, 0.33], color=S["border"], linewidth=1.2,
                  transform=ax_i.transAxes, clip_on=False)

        # ── Section 3: Wall Length Bar Chart ──
        ax_bar = host_ax_bar = host_ax_bar = host_ax_bar = host_ax_bar = None # To avoid lint
        bar_ax = ax_i.inset_axes([0.15, 0.17, 0.8, 0.14], transform=ax_i.transAxes)
        bar_ax.set_facecolor("none")
        for spine in bar_ax.spines.values():
            spine.set_visible(False)
        bar_ax.tick_params(axis='both', which='both', length=0, labelsize=7, colors=S["text_mid"])
        
        walls_to_plot = self.metrics.wall_metrics[:10]
        y_pos = np.arange(len(walls_to_plot))
        lengths = [w["length_mm"]/1000.0 for w in walls_to_plot]
        colors = ["#3B82F6" if w["dir"] == "Hor" else "#F59E0B" for w in walls_to_plot]
        labels = [w["id"] for w in walls_to_plot]
        
        bar_ax.barh(y_pos, lengths, align='center', color=colors, alpha=0.8, height=0.6)
        bar_ax.set_yticks(y_pos)
        bar_ax.set_yticklabels(labels, fontfamily="monospace", fontweight="bold")
        bar_ax.invert_yaxis()
        bar_ax.set_xlabel("Length (m)", fontsize=7, color=S["text_mid"], labelpad=1)
        bar_ax.xaxis.grid(True, linestyle='--', alpha=0.3, color=S["text_mid"])
        bar_ax.set_axisbelow(True)

        ax_i.plot([0, 1], [0.14, 0.14], color=S["border"], linewidth=1.2,
                  transform=ax_i.transAxes, clip_on=False)

        # ── Section 4: Notes & Compass ──
        ax_i.text(0.05, 0.12, "Note:", ha="left", va="top", fontsize=9,
                  fontweight="bold", color=S["text_dark"],
                  transform=ax_i.transAxes)
        ax_i.text(0.05, 0.095,
                  "1. Dims in mm.\n2. Scan auto-aligned.",
                  ha="left", va="top", fontsize=8, color=S["text_dark"],
                  transform=ax_i.transAxes)

        # compass (drawn inline)
        self._draw_compass(ax_i, pos=[0.70, 0.015, 0.25, 0.13])

        # ── Section 5: Sheet Info Box ──
        now = datetime.now().strftime("%d-%m-%Y")
        info = f"SHEET: {sheet_size} ({sheet_w}×{sheet_h} mm)\nDATE: {now}"
        ax_i.add_patch(mpatches.Rectangle(
            (0.05, 0.02), 0.6, 0.045, transform=ax_i.transAxes,
            facecolor="white", edgecolor=S["border"], linewidth=1.0))
        ax_i.text(0.08, 0.05, info, ha="left", va="top",
                  fontsize=7, color=S["text_dark"], fontfamily="monospace",
                  transform=ax_i.transAxes)

        # ── save ────────────────────────────────────────────
        fig.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"       Saved floor plan: {output_path}")

    # ── compass rose ────────────────────────────────────────

    @staticmethod
    def _draw_compass(host_ax, pos):
        """Draw the compass asset, with an inline fallback if the asset is unavailable."""
        cax = host_ax.inset_axes(pos, transform=host_ax.transAxes)
        cax.set_xlim(0, 1); cax.set_ylim(0, 1)
        cax.set_aspect("equal"); cax.axis("off")

        try:
            img = mpimg.imread(FloorPlanRenderer.COMPASS_ICON_PATH)
            cax.imshow(img, extent=[0, 1, 0, 1], origin="upper", aspect="equal")
            return
        except Exception as exc:
            print(f"       Compass icon fallback: {exc}")

        cax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=(1, 1, 1, 0.96), edgecolor="#CBD5E1", linewidth=0.8))
        cax.add_patch(mpatches.Circle(
            (0.5, 0.54), 0.28, facecolor="#F8FAFC",
            edgecolor="#1F2937", linewidth=0.8))
        cax.add_patch(mpatches.Circle(
            (0.5, 0.54), 0.04, facecolor="#1F2937", edgecolor="none"))

        arms = {
            "N": ((0.5, 0.54), (0.5, 0.86), "#CC0000", 12, 1.6),
            "E": ((0.5, 0.54), (0.82, 0.54), "#475467", 9, 1.0),
            "S": ((0.5, 0.54), (0.5, 0.22), "#475467", 9, 1.0),
            "W": ((0.5, 0.54), (0.18, 0.54), "#475467", 9, 1.0),
        }
        lbl_pos = {"N": (0.5, 0.93), "E": (0.91, 0.54),
                   "S": (0.5, 0.12), "W": (0.09, 0.54)}
        for txt, (s, e, col, sc, lw) in arms.items():
            cax.add_patch(mpatches.FancyArrowPatch(
                s, e, arrowstyle="-|>", mutation_scale=sc,
                linewidth=lw, color=col))
            cax.text(lbl_pos[txt][0], lbl_pos[txt][1], txt,
                     ha="center", va="center", fontsize=7,
                     fontweight="bold", color=col, fontfamily="monospace")


class WallElevationsRenderer(BaseRenderer):
    def render(self, output_path="elevations.png"):
        print("[5/5] Rendering wall elevations...")
        S = STYLE
        n = self.metrics.wall_count
        cols = 3
        rows = (n + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 6 + 5, max(rows * 6, 10)), facecolor=S["bg_white"])
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[cols * 6, 5], wspace=0.05)
        
        # --- Left Side: Wall Grid ---
        gs_walls = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs[0], wspace=0.3, hspace=0.3)

        for i, w in enumerate(self.metrics.wall_metrics):
            ax = fig.add_subplot(gs_walls[i // cols, i % cols])
            length = w["length_mm"]
            height = self.metrics.height_mm

            ax.add_patch(mpatches.Rectangle(
                (0, 0), length, height, linewidth=2,
                edgecolor="black", facecolor=S["wall_fill"]))
            ax.set_xlim(-length * 0.2, length * 1.2)
            ax.set_ylim(-height * 0.1, height * 1.1)
            ax.set_aspect("equal")
            ax.set_title(
                f"{w['id']} ({w['dir']}) — {length:.0f}×{height:.0f} mm",
                fontweight="bold")
            ax.text(length / 2, height / 2,
                    f"Area: {w['area_m2']:.2f} m²",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor=S["border"], boxstyle="round,pad=0.5"))

        # --- Right Side: Info Panel ---
        ax_i = fig.add_subplot(gs[1])
        ax_i.axis("off")
        ax_i.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax_i.transAxes,
                                          facecolor="white", edgecolor=S["border"], linewidth=1.0))
        
        # 1. Title
        ax_i.text(0.5, 0.95, "WALL ELEVATIONS METRICS", ha="center", va="center",
                  fontsize=14, fontweight="bold", color=S["border"], transform=ax_i.transAxes)
        ax_i.plot([0, 1], [0.91, 0.91], color=S["border"], linewidth=1.2, transform=ax_i.transAxes, clip_on=False)
        
        # 2. Summary Metrics
        metrics_y = 0.85
        spacing = 0.07
        
        def draw_metric(y, color, label, val_str):
            ax_i.add_patch(mpatches.Rectangle((0.05, y - 0.02), 0.01, 0.04, facecolor=color, edgecolor="none", transform=ax_i.transAxes))
            ax_i.text(0.08, y, label, ha="left", va="center", fontsize=10, transform=ax_i.transAxes)
            ax_i.text(0.48, y, ":", ha="center", va="center", fontsize=10, transform=ax_i.transAxes)
            ax_i.text(0.52, y, val_str, ha="left", va="center", fontsize=11, fontweight="bold", fontfamily="monospace", transform=ax_i.transAxes)

        perimeter_m = self.metrics.perimeter_m
        twa = sum(w["area_m2"] for w in self.metrics.wall_metrics)
        draw_metric(metrics_y, "#3B82F6", "Height", f"{self.metrics.height_mm/1000.0:.2f} m")
        draw_metric(metrics_y - spacing, "#F59E0B", "Wall Area", f"{twa:.2f} m²")
        draw_metric(metrics_y - spacing*2, "#EF4444", "Total Length", f"{perimeter_m:.2f} m")
        draw_metric(metrics_y - spacing*3, "#10B981", "Walls Count", f"{self.metrics.wall_count}")
        
        ax_i.plot([0, 1], [0.58, 0.58], color=S["border"], linewidth=1.2, transform=ax_i.transAxes, clip_on=False)

        # 3. Table
        rows_data = []
        for idx, w in enumerate(self.metrics.wall_metrics):
            if idx >= 15:
                rows_data.append(["…", "…", "…", "…", "…"])
                break
            l_m = w["length_mm"] / 1000.0
            a = w["area_m2"]
            pct = (a / twa) * 100 if twa > 0 else 0
            rows_data.append([w["id"], w["dir"], f"{l_m:.2f}", f"{a:.2f}", f"{pct:.1f}%"])
            
        rows_data.append(["TOTAL", "-", f"{perimeter_m:.2f}", f"{twa:.2f}", "100%"])
            
        cols_headers = ["ID", "Dir", "Len (m)", "Area (m²)", "%"]
        tbl = ax_i.table(cellText=rows_data, colLabels=cols_headers, loc="center",
                         bbox=[0.02, 0.35, 0.96, 0.20], cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.set_zorder(10)
        
        for (r, c_), cell in tbl.get_celld().items():
            cell.set_text_props(fontfamily="monospace")
            cell.set_edgecolor(S["border"])
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor(S["table_header_bg"])
            elif r == len(rows_data):
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#E5E7EB")
            elif r % 2 == 1:
                cell.set_facecolor("#F9FAFB")
            else:
                cell.set_facecolor("white")
                
        ax_i.plot([0, 1], [0.33, 0.33], color=S["border"], linewidth=1.2, transform=ax_i.transAxes, clip_on=False)

        # 4. Bar chart
        bar_ax = ax_i.inset_axes([0.15, 0.12, 0.8, 0.17], transform=ax_i.transAxes)
        bar_ax.set_facecolor("none")
        for spine in bar_ax.spines.values(): spine.set_visible(False)
        bar_ax.tick_params(axis='both', which='both', length=0, labelsize=7, colors=S["text_mid"])
        
        walls_to_plot = self.metrics.wall_metrics[:10]
        y_pos = np.arange(len(walls_to_plot))
        areas = [w["area_m2"] for w in walls_to_plot]
        colors = ["#3B82F6" if w["dir"] == "Hor" else "#F59E0B" for w in walls_to_plot]
        
        bar_ax.barh(y_pos, areas, align='center', color=colors, alpha=0.8, height=0.6)
        bar_ax.set_yticks(y_pos)
        bar_ax.set_yticklabels([w["id"] for w in walls_to_plot])
        bar_ax.invert_yaxis()
        bar_ax.set_xlabel('Area (m²)', fontsize=7, color=S["text_mid"])
        bar_ax.xaxis.grid(True, linestyle='--', alpha=0.5)

        ax_i.plot([0, 1], [0.08, 0.08], color=S["border"], linewidth=1.2, transform=ax_i.transAxes, clip_on=False)
        
        # 5. Legend/Note
        ax_i.text(0.05, 0.05, "Note:", ha="left", va="center", fontsize=9, fontweight="bold", transform=ax_i.transAxes)
        ax_i.text(0.05, 0.03, "Hor = Horizontal (Blue)\nVer = Vertical (Orange)", ha="left", va="top", fontsize=8, color=S["text_mid"], transform=ax_i.transAxes)

        # Draw main Title
        fig.suptitle(f"WALL ELEVATIONS — {self.project_name}", fontsize=22, fontweight="bold", y=0.96)
        plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"       Saved wall elevations: {output_path}")
