# Adapted from 3D-Estimate quality checks with P17 parity thresholds
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.cad.contract import layout_to_metrics_mm


def write_quality_report_json(
    path: str | Path | dict, report: dict | str | Path | None = None
) -> None:
    """Write quality report dictionary to path atomically."""
    if isinstance(path, dict) and isinstance(report, (str, Path)):
        path, report = report, path
    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    temp.replace(target)


def build_quality_report(
    layout: dict | None = None,
    session_dir: str | Path | dict | None = None,
    cloud_num_points: int | None = None,
    *,
    ply_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Build quality report based on P17 thresholds.

    Evaluates layout geometry and reconstruction diagnostics. Missing diagnostic keys
    or geometry keys are skipped without triggering false warnings.
    Never raises an exception.
    """
    try:
        if not isinstance(layout, dict):
            layout = {}

        warnings: list[str] = []

        # Geometry checks (area, bounding box, height)
        has_room = isinstance(layout.get("room"), dict) and bool(layout["room"])
        has_vertices = isinstance(layout.get("vertices"), dict) and bool(layout["vertices"])
        has_walls = isinstance(layout.get("walls"), list) and bool(layout["walls"])
        has_metrics = "bbox_width_mm" in layout and "wall_count" in layout

        if has_room or has_vertices or has_walls or has_metrics:
            if has_metrics:
                metrics = layout
            else:
                try:
                    metrics = layout_to_metrics_mm(layout)
                except Exception:
                    metrics = {}

            # 1. area_m2 < 2.0
            area_m2: float | None = None
            if has_room and layout["room"].get("area_m2") is not None:
                try:
                    area_m2 = float(layout["room"]["area_m2"])
                except (ValueError, TypeError):
                    area_m2 = None
            elif "area_m2" in metrics and metrics["area_m2"] > 0:
                area_m2 = float(metrics["area_m2"])

            if area_m2 is not None and area_m2 < 2.0:
                warnings.append(
                    "Detected floor area is under 2m2; scale or reconstruction is likely wrong."
                )

            # 2. min(bbox_width_mm, bbox_depth_mm) < 800
            w_mm: float | None = None
            d_mm: float | None = None
            if (
                has_room
                and layout["room"].get("width_m") is not None
                and layout["room"].get("depth_m") is not None
            ):
                try:
                    w_mm = float(layout["room"]["width_m"]) * 1000.0
                    d_mm = float(layout["room"]["depth_m"]) * 1000.0
                except (ValueError, TypeError):
                    w_mm, d_mm = None, None
            elif (
                "bbox_width_mm" in metrics
                and "bbox_depth_mm" in metrics
                and metrics["bbox_width_mm"] > 0
                and metrics["bbox_depth_mm"] > 0
            ):
                w_mm = float(metrics["bbox_width_mm"])
                d_mm = float(metrics["bbox_depth_mm"])

            if w_mm is not None and d_mm is not None and min(w_mm, d_mm) < 800.0:
                warnings.append(
                    "Detected bounding box is under 800mm on one side; verify reconstruction scale."
                )

            # 3. height_mm < 1800
            h_mm: float | None = None
            if has_room and layout["room"].get("height_meters") is not None:
                try:
                    h_mm = float(layout["room"]["height_meters"]) * 1000.0
                except (ValueError, TypeError):
                    h_mm = None
            elif any(
                isinstance(w, dict) and w.get("height_meters") is not None
                for w in layout.get("walls", [])
            ):
                for w in layout.get("walls", []):
                    if isinstance(w, dict) and w.get("height_meters") is not None:
                        try:
                            h_mm = float(w["height_meters"]) * 1000.0
                            break
                        except (ValueError, TypeError):
                            continue
            elif "height_mm" in metrics and metrics["height_mm"] > 0:
                h_mm = float(metrics["height_mm"])

            if h_mm is not None and h_mm < 1800.0:
                warnings.append(
                    "Detected room height is under 1800mm; vertical scale may be wrong."
                )

        passed_diag = session_dir if isinstance(session_dir, dict) else {}
        diagnostics = (
            layout.get("diagnostics")
            if isinstance(layout.get("diagnostics"), dict)
            else passed_diag
        )
        if passed_diag and layout.get("diagnostics"):
            diagnostics = {**passed_diag, **layout.get("diagnostics")}

        # 4. Occupancy removed ratio > 0.25 ((raw-kept)/raw when raw > 0)
        raw_points = diagnostics.get("occupancy_raw_points")
        kept_points = diagnostics.get("occupancy_kept_points")
        removed_points: int | None = None
        if raw_points is not None and kept_points is not None:
            try:
                raw_val = int(raw_points)
                kept_val = int(kept_points)
                removed_points = raw_val - kept_val
                if raw_val > 0:
                    removed_ratio = (raw_val - kept_val) / raw_val
                    if removed_ratio > 0.25:
                        warnings.append(
                            "Occupancy cleaning removed more than 25% of wall-slice points; verify scan noise/outliers."
                        )
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        # 5. abs(refined_rotation_deg) > 10
        refined_rotation_deg = diagnostics.get("refined_rotation_deg")
        if refined_rotation_deg is not None:
            try:
                rot_val = float(refined_rotation_deg)
                if abs(rot_val) > 10.0:
                    warnings.append(
                        "Manhattan alignment refinement is large; verify floorplan orientation."
                    )
            except (ValueError, TypeError):
                pass

        # 6. not refine_applied and refine_line_count < 4
        refine_applied = diagnostics.get("refine_applied")
        refine_line_count = diagnostics.get("refine_line_count")
        if refine_applied is not None and refine_line_count is not None:
            try:
                if not bool(refine_applied) and int(refine_line_count) < 4:
                    warnings.append(
                        "Not enough wall lines were detected for Manhattan alignment refinement."
                    )
            except (ValueError, TypeError):
                pass

        # 7 & 8. edge_support bands
        edge_support = diagnostics.get("edge_support")
        if edge_support is not None:
            try:
                es_val = float(edge_support)
                if es_val < 0.45:
                    warnings.append(
                        "Layout edge support is low; floorplan shape may be unreliable."
                    )
                elif 0.45 <= es_val < 0.65:
                    warnings.append(
                        "Layout edge support is moderate; review debug top-down output."
                    )
            except (ValueError, TypeError):
                pass

        # Informational fields (never trigger warnings)
        if ply_size_bytes is None:
            ply_size_bytes = diagnostics.get("ply_size_bytes")
        if ply_size_bytes is None and session_dir is not None and isinstance(session_dir, (str, Path)):
            try:
                ply_path = Path(session_dir) / "reconstructed.ply"
                if ply_path.is_file():
                    ply_size_bytes = ply_path.stat().st_size
            except Exception:
                pass

        resolved_cloud_points = cloud_num_points
        if resolved_cloud_points is None:
            resolved_cloud_points = diagnostics.get("cloud_num_points")

        quality = "warning" if len(warnings) > 0 else "ok"

        report: dict[str, Any] = {
            "ply_size_bytes": ply_size_bytes,
            "cloud_num_points": resolved_cloud_points,
            "quality": quality,
            "warnings": warnings,
            "projection_cleaning": {
                "raw_points": raw_points,
                "removed_points": removed_points,
                "kept_points": kept_points,
            },
            "alignment": {
                "refined_rotation_deg": refined_rotation_deg,
                "refine_applied": refine_applied,
                "refine_line_count": refine_line_count,
            },
            "layout_confidence": {
                "edge_support": edge_support,
                "bbox_fill_ratio": diagnostics.get("bbox_fill_ratio"),
            },
        }

        if session_dir is not None and isinstance(session_dir, (str, Path)):
            try:
                write_quality_report_json(
                    Path(session_dir) / "quality_report.json", report
                )
            except Exception:
                pass

        return report

    except Exception:
        resolved_cloud = (
            cloud_num_points
            if cloud_num_points is not None
            else (
                layout.get("diagnostics", {}).get("cloud_num_points")
                if isinstance(layout, dict) and isinstance(layout.get("diagnostics"), dict)
                else None
            )
        )
        return {
            "ply_size_bytes": None,
            "cloud_num_points": resolved_cloud,
            "quality": "ok",
            "warnings": [],
            "projection_cleaning": {
                "raw_points": None,
                "removed_points": None,
                "kept_points": None,
            },
            "alignment": {
                "refined_rotation_deg": None,
                "refine_applied": None,
                "refine_line_count": None,
            },
            "layout_confidence": {
                "edge_support": None,
                "bbox_fill_ratio": None,
            },
        }
