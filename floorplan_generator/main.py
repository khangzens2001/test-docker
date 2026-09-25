import argparse
import json
import os
import sys

import numpy as np

from floorplan_generator.config import HEIGHT_PERCENTILE
from floorplan_generator.core.pointcloud import PointCloudProcessor
from floorplan_generator.core.geometry import GeometryProcessor
from floorplan_generator.core.metrics import RoomMetrics
from floorplan_generator.core.alignment import align_p2d_manhattan
from floorplan_generator.render.floorplan import FloorPlanRenderer, WallElevationsRenderer
from floorplan_generator.render.debug_topdown import DebugTopDownRenderer


def _metrics_to_dict(metrics):
    return {
        "height_mm": float(metrics.height_mm),
        "area_m2": float(metrics.area_m2),
        "perimeter_m": float(metrics.perimeter_m),
        "bbox_width_mm": float(metrics.bbox_dims[0]),
        "bbox_depth_mm": float(metrics.bbox_dims[1]),
        "shape_name": metrics.shape_name,
        "wall_count": int(metrics.wall_count),
        "vertices_mm": [[float(x), float(y)] for x, y in metrics.vertices],
        "walls": [
            {
                "id": wall["id"],
                "direction": wall["dir"],
                "length_mm": float(wall["length_mm"]),
                "area_m2": float(wall["area_m2"]),
            }
            for wall in metrics.wall_metrics
        ],
    }


def generate_floorplan_from_ply(
    pointcloud_path,
    output_dir=".",
    project_name="ROOM SCAN",
    sheet_size="A3",
    wall_slice_min=800,
    wall_slice_max=1800,
    no_wall_slice=True,
    height_min_percent=None,
    height_max_percent=None,
    debug_grid_resolution_mm=10,
    debug_occupancy_threshold=0,
    debug_image_max_size=1600,
    alignment_mode="off",
    manual_angle_deg=0.0,
    projection_view="xz_neg_y",
):
    """Generate technical floor plan files from a PLY point cloud."""
    if not os.path.exists(pointcloud_path):
        raise FileNotFoundError(f"File not found: {pointcloud_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pointcloud_path))[0]

    pc_proc = PointCloudProcessor(pointcloud_path, projection_view=projection_view)
    pc_proc.load_and_downsample()

    projection_info = {
        "no_wall_slice": bool(no_wall_slice),
        "height_min_percent": None if height_min_percent is None else float(height_min_percent),
        "height_max_percent": None if height_max_percent is None else float(height_max_percent),
        "wall_slice_min_mm": float(wall_slice_min),
        "wall_slice_max_mm": float(wall_slice_max),
        "selected_view": (projection_view or "auto"),
    }

    if no_wall_slice:
        p2d, height_mm = pc_proc.get_2d_projection()
        projection_info["mode"] = "full_projection"
    else:
        if height_min_percent is not None or height_max_percent is not None:
            points = np.asarray(pc_proc.pcd.points)
            floor_z = np.percentile(points[:, 2], HEIGHT_PERCENTILE)
            z_max = np.max(points[:, 2])
            detected_height_mm = (z_max - floor_z) * 1000.0
            height_min = float(0.0 if height_min_percent is None else height_min_percent)
            height_max = float(100.0 if height_max_percent is None else height_max_percent)
            height_min = max(0.0, min(100.0, height_min))
            height_max = max(height_min, min(100.0, height_max))
            wall_slice_min = detected_height_mm * height_min / 100.0
            wall_slice_max = detected_height_mm * height_max / 100.0
            projection_info.update({
                "height_min_percent": height_min,
                "height_max_percent": height_max,
                "wall_slice_min_mm": float(wall_slice_min),
                "wall_slice_max_mm": float(wall_slice_max),
            })
        p2d, height_mm = pc_proc.get_wall_slice_projection(wall_slice_min, wall_slice_max)
        projection_info["mode"] = "height_slice"
    projection_info["projected_points"] = int(len(p2d))
    projection_info["height_mm"] = float(height_mm)
    if pc_proc.alignment_info.get("axis_mapping"):
        projection_info["axis_mapping"] = pc_proc.alignment_info.get("axis_mapping")
    if pc_proc.alignment_info.get("height_axis_source"):
        projection_info["height_axis_source"] = pc_proc.alignment_info.get("height_axis_source")

    p2d, projection_alignment = align_p2d_manhattan(
        p2d,
        mode=alignment_mode,
        manual_angle_deg=manual_angle_deg,
    )

    p2d_raw = p2d.copy()
    use_raw_lfit = os.getenv("PLY_LAYOUT_MODEL", "lfit").lower() in {"lfit", "auto"}
    raw_lfit_vertices, raw_lfit_info = (None, {"selected": False, "reason": "disabled"})
    if use_raw_lfit:
        raw_lfit_vertices, raw_lfit_info = GeometryProcessor.fit_missing_corner_l_shape_from_points(
            p2d_raw,
            source="raw_projection",
        )

    p2d_clean, clean_info = GeometryProcessor.clean_projection(p2d)
    clean_info["raw_lfit"] = raw_lfit_info
    if raw_lfit_vertices is not None:
        verts = raw_lfit_vertices
        GeometryProcessor.last_layout_info = raw_lfit_info
        print(
            "       -> L-shape selected from raw projection "
            f"(corner={raw_lfit_info.get('corner')}, "
            f"notch={raw_lfit_info.get('notch_ratio', 0.0):.3f}, "
            f"edge_support={raw_lfit_info.get('edge_support', 0.0):.3f})."
        )
    else:
        verts = GeometryProcessor.extract_polygon(p2d_clean, clean=False)
    metrics = RoomMetrics(verts, height_mm)

    pdf_path = os.path.join(output_dir, f"{base_name}_FloorPlan.pdf")
    png_path = os.path.join(output_dir, f"{base_name}_FloorPlan.png")
    wall_path = os.path.join(output_dir, f"{base_name}_Walls.png")
    debug_path = os.path.join(output_dir, f"{base_name}_DebugTopDown.png")
    debug_raw_path = os.path.join(output_dir, f"{base_name}_DebugTopDown_raw.png")
    debug_clean_path = os.path.join(output_dir, f"{base_name}_DebugTopDown_cleaned.png")
    metrics_path = os.path.join(output_dir, "metrics.json")
    debug_projection_path = os.path.join(output_dir, "debug_projection_clean.npz")
    debug_projection_raw_path = os.path.join(output_dir, "debug_projection_raw.npz")

    fp_renderer = FloorPlanRenderer(metrics, project_name)
    fp_renderer.render(sheet_size, pdf_path)
    fp_renderer.render(sheet_size, png_path)

    we_renderer = WallElevationsRenderer(metrics, project_name)
    we_renderer.render(wall_path)

    DebugTopDownRenderer(
        p2d=p2d_raw,
        vertices_mm=[],
        grid_resolution_mm=debug_grid_resolution_mm,
        occupancy_threshold=debug_occupancy_threshold,
        image_max_size=debug_image_max_size,
    ).render(debug_raw_path)

    DebugTopDownRenderer(
        p2d=p2d_clean,
        vertices_mm=[],
        grid_resolution_mm=debug_grid_resolution_mm,
        occupancy_threshold=debug_occupancy_threshold,
        image_max_size=debug_image_max_size,
    ).render(debug_clean_path)

    layout_debug_points = p2d_raw if raw_lfit_vertices is not None else p2d_clean
    debug_renderer = DebugTopDownRenderer(
        p2d=layout_debug_points,
        vertices_mm=verts,
        grid_resolution_mm=debug_grid_resolution_mm,
        occupancy_threshold=debug_occupancy_threshold,
        image_max_size=debug_image_max_size,
    )
    debug_renderer.render(debug_path)
    np.savez_compressed(debug_projection_raw_path, p2d_raw=p2d_raw)
    np.savez_compressed(debug_projection_path, p2d_clean=p2d_clean)

    metrics_data = _metrics_to_dict(metrics)
    metrics_data["project_name"] = project_name
    metrics_data["projection"] = projection_info
    metrics_data["projection_cleaning"] = clean_info
    metrics_data["alignment"] = pc_proc.alignment_info
    metrics_data["alignment"]["projection_alignment"] = projection_alignment
    metrics_data["layout_confidence"] = GeometryProcessor.last_layout_info
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)

    return {
        "status": "completed",
        "pointcloud_path": pointcloud_path,
        "output_dir": output_dir,
        "files": {
            "floorplan_pdf": pdf_path,
            "floorplan_png": png_path,
            "wall_elevation_png": wall_path,
            "debug_topdown_png": debug_path,
            "debug_topdown_raw_png": debug_raw_path,
            "debug_topdown_cleaned_png": debug_clean_path,
            "debug_projection_raw_npz": debug_projection_raw_path,
            "debug_projection_npz": debug_projection_path,
            "metrics_json": metrics_path,
        },
        "metrics": metrics_data,
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Technical Floor Plan from 3D Point Cloud")
    parser.add_argument("pointcloud_path", type=str, help="Path to the .ply point cloud file")
    parser.add_argument("--project-name", type=str, default="ROOM SCAN", help="Project name for title block")
    parser.add_argument("--sheet-size", type=str, default="A3", choices=["A0", "A1", "A2", "A3", "A4"])
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")
    
    # Wall Slice arguments
    parser.add_argument("--wall-slice-min", type=float, default=800, help="Wall slice min height in mm (default: 800)")
    parser.add_argument("--wall-slice-max", type=int, default=1800, help="Wall slice end height from floor in mm (default: 1800)")
    parser.add_argument("--no-wall-slice", action="store_true", default=True, help="Disable wall slice and use full point cloud projection")
    parser.add_argument("--use-wall-slice", action="store_false", dest="no_wall_slice", help="Use height slice instead of full projection")
    parser.add_argument("--height-min-percent", type=float, default=None, help="Wall slice start as percent of detected room height")
    parser.add_argument("--height-max-percent", type=float, default=None, help="Wall slice end as percent of detected room height")
    parser.add_argument("--debug-grid-resolution-mm", type=float, default=10, help="Debug top-down grid resolution in mm")
    parser.add_argument("--debug-occupancy-threshold", type=int, default=0, help="Debug top-down occupancy threshold")
    parser.add_argument("--debug-image-max-size", type=int, default=1600, help="Debug top-down max image side in pixels")
    parser.add_argument("--alignment-mode", type=str, default=os.getenv("PLY_PROJECTION_ALIGNMENT", "off"), choices=["auto", "off", "manual"], help="Projection Manhattan alignment mode")
    parser.add_argument("--manual-angle-deg", type=float, default=0.0, help="Manual projection alignment angle in degrees")
    parser.add_argument("--projection-view", type=str, default=os.getenv("PLY_PROJECTION_VIEW", "xz_neg_y"), choices=["auto", "xy_pos_z", "xy_neg_z", "xz_pos_y", "xz_neg_y", "yz_pos_x", "yz_neg_x"], help="Manual projection view for PLY floorplan generation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pointcloud_path):
        print(f"Error: File not found {args.pointcloud_path}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.pointcloud_path))[0]
    
    try:
        generate_floorplan_from_ply(
            pointcloud_path=args.pointcloud_path,
            output_dir=args.output_dir,
            project_name=args.project_name,
            sheet_size=args.sheet_size,
            wall_slice_min=args.wall_slice_min,
            wall_slice_max=args.wall_slice_max,
            no_wall_slice=args.no_wall_slice,
            height_min_percent=args.height_min_percent,
            height_max_percent=args.height_max_percent,
            debug_grid_resolution_mm=args.debug_grid_resolution_mm,
            debug_occupancy_threshold=args.debug_occupancy_threshold,
            debug_image_max_size=args.debug_image_max_size,
            alignment_mode=args.alignment_mode,
            manual_angle_deg=args.manual_angle_deg,
            projection_view=args.projection_view,
        )
        
        print("\nSUCCESS: All technical drawings generated.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
