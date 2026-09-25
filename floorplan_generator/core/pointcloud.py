import os

import numpy as np
import open3d as o3d
from floorplan_generator.config import VOXEL_SIZE, HEIGHT_PERCENTILE

class PointCloudProcessor:
    PROJECTION_VIEWS = {
        "xy_pos_z": {"axes": (0, 1), "signs": (1.0, 1.0), "axis_mapping": {"u": "+x", "v": "+y", "view_direction": "+z"}, "height_axis": 2},
        "xy_neg_z": {"axes": (0, 1), "signs": (1.0, -1.0), "axis_mapping": {"u": "+x", "v": "-y", "view_direction": "-z"}, "height_axis": 2},
        "xz_pos_y": {"axes": (0, 2), "signs": (1.0, 1.0), "axis_mapping": {"u": "+x", "v": "+z", "view_direction": "+y"}, "height_axis": 1},
        "xz_neg_y": {"axes": (0, 2), "signs": (-1.0, 1.0), "axis_mapping": {"u": "-x", "v": "+z", "view_direction": "-y"}, "height_axis": 1},
        "yz_pos_x": {"axes": (1, 2), "signs": (1.0, 1.0), "axis_mapping": {"u": "+y", "v": "+z", "view_direction": "+x"}, "height_axis": 0},
        "yz_neg_x": {"axes": (1, 2), "signs": (-1.0, 1.0), "axis_mapping": {"u": "-y", "v": "+z", "view_direction": "-x"}, "height_axis": 0},
    }

    def __init__(self, file_path, projection_view="xz_neg_y"):
        self.file_path = file_path
        self.pcd = None
        self.height = None
        self.projection_view = (projection_view or "auto").lower()
        self.alignment_info = {}

    def _manual_projection_config(self):
        if self.projection_view in self.PROJECTION_VIEWS:
            return self.PROJECTION_VIEWS[self.projection_view]
        return None

    def _estimate_manhattan_refinement(self, points):
        try:
            import cv2
        except Exception:
            return 0.0, 0

        if len(points) < 1000:
            return 0.0, 0

        floor_z = np.percentile(points[:, 2], HEIGHT_PERCENTILE)
        z_min = floor_z + 0.8
        z_max = floor_z + 1.8
        slice_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
        if len(slice_points) < 1000:
            slice_points = points

        p2d = slice_points[:, :2]
        resolution = 0.01
        x_min, y_min = p2d.min(axis=0)
        x_max, y_max = p2d.max(axis=0)
        width = int(np.ceil((x_max - x_min) / resolution)) + 2
        height = int(np.ceil((y_max - y_min) / resolution)) + 2
        if width < 32 or height < 32 or width > 4000 or height > 4000:
            return 0.0, 0

        grid = np.zeros((height, width), dtype=np.int32)
        idx_x = np.clip(((p2d[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
        idx_y = np.clip(((p2d[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
        np.add.at(grid, (idx_y, idx_x), 1)

        nonzero = grid[grid > 0]
        if len(nonzero) == 0:
            return 0.0, 0
        threshold = max(1, int(np.percentile(nonzero, 35)))
        occupied = (grid > threshold).astype(np.uint8) * 255
        occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        min_line_length = max(30, int(min(width, height) * 0.18))
        lines = cv2.HoughLinesP(
            occupied,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=min_line_length,
            maxLineGap=8,
        )
        if lines is None:
            return 0.0, 0

        residuals = []
        weights = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min_line_length:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            residual = ((angle + 45.0) % 90.0) - 45.0
            if abs(residual) > 20.0:
                continue
            residuals.append(residual)
            weights.append(length)

        if len(residuals) < 4:
            return 0.0, len(residuals)

        residuals = np.asarray(residuals, dtype=float)
        weights = np.asarray(weights, dtype=float)
        correction = -float(np.sum(residuals * weights) / np.sum(weights))
        if abs(correction) < 0.35 or abs(correction) > 12.0:
            return 0.0, len(residuals)
        return correction, len(residuals)

    @staticmethod
    def _detect_up_axis(points):
        ranges = np.ptp(points, axis=0)
        # Indoor room scans usually have the vertical span in a plausible
        # room-height range and smaller than the long floor-plan axis.
        plausible = [(idx, value) for idx, value in enumerate(ranges) if 1.2 <= value <= 5.0]
        if plausible:
            return min(plausible, key=lambda item: item[1])[0], ranges
        return int(np.argmin(ranges)), ranges

    @staticmethod
    def _rotation_matrix_from_vectors(source, target):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        source_norm = np.linalg.norm(source)
        target_norm = np.linalg.norm(target)
        if source_norm < 1e-9 or target_norm < 1e-9:
            return np.eye(3)

        a = source / source_norm
        b = target / target_norm
        cross = np.cross(a, b)
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-9:
            if dot > 0:
                return np.eye(3)
            # 180-degree rotation around any axis orthogonal to source.
            axis = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis - a * np.dot(axis, a)
            axis = axis / (np.linalg.norm(axis) + 1e-9)
            return 2.0 * np.outer(axis, axis) - np.eye(3)

        vx = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        return np.eye(3) + vx + vx @ vx * ((1.0 - dot) / (cross_norm ** 2))

    def _level_points_to_floor(self, points, normals):
        """Align the dominant floor plane normal to +Z to remove roll/pitch tilt."""
        leveling_enabled = os.getenv("PLY_ENABLE_FLOOR_LEVELING", "1") != "0"
        info = {
            "enabled": bool(leveling_enabled),
            "applied": False,
            "method": "floor_ransac",
        }
        if not leveling_enabled or len(points) < 1000:
            info["reason"] = "disabled_or_too_few_points"
            return points, normals, info

        try:
            z = points[:, 2]
            z_low = np.percentile(z, 2)
            z_high = np.percentile(z, 18)
            floor_candidates = points[(z >= z_low) & (z <= z_high)]
            if len(floor_candidates) < 500:
                info["reason"] = "too_few_floor_candidates"
                return points, normals, info

            candidate_pcd = o3d.geometry.PointCloud()
            candidate_pcd.points = o3d.utility.Vector3dVector(floor_candidates)
            plane_model, inliers = candidate_pcd.segment_plane(
                distance_threshold=0.035,
                ransac_n=3,
                num_iterations=800,
            )
            if len(inliers) < max(300, len(floor_candidates) * 0.12):
                info["reason"] = "weak_floor_plane"
                info["inliers"] = int(len(inliers))
                info["candidates"] = int(len(floor_candidates))
                return points, normals, info

            normal = np.asarray(plane_model[:3], dtype=float)
            if normal[2] < 0:
                normal = -normal
            tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2] / (np.linalg.norm(normal) + 1e-9), -1.0, 1.0))))
            if tilt_deg < 0.15:
                info.update({
                    "reason": "already_level",
                    "tilt_deg": tilt_deg,
                    "floor_normal_before": normal.tolist(),
                    "inliers": int(len(inliers)),
                    "candidates": int(len(floor_candidates)),
                })
                return points, normals, info

            if tilt_deg > 20.0:
                info.update({
                    "reason": "tilt_too_large",
                    "tilt_deg": tilt_deg,
                    "floor_normal_before": normal.tolist(),
                    "inliers": int(len(inliers)),
                    "candidates": int(len(floor_candidates)),
                })
                return points, normals, info

            R = self._rotation_matrix_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
            centered = points - points.mean(axis=0)
            leveled_points = centered @ R.T + points.mean(axis=0)
            leveled_normals = normals @ R.T
            info.update({
                "applied": True,
                "tilt_deg": tilt_deg,
                "floor_normal_before": normal.tolist(),
                "inliers": int(len(inliers)),
                "candidates": int(len(floor_candidates)),
            })
            return leveled_points, leveled_normals, info
        except Exception as exc:
            info["reason"] = f"failed: {exc}"
            return points, normals, info
        
    def load_and_downsample(self):
        print(f"[1/5] Loading point cloud from {self.file_path}...")
        self.pcd = o3d.io.read_point_cloud(self.file_path)
        if self.pcd.is_empty():
            raise ValueError(f"Failed to load point cloud: {self.file_path}")
            
        print(f"       Total points: {len(self.pcd.points)}")
        
        print(f"       Downsampling (voxel={VOXEL_SIZE}m)...")
        self.pcd = self.pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        
        print("       Estimating normals and auto-aligning...")
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # 1. Detect Up Axis
        normals = np.asarray(self.pcd.normals)
        points = np.asarray(self.pcd.points)

        manual_projection = self._manual_projection_config()
        if manual_projection is not None and os.getenv("PLY_DISABLE_AUTO_UP_AXIS_FOR_MANUAL_VIEW", "1") != "0":
            self.alignment_info = {
                "enabled": False,
                "reason": "manual_projection_view_preserves_source_axes",
                "projection_view": self.projection_view,
                "axis_mapping": manual_projection["axis_mapping"],
                "source_axis_ranges_m": [float(v) for v in np.ptp(points, axis=0)],
                "initial_rotation_deg": 0.0,
                "refined_rotation_deg": 0.0,
                "refine_line_count": 0,
                "refine_applied": False,
                "total_rotation_deg": 0.0,
            }
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.normals = o3d.utility.Vector3dVector(normals)
            print(f"       Preserving source axes for projection_view={self.projection_view}")
            print(f"       Points after downsample: {len(self.pcd.points)}")
            return self.pcd

        up_axis, axis_ranges = self._detect_up_axis(points)
        axis_names = ['X', 'Y', 'Z']
        print(
            f"       Detected Up Axis: {axis_names[up_axis]} "
            f"(ranges={axis_ranges[0]:.3f},{axis_ranges[1]:.3f},{axis_ranges[2]:.3f}m)"
        )

        if up_axis == 0: # X is up
            points = points[:, [2, 1, 0]]
            normals = normals[:, [2, 1, 0]]
        elif up_axis == 1: # Y is up
            points = points[:, [0, 2, 1]]
            normals = normals[:, [0, 2, 1]]

        points, normals, leveling_info = self._level_points_to_floor(points, normals)
        if leveling_info.get("applied"):
            print(f"       Leveled floor plane: tilt={leveling_info.get('tilt_deg', 0.0):.2f} degrees")
        else:
            print(f"       Floor leveling skipped ({leveling_info.get('reason', 'not_needed')})")
            
        # 2. Detect Wall Rotation in XY plane
        p2d = points[:, :2]
        min_area = float('inf')
        best_angle = 0
        
        for angle_deg in range(0, 90):
            theta = np.radians(angle_deg)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            
            rotated = p2d.dot(R.T)
            min_x, max_x = rotated[:, 0].min(), rotated[:, 0].max()
            min_y, max_y = rotated[:, 1].min(), rotated[:, 1].max()
            
            area = (max_x - min_x) * (max_y - min_y)
            if area < min_area:
                min_area = area
                best_angle = angle_deg
                
        print(f"       Detected Wall Rotation: {best_angle} degrees")
        
        # Apply rotation
        theta = np.radians(best_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        points[:, :2] = points[:, :2].dot(R.T)

        refine_enabled = os.getenv("PLY_ENABLE_MANHATTAN_REFINE", "1") != "0"
        refine_angle, line_count = self._estimate_manhattan_refinement(points) if refine_enabled else (0.0, 0)
        refine_applied = bool(refine_angle)
        if refine_angle:
            theta = np.radians(refine_angle)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            points[:, :2] = points[:, :2].dot(R.T)
            print(
                f"       Refined Wall Rotation: {refine_angle:.2f} degrees "
                f"from {line_count} wall lines"
            )
        else:
            print(f"       Refined Wall Rotation: skipped ({line_count} wall lines)")

        self.alignment_info = {
            "up_axis": axis_names[up_axis],
            "leveling": leveling_info,
            "initial_rotation_deg": float(best_angle),
            "refined_rotation_deg": float(refine_angle),
            "refine_line_count": int(line_count),
            "refine_applied": refine_applied,
            "total_rotation_deg": float(best_angle + refine_angle),
        }
         
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.normals = o3d.utility.Vector3dVector(normals)
        
        print(f"       Points after downsample and align: {len(self.pcd.points)}")
        return self.pcd

    def _height_from_axis(self, points, axis):
        values = points[:, axis]
        low = np.percentile(values, HEIGHT_PERCENTILE)
        high = np.max(values)
        detected_height = (high - low) * 1000.0
        default_height = float(os.getenv("PLY_DEFAULT_HEIGHT_MM", "2400"))
        if detected_height < 1800.0 or detected_height > 4500.0:
            self.height = default_height
            source = "manual_default_due_to_implausible_axis_span"
        else:
            self.height = detected_height
            source = "detected_from_projection_height_axis"
        return low, high, self.height, source

    def _project_points_by_view(self, points):
        config = self._manual_projection_config()
        if config is None:
            return points[:, :2], None
        axes = config["axes"]
        signs = config["signs"]
        p2d = np.empty((len(points), 2), dtype=float)
        p2d[:, 0] = points[:, axes[0]] * signs[0]
        p2d[:, 1] = points[:, axes[1]] * signs[1]
        return p2d, config

    def get_2d_projection(self):
        points = np.asarray(self.pcd.points)
        projected, config = self._project_points_by_view(points)

        if config is not None:
            low, high, height_mm, height_source = self._height_from_axis(points, config["height_axis"])
            print(
                f"       Projection view: {self.projection_view} "
                f"({config['axis_mapping']})"
            )
            print(f"       Height axis range: {low:.3f} to {high:.3f} m ({int(height_mm)} mm, {height_source})")
            self.alignment_info["projection_view"] = self.projection_view
            self.alignment_info["axis_mapping"] = config["axis_mapping"]
            self.alignment_info["height_axis_source"] = height_source
            return projected, self.height
        
        z_min = np.percentile(points[:, 2], HEIGHT_PERCENTILE)
        z_max = np.max(points[:, 2])
        self.height = (z_max - z_min) * 1000  # in mm
        print(f"       Height range: {z_min:.3f} to {z_max:.3f} m ({int(self.height)} mm)")
        
        # Filter floor points (e.g. drop bottom 2%)
        valid_idx = points[:, 2] > z_min
        p2d = points[valid_idx][:, :2]
        
        return p2d, self.height

    def get_wall_slice_projection(self, slice_min_mm, slice_max_mm):
        points = np.asarray(self.pcd.points)
        projected, config = self._project_points_by_view(points)

        if config is not None:
            low, high, height_mm, height_source = self._height_from_axis(points, config["height_axis"])
            print(
                f"       Wall slice is not applied for manual projection_view={self.projection_view}; "
                "using full manual projection."
            )
            self.alignment_info["projection_view"] = self.projection_view
            self.alignment_info["axis_mapping"] = config["axis_mapping"]
            self.alignment_info["height_axis_source"] = height_source
            return projected, height_mm
        
        # Floor baseline
        floor_z = np.percentile(points[:, 2], HEIGHT_PERCENTILE)
        z_max = np.max(points[:, 2])
        self.height = (z_max - floor_z) * 1000  # in mm
        print(f"       Height range: {floor_z:.3f} to {z_max:.3f} m ({int(self.height)} mm)")
        
        # Convert slice mm to meters
        slice_z_min_m = floor_z + (slice_min_mm / 1000.0)
        slice_z_max_m = floor_z + (slice_max_mm / 1000.0)
        
        print(f"       Extracting wall slice: {slice_z_min_m:.3f} to {slice_z_max_m:.3f} m")
        
        valid_idx = (points[:, 2] >= slice_z_min_m) & (points[:, 2] <= slice_z_max_m)
        p2d = points[valid_idx][:, :2]
        
        if len(p2d) < 1000:
            print("       WARNING: Wall slice contains very few points. Falling back to full projection.")
            return self.get_2d_projection()
            
        print(f"       Slice points: {len(p2d)}")
        return p2d, self.height
