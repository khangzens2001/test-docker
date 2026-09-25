import os

import cv2
import numpy as np
from floorplan_generator.config import (
    GRID_RESOLUTION_MM,
    MORPHOLOGY_KERNEL_SIZE,
    BBOX_FILL_RATIO_THRESHOLD,
    CONCAVE_MORPH_KERNEL_SIZE,
    CONCAVE_APPROX_EPSILON,
    DEFAULT_APPROX_EPSILON,
    MIN_EDGE_LENGTH_RATIO,
    MIN_CONCAVE_AREA_RATIO,
)


class GeometryProcessor:
    last_layout_info = {}

    @staticmethod
    def clean_projection(
        p2d,
        resolution_mm=GRID_RESOLUTION_MM,
        density_threshold=None,
        min_clean_points=1000,
    ):
        """Remove sparse/rời rạc top-down outliers before fitting a room polygon."""
        p2d = np.asarray(p2d, dtype=float)
        if p2d.ndim != 2 or p2d.shape[1] < 2 or len(p2d) == 0:
            raise ValueError("No 2D points available for projection cleaning.")
        raw_point_count = int(len(p2d))

        x_min, x_max = p2d[:, 0].min(), p2d[:, 0].max()
        y_min, y_max = p2d[:, 1].min(), p2d[:, 1].max()
        resolution = float(resolution_mm) / 1000.0
        width = int(np.ceil((x_max - x_min) / resolution)) + 2
        height = int(np.ceil((y_max - y_min) / resolution)) + 2

        grid = np.zeros((height, width), dtype=np.int32)
        idx_x = np.clip(((p2d[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
        idx_y = np.clip(((p2d[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
        np.add.at(grid, (idx_y, idx_x), 1)

        nonzero = grid[grid > 0]
        if len(nonzero) == 0:
            return p2d, {"cleaned": False, "reason": "empty_grid"}

        if density_threshold is None:
            density_threshold = max(1, int(np.floor(np.percentile(nonzero, 25))))

        occupied = (grid > density_threshold).astype(np.uint8)
        if np.count_nonzero(occupied) == 0:
            occupied = (grid > 0).astype(np.uint8)
            density_threshold = 0

        def robust_profile_bounds(occupied_mask, min_span_ratio=0.55):
            def axis_bounds(profile, full_size):
                if full_size < 16 or profile.max() <= 0:
                    return 0, full_size - 1, False

                window = max(5, int(round(full_size * 0.04)))
                if window % 2 == 0:
                    window += 1
                kernel = np.ones(window, dtype=np.float32) / window
                smooth = np.convolve(profile.astype(np.float32), kernel, mode="same")
                threshold = max(8.0, float(smooth.max()) * 0.38)
                strong = np.flatnonzero(smooth >= threshold)
                if len(strong) == 0:
                    return 0, full_size - 1, False

                lo = int(strong.min())
                hi = int(strong.max())
                if hi - lo + 1 < full_size * min_span_ratio:
                    return 0, full_size - 1, False

                margin = max(3, int(round(full_size * 0.025)))
                return max(0, lo - margin), min(full_size - 1, hi + margin), True

            col_profile = occupied_mask.sum(axis=0)
            row_profile = occupied_mask.sum(axis=1)
            x0, x1, trimmed_x = axis_bounds(col_profile, occupied_mask.shape[1])
            y0, y1, trimmed_y = axis_bounds(row_profile, occupied_mask.shape[0])
            return x0, x1, y0, y1, trimmed_x or trimmed_y

        profile_trim_enabled = os.getenv("PLY_CLEAN_PROFILE_TRIM", "1") != "0"
        trim_x0, trim_x1, trim_y0, trim_y1, did_profile_trim = robust_profile_bounds(occupied)
        did_profile_trim = did_profile_trim and profile_trim_enabled
        profile_keep = (
            (idx_x >= trim_x0)
            & (idx_x <= trim_x1)
            & (idx_y >= trim_y0)
            & (idx_y <= trim_y1)
        )
        profile_kept_count = int(np.count_nonzero(profile_keep))
        if did_profile_trim and profile_kept_count >= min_clean_points:
            p2d = p2d[profile_keep]
            x_min, x_max = p2d[:, 0].min(), p2d[:, 0].max()
            y_min, y_max = p2d[:, 1].min(), p2d[:, 1].max()
            width = int(np.ceil((x_max - x_min) / resolution)) + 2
            height = int(np.ceil((y_max - y_min) / resolution)) + 2
            grid = np.zeros((height, width), dtype=np.int32)
            idx_x = np.clip(((p2d[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
            idx_y = np.clip(((p2d[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
            np.add.at(grid, (idx_y, idx_x), 1)
            occupied = (grid > density_threshold).astype(np.uint8)

        close_kernel = np.ones((3, 3), np.uint8)
        opened_kernel = np.ones((3, 3), np.uint8)
        connected = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, close_kernel)

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
        if component_count <= 1:
            return p2d, {"cleaned": False, "reason": "no_components"}

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        largest_area = int(areas[largest_label - 1])
        component_mask = (labels == largest_label).astype(np.uint8)
        component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, close_kernel)
        component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_OPEN, opened_kernel)

        keep_points = component_mask[idx_y, idx_x] > 0
        cleaned = p2d[keep_points]
        if len(cleaned) < min_clean_points:
            return p2d, {
                "cleaned": False,
                "reason": "too_few_clean_points",
                "density_threshold": int(density_threshold),
                "clean_points": int(len(cleaned)),
                "raw_points": int(len(p2d)),
            }

        removed = int(raw_point_count - len(cleaned))
        print(
            "       Cleaned top-down projection: "
            f"kept {len(cleaned)}/{raw_point_count} points, removed {removed}, "
            f"density_threshold>{int(density_threshold)}, components={component_count - 1}, "
            f"profile_trim={did_profile_trim}."
        )
        return cleaned, {
            "cleaned": True,
            "raw_points": raw_point_count,
            "clean_points": int(len(cleaned)),
            "removed_points": removed,
            "profile_kept_points": profile_kept_count,
            "density_threshold": int(density_threshold),
            "component_count": int(component_count - 1),
            "largest_component_area_px": largest_area,
            "profile_trimmed": bool(did_profile_trim),
            "profile_trim_bounds_px": [int(trim_x0), int(trim_y0), int(trim_x1), int(trim_y1)],
        }

    @staticmethod
    def _rasterize(p2d):
        """Convert 2D points to an occupancy grid. Returns grid, metadata."""
        x_min, x_max = p2d[:, 0].min(), p2d[:, 0].max()
        y_min, y_max = p2d[:, 1].min(), p2d[:, 1].max()

        resolution = GRID_RESOLUTION_MM / 1000.0
        width = int(np.ceil((x_max - x_min) / resolution)) + 2
        height = int(np.ceil((y_max - y_min) / resolution)) + 2

        grid = np.zeros((height, width), dtype=np.int32)
        idx_x = np.clip(((p2d[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
        idx_y = np.clip(((p2d[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
        np.add.at(grid, (idx_y, idx_x), 1)

        threshold = 2 if len(p2d) > 100000 else 0
        occupied = (grid > threshold).astype(np.uint8) * 255

        return occupied, x_min, y_min, resolution, width, height

    @staticmethod
    def _get_largest_contour(binary_image):
        """Find the largest external contour."""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No room contour found in point cloud!")
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _compute_fill_ratio(contour, image_shape):
        """Compute contour_area / bounding_rect_area to detect concavity."""
        contour_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        if bbox_area == 0:
            return 1.0
        return contour_area / bbox_area

    @staticmethod
    def _filled_contour_mask(contour, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        return mask

    @staticmethod
    def _polygon_iou(mask, polygon):
        poly_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(poly_mask, [np.asarray(polygon, dtype=np.int32)], 255)
        mask_bool = mask > 0
        poly_bool = poly_mask > 0
        union = np.logical_or(mask_bool, poly_bool).sum()
        if union == 0:
            return 0.0
        inter = np.logical_and(mask_bool, poly_bool).sum()
        return inter / union

    @staticmethod
    def _edge_support_ratio(support_mask, polygon, tolerance_px=8):
        if support_mask is None:
            return 0.0, []
        support = (support_mask > 0).astype(np.uint8)
        kernel_size = max(1, int(tolerance_px) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        support = cv2.dilate(support, kernel)

        ratios = []
        for i, point in enumerate(polygon):
            next_point = polygon[(i + 1) % len(polygon)]
            edge_mask = np.zeros(support.shape, dtype=np.uint8)
            cv2.line(
                edge_mask,
                (int(point[0]), int(point[1])),
                (int(next_point[0]), int(next_point[1])),
                1,
                thickness=1,
            )
            edge_pixels = int(np.count_nonzero(edge_mask))
            if edge_pixels == 0:
                ratios.append(0.0)
                continue
            ratios.append(float(np.count_nonzero(edge_mask & support)) / edge_pixels)
        return float(np.mean(ratios)) if ratios else 0.0, ratios

    @staticmethod
    def _l_shape_polygon(x0, y0, x1, y1, xs, ys, missing_corner):
        """Return a 6-point rectilinear L polygon in pixel coordinates.

        The polygon is the bounding rectangle minus one corner rectangle. Image
        y-axis direction does not matter here because all operations are in the
        same grid coordinate system.
        """
        if missing_corner == "top_left":
            return [(xs, y0), (x1, y0), (x1, y1), (x0, y1), (x0, ys), (xs, ys)]
        if missing_corner == "top_right":
            return [(x0, y0), (xs, y0), (xs, ys), (x1, ys), (x1, y1), (x0, y1)]
        if missing_corner == "bottom_right":
            return [(x0, y0), (x1, y0), (x1, ys), (xs, ys), (xs, y1), (x0, y1)]
        if missing_corner == "bottom_left":
            return [(x0, y0), (x1, y0), (x1, y1), (xs, y1), (xs, ys), (x0, ys)]
        raise ValueError(f"Unknown missing corner: {missing_corner}")

    @staticmethod
    def _pixel_vertices_to_mm(vertices, resolution, x_min, y_min):
        final_verts = []
        for px, py in vertices:
            x_mm = (px * resolution + x_min) * 1000
            y_mm = (py * resolution + y_min) * 1000
            final_verts.append([x_mm, y_mm])
        return final_verts

    @staticmethod
    def _fit_l_shape_by_iou(contour, image_shape, resolution, x_min_world, y_min_world, support_mask=None):
        """Fit a Manhattan L shape by maximizing IoU against the filled contour.

        This is more stable than contour simplification for sparse/noisy PLYs:
        rectangle and all plausible 6-corner L candidates are scored against the
        actual occupied room mask, so large missing corners are preserved.
        """
        target_mask = GeometryProcessor._filled_contour_mask(contour, image_shape)
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            return None

        x0, y0 = x, y
        x1, y1 = x + w - 1, y + h - 1
        bbox_area = max(1, w * h)

        rectangle = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        rectangle_iou = GeometryProcessor._polygon_iou(target_mask, rectangle)
        rectangle_support, _ = GeometryProcessor._edge_support_ratio(support_mask, rectangle)

        margin_x = max(2, int(round(w * 0.12)))
        margin_y = max(2, int(round(h * 0.12)))
        if x0 + margin_x >= x1 - margin_x or y0 + margin_y >= y1 - margin_y:
            return None

        # Use contour coordinates and uniform samples as candidate inner corners.
        pts = contour.reshape(-1, 2)
        contour_x = pts[:, 0]
        contour_y = pts[:, 1]
        uniform_x = np.linspace(x0 + margin_x, x1 - margin_x, 28, dtype=int)
        uniform_y = np.linspace(y0 + margin_y, y1 - margin_y, 28, dtype=int)
        quant_x = np.percentile(contour_x, np.linspace(15, 85, 15)).astype(int)
        quant_y = np.percentile(contour_y, np.linspace(15, 85, 15)).astype(int)
        xs_candidates = np.unique(np.clip(np.concatenate([uniform_x, quant_x]), x0 + 2, x1 - 2))
        ys_candidates = np.unique(np.clip(np.concatenate([uniform_y, quant_y]), y0 + 2, y1 - 2))

        best = None
        missing_corners = ("top_left", "top_right", "bottom_right", "bottom_left")
        for xs in xs_candidates:
            for ys in ys_candidates:
                notch_areas = {
                    "top_left": (xs - x0) * (ys - y0),
                    "top_right": (x1 - xs) * (ys - y0),
                    "bottom_right": (x1 - xs) * (y1 - ys),
                    "bottom_left": (xs - x0) * (y1 - ys),
                }
                for corner in missing_corners:
                    notch_ratio = notch_areas[corner] / bbox_area
                    if notch_ratio < MIN_CONCAVE_AREA_RATIO:
                        continue
                    polygon = GeometryProcessor._l_shape_polygon(x0, y0, x1, y1, xs, ys, corner)
                    edge_lengths = []
                    for i, point in enumerate(polygon):
                        next_point = polygon[(i + 1) % len(polygon)]
                        edge_lengths.append(
                            abs(next_point[0] - point[0]) + abs(next_point[1] - point[1])
                        )
                    min_edge = min(edge_lengths)
                    if min_edge < 18:
                        continue
                    edge_support, edge_supports = GeometryProcessor._edge_support_ratio(support_mask, polygon)
                    min_edge_support = float(os.getenv("PLY_LAYOUT_MIN_EDGE_SUPPORT", "0.45"))
                    if min_edge < 35 and edge_support < min_edge_support:
                        continue
                    iou = GeometryProcessor._polygon_iou(target_mask, polygon)
                    score = (iou - rectangle_iou) + 0.10 * (edge_support - rectangle_support)
                    if best is None or score > best["score"]:
                        best = {
                            "polygon": polygon,
                            "iou": iou,
                            "rectangle_iou": rectangle_iou,
                            "score": score,
                            "corner": corner,
                            "notch_ratio": notch_ratio,
                            "edge_support": edge_support,
                            "rectangle_support": rectangle_support,
                            "edge_supports": edge_supports,
                        }

        if best is None:
            return None

        # Require a meaningful improvement over the rectangle; otherwise keep
        # the simpler model to avoid false L-shapes on noisy rectangular rooms.
        if best["score"] < 0.04 and best["iou"] < 0.88:
            return None

        return {
            **best,
            "vertices_mm": GeometryProcessor._pixel_vertices_to_mm(
                best["polygon"], resolution, x_min_world, y_min_world
            ),
        }

    @staticmethod
    def _fit_l_shape_by_missing_corner(occupied, resolution, x_min_world, y_min_world):
        """Fit an axis-aligned L by removing one low-support bbox corner.

        This handles sparse PLY projections where contour approximation tends to
        swallow a real missing corner and return the outer rectangle.
        """
        support = (occupied > 0).astype(np.uint8)
        ys, xs = np.nonzero(support)
        if len(xs) == 0:
            return None

        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bbox_w = x1 - x0
        bbox_h = y1 - y0
        bbox_area = max(1, bbox_w * bbox_h)
        total_occ = int(np.count_nonzero(support[y0:y1, x0:x1]))
        bbox_fill_ratio = total_occ / bbox_area

        min_notch_ratio = float(os.getenv("PLY_LFIT_MIN_NOTCH_RATIO", "0.03"))
        max_occ_outside_ratio = float(os.getenv("PLY_LFIT_MAX_OCC_OUTSIDE_RATIO", "0.035"))
        min_edge_support = float(os.getenv("PLY_LAYOUT_MIN_EDGE_SUPPORT", "0.42"))
        min_notch_side_px = max(10, int(round(min(bbox_w, bbox_h) * 0.08)))
        min_edge_px = max(8, int(round(min(bbox_w, bbox_h) * 0.05)))

        integral = np.pad(support.astype(np.int64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))

        def rect_sum(rx0, ry0, rx1, ry1):
            return int(integral[ry1, rx1] - integral[ry0, rx1] - integral[ry1, rx0] + integral[ry0, rx0])

        def polygon_for_corner(xs_mid, ys_mid, corner):
            if corner == "bottom_right":
                return [(x0, y0), (xs_mid, y0), (xs_mid, ys_mid), (x1, ys_mid), (x1, y1), (x0, y1)]
            if corner == "bottom_left":
                return [(x0, ys_mid), (xs_mid, ys_mid), (xs_mid, y0), (x1, y0), (x1, y1), (x0, y1)]
            if corner == "top_right":
                return [(x0, y0), (x1, y0), (x1, ys_mid), (xs_mid, ys_mid), (xs_mid, y1), (x0, y1)]
            if corner == "top_left":
                return [(x0, y0), (x1, y0), (x1, y1), (xs_mid, y1), (xs_mid, ys_mid), (x0, ys_mid)]
            raise ValueError(f"Unknown missing corner: {corner}")

        x_candidates = np.unique(np.concatenate((
            np.linspace(x0 + min_notch_side_px, x1 - min_notch_side_px, 60).astype(int),
            np.percentile(xs, np.linspace(10, 90, 21)).astype(int),
        )))
        y_candidates = np.unique(np.concatenate((
            np.linspace(y0 + min_notch_side_px, y1 - min_notch_side_px, 90).astype(int),
            np.percentile(ys, np.linspace(10, 90, 21)).astype(int),
        )))
        x_candidates = x_candidates[(x_candidates > x0 + min_edge_px) & (x_candidates < x1 - min_edge_px)]
        y_candidates = y_candidates[(y_candidates > y0 + min_edge_px) & (y_candidates < y1 - min_edge_px)]

        best = None
        for xs_mid in x_candidates.tolist():
            for ys_mid in y_candidates.tolist():
                missing_rects = {
                    "top_left": (x0, ys_mid, xs_mid, y1),
                    "top_right": (xs_mid, ys_mid, x1, y1),
                    "bottom_right": (xs_mid, y0, x1, ys_mid),
                    "bottom_left": (x0, y0, xs_mid, ys_mid),
                }
                for corner, (mx0, my0, mx1, my1) in missing_rects.items():
                    notch_area = max(0, (mx1 - mx0) * (my1 - my0))
                    notch_ratio = notch_area / bbox_area
                    if notch_ratio < min_notch_ratio:
                        continue
                    if (mx1 - mx0) < min_notch_side_px or (my1 - my0) < min_notch_side_px:
                        continue

                    occ_outside = rect_sum(mx0, my0, mx1, my1)
                    occ_outside_ratio = occ_outside / max(total_occ, 1)
                    if occ_outside_ratio > max_occ_outside_ratio:
                        continue

                    polygon = polygon_for_corner(xs_mid, ys_mid, corner)
                    edge_lengths = []
                    for idx, point in enumerate(polygon):
                        next_point = polygon[(idx + 1) % len(polygon)]
                        edge_lengths.append(abs(next_point[0] - point[0]) + abs(next_point[1] - point[1]))
                    if min(edge_lengths) < min_edge_px:
                        continue

                    edge_support, edge_supports = GeometryProcessor._edge_support_ratio(support, polygon, tolerance_px=5)
                    if edge_support < min_edge_support:
                        continue

                    poly_area = bbox_area - notch_area
                    occupied_inside = total_occ - occ_outside
                    empty_inside = poly_area - occupied_inside
                    poly_fill_ratio = occupied_inside / max(poly_area, 1)
                    empty_inside_ratio = empty_inside / max(poly_area, 1)
                    score = (
                        poly_fill_ratio
                        + 0.28 * notch_ratio
                        + 0.14 * edge_support
                        - 2.6 * occ_outside_ratio
                        - 0.08 * empty_inside_ratio
                    )
                    current = {
                        "polygon": polygon,
                        "score": float(score),
                        "corner": corner,
                        "split_px": [int(xs_mid), int(ys_mid)],
                        "bbox_fill_ratio": float(bbox_fill_ratio),
                        "notch_ratio": float(notch_ratio),
                        "occupied_outside_ratio": float(occ_outside_ratio),
                        "poly_fill_ratio": float(poly_fill_ratio),
                        "empty_inside_ratio": float(empty_inside_ratio),
                        "edge_support": float(edge_support),
                        "edge_supports": [float(value) for value in edge_supports],
                    }
                    if best is None or current["score"] > best["score"]:
                        best = current

        if best is None:
            return None
        return {
            **best,
            "vertices_mm": GeometryProcessor._pixel_vertices_to_mm(best["polygon"], resolution, x_min_world, y_min_world),
        }

    @staticmethod
    def fit_missing_corner_l_shape_from_points(p2d, resolution_mm=None, source="raw_projection"):
        """Fit a production L-shape directly from projected 2D points.

        This intentionally runs before heavy morphology/cleaning so real missing
        corners are not closed into a rectangle.
        """
        points = np.asarray(p2d, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
            return None, {"selected": False, "reason": "empty_projection"}

        if resolution_mm is None:
            resolution_mm = float(os.getenv("PLY_RAW_LFIT_RESOLUTION_MM", str(GRID_RESOLUTION_MM)))
        resolution = float(resolution_mm) / 1000.0
        if resolution <= 0:
            return None, {"selected": False, "reason": "invalid_resolution"}

        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        width = int(np.ceil((x_max - x_min) / resolution)) + 2
        height = int(np.ceil((y_max - y_min) / resolution)) + 2
        if width < 16 or height < 16:
            return None, {"selected": False, "reason": "projection_too_small"}

        grid = np.zeros((height, width), dtype=np.int32)
        idx_x = np.clip(((points[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
        idx_y = np.clip(((points[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
        np.add.at(grid, (idx_y, idx_x), 1)

        occupied = (grid > 0).astype(np.uint8) * 255
        fit = GeometryProcessor._fit_l_shape_by_missing_corner(occupied, resolution, x_min, y_min)
        if fit is None:
            return None, {
                "selected": False,
                "reason": "no_l_candidate",
                "source": source,
                "resolution_mm": float(resolution_mm),
                "grid_size_px": [int(width), int(height)],
                "occupied_px": int(np.count_nonzero(occupied)),
            }

        vertices = fit["vertices_mm"]
        edge_lengths_mm = []
        for idx, point in enumerate(vertices):
            next_point = vertices[(idx + 1) % len(vertices)]
            edge_lengths_mm.append(float(np.hypot(next_point[0] - point[0], next_point[1] - point[1])))

        max_bbox_fill = float(os.getenv("PLY_RAW_LFIT_MAX_BBOX_FILL_RATIO", "0.78"))
        min_notch_ratio = float(os.getenv("PLY_RAW_LFIT_MIN_NOTCH_RATIO", "0.05"))
        max_occ_outside = float(os.getenv("PLY_RAW_LFIT_MAX_OCC_OUTSIDE_RATIO", "0.02"))
        min_edge_support = float(os.getenv("PLY_RAW_LFIT_MIN_EDGE_SUPPORT", "0.60"))
        min_poly_fill = float(os.getenv("PLY_RAW_LFIT_MIN_POLY_FILL_RATIO", "0.68"))
        min_edge_mm = float(os.getenv("PLY_RAW_LFIT_MIN_EDGE_MM", "250"))

        rejection_reasons = []
        if fit["bbox_fill_ratio"] > max_bbox_fill:
            rejection_reasons.append("bbox_fill_too_high")
        if fit["notch_ratio"] < min_notch_ratio:
            rejection_reasons.append("notch_too_small")
        if fit["occupied_outside_ratio"] > max_occ_outside:
            rejection_reasons.append("occupied_outside_too_high")
        if fit["edge_support"] < min_edge_support:
            rejection_reasons.append("edge_support_too_low")
        if fit["poly_fill_ratio"] < min_poly_fill:
            rejection_reasons.append("poly_fill_too_low")
        if min(edge_lengths_mm) < min_edge_mm:
            rejection_reasons.append("edge_too_short")

        info = {
            "shape_model": "L-SHAPED",
            "source": source,
            "corner": fit["corner"],
            "score": float(fit["score"]),
            "bbox_fill_ratio": float(fit["bbox_fill_ratio"]),
            "notch_ratio": float(fit["notch_ratio"]),
            "occupied_outside_ratio": float(fit["occupied_outside_ratio"]),
            "poly_fill_ratio": float(fit["poly_fill_ratio"]),
            "empty_inside_ratio": float(fit["empty_inside_ratio"]),
            "edge_support": float(fit["edge_support"]),
            "edge_supports": fit["edge_supports"],
            "edge_lengths_mm": edge_lengths_mm,
            "resolution_mm": float(resolution_mm),
            "grid_size_px": [int(width), int(height)],
            "occupied_px": int(np.count_nonzero(occupied)),
            "selected_by": f"{source}_missing_corner_lfit",
            "selected": not rejection_reasons,
        }
        if rejection_reasons:
            info["reason"] = ",".join(rejection_reasons)
            return None, info
        return vertices, info

    @staticmethod
    def _orthogonalize_contour(approx, resolution, x_min, y_min):
        """Convert an approximated contour to orthogonal (H/V) edges and extract corners.

        Returns list of [x_mm, y_mm] vertices.
        """
        lines = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            if abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]):
                lines.append(('H', (p1[1] + p2[1]) / 2.0))
            else:
                lines.append(('V', (p1[0] + p2[0]) / 2.0))

        # Merge consecutive edges with the same direction
        merged = []
        for line in lines:
            if not merged:
                merged.append(line)
            elif merged[-1][0] == line[0]:
                merged[-1] = (line[0], (merged[-1][1] + line[1]) / 2.0)
            else:
                merged.append(line)

        # Wrap-around merge if first and last have the same direction
        if len(merged) > 1 and merged[0][0] == merged[-1][0]:
            merged[0] = (merged[0][0], (merged[0][1] + merged[-1][1]) / 2.0)
            merged.pop()

        # Compute intersection corners
        corners = []
        for i in range(len(merged)):
            l1 = merged[i]
            l2 = merged[(i + 1) % len(merged)]
            if l1[0] == 'H':
                corners.append((l2[1], l1[1]))
            else:
                corners.append((l1[1], l2[1]))

        # Convert pixel coords to mm
        final_verts = []
        for px, py in corners:
            x_mm = (px * resolution + x_min) * 1000
            y_mm = (py * resolution + y_min) * 1000
            final_verts.append([x_mm, y_mm])

        return final_verts

    @staticmethod
    def _filter_short_edges(vertices, min_ratio):
        """Remove vertices that create edges shorter than min_ratio * perimeter.

        This eliminates noise edges while preserving meaningful concave corners.
        """
        if len(vertices) <= 4:
            return vertices

        # Compute perimeter
        perimeter = 0.0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            dx = vertices[j][0] - vertices[i][0]
            dy = vertices[j][1] - vertices[i][1]
            perimeter += np.sqrt(dx ** 2 + dy ** 2)

        if perimeter == 0:
            return vertices

        min_length = min_ratio * perimeter

        # Iteratively remove the shortest edge until all edges are long enough
        # or we reach 4 vertices (minimum for a rectangle)
        verts = list(vertices)
        changed = True
        while changed and len(verts) > 4:
            changed = False
            n = len(verts)
            edges = []
            for i in range(n):
                j = (i + 1) % n
                dx = verts[j][0] - verts[i][0]
                dy = verts[j][1] - verts[i][1]
                edges.append(np.sqrt(dx ** 2 + dy ** 2))

            # Find shortest edge
            min_idx = int(np.argmin(edges))
            if edges[min_idx] < min_length:
                # Remove the vertex at the end of the shortest edge
                # by merging it with its neighbor
                remove_idx = (min_idx + 1) % len(verts)
                verts.pop(remove_idx)
                changed = True

        return verts

    @staticmethod
    def _validate_concave_polygon(vertices):
        """Ensure the polygon has an even number of vertices (required for
        orthogonal polygons) and at least 4 vertices.

        For orthogonal rooms:
        - 4 vertices = rectangle
        - 6 vertices = L-shape
        - 8 vertices = T/U-shape
        """
        if len(vertices) < 4:
            return vertices

        # Orthogonal polygons must have even number of vertices
        if len(vertices) % 2 != 0:
            # Find and remove the vertex creating the shortest edge
            n = len(vertices)
            min_len = float('inf')
            min_idx = 0
            for i in range(n):
                j = (i + 1) % n
                dx = vertices[j][0] - vertices[i][0]
                dy = vertices[j][1] - vertices[i][1]
                edge_len = np.sqrt(dx ** 2 + dy ** 2)
                if edge_len < min_len:
                    min_len = edge_len
                    min_idx = (i + 1) % n
            vertices.pop(min_idx)

        return vertices

    @staticmethod
    def extract_polygon(p2d, clean=True, **kwargs):
        """Extract an orthogonal boundary polygon from 2D points.

        Uses a two-pass strategy:
        1. First pass with large morphology kernel (fills gaps in walls)
        2. Check bbox_fill_ratio to detect concave rooms
        3. If concave detected, second pass with smaller kernel and tighter
           approximation to preserve L/T/U indentations
        """
        print("[2/5] Detecting orthogonal boundary polygon...")
        GeometryProcessor.last_layout_info = {}

        if clean:
            p2d, _ = GeometryProcessor.clean_projection(p2d)

        occupied, x_min, y_min, resolution, width, height = GeometryProcessor._rasterize(p2d)

        # === Build masks at multiple gap-closing strengths ===
        kernel_large = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
        solid_large = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel_large)
        contour_large = GeometryProcessor._get_largest_contour(solid_large)
        fill_ratio = GeometryProcessor._compute_fill_ratio(contour_large, solid_large.shape)

        print(f"       BBox fill ratio: {fill_ratio:.3f} (threshold: {BBOX_FILL_RATIO_THRESHOLD})")

        kernel_small = np.ones((CONCAVE_MORPH_KERNEL_SIZE, CONCAVE_MORPH_KERNEL_SIZE), np.uint8)
        solid_small = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel_small)
        contour_small = GeometryProcessor._get_largest_contour(solid_small)

        kernel_mid_size = max(3, MORPHOLOGY_KERNEL_SIZE // 3)
        kernel_mid = np.ones((kernel_mid_size, kernel_mid_size), np.uint8)
        solid_mid = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, kernel_mid)
        contour_mid = GeometryProcessor._get_largest_contour(solid_mid)

        explicit_lfit_enabled = os.getenv("PLY_LAYOUT_MODEL", "lfit").lower() in {"lfit", "auto"}
        if explicit_lfit_enabled and fill_ratio < BBOX_FILL_RATIO_THRESHOLD:
            explicit_l = GeometryProcessor._fit_l_shape_by_missing_corner(
                occupied,
                resolution,
                x_min,
                y_min,
            )
            if explicit_l is not None:
                print(
                    "       -> L-shape selected by missing-corner fit "
                    f"(corner={explicit_l['corner']}, notch={explicit_l['notch_ratio']:.3f}, "
                    f"edge_support={explicit_l['edge_support']:.3f})."
                )
                GeometryProcessor.last_layout_info = {
                    "shape_model": "L-SHAPED",
                    "source": "missing_corner_lfit",
                    "corner": explicit_l["corner"],
                    "score": float(explicit_l["score"]),
                    "bbox_fill_ratio": float(explicit_l["bbox_fill_ratio"]),
                    "notch_ratio": float(explicit_l["notch_ratio"]),
                    "occupied_outside_ratio": float(explicit_l["occupied_outside_ratio"]),
                    "poly_fill_ratio": float(explicit_l["poly_fill_ratio"]),
                    "empty_inside_ratio": float(explicit_l["empty_inside_ratio"]),
                    "edge_support": float(explicit_l["edge_support"]),
                    "edge_supports": explicit_l["edge_supports"],
                    "selected_by": "missing_corner_lfit",
                }
                return explicit_l["vertices_mm"]

        # Prefer an explicit L model when it explains the mask better than a
        # rectangle. This prevents large missing corners from being swallowed by
        # morphology or approxPolyDP.
        l_candidates = []
        for name, contour in (
            ("small", contour_small),
            ("mid", contour_mid),
            ("large", contour_large),
        ):
            fit = GeometryProcessor._fit_l_shape_by_iou(
                contour,
                solid_large.shape,
                resolution,
                x_min,
                y_min,
                support_mask=occupied,
            )
            if fit is not None:
                fit["source"] = name
                l_candidates.append(fit)

        if l_candidates:
            best_l = max(l_candidates, key=lambda item: (item["iou"], item["score"]))
            if best_l["score"] > 0.04 or fill_ratio < BBOX_FILL_RATIO_THRESHOLD:
                print(
                    "       -> L-shape selected by IoU "
                    f"(source={best_l['source']}, corner={best_l['corner']}, "
                    f"iou={best_l['iou']:.3f}, rect_iou={best_l['rectangle_iou']:.3f}, "
                    f"edge_support={best_l['edge_support']:.3f})."
                )
                print("       Generated concave orthogonal polygon (6 vertices -> L-shape).")
                GeometryProcessor.last_layout_info = {
                    "shape_model": "L-SHAPED",
                    "source": best_l["source"],
                    "corner": best_l["corner"],
                    "score": float(best_l["score"]),
                    "iou": float(best_l["iou"]),
                    "rect_iou": float(best_l["rectangle_iou"]),
                    "edge_support": float(best_l["edge_support"]),
                    "rectangle_support": float(best_l["rectangle_support"]),
                    "edge_supports": [float(value) for value in best_l.get("edge_supports", [])],
                    "bbox_fill_ratio": float(fill_ratio),
                    "selected_by": "iou_edge_support",
                }
                return best_l["vertices_mm"]

        # === Decide: convex (rectangle) or concave (L/T/U) ===
        if fill_ratio >= BBOX_FILL_RATIO_THRESHOLD:
            # Room looks rectangular — use original aggressive approach
            print("       -> Rectangular room detected, using robust bbox fitting.")
            x, y, w, h = cv2.boundingRect(contour_large)
            rectangle = [(x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1)]
            verts = GeometryProcessor._pixel_vertices_to_mm(rectangle, resolution, x_min, y_min)
            print(f"       Generated orthogonal polygon ({len(verts)} vertices).")
            rectangle_support, edge_supports = GeometryProcessor._edge_support_ratio(occupied, rectangle)
            GeometryProcessor.last_layout_info = {
                "shape_model": "RECTANGULAR",
                "score": 0.0,
                "iou": 1.0,
                "rect_iou": 1.0,
                "edge_support": float(rectangle_support),
                "edge_supports": [float(value) for value in edge_supports],
                "bbox_fill_ratio": float(fill_ratio),
                "selected_by": "robust_bbox",
            }
            return verts

        # === PASS 2: Concave room detected ===
        print("       -> Concave room detected (likely L/T/U shape), using fine-grained fitting.")

        area_small = cv2.contourArea(contour_small)
        area_mid = cv2.contourArea(contour_mid)

        # Choose the contour that is closest in area to the occupied pixels
        # (best represents the actual room shape)
        occupied_pixel_count = np.count_nonzero(occupied)
        candidates = [
            ("small", contour_small, abs(area_small - occupied_pixel_count)),
            ("mid", contour_mid, abs(area_mid - occupied_pixel_count)),
        ]
        candidates.sort(key=lambda x: x[2])
        best_name, best_contour, _ = candidates[0]
        print(f"       Using '{best_name}' kernel contour (area diff from occupancy: {candidates[0][2]:.0f}px).")

        # Use tighter epsilon for concave rooms
        eps = CONCAVE_APPROX_EPSILON * cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, eps, True)

        if len(approx) < 3:
            # Fallback to large kernel result
            print("       WARNING: Fine contour too small, falling back to standard.")
            eps = DEFAULT_APPROX_EPSILON * cv2.arcLength(contour_large, True)
            approx = cv2.approxPolyDP(contour_large, eps, True)
            if len(approx) < 3:
                raise ValueError("Polygon too small after approximation.")
            verts = GeometryProcessor._orthogonalize_contour(approx, resolution, x_min, y_min)
            return verts

        # Orthogonalize the concave contour
        verts = GeometryProcessor._orthogonalize_contour(approx, resolution, x_min, y_min)

        # Filter noise edges but preserve meaningful concave corners
        verts = GeometryProcessor._filter_short_edges(verts, MIN_EDGE_LENGTH_RATIO)

        # Ensure even vertex count for orthogonal polygon
        verts = GeometryProcessor._validate_concave_polygon(verts)

        # Final validation: if we somehow ended up with < 4 vertices, fallback
        if len(verts) < 4:
            print("       WARNING: Concave fitting produced too few vertices, falling back.")
            eps = DEFAULT_APPROX_EPSILON * cv2.arcLength(contour_large, True)
            approx = cv2.approxPolyDP(contour_large, eps, True)
            verts = GeometryProcessor._orthogonalize_contour(approx, resolution, x_min, y_min)

        shape_hint = {4: "rectangle", 6: "L-shape", 8: "T/U-shape"}.get(len(verts), f"{len(verts)}-gon")
        print(f"       Generated concave orthogonal polygon ({len(verts)} vertices -> {shape_hint}).")
        GeometryProcessor.last_layout_info = {
            "shape_model": shape_hint.upper(),
            "source": best_name,
            "bbox_fill_ratio": float(fill_ratio),
            "selected_by": "fine_contour",
        }
        return verts
