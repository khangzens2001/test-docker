"""SPAD LiDAR Occlusion-Aware Depth Fusion Module.

Associates 2D visual feature points with 8x8 SPAD LiDAR depth measurements.
Performs 3x3 local grid window checks to filter out phantom depth associations
at occlusion boundaries (max depth discontinuity delta D > 0.3m).
"""

from typing import Tuple, Optional
import numpy as np


def associate_spad_depth(
    spad_depth: np.ndarray,
    keypoints: np.ndarray,
    K: np.ndarray,
    T_cam_spad: Optional[np.ndarray] = None,
    max_discontinuity: float = 0.3,
    status_mask: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    rays: Optional[np.ndarray] = None,
    ray_frame: str = "lidar_optical",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associates SPAD depth map to 2D feature coordinates with occlusion-aware 3x3 window filtering.

    Args:
        spad_depth: (H_spad, W_spad) array of SPAD depth measurements in meters (e.g. 8x8).
        keypoints: (N, 2) array of [u, v] feature pixel coordinates.
        K: (3, 3) camera intrinsic matrix.
        T_cam_spad: (4, 4) rigid transformation matrix from SPAD frame to camera frame (T_cam_spad).
                    If None, assumes SPAD and Camera are co-aligned (Identity).
        max_discontinuity: Max depth difference threshold in 3x3 window (in meters, default 0.3).
        status_mask: Optional (H_spad, W_spad) array of status flags (status == 5 or True indicates valid).
        image_size: Optional (width, height) tuple. If None, derived as (2*cx, 2*cy).
        min_depth: Minimum valid depth in meters (default 0.1).
        max_depth: Maximum valid depth in meters (default 10.0).

    Returns:
        depths: (N,) array of estimated depth Z in meters for each feature (0.0 if invalid).
        validity: (N,) boolean array indicating whether each depth association is valid.
        variances: (N,) array of depth variances in m^2 (np.inf if invalid).
    """
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if keypoints.ndim == 1:
        keypoints = keypoints.reshape(1, 2)
    
    n_pts = keypoints.shape[0]
    depths = np.zeros(n_pts, dtype=np.float64)
    validity = np.zeros(n_pts, dtype=bool)
    variances = np.full(n_pts, np.inf, dtype=np.float64)
    
    if n_pts == 0:
        return depths, validity, variances

    spad_depth = np.asarray(spad_depth, dtype=np.float64)
    h_spad, w_spad = spad_depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    if image_size is None:
        img_w = float(2.0 * cx)
        img_h = float(2.0 * cy)
    else:
        img_w, img_h = float(image_size[0]), float(image_size[1])

    # Compute rotation R_spad_cam (camera frame -> SPAD frame)
    if T_cam_spad is None:
        R_spad_cam = np.eye(3, dtype=np.float64)
    else:
        T_cam_spad = np.asarray(T_cam_spad, dtype=np.float64)
        R_cam_spad = T_cam_spad[:3, :3]
        R_spad_cam = R_cam_spad.T

    is_bool_status = False
    if status_mask is not None:
        status_mask = np.asarray(status_mask)
        is_bool_status = np.issubdtype(status_mask.dtype, np.bool_)

    for i in range(n_pts):
        u, v = keypoints[i, 0], keypoints[i, 1]

        # 1. Bounds check in image space
        if u < 0 or u >= img_w or v < 0 or v >= img_h:
            continue

        # 2. Unproject pixel to camera frame ray direction
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        ray_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float64)

        # 3. Transform ray direction to SPAD frame
        ray_spad = R_spad_cam @ ray_cam
        xs, ys, zs = ray_spad[0], ray_spad[1], ray_spad[2]

        if zs <= 1e-6:
            continue

        # 4. Project ray in SPAD normalized plane and map to SPAD grid coordinates
        x_norm = xs / zs
        y_norm = ys / zs

        # Re-project to image coordinates in camera frame for grid mapping
        u_proj = x_norm * fx + cx
        v_proj = y_norm * fy + cy

        c_float = (u_proj / img_w) * w_spad
        r_float = (v_proj / img_h) * h_spad

        if c_float < 0 or c_float >= w_spad or r_float < 0 or r_float >= h_spad:
            continue

        r = int(np.floor(r_float))
        c = int(np.floor(c_float))

        r = max(0, min(h_spad - 1, r))
        c = max(0, min(w_spad - 1, c))

        # 5. Extract 3x3 local grid window around (r, c)
        r_min = max(0, r - 1)
        r_max = min(h_spad - 1, r + 1)
        c_min = max(0, c - 1)
        c_max = min(w_spad - 1, c + 1)

        window_depths = spad_depth[r_min:r_max + 1, c_min:c_max + 1]

        # Check status mask if provided
        if status_mask is not None:
            window_status = status_mask[r_min:r_max + 1, c_min:c_max + 1]
            center_status = status_mask[r, c]
            if is_bool_status:
                center_valid = bool(center_status)
                valid_cells_mask = window_status
            else:
                center_valid = center_status in (1, 5, 9)
                valid_cells_mask = (
                    (window_status == 1) | (window_status == 5) | (window_status == 9)
                )

            if not center_valid:
                continue

            valid_window_depths = window_depths[valid_cells_mask]
        else:
            valid_window_depths = window_depths.ravel()

        # Filter positive valid depths
        valid_window_depths = valid_window_depths[valid_window_depths > 0]
        if len(valid_window_depths) == 0:
            continue

        # 6. Check depth discontinuity Delta D in 3x3 window
        min_d = np.min(valid_window_depths)
        max_d = np.max(valid_window_depths)
        delta_d = max_d - min_d

        if delta_d > max_discontinuity:
            # Occlusion boundary detected -> Reject phantom depth
            continue

        center_depth = spad_depth[r, c]
        if center_depth < min_depth or center_depth > max_depth:
            continue

        # If calibrated rays are provided, compute orthogonal camera depth Z_C
        if rays is not None:
            rays_arr = np.asarray(rays, dtype=np.float64)
            if rays_arr.ndim == 2 and rays_arr.shape[0] >= 64 and rays_arr.shape[1] == 3:
                r_vec = rays_arr[r * 8 + c]
            elif rays_arr.ndim == 3 and rays_arr.shape[0] == 8 and rays_arr.shape[1] == 8 and rays_arr.shape[2] == 3:
                r_vec = rays_arr[r, c]
            else:
                r_vec = None

            if r_vec is not None:
                r_norm = float(np.linalg.norm(r_vec))
                if r_norm > 1e-6:
                    cos_theta = abs(float(r_vec[2])) / r_norm
                    assigned_depth = center_depth * cos_theta
                else:
                    assigned_depth = center_depth
            else:
                assigned_depth = center_depth
        else:
            assigned_depth = center_depth

        if assigned_depth < min_depth or assigned_depth > max_depth:
            continue

        # Compute depth variance: sensor noise (0.02 + 0.03 * Z)^2 + local depth variance
        sensor_sigma = 0.02 + 0.03 * assigned_depth
        window_var = float(np.var(valid_window_depths)) if len(valid_window_depths) > 1 else 0.0
        total_variance = (sensor_sigma ** 2) + window_var

        depths[i] = assigned_depth
        validity[i] = True
        variances[i] = total_variance

    return depths, validity, variances


class SpadDepthFusion:
    """Class wrapper for SPAD LiDAR occlusion-aware depth fusion."""

    def __init__(
        self,
        max_discontinuity_m: float = 0.3,
        min_depth_m: float = 0.1,
        max_depth_m: float = 10.0,
    ):
        """Initialize SpadDepthFusion with configuration parameters.

        Args:
            max_discontinuity_m: Max allowed depth discontinuity in 3x3 window (meters).
            min_depth_m: Minimum valid depth in meters.
            max_depth_m: Maximum valid depth in meters.
        """
        self.max_discontinuity_m = max_discontinuity_m
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def associate_depth(
        self,
        spad_depth: np.ndarray,
        keypoints: np.ndarray,
        K: np.ndarray,
        T_cam_spad: Optional[np.ndarray] = None,
        status_mask: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Associate SPAD depth measurements with 2D feature coordinates.

        Returns:
            depths: (N,) array of estimated depth Z in meters.
            validity: (N,) boolean array indicating valid associations.
            variances: (N,) array of depth variances (m^2).
        """
        return associate_spad_depth(
            spad_depth=spad_depth,
            keypoints=keypoints,
            K=K,
            T_cam_spad=T_cam_spad,
            max_discontinuity=self.max_discontinuity_m,
            status_mask=status_mask,
            image_size=image_size,
            min_depth=self.min_depth_m,
            max_depth=self.max_depth_m,
        )
