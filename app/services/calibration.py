import json
import os
import warnings
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

def normalize_quaternions_vectorized(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) to unit length, keeping trajectory continuity.
    
    Safely falls back to identity [0, 0, 0, 1] on NaN, infinite, or degenerate inputs.
    Supports both 1D (4,) and 2D (N, 4) arrays.
    """
    q_arr = np.array(q, dtype=np.float64)
    original_ndim = q_arr.ndim
    
    if original_ndim == 1:
        q_arr = q_arr[np.newaxis, :]
        
    if q_arr.ndim != 2 or q_arr.shape[1] != 4:
        raise ValueError("Quaternion array must be shape (4,) or (N, 4).")
        
    # Guard: Replace NaNs/Infs with identity [0, 0, 0, 1]
    invalid_rows = np.any(~np.isfinite(q_arr), axis=1)
    q_arr[invalid_rows] = np.array([0.0, 0.0, 0.0, 1.0])
    
    norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
    
    # Guard: Handle zero-norm rows using indexing instead of squeeze to prevent shape crashes when N=1
    zero_norm = (norms < 1e-8)[:, 0]
    q_arr[zero_norm] = np.array([0.0, 0.0, 0.0, 1.0])
    norms[zero_norm] = 1.0
    
    q_norm = q_arr / norms
    
    # Trajectory alignment: ensure continuity by preventing antipodal sign jumps
    if q_norm.shape[0] > 1:
        for i in range(1, q_norm.shape[0]):
            if np.dot(q_norm[i], q_norm[i-1]) < 0:
                q_norm[i] = -q_norm[i]
    else:
        # Standalone quaternion: keep qw >= 0
        if q_norm[0, 3] < 0:
            q_norm[0] = -q_norm[0]
            
    if original_ndim == 1:
        return q_norm[0]
    return q_norm

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Wrapper calling the vectorized, trajectory-aware normalizer."""
    return normalize_quaternions_vectorized(q)

def parse_4x4_pose_matrix(flat_matrix: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Extract translation [tx, ty, tz] and normalized quaternion from 4x4 transform matrix.
    
    Assumes standard column-major array convention (ARCore/ARKit/OpenGL).
    Uses SVD-based projection to SO(3) to prevent Rotation.from_matrix from crashing.
    """
    T = flat_matrix.reshape((4, 4), order='F')  # Fix: column-major reshape to prevent coordinate swapping
    t = T[0:3, 3]
    R = T[0:3, 0:3]
    
    # Orthonormalize R using SVD
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        Vt[-1, :] *= -1
        R_ortho = U @ Vt
        
    r_obj = Rotation.from_matrix(R_ortho)
    q = r_obj.as_quat()
    q = normalize_quaternion(q)
    return float(t[0]), float(t[1]), float(t[2]), q


def _is_schema2(data: dict) -> bool:
    return isinstance(data, dict) and "lidar_to_camera" in data


def _T_from_schema2(data: dict) -> tuple[np.ndarray, np.ndarray]:
    l2c = data["lidar_to_camera"]
    R_flat = np.asarray(l2c["rotation_row_major_3x3"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(l2c["translation_meters"], dtype=np.float64)
    T_lidar = np.eye(4)
    T_lidar[0:3, 0:3] = R_flat
    T_lidar[0:3, 3] = t
    T_imu = np.eye(4)
    T_imu[0:3, 0:3] = np.diag([1.0, -1.0, -1.0])
    return T_imu, T_lidar


def _rays_from_schema2(data: dict) -> list[list[float]] | None:
    rays_in = data.get("tof_rays_lidar")
    if not isinstance(rays_in, list) or len(rays_in) != 64:
        return None
    ordered = sorted(rays_in, key=lambda r: int(r["zone"]))
    out = []
    for item in ordered:
        v = np.array([float(item["x"]), float(item["y"]), float(item["z"])], dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("tof_rays_lidar contains a zero vector")
        out.append((v / n).tolist())
    return out


def load_extrinsics(calibration_path: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Load T_imu_camera and T_lidar_camera 4x4 matrices from calibration.json, raising explicit validation errors."""
    default_T_lidar = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.03],
        [0.0, 0.0, 0.0, 1.0]
    ])
    default_T_imu = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0,  0.05],
        [0.0,  0.0, -1.0, -0.01],
        [0.0,  0.0,  0.0,  1.0]
    ])

    if not calibration_path or not os.path.exists(calibration_path):
        # Return defaults when calibration file doesn't exist (expected developer setup)
        return default_T_imu, default_T_lidar

    try:
        with open(calibration_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if _is_schema2(data):
            return _T_from_schema2(data)

        def make_matrix(cfg_key):
            if cfg_key not in data:
                raise KeyError(f"Key '{cfg_key}' missing from calibration config.")
            cfg = data[cfg_key]
            if "quaternion_xyzw" not in cfg or "translation_m" not in cfg:
                raise ValueError(f"quaternion_xyzw or translation_m missing in key '{cfg_key}'.")
            q = np.array(cfg["quaternion_xyzw"])
            if len(q) != 4:
                raise ValueError(f"quaternion_xyzw must be of length 4, got {len(q)}")
            q = normalize_quaternion(q)
            t = np.array(cfg["translation_m"])
            if len(t) != 3:
                raise ValueError(f"translation_m must be of length 3, got {len(t)}")
            R = Rotation.from_quat(q).as_matrix()
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = t
            return T
            
        T_imu = make_matrix("T_imu_camera")
        T_lidar = make_matrix("T_lidar_camera")
        return T_imu, T_lidar
    except Exception as exc:
        raise ValueError(f"Failed to parse calibration file at {calibration_path}: {exc}") from exc


T_GL_CV = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


VL53L8CX_VALID_STATUSES: tuple[int, ...] = (5, 6, 9)


def _display_to_camera_sensor_rotation(rot_deg: int | float | str | None) -> np.ndarray:
    """Return 3x3 rotation R_disp_cam from Camera Sensor frame to Portrait Display frame.

    Matches compute_lidar_zone_mapping:
      - theta = 0 deg:   X_disp = +X_cam, Y_disp = +Y_cam
      - theta = 90 deg:  X_disp = -Y_cam, Y_disp = +X_cam
      - theta = 180 deg: X_disp = -X_cam, Y_disp = -Y_cam
      - theta = 270 deg: X_disp = +Y_cam, Y_disp = -X_cam
    """
    if rot_deg is None:
        return np.eye(3, dtype=np.float64)
    try:
        theta = int(round(float(rot_deg))) % 360
    except (ValueError, TypeError):
        return np.eye(3, dtype=np.float64)
    if theta == 90:
        return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    if theta == 180:
        return np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    if theta == 270:
        return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64)


display_to_camera_sensor_rotation = _display_to_camera_sensor_rotation


def resolve_sensor_to_vio_rotation(
    rot_deg: int | float | str | None,
    img_w: int | None = None,
    img_h: int | None = None,
    R_wc_all: np.ndarray | None = None,
    g_world: np.ndarray | None = None,
    *,
    camera_frame_convention: str | None = None,
    pose_df: pd.DataFrame | None = None,
    already_sensor_frame: bool | None = None,
    pose_source: str | None = None,
) -> np.ndarray:
    """Deterministically resolve the 3x3 rotation R_dc from Camera Sensor frame to VIO Pose frame."""
    if camera_frame_convention == "sensor_opencv" or already_sensor_frame is True:
        return np.eye(3, dtype=np.float64)
    if pose_source == "server_vio":
        return np.eye(3, dtype=np.float64)
    if pose_df is not None and isinstance(pose_df, pd.DataFrame):
        if pose_df.attrs.get("camera_frame_convention") == "sensor_opencv":
            return np.eye(3, dtype=np.float64)
        if pose_df.attrs.get("pose_source") == "server_vio":
            return np.eye(3, dtype=np.float64)

    r_dc_cand = _display_to_camera_sensor_rotation(rot_deg)
    if np.allclose(r_dc_cand, np.eye(3)):
        return np.eye(3, dtype=np.float64)
    if img_w is not None and img_h is not None and int(img_w) <= int(img_h):
        return np.eye(3, dtype=np.float64)
    if R_wc_all is None:
        return r_dc_cand
    R_arr = np.asarray(R_wc_all, dtype=np.float64)
    if R_arr.ndim == 2 and R_arr.shape == (3, 3):
        R_arr = R_arr[None, :, :]
    if R_arr.ndim != 3 or R_arr.shape[0] == 0 or R_arr.shape[1:] != (3, 3):
        return r_dc_cand
    g_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if g_world is not None:
        g_in = np.asarray(g_world, dtype=np.float64).reshape(-1)
        if g_in.size >= 3 and np.all(np.isfinite(g_in[:3])) and float(np.linalg.norm(g_in[:3])) > 1e-6:
            g_vec = g_in[:3] / float(np.linalg.norm(g_in[:3]))
    g_cam = np.einsum("nij,i->nj", R_arr, g_vec)
    non_steep = np.abs(g_cam[:, 2]) < 0.85
    g_sub = g_cam[non_steep] if int(np.count_nonzero(non_steep)) >= 5 else g_cam
    med_x = float(np.median(np.abs(g_sub[:, 0])))
    med_y = float(np.median(np.abs(g_sub[:, 1])))
    if med_y > med_x and med_y >= 0.25:
        return r_dc_cand
    return np.eye(3, dtype=np.float64)


def chain_wahba_to_lidar_camera_extrinsics(
    T_lidar_raw: np.ndarray,
    R_phone_hub: np.ndarray | None,
    rot_deg: int | float | str | None = 90,
    extrinsics_calibrated: bool | None = None,
) -> np.ndarray:
    """Chain R_phone_hub (from Gyro Wahba) into T_lidar_camera for remapped camera-frame zones.

    Because remap_lidar_zones_to_camera_frame already maps the 8x8 VL53L8CX zones from the
    Hub/Display orientation into the Camera Sensor frame, an uncalibrated bootstrap rotation
    of diag(-1, -1, 1) double-flips the LiDAR rays by 180 degrees. When R_phone_hub is available,
    the residual physical tilt between Hub IMU and Phone Display is R_disp_hub = T_GL_CV[:3,:3].T @ R_phone_hub,
    which is mapped into the Camera Sensor frame via R_disp_cam.
    """
    T_out = np.asarray(T_lidar_raw, dtype=np.float64).copy()
    if T_out.shape != (4, 4):
        return np.eye(4, dtype=np.float64)
    R0 = T_out[:3, :3]
    is_inverted_bootstrap = (not bool(extrinsics_calibrated)) and (R0[0, 0] < -0.5) and (R0[1, 1] < -0.5)
    if R_phone_hub is not None:
        R_ph = np.asarray(R_phone_hub, dtype=np.float64)
        if R_ph.shape == (3, 3) and np.all(np.isfinite(R_ph)):
            R_disp_hub = T_GL_CV[:3, :3].T @ R_ph
            # Verify R_disp_hub is close to identity (valid physical mount tilt < 15 deg)
            trace_val = float(np.trace(R_disp_hub))
            if trace_val > 2.80:
                R_dc = _display_to_camera_sensor_rotation(rot_deg)
                R_cam_lidar = R_dc.T @ R_disp_hub @ R_dc
                T_out[:3, :3] = _svd_so3(R_cam_lidar)
                if is_inverted_bootstrap:
                    # Also rotate lever arm translation from display/bootstrap frame to camera sensor frame
                    T_out[:3, 3] = R_dc.T @ T_out[:3, 3]
                return T_out
    if is_inverted_bootstrap:
        R_dc = _display_to_camera_sensor_rotation(rot_deg)
        T_out[:3, :3] = np.eye(3, dtype=np.float64)
        T_out[:3, 3] = R_dc.T @ T_out[:3, 3]
    return T_out


def load_T_bc(session_dir: str) -> np.ndarray:
    ext_path = os.path.join(session_dir, "processed_extrinsics.json")
    manifest_path = os.path.join(session_dir, "manifest.json")
    meaning = ""
    T_raw = None
    rot_deg = None
    if os.path.isfile(ext_path):
        with open(ext_path, encoding="utf-8") as f:
            data = json.load(f)
        meaning = str(data.get("T_imu_camera_meaning") or "")
        arr = np.asarray(data.get("T_imu_camera"), dtype=np.float64)
        if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
            T_raw = arr
        if "camera_sensor_to_display_rotation_deg" in data:
            rot_deg = data.get("camera_sensor_to_display_rotation_deg")
    if rot_deg is None and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as mf:
                mdata = json.load(mf)
            _, _, raw_rot = _extract_raw_lidar_display_metadata(mdata)
            if raw_rot is not None:
                rot_deg = raw_rot
        except Exception:
            pass
    if T_raw is None:
        T_raw, _, meta = load_session_extrinsics(session_dir)
        meaning = str(meta.get("T_imu_camera_meaning") or meaning)
    schema13 = meaning == "phone_imu_android_device_screen_to_arcore_sensor_image"
    if schema13 and np.allclose(T_raw[:3, :3], T_GL_CV[:3, :3], atol=1e-3):
        T_out = T_raw.copy()
    elif schema13:
        T_out = (T_raw @ T_GL_CV).copy()
    else:
        T_out = T_raw.copy()
    if rot_deg is not None:
        R_dc = _display_to_camera_sensor_rotation(rot_deg)
        T_out[:3, :3] = T_out[:3, :3] @ R_dc
    return T_out



def load_camera_intrinsics(path: str) -> np.ndarray:
    """Load package §4.4 camera_matrix.csv as a 3x3 matrix. Fail-closed; no default K."""
    try:
        if not os.path.isfile(path):
            raise OSError("missing camera_matrix.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            arr = np.loadtxt(path, delimiter=",", encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError("Invalid camera_matrix.csv.") from exc

    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError("Invalid camera_matrix.csv.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Invalid camera_matrix.csv.")
    if float(arr[0, 0]) <= 0.0 or float(arr[1, 1]) <= 0.0:
        raise ValueError("Invalid camera_matrix.csv.")
    return arr

def interpolate_vio_poses(vio_timestamps_ns: np.ndarray, translations: np.ndarray, quaternions: np.ndarray, target_timestamps_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate translations and SLERP interpolate quaternions of VIO poses at target timestamps.
    
    Timestamps are internally shifted to zero relative to the first timestamp to prevent
    float64 truncation errors during SLERP.
    """
    if len(target_timestamps_ns) == 0:
        return np.empty((0, 3)), np.empty((0, 4))
        
    sort_idx = np.argsort(vio_timestamps_ns)
    t_vio = vio_timestamps_ns[sort_idx]
    x_vio = translations[sort_idx]
    q_vio = quaternions[sort_idx]
    
    # Guard empty inputs
    if len(t_vio) == 0:
        raise ValueError("VIO dataset contains zero records.")
        
    # Guard: handle single VIO frame datasets or degenerate inputs
    if len(t_vio) < 2:
        x_interp = np.tile(x_vio[0], (len(target_timestamps_ns), 1))
        q_interp = np.tile(q_vio[0], (len(target_timestamps_ns), 1))
        return x_interp, q_interp

    # Deduplicate non-strictly increasing VIO timestamps
    if not np.all(np.diff(t_vio) > 0):
        t_vio, unique_indices = np.unique(t_vio, return_index=True)
        x_vio = x_vio[unique_indices]
        q_vio = q_vio[unique_indices]
        if len(t_vio) < 2:
            x_interp = np.tile(x_vio[0], (len(target_timestamps_ns), 1))
            q_interp = np.tile(q_vio[0], (len(target_timestamps_ns), 1))
            return x_interp, q_interp

    # Shift time origin to 0 to prevent float64 truncation precision loss
    t0 = t_vio[0]
    t_vio_shifted = (t_vio - t0).astype(np.float64)
    t_target_shifted = (target_timestamps_ns - t0).astype(np.float64)
    
    # Ensure targets are within boundaries to avoid extrapolation errors
    t_min, t_max = t_vio_shifted[0], t_vio_shifted[-1]
    t_target_clipped = np.clip(t_target_shifted, t_min, t_max)
    
    # Linear interpolation for translations
    x_interp = np.zeros((len(t_target_clipped), 3))
    for i in range(3):
        x_interp[:, i] = np.interp(t_target_clipped, t_vio_shifted, x_vio[:, i])
        
    # Align VIO quaternions before Slerp fitting to prevent spin artifacts
    q_vio_aligned = normalize_quaternions_vectorized(q_vio)

    # Standardize VIO rotations from OpenGL (+X, +Y, -Z) to OpenCV (+X, -Y, +Z)
    # This represents a 180-degree rotation around the X-axis: q_cv = q_gl * [1, 0, 0, 0]
    q_x_180 = np.array([1.0, 0.0, 0.0, 0.0])
    rot_gl = Rotation.from_quat(q_vio_aligned)
    rot_cv = rot_gl * Rotation.from_quat(q_x_180)
    
    # SLERP interpolation for rotations
    slerp = Slerp(t_vio_shifted, rot_cv)
    rot_interp = slerp(t_target_clipped)
    q_interp = rot_interp.as_quat()
    
    # Re-normalize and preserve temporal continuity (dot_product check)
    q_interp = normalize_quaternions_vectorized(q_interp)
    return x_interp, q_interp

def project_lidar_rays() -> list[list[float]]:
    """Compute 64 normalized unit ray direction vectors.
    
    Adheres strictly to OpenCV conventions (X-right, Y-down, Z-forward) and utilizes
    physically accurate equiangular (spherical) spacing for the VL53L8CX ToF sensor.
    All rays are normalized to unit length to correctly project raw radial ToF range distances.
    """
    fov_deg = 45.0
    grid_size = 8
    angular_step = np.radians(fov_deg / grid_size)  # 5.625 degrees
    half_grid = (grid_size - 1) / 2.0  # 3.5
    
    rays = []
    for r in range(grid_size):
        # elevation: r=0 (top) -> -3.5 * step -> negative Y (up)
        #            r=7 (bottom) -> +3.5 * step -> positive Y (down)
        elevation = (r - half_grid) * angular_step
        for c in range(grid_size):
            # azimuth: c=0 (left) -> -3.5 * step -> negative X (left)
            #          c=7 (right) -> +3.5 * step -> positive X (right)
            azimuth = (c - half_grid) * angular_step
            
            # Compute spherical unit coordinates
            x = np.cos(elevation) * np.sin(azimuth)
            y = np.sin(elevation)
            z = np.cos(elevation) * np.cos(azimuth)
            rays.append([float(x), float(y), float(z)])
    return rays


def load_session_extrinsics(session_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    schema2 = os.path.join(session_dir, "lidar_camera_extrinsics.json")
    calib = os.path.join(session_dir, "calibration.json")
    if os.path.isfile(schema2):
        T_imu, T_lidar = load_extrinsics(schema2)
        with open(schema2, encoding="utf-8") as f:
            data = json.load(f)
        rays = _rays_from_schema2(data)
        if rays is None:
            rays = project_lidar_rays()
            ray_frame, source = "opencv_camera", "analytic_vl53l8cx"
        else:
            ray_frame, source = "lidar_optical", "tof_rays_lidar"
        meaning = "phone_imu_android_device_screen_to_arcore_sensor_image"
    elif os.path.isfile(calib):
        T_imu, T_lidar = load_extrinsics(calib)
        rays = project_lidar_rays()
        ray_frame, source = "opencv_camera", "analytic_vl53l8cx"
        meaning = "calibration_json_T_imu_camera"
        data = {}
    else:
        T_imu, T_lidar = load_extrinsics(None)
        rays = project_lidar_rays()
        ray_frame, source = "opencv_camera", "analytic_vl53l8cx"
        meaning = "default_T_imu_camera"
        data = {}
    return T_imu, T_lidar, {
        "T_imu_camera_meaning": meaning,
        "ray_frame": ray_frame,
        "source": source,
        "rays": rays,
        "extrinsics_source": data.get("source") or data.get("extrinsics_source"),
        "extrinsics_calibrated": data.get("extrinsics_calibrated"),
    }


TOF_Z_MIN_M = 0.05
TOF_Z_MAX_M = 8.0


def _extract_raw_lidar_display_metadata(
    manifest: dict | None,
) -> tuple[bool | None, bool | None, int | None]:
    """Internal helper to extract raw display metadata without applying fallback defaults."""
    if not isinstance(manifest, dict):
        return None, None, None

    streams_rgb = {}
    streams = manifest.get("streams")
    if isinstance(streams, dict) and isinstance(streams.get("rgb"), dict):
        streams_rgb = streams["rgb"]

    flip_h_val = manifest.get("lidar_heatmap_display_flip_h")
    if flip_h_val is None:
        flip_h_val = streams_rgb.get("lidar_heatmap_display_flip_h")
    flip_h = bool(flip_h_val) if flip_h_val is not None else None

    flip_v_val = manifest.get("lidar_heatmap_display_flip_v")
    if flip_v_val is None:
        flip_v_val = streams_rgb.get("lidar_heatmap_display_flip_v")
    flip_v = bool(flip_v_val) if flip_v_val is not None else None

    rot_deg_val = manifest.get("camera_sensor_to_display_rotation_deg")
    if rot_deg_val is None:
        rot_deg_val = streams_rgb.get("camera_sensor_to_display_rotation_deg")
    if rot_deg_val is not None:
        try:
            rot_deg = int(round(float(rot_deg_val)))
        except (ValueError, TypeError):
            rot_deg = None
    else:
        rot_deg = None

    return flip_h, flip_v, rot_deg


def extract_lidar_display_metadata(
    manifest: dict | None,
    default_flip_h: bool = False,
    default_flip_v: bool = False,
    default_rot_deg: int = 90,
) -> tuple[bool, bool, int]:
    """Extract LiDAR heatmap display flip and rotation metadata from manifest.json.
    
    Checks both root level and streams.rgb block:
      - lidar_heatmap_display_flip_h: bool (default: False)
      - lidar_heatmap_display_flip_v: bool (default: False)
      - camera_sensor_to_display_rotation_deg: int in {0, 90, 180, 270} (default: 90)
      
    Returns: (flip_h, flip_v, rot_deg)
    """
    raw_h, raw_v, raw_rot = _extract_raw_lidar_display_metadata(manifest)
    flip_h = default_flip_h if raw_h is None else raw_h
    flip_v = default_flip_v if raw_v is None else raw_v
    rot_deg = default_rot_deg if raw_rot is None else raw_rot
    return flip_h, flip_v, rot_deg


def compute_lidar_zone_mapping(
    flip_h: bool = False,
    flip_v: bool = False,
    rot_deg: int | float | str = 0,
) -> np.ndarray:
    """Compute an index mapping array of length 64 from camera ray index to raw sensor distance index.
    
    The user aligns the LiDAR heatmap on the mobile screen (Display Frame, 8x8 grid with
    r_disp, c_disp in [0..7]).
    
    1. Mapping between user display cell and raw sensor index d_raw:
       r_raw = 7 - r_disp if flip_v else r_disp
       c_raw = 7 - c_disp if flip_h else c_disp
       raw_index = r_raw * 8 + c_raw
       
    2. The mobile display is rotated by theta = rot_deg (clockwise) relative to the raw camera sensor.
       To transform a zone from Display Frame back to Camera Sensor Frame, we rotate by -theta:
       - theta = 0 deg:   r_cam = r_disp,     c_cam = c_disp
       - theta = 90 deg:  r_cam = 7 - c_disp, c_cam = r_disp
       - theta = 180 deg: r_cam = 7 - r_disp, c_cam = 7 - c_disp
       - theta = 270 deg: r_cam = c_disp,     c_cam = 7 - r_disp
       
    3. The distance assigned to ray (r_cam, c_cam) (index r_cam * 8 + c_cam) is d_raw[raw_index].
       Therefore, mapping[cam_index] = raw_index.
    """
    mapping = np.zeros(64, dtype=np.int64)
    try:
        theta = int(round(float(rot_deg))) % 360
    except (ValueError, TypeError):
        theta = 90
    if theta not in (0, 90, 180, 270):
        theta = int(round(theta / 90.0)) * 90 % 360

    for r_disp in range(8):
        for c_disp in range(8):
            r_raw = (7 - r_disp) if flip_v else r_disp
            c_raw = (7 - c_disp) if flip_h else c_disp
            raw_index = r_raw * 8 + c_raw

            if theta == 0:
                r_cam = r_disp
                c_cam = c_disp
            elif theta == 90:
                r_cam = 7 - c_disp
                c_cam = r_disp
            elif theta == 180:
                r_cam = 7 - r_disp
                c_cam = 7 - c_disp
            else:  # theta == 270
                r_cam = c_disp
                c_cam = 7 - r_disp

            cam_index = r_cam * 8 + c_cam
            mapping[cam_index] = raw_index

    return mapping


def remap_lidar_zones_to_camera_frame(
    distances_m: np.ndarray | list[float],
    flip_h: bool = False,
    flip_v: bool = False,
    rot_deg: int | float | str = 0,
) -> np.ndarray:
    """Remap 64 VL53L8CX LiDAR zone distances to match the camera sensor frame."""
    d = np.asarray(distances_m)
    if d.size < 64:
        return d
    try:
        theta = int(round(float(rot_deg))) % 360
    except (ValueError, TypeError):
        theta = 0
    if not flip_h and not flip_v and theta == 0:
        return d
    mapping = compute_lidar_zone_mapping(flip_h=flip_h, flip_v=flip_v, rot_deg=rot_deg)
    orig_shape = d.shape
    flat = d.reshape(-1)
    aligned = np.copy(flat)
    aligned[:64] = flat[mapping]
    return aligned.reshape(orig_shape)


def _resolve_deskew_rotation(
    R_deskew: np.ndarray | None = None,
    omega_cam: np.ndarray | None = None,
    dt_s: float | None = None,
) -> np.ndarray | None:
    if R_deskew is not None:
        try:
            Rd = np.asarray(R_deskew, dtype=np.float64)
            if Rd.shape == (3, 3) and np.all(np.isfinite(Rd)):
                return Rd
        except (ValueError, TypeError):
            pass
    if omega_cam is not None and dt_s is not None:
        try:
            w_vec = np.asarray(omega_cam, dtype=np.float64).reshape(3)
            dt_val = float(dt_s)
            if np.all(np.isfinite(w_vec)) and np.isfinite(dt_val) and abs(dt_val) > 0.0:
                return Rotation.from_rotvec(w_vec * dt_val).as_matrix()
        except (ValueError, TypeError):
            pass
    return None


def probe_session_image_size(
    session_dir: str | None = None,
    rgb_path: str | None = None,
    K: np.ndarray | None = None,
    default_size: tuple[int, int] = (640, 480),
) -> tuple[int, int]:
    """Read (width, height) dynamically from RGB image header first, falling back to K (2*cx, 2*cy)."""
    cand_paths: list[str] = []
    if rgb_path and os.path.isfile(rgb_path):
        cand_paths.append(rgb_path)
    if session_dir and os.path.isdir(session_dir):
        rgb_dir = os.path.join(session_dir, "rgb")
        if os.path.isdir(rgb_dir):
            try:
                for fn in sorted(os.listdir(rgb_dir)):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        fp = os.path.join(rgb_dir, fn)
                        if os.path.isfile(fp):
                            cand_paths.append(fp)
                            break
            except Exception:
                pass
    for p in cand_paths:
        try:
            from PIL import Image

            with Image.open(p) as im:
                w, h = im.size
                if w > 16 and h > 16:
                    return int(w), int(h)
        except Exception:
            pass
    if K is None and session_dir and os.path.isdir(session_dir):
        k_csv = os.path.join(session_dir, "camera_matrix.csv")
        if os.path.isfile(k_csv):
            try:
                K = load_camera_intrinsics(k_csv)
            except Exception:
                K = None
    if K is not None:
        try:
            k_arr = np.asarray(K, dtype=np.float64)
            if k_arr.shape == (3, 3) and k_arr[0, 2] > 16 and k_arr[1, 2] > 16:
                return int(round(float(k_arr[0, 2]) * 2.0)), int(round(float(k_arr[1, 2]) * 2.0))
        except Exception:
            pass
    return int(default_size[0]), int(default_size[1])


def compute_lidar_rgb_deskew_rotation(
    t_lidar_ns: int | float | None,
    t_rgb_ns: int | float | None,
    pose_df: pd.DataFrame | None = None,
    session_R_dc: np.ndarray | None = None,
    max_dt_ns: int = 100_000_000,
    imu_df: pd.DataFrame | None = None,
    R_bc: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ΔR = R_WC(t_rgb)^T @ R_WC(t_lidar) mapping camera frame at t_lidar to t_rgb."""
    if t_lidar_ns is None or t_rgb_ns is None:
        return np.eye(3, dtype=np.float64)
    try:
        tl = float(t_lidar_ns)
        tr = float(t_rgb_ns)
        if not (np.isfinite(tl) and np.isfinite(tr)):
            return np.eye(3, dtype=np.float64)
        if max(abs(tl), abs(tr)) < 1e8 and max(abs(tl), abs(tr)) > 0.0 and abs(tl - tr) < 1e4:
            tl *= 1e9
            tr *= 1e9
        dt_ns = abs(tl - tr)
        if dt_ns == 0.0 or dt_ns > float(max_dt_ns):
            return np.eye(3, dtype=np.float64)

        if pose_df is not None and not pose_df.empty:
            ts_col = (
                "device_timestamp_ns"
                if "device_timestamp_ns" in pose_df.columns
                else ("timestamp" if "timestamp" in pose_df.columns else None)
            )
            if ts_col is not None and all(c in pose_df.columns for c in ("qx", "qy", "qz", "qw")):
                ts = pd.to_numeric(pose_df[ts_col], errors="coerce").to_numpy(dtype=np.float64)
                quats = (
                    pose_df[["qx", "qy", "qz", "qw"]]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=np.float64)
                )
                valid = np.isfinite(ts) & np.all(np.isfinite(quats), axis=1)
                if int(np.count_nonzero(valid)) >= 2:
                    ts_v = ts[valid]
                    dt_v = float(np.median(np.abs(np.diff(ts_v)))) if len(ts_v) >= 2 else 0.0
                    if dt_v < 1e4 and float(np.median(np.abs(ts_v))) < 1e11 and max(abs(tl), abs(tr)) >= 1e8:
                        ts_v = ts_v * 1e9
                    q_v = quats[valid]
                    order = np.argsort(ts_v)
                    ts_v = ts_v[order]
                    q_v = q_v[order]
                    norms = np.linalg.norm(q_v, axis=1, keepdims=True)
                    ok_n = norms.reshape(-1) > 1e-8
                    if int(np.count_nonzero(ok_n)) >= 2:
                        ts_v = ts_v[ok_n]
                        q_v = q_v[ok_n] / norms[ok_n]

                        def _interp_R_wd(t_query: float) -> np.ndarray:
                            if t_query <= ts_v[0]:
                                return Rotation.from_quat(q_v[0]).as_matrix()
                            if t_query >= ts_v[-1]:
                                return Rotation.from_quat(q_v[-1]).as_matrix()
                            idx = int(np.searchsorted(ts_v, t_query))
                            t0, t1 = float(ts_v[idx - 1]), float(ts_v[idx])
                            q0, q1 = q_v[idx - 1], q_v[idx].copy()
                            if float(np.dot(q0, q1)) < 0.0:
                                q1 = -q1
                            if t1 > t0:
                                slerp_pair = Slerp([0.0, t1 - t0], Rotation.from_quat([q0, q1]))
                                return slerp_pair([t_query - t0])[0].as_matrix()
                            return Rotation.from_quat(q0).as_matrix()

                        R_wd_rgb = _interp_R_wd(tr)
                        R_wd_lidar = _interp_R_wd(tl)
                        R_dc = (
                            np.asarray(session_R_dc, dtype=np.float64)
                            if session_R_dc is not None and np.asarray(session_R_dc).shape == (3, 3)
                            else np.eye(3, dtype=np.float64)
                        )
                        R_wc_rgb = R_wd_rgb @ R_dc
                        R_wc_lidar = R_wd_lidar @ R_dc
                        return (R_wc_rgb.T @ R_wc_lidar).astype(np.float64)

        if imu_df is not None and not imu_df.empty and all(c in imu_df.columns for c in ("gx", "gy", "gz")):
            imu_ts_col = next(
                (c for c in ("device_timestamp_ns", "timestamp_nanos", "timestamp") if c in imu_df.columns),
                None,
            )
            if imu_ts_col is not None:
                its = pd.to_numeric(imu_df[imu_ts_col], errors="coerce").to_numpy(dtype=np.float64)
                gyr = imu_df[["gx", "gy", "gz"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
                ok = np.isfinite(its) & np.all(np.isfinite(gyr), axis=1)
                if int(np.count_nonzero(ok)) >= 1:
                    its_v = its[ok]
                    dt_imu = float(np.median(np.abs(np.diff(its_v)))) if len(its_v) >= 2 else 0.0
                    if dt_imu < 1e4 and float(np.median(np.abs(its_v))) < 1e11 and max(abs(tl), abs(tr)) >= 1e8:
                        its_v = its_v * 1e9
                    gyr_v = gyr[ok]
                    t_min, t_max = min(tl, tr), max(tl, tr)
                    in_win = (its_v >= t_min - 10_000_000) & (its_v <= t_max + 10_000_000)
                    w_body = np.mean(gyr_v[in_win], axis=0) if np.any(in_win) else gyr_v[int(np.argmin(np.abs(its_v - 0.5 * (tl + tr))))]
                    R_b2c = (
                        np.asarray(R_bc, dtype=np.float64).T
                        if R_bc is not None and np.asarray(R_bc).shape == (3, 3)
                        else np.eye(3, dtype=np.float64)
                    )
                    w_cam = R_b2c @ w_body
                    dt_s = (tl - tr) * 1e-9
                    return Rotation.from_rotvec(w_cam * dt_s).as_matrix().astype(np.float64)

        return np.eye(3, dtype=np.float64)
    except Exception:
        return np.eye(3, dtype=np.float64)


def project_lidar_frame_to_sparse_tof(
    distances_m: np.ndarray,
    rays: np.ndarray,
    T_lidar_camera: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    flip_h: bool = False,
    flip_v: bool = False,
    rot_deg: int | float | str = 0,
    R_deskew: np.ndarray | None = None,
    omega_cam: np.ndarray | None = None,
    dt_s: float | None = None,
) -> np.ndarray:
    try:
        h = int(height)
        w = int(width)
        if h <= 0 or w <= 0:
            return np.zeros((0, 0), dtype=np.float32)
    except (ValueError, TypeError):
        return np.zeros((0, 0), dtype=np.float32)

    try:
        d = np.asarray(distances_m, dtype=np.float64).reshape(-1)
        r = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(T_lidar_camera, dtype=np.float64)
        Kin = np.asarray(K, dtype=np.float64)
        if T.shape != (4, 4) or Kin.shape != (3, 3) or r.shape[1] != 3:
            return np.zeros((h, w), dtype=np.float32)
    except (ValueError, TypeError):
        return np.zeros((h, w), dtype=np.float32)

    n = min(d.shape[0], r.shape[0], 64)
    if n == 0:
        return np.zeros((h, w), dtype=np.float32)

    if d.shape[0] >= 64:
        d = remap_lidar_zones_to_camera_frame(
            d, flip_h=flip_h, flip_v=flip_v, rot_deg=rot_deg
        )

    R = T[:3, :3]
    t = T[:3, 3]
    fx, fy = float(Kin[0, 0]), float(Kin[1, 1])
    cx, cy = float(Kin[0, 2]), float(Kin[1, 2])
    valid = np.isfinite(d[:n]) & (d[:n] > 0.0)
    if not np.any(valid):
        return np.zeros((h, w), dtype=np.float32)
    pts = (d[:n, None] * r[:n]) @ R.T + t
    R_corr = _resolve_deskew_rotation(R_deskew=R_deskew, omega_cam=omega_cam, dt_s=dt_s)
    if R_corr is not None:
        pts = pts @ R_corr.T
    z = pts[:, 2]
    keep = valid & (z > TOF_Z_MIN_M) & (z < TOF_Z_MAX_M)
    if not np.any(keep):
        return np.zeros((h, w), dtype=np.float32)
    u = fx * pts[keep, 0] / z[keep] + cx
    v = fy * pts[keep, 1] / z[keep] + cy
    u_px = np.rint(u).astype(int)
    v_px = np.rint(v).astype(int)
    z_keep = z[keep]
    inb = (u_px >= 0) & (u_px < w) & (v_px >= 0) & (v_px < h)
    out = np.zeros((h, w), dtype=np.float32)
    for ui, vi, zi in zip(u_px[inb], v_px[inb], z_keep[inb]):
        prev = out[vi, ui]
        if prev == 0.0 or zi < prev:
            out[vi, ui] = np.float32(zi)
    return out


def project_lidar_frame_to_sparse_tof_with_status(
    distances_m: np.ndarray,
    statuses: np.ndarray,
    rays: np.ndarray,
    T_lidar_camera: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    flip_h: bool = False,
    flip_v: bool = False,
    rot_deg: int | float | str = 0,
    R_deskew: np.ndarray | None = None,
    omega_cam: np.ndarray | None = None,
    dt_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 64-zone VL53L8CX distances and hardware status codes (5 vs 9) to image plane."""
    try:
        h = int(height)
        w = int(width)
        if h <= 0 or w <= 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int32)
    except (ValueError, TypeError):
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int32)

    try:
        d = np.asarray(distances_m, dtype=np.float64).reshape(-1)
        s = np.asarray(statuses, dtype=np.float64).reshape(-1)
        r = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(T_lidar_camera, dtype=np.float64)
        Kin = np.asarray(K, dtype=np.float64)
        if T.shape != (4, 4) or Kin.shape != (3, 3) or r.shape[1] != 3:
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.int32)
    except (ValueError, TypeError):
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.int32)

    n = min(d.shape[0], s.shape[0], r.shape[0], 64)
    if n == 0:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.int32)

    if d.shape[0] >= 64:
        d = remap_lidar_zones_to_camera_frame(d, flip_h=flip_h, flip_v=flip_v, rot_deg=rot_deg)
    if s.shape[0] >= 64:
        s = remap_lidar_zones_to_camera_frame(s, flip_h=flip_h, flip_v=flip_v, rot_deg=rot_deg)

    R = T[:3, :3]
    t = T[:3, 3]
    fx, fy = float(Kin[0, 0]), float(Kin[1, 1])
    cx, cy = float(Kin[0, 2]), float(Kin[1, 2])
    s_int = np.where(np.isfinite(s[:n]), np.rint(s[:n]).astype(np.int32), 0)
    valid = np.isfinite(d[:n]) & (d[:n] > 0.0) & np.isin(s_int, VL53L8CX_VALID_STATUSES)
    if not np.any(valid):
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.int32)
    pts = (d[:n, None] * r[:n]) @ R.T + t
    R_corr = _resolve_deskew_rotation(R_deskew=R_deskew, omega_cam=omega_cam, dt_s=dt_s)
    if R_corr is not None:
        pts = pts @ R_corr.T
    z = pts[:, 2]
    keep = valid & (z > TOF_Z_MIN_M) & (z < TOF_Z_MAX_M)
    if not np.any(keep):
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.int32)
    u = fx * pts[keep, 0] / z[keep] + cx
    v = fy * pts[keep, 1] / z[keep] + cy
    u_px = np.rint(u).astype(int)
    v_px = np.rint(v).astype(int)
    z_keep = z[keep]
    s_keep = s_int[keep]
    inb = (u_px >= 0) & (u_px < w) & (v_px >= 0) & (v_px < h)
    out = np.zeros((h, w), dtype=np.float32)
    out_status = np.zeros((h, w), dtype=np.int32)
    for ui, vi, zi, si in zip(u_px[inb], v_px[inb], z_keep[inb], s_keep[inb]):
        prev = out[vi, ui]
        if prev == 0.0 or zi < prev:
            out[vi, ui] = np.float32(zi)
            out_status[vi, ui] = int(si)

    # Project anomalous / dropout zones (status not in {5, 9} or masked distance) into out_status
    # using reference depth along their ray directions without modifying out (sparse_tof).
    anom_mask = ~keep
    if np.any(anom_mask):
        d_ref = float(np.median(d[:n][keep])) if np.any(keep) else 1.5
        d_anom = np.where(
            np.isfinite(d[:n]) & (d[:n] > TOF_Z_MIN_M) & (d[:n] < TOF_Z_MAX_M),
            d[:n],
            d_ref,
        )
        pts_anom = (d_anom[anom_mask, None] * r[:n][anom_mask]) @ R.T + t
        if R_corr is not None:
            pts_anom = pts_anom @ R_corr.T
        z_anom = pts_anom[:, 2]
        ok_z = z_anom > TOF_Z_MIN_M
        if np.any(ok_z):
            u_a = np.rint(fx * pts_anom[ok_z, 0] / z_anom[ok_z] + cx).astype(int)
            v_a = np.rint(fy * pts_anom[ok_z, 1] / z_anom[ok_z] + cy).astype(int)
            s_a = s_int[:n][anom_mask][ok_z]
            inb_a = (u_a >= 0) & (u_a < w) & (v_a >= 0) & (v_a < h)
            for ui, vi, si in zip(u_a[inb_a], v_a[inb_a], s_a[inb_a]):
                if out[vi, ui] == 0.0 and out_status[vi, ui] == 0:
                    out_status[vi, ui] = int(si) if int(si) != 0 else -1
    return out, out_status


LIDAR_RGB_MATCH_NS = 50_000_000


def rematch_lidar_rgb_frames(
    lidar_df: pd.DataFrame | None,
    odo_df: pd.DataFrame | None = None,
    session_dir: str | None = None,
    match_ns: int = LIDAR_RGB_MATCH_NS,
) -> pd.DataFrame | None:
    """Re-match LiDAR rows to RGB frames (frame >= 0) using shifted device_timestamp_ns."""
    if lidar_df is None or not isinstance(lidar_df, pd.DataFrame) or lidar_df.empty:
        return lidar_df
    if "device_timestamp_ns" not in lidar_df.columns:
        return lidar_df

    rgb_frames_list: list[int] = []
    rgb_ts_list: list[float] = []

    if odo_df is not None and isinstance(odo_df, pd.DataFrame) and not odo_df.empty and "frame" in odo_df.columns:
        ts_col = "timestamp" if "timestamp" in odo_df.columns else (
            "device_timestamp_ns" if "device_timestamp_ns" in odo_df.columns else None
        )
        if ts_col is not None:
            fr_arr = pd.to_numeric(odo_df["frame"], errors="coerce").to_numpy(dtype=np.float64)
            ts_arr = pd.to_numeric(odo_df[ts_col], errors="coerce").to_numpy(dtype=np.float64)
            valid_mask = np.isfinite(fr_arr) & (fr_arr >= 0.0) & np.isfinite(ts_arr) & (ts_arr > 0.0)
            seen_frames: set[int] = set()
            for f_val, t_val in zip(fr_arr[valid_mask], ts_arr[valid_mask]):
                fi = int(round(float(f_val)))
                if fi not in seen_frames:
                    seen_frames.add(fi)
                    rgb_frames_list.append(fi)
                    rgb_ts_list.append(float(t_val))

    if not rgb_ts_list and session_dir and os.path.isdir(session_dir):
        for cand_name in ("processed_odometry.csv", "odometry.csv"):
            cand_path = os.path.join(session_dir, cand_name)
            if os.path.isfile(cand_path):
                try:
                    cdf = pd.read_csv(cand_path)
                    if "frame" in cdf.columns:
                        ts_col = "timestamp" if "timestamp" in cdf.columns else (
                            "device_timestamp_ns" if "device_timestamp_ns" in cdf.columns else None
                        )
                        if ts_col is not None:
                            fr_arr = pd.to_numeric(cdf["frame"], errors="coerce").to_numpy(dtype=np.float64)
                            ts_arr = pd.to_numeric(cdf[ts_col], errors="coerce").to_numpy(dtype=np.float64)
                            valid_mask = np.isfinite(fr_arr) & (fr_arr >= 0.0) & np.isfinite(ts_arr) & (ts_arr > 0.0)
                            seen_frames = set()
                            for f_val, t_val in zip(fr_arr[valid_mask], ts_arr[valid_mask]):
                                fi = int(round(float(f_val)))
                                if fi not in seen_frames:
                                    seen_frames.add(fi)
                                    rgb_frames_list.append(fi)
                                    rgb_ts_list.append(float(t_val))
                    if rgb_ts_list:
                        break
                except Exception:
                    pass

    if not rgb_ts_list:
        return lidar_df

    rgb_ts_arr = np.asarray(rgb_ts_list, dtype=np.float64)
    rgb_fr_arr = np.asarray(rgb_frames_list, dtype=np.int64)

    if session_dir and os.path.isdir(session_dir):
        rgb_dir = os.path.join(session_dir, "rgb")
        if os.path.isdir(rgb_dir):
            try:
                disk_frames: set[int] = set()
                for fn in os.listdir(rgb_dir):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        stem = os.path.splitext(fn)[0].replace("frame_", "")
                        if stem.isdigit():
                            val = int(stem)
                            if val < 100_000_000_000:
                                disk_frames.add(val)
                if disk_frames:
                    keep_disk = np.isin(rgb_fr_arr, list(disk_frames))
                    if np.any(keep_disk):
                        rgb_ts_arr = rgb_ts_arr[keep_disk]
                        rgb_fr_arr = rgb_fr_arr[keep_disk]
            except Exception:
                pass

    out = lidar_df.copy()
    lts = pd.to_numeric(out["device_timestamp_ns"], errors="coerce").to_numpy(dtype=np.float64)
    finite_lts = lts[np.isfinite(lts) & (lts > 0.0)]
    if len(finite_lts) > 0 and len(rgb_ts_arr) > 0:
        if float(np.median(rgb_ts_arr)) < 1e11 and float(np.median(finite_lts)) >= 1e11:
            rgb_ts_arr = rgb_ts_arr * 1e9

    order = np.argsort(rgb_ts_arr)
    rgb_ts_arr = rgb_ts_arr[order]
    rgb_fr_arr = rgb_fr_arr[order]

    new_mf = np.full(len(out), -1, dtype=np.int64)
    new_delta = np.zeros(len(out), dtype=np.int64)

    for i, t_val in enumerate(lts):
        if not np.isfinite(t_val) or t_val <= 0.0:
            continue
        j = int(np.searchsorted(rgb_ts_arr, t_val))
        best_ci = None
        best_dt = None
        for ci in (j - 1, j):
            if 0 <= ci < len(rgb_ts_arr):
                dt = abs(float(rgb_ts_arr[ci]) - float(t_val))
                if dt <= float(match_ns) and (best_dt is None or dt < best_dt):
                    best_dt = dt
                    best_ci = ci
        if best_ci is not None and best_dt is not None:
            new_mf[i] = int(rgb_fr_arr[best_ci])
            new_delta[i] = int(round(best_dt))

    out["matched_frame"] = new_mf
    out["match_delta_nanos"] = new_delta
    return out


def should_skip_tof_keyframe(has_lidar: bool, sparse_tof: np.ndarray) -> bool:
    if not has_lidar:
        return False
    return not bool(np.any(np.asarray(sparse_tof) > 0.0))


def lookup_lidar_row(
    lidar_df: pd.DataFrame | None,
    frame: int | None,
    timestamp_ns: int | None,
    frame_index: dict[int, int] | None = None,
    match_ns: int = LIDAR_RGB_MATCH_NS,
) -> pd.Series | None:
    if lidar_df is None or not isinstance(lidar_df, pd.DataFrame) or lidar_df.empty:
        return None
    has_finite_ts = False
    if timestamp_ns is not None and "device_timestamp_ns" in lidar_df.columns:
        ts = pd.to_numeric(lidar_df["device_timestamp_ns"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(ts) & (ts >= 0.0)
        if np.any(finite):
            has_finite_ts = bool(np.any(ts[finite] > 0.0))
            order = np.argsort(ts[finite])
            sorted_ts = ts[finite][order]
            sorted_pos = np.flatnonzero(finite)[order]
            j = int(np.searchsorted(sorted_ts, float(timestamp_ns)))
            candidates = []
            if j < len(sorted_ts):
                candidates.append(j)
            if j > 0:
                candidates.append(j - 1)
            best = None
            best_dt = None
            for ci in candidates:
                dt = abs(float(sorted_ts[ci]) - float(timestamp_ns))
                if dt <= float(match_ns) and (best_dt is None or dt < best_dt):
                    best_dt = dt
                    best = int(sorted_pos[ci])
            if best is not None:
                return lidar_df.iloc[best]
            if frame is None:
                return None

    if has_finite_ts and timestamp_ns is not None:
        return None

    if frame is not None and int(frame) >= 0 and frame_index is not None:
        idx = frame_index.get(int(frame))
        if idx is not None:
            return lidar_df.iloc[int(idx)]
    if frame is not None and int(frame) >= 0 and "matched_frame" in lidar_df.columns:
        mf = pd.to_numeric(lidar_df["matched_frame"], errors="coerce")
        ok = np.isfinite(mf.to_numpy(dtype=np.float64)) & (mf.to_numpy(dtype=np.float64) >= 0.0)
        keyed = mf[ok].round().astype(int)
        hits = keyed[keyed == int(frame)].index
        if len(hits):
            sub = lidar_df.loc[hits]
            if "match_delta_nanos" in sub.columns:
                delta = pd.to_numeric(sub["match_delta_nanos"], errors="coerce").abs()
                return sub.loc[delta.idxmin()]
            return sub.iloc[0]
    return None


def load_processed_lidar_for_tof(session_dir: str) -> dict:
    empty = {
        "df": None,
        "T_lidar_camera": None,
        "rays": None,
        "frame_index": None,
        "flip_h": False,
        "flip_v": False,
        "camera_sensor_to_display_rotation_deg": 90,
    }
    csv_path = os.path.join(session_dir, "processed_lidar.csv")
    ext_path = os.path.join(session_dir, "processed_extrinsics.json")
    ray_path = os.path.join(session_dir, "processed_lidar_intrinsics.json")
    manifest_path = os.path.join(session_dir, "manifest.json")
    try:
        if not (os.path.isfile(csv_path) and os.path.isfile(ext_path) and os.path.isfile(ray_path)):
            return empty
        df = pd.read_csv(csv_path)
        if df.empty:
            return empty
        with open(ext_path, encoding="utf-8") as f:
            ext = json.load(f)
        T = np.asarray(ext.get("T_lidar_camera"), dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return empty
        with open(ray_path, encoding="utf-8") as f:
            intra = json.load(f)
        rays = np.asarray(intra.get("rays"), dtype=np.float64)
        if rays.shape != (64, 3) or not np.all(np.isfinite(rays)):
            return empty
        frame_index: dict[int, int] = {}
        if "matched_frame" in df.columns:
            mf = pd.to_numeric(df["matched_frame"], errors="coerce")
            delta = (
                pd.to_numeric(df["match_delta_nanos"], errors="coerce").abs()
                if "match_delta_nanos" in df.columns
                else pd.Series(np.zeros(len(df)))
            )
            for iloc_i in range(len(df)):
                val = mf.iloc[iloc_i]
                if not np.isfinite(val) or float(val) < 0.0:
                    continue
                key = int(round(float(val)))
                if key not in frame_index:
                    frame_index[key] = iloc_i
                elif float(delta.iloc[iloc_i]) < float(delta.iloc[frame_index[key]]):
                    frame_index[key] = iloc_i

        raw_h = None
        raw_v = None
        raw_rot = None
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    manifest_data = json.load(mf)
                raw_h, raw_v, raw_rot = _extract_raw_lidar_display_metadata(manifest_data)
            except Exception:
                pass

        if raw_h is not None:
            flip_h = raw_h
        else:
            fh = ext.get("lidar_heatmap_display_flip_h", ext.get("flip_h"))
            flip_h = bool(fh) if fh is not None else False

        if raw_v is not None:
            flip_v = raw_v
        else:
            fv = ext.get("lidar_heatmap_display_flip_v", ext.get("flip_v"))
            flip_v = bool(fv) if fv is not None else False

        if raw_rot is not None:
            rot_deg = raw_rot
        else:
            r_deg = ext.get("camera_sensor_to_display_rotation_deg", ext.get("rot_deg"))
            try:
                rot_deg = int(round(float(r_deg))) if r_deg is not None else 90
            except (ValueError, TypeError):
                rot_deg = 90

        return {
            "df": df,
            "T_lidar_camera": T,
            "rays": rays,
            "frame_index": frame_index,
            "flip_h": bool(flip_h),
            "flip_v": bool(flip_v),
            "camera_sensor_to_display_rotation_deg": int(rot_deg),
        }
    except Exception:
        return empty


EM_DWELL_FRAMES = 10
EM_DWELL_MAX_SPAN_M = 0.05
EM_DWELL_MAX_DURATION_S = 2.0
EM_DWELL_MIN_POINTS = 60
EM_TOF_STATUS_OK = frozenset({5, 9})
EM_TOF_RANGE_MIN_M = 0.20
EM_TOF_RANGE_MAX_M = 3.50
EM_MIN_PLANES = 3
EM_MAX_PLANES = 3
EM_MIN_INLIERS_PER_PLANE = 30
EM_MIN_TOF_INLIERS_PER_PLANE = 30
EM_MAX_NORMAL_DOT = 0.38
EM_MAX_COND = 100.0
EM_MAX_ITERS = 5
EM_HUBER_DELTA_M = 0.03
EM_GRAZING_COS_MIN = 0.15
EM_SIGMA0_M = 0.012
EM_SIGMA_SLOPE = 0.008
EM_TRUST_T_M = 0.05
EM_TRUST_R_DEG = 5.0
EM_RANSAC_DIST_M = 0.025
EM_DEPTH_SAMPLE_STRIDE = 4
EM_MIN_TOF_HITS = 40
EM_ASSIGN_MAX_RES_M = 0.15


def find_stationary_dwell(
    pose_df: pd.DataFrame,
    lidar_df: pd.DataFrame | None,
    frame_index: dict[int, int] | None = None,
    imu_df: pd.DataFrame | None = None,
) -> dict | None:
    if pose_df is None or pose_df.empty or lidar_df is None or getattr(lidar_df, "empty", True):
        return None
    if "device_timestamp_ns" not in pose_df.columns:
        return None
    if not all(c in pose_df.columns for c in ("x", "y", "z")):
        return None
    ts = pd.to_numeric(pose_df["device_timestamp_ns"], errors="coerce").to_numpy(dtype=np.float64)
    xyz = pose_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    n_rows = len(pose_df)
    if "is_frozen_interpolated" in pose_df.columns:
        frozen_mask = (
            pd.to_numeric(pose_df["is_frozen_interpolated"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int64)
            != 0
        )
    else:
        frozen_mask = np.zeros(n_rows, dtype=bool)
        if n_rows >= 3:
            step_dp = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            has_q = all(c in pose_df.columns for c in ("qx", "qy", "qz", "qw"))
            q_arr = (
                pose_df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
                if has_q
                else None
            )
            ts_imu = None
            w_imu = None
            if imu_df is not None and not getattr(imu_df, "empty", True):
                t_c = (
                    "device_timestamp_ns"
                    if "device_timestamp_ns" in imu_df.columns
                    else ("timestamp_nanos" if "timestamp_nanos" in imu_df.columns else "timestamp")
                )
                gx_c = "gx" if "gx" in imu_df.columns else ("gyro_x" if "gyro_x" in imu_df.columns else None)
                gy_c = "gy" if "gy" in imu_df.columns else ("gyro_y" if "gyro_y" in imu_df.columns else None)
                gz_c = "gz" if "gz" in imu_df.columns else ("gyro_z" if "gyro_z" in imu_df.columns else None)
                if t_c in imu_df.columns and gx_c and gy_c and gz_c:
                    r_t = pd.to_numeric(imu_df[t_c], errors="coerce").to_numpy(dtype=np.float64)
                    ts_imu = r_t if (len(r_t) > 0 and r_t[0] > 1e12) else (r_t * 1e9)
                    w_imu = np.sqrt(
                        pd.to_numeric(imu_df[gx_c], errors="coerce").to_numpy(dtype=np.float64) ** 2
                        + pd.to_numeric(imu_df[gy_c], errors="coerce").to_numpy(dtype=np.float64) ** 2
                        + pd.to_numeric(imu_df[gz_c], errors="coerce").to_numpy(dtype=np.float64) ** 2
                    )
            idx_s = 1
            while idx_s < n_rows:
                if step_dp[idx_s - 1] >= 1e-7:
                    idx_s += 1
                    continue
                f_start = idx_s - 1
                f_end = idx_s
                while f_end + 1 < n_rows and step_dp[f_end] < 1e-7:
                    f_end += 1
                has_jump_after = (f_end + 1 < n_rows) and (step_dp[f_end] > 0.05)
                if not has_jump_after and q_arr is not None and (f_end + 1 < n_rows):
                    q0 = q_arr[f_end] / max(1e-12, float(np.linalg.norm(q_arr[f_end])))
                    q1 = q_arr[f_end + 1] / max(1e-12, float(np.linalg.norm(q_arr[f_end + 1])))
                    drot = float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(q0, q1))), 0.0, 1.0))))
                    if drot > 5.0:
                        has_jump_after = True
                has_imu_rot = False
                if ts_imu is not None and w_imu is not None:
                    m_i = (ts_imu >= ts[f_start]) & (ts_imu <= ts[min(n_rows - 1, f_end + 1)])
                    if np.any(m_i) and float(np.max(w_imu[m_i])) > 0.08:
                        has_imu_rot = True
                if has_jump_after or has_imu_rot:
                    frozen_mask[f_start : f_end + 1] = True
                idx_s = f_end + 2

    matched_pose = []
    lidar_iloc = []
    for i in range(n_rows):
        if not np.isfinite(ts[i]):
            continue
        frame = None
        if "frame" in pose_df.columns:
            try:
                frame = int(pose_df.iloc[i]["frame"])
            except (TypeError, ValueError):
                frame = None
        row = lookup_lidar_row(
            lidar_df, frame, int(ts[i]), frame_index=frame_index
        )
        if row is None:
            continue
        loc = lidar_df.index.get_loc(row.name)
        if not isinstance(loc, (int, np.integer)):
            loc = int(np.min(np.atleast_1d(loc)))
        matched_pose.append(i)
        lidar_iloc.append(int(loc))
    n = len(matched_pose)
    if n < EM_DWELL_FRAMES:
        return None
    best = None
    for start in range(0, n - EM_DWELL_FRAMES + 1):
        sl = slice(start, start + EM_DWELL_FRAMES)
        pidx = matched_pose[sl]
        if np.any(frozen_mask[pidx]):
            continue
        t0 = float(ts[pidx[0]])
        t1 = float(ts[pidx[-1]])
        if (t1 - t0) / 1e9 > EM_DWELL_MAX_DURATION_S + 1e-9:
            continue
        pts = xyz[pidx]
        span = float(np.max(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)))
        if span >= EM_DWELL_MAX_SPAN_M:
            continue
        n_valid = 0
        for li in lidar_iloc[sl]:
            row = lidar_df.iloc[int(li)]
            for z in range(64):
                col = f"status_{z}"
                if col in row.index and int(row[col]) in EM_TOF_STATUS_OK:
                    n_valid += 1
        if n_valid < EM_DWELL_MIN_POINTS:
            continue
        cand = {
            "t0_ns": int(t0),
            "t1_ns": int(t1),
            "span_m": span,
            "n_lidar_rows": int(EM_DWELL_FRAMES),
            "lidar_iloc": [int(v) for v in lidar_iloc[sl]],
            "pose_iloc": [int(v) for v in pidx],
            "median_ns": int(np.median(ts[pidx])),
            "n_valid_zones": int(n_valid),
        }
        if best is None:
            best = cand
            continue
        if cand["n_valid_zones"] > best["n_valid_zones"]:
            best = cand
        elif cand["n_valid_zones"] == best["n_valid_zones"] and cand["span_m"] < best["span_m"]:
            best = cand
        elif (
            cand["n_valid_zones"] == best["n_valid_zones"]
            and abs(cand["span_m"] - best["span_m"]) < 1e-9
            and cand["t0_ns"] < best["t0_ns"]
        ):
            best = cand
    return best


def gather_dwell_tof_points(
    lidar_df: pd.DataFrame,
    lidar_iloc: list[int],
    rays: np.ndarray,
    *,
    flip_h: bool = False,
    flip_v: bool = False,
    rot_deg: int | float = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    xs, ds, rs = [], [], []
    for li in lidar_iloc:
        row = lidar_df.iloc[int(li)]
        dist = np.array([float(row.get(f"distance_{i}", np.nan)) for i in range(64)], dtype=np.float64)
        stat = np.array([float(row.get(f"status_{i}", np.nan)) for i in range(64)], dtype=np.float64)
        dist = remap_lidar_zones_to_camera_frame(dist, flip_h=flip_h, flip_v=flip_v, rot_deg=rot_deg)
        stat = remap_lidar_zones_to_camera_frame(stat, flip_h=flip_h, flip_v=flip_v, rot_deg=rot_deg)
        n = min(64, rays.shape[0])
        for j in range(n):
            st = stat[j]
            dj = dist[j]
            if not np.isfinite(st) or int(st) not in EM_TOF_STATUS_OK:
                continue
            if not np.isfinite(dj) or dj < EM_TOF_RANGE_MIN_M or dj > EM_TOF_RANGE_MAX_M:
                continue
            xs.append(dj * rays[j])
            ds.append(dj)
            rs.append(rays[j])
    if not xs:
        return np.zeros((0, 3)), np.zeros((0,)), np.zeros((0, 3))
    return np.vstack(xs), np.asarray(ds, dtype=np.float64), np.vstack(rs)


def _solve_weighted_extrinsics_step(
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
    w: np.ndarray | None = None,
    *,
    H_tt: np.ndarray | None = None,
    g_t: np.ndarray | None = None,
    min_sv: float = 1e-3,
    rcond: float = 0.05,
) -> np.ndarray:
    """Solve translation step dt from normal equations H_tt = A^T W A with observability-aware truncated SVD.

    Singular components with S_i < max(min_sv, rcond * S_max) are zeroed out so unobservable directions
    parallel to planes do not drift.
    """
    if H_tt is None:
        if A is None or b is None:
            return np.zeros(3, dtype=np.float64)
        A_arr = np.asarray(A, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64).reshape(-1)
        if len(A_arr) == 0 or len(b_arr) == 0:
            return np.zeros(3, dtype=np.float64)
        if w is None:
            w_arr = np.ones(len(b_arr), dtype=np.float64)
        else:
            w_arr = np.asarray(w, dtype=np.float64).reshape(-1)
        H_tt = A_arr.T @ (w_arr[:, None] * A_arr)
        g_t = A_arr.T @ (w_arr * b_arr)
    else:
        H_tt = np.asarray(H_tt, dtype=np.float64)
        if g_t is None:
            return np.zeros(H_tt.shape[1], dtype=np.float64)
        g_t = np.asarray(g_t, dtype=np.float64).reshape(-1)

    U, S, Vt = np.linalg.svd(H_tt)
    s_max = float(np.max(S)) if len(S) > 0 else 0.0
    cutoff = max(min_sv, rcond * s_max)

    S_inv = np.zeros_like(S)
    valid = S >= cutoff
    S_inv[valid] = 1.0 / S[valid]

    dt = Vt.T @ (S_inv * (U.T @ g_t))
    return dt


def o3d_model_to_nd(model: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(model, dtype=np.float64).reshape(-1)
    if len(arr) < 4:
        return np.zeros(3, dtype=np.float64), 0.0
    a, b, c, d_o3d = [float(v) for v in arr[:4]]
    n = np.array([a, b, c], dtype=np.float64)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-12:
        return n, 0.0
    n = n / nrm
    d = -float(d_o3d) / nrm
    if d < -1e-6:
        n = -n
        d = -d
    d = max(0.0, float(d))
    return n, float(d)


def planes_are_observable(planes: list[dict]) -> bool:
    if len(planes) < EM_MIN_PLANES:
        return False
    use = [p for p in planes if int(p.get("inlier_count", 0)) >= EM_MIN_INLIERS_PER_PLANE]
    if len(use) < EM_MIN_PLANES:
        return False
    use = use[:EM_MAX_PLANES]
    ortho = 0
    for i in range(len(use)):
        for j in range(i + 1, len(use)):
            dot = abs(float(np.dot(use[i]["n"], use[j]["n"])))
            if dot <= EM_MAX_NORMAL_DOT:
                ortho += 1
    if ortho < 2:
        return False
    N = np.stack([p["n"] for p in use])
    try:
        cond = float(np.linalg.cond(N))
        if not np.isfinite(cond) or cond >= EM_MAX_COND:
            return False
    except Exception:
        return False
    return True


def _deterministic_segment_plane(
    pts: np.ndarray,
    distance_threshold: float,
    num_iterations: int = 400,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 3-point RANSAC + SVD plane refinement (independent of OpenMP threads)."""
    pts = np.asarray(pts, dtype=np.float64)
    n_pts = len(pts)
    if n_pts < 3:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64), np.zeros(0, dtype=np.int64)
    if rng is None:
        rng = np.random.default_rng(42)

    idx0 = rng.integers(0, n_pts, size=num_iterations)
    idx1 = (idx0 + rng.integers(1, n_pts, size=num_iterations)) % n_pts
    idx2 = (idx0 + rng.integers(1, n_pts - 1, size=num_iterations)) % n_pts
    same12 = idx2 == idx1
    idx2[same12] = (idx2[same12] + 1) % n_pts

    p0 = pts[idx0]
    v1 = pts[idx1] - p0
    v2 = pts[idx2] - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-9
    if not np.any(valid):
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64), np.zeros(0, dtype=np.int64)

    normals = normals[valid] / norms[valid, None]
    p0_valid = p0[valid]
    d_o3d = -np.sum(normals * p0_valid, axis=1)

    dists = np.abs(normals @ pts.T + d_o3d[:, None])
    inlier_mask = dists <= distance_threshold
    counts = np.sum(inlier_mask, axis=1)
    max_count = int(np.max(counts))
    if max_count < 3:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64), np.zeros(0, dtype=np.int64)

    # Break ties deterministically by lowest mean inlier distance
    inlier_dists_sum = np.sum(np.where(inlier_mask, dists, 0.0), axis=1)
    scores = counts.astype(np.float64) - (inlier_dists_sum / max(distance_threshold * n_pts, 1e-9))
    best_i = int(np.argmax(scores))
    best_inliers = np.flatnonzero(inlier_mask[best_i])

    # SVD least-squares refit on inliers (matching Open3D post-RANSAC refinement)
    inlier_pts = pts[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    n_fit = vt[-1]
    n_norm = float(np.linalg.norm(n_fit))
    if n_norm > 1e-12:
        n_fit = n_fit / n_norm
        d_fit = -float(np.dot(n_fit, centroid))
        refined_inliers = np.flatnonzero(np.abs(pts @ n_fit + d_fit) <= distance_threshold)
        if len(refined_inliers) >= len(best_inliers):
            best_inliers = refined_inliers
            model = np.array([n_fit[0], n_fit[1], n_fit[2], d_fit], dtype=np.float64)
        else:
            n_best = normals[best_i]
            model = np.array([n_best[0], n_best[1], n_best[2], float(d_o3d[best_i])], dtype=np.float64)
    else:
        n_best = normals[best_i]
        model = np.array([n_best[0], n_best[1], n_best[2], float(d_o3d[best_i])], dtype=np.float64)

    return model, best_inliers


def extract_camera_planes_from_dense_depth(
    dense: np.ndarray, K: np.ndarray
) -> list[dict]:
    z = np.asarray(dense, dtype=np.float64)
    if z.ndim != 2 or z.size == 0:
        return []
    h, w = z.shape
    Kin = np.asarray(K, dtype=np.float64)
    if Kin.shape != (3, 3):
        return []
    fx, fy = float(Kin[0, 0]), float(Kin[1, 1])
    cx, cy = float(Kin[0, 2]), float(Kin[1, 2])
    if fx <= 0 or fy <= 0:
        return []
    stride = EM_DEPTH_SAMPLE_STRIDE
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    zz = z[vs, us]
    keep = np.isfinite(zz) & (zz > 0.2) & (zz < 5.0)
    if not np.any(keep):
        return []
    us, vs, zz = us[keep].astype(float), vs[keep].astype(float), zz[keep]
    pts = np.column_stack([
        (us - cx) * zz / fx,
        (vs - cy) * zz / fy,
        zz,
    ])
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(0.02)
    work_pts = np.asarray(pcd.points, dtype=np.float64)
    if len(work_pts) > 1:
        order = np.lexsort((work_pts[:, 2], work_pts[:, 1], work_pts[:, 0]))
        work_pts = work_pts[order]
    rng = np.random.default_rng(42)
    planes = []
    for _ in range(EM_MAX_PLANES):
        if len(work_pts) < 50:
            break
        model, inliers = _deterministic_segment_plane(
            work_pts, distance_threshold=EM_RANSAC_DIST_M, num_iterations=400, rng=rng
        )
        if len(inliers) < EM_MIN_INLIERS_PER_PLANE:
            break
        n, d = o3d_model_to_nd(model)
        inlier_pts = work_pts[inliers].copy()
        planes.append(
            {"n": n, "d": d, "inliers": inlier_pts, "inlier_count": int(len(inliers))}
        )
        mask = np.ones(len(work_pts), dtype=bool)
        mask[inliers] = False
        work_pts = work_pts[mask]
    return planes


def _svd_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_o = U @ Vt
    if np.linalg.det(R_o) < 0:
        Vt[-1, :] *= -1
        R_o = U @ Vt
    return R_o


def _geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = _svd_so3(R_a @ R_b.T)
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))


def _assign(X, ray_dirs, distances, planes, R, t):
    p = X @ R.T + t
    r_cam = ray_dirs @ R.T
    ns = np.stack([pl["n"] for pl in planes])
    ds = np.array([pl["d"] for pl in planes])
    res = p @ ns.T - ds
    idx = np.argmin(np.abs(res), axis=1)
    r = res[np.arange(len(X)), idx]
    n_asg = ns[idx]
    cosb = np.sum(n_asg * r_cam, axis=1)
    keep = (np.abs(cosb) >= EM_GRAZING_COS_MIN) & (np.abs(r) <= EM_ASSIGN_MAX_RES_M)
    sigma = (EM_SIGMA0_M + EM_SIGMA_SLOPE * distances) / np.sqrt(np.maximum(cosb ** 2, 0.05))
    w = 1.0 / (sigma ** 2)
    return idx, r, w, keep, n_asg


def em_point_to_plane(X, ray_dirs, distances, planes, T0) -> dict:
    T0 = np.asarray(T0, dtype=np.float64)
    R = _svd_so3(T0[:3, :3].copy())
    t = T0[:3, 3].copy()
    base = {
        "T": T0.copy(),
        "accepted": False,
        "reason": None,
        "residual_rms": None,
        "residual_rms_T0": None,
        "cond_M": None,
        "iterations": 0,
        "inliers_per_plane": [int(p.get("inlier_count", 0)) for p in planes],
        "delta_t_m": 0.0,
        "delta_r_deg": 0.0,
    }
    if not planes_are_observable(planes) or len(X) < EM_MIN_TOF_HITS:
        base["reason"] = "unobservable"
        return base
    idx0, r0, w0, keep0, _ = _assign(X, ray_dirs, distances, planes, R, t)
    if not np.any(keep0):
        base["reason"] = "no_inliers"
        return base
    rms_T0 = float(np.sqrt(np.average(r0[keep0] ** 2, weights=w0[keep0])))
    base["residual_rms_T0"] = rms_T0

    def pack(R, t):
        T = np.eye(4)
        T[:3, :3] = _svd_so3(R)
        T[:3, 3] = t
        return T

    ds = np.array([p["d"] for p in planes], dtype=np.float64)
    last_cond = None
    for it in range(EM_MAX_ITERS):
        idx, r, w, keep, n_asg = _assign(X, ray_dirs, distances, planes, R, t)
        if int(np.count_nonzero(keep)) < EM_MIN_TOF_HITS:
            base["reason"] = "too_few_tof"
            base["T"] = T0.copy()
            base["iterations"] = it
            return base
        Xk, wk, nk = X[keep], w[keep], n_asg[keep]
        RX = Xk @ R.T
        d_asg = ds[idx[keep]]

        def fun(x):
            tau, omega = x[:3], x[3:]
            Rd = Rotation.from_rotvec(omega).as_matrix() @ R
            td = t + tau
            p = Xk @ Rd.T + td
            r = np.sum(nk * p, axis=1) - d_asg
            abs_r = np.abs(r)
            e_r = np.where(
                abs_r <= EM_HUBER_DELTA_M,
                r,
                np.sign(r) * np.sqrt(np.maximum(0.0, 2.0 * EM_HUBER_DELTA_M * abs_r - EM_HUBER_DELTA_M**2)),
            )
            return e_r * np.sqrt(wk)

        def jac(x):
            tau, omega = x[:3], x[3:]
            Rd = Rotation.from_rotvec(omega).as_matrix() @ R
            td = t + tau
            p = Xk @ Rd.T + td
            r = np.sum(nk * p, axis=1) - d_asg
            abs_r = np.abs(r)
            e_r = np.where(
                abs_r <= EM_HUBER_DELTA_M,
                r,
                np.sign(r) * np.sqrt(np.maximum(0.0, 2.0 * EM_HUBER_DELTA_M * abs_r - EM_HUBER_DELTA_M**2)),
            )
            scale = np.where(
                abs_r <= EM_HUBER_DELTA_M,
                1.0,
                EM_HUBER_DELTA_M / np.maximum(np.abs(e_r), 1e-12),
            )
            J = np.zeros((len(Xk), 6))
            J[:, :3] = nk
            J[:, 3:] = np.cross(RX, nk)
            J *= (np.sqrt(wk) * scale)[:, None]
            return J

        J0 = jac(np.zeros(6))
        M = J0.T @ J0
        d = np.sqrt(np.maximum(np.diag(M), 1e-12))
        D = np.diag(1.0 / d)
        M_scaled = D @ M @ D
        w_eigs = np.linalg.eigvalsh(M_scaled)
        lam_max = float(np.max(np.abs(w_eigs)))
        lam_min = float(np.min(np.abs(w_eigs)))
        last_cond = lam_max / max(lam_min, 1e-12)
        if not np.isfinite(last_cond) or last_cond >= EM_MAX_COND:
            base["reason"] = "ill_conditioned"
            base["cond_M"] = last_cond
            base["iterations"] = it
            base["T"] = T0.copy()
            return base
        sol = least_squares(
            fun, np.zeros(6), jac=jac, loss="linear", max_nfev=40
        )
        omega = sol.x[3:]
        Rd = Rotation.from_rotvec(omega).as_matrix() @ R
        # Linear translation solve with truncated SVD damping
        p_rot = Xk @ Rd.T + t
        b_vec = d_asg - np.sum(nk * p_rot, axis=1)
        tau = _solve_weighted_extrinsics_step(nk, b_vec, wk)
        R = Rd
        t = t + tau
        base["iterations"] = it + 1
        if np.linalg.norm(tau) < 1e-3 and np.degrees(np.linalg.norm(omega)) < 0.05:
            break

    T_star = pack(R, t)
    idx, r, w, keep, _ = _assign(X, ray_dirs, distances, planes, T_star[:3, :3], T_star[:3, 3])
    common_keep = keep & keep0
    eval_keep = common_keep if np.count_nonzero(common_keep) >= EM_MIN_TOF_HITS else (keep if np.any(keep) else keep0)
    rms_T0 = float(np.sqrt(np.average(r0[eval_keep] ** 2, weights=w0[eval_keep])))
    rms = float(np.sqrt(np.average(r[eval_keep] ** 2, weights=w[eval_keep])))
    dt = float(np.linalg.norm(T_star[:3, 3] - T0[:3, 3]))
    dr = _geodesic_deg(T_star[:3, :3], T0[:3, :3])
    counts = [int(np.sum((idx == i) & keep)) for i in range(len(planes))]
    base.update({
        "residual_rms": rms,
        "residual_rms_T0": rms_T0,
        "cond_M": last_cond,
        "inliers_per_plane": counts,
        "delta_t_m": dt,
        "delta_r_deg": dr,
    })
    if rms >= rms_T0 or dt > EM_TRUST_T_M or dr > EM_TRUST_R_DEG:
        base["reason"] = "trust_or_residual"
        base["T"] = T0.copy()
        return base
    if min(counts) < EM_MIN_TOF_INLIERS_PER_PLANE or sum(counts) < EM_MIN_TOF_HITS:
        base["reason"] = "inliers"
        base["T"] = T0.copy()
        return base
    base["accepted"] = True
    base["T"] = T_star
    base["reason"] = None
    return base


def _identity_T() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _valid_T(T) -> bool:
    if T is None:
        return False
    try:
        arr = np.asarray(T, dtype=np.float64)
        return arr.shape == (4, 4) and bool(np.all(np.isfinite(arr)))
    except Exception:
        return False


def _json_sanitize(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.floating, float)):
        return float(val) if np.isfinite(val) else None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, dict):
        return {k: _json_sanitize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_sanitize(v) for v in val]
    return val


def _write_em_json(session_dir: str, payload: dict) -> None:
    path = os.path.join(session_dir, "em_calibration.json")
    serial = dict(payload)
    for key in ("T_lidar_camera", "T0"):
        if key in serial and isinstance(serial[key], np.ndarray):
            serial[key] = serial[key].tolist()
    clean_planes = []
    for p in serial.get("planes", []):
        cp = {
            "n": p["n"].tolist() if isinstance(p.get("n"), np.ndarray) else p.get("n"),
            "d": float(p["d"]) if p.get("d") is not None else None,
            "inlier_count": int(p.get("inlier_count", 0)),
        }
        clean_planes.append(cp)
    serial["planes"] = clean_planes
    if "residual_rms" in serial and "residual_rms_Tstar" in serial:
        del serial["residual_rms"]
    serial = _json_sanitize(serial)
    os.makedirs(session_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2)


def extract_planes_from_sparse_tof_points(p: np.ndarray) -> list[dict]:
    work_pts = np.asarray(p, dtype=np.float64)
    if len(work_pts) < EM_MIN_INLIERS_PER_PLANE:
        return []
    rng = np.random.default_rng(42)
    planes = []
    for _ in range(EM_MAX_PLANES):
        if len(work_pts) < EM_MIN_INLIERS_PER_PLANE:
            break
        model, inliers = _deterministic_segment_plane(
            work_pts, distance_threshold=EM_RANSAC_DIST_M, num_iterations=400, rng=rng
        )
        if len(inliers) < EM_MIN_INLIERS_PER_PLANE:
            break
        n, d = o3d_model_to_nd(model)
        inlier_pts = work_pts[inliers].copy()
        planes.append(
            {"n": n, "d": d, "inliers": inlier_pts, "inlier_count": int(len(inliers))}
        )
        mask = np.ones(len(work_pts), dtype=bool)
        mask[inliers] = False
        work_pts = work_pts[mask]
    return planes


def _persist_accepted_extrinsics(
    session_dir: str,
    result: dict,
    flip_h: bool,
    flip_v: bool,
    rot_deg: int | float,
    source_tag: str = "server_em_dwell",
) -> None:
    refined_path = os.path.join(session_dir, "refined_lidar_camera_extrinsics.json")
    refined_data = {
        "config_schema_version": 2,
        "source": source_tag,
        "lidar_to_camera": {
            "rotation_row_major_3x3": result["T_lidar_camera"][:3, :3].flatten().tolist(),
            "translation_meters": result["T_lidar_camera"][:3, 3].tolist(),
        },
        "metrics": {
            "residual_rms_meters": result["residual_rms_Tstar"],
            "inliers_per_plane": result["inliers_per_plane"],
            "iterations": result["iterations"],
            "converged": True,
        },
    }
    os.makedirs(session_dir, exist_ok=True)
    with open(refined_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(refined_data), f, indent=2)

    processed_path = os.path.join(session_dir, "processed_extrinsics.json")
    proc_data = {}
    if os.path.isfile(processed_path):
        try:
            with open(processed_path, "r", encoding="utf-8") as f:
                proc_data = json.load(f)
        except Exception:
            proc_data = {}
    if not isinstance(proc_data, dict):
        proc_data = {}

    proc_data["T_lidar_camera"] = result["T_lidar_camera"].tolist()
    proc_data["extrinsics_calibrated"] = True
    proc_data["extrinsics_source"] = source_tag
    if "lidar_heatmap_display_flip_h" not in proc_data:
        proc_data["lidar_heatmap_display_flip_h"] = bool(flip_h)
    if "lidar_heatmap_display_flip_v" not in proc_data:
        proc_data["lidar_heatmap_display_flip_v"] = bool(flip_v)
    if "camera_sensor_to_display_rotation_deg" not in proc_data:
        try:
            proc_data["camera_sensor_to_display_rotation_deg"] = int(round(float(rot_deg))) % 360
        except (ValueError, TypeError):
            proc_data["camera_sensor_to_display_rotation_deg"] = 90

    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(proc_data), f, indent=2)


def extract_camera_planes_from_trihedral_depth(
    dense: np.ndarray, K: np.ndarray
) -> list[dict]:
    """Extract up to 3 mutually orthogonal (trihedral) camera planes from a dense depth map."""
    raw_planes = extract_camera_planes_from_dense_depth(dense, K)
    if len(raw_planes) <= 1:
        return raw_planes
    selected: list[dict] = [raw_planes[0]]
    for cand in raw_planes[1:]:
        n_c = np.asarray(cand["n"], dtype=np.float64)
        if all(abs(float(np.dot(n_c, np.asarray(p["n"], dtype=np.float64)))) <= 0.65 for p in selected):
            selected.append(cand)
        if len(selected) >= EM_MAX_PLANES:
            break
    if len(selected) < len(raw_planes):
        for cand in raw_planes:
            if cand not in selected and len(selected) < EM_MAX_PLANES:
                selected.append(cand)
    return selected


class MultiKeyframePlaneCalibration:
    """Multi-keyframe Point-to-Plane + Wahba-prior extrinsic calibration for handheld scans."""

    def __init__(
        self,
        *,
        T0: np.ndarray,
        R_phone_hub: np.ndarray | None = None,
        rot_deg: int | float = 90,
        extrinsics_calibrated: bool | None = None,
        flip_h: bool = False,
        flip_v: bool = False,
    ) -> None:
        self.T0 = np.asarray(T0, dtype=np.float64).copy()
        self.R_phone_hub = R_phone_hub
        self.rot_deg = rot_deg
        self.extrinsics_calibrated = extrinsics_calibrated
        self.flip_h = bool(flip_h)
        self.flip_v = bool(flip_v)
        self.T_prior = chain_wahba_to_lidar_camera_extrinsics(
            self.T0,
            R_phone_hub,
            rot_deg=rot_deg,
            extrinsics_calibrated=extrinsics_calibrated,
        )

    @staticmethod
    def _pose_matrix_from_dict(row: dict | pd.Series) -> np.ndarray | None:
        try:
            if not all(k in row for k in ("x", "y", "z", "qx", "qy", "qz", "qw")):
                return None
            xyz = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
            quat = np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])], dtype=np.float64)
            if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(quat))):
                return None
            nrm = float(np.linalg.norm(quat))
            if nrm < 1e-8:
                return None
            quat /= nrm
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat(quat).as_matrix()
            T[:3, 3] = xyz
            return T
        except Exception:
            return None

    def refine(
        self,
        *,
        keyframes: list[dict],
        lidar_df: pd.DataFrame,
        frame_index: dict[int, int] | None,
        rays: np.ndarray,
        K: np.ndarray,
        depth_service: object | None = None,
        pose_df: pd.DataFrame | None = None,
        g_world: np.ndarray | None = None,
        z_max_ghost: float | None = None,
    ) -> dict | None:
        if len(keyframes) < 5 or lidar_df is None or len(lidar_df) < 10:
            return None

        rays_arr = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        step = max(1, len(keyframes) // 12)
        sampled_kfs = keyframes[::step][:12]

        # Build a pose lookup table for multi-frame local window accumulation
        pose_records: list[tuple[int, np.ndarray, int | None]] = []
        source_rows = []
        if pose_df is not None and isinstance(pose_df, pd.DataFrame) and not pose_df.empty:
            for _, prow in pose_df.iterrows():
                source_rows.append(prow)
        else:
            source_rows = list(keyframes)
        for prow in source_rows:
            ts_val = prow.get("device_timestamp_ns")
            if ts_val is None or not np.isfinite(float(ts_val)):
                continue
            T_wc = self._pose_matrix_from_dict(prow)
            if T_wc is None:
                continue
            frm = prow.get("frame")
            frm_int = int(frm) if frm is not None and np.isfinite(float(frm)) else None
            pose_records.append((int(float(ts_val)), T_wc, frm_int))
        pose_records.sort(key=lambda item: item[0])
        pose_ts_arr = np.array([item[0] for item in pose_records], dtype=np.int64) if pose_records else np.zeros(0, dtype=np.int64)

        R_wc_all_mkf = np.array([item[1][:3, :3] for item in pose_records], dtype=np.float64) if pose_records else None
        first_rgb_p = next(
            (kf.get("_rgb_path") or kf.get("rgb_path") for kf in sampled_kfs if (kf.get("_rgb_path") or kf.get("rgb_path"))),
            None,
        )
        w_est, h_est = probe_session_image_size(rgb_path=first_rgb_p, K=K)
        session_R_dc = resolve_sensor_to_vio_rotation(
            self.rot_deg, img_w=w_est, img_h=h_est, R_wc_all=R_wc_all_mkf, g_world=g_world
        )
        if not np.allclose(session_R_dc, np.eye(3)):
            for idx_rec in range(len(pose_records)):
                ts_r, T_r, frm_r = pose_records[idx_rec]
                T_r_new = T_r.copy()
                T_r_new[:3, :3] = T_r[:3, :3] @ session_R_dc
                pose_records[idx_rec] = (ts_r, T_r_new, frm_r)

        all_X = []
        all_n = []
        all_d = []
        all_w = []
        all_planes = []

        R_prior = _svd_so3(self.T_prior[:3, :3].copy())
        t_prior = self.T_prior[:3, 3].copy()

        for kf in sampled_kfs:
            rgb_path = kf.get("_rgb_path") or kf.get("rgb_path")
            rgb_img = cv2.imread(rgb_path) if (rgb_path and cv2 is not None) else None

            row = lookup_lidar_row(
                lidar_df, kf.get("frame"), kf.get("device_timestamp_ns"), frame_index=frame_index
            )
            if row is None:
                continue
            li = lidar_df.index.get_loc(row.name)
            if not isinstance(li, (int, np.integer)):
                li = int(np.min(np.atleast_1d(li)))
            X_k, dist_k, rdir_k = gather_dwell_tof_points(
                lidar_df, [int(li)], rays_arr, flip_h=self.flip_h, flip_v=self.flip_v, rot_deg=self.rot_deg
            )
            if len(X_k) < 12:
                continue

            T_wc_k = self._pose_matrix_from_dict(kf)
            if T_wc_k is not None and not np.allclose(session_R_dc, np.eye(3)):
                T_wc_k = T_wc_k.copy()
                T_wc_k[:3, :3] = T_wc_k[:3, :3] @ session_R_dc
            window_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
                (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), X_k, dist_k, rdir_k)
            ]

            # Gather nearby frames within +/- 800 ms using relative camera pose T_{c_k c_m}
            kf_ts = kf.get("device_timestamp_ns")
            if T_wc_k is not None and kf_ts is not None and len(pose_ts_arr) > 1:
                T_ck_w = np.linalg.inv(T_wc_k)
                dt_ns = np.abs(pose_ts_arr - int(float(kf_ts)))
                near_idx = np.flatnonzero((dt_ns > 20_000_000) & (dt_ns <= 800_000_000))
                if len(near_idx) > 6:
                    sel = np.linspace(0, len(near_idx) - 1, 6, dtype=int)
                    near_idx = near_idx[sel]
                used_li = {int(li)}
                for ni in near_idx:
                    ts_m, T_wc_m, frm_m = pose_records[int(ni)]
                    row_m = lookup_lidar_row(lidar_df, frm_m, ts_m, frame_index=frame_index)
                    if row_m is None:
                        continue
                    li_m = lidar_df.index.get_loc(row_m.name)
                    if not isinstance(li_m, (int, np.integer)):
                        li_m = int(np.min(np.atleast_1d(li_m)))
                    if int(li_m) in used_li:
                        continue
                    used_li.add(int(li_m))
                    X_m, dist_m, rdir_m = gather_dwell_tof_points(
                        lidar_df, [int(li_m)], rays_arr, flip_h=self.flip_h, flip_v=self.flip_v, rot_deg=self.rot_deg
                    )
                    if len(X_m) < 12:
                        continue
                    T_ck_cm = T_ck_w @ T_wc_m
                    R_ck_cm = T_ck_cm[:3, :3]
                    t_ck_cm = T_ck_cm[:3, 3]
                    if float(np.linalg.norm(t_ck_cm)) <= 0.50:
                        window_frames.append((R_ck_cm, t_ck_cm, X_m, dist_m, rdir_m))

            planes_k = []
            if rgb_img is not None and depth_service is not None and hasattr(depth_service, "run_depthor_plus"):
                try:
                    h, w = rgb_img.shape[:2]
                    dists_raw = np.array([float(row.get(f"distance_{i}", np.nan)) for i in range(64)], dtype=np.float64)
                    sparse_tof = project_lidar_frame_to_sparse_tof(
                        dists_raw, rays_arr, self.T_prior, K, h, w,
                        flip_h=self.flip_h, flip_v=self.flip_v, rot_deg=self.rot_deg
                    )
                    g_cam = None
                    if g_world is not None and all(k in kf for k in ("qx", "qy", "qz", "qw")):
                        R_wc = Rotation.from_quat([kf["qx"], kf["qy"], kf["qz"], kf["qw"]]).as_matrix() @ session_R_dc
                        g_cam = R_wc.T @ np.asarray(g_world, dtype=float)
                    kf_rgb_path = (
                        kf.get("_rgb_path")
                        or kf.get("rgb_path")
                        or kf.get("image_path")
                        or kf.get("filename")
                    )
                    dense = depth_service.run_depthor_plus(
                        rgb_img,
                        sparse_tof,
                        K,
                        g_cam=g_cam,
                        z_max_ghost=z_max_ghost,
                        rgb_path=kf_rgb_path,
                    )
                    if dense is not None:
                        planes_k = extract_camera_planes_from_trihedral_depth(dense, K)
                except Exception:
                    planes_k = []

            if not planes_k:
                # Extract planes from the multi-frame accumulated dToF cloud in keyframe c_k
                pts_ck_list = []
                for R_rel, t_rel, X_m, _, _ in window_frames:
                    p_cm = X_m @ R_prior.T + t_prior
                    p_ck = p_cm @ R_rel.T + t_rel
                    pts_ck_list.append(p_ck)
                p_cam_window = np.vstack(pts_ck_list)
                planes_k = extract_planes_from_sparse_tof_points(p_cam_window)
            if not planes_k:
                continue

            for pl in planes_k:
                all_planes.append(pl)

            # Pull planes (n_k, d_k) in c_k back to each frame c_m in window_frames:
            # n'_{k,m} = R_{c_k c_m}^T n_k, d'_{k,m} = d_k - n_k^T t_{c_k c_m}
            for R_rel, t_rel, X_m, dist_m, rdir_m in window_frames:
                planes_m = []
                for pl in planes_k:
                    n_k = np.asarray(pl["n"], dtype=np.float64)
                    d_k = float(pl["d"])
                    n_m = R_rel.T @ n_k
                    d_m = d_k - float(np.dot(n_k, t_rel))
                    if d_m < 0.0:
                        n_m = -n_m
                        d_m = -d_m
                    if d_m > 0.10:
                        planes_m.append({"n": n_m, "d": d_m, "inlier_count": pl.get("inlier_count", 0)})
                if not planes_m:
                    continue
                idx_m, r_m, w_m, keep_m, n_asg_m = _assign(
                    X_m, rdir_m, dist_m, planes_m, R_prior, t_prior
                )
                if np.count_nonzero(keep_m) < 8:
                    continue
                ds_m = np.array([p["d"] for p in planes_m], dtype=np.float64)
                all_X.append(X_m[keep_m])
                all_n.append(n_asg_m[keep_m])
                all_d.append(ds_m[idx_m[keep_m]])
                all_w.append(w_m[keep_m])

        if not all_X:
            return None

        X_pool = np.vstack(all_X)
        n_pool = np.vstack(all_n)
        d_pool = np.concatenate(all_d)
        w_pool = np.concatenate(all_w)
        if len(X_pool) < EM_MIN_TOF_HITS:
            return None

        p0 = X_pool @ R_prior.T + t_prior
        r0 = np.sum(n_pool * p0, axis=1) - d_pool
        rms_prior = float(np.sqrt(np.average(r0 ** 2, weights=w_pool)))

        # Tikhonov-regularized 6-DoF Gauss-Newton refinement around T_prior
        reg_t = 1.0 / (0.025 ** 2)
        reg_r = 1.0 / (np.radians(2.0) ** 2)
        RX = X_pool @ R_prior.T

        def fun_reg(x):
            tau, omega = x[:3], x[3:]
            Rd = Rotation.from_rotvec(omega).as_matrix() @ R_prior
            td = t_prior + tau
            p = X_pool @ Rd.T + td
            r = np.sum(n_pool * p, axis=1) - d_pool
            abs_r = np.abs(r)
            e_r = np.where(
                abs_r <= EM_HUBER_DELTA_M,
                r,
                np.sign(r) * np.sqrt(np.maximum(0.0, 2.0 * EM_HUBER_DELTA_M * abs_r - EM_HUBER_DELTA_M ** 2)),
            )
            res_data = e_r * np.sqrt(w_pool)
            res_prior = np.concatenate([tau * np.sqrt(reg_t), omega * np.sqrt(reg_r)])
            return np.concatenate([res_data, res_prior])

        sol = least_squares(fun_reg, np.zeros(6), loss="linear", max_nfev=35)
        omega = sol.x[3:]
        Rd = Rotation.from_rotvec(omega).as_matrix() @ R_prior
        p_rot = X_pool @ Rd.T + t_prior
        b_vec = d_pool - np.sum(n_pool * p_rot, axis=1)
        tau = _solve_weighted_extrinsics_step(n_pool, b_vec, w_pool)
        dt = float(np.linalg.norm(tau))
        dr = float(np.degrees(np.linalg.norm(omega)))
        trust_ok = (dt <= EM_TRUST_T_M) and (dr <= EM_TRUST_R_DEG)
        if trust_ok:
            R_cand = _svd_so3(Rotation.from_rotvec(omega).as_matrix() @ R_prior)
            t_cand = t_prior + tau
            p_cand = X_pool @ R_cand.T + t_cand
            r_cand = np.sum(n_pool * p_cand, axis=1) - d_pool
            rms_cand = float(np.sqrt(np.average(r_cand ** 2, weights=w_pool)))
            improved = rms_cand < rms_prior - 1e-6
        else:
            improved = False

        if trust_ok and improved:
            R_star = R_cand
            t_star = t_cand
            rms_star = rms_cand
            accepted = True
            reason = None
        else:
            R_star = R_prior
            t_star = t_prior
            rms_star = rms_prior
            dt = 0.0
            dr = 0.0
            accepted = False
            reason = "trust_region_exceeded" if not trust_ok else "no_improvement"

        T_star = np.eye(4, dtype=np.float64)
        T_star[:3, :3] = R_star
        T_star[:3, 3] = t_star

        # Report total delta and RMS improvement relative to the input T0 as well as T_prior
        p_t0 = X_pool @ _svd_so3(self.T0[:3, :3]).T + self.T0[:3, 3]
        r_t0 = np.sum(n_pool * p_t0, axis=1) - d_pool
        rms_t0_raw = float(np.sqrt(np.average(r_t0 ** 2, weights=w_pool)))
        rms_T0_report = rms_t0_raw
        dt_total = float(np.linalg.norm(T_star[:3, 3] - self.T0[:3, 3]))
        dr_total = _geodesic_deg(T_star[:3, :3], self.T0[:3, :3])
        if dt_total > EM_TRUST_T_M or dr_total > EM_TRUST_R_DEG:
            dt_total = dt
            dr_total = dr

        J_geom = np.hstack([n_pool, np.cross(RX, n_pool)]) * np.sqrt(w_pool)[:, None]
        M = J_geom.T @ J_geom + np.diag([reg_t, reg_t, reg_t, reg_r, reg_r, reg_r])
        d_diag = np.sqrt(np.maximum(np.diag(M), 1e-12))
        D = np.diag(1.0 / d_diag)
        eigs = np.linalg.eigvalsh(D @ M @ D)
        cond_M = float(np.max(np.abs(eigs)) / max(float(np.min(np.abs(eigs))), 1e-12))

        return {
            "T_lidar_camera": T_star,
            "accepted": accepted,
            "reason": reason,
            "plane_source": "multi_keyframe_em",
            "T0": self.T_prior,
            "dwell": {
                "t0_ns": int(sampled_kfs[0].get("device_timestamp_ns", 0)),
                "t1_ns": int(sampled_kfs[-1].get("device_timestamp_ns", 0)),
                "n_lidar_rows": len(sampled_kfs),
                "span_m": 0.0,
            },
            "planes": all_planes[:6],
            "inliers_per_plane": [int(p.get("inlier_count", 0)) for p in all_planes[:6]],
            "iterations": int(getattr(sol, "nfev", 1)),
            "cond_M": cond_M,
            "residual_rms_T0": rms_T0_report,
            "residual_rms_Tstar": rms_star,
            "residual_rms": rms_star,
            "delta_t_m": dt_total if dt_total > 0 else dt,
            "delta_r_deg": dr_total if dr_total > 0 else dr,
        }


def _refine_multi_keyframe_extrinsics(
    *,
    T0: np.ndarray,
    keyframes: list[dict],
    lidar_df: pd.DataFrame,
    frame_index: dict[int, int] | None,
    rays: np.ndarray,
    K: np.ndarray,
    depth_service: object,
    flip_h: bool,
    flip_v: bool,
    rot_deg: int | float,
    session_dir: str,
    g_world: np.ndarray | None = None,
    z_max_ghost: float | None = None,
    pose_df: pd.DataFrame | None = None,
) -> dict | None:
    """Multi-keyframe Point-to-Plane + Wahba-prior extrinsic calibration for handheld scans."""
    if len(keyframes) < 5 or lidar_df is None or len(lidar_df) < 10:
        return None

    proc_path = os.path.join(session_dir, "processed_extrinsics.json")
    R_phone_hub = None
    ext_cal = None
    if os.path.isfile(proc_path):
        try:
            with open(proc_path, "r", encoding="utf-8") as f:
                pdata = json.load(f)
            if pdata.get("R_phone_hub") is not None:
                R_phone_hub = np.asarray(pdata["R_phone_hub"], dtype=np.float64)
            ext_cal = pdata.get("extrinsics_calibrated")
        except Exception:
            pass

    calibrator = MultiKeyframePlaneCalibration(
        T0=T0,
        R_phone_hub=R_phone_hub,
        rot_deg=rot_deg,
        extrinsics_calibrated=ext_cal,
        flip_h=flip_h,
        flip_v=flip_v,
    )
    return calibrator.refine(
        keyframes=keyframes,
        lidar_df=lidar_df,
        frame_index=frame_index,
        rays=rays,
        K=K,
        depth_service=depth_service,
        pose_df=pose_df,
        g_world=g_world,
        z_max_ghost=z_max_ghost,
    )



def refine_lidar_camera_extrinsics_em(
    *,
    T0: np.ndarray | None,
    pose_df: pd.DataFrame,
    keyframes: list[dict],
    lidar_pack: dict,
    K: np.ndarray,
    depth_service: object,
    flip_h: bool,
    flip_v: bool,
    rot_deg: int | float,
    session_dir: str,
    g_world: np.ndarray | None = None,
    z_max_ghost: float | None = None,
) -> dict:
    T_fallback = T0 if _valid_T(T0) else _identity_T()
    result = {
        "T_lidar_camera": np.asarray(T_fallback, dtype=np.float64).copy(),
        "accepted": False,
        "reason": "no_dwell",
        "plane_source": None,
        "T0": np.asarray(T_fallback, dtype=np.float64).copy(),
        "dwell": {"t0_ns": None, "t1_ns": None, "n_lidar_rows": 0, "span_m": None},
        "planes": [],
        "inliers_per_plane": [],
        "iterations": 0,
        "cond_M": None,
        "residual_rms_T0": None,
        "residual_rms_Tstar": None,
        "residual_rms": None,
        "delta_t_m": None,
        "delta_r_deg": None,
    }

    try:
        if not _valid_T(T0):
            result["reason"] = "invalid_T0"
            _write_em_json(session_dir, result)
            return result

        lidar_df = lidar_pack.get("df") if isinstance(lidar_pack, dict) else None
        frame_index = lidar_pack.get("frame_index") if isinstance(lidar_pack, dict) else None
        rays = lidar_pack.get("rays") if isinstance(lidar_pack, dict) else None
        if rays is None:
            rays = project_lidar_rays()

        dwell = find_stationary_dwell(pose_df, lidar_df, frame_index=frame_index)
        if dwell is None:
            mkf = _refine_multi_keyframe_extrinsics(
                T0=np.asarray(T0, dtype=np.float64),
                keyframes=keyframes,
                lidar_df=lidar_df,
                frame_index=frame_index,
                rays=rays,
                K=K,
                depth_service=depth_service,
                flip_h=flip_h,
                flip_v=flip_v,
                rot_deg=rot_deg,
                session_dir=session_dir,
                g_world=g_world,
                z_max_ghost=z_max_ghost,
                pose_df=pose_df,
            )
            if mkf is not None and mkf.get("accepted"):
                result.update(mkf)
                _persist_accepted_extrinsics(session_dir, result, flip_h, flip_v, rot_deg)
                _write_em_json(session_dir, result)
                return result
            result["reason"] = "no_dwell"
            _write_em_json(session_dir, result)
            return result

        result["dwell"] = {
            "t0_ns": dwell.get("t0_ns"),
            "t1_ns": dwell.get("t1_ns"),
            "n_lidar_rows": dwell.get("n_lidar_rows", 0),
            "span_m": dwell.get("span_m"),
        }

        X, distances, ray_dirs = gather_dwell_tof_points(
            lidar_df,
            dwell.get("lidar_iloc", []),
            rays,
            flip_h=flip_h,
            flip_v=flip_v,
            rot_deg=rot_deg,
        )
        if len(X) == 0:
            result["reason"] = "dwell_tof_empty"
            _write_em_json(session_dir, result)
            return result

        if len(X) < EM_MIN_TOF_HITS:
            result["reason"] = "too_few_tof"
            _write_em_json(session_dir, result)
            return result

        best_kf = None
        min_dt = None
        median_ns = dwell.get("median_ns")
        if median_ns is not None:
            for kf in keyframes:
                rgb_path = kf.get("_rgb_path") or kf.get("rgb_path")
                ts = kf.get("device_timestamp_ns")
                if not rgb_path or ts is None:
                    continue
                try:
                    dt = abs(float(ts) - float(median_ns))
                except (ValueError, TypeError):
                    continue
                if min_dt is None or dt < min_dt:
                    min_dt = dt
                    best_kf = kf

        if best_kf is None:
            result["reason"] = "no_dwell_rgb"
            _write_em_json(session_dir, result)
            return result

        rgb_path = best_kf.get("_rgb_path") or best_kf.get("rgb_path")
        rgb_img = cv2.imread(rgb_path) if (cv2 is not None and rgb_path) else None
        if rgb_img is None:
            result["reason"] = "no_dwell_rgb"
            _write_em_json(session_dir, result)
            return result

        h, w = rgb_img.shape[:2]
        sparse_tof = np.zeros((h, w), dtype=np.float32)
        row = lookup_lidar_row(
            lidar_df,
            best_kf.get("frame"),
            best_kf.get("device_timestamp_ns"),
            frame_index=frame_index,
        )
        if row is not None and rays is not None:
            dists = np.array([float(row.get(f"distance_{i}", np.nan)) for i in range(64)], dtype=np.float64)
            sparse_tof = project_lidar_frame_to_sparse_tof(
                dists,
                rays,
                T0,
                K,
                h,
                w,
                flip_h=flip_h,
                flip_v=flip_v,
                rot_deg=rot_deg,
            )

        planes = []
        plane_source = None
        if depth_service is not None and hasattr(depth_service, "run_depthor_plus"):
            try:
                g_cam = None
                if g_world is not None and all(k in best_kf for k in ("qx", "qy", "qz", "qw")):
                    R_wc_all_em = None
                    if pose_df is not None and all(c in pose_df.columns for c in ("qx", "qy", "qz", "qw")) and len(pose_df) > 0:
                        try:
                            R_wc_all_em = Rotation.from_quat(
                                pose_df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
                            ).as_matrix()
                        except Exception:
                            R_wc_all_em = None
                    R_dc_em = resolve_sensor_to_vio_rotation(
                        rot_deg, img_w=w, img_h=h, R_wc_all=R_wc_all_em, g_world=g_world
                    )
                    R_dw = (
                        Rotation.from_quat(
                            [best_kf["qx"], best_kf["qy"], best_kf["qz"], best_kf["qw"]]
                        ).as_matrix()
                        @ R_dc_em
                    )
                    g_cam = R_dw.T @ np.asarray(g_world, dtype=float)
                best_kf_rgb_path = (
                    best_kf.get("_rgb_path")
                    or best_kf.get("rgb_path")
                    or best_kf.get("image_path")
                    or best_kf.get("filename")
                )
                dense = depth_service.run_depthor_plus(
                    rgb_img,
                    sparse_tof,
                    K,
                    g_cam=g_cam,
                    z_max_ghost=z_max_ghost,
                    rgb_path=best_kf_rgb_path,
                )
                if dense is not None:
                    dense_planes = extract_camera_planes_from_dense_depth(dense, K)
                    if planes_are_observable(dense_planes):
                        planes = dense_planes
                        plane_source = "dense_dwell"
            except Exception:
                pass

        if not planes_are_observable(planes):
            R0 = T0[:3, :3]
            t0 = T0[:3, 3]
            p_cam = X @ R0.T + t0
            sparse_planes = extract_planes_from_sparse_tof_points(p_cam)
            if planes_are_observable(sparse_planes):
                planes = sparse_planes
                plane_source = "sparse_tof_fallback"

        if not planes_are_observable(planes):
            mkf = _refine_multi_keyframe_extrinsics(
                T0=np.asarray(T0, dtype=np.float64),
                keyframes=keyframes,
                lidar_df=lidar_df,
                frame_index=frame_index,
                rays=rays,
                K=K,
                depth_service=depth_service,
                flip_h=flip_h,
                flip_v=flip_v,
                rot_deg=rot_deg,
                session_dir=session_dir,
                g_world=g_world,
                z_max_ghost=z_max_ghost,
                pose_df=pose_df,
            )
            if mkf is not None and mkf.get("accepted"):
                result.update(mkf)
                _persist_accepted_extrinsics(session_dir, result, flip_h, flip_v, rot_deg)
                _write_em_json(session_dir, result)
                return result
            result["reason"] = "unobservable"
            _write_em_json(session_dir, result)
            return result

        result["planes"] = planes
        em_out = em_point_to_plane(X, ray_dirs, distances, planes, T0)

        if em_out.get("accepted") and plane_source == "sparse_tof_fallback":
            rms = em_out.get("residual_rms")
            rms0 = em_out.get("residual_rms_T0")
            if rms is None or rms0 is None or rms >= 0.85 * rms0:
                em_out["accepted"] = False
                em_out["reason"] = "trust_or_residual"
                em_out["T"] = T0.copy()

        result["accepted"] = bool(em_out.get("accepted", False))
        result["reason"] = em_out.get("reason")
        result["plane_source"] = plane_source
        result["inliers_per_plane"] = em_out.get("inliers_per_plane", [])
        result["iterations"] = int(em_out.get("iterations", 0))
        result["cond_M"] = em_out.get("cond_M")
        result["residual_rms_T0"] = em_out.get("residual_rms_T0")
        result["residual_rms_Tstar"] = em_out.get("residual_rms")
        result["residual_rms"] = em_out.get("residual_rms")
        result["delta_t_m"] = em_out.get("delta_t_m")
        result["delta_r_deg"] = em_out.get("delta_r_deg")
        result["T_lidar_camera"] = em_out.get("T", T_fallback).copy()

        if not result["accepted"]:
            mkf = _refine_multi_keyframe_extrinsics(
                T0=np.asarray(T0, dtype=np.float64),
                keyframes=keyframes,
                lidar_df=lidar_df,
                frame_index=frame_index,
                rays=rays,
                K=K,
                depth_service=depth_service,
                flip_h=flip_h,
                flip_v=flip_v,
                rot_deg=rot_deg,
                session_dir=session_dir,
                g_world=g_world,
                z_max_ghost=z_max_ghost,
                pose_df=pose_df,
            )
            if mkf is not None and mkf.get("accepted"):
                result.update(mkf)

        if result["accepted"]:
            _persist_accepted_extrinsics(session_dir, result, flip_h, flip_v, rot_deg)

    except Exception:
        result["reason"] = "exception"
        result["accepted"] = False
        result["T_lidar_camera"] = np.asarray(T_fallback, dtype=np.float64).copy()

    try:
        _write_em_json(session_dir, result)
    except Exception:
        pass
    return result


def refine_extrinsics_from_session(
    session_dir: str,
    *,
    depth_service: object | None = None,
    prefer_server_vio: bool = False,
    rot_deg: int | float | None = None,
) -> dict:
    """High-level entry point to refine and persist LiDAR-Camera extrinsics for a session directory."""
    lidar_pack = load_processed_lidar_for_tof(session_dir)
    T0 = lidar_pack.get("T_lidar_camera")
    flip_h = bool(lidar_pack.get("flip_h", False))
    flip_v = bool(lidar_pack.get("flip_v", False))
    if rot_deg is None:
        rot_deg = int(lidar_pack.get("camera_sensor_to_display_rotation_deg", 90))

    cam_csv = os.path.join(session_dir, "camera_matrix.csv")
    K = load_camera_intrinsics(cam_csv) if os.path.isfile(cam_csv) else np.eye(3, dtype=np.float64)

    pose_df = pd.DataFrame()
    pose_candidates = (
        ["processed_vio.csv", "processed_odometry.csv", "odometry.csv"]
        if prefer_server_vio
        else ["processed_odometry.csv", "processed_vio.csv", "odometry.csv"]
    )
    chosen_file = None
    for fname in pose_candidates:
        fpath = os.path.join(session_dir, fname)
        if os.path.isfile(fpath):
            try:
                df = pd.read_csv(fpath)
                if not df.empty:
                    if "timestamp" in df.columns and "device_timestamp_ns" not in df.columns:
                        ts_num = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
                        finite_ts = ts_num[np.isfinite(ts_num) & (ts_num > 0.0)]
                        scale_ts = 1e9 if (len(finite_ts) > 0 and float(np.median(finite_ts)) < 1e11) else 1.0
                        df["device_timestamp_ns"] = np.rint(ts_num * scale_ts).astype(np.int64)
                    pose_df = df
                    chosen_file = fname
                    break
            except Exception:
                continue

    if chosen_file == "processed_vio.csv":
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        pose_df.attrs["pose_source"] = "server_vio"
        pose_df.attrs["camera_frame_convention"] = "sensor_opencv"
    else:
        g_world = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        pose_df.attrs["pose_source"] = "arcore_odometry"
        pose_df.attrs["camera_frame_convention"] = "sensor_opencv"
        if not pose_df.empty and all(c in pose_df.columns for c in ("qx", "qy", "qz", "qw")):
            q_gl = normalize_quaternions_vectorized(
                pose_df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
            )
            q_x_180 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            q_cv = (Rotation.from_quat(q_gl) * Rotation.from_quat(q_x_180)).as_quat()
            if rot_deg is not None and int(round(float(rot_deg))) % 360 != 0:
                R_dc = _display_to_camera_sensor_rotation(rot_deg)
                q_sensor = (Rotation.from_quat(q_cv) * Rotation.from_matrix(R_dc)).as_quat()
            else:
                q_sensor = q_cv
            q_norm = normalize_quaternions_vectorized(q_sensor)
            pose_df["qx"] = q_norm[:, 0]
            pose_df["qy"] = q_norm[:, 1]
            pose_df["qz"] = q_norm[:, 2]
            pose_df["qw"] = q_norm[:, 3]

    keyframes: list[dict] = []
    if not pose_df.empty:
        kf_source_df = pose_df
        if "frame" in pose_df.columns:
            fr_num = pd.to_numeric(pose_df["frame"], errors="coerce")
            nonneg = pose_df[fr_num >= 0]
            if not nonneg.empty:
                kf_source_df = nonneg
        step = max(1, len(kf_source_df) // 60)
        rgb_dir = os.path.join(session_dir, "rgb")
        for _, row in kf_source_df.iloc[::step].iterrows():
            item = row.to_dict()
            frm = item.get("frame")
            if frm is not None and np.isfinite(float(frm)) and float(frm) >= 0.0:
                fidx = int(round(float(frm)))
                for pat in (f"frame_{fidx:06d}.jpg", f"frame_{fidx}.jpg", f"{fidx:06d}.jpg"):
                    cand = os.path.join(rgb_dir, pat)
                    if os.path.isfile(cand):
                        item["_rgb_path"] = cand
                        break
            keyframes.append(item)

    return refine_lidar_camera_extrinsics_em(
        T0=T0,
        pose_df=pose_df,
        keyframes=keyframes,
        lidar_pack=lidar_pack,
        K=K,
        depth_service=depth_service,
        flip_h=flip_h,
        flip_v=flip_v,
        rot_deg=rot_deg,
        session_dir=session_dir,
        g_world=g_world,
    )




