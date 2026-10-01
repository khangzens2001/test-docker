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


def load_T_bc(session_dir: str) -> np.ndarray:
    ext_path = os.path.join(session_dir, "processed_extrinsics.json")
    meaning = ""
    T_raw = None
    if os.path.isfile(ext_path):
        with open(ext_path, encoding="utf-8") as f:
            data = json.load(f)
        meaning = str(data.get("T_imu_camera_meaning") or "")
        arr = np.asarray(data.get("T_imu_camera"), dtype=np.float64)
        if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
            T_raw = arr
    if T_raw is None:
        T_raw, _, meta = load_session_extrinsics(session_dir)
        meaning = str(meta.get("T_imu_camera_meaning") or meaning)
    schema13 = meaning == "phone_imu_android_device_screen_to_arcore_sensor_image"
    if schema13 and np.allclose(T_raw[:3, :3], T_GL_CV[:3, :3], atol=1e-3):
        return T_raw.copy()
    if schema13:
        return T_raw @ T_GL_CV
    return T_raw.copy()


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


LIDAR_RGB_MATCH_NS = 50_000_000


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
    if frame is not None and frame_index is not None:
        idx = frame_index.get(int(frame))
        if idx is not None:
            return lidar_df.iloc[int(idx)]
    if frame is not None and "matched_frame" in lidar_df.columns:
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
    if timestamp_ns is None or "device_timestamp_ns" not in lidar_df.columns:
        return None
    ts = pd.to_numeric(lidar_df["device_timestamp_ns"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(ts)
    if not np.any(finite):
        return None
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
    if best is None:
        return None
    return lidar_df.iloc[best]


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
) -> dict | None:
    if pose_df is None or pose_df.empty or lidar_df is None or getattr(lidar_df, "empty", True):
        return None
    if "device_timestamp_ns" not in pose_df.columns:
        return None
    if not all(c in pose_df.columns for c in ("x", "y", "z")):
        return None
    ts = pd.to_numeric(pose_df["device_timestamp_ns"], errors="coerce").to_numpy(dtype=np.float64)
    xyz = pose_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    matched_pose = []
    lidar_iloc = []
    for i in range(len(pose_df)):
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
    work = pcd
    planes = []
    for _ in range(EM_MAX_PLANES):
        if len(work.points) < 50:
            break
        model, inliers = work.segment_plane(
            distance_threshold=EM_RANSAC_DIST_M, ransac_n=3, num_iterations=400
        )
        if len(inliers) < EM_MIN_INLIERS_PER_PLANE:
            break
        n, d = o3d_model_to_nd(model)
        inlier_pts = np.asarray(work.select_by_index(inliers).points, dtype=float)
        planes.append(
            {"n": n, "d": d, "inliers": inlier_pts, "inlier_count": int(len(inliers))}
        )
        work = work.select_by_index(inliers, invert=True)
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
        tau, omega = sol.x[:3], sol.x[3:]
        R = Rotation.from_rotvec(omega).as_matrix() @ R
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
    p = np.asarray(p, dtype=np.float64)
    if len(p) < EM_MIN_INLIERS_PER_PLANE:
        return []
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    work = pcd
    planes = []
    for _ in range(EM_MAX_PLANES):
        if len(work.points) < EM_MIN_INLIERS_PER_PLANE:
            break
        model, inliers = work.segment_plane(
            distance_threshold=EM_RANSAC_DIST_M, ransac_n=3, num_iterations=400
        )
        if len(inliers) < EM_MIN_INLIERS_PER_PLANE:
            break
        n, d = o3d_model_to_nd(model)
        inlier_pts = np.asarray(work.select_by_index(inliers).points, dtype=float)
        planes.append(
            {"n": n, "d": d, "inliers": inlier_pts, "inlier_count": int(len(inliers))}
        )
        work = work.select_by_index(inliers, invert=True)
    return planes


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
        dwell = find_stationary_dwell(pose_df, lidar_df, frame_index=frame_index)
        if dwell is None:
            result["reason"] = "no_dwell"
            _write_em_json(session_dir, result)
            return result

        result["dwell"] = {
            "t0_ns": dwell.get("t0_ns"),
            "t1_ns": dwell.get("t1_ns"),
            "n_lidar_rows": dwell.get("n_lidar_rows", 0),
            "span_m": dwell.get("span_m"),
        }

        rays = lidar_pack.get("rays") if isinstance(lidar_pack, dict) else None
        if rays is None:
            rays = project_lidar_rays()

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
                    R_dw = Rotation.from_quat(
                        [best_kf["qx"], best_kf["qy"], best_kf["qz"], best_kf["qw"]]
                    ).as_matrix()
                    g_cam = R_dw.T @ np.asarray(g_world, dtype=float)
                dense = depth_service.run_depthor_plus(
                    rgb_img, sparse_tof, K, g_cam=g_cam, z_max_ghost=z_max_ghost
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

        if result["accepted"]:
            refined_path = os.path.join(session_dir, "refined_lidar_camera_extrinsics.json")
            refined_data = {
                "config_schema_version": 2,
                "source": "server_em_dwell",
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
            proc_data["extrinsics_source"] = "server_em_dwell"
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

    except Exception:
        result["reason"] = "exception"
        result["accepted"] = False
        result["T_lidar_camera"] = np.asarray(T_fallback, dtype=np.float64).copy()

    try:
        _write_em_json(session_dir, result)
    except Exception:
        pass
    return result



