"""Standalone Algorithm Pipeline Runner.

Orchestrates the pure algorithmic pipeline for 3D room reconstruction,
spatial measurement, and floorplan generation without any dependency on
Celery @shared_task or SQLAlchemy Database.

Pipeline Sequence:
TimeSync / Ingestion -> DELTAR EM Calibration -> VIO -> Neural Depth Completion ONNX ->
TSDF Fusion -> VGGT Prior -> Plane Segmentation -> Manhattan Dual-Stage & Occupancy Layout ->
Room Metrics Extraction -> CAD Floorplan & Wall Elevations rendering & 3D GLB export ->
Material & Bundle Estimation.

Designed to be mirrored 1:1 to deployment environments (e.g., RunPod feat-full-pipeline/app/).
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

from app.core.config import settings
from app.services.alignment import (
    WAHBA_OMEGA_MIN_RAD_S,
    apply_butterworth_lowpass,
    calculate_robust_jitter,
    estimate_R_phone_hub,
    pchip_resample,
)
from app.services.bundle import export_session_bundle
from app.services.calibration import (
    extract_lidar_display_metadata,
    load_camera_intrinsics,
    load_processed_lidar_for_tof,
    load_session_extrinsics,
    lookup_lidar_row,
    normalize_quaternions_vectorized,
    project_lidar_frame_to_sparse_tof,
    refine_lidar_camera_extrinsics_em,
    should_skip_tof_keyframe,
)
from app.services.depth_completion import DepthCompletionService, GHOST_RANGE_FLOOR_M
from app.services.estimation import DEFAULT_RECALCULATE_PARAMS
from app.services.floorplan import export_artifacts
from app.services.ingestion import (
    assign_primary_imu,
    assign_seq_num,
    detect_time_unit,
    hub_gap_stats,
    map_hub_android_from_lidar,
    mask_lidar_zones,
    parse_odometry_csv,
    parse_phone_imu,
    parse_telemetry_csv,
    reconstruct_mcu_rollover,
    reorder_bounded_dejitter_buffer,
    resolve_imu_scale,
)
from app.services.plane_segmentation import is_usable_gravity, resolve_gravity_vector
from app.services.reconstruction import ODOMETRY_RGB_MATCH_NS, ReconstructionService
from app.services.room_model import export_room_model_glb
from app.services.room_model_texture import export_room_model_texture_glb
from app.services.vggt_prior import attach_vggt_diagnostics, load_vggt_prior
from app.services.vio import VioEstimator, publish_session_vio

logger = logging.getLogger(__name__)

# Column mappings
IMU_MAPPINGS = {
    "mcu_timestamp_ns": ["imu_mcu_timestamp_ns", "mcu_timestamp_ns"],
    "ax": ["ax", "acc_x", "linear_acceleration_x"],
    "ay": ["ay", "acc_y", "linear_acceleration_y"],
    "az": ["az", "acc_z", "linear_acceleration_z"],
    "gx": ["gx", "gyro_x", "rotation_rate_x"],
    "gy": ["gy", "gyro_y", "rotation_rate_y"],
    "gz": ["gz", "gyro_z", "rotation_rate_z"],
    "seq_num": ["imu_seq_num", "seq"],
    "receive_timestamp_ns": ["mobile_receive_timestamp_nanos"],
    "legacy_device_timestamp_ns": ["imu_motion_sync_android_nanos", "timestamp"],
}
IMU_OPTIONAL = frozenset({"seq_num", "receive_timestamp_ns", "legacy_device_timestamp_ns"})

LIDAR_MAPPINGS = {
    "mcu_timestamp_ns": ["lidar_mcu_timestamp_ns", "mcu_timestamp_ns"],
    "device_timestamp_ns": [
        "lidar_android_timestamp_nanos",
        "time_sync_mapped_android_nanos",
        "android_timestamp_nanos",
        "timestamp",
    ],
    "distance_{0..63}": [],
    "status_{0..63}": [],
    "seq_num": ["lidar_seq_num", "seq"],
    "matched_frame": ["matched_frame"],
    "match_delta_nanos": ["match_delta_nanos"],
    "receive_timestamp_ns": ["mobile_receive_timestamp_nanos"],
}
LIDAR_OPTIONAL = frozenset({
    "seq_num",
    "matched_frame",
    "match_delta_nanos",
    "receive_timestamp_ns",
    "device_timestamp_ns",
})

VIO_MAPPINGS = {
    "device_timestamp_ns": ["timestamp"],
    "frame": ["frame"],
    "x": ["x", "pos_x"],
    "y": ["y", "pos_y"],
    "z": ["z", "pos_z"],
    "qx": ["qx", "quat_x"],
    "qy": ["qy", "quat_y"],
    "qz": ["qz", "quat_z"],
    "qw": ["qw", "quat_w"],
}

ODO_JUMP_SPEED_MPS = 2.5
ODO_JUMP_DIST_M = 0.20
ODO_JUMP_DT_S = 0.10
ODO_JUMP_ROT_DEG = 25.0
ODO_FREEZE_FRAMES = 5
ODO_MIN_ISLAND_FRAMES = 30
ODO_FREEZE_POS_EPS_M = 1e-4
GHOST_TRAJ_MARGIN_M = 1.00


def _decimate_keyframes(keyframes: list[dict], max_limit: int = 150) -> list[dict]:
    if len(keyframes) <= max_limit:
        return keyframes
    indices = np.unique(np.linspace(0, len(keyframes) - 1, max_limit, dtype=int))
    return [keyframes[i] for i in indices]


def _geodesic_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    r0 = Rotation.from_quat(q0)
    r1 = Rotation.from_quat(q1)
    return float(np.degrees((r0.inv() * r1).magnitude()))


def _yaw_rad(qx, qy, qz, qw) -> float:
    from scipy.spatial.transform import Rotation

    fwd = Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ np.array([0.0, 0.0, 1.0])
    return float(np.arctan2(fwd[0], fwd[2]))


def _yaw_span_deg(yaws: np.ndarray) -> float:
    if yaws.size < 2:
        return 0.0
    degs = np.sort(np.degrees(yaws) % 360.0)
    gaps = np.diff(degs)
    wrap = float(degs[0] + 360.0 - degs[-1])
    largest = float(max(float(gaps.max()) if len(gaps) else 0.0, wrap))
    return float(np.clip(360.0 - largest, 0.0, 360.0))


def _apply_island_attrs(out: pd.DataFrame, src: pd.DataFrame, extra: dict) -> pd.DataFrame:
    out.attrs.clear()
    out.attrs["pose_source"] = src.attrs.get("pose_source", "arcore_odometry")
    for k, v in extra.items():
        if v is not None:
            out.attrs[k] = v
    return out


def select_best_tracking_island(
    pose_df: pd.DataFrame, imu_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    cols = ["device_timestamp_ns", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"]
    req_cols = ["device_timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"]
    if pose_df is None or pose_df.empty:
        out = pose_df.copy() if pose_df is not None else pd.DataFrame(columns=cols)
        return _apply_island_attrs(
            out,
            pose_df if pose_df is not None else out,
            {"tracking_islands_detected": 0, "discarded_frames": 0, "island_fail_open_reason": "empty"},
        )
    df = pose_df.copy()
    for c in req_cols:
        if c not in df.columns:
            return _apply_island_attrs(
                pose_df.copy(),
                pose_df,
                {"tracking_islands_detected": 1, "discarded_frames": 0, "island_fail_open_reason": "too_short"},
            )
    finite = np.all(np.isfinite(df[req_cols].to_numpy(dtype=np.float64)), axis=1)
    df = df.loc[finite].sort_values("device_timestamp_ns").reset_index(drop=True)
    if len(df) < 2:
        reason = "empty" if len(df) == 0 else "too_short"
        detected = 0 if len(df) == 0 else 1
        return _apply_island_attrs(
            df if len(df) else pose_df.copy(),
            pose_df,
            {"tracking_islands_detected": detected, "discarded_frames": 0, "island_fail_open_reason": reason},
        )

    from scipy.spatial.transform import Rotation

    ts = df["device_timestamp_ns"].to_numpy(dtype=np.float64)
    p = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    q = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    n = len(df)

    has_gyro = False
    ts_imu, gx, gy, gz = None, None, None, None
    if imu_df is not None and not imu_df.empty:
        ts_col = (
            "device_timestamp_ns"
            if "device_timestamp_ns" in imu_df.columns
            else ("timestamp_nanos" if "timestamp_nanos" in imu_df.columns else "timestamp")
        )
        gx_col = "gx" if "gx" in imu_df.columns else ("gyro_x" if "gyro_x" in imu_df.columns else None)
        gy_col = "gy" if "gy" in imu_df.columns else ("gyro_y" if "gyro_y" in imu_df.columns else None)
        gz_col = "gz" if "gz" in imu_df.columns else ("gyro_z" if "gyro_z" in imu_df.columns else None)
        if ts_col in imu_df.columns and gx_col and gy_col and gz_col:
            raw_ts = imu_df[ts_col].to_numpy(dtype=np.float64)
            ts_imu = raw_ts if raw_ts[0] > 1e12 else (raw_ts * 1e9)
            gx = imu_df[gx_col].to_numpy(dtype=np.float64)
            gy = imu_df[gy_col].to_numpy(dtype=np.float64)
            gz = imu_df[gz_col].to_numpy(dtype=np.float64)
            has_gyro = True

    split_before = np.zeros(n, dtype=bool)
    freeze_run = 1
    for i in range(1, n):
        dt = (ts[i] - ts[i - 1]) / 1e9
        dist = float(np.linalg.norm(p[i] - p[i - 1]))
        rot_deg = _geodesic_deg(q[i - 1], q[i])

        if has_gyro:
            m = (ts_imu >= ts[i - 1]) & (ts_imu <= ts[i])
            if np.any(m):
                w_norm = np.sqrt(gx[m] ** 2 + gy[m] ** 2 + gz[m] ** 2)
                gyro_rot_deg = float(np.sum(w_norm) * (dt / max(1, np.sum(m))) * (180.0 / np.pi))
            else:
                gyro_rot_deg = 0.0

            rot_diff = abs(rot_deg - gyro_rot_deg)
            is_true_jump = (rot_diff > 20.0 and rot_deg > 25.0) or (dist > 0.80 and dt <= 0.15)
            if is_true_jump:
                split_before[i] = True
        else:
            speed_hit = dt > 0.0 and (dist / dt) > ODO_JUMP_SPEED_MPS
            dist_hit = dist > ODO_JUMP_DIST_M and dt <= ODO_JUMP_DT_S
            rot_hit = rot_deg > ODO_JUMP_ROT_DEG
            if float(np.linalg.norm(p[i] - p[i - 1])) < ODO_FREEZE_POS_EPS_M:
                freeze_run += 1
            else:
                freeze_hit = freeze_run >= ODO_FREEZE_FRAMES and (speed_hit or dist_hit or rot_hit)
                freeze_run = 1
                if freeze_hit or speed_hit or dist_hit or rot_hit:
                    split_before[i] = True
                continue
            if speed_hit or dist_hit or rot_hit:
                split_before[i] = True

    starts = [0] + [i for i in range(1, n) if split_before[i]]
    ends = [s - 1 for s in starts[1:]] + [n - 1]
    islands = list(zip(starts, ends))

    # If IMU gyro is available, stitch valid multi-islands
    if has_gyro and len(islands) > 1:
        valid_islands = [(s, e) for s, e in islands if (e - s + 1) >= ODO_MIN_ISLAND_FRAMES]
        if valid_islands:
            s0, e0 = valid_islands[0]
            stitched_p = [p[i].copy() for i in range(s0, e0 + 1)]
            stitched_q = [Rotation.from_quat(q[i]) for i in range(s0, e0 + 1)]
            stitched_indices = list(range(s0, e0 + 1))

            for k in range(1, len(valid_islands)):
                prev_s, prev_e = valid_islands[k - 1]
                next_s, next_e = valid_islands[k]

                t_prev = ts[prev_e]
                t_next = ts[next_s]
                dt_gap = (t_next - t_prev) / 1e9

                m = (ts_imu >= t_prev) & (ts_imu <= t_next)
                if np.any(m):
                    omega = np.array([gx[m].mean(), gy[m].mean(), gz[m].mean()]) * dt_gap
                    delta_R = Rotation.from_rotvec(omega)
                else:
                    delta_R = Rotation.identity()

                R_target = stitched_q[-1] * delta_R
                R_jump = Rotation.from_quat(q[next_s])
                R_corr = R_target * R_jump.inv()

                p_anchor = stitched_p[-1]
                p_jump = p[next_s]

                for i in range(next_s, next_e + 1):
                    p_rel = p[i] - p_jump
                    stitched_p.append(p_anchor + R_corr.apply(p_rel))
                    stitched_q.append(R_corr * Rotation.from_quat(q[i]))
                    stitched_indices.append(i)

            stitched_df = df.iloc[stitched_indices].copy().reset_index(drop=True)
            stitched_p_arr = np.array(stitched_p)
            stitched_q_arr = np.array([q_obj.as_quat() for q_obj in stitched_q])
            stitched_df["x"] = stitched_p_arr[:, 0]
            stitched_df["y"] = stitched_p_arr[:, 1]
            stitched_df["z"] = stitched_p_arr[:, 2]
            stitched_df["qx"] = stitched_q_arr[:, 0]
            stitched_df["qy"] = stitched_q_arr[:, 1]
            stitched_df["qz"] = stitched_q_arr[:, 2]
            stitched_df["qw"] = stitched_q_arr[:, 3]

            yaws = np.array(
                [_yaw_rad(r.qx, r.qy, r.qz, r.qw) for r in stitched_df.itertuples(index=False)],
                dtype=float,
            )
            span = _yaw_span_deg(yaws)
            ratio = span / 360.0
            fr = (
                stitched_df["frame"].to_numpy()
                if "frame" in stitched_df.columns
                else np.array([0, len(stitched_df) - 1])
            )
            extra = {
                "tracking_islands_detected": len(islands),
                "selected_island_frames": (int(fr[0]), int(fr[-1])),
                "discarded_frames": int(n - len(stitched_df)),
                "selected_island_score": float(len(stitched_df) * ratio),
                "selected_island_yaw_coverage": float(ratio),
                "stitched_islands": True,
            }
            return _apply_island_attrs(stitched_df, pose_df, extra)

    scored = []
    for s, e in islands:
        sl = df.iloc[s : e + 1]
        yaws = np.array(
            [_yaw_rad(r.qx, r.qy, r.qz, r.qw) for r in sl.itertuples(index=False)],
            dtype=float,
        )
        span = _yaw_span_deg(yaws)
        ratio = span / 360.0
        quads = len({int(((np.degrees(y) % 360.0) // 90.0) % 4) for y in yaws})
        scored.append((len(sl) * ratio, quads, len(sl), -s, s, e, ratio))
    scored.sort(reverse=True)
    _score, _q, n_win, _ns, s, e, ratio = scored[0]
    n_islands = len(islands)
    if n_win < ODO_MIN_ISLAND_FRAMES:
        return _apply_island_attrs(
            pose_df.copy(),
            pose_df,
            {
                "tracking_islands_detected": n_islands,
                "discarded_frames": 0,
                "island_fail_open_reason": "short_island",
            },
        )
    out = df.iloc[s : e + 1].copy()
    fr = out["frame"].to_numpy() if "frame" in out.columns else np.array([s, e])
    extra = {
        "tracking_islands_detected": n_islands,
        "selected_island_frames": (int(fr[0]), int(fr[-1])),
        "discarded_frames": int(n - (e - s + 1)),
        "selected_island_score": float(_score),
        "selected_island_yaw_coverage": float(ratio),
    }
    return _apply_island_attrs(out, pose_df, extra)


def resolve_keyframe_rgb_path(session_dir: str, kf: dict) -> str | None:
    ts = kf.get("device_timestamp_ns")
    candidates: list[str] = []
    if ts is not None:
        candidates.append(os.path.join(session_dir, "rgb", f"frame_{ts}.jpg"))
    if kf.get("frame") is not None:
        candidates.append(os.path.join(session_dir, "rgb", f"{int(kf['frame']):06d}.jpg"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_pose_table_for_tsdf(
    session_dir: str,
    *,
    apply_island_filter: bool = True,
    prefer_server_vio: bool = True,
) -> pd.DataFrame:
    vio_path = os.path.join(session_dir, "processed_vio.csv")
    odo_path = os.path.join(session_dir, "odometry.csv")
    vio_required = {"device_timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if prefer_server_vio and os.path.isfile(vio_path) and os.path.getsize(vio_path) > 0:
        vio_df = pd.read_csv(vio_path)
        if not vio_df.empty and vio_required.issubset(set(vio_df.columns)):
            vio_cols = list(vio_required)
            if np.all(np.isfinite(vio_df[vio_cols].to_numpy(dtype=np.float64))):
                vio_df.attrs["pose_source"] = "server_vio"
                return vio_df
    if not os.path.isfile(odo_path) or os.path.getsize(odo_path) == 0:
        return pd.DataFrame()
    odo_df = pd.read_csv(odo_path)
    required = {"timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if odo_df.empty or not required.issubset(set(odo_df.columns)):
        return pd.DataFrame()
    from scipy.spatial.transform import Rotation

    ts_ns = np.rint(odo_df["timestamp"].to_numpy(dtype=np.float64) * 1e9).astype(np.int64)
    q_gl = normalize_quaternions_vectorized(
        odo_df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    )
    q_x_180 = np.array([1.0, 0.0, 0.0, 0.0])
    q_cv = (Rotation.from_quat(q_gl) * Rotation.from_quat(q_x_180)).as_quat()
    q_cv = normalize_quaternions_vectorized(q_cv)
    out = pd.DataFrame(
        {
            "device_timestamp_ns": ts_ns,
            "frame": odo_df["frame"].to_numpy(),
            "x": odo_df["x"].to_numpy(dtype=np.float64),
            "y": odo_df["y"].to_numpy(dtype=np.float64),
            "z": odo_df["z"].to_numpy(dtype=np.float64),
            "qx": q_cv[:, 0],
            "qy": q_cv[:, 1],
            "qz": q_cv[:, 2],
            "qw": q_cv[:, 3],
        }
    )
    out.attrs["pose_source"] = "arcore_odometry"
    if apply_island_filter:
        imu_df = None
        for cand in ["phone_imu.csv", "processed_phone_imu.csv"]:
            p = os.path.join(session_dir, cand)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                try:
                    imu_df = pd.read_csv(p)
                    break
                except Exception:
                    pass
        out = select_best_tracking_island(out, imu_df=imu_df)
    return out


def _attach_odometry_frame_indices(keyframes: list[dict], session_dir: str) -> None:
    odo_path = os.path.join(session_dir, "odometry.csv")
    if not keyframes or not os.path.isfile(odo_path) or os.path.getsize(odo_path) == 0:
        return
    odo = pd.read_csv(odo_path)
    if odo.empty or "timestamp" not in odo.columns or "frame" not in odo.columns:
        return
    odo = odo.dropna(subset=["timestamp", "frame"])
    if odo.empty:
        return
    odo_ns = np.rint(odo["timestamp"].to_numpy(dtype=np.float64) * 1e9)
    odo_frames = odo["frame"].to_numpy()
    for kf in keyframes:
        ts = kf.get("device_timestamp_ns")
        if ts is None:
            continue
        diffs = np.abs(odo_ns - int(ts))
        j = int(np.argmin(diffs))
        if float(diffs[j]) <= ODOMETRY_RGB_MATCH_NS:
            kf["frame"] = int(odo_frames[j])


def _run_legacy_reconstruction(session_dir: str) -> None:
    stub_ply = "app/static/stubs/reconstructed.ply"
    stub_glb = "app/static/stubs/reconstructed.glb"

    ply_tmp = os.path.join(session_dir, "reconstructed.ply.tmp")
    ply_final = os.path.join(session_dir, "reconstructed.ply")
    if os.path.exists(stub_ply):
        shutil.copy(stub_ply, ply_tmp)
    else:
        with open(ply_tmp, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    os.replace(ply_tmp, ply_final)

    glb_tmp = os.path.join(session_dir, "reconstructed.glb.tmp")
    glb_final = os.path.join(session_dir, "reconstructed.glb")
    if os.path.exists(stub_glb):
        shutil.copy(stub_glb, glb_tmp)
    else:
        with open(glb_tmp, "wb") as f:
            f.write(b"glTFdummycontent")
    os.replace(glb_tmp, glb_final)


@dataclass
class PipelineMetrics:
    room_area_m2: float = 0.0
    room_perimeter_m: float = 0.0
    room_height_m: float = 0.0
    room_width_m: float = 0.0
    room_depth_m: float = 0.0
    wall_count: int = 0
    portal_count: int = 0
    mesh_points_count: int = 0
    mesh_bbox_span: list[float] = field(default_factory=list)
    trajectory_poses_count: int = 0
    trajectory_path_length_m: float = 0.0
    trajectory_closure_drift_m: float = 0.0
    materials: dict[str, Any] = field(default_factory=dict)
    tatami: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    additional: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        if not isinstance(item, str):
            raise TypeError(f"attribute name must be string, not {type(item).__name__!r}")
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def __contains__(self, item: Any) -> bool:
        if not isinstance(item, str):
            return False
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        if not isinstance(item, str):
            return default
        return getattr(self, item, default)

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]


@dataclass
class PipelineResult:
    success: bool
    session_dir: str
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    artifacts: dict[str, str] = field(default_factory=dict)
    floorplan: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if isinstance(self.metrics, PipelineMetrics):
            data["metrics"] = self.metrics.to_dict()
        return data

    def __getitem__(self, item: str) -> Any:
        if not isinstance(item, str):
            raise TypeError(f"attribute name must be string, not {type(item).__name__!r}")
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def __contains__(self, item: Any) -> bool:
        if not isinstance(item, str):
            return False
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        if not isinstance(item, str):
            return default
        return getattr(self, item, default)

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]


def run_sensor_ingestion(session_dir: str) -> dict[str, Any]:
    """Execute pure TimeSync, clock synchronization, and sensor ingestion on session_dir."""
    manifest_path = os.path.join(session_dir, "manifest.json")
    manifest = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = None
    manifest_dict = manifest if isinstance(manifest, dict) else {}
    flip_h, flip_v, rot_deg = extract_lidar_display_metadata(manifest_dict)
    kinematics = manifest_dict.get("kinematics") if isinstance(manifest_dict.get("kinematics"), dict) else {}
    max_hold_ns = int(kinematics.get("max_hold_ns") or 50_000_000) if kinematics.get("max_hold_ns") else 50_000_000
    imu_scale = resolve_imu_scale(manifest)

    T_imu, T_lidar, extra_meta = load_session_extrinsics(session_dir)
    projection = manifest_dict.get("projection") if isinstance(manifest_dict.get("projection"), dict) else {}
    if "extrinsics_source" in projection:
        extra_meta["extrinsics_source"] = projection["extrinsics_source"]
    if "extrinsics_calibrated" in projection:
        extra_meta["extrinsics_calibrated"] = projection["extrinsics_calibrated"]

    # Parse Hub IMU
    imu_csv = os.path.join(session_dir, "imu.csv")
    if os.path.isfile(imu_csv):
        with open(imu_csv, "rb") as f:
            imu_data = parse_telemetry_csv(
                io.BytesIO(f.read()),
                IMU_MAPPINGS,
                optional_columns=IMU_OPTIONAL,
                imu_scale=imu_scale,
            )
        imu_data = assign_seq_num(imu_data)
        if len(imu_data) > 0 and "mcu_timestamp_ns" in imu_data.columns:
            imu_data["mcu_timestamp_ns"] = reconstruct_mcu_rollover(imu_data["mcu_timestamp_ns"])
            imu_scale_mcu = detect_time_unit(imu_data["mcu_timestamp_ns"], is_mcu=True)
            imu_data["mcu_timestamp_ns"] = imu_data["mcu_timestamp_ns"].astype(np.int64) * int(imu_scale_mcu)
            imu_data = reorder_bounded_dejitter_buffer(imu_data, ts_col="mcu_timestamp_ns", window_ns=100_000_000)
    else:
        imu_data = pd.DataFrame()

    # Parse LiDAR
    lidar_csv = os.path.join(session_dir, "lidar.csv")
    if os.path.isfile(lidar_csv):
        with open(lidar_csv, "rb") as f:
            lidar_data = parse_telemetry_csv(
                io.BytesIO(f.read()),
                LIDAR_MAPPINGS,
                optional_columns=LIDAR_OPTIONAL,
                imu_scale="identity",
            )
        lidar_data = assign_seq_num(lidar_data)
        if len(lidar_data) > 0 and "mcu_timestamp_ns" in lidar_data.columns:
            lidar_data["mcu_timestamp_ns"] = reconstruct_mcu_rollover(lidar_data["mcu_timestamp_ns"])
            lidar_scale_mcu = detect_time_unit(lidar_data["mcu_timestamp_ns"], is_mcu=True)
            lidar_data["mcu_timestamp_ns"] = lidar_data["mcu_timestamp_ns"].astype(np.int64) * int(lidar_scale_mcu)
        if len(lidar_data) > 0 and "device_timestamp_ns" in lidar_data.columns:
            lidar_scale_dev = detect_time_unit(lidar_data["device_timestamp_ns"], is_mcu=False)
            lidar_data["device_timestamp_ns"] = lidar_data["device_timestamp_ns"].astype(np.int64) * int(lidar_scale_dev)
        if len(lidar_data) > 0 and "mcu_timestamp_ns" in lidar_data.columns:
            lidar_data["_orig_idx"] = np.arange(len(lidar_data), dtype=np.int64)
            lidar_data = lidar_data.sort_values(by=["mcu_timestamp_ns", "_orig_idx"], kind="mergesort").reset_index(drop=True)
            lidar_data = lidar_data.drop(columns=["_orig_idx"])
    else:
        lidar_data = pd.DataFrame()

    warnings_list: list[str] = []
    if "device_timestamp_ns" in lidar_data.columns and len(lidar_data) > 1:
        dev_ts = lidar_data["device_timestamp_ns"].to_numpy()
        if np.any(np.diff(dev_ts) <= 0):
            warnings_list.append("LiDAR device timestamps are not strictly increasing after MCU sort.")

    if len(lidar_data) < 2:
        raise ValueError("LiDAR telemetry must contain at least 2 rows.")

    if len(imu_data) >= 2 and "mcu_timestamp_ns" in imu_data.columns:
        if np.any(np.diff(imu_data["mcu_timestamp_ns"].to_numpy()) <= 0):
            warnings_list.append("IMU MCU timestamps are not strictly increasing.")

    lidar_mcu = lidar_data["mcu_timestamp_ns"].to_numpy()
    if np.any(np.diff(lidar_mcu) > 1_000_000_000):
        raise ValueError("LiDAR has temporal gaps exceeding 1000 ms.")

    phone_csv = os.path.join(session_dir, "phone_imu.csv")
    if os.path.isfile(phone_csv):
        phone_data = parse_phone_imu(session_dir)
    else:
        phone_data = pd.DataFrame()

    phone_imu_max_gap_ns = 0
    if len(phone_data) >= 2 and "device_timestamp_ns" in phone_data.columns:
        phone_diffs = np.diff(phone_data["device_timestamp_ns"].to_numpy())
        phone_imu_max_gap_ns = int(np.max(phone_diffs))
        if phone_imu_max_gap_ns > 100_000_000:
            warnings_list.append(f"Phone IMU has temporal gaps exceeding 100 ms (max gap: {phone_imu_max_gap_ns} ns).")

    odo_csv = os.path.join(session_dir, "odometry.csv")
    if os.path.isfile(odo_csv):
        with open(odo_csv, "rb") as f:
            odo_data = parse_odometry_csv(io.BytesIO(f.read()))
    else:
        odo_data = pd.DataFrame()

    vio_csv = os.path.join(session_dir, "vio.csv")
    vio_data = None
    if os.path.isfile(vio_csv) and not os.path.isfile(odo_csv):
        with open(vio_csv, "rb") as f:
            vio_data = parse_telemetry_csv(io.BytesIO(f.read()), VIO_MAPPINGS)
        if len(vio_data) < 2:
            raise ValueError("VIO telemetry file must contain at least 2 rows.")

    primary, hub_role = assign_primary_imu(
        phone_rows=len(phone_data),
        hub_rows=len(imu_data),
        odo_rows=len(odo_data),
    )
    if primary == "hub" and len(imu_data) < 2:
        raise ValueError("Primary IMU telemetry must contain at least 2 rows.")
    if primary == "none" and hub_role == "none" and len(odo_data) < 2:
        raise ValueError("No usable inertial or trajectory stream (phone_imu, imu, odometry).")
    if primary == "none" and hub_role == "none":
        warnings_list.append("Kinematics skipped: no usable inertial stream.")

    # TimeSync Clock Bridge
    clock_bridge = "mcu_identity"
    clock_slope_s = 1.0
    clock_offset_ns = 0
    if "device_timestamp_ns" in lidar_data.columns and len(imu_data) > 0 and "mcu_timestamp_ns" in imu_data.columns:
        hub_and, s, c_rel, bridge = map_hub_android_from_lidar(
            imu_data["mcu_timestamp_ns"].to_numpy(),
            lidar_data["mcu_timestamp_ns"].to_numpy(),
            lidar_data["device_timestamp_ns"].to_numpy(),
        )
        if bridge == "lidar_mcu_android":
            imu_data["device_timestamp_ns"] = hub_and
            clock_bridge = "lidar_mcu_android"
            clock_slope_s = float(s)
            clock_offset_ns = int(c_rel)
        else:
            legacy_col = "legacy_device_timestamp_ns" if "legacy_device_timestamp_ns" in imu_data.columns else ("device_timestamp_ns" if "device_timestamp_ns" in imu_data.columns else None)
            if legacy_col is not None:
                scale_leg = detect_time_unit(imu_data[legacy_col], is_mcu=False)
                imu_data["device_timestamp_ns"] = imu_data[legacy_col].astype(np.int64) * int(scale_leg)
                clock_bridge = "legacy_hub_device"
            else:
                imu_data["device_timestamp_ns"] = imu_data["mcu_timestamp_ns"]
                clock_bridge = "mcu_identity"
    else:
        if len(imu_data) > 0:
            legacy_col = "legacy_device_timestamp_ns" if "legacy_device_timestamp_ns" in imu_data.columns else ("device_timestamp_ns" if "device_timestamp_ns" in imu_data.columns else None)
            if legacy_col is not None:
                scale_leg = detect_time_unit(imu_data[legacy_col], is_mcu=False)
                imu_data["device_timestamp_ns"] = imu_data[legacy_col].astype(np.int64) * int(scale_leg)
                clock_bridge = "legacy_hub_device"
            elif "mcu_timestamp_ns" in imu_data.columns:
                imu_data["device_timestamp_ns"] = imu_data["mcu_timestamp_ns"]
                clock_bridge = "mcu_identity"

    # Overlap verification
    if primary == "phone" and len(phone_data) > 0 and "device_timestamp_ns" in phone_data.columns:
        traj_ts = phone_data["device_timestamp_ns"].to_numpy()
    elif len(odo_data) > 0 and "device_timestamp_ns" in odo_data.columns:
        traj_ts = odo_data["device_timestamp_ns"].to_numpy()
    elif len(imu_data) > 0 and "device_timestamp_ns" in imu_data.columns:
        traj_ts = imu_data["device_timestamp_ns"].to_numpy()
    else:
        raise ValueError("No usable inertial or trajectory stream (phone_imu, imu, odometry).")

    if "device_timestamp_ns" in lidar_data.columns:
        l_dev = lidar_data["device_timestamp_ns"].to_numpy()
        valid_l = np.isfinite(l_dev.astype(np.float64)) & (l_dev > 0)
        if np.count_nonzero(valid_l) >= 2:
            lidar_ts = l_dev[valid_l]
        else:
            lidar_ts = lidar_data["mcu_timestamp_ns"].to_numpy()
    else:
        lidar_ts = lidar_data["mcu_timestamp_ns"].to_numpy()

    t_start = max(int(np.min(traj_ts)), int(np.min(lidar_ts)))
    t_end = min(int(np.max(traj_ts)), int(np.max(lidar_ts)))
    overlap_ns = t_end - t_start
    if overlap_ns < 1_000_000_000:
        raise ValueError("Overlap duration between trajectory clock and LiDAR must be >= 1.0 s.")

    # VIO interpolation if legacy vio_data exists
    if vio_data is not None:
        vio_scale_dev = detect_time_unit(vio_data["device_timestamp_ns"], is_mcu=False)
        vio_data["device_timestamp_ns"] = vio_data["device_timestamp_ns"].astype(np.int64) * int(vio_scale_dev)
        if not vio_data["device_timestamp_ns"].is_monotonic_increasing:
            raise ValueError("VIO device timestamps must be strictly monotonic.")
        vio_data = vio_data.sort_values(by="device_timestamp_ns").reset_index(drop=True)
        if np.any(np.diff(vio_data["device_timestamp_ns"].to_numpy()) > 1000 * 1e6):
            raise ValueError("VIO has temporal gaps exceeding 1000 ms.")
        from app.services.calibration import interpolate_vio_poses

        vio_ts = vio_data["device_timestamp_ns"].to_numpy(dtype=np.int64)
        translations = vio_data[["x", "y", "z"]].to_numpy(dtype=np.float64)
        raw_quats = vio_data[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
        quaternions = normalize_quaternions_vectorized(raw_quats)

        vio_t_min, vio_t_max = vio_ts.min(), vio_ts.max()
        in_bounds = (lidar_data["device_timestamp_ns"] >= vio_t_min) & (lidar_data["device_timestamp_ns"] <= vio_t_max)
        lidar_data = lidar_data[in_bounds].copy()
        if len(lidar_data) < 2:
            raise ValueError("LiDAR telemetry has fewer than 2 frames overlapping with VIO tracking window.")

        target_ts = lidar_data["device_timestamp_ns"].to_numpy()
        x_interp, q_interp = interpolate_vio_poses(vio_ts, translations, quaternions, target_ts)
        processed_vio = pd.DataFrame({
            "device_timestamp_ns": target_ts,
            "x": x_interp[:, 0], "y": x_interp[:, 1], "z": x_interp[:, 2],
            "qx": q_interp[:, 0], "qy": q_interp[:, 1], "qz": q_interp[:, 2], "qw": q_interp[:, 3],
        })
        processed_vio.to_csv(os.path.join(session_dir, "processed_vio.csv"), index=False)

    # Mask LiDAR zones
    lidar_data = mask_lidar_zones(lidar_data)
    dist_cols = [f"distance_{i}" for i in range(64)]
    total_frames = len(lidar_data)
    all_nan_frames = int(lidar_data[dist_cols].isna().all(axis=1).sum())
    all_nan_ratio = float(all_nan_frames / total_frames) if total_frames > 0 else 0.0
    if all_nan_ratio > 0.10:
        raise ValueError("LiDAR quality failure: >10% completely invalid frames.")

    if len(imu_data) > 0 and "mcu_timestamp_ns" in imu_data.columns:
        imu_gap_info = hub_gap_stats(imu_data["mcu_timestamp_ns"].to_numpy(), max_hold_ns)
    else:
        imu_gap_info = {
            "sample_count": 0,
            "gaps_over_max_hold": 0,
            "gaps_over_100ms": 0,
            "max_gap_ns": 0,
            "dropped_intervals": [],
        }

    hub_recv_dup = 0
    if "receive_timestamp_ns" in imu_data.columns and len(imu_data) > 0:
        hub_recv_dup = int(imu_data["receive_timestamp_ns"].duplicated().sum())
        if hub_recv_dup > 0:
            warnings_list.append(f"Hub IMU has {hub_recv_dup} duplicate receive timestamps.")

    cols_imu = ["device_timestamp_ns", "mcu_timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"]
    if "seq_num" in imu_data.columns:
        cols_imu.append("seq_num")
    cols_imu = [c for c in cols_imu if c in imu_data.columns]
    processed_imu = imu_data[cols_imu].copy()
    processed_imu.to_csv(os.path.join(session_dir, "processed_imu.csv"), index=False)

    processed_phone = None
    if len(phone_data) > 0 and "device_timestamp_ns" in phone_data.columns:
        t_phone = phone_data["device_timestamp_ns"].to_numpy(dtype=np.int64)
        if len(t_phone) >= 2:
            phone_dts = np.diff(t_phone)
            phone_jitter = calculate_robust_jitter(phone_dts.astype(np.float64))
            phone_median_dt = float(np.median(phone_dts))
            if phone_median_dt > 0 and phone_jitter > 0.05 * phone_median_dt:
                t0 = t_phone[0]
                t1 = t_phone[-1]
                target_uniform = np.arange(t0, t1, 10_000_000, dtype=np.int64)
                timeline_rel = (t_phone - t0).astype(np.float64)
                target_rel = (target_uniform - t0).astype(np.float64)

                fs = 1e9 / phone_median_dt
                accel_cols = ["ax", "ay", "az"]
                gyro_cols = ["gx", "gy", "gz"]
                accel_data = phone_data[accel_cols].to_numpy(dtype=np.float64)
                gyro_data = phone_data[gyro_cols].to_numpy(dtype=np.float64)
                if fs > 100.0:
                    accel_data = apply_butterworth_lowpass(accel_data, fs, timeline_rel)
                    gyro_data = apply_butterworth_lowpass(gyro_data, fs, timeline_rel)
                resampled_accel = pchip_resample(timeline_rel, accel_data, target_rel)
                resampled_gyro = pchip_resample(timeline_rel, gyro_data, target_rel)

                processed_phone = pd.DataFrame({
                    "device_timestamp_ns": target_uniform,
                    "ax": resampled_accel[:, 0], "ay": resampled_accel[:, 1], "az": resampled_accel[:, 2],
                    "gx": resampled_gyro[:, 0], "gy": resampled_gyro[:, 1], "gz": resampled_gyro[:, 2],
                })
            else:
                phone_cols = ["device_timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"]
                processed_phone = phone_data[[c for c in phone_cols if c in phone_data.columns]].copy()
        else:
            phone_cols = ["device_timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"]
            processed_phone = phone_data[[c for c in phone_cols if c in phone_data.columns]].copy()
        processed_phone.to_csv(os.path.join(session_dir, "processed_phone_imu.csv"), index=False)

    hub_R_diag = {
        "hub_R_accepted": False,
        "hub_R_reason": "hub_not_constraint" if hub_role != "constraint" else "missing_stream",
        "hub_R_n_pairs": 0,
        "hub_R_omega_min_rad_s": float(WAHBA_OMEGA_MIN_RAD_S),
        "hub_R_residual_rms_deg": None,
        "hub_R_sv_ratio": None,
    }
    hub_R_accepted_matrix = None
    if (
        hub_role == "constraint"
        and processed_phone is not None
        and len(processed_phone) > 0
        and len(processed_imu) > 0
        and all(c in processed_phone.columns for c in ("device_timestamp_ns", "gx", "gy", "gz"))
        and all(c in processed_imu.columns for c in ("device_timestamp_ns", "gx", "gy", "gz"))
    ):
        try:
            wahba = estimate_R_phone_hub(
                processed_phone["device_timestamp_ns"].to_numpy(dtype=np.int64),
                processed_phone[["gx", "gy", "gz"]].to_numpy(dtype=np.float64),
                processed_imu["device_timestamp_ns"].to_numpy(dtype=np.int64),
                processed_imu[["gx", "gy", "gz"]].to_numpy(dtype=np.float64),
            )
            hub_R_diag["hub_R_accepted"] = bool(wahba["accepted"])
            hub_R_diag["hub_R_reason"] = wahba["reason"]
            hub_R_diag["hub_R_n_pairs"] = int(wahba["n_pairs"])
            hub_R_diag["hub_R_residual_rms_deg"] = (
                None if wahba["residual_rms_deg"] is None else float(wahba["residual_rms_deg"])
            )
            hub_R_diag["hub_R_sv_ratio"] = (
                None if wahba["sv_ratio"] is None else float(wahba["sv_ratio"])
            )
            if wahba["accepted"] and wahba["R"] is not None:
                hub_R_accepted_matrix = np.asarray(wahba["R"], dtype=np.float64)
        except Exception as exc:
            logger.warning("Hub IMU Wahba estimation failed with %s: %s", type(exc).__name__, exc)

    if len(odo_data) > 0:
        odo_data.to_csv(os.path.join(session_dir, "processed_odometry.csv"), index=False)

    lidar_cols = ["device_timestamp_ns", "mcu_timestamp_ns"]
    lidar_cols += [f"distance_{i}" for i in range(64)]
    lidar_cols += [f"status_{i}" for i in range(64)]
    for opt_c in ["matched_frame", "match_delta_nanos", "seq_num"]:
        if opt_c in lidar_data.columns:
            lidar_cols.append(opt_c)
    lidar_cols = [c for c in lidar_cols if c in lidar_data.columns]
    lidar_data[lidar_cols].to_csv(os.path.join(session_dir, "processed_lidar.csv"), index=False)

    processed_extrinsics = {
        "T_imu_camera": T_imu.tolist(),
        "T_lidar_camera": T_lidar.tolist(),
        "T_imu_camera_meaning": extra_meta.get("T_imu_camera_meaning"),
        "ray_frame": extra_meta.get("ray_frame"),
        "source": extra_meta.get("source"),
        "extrinsics_source": extra_meta.get("extrinsics_source"),
        "extrinsics_calibrated": extra_meta.get("extrinsics_calibrated"),
        "lidar_heatmap_display_flip_h": flip_h,
        "lidar_heatmap_display_flip_v": flip_v,
        "camera_sensor_to_display_rotation_deg": rot_deg,
    }
    if hub_R_accepted_matrix is not None:
        R = hub_R_accepted_matrix
        processed_extrinsics["R_phone_hub"] = R.tolist()
        processed_extrinsics["T_phone_hub"] = [
            [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), 0.0],
            [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), 0.0],
            [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        processed_extrinsics["hub_imu_extrinsics_source"] = "gyro_wahba"
    with open(os.path.join(session_dir, "processed_extrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(processed_extrinsics, f)

    processed_intrinsics = {
        "frame": extra_meta.get("ray_frame"),
        "source": extra_meta.get("source"),
        "rays": extra_meta.get("rays"),
    }
    with open(os.path.join(session_dir, "processed_lidar_intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(processed_intrinsics, f)

    diagnostics = {
        "schema_version": manifest_dict.get("schema_version") or "legacy",
        "contract_revision": manifest_dict.get("contract_revision") or ("p2b-2026-08" if manifest_dict.get("schema_version") == "1.3" else "legacy"),
        "primary_imu": primary,
        "hub_role": hub_role,
        "max_hold_ns": int(max_hold_ns),
        "clock_bridge": clock_bridge,
        "clock_slope_s": float(clock_slope_s),
        "clock_offset_ns": int(clock_offset_ns),
        "hub_imu_sample_count": int(imu_gap_info["sample_count"]),
        "hub_imu_gaps_over_max_hold": int(imu_gap_info["gaps_over_max_hold"]),
        "hub_imu_gaps_over_100ms": int(imu_gap_info["gaps_over_100ms"]),
        "hub_imu_max_gap_ns": int(imu_gap_info["max_gap_ns"]),
        "hub_imu_receive_duplicate_count": int(hub_recv_dup),
        "phone_imu_sample_count": int(len(phone_data)),
        "phone_imu_max_gap_ns": int(phone_imu_max_gap_ns),
        "lidar_frame_count": int(len(lidar_data)),
        "lidar_all_nan_frame_ratio": float(all_nan_ratio),
        "lidar_heatmap_display_flip_h": flip_h,
        "lidar_heatmap_display_flip_v": flip_v,
        "camera_sensor_to_display_rotation_deg": rot_deg,
        "warnings": warnings_list,
    }
    diagnostics.update(hub_R_diag)
    with open(os.path.join(session_dir, "ingestion_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f)

    return diagnostics


def run_3d_reconstruction(
    session_dir: str,
    *,
    enable_server_vio: bool = True,
    enable_tof_pipeline: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Execute pure 3D reconstruction and spatial measurement chain on session_dir."""
    if not enable_tof_pipeline:
        logger.info("ToF pipeline disabled. Running legacy 3D reconstruction.")
        _run_legacy_reconstruction(session_dir)
        return {"status": "reconstructed", "mode": "legacy"}

    manifest_path = os.path.join(session_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.warning("Manifest missing. Falling back to legacy reconstruction.")
        _run_legacy_reconstruction(session_dir)
        return {"status": "reconstructed", "mode": "legacy"}

    k = load_camera_intrinsics(os.path.join(session_dir, "camera_matrix.csv"))

    if enable_server_vio:
        publish_session_vio(session_dir)
        gc.collect()

    estimator = VioEstimator()
    pose_df = load_pose_table_for_tsdf(session_dir, prefer_server_vio=enable_server_vio)
    raw_keyframes = estimator.select_keyframes(pose_df)
    if pose_df.attrs.get("pose_source") != "server_vio":
        _attach_odometry_frame_indices(raw_keyframes, session_dir)

    resolved: list[dict] = []
    for kf in raw_keyframes:
        path = resolve_keyframe_rgb_path(session_dir, kf)
        if path is None:
            continue
        rgb = cv2.imread(path) if cv2 is not None else None
        if rgb is None:
            continue
        item = dict(kf)
        item["_rgb_path"] = path
        resolved.append(item)

    if not resolved and int(pose_df.attrs.get("discarded_frames", 0)) > 0:
        logger.info("island_rgb_fail_open")
        pose_df = load_pose_table_for_tsdf(session_dir, apply_island_filter=False, prefer_server_vio=enable_server_vio)
        raw_keyframes = estimator.select_keyframes(pose_df)
        _attach_odometry_frame_indices(raw_keyframes, session_dir)
        resolved = []
        for kf in raw_keyframes:
            path = resolve_keyframe_rgb_path(session_dir, kf)
            if path is None:
                continue
            rgb = cv2.imread(path) if cv2 is not None else None
            if rgb is None:
                continue
            item = dict(kf)
            item["_rgb_path"] = path
            resolved.append(item)

    keyframes = _decimate_keyframes(resolved, max_limit=150)

    manifest_data = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = None
    m_flip_h, m_flip_v, m_rot_deg = extract_lidar_display_metadata(manifest_data)

    lidar_pack = load_processed_lidar_for_tof(session_dir)
    has_lidar = bool(lidar_pack.get("df") is not None and not lidar_pack["df"].empty)
    flip_h = lidar_pack.get("flip_h", m_flip_h)
    flip_v = lidar_pack.get("flip_v", m_flip_v)
    rot_deg = lidar_pack.get("camera_sensor_to_display_rotation_deg", m_rot_deg)

    depths: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    colors: list[np.ndarray] = []

    from scipy.spatial.transform import Rotation

    gravity_vector = resolve_gravity_vector(pose_df, source=pose_df.attrs.get("pose_source", "unknown"))
    g_world = gravity_vector if is_usable_gravity(gravity_vector) else None

    if keyframes:
        P = np.array([[kf["x"], kf["y"], kf["z"]] for kf in keyframes], dtype=float)
        diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    else:
        diag = 0.0
    z_max_ghost = max(GHOST_RANGE_FLOOR_M, diag + GHOST_TRAJ_MARGIN_M)

    depth_service = DepthCompletionService()

    try:
        em = refine_lidar_camera_extrinsics_em(
            T0=lidar_pack.get("T_lidar_camera"),
            pose_df=pose_df,
            keyframes=keyframes,
            lidar_pack=lidar_pack,
            K=k,
            depth_service=depth_service,
            flip_h=flip_h,
            flip_v=flip_v,
            rot_deg=rot_deg,
            session_dir=session_dir,
            g_world=g_world,
            z_max_ghost=z_max_ghost,
        )
        if lidar_pack.get("T_lidar_camera") is not None:
            lidar_pack["T_lidar_camera"] = em["T_lidar_camera"]
    except Exception:
        logger.exception("EM extrinsic refinement failed for session; keeping T0")

    for idx, kf in enumerate(keyframes):
        if progress_callback:
            progress_callback(idx, len(keyframes))

        rgb = cv2.imread(kf["_rgb_path"]) if cv2 is not None else None
        if rgb is None:
            continue

        sparse_tof = np.zeros(rgb.shape[:2], dtype=np.float32)
        row = lookup_lidar_row(
            lidar_pack.get("df"),
            kf.get("frame"),
            kf.get("device_timestamp_ns"),
            frame_index=lidar_pack.get("frame_index"),
        )
        if row is not None and lidar_pack.get("rays") is not None:
            dist_cols = [f"distance_{i}" for i in range(64)]
            dists = row[dist_cols].to_numpy(dtype=np.float64)
            h, w = rgb.shape[:2]
            sparse_tof = project_lidar_frame_to_sparse_tof(
                dists,
                lidar_pack["rays"],
                lidar_pack["T_lidar_camera"],
                k,
                h,
                w,
                flip_h=flip_h,
                flip_v=flip_v,
                rot_deg=rot_deg,
            )
        if should_skip_tof_keyframe(has_lidar, sparse_tof):
            continue

        R_wc = Rotation.from_quat([kf["qx"], kf["qy"], kf["qz"], kf["qw"]]).as_matrix()
        g_cam = None if g_world is None else (R_wc.T @ np.asarray(g_world, dtype=float))
        dense_depth = depth_service.run_depthor_plus(rgb, sparse_tof, k, g_cam=g_cam, z_max_ghost=z_max_ghost)
        if np.count_nonzero(dense_depth > 0) == 0:
            continue
        depths.append(dense_depth)
        colors.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        pose = np.eye(4)
        pose[0:3, 3] = [kf["x"], kf["y"], kf["z"]]
        pose[0:3, 0:3] = R_wc
        poses.append(pose)

    trajectory = (
        pose_df[["x", "y", "z"]].to_numpy(dtype=float)
        if (pose_df is not None and not pose_df.empty and {"x", "y", "z"}.issubset(pose_df.columns))
        else (
            np.array([p[0:3, 3] for p in poses], dtype=float)
            if poses
            else (
                np.array([[kf["x"], kf["y"], kf["z"]] for kf in keyframes], dtype=float)
                if keyframes
                else None
            )
        )
    )

    reconstruction_service = ReconstructionService()
    mesh = reconstruction_service.integrate_tsdf(depths, poses, k, colors)
    del depths, poses, colors
    if "depth_service" in locals():
        del depth_service
    try:
        from app.services.depth_completion import DepthorModelManager
        DepthorModelManager.release()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("TSDF produced empty mesh.")

    reconstruction_service.export_mesh_artifacts(mesh, session_dir)

    enable_vggt = (os.getenv("ENABLE_VGGT", "1") != "0")
    vggt_prior = None
    if enable_vggt:
        try:
            from app.services.vggt_runner import run_vggt_inference
            force_vggt = (pose_df.attrs.get("pose_source") == "server_vio")
            run_vggt_inference(session_dir, force_recompute=force_vggt)
        except Exception:
            logger.exception("VGGT inference attempt failed; falling back to standard reconstruction")

        vggt_result = load_vggt_prior(session_dir)
        vggt_prior = vggt_result.prior
    else:
        from app.services.vggt_prior import VggtPriorResult
        vggt_result = VggtPriorResult(prior=None, skip_reason="disabled_by_config")

    try:
        layout, whiteflat_pcd = reconstruction_service.segment_planes(
            mesh,
            gravity_vector,
            trajectory=trajectory,
            vggt_prior=vggt_result.prior,
            session_dir=session_dir,
        )
    except TypeError as err:
        if "trajectory" in str(err) or "session_dir" in str(err):
            layout, whiteflat_pcd = reconstruction_service.segment_planes(mesh, gravity_vector)
        else:
            raise

    attach_vggt_diagnostics(layout, vggt_result)
    reconstruction_service.export_whiteflat_ply(whiteflat_pcd, session_dir)
    reconstruction_service.export_visual_artifacts(
        mesh, layout, gravity_vector, session_dir, vggt_prior=vggt_result.prior
    )

    export_room_model_glb(layout, session_dir)
    try:
        export_room_model_texture_glb(session_dir, layout=layout)
    except Exception:
        logger.exception("Failed to export room_model_texture.glb")

    export_artifacts(layout, session_dir)

    return {
        "status": "reconstructed",
        "layout": layout,
        "mesh": mesh,
        "pose_df": pose_df,
        "keyframes_count": len(keyframes),
    }


def run_material_estimation(
    session_dir: str,
    params: dict[str, Any] | None = None,
    *,
    allow_partial: bool = True,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Execute floorplan bundle and material estimation on session_dir."""
    floorplan_path = os.path.join(session_dir, "floorplan.json")
    if not os.path.isfile(floorplan_path):
        raise FileNotFoundError(f"Floorplan not found: {floorplan_path}")

    with open(floorplan_path, "r", encoding="utf-8") as f:
        floorplan_data = json.load(f)

    active_params = dict(DEFAULT_RECALCULATE_PARAMS)
    if params:
        active_params.update(params)

    return export_session_bundle(
        floorplan_data,
        session_dir,
        active_params,
        render_cad=True,
        render_materials="all",
        allow_partial=allow_partial,
        session_id=session_id or os.path.basename(session_dir),
    )


def extract_pipeline_metrics(
    session_dir: str,
    layout: dict | None = None,
    pose_df: pd.DataFrame | None = None,
    mat_res: dict | None = None,
) -> PipelineMetrics:
    """Extract metrics from floorplan, mesh, trajectory, and materials."""
    metrics = PipelineMetrics()

    # Floorplan metrics
    fp_path = os.path.join(session_dir, "floorplan.json")
    if os.path.isfile(fp_path):
        try:
            with open(fp_path, "r", encoding="utf-8") as f:
                fp_data = json.load(f)
            room = fp_data.get("room", {})
            metrics.room_area_m2 = float(room.get("area_m2", 0.0))
            metrics.room_perimeter_m = float(room.get("perimeter_m", 0.0))
            metrics.room_height_m = float(room.get("height_meters", 0.0))
            metrics.room_width_m = float(room.get("width_m", 0.0))
            metrics.room_depth_m = float(room.get("depth_m", 0.0))
            metrics.wall_count = len(fp_data.get("walls", []))
            metrics.portal_count = len(fp_data.get("portals", []))
        except Exception as e:
            logger.warning("Error reading floorplan.json metrics: %s", e)

    # Mesh metrics
    ply_path = os.path.join(session_dir, "reconstructed.ply")
    if os.path.isfile(ply_path):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(ply_path)
            pts = np.asarray(pcd.points)
            if len(pts) > 0:
                metrics.mesh_points_count = len(pts)
                span = pts.max(axis=0) - pts.min(axis=0)
                metrics.mesh_bbox_span = [round(float(v), 3) for v in span]
        except Exception:
            pass

    # Trajectory metrics
    if pose_df is None or pose_df.empty:
        try:
            pose_df = load_pose_table_for_tsdf(session_dir)
        except Exception:
            pose_df = None

    if pose_df is not None and not pose_df.empty and {"x", "y", "z"}.issubset(pose_df.columns):
        metrics.trajectory_poses_count = len(pose_df)
        xyz = pose_df[["x", "y", "z"]].to_numpy(dtype=float)
        diffs = np.diff(xyz, axis=0)
        metrics.trajectory_path_length_m = round(float(np.sum(np.linalg.norm(diffs, axis=1))), 3)
        metrics.trajectory_closure_drift_m = round(float(np.linalg.norm(xyz[-1] - xyz[0])), 3)

    # Materials / Bundle metrics from mat_res dict if passed
    if mat_res and isinstance(mat_res, dict):
        if "tatami" in mat_res and mat_res["tatami"]:
            metrics.tatami = mat_res["tatami"]
        for k in ["wallpaper", "neda", "tiling", "plywood", "cf", "combined"]:
            if k in mat_res and mat_res[k]:
                metrics.materials[k] = mat_res[k]

    # Overlay / load from disk
    mat_path = os.path.join(session_dir, "materials.json")
    if os.path.isfile(mat_path):
        try:
            with open(mat_path, "r", encoding="utf-8") as f:
                mat_data = json.load(f)
            if isinstance(mat_data, dict):
                metrics.materials.update(mat_data)
        except Exception:
            pass

    tatami_path = os.path.join(session_dir, "tatami_layout.json")
    if os.path.isfile(tatami_path):
        try:
            with open(tatami_path, "r", encoding="utf-8") as f:
                tat_data = json.load(f)
            if isinstance(tat_data, dict):
                metrics.tatami = tat_data
        except Exception:
            pass

    return metrics


def collect_pipeline_artifacts(session_dir: str) -> dict[str, str]:
    """Scan session_dir and return dictionary mapping artifact names to absolute paths."""
    artifacts: dict[str, str] = {}
    known_filenames = {
        "reconstructed.ply",
        "reconstructed.glb",
        "room_model.glb",
        "room_model_texture.glb",
        "reconstructed_visual.ply",
        "whiteflat.ply",
        "floorplan.json",
        "floorplan.svg",
        "floorplan.dxf",
        "materials.json",
        "tatami_layout.json",
        "session_bundle.zip",
        "ingestion_diagnostics.json",
        "processed_lidar.csv",
        "processed_imu.csv",
        "processed_phone_imu.csv",
        "processed_odometry.csv",
        "processed_vio.csv",
        "processed_extrinsics.json",
        "processed_lidar_intrinsics.json",
    }
    valid_extensions = (".ply", ".glb", ".svg", ".dxf", ".json", ".zip", ".png", ".pdf")
    if os.path.isdir(session_dir):
        for item in os.listdir(session_dir):
            p = os.path.join(session_dir, item)
            if os.path.isfile(p):
                if item in known_filenames or item.endswith(valid_extensions):
                    artifacts[item] = os.path.abspath(p)
    return artifacts


class PipelineRunner:
    """Standalone Algorithm Pipeline Engine Runner.

    Coordinates the pure sequential processing:
    TimeSync -> DELTAR EM Calibration -> VIO -> Neural Depth Completion ONNX ->
    TSDF Fusion -> VGGT Prior -> Plane Segmentation -> Manhattan Dual-Stage & Occupancy Layout ->
    Room Metrics Extraction -> CAD Floorplan & Wall Elevations rendering & 3D GLB export ->
    Material & Bundle Estimation.
    """

    def __init__(
        self,
        *,
        enable_server_vio: bool = True,
        enable_tof_pipeline: bool = True,
        recalculate_params: dict[str, Any] | None = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force_ingest: bool = False,
    ):
        self.enable_server_vio = enable_server_vio
        self.enable_tof_pipeline = enable_tof_pipeline
        self.recalculate_params = recalculate_params or dict(DEFAULT_RECALCULATE_PARAMS)
        self.progress_callback = progress_callback
        self.force_ingest = force_ingest

    def run_timesync(self, session_dir: str) -> dict[str, Any]:
        return run_sensor_ingestion(session_dir)

    def run_reconstruction(
        self,
        session_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, Any]:
        return run_3d_reconstruction(
            session_dir,
            enable_server_vio=self.enable_server_vio,
            enable_tof_pipeline=self.enable_tof_pipeline,
            progress_callback=progress_callback or self.progress_callback,
        )

    def run_materials(
        self,
        session_dir: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return run_material_estimation(
            session_dir,
            params=params or self.recalculate_params,
        )

    def run(self, session_dir: str) -> PipelineResult:
        """Run the complete end-to-end algorithmic pipeline synchronously."""
        session_dir = os.path.abspath(session_dir)
        t_start = time.time()
        timings: dict[str, float] = {}

        try:
            # 1. TimeSync / Sensor Ingestion (if raw imu/lidar exists and not already processed)
            t0 = time.time()
            already_ingested = os.path.isfile(os.path.join(session_dir, "processed_lidar.csv"))
            has_raw_lidar = os.path.isfile(os.path.join(session_dir, "lidar.csv"))
            needs_ingest = self.force_ingest or (has_raw_lidar and not already_ingested)
            diag: dict[str, Any] = {}
            if needs_ingest:
                logger.info("PipelineRunner: Executing TimeSync & Ingestion for %s", session_dir)
                diag = self.run_timesync(session_dir)
            elif os.path.isfile(os.path.join(session_dir, "ingestion_diagnostics.json")):
                try:
                    with open(os.path.join(session_dir, "ingestion_diagnostics.json"), "r", encoding="utf-8") as f:
                        diag = json.load(f)
                except Exception:
                    diag = {}
            timings["ingestion_sec"] = round(time.time() - t0, 3)

            # 2. 3D Reconstruction Chain
            t1 = time.time()
            logger.info("PipelineRunner: Executing 3D Reconstruction & Floorplan for %s", session_dir)
            recon_res = self.run_reconstruction(session_dir, progress_callback=self.progress_callback)
            timings["reconstruction_sec"] = round(time.time() - t1, 3)

            # 3. Material Estimation & Bundle Export
            t2 = time.time()
            logger.info("PipelineRunner: Executing Material & Tatami Estimation for %s", session_dir)
            mat_res: dict[str, Any] = {}
            if os.path.isfile(os.path.join(session_dir, "floorplan.json")):
                try:
                    mat_res = self.run_materials(session_dir)
                except Exception as exc:
                    logger.warning("PipelineRunner: Material estimation warning for %s: %s", session_dir, exc)
                    diag["material_estimation_warning"] = str(exc)
            timings["materials_sec"] = round(time.time() - t2, 3)
            timings["total_sec"] = round(time.time() - t_start, 3)

            # 4. Extract Metrics & Collect Artifacts
            metrics = extract_pipeline_metrics(
                session_dir,
                layout=recon_res.get("layout"),
                pose_df=recon_res.get("pose_df"),
                mat_res=mat_res,
            )
            metrics.timings = timings

            artifacts = collect_pipeline_artifacts(session_dir)

            floorplan_data = {}
            fp_path = os.path.join(session_dir, "floorplan.json")
            if os.path.isfile(fp_path):
                with open(fp_path, "r", encoding="utf-8") as f:
                    floorplan_data = json.load(f)

            return PipelineResult(
                success=True,
                session_dir=session_dir,
                metrics=metrics,
                artifacts=artifacts,
                floorplan=floorplan_data,
                diagnostics=diag,
                error_message=None,
            )
        except Exception as exc:
            logger.exception("PipelineRunner execution failed for %s: %s", session_dir, exc)
            timings["total_sec"] = round(time.time() - t_start, 3)
            return PipelineResult(
                success=False,
                session_dir=session_dir,
                metrics=PipelineMetrics(timings=timings),
                artifacts=collect_pipeline_artifacts(session_dir),
                error_message=str(exc),
            )


def run_pipeline(session_dir: str, **kwargs) -> PipelineResult:
    """Convenience functional entry point for PipelineRunner."""
    runner = PipelineRunner(**kwargs)
    return runner.run(session_dir)
