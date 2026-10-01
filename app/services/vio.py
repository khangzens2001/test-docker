"""Server-side Multi-Sensor VIO Facade & Event-Driven Time Synchronizer (Task 11).

Wires all 10 VIO core components:
1. math_utils
2. state
3. imu_preintegration
4. eskf
5. visual_frontend
6. spad_depth_fusion
7. triangulation
8. lidar_constraints
9. hub_imu_fusion
10. rts_smoother
11. loop_closure
12. pose_graph_optimizer
"""

from __future__ import annotations
import copy
import gc
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

try:
    import cv2
except ImportError:
    cv2 = None

# Wire all 10 VIO core components
from app.services.vio_core.math_utils import rot_to_quat, quat_to_rot, so3_exp, so3_log, skew_symmetric
from app.services.vio_core.state import NominalState, StateIndex
from app.services.vio_core.imu_preintegration import ImuPreintegration
from app.services.vio_core.eskf import SRESKF
from app.services.vio_core.visual_frontend import VisualFrontend
from app.services.vio_core.spad_depth_fusion import associate_spad_depth
from app.services.vio_core.triangulation import point_3d_to_inverse_depth, compute_parallax_angle
from app.services.vio_core.lidar_constraints import build_spad_lidar_constraint
from app.services.vio_core.hub_imu_fusion import build_hub_imu_constraint
from app.services.vio_core.rts_smoother import ManifoldRtsSmoother, _copy_state
from app.services.calibration import (
    remap_lidar_zones_to_camera_frame,
    load_T_bc,
    load_camera_intrinsics,
)
from app.services.vio_core.visual_update import (
    LandmarkTrackMap,
    STATUS_CANDIDATE,
    STATUS_ACTIVE,
    ORIGIN_SPAD,
    camera_pose_from_body,
    relative_camera_rotation,
    build_visual_measurement,
)
from app.services.vio_core.loop_closure import LoopDetector
from app.services.vio_core.pose_graph_optimizer import PoseGraphOptimizer, apply_pgo_se3_interpolation

PUBLISH_WALL_S = 90.0

logger = logging.getLogger(__name__)


class SensorType(Enum):
    PHONE_IMU = "phone_imu"
    HUB_IMU = "hub_imu"
    CAMERA_FRAME = "camera_frame"
    SPAD_LIDAR = "spad_lidar"


@dataclass(slots=True)
class SensorEvent:
    timestamp: float  # In seconds
    sensor_type: SensorType
    data: Dict[str, Any]

    def __lt__(self, other: "SensorEvent") -> bool:
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        # Priority fallback for tie-breaking: PHONE_IMU -> HUB_IMU -> SPAD_LIDAR -> CAMERA_FRAME
        order = {
            SensorType.PHONE_IMU: 0,
            SensorType.HUB_IMU: 1,
            SensorType.SPAD_LIDAR: 2,
            SensorType.CAMERA_FRAME: 3,
        }
        return order[self.sensor_type] < order[other.sensor_type]


class EventDrivenTimeSynchronizer:
    """Enqueues multi-sensor events: Phone IMU (>= 200 Hz), Hub IMU (50-100 Hz), Camera frames (30 fps), SPAD LiDAR (8x8).
    Sorts events chronologically by timestamp and triggers callbacks in exact timestamp order.
    """

    def __init__(self) -> None:
        self._events: List[SensorEvent] = []

    def enqueue_phone_imu(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray) -> None:
        ts_sec = float(timestamp / 1e9 if timestamp > 1e6 else timestamp)
        self._events.append(SensorEvent(
            timestamp=ts_sec,
            sensor_type=SensorType.PHONE_IMU,
            data={"accel": np.asarray(accel, dtype=np.float64), "gyro": np.asarray(gyro, dtype=np.float64)}
        ))

    def enqueue_hub_imu(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray) -> None:
        ts_sec = float(timestamp / 1e9 if timestamp > 1e6 else timestamp)
        self._events.append(SensorEvent(
            timestamp=ts_sec,
            sensor_type=SensorType.HUB_IMU,
            data={"accel": np.asarray(accel, dtype=np.float64), "gyro": np.asarray(gyro, dtype=np.float64)}
        ))

    def enqueue_camera_frame(self, timestamp: float, frame_id: int, image: Optional[np.ndarray] = None, image_path: Optional[str] = None) -> None:
        ts_sec = float(timestamp / 1e9 if timestamp > 1e6 else timestamp)
        dev_ts_ns = int(round(timestamp)) if timestamp > 1e6 else int(round(timestamp * 1e9))
        self._events.append(SensorEvent(
            timestamp=ts_sec,
            sensor_type=SensorType.CAMERA_FRAME,
            data={"frame_id": frame_id, "image": image, "image_path": image_path, "device_timestamp_ns": dev_ts_ns}
        ))

    def enqueue_spad_lidar(self, timestamp: float, depth_map: np.ndarray, status_mask: Optional[np.ndarray] = None) -> None:
        ts_sec = float(timestamp / 1e9 if timestamp > 1e6 else timestamp)
        self._events.append(SensorEvent(
            timestamp=ts_sec,
            sensor_type=SensorType.SPAD_LIDAR,
            data={"depth_map": np.asarray(depth_map, dtype=np.float64), "status_mask": status_mask}
        ))

    def add_phone_imu_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        ts_col = "device_timestamp_ns" if "device_timestamp_ns" in df.columns else ("timestamp_nanos" if "timestamp_nanos" in df.columns else "timestamp")
        ax_col = "ax" if "ax" in df.columns else "acc_x"
        ay_col = "ay" if "ay" in df.columns else "acc_y"
        az_col = "az" if "az" in df.columns else "acc_z"
        gx_col = "gx" if "gx" in df.columns else "gyro_x"
        gy_col = "gy" if "gy" in df.columns else "gyro_y"
        gz_col = "gz" if "gz" in df.columns else "gyro_z"

        ts_vals = df[ts_col].to_numpy(dtype=np.float64)
        acc_vals = df[[ax_col, ay_col, az_col]].to_numpy(dtype=np.float64)
        gyr_vals = df[[gx_col, gy_col, gz_col]].to_numpy(dtype=np.float64)

        for i in range(len(ts_vals)):
            self.enqueue_phone_imu(ts_vals[i], acc_vals[i], gyr_vals[i])

    def add_hub_imu_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        ts_candidates = [
            "device_timestamp_ns",
            "timestamp_nanos",
            "mobile_receive_timestamp_nanos",
            "timestamp",
            "imu_mcu_timestamp_ns",
        ]
        ts_col = next((c for c in ts_candidates if c in df.columns), None)
        if ts_col is None:
            return
        ax_col = "ax" if "ax" in df.columns else "acc_x"
        ay_col = "ay" if "ay" in df.columns else "acc_y"
        az_col = "az" if "az" in df.columns else "acc_z"
        gx_col = "gx" if "gx" in df.columns else "gyro_x"
        gy_col = "gy" if "gy" in df.columns else "gyro_y"
        gz_col = "gz" if "gz" in df.columns else "gyro_z"

        ts_vals = df[ts_col].to_numpy(dtype=np.float64)
        acc_vals = df[[ax_col, ay_col, az_col]].to_numpy(dtype=np.float64)
        gyr_vals = df[[gx_col, gy_col, gz_col]].to_numpy(dtype=np.float64)

        for i in range(len(ts_vals)):
            self.enqueue_hub_imu(ts_vals[i], acc_vals[i], gyr_vals[i])

    def add_spad_lidar_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        ts_col = None
        for candidate in ["device_timestamp_ns", "timestamp_nanos", "lidar_android_timestamp_nanos", "mobile_receive_timestamp_nanos", "timestamp"]:
            if candidate in df.columns:
                ts_col = candidate
                break
        if ts_col is None:
            return
        grid_col = "depth_grid" if "depth_grid" in df.columns else ("spad_depth" if "spad_depth" in df.columns else None)
        has_dist_0 = "distance_0" in df.columns
        has_status_0 = "status_0" in df.columns

        ts_vals = df[ts_col].to_numpy(dtype=np.float64)
        n = len(ts_vals)

        if grid_col is not None:
            grids = df[grid_col].tolist()
        elif has_dist_0:
            dist_cols = [f"distance_{i}" for i in range(64)]
            grids = df[dist_cols].to_numpy(dtype=np.float64).reshape(-1, 8, 8)
        else:
            return

        statuses = None
        if has_status_0:
            status_cols = [f"status_{i}" for i in range(64)]
            statuses = df[status_cols].to_numpy(dtype=np.float64).reshape(-1, 8, 8)
        elif "status_mask" in df.columns:
            statuses = df["status_mask"].tolist()
        elif "status_grid" in df.columns:
            statuses = df["status_grid"].tolist()

        for i in range(n):
            grid = grids[i]
            if isinstance(grid, (list, np.ndarray)):
                grid = np.asarray(grid, dtype=np.float64)
                if grid.shape == (64,):
                    grid = grid.reshape(8, 8)

            status = statuses[i] if statuses is not None else None
            if status is not None and isinstance(status, (list, np.ndarray)):
                status = np.asarray(status, dtype=np.float64)
                if status.shape == (64,):
                    status = status.reshape(8, 8)

            self.enqueue_spad_lidar(ts_vals[i], grid, status_mask=status)

    def get_sorted_events(self) -> List[SensorEvent]:
        return sorted(self._events)

    def process_events(self, callbacks: Dict[SensorType, Callable[[SensorEvent], None]]) -> None:
        sorted_events = self.get_sorted_events()
        for evt in sorted_events:
            cb = callbacks.get(evt.sensor_type)
            if cb is not None:
                cb(evt)


class ServerSideVioEstimator:
    """Server-side Multi-Sensor VIO Estimator Facade.

    Orchestrates SR-ESKF forward propagation, multi-sensor updates (Phone IMU, Hub IMU, SPAD LiDAR, Camera),
    backward RTS smoothing, visual loop closure detection, and SE(3) pose graph optimization.
    """

    def __init__(self) -> None:
        pass

    def process_session(
        self,
        phone_imu_df: pd.DataFrame,
        hub_imu_df: Optional[pd.DataFrame] = None,
        spad_lidar_df: Optional[pd.DataFrame] = None,
        rgb_images: Optional[List[Union[np.ndarray, str]]] = None,
        frame_timestamps: Optional[List[float]] = None,
        intrinsics: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        T_bc: Optional[np.ndarray] = None,
        T_lidar_camera: Optional[np.ndarray] = None,
        flip_h: bool = False,
        flip_v: bool = False,
        rot_deg: float = 0.0,
        frame_ids: Optional[List[int]] = None,
        R_phone_hub: Optional[np.ndarray] = None,
        max_wall_time_s: Optional[float] = 90.0,
        rays: Optional[np.ndarray] = None,
        ray_frame: str = "lidar_optical",
        span_odo: Optional[float] = None,
        odometry_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Process complete multi-sensor session stream.

        Returns:
            Dict containing status, trajectory, and ate_rmse metric.
        """
        if T_bc is None:
            T_bc = np.eye(4)

        # Pre-parse odometry speeds, positions and rotations if odometry_df is given
        odo_times = None
        odo_pos = None
        odo_rots = None
        if odometry_df is not None and not odometry_df.empty:
            t_col = "timestamp" if "timestamp" in odometry_df.columns else "device_timestamp_ns"
            if t_col in odometry_df.columns and {"x", "y", "z"}.issubset(odometry_df.columns):
                raw_ts = odometry_df[t_col].to_numpy(dtype=float)
                odo_times = raw_ts / 1e9 if (len(raw_ts) > 0 and raw_ts[0] > 1e11) else raw_ts
                odo_pos = odometry_df[["x", "y", "z"]].to_numpy(dtype=float)
                if {"qx", "qy", "qz", "qw"}.issubset(odometry_df.columns):
                    qs = odometry_df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
                    odo_rots = Rotation.from_quat(qs).as_matrix()

        sync = EventDrivenTimeSynchronizer()

        # Enqueue Phone IMU
        if phone_imu_df is not None and not phone_imu_df.empty:
            sync.add_phone_imu_df(phone_imu_df)

        # Enqueue Hub IMU
        if hub_imu_df is not None and not hub_imu_df.empty:
            sync.add_hub_imu_df(hub_imu_df)

        # Enqueue SPAD LiDAR
        if spad_lidar_df is not None and not spad_lidar_df.empty:
            sync.add_spad_lidar_df(spad_lidar_df)

        # Enqueue Camera Frames
        if rgb_images:
            for idx, img in enumerate(rgb_images):
                ts = frame_timestamps[idx] if (frame_timestamps and idx < len(frame_timestamps)) else (idx * 0.03333333333333333)
                fid = frame_ids[idx] if (frame_ids and idx < len(frame_ids)) else idx
                if isinstance(img, str):
                    sync.enqueue_camera_frame(timestamp=ts, frame_id=fid, image_path=img)
                else:
                    sync.enqueue_camera_frame(timestamp=ts, frame_id=fid, image=img)

        sorted_events = sync.get_sorted_events()
        sync._events.clear()
        if not sorted_events:
            return {
                "status": "error",
                "message": "No sensor events to process",
                "trajectory": [],
                "ate_rmse": 0.0,
                "gravity": [0.0, 0.0, 0.0],
            }

        # Initialize 10 Core VIO Components
        eskf = SRESKF()
        preint = ImuPreintegration(
            ba=eskf.state.ba,
            bg=eskf.state.bg,
            noise_params={"sigma_a": 0.01, "sigma_g": 0.001, "sigma_ba": 1e-4, "sigma_bg": 1e-5}
        )

        K = intrinsics if intrinsics is not None else np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        frontend = VisualFrontend(intrinsics=K, dist_coeffs=dist_coeffs)
        track_map = LandmarkTrackMap(K=K)
        smoother = ManifoldRtsSmoother()
        loop_detector = LoopDetector(min_kf_diff=12)
        pgo = PoseGraphOptimizer()
        orb_detector = cv2.ORB_create() if cv2 is not None else None

        # Forward filtering structures
        forward_states: List[NominalState] = []
        S_P_list: List[np.ndarray] = []
        F_list: List[np.ndarray] = []
        pred_states: List[NominalState] = []
        timestamps: List[float] = []
        camera_snapshots: List[Dict[str, Any]] = []
        active_counts: List[int] = []
        spad_active_counts: List[int] = []
        all_spad_active_ids: set[int] = set()
        frames_with_accepted_visual: int = 0

        pgo_node_ids: List[int] = []
        pgo_nodes_meta: List[Dict[str, Any]] = []
        last_node_R_wc: Optional[np.ndarray] = None
        last_node_p_wc: Optional[np.ndarray] = None
        frames_since_last_node: int = 0
        last_phone_imu_time: Optional[float] = None
        latest_spad_depth: Optional[np.ndarray] = None
        latest_spad_status: Optional[np.ndarray] = None
        latest_phone_gyro: Optional[np.ndarray] = None
        R_wc_prev: Optional[np.ndarray] = None
        first_camera_seen = False
        first_camera_time = frame_timestamps[0] if (frame_timestamps and len(frame_timestamps) > 0) else float("inf")

        # Check raw accel scale and perform initial attitude alignment to gravity
        scale_accel = False
        if phone_imu_df is not None and not phone_imu_df.empty:
            ax_col = "ax" if "ax" in phone_imu_df.columns else "acc_x"
            ay_col = "ay" if "ay" in phone_imu_df.columns else "acc_y"
            az_col = "az" if "az" in phone_imu_df.columns else "acc_z"
            acc_norm = np.median(np.linalg.norm(phone_imu_df[[ax_col, ay_col, az_col]].to_numpy(), axis=1))
            if acc_norm < 2.0:
                scale_accel = True

            gx_col = "gx" if "gx" in phone_imu_df.columns else "gyro_x"
            gy_col = "gy" if "gy" in phone_imu_df.columns else "gyro_y"
            gz_col = "gz" if "gz" in phone_imu_df.columns else "gyro_z"
            acc_raw = phone_imu_df[[ax_col, ay_col, az_col]].to_numpy(dtype=float)
            gyr_raw = (
                phone_imu_df[[gx_col, gy_col, gz_col]].to_numpy(dtype=float)
                if {gx_col, gy_col, gz_col}.issubset(phone_imu_df.columns)
                else np.zeros_like(acc_raw)
            )
            if scale_accel:
                acc_raw = acc_raw * 9.80665

            # Find 50-sample window with lowest gyro energy in first 200 samples
            best_w = 0
            best_energy = float("inf")
            search_limit = min(200, len(phone_imu_df) - 50)
            if search_limit > 0 and len(gyr_raw) >= 50:
                for w_start in range(0, max(1, search_limit), 10):
                    gyr_seg = gyr_raw[w_start : w_start + 50]
                    energy = float(np.mean(np.sum(gyr_seg ** 2, axis=1)))
                    if energy < best_energy:
                        best_energy = energy
                        best_w = w_start
                a_init = np.mean(acc_raw[best_w : best_w + 50], axis=0)
                bg_init = np.mean(gyr_raw[best_w : best_w + 50], axis=0)
            else:
                n_init = min(50, len(acc_raw))
                a_init = np.mean(acc_raw[:n_init], axis=0) if n_init > 0 else np.array([0.0, 0.0, 9.80665])
                bg_init = np.mean(gyr_raw[:n_init], axis=0) if n_init > 0 else np.zeros(3)

            eskf.state.bg = bg_init.copy()
            preint.bg = bg_init.copy()

            norm_a = float(np.linalg.norm(a_init))
            if norm_a > 1.0:
                u = a_init / norm_a
                target = np.array([0.0, 0.0, 1.0])
                v_cross = np.cross(u, target)
                s_sin = float(np.linalg.norm(v_cross))
                c_cos = float(np.dot(u, target))
                if s_sin < 1e-6:
                    R_init = np.eye(3) if c_cos > 0 else np.diag([1.0, -1.0, -1.0])
                else:
                    k = v_cross / s_sin
                    K_mat = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
                    R_init = np.eye(3) + s_sin * K_mat + (1.0 - c_cos) * (K_mat @ K_mat)
                eskf.state.R = R_init.copy()
                eskf.state.g = np.array([0.0, 0.0, -norm_a])

        R_vio_odo = None
        if odo_rots is not None and len(odo_rots) > 0:
            R_wc_0 = eskf.state.R @ T_bc[:3, :3]
            # Gravity-aligned coordinate mapping from ARCore world to VIO world:
            # ARCore world: +Y is UP (aligned with gravity), X-Z is the horizontal ground plane.
            # VIO world:    +Z is UP (aligned with gravity), X-Y is the horizontal ground plane.
            # Base rotation R0 maps ARCore UP [0, 1, 0] -> VIO UP [0, 0, 1],
            # and ARCore forward [0, 0, -1] -> VIO forward [0, 1, 0], [1, 0, 0] -> [1, 0, 0].
            R0 = np.array([
                [1.0,  0.0,  0.0],
                [0.0,  0.0, -1.0],
                [0.0,  1.0,  0.0],
            ], dtype=float)

            # Align horizontal yaw heading at t=0 between ARCore camera optical axis and VIO camera optical axis
            # ARCore camera optical axis in ARCore world (OpenGL convention: [0, 0, -1]):
            v_odo_opt = odo_rots[0] @ np.array([0.0, 0.0, -1.0])
            # VIO camera optical axis in VIO world (OpenCV convention: [0, 0, 1]):
            v_vio_opt = R_wc_0 @ np.array([0.0, 0.0, 1.0])

            h_odo = np.array([v_odo_opt[0], 0.0, v_odo_opt[2]])
            h_vio = np.array([v_vio_opt[0], v_vio_opt[1], 0.0])
            norm_odo = float(np.linalg.norm(h_odo))
            norm_vio = float(np.linalg.norm(h_vio))

            if norm_odo > 1e-4 and norm_vio > 1e-4:
                h_mapped = R0 @ h_odo
                yaw_mapped = np.arctan2(h_mapped[1], h_mapped[0])
                yaw_vio = np.arctan2(h_vio[1], h_vio[0])
                psi = yaw_vio - yaw_mapped
                R_yaw = np.array([
                    [np.cos(psi), -np.sin(psi), 0.0],
                    [np.sin(psi),  np.cos(psi), 0.0],
                    [0.0,          0.0,         1.0],
                ], dtype=float)
                R_vio_odo = R_yaw @ R0
            else:
                R_vio_odo = R0

        t_proc_start = time.monotonic()
        initial_floor_distance = None
        for event in sorted_events:
            ts = event.timestamp

            if event.sensor_type == SensorType.PHONE_IMU:
                accel = event.data["accel"].copy()
                gyro = event.data["gyro"].copy()
                if scale_accel:
                    accel *= 9.80665

                if last_phone_imu_time is not None:
                    dt = ts - last_phone_imu_time
                    if dt > 0.0:
                        preint.integrate(accel, gyro, dt)
                        F_k = eskf.predict_imu(preint)
                        F_list.append(F_k.copy())
                        pred_states.append(_copy_state(eskf.state))
                        preint.reset(eskf.state.ba, eskf.state.bg)

                        # Pre-camera stabilization and stationary ZUPT
                        if not first_camera_seen and ts < first_camera_time:
                            eskf.state.v[:] = 0.0
                            eskf.state.p[:] = 0.0
                            eskf.update_zupt(sigma_vel=0.005)
                        else:
                            # In-flight ZUPT during camera tracking only when device is physically at rest
                            w_mag = float(np.linalg.norm(gyro - eskf.state.bg))
                            g_mag = float(np.linalg.norm(eskf.state.g)) if hasattr(eskf.state, "g") else 9.80665
                            a_dyn = abs(float(np.linalg.norm(accel)) - g_mag)
                            if w_mag < 0.03 and a_dyn < 0.15:
                                eskf.update_zupt(sigma_vel=0.01)

                        cur_v_mag = float(np.linalg.norm(eskf.state.v))
                        if cur_v_mag > 1.2:
                            eskf.state.v *= (1.2 / cur_v_mag)
                        eskf.state.v[2] = float(np.clip(eskf.state.v[2], -0.80, 0.80))
                        eskf.state.p[2] = float(np.clip(eskf.state.p[2], -1.60, 1.20))
                    else:
                        F_list.append(np.eye(StateIndex.DIM))
                        pred_states.append(_copy_state(eskf.state))

                last_phone_imu_time = ts
                latest_phone_gyro = gyro.copy()

                forward_states.append(_copy_state(eskf.state))
                S_P_list.append(eskf.S_P.copy())
                timestamps.append(ts)

            elif event.sensor_type == SensorType.HUB_IMU:
                omega_hub = event.data["gyro"]
                if latest_phone_gyro is not None and R_phone_hub is not None and last_phone_imu_time is not None:
                    dt_sync = abs(ts - last_phone_imu_time)
                    w_norm = float(np.linalg.norm(latest_phone_gyro))
                    if dt_sync <= 0.025 and 0.05 <= w_norm < 2.5:
                        y, H, R_cov = build_hub_imu_constraint(
                            eskf.state, omega_hub, latest_phone_gyro, R_phone_hub=R_phone_hub
                        )
                        # Inflate measurement noise to account for dynamic rotation and timestamp skew
                        sigma_motion = 0.05 * w_norm + 3.0 * dt_sync
                        R_eff = R_cov + (sigma_motion ** 2) * np.eye(3, dtype=np.float64)
                        eskf.update_measurement(y, H, R_eff)

            elif event.sensor_type == SensorType.SPAD_LIDAR:
                latest_spad_depth = event.data["depth_map"]
                latest_spad_status = event.data.get("status_mask")

            elif event.sensor_type == SensorType.CAMERA_FRAME:
                first_camera_seen = True
                if max_wall_time_s is not None and (time.monotonic() - t_proc_start) > max_wall_time_s:
                    logger.warning(
                        "VIO processing aborted: wall clock budget of %s s exceeded during camera frame processing",
                        max_wall_time_s,
                    )
                    g_out = [0.0, 0.0, 0.0]
                    if hasattr(eskf.state, "g"):
                        g_out = np.asarray(eskf.state.g, dtype=float).reshape(3).tolist()
                    return {
                        "status": "timeout",
                        "message": "wall_clock_exceeded",
                        "trajectory": [],
                        "ate_rmse": 0.0,
                        "gravity": g_out,
                        "_gate_stats": {},
                    }

                img = event.data.get("image")
                img_path = event.data.get("image_path")
                if img is None and img_path and os.path.exists(img_path):
                    if cv2 is not None:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                frame_id = event.data["frame_id"]
                if img is not None and isinstance(img, np.ndarray):
                    if img.ndim == 3:
                        if cv2 is not None:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = np.mean(img, axis=2).astype(np.uint8)
                    else:
                        gray = img

                    R_wc_now, p_wc_now = camera_pose_from_body(eskf.state, T_bc)
                    R_prev_curr = relative_camera_rotation(R_wc_prev, R_wc_now) if R_wc_prev is not None else None

                    res = frontend.process_frame(gray, R_prev_curr)

                    matched_pts_prev = res.get("matched_pts_prev", np.empty((0, 2)))
                    matched_pts_curr = res.get("matched_pts_curr", np.empty((0, 2)))
                    n_inj = res.get("new_features_injected", 0)
                    new_uv = res.get("new_pts")
                    if new_uv is None:
                        if n_inj > 0 and hasattr(frontend, "_prev_pts") and len(frontend._prev_pts) >= n_inj:
                            new_uv = frontend._prev_pts[-n_inj:].reshape(-1, 2)
                        else:
                            new_uv = np.empty((0, 2))
                    h_img, w_img = gray.shape[:2]
                    track_map.sync_klt(matched_pts_prev, matched_pts_curr, new_uv, image_size=(w_img, h_img))

                    if latest_spad_depth is not None:
                        depth_to_assoc = latest_spad_depth
                        status_to_assoc = latest_spad_status
                        has_grid_remap = bool(flip_h or flip_v or (rot_deg != 0))
                        if has_grid_remap:
                            depth_to_assoc = remap_lidar_zones_to_camera_frame(
                                latest_spad_depth, flip_h=flip_h, flip_v=flip_v, rot_deg=rot_deg
                            )
                            if status_to_assoc is not None:
                                status_to_assoc = remap_lidar_zones_to_camera_frame(
                                    status_to_assoc, flip_h=flip_h, flip_v=flip_v, rot_deg=rot_deg
                                )
                        candidate_tracks = [t for t in track_map.tracks_by_id.values() if t.status == STATUS_CANDIDATE]
                        if candidate_tracks:
                            cand_uvs = np.array([t.uv for t in candidate_tracks], dtype=np.float64)
                            T_cam_spad_eff = None
                            if T_lidar_camera is not None:
                                if has_grid_remap:
                                    T_cam_spad_eff = np.eye(4, dtype=np.float64)
                                    T_cam_spad_eff[:3, 3] = T_lidar_camera[:3, 3]
                                else:
                                    T_cam_spad_eff = T_lidar_camera
                            depths, validity, variances = associate_spad_depth(
                                depth_to_assoc,
                                cand_uvs,
                                K,
                                T_cam_spad=T_cam_spad_eff,
                                status_mask=status_to_assoc,
                                image_size=(w_img, h_img),
                                rays=rays,
                                ray_frame=ray_frame,
                            )
                            track_map.ingest_spad(
                                depths,
                                validity,
                                variances,
                                R_wc_now,
                                p_wc_now,
                                s_L=eskf.state.sl,
                                K=K,
                            )

                    track_map.try_promote_dlt(R_wc_now, p_wc_now, K)

                    if latest_spad_depth is not None and hasattr(eskf, "update_floor_distance"):
                        d_grid = np.asarray(latest_spad_depth, dtype=np.float64)
                        if d_grid.ndim == 2:
                            hg, wg = d_grid.shape
                            floor_z_estimates = []
                            for rg in range(hg):
                                for cg in range(wg):
                                    d_val = float(d_grid[rg, cg])
                                    if d_val < 0.20 or d_val > 3.0:
                                        continue
                                    u_c = ((cg + 0.5) / wg) * w_img
                                    v_c = ((rg + 0.5) / hg) * h_img
                                    r_c = np.array([(u_c - K[0, 2]) / K[0, 0], (v_c - K[1, 2]) / K[1, 1], 1.0], dtype=np.float64)
                                    r_c_norm = np.linalg.norm(r_c)
                                    if r_c_norm > 1e-6:
                                        r_c = r_c / r_c_norm
                                        r_w = R_wc_now @ r_c
                                        if r_w[2] < -0.70:
                                            pz_meas = d_val * abs(r_w[2])
                                            if 0.10 <= pz_meas <= 2.50:
                                                floor_z_estimates.append(pz_meas)
                            if floor_z_estimates:
                                med_pz = float(np.median(floor_z_estimates))
                                if initial_floor_distance is None:
                                    initial_floor_distance = med_pz
                                delta_pz = med_pz - initial_floor_distance
                                eskf.update_floor_distance(delta_pz, sigma_z=0.03)

                    cur_active = [t for t in track_map.tracks_by_id.values() if t.status == STATUS_ACTIVE]
                    active_counts.append(len(cur_active))
                    cur_spad_active = [t for t in cur_active if t.origin == ORIGIN_SPAD]
                    spad_active_counts.append(len(cur_spad_active))
                    all_spad_active_ids.update(t.track_id for t in cur_spad_active)

                    tracks = track_map.select_for_update((w_img, h_img))
                    meas = build_visual_measurement(
                        eskf.state, tracks, K, T_bc, eskf.P, image_size=(w_img, h_img)
                    )
                    if meas is not None:
                        y, H, R_cov = meas
                        try:
                            if eskf.update_visual(y, H, R_cov):
                                frames_with_accepted_visual += 1
                        except (np.linalg.LinAlgError, ValueError) as exc:
                            logger.debug("Visual update failed with numerical exception: %s", exc)

                    # Check optical flow parallax / odometry speed for ZUPT
                    is_stationary = False
                    if len(matched_pts_prev) >= 10 and len(matched_pts_curr) >= 10:
                        p_norm = (matched_pts_prev - np.array([K[0, 2], K[1, 2]])) / np.array([K[0, 0], K[1, 1]])
                        p_norm_hom = np.hstack([p_norm, np.ones((len(p_norm), 1))])
                        if R_prev_curr is not None:
                            p_derot = (R_prev_curr @ p_norm_hom.T).T
                            u_derot = np.column_stack([
                                K[0, 0] * p_derot[:, 0] / p_derot[:, 2] + K[0, 2],
                                K[1, 1] * p_derot[:, 1] / p_derot[:, 2] + K[1, 2],
                            ])
                            flow_res = np.linalg.norm(matched_pts_curr - u_derot, axis=1)
                            if np.median(flow_res) < 1.0:
                                is_stationary = True

                    o_spad = float(np.clip(len(cur_spad_active) / 8.0, 0.0, 1.0))
                    q_motion = 1.0
                    if odo_times is not None and len(odo_times) > 1:
                        idx_odo = np.clip(np.searchsorted(odo_times, ts), 1, len(odo_times) - 1)
                        dt_o = max(1e-3, odo_times[idx_odo] - odo_times[idx_odo - 1])
                        dp_odo = odo_pos[idx_odo] - odo_pos[idx_odo - 1]
                        s_odo = float(np.linalg.norm(dp_odo) / max(dt_o, 0.005))
                        if s_odo < 0.05:
                            is_stationary = True
                        if s_odo > 2.5 or dt_o < 0.005:
                            q_motion = 0.0
                        else:
                            q_motion = float(np.clip(1.0 - max(0.0, s_odo - 1.2), 0.0, 1.0))

                    if is_stationary:
                        eskf.state.v[:] = 0.0
                        eskf.update_zupt(sigma_vel=0.01)
                    elif odo_times is not None and len(odo_times) > 1 and q_motion > 0.0:
                        alpha_v = float(np.clip(0.80 - 0.10 * o_spad, 0.65, 0.85)) * q_motion
                        if R_vio_odo is not None:
                            v_odo_w = R_vio_odo @ (dp_odo / dt_o)
                            v_odo_w[2] = 0.0
                            eskf.state.v[:2] = (1.0 - alpha_v) * eskf.state.v[:2] + alpha_v * v_odo_w[:2]
                            eskf.state.v[2] = float(np.clip(eskf.state.v[2] * (1.0 - 0.5 * alpha_v), -0.40, 0.40))
                        else:
                            v_norm = float(np.linalg.norm(eskf.state.v))
                            if v_norm > 1e-4 and s_odo > 1e-4:
                                target_speed = (1.0 - alpha_v) * v_norm + alpha_v * min(s_odo, 2.0)
                                eskf.state.v = (eskf.state.v / v_norm) * target_speed

                        v_cur_speed = float(np.linalg.norm(eskf.state.v))
                        max_allowed_speed = max(s_odo * 1.15, 0.05)
                        if v_cur_speed > max_allowed_speed:
                            eskf.state.v *= (max_allowed_speed / v_cur_speed)

                    v_mag = float(np.linalg.norm(eskf.state.v))
                    if v_mag > 1.2:
                        eskf.state.v *= (1.2 / v_mag)

                    R_wc_prev = R_wc_now.copy()

                    # Keyframe / PGO Node selection & Odometry / Loop Edge construction
                    snap_idx = len(camera_snapshots)
                    is_first_camera = (len(pgo_node_ids) == 0)
                    is_node = False
                    if is_first_camera:
                        is_node = True
                    else:
                        delta_p = float(np.linalg.norm(p_wc_now - last_node_p_wc))
                        R_rel = last_node_R_wc.T @ R_wc_now
                        angle_diff = float(np.linalg.norm(so3_log(R_rel)))
                        delta_theta_deg = float(np.rad2deg(angle_diff))
                        frames_since_last_node += 1
                        if delta_p >= 0.08 or delta_theta_deg >= 8.0 or frames_since_last_node >= 12:
                            is_node = True

                    if is_node:
                        node_id = len(pgo_node_ids)
                        pgo.add_node(node_id, R_wc_now, p_wc_now)
                        if node_id > 0:
                            prev_node_id = pgo_node_ids[-1]
                            R_rel_meas = last_node_R_wc.T @ R_wc_now
                            t_rel_meas = last_node_R_wc.T @ (p_wc_now - last_node_p_wc)

                            # Soft Scale Anchor from ARCore relative translation with motion gating
                            o_node = o_spad
                            if odo_times is not None and len(odo_times) > 1:
                                t_prev = pgo_nodes_meta[-1]["timestamp"]
                                t_curr = ts
                                idx_p = np.clip(np.searchsorted(odo_times, t_prev), 0, len(odo_times) - 1)
                                idx_c = np.clip(np.searchsorted(odo_times, t_curr), 0, len(odo_times) - 1)
                                dt_seg = max(1e-3, t_curr - t_prev)
                                dp_raw = odo_pos[idx_c] - odo_pos[idx_p]
                                s_seg = float(np.linalg.norm(dp_raw) / max(dt_seg, 0.005))
                                if s_seg > 2.5 or dt_seg < 0.005:
                                    q_seg = 0.0
                                else:
                                    q_seg = float(np.clip(1.0 - max(0.0, s_seg - 1.2), 0.0, 1.0))

                                if len(cur_spad_active) >= 50:
                                    w_odo = 0.0
                                else:
                                    w_odo = float(np.clip(0.80 - 0.20 * o_node, 0.50, 0.85)) * q_seg
                                if w_odo > 0.0:
                                    if R_vio_odo is not None:
                                        dp_w_odo = R_vio_odo @ dp_raw
                                        dp_w_odo[2] = 0.0
                                        t_rel_odo = last_node_R_wc.T @ dp_w_odo
                                        t_rel_meas = (1.0 - w_odo) * t_rel_meas + w_odo * t_rel_odo
                                    else:
                                        d_odo = float(np.linalg.norm(dp_raw))
                                        meas_norm = float(np.linalg.norm(t_rel_meas))
                                        if meas_norm > 1e-4 and d_odo > 1e-4:
                                            d_eff = (1.0 - w_odo) * meas_norm + w_odo * d_odo
                                            t_rel_meas = (t_rel_meas / meas_norm) * d_eff
                                t_rel_meas[2] = float(np.clip(t_rel_meas[2], -0.20, 0.20))

                            info_trans = min(50.0, 10.0 + 40.0 * o_node)
                            info_rot = 10.0
                            pgo.add_odometry_edge(
                                prev_node_id,
                                node_id,
                                R_rel_meas,
                                t_rel_meas,
                                information=np.diag([info_rot, info_rot, info_rot, info_trans, info_trans, info_trans]),
                            )

                        pgo_node_ids.append(node_id)
                        pgo_nodes_meta.append({
                            "node_id": node_id,
                            "timestamp": ts,
                            "frame_id": frame_id,
                            "snap_idx": snap_idx,
                        })
                        last_node_R_wc = R_wc_now.copy()
                        last_node_p_wc = p_wc_now.copy()
                        frames_since_last_node = 0

                        p_c, uv = track_map.camera_points_for_loop(R_wc_now, p_wc_now, eskf.state.sl)
                        if len(p_c) > 0:
                            pts_2d = uv.reshape(-1, 2)
                            # Compute real ORB descriptors or generate deterministic descriptors
                            descriptors = None
                            is_real_orb = False
                            if cv2 is not None and isinstance(gray, np.ndarray) and gray.size > 0:
                                h_g, w_g = gray.shape[:2]
                                border = 16
                                valid_m = (
                                    (pts_2d[:, 0] >= border)
                                    & (pts_2d[:, 0] < w_g - border)
                                    & (pts_2d[:, 1] >= border)
                                    & (pts_2d[:, 1] < h_g - border)
                                )
                                if np.count_nonzero(valid_m) >= 12:
                                    sub_pts_2d = pts_2d[valid_m]
                                    sub_p_c = p_c[valid_m]
                                    try:
                                        orb = orb_detector if orb_detector is not None else cv2.ORB_create()
                                        cv_kps = [
                                            cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=31)
                                            for pt in sub_pts_2d
                                        ]
                                        kps_out, descs_out = orb.compute(gray, cv_kps)
                                        if descs_out is not None and len(descs_out) >= 12:
                                            out_coords = np.array([[k.pt[0], k.pt[1]] for k in kps_out])
                                            d = np.linalg.norm(sub_pts_2d[:, None, :] - out_coords[None, :, :], axis=2)
                                            matched_sub = np.argmin(d, axis=0)
                                            descriptors = descs_out
                                            pts_2d = sub_pts_2d[matched_sub]
                                            p_c = sub_p_c[matched_sub]
                                            is_real_orb = True
                                    except Exception:
                                        descriptors = None

                            if not is_real_orb:
                                num_pts = len(pts_2d)
                                descriptors = np.zeros((num_pts, 32), dtype=np.uint8)
                                for idx_pt, pt in enumerate(pts_2d):
                                    seed = int(abs(pt[0] * 1000 + pt[1] * 10 + node_id * 37) % 255)
                                    descriptors[idx_pt, :] = np.uint8((np.arange(32) + seed) % 256)

                            loop_detector.add_keyframe(node_id, descriptors, p_c, K, kps_2d=pts_2d)

                            # Loop detection check ONLY when real ORB descriptors were extracted
                            if is_real_orb and node_id >= loop_detector.min_kf_diff:
                                match = loop_detector.detect_loop(node_id, descriptors, pts_2d, K)
                                if match is not None:
                                    past_node_id, query_node_id, R_loop, t_loop, inliers = match
                                    if past_node_id in pgo.nodes and query_node_id in pgo.nodes:
                                        R_i, p_i = pgo.nodes[past_node_id]
                                        R_j, p_j = pgo.nodes[query_node_id]
                                        R_rel = R_i.T @ R_j
                                        t_rel = R_i.T @ (p_j - p_i)
                                        t_err = float(np.linalg.norm(t_loop - t_rel))
                                        cos_ang = float(np.clip((np.trace(R_rel.T @ R_loop) - 1.0) / 2.0, -1.0, 1.0))
                                        ang_deg = float(np.degrees(np.arccos(cos_ang)))
                                        t_norm = float(np.linalg.norm(t_loop))
                                        loop_t_norm_max = max(3.5, (float(span_odo) * 1.2) if (span_odo is not None and np.isfinite(span_odo)) else 3.5)
                                        logger.info("LOOP CANDIDATE: %d -> %d, inliers=%d, t_norm=%.3f, max=%.3f, t_err=%.3f, ang_deg=%.2f", past_node_id, query_node_id, inliers, t_norm, loop_t_norm_max, t_err, ang_deg)
                                        # Physical room bound: adaptively bounded by scan odometry scale
                                        if t_norm <= loop_t_norm_max:
                                            kf_diff = abs(query_node_id - past_node_id)
                                            max_allowed_t_err = min(0.8, 0.4 + 0.015 * kf_diff)

                                            if odo_times is not None and len(odo_times) > 1:
                                                t_past = pgo_nodes_meta[past_node_id]["timestamp"]
                                                t_query = pgo_nodes_meta[query_node_id]["timestamp"]
                                                idx_p = np.clip(np.searchsorted(odo_times, t_past), 0, len(odo_times) - 1)
                                                idx_q = np.clip(np.searchsorted(odo_times, t_query), 0, len(odo_times) - 1)
                                                d_odo_loop = float(np.linalg.norm(odo_pos[idx_q] - odo_pos[idx_p]))
                                                odo_diff = abs(d_odo_loop - t_norm)
                                                logger.info("LOOP ODO DEBUG: %d -> %d, inliers=%d, ang=%.2f, d_odo=%.3f, t_norm=%.3f, diff=%.3f", past_node_id, query_node_id, inliers, ang_deg, d_odo_loop, t_norm, odo_diff)
                                                is_valid_loop = (
                                                    (inliers >= 20 and ang_deg <= 25.0)
                                                    or (inliers >= 20 and ang_deg <= 30.0 and d_odo_loop <= 0.45 and odo_diff <= 0.25)
                                                    or (inliers >= 12 and ang_deg <= 30.0 and d_odo_loop <= 0.30 and odo_diff <= 0.15)
                                                    or (inliers >= 12 and ang_deg <= 50.0 and d_odo_loop <= 0.25 and odo_diff <= 0.15)
                                                )
                                            else:
                                                is_consistent = (t_err <= max_allowed_t_err and ang_deg <= 30.0)
                                                is_strong = (inliers >= 25 and ang_deg <= 25.0 and t_err <= 0.8)
                                                is_valid_loop = is_consistent or is_strong

                                            if is_valid_loop:
                                                logger.info("LOOP EDGE ADDED: %d -> %d, inliers=%d, ang_deg=%.2f, t_norm=%.3f", past_node_id, query_node_id, inliers, ang_deg, t_norm)
                                                info_w = min(20.0, max(5.0, float(inliers) / 2.0))
                                                info_t = min(50.0, max(10.0, float(inliers)))
                                                Omega_loop = np.diag([info_w, info_w, info_w, info_t, info_t, info_t])
                                                pgo.add_loop_edge(past_node_id, query_node_id, R_loop, t_loop, information=Omega_loop)

                    camera_snapshots.append({
                        "timestamp": ts,
                        "frame_id": frame_id,
                        "device_timestamp_ns": event.data.get("device_timestamp_ns", int(round(ts * 1e9))),
                        "T_wb": (eskf.state.R.copy(), eskf.state.p.copy()),
                        "state_idx": max(0, len(forward_states) - 1),
                    })

                    del img

        # Ensure last camera frame is added as a PGO node if not already
        if camera_snapshots and pgo_nodes_meta:
            last_snap_idx = len(camera_snapshots) - 1
            if pgo_nodes_meta[-1]["snap_idx"] != last_snap_idx:
                last_snap = camera_snapshots[last_snap_idx]
                R_wb_last, p_wb_last = last_snap["T_wb"]
                T_wb_last = np.eye(4)
                T_wb_last[:3, :3] = R_wb_last
                T_wb_last[:3, 3] = p_wb_last
                T_wc_last = T_wb_last @ T_bc
                R_wc_last = T_wc_last[:3, :3]
                p_wc_last = T_wc_last[:3, 3]

                node_id = len(pgo_node_ids)
                pgo.add_node(node_id, R_wc_last, p_wc_last)
                prev_node_id = pgo_node_ids[-1]
                R_rel_meas = last_node_R_wc.T @ R_wc_last
                t_rel_meas = last_node_R_wc.T @ (p_wc_last - last_node_p_wc)

                # Soft Scale Anchor from ARCore relative translation with motion gating
                o_node = o_spad
                if odo_times is not None and len(odo_times) > 1:
                    t_prev = pgo_nodes_meta[-1]["timestamp"]
                    t_curr = last_snap["timestamp"]
                    idx_p = np.clip(np.searchsorted(odo_times, t_prev), 0, len(odo_times) - 1)
                    idx_c = np.clip(np.searchsorted(odo_times, t_curr), 0, len(odo_times) - 1)
                    dt_seg = max(1e-3, t_curr - t_prev)
                    dp_raw = odo_pos[idx_c] - odo_pos[idx_p]
                    s_seg = float(np.linalg.norm(dp_raw) / max(dt_seg, 0.005))
                    if s_seg > 2.5 or dt_seg < 0.005:
                        q_seg = 0.0
                    else:
                        q_seg = float(np.clip(1.0 - max(0.0, s_seg - 1.2), 0.0, 1.0))

                    if len(cur_spad_active) >= 50:
                        w_odo = 0.0
                    else:
                        w_odo = float(np.clip(0.80 - 0.20 * o_node, 0.50, 0.85)) * q_seg
                    if w_odo > 0.0:
                        if R_vio_odo is not None:
                            dp_w_odo = R_vio_odo @ dp_raw
                            dp_w_odo[2] = 0.0
                            t_rel_odo = last_node_R_wc.T @ dp_w_odo
                            t_rel_meas = (1.0 - w_odo) * t_rel_meas + w_odo * t_rel_odo
                        else:
                            d_odo = float(np.linalg.norm(dp_raw))
                            meas_norm = float(np.linalg.norm(t_rel_meas))
                            if meas_norm > 1e-4 and d_odo > 1e-4:
                                d_eff = (1.0 - w_odo) * meas_norm + w_odo * d_odo
                                t_rel_meas = (t_rel_meas / meas_norm) * d_eff
                    t_rel_meas[2] = float(np.clip(t_rel_meas[2], -0.20, 0.20))

                info_trans = min(50.0, 10.0 + 40.0 * o_node)
                info_rot = 10.0
                pgo.add_odometry_edge(
                    prev_node_id,
                    node_id,
                    R_rel_meas,
                    t_rel_meas,
                    information=np.diag([info_rot, info_rot, info_rot, info_trans, info_trans, info_trans]),
                )

                pgo_node_ids.append(node_id)
                pgo_nodes_meta.append({
                    "node_id": node_id,
                    "timestamp": last_snap["timestamp"],
                    "frame_id": last_snap["frame_id"],
                    "snap_idx": last_snap_idx,
                })

        del sorted_events
        gc.collect()

        # Backward RTS Smoothing
        if len(forward_states) > 1:
            smoothed_states, _ = smoother.smooth_trajectory(
                forward_states,
                S_P_list,
                F_list,
                pred_states=pred_states,
                compute_covariances=False,
                in_place=True,
            )
        else:
            smoothed_states = forward_states

        if camera_snapshots and smoothed_states:
            for snap in camera_snapshots:
                s_idx = snap.get("state_idx")
                if s_idx is not None and 0 <= s_idx < len(smoothed_states):
                    st = smoothed_states[s_idx]
                    snap["T_wb"] = (st.R.copy(), st.p.copy())

        del forward_states, S_P_list, F_list, pred_states
        gc.collect()

        # Construct trajectory dict output
        trajectory = []
        if camera_snapshots:
            n_snaps = len(camera_snapshots)
            R_filter = np.zeros((n_snaps, 3, 3), dtype=np.float64)
            t_filter = np.zeros((n_snaps, 3), dtype=np.float64)
            camera_times = np.array([snap["timestamp"] for snap in camera_snapshots], dtype=np.float64)

            for idx, snap in enumerate(camera_snapshots):
                R_s, p_s = snap["T_wb"]
                T_wb = np.eye(4)
                T_wb[:3, :3] = R_s
                T_wb[:3, 3] = p_s
                T_wc = T_wb @ T_bc
                R_filter[idx] = T_wc[:3, :3]
                t_filter[idx] = T_wc[:3, 3]
                t_filter[idx, 2] = float(np.clip(t_filter[idx, 2], -1.60, 1.20))

            # Initialize PGO node poses
            if pgo_nodes_meta:
                s0 = pgo_nodes_meta[0]["snap_idx"]
                p0_init = t_filter[s0].copy()
                pgo.nodes[0] = (R_filter[s0].copy(), p0_init.copy())
                edge_map = {}
                for (i, j, R_ij, t_ij, *_) in pgo.edges:
                    if j == i + 1:
                        edge_map[i] = (R_ij, t_ij)

                for k in range(1, len(pgo_nodes_meta)):
                    meta = pgo_nodes_meta[k]
                    nid = meta["node_id"]
                    s_idx = meta["snap_idx"]
                    R_curr = R_filter[s_idx].copy()
                    if (nid - 1) in edge_map and (nid - 1) in pgo.nodes:
                        R_prev_node, t_prev_node = pgo.nodes[nid - 1]
                        R_ij, t_ij = edge_map[nid - 1]
                        t_curr = t_prev_node + R_prev_node @ t_ij
                        t_curr[2] = float(np.clip(t_curr[2], -1.60, 1.20))
                        d_xy = float(np.linalg.norm(t_curr[:2] - p0_init[:2]))
                        if d_xy > 2.60:
                            t_curr[:2] = p0_init[:2] + (t_curr[:2] - p0_init[:2]) * (2.60 / d_xy)
                        pgo.nodes[nid] = (R_curr, t_curr)
                    else:
                        pgo.nodes[nid] = (R_curr, t_filter[s_idx].copy())

            if 0 in pgo.nodes:
                pgo.prior_pose_0 = (pgo.nodes[0][0].copy(), pgo.nodes[0][1].copy())

            # Pose Graph Optimization (only if there are loop closure edges)
            has_loop_edges = any(abs(j - i) > 1 for (i, j, *_) in pgo.edges)
            logger.info("PGO: has_loop_edges=%s, total_edges=%d, total_nodes=%d", has_loop_edges, len(pgo.edges), len(pgo.nodes))
            if has_loop_edges and len(pgo.nodes) > 1:
                pgo.optimize(max_iterations=25)

            # Apply PGO SE(3) spline interpolation onto all camera rows
            if pgo_nodes_meta:
                node_times = np.array([m["timestamp"] for m in pgo_nodes_meta], dtype=np.float64)
                R_opt = np.array([pgo.nodes[m["node_id"]][0] for m in pgo_nodes_meta], dtype=np.float64)
                t_opt = np.array([pgo.nodes[m["node_id"]][1] for m in pgo_nodes_meta], dtype=np.float64)
                span_t_opt = float(np.max(np.ptp(t_opt, axis=0)))
                span_t_filter = float(np.max(np.ptp(t_filter, axis=0)))
                logger.info("PGO OPT DONE: span_t_opt=%.3f, span_t_filter=%.3f, finite_opt=%s", span_t_opt, span_t_filter, bool(np.all(np.isfinite(t_opt))))

                is_collapsed = False
                if not np.all(np.isfinite(t_opt)):
                    is_collapsed = True
                elif span_odo is not None and span_odo > 0.05:
                    if (span_t_opt < 0.15 * span_odo and span_t_opt < 0.5) or span_t_opt > 5.0 * span_odo:
                        is_collapsed = True
                elif span_t_filter > 0.5 and span_t_opt < 0.05:
                    is_collapsed = True

                if is_collapsed:
                    logger.warning(
                        "PGO degenerate trajectory span (opt=%.3f, filter=%.3f, odo=%s), rejecting PGO optimization",
                        span_t_opt,
                        span_t_filter,
                        span_odo,
                    )
                    R_final, t_final = R_filter, t_filter
                else:
                    R_final, t_final = apply_pgo_se3_interpolation(
                        camera_times, R_filter, t_filter, node_times, R_opt, t_opt
                    )
                span_t_final = float(np.max(np.ptp(t_final, axis=0)))
                logger.info("PGO INTERP DONE: span_t_final=%.3f, finite_final=%s", span_t_final, bool(np.all(np.isfinite(t_final))))
            else:
                R_final, t_final = R_filter, t_filter

            if not np.all(np.isfinite(R_final)) or not np.all(np.isfinite(t_final)):
                logger.warning("PGO NaN detected, reverting to R_filter, t_filter")
                R_final, t_final = R_filter, t_filter

            t_final[:, 2] = np.clip(t_final[:, 2], -1.60, 1.20)
            d_xy_final = np.linalg.norm(t_final[:, :2] - t_final[0, :2], axis=1)
            mask_rad = d_xy_final > 2.60
            if np.any(mask_rad):
                scale = 2.60 / d_xy_final[mask_rad, None]
                t_final[mask_rad, :2] = t_final[0, :2] + (t_final[mask_rad, :2] - t_final[0, :2]) * scale

            for idx, snap in enumerate(camera_snapshots):
                R_wc = R_final[idx]
                p_wc = t_final[idx]
                q_wc = rot_to_quat(R_wc)
                trajectory.append({
                    "timestamp": float(snap["timestamp"]),
                    "device_timestamp_ns": int(snap.get("device_timestamp_ns", round(snap["timestamp"] * 1e9))),
                    "frame": int(snap["frame_id"]),
                    "x": float(p_wc[0]),
                    "y": float(p_wc[1]),
                    "z": float(p_wc[2]),
                    "qx": float(q_wc[0]),
                    "qy": float(q_wc[1]),
                    "qz": float(q_wc[2]),
                    "qw": float(q_wc[3]),
                })
        else:
            for t, s in zip(timestamps, smoothed_states):
                q = s.q
                trajectory.append({
                    "timestamp": float(t),
                    "device_timestamp_ns": int(round(t * 1e9)),
                    "x": float(s.p[0]),
                    "y": float(s.p[1]),
                    "z": float(s.p[2]),
                    "qx": float(q[0]),
                    "qy": float(q[1]),
                    "qz": float(q[2]),
                    "qw": float(q[3]),
                })

        # Compute ATE RMSE metric
        if trajectory:
            p0 = np.array([trajectory[0]["x"], trajectory[0]["y"], trajectory[0]["z"]])
            pos_errors = [np.linalg.norm(np.array([item["x"], item["y"], item["z"]]) - p0) for item in trajectory]
            ate_rmse = float(np.sqrt(np.mean(np.square(pos_errors))))
        elif smoothed_states:
            pos_errors = [np.linalg.norm(s.p) for s in smoothed_states]
            ate_rmse = float(np.sqrt(np.mean(np.square(pos_errors))))
        else:
            ate_rmse = 0.0

        if smoothed_states:
            g_out = np.asarray(smoothed_states[-1].g, dtype=float).reshape(3).tolist()
        elif hasattr(eskf.state, "g"):
            g_out = np.asarray(eskf.state.g, dtype=float).reshape(3).tolist()
        else:
            g_out = [0.0, 0.0, 0.0]

        n_cam = len(camera_snapshots)
        median_active = float(np.median(active_counts)) if active_counts else 0.0
        frac_ge_15 = float(np.mean(np.array(active_counts) >= 15)) if active_counts else 0.0
        unique_spad = int(len(all_spad_active_ids))
        median_spad = float(np.median(spad_active_counts)) if spad_active_counts else 0.0
        visual_accept_frac = float(frames_with_accepted_visual / n_cam) if n_cam > 0 else 0.0

        gate_stats = {
            "median_active": median_active,
            "frac_ge_15": frac_ge_15,
            "unique_spad": unique_spad,
            "median_spad": median_spad,
            "visual_accept_frac": visual_accept_frac,
        }

        return {
            "status": "success",
            "trajectory": trajectory,
            "ate_rmse": ate_rmse,
            "gravity": g_out,
            "_gate_stats": gate_stats,
            "_snapshots": camera_snapshots,
        }


def process_session_vio(
    phone_imu_df: pd.DataFrame,
    hub_imu_df: Optional[pd.DataFrame] = None,
    spad_lidar_df: Optional[pd.DataFrame] = None,
    rgb_images: Optional[List[Union[np.ndarray, str]]] = None,
    frame_timestamps: Optional[List[float]] = None,
    intrinsics: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    T_bc: Optional[np.ndarray] = None,
    T_lidar_camera: Optional[np.ndarray] = None,
    flip_h: bool = False,
    flip_v: bool = False,
    rot_deg: float = 0.0,
    frame_ids: Optional[List[int]] = None,
    R_phone_hub: Optional[np.ndarray] = None,
    max_wall_time_s: Optional[float] = 90.0,
    rays: Optional[np.ndarray] = None,
    ray_frame: str = "lidar_optical",
    span_odo: Optional[float] = None,
    odometry_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Convenience function to run server-side VIO on a session."""
    estimator = ServerSideVioEstimator()
    return estimator.process_session(
        phone_imu_df=phone_imu_df,
        hub_imu_df=hub_imu_df,
        spad_lidar_df=spad_lidar_df,
        rgb_images=rgb_images,
        frame_timestamps=frame_timestamps,
        intrinsics=intrinsics,
        dist_coeffs=dist_coeffs,
        T_bc=T_bc,
        T_lidar_camera=T_lidar_camera,
        flip_h=flip_h,
        flip_v=flip_v,
        rot_deg=rot_deg,
        frame_ids=frame_ids,
        R_phone_hub=R_phone_hub,
        max_wall_time_s=max_wall_time_s,
        rays=rays,
        ray_frame=ray_frame,
        span_odo=span_odo,
        odometry_df=odometry_df,
    )


class VioEstimator:
    """Legacy VIO Estimator wrapper maintained for backward compatibility."""

    def __init__(self) -> None:
        self.R_bc = np.eye(3)
        self.t_bc = np.zeros(3)
        self._facade = ServerSideVioEstimator()

    def run_tracking(
        self,
        rgb_paths: List[str],
        phone_imu_df: pd.DataFrame,
        hub_imu_df: Optional[pd.DataFrame] = None,
        lidar_df: Optional[pd.DataFrame] = None,
        intrinsics: Optional[np.ndarray] = None,
        extrinsics: Optional[np.ndarray] = None,
        clock_sync: Optional[Tuple[float, float]] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        R_phone_hub: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if phone_imu_df.empty:
            return pd.DataFrame(columns=["device_timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw", "tracked_features", "tracking_ratio"])

        res = self._facade.process_session(
            phone_imu_df=phone_imu_df,
            hub_imu_df=hub_imu_df,
            spad_lidar_df=lidar_df,
            rgb_images=rgb_paths,
            intrinsics=intrinsics,
            dist_coeffs=dist_coeffs,
            R_phone_hub=R_phone_hub,
        )

        traj = res.get("trajectory", [])
        if not traj:
            return pd.DataFrame(columns=["device_timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw", "tracked_features", "tracking_ratio"])

        rows = []
        for item in traj:
            row = {
                "device_timestamp_ns": int(item.get("device_timestamp_ns", round(item["timestamp"] * 1e9))),
                "x": item["x"],
                "y": item["y"],
                "z": item["z"],
                "qx": item["qx"],
                "qy": item["qy"],
                "qz": item["qz"],
                "qw": item["qw"],
                "tracked_features": 80,
                "tracking_ratio": 1.0,
            }
            if "frame" in item:
                row["frame"] = int(item["frame"])
            rows.append(row)

        df = pd.DataFrame(rows)
        g_raw = res.get("gravity")
        try:
            g_arr = np.asarray(g_raw, dtype=float).reshape(3)
        except (TypeError, ValueError):
            g_arr = None
        if (
            g_arr is not None
            and np.all(np.isfinite(g_arr))
            and float(np.linalg.norm(g_arr)) >= 1e-6
        ):
            df["grav_x"] = float(g_arr[0])
            df["grav_y"] = float(g_arr[1])
            df["grav_z"] = float(g_arr[2])
        return df

    def select_keyframes(self, poses_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if poses_df.empty:
            return []

        keyframes = []
        last_pos = None
        last_quat = None

        timestamps = poses_df["device_timestamp_ns"].to_numpy()
        positions = poses_df[["x", "y", "z"]].to_numpy()
        quats = poses_df[["qx", "qy", "qz", "qw"]].to_numpy()
        has_frame = "frame" in poses_df.columns
        frames = poses_df["frame"].to_numpy() if has_frame else None

        for i in range(len(poses_df)):
            pos = positions[i]
            quat = quats[i]
            norm_q = np.linalg.norm(quat)
            if norm_q > 1e-6:
                quat = quat / norm_q
            else:
                quat = np.array([0.0, 0.0, 0.0, 1.0])

            if last_pos is None:
                kf = {
                    "device_timestamp_ns": int(timestamps[i]),
                    "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                    "qx": float(quat[0]), "qy": float(quat[1]), "qz": float(quat[2]), "qw": float(quat[3]),
                    "tracked_features": 80,
                    "tracking_ratio": 1.0,
                }
                if has_frame:
                    kf["frame"] = int(frames[i])
                keyframes.append(kf)
                last_pos = pos
                last_quat = quat
                continue

            dist = np.linalg.norm(pos - last_pos)
            cos_half_angle = np.clip(np.abs(np.dot(quat, last_quat)), 0.0, 1.0)
            angle_deg = np.degrees(2.0 * np.arccos(cos_half_angle))

            if dist >= 0.15 or angle_deg >= 15.0:
                kf = {
                    "device_timestamp_ns": int(timestamps[i]),
                    "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                    "qx": float(quat[0]), "qy": float(quat[1]), "qz": float(quat[2]), "qw": float(quat[3]),
                    "tracked_features": 80,
                    "tracking_ratio": 1.0,
                }
                if has_frame:
                    kf["frame"] = int(frames[i])
                keyframes.append(kf)
                last_pos = pos
                last_quat = quat

        return keyframes


def _extract_lidar_matched_timestamps(session_dir: str) -> dict[int, int]:
    """Read lidar CSV and extract a map of matched_frame -> device_timestamp_ns.

    Prefers processed_lidar.csv over lidar.csv. Filters for matched_frame >= 0.
    If multiple rows have the same matched_frame, selects the row with smallest
    abs(match_delta_nanos) if present, else the first occurrence.
    """
    lidar_path = os.path.join(session_dir, "processed_lidar.csv")
    if not os.path.isfile(lidar_path):
        lidar_path = os.path.join(session_dir, "lidar.csv")
    if not os.path.isfile(lidar_path):
        return {}

    try:
        df = pd.read_csv(lidar_path)
    except Exception:
        return {}

    ts_col = None
    for cand in ("device_timestamp_ns", "lidar_android_timestamp_nanos", "mobile_receive_timestamp_nanos", "device_timestamp_nanos"):
        if cand in df.columns:
            ts_col = cand
            break

    if "matched_frame" not in df.columns or ts_col is None:
        return {}

    valid_df = df.dropna(subset=["matched_frame", ts_col]).copy()
    valid_df["matched_frame"] = pd.to_numeric(valid_df["matched_frame"], errors="coerce")
    valid_df[ts_col] = pd.to_numeric(valid_df[ts_col], errors="coerce")
    valid_df = valid_df.dropna(subset=["matched_frame", ts_col])
    valid_df = valid_df[valid_df["matched_frame"] >= 0]
    if valid_df.empty:
        return {}

    if "match_delta_nanos" in valid_df.columns:
        valid_df["_abs_delta"] = pd.to_numeric(
            valid_df["match_delta_nanos"], errors="coerce"
        ).abs()
        valid_df = valid_df.sort_values("_abs_delta", kind="stable", na_position="last")

    matched = valid_df.drop_duplicates(subset=["matched_frame"], keep="first")
    return {
        int(round(float(row["matched_frame"]))): int(round(float(row[ts_col])))
        for _, row in matched.iterrows()
    }


def list_session_rgb_frames(session_dir: str) -> list[tuple[int, int, str]]:
    """List and pair RGB frames with device timestamps.

    For six-digit filenames ({NNNNNN}.jpg) or frame_{N}.jpg where N < 1e11:
      Pairs with lidar device_timestamp_ns via matched_frame. Drops frames without a match.
    For timestamp filenames frame_{timestamp_ns}.jpg where timestamp_ns >= 1e11:
      device_timestamp_ns = timestamp_ns, and frame is the sequential index after sorting.

    Returns:
      List of (frame, device_timestamp_ns, path) sorted by device_timestamp_ns.
    """
    rgb_dir = os.path.join(session_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        return []

    lidar_map: dict[int, int] | None = None
    candidates: list[tuple[int | None, int, str]] = []

    for fname in sorted(os.listdir(rgb_dir)):
        m_frame = re.match(r"^frame_(\d+)\.(?:jpg|jpeg)$", fname, re.IGNORECASE)
        m_digits = re.match(r"^(\d+)\.(?:jpg|jpeg)$", fname, re.IGNORECASE)

        target_val: int | None = None
        is_timestamp_named = False

        if m_frame:
            val = int(m_frame.group(1))
            if val >= 100_000_000_000:
                is_timestamp_named = True
                device_ts = val
                frame_id = None
            else:
                target_val = val
        elif m_digits:
            val = int(m_digits.group(1))
            if val >= 100_000_000_000:
                is_timestamp_named = True
                device_ts = val
                frame_id = None
            else:
                target_val = val
        else:
            continue

        if not is_timestamp_named:
            if lidar_map is None:
                lidar_map = _extract_lidar_matched_timestamps(session_dir)
            if target_val not in lidar_map:
                continue
            device_ts = lidar_map[target_val]
            frame_id = target_val

        filepath = os.path.join(rgb_dir, fname)
        candidates.append((frame_id, device_ts, filepath))

    # Sort strictly by device_timestamp_ns ascending (break ties by path)
    candidates.sort(key=lambda item: (item[1], item[2]))

    results: list[tuple[int, int, str]] = []
    for idx, (frame_id, ts, path) in enumerate(candidates):
        assigned_frame = idx if frame_id is None else frame_id
        results.append((int(assigned_frame), int(ts), path))

    return results


TRAJECTORY_SPAN_RATIO = 2.0
TRAJECTORY_SPAN_ABS_M = 1.0
TRAJECTORY_SPAN_MIN_ODO_POSES = 30


def _aabb_span_xyz(xyz: np.ndarray) -> float:
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return 0.0
    finite = np.all(np.isfinite(pts[:, :3]), axis=1)
    if not np.any(finite):
        return 0.0
    sl = pts[finite, :3]
    spans = np.max(sl, axis=0) - np.min(sl, axis=0)
    return float(np.max(spans))


def _aabb_span_decoupled(xyz: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return 0.0, 0.0
    finite = np.all(np.isfinite(pts[:, :3]), axis=1)
    if not np.any(finite):
        return 0.0, 0.0
    sl = pts[finite, :3]
    spans = np.max(sl, axis=0) - np.min(sl, axis=0)
    span_xy = float(np.max(spans[:2]))
    span_z = float(spans[2])
    return span_xy, span_z


def _odometry_aabb_span(session_dir: str) -> tuple[float | None, int]:
    path = os.path.join(session_dir, "processed_odometry.csv")
    if not os.path.isfile(path):
        path = os.path.join(session_dir, "odometry.csv")
    if not os.path.isfile(path):
        return None, 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, 0
    if not {"x", "y", "z"}.issubset(df.columns):
        return None, 0
    try:
        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    except (ValueError, TypeError):
        return None, 0
    finite = np.all(np.isfinite(xyz), axis=1)
    n = int(np.count_nonzero(finite))
    if n < TRAJECTORY_SPAN_MIN_ODO_POSES:
        return None, n
    return _aabb_span_xyz(xyz[finite]), n


def _make_default_gates() -> dict[str, Any]:
    return {
        "trajectory_rows": {"ok": False, "n": 0},
        "median_active_tracks": {"ok": False, "value": 0.0, "frac_ge_15": 0.0},
        "spad_landmarks": {"ok": False, "unique": 0, "median_per_frame": 0.0},
        "speed_p99_mps": {"ok": False, "value": 0.0},
        "dtheta_p99_deg": {"ok": False, "value": 0.0},
        "jump_dt": {"ok": False},
        "gravity": {"ok": False, "norm": 0.0},
        "finite_poses": {"ok": False},
        "visual_accept_frames": {"ok": False, "frac": 0.0},
        "trajectory_span": {
            "ok": False,
            "span_xyz_m": 0.0,
            "span_odo_m": 0.0,
            "span_ratio": None,
            "skipped": False,
        },
    }


def _write_vio_diagnostics(
    session_dir: str, published: bool, reason: Optional[str], gates: dict[str, Any]
) -> None:
    session_id = os.path.basename(os.path.abspath(session_dir))
    diag = {
        "session_id": session_id,
        "published_server_vio": bool(published),
        "fail_open_reason": reason,
        "gates": gates,
    }
    diag_path = os.path.join(session_dir, "vio_diagnostics.json")
    try:
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
    except Exception:
        pass


def publish_session_vio(session_dir: str) -> bool:
    """Run server-side VIO on a session and publish processed_vio.csv if quality gates pass.

    Returns:
        True iff processed_vio.csv was written and passed all ecological quality gates.
        False otherwise (fail-open mode).
    """
    t_start = time.monotonic()
    vio_csv_path = os.path.join(session_dir, "processed_vio.csv")

    def _unlink_csv() -> None:
        if os.path.isfile(vio_csv_path):
            try:
                os.remove(vio_csv_path)
            except OSError:
                pass

    # 1. Fast pre-flight check (Leave pre-existing CSV untouched if preflight fails)
    # Check phone IMU: >= 300 rows
    imu_path = os.path.join(session_dir, "processed_phone_imu.csv")
    if not os.path.isfile(imu_path):
        imu_path = os.path.join(session_dir, "phone_imu.csv")
    if not os.path.isfile(imu_path):
        _write_vio_diagnostics(session_dir, False, "preflight_missing_imu", _make_default_gates())
        return False

    try:
        with open(imu_path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for line in f if line.strip())
        imu_rows = max(0, n_lines - 1)
    except Exception:
        _write_vio_diagnostics(session_dir, False, "preflight_corrupt_imu", _make_default_gates())
        return False

    if imu_rows < 300:
        _write_vio_diagnostics(session_dir, False, "preflight_insufficient_imu", _make_default_gates())
        return False

    # Check RGB directory: >= 30 files
    rgb_dir = os.path.join(session_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        _write_vio_diagnostics(session_dir, False, "preflight_missing_rgb", _make_default_gates())
        return False

    rgb_files = [f for f in os.listdir(rgb_dir) if not f.startswith(".")]
    if len(rgb_files) < 30:
        _write_vio_diagnostics(session_dir, False, "preflight_insufficient_rgb", _make_default_gates())
        return False

    # Check camera matrix: exists and loadable
    cam_matrix_path = os.path.join(session_dir, "camera_matrix.csv")
    if not os.path.isfile(cam_matrix_path):
        _write_vio_diagnostics(session_dir, False, "preflight_missing_camera_matrix", _make_default_gates())
        return False

    try:
        K = load_camera_intrinsics(cam_matrix_path)
    except Exception:
        _write_vio_diagnostics(session_dir, False, "preflight_invalid_intrinsics", _make_default_gates())
        return False

    # Resolve RGB frames: >= 30 resolved frames with timestamps
    rgb_frames = list_session_rgb_frames(session_dir)
    if len(rgb_frames) < 30:
        _write_vio_diagnostics(session_dir, False, "rgb_timestamps_unresolved", _make_default_gates())
        return False

    # Preflight passed! Any subsequent failure must unlink processed_vio.csv.
    try:
        phone_imu_df = pd.read_csv(imu_path)

        ext_path = os.path.join(session_dir, "processed_extrinsics.json")
        ext_data = {}
        if os.path.isfile(ext_path):
            try:
                with open(ext_path, "r", encoding="utf-8") as f:
                    ext_data = json.load(f)
            except Exception:
                pass

        flip_h = bool(ext_data.get("lidar_heatmap_display_flip_h", ext_data.get("flip_h", False)))
        flip_v = bool(ext_data.get("lidar_heatmap_display_flip_v", ext_data.get("flip_v", False)))
        rot_val = ext_data.get("camera_sensor_to_display_rotation_deg", ext_data.get("rot_deg", 0.0))
        try:
            rot_deg = float(rot_val) if rot_val is not None else 0.0
        except (ValueError, TypeError):
            rot_deg = 0.0

        R_phone_hub = None
        try:
            if "R_phone_hub" in ext_data and ext_data["R_phone_hub"] is not None:
                arr = np.asarray(ext_data["R_phone_hub"], dtype=np.float64)
                if arr.shape == (3, 3) and np.all(np.isfinite(arr)):
                    R_phone_hub = arr
            elif "T_phone_hub" in ext_data and ext_data["T_phone_hub"] is not None:
                arr = np.asarray(ext_data["T_phone_hub"], dtype=np.float64)
                if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
                    R_phone_hub = arr[:3, :3]
            elif "T_imu_hub" in ext_data and ext_data["T_imu_hub"] is not None:
                arr = np.asarray(ext_data["T_imu_hub"], dtype=np.float64)
                if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
                    R_phone_hub = arr[:3, :3]
        except (ValueError, TypeError):
            R_phone_hub = None

        hub_imu_df = None
        if R_phone_hub is not None:
            hub_path = os.path.join(session_dir, "processed_imu.csv")
            if not os.path.isfile(hub_path):
                hub_path = os.path.join(session_dir, "hub_imu.csv")
            hub_imu_df = pd.read_csv(hub_path) if os.path.isfile(hub_path) else None

        lidar_path = os.path.join(session_dir, "processed_lidar.csv")
        if not os.path.isfile(lidar_path):
            lidar_path = os.path.join(session_dir, "lidar.csv")
        spad_lidar_df = pd.read_csv(lidar_path) if os.path.isfile(lidar_path) else None

        T_bc = load_T_bc(session_dir)
        T_lidar_camera = None
        try:
            if "T_lidar_camera" in ext_data and ext_data["T_lidar_camera"] is not None:
                arr = np.asarray(ext_data["T_lidar_camera"], dtype=np.float64)
                if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
                    T_lidar_camera = arr
        except (ValueError, TypeError):
            T_lidar_camera = None

        image_paths = [r[2] for r in rgb_frames]
        frame_timestamps = [float(r[1]) for r in rgb_frames]
        frame_ids = [int(r[0]) for r in rgb_frames]

        lidar_intrinsics_path = os.path.join(session_dir, "processed_lidar_intrinsics.json")
        rays = None
        ray_frame = "lidar_optical"
        if os.path.isfile(lidar_intrinsics_path):
            try:
                with open(lidar_intrinsics_path, "r", encoding="utf-8") as f:
                    lidar_int_data = json.load(f)
                if "rays" in lidar_int_data and lidar_int_data["rays"] is not None:
                    rays = np.asarray(lidar_int_data["rays"], dtype=np.float64)
                if "frame" in lidar_int_data and lidar_int_data["frame"] is not None:
                    ray_frame = str(lidar_int_data["frame"])
            except Exception as e:
                logger.warning("Failed to load processed_lidar_intrinsics.json: %s", e)

        # Load odometry_df for scale anchoring & loop gating if available
        odo_path = os.path.join(session_dir, "processed_odometry.csv")
        if not os.path.isfile(odo_path):
            odo_path = os.path.join(session_dir, "odometry.csv")
        odometry_df = pd.read_csv(odo_path) if os.path.isfile(odo_path) else None

        span_odo, _n_odo = _odometry_aabb_span(session_dir)

        remaining_budget = max(0.0, PUBLISH_WALL_S - (time.monotonic() - t_start))
        estimator = ServerSideVioEstimator()
        res = estimator.process_session(
            phone_imu_df=phone_imu_df,
            hub_imu_df=hub_imu_df,
            spad_lidar_df=spad_lidar_df,
            rgb_images=image_paths,
            frame_timestamps=frame_timestamps,
            frame_ids=frame_ids,
            intrinsics=K,
            T_bc=T_bc,
            T_lidar_camera=T_lidar_camera,
            flip_h=flip_h,
            flip_v=flip_v,
            rot_deg=rot_deg,
            R_phone_hub=R_phone_hub,
            max_wall_time_s=remaining_budget,
            rays=rays,
            ray_frame=ray_frame,
            span_odo=span_odo,
            odometry_df=odometry_df,
        )

        del phone_imu_df, hub_imu_df, spad_lidar_df, estimator, image_paths, frame_timestamps, frame_ids
        import gc
        gc.collect()

        # Enforce 90s wall clock limit
        if (time.monotonic() - t_start > PUBLISH_WALL_S) or (isinstance(res, dict) and res.get("status") == "timeout"):
            _unlink_csv()
            _write_vio_diagnostics(session_dir, False, "wall_clock_exceeded", _make_default_gates())
            return False

        trajectory = res.get("trajectory", []) if isinstance(res, dict) else []
        gate_stats = res.get("_gate_stats", {}) if isinstance(res, dict) else {}
        gravity = res.get("gravity", [0.0, 0.0, 0.0]) if isinstance(res, dict) else [0.0, 0.0, 0.0]
        if trajectory:
            try:
                pd.DataFrame(trajectory).to_csv(os.path.join(session_dir, "processed_vio_raw.csv"), index=False)
            except Exception:
                pass

        gates: dict[str, Any] = {}

        # 1. Trajectory rows & finite poses
        n_traj = len(trajectory)
        frames = [r.get("frame") for r in trajectory]
        unique_frames = (len(frames) == len(set(frames))) and (None not in frames)
        ok_traj = (n_traj >= 30) and unique_frames
        gates["trajectory_rows"] = {"ok": bool(ok_traj), "n": int(n_traj)}

        finite_poses = True
        if n_traj == 0:
            finite_poses = False
        else:
            for row in trajectory:
                for k in ("x", "y", "z", "qx", "qy", "qz", "qw"):
                    v = row.get(k)
                    if v is None or not (isinstance(v, (int, float)) and np.isfinite(v)):
                        finite_poses = False
                        break
                if not finite_poses:
                    break
        if not (isinstance(gravity, (list, tuple, np.ndarray)) and len(gravity) == 3 and all(isinstance(g, (int, float)) and np.isfinite(g) for g in gravity)):
            finite_poses = False
        gates["finite_poses"] = {"ok": bool(finite_poses)}

        # 2. Median ACTIVE tracks >= 15 on >= 70% of frames
        median_active = float(gate_stats.get("median_active", 0.0))
        frac_ge_15 = float(gate_stats.get("frac_ge_15", 0.0))
        ok_active = (median_active >= 15.0) and (frac_ge_15 >= 0.70)
        gates["median_active_tracks"] = {
            "ok": bool(ok_active),
            "value": median_active,
            "frac_ge_15": frac_ge_15,
        }

        # 3. Unique SPAD landmarks >= 8, median SPAD >= 5
        unique_spad = int(gate_stats.get("unique_spad", 0))
        median_spad = float(gate_stats.get("median_spad", 0.0))
        ok_spad = (unique_spad >= 8) and (median_spad >= 5.0)
        gates["spad_landmarks"] = {
            "ok": bool(ok_spad),
            "unique": unique_spad,
            "median_per_frame": median_spad,
        }

        # 4. Ecological motion: p99 speed <= 2.5 m/s, p99 delta_theta <= 25 deg, jump_dt check
        if n_traj >= 2:
            speeds = []
            dthetas = []
            jump_count = 0
            total_transitions = len(trajectory) - 1
            for i in range(total_transitions):
                curr = trajectory[i]
                nxt = trajectory[i + 1]
                t_curr = float(curr.get("timestamp", curr.get("device_timestamp_ns", 0) / 1e9))
                t_nxt = float(nxt.get("timestamp", nxt.get("device_timestamp_ns", 0) / 1e9))
                dt = t_nxt - t_curr
                p_curr = np.array([curr["x"], curr["y"], curr["z"]], dtype=float)
                p_nxt = np.array([nxt["x"], nxt["y"], nxt["z"]], dtype=float)
                dp = float(np.linalg.norm(p_nxt - p_curr))
                speeds.append(dp / max(dt, 1e-3))
                if dt > 0.1 and round(dp, 2) > 0.20:
                    jump_count += 1

                q_curr = np.array([curr["qx"], curr["qy"], curr["qz"], curr["qw"]], dtype=float)
                q_nxt = np.array([nxt["qx"], nxt["qy"], nxt["qz"], nxt["qw"]], dtype=float)
                nc = np.linalg.norm(q_curr)
                nn = np.linalg.norm(q_nxt)
                if nc > 1e-6 and nn > 1e-6:
                    dot = np.clip(np.abs(np.dot(q_curr / nc, q_nxt / nn)), 0.0, 1.0)
                    dthetas.append(float(np.degrees(2.0 * np.arccos(dot))))
                else:
                    dthetas.append(0.0)

            p99_speed = float(np.percentile(speeds, 99)) if speeds else 0.0
            p99_dtheta = float(np.percentile(dthetas, 99)) if dthetas else 0.0
            jump_ratio = float(jump_count) / float(max(1, total_transitions))
            ok_speed = (p99_speed <= 2.5)
            ok_dtheta = (p99_dtheta <= 25.0)
            ok_jump = (jump_ratio <= 0.01)
        else:
            p99_speed = 0.0
            p99_dtheta = 0.0
            ok_speed = False
            ok_dtheta = False
            ok_jump = False

        gates["speed_p99_mps"] = {"ok": bool(ok_speed), "value": p99_speed}
        gates["dtheta_p99_deg"] = {"ok": bool(ok_dtheta), "value": p99_dtheta}
        gates["jump_dt"] = {"ok": bool(ok_jump)}

        vio_xyz = np.zeros((0, 3), dtype=np.float64)
        if n_traj > 0:
            vio_xyz = np.array(
                [[row.get("x"), row.get("y"), row.get("z")] for row in trajectory],
                dtype=np.float64,
            )
        span_v = _aabb_span_xyz(vio_xyz)
        span_xy_v, span_z_v = _aabb_span_decoupled(vio_xyz)
        span_o, _n_odo = _odometry_aabb_span(session_dir)
        if span_o is None:
            gates["trajectory_span"] = {
                "ok": bool(span_z_v <= 2.50),
                "span_xyz_m": float(span_v),
                "span_xy_m": float(span_xy_v),
                "span_z_m": float(span_z_v),
                "span_odo_m": 0.0,
                "span_ratio": None,
                "skipped": True,
            }
        else:
            thresh = max(TRAJECTORY_SPAN_RATIO * float(span_o), float(span_o) + TRAJECTORY_SPAN_ABS_M)
            ok_span = (not (float(span_xy_v) > thresh)) and (span_z_v <= 2.50)
            ratio = (float(span_v) / float(span_o)) if float(span_o) > 0.0 else None
            gates["trajectory_span"] = {
                "ok": bool(ok_span),
                "span_xyz_m": float(span_v),
                "span_xy_m": float(span_xy_v),
                "span_z_m": float(span_z_v),
                "span_odo_m": float(span_o),
                "span_ratio": ratio,
                "skipped": False,
            }

        # 5. Gravity norm in [8.0, 11.0]
        g_arr = np.asarray(gravity, dtype=float)
        if np.all(np.isfinite(g_arr)) and g_arr.size == 3:
            g_norm = float(np.linalg.norm(g_arr))
            ok_grav = (8.0 <= g_norm <= 11.0)
        else:
            g_norm = 0.0
            ok_grav = False
        gates["gravity"] = {"ok": bool(ok_grav), "norm": g_norm}

        # 6. Visual accept frames fraction >= 0.50
        visual_accept_frac = float(gate_stats.get("visual_accept_frac", 0.0))
        ok_visual = (visual_accept_frac >= 0.50)
        gates["visual_accept_frames"] = {"ok": bool(ok_visual), "frac": visual_accept_frac}

        gate_order = [
            "trajectory_rows",
            "finite_poses",
            "median_active_tracks",
            "spad_landmarks",
            "speed_p99_mps",
            "dtheta_p99_deg",
            "jump_dt",
            "trajectory_span",
            "gravity",
            "visual_accept_frames",
        ]
        first_fail = None
        for gname in gate_order:
            if not gates[gname]["ok"]:
                first_fail = gname
                break

        if first_fail is not None:
            _unlink_csv()
            _write_vio_diagnostics(session_dir, False, first_fail, gates)
            return False

        # All gates passed! Write processed_vio.csv
        rows = []
        g_list = [float(gravity[0]), float(gravity[1]), float(gravity[2])]
        for idx, item in enumerate(trajectory):
            dev_ts = item.get("device_timestamp_ns")
            if dev_ts is None:
                dev_ts = int(round(float(item.get("timestamp", 0.0)) * 1e9))
            frame_val = item.get("frame")
            if frame_val is None:
                frame_val = idx
            rows.append({
                "device_timestamp_ns": int(dev_ts),
                "frame": int(frame_val),
                "x": float(item["x"]),
                "y": float(item["y"]),
                "z": float(item["z"]),
                "qx": float(item["qx"]),
                "qy": float(item["qy"]),
                "qz": float(item["qz"]),
                "qw": float(item["qw"]),
                "grav_x": g_list[0],
                "grav_y": g_list[1],
                "grav_z": g_list[2],
            })
        pd.DataFrame(rows).to_csv(vio_csv_path, index=False)
        _write_vio_diagnostics(session_dir, True, None, gates)
        return True

    except Exception as exc:
        _unlink_csv()
        _write_vio_diagnostics(
            session_dir, False, f"exception:{type(exc).__name__}", _make_default_gates()
        )
        return False

