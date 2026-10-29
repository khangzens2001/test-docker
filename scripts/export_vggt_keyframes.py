"""Export ≤44 VGGT keyframes from a session tracking island.

Producer contract (Colab checklist, spec §9):
1. Keep input filenames as vggt_filename (000000.jpg …).
2. After predictions = model(images), decode cameras with
   pose_encoding_to_extri_intri on the same tensors used for
   predictions["world_points"].
3. Write vggt/cameras.json (OpenCV w2c extrinsic_3x4 + center = -R.T @ t).
4. Write NPZ key points from those world points before align_clean_ply_to_manhattan.
5. One forward pass, 44 images, bfloat16 on Ampere+.

This repo does not run VGGT-1B. Drop Colab outputs into session_dir/vggt/.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from app.services.vggt_prior import N_VGGT_MAX, YAW_BINS
try:
    from app.pipeline.runner import load_pose_table_for_tsdf
except ImportError:
    from app.tasks.pipeline import load_pose_table_for_tsdf

logger = logging.getLogger(__name__)


@dataclass
class KeyframeExport:
    n_selected: int
    n_sent_to_vggt: int
    manifest_path: str | None
    island_frame_start: int | None
    island_frame_end: int | None
    pose_source: str | None


def extract_yaw_deg(qx: float, qy: float, qz: float, qw: float, is_server_vio: bool = False) -> float:
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    if is_server_vio:
        # VIO world (+Z up), OpenCV camera convention (+Z optical axis forward)
        fwd = R @ np.array([0.0, 0.0, 1.0])
        return float(np.degrees(np.arctan2(fwd[1], fwd[0])) % 360.0)
    else:
        # ARCore world (+Y up), OpenGL camera convention (-Z optical axis forward)
        fwd = R @ np.array([0.0, 0.0, -1.0])
        return float(np.degrees(np.arctan2(fwd[0], fwd[2])) % 360.0)


def extract_elevation_deg(qx: float, qy: float, qz: float, qw: float, is_server_vio: bool = False) -> float:
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    if is_server_vio:
        # VIO world (+Z up), OpenCV camera convention (+Z optical axis forward)
        fwd = R @ np.array([0.0, 0.0, 1.0])
        return float(np.degrees(np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))))
    else:
        # ARCore world (+Y up), OpenGL camera convention (-Z optical axis forward)
        fwd = R @ np.array([0.0, 0.0, -1.0])
        return float(np.degrees(np.arctan2(fwd[1], np.hypot(fwd[0], fwd[2]))))


def opengl_yaw_deg(qx, qy, qz, qw) -> float:
    return extract_yaw_deg(qx, qy, qz, qw, is_server_vio=False)


def _elevation_bin(elev_deg: float) -> int:
    if elev_deg < -42.0:
        return 0  # Downward: floor & floor basins
    if elev_deg > 14.0:
        return 2  # Upward: upper walls, ventilation louver, exhaust fan, shower head
    return 1      # Horizontal/mid: walls, mirror, shelves, shower mixer, door, toilet


def yaw_coverage_subsample(
    records: list[dict],
    max_n: int = N_VGGT_MAX,
    target_yaw_deg: float | None = None,
    min_target_count: int = 6,
    max_safe_gap: int = 120,
) -> list[dict]:
    kept = list(records)
    if len(kept) <= max_n:
        return kept

    target_b = None
    if target_yaw_deg is not None:
        target_b = int(float(target_yaw_deg % 360.0) // (360.0 / YAW_BINS)) % YAW_BINS

    has_all_src = len(kept) > 2 and all("source_frame_index" in r for r in kept)
    total_span = (
        int(kept[-1]["source_frame_index"]) - int(kept[0]["source_frame_index"])
        if has_all_src
        else 0
    )
    eff_safe_gap = 65 if total_span > 500 else max_safe_gap
    min_frame_gap = max(8, int(total_span / (max_n * 2.2))) if total_span > 500 else 8
    eff_min_target = 5 if total_span > 500 else min_target_count
    has_elev = any(abs(float(r.get("elevation_deg", 0.0))) > 1e-3 for r in kept)
    eb_min_quota = {0: 1, 1: 3, 2: 1}

    while len(kept) > max_n:
        bins = [[] for _ in range(YAW_BINS)]
        grid_bins = {(yb, eb): [] for yb in range(YAW_BINS) for eb in range(3)}
        for i, r in enumerate(kept):
            b = int(float(r["yaw_deg"]) // (360.0 / YAW_BINS)) % YAW_BINS
            eb = _elevation_bin(float(r.get("elevation_deg", 0.0))) if has_elev else 1
            bins[b].append(i)
            grid_bins[(b, eb)].append(i)

        cand_bins = sorted(range(YAW_BINS), key=lambda b: len(bins[b]), reverse=True)
        chosen_b = cand_bins[0]
        # Protect target bin if it has <= eff_min_target frames and another bin has frames to spare
        if target_b is not None and chosen_b == target_b and len(bins[target_b]) <= eff_min_target:
            for alt_b in cand_bins[1:]:
                if len(bins[alt_b]) > 0:
                    chosen_b = alt_b
                    break

        def _get_bin_cands(yb: int) -> list[int]:
            if not has_elev:
                return bins[yb]
            eb_order = sorted(
                range(3),
                key=lambda eb: (
                    len(grid_bins[(yb, eb)]) - eb_min_quota[eb],
                    len(grid_bins[(yb, eb)]),
                ),
                reverse=True,
            )
            for eb in eb_order:
                cands_eb = grid_bins[(yb, eb)]
                if len(cands_eb) > eb_min_quota[eb] or (
                    len(cands_eb) >= 1
                    and min(float(kept[i]["sharpness_score"]) for i in cands_eb) < 15.0
                ):
                    return cands_eb
            return bins[yb]

        bin_cands = _get_bin_cands(chosen_b)
        has_src = all("source_frame_index" in kept[i] for i in bin_cands)
        if has_src and len(kept) > 2:
            def _cand_key(i: int):
                gap_l = (
                    kept[i]["source_frame_index"] - kept[i - 1]["source_frame_index"]
                    if i > 0
                    else float("inf")
                )
                gap_r = (
                    kept[i + 1]["source_frame_index"] - kept[i]["source_frame_index"]
                    if i + 1 < len(kept)
                    else float("inf")
                )
                min_g = min(gap_l, gap_r)
                is_dup = 0 if min_g < 8 else (1 if min_g < min_frame_gap else 2)
                return (is_dup, float(kept[i]["sharpness_score"]), min_g)

            safe_cands = [
                i
                for i in bin_cands
                if 0 < i < len(kept) - 1
                and (kept[i + 1]["source_frame_index"] - kept[i - 1]["source_frame_index"] <= eff_safe_gap)
            ]
            if safe_cands:
                drop_i = min(safe_cands, key=_cand_key)
            else:
                found_alt = False
                for alt_b in cand_bins[1:]:
                    if target_b is not None and alt_b == target_b and len(bins[target_b]) <= eff_min_target:
                        continue
                    if len(bins[alt_b]) > 2:
                        alt_bin_cands = _get_bin_cands(alt_b)
                        alt_safe = [
                            i
                            for i in alt_bin_cands
                            if 0 < i < len(kept) - 1
                            and (kept[i + 1]["source_frame_index"] - kept[i - 1]["source_frame_index"] <= eff_safe_gap)
                        ]
                        if alt_safe:
                            drop_i = min(alt_safe, key=_cand_key)
                            found_alt = True
                            break
                if not found_alt:
                    all_safe = [
                        i
                        for i in bins[chosen_b]
                        if 0 < i < len(kept) - 1
                        and (kept[i + 1]["source_frame_index"] - kept[i - 1]["source_frame_index"] <= eff_safe_gap)
                    ]
                    if all_safe:
                        drop_i = min(all_safe, key=_cand_key)
                    else:
                        mid_cands = [i for i in bin_cands if 0 < i < len(kept) - 1]
                        cands = mid_cands if mid_cands else bin_cands
                        drop_i = min(cands, key=_cand_key)
        else:
            drop_i = min(bin_cands, key=lambda i: float(kept[i]["sharpness_score"]))

        kept.pop(drop_i)
    return kept


def uniform_trajectory_subsample(
    records: list[dict],
    max_n: int = N_VGGT_MAX,
) -> list[dict]:
    """Sample <=max_n keyframes with uniform linspace spacing along the trajectory."""
    kept = list(records)
    if len(kept) <= max_n:
        return kept

    indices = np.round(np.linspace(0, len(kept) - 1, max_n)).astype(int)
    return [kept[idx] for idx in indices]


class _FallbackKeyframeRecord:
    def __init__(self, frame_index: int, filename: str, sharpness: float):
        self.source_frame_index = frame_index
        self.source_filename = filename
        self.sharpness_score = sharpness


class _FallbackKeyframeResult:
    def __init__(self, records: list):
        self.kept_records = records
        self.kept_count = len(records)


def _score_frame_quality(fpath: str) -> float:
    """Compute frame quality using Laplacian sharpness variance."""
    bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
    if bgr is None:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _fallback_select_keyframes(
    staging_dir: str,
    window_translation_m: float = 0.25,
    window_rotation_deg: float = 25.0,
    max_window_frames: int = 24,
    **kwargs,
):
    odo_path = os.path.join(staging_dir, "odometry.csv")
    rgb_dir = os.path.join(staging_dir, "rgb")
    if not os.path.isfile(odo_path) or not os.path.isdir(rgb_dir):
        return _FallbackKeyframeResult([])
    df = pd.read_csv(odo_path)
    if df.empty or not {"frame", "x", "y", "z"}.issubset(df.columns):
        return _FallbackKeyframeResult([])

    eff_max_win = min(int(max_window_frames), 24) if max_window_frames else 24
    windows: list[list[tuple[int, str, str]]] = []
    cur_win: list[tuple[int, str, str]] = []
    win_start_pos = None
    win_start_quat = None

    for _, row in df.iterrows():
        if "is_frozen_interpolated" in row.index and pd.notna(row["is_frozen_interpolated"]) and int(row["is_frozen_interpolated"]) != 0:
            continue
        f_idx = int(row["frame"])
        if f_idx < 0:
            continue
        fname = f"{f_idx:06d}.jpg"
        fpath = os.path.join(rgb_dir, fname)
        if not os.path.isfile(fpath):
            continue

        pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        quat = np.array(
            [
                float(row.get("qx", 0.0)),
                float(row.get("qy", 0.0)),
                float(row.get("qz", 0.0)),
                float(row.get("qw", 1.0)),
            ]
        )

        if win_start_pos is None:
            win_start_pos = pos
            win_start_quat = quat
            cur_win.append((f_idx, fname, fpath))
        else:
            dist = float(np.linalg.norm(pos - win_start_pos))
            cos_half = np.clip(np.abs(np.dot(quat, win_start_quat)), 0.0, 1.0)
            rot_deg = float(np.degrees(2.0 * np.arccos(cos_half)))
            if (
                dist >= window_translation_m
                or rot_deg >= window_rotation_deg
                or len(cur_win) >= eff_max_win
            ):
                windows.append(cur_win)
                cur_win = [(f_idx, fname, fpath)]
                win_start_pos = pos
                win_start_quat = quat
            else:
                cur_win.append((f_idx, fname, fpath))

    if cur_win:
        windows.append(cur_win)

    records = []
    last_selected_frame = -999
    for win in windows:
        cands = [item for item in win if item[0] - last_selected_frame >= 8]
        if not cands:
            cands = win
        best_item = None
        best_score = -1.0
        for item in cands:
            sc = _score_frame_quality(item[2])
            if sc > best_score:
                best_score = sc
                best_item = item
        if best_item is not None:
            records.append(_FallbackKeyframeRecord(best_item[0], best_item[1], best_score))
            last_selected_frame = best_item[0]

    return _FallbackKeyframeResult(records)


def _select_keyframes_dry(session_dir, output_dir=None, **kwargs):
    return _fallback_select_keyframes(session_dir, **kwargs)


def _stage_island(session_dir: str, pose_df: pd.DataFrame, staging: str) -> None:
    rgb_src = os.path.join(session_dir, "rgb")
    rgb_dst = os.path.join(staging, "rgb")
    os.makedirs(rgb_dst, exist_ok=True)
    if not os.path.isdir(rgb_src):
        return

    if "frame" in pose_df.columns:
        frames = set(int(f) for f in pose_df["frame"].to_numpy() if np.isfinite(f) and int(f) >= 0)
    else:
        frames = None

    pose_df.to_csv(os.path.join(staging, "odometry.csv"), index=False)

    for name in os.listdir(rgb_src):
        src = os.path.join(rgb_src, name)
        if not os.path.isfile(src):
            continue
        stem, ext = os.path.splitext(name)
        try:
            idx = int(stem.replace("frame_", ""))
        except ValueError:
            continue
        if frames is not None and idx not in frames:
            continue
        dst_name = f"{idx:06d}{ext.lower() if ext else '.jpg'}"
        shutil.copy2(src, os.path.join(rgb_dst, dst_name))


def compute_spherical_fov_grid_dims(
    session_dir: str | None = None,
    K: np.ndarray | None = None,
    img_h: int | None = None,
    img_w: int | None = None,
    rot_deg: int | None = None,
) -> tuple[int, int, float]:
    """Derive spherical grid (N_yaw, N_pitch, cone_half_angle_deg) from camera intrinsics K
    and upright image dimensions (H_up, W_up) for target MVS overlap rho_yaw=0.43, rho_pitch=0.64.
    Defaults cleanly to (12, 5, 25.0) when intrinsics are unavailable."""
    default_grid = (12, 5, 25.0)
    try:
        if K is None and session_dir is not None:
            k_path = os.path.join(session_dir, "camera_matrix.csv")
            if os.path.isfile(k_path):
                k_arr = np.loadtxt(k_path, delimiter=",", dtype=float)
                if k_arr.shape == (3, 3):
                    K = k_arr
        if K is None or K.shape != (3, 3):
            return default_grid

        if rot_deg is None and session_dir is not None:
            man_p = os.path.join(session_dir, "manifest.json")
            if os.path.isfile(man_p):
                with open(man_p, "r", encoding="utf-8") as mf:
                    rot_deg = int(json.load(mf).get("camera_sensor_to_display_rotation_deg", 90) or 90) % 360
        if rot_deg is None:
            rot_deg = 90

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        if fx <= 10.0 or fy <= 10.0:
            return default_grid

        if (img_w is None or img_w <= 16 or img_h is None or img_h <= 16) and session_dir is not None:
            rgb_dir_probe = os.path.join(session_dir, "rgb")
            if os.path.isdir(rgb_dir_probe):
                try:
                    for fn_p in sorted(os.listdir(rgb_dir_probe)):
                        if fn_p.lower().endswith((".jpg", ".jpeg", ".png")):
                            with Image.open(os.path.join(rgb_dir_probe, fn_p)) as im_p:
                                w_p, h_p = im_p.size
                                if w_p > 16 and h_p > 16:
                                    img_w, img_h = w_p, h_p
                            break
                except Exception:
                    pass
        w_sens = float(img_w) if (img_w is not None and img_w > 16) else (2.0 * cx if cx > 10.0 else 640.0)
        h_sens = float(img_h) if (img_h is not None and img_h > 16) else (2.0 * cy if cy > 10.0 else 480.0)
        if rot_deg in (90, 270):
            w_up, h_up = h_sens, w_sens
            f_up_x, f_up_y = fy, fx
        else:
            w_up, h_up = w_sens, h_sens
            f_up_x, f_up_y = fx, fy

        fov_yaw_deg = float(np.degrees(2.0 * np.arctan(w_up / (2.0 * f_up_x))))
        fov_pitch_deg = float(np.degrees(2.0 * np.arctan(h_up / (2.0 * f_up_y))))
        if not (25.0 <= fov_yaw_deg <= 110.0 and 25.0 <= fov_pitch_deg <= 110.0):
            return default_grid

        step_yaw = (1.0 - 0.43) * fov_yaw_deg
        step_pitch = (1.0 - 0.64) * fov_pitch_deg
        n_yaw = int(np.clip(round(360.0 / max(step_yaw, 15.0)), 8, 16))
        n_pitch = int(np.clip(round(120.0 / max(step_pitch, 12.0)), 3, 7))
        # Normalize to standard 12x5 when within typical smartphone FoV range [38..56] x [58..72]
        if 38.0 <= fov_yaw_deg <= 56.0:
            n_yaw = 12
        if 58.0 <= fov_pitch_deg <= 72.0:
            n_pitch = 5
        cone_deg = float(np.clip(0.5 * fov_yaw_deg, 20.0, 28.0))
        if abs(cone_deg - 25.0) <= 2.5:
            cone_deg = 25.0
        return n_yaw, n_pitch, cone_deg
    except Exception:
        return default_grid


def select_two_tier_spherical_keyframes(
    session_dir: str,
    pose_df: pd.DataFrame,
    n_vggt: int = 72,
    n_tsdf: int = 160,
    sigma_omega: float = 0.60,
    sigma_vel: float = 0.35,
    omega_max: float = 1.20,
    vel_max: float = 0.80,
    lambda_revisit: float = 0.35,
    t_drift_s: float = 20.0,
) -> tuple[list[dict], list[dict]]:
    """Zero-copy Two-Tier Spherical FoV (12x5) + Kinematic Motion-Blur Gate +
    Temporal-Locality Revisit Penalty keyframe selector.
    Returns (vggt_keyframes, tsdf_keyframes) with vggt_keyframes ⊂ tsdf_keyframes."""
    rgb_dir = os.path.join(session_dir, "rgb")
    if pose_df is None or pose_df.empty or not os.path.isdir(rgb_dir):
        return [], []

    is_server_vio = pose_df.attrs.get("pose_source") == "server_vio"
    n_yaw_bins, n_pitch_bins, cone_deg = compute_spherical_fov_grid_dims(session_dir=session_dir)
    cos_cone = float(np.cos(np.radians(cone_deg)))

    # Map available RGB files without copying to /tmp
    rgb_by_frame: dict[int, tuple[str, str]] = {}
    rgb_by_ts: dict[int, tuple[str, str]] = {}
    for fname in os.listdir(rgb_dir):
        fpath = os.path.join(rgb_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stem, _ = os.path.splitext(fname)
        try:
            if stem.startswith("frame_"):
                rgb_by_ts[int(stem.replace("frame_", ""))] = (fname, fpath)
            else:
                rgb_by_frame[int(stem)] = (fname, fpath)
        except ValueError:
            continue

    if not rgb_by_frame and not rgb_by_ts:
        return [], []

    # Load dToF status==5 zone count per timestamp if processed_lidar.csv exists
    lidar_ts = None
    lidar_s5_counts = None
    lidar_path = os.path.join(session_dir, "processed_lidar.csv")
    if os.path.isfile(lidar_path) and os.path.getsize(lidar_path) > 0:
        try:
            ldf = pd.read_csv(lidar_path)
            ts_c = (
                "device_timestamp_ns"
                if "device_timestamp_ns" in ldf.columns
                else ("timestamp" if "timestamp" in ldf.columns else ldf.columns[0])
            )
            stat_cols = [f"status_{z}" for z in range(64) if f"status_{z}" in ldf.columns]
            dist_cols = [f"distance_{z}" for z in range(64) if f"distance_{z}" in ldf.columns]
            if len(dist_cols) == 64:
                lidar_ts = ldf[ts_c].to_numpy(dtype=np.float64)
                d_mat = ldf[dist_cols].to_numpy(dtype=np.float64)
                if np.nanmedian(d_mat) > 20.0:
                    d_mat = d_mat / 1000.0
                valid_d = np.isfinite(d_mat) & (d_mat > 0.15) & (d_mat < 4.5)
                if len(stat_cols) == 64:
                    s_mat = ldf[stat_cols].to_numpy(dtype=np.float64)
                    valid_s5 = valid_d & (s_mat == 5)
                else:
                    valid_s5 = valid_d
                lidar_s5_counts = np.sum(valid_s5, axis=1).astype(np.float64)
        except Exception:
            lidar_ts = None
            lidar_s5_counts = None

    df = pose_df.reset_index(drop=True)
    n_rows = len(df)
    if "device_timestamp_ns" in df.columns:
        ts_s = df["device_timestamp_ns"].to_numpy(dtype=np.float64) * 1e-9
        ts_ns_arr = df["device_timestamp_ns"].to_numpy(dtype=np.int64)
    elif "timestamp" in df.columns:
        raw_t = df["timestamp"].to_numpy(dtype=np.float64)
        ts_s = raw_t * 1e-9 if (len(raw_t) > 0 and raw_t[0] > 1e12) else raw_t
        ts_ns_arr = np.rint(ts_s * 1e9).astype(np.int64)
    else:
        ts_s = np.arange(n_rows, dtype=np.float64) / 30.0
        ts_ns_arr = np.rint(ts_s * 1e9).astype(np.int64)

    pos_arr = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    quat_arr = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    q_norms = np.linalg.norm(quat_arr, axis=1, keepdims=True)
    quat_arr = np.where(q_norms > 1e-9, quat_arr / np.maximum(q_norms, 1e-9), np.array([0.0, 0.0, 0.0, 1.0]))
    rots = Rotation.from_quat(quat_arr)
    rot_mats = rots.as_matrix()

    # Compute angular velocity ||omega_i|| (rad/s) and linear velocity ||v_i|| (m/s)
    omega_arr = np.zeros(n_rows, dtype=np.float64)
    vel_arr = np.zeros(n_rows, dtype=np.float64)
    if n_rows >= 2:
        dt = np.maximum(np.diff(ts_s), 1e-3)
        dpos = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1) / dt
        vel_arr[:-1] = dpos
        vel_arr[-1] = dpos[-1]
        rel_rots = rots[:-1].inv() * rots[1:]
        dang = rel_rots.magnitude() / dt
        omega_arr[:-1] = dang
        omega_arr[-1] = dang[-1]

    # Segment trajectory into candidate windows (0.18m translation, 15 deg rotation, or 18 frames)
    windows: list[list[int]] = []
    cur_win: list[int] = []
    w_pos0 = None
    w_quat0 = None
    has_frame_col = "frame" in df.columns
    frame_col = (
        pd.to_numeric(df["frame"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        if has_frame_col
        else np.arange(n_rows, dtype=np.int64)
    )
    frozen_col = (
        pd.to_numeric(df["is_frozen_interpolated"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        if "is_frozen_interpolated" in df.columns
        else np.zeros(n_rows, dtype=np.int64)
    )

    for r_i in range(n_rows):
        if frozen_col[r_i] != 0:
            continue
        f_idx = int(frame_col[r_i])
        if has_frame_col and f_idx < 0:
            continue
        t_ns = int(ts_ns_arr[r_i])
        if f_idx not in rgb_by_frame and t_ns not in rgb_by_ts:
            continue
        p_i = pos_arr[r_i]
        q_i = quat_arr[r_i]
        if w_pos0 is None:
            w_pos0 = p_i
            w_quat0 = q_i
            cur_win.append(r_i)
        else:
            dist = float(np.linalg.norm(p_i - w_pos0))
            cos_h = float(np.clip(abs(np.dot(q_i, w_quat0)), 0.0, 1.0))
            r_deg = float(np.degrees(2.0 * np.arccos(cos_h)))
            if dist >= 0.18 or r_deg >= 15.0 or len(cur_win) >= 18:
                windows.append(cur_win)
                cur_win = [r_i]
                w_pos0 = p_i
                w_quat0 = q_i
            else:
                cur_win.append(r_i)
    if cur_win:
        windows.append(cur_win)

    if not windows:
        return [], []

    # Pick top candidates per window (preferring frames within hard kinematic limits omega_max, vel_max)
    cand_rows: list[int] = []
    for win in windows:
        kin_ok = [r_i for r_i in win if omega_arr[r_i] <= omega_max and vel_arr[r_i] <= vel_max]
        pool = kin_ok if kin_ok else win
        # Sample up to 3 evenly spaced frames from each micro-window to evaluate Laplacian sharpness
        if len(pool) <= 3:
            cand_rows.extend(pool)
        else:
            idxs = np.unique(np.linspace(0, len(pool) - 1, 3, dtype=int))
            cand_rows.extend([pool[ii] for ii in idxs])

    cand_rows = sorted(set(cand_rows))
    candidates: list[dict] = []
    raw_sharps: list[float] = []

    for r_i in cand_rows:
        f_idx = int(frame_col[r_i])
        t_ns = int(ts_ns_arr[r_i])
        fname, fpath = rgb_by_frame.get(f_idx) or rgb_by_ts.get(t_ns, (None, None))
        if fpath is None:
            continue
        gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape[0] < 16 or gray.shape[1] < 16:
            continue
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var <= 1e-4:
            continue
        raw_sharps.append(lap_var)

        R_m = rot_mats[r_i]
        fwd = R_m @ (np.array([0.0, 0.0, 1.0]) if is_server_vio else np.array([0.0, 0.0, -1.0]))
        fnorm = float(np.linalg.norm(fwd))
        if fnorm > 1e-9:
            fwd = fwd / fnorm
        if is_server_vio:
            yaw_deg = float(np.degrees(np.arctan2(fwd[1], fwd[0])) % 360.0)
            pitch_deg = float(np.degrees(np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))))
        else:
            yaw_deg = float(np.degrees(np.arctan2(fwd[0], fwd[2])) % 360.0)
            pitch_deg = float(np.degrees(np.arctan2(fwd[1], np.hypot(fwd[0], fwd[2]))))

        yb = int(yaw_deg // (360.0 / n_yaw_bins)) % n_yaw_bins
        pitch_clamped = float(np.clip(pitch_deg, -59.99, 59.99))
        pb = int((pitch_clamped + 60.0) // (120.0 / n_pitch_bins))
        pb = int(np.clip(pb, 0, n_pitch_bins - 1))

        s5_cnt = 56.0
        if lidar_ts is not None and lidar_s5_counts is not None and len(lidar_ts) > 0:
            l_idx = int(np.argmin(np.abs(lidar_ts - float(t_ns))))
            s5_cnt = float(lidar_s5_counts[l_idx])

        candidates.append({
            "row_idx": r_i,
            "source_frame_index": f_idx,
            "source_filename": fname,
            "rgb_path": fpath,
            "device_timestamp_ns": t_ns,
            "time_s": float(ts_s[r_i]),
            "x": float(pos_arr[r_i, 0]),
            "y": float(pos_arr[r_i, 1]),
            "z": float(pos_arr[r_i, 2]),
            "qx": float(quat_arr[r_i, 0]),
            "qy": float(quat_arr[r_i, 1]),
            "qz": float(quat_arr[r_i, 2]),
            "qw": float(quat_arr[r_i, 3]),
            "fwd": fwd,
            "yaw_deg": yaw_deg,
            "elevation_deg": pitch_deg,
            "yb": yb,
            "pb": pb,
            "sharpness_score": lap_var,
            "omega": float(omega_arr[r_i]),
            "vel": float(vel_arr[r_i]),
            "s5_count": s5_cnt,
            "kin_hard_ok": bool(omega_arr[r_i] <= omega_max and vel_arr[r_i] <= vel_max),
        })

    if not candidates:
        return [], []

    s_arr = np.asarray(raw_sharps, dtype=np.float64)
    p5 = float(np.percentile(s_arr, 5.0))
    p95 = float(np.percentile(s_arr, 95.0))
    s_span = max(p95 - p5, 1e-6)

    for c in candidates:
        s_norm = float(np.clip((c["sharpness_score"] - p5) / s_span, 0.0, 1.0))
        w_rot = float(np.exp(-(c["omega"] ** 2) / (2.0 * (sigma_omega ** 2))))
        w_vel = float(np.exp(-(c["vel"] ** 2) / (2.0 * (sigma_vel ** 2))))
        w_tof = float(np.clip(c["s5_count"] / 64.0, 0.0, 1.0))
        q_score = 0.45 * s_norm + 0.25 * w_rot + 0.15 * w_vel + 0.15 * w_tof
        if not c["kin_hard_ok"]:
            q_score *= 0.35
        c["q_score"] = q_score

    # Identify dominant scan loop time center to anchor temporal locality
    all_times = np.array([c["time_s"] for c in candidates], dtype=np.float64)
    t_med = float(np.median(all_times))

    def _effective_score(cand: dict, selected_list: list[dict], min_frame_gap: int) -> float:
        f_c = cand["source_frame_index"]
        for s in selected_list:
            if abs(f_c - s["source_frame_index"]) < min_frame_gap:
                return -1e9
        if not selected_list:
            return cand["q_score"] - 0.05 * float(
                1.0 - np.exp(-((cand["time_s"] - t_med) ** 2) / (2.0 * (t_drift_s ** 2)))
            )
        fwd_c = cand["fwd"]
        t_c = cand["time_s"]
        max_dt_cone = 0.0
        has_cone_nb = False
        for s in selected_list:
            if float(np.dot(fwd_c, s["fwd"])) >= cos_cone:
                has_cone_nb = True
                dt_s = abs(t_c - s["time_s"])
                if dt_s > max_dt_cone:
                    max_dt_cone = dt_s
        if not has_cone_nb:
            # Weak global temporal locality relative to nearest selected in time
            min_dt_any = min(abs(t_c - s["time_s"]) for s in selected_list)
            pen = 0.10 * float(1.0 - np.exp(-(min_dt_any ** 2) / (2.0 * (t_drift_s ** 2))))
            return cand["q_score"] - pen
        revisit_pen = lambda_revisit * float(
            1.0 - np.exp(-(max_dt_cone ** 2) / (2.0 * (t_drift_s ** 2)))
        )
        return cand["q_score"] - revisit_pen

    # Group candidates by spherical bin (yb, pb)
    bins_map: dict[tuple[int, int], list[int]] = {}
    for idx_c, c in enumerate(candidates):
        bins_map.setdefault((c["yb"], c["pb"]), []).append(idx_c)

    selected_vggt: list[dict] = []
    selected_idx_set: set[int] = set()

    # Pass 1: Cover every populated spherical bin (12 x 5 = up to 60 bins) with its highest effective-score frame
    # Order bins from most-populated/central pitch to ensure the temporal anchor starts cleanly
    bin_keys = sorted(
        bins_map.keys(),
        key=lambda k: (-len(bins_map[k]), abs(k[1] - (n_pitch_bins // 2)), k[0]),
    )
    for bkey in bin_keys:
        if len(selected_vggt) >= n_vggt:
            break
        c_indices = [i for i in bins_map[bkey] if i not in selected_idx_set]
        if not c_indices:
            continue
        best_i = max(c_indices, key=lambda i: _effective_score(candidates[i], selected_vggt, min_frame_gap=5))
        if _effective_score(candidates[best_i], selected_vggt, min_frame_gap=3) > -1e8:
            selected_vggt.append(candidates[best_i])
            selected_idx_set.add(best_i)

    # Pass 2: Fill remaining budget up to n_vggt (72) by round-robin across yaw sectors and pitch bins
    while len(selected_vggt) < min(n_vggt, len(candidates)):
        # Count current per-yaw-bin and per-spherical-bin coverage
        yb_counts = np.zeros(n_yaw_bins, dtype=int)
        sb_counts: dict[tuple[int, int], int] = {k: 0 for k in bins_map}
        for s in selected_vggt:
            yb_counts[s["yb"]] += 1
            sb_counts[(s["yb"], s["pb"])] = sb_counts.get((s["yb"], s["pb"]), 0) + 1

        rem_indices = [i for i in range(len(candidates)) if i not in selected_idx_set]
        if not rem_indices:
            break

        def _tier1_fill_key(i: int) -> tuple[float, float]:
            c = candidates[i]
            eff_q = _effective_score(c, selected_vggt, min_frame_gap=5)
            if eff_q <= -1e8:
                return (-1e9, -1e9)
            sc_cnt = sb_counts.get((c["yb"], c["pb"]), 0)
            yb_cnt = int(yb_counts[c["yb"]])
            # Prioritize spherical bins with only 1 keyframe and less-dense yaw sectors
            rarity_bonus = 0.50 / (1.0 + sc_cnt) + 0.25 / (1.0 + yb_cnt)
            return (eff_q + rarity_bonus, eff_q)

        best_i = max(rem_indices, key=_tier1_fill_key)
        if _tier1_fill_key(best_i)[0] <= -1e8:
            # Relax frame gap if needed
            best_i = max(rem_indices, key=lambda i: _effective_score(candidates[i], selected_vggt, min_frame_gap=1))
        selected_vggt.append(candidates[best_i])
        selected_idx_set.add(best_i)

    # Tier 2: Expand selected_vggt (72) -> selected_tsdf (160) so every spherical bin gets >= 2 keyframes
    selected_tsdf = list(selected_vggt)
    while len(selected_tsdf) < min(n_tsdf, len(candidates)):
        sb_counts_2: dict[tuple[int, int], int] = {k: 0 for k in bins_map}
        yb_counts_2 = np.zeros(n_yaw_bins, dtype=int)
        for s in selected_tsdf:
            sb_counts_2[(s["yb"], s["pb"])] = sb_counts_2.get((s["yb"], s["pb"]), 0) + 1
            yb_counts_2[s["yb"]] += 1

        rem_indices = [i for i in range(len(candidates)) if i not in selected_idx_set]
        if not rem_indices:
            break

        def _tier2_fill_key(i: int) -> tuple[int, float]:
            c = candidates[i]
            eff_q = _effective_score(c, selected_tsdf, min_frame_gap=3)
            if eff_q <= -1e8:
                return (-999, -1e9)
            sc_cnt = sb_counts_2.get((c["yb"], c["pb"]), 0)
            needs_second = 1 if sc_cnt < 2 else 0
            rarity = 0.35 / (1.0 + sc_cnt) + 0.15 / (1.0 + int(yb_counts_2[c["yb"]]))
            return (needs_second, eff_q + rarity)

        best_i = max(rem_indices, key=_tier2_fill_key)
        if _tier2_fill_key(best_i)[0] == -999:
            best_i = max(rem_indices, key=lambda i: candidates[i]["q_score"])
        selected_tsdf.append(candidates[best_i])
        selected_idx_set.add(best_i)

    selected_vggt.sort(key=lambda r: int(r["source_frame_index"]))
    selected_tsdf.sort(key=lambda r: int(r["source_frame_index"]))
    return selected_vggt, selected_tsdf


def export_vggt_keyframes(
    session_dir: str,
    subsample_method: str = "yaw",
    prefer_server_vio: bool = True,
    max_n: int | None = None,
) -> KeyframeExport:
    pose_df = load_pose_table_for_tsdf(session_dir, prefer_server_vio=prefer_server_vio)
    empty = KeyframeExport(0, 0, None, None, None, pose_df.attrs.get("pose_source") if hasattr(pose_df, "attrs") else None)
    if pose_df is None or pose_df.empty:
        return empty
    island = pose_df.attrs.get("selected_island_frames")
    island_start = int(island[0]) if island else None
    island_end = int(island[1]) if island else None

    if subsample_method == "spherical":
        eff_max_n = int(max_n) if (max_n is not None and int(max_n) > 0) else 72
        eff_tsdf_n = max(160, eff_max_n)
        vggt_kfs, tsdf_kfs = select_two_tier_spherical_keyframes(
            session_dir, pose_df, n_vggt=eff_max_n, n_tsdf=eff_tsdf_n
        )
        if not vggt_kfs:
            return empty
        img_dir = os.path.join(session_dir, "keyframes_selected", "images")
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)
        frames_out = []
        for i, rec in enumerate(vggt_kfs):
            vggt_filename = f"{i:06d}.jpg"
            dst_path = os.path.join(img_dir, vggt_filename)
            shutil.copy2(rec["rgb_path"], dst_path)
            frames_out.append({
                "vggt_filename": vggt_filename,
                "source_filename": rec["source_filename"],
                "source_frame_index": int(rec["source_frame_index"]),
                "device_timestamp_ns": int(rec["device_timestamp_ns"]),
                "x": float(rec["x"]),
                "y": float(rec["y"]),
                "z": float(rec["z"]),
                "qx": float(rec["qx"]),
                "qy": float(rec["qy"]),
                "qz": float(rec["qz"]),
                "qw": float(rec["qw"]),
            })
        vggt_dir = os.path.join(session_dir, "vggt")
        os.makedirs(vggt_dir, exist_ok=True)
        man_path = os.path.join(vggt_dir, "keyframe_manifest.json")
        payload = {
            "pose_source": pose_df.attrs.get("pose_source"),
            "island_frame_start": island_start,
            "island_frame_end": island_end,
            "n_selected": len(tsdf_kfs),
            "n_sent_to_vggt": len(frames_out),
            "tsdf_source_frame_indices": [int(r["source_frame_index"]) for r in tsdf_kfs],
            "frames": frames_out,
        }
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return KeyframeExport(
            n_selected=len(tsdf_kfs),
            n_sent_to_vggt=len(frames_out),
            manifest_path=man_path,
            island_frame_start=island_start,
            island_frame_end=island_end,
            pose_source=pose_df.attrs.get("pose_source"),
        )

    eff_limit = int(max_n) if (max_n is not None and int(max_n) > 0) else N_VGGT_MAX
    with tempfile.TemporaryDirectory(prefix="vggt_kf_") as staging:
        _stage_island(session_dir, pose_df, staging)
        staged_rgb = os.path.join(staging, "rgb")
        if not os.path.isdir(staged_rgb) or not os.listdir(staged_rgb):
            return empty
        result = _select_keyframes_dry(
            staging,
            output_dir=os.path.join(staging, "keyframes_selected"),
            window_translation_m=0.25,
            window_rotation_deg=25.0,
            overlap=0.0,
            max_window_frames=35,
        )
        kept = sorted(
            [r for r in result.kept_records],
            key=lambda r: int(getattr(r, "source_frame_index", 0)),
        )
        n_selected = len(kept)
        odo_path = os.path.join(staging, "odometry.csv")
        odo = pd.read_csv(odo_path) if os.path.isfile(odo_path) else pd.DataFrame()
        odo_by_frame = {}
        if not odo.empty and "frame" in odo.columns:
            for _, row in odo.iterrows():
                f_val = row["frame"]
                if np.isfinite(f_val) and int(f_val) >= 0:
                    odo_by_frame[int(f_val)] = row
        recs = []
        is_server_vio = (pose_df.attrs.get("pose_source") == "server_vio")
        for r in kept:
            row = odo_by_frame.get(int(r.source_frame_index))
            if row is None:
                yaw = 0.0
                elev = 0.0
            else:
                yaw = extract_yaw_deg(row["qx"], row["qy"], row["qz"], row["qw"], is_server_vio=is_server_vio)
                elev = extract_elevation_deg(row["qx"], row["qy"], row["qz"], row["qw"], is_server_vio=is_server_vio)
            recs.append(
                {
                    "record": r,
                    "yaw_deg": yaw,
                    "elevation_deg": elev,
                    "sharpness_score": float(getattr(r, "sharpness_score", 0.0)),
                    "source_frame_index": int(r.source_frame_index),
                    "source_filename": r.source_filename,
                }
            )
        target_yaw = None
        if len(pose_df) >= 2 and {"x", "y"}.issubset(pose_df.columns):
            dx = float(pose_df["x"].iloc[-1] - pose_df["x"].iloc[0])
            dy = float(pose_df["y"].iloc[-1] - pose_df["y"].iloc[0])
            if np.hypot(dx, dy) >= 0.3:
                target_yaw = float(np.degrees(np.arctan2(dy, dx)) % 360.0) if is_server_vio else float(np.degrees(np.arctan2(dx, dy)) % 360.0)
            else:
                try:
                    xy = pose_df[["x", "y"]].to_numpy(dtype=float)
                    xy_valid = xy[np.all(np.isfinite(xy), axis=1)]
                    if len(xy_valid) >= 5:
                        xy_c = xy_valid - xy_valid.mean(axis=0)
                        _, _, Vt = np.linalg.svd(xy_c, full_matrices=False)
                        v_pca = Vt[0]
                        if np.hypot(v_pca[0], v_pca[1]) > 1e-4:
                            target_yaw = float(np.degrees(np.arctan2(v_pca[1], v_pca[0])) % 360.0) if is_server_vio else float(np.degrees(np.arctan2(v_pca[0], v_pca[1])) % 360.0)
                except Exception:
                    target_yaw = None

        if subsample_method == "yaw":
            sent = (
                yaw_coverage_subsample(recs, max_n=eff_limit, target_yaw_deg=target_yaw, min_target_count=8)
                if n_selected > eff_limit
                else recs
            )
        else:
            sent = (
                uniform_trajectory_subsample(recs, max_n=eff_limit)
                if n_selected > eff_limit
                else recs
            )

        img_dir = os.path.join(session_dir, "keyframes_selected", "images")
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)
        pose_by_frame = {}
        if "frame" in pose_df.columns:
            for _, row in pose_df.iterrows():
                f_val = row["frame"]
                if np.isfinite(f_val) and int(f_val) >= 0:
                    pose_by_frame[int(f_val)] = row
        frames_out = []
        for i, rec in enumerate(sent):
            vggt_filename = f"{i:06d}.jpg"
            src_idx = rec["source_frame_index"]
            src_name = rec["source_filename"]
            src_path = os.path.join(staging, "rgb", src_name)
            if not os.path.isfile(src_path):
                src_path = os.path.join(staging, "rgb", f"{src_idx:06d}.jpg")
            dst_path = os.path.join(img_dir, vggt_filename)
            shutil.copy2(src_path, dst_path)

            prow = pose_by_frame.get(src_idx)
            if prow is None and len(pose_df):
                prow = pose_df.iloc[min(i, len(pose_df) - 1)]
            frames_out.append(
                {
                    "vggt_filename": vggt_filename,
                    "source_filename": src_name,
                    "source_frame_index": src_idx,
                    "x": float(prow["x"]) if prow is not None else 0.0,
                    "y": float(prow["y"]) if prow is not None else 0.0,
                    "z": float(prow["z"]) if prow is not None else 0.0,
                    "qx": float(prow["qx"]) if prow is not None else 0.0,
                    "qy": float(prow["qy"]) if prow is not None else 0.0,
                    "qz": float(prow["qz"]) if prow is not None else 0.0,
                    "qw": float(prow["qw"]) if prow is not None else 1.0,
                }
            )
    vggt_dir = os.path.join(session_dir, "vggt")
    os.makedirs(vggt_dir, exist_ok=True)
    man_path = os.path.join(vggt_dir, "keyframe_manifest.json")
    payload = {
        "pose_source": pose_df.attrs.get("pose_source"),
        "island_frame_start": island_start,
        "island_frame_end": island_end,
        "n_selected": n_selected,
        "n_sent_to_vggt": len(frames_out),
        "frames": frames_out,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return KeyframeExport(
        n_selected=n_selected,
        n_sent_to_vggt=len(frames_out),
        manifest_path=man_path,
        island_frame_start=island_start,
        island_frame_end=island_end,
        pose_source=pose_df.attrs.get("pose_source"),
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_dir")
    p.add_argument("--method", default="spherical", choices=["spherical", "yaw", "uniform"])
    p.add_argument("--max-n", type=int, default=72)
    args = p.parse_args()
    out = export_vggt_keyframes(args.session_dir, subsample_method=args.method, max_n=args.max_n)
    print(f"selected={out.n_selected} sent={out.n_sent_to_vggt} manifest={out.manifest_path}")


if __name__ == "__main__":
    main()

