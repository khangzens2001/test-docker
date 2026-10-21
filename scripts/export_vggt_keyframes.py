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
                gap_l = kept[i]["source_frame_index"] - kept[i - 1]["source_frame_index"]
                gap_r = kept[i + 1]["source_frame_index"] - kept[i]["source_frame_index"]
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
    """Compute composite frame quality combining Laplacian sharpness with unclipped colored object completeness."""
    bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
    if bgr is None:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    small = cv2.cvtColor(bgr[::4, ::4], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = small.shape[:2]
    if h < 8 or w < 8:
        return sharp
    sat = np.ptp(small, axis=2)
    rg = small[:, :, 0] - small[:, :, 1]
    warm = (rg > 0.14) | ((sat > 0.18) & (small[:, :, 0] > small[:, :, 2]) & (rg > 0.08))
    n_warm = int(np.count_nonzero(warm))
    warm_frac = n_warm / float(h * w)
    if warm_frac >= 0.04:
        bdr = np.zeros((h, w), dtype=bool)
        bdr[:3, :] = True
        bdr[-3:, :] = True
        bdr[:, :3] = True
        bdr[:, -3:] = True
        n_bdr = int(np.count_nonzero(warm & bdr))
        bdr_ratio = n_bdr / max(1.0, float(n_warm))
        ys, xs = np.nonzero(warm)
        cy = float(np.mean(ys)) / h - 0.5
        cx = float(np.mean(xs)) / w - 0.5
        center_bonus = float(np.exp(-4.0 * (cx * cx + cy * cy)))
        completeness = (warm_frac ** 2.0) * np.exp(-35.0 * bdr_ratio) * center_bonus
        return float(sharp * np.exp(-25.0 * bdr_ratio) + 4000.0 * completeness)
    return sharp


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
        f_idx = int(row["frame"])
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
        frames = set(int(f) for f in pose_df["frame"].to_numpy() if np.isfinite(f))
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


def export_vggt_keyframes(
    session_dir: str,
    subsample_method: str = "yaw",
    prefer_server_vio: bool = True,
) -> KeyframeExport:
    pose_df = load_pose_table_for_tsdf(session_dir, prefer_server_vio=prefer_server_vio)
    empty = KeyframeExport(0, 0, None, None, None, pose_df.attrs.get("pose_source") if hasattr(pose_df, "attrs") else None)
    if pose_df is None or pose_df.empty:
        return empty
    island = pose_df.attrs.get("selected_island_frames")
    island_start = int(island[0]) if island else None
    island_end = int(island[1]) if island else None
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
                odo_by_frame[int(row["frame"])] = row
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
                yaw_coverage_subsample(recs, max_n=N_VGGT_MAX, target_yaw_deg=target_yaw, min_target_count=8)
                if n_selected > N_VGGT_MAX
                else recs
            )
        else:
            sent = (
                uniform_trajectory_subsample(recs, max_n=N_VGGT_MAX)
                if n_selected > N_VGGT_MAX
                else recs
            )

        img_dir = os.path.join(session_dir, "keyframes_selected", "images")
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)
        pose_by_frame = {}
        if "frame" in pose_df.columns:
            for _, row in pose_df.iterrows():
                pose_by_frame[int(row["frame"])] = row
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
    args = p.parse_args()
    out = export_vggt_keyframes(args.session_dir)
    print(f"selected={out.n_selected} sent={out.n_sent_to_vggt} manifest={out.manifest_path}")


if __name__ == "__main__":
    main()
