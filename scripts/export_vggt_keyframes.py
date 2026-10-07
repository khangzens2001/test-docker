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


def opengl_yaw_deg(qx, qy, qz, qw) -> float:
    return extract_yaw_deg(qx, qy, qz, qw, is_server_vio=False)


def yaw_coverage_subsample(
    records: list[dict],
    max_n: int = N_VGGT_MAX,
    target_yaw_deg: float | None = None,
    min_target_count: int = 6,
) -> list[dict]:
    kept = list(records)
    if len(kept) <= max_n:
        return kept

    target_b = None
    if target_yaw_deg is not None:
        target_b = int(float(target_yaw_deg % 360.0) // (360.0 / YAW_BINS)) % YAW_BINS

    while len(kept) > max_n:
        bins = [[] for _ in range(YAW_BINS)]
        for i, r in enumerate(kept):
            b = int(float(r["yaw_deg"]) // (360.0 / YAW_BINS)) % YAW_BINS
            bins[b].append(i)

        cand_bins = sorted(range(YAW_BINS), key=lambda b: len(bins[b]), reverse=True)
        chosen_b = cand_bins[0]
        # Protect target bin if it has <= min_target_count frames and another bin has frames to spare
        if target_b is not None and chosen_b == target_b and len(bins[target_b]) <= min_target_count:
            for alt_b in cand_bins[1:]:
                if len(bins[alt_b]) > 0:
                    chosen_b = alt_b
                    break

        drop_i = min(bins[chosen_b], key=lambda i: float(kept[i]["sharpness_score"]))
        kept.pop(drop_i)
    return kept


class _FallbackKeyframeRecord:
    def __init__(self, frame_index: int, filename: str, sharpness: float):
        self.source_frame_index = frame_index
        self.source_filename = filename
        self.sharpness_score = sharpness


class _FallbackKeyframeResult:
    def __init__(self, records: list):
        self.kept_records = records
        self.kept_count = len(records)


def _fallback_select_keyframes(staging_dir: str, window_translation_m: float = 0.5, window_rotation_deg: float = 45.0, **kwargs):
    import cv2
    odo_path = os.path.join(staging_dir, "odometry.csv")
    rgb_dir = os.path.join(staging_dir, "rgb")
    if not os.path.isfile(odo_path) or not os.path.isdir(rgb_dir):
        return _FallbackKeyframeResult([])
    df = pd.read_csv(odo_path)
    if df.empty or not {"frame", "x", "y", "z"}.issubset(df.columns):
        return _FallbackKeyframeResult([])
    
    records = []
    last_pos = None
    last_quat = None
    
    for _, row in df.iterrows():
        f_idx = int(row["frame"])
        fname = f"{f_idx:06d}.jpg"
        fpath = os.path.join(rgb_dir, fname)
        if not os.path.isfile(fpath):
            continue
        
        pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        quat = np.array([float(row.get("qx", 0.0)), float(row.get("qy", 0.0)), float(row.get("qz", 0.0)), float(row.get("qw", 1.0))])
        
        take = False
        if last_pos is None:
            take = True
        else:
            dist = float(np.linalg.norm(pos - last_pos))
            cos_half = np.clip(np.abs(np.dot(quat, last_quat)), 0.0, 1.0)
            rot_deg = float(np.degrees(2.0 * np.arccos(cos_half)))
            if dist >= window_translation_m or rot_deg >= window_rotation_deg:
                take = True
                
        if take:
            gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0
            records.append(_FallbackKeyframeRecord(f_idx, fname, sharpness))
            last_pos = pos
            last_quat = quat
            
    return _FallbackKeyframeResult(records)


def _select_keyframes_dry(session_dir, output_dir=None, **kwargs):
    try:
        from keyframe_select import select_keyframes_dry
        return select_keyframes_dry(session_dir, output_dir=output_dir, **kwargs)
    except ImportError:
        return _fallback_select_keyframes(session_dir, **kwargs)


def _stage_island(session_dir: str, pose_df: pd.DataFrame, staging: str) -> None:
    rgb_src = os.path.join(session_dir, "rgb")
    rgb_dst = os.path.join(staging, "rgb")
    os.makedirs(rgb_dst, exist_ok=True)
    if not os.path.isdir(rgb_src):
        return

    # Write the active trajectory (from pose_df, which contains server_vio or island odometry)
    # as odometry.csv in staging directory so keyframe selector evaluates the actual trajectory.
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


def export_vggt_keyframes(session_dir: str) -> KeyframeExport:
    pose_df = load_pose_table_for_tsdf(session_dir)
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
            window_translation_m=0.5,
            window_rotation_deg=45.0,
        )
        kept = [r for r in result.kept_records]
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
            else:
                yaw = extract_yaw_deg(row["qx"], row["qy"], row["qz"], row["qw"], is_server_vio=is_server_vio)
            recs.append(
                {
                    "record": r,
                    "yaw_deg": yaw,
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

        sent = (
            yaw_coverage_subsample(recs, max_n=N_VGGT_MAX, target_yaw_deg=target_yaw, min_target_count=8)
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
            shutil.copy2(src_path, os.path.join(img_dir, vggt_filename))
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
