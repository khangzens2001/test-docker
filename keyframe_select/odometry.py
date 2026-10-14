import logging
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .models import FramePose

logger = logging.getLogger(__name__)

_FRAME_NAME_RE = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def list_rgb_frames(session_dir: Path) -> List[Tuple[int, Path]]:
    rgb_dir = session_dir / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"No rgb/ directory found in {session_dir}")

    frames = []
    for p in rgb_dir.iterdir():
        m = _FRAME_NAME_RE.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda t: t[0])
    if not frames:
        raise FileNotFoundError(f"No numbered image files found in {rgb_dir}")
    return frames


def load_odometry(session_dir: Path) -> pd.DataFrame:
    odom_path = session_dir / "odometry.csv"
    if not odom_path.is_file():
        raise FileNotFoundError(f"odometry.csv not found in {session_dir}")
    return pd.read_csv(odom_path)


def join_frames_with_odometry(
    session_dir: Path, min_coverage: float
) -> Tuple[List[FramePose], float, bool, str]:
    """
    Returns (frame_poses, coverage, use_fallback, warning_message).
    frame_poses is ordered by frame_index and always covers every rgb file,
    with odometry_matched=False + None pose fields for unmatched frames.
    """
    rgb_frames = list_rgb_frames(session_dir)
    total = len(rgb_frames)

    try:
        odom = load_odometry(session_dir)
    except FileNotFoundError as e:
        return (
            [FramePose(idx, path) for idx, path in rgb_frames],
            0.0,
            True,
            f"odometry.csv missing: {e}. Falling back to frame-index windowing.",
        )

    if "frame" in odom.columns:
        pose_by_frame = {
            int(row["frame"]): row for _, row in odom.iterrows()
        }
        matched = 0
        poses = []
        for idx, path in rgb_frames:
            row = pose_by_frame.get(idx)
            if row is not None:
                matched += 1
                poses.append(
                    FramePose(
                        idx,
                        path,
                        x=float(row["x"]),
                        y=float(row["y"]),
                        z=float(row["z"]),
                        qx=float(row["qx"]),
                        qy=float(row["qy"]),
                        qz=float(row["qz"]),
                        qw=float(row["qw"]),
                        odometry_matched=True,
                    )
                )
            else:
                poses.append(FramePose(idx, path))
        coverage = matched / total if total else 0.0
    elif len(odom) > 0:
        # No explicit frame column: zip position-ordered rows with sorted rgb
        # files as a lower-confidence join.
        logger.warning(
            "odometry.csv has no 'frame' column; joining by row order "
            "(lower confidence)."
        )
        poses = []
        matched = 0
        n = min(len(odom), total)
        for i, (idx, path) in enumerate(rgb_frames):
            if i < n:
                row = odom.iloc[i]
                matched += 1
                poses.append(
                    FramePose(
                        idx,
                        path,
                        x=float(row["x"]),
                        y=float(row["y"]),
                        z=float(row["z"]),
                        qx=float(row["qx"]),
                        qy=float(row["qy"]),
                        qz=float(row["qz"]),
                        qw=float(row["qw"]),
                        odometry_matched=True,
                    )
                )
            else:
                poses.append(FramePose(idx, path))
        coverage = matched / total if total else 0.0
    else:
        poses = [FramePose(idx, path) for idx, path in rgb_frames]
        coverage = 0.0

    if coverage < min_coverage:
        warning = (
            f"odometry coverage {coverage:.1%} below min_odometry_coverage "
            f"({min_coverage:.1%}). Falling back to frame-index windowing."
        )
        return poses, coverage, True, warning

    return poses, coverage, False, ""
