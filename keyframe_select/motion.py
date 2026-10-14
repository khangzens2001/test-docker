import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config
from .models import FramePose, Window


def compute_step_deltas(poses: List[FramePose]) -> Tuple[List[float], List[float]]:
    """Per-frame (dt_translation_m, dt_rotation_deg) vs. the previous frame.

    Index 0 is always 0.0/0.0. A step where either endpoint lacks a matched
    odometry pose is treated as zero motion (conservative: never fabricates
    motion across a gap).
    """
    n = len(poses)
    dt_trans = [0.0] * n
    dt_rot = [0.0] * n
    for i in range(1, n):
        a, b = poses[i - 1], poses[i]
        if not (a.odometry_matched and b.odometry_matched):
            continue
        dt_trans[i] = math.sqrt(
            (b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2
        )
        ra = Rotation.from_quat([a.qx, a.qy, a.qz, a.qw])
        rb = Rotation.from_quat([b.qx, b.qy, b.qz, b.qw])
        rel = ra.inv() * rb
        dt_rot[i] = abs(rel.magnitude()) * 180.0 / math.pi
    return dt_trans, dt_rot


def _clean_poses(
    poses: List[FramePose], outlier_jump_factor: float
) -> Tuple[List[Optional[np.ndarray]], List[Optional[Rotation]]]:
    """Position/rotation per frame, with unmatched frames and single-frame
    odometry glitches (a step delta far above the session's typical step)
    forward/backward-filled from the nearest trustworthy neighbor.

    This makes the window-building distance calculation robust to isolated
    bad odometry samples, without needing to clamp/guess a magnitude for them.
    """
    n = len(poses)
    dt_trans, dt_rot = compute_step_deltas(poses)

    def _outlier_mask(deltas: List[float]) -> np.ndarray:
        arr = np.array(deltas, dtype=float)
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            return np.zeros(len(arr), dtype=bool)
        median = float(np.median(nonzero))
        if median <= 0:
            return np.zeros(len(arr), dtype=bool)
        return arr > median * outlier_jump_factor

    bad_step = _outlier_mask(dt_trans) | _outlier_mask(dt_rot)

    positions: List[Optional[np.ndarray]] = [None] * n
    rotations: List[Optional[Rotation]] = [None] * n
    for i, p in enumerate(poses):
        if not p.odometry_matched:
            continue
        if bad_step[i]:  # the step INTO frame i looked like a glitch
            continue
        positions[i] = np.array([p.x, p.y, p.z])
        rotations[i] = Rotation.from_quat([p.qx, p.qy, p.qz, p.qw])

    last_pos, last_rot = None, None
    for i in range(n):
        if positions[i] is not None:
            last_pos, last_rot = positions[i], rotations[i]
        elif last_pos is not None:
            positions[i], rotations[i] = last_pos, last_rot

    next_pos, next_rot = None, None
    for i in range(n - 1, -1, -1):
        if positions[i] is not None:
            next_pos, next_rot = positions[i], rotations[i]
        elif next_pos is not None:
            positions[i], rotations[i] = next_pos, next_rot

    return positions, rotations


def _frames_slice(poses: List[FramePose], start: int, end: int) -> list:
    return [p.frame_index for p in poses[start : end + 1]]


def build_windows_by_motion(poses: List[FramePose], cfg: Config) -> List[Window]:
    """Segments the trajectory using NET displacement from each window's
    start frame (translation distance + rotation angle directly between the
    start pose and the candidate pose) rather than the path-length/summed
    rotation traversed along the way.

    This distinction matters: a camera that rotates back and forth while
    examining one stationary subject racks up a large *sum* of per-step
    rotation magnitudes (each step counted positively regardless of
    direction) without the viewpoint actually changing much. Net distance
    from the window's start cancels that oscillation out and only closes a
    window once the viewpoint has genuinely moved away from where it began.
    """
    n = len(poses)
    if n <= 1:
        return [Window(index=0, frame_start=0, frame_end=n - 1, frames=_frames_slice(poses, 0, n - 1))]

    positions, rotations = _clean_poses(poses, cfg.outlier_jump_factor)

    windows: List[Window] = []
    s = 0
    while True:
        p_start, r_start = positions[s], rotations[s]
        next_start = None
        end = s
        net_trans = 0.0
        net_rot = 0.0
        for i in range(s + 1, n):
            net_trans = float(np.linalg.norm(positions[i] - p_start))
            net_rot = abs((r_start.inv() * rotations[i]).magnitude()) * 180.0 / math.pi
            progress = max(
                net_trans / cfg.window_translation_m,
                net_rot / cfg.window_rotation_deg,
            )
            if next_start is None and progress >= (1 - cfg.overlap):
                next_start = i
            if progress >= 1.0 or (i - s + 1) >= cfg.max_window_frames:
                end = i
                break
            end = i
        else:
            end = n - 1

        windows.append(
            Window(
                index=len(windows),
                frame_start=s,
                frame_end=end,
                frames=_frames_slice(poses, s, end),
                cumulative_translation_m=net_trans,
                cumulative_rotation_deg=net_rot,
            )
        )

        if end >= n - 1:
            break
        if next_start is None:
            # Motion never reached the (1 - overlap) progress mark before the
            # window was cut off by max_window_frames (e.g. a stationary or
            # oscillating segment) — space the next window using the same
            # overlap fraction applied to the frames actually covered, so we
            # don't degenerate into a ~1-frame stride across the whole
            # segment.
            window_width = end - s + 1
            next_start = s + max(1, int(round(window_width * (1 - cfg.overlap))))
        elif next_start <= s:
            next_start = s + 1
        s = next_start

    _merge_undersized_trailing_window(windows, poses, cfg.min_window_frames)
    return windows


def build_windows_by_frame_index(poses: List[FramePose], cfg: Config) -> List[Window]:
    n = len(poses)
    if n <= 1:
        return [Window(index=0, frame_start=0, frame_end=n - 1, frames=_frames_slice(poses, 0, n - 1))]

    window_size = max(1, cfg.fallback_window_frames)
    stride = max(1, int(round(window_size * (1 - cfg.overlap))))

    windows: List[Window] = []
    s = 0
    while True:
        end = min(s + window_size - 1, n - 1)
        windows.append(
            Window(index=len(windows), frame_start=s, frame_end=end, frames=_frames_slice(poses, s, end))
        )
        if end >= n - 1:
            break
        s += stride

    _merge_undersized_trailing_window(windows, poses, cfg.min_window_frames)
    return windows


def _merge_undersized_trailing_window(windows: List[Window], poses: List[FramePose], min_window_frames: int) -> None:
    if len(windows) < 2:
        return
    last = windows[-1]
    if (last.frame_end - last.frame_start + 1) < min_window_frames:
        prev = windows[-2]
        prev.frame_end = last.frame_end
        prev.frames = _frames_slice(poses, prev.frame_start, prev.frame_end)
        windows.pop()


def build_windows(poses: List[FramePose], use_fallback: bool, cfg: Config) -> List[Window]:
    if use_fallback:
        return build_windows_by_frame_index(poses, cfg)
    return build_windows_by_motion(poses, cfg)
