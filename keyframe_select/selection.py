import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import Config
from .dedup import dedup_window_winners
from .models import FramePose, KeyframeRecord, Window
from .motion import build_windows
from .odometry import join_frames_with_odometry
from .sharpness import compute_sharpness

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of keyframe selection.

    Contains the full list of candidate records (kept and dropped),
    diagnostics, and convenience properties for quick access to the
    selected keyframes.
    """

    records: List[KeyframeRecord]
    warnings: List[str]
    odometry_coverage: float
    used_fallback: bool
    total_source_frames: int

    # --- Convenience properties ---

    @property
    def kept_records(self) -> List[KeyframeRecord]:
        """Only the records that were kept (not dropped by dedup)."""
        return [r for r in self.records if r.kept]

    @property
    def kept_count(self) -> int:
        """Number of kept keyframes."""
        return sum(1 for r in self.records if r.kept)

    @property
    def kept_source_filenames(self) -> List[str]:
        """Source filenames of kept keyframes, in selection order."""
        return [r.source_filename for r in self.records if r.kept]

    # --- Methods ---

    def summary(self) -> str:
        """One-line human-readable summary of the selection result."""
        return (
            f"Windows: {len(self.records)}  "
            f"Kept keyframes: {self.kept_count}  "
            f"Source frames: {self.total_source_frames}  "
            f"Odometry coverage: {self.odometry_coverage:.1%}  "
            f"Fallback used: {self.used_fallback}"
        )

    def to_dataframe(self):
        """Convert records to a ``pandas.DataFrame`` for analysis.

        Raises ``ImportError`` if pandas is not installed.
        """
        import pandas as pd
        from dataclasses import asdict

        return pd.DataFrame([asdict(r) for r in self.records])


def _pick_sharpest(poses: List[FramePose], window: Window, cfg: Config) -> tuple:
    best_path = None
    best_frame_index = None
    best_score = float("-inf")
    for p in poses[window.frame_start : window.frame_end + 1]:
        score = compute_sharpness(p.path, normalize=cfg.sharpness_normalize)
        if score > best_score:
            best_score = score
            best_path = p.path
            best_frame_index = p.frame_index
        elif score == best_score and (
            best_frame_index is None or p.frame_index < best_frame_index
        ):
            best_path = p.path
            best_frame_index = p.frame_index
    return best_frame_index, best_path, best_score


def run_selection(cfg: Config) -> SelectionResult:
    warnings: List[str] = []

    poses, coverage, use_fallback, warn_msg = join_frames_with_odometry(
        cfg.session_dir, cfg.min_odometry_coverage
    )
    if warn_msg:
        warnings.append(warn_msg)
        logger.warning(warn_msg)

    windows = build_windows(poses, use_fallback, cfg)
    if len(windows) == 1:
        warnings.append(
            "Session shorter than one window — returning a single keyframe."
        )

    poses_by_index = {p.frame_index: p for p in poses}

    winners = []  # (window_index, path, score) for dedup
    prelim = {}  # window_index -> KeyframeRecord (kept True initially)

    for w in windows:
        frame_index, path, score = _pick_sharpest(poses, w, cfg)
        p = poses_by_index[frame_index]
        rec = KeyframeRecord(
            new_index=-1,
            new_filename="",
            source_frame_index=frame_index,
            source_filename=p.path.name,
            window_index=w.index,
            window_frame_start=poses[w.frame_start].frame_index,
            window_frame_end=poses[w.frame_end].frame_index,
            cumulative_translation_m=w.cumulative_translation_m,
            cumulative_rotation_deg=w.cumulative_rotation_deg,
            sharpness_score=score,
            odometry_matched=p.odometry_matched,
            kept=True,
        )
        prelim[w.index] = rec
        winners.append((w.index, path, score))

    dropped = dedup_window_winners(winners, cfg)
    for wid, reason in dropped.items():
        prelim[wid].kept = False
        prelim[wid].dropped_reason = reason

    records = [prelim[w.index] for w in windows]

    new_index = 0
    for rec in records:
        if rec.kept:
            rec.new_index = new_index
            rec.new_filename = cfg.rename_format.format(index=new_index)
            new_index += 1

    return SelectionResult(
        records=records,
        warnings=warnings,
        odometry_coverage=coverage,
        used_fallback=use_fallback,
        total_source_frames=len(poses),
    )
