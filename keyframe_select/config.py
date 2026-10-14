from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

_EXPOSURE_STRATEGIES = ("gamma", "linear", "clahe")


@dataclass
class Config:
    session_dir: Path
    output_dir: Path

    window_translation_m: float = 0.5
    window_rotation_deg: float = 45.0
    overlap: float = 0.35
    max_window_frames: int = 90
    min_window_frames: int = 2
    outlier_jump_factor: float = 5.0

    sharpness_normalize: bool = False

    dedup_hash_threshold: int = 6
    dedup_scope: Literal["local", "global"] = "local"
    dedup_lookback: int = 5

    fallback_window_frames: int = 20
    min_odometry_coverage: float = 0.8

    copy_mode: Literal["copy", "symlink", "hardlink"] = "copy"
    rename_format: str = "{index:06d}.jpg"
    max_dimension: Optional[int] = None

    # --- Exposure normalisation (opt-in) ---
    exposure_normalize: bool = False
    exposure_strategy: Literal["gamma", "linear", "clahe"] = "gamma"
    exposure_target: Optional[float] = None  # None = auto (median of kept frames)
    exposure_quality: int = 95              # JPEG output quality [0, 100]

    dry_run: bool = False
    log_level: str = "INFO"

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = str(v) if isinstance(v, Path) else v
        return d
