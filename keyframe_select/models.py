from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FramePose:
    frame_index: int
    path: Path
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    qx: Optional[float] = None
    qy: Optional[float] = None
    qz: Optional[float] = None
    qw: Optional[float] = None
    odometry_matched: bool = False


@dataclass
class Window:
    index: int
    frame_start: int
    frame_end: int
    frames: list = field(default_factory=list)
    cumulative_translation_m: float = 0.0
    cumulative_rotation_deg: float = 0.0


@dataclass
class KeyframeRecord:
    new_index: int
    new_filename: str
    source_frame_index: int
    source_filename: str
    window_index: int
    window_frame_start: int
    window_frame_end: int
    cumulative_translation_m: float
    cumulative_rotation_deg: float
    sharpness_score: float
    odometry_matched: bool
    kept: bool
    dropped_reason: str = ""
    # Exposure normalisation metadata (populated by manifest.py)
    source_mean_brightness: Optional[float] = None
    exposure_normalized: bool = False
