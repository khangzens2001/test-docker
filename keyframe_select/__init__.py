"""Keyframe selection for continuous 3D-scan RGB sequences.

Select the best representative frames from a scan session using
motion-based windowing, sharpness scoring, and perceptual deduplication.

Quick start::

    from keyframe_select import select_keyframes

    result = select_keyframes("/path/to/scan_session")
    print(result.summary())
"""

__version__ = "0.1.0"

from .api import select_keyframes, select_keyframes_dry
from .config import Config
from .models import KeyframeRecord
from .selection import SelectionResult

__all__ = [
    # --- Primary API ---
    "select_keyframes",
    "select_keyframes_dry",
    # --- Configuration ---
    "Config",
    # --- Result types ---
    "SelectionResult",
    "KeyframeRecord",
    # --- Metadata ---
    "__version__",
]
