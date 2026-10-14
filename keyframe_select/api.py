"""Public API for keyframe selection.

This module provides high-level facade functions that wrap the entire
keyframe-selection pipeline into simple, one-call interfaces.

Typical usage::

    from keyframe_select import select_keyframes

    result = select_keyframes("./my_scan_session")
    print(result.summary())
    for path in result.kept_paths:
        print(path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .config import Config
from .manifest import write_output
from .selection import SelectionResult, run_selection

__all__ = ["select_keyframes", "select_keyframes_dry"]

logger = logging.getLogger(__name__)


def _build_config(
    session_dir: Union[str, Path],
    output_dir: Union[str, Path, None],
    dry_run: bool,
    **kwargs,
) -> Config:
    """Construct a ``Config`` from user-friendly arguments.

    ``output_dir`` defaults to ``<session_dir>/keyframes_selected`` when not
    provided — following the convention of keeping outputs next to the source
    data (alongside ``rgb/``, ``depth/``, etc.).
    """
    session_dir = Path(session_dir)
    if output_dir is None:
        output_dir = session_dir / "keyframes_selected"
    else:
        output_dir = Path(output_dir)

    return Config(session_dir=session_dir, output_dir=output_dir, dry_run=dry_run, **kwargs)


def select_keyframes(
    session_dir: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
    *,
    # --- Window parameters ---
    window_translation_m: float = 0.5,
    window_rotation_deg: float = 45.0,
    overlap: float = 0.35,
    max_window_frames: int = 90,
    min_window_frames: int = 2,
    # --- Sharpness ---
    sharpness_normalize: bool = False,
    # --- Dedup ---
    dedup_hash_threshold: int = 6,
    dedup_scope: str = "local",
    dedup_lookback: int = 5,
    # --- Fallback ---
    fallback_window_frames: int = 20,
    min_odometry_coverage: float = 0.8,
    # --- Output ---
    copy_mode: str = "copy",
    rename_format: str = "{index:06d}.jpg",
    max_dimension: Optional[int] = None,
    # --- Exposure normalisation ---
    exposure_normalize: bool = False,
    exposure_strategy: str = "gamma",
    exposure_target: Optional[float] = None,
    exposure_quality: int = 95,
    # --- Misc ---
    outlier_jump_factor: float = 5.0,
    log_level: str = "INFO",
) -> SelectionResult:
    """Select keyframes from a 3D-scan session and write results to disk.

    This is the primary entry point for the keyframe-selection pipeline.
    It reads RGB images and (optionally) odometry data from ``session_dir``,
    selects the best representative frames, and writes the output (images +
    manifest CSV + config JSON) into ``output_dir``.

    Parameters
    ----------
    session_dir : str or Path
        Scan session directory containing ``rgb/`` (numbered images) and
        optionally ``odometry.csv``.
    output_dir : str, Path, or None
        Where to write selected keyframes. Defaults to
        ``<session_dir>/keyframes_selected``.
    window_translation_m : float
        Translation threshold (meters) to close a motion window.
    window_rotation_deg : float
        Rotation threshold (degrees) to close a motion window.
    overlap : float
        Fractional overlap between consecutive windows (0.0–1.0).
    max_window_frames : int
        Maximum number of frames per window (caps stationary segments).
    min_window_frames : int
        Minimum window size; trailing undersized windows are merged.
    sharpness_normalize : bool
        If True, normalize Laplacian variance by pixel-intensity variance.
    dedup_hash_threshold : int
        Hamming distance threshold for perceptual-hash deduplication.
    dedup_scope : str
        ``"local"`` (compare with recent kept frames only) or ``"global"``.
    dedup_lookback : int
        Number of recently kept frames to compare against (local scope).
    fallback_window_frames : int
        Fixed window size when odometry is unavailable.
    min_odometry_coverage : float
        Minimum fraction of frames matched to odometry before falling back.
    copy_mode : str
        ``"copy"``, ``"symlink"``, or ``"hardlink"`` for output images.
    rename_format : str
        Format string for renaming output images (receives ``index``).
    max_dimension : int or None
        If set, downscale output images so longer side ≤ this many pixels.
    outlier_jump_factor : float
        Factor above median step-delta to flag an odometry glitch.
    log_level : str
        Logging level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).

    Returns
    -------
    SelectionResult
        Dataclass with full results: records, warnings, statistics, and
        convenience properties (``kept_records``, ``kept_paths``, etc.).

    Examples
    --------
    Typical usage::

        from keyframe_select import select_keyframes

        # Basic
        result = select_keyframes("/path/to/scan_session")

        # With exposure normalisation (recommended for 3D reconstruction)
        result = select_keyframes(
            "/path/to/scan_session",
            exposure_normalize=True,
            exposure_strategy="gamma",   # "gamma" | "linear" | "clahe"
        )

        # Custom thresholds for close-range object scanning
        result = select_keyframes(
            "/path/to/scan_session",
            window_rotation_deg=30.0,
            window_translation_m=0.3,
            dedup_hash_threshold=4,
        )

        # Output to a specific directory
        result = select_keyframes(
            "/path/to/scan_session",
            output_dir="/path/to/my_output",
            copy_mode="symlink",
        )
    """
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = _build_config(
        session_dir=session_dir,
        output_dir=output_dir,
        dry_run=False,
        window_translation_m=window_translation_m,
        window_rotation_deg=window_rotation_deg,
        overlap=overlap,
        max_window_frames=max_window_frames,
        min_window_frames=min_window_frames,
        sharpness_normalize=sharpness_normalize,
        dedup_hash_threshold=dedup_hash_threshold,
        dedup_scope=dedup_scope,
        dedup_lookback=dedup_lookback,
        fallback_window_frames=fallback_window_frames,
        min_odometry_coverage=min_odometry_coverage,
        copy_mode=copy_mode,
        rename_format=rename_format,
        max_dimension=max_dimension,
        exposure_normalize=exposure_normalize,
        exposure_strategy=exposure_strategy,
        exposure_target=exposure_target,
        exposure_quality=exposure_quality,
        outlier_jump_factor=outlier_jump_factor,
        log_level=log_level,
    )

    result = run_selection(cfg)
    write_output(result, cfg)

    logger.info(result.summary())
    return result


def select_keyframes_dry(
    session_dir: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
    **kwargs,
) -> SelectionResult:
    """Run keyframe selection without writing images to disk.

    Behaves identically to :func:`select_keyframes` but sets
    ``dry_run=True``: the manifest CSV and run-config JSON are written,
    but no images are copied. Useful for previewing results before
    committing disk space.

    Parameters
    ----------
    session_dir : str or Path
        Scan session directory.
    output_dir : str, Path, or None
        Output directory for manifest/config (defaults as usual).
    **kwargs
        All keyword arguments accepted by :func:`select_keyframes`
        except ``copy_mode`` and ``max_dimension`` (irrelevant in dry run).

    Returns
    -------
    SelectionResult
        Same as :func:`select_keyframes`.

    Examples
    --------
    Preview before committing::

        from keyframe_select import select_keyframes_dry, select_keyframes

        result = select_keyframes_dry("/path/to/session")
        print(f"Would keep {len(result.kept_records)} keyframes")

        # Happy with the count? Run for real:
        result = select_keyframes("/path/to/session")
    """
    log_level = kwargs.pop("log_level", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = _build_config(
        session_dir=session_dir,
        output_dir=output_dir,
        dry_run=True,
        log_level=log_level,
        **kwargs,
    )

    result = run_selection(cfg)
    write_output(result, cfg)

    logger.info(result.summary())
    return result
