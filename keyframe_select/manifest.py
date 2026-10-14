import csv
import json
import logging
import os
import shutil
from dataclasses import asdict, fields
from pathlib import Path

from PIL import Image

from .config import Config
from .models import KeyframeRecord
from .selection import SelectionResult

logger = logging.getLogger(__name__)


def _copy_file(src, dst, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown copy_mode: {mode}")


def _save_resized(src, dst, max_dimension: int) -> None:
    img = Image.open(src)
    scale = max_dimension / max(img.width, img.height)
    if scale < 1.0:
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    img.save(dst)


def _copy_or_resize(src: Path, dst: Path, cfg: Config) -> None:
    """Copy or resize a single source image to dst, honouring cfg settings.

    This is the standard (non-exposure-normalised) path used when
    ``cfg.exposure_normalize`` is False or when normalisation fails.
    """
    if cfg.max_dimension:
        # Resizing writes new pixel data, so copy_mode (symlink/hardlink)
        # doesn't apply — always materializes a real (resized) file.
        _save_resized(src, dst, cfg.max_dimension)
    else:
        _copy_file(src, dst, cfg.copy_mode)


def write_manifest_csv(result: SelectionResult, cfg: Config) -> None:
    manifest_path = cfg.output_dir / "keyframes_manifest.csv"
    fieldnames = [f.name for f in fields(KeyframeRecord)]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in result.records:
            writer.writerow(asdict(rec))


def write_run_config(result: SelectionResult, cfg: Config) -> None:
    payload = cfg.to_dict()
    payload["warnings"] = result.warnings
    payload["odometry_coverage"] = result.odometry_coverage
    payload["used_fallback"] = result.used_fallback
    payload["total_source_frames"] = result.total_source_frames
    payload["total_kept_keyframes"] = sum(1 for r in result.records if r.kept)
    with open(cfg.output_dir / "run_config.json", "w") as f:
        json.dump(payload, f, indent=2)


def copy_kept_images(result: SelectionResult, cfg: Config) -> None:
    images_dir = cfg.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = cfg.session_dir / "rgb"

    # ------------------------------------------------------------------ #
    # Exposure normalisation — compute shared target brightness once for  #
    # all kept frames so that every output image is normalised to the     #
    # same reference value.                                               #
    # ------------------------------------------------------------------ #
    use_exposure = cfg.exposure_normalize and not cfg.dry_run
    exposure_target: float | None = None

    if use_exposure:
        from .exposure import compute_target_brightness, normalize_exposure

        if cfg.exposure_target is not None:
            exposure_target = float(cfg.exposure_target)
            logger.info("Exposure normalisation: using explicit target=%.1f", exposure_target)
        else:
            kept_paths = [
                rgb_dir / rec.source_filename
                for rec in result.records
                if rec.kept
            ]
            if kept_paths:
                exposure_target = compute_target_brightness(kept_paths)
                logger.info(
                    "Exposure normalisation: auto target=%.1f (median of %d frames, strategy=%s)",
                    exposure_target,
                    len(kept_paths),
                    cfg.exposure_strategy,
                )

    for rec in result.records:
        if not rec.kept:
            continue
        src = rgb_dir / rec.source_filename
        dst = images_dir / rec.new_filename

        if use_exposure and exposure_target is not None:
            # Normalize overwrites any resize — both pixel ops, normalize wins.
            # If max_dimension is also set, normalize first then resize.
            try:
                orig_mean = normalize_exposure(
                    src, dst,
                    target_brightness=exposure_target,
                    strategy=cfg.exposure_strategy,
                    quality=cfg.exposure_quality,
                )
                rec.source_mean_brightness = round(orig_mean, 2)
                rec.exposure_normalized = True

                # Optional resize pass on already-written output
                if cfg.max_dimension:
                    _save_resized(dst, dst, cfg.max_dimension)

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Exposure normalisation failed for %s: %s — falling back to copy.",
                    src.name, exc,
                )
                _copy_or_resize(src, dst, cfg)

        else:
            _copy_or_resize(src, dst, cfg)


def write_output(result: SelectionResult, cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.dry_run:
        copy_kept_images(result, cfg)
    write_manifest_csv(result, cfg)
    write_run_config(result, cfg)
