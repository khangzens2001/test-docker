"""CLI entry point for keyframe selection.

Usage::

    keyframe-select /path/to/session_dir --output /path/to/output
"""

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from .api import _build_config
from .manifest import write_output
from .selection import run_selection

app = typer.Typer(add_completion=False)


@app.command()
def main(
    session_dir: Annotated[Path, typer.Argument(help="Scan session dir containing rgb/ and odometry.csv")],
    output: Annotated[Path, typer.Option("--output", help="Output directory")],
    window_translation_m: float = 0.5,
    window_rotation_deg: float = 45.0,
    overlap: float = 0.35,
    max_window_frames: int = 90,
    min_window_frames: int = 2,
    outlier_jump_factor: float = 5.0,
    sharpness_normalize: bool = False,
    dedup_hash_threshold: int = 6,
    dedup_scope: str = typer.Option("local", help="local or global"),
    dedup_lookback: int = 5,
    fallback_window_frames: int = 20,
    min_odometry_coverage: float = 0.8,
    copy_mode: str = typer.Option("copy", help="copy, symlink, or hardlink"),
    rename_format: str = "{index:06d}.jpg",
    max_dimension: Optional[int] = typer.Option(
        None, help="Downscale kept images so their longer side is at most this many pixels (keeps aspect ratio, never upscales)"
    ),
    exposure_normalize: bool = typer.Option(False, help="Normalise brightness of output images"),
    exposure_strategy: str = typer.Option("gamma", help="Normalisation strategy: gamma, linear, or clahe"),
    exposure_target: Optional[float] = typer.Option(None, help="Target mean brightness [0-255]; auto (median) if omitted"),
    exposure_quality: int = typer.Option(95, help="JPEG quality for normalised output [0-100]"),
    dry_run: bool = False,
    log_level: str = "INFO",
):
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    cfg = _build_config(
        session_dir=session_dir,
        output_dir=output,
        dry_run=dry_run,
        window_translation_m=window_translation_m,
        window_rotation_deg=window_rotation_deg,
        overlap=overlap,
        max_window_frames=max_window_frames,
        min_window_frames=min_window_frames,
        outlier_jump_factor=outlier_jump_factor,
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
        log_level=log_level,
    )

    result = run_selection(cfg)
    write_output(result, cfg)

    typer.echo(result.summary())
    for w in result.warnings:
        typer.secho(f"WARNING: {w}", fg=typer.colors.YELLOW)


def cli():
    app()


if __name__ == "__main__":
    cli()
