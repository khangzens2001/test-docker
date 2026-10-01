"""CAD rendering and quality inspection package."""

from typing import Any

__all__ = [
    "ARTIFACT_MAP",
    "FloorPlanRenderer",
    "SHEET_SIZES",
    "STYLE",
    "WALL_THICKNESS_MM",
    "WallElevationsRenderer",
    "_resolve_cjk_font",
    "artifact_url",
    "build_quality_report",
    "configure_matplotlib_agg",
    "layout_to_metrics_mm",
    "render_floorplan",
    "render_wall_elevations",
    "resolve_artifact_file",
    "shape_name_for_n",
    "write_metrics_json",
    "write_quality_report_json",
]

_SUBMODULE_MAP = {
    "ARTIFACT_MAP": "app.services.cad.artifacts",
    "artifact_url": "app.services.cad.artifacts",
    "resolve_artifact_file": "app.services.cad.artifacts",
    "layout_to_metrics_mm": "app.services.cad.contract",
    "shape_name_for_n": "app.services.cad.contract",
    "write_metrics_json": "app.services.cad.contract",
    "FloorPlanRenderer": "app.services.cad.floorplan_render",
    "configure_matplotlib_agg": "app.services.cad.floorplan_render",
    "render_floorplan": "app.services.cad.floorplan_render",
    "_resolve_cjk_font": "app.services.cad.fonts",
    "build_quality_report": "app.services.cad.quality",
    "write_quality_report_json": "app.services.cad.quality",
    "SHEET_SIZES": "app.services.cad.style",
    "STYLE": "app.services.cad.style",
    "WALL_THICKNESS_MM": "app.services.cad.style",
    "WallElevationsRenderer": "app.services.cad.wall_elevations",
    "render_wall_elevations": "app.services.cad.wall_elevations",
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULE_MAP:
        mod_name = _SUBMODULE_MAP[name]
        import importlib

        mod = importlib.import_module(mod_name)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


