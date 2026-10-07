"""Font resolution and Matplotlib Agg configuration.

Resolves CJK fonts in priority order:
1. app/static/fonts/
2. /usr/share/fonts/opentype/noto/
3. macOS system/library fonts (/System/Library/Fonts/, /Library/Fonts/)
4. Fallback to DejaVu Sans.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_RESOLVED_FONT: str | None = None
_AGG_CONFIGURED: bool = False


def _resolve_cjk_font() -> str | None:
    """Resolve CJK font path in order of priority."""
    global _RESOLVED_FONT
    if _RESOLVED_FONT is not None:
        return _RESOLVED_FONT

    import matplotlib.font_manager as fm

    search_dirs = [
        Path(__file__).resolve().parents[2] / "static" / "fonts",
        Path("/usr/share/fonts/opentype/noto"),
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    ]

    cjk_priority_keywords = ("notosanscjk", "hiragino", "ipaex", "jp")
    cjk_fallback_keywords = (
        "cjk",
        "pingfang",
        "gothic",
        "myungjo",
        "meiryo",
        "songti",
        "heiti",
        "kaiti",
    )
    generic_keywords = ("noto",)

    def _font_rank(p: Path) -> tuple[int, str]:
        nl = p.name.lower()
        if "symbol" in nl:
            return (99, nl)
        if any(k in nl for k in cjk_priority_keywords):
            return (0, nl)
        if any(k in nl for k in cjk_fallback_keywords):
            return (1, nl)
        if any(k in nl for k in generic_keywords):
            return (2, nl)
        return (3, nl)

    for directory in search_dirs:
        if not directory.is_dir():
            continue

        font_files: list[Path] = []
        for ext in ("*.ttc", "*.otf", "*.ttf"):
            font_files.extend(directory.glob(ext))

        # First try CJK prioritized matches
        for f in sorted(font_files, key=_font_rank):
            name_lower = f.name.lower()
            if (
                any(kw in name_lower for kw in cjk_priority_keywords)
                or any(kw in name_lower for kw in cjk_fallback_keywords)
                or any(kw in name_lower for kw in generic_keywords)
            ):
                try:
                    fm.fontManager.addfont(str(f))
                    prop = fm.FontProperties(fname=str(f))
                    font_name = prop.get_name()
                    if font_name:
                        _RESOLVED_FONT = font_name
                        return font_name
                except Exception as exc:
                    logger.debug("Failed adding font %s: %s", f, exc)
                    continue

        # If in static/fonts and has any font file, try using it
        if directory == search_dirs[0] and font_files:
            for f in sorted(font_files):
                try:
                    fm.fontManager.addfont(str(f))
                    prop = fm.FontProperties(fname=str(f))
                    font_name = prop.get_name()
                    if font_name:
                        _RESOLVED_FONT = font_name
                        return font_name
                except Exception:
                    continue

    _RESOLVED_FONT = "DejaVu Sans"
    return _RESOLVED_FONT


def configure_matplotlib_agg() -> None:
    """Configure matplotlib with Agg headless backend and resolved fonts.

    Must call matplotlib.use("Agg") before importing pyplot.
    """
    global _AGG_CONFIGURED
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not _AGG_CONFIGURED:
        plt.rcParams["axes.unicode_minus"] = False
        font_name = _resolve_cjk_font()
        if font_name and font_name != "DejaVu Sans":
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "Arial", "sans-serif"]
        else:
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
        _AGG_CONFIGURED = True
