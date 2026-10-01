"""CAD styling constants and sheet dimensions.

Ported from 3D-Estimate backend/floorplan_generator/config.py.
"""

WALL_THICKNESS_MM: float = 40.0

STYLE: dict[str, str] = {
    "bg_white": "#FFFFFF",
    "border": "#1a1a2e",
    "floor_fill": "#F8F4EC",
    "wall_fill": "#E0D8CC",
    "wall_stroke": "#1a1a1a",
    "hatch_pattern": "////",
    "dim_wall_color": "#CC0000",
    "dim_bbox_color": "#0066CC",
    "text_dark": "#1a1a2e",
    "text_mid": "#475467",
    "badge_bg": "#FFFDF8",
    "badge_border": "#1F2937",
    "info_bg": "#FAFAFA",
    "table_header_bg": "#F0F0F0",
    "compass_north": "#CC0000",
}

SHEET_SIZES: dict[str, tuple[float, float]] = {
    "A0": (1189.0, 841.0),
    "A1": (841.0, 594.0),
    "A2": (594.0, 420.0),
    "A3": (420.0, 297.0),
    "A4": (297.0, 210.0),
}
