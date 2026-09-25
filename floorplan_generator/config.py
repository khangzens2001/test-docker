# floorplan_generator/config.py

# ==========================================
# 3D POINT CLOUD & GEOMETRY SETTINGS
# ==========================================
VOXEL_SIZE = 0.015         # 1.5 cm voxel for downsampling
HEIGHT_PERCENTILE = 2      # Remove bottom 2% of points (floor noise)

# ==========================================
# 2D PROJECTION & MORPHOLOGY
# ==========================================
GRID_RESOLUTION_MM = 10    # 10mm per pixel for 2D grid
MORPHOLOGY_KERNEL_SIZE = 20 # Dilation/Erosion kernel size (increased for wall slice gaps)

# ==========================================
# CONCAVE ROOM DETECTION (L-shape, etc.)
# ==========================================
BBOX_FILL_RATIO_THRESHOLD = 0.82  # If contour_area/bbox_area < this, room is likely concave (L/T/U)
CONCAVE_MORPH_KERNEL_SIZE = 5     # Smaller kernel for concave detection to preserve indentations
CONCAVE_APPROX_EPSILON = 0.005    # Tighter approxPolyDP epsilon for concave rooms
DEFAULT_APPROX_EPSILON = 0.02     # Default approxPolyDP epsilon for convex/rectangular rooms
MIN_EDGE_LENGTH_RATIO = 0.04      # Minimum edge length as ratio of perimeter (filter noise edges)
MIN_CONCAVE_AREA_RATIO = 0.03     # Minimum concave notch area as ratio of bbox to keep
# ==========================================
# ALIGNMENT & RENDERING SETTINGS
# ==========================================
WALL_THICKNESS_MM = 40     # Visual wall thickness for hatch rendering on plan

# ==========================================
# RENDERING COLORS & STYLES
# ==========================================
STYLE = {
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

# ==========================================
# SHEET SIZES (Width, Height in mm)
# ==========================================
SHEET_SIZES = {
    "A0": (1189, 841),
    "A1": (841, 594),
    "A2": (594, 420),
    "A3": (420, 297),
    "A4": (297, 210)
}
