DEFAULT_WALLPAPER_TRIM_M = 0.10
DEFAULT_WALLPAPER_OVERLAP_M = 0.02
DEFAULT_WALLPAPER_REPEAT_M = 0.0

WALLPAPER_STANDARDS: tuple[dict[str, float | str], ...] = (
    {"name": "Standard Width (JP)", "width_m": 0.92, "roll_length_m": 50.0},
    {"name": "Narrow Width (JP)", "width_m": 0.46, "roll_length_m": 10.0},
    {"name": "European Standard", "width_m": 0.53, "roll_length_m": 10.05},
)
