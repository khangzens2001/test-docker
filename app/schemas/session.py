from datetime import datetime
from enum import Enum
from typing import Any, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TatamiStandardEnum(str, Enum):
    Kyoma = "Kyoma"
    Chukyoma = "Chukyoma"
    Edoma = "Edoma"
    Danchima = "Danchima"
    Ryukyu = "Ryukyu"


class TatamiLayoutEnum(str, Enum):
    Shugikyo = "Shugikyo"
    Fushugikyo = "Fushugikyo"
    Ryukyu = "Ryukyu"


class WallpaperMatchTypeEnum(str, Enum):
    free = "free"
    straight = "straight"
    half_drop = "half-drop"


class WallOverride(BaseModel):
    wall_id: str = Field(..., pattern=r"^W\d+$")
    length_mm: float = Field(..., gt=0)


class RecalculateRequest(BaseModel):
    wallpaper_width_m: Optional[float] = Field(None, gt=0)
    wallpaper_roll_length_m: Optional[float] = Field(None, gt=0)
    wallpaper_trim_m: Optional[float] = Field(None, ge=0)
    wallpaper_overlap_m: Optional[float] = Field(None, ge=0)
    wallpaper_repeat_m: Optional[float] = Field(None, ge=0)
    wallpaper_match_type: Optional[WallpaperMatchTypeEnum] = None
    tatami_standard: Optional[TatamiStandardEnum] = None
    tatami_layout_type: Optional[TatamiLayoutEnum] = None
    wall_overrides: Optional[list[WallOverride]] = None
    update_mode: Optional[str] = None
    height_mm: Optional[float] = Field(None, gt=0)
    joist_pitch_mm: Optional[float] = Field(None, gt=0)
    border_width_mm: Optional[float] = Field(None, gt=0)
    board_length_mm: Optional[float] = Field(None, gt=0)
    board_width_mm: Optional[float] = Field(None, gt=0)
    cf_roll_width_mm: Optional[float] = Field(None, gt=0)
    tile_length_mm: Optional[float] = Field(None, gt=0)
    tile_width_mm: Optional[float] = Field(None, gt=0)
    tile_joint_width_mm: Optional[float] = Field(None, ge=0)
    waste_rate: Optional[float] = Field(None, ge=0)

    @model_validator(mode="after")
    def _width_gt_overlap(self) -> Self:
        if self.wallpaper_width_m is not None and self.wallpaper_overlap_m is not None:
            if self.wallpaper_width_m <= self.wallpaper_overlap_m:
                raise ValueError("wallpaper_width_m must be greater than wallpaper_overlap_m")
        return self


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    status: str
    progress_percentage: int = 0
    error_message: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    results: Optional[dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SessionStatus(str, Enum):
    received = "received"
    validating = "validating"
    running_pipeline = "running_pipeline"
    reconstructing = "reconstructing"
    completed = "completed"
    failed = "failed"


class ArtifactTypeEnum(str, Enum):
    point_cloud = "point_cloud"
    whiteflat_point_cloud = "whiteflat_point_cloud"
    mesh = "mesh"
    floorplan_json = "floorplan_json"
    floorplan_svg = "floorplan_svg"
    floorplan_dxf = "floorplan_dxf"
    visual_point_cloud = "visual_point_cloud"
    dollhouse_point_cloud = "dollhouse_point_cloud"
    room_model = "room_model"
    room_model_texture = "room_model_texture"
    cad_drawing_a3_png = "cad_drawing_a3_png"
    cad_drawing_a3_pdf = "cad_drawing_a3_pdf"
    cad_drawing_a4_png = "cad_drawing_a4_png"
    cad_drawing_a4_pdf = "cad_drawing_a4_pdf"
    cad_wall_elevations = "cad_wall_elevations"
    cad_quality_report = "cad_quality_report"
    metrics_mm = "metrics_mm"
    floor_neda_png = "floor_neda_png"
    floor_neda_pdf = "floor_neda_pdf"
    floor_tiling_png = "floor_tiling_png"
    floor_tiling_pdf = "floor_tiling_pdf"
    floor_plywood_png = "floor_plywood_png"
    floor_plywood_pdf = "floor_plywood_pdf"
    floor_cf_png = "floor_cf_png"
    floor_cf_pdf = "floor_cf_pdf"
    floor_combined_png = "floor_combined_png"



class SessionUploadResponse(BaseModel):
    session_id: str
    status: SessionStatus
    created_at: str

