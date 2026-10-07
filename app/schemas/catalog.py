from typing import List

from pydantic import BaseModel


class TatamiStandard(BaseModel):
    name: str
    width_m: float
    length_m: float


class TatamiCatalogResponse(BaseModel):
    standards: List[TatamiStandard]
    available_layouts: List[str]
    soft_limit_mm: float
    hard_limit_mm: float
    max_skew_angle_deg: float


class WallpaperStandard(BaseModel):
    name: str
    width_m: float
    roll_length_m: float


class WallpaperCatalogResponse(BaseModel):
    standards: List[WallpaperStandard]
    supported_match_types: List[str]
    default_trim_m: float
    default_overlap_m: float
