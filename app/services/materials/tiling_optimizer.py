# Ported from 3D-Estimate backend/api/tiling_optimizer.py
"""
Tiling Optimizer Engine
========================
Dense grid search optimizer for floor tiling layout.

Search space:
- offset_x: [0, tile_module_x) with configurable step
- offset_y: [0, tile_module_y) with configurable step
- orientation: landscape (0°) and portrait (90°) if allowed

For each candidate layout:
1. Place grid of tiles clipped to floor polygon (Shapely)
2. Classify each piece: full / cut (with precise geometry)
3. Detect small and forbidden pieces by configurable thresholds
4. Score using weighted multi-criteria function
5. Return best layout + detailed cutting plan
"""

import math
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

try:
    from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
    from shapely.ops import orient, unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class TilePiece:
    """A single tile piece placed on the floor."""
    piece_id: str                    # "F1", "C1", etc.
    rect: Tuple[float, float, float, float]  # (x, y, w, h) of the original tile grid cell
    piece_type: str                  # "full" or "cut"
    width_mm: float                  # Actual piece width (bounding box)
    height_mm: float                 # Actual piece height (bounding box)
    area_mm2: float                  # Actual area of the piece
    shape_type: str = "Rect"         # "Rect", "Triangle", "L-Shape", "Complex"
    vertices: List = field(default_factory=list)
    edges: List = field(default_factory=list)
    intersection: Any = None         # Shapely geometry for rendering
    min_side_mm: float = 0.0         # min(width, height) — for penalty checks
    source_stock_tile: str = ""      # "S1", "S2" — set by CuttingPlanGenerator

    def __post_init__(self):
        self.min_side_mm = min(self.width_mm, self.height_mm)


@dataclass
class TileLayout:
    """One candidate layout: all placed pieces + statistics."""
    pieces: List[TilePiece] = field(default_factory=list)
    orientation: str = "landscape"   # "landscape" or "portrait"
    offset_x: float = 0.0
    offset_y: float = 0.0
    tile_length_mm: float = 0.0     # Effective length (may be swapped for portrait)
    tile_width_mm: float = 0.0      # Effective width
    room_area_mm2: float = 0.0
    score: float = -float('inf')

    # Computed stats
    full_count: int = 0
    cut_count: int = 0
    total_placed_area: float = 0.0
    num_small_pieces: int = 0        # Pieces with min_side < avoid_threshold
    num_forbidden_pieces: int = 0    # Pieces with min_side < forbidden_threshold
    num_cut_groups: int = 0          # Number of distinct cut groups
    coverage_percent: float = 0.0
    uncovered_area_mm2: float = 0.0


@dataclass
class StockTileCut:
    """One stock tile and the pieces cut from it."""
    stock_tile_id: str
    pieces: List[Dict] = field(default_factory=list)  # [{piece_id, w, h, x, y}]
    waste_mm2: float = 0.0
    waste_percent: float = 0.0


@dataclass
class CuttingPlan:
    """Complete cutting plan mapping stock tiles → cut pieces."""
    stock_tiles: List[StockTileCut] = field(default_factory=list)
    total_stock_tiles: int = 0
    total_waste_m2: float = 0.0
    total_waste_percent: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Layout Scorer
# ═══════════════════════════════════════════════════════════════

class LayoutScorer:
    """
    Multi-criteria scoring with configurable weights.

    Priority order (center-aligned, symmetry-first):
    1. Zero uncovered area
    2. No forbidden pieces (< forbidden_threshold)
    3. No small pieces (< avoid_threshold)
    4. Symmetric opposing edges (center-aligned)
    5. Fewer cut groups
    6. More full tiles
    7. Fewer stock tiles to cut
    8. Less waste
    """

    def __init__(self, avoid_threshold_mm: float = 100.0,
                 forbidden_threshold_mm: float = 50.0):
        self.avoid_threshold = avoid_threshold_mm
        self.forbidden_threshold = forbidden_threshold_mm

    def score(self, layout: TileLayout, min_x: float = 0.0,
              min_y: float = 0.0, step_x: float = 1.0,
              step_y: float = 1.0, max_x: float = 0.0,
              max_y: float = 0.0) -> float:
        s = 0.0

        # 1. Uncovered area — absolute priority
        if layout.room_area_mm2 > 0:
            uncovered_ratio = layout.uncovered_area_mm2 / layout.room_area_mm2
            s -= 1_000_000 * uncovered_ratio

        # 2. Forbidden pieces — very heavy penalty
        s -= 50_000 * layout.num_forbidden_pieces

        # 3. Small pieces — heavy penalty
        s -= 5_000 * layout.num_small_pieces

        # 3b. Prefer larger cut pieces (penalize by inverse of min-side)
        for piece in layout.pieces:
            if piece.piece_type == "cut":
                if piece.min_side_mm < self.avoid_threshold:
                    s -= 0.5 * (self.avoid_threshold - piece.min_side_mm)

        # 4. Symmetry — opposing edges should have equal-sized cuts
        if max_x > min_x and max_y > min_y:
            symmetry_diff = self._compute_symmetry_diff(
                layout.pieces, min_x, min_y, max_x, max_y, step_x, step_y)
            s -= 200 * symmetry_diff

        # 5. Fewer cut groups = fewer unique cut operations
        s -= 100 * layout.num_cut_groups

        # 6. More full tiles = better
        s += 10 * layout.full_count

        # 7. Fewer cut pieces (proxy for stock tiles to cut)
        s -= 8 * layout.cut_count

        # 8. Less waste (per mm²)
        waste_mm2 = sum(
            (p.rect[2] * p.rect[3]) - p.area_mm2
            for p in layout.pieces if p.piece_type == "cut"
        )
        s -= 0.001 * waste_mm2

        return s

    def _compute_symmetry_diff(self, pieces, min_x, min_y, max_x, max_y,
                               step_x, step_y):
        """Compute asymmetry between opposing edge cuts (lower = more symmetric)."""
        edge_tol_x = step_x * 0.5
        edge_tol_y = step_y * 0.5

        left_widths = []
        right_widths = []
        top_heights = []
        bottom_heights = []

        for p in pieces:
            if p.piece_type != "cut":
                continue
            px, py, pw, ph = p.rect
            # Left edge
            if px < min_x + edge_tol_x:
                left_widths.append(p.width_mm)
            # Right edge
            if px + pw > max_x - edge_tol_x:
                right_widths.append(p.width_mm)
            # Top edge (smaller y)
            if py < min_y + edge_tol_y:
                top_heights.append(p.height_mm)
            # Bottom edge (larger y)
            if py + ph > max_y - edge_tol_y:
                bottom_heights.append(p.height_mm)

        diff = 0.0
        if left_widths and right_widths:
            avg_l = sum(left_widths) / len(left_widths)
            avg_r = sum(right_widths) / len(right_widths)
            diff += abs(avg_l - avg_r)
        if top_heights and bottom_heights:
            avg_t = sum(top_heights) / len(top_heights)
            avg_b = sum(bottom_heights) / len(bottom_heights)
            diff += abs(avg_t - avg_b)

        return diff


# ═══════════════════════════════════════════════════════════════
#  Merged Piece (Virtual Piece wrapper)
# ═══════════════════════════════════════════════════════════════

class MergedPiece:
    def __init__(self, pieces: List[TilePiece], width_mm: float, height_mm: float, merge_axis: str):
        self.pieces = pieces
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.area_mm2 = sum(p.area_mm2 for p in pieces)
        self.merge_axis = merge_axis  # "x", "y", or "grid"
        self.piece_id = "+".join(p.piece_id for p in pieces)


# ═══════════════════════════════════════════════════════════════
#  Cutting Plan Generator
# ═══════════════════════════════════════════════════════════════

class CuttingPlanGenerator:
    """
    2D guillotine bin-packing with rotation for cut pieces.

    Maps cut pieces → stock tiles, computing waste per stock tile.
    """

    def __init__(self, tile_length_mm: float, tile_width_mm: float):
        self.tile_w = tile_length_mm
        self.tile_h = tile_width_mm
        self.tile_area = tile_length_mm * tile_width_mm

    def _merge_compatible_pieces(self, cut_pieces: List[TilePiece]) -> List[Any]:
        # Group pieces by exact rounded dimensions and shape
        groups = defaultdict(list)
        for p in cut_pieces:
            key = (
                round(p.width_mm / 5) * 5,
                round(p.height_mm / 5) * 5,
                p.shape_type
            )
            groups[key].append(p)

        merged_items = []
        for key, pieces in groups.items():
            rw, rh, shape_type = key
            
            # How many can fit along X or Y axis?
            max_nx = int(math.floor(self.tile_w / rw)) if rw > 0 else 1
            max_ny = int(math.floor(self.tile_h / rh)) if rh > 0 else 1
            # Check rotation
            max_nx_rot = int(math.floor(self.tile_w / rh)) if rh > 0 else 1
            max_ny_rot = int(math.floor(self.tile_h / rw)) if rw > 0 else 1
            
            # Max pieces that can fit in one stock tile without rotation vs with rotation
            capacity_normal = max_nx * max_ny
            capacity_rotated = max_nx_rot * max_ny_rot
            
            best_capacity = max(capacity_normal, capacity_rotated)
            if best_capacity <= 1:
                # Can't even merge 2, just append individually
                merged_items.extend(pieces)
                continue
                
            # We will merge up to `best_capacity` pieces together
            idx = 0
            while idx < len(pieces):
                chunk_size = min(best_capacity, len(pieces) - idx)
                if chunk_size == 1:
                    merged_items.append(pieces[idx])
                    idx += 1
                else:
                    chunk = pieces[idx : idx + chunk_size]
                    
                    # Compute union bounds purely for the packing representation
                    # We will lay them out in a grid
                    if capacity_normal >= capacity_rotated:
                        cols, rows = max_nx, max_ny
                    else:
                        cols, rows = max_nx_rot, max_ny_rot
                        # Swap virtual item dims to represent rotated pack
                        rw, rh = rh, rw
                        
                    # Calculate bounding box of the packed chunk
                    actual_cols = min(cols, chunk_size)
                    actual_rows = int(math.ceil(chunk_size / cols))
                    
                    mp = MergedPiece(
                        pieces=chunk,
                        width_mm=actual_cols * rw,
                        height_mm=actual_rows * rh,
                        merge_axis="grid"  # Handled in _place_merged_piece
                    )
                    # Store packing info for _place_merged_piece
                    mp._pack_cols = cols
                    mp._pack_rw = rw
                    mp._pack_rh = rh
                    mp._is_rotated = (capacity_rotated > capacity_normal)
                    
                    merged_items.append(mp)
                    idx += chunk_size
                    
        return merged_items

    def _place_merged_piece(self, st: StockTileCut, piece: MergedPiece, origin_x: float, origin_y: float, is_rot: bool, stock_id: str):
        if piece.merge_axis == "grid":
            cols = piece._pack_cols
            pw = piece._pack_rw
            ph = piece._pack_rh
            pack_rotated = piece._is_rotated

            # If the bin packer rotated the WHOLE block on top of its internal rotation
            # we need to swap things
            final_rot = (pack_rotated != is_rot)

            for i, p in enumerate(piece.pieces):
                row = i // cols
                col = i % cols
                if not final_rot:
                    cx = origin_x + col * pw
                    cy = origin_y + row * ph
                    st.pieces.append({
                        "piece_id": p.piece_id, "w": round(p.width_mm, 1), "h": round(p.height_mm, 1),
                        "x": round(cx, 1), "y": round(cy, 1), "is_merged": True
                    })
                else:
                    # Swap axes for placement
                    cx = origin_x + row * ph
                    cy = origin_y + col * pw
                    st.pieces.append({
                        "piece_id": p.piece_id, "w": round(p.height_mm, 1), "h": round(p.width_mm, 1),
                        "x": round(cx, 1), "y": round(cy, 1), "is_merged": True
                    })
                p.source_stock_tile = stock_id
        else:
            p1, p2 = piece.pieces
            if not is_rot:
                if piece.merge_axis == "x":
                    st.pieces.append({"piece_id": p1.piece_id, "w": round(p1.width_mm, 1), "h": round(p1.height_mm, 1), "x": round(origin_x, 1), "y": round(origin_y, 1), "is_merged": True})
                    st.pieces.append({"piece_id": p2.piece_id, "w": round(p2.width_mm, 1), "h": round(p2.height_mm, 1), "x": round(origin_x + p1.width_mm, 1), "y": round(origin_y, 1), "is_merged": True})
                else:
                    st.pieces.append({"piece_id": p1.piece_id, "w": round(p1.width_mm, 1), "h": round(p1.height_mm, 1), "x": round(origin_x, 1), "y": round(origin_y, 1), "is_merged": True})
                    st.pieces.append({"piece_id": p2.piece_id, "w": round(p2.width_mm, 1), "h": round(p2.height_mm, 1), "x": round(origin_x, 1), "y": round(origin_y + p1.height_mm, 1), "is_merged": True})
            else:
                if piece.merge_axis == "x":
                    st.pieces.append({"piece_id": p1.piece_id, "w": round(p1.height_mm, 1), "h": round(p1.width_mm, 1), "x": round(origin_x, 1), "y": round(origin_y, 1), "is_merged": True})
                    st.pieces.append({"piece_id": p2.piece_id, "w": round(p2.height_mm, 1), "h": round(p2.width_mm, 1), "x": round(origin_x + p1.width_mm, 1), "y": round(origin_y, 1), "is_merged": True})
                else:
                    st.pieces.append({"piece_id": p1.piece_id, "w": round(p1.height_mm, 1), "h": round(p1.width_mm, 1), "x": round(origin_x, 1), "y": round(origin_y, 1), "is_merged": True})
                    st.pieces.append({"piece_id": p2.piece_id, "w": round(p2.height_mm, 1), "h": round(p2.width_mm, 1), "x": round(origin_x, 1), "y": round(origin_y + p1.height_mm, 1), "is_merged": True})
            p1.source_stock_tile = stock_id
            p2.source_stock_tile = stock_id

    def generate(self, cut_pieces: List[TilePiece]) -> CuttingPlan:
        if not cut_pieces:
            return CuttingPlan()

        merged_pieces = self._merge_compatible_pieces(cut_pieces)
        sorted_pieces = sorted(merged_pieces, key=lambda p: p.area_mm2, reverse=True)

        remnants: List[Dict] = []
        stock_tiles: List[StockTileCut] = []
        stock_count = 0

        for piece in sorted_pieces:
            pw = piece.width_mm
            ph = piece.height_mm
            if pw <= 0 or ph <= 0:
                continue

            placed = False
            for idx, rem in enumerate(remnants):
                fit_normal = (pw <= rem["w"] + 0.5 and ph <= rem["h"] + 0.5)
                fit_rotated = (ph <= rem["w"] + 0.5 and pw <= rem["h"] + 0.5)

                if fit_normal or fit_rotated:
                    fit_w, fit_h = (pw, ph) if fit_normal else (ph, pw)
                    is_rot = not fit_normal
                    
                    stock_id = rem["stock_id"]
                    st = next(s for s in stock_tiles if s.stock_tile_id == stock_id)
                    
                    if isinstance(piece, MergedPiece):
                        self._place_merged_piece(st, piece, rem["x"], rem["y"], is_rot, stock_id)
                    else:
                        st.pieces.append({
                            "piece_id": piece.piece_id,
                            "w": round(fit_w, 1), "h": round(fit_h, 1),
                            "x": round(rem["x"], 1), "y": round(rem["y"], 1),
                        })
                        piece.source_stock_tile = stock_id

                    rw, rh = rem["w"], rem["h"]
                    rx, ry = rem["x"], rem["y"]
                    remnants.pop(idx)

                    if rw - fit_w > 2.0:
                        remnants.append({"w": rw - fit_w, "h": rh, "stock_id": stock_id, "x": rx + fit_w, "y": ry})
                    if rh - fit_h > 2.0:
                        remnants.append({"w": fit_w, "h": rh - fit_h, "stock_id": stock_id, "x": rx, "y": ry + fit_h})
                    
                    placed = True
                    break

            if not placed:
                stock_count += 1
                stock_id = f"S{stock_count}"
                st = StockTileCut(stock_tile_id=stock_id)
                stock_tiles.append(st)

                rw, rh = self.tile_w, self.tile_h
                if pw <= rw + 0.5 and ph <= rh + 0.5:
                    fit_w, fit_h = pw, ph
                    is_rot = False
                elif ph <= rw + 0.5 and pw <= rh + 0.5:
                    fit_w, fit_h = ph, pw
                    is_rot = True
                else:
                    fit_w, fit_h = min(pw, rw), min(ph, rh)
                    is_rot = False

                if isinstance(piece, MergedPiece):
                    self._place_merged_piece(st, piece, 0.0, 0.0, is_rot, stock_id)
                else:
                    st.pieces.append({
                        "piece_id": piece.piece_id,
                        "w": round(fit_w, 1), "h": round(fit_h, 1),
                        "x": 0.0, "y": 0.0,
                    })
                    piece.source_stock_tile = stock_id

                if rw - fit_w > 2.0:
                    remnants.append({"w": rw - fit_w, "h": rh, "stock_id": stock_id, "x": fit_w, "y": 0.0})
                if rh - fit_h > 2.0:
                    remnants.append({"w": fit_w, "h": rh - fit_h, "stock_id": stock_id, "x": 0.0, "y": fit_h})

        total_piece_area = 0.0
        for st in stock_tiles:
            pieces_area = sum(p["w"] * p["h"] for p in st.pieces)
            total_piece_area += pieces_area
            st.waste_mm2 = max(0, self.tile_area - pieces_area)
            st.waste_percent = round((st.waste_mm2 / self.tile_area * 100) if self.tile_area > 0 else 0, 1)

        total_waste_mm2 = sum(st.waste_mm2 for st in stock_tiles)
        total_stock_area = stock_count * self.tile_area

        return CuttingPlan(
            stock_tiles=stock_tiles,
            total_stock_tiles=stock_count,
            total_waste_m2=round(total_waste_mm2 / 1e6, 4),
            total_waste_percent=round((total_waste_mm2 / total_stock_area * 100) if total_stock_area > 0 else 0, 1)
        )


class TilingOptimizer:
    """
    Dense grid search optimizer for floor tiling.
    """

    SEARCH_MODES = {
        "fast":     {"step_divisor": 6},
        "balanced": {"step_divisor": 12},
        "precise":  {"step_divisor": 24},
        "japanese_standard": {"step_divisor": 1},
    }

    def __init__(
        self,
        tile_length_mm: float,
        tile_width_mm: float,
        grout_gap_mm: float = 2.0,
        avoid_piece_side_under_mm: float = 100.0,
        forbidden_piece_side_under_mm: float = 50.0,
        search_step_mm: Optional[float] = None,
        allow_rotation: bool = True,
        search_mode: str = "balanced",
    ):
        self.tile_length_mm = tile_length_mm
        self.tile_width_mm = tile_width_mm
        self.grout_gap_mm = grout_gap_mm
        self.avoid_threshold = avoid_piece_side_under_mm
        self.forbidden_threshold = forbidden_piece_side_under_mm
        self.allow_rotation = allow_rotation
        self.search_mode = search_mode

        if search_step_mm is not None and search_step_mm > 0:
            self.search_step = search_step_mm
        else:
            mode_cfg = self.SEARCH_MODES.get(search_mode, self.SEARCH_MODES["balanced"])
            if search_mode == "japanese_standard":
                self.search_step = float('inf')
            else:
                module_x = tile_length_mm + grout_gap_mm
                self.search_step = max(5.0, module_x / mode_cfg["step_divisor"])

        self.scorer = LayoutScorer(
            avoid_threshold_mm=avoid_piece_side_under_mm,
            forbidden_threshold_mm=forbidden_piece_side_under_mm
        )

    def _prepare_room(self, polygon_points_mm: List[Tuple[float, float]]) -> Tuple["ShapelyPolygon", float, float, float, float, float]:
        if not HAS_SHAPELY:
            raise RuntimeError("Shapely is required for tiling optimization. Install with: pip install shapely")

        pts = np.array(polygon_points_mm, dtype=float)
        room_poly = ShapelyPolygon(pts)
        if not room_poly.is_valid:
            room_poly = room_poly.buffer(0)
        room_area = room_poly.area
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        return room_poly, room_area, min_x, min_y, max_x, max_y

    def _build_orientations(self) -> List[Tuple[str, float, float]]:
        orientations = [("landscape", self.tile_length_mm, self.tile_width_mm)]
        if self.allow_rotation and self.tile_length_mm != self.tile_width_mm:
            orientations.append(("portrait", self.tile_width_mm, self.tile_length_mm))
        return orientations

    def _build_allowed_offsets(self, min_coord, max_coord, module_size):
        if self.search_mode == "japanese_standard":
            return [0.0]
        return self._build_center_aligned_offsets(min_coord, max_coord, module_size)

    def _build_center_aligned_offsets(self, min_coord, max_coord, module_size):
        """
        Compute offsets that center the tile grid within [min_coord, max_coord],
        so opposing edge remnants are equal.
        """
        room_span = max_coord - min_coord
        if room_span <= 0 or module_size <= 0:
            return [0.0]

        n_full = int(math.floor(room_span / module_size))
        remainder = room_span - n_full * module_size
        center_offset = remainder / 2.0

        candidates = {center_offset}

        for delta in [self.grout_gap_mm, -self.grout_gap_mm,
                      self.grout_gap_mm * 2, -self.grout_gap_mm * 2]:
            candidate = center_offset + delta
            if 0 <= candidate < module_size:
                candidates.add(candidate)

        candidates.add(0.0)
        return sorted(candidates)

    def _build_result(self, best_layout: TileLayout, room_area: float, candidates_evaluated: int, extra_stats: Dict = None) -> Dict[str, Any]:
        cut_pieces = [p for p in best_layout.pieces if p.piece_type == "cut"]
        cpg = CuttingPlanGenerator(self.tile_length_mm, self.tile_width_mm)
        cutting_plan = cpg.generate(cut_pieces)

        total_buy_tiles = best_layout.full_count + cutting_plan.total_stock_tiles
        warnings = self._generate_warnings(best_layout)
        cut_groups = self._group_cut_tiles(cut_pieces, self.tile_length_mm, self.tile_width_mm)

        tile_area = self.tile_length_mm * self.tile_width_mm
        cut_waste_mm2 = sum(tile_area - p.area_mm2 for p in cut_pieces)
        total_buy_area_m2 = (total_buy_tiles * tile_area) / 1e6
        floor_area_m2 = round(room_area / 1e6, 4)
        gross_waste_m2 = max(0.0, total_buy_area_m2 - (room_area / 1e6))
        gross_waste_percent = (gross_waste_m2 / (room_area / 1e6) * 100) if room_area > 0 else 0.0

        tiles_legacy = []
        for p in best_layout.pieces:
            t = {
                "rect": p.rect, "type": p.piece_type, "area_mm2": p.area_mm2,
                "shape_type": p.shape_type, "vertices": p.vertices, "edges": p.edges,
            }
            if p.piece_type == "cut":
                t["cut_length_mm"] = round(p.width_mm, 1)
                t["cut_width_mm"] = round(p.height_mm, 1)
            tiles_legacy.append(t)

        stats = {
            "total_tiles": best_layout.full_count + best_layout.cut_count,
            "full_tiles": best_layout.full_count,
            "cut_tiles": best_layout.cut_count,
            "total_buy_tiles": total_buy_tiles,
            "cut_tile_groups": cut_groups,
            "total_tile_area_m2": round(best_layout.total_placed_area / 1e6, 3),
            "total_buy_area_m2": round(total_buy_area_m2, 3),
            "waste_area_m2": round(max(0, cut_waste_mm2 / 1e6), 3),
            "gross_waste_area_m2": round(gross_waste_m2, 3),
            "gross_waste_percent": round(gross_waste_percent, 1),
            "coverage_percent": round(best_layout.coverage_percent, 1),
            "best_offset": (round(best_layout.offset_x, 1), round(best_layout.offset_y, 1)),
            "best_orientation": best_layout.orientation,
            "best_offset_x_mm": round(best_layout.offset_x, 1),
            "best_offset_y_mm": round(best_layout.offset_y, 1),
            "offset_x": round(best_layout.offset_x, 1),
            "offset_y": round(best_layout.offset_y, 1),
            "orientation": best_layout.orientation,
            "floor_area_m2": floor_area_m2,
            "room_area_m2": floor_area_m2,
            "search_candidates_evaluated": candidates_evaluated,
            "num_cut_groups": best_layout.num_cut_groups,
            "num_small_pieces": best_layout.num_small_pieces,
            "num_forbidden_pieces": best_layout.num_forbidden_pieces,
            "effective_tile_length_mm": best_layout.tile_length_mm,
            "effective_tile_width_mm": best_layout.tile_width_mm,
            "total_stock_tiles": cutting_plan.total_stock_tiles,
            "total_waste_percent": cutting_plan.total_waste_percent,
        }
        
        if extra_stats:
            stats.update(extra_stats)

        piece_dimensions = []
        for p in best_layout.pieces:
            if p.piece_type == "cut":
                piece_dimensions.append({
                    "piece_id": p.piece_id,
                    "width_mm": round(p.width_mm, 1),
                    "height_mm": round(p.height_mm, 1),
                    "shape_type": p.shape_type,
                    "source_stock_tile": p.source_stock_tile,
                })

        cutting_plan_dict = {
            "stock_tiles": [
                {
                    "stock_tile_id": st.stock_tile_id,
                    "pieces": st.pieces,
                    "waste_mm2": round(st.waste_mm2, 1),
                    "waste_percent": st.waste_percent,
                } for st in cutting_plan.stock_tiles
            ],
            "total_stock_tiles": cutting_plan.total_stock_tiles,
            "total_waste_m2": cutting_plan.total_waste_m2,
            "total_waste_percent": cutting_plan.total_waste_percent,
        }

        return {
            "tiles": tiles_legacy,
            "stats": stats,
            "cutting_plan": cutting_plan_dict,
            "warnings": warnings,
            "piece_dimensions": piece_dimensions,
        }

    def optimize(
        self,
        polygon_points_mm: List[Tuple[float, float]],
        deadline_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        if deadline_time is not None and time.monotonic() >= deadline_time:
            return {"skipped": True, "reason": "timeout"}

        room_poly, room_area, min_x, min_y, max_x, max_y = self._prepare_room(polygon_points_mm)
        orientations = self._build_orientations()

        best_layout: Optional[TileLayout] = None
        best_score = -float('inf')
        best_zero_forbidden_layout: Optional[TileLayout] = None
        best_zero_forbidden_score = -float('inf')
        candidates_evaluated = 0

        for orient_name, eff_length, eff_width in orientations:
            module_x = eff_length + self.grout_gap_mm
            module_y = eff_width + self.grout_gap_mm

            allowed_offsets_x = self._build_allowed_offsets(min_x, max_x, module_x)
            allowed_offsets_y = self._build_allowed_offsets(min_y, max_y, module_y)

            for ox in allowed_offsets_x:
                for oy in allowed_offsets_y:
                    if deadline_time is not None and time.monotonic() >= deadline_time:
                        return {"skipped": True, "reason": "timeout"}

                    layout = self._generate_layout(
                        room_poly, room_area, eff_length, eff_width,
                        self.grout_gap_mm, min_x, min_y, max_x, max_y,
                        ox, oy, orient_name
                    )
                    layout.score = self.scorer.score(
                        layout, min_x=min_x, min_y=min_y, step_x=module_x, step_y=module_y, max_x=max_x, max_y=max_y
                    )
                    candidates_evaluated += 1

                    if layout.num_forbidden_pieces == 0:
                        if layout.score > best_zero_forbidden_score:
                            best_zero_forbidden_score = layout.score
                            best_zero_forbidden_layout = layout

                    if layout.score > best_score:
                        best_score = layout.score
                        best_layout = layout

        # If some candidate has num_forbidden_pieces == 0, prefer it
        chosen_layout = best_zero_forbidden_layout if best_zero_forbidden_layout is not None else best_layout

        if chosen_layout is None:
            chosen_layout = self._generate_layout(
                room_poly, room_area, self.tile_length_mm, self.tile_width_mm,
                self.grout_gap_mm, min_x, min_y, max_x, max_y,
                0.0, 0.0, "landscape"
            )
            chosen_layout.score = self.scorer.score(chosen_layout, max_x=max_x, max_y=max_y)
            candidates_evaluated = 1

        return self._build_result(chosen_layout, room_area, candidates_evaluated)

    def _finalize_layout(self, layout: TileLayout, pieces: List[TilePiece], room_area: float, step_y: float):
        num_small = 0
        num_forbidden = 0
        for p in pieces:
            if p.piece_type == "cut":
                if p.min_side_mm < self.forbidden_threshold:
                    num_forbidden += 1
                elif p.min_side_mm < self.avoid_threshold:
                    num_small += 1

        pieces.sort(key=lambda p: (round(p.rect[1] / step_y), p.rect[0]))

        full_count = 0
        cut_count = 0
        for p in pieces:
            if p.piece_type == "full":
                full_count += 1
                p.piece_id = f"F{full_count}"
            else:
                cut_count += 1
                p.piece_id = f"C{cut_count}"

        layout.pieces = pieces
        layout.full_count = full_count
        layout.cut_count = cut_count
        layout.num_small_pieces = num_small
        layout.num_forbidden_pieces = num_forbidden

        cut_group_keys = set()
        for p in pieces:
            if p.piece_type == "cut":
                key = (round(p.width_mm / 5) * 5, round(p.height_mm / 5) * 5, p.shape_type)
                cut_group_keys.add(key)
        layout.num_cut_groups = len(cut_group_keys)

        layout.coverage_percent = (layout.total_placed_area / room_area * 100) if room_area > 0 else 0
        layout.uncovered_area_mm2 = max(0, room_area - layout.total_placed_area)
        return layout

    def _generate_layout(
        self, room_poly: "ShapelyPolygon", room_area: float, eff_length: float, eff_width: float,
        grout_gap: float, min_x: float, min_y: float, max_x: float, max_y: float,
        offset_x: float, offset_y: float, orientation: str,
    ) -> TileLayout:
        tile_area = eff_length * eff_width
        step_x = eff_length + grout_gap
        step_y = eff_width + grout_gap

        layout = TileLayout(
            orientation=orientation, offset_x=offset_x, offset_y=offset_y,
            tile_length_mm=eff_length, tile_width_mm=eff_width, room_area_mm2=room_area,
        )

        total_placed_area = 0.0
        pieces = []

        x = min_x - offset_x
        while x < max_x + step_x:
            y = min_y - offset_y
            while y < max_y + step_y:
                tile_rect = shapely_box(x, y, x + eff_length, y + eff_width)
                intersection = room_poly.intersection(tile_rect)

                if intersection.is_empty or intersection.area < 1.0:
                    y += step_y
                    continue

                coverage_ratio = intersection.area / tile_area
                bounds = intersection.bounds
                cut_w = bounds[2] - bounds[0]
                cut_h = bounds[3] - bounds[1]
                dims_match = (abs(cut_w - eff_length) < 5.0 and abs(cut_h - eff_width) < 5.0)

                if coverage_ratio > 0.99 and dims_match:
                    pieces.append(TilePiece(piece_id="", rect=(x, y, eff_length, eff_width), piece_type="full", width_mm=eff_length, height_mm=eff_width, area_mm2=tile_area, shape_type="Rect"))
                    total_placed_area += tile_area
                else:
                    actual_area = intersection.area
                    shape_type, vertices, edges = self._extract_geometry(intersection, eff_length, eff_width)
                    pieces.append(TilePiece(piece_id="", rect=(x, y, eff_length, eff_width), piece_type="cut", width_mm=cut_w, height_mm=cut_h, area_mm2=actual_area, shape_type=shape_type, vertices=vertices, edges=edges, intersection=intersection))
                    total_placed_area += actual_area

                y += step_y
            x += step_x

        pieces = self._merge_adjacent_pieces(pieces, eff_length, eff_width)
        layout.total_placed_area = sum(p.area_mm2 for p in pieces)
        return self._finalize_layout(layout, pieces, room_area, step_y)

    def _merge_adjacent_pieces(self, pieces: List[TilePiece], eff_length: float, eff_width: float) -> List[TilePiece]:
        if not HAS_SHAPELY:
            return pieces

        full_tiles = [p for p in pieces if p.piece_type == "full"]
        cut_tiles = [p for p in pieces if p.piece_type == "cut"]

        if len(cut_tiles) <= 1:
            return pieces

        n = len(cut_tiles)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_adjacent(cut_tiles[i], cut_tiles[j]):
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        components = []
        for start in range(n):
            if visited[start]:
                continue
            component = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component)

        merged_cuts = []
        for comp in components:
            if len(comp) == 1:
                merged_cuts.append(cut_tiles[comp[0]])
                continue

            comp_pieces = [cut_tiles[idx] for idx in comp]

            all_have_geom = all(p.intersection is not None for p in comp_pieces)
            if not all_have_geom:
                merged_cuts.extend(comp_pieces)
                continue

            try:
                combo = unary_union([p.intersection for p in comp_pieces])
                if combo.is_empty:
                    merged_cuts.extend(comp_pieces)
                    continue
            except Exception:
                merged_cuts.extend(comp_pieces)
                continue

            bounds = combo.bounds
            c_w = bounds[2] - bounds[0]
            c_h = bounds[3] - bounds[1]

            fits = False
            if c_w <= eff_length + 1.0 and c_h <= eff_width + 1.0:
                fits = True
            elif c_w <= eff_width + 1.0 and c_h <= eff_length + 1.0:
                fits = True

            if fits and combo.geom_type in ('Polygon', 'MultiPolygon', 'GeometryCollection'):
                new_rect = (
                    min(p.rect[0] for p in comp_pieces),
                    min(p.rect[1] for p in comp_pieces),
                    max(p.rect[0] + p.rect[2] for p in comp_pieces) - min(p.rect[0] for p in comp_pieces),
                    max(p.rect[1] + p.rect[3] for p in comp_pieces) - min(p.rect[1] for p in comp_pieces),
                )
                shape_type, vertices, edges = self._extract_geometry(combo, eff_length, eff_width)
                new_piece = TilePiece(
                    piece_id="",
                    rect=new_rect,
                    piece_type="cut",
                    width_mm=c_w,
                    height_mm=c_h,
                    area_mm2=combo.area,
                    shape_type=shape_type,
                    vertices=vertices,
                    edges=edges,
                    intersection=combo,
                )
                merged_cuts.append(new_piece)
            else:
                merged_cuts.extend(self._pairwise_merge(comp_pieces, eff_length, eff_width))

        return full_tiles + merged_cuts

    def _are_adjacent(self, p1: TilePiece, p2: TilePiece) -> bool:
        gap_tol = self.grout_gap_mm + 1.0
        h_adj = (abs(p1.rect[0] + p1.rect[2] - p2.rect[0]) < gap_tol and abs(p1.rect[1] - p2.rect[1]) < gap_tol)
        h_adj_rev = (abs(p2.rect[0] + p2.rect[2] - p1.rect[0]) < gap_tol and abs(p2.rect[1] - p1.rect[1]) < gap_tol)
        v_adj = (abs(p1.rect[1] + p1.rect[3] - p2.rect[1]) < gap_tol and abs(p1.rect[0] - p2.rect[0]) < gap_tol)
        v_adj_rev = (abs(p2.rect[1] + p2.rect[3] - p1.rect[1]) < gap_tol and abs(p2.rect[0] - p1.rect[0]) < gap_tol)
        return h_adj or h_adj_rev or v_adj or v_adj_rev

    def _pairwise_merge(self, cut_pieces: List[TilePiece], eff_length: float, eff_width: float) -> List[TilePiece]:
        result = list(cut_pieces)
        merged_any = True
        while merged_any:
            merged_any = False
            i = 0
            while i < len(result):
                j = i + 1
                while j < len(result):
                    p1 = result[i]
                    p2 = result[j]

                    if not self._are_adjacent(p1, p2):
                        j += 1
                        continue

                    if p1.intersection is None or p2.intersection is None:
                        j += 1
                        continue

                    combo = unary_union([p1.intersection, p2.intersection])
                    if combo.is_empty:
                        j += 1
                        continue

                    bounds = combo.bounds
                    c_w = bounds[2] - bounds[0]
                    c_h = bounds[3] - bounds[1]

                    fits = False
                    if c_w <= eff_length + 1.0 and c_h <= eff_width + 1.0:
                        fits = True
                    elif c_w <= eff_width + 1.0 and c_h <= eff_length + 1.0:
                        fits = True

                    if fits and combo.geom_type in ('Polygon', 'MultiPolygon'):
                        new_rect = (
                            min(p1.rect[0], p2.rect[0]),
                            min(p1.rect[1], p2.rect[1]),
                            max(p1.rect[0]+p1.rect[2], p2.rect[0]+p2.rect[2]) - min(p1.rect[0], p2.rect[0]),
                            max(p1.rect[1]+p1.rect[3], p2.rect[1]+p2.rect[3]) - min(p1.rect[1], p2.rect[1]),
                        )
                        shape_type, vertices, edges = self._extract_geometry(combo, eff_length, eff_width)
                        new_piece = TilePiece(
                            piece_id="",
                            rect=new_rect,
                            piece_type="cut",
                            width_mm=c_w,
                            height_mm=c_h,
                            area_mm2=combo.area,
                            shape_type=shape_type,
                            vertices=vertices,
                            edges=edges,
                            intersection=combo,
                        )
                        result.pop(j)
                        result[i] = new_piece
                        merged_any = True
                        break
                    j += 1
                if merged_any:
                    break
                i += 1
        return result

    def _extract_geometry(self, intersection, eff_length, eff_width):
        shape_type = "Rect"
        vertices = []
        edges = []

        try:
            poly_geom = None
            if intersection.geom_type == 'Polygon':
                poly_geom = intersection
            elif intersection.geom_type == 'MultiPolygon':
                if not intersection.is_empty:
                    poly_geom = max(intersection.geoms, key=lambda g: g.area)
            elif intersection.geom_type == 'GeometryCollection':
                polys = [g for g in intersection.geoms if g.geom_type == 'Polygon']
                if polys:
                    poly_geom = max(polys, key=lambda g: g.area)
                else:
                    multipolys = [g for g in intersection.geoms if g.geom_type == 'MultiPolygon']
                    if multipolys:
                        largest_mp = max(multipolys, key=lambda g: g.area)
                        if not largest_mp.is_empty:
                            poly_geom = max(largest_mp.geoms, key=lambda g: g.area)

            if poly_geom is not None and not poly_geom.is_empty:
                poly_geom = orient(poly_geom, sign=1.0)
                simplified = poly_geom.simplify(2.0, preserve_topology=True)
                coords = list(simplified.exterior.coords)
                vertices = coords[:-1]
                num_v = len(vertices)

                if num_v == 3:
                    shape_type = "Triangle"
                elif num_v == 4:
                    shape_type = "Rect"
                elif num_v in (5, 6):
                    shape_type = "L-Shape"
                else:
                    shape_type = "Complex"

                for i in range(len(coords) - 1):
                    p1 = coords[i]
                    p2 = coords[i + 1]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = math.sqrt(dx * dx + dy * dy)
                    orientation = "H" if abs(dy) < abs(dx) else "V"
                    edges.append({"from": p1, "to": p2, "length_mm": round(length, 1), "orientation": orientation})
        except Exception:
            shape_type = "Rect"

        return shape_type, vertices, edges

    def _generate_warnings(self, layout: TileLayout) -> List[str]:
        warnings = []
        if layout.num_forbidden_pieces > 0:
            warnings.append(f"! {layout.num_forbidden_pieces} piece(s) have a side shorter than {self.forbidden_threshold:.0f}mm (forbidden threshold). Consider adjusting tile offset or size.")
        if layout.num_small_pieces > 0:
            warnings.append(f"! {layout.num_small_pieces} piece(s) have a side shorter than {self.avoid_threshold:.0f}mm (avoid threshold).")
        if layout.coverage_percent < 99.5:
            warnings.append(f"! Coverage is {layout.coverage_percent:.1f}% — some floor area may not be covered.")
        return warnings

    def _group_cut_tiles(self, cut_pieces: List[TilePiece], tile_length_mm: float, tile_width_mm: float) -> List[Dict]:
        group_counts = defaultdict(int)
        group_reps = {}
        for p in cut_pieces:
            key = (round(p.width_mm / 5) * 5, round(p.height_mm / 5) * 5, p.shape_type)
            group_counts[key] += 1
            if key not in group_reps:
                group_reps[key] = p

        result = []
        for key, count in sorted(group_counts.items(), key=lambda x: -x[1]):
            rep_piece = group_reps[key]
            exact_cl = round(rep_piece.width_mm, 1)
            exact_cw = round(rep_piece.height_mm, 1)
            edges_str = "/".join(f"{e['length_mm']:.1f}" for e in rep_piece.edges) if rep_piece.edges else ""
                
            result.append({
                "cut_length_mm": exact_cl,
                "cut_width_mm": exact_cw,
                "shape_type": key[2],
                "count": count,
                "area_per_piece_m2": round((exact_cl * exact_cw) / 1e6, 4),
                "edges_str": edges_str
            })
        return result


class BattenAlignedTilingOptimizer(TilingOptimizer):
    def __init__(
        self,
        tile_length_mm: float,
        tile_width_mm: float,
        batten_width_mm: float,
        batten_spacing_mm: float,
        tiling_direction: str = "auto",
        grout_gap_mm: float = 2.0,
        avoid_piece_side_under_mm: float = 100.0,
        forbidden_piece_side_under_mm: float = 50.0,
        search_step_mm: Optional[float] = None,
        allow_rotation: bool = True,
        search_mode: str = "balanced",
    ):
        super().__init__(
            tile_length_mm=tile_length_mm,
            tile_width_mm=tile_width_mm,
            grout_gap_mm=grout_gap_mm,
            avoid_piece_side_under_mm=avoid_piece_side_under_mm,
            forbidden_piece_side_under_mm=forbidden_piece_side_under_mm,
            search_step_mm=search_step_mm,
            allow_rotation=allow_rotation,
            search_mode=search_mode,
        )
        self.batten_width_mm = batten_width_mm
        self.batten_spacing_mm = batten_spacing_mm
        self.tiling_direction = tiling_direction

    def _build_orientations(self) -> List[Tuple[str, float, float]]:
        orientations = []
        if self.tiling_direction == "horizontal":
            orientations.append(("landscape", self.tile_length_mm, self.tile_width_mm))
        elif self.tiling_direction == "vertical":
            orientations.append(("portrait", self.tile_width_mm, self.tile_length_mm))
        else:
            orientations.append(("landscape", self.tile_length_mm, self.tile_width_mm))
            if self.allow_rotation and self.tile_length_mm != self.tile_width_mm:
                orientations.append(("portrait", self.tile_width_mm, self.tile_length_mm))
        return orientations

    def _compute_batten_positions(self, min_coord: float, max_coord: float) -> List[float]:
        half_w = self.batten_width_mm / 2.0
        positions = [min_coord + half_w]
        pos = min_coord + half_w + self.batten_spacing_mm
        last = max_coord - half_w
        while pos < last - 1.0:
            positions.append(pos)
            pos += self.batten_spacing_mm
        positions.append(last)
        return positions

    def _build_allowed_offsets(self, min_coord, max_coord, module_size, orient_name="landscape", axis="x"):
        half_w = self.batten_width_mm / 2.0
        if (orient_name == "landscape" and axis == "x") or (orient_name == "portrait" and axis == "y"):
            allowed_offsets = []
            for k in range(-100, 100):
                o = (half_w + k * self.batten_spacing_mm) % module_size
                if not any(abs(o - existing) < 0.5 for existing in allowed_offsets):
                    allowed_offsets.append(o)
            allowed_offsets.sort()
            return allowed_offsets
        else:
            return self._build_center_aligned_offsets(min_coord, max_coord, module_size)

    def optimize(
        self,
        polygon_points_mm: List[Tuple[float, float]],
        deadline_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        if deadline_time is not None and time.monotonic() >= deadline_time:
            return {"skipped": True, "reason": "timeout"}

        room_poly, room_area, min_x, min_y, max_x, max_y = self._prepare_room(polygon_points_mm)
        orientations = self._build_orientations()

        best_layout: Optional[TileLayout] = None
        best_score = -float('inf')
        best_zero_forbidden_layout: Optional[TileLayout] = None
        best_zero_forbidden_score = -float('inf')
        candidates_evaluated = 0

        for orient_name, eff_length, eff_width in orientations:
            module_x = eff_length + self.grout_gap_mm
            module_y = eff_width + self.grout_gap_mm

            allowed_offsets_x = self._build_allowed_offsets(min_x, max_x, module_x, orient_name, "x")
            allowed_offsets_y = self._build_allowed_offsets(min_y, max_y, module_y, orient_name, "y")

            for ox in allowed_offsets_x:
                for oy in allowed_offsets_y:
                    if deadline_time is not None and time.monotonic() >= deadline_time:
                        return {"skipped": True, "reason": "timeout"}

                    layout = self._generate_layout(
                        room_poly, room_area, eff_length, eff_width,
                        self.grout_gap_mm, min_x, min_y, max_x, max_y,
                        ox, oy, orient_name
                    )
                    layout.score = self.scorer.score(
                        layout, min_x=min_x, min_y=min_y, step_x=module_x, step_y=module_y, max_x=max_x, max_y=max_y
                    )
                    candidates_evaluated += 1

                    if layout.num_forbidden_pieces == 0:
                        if layout.score > best_zero_forbidden_score:
                            best_zero_forbidden_score = layout.score
                            best_zero_forbidden_layout = layout

                    if layout.score > best_score:
                        best_score = layout.score
                        best_layout = layout

        chosen_layout = best_zero_forbidden_layout if best_zero_forbidden_layout is not None else best_layout

        if chosen_layout is None:
            chosen_layout = self._generate_layout(
                room_poly, room_area, self.tile_length_mm, self.tile_width_mm,
                self.grout_gap_mm, min_x, min_y, max_x, max_y,
                0.0, 0.0, "landscape"
            )
            chosen_layout.score = self.scorer.score(
                chosen_layout, min_x=min_x, min_y=min_y,
                step_x=self.tile_length_mm + self.grout_gap_mm,
                step_y=self.tile_width_mm + self.grout_gap_mm,
                max_x=max_x, max_y=max_y
            )
            candidates_evaluated = 1

        if chosen_layout.orientation == "landscape":
            batten_positions = self._compute_batten_positions(min_x, max_x)
        else:
            batten_positions = self._compute_batten_positions(min_y, max_y)

        extra_stats = {
            "batten_aligned": True,
            "batten_spacing_mm": self.batten_spacing_mm,
            "tiling_direction": "horizontal" if chosen_layout.orientation == "landscape" else "vertical",
            "batten_positions_mm": batten_positions,
        }
        
        return self._build_result(chosen_layout, room_area, candidates_evaluated, extra_stats)

    def _generate_layout(
        self, room_poly: "ShapelyPolygon", room_area: float, eff_length: float, eff_width: float,
        grout_gap: float, min_x: float, min_y: float, max_x: float, max_y: float,
        offset_x: float, offset_y: float, orientation: str,
    ) -> TileLayout:
        tile_area = eff_length * eff_width
        step_x = eff_length + grout_gap
        step_y = eff_width + grout_gap

        layout = TileLayout(
            orientation=orientation, offset_x=offset_x, offset_y=offset_y,
            tile_length_mm=eff_length, tile_width_mm=eff_width, room_area_mm2=room_area,
        )

        total_placed_area = 0.0
        pieces = []

        if orientation == "landscape":
            batten_centers = self._compute_batten_positions(min_x, max_x)
            
            nominal_xs = []
            cur_x = min_x + offset_x
            while cur_x > min_x: cur_x -= step_x
            while cur_x < max_x + step_x:
                nominal_xs.append(cur_x)
                cur_x += step_x
            
            x_coords = []
            for x in nominal_xs:
                if x <= min_x + 1.0: x_coords.append(min_x)
                elif x >= max_x - 1.0: x_coords.append(max_x)
                else: x_coords.append(min(batten_centers, key=lambda c: abs(c - x)))
            x_coords = sorted(list(set(x_coords)))
            if min_x not in x_coords: x_coords.insert(0, min_x)
            if max_x not in x_coords: x_coords.append(max_x)
            x_coords = sorted(list(set(x_coords)))

            nominal_ys = []
            cur_y = min_y + offset_y
            while cur_y > min_y: cur_y -= step_y
            while cur_y < max_y + step_y:
                nominal_ys.append(cur_y)
                cur_y += step_y
            y_coords = sorted(list(set([max(min_y, min(max_y, y)) for y in nominal_ys])))
        else:
            batten_centers = self._compute_batten_positions(min_y, max_y)
            
            nominal_ys = []
            cur_y = min_y + offset_y
            while cur_y > min_y: cur_y -= step_y
            while cur_y < max_y + step_y:
                nominal_ys.append(cur_y)
                cur_y += step_y
            
            y_coords = []
            for y in nominal_ys:
                if y <= min_y + 1.0: y_coords.append(min_y)
                elif y >= max_y - 1.0: y_coords.append(max_y)
                else: y_coords.append(min(batten_centers, key=lambda c: abs(c - y)))
            y_coords = sorted(list(set(y_coords)))
            if min_y not in y_coords: y_coords.insert(0, min_y)
            if max_y not in y_coords: y_coords.append(max_y)
            y_coords = sorted(list(set(y_coords)))

            nominal_xs = []
            cur_x = min_x + offset_x
            while cur_x > min_x: cur_x -= step_x
            while cur_x < max_x + step_x:
                nominal_xs.append(cur_x)
                cur_x += step_x
            x_coords = sorted(list(set([max(min_x, min(max_x, x)) for x in nominal_xs])))

        for i in range(len(x_coords) - 1):
            x1, x2 = x_coords[i], x_coords[i+1]
            w_i = x2 - x1
            if w_i < 1.0: continue
            for j in range(len(y_coords) - 1):
                y1, y2 = y_coords[j], y_coords[j+1]
                h_j = y2 - y1
                if h_j < 1.0: continue

                tile_rect = shapely_box(x1, y1, x2, y2)
                intersection = room_poly.intersection(tile_rect)

                if intersection.is_empty or intersection.area < 1.0:
                    continue

                coverage_ratio = intersection.area / (w_i * h_j)
                dims_match = (abs(w_i - eff_length) < 5.0 and abs(h_j - eff_width) < 5.0)

                if coverage_ratio > 0.99 and dims_match:
                    pieces.append(TilePiece(piece_id="", rect=(x1, y1, w_i, h_j), piece_type="full", width_mm=w_i, height_mm=h_j, area_mm2=w_i * h_j, shape_type="Rect"))
                    total_placed_area += w_i * h_j
                else:
                    actual_area = intersection.area
                    shape_type, vertices, edges = self._extract_geometry(intersection, eff_length, eff_width)
                    pieces.append(TilePiece(piece_id="", rect=(x1, y1, w_i, h_j), piece_type="cut", width_mm=w_i, height_mm=h_j, area_mm2=actual_area, shape_type=shape_type, vertices=vertices, edges=edges, intersection=intersection))
                    total_placed_area += actual_area

        pieces = self._merge_adjacent_pieces(pieces, eff_length, eff_width)
        layout.total_placed_area = sum(p.area_mm2 for p in pieces)
        return self._finalize_layout(layout, pieces, room_area, step_y)
