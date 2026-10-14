from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imagehash
from PIL import Image

from .config import Config


def _exposure_normalized_hash(path: Path) -> imagehash.ImageHash:
    """phash on a histogram-equalized grayscale version of the image.

    Consecutive frames of the *same* physical subject can vary widely in
    auto-exposure brightness (observed on real scan data: mean intensity
    swinging from ~75 to ~165 across a handful of frames of one object).
    Raw phash is somewhat sensitive to that — equalizing brightness/contrast
    first makes the hash reflect structure rather than exposure, so true
    near-duplicates shot under different exposure are still recognized.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {path}")
    equalized = cv2.equalizeHist(gray)
    return imagehash.phash(Image.fromarray(equalized))


def dedup_window_winners(
    winners: List[Tuple[int, Path, float]], cfg: Config
) -> Dict[int, str]:
    """winners: list of (window_index, image_path, sharpness_score) in
    temporal (window) order. Returns {window_index: dropped_reason} for
    winners that should be dropped as near-duplicates.
    """
    hashes = [_exposure_normalized_hash(path) for _, path, _ in winners]

    dropped: Dict[int, str] = {}
    kept_positions: List[int] = []  # positions (into `winners`) currently kept

    for i, (wid, _path, score) in enumerate(winners):
        candidates = (
            kept_positions[-cfg.dedup_lookback :]
            if cfg.dedup_scope == "local"
            else kept_positions
        )
        dup_pos = None
        for j in candidates:
            if hashes[i] - hashes[j] <= cfg.dedup_hash_threshold:
                dup_pos = j
                break

        if dup_pos is None:
            kept_positions.append(i)
            continue

        dup_wid = winners[dup_pos][0]
        dup_score = winners[dup_pos][2]
        if score > dup_score:
            dropped[dup_wid] = f"near_duplicate_of={wid}"
            kept_positions.remove(dup_pos)
            kept_positions.append(i)
        else:
            dropped[wid] = f"near_duplicate_of={dup_wid}"

    return dropped
