"""ANMS-SSC Grid Bucketing feature selection module.

Implements Adaptive Non-Maximal Suppression via Efficient Spatial Suppression (SSC)
over a spatial grid (6x8 default) to select keypoints with high spatial distribution
and uniform coverage across image buckets.
"""

import numpy as np

from app.services.vio_core import numba_accelerated as _nba
from app.services.vio_core.numba_accelerated import JIT_ENABLED


def anms_radii_numpy(sorted_kps: np.ndarray, sorted_resp: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Pairwise-matrix twin of anms_radii_nb for the no-JIT path. Same dtype semantics:
    comparisons in the response dtype, distances in the keypoint dtype."""
    n_pts = sorted_kps.shape[0]
    radii = np.full(n_pts, np.inf, dtype=np.float64)
    if n_pts <= 1:
        return radii
    stronger = sorted_resp[None, :] > thr[:, None]           # stronger[i, j]: resp_j > c * resp_i
    earlier = np.tri(n_pts, n_pts, -1, dtype=bool)           # earlier[i, j]:  j < i
    mask = stronger & earlier
    dists = np.linalg.norm(sorted_kps[:, None, :] - sorted_kps[None, :, :], axis=2)
    dists = np.where(mask, dists, np.inf)
    row_min = dists.min(axis=1)
    has_any = mask.any(axis=1)
    radii[has_any] = row_min[has_any]
    return radii


def _anms_radii(sorted_kps: np.ndarray, sorted_resp: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Dispatch to the numba kernel when it compiled, else to the pairwise-matrix numpy twin
    (which is faster than the kernel's pure-Python body). KERNEL_STATUS is filled by warmup;
    before warmup the kernel is assumed good and compiles lazily on first call."""
    if JIT_ENABLED and _nba.KERNEL_STATUS.get("anms_radii", True):
        return _nba.anms_radii_nb(sorted_kps, sorted_resp, thr)
    return anms_radii_numpy(sorted_kps, sorted_resp, thr)



def select_anms_ssc(
    keypoints: np.ndarray,
    responses: np.ndarray,
    image_size: tuple[int, int],
    max_features: int = 300,
    min_features: int = 80,
    num_grid_rows: int = 6,
    num_grid_cols: int = 8,
    c_robust: float = 0.9,
) -> np.ndarray:
    """Select keypoints using ANMS-SSC with 6x8 spatial grid bucketing.

    Args:
        keypoints: Array of shape (N, 2) containing [x, y] coordinates.
        responses: Array of shape (N,) containing corner response scores.
        image_size: Tuple (width, height) of image dimensions.
        max_features: Maximum number of features to select.
        min_features: Minimum features threshold.
        num_grid_rows: Number of grid rows (default 6).
        num_grid_cols: Number of grid columns (default 8).
        c_robust: Robustness parameter for ANMS response comparison (default 0.9).

    Returns:
        1D integer numpy array of selected keypoint indices in original array.
    """
    n_pts = len(keypoints)
    if n_pts == 0:
        return np.empty((0,), dtype=int)
    
    if n_pts <= min_features:
        return np.arange(n_pts, dtype=int)

    # Sort keypoints by response strength descending
    order = np.argsort(-responses)
    if n_pts > 1000:
        order = order[:1000]
        n_pts = 1000

    sorted_kps = keypoints[order]
    sorted_resp = responses[order]

    # Compute ANMS suppression radius for each keypoint.
    # Thresholds c * resp_i are evaluated as numpy scalars and stored in the response dtype, which is
    # the dtype NumPy compared in before; the radii kernel keeps distances in the keypoint dtype.
    thr = np.empty(n_pts, dtype=sorted_resp.dtype)
    for i in range(n_pts):
        thr[i] = c_robust * sorted_resp[i]
    radii = np.asarray(_anms_radii(sorted_kps, sorted_resp, thr), dtype=np.float64)
    radii[0] = np.inf

    # Spatial grid bucketing (6x8). Cell indices use float64 floor-division exactly like the
    # scalar expression `sorted_kps[i, 0] // w_cell` did; insertion order (ascending i) is kept.
    width, height = image_size
    w_cell = width / num_grid_cols if num_grid_cols > 0 else width
    h_cell = height / num_grid_rows if num_grid_rows > 0 else height

    cols = np.clip(np.floor_divide(sorted_kps[:, 0].astype(np.float64), w_cell), 0, num_grid_cols - 1).astype(int)
    rows = np.clip(np.floor_divide(sorted_kps[:, 1].astype(np.float64), h_cell), 0, num_grid_rows - 1).astype(int)

    grid_buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n_pts):
        grid_buckets.setdefault((int(rows[i]), int(cols[i])), []).append(i)

    # Sort indices within each cell by suppression radius descending
    for key in grid_buckets:
        grid_buckets[key].sort(key=lambda idx: radii[idx], reverse=True)

    # First pass: pick top features per grid cell evenly
    target_per_cell = max(1, max_features // (num_grid_rows * num_grid_cols))
    selected_sorted_indices: list[int] = []
    selected_set: set[int] = set()

    for cell, idx_list in grid_buckets.items():
        take = idx_list[:target_per_cell]
        for idx in take:
            selected_sorted_indices.append(idx)
            selected_set.add(idx)

    # Second pass: fill remaining quota up to max_features based on global radius
    if len(selected_sorted_indices) < max_features:
        remaining_indices = [i for i in range(n_pts) if i not in selected_set]
        remaining_indices.sort(key=lambda idx: radii[idx], reverse=True)
        
        quota = max_features - len(selected_sorted_indices)
        for idx in remaining_indices[:quota]:
            selected_sorted_indices.append(idx)
            selected_set.add(idx)

    # Map back to original keypoint indices
    original_indices = order[np.array(selected_sorted_indices, dtype=int)]
    return original_indices
