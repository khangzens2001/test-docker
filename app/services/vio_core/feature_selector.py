"""ANMS-SSC Grid Bucketing feature selection module.

Implements Adaptive Non-Maximal Suppression via Efficient Spatial Suppression (SSC)
over a spatial grid (6x8 default) to select keypoints with high spatial distribution
and uniform coverage across image buckets.
"""

import numpy as np


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

    # Compute ANMS suppression radius for each keypoint
    radii = np.full(n_pts, np.inf, dtype=np.float64)
    for i in range(1, n_pts):
        mask = sorted_resp[:i] > (c_robust * sorted_resp[i])
        if np.any(mask):
            candidates = sorted_kps[:i][mask]
            dists = np.linalg.norm(candidates - sorted_kps[i], axis=1)
            radii[i] = np.min(dists)

    # Spatial grid bucketing (6x8)
    width, height = image_size
    w_cell = width / num_grid_cols if num_grid_cols > 0 else width
    h_cell = height / num_grid_rows if num_grid_rows > 0 else height

    grid_buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n_pts):
        c = int(np.clip(sorted_kps[i, 0] // w_cell, 0, num_grid_cols - 1))
        r = int(np.clip(sorted_kps[i, 1] // h_cell, 0, num_grid_rows - 1))
        grid_buckets.setdefault((r, c), []).append(i)

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
