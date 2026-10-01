"""DLT / Inverse Depth Triangulation (rho = 1/Z) for Multi-View VIO."""

from typing import NamedTuple, Union
import numpy as np


class TriangulationResult(NamedTuple):
    """Container for DLT triangulation result."""
    point_3d: np.ndarray     # 3D point in world frame (3,)
    rho: float              # Inverse depth rho = 1/Z in host frame
    valid: bool             # Validity flag (parallax >= 1.0 deg and rho > 0)
    parallax_deg: float     # Ray parallax angle in degrees
    state: np.ndarray       # Bearing angles + inverse depth state [alpha, beta, rho]^T


def compute_parallax_angle(P_w: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> float:
    """Computes the ray parallax angle theta in degrees between two optical centers.

    Args:
        P_w: 3D point position in world frame (3,).
        t1: Optical center position of camera 1 in world frame (3,).
        t2: Optical center position of camera 2 in world frame (3,).

    Returns:
        Parallax angle in degrees.
    """
    r1 = P_w - t1
    r2 = P_w - t2

    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)

    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0

    r1_hat = r1 / n1
    r2_hat = r2 / n2

    cos_theta = np.clip(np.dot(r1_hat, r2_hat), -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return float(np.degrees(theta_rad))


def point_3d_to_inverse_depth(
    P_w: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray
) -> np.ndarray:
    """Converts a 3D world point to host camera inverse depth state [alpha, beta, rho]^T.

    Args:
        P_w: 3D point in world frame (3,).
        R_wc: Camera rotation matrix (camera to world) (3, 3).
        t_wc: Camera position in world frame (3,).

    Returns:
        State vector [alpha, beta, rho]^T.
    """
    p_cam = R_wc.T @ (P_w - t_wc)
    x, y, z = p_cam[0], p_cam[1], p_cam[2]

    if z <= 1e-8:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    rho = 1.0 / z
    alpha = x / z
    beta = y / z
    return np.array([alpha, beta, rho], dtype=np.float64)


def inverse_depth_to_3d(
    alpha: float, beta: float, rho: float, R_wc: np.ndarray, t_wc: np.ndarray
) -> np.ndarray:
    """Converts inverse depth state [alpha, beta, rho]^T back to a 3D world point.

    Args:
        alpha: Bearing coordinate x/z.
        beta: Bearing coordinate y/z.
        rho: Inverse depth 1/z.
        R_wc: Camera rotation matrix (camera to world) (3, 3).
        t_wc: Camera position in world frame (3,).

    Returns:
        3D point in world frame (3,).
    """
    if abs(rho) <= 1e-8:
        z = 1e8
    else:
        z = 1.0 / rho

    p_cam = np.array([alpha * z, beta * z, z], dtype=np.float64)
    return R_wc @ p_cam + t_wc


def triangulate_dlt_single(
    pts1: np.ndarray,
    pts2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    K: np.ndarray,
    min_parallax_deg: float = 1.0,
) -> TriangulationResult:
    """Performs DLT triangulation (SVD on 4x4 matrix) for a single 2D keypoint pair.

    Args:
        pts1: 2D keypoint in image 1 (u, v) (2,).
        pts2: 2D keypoint in image 2 (u, v) (2,).
        R1: Camera 1 rotation matrix R_wc1 (3, 3).
        t1: Camera 1 translation t_wc1 (3,).
        R2: Camera 2 rotation matrix R_wc2 (3, 3).
        t2: Camera 2 translation t_wc2 (3,).
        K: Camera intrinsic matrix (3, 3).
        min_parallax_deg: Minimum required parallax angle in degrees (default 1.0).

    Returns:
        TriangulationResult container.
    """
    K_inv = np.linalg.inv(K)

    p1_norm = K_inv @ np.array([pts1[0], pts1[1], 1.0])
    p2_norm = K_inv @ np.array([pts2[0], pts2[1], 1.0])

    x1, y1 = p1_norm[0], p1_norm[1]
    x2, y2 = p2_norm[0], p2_norm[1]

    # Camera projection matrices (world to camera projection: P = [R^T | -R^T * t])
    P1 = R1.T @ np.hstack([np.eye(3), -t1.reshape(3, 1)])
    P2 = R2.T @ np.hstack([np.eye(3), -t2.reshape(3, 1)])

    # Construct DLT 4x4 linear system A * X = 0
    A = np.zeros((4, 4), dtype=np.float64)
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]

    if abs(X_h[3]) < 1e-12:
        return TriangulationResult(
            point_3d=np.zeros(3),
            rho=0.0,
            valid=False,
            parallax_deg=0.0,
            state=np.zeros(3),
        )

    P_w = X_h[:3] / X_h[3]

    # Transform 3D point to camera frames
    p_c1 = R1.T @ (P_w - t1)
    p_c2 = R2.T @ (P_w - t2)

    z1 = p_c1[2]
    z2 = p_c2[2]

    # Inverse depth & state vector
    state = point_3d_to_inverse_depth(P_w, R1, t1)
    rho = float(state[2])

    # Parallax angle check
    parallax_deg = compute_parallax_angle(P_w, t1, t2)

    # Validity flag check: positive depths in both cameras, positive inverse depth, and sufficient parallax
    valid = bool((z1 > 0.01) and (z2 > 0.01) and (rho > 0.0) and (parallax_deg >= min_parallax_deg))

    return TriangulationResult(
        point_3d=P_w,
        rho=rho,
        valid=valid,
        parallax_deg=parallax_deg,
        state=state,
    )


def triangulate_dlt(
    pts1: np.ndarray,
    pts2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    K: np.ndarray,
    min_parallax_deg: float = 1.0,
) -> Union[TriangulationResult, list[TriangulationResult]]:
    """DLT triangulation for keypoints (supports single point pair or batch of point pairs).

    Args:
        pts1: Keypoints in image 1 (2,) or (N, 2).
        pts2: Keypoints in image 2 (2,) or (N, 2).
        R1: Camera 1 rotation matrix R_wc1 (3, 3).
        t1: Camera 1 position t_wc1 (3,).
        R2: Camera 2 rotation matrix R_wc2 (3, 3).
        t2: Camera 2 position t_wc2 (3,).
        K: Camera intrinsic matrix (3, 3).
        min_parallax_deg: Minimum parallax angle threshold (default 1.0 deg).

    Returns:
        TriangulationResult for single point input, or list[TriangulationResult] for batch input.
    """
    pts1_arr = np.asarray(pts1, dtype=np.float64)
    pts2_arr = np.asarray(pts2, dtype=np.float64)

    if pts1_arr.ndim == 2:
        results = []
        for p1, p2 in zip(pts1_arr, pts2_arr):
            results.append(
                triangulate_dlt_single(p1, p2, R1, t1, R2, t2, K, min_parallax_deg)
            )
        return results
    else:
        return triangulate_dlt_single(pts1_arr, pts2_arr, R1, t1, R2, t2, K, min_parallax_deg)


def triangulate_point(
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    K: np.ndarray,
    min_parallax_deg: float = 1.0,
) -> tuple[np.ndarray, float] | None:
    """Multi-view DLT triangulation for a list of observations (R_wc, t_wc, uv).

    Args:
        observations: List of (R_wc, t_wc, uv) tuples for each camera view.
        K: Camera intrinsic matrix (3, 3).
        min_parallax_deg: Minimum required parallax angle (default 1.0 deg).

    Returns:
        (P_w, mean_reprojection_error) if valid, or None if invalid.
    """
    if len(observations) < 2:
        return None

    K_inv = np.linalg.inv(K)
    A_rows = []

    for R_wc, t_wc, uv in observations:
        P = R_wc.T @ np.hstack([np.eye(3), -t_wc.reshape(3, 1)])
        p_norm = K_inv @ np.array([uv[0], uv[1], 1.0])
        x, y = p_norm[0], p_norm[1]
        A_rows.append(x * P[2] - P[0])
        A_rows.append(y * P[2] - P[1])

    A = np.array(A_rows, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]

    if abs(X_h[3]) < 1e-12:
        return None

    P_w = X_h[:3] / X_h[3]

    total_err = 0.0
    for R_wc, t_wc, uv in observations:
        p_cam = R_wc.T @ (P_w - t_wc)
        if p_cam[2] < 0.01:
            return None
        if np.linalg.norm(P_w - t_wc) > 100.0:
            return None
        proj = K @ p_cam
        proj_uv = proj[:2] / proj[2]
        total_err += float(np.linalg.norm(proj_uv - uv))

    mean_err = total_err / len(observations)
    if mean_err > 5.0:
        return None

    # Parallax angle check between first and last camera position
    t_first = observations[0][1]
    t_last = observations[-1][1]
    parallax = compute_parallax_angle(P_w, t_first, t_last)
    if parallax < min_parallax_deg:
        return None

    return P_w, mean_err


def triangulate_tracks(
    tracks: list[list[tuple]],
    poses: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Batch triangulates feature tracks across camera poses.

    Args:
        tracks: List of feature tracks, where each track is a list of (frame_id, u, v).
        poses: List of camera poses (R_wc, t_wc) indexed by frame_id.
        K: Camera intrinsic matrix (3, 3).

    Returns:
        List of (track_id, 3D point in world frame).
    """
    results = []
    for tid, track in enumerate(tracks):
        if len(track) < 2:
            continue
        obs = []
        for frame_id, u, v in track:
            if frame_id < len(poses):
                R, t = poses[frame_id]
                obs.append((R, t, np.array([u, v], dtype=np.float64)))
        if len(obs) >= 2:
            result = triangulate_point(obs, K)
            if result is not None:
                results.append((tid, result[0]))
    return results
