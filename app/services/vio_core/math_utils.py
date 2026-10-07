"""SO(3) Math Utilities - Right Error State & Robust Lie Algebra."""

import numpy as np


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Computes the 3x3 skew-symmetric matrix (hat operator) for a 3D vector.

    Args:
        v: 3D vector array [v0, v1, v2].

    Returns:
        3x3 skew-symmetric matrix.
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues formula for exponential map so(3) -> SO(3).

    Args:
        phi: Rotation vector (axis * angle) of shape (3,).

    Returns:
        3x3 rotation matrix R.
    """
    if not np.all(np.isfinite(phi)):
        return np.eye(3)
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3) + skew_symmetric(phi)
    axis = phi / angle
    K = skew_symmetric(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map SO(3) -> so(3) with robust handling near pi singularity.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Rotation vector phi of shape (3,).
    """
    if not np.all(np.isfinite(R)):
        return np.zeros(3)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    if angle < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if abs(angle - np.pi) < 1e-4:
        diag = np.diag(R)
        k = int(np.argmax(diag))
        u = R[:, k].copy()
        u[k] += 1.0
        u = u / np.linalg.norm(u)
        diff = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(u, diff) < 0:
            u = -u
        return u * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Computes the 3x3 Right Jacobian of SO(3).

    Args:
        phi: Rotation vector of shape (3,).

    Returns:
        3x3 Right Jacobian matrix Jr(phi).
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3) - 0.5 * skew_symmetric(phi)
    axis = phi / angle
    K = skew_symmetric(axis)
    return np.eye(3) - ((1.0 - np.cos(angle)) / angle) * K + ((angle - np.sin(angle)) / angle) * (K @ K)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Converts quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.

    Args:
        q: Quaternion vector of shape (4,), format [qx, qy, qz, qw].

    Returns:
        3x3 rotation matrix R.
    """
    q_norm = q / np.linalg.norm(q)
    x, y, z, w = q_norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Converts 3x3 rotation matrix to quaternion [qx, qy, qz, qw] with positive scalar sign (qw >= 0).

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Quaternion vector of shape (4,), format [qx, qy, qz, qw].
    """
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        ])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[2, 1] - R[1, 2]) / s,
        ])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
            (R[0, 2] - R[2, 0]) / s,
        ])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    if q[3] < 0.0:
        q = -q
    return q / np.linalg.norm(q)
