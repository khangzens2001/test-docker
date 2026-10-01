"""SPAD LiDAR Depth Constraint Builder for 23D SR-ESKF (Task 7B)."""

from typing import Tuple, Optional
import numpy as np
from app.services.vio_core.state import NominalState, StateIndex as I


def compute_adaptive_ray_noise(depth: float, alpha_deg: float = 1.0) -> float:
    """Compute adaptive ray noise model R_lidar(alpha, d) = sigma_range^2 + (d * tan(alpha))^2.

    Args:
        depth (float): SPAD depth measurement in meters.
        alpha_deg (float): Field of view per zone in degrees (default: 1.0).

    Returns:
        float: Calculated ray noise variance in m^2.
    """
    d = max(0.0, float(depth))
    alpha_rad = np.deg2rad(alpha_deg)
    sigma_range = 0.02 + 0.03 * d
    r_noise = (sigma_range ** 2) + ((d * np.tan(alpha_rad)) ** 2)
    return float(r_noise)


def build_spad_lidar_constraint(
    p_spad: np.ndarray,
    d_spad: float,
    s_L: float = 1.0,
    alpha_deg: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build single SPAD LiDAR depth measurement constraint.

    Args:
        p_spad (np.ndarray): 3D predicted position vector in body/sensor frame (3,).
        d_spad (float): Measured SPAD depth in meters.
        s_L (float): LiDAR scale factor (default: 1.0).
        alpha_deg (float): FOV per zone in degrees (default: 1.0).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - y: Residual (1,) = d_spad - d_pred
            - H: Jacobian (1, 23) w.r.t 23D state error vector
            - R_cov: Covariance matrix (1, 1)
    """
    p = np.asarray(p_spad, dtype=np.float64)
    norm_p = float(np.linalg.norm(p))
    d_pred = float(s_L * norm_p)

    y = np.array([float(d_spad) - d_pred], dtype=np.float64)

    H = np.zeros((1, I.DIM), dtype=np.float64)
    if norm_p > 1e-9:
        H[0, I.POS] = s_L * (p / norm_p)
    else:
        H[0, I.POS] = np.zeros(3, dtype=np.float64)

    # Orientation derivative for distance ||p|| is zero vector (3,)
    H[0, I.ORI] = np.zeros(3, dtype=np.float64)

    # Scale factor derivative w.r.t s_L: norm_p
    H[0, I.SL] = norm_p

    r_var = compute_adaptive_ray_noise(d_spad, alpha_deg)
    R_cov = np.array([[r_var]], dtype=np.float64)

    return y, H, R_cov


class LidarConstraintBuilder:
    """Builder for incorporating 8x8 SPAD LiDAR depth measurements into the 23D SR-ESKF."""

    def __init__(
        self,
        T_lidar_camera: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
        alpha_deg: float = 1.0,
    ) -> None:
        """Initialize LidarConstraintBuilder.

        Args:
            T_lidar_camera (Optional[np.ndarray]): 4x4 Extrinsic transform from SPAD LiDAR to Camera.
            K (Optional[np.ndarray]): 3x3 Camera intrinsic matrix.
            alpha_deg (float): FOV per zone in degrees (default: 1.0).
        """
        if T_lidar_camera is None:
            self.T_lidar_camera = np.eye(4, dtype=np.float64)
        else:
            self.T_lidar_camera = np.asarray(T_lidar_camera, dtype=np.float64)

        self.R_lc = self.T_lidar_camera[:3, :3]
        self.t_lc = self.T_lidar_camera[:3, 3]

        if K is None:
            self.K = np.eye(3, dtype=np.float64)
        else:
            self.K = np.asarray(K, dtype=np.float64)

        self.alpha_deg = float(alpha_deg)

    def zone_to_ray(self, zone_idx: int) -> np.ndarray:
        """Convert 8x8 SPAD grid zone index (0..63) to a unit ray direction vector in LiDAR frame.

        Args:
            zone_idx (int): Zone index from 0 to 63.

        Returns:
            np.ndarray: 3D unit direction ray vector (3,) pointing in +z direction.
        """
        row = zone_idx // 8
        col = zone_idx % 8
        fov_rad = np.deg2rad(self.alpha_deg)

        theta_x = (col - 3.5) * fov_rad
        theta_y = (row - 3.5) * fov_rad

        ray = np.array([np.tan(theta_x), np.tan(theta_y), 1.0], dtype=np.float64)
        norm_ray = np.linalg.norm(ray)
        if norm_ray > 1e-9:
            ray = ray / norm_ray
        return ray

    def build_measurement(
        self,
        state: NominalState,
        lidar_zones: np.ndarray,
        lidar_status: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build batch measurement innovation, Jacobian, and covariance for valid SPAD zones.

        Args:
            state (NominalState): Current 23D ESKF nominal state.
            lidar_zones (np.ndarray): Array of 64 SPAD depth readings.
            lidar_status (Optional[np.ndarray]): Array of 64 status codes (5 = valid).

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
                - innovations: (N,) depth residuals
                - H: (N, 23) stacked measurement Jacobians
                - R_cov: (N, N) measurement covariance matrix
                Or None if no valid zones are present.
        """
        zones = np.asarray(lidar_zones, dtype=np.float64).flatten()
        valid_mask = (zones >= 0.1) & (zones <= 8.0)

        if lidar_status is not None:
            status = np.asarray(lidar_status, dtype=int).flatten()
            valid_mask &= (status == 5)

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None

        n_valid = len(valid_indices)
        innovations = np.zeros(n_valid, dtype=np.float64)
        H = np.zeros((n_valid, I.DIM), dtype=np.float64)
        R_cov = np.zeros((n_valid, n_valid), dtype=np.float64)

        for i, idx in enumerate(valid_indices):
            z_meas = float(zones[idx])
            ray_l = self.zone_to_ray(idx)
            p_l = ray_l * z_meas
            p_c = self.R_lc @ p_l + self.t_lc

            y_single, H_single, R_single = build_spad_lidar_constraint(
                p_spad=p_c,
                d_spad=z_meas,
                s_L=state.sl,
                alpha_deg=self.alpha_deg,
            )

            innovations[i] = y_single[0]
            H[i, :] = H_single[0, :]
            R_cov[i, i] = R_single[0, 0]

        return innovations, H, R_cov
