"""Square-Root ESKF 23D Core Engine with Potter QR Updates & Chi-Square Gating."""

import numpy as np
from scipy.linalg import qr
from app.services.vio_core.state import NominalState, StateIndex as I
from app.services.vio_core.math_utils import skew_symmetric


class SRESKF:
    """Square-Root Error State Kalman Filter (23D) engine."""

    CHI2_THRESHOLD_3D = 7.815

    def __init__(self) -> None:
        self.state = NominalState()
        self.S_P = np.diag([
            *([0.1] * 3),     # POS
            *([0.1] * 3),     # VEL
            *([0.01] * 3),    # ORI
            *([0.1] * 3),     # BA: 0.1 m/s^2 uncertainty
            *([0.005] * 3),   # BG
            *([0.005] * 3),   # BG_HUB
            *([0.001] * 3),   # GRAV: well-known magnitude/direction
            0.001,            # TD
            0.01,             # SL
        ])
        self.Q_sqrt = np.diag([
            *([0.005] * 3),   # POS
            *([0.005] * 3),   # VEL
            *([0.001] * 3),   # ORI
            *([1e-4] * 3),    # BA
            *([1e-5] * 3),    # BG
            *([1e-5] * 3),    # BG_HUB
            *([1e-7] * 3),    # GRAV: nearly zero process noise
            1e-6,             # TD
            1e-5,             # SL
        ])

    def predict_imu(self, preint) -> np.ndarray:
        """Predict nominal state and propagate covariance square root using IMU preintegration."""
        dt = preint.delta_t
        s = self.state
        R_prev = s.R.copy()
        s.p += s.v * dt + 0.5 * s.g * dt**2 + R_prev @ preint.delta_p
        s.v += s.g * dt + R_prev @ preint.delta_v
        s.R = R_prev @ preint.delta_R

        F = np.eye(I.DIM)
        F[I.POS, I.VEL] = np.eye(3) * dt
        F[I.POS, I.ORI] = -R_prev @ skew_symmetric(preint.delta_p)
        F[I.VEL, I.ORI] = -R_prev @ skew_symmetric(preint.delta_v)
        F[I.ORI, I.ORI] = preint.delta_R.T
        F[I.POS, I.GRAV] = 0.5 * np.eye(3) * dt**2
        F[I.VEL, I.GRAV] = np.eye(3) * dt

        # Jacobians with respect to Accel Bias (ba) and Gyro Bias (bg)
        if hasattr(preint, "J_ba") and preint.J_ba is not None:
            F[I.POS, I.BA] = R_prev @ preint.J_ba[0:3, :]
            F[I.VEL, I.BA] = R_prev @ preint.J_ba[3:6, :]
        else:
            F[I.POS, I.BA] = -0.5 * R_prev * (dt**2)
            F[I.VEL, I.BA] = -R_prev * dt

        if hasattr(preint, "J_bg") and preint.J_bg is not None:
            F[I.POS, I.BG] = R_prev @ preint.J_bg[0:3, :]
            F[I.VEL, I.BG] = R_prev @ preint.J_bg[3:6, :]
            F[I.ORI, I.BG] = preint.J_bg[6:9, :]
        else:
            F[I.ORI, I.BG] = -np.eye(3) * dt

        q_scale = 5.0 if getattr(preint, "is_inflated", False) else 1.0
        M = np.hstack([F @ self.S_P, self.Q_sqrt * np.sqrt(dt) * q_scale])
        _, R_qr = qr(M.T)
        self.S_P = R_qr[: I.DIM, : I.DIM].T
        return F

    def update_zupt(self, sigma_vel: float = 0.02) -> bool:
        """Zero-Velocity Update (ZUPT). Constrains linear velocity to 0."""
        H = np.zeros((3, I.DIM))
        H[:, I.VEL] = np.eye(3)
        y = -self.state.v
        R_cov = (sigma_vel ** 2) * np.eye(3)
        return self.update_measurement(y, H, R_cov, chi2_mult=3.0)

    def update_floor_distance(self, p_z_meas: float, sigma_z: float = 0.03) -> bool:
        """Direct 1D metric floor distance constraint to lock vertical position p_z."""
        H = np.zeros((1, I.DIM))
        H[0, I.POS.start + 2] = 1.0
        y = np.array([float(p_z_meas) - float(self.state.p[2])])
        R_cov = np.array([[float(sigma_z) ** 2]])
        return self.update_measurement(y, H, R_cov, chi2_mult=3.0)

    def update_measurement(
        self, innovation: np.ndarray, H: np.ndarray, R_cov: np.ndarray, chi2_mult: float = 1.0
    ) -> bool:
        """Update state using measurement innovation with Potter QR update and Chi-square gating.

        Returns:
            bool: True if measurement was accepted and updated; False if rejected by gating.
        """
        m = len(innovation)
        S_P = self.S_P
        P = S_P @ S_P.T
        S_meas = H @ P @ H.T + R_cov

        d_mahalanobis2 = innovation.T @ np.linalg.solve(S_meas, innovation)
        if d_mahalanobis2 > self.CHI2_THRESHOLD_3D * (m / 3.0) * chi2_mult:
            return False

        K = P @ H.T @ np.linalg.solve(S_meas, np.eye(m))
        dx = K @ innovation
        self.state.inject(dx)

        S_R = np.linalg.cholesky(R_cov)
        M = np.block([[S_R, H @ S_P], [np.zeros((I.DIM, m)), S_P]])
        _, R_qr = qr(M.T)
        self.S_P = R_qr[m:, m:].T
        return True

    @property
    def P(self) -> np.ndarray:
        return self.S_P @ self.S_P.T

    def update_visual(self, innovation: np.ndarray, H: np.ndarray, R_cov: np.ndarray) -> bool:
        return self.update_measurement(innovation, H, R_cov)

    def update_lidar(self, innovation: np.ndarray, H: np.ndarray, R_cov: np.ndarray) -> bool:
        return self.update_measurement(innovation, H, R_cov)

    def update_hub_imu(self, innovation: np.ndarray, H: np.ndarray, R_cov: np.ndarray) -> bool:
        return self.update_measurement(innovation, H, R_cov)

    def get_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.state.p.copy(), self.state.q


ESKF = SRESKF

