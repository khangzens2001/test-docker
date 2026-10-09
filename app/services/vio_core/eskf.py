"""Square-Root ESKF 23D Core Engine with Potter QR Updates & Chi-Square Gating."""

import numpy as np
from scipy.linalg import qr
from app.services.vio_core import numba_accelerated as _nba
from app.services.vio_core.state import NominalState, StateIndex as I


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
        dt = float(preint.delta_t)
        s = self.state

        # Jacobians with respect to Accel Bias (ba) and Gyro Bias (bg); legacy fallback when the
        # preintegration object carries none (first-order constant-rate model).
        J_ba = getattr(preint, "J_ba", None)
        J_bg = getattr(preint, "J_bg", None)
        if J_ba is None:
            J_ba = np.zeros((9, 3), dtype=np.float64)
            J_ba[0:3, :] = -0.5 * np.eye(3) * (dt**2)
            J_ba[3:6, :] = -np.eye(3) * dt
        if J_bg is None:
            J_bg = np.zeros((9, 3), dtype=np.float64)
            J_bg[6:9, :] = -np.eye(3) * dt

        q_scale = 5.0 if getattr(preint, "is_inflated", False) else 1.0

        p_new, v_new, R_new, F, S_P_new = _nba.eskf_predict(
            np.ascontiguousarray(s.p, dtype=np.float64),
            np.ascontiguousarray(s.v, dtype=np.float64),
            np.ascontiguousarray(s.R, dtype=np.float64),
            np.ascontiguousarray(s.g, dtype=np.float64),
            np.ascontiguousarray(self.S_P, dtype=np.float64),
            np.ascontiguousarray(preint.delta_R, dtype=np.float64),
            np.ascontiguousarray(preint.delta_v, dtype=np.float64),
            np.ascontiguousarray(preint.delta_p, dtype=np.float64),
            np.ascontiguousarray(J_ba, dtype=np.float64),
            np.ascontiguousarray(J_bg, dtype=np.float64),
            dt,
            np.ascontiguousarray(self.Q_sqrt, dtype=np.float64),
            float(q_scale),
        )
        s.p[:] = p_new
        s.v[:] = v_new
        s.R = R_new
        self.S_P = S_P_new
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

        When m > I.DIM (23), applies whitening and measurement compression via thin QR to reduce
        the measurement to 23D, turning O(m^3) matrix inversions and large QR updates into fast
        O(m * n^2) operations with mathematical equivalence.

        Returns:
            bool: True if measurement was accepted and updated; False if rejected by gating.
        """
        m = len(innovation)
        S_P = self.S_P
        P = S_P @ S_P.T

        if m > I.DIM:
            try:
                # Fast vectorized 2x2 block Cholesky whitening if block-diagonal structure (visual updates)
                if m % 2 == 0 and R_cov.shape == (m, m):
                    nb = m // 2
                    bi = np.arange(nb)
                    Rb = np.empty((nb, 2, 2), dtype=R_cov.dtype)
                    Rb[:, 0, 0] = R_cov[2 * bi, 2 * bi]
                    Rb[:, 0, 1] = R_cov[2 * bi, 2 * bi + 1]
                    Rb[:, 1, 0] = R_cov[2 * bi + 1, 2 * bi]
                    Rb[:, 1, 1] = R_cov[2 * bi + 1, 2 * bi + 1]
                    L00 = np.sqrt(np.maximum(1e-12, Rb[:, 0, 0]))
                    L10 = Rb[:, 1, 0] / L00
                    L11 = np.sqrt(np.maximum(1e-12, Rb[:, 1, 1] - L10**2))

                    yb = innovation.reshape(-1, 2)
                    yw0 = yb[:, 0] / L00
                    yw1 = (yb[:, 1] - L10 * yw0) / L11
                    y_white = np.empty(m, dtype=np.float64)
                    y_white[0::2] = yw0
                    y_white[1::2] = yw1

                    Hb = H.reshape(-1, 2, I.DIM)
                    Hw0 = Hb[:, 0, :] / L00[:, None]
                    Hw1 = (Hb[:, 1, :] - L10[:, None] * Hw0) / L11[:, None]
                    H_white = np.empty((m, I.DIM), dtype=np.float64)
                    H_white[0::2] = Hw0
                    H_white[1::2] = Hw1
                else:
                    S_R = np.linalg.cholesky(R_cov)
                    H_white = np.linalg.solve(S_R, H)
                    y_white = np.linalg.solve(S_R, innovation)

                Q1, T_H = np.linalg.qr(H_white, mode="reduced")
                k = T_H.shape[0]  # min(m, I.DIM) == 23
                z1 = Q1.T @ y_white
                z2_sq = float(np.dot(y_white, y_white) - np.dot(z1, z1))

                S_meas_comp = T_H @ P @ T_H.T + np.eye(k)
                inv_S_z1 = np.linalg.solve(S_meas_comp, z1)
                d_mahalanobis2 = float(z1.T @ inv_S_z1) + max(0.0, z2_sq)

                if d_mahalanobis2 > self.CHI2_THRESHOLD_3D * (m / 3.0) * chi2_mult:
                    return False

                dx = P @ T_H.T @ inv_S_z1
                self.state.inject(dx)

                M = np.block([[np.eye(k), T_H @ S_P], [np.zeros((I.DIM, k)), S_P]])
                _, R_qr = qr(M.T)
                self.S_P = R_qr[k:, k:].T
                return True
            except (np.linalg.LinAlgError, ValueError):
                return False

        # Low-dimensional measurement (m <= 23, e.g. ZUPT m=3, floor distance m=1, Hub IMU m=3)
        S_meas = H @ P @ H.T + R_cov
        try:
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
        except (np.linalg.LinAlgError, ValueError):
            return False

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

