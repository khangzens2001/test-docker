"""Right-Error IMU Pre-integration with PCHIP & Covariance Inflation."""

import numpy as np
from scipy.interpolate import PchipInterpolator
from app.services.vio_core.math_utils import so3_exp, skew_symmetric, right_jacobian_so3


def interpolate_pchip(timestamps: np.ndarray, measurements: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    """PCHIP monotonic cubic spline interpolation for resampled IMU data.

    Args:
        timestamps: 1D array of original timestamps.
        measurements: 2D array of shape (N, D) containing IMU measurements (accel or gyro).
        target_timestamps: 1D array of target timestamps to interpolate.

    Returns:
        2D array of shape (len(target_timestamps), D) with interpolated values.
    """
    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamp samples for PCHIP interpolation.")
    
    interp_fn = PchipInterpolator(timestamps, measurements, axis=0)
    return interp_fn(target_timestamps)


class ImuPreintegration:
    """Pre-integrates IMU measurements between keyframes using Right Error State representation.

    Tracks delta rotation (delta_R), delta velocity (delta_v), delta position (delta_p),
    first-order bias Jacobians (J_ba, J_bg), and 9x9 pre-integration error covariance.
    Inflates covariance when sample gaps exceed 0.1s (100ms-500ms).
    """

    def __init__(self, ba: np.ndarray, bg: np.ndarray, noise_params: dict):
        """Initialize IMU pre-integration module.

        Args:
            ba: 3D accelerometer bias vector.
            bg: 3D gyroscope bias vector.
            noise_params: Dictionary with noise standard deviations ('sigma_a', 'sigma_g', 'sigma_ba', 'sigma_bg').
        """
        self.sigma_a = float(noise_params.get("sigma_a", 0.01))
        self.sigma_g = float(noise_params.get("sigma_g", 0.001))
        self.sigma_ba = float(noise_params.get("sigma_ba", 1e-4))
        self.sigma_bg = float(noise_params.get("sigma_bg", 1e-5))
        self.is_inflated = False
        self.reset(ba, bg)

    def reset(self, ba: np.ndarray, bg: np.ndarray) -> None:
        """Reset pre-integration states and Jacobians for a new frame interval.

        Args:
            ba: 3D accelerometer bias vector.
            bg: 3D gyroscope bias vector.
        """
        self.ba = np.array(ba, dtype=np.float64).copy()
        self.bg = np.array(bg, dtype=np.float64).copy()
        self.delta_R = np.eye(3, dtype=np.float64)
        self.delta_v = np.zeros(3, dtype=np.float64)
        self.delta_p = np.zeros(3, dtype=np.float64)
        self.delta_t = 0.0
        self.covariance = np.zeros((9, 9), dtype=np.float64)
        
        # Jacobians for [pos (0:3), vel (3:6), ori (6:9)]
        self.J_ba = np.zeros((9, 3), dtype=np.float64)
        self.J_bg = np.zeros((9, 3), dtype=np.float64)
        
        self.is_inflated = False
        self._prev_accel = None
        self._prev_gyro = None

    def integrate(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """Integrate a single IMU sample (accel, gyro) over interval dt.

        Args:
            accel: 3D raw accelerometer measurement (m/s^2).
            gyro: 3D raw gyroscope measurement (rad/s).
            dt: Time step duration in seconds.
        """
        if dt <= 0.0:
            return

        # Sample gap inflation for dt > 0.1s (100ms - 500ms)
        inflation_factor = 1.0
        if dt > 0.1:
            self.is_inflated = True
            inflation_factor = (dt / 0.005) ** 2

        a_corr = np.array(accel, dtype=np.float64) - self.ba
        w_corr = np.array(gyro, dtype=np.float64) - self.bg

        if self._prev_accel is not None:
            a_mid = 0.5 * (self._prev_accel + a_corr)
            w_mid = 0.5 * (self._prev_gyro + w_corr)
        else:
            a_mid = a_corr
            w_mid = w_corr

        self._prev_accel = a_corr.copy()
        self._prev_gyro = w_corr.copy()

        phi = w_mid * dt
        Jr = right_jacobian_so3(phi)
        dR_step = so3_exp(phi)
        acc_world = self.delta_R @ a_mid

        # Jacobians update for [pos(0:3), vel(3:6), ori(6:9)]
        J_bg_R_prev = self.J_bg[6:9, :].copy()
        J_ba_v_prev = self.J_ba[3:6, :].copy()
        J_bg_v_prev = self.J_bg[3:6, :].copy()

        # Orientation Jacobian wrt bg
        self.J_bg[6:9, :] = dR_step.T @ J_bg_R_prev - Jr * dt

        # Velocity Jacobians
        self.J_ba[3:6, :] = self.J_ba[3:6, :] - self.delta_R * dt
        self.J_bg[3:6, :] = self.J_bg[3:6, :] - skew_symmetric(acc_world) @ J_bg_R_prev * dt

        # Position Jacobians
        self.J_ba[0:3, :] = self.J_ba[0:3, :] + J_ba_v_prev * dt - 0.5 * self.delta_R * (dt ** 2)
        self.J_bg[0:3, :] = self.J_bg[0:3, :] + J_bg_v_prev * dt - 0.5 * skew_symmetric(acc_world) @ J_bg_R_prev * (dt ** 2)

        # Covariance propagation
        # State order for covariance: [pos(0:3), vel(3:6), ori(6:9)]
        F = np.eye(9, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * skew_symmetric(acc_world) * (dt ** 2)
        F[3:6, 6:9] = -skew_symmetric(acc_world) * dt
        F[6:9, 6:9] = dR_step.T

        G = np.zeros((9, 6), dtype=np.float64)
        G[0:3, 0:3] = 0.5 * self.delta_R * (dt ** 2)
        G[3:6, 0:3] = self.delta_R * dt
        G[6:9, 3:6] = Jr * dt

        Q_d = np.diag([
            *([self.sigma_a ** 2] * 3),
            *([self.sigma_g ** 2] * 3),
        ]) * inflation_factor

        self.covariance = F @ self.covariance @ F.T + G @ Q_d @ G.T

        # Update integrated motion state
        self.delta_p += self.delta_v * dt + 0.5 * acc_world * (dt ** 2)
        self.delta_v += acc_world * dt
        self.delta_R = self.delta_R @ dR_step
        self.delta_t += dt

    def correct_bias(self, dba: np.ndarray, dbg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply first-order bias updates to pre-integrated measurements.

        Args:
            dba: Change in accelerometer bias (3D).
            dbg: Change in gyroscope bias (3D).

        Returns:
            Tuple of (corrected_dR, corrected_dv, corrected_dp).
        """
        dba = np.array(dba, dtype=np.float64)
        dbg = np.array(dbg, dtype=np.float64)

        dR_corr = self.delta_R @ so3_exp(self.J_bg[6:9, :] @ dbg)
        dv_corr = self.delta_v + self.J_ba[3:6, :] @ dba + self.J_bg[3:6, :] @ dbg
        dp_corr = self.delta_p + self.J_ba[0:3, :] @ dba + self.J_bg[0:3, :] @ dbg

        return dR_corr, dv_corr, dp_corr
