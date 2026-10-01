"""Hub IMU Gyro Cross-Validation Constraint Builder (Task 8).

Fuses Hub IMU gyro readings with Phone IMU gyro to estimate phone and hub gyro biases
and cross-validate rotational dynamics between phone and external hub sensor.
"""

from typing import Tuple, Optional
import numpy as np
from app.services.vio_core.state import NominalState, StateIndex as I
from app.services.vio_core.math_utils import skew_symmetric


def build_hub_imu_constraint(
    state: NominalState,
    omega_hub: np.ndarray,
    omega_phone: np.ndarray,
    R_phone_hub: Optional[np.ndarray] = None,
    sigma_g_phone: float = 0.01,
    sigma_g_hub: float = 0.01,
    estimate_orientation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds Hub IMU gyro cross-validation measurement constraint.

    Residual y = omega_phone - (R_phone_hub @ (omega_hub - bg_hub) + bg)

    Measurement matrix H (3x23):
        - H[:, I.BG]     (slice 12:15): -I_3 (Phone gyro bias bg)
        - H[:, I.BG_HUB] (slice 15:18): +R_phone_hub (Hub gyro bias bg_hub)
        - H[:, I.ORI]    (slice 6:9):   -[ (R_phone_hub @ (omega_hub - bg_hub)) x ] if estimate_orientation else 0_3x3
        - All other slices zero.

    Measurement noise covariance R_cov = (sigma_g_phone^2 + sigma_g_hub^2) * I_3

    Args:
        state: Current ESKF NominalState containing phone bias bg (slice 12:15) and hub bias bg_hub (slice 15:18).
        omega_hub: (3,) array of raw angular velocity from Hub IMU.
        omega_phone: (3,) array of raw angular velocity from Phone IMU.
        R_phone_hub: (3, 3) rotation matrix from Hub frame to Phone frame. If None, defaults to Identity.
        sigma_g_phone: Phone gyro noise std dev in rad/s (default 0.01).
        sigma_g_hub: Hub gyro noise std dev in rad/s (default 0.01).
        estimate_orientation: If True, computes orientation Jacobian H[:, I.ORI]. Default False.

    Returns:
        y: (3,) residual vector.
        H: (3, 23) measurement Jacobian matrix.
        R_cov: (3, 3) measurement noise covariance matrix.
    """
    omega_hub = np.asarray(omega_hub, dtype=np.float64)
    omega_phone = np.asarray(omega_phone, dtype=np.float64)

    if R_phone_hub is None:
        R_phone_hub = np.eye(3, dtype=np.float64)
    else:
        R_phone_hub = np.asarray(R_phone_hub, dtype=np.float64)

    # Correct hub gyro with hub bias state (bg_hub)
    bg_hub = state.bg_hub
    w_h_corrected = omega_hub - bg_hub

    # Rotate corrected hub gyro to phone frame
    w_hub_in_phone = R_phone_hub @ w_h_corrected

    # Expected phone gyro measurement: w_hub_in_phone + phone_bias (bg)
    bg_phone = state.bg
    w_phone_pred = w_hub_in_phone + bg_phone

    # Residual y = omega_phone - w_phone_pred
    y = omega_phone - w_phone_pred

    # Initialize 3x23 Jacobian matrix
    H = np.zeros((3, I.DIM), dtype=np.float64)

    # Slice 12:15 (I.BG): Phone gyro bias derivative -I_3
    H[:, I.BG] = -np.eye(3, dtype=np.float64)

    # Slice 15:18 (I.BG_HUB): Hub gyro bias derivative +R_phone_hub
    H[:, I.BG_HUB] = R_phone_hub

    # Slice 6:9 (I.ORI): Orientation derivative if enabled
    if estimate_orientation:
        H[:, I.ORI] = -skew_symmetric(w_hub_in_phone)

    # Measurement noise covariance
    r_variance = (sigma_g_phone ** 2) + (sigma_g_hub ** 2)
    R_cov = r_variance * np.eye(3, dtype=np.float64)

    return y, H, R_cov


class HubImuFusionBuilder:
    """Builder class for Hub IMU gyro cross-validation measurement constraints."""

    def __init__(
        self,
        R_phone_hub: Optional[np.ndarray] = None,
        sigma_g_phone: float = 0.01,
        sigma_g_hub: float = 0.01,
        estimate_orientation: bool = False,
    ):
        """Initialize HubImuFusionBuilder.

        Args:
            R_phone_hub: Optional (3, 3) rotation matrix from Hub frame to Phone frame.
            sigma_g_phone: Phone gyro noise std dev (default 0.01 rad/s).
            sigma_g_hub: Hub gyro noise std dev (default 0.01 rad/s).
            estimate_orientation: Whether to estimate orientation sensitivity in H[:, I.ORI].
        """
        self.R_phone_hub = R_phone_hub
        self.sigma_g_phone = sigma_g_phone
        self.sigma_g_hub = sigma_g_hub
        self.estimate_orientation = estimate_orientation

    def build_constraint(
        self,
        state: NominalState,
        omega_hub: np.ndarray,
        omega_phone: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build constraint for given state and gyro measurements."""
        return build_hub_imu_constraint(
            state=state,
            omega_hub=omega_hub,
            omega_phone=omega_phone,
            R_phone_hub=self.R_phone_hub,
            sigma_g_phone=self.sigma_g_phone,
            sigma_g_hub=self.sigma_g_hub,
            estimate_orientation=self.estimate_orientation,
        )

    def build_measurement(
        self,
        state: NominalState,
        omega_hub: np.ndarray,
        omega_phone: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Alias for build_constraint."""
        return self.build_constraint(state, omega_hub, omega_phone)


# Backward compatibility alias
HubImuFusion = HubImuFusionBuilder
