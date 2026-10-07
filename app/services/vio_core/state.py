"""23-Dimensional State Vector Definition for SR-ESKF."""

import numpy as np
from app.services.vio_core.math_utils import rot_to_quat, so3_exp


class StateIndex:
    POS = slice(0, 3)  # Position (3D)
    VEL = slice(3, 6)  # Velocity (3D)
    ORI = slice(6, 9)  # Orientation Right Error (3D)
    BA = slice(9, 12)  # Phone Accel Bias (3D)
    BG = slice(12, 15)  # Phone Gyro Bias (3D)
    BG_HUB = slice(15, 18)  # Hub Gyro Bias (3D)
    GRAV = slice(18, 21)  # Gravity (3D)
    TD = slice(21, 22)  # Camera-IMU Time Offset (1D)
    SL = slice(22, 23)  # LiDAR Scale (1D)
    DIM = 23


class NominalState:
    __slots__ = ("p", "v", "R", "ba", "bg", "bg_hub", "g", "td", "sl")

    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.bg_hub = np.zeros(3)
        self.g = np.array([0.0, 0.0, -9.80665])
        self.td = 0.0
        self.sl = 1.0

    @property
    def q(self) -> np.ndarray:
        return rot_to_quat(self.R)

    def inject(self, dx: np.ndarray) -> None:
        I = StateIndex
        self.p += dx[I.POS]
        self.v += dx[I.VEL]
        self.R = self.R @ so3_exp(dx[I.ORI])  # Right Error State Injection
        self.ba += dx[I.BA]
        self.bg += dx[I.BG]
        self.bg_hub += dx[I.BG_HUB]
        self.g += dx[I.GRAV]
        self.td += float(dx[I.TD][0])
        self.sl += float(dx[I.SL][0])

    def copy(self) -> 'NominalState':
        new_s = NominalState()
        new_s.p = self.p.copy()
        new_s.v = self.v.copy()
        new_s.R = self.R.copy()
        new_s.ba = self.ba.copy()
        new_s.bg = self.bg.copy()
        new_s.bg_hub = self.bg_hub.copy()
        new_s.g = self.g.copy()
        new_s.td = float(self.td)
        new_s.sl = float(self.sl)
        return new_s


