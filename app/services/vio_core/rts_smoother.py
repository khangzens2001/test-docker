"""Manifold SR-RTS Backward Smoother with SO(3) Retraction."""

import copy
import numpy as np
from app.services.vio_core.math_utils import so3_log
from app.services.vio_core.state import NominalState, StateIndex as I


def _copy_state(state: NominalState) -> NominalState:
    """Creates a deep copy of a NominalState object."""
    new_s = NominalState()
    new_s.p = state.p.copy()
    new_s.v = state.v.copy()
    new_s.R = state.R.copy()
    new_s.ba = state.ba.copy()
    new_s.bg = state.bg.copy()
    new_s.bg_hub = state.bg_hub.copy()
    new_s.g = state.g.copy()
    new_s.td = float(state.td)
    new_s.sl = float(state.sl)
    return new_s


class ManifoldRtsSmoother:
    """Rauch-Tung-Striebel (RTS) Backward Smoother on SO(3) Manifold."""

    def __init__(self) -> None:
        # Default process noise covariance matrix for 23D state
        self.default_Q = np.diag([
            *([0.01**2] * 3),     # Position
            *([0.01**2] * 3),     # Velocity
            *([0.001**2] * 3),    # Orientation
            *([1e-4**2] * 3),     # Phone Accel Bias
            *([1e-5**2] * 3),     # Phone Gyro Bias
            *([1e-5**2] * 3),     # Hub Gyro Bias
            *([1e-4**2] * 3),     # Gravity
            1e-6**2,              # Time offset
            1e-5**2,              # LiDAR scale
        ])

    def smooth_trajectory(
        self,
        states: list[NominalState],
        S_P_list: list[np.ndarray],
        F_list: list[np.ndarray],
        Q_list: list[np.ndarray] | np.ndarray | None = None,
        pred_states: list[NominalState] | None = None,
        compute_covariances: bool = True,
        in_place: bool = False,
    ) -> tuple[list[NominalState], list[np.ndarray]]:
        """Performs backward RTS smoothing on a sequence of nominal states and square-root covariances.

        Args:
            states: List of nominal states from forward ESKF run of length N.
            S_P_list: List of Cholesky factors S_P (where P = S_P @ S_P.T) of length N.
            F_list: List of state transition Jacobians F_k from step k to k+1 of length N-1.
            Q_list: Process noise covariance matrix (single 2D array, list of N-1 2D arrays, or None).
            pred_states: Optional list of predicted nominal states at step k+1|k of length N-1.
            compute_covariances: Whether to compute and return smoothed covariance Cholesky factors.
            in_place: Whether to update states in-place to save memory.

        Returns:
            tuple[list[NominalState], list[np.ndarray]]: Smoothed states and smoothed Cholesky factors.
        """
        N = len(states)
        if N <= 1:
            res_states = states if in_place else [_copy_state(s) for s in states]
            res_sp = [S.copy() for S in S_P_list] if compute_covariances else []
            return res_states, res_sp

        if len(S_P_list) != N:
            raise ValueError(f"S_P_list length ({len(S_P_list)}) must match states length ({N})")
        if len(F_list) != N - 1:
            raise ValueError(f"F_list length ({len(F_list)}) must be N-1 ({N - 1})")
        if pred_states is not None and len(pred_states) != N - 1:
            raise ValueError(f"pred_states length ({len(pred_states)}) must be N-1 ({N - 1})")

        smoothed_states = states if in_place else [_copy_state(s) for s in states]
        smoothed_S_P = [S.copy() for S in S_P_list] if compute_covariances else []

        for k in range(N - 2, -1, -1):
            S_P_k = S_P_list[k]
            P_k = S_P_k @ S_P_k.T
            F_k = F_list[k]

            # Get process noise Q_k
            if Q_list is None:
                Q_k = self.default_Q
            elif isinstance(Q_list, list):
                Q_k = Q_list[k]
            elif isinstance(Q_list, np.ndarray) and Q_list.ndim == 2:
                Q_k = Q_list
            else:
                Q_k = Q_list[k]

            # Predicted covariance P_{k+1|k}
            P_next_pred = F_k @ P_k @ F_k.T + Q_k
            P_next_pred = 0.5 * (P_next_pred + P_next_pred.T)

            # Smoother gain C_k = P_k F_k^T P_{k+1|k}^{-1}
            # Solved via P_{k+1|k} C_k^T = F_k P_k
            C_k = np.linalg.solve(P_next_pred, F_k @ P_k).T

            # Difference vector delta_x_{k+1} = x_{k+1}^s - x_{k+1|k}
            s_smooth_next = smoothed_states[k + 1]
            s_pred_next = pred_states[k] if pred_states is not None else states[k + 1]

            dx_next = np.zeros(I.DIM)
            dx_next[I.POS] = s_smooth_next.p - s_pred_next.p
            dx_next[I.VEL] = s_smooth_next.v - s_pred_next.v
            dx_next[I.ORI] = so3_log(s_pred_next.R.T @ s_smooth_next.R)
            dx_next[I.BA] = s_smooth_next.ba - s_pred_next.ba
            dx_next[I.BG] = s_smooth_next.bg - s_pred_next.bg
            dx_next[I.BG_HUB] = s_smooth_next.bg_hub - s_pred_next.bg_hub
            dx_next[I.GRAV] = s_smooth_next.g - s_pred_next.g
            dx_next[I.TD] = np.array([s_smooth_next.td - s_pred_next.td])
            dx_next[I.SL] = np.array([s_smooth_next.sl - s_pred_next.sl])

            # Error state injection delta_x_k = C_k @ delta_x_{k+1}
            dx_k = C_k @ dx_next
            smoothed_states[k].inject(dx_k)

            # Smoothed covariance matrix P_k^s = P_k + C_k (P_{k+1}^s - P_{k+1|k}) C_k^T
            if compute_covariances:
                S_P_next_smooth = smoothed_S_P[k + 1]
                P_next_smooth = S_P_next_smooth @ S_P_next_smooth.T

                P_k_smooth = P_k + C_k @ (P_next_smooth - P_next_pred) @ C_k.T
                P_k_smooth = 0.5 * (P_k_smooth + P_k_smooth.T)

                # Cholesky factor extraction
                try:
                    S_P_k_smooth = np.linalg.cholesky(P_k_smooth)
                except np.linalg.LinAlgError:
                    jitter = 1e-12 * np.eye(I.DIM)
                    S_P_k_smooth = np.linalg.cholesky(P_k_smooth + jitter)

                smoothed_S_P[k] = S_P_k_smooth

        return smoothed_states, smoothed_S_P


def rts_smooth(
    forward_states: list[NominalState],
    forward_covs: list[np.ndarray],
    predict_states: list[NominalState] | None = None,
    predict_covs: list[np.ndarray] | None = None,
    transition_matrices: list[np.ndarray] | None = None,
) -> tuple[list[NominalState], list[np.ndarray]]:
    """Helper wrapper for ManifoldRtsSmoother."""
    smoother = ManifoldRtsSmoother()
    if transition_matrices is None:
        transition_matrices = [np.eye(I.DIM) for _ in range(len(forward_states) - 1)]

    # Extract S_P factors if 2D covariance matrices were provided
    S_P_list = []
    for C in forward_covs:
        if C.shape == (I.DIM, I.DIM):
            try:
                S_P_list.append(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                S_P_list.append(np.linalg.cholesky(C + 1e-12 * np.eye(I.DIM)))
        else:
            S_P_list.append(C)

    smoothed_states, smoothed_S_P = smoother.smooth_trajectory(
        states=forward_states,
        S_P_list=S_P_list,
        F_list=transition_matrices,
        pred_states=predict_states,
    )
    smoothed_covs = [S @ S.T for S in smoothed_S_P]
    return smoothed_states, smoothed_covs

