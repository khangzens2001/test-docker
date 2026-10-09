"""Manifold SR-RTS Backward Smoother with SO(3) Retraction."""

import copy
import logging
import os
import numpy as np
from app.services.vio_core.math_utils import so3_log
from app.services.vio_core import numba_accelerated as _nba
from app.services.vio_core.state import NominalState, StateIndex as I

logger = logging.getLogger(__name__)

RTS_CHUNK = 2048  # steps per gain-precompute chunk (bounds peak memory to a few 10 MB on either device)


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


def resolve_rts_device() -> str:
    """Pick the device for the RTS gain precompute from VIO_RTS_DEVICE (auto|cuda|cpu)."""
    pref = os.environ.get("VIO_RTS_DEVICE", "auto").strip().lower()
    if pref not in ("auto", "cuda", "cpu"):
        logger.warning("Unrecognised VIO_RTS_DEVICE=%r (expected auto|cuda|cpu); treating as auto", pref)
        pref = "auto"
    if pref == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if pref == "cuda":
            logger.warning("VIO_RTS_DEVICE=cuda requested but torch is not installed; using cpu")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        logger.warning("VIO_RTS_DEVICE=cuda requested but torch.cuda.is_available() is False; using cpu")
    return "cpu"


def rts_gains_numpy(S: np.ndarray, F: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """C_k = (P_{k+1|k}^{-1} F_k P_k)^T for stacked (M,23,23) inputs, same formulas as the loop."""
    P = S @ np.transpose(S, (0, 2, 1))
    FP = F @ P
    P_pred = FP @ np.transpose(F, (0, 2, 1)) + Q
    P_pred = 0.5 * (P_pred + np.transpose(P_pred, (0, 2, 1)))
    return np.transpose(np.linalg.solve(P_pred, FP), (0, 2, 1))


def rts_gains_torch(S: np.ndarray, F: np.ndarray, Q: np.ndarray, device: str) -> np.ndarray:
    """Same as rts_gains_numpy, computed in float64 on the given torch device."""
    import torch

    with torch.no_grad():
        S_t = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float64)).to(device)
        F_t = torch.from_numpy(np.ascontiguousarray(F, dtype=np.float64)).to(device)
        Q_t = torch.from_numpy(np.ascontiguousarray(Q, dtype=np.float64)).to(device)
        P = S_t @ S_t.transpose(1, 2)
        FP = F_t @ P
        P_pred = FP @ F_t.transpose(1, 2) + Q_t
        P_pred = 0.5 * (P_pred + P_pred.transpose(1, 2))
        C = torch.linalg.solve(P_pred, FP).transpose(1, 2).contiguous()
        return C.cpu().numpy()


def _pack_states(states: list[NominalState]) -> tuple[np.ndarray, np.ndarray]:
    N = len(states)
    x = np.empty((N, 20), dtype=np.float64)
    R = np.empty((N, 3, 3), dtype=np.float64)
    for i, s in enumerate(states):
        x[i, 0:3] = s.p
        x[i, 3:6] = s.v
        x[i, 6:9] = s.ba
        x[i, 9:12] = s.bg
        x[i, 12:15] = s.bg_hub
        x[i, 15:18] = s.g
        x[i, 18] = s.td
        x[i, 19] = s.sl
        R[i] = s.R
    return x, R


def _unpack_states(states: list[NominalState], x: np.ndarray, R: np.ndarray) -> None:
    """Write packed arrays back into the NominalState objects (in-place on their arrays, like inject())."""
    for i, s in enumerate(states):
        s.p[...] = x[i, 0:3]
        s.v[...] = x[i, 3:6]
        s.ba[...] = x[i, 6:9]
        s.bg[...] = x[i, 9:12]
        s.bg_hub[...] = x[i, 12:15]
        s.g[...] = x[i, 15:18]
        s.td = float(x[i, 18])
        s.sl = float(x[i, 19])
        s.R[...] = R[i]



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

        if not compute_covariances:
            return self._smooth_states_batched(states, S_P_list, F_list, Q_list, pred_states, in_place), []

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

    def _q_block(self, Q_list, k0: int, k1: int) -> np.ndarray:
        """Process-noise matrices for steps k0..k1-1 as an (M,23,23) array (view when constant)."""
        m = k1 - k0
        if Q_list is None:
            return np.broadcast_to(self.default_Q, (m, I.DIM, I.DIM))
        if isinstance(Q_list, list):
            return np.stack(Q_list[k0:k1])
        if isinstance(Q_list, np.ndarray) and Q_list.ndim == 2:
            return np.broadcast_to(Q_list, (m, I.DIM, I.DIM))
        return np.asarray(Q_list[k0:k1], dtype=np.float64)

    def _smooth_states_batched(
        self,
        states: list[NominalState],
        S_P_list: list[np.ndarray],
        F_list: list[np.ndarray],
        Q_list,
        pred_states: list[NominalState] | None,
        in_place: bool,
    ) -> list[NominalState]:
        """States-only RTS pass: chunked gain precompute (numpy or CUDA) + numba backward recursion."""
        N = len(states)
        out_states = states if in_place else [_copy_state(s) for s in states]
        x_s, R_s = _pack_states(out_states)

        if pred_states is not None:
            x_pred, R_pred = _pack_states(pred_states)
            pred_offset = 0
        elif in_place:
            # The loop reads states[k+1] which, in place, is the already-smoothed state: alias.
            x_pred, R_pred = x_s, R_s
            pred_offset = 1
        else:
            x_pred, R_pred = _pack_states(states)
            pred_offset = 1

        device = resolve_rts_device()
        k1 = N - 1
        while k1 > 0:
            k0 = max(0, k1 - RTS_CHUNK)
            S = np.stack(S_P_list[k0:k1]).astype(np.float64, copy=False)
            F = np.stack(F_list[k0:k1]).astype(np.float64, copy=False)
            Q = self._q_block(Q_list, k0, k1)
            if device == "cuda":
                C = rts_gains_torch(S, F, Q, device)
            else:
                C = rts_gains_numpy(S, F, Q)
            _nba.rts_backward_chunk(np.ascontiguousarray(C), x_s, R_s, x_pred, R_pred, pred_offset, k0, k1)
            k1 = k0

        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

        _unpack_states(out_states, x_s, R_s)
        return out_states


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

