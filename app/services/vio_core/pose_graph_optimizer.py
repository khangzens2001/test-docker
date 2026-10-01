import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from app.services.vio_core.math_utils import right_jacobian_so3, skew_symmetric, so3_exp, so3_log

logger = logging.getLogger(__name__)


class PoseGraphOptimizer:
    """Pose Graph Optimizer on SE(3) manifold with Gauge Anchoring on Pose 0."""

    def __init__(self) -> None:
        # Dict storing keyframe nodes: kf_id -> (R (3x3), t (3,))
        self.nodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Initial prior pose for node 0 anchor
        self.prior_pose_0: tuple[np.ndarray, np.ndarray] | None = None
        # List of edges: tuple (i, j, R_meas, t_meas, information_matrix)
        self.edges: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []

    def add_node(self, kf_id: int, R: np.ndarray, t: np.ndarray) -> None:
        """Adds or updates a keyframe node pose in the graph.

        Args:
            kf_id: Keyframe ID.
            R: 3x3 rotation matrix.
            t: 3D position vector.
        """
        R_copy = R.copy()
        t_copy = t.copy().reshape(3)
        self.nodes[kf_id] = (R_copy, t_copy)

        if kf_id == 0 and self.prior_pose_0 is None:
            self.prior_pose_0 = (R_copy.copy(), t_copy.copy())

    def add_keyframe_pose(self, kf_id: int, R: np.ndarray, t: np.ndarray) -> None:
        """Alias for add_node."""
        self.add_node(kf_id, R, t)

    def add_odometry_edge(
        self,
        i: int,
        j: int,
        R_ij: np.ndarray,
        t_ij: np.ndarray,
        information: np.ndarray | None = None,
    ) -> None:
        """Adds a sequential odometry edge between keyframe i and keyframe j.

        Args:
            i: Start keyframe ID.
            j: End keyframe ID.
            R_ij: Measured 3x3 relative rotation from i to j.
            t_ij: Measured 3D relative translation from i to j.
            information: 6x6 information matrix. Defaults to Identity(6).
        """
        if information is None:
            information = 10.0 * np.eye(6, dtype=np.float64)
        self.edges.append((i, j, R_ij.copy(), t_ij.copy().reshape(3), information.copy()))

    def add_loop_edge(
        self,
        i: int,
        j: int,
        R_ij: np.ndarray,
        t_ij: np.ndarray,
        information: np.ndarray | None = None,
        max_trans: float = 6.0,
    ) -> None:
        """Adds a loop closure edge between keyframe i and keyframe j with sanity validation.

        Args:
            i: Start keyframe ID.
            j: End keyframe ID.
            R_ij: Measured 3x3 relative rotation from i to j.
            t_ij: Measured 3D relative translation from i to j.
            information: 6x6 information matrix. Defaults to Identity(6).
            max_trans: Maximum plausible translation in meters (rejects degenerate PnP).
        """
        R_arr = np.asarray(R_ij, dtype=np.float64)
        t_arr = np.asarray(t_ij, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(R_arr)) or not np.all(np.isfinite(t_arr)):
            return
        if float(np.linalg.norm(t_arr)) > max_trans:
            return
        if information is None:
            information = np.eye(6, dtype=np.float64)
        self.edges.append((i, j, R_arr.copy(), t_arr.copy(), information.copy()))

    def _inverse_right_jacobian_so3(self, phi: np.ndarray) -> np.ndarray:
        """Inverse of SO(3) Right Jacobian Jr_inv(phi)."""
        angle = np.linalg.norm(phi)
        if angle < 1e-8:
            return np.eye(3) + 0.5 * skew_symmetric(phi)
        axis = phi / angle
        K = skew_symmetric(axis)
        cot_half = 1.0 / np.tan(angle / 2.0)
        return (
            (angle / 2.0) * cot_half * np.eye(3)
            + 0.5 * K
            + (1.0 - (angle / 2.0) * cot_half) * np.outer(axis, axis)
        )

    def optimize(
        self,
        max_iterations: int = 20,
        tol: float = 1e-6,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Performs Gauss-Newton SE(3) pose graph optimization with Gauge Anchoring on Pose 0.

        Args:
            max_iterations: Maximum Gauss-Newton iterations.
            tol: Convergence tolerance on step norm.

        Returns:
            dict mapping kf_id -> (R_opt, t_opt).
        """
        if not self.nodes:
            return {}

        node_ids = sorted(list(self.nodes.keys()))
        id_to_idx = {kf_id: idx for idx, kf_id in enumerate(node_ids)}
        N = len(node_ids)
        M = 6 * N

        # Pose 0 Anchor Information Matrix Omega_0 = 10^9 * I_6
        Omega_0 = 1e9 * np.eye(6, dtype=np.float64)

        b = np.zeros(M, dtype=np.float64)

        for iteration in range(max_iterations):
            H_lil = sp.lil_matrix((M, M), dtype=np.float64)
            b = np.zeros(M, dtype=np.float64)

            # 1. Accumulate edge residuals and Jacobians
            for i, j, R_meas, t_meas, Omega in self.edges:
                if i not in id_to_idx or j not in id_to_idx:
                    continue

                idx_i = 6 * id_to_idx[i]
                idx_j = 6 * id_to_idx[j]

                R_i, t_i = self.nodes[i]
                R_j, t_j = self.nodes[j]

                # T_err = T_meas^-1 * T_i^-1 * T_j
                R_err = R_meas.T @ R_i.T @ R_j
                t_err = R_meas.T @ (R_i.T @ (t_j - t_i) - t_meas)
                phi_err = so3_log(R_err)

                e_ij = np.hstack([t_err, phi_err])  # 6D error vector

                # Jacobians
                Jr_inv_err = self._inverse_right_jacobian_so3(phi_err)
                Jl_inv_err = Jr_inv_err.T

                J_j = np.zeros((6, 6), dtype=np.float64)
                J_j[0:3, 0:3] = R_meas.T @ R_i.T
                J_j[3:6, 3:6] = Jr_inv_err

                J_i = np.zeros((6, 6), dtype=np.float64)
                J_i[0:3, 0:3] = -R_meas.T @ R_i.T
                J_i[0:3, 3:6] = R_meas.T @ skew_symmetric(R_i.T @ (t_j - t_i))
                J_i[3:6, 3:6] = -Jl_inv_err @ R_meas.T

                # Huber robust cost on loop edges (abs(j - i) > 1)
                is_loop = abs(j - i) > 1
                if is_loop:
                    e_norm = float(np.linalg.norm(e_ij))
                    delta_huber = 1.5
                    w_huber = min(1.0, delta_huber / max(e_norm, 1e-9))
                    Omega_eff = Omega * w_huber
                else:
                    Omega_eff = Omega

                # Accumulate linear system H dx = -b
                b[idx_i : idx_i + 6] += J_i.T @ Omega_eff @ e_ij
                b[idx_j : idx_j + 6] += J_j.T @ Omega_eff @ e_ij

                H_lil[idx_i : idx_i + 6, idx_i : idx_i + 6] += J_i.T @ Omega_eff @ J_i
                H_lil[idx_i : idx_i + 6, idx_j : idx_j + 6] += J_i.T @ Omega_eff @ J_j
                H_lil[idx_j : idx_j + 6, idx_i : idx_i + 6] += J_j.T @ Omega_eff @ J_i
                H_lil[idx_j : idx_j + 6, idx_j : idx_j + 6] += J_j.T @ Omega_eff @ J_j

            # 2. Accumulate Anchor Prior on Pose 0
            if 0 in id_to_idx and self.prior_pose_0 is not None:
                idx_0 = 6 * id_to_idx[0]
                R_0_prior, t_0_prior = self.prior_pose_0
                R_0_curr, t_0_curr = self.nodes[0]

                e_0_t = t_0_curr - t_0_prior
                e_0_r = so3_log(R_0_prior.T @ R_0_curr)
                e_0 = np.hstack([e_0_t, e_0_r])

                b[idx_0 : idx_0 + 6] += Omega_0 @ e_0
                H_lil[idx_0 : idx_0 + 6, idx_0 : idx_0 + 6] += Omega_0

            # Solve sparse linear system H dx = -b with Tikhonov regularizer
            H_csr = H_lil.tocsr()
            H_csr.setdiag(H_csr.diagonal() + 1e-4)

            dx = None
            try:
                dx = splinalg.spsolve(H_csr, -b)
            except Exception:
                try:
                    H_dense = H_csr.toarray()
                    dx = np.linalg.lstsq(H_dense, -b, rcond=1e-5)[0]
                except Exception:
                    break

            if dx is None or not np.all(np.isfinite(dx)):
                break

            step_norm = float(np.linalg.norm(dx))
            max_step = max(300.0, np.sqrt(N) * 20.0)
            logger.info("PGO iter %d: step_norm=%.3f, max_step=%.1f", iteration, step_norm, max_step)
            if not np.isfinite(step_norm) or step_norm > max_step:
                logger.warning("PGO step norm exceeded: %.3f > %.1f", step_norm, max_step)
                break
            if step_norm < tol:
                self._update_nodes(node_ids, id_to_idx, dx)
                break

            max_dphi = 0.0
            max_dp = 0.0
            for kf_id in node_ids:
                idx_k = 6 * id_to_idx[kf_id]
                dp_n = float(np.linalg.norm(dx[idx_k : idx_k + 3]))
                dphi_n = float(np.linalg.norm(dx[idx_k + 3 : idx_k + 6]))
                if dp_n > max_dp:
                    max_dp = dp_n
                if dphi_n > max_dphi:
                    max_dphi = dphi_n

            step_scale = 1.0
            if max_dphi > 0.20:
                step_scale = min(step_scale, 0.20 / max_dphi)
            if max_dp > 0.50:
                step_scale = min(step_scale, 0.50 / max_dp)

            # Retract state update on manifold with trust region scaling
            self._update_nodes(node_ids, id_to_idx, dx * step_scale)

        return self.nodes

    def _update_nodes(
        self,
        node_ids: list[int],
        id_to_idx: dict[int, int],
        dx: np.ndarray,
    ) -> None:
        """Applies manifold updates dx to node poses."""
        for kf_id in node_ids:
            idx = 6 * id_to_idx[kf_id]
            dp = dx[idx : idx + 3]
            dphi = dx[idx + 3 : idx + 6]
            if not np.all(np.isfinite(dp)) or not np.all(np.isfinite(dphi)):
                continue

            R_curr, t_curr = self.nodes[kf_id]
            t_updated = t_curr + dp
            R_updated = R_curr @ so3_exp(dphi)
            if np.all(np.isfinite(t_updated)) and np.all(np.isfinite(R_updated)):
                self.nodes[kf_id] = (R_updated, t_updated)


def apply_pgo_se3_interpolation(
    times: np.ndarray,
    R_filter: np.ndarray,
    t_filter: np.ndarray,
    node_times: np.ndarray,
    R_opt: np.ndarray,
    t_opt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies SE(3) pose graph optimization correction spline to filter poses.

    For each PGO node k:
        Delta_T_k = T_opt_k @ inv(T_filter_k)
        xi_k = (so3_log(Delta_R_k), Delta_t_k)

    For camera time t between node k and node k+1:
        tau = (t - t_k) / (t_{k+1} - t_k)
        xi(tau) = (1 - tau) * xi_k + tau * xi_{k+1}
        T_out(t) = exp(xi(tau)) @ T_filter(t)

    Handles boundary clamping:
        for t <= node_times[0], uses Delta_T_0
        for t >= node_times[-1], uses Delta_T_{K-1}

    Args:
        times: (N,) camera frame timestamps in seconds.
        R_filter: (N, 3, 3) rotation matrices from filter.
        t_filter: (N, 3) translations from filter.
        node_times: (K,) node timestamps in seconds.
        R_opt: (K, 3, 3) optimized rotation matrices at nodes.
        t_opt: (K, 3) optimized translations at nodes.

    Returns:
        R_out: (N, 3, 3) corrected rotation matrices.
        t_out: (N, 3) corrected translations.
    """
    times = np.asarray(times, dtype=np.float64)
    R_filter = np.asarray(R_filter, dtype=np.float64)
    t_filter = np.asarray(t_filter, dtype=np.float64)
    node_times = np.asarray(node_times, dtype=np.float64)
    R_opt = np.asarray(R_opt, dtype=np.float64)
    t_opt = np.asarray(t_opt, dtype=np.float64)

    N = len(times)
    if N == 0:
        return np.empty((0, 3, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    K = len(node_times)
    if K == 0:
        return R_filter.copy(), t_filter.copy()

    # Find or interpolate filter poses at node timestamps
    R_node_f = np.zeros((K, 3, 3), dtype=np.float64)
    t_node_f = np.zeros((K, 3), dtype=np.float64)
    for k, nt in enumerate(node_times):
        idx = int(np.argmin(np.abs(times - nt)))
        if np.abs(times[idx] - nt) < 1e-4 or N == 1:
            R_node_f[k] = R_filter[idx]
            t_node_f[k] = t_filter[idx]
        else:
            t_node_f[k] = np.array([np.interp(nt, times, t_filter[:, c]) for c in range(3)])
            R_node_f[k] = R_filter[idx]

    R_out = np.zeros_like(R_filter)
    t_out = np.zeros_like(t_filter)

    if K == 1:
        delta_R_0 = R_opt[0] @ R_node_f[0].T
        delta_t_0 = t_opt[0] - t_node_f[0]
        for i in range(N):
            R_out[i] = delta_R_0 @ R_filter[i]
            t_out[i] = t_filter[i] + delta_t_0
        return R_out, t_out

    delta_R = np.zeros((K, 3, 3), dtype=np.float64)
    delta_t = np.zeros((K, 3), dtype=np.float64)
    omega = np.zeros((K, 3), dtype=np.float64)

    for k in range(K):
        delta_R[k] = R_opt[k] @ R_node_f[k].T
        delta_t[k] = t_opt[k] - t_node_f[k]
        omega[k] = so3_log(delta_R[k])

    t_start = node_times[0]
    t_end = node_times[-1]

    for i in range(N):
        t_curr = times[i]
        if t_curr <= t_start:
            R_delta = delta_R[0]
            v_interp = delta_t[0]
        elif t_curr >= t_end:
            R_delta = delta_R[-1]
            v_interp = delta_t[-1]
        else:
            k = int(np.searchsorted(node_times, t_curr, side="right") - 1)
            k = max(0, min(k, K - 2))
            dt = node_times[k + 1] - node_times[k]
            tau = (t_curr - node_times[k]) / dt if dt > 1e-9 else 0.0
            tau = max(0.0, min(1.0, tau))

            omega_interp = (1.0 - tau) * omega[k] + tau * omega[k + 1]
            v_interp = (1.0 - tau) * delta_t[k] + tau * delta_t[k + 1]
            R_delta = so3_exp(omega_interp)

        R_out[i] = R_delta @ R_filter[i]
        t_out[i] = t_filter[i] + v_interp

    if not np.all(np.isfinite(R_out)) or not np.all(np.isfinite(t_out)):
        return R_filter.copy(), t_filter.copy()

    return R_out, t_out
