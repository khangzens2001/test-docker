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

        def _eval_cost() -> float:
            total_c = 0.0
            for i_e, j_e, R_m, t_m, Om in self.edges:
                if i_e not in id_to_idx or j_e not in id_to_idx:
                    continue
                R_ie, t_ie = self.nodes[i_e]
                R_je, t_je = self.nodes[j_e]
                R_er = R_m.T @ R_ie.T @ R_je
                t_er = R_m.T @ (R_ie.T @ (t_je - t_ie) - t_m)
                phi_er = so3_log(R_er)
                e_vec = np.hstack([t_er, phi_er])
                if abs(j_e - i_e) > 1:
                    en = float(np.linalg.norm(e_vec))
                    wh = min(1.0, 1.5 / max(en, 1e-9))
                    Om_e = Om * wh
                else:
                    Om_e = Om
                total_c += 0.5 * float(e_vec @ Om_e @ e_vec)
            if 0 in id_to_idx and self.prior_pose_0 is not None:
                R_0p, t_0p = self.prior_pose_0
                R_0c, t_0c = self.nodes[0]
                e_0v = np.hstack([t_0c - t_0p, so3_log(R_0p.T @ R_0c)])
                total_c += 0.5 * float(e_0v @ Omega_0 @ e_0v)
            return total_c

        lambda_lm = 1e-3
        curr_cost = _eval_cost()

        use_dense = M < 600
        if not use_dense:
            r6, c6 = np.meshgrid(np.arange(6, dtype=np.int32), np.arange(6, dtype=np.int32), indexing="ij")
            r6 = r6.ravel()
            c6 = c6.ravel()

        for iteration in range(max_iterations):
            if use_dense:
                H_mat = np.zeros((M, M), dtype=np.float64)
            else:
                row_blocks = []
                col_blocks = []
                val_blocks = []
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
                is_loop = abs(j - i) > 1

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
                if is_loop:
                    e_norm = float(np.linalg.norm(e_ij))
                    delta_huber = 1.5
                    w_huber = min(1.0, delta_huber / max(e_norm, 1e-9))
                    Omega_eff = Omega * w_huber
                else:
                    Omega_eff = Omega

                # Accumulate linear system H dx = -b
                JtO_i = J_i.T @ Omega_eff
                JtO_j = J_j.T @ Omega_eff
                b[idx_i : idx_i + 6] += JtO_i @ e_ij
                b[idx_j : idx_j + 6] += JtO_j @ e_ij

                H_ii = JtO_i @ J_i
                H_ij = JtO_i @ J_j
                H_ji = JtO_j @ J_i
                H_jj = JtO_j @ J_j
                if use_dense:
                    H_mat[idx_i : idx_i + 6, idx_i : idx_i + 6] += H_ii
                    H_mat[idx_i : idx_i + 6, idx_j : idx_j + 6] += H_ij
                    H_mat[idx_j : idx_j + 6, idx_i : idx_i + 6] += H_ji
                    H_mat[idx_j : idx_j + 6, idx_j : idx_j + 6] += H_jj
                else:
                    row_blocks.extend((r6 + idx_i, r6 + idx_i, r6 + idx_j, r6 + idx_j))
                    col_blocks.extend((c6 + idx_i, c6 + idx_j, c6 + idx_i, c6 + idx_j))
                    val_blocks.extend((H_ii.ravel(), H_ij.ravel(), H_ji.ravel(), H_jj.ravel()))

            # 2. Accumulate Anchor Prior on Pose 0
            if 0 in id_to_idx and self.prior_pose_0 is not None:
                idx_0 = 6 * id_to_idx[0]
                R_0_prior, t_0_prior = self.prior_pose_0
                R_0_curr, t_0_curr = self.nodes[0]

                e_0_t = t_0_curr - t_0_prior
                e_0_r = so3_log(R_0_prior.T @ R_0_curr)
                e_0 = np.hstack([e_0_t, e_0_r])

                b[idx_0 : idx_0 + 6] += Omega_0 @ e_0
                if use_dense:
                    H_mat[idx_0 : idx_0 + 6, idx_0 : idx_0 + 6] += Omega_0
                else:
                    row_blocks.append(r6 + idx_0)
                    col_blocks.append(c6 + idx_0)
                    val_blocks.append(Omega_0.ravel())

            # Solve linear system H dx = -b with Levenberg-Marquardt regularizer
            dx = None
            if use_dense:
                H_mat[np.diag_indices(M)] += lambda_lm
                try:
                    dx = np.linalg.solve(H_mat, -b)
                except Exception:
                    try:
                        dx = np.linalg.lstsq(H_mat, -b, rcond=1e-5)[0]
                    except Exception:
                        break
            else:
                diag_idx = np.arange(M, dtype=np.int32)
                row_blocks.append(diag_idx)
                col_blocks.append(diag_idx)
                val_blocks.append(np.full(M, lambda_lm, dtype=np.float64))
                H_csr = sp.coo_matrix(
                    (np.concatenate(val_blocks), (np.concatenate(row_blocks), np.concatenate(col_blocks))),
                    shape=(M, M),
                    dtype=np.float64,
                ).tocsr()
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
            if max_dphi > 0.50:
                step_scale = min(step_scale, 0.50 / max_dphi)
            if max_dp > 0.80:
                step_scale = min(step_scale, 0.80 / max_dp)

            saved_nodes = {k: (v[0].copy(), v[1].copy()) for k, v in self.nodes.items()}
            self._update_nodes(node_ids, id_to_idx, dx * step_scale)
            new_cost = _eval_cost()
            if new_cost <= curr_cost + 1e-6:
                curr_cost = new_cost
                lambda_lm = max(1e-4, lambda_lm * 0.5)
            else:
                self.nodes = saved_nodes
                lambda_lm = min(1e4, lambda_lm * 10.0)

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
            disp_0 = t_filter[i] - t_node_f[0]
            t_out[i] = t_opt[0] + delta_R_0 @ disp_0
        return R_out, t_out

    delta_R = np.zeros((K, 3, 3), dtype=np.float64)
    omega = np.zeros((K, 3), dtype=np.float64)

    for k in range(K):
        delta_R[k] = R_opt[k] @ R_node_f[k].T
        omega[k] = so3_log(delta_R[k])

    t_start = node_times[0]
    t_end = node_times[-1]

    for i in range(N):
        t_curr = times[i]
        if t_curr <= t_start:
            R_delta = delta_R[0]
            disp = t_filter[i] - t_node_f[0]
            delta_t_0 = t_opt[0] - t_node_f[0]
            t_out[i] = t_node_f[0] + delta_t_0 + R_delta @ disp
        elif t_curr >= t_end:
            R_delta = delta_R[-1]
            disp = t_filter[i] - t_node_f[-1]
            delta_t_end = t_opt[-1] - t_node_f[-1]
            t_out[i] = t_node_f[-1] + delta_t_end + R_delta @ disp
        else:
            k = int(np.searchsorted(node_times, t_curr, side="right") - 1)
            k = max(0, min(k, K - 2))
            dt = node_times[k + 1] - node_times[k]
            tau = (t_curr - node_times[k]) / dt if dt > 1e-9 else 0.0
            tau = max(0.0, min(1.0, tau))

            omega_interp = (1.0 - tau) * omega[k] + tau * omega[k + 1]
            R_delta = so3_exp(omega_interp)

            # Node endpoints with rotational displacement
            p_rot_k = t_node_f[k]
            p_rot_next = t_node_f[k] + delta_R[k + 1] @ (t_node_f[k + 1] - t_node_f[k])

            # Residual translation corrections at endpoints
            delta_t_k = t_opt[k] - p_rot_k
            delta_t_next = t_opt[k + 1] - p_rot_next

            # Interpolated residual translation
            v_interp = (1.0 - tau) * delta_t_k + tau * delta_t_next

            # Rotated displacement for intermediate frame i
            disp = t_filter[i] - t_node_f[k]
            p_rot_i = t_node_f[k] + R_delta @ disp

            t_out[i] = p_rot_i + v_interp

        R_out[i] = R_delta @ R_filter[i]

    if not np.all(np.isfinite(R_out)) or not np.all(np.isfinite(t_out)):
        return R_filter.copy(), t_filter.copy()

    return R_out, t_out
