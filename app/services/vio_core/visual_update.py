"""Visual measurement update for SR-ESKF with Huber weighting and Chi-square gating."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import block_diag

from app.services.vio_core.math_utils import skew_symmetric
from app.services.vio_core.state import NominalState, StateIndex as I
from app.services.vio_core.triangulation import triangulate_dlt_batch

CHI2_VIS_2 = 5.99
HUBER_DELTA_PX = 2.0
SIGMA_PIX = 1.0
N_MAX_TRACKS = 60
MIN_PARALLAX_DEG = 1.5
MISS_RETIRE = 3

ORIGIN_SPAD = "spad"
ORIGIN_DLT = "dlt"

STATUS_CANDIDATE = "CANDIDATE"
STATUS_ACTIVE = "ACTIVE"
STATUS_RETIRED = "RETIRED"


@dataclass(slots=True)
class LandmarkTrack:
    track_id: int
    status: str
    origin: str
    uv: np.ndarray
    u_cam: np.ndarray | None = None
    Z_spad: float | None = None
    R_host: np.ndarray | None = None
    p_host: np.ndarray | None = None
    p_w: np.ndarray | None = None
    Sigma_pf: np.ndarray | None = None
    misses: int = 0
    uv_host: np.ndarray | None = None


def camera_pose_from_body(state: NominalState, T_bc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute camera pose in world frame from IMU body state and T_bc extrinsics.

    Args:
        state: Nominal state containing body position p, velocity v, orientation R, time offset td.
        T_bc: 4x4 rigid transformation mapping camera frame to IMU body frame.

    Returns:
        (R_wc, p_wc): Camera orientation matrix (3x3) and camera position vector (3,).
    """
    R_bc = T_bc[:3, :3]
    t_bc = T_bc[:3, 3]
    R_wc = state.R @ R_bc
    p_wc = state.p + state.v * state.td + state.R @ t_bc
    return R_wc, p_wc


def relative_camera_rotation(R_prev: np.ndarray, R_now: np.ndarray) -> np.ndarray:
    """Compute relative camera rotation R_prev_curr = R_now.T @ R_prev.

    Transforms vectors from previous camera frame to current camera frame: v_curr = R_prev_curr @ v_prev.

    Args:
        R_prev: 3x3 orientation matrix of previous camera frame in world frame.
        R_now: 3x3 orientation matrix of current camera frame in world frame.

    Returns:
        3x3 relative rotation matrix R_prev_curr = R_now.T @ R_prev.
    """
    return R_now.T @ R_prev


def spad_world_point(track: LandmarkTrack, s_L: float) -> np.ndarray:
    """Compute 3D landmark position in world frame from SPAD ray+depth anchor.

    p_w(s_L) = p_host + R_host @ (s_L * Z_spad * u_cam)

    Args:
        track: SPAD LandmarkTrack with host pose and camera ray.
        s_L: LiDAR scale factor from nominal state.

    Returns:
        3D world coordinate array of shape (3,).
    """
    return track.p_host + track.R_host @ (s_L * track.Z_spad * track.u_cam)


def build_visual_measurement(
    state: NominalState,
    tracks: list[LandmarkTrack],
    K: np.ndarray,
    T_bc: np.ndarray,
    P: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build stacked visual measurement residual, Jacobian, and effective covariance.

    Applies Huber robust weighting and per-track chi-square (2 DOF) pre-gating.
    Limits measurement to at most N_MAX_TRACKS, prioritizing SPAD-anchored tracks.

    Args:
        state: Current nominal state.
        tracks: List of landmark tracks to evaluate.
        K: 3x3 camera intrinsic matrix.
        T_bc: 4x4 camera-to-body extrinsics.
        P: 23x23 error-state covariance matrix.
        image_size: Optional (width, height) of camera image.

    Returns:
        tuple (y, H, R_cov) or None if no tracks pass pre-gating:
            y: Stacked residual vector of shape (2N,).
            H: Stacked measurement Jacobian of shape (2N, 23).
            R_cov: Block-diagonal effective measurement covariance of shape (2N, 2N).
    """
    R_bc = T_bc[:3, :3]
    t_bc = T_bc[:3, 3]
    R_wc, p_wc = camera_pose_from_body(state, T_bc)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if image_size is not None:
        img_w, img_h = float(image_size[0]), float(image_size[1])
    else:
        img_w, img_h = float(2.0 * cx), float(2.0 * cy)

    # Prioritize SPAD tracks, then DLT, capped at N_MAX_TRACKS
    tracks_to_process = sorted(tracks, key=lambda t: 0 if t.origin == ORIGIN_SPAD else 1)[:N_MAX_TRACKS]

    y_list: list[np.ndarray] = []
    H_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []

    cols_spad = np.array([0, 1, 2, 6, 7, 8, 21, 22], dtype=np.int64)
    cols_dlt = np.array([0, 1, 2, 6, 7, 8, 21], dtype=np.int64)
    P_spad = P[np.ix_(cols_spad, cols_spad)]
    P_dlt = P[np.ix_(cols_dlt, cols_dlt)]
    R_state_T = state.R.T

    for tr in tracks_to_process:
        if tr.status != STATUS_ACTIVE:
            continue
        if tr.origin == ORIGIN_SPAD:
            if tr.Z_spad is None or tr.u_cam is None or tr.p_host is None or tr.R_host is None:
                continue
            p_w = spad_world_point(tr, state.sl)
        elif tr.origin == ORIGIN_DLT:
            if tr.p_w is None:
                continue
            p_w = tr.p_w
        else:
            continue

        p_c = R_wc.T @ (p_w - p_wc)
        X, Y, Z = p_c[0], p_c[1], p_c[2]
        if Z <= 0.2:
            continue

        u_proj = fx * (X / Z) + cx
        v_proj = fy * (Y / Z) + cy

        # Reject tracks whose projected points leave camera image bounds (with margin)
        margin = 20.0
        if u_proj < -margin or u_proj > img_w + margin or v_proj < -margin or v_proj > img_h + margin:
            continue

        # Reject tracks whose measured points are outside camera bounds
        if tr.uv[0] < -margin or tr.uv[0] > img_w + margin or tr.uv[1] < -margin or tr.uv[1] > img_h + margin:
            continue

        pi_c = np.array([u_proj, v_proj])

        # Residual innovation y = z - h(x)
        y_i = tr.uv - pi_c
        r_norm = float(np.linalg.norm(y_i))

        # Camera projection Jacobian J_pi (2x3)
        inv_Z = 1.0 / Z
        inv_Z2 = inv_Z * inv_Z
        J_pi = np.array([
            [fx * inv_Z, 0.0, -fx * X * inv_Z2],
            [0.0, fy * inv_Z, -fy * Y * inv_Z2],
        ])
        J_b = J_pi @ R_bc.T
        J_R = J_b @ R_state_T

        # Measurement Jacobian H_i (2x23)
        H_i = np.zeros((2, 23))
        # Position error state: -J_R
        H_i[:, I.POS] = -J_R
        # Orientation error state: J_pi @ R_bc.T @ [p_b]_x
        p_b = R_bc @ p_c + t_bc
        H_i[:, I.ORI] = J_b @ skew_symmetric(p_b)
        # Time-offset error state: -J_R @ v
        H_i[:, I.TD] = (-J_R @ state.v).reshape(2, 1)
        # Scale error state (SPAD only): J_R @ R_host @ u_cam * Z_spad
        is_spad = (tr.origin == ORIGIN_SPAD)
        if is_spad:
            H_i[:, I.SL] = (J_R @ (tr.R_host @ tr.u_cam) * tr.Z_spad).reshape(2, 1)

        # Geometric covariance: sigma_pix^2 * I_2 + J_R @ Sigma_pf @ J_R.T
        if tr.Sigma_pf is not None:
            R_geom = (SIGMA_PIX ** 2) * np.eye(2) + (J_R @ tr.Sigma_pf) @ J_R.T
        else:
            R_geom = np.diag([SIGMA_PIX ** 2, SIGMA_PIX ** 2])

        # Huber robust weighting
        w = min(1.0, HUBER_DELTA_PX / max(r_norm, 1e-9))
        R_eff = (1.0 / w) * R_geom

        # Chi-square innovation pre-gating with fast sub-covariance projection
        if is_spad:
            H_sub = H_i[:, cols_spad]
            S_i = H_sub @ P_spad @ H_sub.T + R_eff
        else:
            H_sub = H_i[:, cols_dlt]
            S_i = H_sub @ P_dlt @ H_sub.T + R_eff

        det_S = S_i[0, 0] * S_i[1, 1] - S_i[0, 1] * S_i[1, 0]
        if det_S <= 1e-12:
            continue
        inv_det = 1.0 / det_S
        d2 = float(
            (
                y_i[0] * (S_i[1, 1] * y_i[0] - S_i[0, 1] * y_i[1])
                + y_i[1] * (-S_i[1, 0] * y_i[0] + S_i[0, 0] * y_i[1])
            )
            * inv_det
        )

        if d2 >= CHI2_VIS_2:
            continue

        y_list.append(y_i)
        H_list.append(H_i)
        R_list.append(R_eff)

    if not y_list:
        return None

    y_stacked = np.concatenate(y_list, axis=0)
    H_stacked = np.vstack(H_list)
    n_meas = len(y_list)
    R_stacked = np.zeros((2 * n_meas, 2 * n_meas), dtype=np.float64)
    R_arr = np.asarray(R_list)
    bi = np.arange(n_meas)
    R_stacked[2 * bi, 2 * bi] = R_arr[:, 0, 0]
    R_stacked[2 * bi, 2 * bi + 1] = R_arr[:, 0, 1]
    R_stacked[2 * bi + 1, 2 * bi] = R_arr[:, 1, 0]
    R_stacked[2 * bi + 1, 2 * bi + 1] = R_arr[:, 1, 1]
    return y_stacked, H_stacked, R_stacked


class LandmarkTrackMap:
    """Manages visual landmark feature tracks across camera frames with SPAD-first DLT lifecycle."""

    def __init__(self, K: np.ndarray | None = None) -> None:
        self.tracks_by_id: dict[int, LandmarkTrack] = {}
        self._next_id: int = 1
        self.K: np.ndarray | None = K

    def sync_klt(
        self,
        matched_pts_prev: np.ndarray,
        matched_pts_curr: np.ndarray,
        new_uv: np.ndarray,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        """Synchronize tracks with visual frontend KLT tracking output.

        Matches matched_pts_prev to stored uv (nearest, <= 1 px), updates uv for survivors,
        increments misses for unobserved alive tracks (retiring at MISS_RETIRE), and
        registers new_uv with monotonic track IDs as CANDIDATE.
        """
        pts_prev = np.asarray(matched_pts_prev, dtype=float).reshape(-1, 2)
        pts_curr = np.asarray(matched_pts_curr, dtype=float).reshape(-1, 2)
        new_pts = np.asarray(new_uv, dtype=float).reshape(-1, 2)

        w_bound = None
        h_bound = None
        if image_size is not None:
            w_bound, h_bound = float(image_size[0]), float(image_size[1])
        elif self.K is not None:
            w_bound, h_bound = float(2.0 * self.K[0, 2]), float(2.0 * self.K[1, 2])

        alive_tracks = [t for t in self.tracks_by_id.values() if t.status != STATUS_RETIRED]
        matched_track_ids: set[int] = set()

        if len(pts_prev) > 0 and len(alive_tracks) > 0:
            track_uvs = np.array([t.uv for t in alive_tracks])
            diff = pts_prev[:, None, :] - track_uvs[None, :, :]
            dists_sq = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2

            thresh_sq = (1.0 + 1e-4) ** 2
            i_indices, j_indices = np.where(dists_sq <= thresh_sq)
            if len(i_indices) > 0:
                dist_vals = dists_sq[i_indices, j_indices]
                order = np.argsort(dist_vals)
                used_prev = set()
                used_tracks = set()
                for idx in order:
                    i = int(i_indices[idx])
                    j = int(j_indices[idx])
                    if i not in used_prev and j not in used_tracks:
                        used_prev.add(i)
                        used_tracks.add(j)
                        t = alive_tracks[j]
                        t.uv = pts_curr[i].copy()
                        t.misses = 0
                        if w_bound is not None and h_bound is not None:
                            if t.uv[0] < 0.0 or t.uv[0] >= w_bound or t.uv[1] < 0.0 or t.uv[1] >= h_bound:
                                t.status = STATUS_RETIRED
                        matched_track_ids.add(t.track_id)

        # Increment misses for unobserved alive tracks and retire if misses >= MISS_RETIRE
        for t in alive_tracks:
            if t.track_id not in matched_track_ids:
                t.misses += 1
                if t.misses >= MISS_RETIRE:
                    t.status = STATUS_RETIRED

        # Prune retired tracks when track map grows to prevent memory and triangulation slowdown
        if len(self.tracks_by_id) > 200:
            self.tracks_by_id = {
                tid: tr for tid, tr in self.tracks_by_id.items()
                if tr.status != STATUS_RETIRED
            }

        # Register new UV points
        for pt in new_pts:
            if w_bound is not None and h_bound is not None:
                if pt[0] < 0.0 or pt[0] >= w_bound or pt[1] < 0.0 or pt[1] >= h_bound:
                    continue
            tid = self._next_id
            self._next_id += 1
            t = LandmarkTrack(
                track_id=tid,
                status=STATUS_CANDIDATE,
                origin=ORIGIN_DLT,
                uv=pt.copy(),
                uv_host=pt.copy(),
                misses=0,
            )
            self.tracks_by_id[tid] = t

    def ingest_spad(
        self,
        depths: np.ndarray,
        validity: np.ndarray,
        variances: np.ndarray,
        R_wc: np.ndarray,
        p_wc: np.ndarray,
        s_L: float,
        K: np.ndarray | None = None,
    ) -> None:
        """Promote candidate tracks with valid SPAD associations to ACTIVE SPAD tracks."""
        if K is not None:
            self.K = K

        depths_arr = np.asarray(depths, dtype=float)
        valid_arr = np.asarray(validity, dtype=bool)
        var_arr = np.asarray(variances, dtype=float)

        candidates = [t for t in self.tracks_by_id.values() if t.status == STATUS_CANDIDATE]
        alive = [t for t in self.tracks_by_id.values() if t.status != STATUS_RETIRED]

        if len(depths_arr) == len(candidates):
            target_tracks = candidates
        elif len(depths_arr) == len(alive):
            target_tracks = alive
        elif len(depths_arr) == len(self.tracks_by_id):
            target_tracks = list(self.tracks_by_id.values())
        else:
            target_tracks = candidates[: len(depths_arr)]

        K_use = self.K
        fx, fy, cx, cy = (K_use[0, 0], K_use[1, 1], K_use[0, 2], K_use[1, 2]) if K_use is not None else (None, None, None, None)
        tan_sigma_sq = float(np.tan(np.deg2rad(1.0)) ** 2)

        for i, t in enumerate(target_tracks):
            if i >= len(depths_arr):
                break
            if t.status != STATUS_CANDIDATE:
                continue
            if not valid_arr[i] or depths_arr[i] <= 0:
                continue

            t.status = STATUS_ACTIVE
            t.origin = ORIGIN_SPAD
            t.Z_spad = float(depths_arr[i])
            t.R_host = R_wc.copy()
            t.p_host = p_wc.copy()

            if fx is not None:
                u, v = t.uv[0], t.uv[1]
                t.u_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
            elif t.u_cam is None:
                t.u_cam = np.array([0.0, 0.0, 1.0], dtype=float)

            var_val = float(var_arr[i]) if (i < len(var_arr) and np.isfinite(var_arr[i])) else 0.01
            u_norm_sq = float(t.u_cam[0] ** 2 + t.u_cam[1] ** 2 + t.u_cam[2] ** 2)
            if u_norm_sq > 1e-12:
                u_dir = t.u_cam / np.sqrt(u_norm_sq)
            else:
                u_dir = np.array([0.0, 0.0, 1.0])
            var_ang = (t.Z_spad ** 2) * tan_sigma_sq
            u_world = t.R_host @ u_dir
            t.Sigma_pf = var_ang * np.eye(3) + (var_val - var_ang) * np.outer(u_world, u_world)

    def try_promote_dlt(
        self,
        R_wc: np.ndarray,
        p_wc: np.ndarray,
        K: np.ndarray,
    ) -> None:
        """Triangulate candidate tracks against host pose and promote if parallax >= 1.5 deg.

        Tracks without a host pose get the current pose as host. All remaining candidates are
        triangulated in one stacked DLT call (numerically identical to the per-track version).
        """
        self.K = K

        batch_tracks: list[LandmarkTrack] = []
        for t in list(self.tracks_by_id.values()):
            if t.status != STATUS_CANDIDATE:
                continue
            if t.R_host is None or t.p_host is None:
                t.R_host = R_wc.copy()
                t.p_host = p_wc.copy()
                t.uv_host = t.uv.copy()
                continue
            batch_tracks.append(t)

        if not batch_tracks:
            return

        uv1 = np.array([t.uv_host if t.uv_host is not None else t.uv for t in batch_tracks], dtype=np.float64).reshape(-1, 2)
        uv2 = np.array([t.uv for t in batch_tracks], dtype=np.float64).reshape(-1, 2)
        R1 = np.stack([t.R_host for t in batch_tracks]).astype(np.float64)
        t1 = np.stack([t.p_host for t in batch_tracks]).astype(np.float64)

        res = triangulate_dlt_batch(uv1, uv2, R1, t1, R_wc, p_wc, K, min_parallax_deg=MIN_PARALLAX_DEG)

        for i in np.flatnonzero(res.valid):
            t = batch_tracks[int(i)]
            t.status = STATUS_ACTIVE
            t.origin = ORIGIN_DLT
            t.p_w = res.point_3d[i].copy()
            sin_px = np.sin(np.radians(max(float(res.parallax_deg[i]), 0.1)))
            var_dlt = float(np.clip((0.05 / sin_px) ** 2, 1e-4, 1.0))
            t.Sigma_pf = np.eye(3) * var_dlt

    def select_for_update(self, image_size: tuple[int, int]) -> list[LandmarkTrack]:
        """Select up to N_MAX_TRACKS active tracks, prioritizing SPAD with 8x6 grid distribution."""
        active = [t for t in self.tracks_by_id.values() if t.status == STATUS_ACTIVE]
        spad_tracks = [t for t in active if t.origin == ORIGIN_SPAD]
        dlt_tracks = [t for t in active if t.origin == ORIGIN_DLT]

        if len(spad_tracks) >= N_MAX_TRACKS:
            return spad_tracks[:N_MAX_TRACKS]

        chosen: list[LandmarkTrack] = list(spad_tracks)
        remaining_slots = N_MAX_TRACKS - len(chosen)

        if remaining_slots > 0 and dlt_tracks:
            w, h = image_size
            w_cell = max(w / 8.0, 1e-6)
            h_cell = max(h / 6.0, 1e-6)

            cell_counts: dict[tuple[int, int], int] = {}
            remaining_dlt: list[LandmarkTrack] = []

            for t in dlt_tracks:
                c = int(np.clip(t.uv[0] // w_cell, 0, 7))
                r = int(np.clip(t.uv[1] // h_cell, 0, 5))
                cell = (c, r)
                if cell_counts.get(cell, 0) < 1:
                    chosen.append(t)
                    cell_counts[cell] = cell_counts.get(cell, 0) + 1
                    if len(chosen) == N_MAX_TRACKS:
                        return chosen
                else:
                    remaining_dlt.append(t)

            for t in remaining_dlt:
                if len(chosen) >= N_MAX_TRACKS:
                    break
                chosen.append(t)

        return chosen

    def camera_points_for_loop(
        self,
        R_wc: np.ndarray,
        p_wc: np.ndarray,
        s_L: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 3D landmark points in current camera frame p_c and their uv coordinates."""
        p_c_list: list[np.ndarray] = []
        uv_list: list[np.ndarray] = []

        for t in self.tracks_by_id.values():
            if t.status != STATUS_ACTIVE:
                continue

            if t.origin == ORIGIN_SPAD:
                if t.p_host is None or t.R_host is None or t.Z_spad is None or t.u_cam is None:
                    continue
                p_w = spad_world_point(t, s_L)
            elif t.origin == ORIGIN_DLT:
                if t.p_w is None:
                    continue
                p_w = t.p_w
            else:
                continue

            p_c = R_wc.T @ (p_w - p_wc)
            p_c_list.append(p_c)
            uv_list.append(t.uv)

        if not p_c_list:
            return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=float)

        return np.array(p_c_list, dtype=float), np.array(uv_list, dtype=float)

