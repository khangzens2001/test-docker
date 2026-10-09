import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# Configure Numba disk cache directory to persistent /tmp/numba_cache
NUMBA_CACHE_DIR = "/tmp/numba_cache"
os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR
os.makedirs(NUMBA_CACHE_DIR, exist_ok=True)

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    # Fallback dummy decorator if numba is unavailable: kernels run as plain numpy.
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# True when kernels actually run compiled. NUMBA_DISABLE_JIT=1 keeps numba importable but turns every
# @njit function into plain Python, where a double loop would be far slower than numpy.
JIT_ENABLED = HAVE_NUMBA and os.environ.get("NUMBA_DISABLE_JIT", "0").strip() != "1"

@njit(cache=True, fastmath=True)
def fast_vector_norm(vec: np.ndarray) -> float:
    """Compute L2 norm of a 1D vector with Numba JIT compilation."""
    acc = 0.0
    for i in range(vec.shape[0]):
        acc += vec[i] * vec[i]
    return np.sqrt(acc)

@njit(cache=True, fastmath=True)
def fast_quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Perform quaternion multiplication q1 * q2 [x, y, z, w]."""
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    
    out = np.empty(4, dtype=np.float64)
    out[0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out[3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return out

@njit(cache=True, fastmath=True)
def fast_imu_step_integration(p: np.ndarray, v: np.ndarray, a: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Integrate IMU acceleration step p_next = p + v*dt + 0.5*a*dt^2, v_next = v + a*dt."""
    p_next = np.empty(3, dtype=np.float64)
    v_next = np.empty(3, dtype=np.float64)
    for i in range(3):
        p_next[i] = p[i] + v[i] * dt + 0.5 * a[i] * dt * dt
        v_next[i] = v[i] + a[i] * dt
    return p_next, v_next


# ---------------------------------------------------------------------------
# SO(3) twins of math_utils (no fastmath: outputs must match numpy to 1e-12)
# ---------------------------------------------------------------------------

@njit(cache=True)
def nb_skew(v):
    K = np.zeros((3, 3))
    K[0, 1] = -v[2]
    K[0, 2] = v[1]
    K[1, 0] = v[2]
    K[1, 2] = -v[0]
    K[2, 0] = -v[1]
    K[2, 1] = v[0]
    return K


@njit(cache=True)
def nb_so3_exp(phi):
    for i in range(3):
        if not np.isfinite(phi[i]):
            return np.eye(3)
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3) + nb_skew(phi)
    axis = phi / angle
    K = nb_skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


@njit(cache=True)
def nb_so3_log(R):
    for i in range(3):
        for j in range(3):
            if not np.isfinite(R[i, j]):
                return np.zeros(3)
    tr = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    if tr > 1.0:
        tr = 1.0
    if tr < -1.0:
        tr = -1.0
    angle = np.arccos(tr)
    diff = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-8:
        return diff * 0.5
    if abs(angle - np.pi) < 1e-4:
        k = 0
        if R[1, 1] > R[k, k]:
            k = 1
        if R[2, 2] > R[k, k]:
            k = 2
        u = R[:, k].copy()
        u[k] += 1.0
        u = u / np.linalg.norm(u)
        if np.dot(u, diff) < 0:
            u = -u
        return u * angle
    axis = diff / (2.0 * np.sin(angle))
    return axis * angle


@njit(cache=True)
def nb_right_jacobian_so3(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3) - 0.5 * nb_skew(phi)
    axis = phi / angle
    K = nb_skew(axis)
    return np.eye(3) - ((1.0 - np.cos(angle)) / angle) * K + ((angle - np.sin(angle)) / angle) * (K @ K)


# ---------------------------------------------------------------------------
# ANMS-SSC suppression radii (feature_selector.select_anms_ssc inner loop)
# ---------------------------------------------------------------------------

@njit(cache=True)
def anms_radii_nb(sorted_kps, sorted_resp, thr):
    """radii[i] = min distance to an earlier point j < i with resp_j > thr_i (inf when none).

    Arithmetic stays in the input dtype (float32 for FAST keypoints), exactly like
    np.linalg.norm(candidates - kps[i], axis=1) did: sqrt(dx*dx + dy*dy), no contraction."""
    n = sorted_kps.shape[0]
    radii = np.full(n, np.inf)
    for i in range(1, n):
        xi = sorted_kps[i, 0]
        yi = sorted_kps[i, 1]
        best = np.inf
        found = False
        for j in range(i):
            if sorted_resp[j] > thr[i]:
                dx = sorted_kps[j, 0] - xi
                dy = sorted_kps[j, 1] - yi
                d = np.sqrt(dx * dx + dy * dy)
                if (not found) or d < best:
                    best = d
                    found = True
        if found:
            radii[i] = best
    return radii


# ---------------------------------------------------------------------------
# IMU preintegration step (ImuPreintegration.integrate body, pure arrays)
# ---------------------------------------------------------------------------

@njit(cache=True)
def preint_step(delta_R, delta_v, delta_p, cov, J_ba, J_bg, accel, gyro, dt, ba, bg,
                sigma_a, sigma_g, prev_accel, prev_gyro, has_prev):
    """One ImuPreintegration.integrate step. Returns new arrays; expression order mirrors the
    original numpy code so results agree to the last few ulps."""
    inflation_factor = 1.0
    inflated = False
    if dt > 0.1:
        inflated = True
        inflation_factor = (dt / 0.005) ** 2

    a_corr = accel - ba
    w_corr = gyro - bg
    if has_prev:
        a_mid = 0.5 * (prev_accel + a_corr)
        w_mid = 0.5 * (prev_gyro + w_corr)
    else:
        a_mid = a_corr
        w_mid = w_corr

    phi = w_mid * dt
    Jr = nb_right_jacobian_so3(phi)
    dR_step = nb_so3_exp(phi)
    acc_world = delta_R @ a_mid
    acc_skew = nb_skew(acc_world)
    dt2 = dt ** 2

    J_bg_R_prev = J_bg[6:9, :].copy()
    J_ba_v_prev = J_ba[3:6, :].copy()
    J_bg_v_prev = J_bg[3:6, :].copy()

    J_ba_new = J_ba.copy()
    J_bg_new = J_bg.copy()
    J_bg_new[6:9, :] = dR_step.T @ J_bg_R_prev - Jr * dt
    J_ba_new[3:6, :] = J_ba[3:6, :] - delta_R * dt
    J_bg_new[3:6, :] = J_bg[3:6, :] - acc_skew @ J_bg_R_prev * dt
    J_ba_new[0:3, :] = J_ba[0:3, :] + J_ba_v_prev * dt - 0.5 * delta_R * dt2
    J_bg_new[0:3, :] = J_bg[0:3, :] + J_bg_v_prev * dt - 0.5 * acc_skew @ J_bg_R_prev * dt2

    F = np.eye(9)
    F[0:3, 3:6] = np.eye(3) * dt
    F[0:3, 6:9] = -0.5 * acc_skew * dt2
    F[3:6, 6:9] = -acc_skew * dt
    F[6:9, 6:9] = dR_step.T

    G = np.zeros((9, 6))
    G[0:3, 0:3] = 0.5 * delta_R * dt2
    G[3:6, 0:3] = delta_R * dt
    G[6:9, 3:6] = Jr * dt

    Q_d = np.zeros((6, 6))
    for i in range(3):
        Q_d[i, i] = sigma_a ** 2 * inflation_factor
        Q_d[3 + i, 3 + i] = sigma_g ** 2 * inflation_factor

    cov_new = F @ cov @ F.T + G @ Q_d @ G.T

    delta_p_new = delta_p + (delta_v * dt + 0.5 * acc_world * dt2)
    delta_v_new = delta_v + acc_world * dt
    delta_R_new = delta_R @ dR_step
    return delta_R_new, delta_v_new, delta_p_new, cov_new, J_ba_new, J_bg_new, a_corr, w_corr, inflated


# ---------------------------------------------------------------------------
# SR-ESKF prediction (SRESKF.predict_imu body, pure arrays). QR via LAPACK dgeqrf,
# the same routine scipy.linalg.qr used before.
# ---------------------------------------------------------------------------

@njit(cache=True)
def eskf_predict(p, v, R, g, S_P, delta_R, delta_v, delta_p, J_ba, J_bg, dt, Q_sqrt, q_scale):
    R_prev = R.copy()
    p_new = p + (v * dt + 0.5 * g * dt ** 2 + R_prev @ delta_p)
    v_new = v + (g * dt + R_prev @ delta_v)
    R_new = R_prev @ delta_R

    F = np.eye(23)
    F[0:3, 3:6] = np.eye(3) * dt                       # POS <- VEL
    F[0:3, 6:9] = -R_prev @ nb_skew(delta_p)           # POS <- ORI
    F[3:6, 6:9] = -R_prev @ nb_skew(delta_v)           # VEL <- ORI
    F[6:9, 6:9] = delta_R.T                            # ORI <- ORI
    F[0:3, 18:21] = 0.5 * np.eye(3) * dt ** 2          # POS <- GRAV
    F[3:6, 18:21] = np.eye(3) * dt                     # VEL <- GRAV
    F[0:3, 9:12] = R_prev @ J_ba[0:3, :]               # POS <- BA
    F[3:6, 9:12] = R_prev @ J_ba[3:6, :]               # VEL <- BA
    F[0:3, 12:15] = R_prev @ J_bg[0:3, :]              # POS <- BG
    F[3:6, 12:15] = R_prev @ J_bg[3:6, :]              # VEL <- BG
    F[6:9, 12:15] = J_bg[6:9, :]                       # ORI <- BG

    M = np.hstack((F @ S_P, Q_sqrt * np.sqrt(dt) * q_scale))   # (23, 46)
    Mt = np.ascontiguousarray(M.T)                                # (46, 23)
    _, R_qr = np.linalg.qr(Mt)                                    # reduced: R_qr is (23, 23)
    S_P_new = np.ascontiguousarray(R_qr[0:23, 0:23].T)
    return p_new, v_new, R_new, F, S_P_new


# ---------------------------------------------------------------------------
# RTS backward recursion over packed state arrays
# x layout (20): p 0:3, v 3:6, ba 6:9, bg 9:12, bg_hub 12:15, g 15:18, td 18, sl 19
# error-state layout (23): POS 0:3, VEL 3:6, ORI 6:9, BA 9:12, BG 12:15, BG_HUB 15:18, GRAV 18:21, TD 21, SL 22
# ---------------------------------------------------------------------------

@njit(cache=True)
def rts_backward_chunk(C, x_s, R_s, x_pred, R_pred, pred_offset, k0, k1):
    """Apply smoother gains C[k - k0] for k = k1-1 down to k0, in place on (x_s, R_s).

    x_pred/R_pred hold the prediction used for step k at index k + pred_offset
    (pred_offset = 0 for an explicit pred_states list, 1 when states[k+1] is used)."""
    for k in range(k1 - 1, k0 - 1, -1):
        kp = k + pred_offset
        dx_next = np.zeros(23)
        lin = x_s[k + 1] - x_pred[kp]
        dx_next[0:6] = lin[0:6]
        dx_next[6:9] = nb_so3_log(R_pred[kp].T @ R_s[k + 1])
        dx_next[9:23] = lin[6:20]

        dx_k = C[k - k0] @ dx_next

        x_s[k, 0:6] += dx_k[0:6]
        R_s[k] = R_s[k] @ nb_so3_exp(dx_k[6:9])
        x_s[k, 6:20] += dx_k[9:23]


KERNEL_STATUS: dict[str, bool] = {}

# Module-level names rebound to their pure-Python twin when a kernel fails to compile at warmup.
_KERNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "so3": ("nb_skew", "nb_so3_exp", "nb_so3_log", "nb_right_jacobian_so3"),
    "preint_step": ("preint_step",),
    "eskf_predict": ("eskf_predict",),
    "anms_radii": ("anms_radii_nb",),
    "rts_backward_chunk": ("rts_backward_chunk",),
}


def _install_python_fallback(group: str) -> None:
    """Rebind every kernel in `group` to its `.py_func` (the undecorated Python body).

    Callers reference kernels through this module (`numba_accelerated.eskf_predict(...)`), so the
    rebinding takes effect for them. The Python twins are written in numpy-array style and run at
    roughly the pre-acceleration speed (measured: whole session ≈63 s with NUMBA_DISABLE_JIT=1)."""
    g = globals()
    for name in _KERNEL_GROUPS[group]:
        py = getattr(g[name], "py_func", None)
        if py is not None:
            g[name] = py
            logger.warning("numba kernel %s replaced by its pure-Python fallback", name)


def warmup_numba_jit() -> None:
    """Compile every JIT function with dummy arrays so the disk cache is populated at worker start.

    Each kernel is compiled independently; a failure is logged, recorded in KERNEL_STATUS, and the
    kernel is replaced by its pure-Python twin (spec §6: "verified at warmup, otherwise the fallback
    path is used"), so a numba/LAPACK binding problem degrades VIO to the pre-acceleration speed
    instead of failing every session open."""
    dummy_vec = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    _ = fast_vector_norm(dummy_vec)

    q1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    _ = fast_quaternion_multiply(q1, q2)

    p = np.zeros(3, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    a = np.array([0.0, 0.0, 9.81], dtype=np.float64)
    _ = fast_imu_step_integration(p, v, a, 0.01)

    eye3 = np.eye(3)
    z3 = np.zeros(3)
    phi = np.array([0.01, -0.02, 0.03])

    def _try(name, fn):
        try:
            fn()
            KERNEL_STATUS[name] = True
        except Exception as exc:  # numba TypingError / LAPACK binding problems
            KERNEL_STATUS[name] = False
            logger.warning("numba warmup failed for %s: %s", name, exc)
            _install_python_fallback(name)

    _try("so3", lambda: (nb_skew(phi), nb_so3_exp(phi), nb_so3_log(nb_so3_exp(phi)), nb_right_jacobian_so3(phi)))
    _try("preint_step", lambda: preint_step(
        eye3, z3, z3, np.zeros((9, 9)), np.zeros((9, 3)), np.zeros((9, 3)),
        a, z3, 0.005, z3, z3, 0.01, 0.001, z3, z3, False))
    _try("eskf_predict", lambda: eskf_predict(
        z3.copy(), z3.copy(), eye3, np.array([0.0, 0.0, -9.81]), np.eye(23) * 0.1,
        eye3, z3, z3, np.zeros((9, 3)), np.zeros((9, 3)), 0.005, np.eye(23) * 1e-3, 1.0))
    kps32 = np.array([[1.0, 2.0], [3.0, 5.0], [8.0, 1.0]], dtype=np.float32)
    resp32 = np.array([9.0, 7.0, 5.0], dtype=np.float32)
    _try("anms_radii", lambda: (
        anms_radii_nb(kps32, resp32, (0.9 * resp32).astype(np.float32)),
        anms_radii_nb(kps32.astype(np.float64), resp32.astype(np.float64), 0.9 * resp32.astype(np.float64)),
    ))
    _try("rts_backward_chunk", lambda: rts_backward_chunk(
        np.zeros((1, 23, 23)), np.zeros((2, 20)), np.stack([eye3, eye3]),
        np.zeros((1, 20)), np.stack([eye3]), 0, 0, 1))


