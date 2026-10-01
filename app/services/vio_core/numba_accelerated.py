import os
import numpy as np

# Configure Numba disk cache directory to persistent /tmp/numba_cache
NUMBA_CACHE_DIR = "/tmp/numba_cache"
os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR
os.makedirs(NUMBA_CACHE_DIR, exist_ok=True)

try:
    from numba import njit
except ImportError:
    # Fallback dummy decorator if numba is unavailable
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

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

def warmup_numba_jit() -> None:
    """Warmup trigger executing all JIT functions with dummy arrays to populate disk cache."""
    dummy_vec = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    _ = fast_vector_norm(dummy_vec)

    q1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    _ = fast_quaternion_multiply(q1, q2)

    p = np.zeros(3, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    a = np.array([0.0, 0.0, 9.81], dtype=np.float64)
    _ = fast_imu_step_integration(p, v, a, 0.01)
