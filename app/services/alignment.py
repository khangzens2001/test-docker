import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, correlate as scipy_correlate

WAHBA_OMEGA_MIN_RAD_S = 0.3
WAHBA_MAX_PAIR_DT_NS = 10_000_000
WAHBA_MIN_PAIRS = 100
WAHBA_MAX_RMS_DEG = 10.0
WAHBA_MIN_SV_RATIO = 0.05


def calculate_robust_jitter(delta_t: np.ndarray) -> float:
    """Calculate the Median Absolute Deviation (MAD) of the sampling intervals."""
    # Convert input to float64 array to handle potential NaNs and float math
    delta_t_arr = np.asarray(delta_t, dtype=np.float64)
    valid_deltas = delta_t_arr[~np.isnan(delta_t_arr)]
    if len(valid_deltas) == 0:
        return 0.0
    nominal_interval = np.median(valid_deltas)
    mad = np.median(np.abs(valid_deltas - nominal_interval))
    return float(1.4826 * mad)

def fit_clock_alignment_model(t_device_rel: np.ndarray, t_mcu_rel: np.ndarray, robust_jitter: float) -> tuple[float, float]:
    """Fit robust linear relation t_device_rel = s * t_mcu_rel + c_rel.
    
    Inputs must be relative timelines (absolute nanosecond epochs cast to float64 truncate significand precision).
    Timestamps are internally scaled to seconds to prevent numerical ill-conditioning in least_squares.
    """
    t_device_rel_arr = np.asarray(t_device_rel, dtype=np.float64)
    t_mcu_rel_arr = np.asarray(t_mcu_rel, dtype=np.float64)
    
    valid = ~np.isnan(t_device_rel_arr) & ~np.isnan(t_mcu_rel_arr)
    y = t_device_rel_arr[valid]
    x = t_mcu_rel_arr[valid]
    
    if len(y) < 2:
        return 1.0, 0.0
        
    # Scale to seconds to avoid least_squares conditioning failure
    y_sec = y / 1e9
    x_sec = x / 1e9
    jitter_sec = robust_jitter / 1e9
    
    # 2D Analytical Jacobian for speed and accuracy
    def residuals(params, x_val, y_val):
        s, c = params
        return (s * x_val + c) - y_val

    def jacobian(params, x_val, y_val):
        jac = np.zeros((len(x_val), 2))
        jac[:, 0] = x_val
        jac[:, 1] = 1.0
        return jac

    # Scale successive differences variance to standard deviation of timestamp noise
    sigma_t = jitter_sec / np.sqrt(2.0)
    f_scale = max(1.5 * sigma_t, 1e-5)  # 10 microsecond safety floor in seconds

    c0 = float(np.median(y_sec - x_sec))

    # Pass 1: Huber least-squares to fit the drift rate (slope) robustly on all valid data
    res = least_squares(
        residuals,
        x0=[1.0, c0],
        args=(x_sec, y_sec),
        jac=jacobian,
        loss="huber",
        f_scale=f_scale,
        x_scale="jac",  # Stabilizes optimization against parameter scale mismatches
    )
    s_est, c_est_sec = res.x

    # Pass 1.5: Outlier rejection and linear fit refinement on clean data
    r = (s_est * x_sec + c_est_sec) - y_sec
    clean = np.abs(r) < 3.0 * f_scale
    if np.sum(clean) >= 2:
        res_clean = least_squares(
            residuals,
            x0=[s_est, c_est_sec],
            args=(x_sec[clean], y_sec[clean]),
            jac=jacobian,
            loss="linear",
        )
        s_est, _ = res_clean.x

    # Pass 2: Estimate offset c using the 10th percentile of the residuals
    # This captures the minimum-latency envelope without introducing symmetric truncation bias
    residuals_raw = y_sec - s_est * x_sec
    c_envelope_sec = np.percentile(residuals_raw, 10)

    # Plausibility check: drift slope within 1000 ppm
    if abs(s_est - 1.0) < 0.001:
        return float(s_est), float(c_envelope_sec * 1e9)
    else:
        # Fallback to nominal slope and 10th percentile offset to capture minimum latency
        s_fallback = 1.0
        c_fallback = float(np.percentile(y - x, 10))
        return s_fallback, c_fallback

def apply_butterworth_lowpass(data: np.ndarray, fs: float, timeline: np.ndarray, cutoff_ratio: float = 0.4) -> np.ndarray:
    """Apply 2nd-order zero-phase Butterworth low-pass filter forward-backward.
    
    Supports 1D and 2D arrays. Safe copy is returned to avoid side effects on original arrays.
    """
    original_ndim = data.ndim
    filtered_data = np.array(data, dtype=np.float64, copy=True)
    if original_ndim == 1:
        filtered_data = filtered_data[:, np.newaxis]
        
    nans = np.isnan(filtered_data)
    if np.any(nans):
        for col in range(filtered_data.shape[1]):
            col_data = filtered_data[:, col]
            col_nans = nans[:, col]
            if np.all(col_nans):
                filtered_data[:, col] = 0.0
                continue
            x = timeline[~col_nans]
            y = col_data[~col_nans]
            filtered_data[:, col] = np.interp(timeline, x, y)

    # Guard: filtfilt requires at least 9 samples for a 2nd-order filter to prevent crashes
    if len(filtered_data) < 9:
        if original_ndim == 1:
            return filtered_data.squeeze(axis=1)
        return filtered_data
        
    target_nyq = 50.0  # Resampling target grid is 100 Hz, so Nyquist limit is 50 Hz
    cutoff = min(cutoff_ratio * fs, 0.8 * target_nyq)  # Anti-aliasing cutoff capped relative to target Nyquist
    nyq = 0.5 * fs
    # SRE: Enforce 5% safety margin relative to Nyquist frequency to avoid scipy filter design crashes
    if nyq <= 0.0 or cutoff <= 0.0 or cutoff >= 0.95 * nyq:
        if original_ndim == 1:
            return filtered_data.squeeze(axis=1)
        return filtered_data
        
    low = cutoff / nyq
    b, a = butter(2, low, btype="low")
    res = filtfilt(b, a, filtered_data, axis=0)
    if original_ndim == 1:
        return res.squeeze(axis=1)
    return res

def pchip_resample(timeline_mcu: np.ndarray, data: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    """Resample each column of data sequentially using PCHIP Interpolation."""
    original_ndim = data.ndim
    timeline_mcu_arr = np.asarray(timeline_mcu, dtype=np.float64)
    data_arr = np.array(data, dtype=np.float64, copy=True)
    target_grid_arr = np.asarray(target_grid, dtype=np.float64)
    
    if original_ndim == 1:
        data_arr = data_arr[:, np.newaxis]
        
    if len(timeline_mcu_arr) < 2 or len(target_grid_arr) == 0:
        res = np.zeros((len(target_grid_arr), data_arr.shape[1]))
        return res.squeeze(axis=1) if original_ndim == 1 else res

    # SRE: Fill NaNs to prevent PchipInterpolator crash or NaN propagation
    nans = np.isnan(data_arr)
    if np.any(nans):
        for col in range(data_arr.shape[1]):
            col_data = data_arr[:, col]
            col_nans = nans[:, col]
            if np.all(col_nans):
                data_arr[:, col] = 0.0
                continue
            x = timeline_mcu_arr[~col_nans]
            y = col_data[~col_nans]
            data_arr[:, col] = np.interp(timeline_mcu_arr, x, y)

    # Deduplicate non-strictly increasing timestamps by averaging duplicate points
    if not np.all(np.diff(timeline_mcu_arr) > 0):
        unique_t, indices, counts = np.unique(timeline_mcu_arr, return_inverse=True, return_counts=True)
        if len(unique_t) < 2:
            raise ValueError("Not enough unique timestamps to resample.")
            
        deduped_data = np.zeros((len(unique_t), data_arr.shape[1]))
        for col in range(data_arr.shape[1]):
            deduped_data[:, col] = np.bincount(indices, weights=data_arr[:, col]) / counts
        timeline_mcu_arr = unique_t
        data_arr = deduped_data

    # Guard: clamp target grid to prevent extrapolation crashes
    t_min, t_max = timeline_mcu_arr[0], timeline_mcu_arr[-1]
    clamped_grid = np.clip(target_grid_arr, t_min, t_max)

    # Performance: Vectorize PchipInterpolator to run across data columns axis simultaneously in C code
    interp = PchipInterpolator(timeline_mcu_arr, data_arr, axis=0)
    res = interp(clamped_grid)
    return res.squeeze(axis=1) if original_ndim == 1 else res

def align_gyroscope_signals(t_phone: np.ndarray, w_phone: np.ndarray, t_hub: np.ndarray, w_hub: np.ndarray) -> tuple[float, float]:
    if len(t_phone) < 2 or len(t_hub) < 2:
        return 1.0, 0.0
        
    # Ensure monotonic timestamp ordering
    sort_phone = np.argsort(t_phone)
    t_phone = t_phone[sort_phone]
    w_phone = w_phone[sort_phone]
    
    sort_hub = np.argsort(t_hub)
    t_hub = t_hub[sort_hub]
    w_hub = w_hub[sort_hub]
    
    # Auto-detect nanoseconds using median delta interval (> 1ms)
    if np.median(np.diff(t_phone)) < 1e6:
        t_phone = t_phone * 1e9
    if np.median(np.diff(t_hub)) < 1e6:
        t_hub = t_hub * 1e9
        
    # Lowpass filter signals to prevent high-frequency noise amplification in gradient magnitude
    fs_phone = 1.0 / float(np.median(np.diff(t_phone / 1e9))) if len(t_phone) > 1 else 100.0
    fs_hub = 1.0 / float(np.median(np.diff(t_hub / 1e9))) if len(t_hub) > 1 else 100.0
    
    t_phone_sec = t_phone / 1e9
    t_hub_sec = t_hub / 1e9

    w_phone_filt = apply_butterworth_lowpass(w_phone, fs=fs_phone, timeline=t_phone_sec, cutoff_ratio=0.1)
    w_hub_filt = apply_butterworth_lowpass(w_hub, fs=fs_hub, timeline=t_hub_sec, cutoff_ratio=0.1)
    
    # Compute derivative magnitudes using central gradient to prevent initial sample diff artifacts
    diff_w_phone = np.gradient(w_phone_filt, axis=0)
    diff_w_hub = np.gradient(w_hub_filt, axis=0)
    mag_phone = np.linalg.norm(diff_w_phone, axis=1)
    mag_hub = np.linalg.norm(diff_w_hub, axis=1)
    
    t_phone_zero = t_phone_sec - t_phone_sec[0]
    t_hub_zero = t_hub_sec - t_hub_sec[0]
    
    duration = max(t_phone_zero[-1], t_hub_zero[-1])
    if duration <= 0 or duration > 3600 or np.isnan(duration) or np.isinf(duration):
        raise ValueError(f"Invalid signal duration: {duration} seconds.")
        
    if np.std(mag_phone) < 1e-6 or np.std(mag_hub) < 1e-6:
        return 1.0, float((t_hub_sec[0] - t_phone_sec[0]) * 1e9)
        
    t_phone_coarse = np.arange(0, t_phone_zero[-1], 1.0/15.0)
    t_hub_coarse = np.arange(0, t_hub_zero[-1], 1.0/15.0)
    
    if len(t_phone_coarse) < 15 or len(t_hub_coarse) < 15:
        raise ValueError("Insufficient signal duration for alignment (< 1.0s).")
        
    mag_phone_coarse = np.interp(t_phone_coarse, t_phone_zero, mag_phone)
    mag_hub_coarse = np.interp(t_hub_coarse, t_hub_zero, mag_hub)
    
    corr = scipy_correlate(mag_phone_coarse - np.mean(mag_phone_coarse), mag_hub_coarse - np.mean(mag_hub_coarse), mode="full", method="fft")
    
    # Correct full correlation lag grid orientation
    lags = np.arange(-len(mag_hub_coarse) + 1, len(mag_phone_coarse))
    coarse_lag_grid = lags[np.argmax(corr)] / 15.0
    coarse_offset = (t_hub_sec[0] - t_phone_sec[0]) - coarse_lag_grid
    
    t_phone_aligned = t_phone_sec + coarse_offset
    t_start = max(t_phone_aligned[0], t_hub_sec[0])
    t_end = min(t_phone_aligned[-1], t_hub_sec[-1])
    
    if (t_end - t_start) < 1.0:
        return 1.0, float(coarse_offset * 1e9)
        
    t_fine = np.arange(t_start, t_end, 1.0/100.0)
    mag_phone_fine = np.interp(t_fine - coarse_offset, t_phone_sec, mag_phone)
    mag_hub_fine = np.interp(t_fine, t_hub_sec, mag_hub)
    
    corr_fine = scipy_correlate(mag_phone_fine - np.mean(mag_phone_fine), mag_hub_fine - np.mean(mag_hub_fine), mode="full", method="fft")
    lags_fine = np.arange(-len(mag_hub_fine) + 1, len(mag_phone_fine))
    
    center_idx = len(mag_hub_fine) - 1
    search_radius = 50
    start_idx = max(0, center_idx - search_radius)
    end_idx = min(len(corr_fine), center_idx + search_radius + 1)
    
    sub_corr = corr_fine[start_idx:end_idx]
    sub_lags = lags_fine[start_idx:end_idx]
    
    peak_sub_idx = np.argmax(sub_corr)
    peak_idx = start_idx + peak_sub_idx
    
    if 0 < peak_idx < len(corr_fine) - 1:
        y1, y2, y3 = corr_fine[peak_idx - 1], corr_fine[peak_idx], corr_fine[peak_idx + 1]
        denom = y1 - 2*y2 + y3
        if denom < -1e-8:
            fine_adjustment = -0.5 * (y3 - y1) / denom
        else:
            fine_adjustment = 0.0
    else:
        fine_adjustment = 0.0
        
    offset_sec = coarse_offset - (lags_fine[peak_idx] + fine_adjustment) / 100.0
    
    # Segment signals into sliding windows to fit clock drift slope s and intercept c via linear regression
    chunk_size_sec = 4.0
    step_sec = 1.0
    fs_grid = 1000.0
    
    if duration >= chunk_size_sec:
        num_chunks = int((duration - chunk_size_sec) / step_sec) + 1
        t_chunks = []
        offsets_chunks = []
        for k in range(num_chunks):
            t0 = t_phone_sec[0] + k * step_sec
            t1 = t0 + chunk_size_sec
            mask_phone = (t_phone_sec >= t0) & (t_phone_sec < t1)
            if np.sum(mask_phone) > 10 and np.std(mag_phone[mask_phone]) > 1e-5:
                # Interpolate chunk signals onto fine 1kHz grid using coarse_offset
                t_grid = np.arange(0, chunk_size_sec, 1.0 / fs_grid)
                chunk_p = np.interp(t0 + t_grid, t_phone_sec, mag_phone)
                chunk_h = np.interp(t0 + coarse_offset + t_grid, t_hub_sec, mag_hub)
                
                chunk_p_norm = chunk_p - np.mean(chunk_p)
                chunk_h_norm = chunk_h - np.mean(chunk_h)
                
                if np.std(chunk_p_norm) > 1e-5 and np.std(chunk_h_norm) > 1e-5:
                    corr_k = scipy_correlate(chunk_p_norm, chunk_h_norm, mode="full", method="fft")
                    lags_k = np.arange(-len(chunk_h_norm) + 1, len(chunk_p_norm))
                    center_k = len(chunk_h_norm) - 1
                    
                    overlap = len(chunk_p_norm) - np.abs(lags_k)
                    corr_norm = corr_k / overlap
                    
                    search_r = int(0.3 * fs_grid)
                    sub_corr_k = corr_norm[center_k - search_r : center_k + search_r + 1]
                    peak_sub_k = np.argmax(sub_corr_k)
                    peak_k = center_k - search_r + peak_sub_k
                    
                    if 0 < peak_k < len(corr_norm) - 1:
                        y1, y2, y3 = corr_norm[peak_k - 1], corr_norm[peak_k], corr_norm[peak_k + 1]
                        denom = y1 - 2*y2 + y3
                        fine_adj_k = -0.5 * (y3 - y1) / denom if denom < -1e-8 else 0.0
                    else:
                        fine_adj_k = 0.0
                        
                    local_lag = (lags_k[peak_k] + fine_adj_k) / fs_grid
                    local_offset = coarse_offset - local_lag
                    
                    t_chunks.append(float(np.mean(t_phone_sec[mask_phone])))
                    offsets_chunks.append(float(local_offset))
                    
        if len(t_chunks) >= 2:
            poly = np.polyfit(t_chunks, offsets_chunks, 1)
            s = float(1.0 + poly[0])
            c = float(poly[1] * 1e9)
            return s, c
            
    return 1.0, float(offset_sec * 1e9)


def _wahba_result(accepted, R, reason, n_pairs, residual_rms_deg, sv_ratio):
    return {
        "accepted": bool(accepted),
        "R": None if R is None else np.asarray(R, dtype=np.float64),
        "reason": reason,
        "n_pairs": int(n_pairs),
        "residual_rms_deg": None if residual_rms_deg is None else float(residual_rms_deg),
        "sv_ratio": None if sv_ratio is None else float(sv_ratio),
    }


def estimate_R_phone_hub(
    phone_ts_ns: np.ndarray,
    phone_gyro: np.ndarray,
    hub_ts_ns: np.ndarray,
    hub_gyro: np.ndarray,
) -> dict:
    phone_ts_arr = np.asarray(phone_ts_ns).reshape(-1)
    hub_ts_arr = np.asarray(hub_ts_ns).reshape(-1)
    phone_g = np.asarray(phone_gyro, dtype=np.float64).reshape(-1, 3)
    hub_g = np.asarray(hub_gyro, dtype=np.float64).reshape(-1, 3)
    if phone_ts_arr.size != phone_g.shape[0] or hub_ts_arr.size != hub_g.shape[0]:
        return _wahba_result(False, None, "missing_stream", 0, None, None)

    if np.issubdtype(phone_ts_arr.dtype, np.floating):
        p_ts_ok = np.isfinite(phone_ts_arr)
    elif np.issubdtype(phone_ts_arr.dtype, np.integer):
        p_ts_ok = np.ones(phone_ts_arr.shape, dtype=bool)
    else:
        p_ts_ok = np.isfinite(phone_ts_arr.astype(np.float64))

    if np.issubdtype(hub_ts_arr.dtype, np.floating):
        h_ts_ok = np.isfinite(hub_ts_arr)
    elif np.issubdtype(hub_ts_arr.dtype, np.integer):
        h_ts_ok = np.ones(hub_ts_arr.shape, dtype=bool)
    else:
        h_ts_ok = np.isfinite(hub_ts_arr.astype(np.float64))

    p_ok = p_ts_ok & np.all(np.isfinite(phone_g), axis=1)
    h_ok = h_ts_ok & np.all(np.isfinite(hub_g), axis=1)
    phone_ts = phone_ts_arr[p_ok].astype(np.int64)
    phone_g = phone_g[p_ok]
    hub_ts = hub_ts_arr[h_ok].astype(np.int64)
    hub_g = hub_g[h_ok]
    if phone_ts.size < 2 or hub_ts.size < 2:
        return _wahba_result(False, None, "missing_stream", 0, None, None)

    order = np.argsort(phone_ts, kind="mergesort")
    phone_ts = phone_ts[order]
    phone_g = phone_g[order]
    uniq, inv = np.unique(phone_ts, return_inverse=True)
    if uniq.size != phone_ts.size:
        sums = np.zeros((uniq.size, 3), dtype=np.float64)
        counts = np.zeros((uniq.size, 1), dtype=np.float64)
        np.add.at(sums, inv, phone_g)
        np.add.at(counts, inv, 1.0)
        phone_ts = uniq
        phone_g = sums / counts
    if phone_ts.size < 2:
        return _wahba_result(False, None, "missing_stream", 0, None, None)

    t_first = int(phone_ts[0])
    t_last = int(phone_ts[-1])
    kept_phone = []
    kept_hub = []
    n_phone = int(phone_ts.size)
    for k in range(int(hub_ts.size)):
        t = int(hub_ts[k])
        if t < t_first or t > t_last:
            continue
        i = int(np.searchsorted(phone_ts, t, side="left"))
        if i >= n_phone:
            continue
        if i == 0 or int(phone_ts[i]) == t:
            w_p = phone_g[i]
        elif 0 < i < n_phone:
            t0 = int(phone_ts[i - 1])
            t1 = int(phone_ts[i])
            if min(t - t0, t1 - t) > WAHBA_MAX_PAIR_DT_NS:
                continue
            if t1 > t0:
                alpha = (t - t0) / (t1 - t0)
                w_p = (1.0 - alpha) * phone_g[i - 1] + alpha * phone_g[i]
            else:
                w_p = phone_g[i]
        else:
            continue
        w_h = hub_g[k]
        if float(np.linalg.norm(w_p)) >= WAHBA_OMEGA_MIN_RAD_S and float(np.linalg.norm(w_h)) >= WAHBA_OMEGA_MIN_RAD_S:
            kept_phone.append(np.asarray(w_p, dtype=np.float64))
            kept_hub.append(np.asarray(w_h, dtype=np.float64))

    n_pairs = len(kept_phone)
    if n_pairs < WAHBA_MIN_PAIRS:
        return _wahba_result(False, None, "too_few_pairs", n_pairs, None, None)

    w_p = np.stack(kept_phone, axis=0)
    w_h = np.stack(kept_hub, axis=0)
    u_p = w_p / np.linalg.norm(w_p, axis=1, keepdims=True)
    u_h = w_h / np.linalg.norm(w_h, axis=1, keepdims=True)
    H = u_h.T @ u_p
    U, S, Vt = np.linalg.svd(H)
    if (not np.all(np.isfinite(S))) or float(S[0]) <= 0.0:
        return _wahba_result(False, None, "too_few_pairs", n_pairs, None, None)
    det_H = float(np.linalg.det(Vt.T @ U.T))
    if abs(det_H) < 1e-6:
        return _wahba_result(False, None, "too_few_pairs", n_pairs, None, None)
    d = 1.0 if det_H > 0.0 else -1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    sv_ratio = float(S[1] / S[0])
    if sv_ratio < WAHBA_MIN_SV_RATIO:
        return _wahba_result(False, None, "insufficient_excitation", n_pairs, None, sv_ratio)

    a = (R @ w_h.T).T
    b = w_p
    sin_th = np.linalg.norm(np.cross(a, b), axis=1)
    cos_th = np.sum(a * b, axis=1)
    theta_deg = np.degrees(np.arctan2(sin_th, cos_th))
    residual_rms_deg = float(np.sqrt(np.mean(theta_deg ** 2)))
    if residual_rms_deg > WAHBA_MAX_RMS_DEG:
        return _wahba_result(False, None, "residual_rms", n_pairs, residual_rms_deg, sv_ratio)
    return _wahba_result(True, R, None, n_pairs, residual_rms_deg, sv_ratio)

