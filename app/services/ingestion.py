import io
import os
import re

import numpy as np
import pandas as pd

from app.core.exceptions import SensorDataIngestionError
from app.services.alignment import calculate_robust_jitter, fit_clock_alignment_model

IMU_CANONICAL_MAPPINGS: dict[str, list[str]] = {
    "timestamp_ns": [
        "timestamp_ns",
        "timestamp",
        "#timestamp",
        "time",
        "t",
        "mcu_timestamp_ns",
        "device_timestamp_ns",
        "header.stamp",
        "header_stamp",
    ],
    "accel_x": [
        "accel_x",
        "ax",
        "a_x",
        "linear_acceleration.x",
        "accelerometer_x",
        "acc_x",
    ],
    "accel_y": [
        "accel_y",
        "ay",
        "a_y",
        "linear_acceleration.y",
        "accelerometer_y",
        "acc_y",
    ],
    "accel_z": [
        "accel_z",
        "az",
        "a_z",
        "linear_acceleration.z",
        "accelerometer_z",
        "acc_z",
    ],
    "gyro_x": [
        "gyro_x",
        "gx",
        "g_x",
        "angular_velocity.x",
        "gyroscope_x",
        "gyr_x",
        "rotation_rate_x",
    ],
    "gyro_y": [
        "gyro_y",
        "gy",
        "g_y",
        "angular_velocity.y",
        "gyroscope_y",
        "gyr_y",
        "rotation_rate_y",
    ],
    "gyro_z": [
        "gyro_z",
        "gz",
        "g_z",
        "angular_velocity.z",
        "gyroscope_z",
        "gyr_z",
        "rotation_rate_z",
    ],
}

PHONE_IMU_MAPPINGS: dict[str, list[str]] = {
    "device_timestamp_ns": ["timestamp_nanos", "timestamp_ns", "timestamp", "time"],
    "ax": ["ax", "a_x", "acc_x", "accel_x"],
    "ay": ["ay", "a_y", "acc_y", "accel_y"],
    "az": ["az", "a_z", "acc_z", "accel_z"],
    "gx": ["gx", "g_x", "gyro_x"],
    "gy": ["gy", "g_y", "gyro_y"],
    "gz": ["gz", "g_z", "gyro_z"],
}


def resolve_imu_scale(manifest: dict | None) -> str:
    if not isinstance(manifest, dict):
        return "mag_heuristic"
    streams = manifest.get("streams") if isinstance(manifest.get("streams"), dict) else {}
    imu = streams.get("imu") if isinstance(streams.get("imu"), dict) else {}
    unit = imu.get("unit_status")
    ver = manifest.get("schema_version")
    if unit == "si_bno08x_div_1000":
        return "identity"
    if unit == "raw_counts_unconfirmed":
        return "div_1000"
    ver_s = str(ver) if ver is not None else None
    if ver_s in {"1.3"}:
        return "identity"
    if ver_s in {"1.1", "1.2"}:
        return "div_1000"
    if ver_s in {"1", "1.0"}:
        return "identity"
    if ver_s is None:
        return "mag_heuristic"
    return "mag_heuristic"


def parse_telemetry_csv(
    file_stream: io.BytesIO,
    column_mappings: dict[str, list[str]],
    *,
    imu_scale: str = "mag_heuristic",
    optional_columns: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Read CSV, strip BOM, trim whitespace, map case-insensitive aliases using PyArrow."""
    file_stream.seek(0)
    row_count = 0
    first_line_bytes = file_stream.readline()
    if first_line_bytes:
        row_count += 1
        if len(first_line_bytes) > 65536:
            raise ValueError("CSV line exceeds 65536 byte limit.")

        first_line_cleaned = first_line_bytes
        if first_line_cleaned.startswith(b"\xef\xbb\xbf"):
            first_line_cleaned = first_line_cleaned[3:]

        first_line_str = first_line_cleaned.decode("utf-8", errors="replace").rstrip("\r\n")
        header_cols = [c.strip().lower() for c in first_line_str.split(",")]
        if len(header_cols) > 150:
            raise ValueError("CSV exceeds maximum columns limit of 150.")

    for line in file_stream:
        row_count += 1
        if row_count > 120001:
            raise ValueError("CSV exceeds maximum rows limit of 120,000.")
        if len(line) > 65536:
            raise ValueError("CSV line exceeds 65536 byte limit.")

    file_stream.seek(0)
    content = file_stream.read()

    # Strip UTF-8 BOM if present
    if content.startswith(b"\xef\xbb\xbf"):
        content = content[3:]

    df = pd.read_csv(
        io.BytesIO(content),
        engine="pyarrow",
        dtype_backend="pyarrow",
    )

    # Security: check payload limits to prevent OOM / resource exhaustion crashes
    if len(df.columns) > 150:
        raise ValueError("CSV exceeds maximum columns limit of 150.")
    if len(df) > 120000:
        raise ValueError("CSV exceeds maximum rows limit of 120,000.")

    # Convert columns to string lower
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Detect 16-element pose matrix columns to automatically reconstruct translations & quaternions
    matrix_cols = [f"transform_{i}" for i in range(16)]
    alt_matrix_cols = [f"m{i}{j}" for i in range(4) for j in range(4)]
    has_matrix = all(col in df.columns for col in matrix_cols)
    has_alt_matrix = all(col in df.columns for col in alt_matrix_cols)
    if has_matrix or has_alt_matrix:
        cols_to_use = matrix_cols if has_matrix else alt_matrix_cols
        # Use import here to avoid circular dependencies
        from app.services.calibration import parse_4x4_pose_matrix

        pose_data = df[cols_to_use].to_numpy(dtype=float)
        translations = []
        quaternions = []
        for row_idx in range(len(pose_data)):
            tx, ty, tz, q = parse_4x4_pose_matrix(pose_data[row_idx])
            translations.append([tx, ty, tz])
            quaternions.append(q)
        translations = np.array(translations)
        quaternions = np.array(quaternions)
        df["x"] = translations[:, 0]
        df["y"] = translations[:, 1]
        df["z"] = translations[:, 2]
        df["qx"] = quaternions[:, 0]
        df["qy"] = quaternions[:, 1]
        df["qz"] = quaternions[:, 2]
        df["qw"] = quaternions[:, 3]

    mapped_data = {}
    optional = optional_columns or frozenset()

    for target_col, aliases in column_mappings.items():
        if target_col.endswith("_{0..63}"):
            base_name = target_col.replace("_{0..63}", "").lower()
            for i in range(64):
                col_key = f"{base_name}_{i}"
                possible_aliases = [
                    col_key,
                    f"{base_name}_{i:02d}",
                    f"{base_name[0]}{i}",
                    f"{base_name[0]}{i:02d}",
                    f"{base_name}{i}",
                    f"{base_name}{i:02d}",
                ]
                found_col = next((a for a in possible_aliases if a in df.columns), None)
                if found_col is not None:
                    mapped_data[col_key] = df[found_col]
                else:
                    raise ValueError(f"Missing column required for pattern: {col_key}")
            continue

        matched_col = None
        for alias in [target_col] + aliases:
            alias_lower = alias.lower()
            if alias_lower in df.columns:
                matched_col = alias_lower
                break

        if matched_col is None:
            if target_col in optional:
                continue
            raise ValueError(f"Required column target mapping not found: {target_col}")
        else:
            mapped_data[target_col] = df[matched_col]

    out_df = pd.DataFrame(mapped_data)

    # Drop any rows where mcu_timestamp_ns, device_timestamp_ns, or timestamp_ns columns contain NaN values
    ts_cols = [
        col
        for col in ["mcu_timestamp_ns", "device_timestamp_ns", "timestamp_ns"]
        if col in out_df.columns
    ]
    if ts_cols:
        out_df = out_df.dropna(subset=ts_cols)

    accel_pairs = [
        (["accel_x", "accel_y", "accel_z"], ["gyro_x", "gyro_y", "gyro_z"]),
        (["ax", "ay", "az"], ["gx", "gy", "gz"]),
    ]
    if imu_scale not in {"identity", "div_1000", "mag_heuristic"}:
        raise ValueError(f"Unknown imu_scale: {imu_scale}")
    if imu_scale != "identity":
        for accel_cols, gyro_cols in accel_pairs:
            if not all(c in out_df.columns for c in accel_cols):
                continue
            if imu_scale == "div_1000":
                for c in accel_cols:
                    out_df[c] = out_df[c].astype(float) / 1000.0
                for c in gyro_cols:
                    if c in out_df.columns:
                        out_df[c] = out_df[c].astype(float) / 1000.0
            elif imu_scale == "mag_heuristic":
                accel_data = out_df[accel_cols].astype(float).to_numpy()
                accel_mag = np.linalg.norm(accel_data, axis=1)
                if (
                    len(accel_mag) > 0
                    and np.any(~np.isnan(accel_mag))
                    and np.nanmedian(accel_mag) > 500.0
                ):
                    for c in accel_cols:
                        out_df[c] = (out_df[c].astype(float) / 1000.0) * 9.80665
                    for c in gyro_cols:
                        if c in out_df.columns:
                            out_df[c] = (out_df[c].astype(float) / 1000.0) * (np.pi / 180.0)
            break

    # Enforce float finiteness for telemetry metrics
    for col in out_df.columns:
        if pd.api.types.is_float_dtype(out_df[col]):
            vals = out_df[col].to_numpy(dtype=float, na_value=np.nan)
            if not np.all(np.isfinite(vals[~np.isnan(vals)])):
                raise ValueError(f"Non-finite float values detected in column: {col}")

    return out_df


def parse_phone_imu(session_dir: str) -> pd.DataFrame:
    phone_imu_path = os.path.join(session_dir, "phone_imu.csv")
    if not os.path.exists(phone_imu_path):
        raise SensorDataIngestionError(
            f"Required IMU file 'phone_imu.csv' missing in session directory: {session_dir}"
        )
    try:
        with open(phone_imu_path, "rb") as f:
            stream = io.BytesIO(f.read())
        return parse_telemetry_csv(
            stream, PHONE_IMU_MAPPINGS, imu_scale="identity"
        )
    except SensorDataIngestionError:
        raise
    except Exception as exc:
        raise SensorDataIngestionError(f"Failed to ingest IMU telemetry: {exc}") from exc


def detect_time_unit(timestamps: pd.Series, is_mcu: bool) -> float:
    """Return scaling factor to multiply by to yield nanoseconds."""
    t_arr = timestamps.dropna().to_numpy(dtype=np.float64)
    if len(t_arr) < 2:
        return 1.0

    if not is_mcu:
        deltas = np.diff(t_arr)
        dt_median = np.median(deltas) if len(deltas) > 0 else 0.0
        # If delta between consecutive samples is >= 2,000,000 ticks (> 2 ms),
        # this is unambiguously a nanosecond stream (e.g. Android monotonic uptime
        # reaching 14-28 days or Unix nanoseconds). Any other unit (s, ms, us)
        # with dt >= 2e6 would mean > 2 seconds between samples.
        if dt_median >= 2_000_000:
            return 1.0

        t_mean = np.mean(t_arr)
        if 1.2e9 <= t_mean < 2.5e9:
            return 1e9
        elif 1.2e12 <= t_mean < 2.5e12:
            return 1e6
        elif 1.2e15 <= t_mean < 2.5e15:
            return 1e3
        else:
            return 1.0
    else:
        deltas = np.diff(t_arr)
        dt_k = np.median(deltas)
        if dt_k < 0.5:
            return 1e9
        elif 0.5 <= dt_k < 500.0:
            return 1e6
        elif 500.0 <= dt_k < 500000.0:
            return 1e3
        else:
            return 1.0


def reconstruct_mcu_rollover(timestamps: list[int] | np.ndarray) -> np.ndarray:
    """Unwrap 32-bit raw timer rollovers based on 2^32 bounds BEFORE unit scaling is applied."""
    ts_arr = np.array(timestamps, dtype=np.float64)
    if len(ts_arr) < 2:
        return np.nan_to_num(ts_arr).astype(np.int64)

    valid_mask = ~np.isnan(ts_arr) & np.isfinite(ts_arr)
    if not np.any(valid_mask):
        return np.zeros_like(ts_arr, dtype=np.int64)

    valid_idx = np.flatnonzero(valid_mask)
    valid_ts = ts_arr[valid_mask].astype(np.int64)

    if len(valid_ts) >= 2:
        diffs = np.diff(valid_ts)
        rollovers = diffs < -2147483648
        carry = np.cumsum(rollovers) * 4294967296
        unwrapped_valid = np.zeros_like(valid_ts, dtype=np.int64)
        unwrapped_valid[0] = valid_ts[0]
        unwrapped_valid[1:] = valid_ts[1:] + carry
    else:
        unwrapped_valid = valid_ts

    unwrapped_full = np.zeros_like(ts_arr, dtype=np.float64)
    unwrapped_full[valid_mask] = unwrapped_valid.astype(np.float64)

    nan_mask = ~valid_mask
    if np.any(nan_mask):
        unwrapped_full[nan_mask] = np.interp(
            np.flatnonzero(nan_mask), valid_idx, unwrapped_valid.astype(np.float64)
        )

    return unwrapped_full.astype(np.int64)


def parse_odometry_csv(file_stream: io.BytesIO) -> pd.DataFrame:
    mappings = {
        "timestamp": ["timestamp"],
        "frame": ["frame"],
        "x": ["x", "pos_x"],
        "y": ["y", "pos_y"],
        "z": ["z", "pos_z"],
        "qx": ["qx", "quat_x"],
        "qy": ["qy", "quat_y"],
        "qz": ["qz", "quat_z"],
        "qw": ["qw", "quat_w"],
    }
    df = parse_telemetry_csv(file_stream, mappings, imu_scale="identity")
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    if np.nanmedian(ts) < 1e15:
        device = np.rint(ts * 1e9).astype(np.int64)
    else:
        device = np.rint(ts).astype(np.int64)
    out = pd.DataFrame({
        "device_timestamp_ns": device,
        "frame": df["frame"].to_numpy(),
        "x": df["x"].to_numpy(dtype=np.float64),
        "y": df["y"].to_numpy(dtype=np.float64),
        "z": df["z"].to_numpy(dtype=np.float64),
        "qx": df["qx"].to_numpy(dtype=np.float64),
        "qy": df["qy"].to_numpy(dtype=np.float64),
        "qz": df["qz"].to_numpy(dtype=np.float64),
        "qw": df["qw"].to_numpy(dtype=np.float64),
    })
    return out


def assign_seq_num(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "seq_num" not in out.columns:
        out["seq_num"] = np.arange(len(out), dtype=np.int64)
    return out


def assign_primary_imu(*, phone_rows: int, hub_rows: int, odo_rows: int) -> tuple[str, str]:
    if phone_rows >= 2:
        return "phone", "constraint"
    if hub_rows >= 2:
        return "hub", "primary"
    if odo_rows >= 2:
        return "none", "none"
    return "none", "none"


def map_hub_android_from_lidar(
    hub_mcu_ns: np.ndarray,
    lidar_mcu_ns: np.ndarray,
    lidar_android_ns: np.ndarray,
) -> tuple[np.ndarray, float, int, str]:
    if len(lidar_mcu_ns) != len(lidar_android_ns):
        raise ValueError("lidar_mcu_ns and lidar_android_ns must have equal length")
    hub_mcu_ns = np.asarray(hub_mcu_ns, dtype=np.int64)
    lidar_mcu_ns = np.asarray(lidar_mcu_ns, dtype=np.int64)
    lidar_android_ns = np.asarray(lidar_android_ns, dtype=np.int64)
    valid = np.isfinite(lidar_android_ns.astype(np.float64)) & (lidar_android_ns > 0)
    if int(np.count_nonzero(valid)) < 2:
        return np.zeros(len(hub_mcu_ns), dtype=np.int64), 1.0, 0, "insufficient_anchors"
    lm = lidar_mcu_ns[valid]
    la = lidar_android_ns[valid]
    t0_mcu = int(lm[0])
    t0_and = int(la[0])
    mcu_rel = (lm - t0_mcu).astype(np.float64)
    and_rel = (la - t0_and).astype(np.float64)
    dts = np.diff(la)
    jitter = calculate_robust_jitter(dts.astype(np.float64))
    s, c_rel = fit_clock_alignment_model(and_rel, mcu_rel, jitter)
    hub_rel = hub_mcu_ns.astype(np.int64) - t0_mcu
    drift = ((s - 1.0) * hub_rel.astype(np.float64)).astype(np.int64)
    hub_and = t0_and + int(c_rel) + hub_rel + drift
    return hub_and.astype(np.int64), float(s), int(c_rel), "lidar_mcu_android"


def hub_gap_stats(mcu_ns: np.ndarray, max_hold_ns: int) -> dict:
    mcu_ns = np.asarray(mcu_ns, dtype=np.int64)
    if len(mcu_ns) < 2:
        return {
            "sample_count": int(len(mcu_ns)),
            "gaps_over_max_hold": 0,
            "gaps_over_100ms": 0,
            "max_gap_ns": 0,
            "dropped_intervals": [],
        }
    dt = np.diff(mcu_ns)
    dropped = []
    for i, d in enumerate(dt):
        if int(d) > int(max_hold_ns):
            dropped.append({"from_mcu": int(mcu_ns[i]), "to_mcu": int(mcu_ns[i + 1]), "dt_ns": int(d)})
    return {
        "sample_count": int(len(mcu_ns)),
        "gaps_over_max_hold": int(np.count_nonzero(dt > max_hold_ns)),
        "gaps_over_100ms": int(np.count_nonzero(dt > 100_000_000)),
        "max_gap_ns": int(dt.max()),
        "dropped_intervals": dropped,
    }


def mask_lidar_zones(lidar_df: pd.DataFrame) -> pd.DataFrame:
    out = lidar_df.copy()
    for i in range(64):
        dist_col = f"distance_{i}"
        status_col = f"status_{i}"
        out[dist_col] = out[dist_col].astype(float) / 1000.0
        invalid = ~out[status_col].isin([5, 9]) | (out[dist_col] < 0.020) | (out[dist_col] > 4.0)
        out.loc[invalid, dist_col] = np.nan
    return out


def reorder_bounded_dejitter_buffer(
    df: pd.DataFrame,
    ts_col: str = "mcu_timestamp_ns",
    window_ns: int = 100_000_000,
) -> pd.DataFrame:
    """Bounded de-jitter reordering buffer and MCU clock reset/rollover compensation.

    1. Detect MCU clock resets/rollovers where dt < -window_ns, shifting subsequent timestamps.
    2. Stable-sort (mergesort) by [ts_col, '_orig_idx'] to resolve bounded reordering/jitter.
    3. Ensure strictly monotonic: if ts[i] <= ts[i-1], bump ts[i] = ts[i-1] + 1000 ns (1 us).
    """
    if df is None or len(df) == 0 or ts_col not in df.columns:
        return df

    out = df.copy()
    ts = out[ts_col].to_numpy(dtype=np.int64).copy()
    n = len(ts)
    if n < 2:
        return out

    # 1. MCU clock reset/rollover detection
    diffs = np.diff(ts)
    pos_diffs = diffs[diffs > 0]
    nominal_dt = int(np.median(pos_diffs)) if len(pos_diffs) > 0 else 2_500_000
    if nominal_dt <= 0:
        nominal_dt = 1000

    shift = 0
    for i in range(1, n):
        d = ts[i] - ts[i - 1]
        if d < -window_ns:
            # Substantial backward clock jump -> MCU reset or unexpected rollover
            shift += int((ts[i - 1] - ts[i]) + nominal_dt)
        if shift > 0:
            ts[i] += shift
    out[ts_col] = ts

    # 2. Stable-sort by [ts_col, '_orig_idx']
    out["_orig_idx"] = np.arange(n, dtype=np.int64)
    out = out.sort_values(by=[ts_col, "_orig_idx"], kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["_orig_idx"])

    # 3. Strictly monotonic guarantee
    ts_sorted = out[ts_col].to_numpy(dtype=np.int64).copy()
    for i in range(1, n):
        if ts_sorted[i] <= ts_sorted[i - 1]:
            ts_sorted[i] = ts_sorted[i - 1] + 1000
    out[ts_col] = ts_sorted

    return out

