"""Producer-side visual frontend for server-side VIO.

The frontend (CLAHE, KLT, RANSAC, FAST/ANMS) runs ahead of the filter in its own thread and
publishes immutable FrontendResult records. Its KLT initial-flow rotation comes from raw gyro
samples (design doc §3.2), which is the only algorithmic difference to the inline frontend call.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - CPU images always ship OpenCV
    cv2 = None

from app.services.vio_core.math_utils import so3_exp

logger = logging.getLogger(__name__)

QUEUE_MAXSIZE = int(os.environ.get("VIO_FRONTEND_QUEUE_SIZE", "256"))


def gyro_relative_camera_rotation(
    gyro_t: np.ndarray,
    gyro_w: np.ndarray,
    t_prev: float,
    t_now: float,
    bg0: np.ndarray,
    R_bc: np.ndarray,
) -> np.ndarray:
    """R_prev_curr = R_bc^T ΔR_body^T R_bc with ΔR_body integrated over gyro samples in (t_prev, t_now].

    Each sample i contributes so3_exp((ω_i − bg0)·(t_i − t_{i−1})), the same first-order step the
    ESKF applies per IMU event. Sample 0 has no predecessor and is skipped. An empty window yields
    the identity.
    """
    if t_now <= t_prev:
        return np.eye(3)
    i0 = int(np.searchsorted(gyro_t, t_prev, side="right"))   # first index with t_i > t_prev
    i1 = int(np.searchsorted(gyro_t, t_now, side="right"))    # first index with t_i > t_now
    dR = np.eye(3)
    for i in range(max(i0, 1), i1):
        dt_i = float(gyro_t[i] - gyro_t[i - 1])
        if dt_i <= 0.0:
            continue
        dR = dR @ so3_exp((gyro_w[i] - bg0) * dt_i)
    if i1 <= max(i0, 1):
        return np.eye(3)
    return R_bc.T @ dR.T @ R_bc


@dataclass(slots=True)
class FrontendFrame:
    frame_id: int
    timestamp: float
    image: Union[np.ndarray, str, None]   # decoded array (gray or BGR), image path, or None


@dataclass(slots=True)
class FrontendResult:
    frame_id: int
    timestamp: float
    matched_pts_prev: np.ndarray
    matched_pts_curr: np.ndarray
    new_pts: np.ndarray
    num_tracked: int
    new_features_injected: int
    gray: Optional[np.ndarray]
    image_size: Optional[tuple[int, int]]      # (width, height)
    R_prev_curr: Optional[np.ndarray]
    skipped: bool = False                      # image missing/undecodable: consumer skips the frame
    compute_time_s: float = 0.0


@dataclass(slots=True)
class FrontendError:
    frame_id: int
    exc: BaseException


class FrontendTimeout(TimeoutError):
    """Raised by FrontendPipeline.get when no result arrived within the wall-clock budget."""


@dataclass(slots=True)
class _EndSentinel:
    compute_time_s: float = 0.0


_END = _EndSentinel()


def decode_gray(image: Union[np.ndarray, str, None]) -> Optional[np.ndarray]:
    """Same decode rules as the former inline camera-frame handler."""
    img = image
    if isinstance(img, str):
        if cv2 is None or not os.path.exists(img):
            return None
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if img is None or not isinstance(img, np.ndarray):
        return None
    if img.ndim == 3:
        if cv2 is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(img, axis=2).astype(np.uint8)
    return img


def _process_worker(
    frames: list[FrontendFrame],
    gyro_t: np.ndarray,
    gyro_w: np.ndarray,
    bg0: np.ndarray,
    R_bc: np.ndarray,
    frontend_cls,
    frontend_kwargs: dict,
    out_q,
    stop_evt,
) -> None:
    if cv2 is not None:
        raw_threads = os.environ.get("VIO_CV_THREADS", "4").strip()
        try:
            cv_threads = max(0, int(raw_threads)) if raw_threads else 4
        except ValueError:
            cv_threads = 4
        cv2.setNumThreads(cv_threads)
    frontend = frontend_cls(**frontend_kwargs)
    t_last = None
    compute_time_s = 0.0
    for frame in frames:
        if stop_evt.is_set():
            if hasattr(out_q, "cancel_join_thread"):
                try:
                    out_q.cancel_join_thread()
                except Exception:
                    pass
            return
        t0 = time.perf_counter()
        try:
            gray = decode_gray(frame.image)
            if gray is None:
                compute_time_s += time.perf_counter() - t0
                empty = np.empty((0, 2), dtype=np.float32)
                item = FrontendResult(
                    frame.frame_id, frame.timestamp, empty, empty, empty, 0, 0, None, None, None, skipped=True, compute_time_s=compute_time_s
                )
            else:
                if t_last is None:
                    R_prev_curr = None
                else:
                    R_prev_curr = gyro_relative_camera_rotation(gyro_t, gyro_w, t_last, frame.timestamp, bg0, R_bc)
                res = frontend.process_frame(gray, R_prev_curr)
                n_inj = int(res.get("new_features_injected", 0))
                new_pts = res.get("new_pts")
                if new_pts is None:
                    prev_pts = getattr(frontend, "_prev_pts", None)
                    if n_inj > 0 and prev_pts is not None and len(prev_pts) >= n_inj:
                        new_pts = np.array(prev_pts[-n_inj:], copy=True).reshape(-1, 2)
                    else:
                        new_pts = np.empty((0, 2))
                h, w = gray.shape[:2]
                t_last = frame.timestamp
                compute_time_s += time.perf_counter() - t0
                item = FrontendResult(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    matched_pts_prev=res.get("matched_pts_prev", np.empty((0, 2))),
                    matched_pts_curr=res.get("matched_pts_curr", np.empty((0, 2))),
                    new_pts=new_pts,
                    num_tracked=int(res.get("num_tracked", 0)),
                    new_features_injected=n_inj,
                    gray=None,
                    image_size=(int(w), int(h)),
                    R_prev_curr=R_prev_curr,
                    compute_time_s=compute_time_s,
                )
        except Exception as exc:
            while not stop_evt.is_set():
                try:
                    out_q.put(FrontendError(frame.frame_id, exc), timeout=0.1)
                    break
                except Exception:
                    continue
            if hasattr(out_q, "close"):
                try:
                    out_q.close()
                except Exception:
                    pass
            if hasattr(out_q, "join_thread"):
                try:
                    out_q.join_thread()
                except Exception:
                    pass
            return

        while not stop_evt.is_set():
            try:
                out_q.put(item, timeout=0.1)
                break
            except Exception:
                continue

    while not stop_evt.is_set():
        try:
            out_q.put(_EndSentinel(compute_time_s=compute_time_s), timeout=0.1)
            break
        except Exception:
            continue

    if hasattr(out_q, "close"):
        try:
            out_q.close()
        except Exception:
            pass
    if hasattr(out_q, "join_thread"):
        try:
            out_q.join_thread()
        except Exception:
            pass


class FrontendPipeline:
    """Runs VisualFrontend.process_frame for an ordered frame list, in a producer thread or process or inline.

    Threaded, process, and inline modes share _process_one, so the only difference between them is execution model.
    The VisualFrontend instance is touched by exactly one execution worker; consumers only read immutable FrontendResult records.
    """

    def __init__(
        self,
        frames: Sequence[FrontendFrame],
        gyro_t: np.ndarray,
        gyro_w: np.ndarray,
        bg0: np.ndarray,
        T_bc: np.ndarray,
        frontend,
        threaded: bool = True,
        maxsize: int = QUEUE_MAXSIZE,
        use_process: bool = False,
    ) -> None:
        self._frames = list(frames)
        self._gyro_t = np.asarray(gyro_t, dtype=np.float64).reshape(-1)
        self._gyro_w = np.asarray(gyro_w, dtype=np.float64).reshape(-1, 3)
        self._bg0 = np.asarray(bg0, dtype=np.float64).reshape(3).copy()
        self._R_bc = np.asarray(T_bc, dtype=np.float64)[:3, :3].copy()
        self._frontend = frontend
        self.threaded = bool(threaded)
        self.use_process = bool(use_process) and self.threaded
        self._proc = None
        self._thread: Optional[threading.Thread] = None

        if self.use_process:
            try:
                import billiard as mp
            except ImportError:
                import multiprocessing as mp
            ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
            self._queue = ctx.Queue(maxsize=maxsize)
            self._stop = ctx.Event()
        else:
            self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
            self._stop = threading.Event()

        self._cursor = 0
        self._t_last_processed: Optional[float] = None
        self.compute_time_s = 0.0

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> "FrontendPipeline":
        if not self.threaded:
            return self

        if self.use_process and self._proc is None:
            try:
                import billiard as mp
            except ImportError:
                import multiprocessing as mp
            ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
            frontend_cls = type(self._frontend)
            frontend_kwargs = {
                "intrinsics": getattr(self._frontend, "K", None),
                "dist_coeffs": getattr(self._frontend, "dist_coeffs", None),
                "max_features": getattr(self._frontend, "max_features", 300),
                "min_features": getattr(self._frontend, "min_features", 80),
                "decimation_cap": getattr(self._frontend, "decimation_cap", 150),
            }
            self._proc = ctx.Process(
                target=_process_worker,
                name="vio-frontend-producer-proc",
                args=(
                    self._frames,
                    self._gyro_t,
                    self._gyro_w,
                    self._bg0,
                    self._R_bc,
                    frontend_cls,
                    frontend_kwargs,
                    self._queue,
                    self._stop,
                ),
            )
            self._proc.daemon = True
            self._proc.start()
        elif not self.use_process and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="vio-frontend-producer", daemon=True)
            self._thread.start()
        return self

    def stop(self, join_timeout: float = 30.0) -> None:
        self._stop.set()
        if self._proc is not None:
            if hasattr(self._queue, "cancel_join_thread"):
                try:
                    self._queue.cancel_join_thread()
                except Exception:
                    pass
            self._proc.join(timeout=join_timeout)
            if self._proc.is_alive():
                logger.warning("frontend producer process did not stop within %.1f s; terminating", join_timeout)
                try:
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
                except Exception:
                    pass
            if hasattr(self._queue, "close"):
                try:
                    self._queue.close()
                except Exception:
                    pass
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                # The thread is still inside OpenCV and keeps using `frontend`. We drop our handle so a
                # repeat stop() is a no-op; a get() after stop() would report "producer thread died",
                # which is fine because process_session never reads after stopping.
                logger.warning("frontend producer thread did not stop within %.1f s", join_timeout)
            self._thread = None

    def __enter__(self) -> "FrontendPipeline":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    @property
    def is_alive(self) -> bool:
        if self._proc is not None:
            return self._proc.is_alive()
        return self._thread is not None and self._thread.is_alive()

    # ---- producer ----------------------------------------------------------
    def _process_one(self, frame: FrontendFrame) -> FrontendResult:
        t0 = time.perf_counter()
        gray = decode_gray(frame.image)
        if gray is None:
            self.compute_time_s += time.perf_counter() - t0
            empty = np.empty((0, 2), dtype=np.float32)
            return FrontendResult(frame.frame_id, frame.timestamp, empty, empty, empty, 0, 0, None, None, None, skipped=True)

        if self._t_last_processed is None:
            R_prev_curr = None
        else:
            R_prev_curr = gyro_relative_camera_rotation(
                self._gyro_t, self._gyro_w, self._t_last_processed, frame.timestamp, self._bg0, self._R_bc
            )

        res = self._frontend.process_frame(gray, R_prev_curr)

        n_inj = int(res.get("new_features_injected", 0))
        new_pts = res.get("new_pts")
        if new_pts is None:
            prev_pts = getattr(self._frontend, "_prev_pts", None)
            if n_inj > 0 and prev_pts is not None and len(prev_pts) >= n_inj:
                new_pts = np.array(prev_pts[-n_inj:], copy=True).reshape(-1, 2)   # snapshot, not a view
            else:
                new_pts = np.empty((0, 2))

        h, w = gray.shape[:2]
        self._t_last_processed = frame.timestamp
        self.compute_time_s += time.perf_counter() - t0
        return FrontendResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            matched_pts_prev=res.get("matched_pts_prev", np.empty((0, 2))),
            matched_pts_curr=res.get("matched_pts_curr", np.empty((0, 2))),
            new_pts=new_pts,
            num_tracked=int(res.get("num_tracked", 0)),
            new_features_injected=n_inj,
            gray=gray,
            image_size=(int(w), int(h)),
            R_prev_curr=R_prev_curr,
        )

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        for frame in self._frames:
            if self._stop.is_set():
                return
            try:
                item = self._process_one(frame)
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer, which re-raises it
                self._put(FrontendError(frame.frame_id, exc))
                return
            if not self._put(item):
                return
        self._put(_END)

    # ---- consumer ------------------------------------------------------------
    def get(self, frame_id: int, timeout: Optional[float] = None) -> FrontendResult:
        if not self.threaded:
            if self._cursor >= len(self._frames):
                raise RuntimeError(f"frontend pipeline exhausted: no frame left for frame_id {frame_id}")
            frame = self._frames[self._cursor]
            self._cursor += 1
            if frame.frame_id != frame_id:
                raise RuntimeError(f"frontend pipeline frame mismatch: producer {frame.frame_id}, consumer {frame_id}")
            return self._process_one(frame)

        if self.threaded and self._thread is None and self._proc is None and not self._stop.is_set():
            self.start()

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if not self.is_alive and self._queue.empty():
                if self._proc is not None and self._proc.exitcode not in (0, None):
                    raise RuntimeError(f"frontend pipeline producer process died unexpectedly with exitcode {self._proc.exitcode} before frame {frame_id}")
                if self._thread is not None:
                    raise RuntimeError(f"frontend pipeline producer thread died unexpectedly before frame {frame_id}")
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            chunk = 0.05 if remaining is None else min(0.05, remaining)
            try:
                item = self._queue.get(timeout=chunk)
                break
            except queue.Empty:
                if not self.is_alive:
                    if self._proc is not None and self._proc.exitcode not in (0, None):
                        raise RuntimeError(f"frontend pipeline producer process died unexpectedly with exitcode {self._proc.exitcode} before frame {frame_id}")
                    if self._thread is not None and self._queue.empty():
                        raise RuntimeError(f"frontend pipeline producer thread died unexpectedly before frame {frame_id}")
                if deadline is not None and time.monotonic() >= deadline:
                    raise FrontendTimeout(f"frontend result for frame {frame_id} not available within {timeout} s") from None

        if item is _END or isinstance(item, _EndSentinel):
            if isinstance(item, _EndSentinel) and item.compute_time_s > 0.0:
                self.compute_time_s = item.compute_time_s
            self._queue.put(_END)  # keep the sentinel for any further call
            raise RuntimeError(f"frontend pipeline exhausted: no frame left for frame_id {frame_id}")
        if isinstance(item, FrontendError):
            self._queue.put(item)
            raise item.exc
        if getattr(item, "compute_time_s", 0.0) > 0.0:
            self.compute_time_s = item.compute_time_s
        if item.frame_id != frame_id:
            raise RuntimeError(f"frontend pipeline frame mismatch: producer {item.frame_id}, consumer {frame_id}")
        return item
