"""Adaptive Hardware Detection and Concurrency Management for 3D Reconstruction.

Coordinates running GPU-intensive Depthor+ (ONNX Runtime CUDA) and VGGT
(PyTorch CUDA FP16) concurrently using isolated multiprocessing (`spawn` context)
to eliminate CUDA context contention, minimize execution time, and ensure
fail-safe graceful degradation.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import time
from typing import Any, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)


def is_vggt_enabled() -> bool:
    """Check if VGGT inference is enabled via env var or settings."""
    env_val = os.getenv("ENABLE_VGGT")
    if env_val is not None:
        return env_val.strip().lower() not in ("0", "false", "no", "off")
    return getattr(settings, "ENABLE_VGGT", True)


def is_parallel_reconstruction_enabled() -> bool:
    """Check if parallel reconstruction mode is enabled via env var or settings."""
    env_val = os.getenv("PARALLEL_RECONSTRUCTION_ENABLED")
    if env_val is not None:
        return env_val.strip().lower() not in ("0", "false", "no", "off")
    return getattr(settings, "PARALLEL_RECONSTRUCTION_ENABLED", True)


def get_min_parallel_vram_gb() -> float:
    """Get minimum GPU VRAM threshold in GB required for parallel reconstruction."""
    env_val = os.getenv("MIN_PARALLEL_VRAM_GB")
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            pass
    return float(getattr(settings, "MIN_PARALLEL_VRAM_GB", 11.0))


def get_parallel_reconstruction_timeout_s() -> float:
    """Get maximum timeout in seconds for parallel VGGT inference worker."""
    env_val = os.getenv("PARALLEL_RECONSTRUCTION_TIMEOUT_S")
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            pass
    return float(getattr(settings, "PARALLEL_RECONSTRUCTION_TIMEOUT_S", 300.0))


def get_gpu_vram_gb(device_id: int = 0) -> float:
    """Return total VRAM in gigabytes (GiB) for device_id, or 0.0 if unavailable."""
    # 1. Try PyTorch CUDA properties
    try:
        import torch

        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            total_bytes = torch.cuda.get_device_properties(device_id).total_memory
            return float(total_bytes / (1024**3))
    except Exception:
        pass

    # 2. Fallback to nvidia-smi query
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            text=True,
            timeout=2.0,
        ).strip().splitlines()
        if res and device_id < len(res):
            mb = float(res[device_id].strip())
            return float(mb / 1024.0)
    except Exception:
        pass

    return 0.0


def check_parallel_reconstruction_hardware(
    device_id: Optional[int] = None,
    min_vram_gb: Optional[float] = None,
    ignore_config: bool = False,
) -> Tuple[bool, str]:
    """Evaluate whether system hardware and configuration support parallel 3D reconstruction.

    Checks:
    1. VGGT enabled (ENABLE_VGGT != 0)
    2. Parallel mode enabled (PARALLEL_RECONSTRUCTION_ENABLED != 0, unless ignore_config=True)
    3. CUDA availability (torch.cuda.is_available)
    4. Total GPU VRAM >= threshold (default 11.0 GB)

    Returns:
        (is_eligible, reason_string)
    """
    if not is_vggt_enabled():
        return False, "VGGT is disabled via ENABLE_VGGT configuration"

    if not ignore_config and not is_parallel_reconstruction_enabled():
        return False, "Parallel reconstruction is disabled via PARALLEL_RECONSTRUCTION_ENABLED configuration"

    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if not cuda_available:
        return False, "CUDA is not available on this system"

    dev_id = device_id if device_id is not None else getattr(settings, "CUDA_DEVICE_ID", 0)
    threshold = min_vram_gb if min_vram_gb is not None else get_min_parallel_vram_gb()
    actual_vram = get_gpu_vram_gb(dev_id)

    if actual_vram < threshold:
        return (
            False,
            f"Insufficient GPU VRAM for parallel reconstruction: {actual_vram:.2f} GB available < {threshold:.2f} GB required",
        )

    return (
        True,
        f"Hardware check passed: CUDA available, {actual_vram:.2f} GB VRAM >= {threshold:.2f} GB threshold",
    )


def _vggt_worker_entrypoint(
    session_dir: str,
    force_recompute: bool,
    result_queue: Any,
    device_id: Optional[int] = None,
) -> None:
    """Worker entrypoint executed in an isolated spawned child process."""
    import time
    import logging

    worker_logger = logging.getLogger("reconstruction_concurrency.worker")
    t0 = time.time()
    try:
        dev_id = device_id if device_id is not None else getattr(settings, "CUDA_DEVICE_ID", 0)
        try:
            import torch
            if torch.cuda.is_available() and dev_id < torch.cuda.device_count():
                torch.cuda.set_device(dev_id)
        except Exception as dev_err:
            worker_logger.warning("Could not set CUDA device %s: %s", dev_id, dev_err)

        from app.services.vggt_runner import run_vggt_inference

        success = run_vggt_inference(session_dir, force_recompute=force_recompute)
        dur = time.time() - t0
        err = None if success else "run_vggt_inference returned False"
        result_queue.put({"success": bool(success), "duration_s": dur, "error": err})
    except BaseException as e:
        dur = time.time() - t0
        worker_logger.exception("VGGT worker process encountered unhandled exception: %s", e)
        try:
            result_queue.put({"success": False, "duration_s": dur, "error": str(e)})
        except Exception:
            pass
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise


class ParallelVGGTWorker:
    """Manages an isolated spawned worker process for VGGT 3D inference.

    Executes VGGT in a separate multiprocessing process using the 'spawn' context
    to ensure complete CUDA context isolation from ONNX Runtime Depthor+.
    Provides synchronization barrier with timeout and graceful error reporting.
    """

    def __init__(
        self,
        session_dir: str,
        force_recompute: bool = False,
        timeout_s: Optional[float] = None,
        device_id: Optional[int] = None,
    ):
        self.session_dir = session_dir
        self.force_recompute = force_recompute
        self.timeout_s = (
            timeout_s if timeout_s is not None else get_parallel_reconstruction_timeout_s()
        )
        self.device_id = device_id if device_id is not None else getattr(settings, "CUDA_DEVICE_ID", 0)
        self._ctx = mp.get_context("spawn")
        self._queue: Optional[Any] = None
        self._proc: Optional[mp.Process] = None
        self._started: bool = False
        self._joined: bool = False
        self._start_time: Optional[float] = None
        self._result: Optional[dict[str, Any]] = None

    def start(self) -> None:
        """Launch the VGGT worker process in the background."""
        if self._started:
            return
        self._queue = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_vggt_worker_entrypoint,
            args=(self.session_dir, self.force_recompute, self._queue, self.device_id),
            name="ParallelVGGTWorker",
        )
        self._proc.start()
        self._started = True
        self._start_time = time.time()
        logger.info(
            "Spawned parallel VGGT worker process (pid=%s, device_id=%s) for session: %s (force_recompute=%s)",
            self._proc.pid,
            self.device_id,
            self.session_dir,
            self.force_recompute,
        )

    def is_alive(self) -> bool:
        """Check whether the worker process is currently running."""
        return self._proc.is_alive() if self._proc is not None else False

    def join(self, timeout: Optional[float] = None) -> dict[str, Any]:
        """Wait for worker process to finish with timeout barrier.

        Returns a dictionary with keys:
          - 'success': bool
          - 'duration_s': float
          - 'error': Optional[str]
        """
        if not self._started or self._proc is None:
            return {"success": False, "duration_s": 0.0, "error": "Worker not started"}
        if self._joined and self._result is not None:
            return self._result

        if timeout is not None:
            remaining_timeout = timeout
        else:
            elapsed = time.time() - self._start_time if self._start_time else 0.0
            remaining_timeout = max(0.5, self.timeout_s - elapsed)

        logger.info(
            "Waiting for VGGT worker (pid=%s) at barrier with remaining timeout=%.1fs...",
            self._proc.pid,
            remaining_timeout,
        )
        self._proc.join(timeout=remaining_timeout)

        if self._proc.is_alive():
            logger.warning(
                "VGGT worker process (pid=%s) timed out after %.1fs. Force terminating...",
                self._proc.pid,
                remaining_timeout,
            )
            self._terminate_proc()
            self._result = {
                "success": False,
                "duration_s": (
                    (time.time() - self._start_time)
                    if self._start_time
                    else remaining_timeout
                ),
                "error": f"Timeout after {remaining_timeout:.1f}s",
            }
            self._joined = True
            self._cleanup_queue()
            return self._result

        exitcode = self._proc.exitcode
        dur = (time.time() - self._start_time) if self._start_time else 0.0
        res = None
        if self._queue is not None:
            try:
                # If process died abnormally (exitcode != 0), do not wait 2s on empty queue
                if exitcode != 0:
                    res = self._queue.get_nowait() if not self._queue.empty() else None
                else:
                    res = self._queue.get(timeout=1.0)
            except Exception:
                res = None

        if res is None:
            res = {
                "success": False,
                "duration_s": dur,
                "error": (
                    "Worker process exited without returning a result"
                    if exitcode == 0
                    else f"Worker process terminated abnormally with exit code {exitcode}"
                ),
            }

        self._result = res
        self._joined = True
        self._cleanup_queue()
        return self._result

    def terminate(self) -> None:
        """Safely terminate child process if still alive and clean up resources."""
        self._terminate_proc()
        self._cleanup_queue()

    def _terminate_proc(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                logger.warning("Terminating VGGT worker process (pid=%s)", self._proc.pid)
                try:
                    self._proc.terminate()
                    self._proc.join(timeout=3.0)
                    if self._proc.is_alive():
                        logger.warning(
                            "VGGT worker (pid=%s) still alive after terminate; killing...",
                            self._proc.pid,
                        )
                        self._proc.kill()
                        self._proc.join(timeout=1.0)
                except Exception as e:
                    logger.warning("Exception during VGGT process termination: %s", e)
            else:
                try:
                    self._proc.join(timeout=0.2)
                except Exception:
                    pass

    def _cleanup_queue(self) -> None:
        if self._queue is not None:
            try:
                self._queue.cancel_join_thread()
                self._queue.close()
            except Exception:
                pass
            self._queue = None

    def __enter__(self) -> "ParallelVGGTWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.terminate()
        elif not self._joined:
            self.join()
