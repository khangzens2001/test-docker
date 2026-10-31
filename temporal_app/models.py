"""Plain dataclasses exchanged between workflow, activities and callers (JSON-serialisable)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Stage names emitted by the GPU handler (see app/pipeline/runner.py stage_callback)
STAGES = [
    "queued",
    "resolve_input",
    "gpu_submitted",
    "ingestion",
    "vio",
    "keyframe_selection",
    "em_refinement",
    "depth_completion",
    "tsdf_integration",
    "vggt_prior",
    "plane_segmentation",
    "export_artifacts",
    "materials",
    "build_result",
    "notify_backend",
    "done",
]

# Error types that must never be retried (bad input / domain errors)
NON_RETRYABLE_ERROR_TYPES = [
    "invalid_input",
    "ZipUnsafe",
    "SensorDataIngestionError",
    "TrackingLostError",
    "VioInitializationError",
    "MaterialEstimationError",
    "TatamiShavingLimitExceededError",
    "TatamiSkewAngleExceededError",
    "RunPodConfigError",
    "PipelineFailed",
]


@dataclass
class RoomScanInput:
    task_id: str
    zip_url: str
    batch_id: str = "batch"
    user_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)  # estimation overrides (pitch_mm, ...)
    gpu_params: dict[str, Any] = field(default_factory=dict)  # conf_threshold, max_points, ...
    enable_parallel: Optional[bool] = None
    use_search_attributes: bool = True
    gpu_task_queue: str = "lidar-gpu-bridge"
    gpu_timeout_s: int = 1500
    notify: bool = True


@dataclass
class ResolvedInput:
    task_id: str
    job_id: str
    zip_url: str
    batch_id: str
    params: dict[str, Any] = field(default_factory=dict)
    gpu_params: dict[str, Any] = field(default_factory=dict)
    enable_parallel: Optional[bool] = None


@dataclass
class StageEvent:
    stage: str
    pct: int = 0
    stage_index: int = -1
    source: str = "gpu"


@dataclass
class ScanStatus:
    workflow_id: str
    state: str = "pending"  # pending|running|succeeded|failed|cancelled
    current_stage: str = "queued"
    pct: int = 0
    stage_log: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    runpod_job_id: Optional[str] = None


# Job-level failures worth one resubmission (fresh RunPod job, sequential GPU mode).
# Generic deterministic failures ("PipelineFailed") are NOT resubmitted: they would burn GPU twice for the same error.
RESUBMIT_ERROR_TYPES = ["oom", "RunPodTimedOut", "RunPodCancelled", "RunPodWorkerLost", "transient_io"]
