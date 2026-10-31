"""Environment-driven configuration for the Temporal layer (no heavy imports; sandbox-safe)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class TemporalSettings:
    address: str = _f("TEMPORAL_ADDRESS", "localhost:7233")
    namespace: str = _f("TEMPORAL_NAMESPACE", "default")
    api_key: str = _f("TEMPORAL_API_KEY", "")
    tls: bool = _f("TEMPORAL_TLS", "0") == "1"
    cpu_task_queue: str = _f("TEMPORAL_CPU_TASK_QUEUE", "lidar-cpu")
    gpu_task_queue: str = _f("TEMPORAL_GPU_TASK_QUEUE", "lidar-gpu-bridge")
    metrics_bind: str = _f("TEMPORAL_METRICS_BIND", "0.0.0.0:9464")
    otlp_endpoint: str = _f("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    # comma separated: cpu,gpu
    worker_roles: str = _f("WORKER_ROLES", "cpu,gpu")
    cpu_max_concurrent_activities: int = int(_f("CPU_MAX_CONCURRENT_ACTIVITIES", "4"))
    # RunPod bridge: each in-flight activity == one RunPod job; cap it to protect GPU quota.
    gpu_max_concurrent_activities: int = int(_f("GPU_MAX_CONCURRENT_ACTIVITIES", "3"))
    gpu_activities_per_second: float = float(_f("GPU_ACTIVITIES_PER_SECOND", "1"))


@dataclass(frozen=True)
class RunPodSettings:
    api_key: str = _f("RUNPOD_API_KEY", "")
    endpoint_id: str = _f("RUNPOD_ENDPOINT_ID", "")
    base_url: str = _f("RUNPOD_BASE_URL", "https://api.runpod.ai/v2")
    poll_interval_s: float = float(_f("RUNPOD_POLL_INTERVAL_S", "3"))
    execution_timeout_s: int = int(_f("RUNPOD_EXECUTION_TIMEOUT_S", "1200"))
    queue_ttl_s: int = int(_f("RUNPOD_QUEUE_TTL_S", "900"))
    http_timeout_s: float = float(_f("RUNPOD_HTTP_TIMEOUT_S", "30"))


@dataclass(frozen=True)
class CallbackSettings:
    url: str = _f("BACKEND_CALLBACK_URL", "")
    secret: str = _f("BACKEND_CALLBACK_SECRET", "")
    timeout_s: float = float(_f("BACKEND_CALLBACK_TIMEOUT_S", "15"))


def temporal_settings() -> TemporalSettings:
    return TemporalSettings()


def runpod_settings() -> RunPodSettings:
    return RunPodSettings()


def callback_settings() -> CallbackSettings:
    return CallbackSettings()
