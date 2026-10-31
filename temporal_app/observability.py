from __future__ import annotations

import json
import logging
import sys
import time

from temporalio.runtime import PrometheusConfig, Runtime, TelemetryConfig

from temporal_app.config import TemporalSettings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # temporalio adds workflow/activity context under these attributes
        for attr in ("temporal_workflow", "temporal_activity"):
            ctx = getattr(record, attr, None)
            if ctx:
                data[attr] = ctx
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)


def build_runtime(cfg: TemporalSettings) -> Runtime:
    """Expose SDK + custom metrics (lidar_stage_duration_seconds, ...) on a Prometheus endpoint."""
    return Runtime(telemetry=TelemetryConfig(metrics=PrometheusConfig(bind_address=cfg.metrics_bind)))


def build_interceptors(cfg: TemporalSettings) -> list:
    """OpenTelemetry tracing is optional: enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set."""
    if not cfg.otlp_endpoint:
        return []
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from temporalio.contrib.opentelemetry import TracingInterceptor

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint)))
        trace.set_tracer_provider(provider)
        return [TracingInterceptor()]
    except ImportError:
        logging.getLogger(__name__).warning("OTEL endpoint set but opentelemetry packages missing; tracing disabled")
        return []
