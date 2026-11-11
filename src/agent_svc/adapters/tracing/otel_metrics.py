"""OpenTelemetry metrics export (OTLP/HTTP), sharing the tracer's endpoint and headers."""

from __future__ import annotations

from typing import Any

from agent_svc import __version__
from agent_svc.config.settings import Settings
from agent_svc.domain.exceptions import ConfigurationError


def build_meter_provider(settings: Settings, reader: Any = None) -> Any:
    """A ``MeterProvider`` exporting over OTLP, or over ``reader`` when one is given (tests)."""
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except ImportError as exc:
        raise ConfigurationError(
            "metrics need the OpenTelemetry SDK: uv sync --extra otel "
            "(or set AGENT_METRICS__ENABLED=false)"
        ) from exc
    t = settings.tracing
    if reader is None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=t.otlp_endpoint.rstrip("/") + "/v1/metrics", headers=t.otlp_headers or None
            ),
            export_interval_millis=int(settings.metrics.export_interval_seconds * 1000),
        )
    return MeterProvider(
        resource=Resource.create({"service.name": t.service_name, "service.version": __version__}),
        metric_readers=[reader],
    )
