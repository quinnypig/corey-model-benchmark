from __future__ import annotations

import atexit
import ctypes
import gc
import logging
import os
import threading
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode


LOGGER = logging.getLogger(__name__)
_LOCK = threading.Lock()
_PROVIDER: TracerProvider | None = None
_FLASK_APPS: set[int] = set()


def initialize() -> bool:
    """Configure one process-wide OTLP exporter when an endpoint or API key exists."""
    global _PROVIDER
    with _LOCK:
        if _PROVIDER is not None:
            return True

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
        base_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        api_key = os.environ.get("HONEYCOMB_API_KEY", "").strip()
        if not endpoint and base_endpoint:
            endpoint = base_endpoint.rstrip("/") + "/v1/traces"
        if not endpoint and api_key:
            endpoint = "https://api.honeycomb.io/v1/traces"
        if not endpoint:
            return False

        headers = {}
        if api_key and "api.honeycomb.io" in endpoint:
            headers["x-honeycomb-team"] = api_key

        try:
            package_version = version("quinnferno")
        except PackageNotFoundError:
            package_version = "development"
        resource = Resource.create(
            {
                SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "quinnferno"),
                SERVICE_VERSION: os.environ.get("QUINNFERNO_VERSION", package_version),
            }
        )
        try:
            exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(
                BatchSpanProcessor(
                    exporter,
                    max_queue_size=4096,
                    max_export_batch_size=512,
                    schedule_delay_millis=2000,
                )
            )
            trace.set_tracer_provider(provider)
            _PROVIDER = provider
            atexit.register(_shutdown)
        except Exception:
            LOGGER.exception("OpenTelemetry initialization failed; continuing without export")
            return False
    return True


def instrument_flask(app: Any) -> None:
    if not initialize():
        return
    identity = id(app)
    with _LOCK:
        if identity in _FLASK_APPS:
            return
        _FLASK_APPS.add(identity)
    FlaskInstrumentor().instrument_app(
        app,
        tracer_provider=_PROVIDER,
        excluded_urls=r".*/healthz,.*/readyz",
    )


def tracer() -> Any:
    return trace.get_tracer("quinnferno")


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Iterator[Any]:
    with tracer().start_as_current_span(name, kind=kind, attributes=_clean(attributes or {})) as current:
        try:
            yield current
        except BaseException as exc:
            current.record_exception(exc)
            current.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def set_attributes(current: Any, attributes: dict[str, Any]) -> None:
    current.set_attributes(_clean(attributes))


def runtime_attributes(prefix: str = "process") -> dict[str, Any]:
    attributes: dict[str, Any] = {
        f"{prefix}.thread_count": threading.active_count(),
    }
    for name, path in (
        ("memory.current_bytes", "/sys/fs/cgroup/memory.current"),
        ("memory.peak_bytes", "/sys/fs/cgroup/memory.peak"),
        ("memory.limit_bytes", "/sys/fs/cgroup/memory.max"),
    ):
        value = _read_int(path)
        if value is not None:
            attributes[f"{prefix}.{name}"] = value
    rss = _proc_rss()
    if rss is not None:
        attributes[f"{prefix}.memory.rss_bytes"] = rss
    return attributes


def trim_memory() -> dict[str, Any]:
    """Return freed Python/glibc arenas after large report materializations."""
    before_rss = _proc_rss()
    before_current = _read_int("/sys/fs/cgroup/memory.current")
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (AttributeError, OSError):
        pass
    after_rss = _proc_rss()
    after_current = _read_int("/sys/fs/cgroup/memory.current")
    result = {
        "memory.before_rss_bytes": before_rss,
        "memory.after_rss_bytes": after_rss,
        "memory.before_cgroup_bytes": before_current,
        "memory.after_cgroup_bytes": after_current,
    }
    if isinstance(before_rss, int) and isinstance(after_rss, int):
        result["memory.released_rss_bytes"] = max(0, before_rss - after_rss)
    if isinstance(before_current, int) and isinstance(after_current, int):
        result["memory.released_cgroup_bytes"] = max(0, before_current - after_current)
    return result


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in attributes.items()
        if value is not None and isinstance(value, (bool, str, int, float, list, tuple))
    }


def _read_int(path: str) -> int | None:
    try:
        raw = Path(path).read_text(encoding="ascii").strip()
        return int(raw) if raw != "max" else None
    except (OSError, ValueError):
        return None


def _proc_rss() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _shutdown() -> None:
    global _PROVIDER
    provider = _PROVIDER
    _PROVIDER = None
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:
            LOGGER.exception("OpenTelemetry shutdown failed")
