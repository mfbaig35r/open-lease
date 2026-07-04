"""Structured JSON logging with correlation IDs (spec §12).

JSON goes to stderr (or a file); human-readable rich output is a CLI rendering concern, never here.
Every operation carries a ``correlation_id`` so a single deploy can be traced end to end across the
reconciler and provider calls. The id lives in a ``ContextVar`` and is injected into every record;
a record may also carry its own ``correlation_id`` (events do), which takes precedence.

OpenTelemetry is not integrated in Phase 1, but spans would map 1:1 to these boundaries, so no
restructuring is needed later.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TextIO

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Standard LogRecord attributes we never want duplicated into the JSON "extra" bag.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
    "correlation_id",
    "taskName",
}


def set_correlation_id(correlation_id: str | None) -> None:
    _correlation_id.set(correlation_id)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


@contextmanager
def correlation_context(correlation_id: str) -> Iterator[None]:
    """Bind a correlation id for the duration of a block, restoring the previous one after."""
    token = _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        # A record's own correlation_id (e.g. an event's) wins over the ambient contextvar.
        correlation_id = getattr(record, "correlation_id", None) or _correlation_id.get()
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id,
        }
        # Fold in structured extras passed via logger.info(..., extra={...}).
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """Install the JSON formatter on the package root logger. Idempotent."""
    logger = logging.getLogger("gpu_orchestrator")
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    # Replace any handlers we previously installed so repeated calls do not stack duplicates.
    logger.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger (``gpu_orchestrator.<name>``)."""
    return logging.getLogger(f"gpu_orchestrator.{name}")
