"""Step-2 logging tests: JSON shape and correlation-id propagation (spec §12)."""

from __future__ import annotations

import io
import json

from gpu_orchestrator.logging import (
    configure_logging,
    correlation_context,
    get_logger,
)


def _emit_and_read(fn) -> dict:
    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream)
    fn(get_logger("test"))
    line = stream.getvalue().strip().splitlines()[-1]
    return json.loads(line)


def test_json_line_has_expected_shape():
    entry = _emit_and_read(lambda log: log.info("hello", extra={"foo": "bar"}))
    assert entry["message"] == "hello"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "gpu_orchestrator.test"
    assert entry["foo"] == "bar"
    assert "ts" in entry


def test_correlation_context_propagates():
    def fn(log):
        with correlation_context("corr-xyz"):
            log.info("inside")

    entry = _emit_and_read(fn)
    assert entry["correlation_id"] == "corr-xyz"


def test_record_correlation_id_beats_contextvar():
    def fn(log):
        with correlation_context("ambient"):
            log.info("event", extra={"correlation_id": "explicit"})

    entry = _emit_and_read(fn)
    assert entry["correlation_id"] == "explicit"


def test_no_correlation_id_is_null():
    entry = _emit_and_read(lambda log: log.info("bare"))
    assert entry["correlation_id"] is None
