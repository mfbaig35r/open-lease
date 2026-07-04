"""Step-2 event-log tests: emit persists + logs, query delegates and filters (spec §12)."""

from __future__ import annotations

import pytest

from gpu_orchestrator.events import EventLog
from gpu_orchestrator.models import EventKind
from gpu_orchestrator.store import Store
from tests.fixtures.events import HAPPY_PATH, make_event


@pytest.fixture
def log(tmp_path):
    store = Store(tmp_path / "state.db")
    yield EventLog(store)
    store.close()


def test_emit_persists_and_is_queryable(log):
    for e in HAPPY_PATH:
        log.emit(e)
    out = log.query(deployment_id=HAPPY_PATH[0].deployment_id)
    assert len(out) == len(HAPPY_PATH)


def test_query_filters_by_kind(log):
    for e in HAPPY_PATH:
        log.emit(e)
    ready = log.query(kind=EventKind.DEPLOYMENT_READY)
    assert len(ready) == 1
    assert ready[0].kind == EventKind.DEPLOYMENT_READY


def test_emit_writes_a_structured_log_line(log, caplog):
    import logging

    # Attach caplog's handler directly so capture does not depend on the events logger's
    # propagate setting (configure_logging may set propagate=False elsewhere in the suite).
    logger = logging.getLogger("gpu_orchestrator.events")
    logger.addHandler(caplog.handler)
    old_propagate = logger.propagate
    logger.propagate = False  # capture exactly once via the directly-attached handler
    logger.setLevel(logging.INFO)
    try:
        event = make_event(EventKind.DEPLOYMENT_READY, seq=1)
        log.emit(event)
    finally:
        logger.removeHandler(caplog.handler)
        logger.propagate = old_propagate

    records = [r for r in caplog.records if r.name == "gpu_orchestrator.events"]
    assert len(records) == 1
    assert records[0].correlation_id == event.correlation_id
    assert records[0].event_kind == EventKind.DEPLOYMENT_READY.value
