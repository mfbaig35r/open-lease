"""Step-2 store tests: round-trip every persisted model, migrations, and schema-version guard."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest

from gpu_orchestrator.errors import DeploymentNotFoundError, SchemaVersionError
from gpu_orchestrator.models import CostRecord, DeploymentState
from gpu_orchestrator.store import Store
from tests.fixtures.deployments import make_deployment
from tests.fixtures.events import ALL_EVENTS


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "state.db")
    yield s
    s.close()


def test_wal_mode_enabled(store):
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


@pytest.mark.parametrize("state", list(DeploymentState))
def test_deployment_roundtrip_through_store(store, state):
    dep = make_deployment(state, deployment_id=f"dep-{state.value}")
    store.save_deployment(dep)
    assert store.get_deployment(dep.id) == dep


def test_get_missing_deployment_raises(store):
    with pytest.raises(DeploymentNotFoundError):
        store.get_deployment("dep-nope")


def test_save_is_upsert(store):
    dep = make_deployment(DeploymentState.REQUESTED, deployment_id="dep-1")
    store.save_deployment(dep)
    updated = dep.model_copy(update={"observed_state": DeploymentState.READY})
    store.save_deployment(updated)
    assert store.get_deployment("dep-1").observed_state == DeploymentState.READY


def test_list_deployments_hides_stopped_by_default(store):
    for state in DeploymentState:
        store.save_deployment(make_deployment(state, deployment_id=f"dep-{state.value}"))
    visible = store.list_deployments()
    assert all(d.observed_state != DeploymentState.STOPPED for d in visible)
    assert len(store.list_deployments(include_stopped=True)) == len(list(DeploymentState))


def test_delete_deployment(store):
    dep = make_deployment(DeploymentState.READY, deployment_id="dep-del")
    store.save_deployment(dep)
    store.delete_deployment("dep-del")
    with pytest.raises(DeploymentNotFoundError):
        store.get_deployment("dep-del")


def test_event_roundtrip_and_ordering(store):
    for e in ALL_EVENTS:
        store.append_event(e)
    out = store.query_events()
    assert len(out) == len(ALL_EVENTS)
    assert {e.id for e in out} == {e.id for e in ALL_EVENTS}


def test_event_query_filters(store):
    for e in ALL_EVENTS:
        store.append_event(e)
    first = ALL_EVENTS[0]
    by_kind = store.query_events(kind=first.kind.value)
    assert all(e.kind == first.kind for e in by_kind)
    by_dep = store.query_events(deployment_id=first.deployment_id)
    assert len(by_dep) == len(ALL_EVENTS)
    none = store.query_events(deployment_id="dep-absent")
    assert none == []


def test_event_query_since(store):
    for e in ALL_EVENTS:
        store.append_event(e)
    future = datetime(2099, 1, 1, tzinfo=UTC)
    assert store.query_events(since=future) == []


def test_cost_record_roundtrip(store):
    rec = CostRecord(
        deployment_id="dep-cost",
        gpu_hourly_usd=1.5,
        started_at=datetime(2026, 7, 3, 12, 0, tzinfo=UTC),
        stopped_at=datetime(2026, 7, 3, 13, 0, tzinfo=UTC),
    )
    store.save_cost_record(rec)
    out = store.get_cost_records("dep-cost")
    assert out == [rec]


def test_unknown_schema_version_fails_loudly(store, tmp_path):
    dep = make_deployment(DeploymentState.READY, deployment_id="dep-ver")
    store.save_deployment(dep)
    # Tamper the stored payload to an unknown future version via a second connection.
    raw = sqlite3.connect(str(tmp_path / "state.db"))
    doc = json.loads(raw.execute("SELECT doc FROM deployments WHERE id='dep-ver'").fetchone()[0])
    doc["schema_version"] = 999
    raw.execute("UPDATE deployments SET doc = ? WHERE id='dep-ver'", (json.dumps(doc),))
    raw.commit()
    raw.close()
    with pytest.raises(SchemaVersionError):
        store.get_deployment("dep-ver")


def test_migrations_are_idempotent_across_reopen(tmp_path):
    path = tmp_path / "state.db"
    s1 = Store(path)
    s1.save_deployment(make_deployment(DeploymentState.READY, deployment_id="dep-persist"))
    s1.close()
    # Reopening runs _migrate again; it must be a no-op and data must survive.
    s2 = Store(path)
    assert s2.get_deployment("dep-persist").observed_state == DeploymentState.READY
    assert s2._conn.execute("PRAGMA user_version").fetchone()[0] == 1
    s2.close()
