"""Step-1 contract tests: the domain models round-trip and the fixtures are well-formed.

Store-backed round-trips arrive at build step 2; here we lock the JSON contract and schema_version
so downstream steps build on a stable shape.
"""

from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from gpu_orchestrator.models import (
    SCHEMA_VERSION,
    CostRecord,
    Deployment,
    DeploymentState,
    Event,
    EventKind,
    RuntimeProfile,
)
from tests.fixtures.catalog import PROFILES, SPECS
from tests.fixtures.deployments import BY_STATE, FAILED_DEPLOYMENT
from tests.fixtures.events import ALL_EVENTS


def test_every_deployment_state_has_a_fixture():
    assert set(BY_STATE) == set(DeploymentState)


def test_every_event_kind_has_a_fixture():
    assert {e.kind for e in ALL_EVENTS} == set(EventKind)


@pytest.mark.parametrize("state", list(DeploymentState))
def test_deployment_json_roundtrip_preserves_shape(state):
    original = BY_STATE[state]
    restored = Deployment.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.schema_version == SCHEMA_VERSION


def test_event_roundtrip():
    original = ALL_EVENTS[0]
    restored = Event.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.schema_version == SCHEMA_VERSION


def test_persisted_models_carry_schema_version():
    for model in (Deployment, Event, CostRecord):
        assert "schema_version" in model.model_fields


def test_failed_deployment_has_no_live_instance():
    # The cost-safety invariant in data form: FAILED implies the instance was destroyed (spec §7.3).
    assert FAILED_DEPLOYMENT.instance is None
    assert FAILED_DEPLOYMENT.failure is not None
    assert FAILED_DEPLOYMENT.failure.retryable is False


def test_cost_record_math_is_deterministic_once_stopped():
    from datetime import datetime

    rec = CostRecord(
        deployment_id="dep-a1b2c3",
        gpu_hourly_usd=2.0,
        started_at=datetime(2026, 7, 3, 12, 0, tzinfo=UTC),
        stopped_at=datetime(2026, 7, 3, 14, 30, tzinfo=UTC),
    )
    assert rec.accrued_usd == 5.0  # 2.0 * 2.5h
    assert rec.estimated_monthly_usd == pytest.approx(2.0 * 24 * 30)


def test_cost_record_accrues_while_running():
    from datetime import datetime

    rec = CostRecord(
        deployment_id="dep-a1b2c3",
        gpu_hourly_usd=1.0,
        started_at=datetime(2020, 1, 1, tzinfo=UTC),  # long ago
    )
    assert rec.stopped_at is None
    assert rec.accrued_usd > 0


def test_runtime_profile_requires_validation_metadata():
    # A profile without validation metadata must not construct (spec §14).
    with pytest.raises(ValidationError):
        RuntimeProfile(
            model_id="x",
            image="img",
            recommended_gpu="A100-80GB",
            min_disk_gb=20,
        )


def test_catalog_fixtures_are_consistent():
    for model_id, spec in SPECS.items():
        assert spec.id == model_id
        assert model_id in PROFILES
        assert PROFILES[model_id].validation.startup_timeout_seconds > 0
