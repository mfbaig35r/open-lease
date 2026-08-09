"""Stage-budget selection: which timeout a deployment is actually held to (spec §7.3).

The STARTING stage is the one that matters in practice. ``map_to_observed_state`` never yields
DOWNLOADING (a running pod whose runtime is not answering yet reads as STARTING), so a cold start
pulls its entire model inside STARTING and must be held to the model's declared
``startup_timeout_seconds``, not to the global default meant for a small model.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.outcomes import apply_stage_budget
from gpu_orchestrator.models import DeploymentState
from tests.fixtures.catalog import QWEN3_06B_PROFILE, QWEN3_32B_PROFILE
from tests.fixtures.deployments import make_deployment

S = DeploymentState
_T0 = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)


def _elapsed(seconds: int) -> datetime:
    """The fixture's state transition is stamped at _T0, so this is time-in-stage."""
    return _T0 + timedelta(seconds=seconds)


def test_starting_honors_per_model_budget_over_global_default():
    # qwen3-32b declares 2400s. The global timeout_starting default is 300s.
    config = Config()
    assert config.timeout_starting < QWEN3_32B_PROFILE.validation.startup_timeout_seconds

    dep = make_deployment(S.STARTING, profile=QWEN3_32B_PROFILE)
    apply_stage_budget(dep, S.STARTING, config, _elapsed(config.timeout_starting + 60))

    assert dep.failure is None, "a 65GB cold pull was torn down at the small-model default"


def test_starting_still_trips_past_the_declared_budget():
    config = Config()
    dep = make_deployment(S.STARTING, profile=QWEN3_32B_PROFILE)
    budget = QWEN3_32B_PROFILE.validation.startup_timeout_seconds

    apply_stage_budget(dep, S.STARTING, config, _elapsed(budget + 60))

    assert dep.failure is not None
    assert dep.failure.stage is S.STARTING
    assert dep.failure.retryable is True
    assert str(budget) in dep.failure.message


def test_each_model_gets_its_own_starting_budget():
    # The small model declares 600s, so it must trip well before the large model's 2400s.
    config = Config()
    small_budget = QWEN3_06B_PROFILE.validation.startup_timeout_seconds
    dep = make_deployment(S.STARTING, profile=QWEN3_06B_PROFILE)

    apply_stage_budget(dep, S.STARTING, config, _elapsed(small_budget + 60))

    assert dep.failure is not None
    assert str(small_budget) in dep.failure.message


def test_non_starting_stages_still_use_the_global_config():
    # Only STARTING carries the per-model override; PROVISIONING is provider-side, not model-side.
    config = Config()
    dep = make_deployment(S.PROVISIONING, profile=QWEN3_32B_PROFILE)

    apply_stage_budget(dep, S.PROVISIONING, config, _elapsed(config.timeout_provisioning + 60))

    assert dep.failure is not None
    assert dep.failure.stage is S.PROVISIONING
    assert str(config.timeout_provisioning) in dep.failure.message


def test_regression_cold_h100_pull_survives_the_five_minute_mark():
    """Repro of the 2026-08-03 crash-loop: three qwen3-32b pods on an H100 were destroyed at
    317s, 319s, and 262s in STARTING, each restarting a ~17m/65GB download from zero, so the
    deployment could never converge and burned $0.76 before failing."""
    config = Config()
    dep = make_deployment(S.STARTING, profile=QWEN3_32B_PROFILE)

    for observed_teardown in (317, 319):
        apply_stage_budget(dep, S.STARTING, config, _elapsed(observed_teardown))

    assert dep.failure is None
