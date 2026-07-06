"""Step-5 pure-core tests: map_to_observed_state and next_step.

These two functions carry no I/O and no clock, so they are tested exhaustively against the state
matrix (spec §7.3, §18). If the reconciler has a bug, it should show up here first.
"""

from __future__ import annotations

import pytest

from gpu_orchestrator.core.reconciler import map_to_observed_state, next_step
from gpu_orchestrator.models import (
    CheckResult,
    DeploymentState,
    FailureInfo,
    HealthState,
    HealthStatus,
    Instance,
    ReconcileAction,
)
from tests.fixtures.deployments import make_deployment

S = DeploymentState
A = ReconcileAction


def _instance(state: str) -> Instance:
    return Instance(provider_instance_id="pod-1", provider="mock", gpu_type="MOCK-GPU", state=state)


def _health(status: HealthState) -> HealthStatus:
    return HealthStatus(status=status, checks={"x": CheckResult(ok=status is HealthState.HEALTHY)})


# --- map_to_observed_state ------------------------------------------------------------


def test_map_no_instance_is_requested():
    assert map_to_observed_state(None, None) == S.REQUESTED


def test_map_dead_instance_folds_to_requested():
    for token in ("EXITED", "TERMINATED", "DEAD", "TERMINATING", "FAILED"):
        assert map_to_observed_state(_instance(token), None) == S.REQUESTED


def test_map_not_yet_running_is_provisioning():
    assert map_to_observed_state(_instance("PROVISIONING"), None) == S.PROVISIONING


def test_map_running_without_health_is_starting():
    assert map_to_observed_state(_instance("RUNNING"), None) == S.STARTING


def test_map_running_healthy_is_ready():
    assert map_to_observed_state(_instance("RUNNING"), _health(HealthState.HEALTHY)) == S.READY


def test_map_running_booting_is_starting():
    assert map_to_observed_state(_instance("RUNNING"), _health(HealthState.BOOTING)) == S.STARTING


def test_map_running_unhealthy_is_degraded():
    for status in (HealthState.DEGRADED, HealthState.FAILED):
        assert map_to_observed_state(_instance("RUNNING"), _health(status)) == S.DEGRADED


# --- next_step: happy path toward READY -----------------------------------------------


def _dep(desired: S, current: S, failure: FailureInfo | None = None):
    d = make_deployment(S.READY)
    d.desired_state = desired
    d.observed_state = current
    d.failure = failure
    return d


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (S.REQUESTED, A.CREATE_INSTANCE),
        (S.PROVISIONING, A.WAIT_FOR_PROVIDER),
        (S.BOOTING, A.WAIT_FOR_RUNTIME),
        (S.DOWNLOADING, A.WAIT_FOR_RUNTIME),
        (S.STARTING, A.WAIT_FOR_RUNTIME),
    ],
)
def test_next_step_progress_toward_ready(observed, expected):
    assert next_step(_dep(S.READY, S.REQUESTED), observed) == expected


def test_next_step_marks_ready_once():
    assert next_step(_dep(S.READY, S.STARTING), S.READY) == A.MARK_READY


def test_next_step_ready_steady_state_is_none():
    assert next_step(_dep(S.READY, S.READY), S.READY) == A.NONE


def test_next_step_marks_degraded_once():
    assert next_step(_dep(S.READY, S.READY), S.DEGRADED) == A.MARK_DEGRADED


def test_next_step_degraded_steady_state_is_none():
    assert next_step(_dep(S.READY, S.DEGRADED), S.DEGRADED) == A.NONE


# --- next_step: stop / cost safety ----------------------------------------------------


def test_next_step_stop_with_instance_destroys():
    assert next_step(_dep(S.STOPPED, S.READY), S.READY) == A.DESTROY_INSTANCE


def test_next_step_stop_without_instance_is_none():
    assert next_step(_dep(S.STOPPED, S.STOPPING), S.REQUESTED) == A.NONE


# --- next_step: failure handling ------------------------------------------------------


def _fail(retryable: bool, attempts: int) -> FailureInfo:
    return FailureInfo(stage=S.PROVISIONING, message="boom", retryable=retryable, attempts=attempts)


def test_next_step_retryable_without_instance_retries():
    dep = _dep(S.READY, S.REQUESTED, _fail(True, 1))
    assert next_step(dep, S.REQUESTED, max_attempts=3) == A.RETRY


def test_next_step_retryable_with_partial_instance_destroys_first():
    dep = _dep(S.READY, S.PROVISIONING, _fail(True, 1))
    assert next_step(dep, S.PROVISIONING, max_attempts=3) == A.DESTROY_INSTANCE


def test_next_step_exhausted_retries_marks_failed():
    dep = _dep(S.READY, S.PROVISIONING, _fail(True, 3))
    assert next_step(dep, S.REQUESTED, max_attempts=3) == A.MARK_FAILED


def test_next_step_terminal_failure_destroys_lingering_instance():
    # Cost safety: even a terminally-failed deployment must not keep a running instance.
    dep = _dep(S.READY, S.READY, _fail(False, 1))
    assert next_step(dep, S.READY) == A.DESTROY_INSTANCE


# --- next_step: runtime crash loop (gauntlet §18 #4) ----------------------------------


def test_next_step_runtime_crash_loop_marks_failed():
    # A pod created fine but crashing before READY every time (OOM loop): once the crash count hits
    # the cap and no pod is held, give up instead of recreating forever.
    dep = _dep(S.READY, S.STARTING)
    dep.instance = None
    dep.runtime_failures = 3
    assert next_step(dep, S.REQUESTED, max_attempts=3) == A.MARK_FAILED


def test_next_step_runtime_crash_loop_destroys_held_pod_first():
    # Cost safety: a pod still held when the crash cap trips is torn down first, even a dead one.
    dep = _dep(S.READY, S.STARTING)
    dep.instance = _instance("EXITED")
    dep.runtime_failures = 3
    assert next_step(dep, S.REQUESTED, max_attempts=3) == A.DESTROY_INSTANCE


def test_next_step_runtime_failures_below_cap_recreates():
    # Under the cap, a vanished pod is still recreated -- transient single deaths self-heal.
    dep = _dep(S.READY, S.STARTING)
    dep.instance = None
    dep.runtime_failures = 2
    assert next_step(dep, S.REQUESTED, max_attempts=3) == A.CREATE_INSTANCE


def test_next_step_runtime_failed_terminal_rests():
    # Once terminally failed with no pod, rest -- do not oscillate on daemon re-ticks.
    dep = _dep(S.READY, S.FAILED)
    dep.instance = None
    dep.runtime_failures = 3
    assert next_step(dep, S.REQUESTED, max_attempts=3) == A.NONE


def test_next_step_terminal_failure_without_instance_marks_failed():
    dep = _dep(S.READY, S.STARTING, _fail(False, 1))
    assert next_step(dep, S.REQUESTED) == A.MARK_FAILED


def test_next_step_already_failed_is_none():
    dep = _dep(S.READY, S.FAILED, _fail(False, 1))
    assert next_step(dep, S.REQUESTED) == A.NONE
