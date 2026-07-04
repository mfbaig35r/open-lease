"""Recording layer for the reconcile loop: apply a decided action's outcome to the Deployment
record and emit the matching events (spec §7.3, §12).

This is the "record" half of the tick that ``reconciler.reconcile_once`` orchestrates. It is kept
separate from the decision core (``next_step``) and the side-effecting dispatcher (``execute``) so
each stays small and single-purpose. Nothing here talks to a provider or runtime; it only mutates
the record and appends events. ``resulting_state`` is pure; the rest also read/write the clock via
their ``now`` argument, never by calling it directly.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from ..config import Config
from ..errors import InstanceCreationError, ProviderAPIError
from ..events import EventLog
from ..models import (
    Deployment,
    DeploymentState,
    Event,
    EventKind,
    FailureInfo,
    ReconcileAction,
    StateTransition,
)


def resulting_state(
    action: ReconcileAction, observed: DeploymentState, deployment: Deployment
) -> DeploymentState:
    """The observed_state the deployment holds after this action's side effect (pure)."""
    if action == ReconcileAction.MARK_READY:
        return DeploymentState.READY
    if action == ReconcileAction.MARK_DEGRADED:
        return DeploymentState.DEGRADED
    if action == ReconcileAction.MARK_FAILED:
        return DeploymentState.FAILED
    if action in (ReconcileAction.CREATE_INSTANCE, ReconcileAction.RETRY):
        return DeploymentState.PROVISIONING
    if action == ReconcileAction.WAIT_FOR_PROVIDER:
        return DeploymentState.PROVISIONING
    if action == ReconcileAction.WAIT_FOR_RUNTIME:
        return DeploymentState.STARTING
    if action == ReconcileAction.DESTROY_INSTANCE:
        return (
            DeploymentState.STOPPING
            if deployment.desired_state == DeploymentState.STOPPED
            else DeploymentState.REQUESTED
        )
    # NONE / ADOPT_INSTANCE: no new stage, except the terminal "no instance + desired STOPPED" case
    # which is how a stop finally settles (map never emits STOPPED; next_step returns NONE here).
    if (
        deployment.desired_state == DeploymentState.STOPPED
        and observed == DeploymentState.REQUESTED
    ):
        return DeploymentState.STOPPED
    return observed


def settle(
    deployment: Deployment, action: ReconcileAction, result_state: DeploymentState, now: datetime
) -> None:
    transition(deployment, result_state, f"reconcile:{action.value}", now)
    # A successful create/retry/ready means we are back on the happy path: drop any prior failure so
    # the next tick does not re-read it and tear the fresh instance down. DESTROY during a retry
    # deliberately keeps the failure so the following tick knows to recreate.
    if action in (
        ReconcileAction.CREATE_INSTANCE,
        ReconcileAction.RETRY,
        ReconcileAction.MARK_READY,
    ):
        deployment.failure = None


def record_failure(
    deployment: Deployment, stage: DeploymentState, exc: Exception, config: Config, now: datetime
) -> None:
    """Convert a provider/runtime error into a FailureInfo. Provider API / creation errors are
    retryable up to the attempt cap; every other error type is terminal (spec §7.3)."""
    retryable = isinstance(exc, ProviderAPIError | InstanceCreationError)
    prior = deployment.failure.attempts if deployment.failure else 0
    deployment.failure = FailureInfo(
        stage=stage,
        message=str(exc),
        retryable=retryable,
        attempts=prior + 1,
        last_attempt_at=now,
    )
    if not retryable or deployment.failure.attempts >= config.retry_max_attempts:
        transition(deployment, DeploymentState.FAILED, f"failed:{type(exc).__name__}", now)


def retry_backoff_elapsed(deployment: Deployment, config: Config, now: datetime) -> bool:
    """Whether enough time has passed since the last failed attempt to try again. Capped exponential
    backoff (``retry_backoff_min`` doubling up to ``retry_backoff_max``, spec §7.3). Returns True
    when there is no prior attempt timestamp (nothing to wait on)."""
    failure = deployment.failure
    if failure is None or failure.last_attempt_at is None:
        return True
    delay = min(
        config.retry_backoff_max,
        config.retry_backoff_min * (2 ** max(0, failure.attempts - 1)),
    )
    return (now - failure.last_attempt_at).total_seconds() >= delay


def apply_stage_budget(
    deployment: Deployment, observed: DeploymentState, config: Config, now: datetime
) -> None:
    """Cost-safety net: a deployment stuck in a provisioning stage past its budget becomes a
    retryable failure, so the reconciler tears the pod down instead of paying for a stalled boot
    forever (spec §7.3). Only fires once we have actually been sitting in ``observed`` (its
    transition is already recorded); a state entered this same tick is not yet over budget."""
    budget = _stage_budget(observed, config)
    if budget is None or deployment.observed_state != observed:
        return
    if deployment.failure is not None and deployment.failure.stage == observed:
        return  # already flagged this stage; do not re-increment every tick
    entered = _entered_at(deployment, observed)
    if entered is not None and (now - entered).total_seconds() > budget:
        prior = deployment.failure.attempts if deployment.failure else 0
        deployment.failure = FailureInfo(
            stage=observed,
            message=f"{observed.value} exceeded {budget}s budget",
            retryable=True,
            attempts=prior + 1,
        )


def transition(
    deployment: Deployment, to_state: DeploymentState, reason: str, now: datetime
) -> None:
    if deployment.observed_state == to_state:
        return
    deployment.state_history.append(
        StateTransition(
            from_state=deployment.observed_state, to_state=to_state, at=now, reason=reason
        )
    )
    deployment.observed_state = to_state
    deployment.updated_at = now


# --- events ---------------------------------------------------------------------------

_ACTION_EVENT: dict[ReconcileAction, EventKind] = {
    ReconcileAction.CREATE_INSTANCE: EventKind.INSTANCE_CREATED,
    ReconcileAction.RETRY: EventKind.INSTANCE_CREATED,
    ReconcileAction.MARK_READY: EventKind.DEPLOYMENT_READY,
    ReconcileAction.MARK_DEGRADED: EventKind.HEALTH_DEGRADED,
    ReconcileAction.MARK_FAILED: EventKind.DEPLOYMENT_FAILED,
}


def emit_action_events(events: EventLog, deployment: Deployment, action: ReconcileAction) -> None:
    if action == ReconcileAction.NONE:
        return
    emit(events, deployment, EventKind.RECONCILE_ACTION, {"action": action.value})
    milestone = _ACTION_EVENT.get(action)
    if milestone is not None:
        emit(events, deployment, milestone, {"instance": instance_id(deployment)})
    if action == ReconcileAction.DESTROY_INSTANCE and deployment.observed_state in (
        DeploymentState.STOPPING,
        DeploymentState.STOPPED,
    ):
        emit(events, deployment, EventKind.DEPLOYMENT_STOPPED, {})


def emit(events: EventLog, deployment: Deployment, kind: EventKind, payload: dict) -> None:
    events.emit(
        Event(
            id=f"evt-{uuid4().hex[:12]}",
            correlation_id=deployment.id,
            deployment_id=deployment.id,
            kind=kind,
            payload=payload,
        )
    )


def instance_id(deployment: Deployment) -> str | None:
    return deployment.instance.provider_instance_id if deployment.instance else None


def _stage_budget(observed: DeploymentState, config: Config) -> int | None:
    return {
        DeploymentState.PROVISIONING: config.timeout_provisioning,
        DeploymentState.BOOTING: config.timeout_booting,
        DeploymentState.DOWNLOADING: config.timeout_download,
        DeploymentState.STARTING: config.timeout_starting,
    }.get(observed)


def _entered_at(deployment: Deployment, state: DeploymentState) -> datetime | None:
    for state_transition in reversed(deployment.state_history):
        if state_transition.to_state == state:
            return state_transition.at
    return None
