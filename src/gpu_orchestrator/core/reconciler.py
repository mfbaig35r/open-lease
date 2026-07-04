"""The reconcile loop: the heart of the system (spec §7.3).

The design is a hard split between DECIDING and DOING:

* Two PURE functions (no I/O, no clock) form the decision core and are exhaustively unit-tested:
  - ``map_to_observed_state(instance, runtime_health)`` is the ONE place a provider-native reality
    is translated into a ``DeploymentState`` (spec §8.1). The provider never learns DeploymentState
    exists; every weird provider semantic is handled and tested here.
  - ``next_step(deployment, observed)`` returns the ONE ``ReconcileAction`` to take this tick.

* Two ASYNC functions form the I/O boundary:
  - ``observe`` asks the provider and runtime what actually exists right now. Adoption (spec §7.5)
    lives here because it needs a provider lookup.
  - ``execute`` is the ONLY place side effects happen: a thin dispatcher, one call per action.

``reconcile_once`` ties them together for a single deployment and persists the result. It is the
unit the daemon loop wraps (loop ownership = daemon, CLAUDE.md) and is also callable directly from
tests and a future serverless trigger (spec §7.3). One step per tick: no stage is ever chained
inside a single pass, which is what makes interruption and resume free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import Config
from ..core.catalog import Catalog
from ..errors import ProviderError, ReconcileError, RuntimeError_
from ..events import EventLog
from ..logging import get_logger
from ..models import (
    Deployment,
    DeploymentState,
    EventKind,
    GPUType,
    HealthState,
    HealthStatus,
    Instance,
    InstanceRequest,
    ReconcileAction,
    _utcnow,
)
from ..providers.base import Provider
from ..runtimes.base import Runtime
from ..store import Store
from . import costs, outcomes

_log = get_logger("reconciler")

# Provider-native pod-state tokens, interpreted in exactly one place (spec §8.1). RunPod reports
# ``desiredStatus`` and the mock reports its own string; both use "RUNNING" for a live pod. Anything
# in ``_DEAD`` is a pod the provider considers gone; we fold it back to "no instance" so next_step
# recreates (desired READY) or finishes teardown (desired STOPPED), and the orphan sweep reaps any
# provider object that outlives the record.
_RUNNING = {"RUNNING"}
_DEAD = {"EXITED", "TERMINATED", "TERMINATING", "DEAD", "FAILED"}


# =====================================================================================
# PURE decision core (no I/O, no clock) -- the most-tested code in the repo (spec §7.3)
# =====================================================================================


def map_to_observed_state(
    instance: Instance | None, runtime_health: HealthStatus | None
) -> DeploymentState:
    """Translate raw provider + runtime reality into a DeploymentState. Pure; the single source of
    truth for this mapping (spec §8.1). ``instance is None`` (or provider-dead) => nothing is
    running, which reads as REQUESTED: square one, from which next_step decides create vs done."""
    if instance is None:
        return DeploymentState.REQUESTED
    token = instance.state.upper()
    if token in _DEAD:
        return DeploymentState.REQUESTED
    if token not in _RUNNING:
        return DeploymentState.PROVISIONING
    # Compute is up. The runtime health decides the rest.
    if runtime_health is None:
        return DeploymentState.STARTING  # endpoint not routable / not probed yet
    if runtime_health.status is HealthState.HEALTHY:
        return DeploymentState.READY
    if runtime_health.status is HealthState.BOOTING:
        return DeploymentState.STARTING
    return DeploymentState.DEGRADED  # alive but unhealthy (spec §10)


def next_step(
    deployment: Deployment, observed: DeploymentState, *, max_attempts: int = 3
) -> ReconcileAction:
    """The one action to take this tick, as a pure function of (deployment, observed). Exhaustively
    unit-tested against the full desired x observed x failure matrix (spec §7.3, §18)."""
    desired = deployment.desired_state
    failure = deployment.failure
    has_instance = observed != DeploymentState.REQUESTED

    # Terminal failure: not retryable, or retries exhausted. Enforce cost safety, then rest.
    if failure is not None and (not failure.retryable or failure.attempts >= max_attempts):
        if has_instance:
            return ReconcileAction.DESTROY_INSTANCE
        return (
            ReconcileAction.MARK_FAILED
            if deployment.observed_state != DeploymentState.FAILED
            else ReconcileAction.NONE
        )

    # User wants it stopped: tear the instance down. The cost-safety invariant lives here (§7.3).
    if desired == DeploymentState.STOPPED:
        return ReconcileAction.DESTROY_INSTANCE if has_instance else ReconcileAction.NONE

    # A retryable failure that has budget left: clear any partial instance first, then retry.
    if failure is not None and failure.retryable:
        return ReconcileAction.DESTROY_INSTANCE if has_instance else ReconcileAction.RETRY

    # Happy path toward READY.
    if observed == DeploymentState.REQUESTED:
        return ReconcileAction.CREATE_INSTANCE
    if observed == DeploymentState.PROVISIONING:
        return ReconcileAction.WAIT_FOR_PROVIDER
    if observed in (
        DeploymentState.BOOTING,
        DeploymentState.DOWNLOADING,
        DeploymentState.STARTING,
    ):
        return ReconcileAction.WAIT_FOR_RUNTIME
    if observed == DeploymentState.READY:
        return (
            ReconcileAction.NONE
            if deployment.observed_state == DeploymentState.READY
            else ReconcileAction.MARK_READY
        )
    if observed == DeploymentState.DEGRADED:
        return (
            ReconcileAction.NONE
            if deployment.observed_state == DeploymentState.DEGRADED
            else ReconcileAction.MARK_DEGRADED
        )
    return ReconcileAction.NONE


# =====================================================================================
# ASYNC I/O boundary: observe (read reality) and execute (the only side effects)
# =====================================================================================


@dataclass
class Observation:
    """What ``observe`` learned this tick. ``adopted`` is True when an instance was recovered by tag
    (spec §7.5) rather than carried on the record."""

    observed_state: DeploymentState
    instance: Instance | None
    endpoint_url: str | None
    health: HealthStatus | None
    adopted: bool = False


async def observe(deployment: Deployment, provider: Provider, runtime: Runtime) -> Observation:
    """Ask the provider and runtime what actually exists. Trusts reality, not our records: if the
    provider says the pod is gone, it is gone regardless of what SQLite holds (spec §7.3)."""
    instance = deployment.instance
    adopted = False
    if instance is None:
        # No instance on record: it may still exist under our deterministic name (interrupted
        # deploy). Adoption makes deploy_model safe to interrupt at any point (spec §7.5).
        instance = await provider.find_instance_by_deployment_id(deployment.id)
        adopted = instance is not None
    else:
        instance = await provider.get_instance(instance.provider_instance_id)

    endpoint_url: str | None = None
    health: HealthStatus | None = None
    if instance is not None:
        endpoint_url = await provider.resolve_endpoint_url(instance, runtime.serving_port)
        # Once a deployment is serving (READY/DEGRADED), runtime health is the health engine's job,
        # with flap absorption (§10). observe here only confirms the instance is alive and preserves
        # the current serving state; a single blip must never regress it through this path.
        if deployment.observed_state in (DeploymentState.READY, DeploymentState.DEGRADED):
            return Observation(deployment.observed_state, instance, endpoint_url, None, adopted)
        if endpoint_url is not None:
            health = await _probe_health(runtime, endpoint_url, deployment.model_id)
    observed = map_to_observed_state(instance, health)
    return Observation(observed, instance, endpoint_url, health, adopted)


async def _probe_health(runtime: Runtime, endpoint_url: str, model_id: str) -> HealthStatus:
    """Bring-up readiness probe: is the server up and serving the model yet? Returns HEALTHY once
    both pass, else BOOTING. Ongoing degradation of an already-READY deployment is the health
    engine's concern (core/health.py), not this path's."""
    alive = await runtime.health_check(endpoint_url)
    if not alive.ok:
        return HealthStatus(status=HealthState.BOOTING, checks={"http_alive": alive})
    ready = await runtime.model_ready(endpoint_url, model_id)
    status = HealthState.HEALTHY if ready.ok else HealthState.BOOTING
    return HealthStatus(status=status, checks={"http_alive": alive, "model_ready": ready})


async def execute(
    action: ReconcileAction,
    deployment: Deployment,
    obs: Observation,
    *,
    provider: Provider,
    runtime: Runtime,
    catalog: Catalog,
    config: Config,
    store: Store,
    now: datetime,
) -> None:
    """The ONLY place side effects happen: a thin dispatcher, one provider/runtime call per action.
    Instance creation/destruction also opens/closes the cost accrual (§11), tying the money to the
    compute. WAIT_*, MARK_*, and NONE have no side effect here -- the state is recorded by
    reconcile_once; for a waiting action, the passage of time is the "action"."""
    if action in (ReconcileAction.CREATE_INSTANCE, ReconcileAction.RETRY):
        await _create_instance(deployment, provider, runtime, catalog, config, store, now)
    elif action == ReconcileAction.DESTROY_INSTANCE:
        await _destroy_instance(deployment, provider, store, now)
    elif action == ReconcileAction.ADOPT_INSTANCE:
        deployment.instance = obs.instance
    elif action == ReconcileAction.MARK_READY:
        deployment.endpoint_url = obs.endpoint_url


async def _create_instance(
    deployment: Deployment,
    provider: Provider,
    runtime: Runtime,
    catalog: Catalog,
    config: Config,
    store: Store,
    now: datetime,
) -> None:
    spec = catalog.get_spec(deployment.model_id)
    profile = deployment.profile
    gpu = await _resolve_gpu(provider, profile.recommended_gpu)
    name = config.instance_name(deployment.id)
    request = runtime.build_instance_request(spec, profile, gpu, name=name)
    request = _inject_secrets(request, config)
    deployment.instance = await provider.create_instance(request)
    # Cost accrues from the moment the instance exists (§11), at the resolved GPU's rate.
    costs.open_record(deployment, gpu.hourly_usd, now, store)


async def _destroy_instance(
    deployment: Deployment, provider: Provider, store: Store, now: datetime
) -> None:
    if deployment.instance is not None:
        await provider.destroy_instance(deployment.instance.provider_instance_id)
    costs.close_open_records(deployment.id, now, store)
    deployment.instance = None
    deployment.endpoint_url = None


async def _resolve_gpu(provider: Provider, wanted: str) -> GPUType:
    caps = await provider.capabilities()
    for gpu in caps.gpu_types:
        if wanted in (gpu.id, gpu.provider_sku):
            return gpu
    raise ReconcileError(
        f"provider {provider.name!r} offers no GPU matching profile requirement {wanted!r}"
    )


def _inject_secrets(request: InstanceRequest, config: Config) -> InstanceRequest:
    """Credential handling lives in one place: the orchestrator, never the runtime (spec §9)."""
    if config.hf_token is None:
        return request
    env = dict(request.env)
    env.setdefault("HF_TOKEN", config.hf_token.get_secret_value())
    return request.model_copy(update={"env": env})


# =====================================================================================
# reconcile_once: one tick for one deployment (observe -> decide -> execute -> record)
# =====================================================================================


async def reconcile_once(
    deployment: Deployment,
    *,
    provider: Provider,
    runtime: Runtime,
    catalog: Catalog,
    config: Config,
    store: Store,
    events: EventLog,
    now: datetime | None = None,
) -> Deployment:
    """Advance one deployment by exactly one step and persist it. Safe to call repeatedly; each call
    re-reads reality, so a crashed/restarted process simply resumes from wherever the pod is."""
    now = now or _utcnow()
    obs = await observe(deployment, provider, runtime)
    deployment.instance = obs.instance
    if obs.endpoint_url is not None:
        deployment.endpoint_url = obs.endpoint_url
    if obs.adopted:
        outcomes.emit(
            events,
            deployment,
            EventKind.INSTANCE_ADOPTED,
            {"instance": outcomes.instance_id(deployment)},
        )

    observed = obs.observed_state
    outcomes.apply_stage_budget(deployment, observed, config, now)

    # Steady state: already where we want to be, nothing pending. Do not churn the store.
    if (
        observed == deployment.desired_state
        and deployment.observed_state == observed
        and deployment.failure is None
    ):
        return deployment

    action = next_step(deployment, observed, max_attempts=config.retry_max_attempts)
    try:
        await execute(
            action,
            deployment,
            obs,
            provider=provider,
            runtime=runtime,
            catalog=catalog,
            config=config,
            store=store,
            now=now,
        )
    except (ProviderError, RuntimeError_) as exc:
        _record_and_emit_failure(deployment, observed, exc, config, now, store, events)
        return deployment

    result_state = outcomes.resulting_state(action, observed, deployment)
    outcomes.settle(deployment, action, result_state, now)
    store.save_deployment(deployment)
    outcomes.emit_action_events(events, deployment, action)
    return deployment


def _record_and_emit_failure(
    deployment: Deployment,
    stage: DeploymentState,
    exc: Exception,
    config: Config,
    now: datetime,
    store: Store,
    events: EventLog,
) -> None:
    outcomes.record_failure(deployment, stage, exc, config, now)
    store.save_deployment(deployment)
    if deployment.observed_state == DeploymentState.FAILED:
        outcomes.emit(events, deployment, EventKind.DEPLOYMENT_FAILED, {"error": str(exc)})
    else:
        # Retryable: not a deployment failure yet, just a recorded attempt.
        outcomes.emit(
            events,
            deployment,
            EventKind.RECONCILE_ACTION,
            {
                "action": "retry_scheduled",
                "error": str(exc),
                "attempt": deployment.failure.attempts,
            },
        )
