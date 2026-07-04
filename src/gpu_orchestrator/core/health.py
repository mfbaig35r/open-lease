"""Health engine (spec §10): monitor deployments that have reached serving state and declare
DEGRADED with flap absorption.

Ownership split with the reconciler (§7.3 vs §10): the reconciler owns instance lifecycle and the
bring-up progression (is it READY yet). Once a deployment is READY, THIS engine owns runtime-health
monitoring. The two do not race: the reconciler's ``observe`` stops re-probing runtime health once a
deployment is READY/DEGRADED (it only confirms the instance is alive), leaving the DEGRADED verdict
to the monitor, which requires ``health_failure_threshold`` consecutive failures before flipping,
so a single blip cannot regress a healthy deployment.

Phase 1 is report-only: DEGRADED is surfaced (``health_degraded`` event + observed_state), never
auto-restarted. ``restart`` is a user action, and it is expensive on ephemeral disks (§10, §14).

``run_checks`` is a one-shot instantaneous probe (used by ``Orchestrator.get_health``).
``HealthMonitor.check_once`` is one poll tick for one deployment: it applies the flap threshold and
mutates/persists state. It is callable directly and daemon-wrapped in step 7, mirroring
``reconcile_once``.
"""

from __future__ import annotations

from datetime import datetime

from ..config import Config
from ..events import EventLog
from ..models import (
    CheckResult,
    Deployment,
    DeploymentState,
    EventKind,
    HealthState,
    HealthStatus,
    _utcnow,
)
from ..providers.base import Provider
from ..runtimes.base import Runtime
from ..store import Store
from . import outcomes

# Deployments the monitor acts on: only those that have reached serving state (spec §10).
_MONITORED = (DeploymentState.READY, DeploymentState.DEGRADED)


async def run_checks(deployment: Deployment, provider: Provider, runtime: Runtime) -> HealthStatus:
    """The four checks of §10 as one instantaneous snapshot (no flap logic, no state mutation).

    ``instance_alive`` failing yields status FAILED: the pod is gone, which is the reconciler's job,
    not ours. Otherwise the instance is alive and the runtime checks decide healthy vs degraded."""
    instance = None
    if deployment.instance is not None:
        instance = await provider.get_instance(deployment.instance.provider_instance_id)
    if instance is None:
        return HealthStatus(
            status=HealthState.FAILED,
            checks={"instance_alive": CheckResult(ok=False, detail="pod gone")},
        )

    checks: dict[str, CheckResult] = {"instance_alive": CheckResult(ok=True)}
    endpoint_url = deployment.endpoint_url or await provider.resolve_endpoint_url(
        instance, runtime.serving_port
    )
    if endpoint_url is None:
        checks["http_alive"] = CheckResult(ok=False, detail="endpoint not routable")
        return HealthStatus(status=HealthState.DEGRADED, checks=checks)

    http = await runtime.health_check(endpoint_url)
    model = await runtime.model_ready(endpoint_url, deployment.model_id)
    checks["http_alive"] = http
    checks["model_loaded"] = model
    # Latency is a degradation signal, recorded but not itself a failure in Phase 1 (§10).
    checks["latency"] = CheckResult(ok=True, latency_ms=model.latency_ms, detail="latency signal")

    status = HealthState.HEALTHY if (http.ok and model.ok) else HealthState.DEGRADED
    return HealthStatus(status=status, checks=checks)


class HealthMonitor:
    """Owns the per-deployment consecutive-failure counters that absorb flapping. Phase 1 keeps them
    in memory (the poll loop is a single process); durability across restart is not required because
    a restarted monitor simply re-observes and re-counts from zero."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._consecutive_failures: dict[str, int] = {}

    async def check_once(
        self,
        deployment: Deployment,
        *,
        provider: Provider,
        runtime: Runtime,
        store: Store,
        events: EventLog,
        now: datetime | None = None,
    ) -> HealthStatus:
        """One health-poll tick for one deployment. No-op for anything not yet in serving state."""
        now = now or _utcnow()
        if deployment.observed_state not in _MONITORED:
            return HealthStatus(status=HealthState.BOOTING, checks={})

        status = await run_checks(deployment, provider, runtime)

        if status.status is HealthState.FAILED:
            # Pod gone: not our call. Reset the counter and let the reconciler collapse/recreate.
            self._consecutive_failures.pop(deployment.id, None)
            return status

        if status.status is HealthState.HEALTHY:
            self._consecutive_failures.pop(deployment.id, None)
            if deployment.observed_state is DeploymentState.DEGRADED:
                self._recover(deployment, store, events, now)
            return status

        # Raw degraded: only DECLARE degraded after enough consecutive failures (flap absorption).
        strikes = self._consecutive_failures.get(deployment.id, 0) + 1
        self._consecutive_failures[deployment.id] = strikes
        if (
            strikes >= self._config.health_failure_threshold
            and deployment.observed_state is not DeploymentState.DEGRADED
        ):
            self._declare_degraded(deployment, store, events, now, strikes)
        return status

    def _declare_degraded(
        self, deployment: Deployment, store: Store, events: EventLog, now: datetime, strikes: int
    ) -> None:
        outcomes.transition(deployment, DeploymentState.DEGRADED, "health:degraded", now)
        store.save_deployment(deployment)
        outcomes.emit(
            events, deployment, EventKind.HEALTH_DEGRADED, {"consecutive_failures": strikes}
        )

    def _recover(
        self, deployment: Deployment, store: Store, events: EventLog, now: datetime
    ) -> None:
        outcomes.transition(deployment, DeploymentState.READY, "health:recovered", now)
        store.save_deployment(deployment)
        outcomes.emit(events, deployment, EventKind.HEALTH_PASSED, {})
