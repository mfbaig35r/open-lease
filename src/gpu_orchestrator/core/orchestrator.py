"""The Orchestrator facade: the single entry point every interface uses (spec §7.1).

Interfaces (CLI, API, MCP, Swamp) call these methods and nothing else in the core; all business
logic lives behind here. The facade owns the long-lived collaborators (store, event log, catalog)
and composes a Provider with a Runtime -- the only place in the system those two seams meet.

``deploy_model`` reads top-to-bottom as the deploy flow (E2): validate model -> resolve profile ->
apply overrides -> create the Deployment record (desired=READY) -> emit -> hand to the reconciler ->
return (non-blocking) or wait. The reconcile loop itself is owned by the daemon (CLAUDE.md); when
a caller passes ``wait=True`` the facade drives ``reconcile_once`` inline until it settles.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime
from uuid import uuid4

from ..config import Config
from ..errors import OrchestratorError, ReconcileError
from ..events import EventLog
from ..logging import correlation_context, get_logger
from ..models import (
    CostEstimate,
    CostRecord,
    Deployment,
    DeploymentState,
    Event,
    EventKind,
    GPUType,
    HealthStatus,
    ModelSpec,
    ProviderInfo,
    RuntimeOverrides,
    RuntimeProfile,
)
from ..providers.base import PROVIDERS, Provider
from ..runtimes.base import RUNTIMES, Runtime
from ..store import Store
from . import health
from .catalog import Catalog, load_catalog
from .reconciler import reconcile_once

_log = get_logger("orchestrator")

# A wait/drive safety cap: reconcile_once is one step per tick, so a deployment reaches a terminal
# state in a bounded number of ticks. This guards the inline ``wait=True`` path against a stuck
# provider; the daemon uses the real clock instead.
_MAX_DRIVE_TICKS = 200

_TERMINAL_READY = {DeploymentState.READY, DeploymentState.FAILED}
_TERMINAL_STOP = {DeploymentState.STOPPED, DeploymentState.FAILED}


class Orchestrator:
    def __init__(
        self,
        config: Config | None = None,
        *,
        catalog: Catalog | None = None,
        provider: Provider | None = None,
        runtime: Runtime | None = None,
    ) -> None:
        # ``provider``/``runtime`` injection is the seam tests use to run against the mock provider
        # without touching config or the network; production leaves them None and builds by name.
        self._config = config or Config()
        self._store = Store(self._config.state_db)
        self._events = EventLog(self._store)
        self._catalog = catalog or load_catalog()
        self._injected_provider = provider
        self._injected_runtime = runtime

    @property
    def config(self) -> Config:
        return self._config

    def close(self) -> None:
        self._store.close()

    # --- deploy / lifecycle ---------------------------------------------------------

    async def deploy_model(
        self,
        model_id: str,
        *,
        provider: str = "runpod",
        gpu: str | None = None,
        wait: bool = False,
        overrides: RuntimeOverrides | None = None,
    ) -> Deployment:
        spec = self._catalog.get_spec(model_id)  # raises ModelNotFoundError if unknown
        profile = _apply_overrides(self._catalog.get_profile(model_id), gpu, overrides)
        deployment = Deployment(
            id=_new_deployment_id(),
            model_id=spec.id,
            provider=provider,
            desired_state=DeploymentState.READY,
            observed_state=DeploymentState.REQUESTED,
            profile=profile,
        )
        with correlation_context(deployment.id):
            self._store.save_deployment(deployment)
            self._emit(deployment, EventKind.DEPLOYMENT_REQUESTED, {"model_id": spec.id})
            if wait:
                deployment = await self._drive(deployment, _TERMINAL_READY)
        return deployment

    async def stop_deployment(self, deployment_id: str) -> Deployment:
        deployment = self._store.get_deployment(deployment_id)
        deployment.desired_state = DeploymentState.STOPPED
        self._store.save_deployment(deployment)
        return await self._drive(deployment, _TERMINAL_STOP)

    async def delete_deployment(self, deployment_id: str) -> None:
        deployment = self._store.get_deployment(deployment_id)
        deployment.desired_state = DeploymentState.STOPPED
        self._store.save_deployment(deployment)
        # Cost safety: never delete a record while its instance may still be running (spec §7.3).
        await self._drive(deployment, _TERMINAL_STOP)
        self._store.delete_deployment(deployment_id)
        self._emit(deployment, EventKind.DEPLOYMENT_DELETED, {})

    async def restart_deployment(self, deployment_id: str) -> Deployment:
        deployment = self._store.get_deployment(deployment_id)
        # An honest restart is a full re-provision (spec §10): tear down, then bring up fresh.
        deployment.desired_state = DeploymentState.STOPPED
        self._store.save_deployment(deployment)
        await self._drive(deployment, _TERMINAL_STOP)
        deployment.desired_state = DeploymentState.READY
        deployment.failure = None
        self._store.save_deployment(deployment)
        return await self._drive(deployment, _TERMINAL_READY)

    # --- reads ----------------------------------------------------------------------

    def get_deployment(self, deployment_id: str) -> Deployment:
        return self._store.get_deployment(deployment_id)

    def list_deployments(self, *, include_stopped: bool = False) -> list[Deployment]:
        return self._store.list_deployments(include_stopped=include_stopped)

    def list_models(self) -> list[ModelSpec]:
        return self._catalog.list_models()

    def events(
        self, deployment_id: str | None = None, *, since: datetime | None = None
    ) -> list[Event]:
        return self._events.query(deployment_id, since=since)

    def get_costs(self, deployment_id: str | None = None) -> list[CostRecord]:
        return self._store.get_cost_records(deployment_id)

    async def get_health(self, deployment_id: str) -> HealthStatus:
        deployment = self._store.get_deployment(deployment_id)
        return await health.run_checks(
            deployment, self._provider(deployment.provider), self._runtime()
        )

    async def get_logs(
        self, deployment_id: str, *, tail: int = 100, follow: bool = False
    ) -> Iterator[str]:
        deployment = self._store.get_deployment(deployment_id)
        if deployment.instance is None:
            return iter(())
        lines = await self._provider(deployment.provider).get_logs(
            deployment.instance.provider_instance_id, tail
        )
        return iter(lines)

    async def list_providers(self) -> list[ProviderInfo]:
        out: list[ProviderInfo] = []
        for name in PROVIDERS:
            try:
                caps = await self._provider(name).capabilities()
            except OrchestratorError:
                continue  # e.g. RunPod with no API key configured on this install
            out.append(ProviderInfo(name=name, capabilities=caps))
        return out

    async def estimate_cost(
        self, model_id: str, *, provider: str = "runpod", hours: float = 1.0
    ) -> CostEstimate:
        profile = self._catalog.get_profile(model_id)
        caps = await self._provider(provider).capabilities()
        gpu = _match_gpu(caps.gpu_types, profile.recommended_gpu)
        return CostEstimate(
            model_id=model_id,
            provider=provider,
            gpu_type=gpu.id,
            gpu_hourly_usd=gpu.hourly_usd,
            hours=hours,
            estimated_usd=round(gpu.hourly_usd * hours, 4),
        )

    # --- internals ------------------------------------------------------------------

    async def _drive(self, deployment: Deployment, until: set[DeploymentState]) -> Deployment:
        """Inline reconcile loop for the ``wait=True`` path and for stop/delete/restart. Paced by
        ``config.reconcile_interval`` so it can follow a real provider (minutes to READY) without
        hammering the API; tests set the interval to 0. The daemon owns the loop for non-blocking
        deploys -- this is the caller-blocks path."""
        provider = self._provider(deployment.provider)
        runtime = self._runtime()
        for _ in range(_MAX_DRIVE_TICKS):
            if deployment.observed_state in until:
                return deployment
            deployment = await reconcile_once(
                deployment,
                provider=provider,
                runtime=runtime,
                catalog=self._catalog,
                config=self._config,
                store=self._store,
                events=self._events,
            )
            if deployment.observed_state in until:
                return deployment
            if self._config.reconcile_interval:
                await asyncio.sleep(self._config.reconcile_interval)
        raise ReconcileError(
            f"deployment {deployment.id} did not settle within {_MAX_DRIVE_TICKS} ticks "
            f"(observed={deployment.observed_state.value})"
        )

    def _provider(self, name: str) -> Provider:
        return self._injected_provider or build_provider(self._config, name)

    def _runtime(self, name: str = "vllm") -> Runtime:
        return self._injected_runtime or build_runtime(name)

    def _emit(self, deployment: Deployment, kind: EventKind, payload: dict) -> None:
        self._events.emit(
            Event(
                id=f"evt-{uuid4().hex[:12]}",
                correlation_id=deployment.id,
                deployment_id=deployment.id,
                kind=kind,
                payload=payload,
            )
        )


# =====================================================================================
# Small pure helpers
# =====================================================================================


def _new_deployment_id() -> str:
    return f"dep-{uuid4().hex[:6]}"


def _apply_overrides(
    profile: RuntimeProfile, gpu: str | None, overrides: RuntimeOverrides | None
) -> RuntimeProfile:
    """Fold CLI overrides into a copy of the catalog profile. The profile decides by default; an
    explicit ``--gpu`` or ``overrides`` is the user overriding that decision (spec §7.1, §15)."""
    updates: dict[str, object] = {}
    chosen_gpu = gpu or (overrides.gpu if overrides else None)
    if chosen_gpu:
        updates["recommended_gpu"] = chosen_gpu
    if overrides and overrides.launch_args:
        updates["launch_args"] = {**profile.launch_args, **overrides.launch_args}
    if overrides and overrides.env:
        updates["env"] = {**profile.env, **overrides.env}
    return profile.model_copy(update=updates) if updates else profile


def _match_gpu(gpu_types: list[GPUType], wanted: str) -> GPUType:
    for gpu in gpu_types:
        if wanted in (gpu.id, gpu.provider_sku):
            return gpu
    raise ReconcileError(f"no GPU matching {wanted!r} in provider menu")


def build_provider(config: Config, name: str) -> Provider:
    """Construct a provider by name from config. Shared by the Orchestrator and the daemon so the
    RunPod-key wiring lives in exactly one place."""
    cls = PROVIDERS.get(name)
    if cls is None:
        raise ReconcileError(f"unknown provider {name!r}")
    if name == "runpod":
        key = config.runpod_api_key
        return cls(
            namespace=config.namespace,
            api_key=key.get_secret_value() if key is not None else None,
        )
    return cls(namespace=config.namespace)


def build_runtime(name: str = "vllm") -> Runtime:
    cls = RUNTIMES.get(name)
    if cls is None:
        raise ReconcileError(f"unknown runtime {name!r}")
    return cls()
