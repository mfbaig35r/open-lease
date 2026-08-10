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
from collections.abc import Callable, Iterator
from datetime import datetime
from uuid import uuid4

import httpx

from ..config import Config
from ..errors import BudgetExceededError, ModelNotFoundError, OrchestratorError, ReconcileError
from ..events import EventLog
from ..logging import correlation_context, get_logger
from ..models import (
    AutoscalePolicy,
    Budget,
    BudgetAction,
    BudgetWindow,
    CostEstimate,
    CostRecord,
    Deployment,
    DeploymentState,
    Event,
    EventKind,
    GpuAvailability,
    GPUType,
    HealthStatus,
    ModelSpec,
    ProviderInfo,
    RuntimeOverrides,
    RuntimeProfile,
    Schedule,
    UsageSummary,
    ValidationMetadata,
    VolumeInfo,
    _utcnow,
)
from ..providers.base import PROVIDERS, Provider
from ..runtimes.base import RUNTIMES, Runtime
from ..store import Store
from . import autoscale, batch, budgets, health, modelinfo, usage
from .catalog import Catalog, load_catalog
from .reconciler import reconcile_once

_log = get_logger("orchestrator")

# A wait/drive safety cap: reconcile_once is one step per tick, so a deployment reaches a terminal
# state in a bounded number of ticks. This guards the inline ``wait=True`` path against a stuck
# provider; the daemon uses the real clock instead.
_MAX_DRIVE_TICKS = 200

_TERMINAL_READY = {DeploymentState.READY, DeploymentState.FAILED}
_TERMINAL_STOP = {DeploymentState.STOPPED, DeploymentState.FAILED}

# Defaults for an ad-hoc (--hf-repo) deploy that has no catalog recipe.
# Pinned to a recent stable vLLM release (reproducible), bumped deliberately rather than tracking
# `latest`. Old pins silently fail to load newer model architectures, so keep this current; override
# per deploy with `--image` when a model needs a different (or nightly) build.
_ADHOC_IMAGE = "vllm/vllm-openai:v0.25.1"
# Per-runtime default image for an ad-hoc deploy. Without this, picking a runtime would also mean
# hand-passing its image, and llamacpp would stay unreachable outside a curated catalog entry.
_ADHOC_IMAGES = {
    "vllm": _ADHOC_IMAGE,
    "llamacpp": "ghcr.io/ggml-org/llama.cpp:server-cuda",
}
_ADHOC_DISK_GB = 60
_ADHOC_STARTUP_SECONDS = 1800  # generous: an unknown model may be large / slow to download


class Orchestrator:
    def __init__(
        self,
        config: Config | None = None,
        *,
        catalog: Catalog | None = None,
        provider: Provider | None = None,
        runtime: Runtime | None = None,
        hf_transport: httpx.BaseTransport | None = None,
    ) -> None:
        # ``provider``/``runtime`` injection is the seam tests use to run against the mock provider
        # without touching config or the network; production leaves them None and builds by name.
        self._config = config or Config()
        self._store = Store(self._config.state_db)
        self._events = EventLog(self._store)
        self._catalog = catalog or load_catalog()
        self._injected_provider = provider
        self._injected_runtime = runtime
        # Same seam as provider/runtime: lets tests size a model without touching the hub.
        self._hf_transport = hf_transport

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
        """Deploy a catalog model by id (raises ModelNotFoundError if unknown)."""
        spec = self._catalog.get_spec(model_id)
        profile = _apply_overrides(self._catalog.get_profile(model_id), gpu, overrides)
        return await self._launch(
            model_id=spec.id,
            provider=provider,
            profile=profile,
            hf_repo=spec.hf_repo,
            context_window=spec.context_window,
            wait=wait,
        )

    async def deploy_adhoc(
        self,
        *,
        hf_repo: str,
        gpu: str | None = None,
        provider: str = "runpod",
        context_window: int = 0,
        image: str | None = None,
        disk_gb: int | None = None,
        gpu_count: int | None = None,
        runtime: str = "vllm",
        wait: bool = False,
        overrides: RuntimeOverrides | None = None,
    ) -> Deployment:
        """Deploy any vLLM-servable HF repo with no catalog entry. The engine is model-neutral; the
        catalog only supplies tuned recipes. ``context_window`` 0 lets the engine auto-detect. The
        deployment carries its own hf_repo, so reconcile and the proxy need no catalog lookup.

        ``gpu`` is optional (issue #24): when omitted, the model's size is read from its Hugging
        Face metadata and the cheapest sufficient GPU is chosen, including a multi-GPU pod when no
        single card fits. An explicit ``gpu`` always wins, and an unreadable model raises rather
        than guessing. ``runtime`` picks the serving engine (issue #23); its default image follows
        from that choice."""
        if runtime not in RUNTIMES:
            raise ReconcileError(f"unknown runtime {runtime!r} (have: {sorted(RUNTIMES)})")
        pinned = gpu is not None  # explicit choice; a hub-sized GPU stays substitutable
        if gpu is None:
            gpu, sized_count = await self._size_from_hub(hf_repo, provider)
            gpu_count = gpu_count or sized_count
        gpu_count = gpu_count or 1
        img = image or _ADHOC_IMAGES.get(runtime, _ADHOC_IMAGE)
        profile = RuntimeProfile(
            model_id=_adhoc_model_id(hf_repo),
            runtime=runtime,
            image=img,
            recommended_gpu=gpu,
            gpu_pinned=pinned,
            tensor_parallel=gpu_count,
            min_disk_gb=disk_gb or _ADHOC_DISK_GB,
            validation=ValidationMetadata(
                validated_at="",
                validated_provider=provider,
                validated_gpu=gpu,
                validated_image=img,
                startup_timeout_seconds=_ADHOC_STARTUP_SECONDS,
                notes="ad-hoc --hf-repo deploy (no catalog entry)",
            ),
        )
        profile = _apply_overrides(profile, None, overrides)  # fold in --set launch_args
        return await self._launch(
            model_id=profile.model_id,
            provider=provider,
            profile=profile,
            hf_repo=hf_repo,
            context_window=context_window,
            wait=wait,
        )

    async def _launch(
        self,
        *,
        model_id: str,
        provider: str,
        profile: RuntimeProfile,
        hf_repo: str,
        context_window: int,
        wait: bool,
    ) -> Deployment:
        """Shared tail of deploy_model / deploy_adhoc: create the record, persist, emit, drive."""
        self._enforce_budget_admission()
        deployment = Deployment(
            id=_new_deployment_id(),
            model_id=model_id,
            provider=provider,
            hf_repo=hf_repo,
            context_window=context_window,
            desired_state=DeploymentState.READY,
            observed_state=DeploymentState.REQUESTED,
            profile=profile,
        )
        with correlation_context(deployment.id):
            self._store.save_deployment(deployment)
            self._emit(deployment, EventKind.DEPLOYMENT_REQUESTED, {"model_id": model_id})
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

    async def scale(self, model_id: str, replicas: int, *, wait: bool = False) -> list[Deployment]:
        """Ensure ``replicas`` deployments serve ``model_id`` (Tier B). The proxy load-balances over
        whatever is READY, so a replica is just another deployment of the same model: the reconciler
        and next_step are untouched. Scale up by cloning an existing member (its profile, limits,
        and schedule); scale down by stopping the newest surplus, keeping the established
        ones. Requires an existing member to clone when scaling up from zero (use
        `gpu deploy --replicas` for the initial pool)."""
        if replicas < 0:
            raise ReconcileError("replicas cannot be negative")
        members = sorted(
            (
                d
                for d in self._store.list_deployments(include_stopped=False)
                if d.model_id == model_id and d.desired_state != DeploymentState.STOPPED
            ),
            key=lambda d: d.created_at,
        )
        if replicas > len(members):
            if not members:
                raise ReconcileError(
                    f"no active deployment of {model_id!r} to scale; deploy one first"
                )
            template = members[-1]
            for _ in range(replicas - len(members)):
                members.append(await self._launch_clone(template, wait=wait))
        elif replicas < len(members):
            for surplus in members[replicas:]:  # stop the newest surplus, keep the established ones
                await self.stop_deployment(surplus.id)
            members = members[:replicas]
        return members

    async def _launch_clone(self, template: Deployment, *, wait: bool) -> Deployment:
        """Create one more replica from a template deployment (new id, same model/profile/limits/
        schedule), persist and emit it, and optionally drive it to READY."""
        clone = autoscale.replica_from(template, _new_deployment_id())
        with correlation_context(clone.id):
            self._store.save_deployment(clone)
            self._emit(clone, EventKind.DEPLOYMENT_REQUESTED, {"model_id": clone.model_id})
            if wait:
                clone = await self._drive(clone, _TERMINAL_READY)
        return clone

    async def set_schedule(self, deployment_id: str, schedule: Schedule) -> Deployment:
        """Attach or replace a deployment's operating schedule (capacity plan, Tier A1). The running
        daemon resolves the posture each tick and drives capacity to match; nothing is brought up
        or torn down in this call. The schedule is authoritative over manual stop/start until
        cleared."""
        deployment = self._store.get_deployment(deployment_id)
        deployment.schedule = schedule
        self._store.save_deployment(deployment)
        return deployment

    async def clear_schedule(self, deployment_id: str) -> Deployment:
        """Remove a deployment's schedule, returning it to manual control (deploy/stop own desired
        state again). Leaves the current desired_state as-is."""
        deployment = self._store.get_deployment(deployment_id)
        deployment.schedule = None
        self._store.save_deployment(deployment)
        return deployment

    async def set_limits(
        self,
        deployment_id: str,
        *,
        max_concurrency: int | None,
        max_queue: int = 0,
        queue_timeout_s: float = 30.0,
    ) -> Deployment:
        """Set (or clear, with ``max_concurrency=None``) a deployment's concurrency envelope (Tier
        A3), enforced by the OpenAI proxy. Re-validated through the model so a bad value is rejected
        now, not on the next load. A live proxy picks up the change on its next restart."""
        current = self._store.get_deployment(deployment_id)
        updated = Deployment.model_validate(
            {
                **current.model_dump(),
                "max_concurrency": max_concurrency,
                "max_queue": max_queue,
                "queue_timeout_s": queue_timeout_s,
            }
        )
        self._store.save_deployment(updated)
        return updated

    async def set_budget(
        self,
        *,
        limit_usd: float,
        window: BudgetWindow,
        on_exceed: BudgetAction = BudgetAction.WARN,
        deployment_id: str | None = None,
        warn_fraction: float = 0.8,
    ) -> Budget:
        """Create a spend ceiling (capacity plan, Tier A2). ``deployment_id`` None is account-wide.
        The running daemon evaluates it each tick and enforces ``on_exceed``; a `block_new` ceiling
        is also checked here before every new deploy."""
        budget = Budget(
            id=_new_budget_id(),
            deployment_id=deployment_id,
            window=window,
            limit_usd=limit_usd,
            on_exceed=on_exceed,
            warn_fraction=warn_fraction,
        )
        self._store.save_budget(budget)
        return budget

    async def remove_budget(self, budget_id: str) -> bool:
        """Delete a budget; returns True if one was removed. Any budget_hold it set clears on the
        daemon's next budget tick."""
        return self._store.delete_budget(budget_id)

    async def set_autoscale(
        self,
        *,
        model_id: str,
        max_replicas: int,
        target_rpm_per_replica: float,
        min_replicas: int = 1,
    ) -> AutoscalePolicy:
        """Create or replace a model's autoscaling policy (capacity plan, Tier B2). The running
        daemon keeps the replica count matched to served request rate within these bounds."""
        policy = AutoscalePolicy(
            model_id=model_id,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target_rpm_per_replica=target_rpm_per_replica,
        )
        self._store.save_autoscale_policy(policy)
        return policy

    async def remove_autoscale(self, model_id: str) -> bool:
        """Delete a model's autoscaling policy; returns True if one was removed."""
        return self._store.delete_autoscale_policy(model_id)

    def list_autoscale(self) -> list[AutoscalePolicy]:
        return self._store.list_autoscale_policies()

    def _enforce_budget_admission(self) -> None:
        """Refuse a new deploy while an account `block_new` budget is over its ceiling (a policy
        boundary). A per-deployment budget does not gate a brand-new deploy (nothing to bind to)."""
        blocker = budgets.admission_blocked(
            self._store.list_budgets(), self._store.get_cost_records(), _utcnow()
        )
        if blocker is not None:
            raise BudgetExceededError(
                f"account budget {blocker.id} is over its {blocker.window.value} ceiling of "
                f"${blocker.limit_usd:.2f}; new deploys are blocked until the window resets"
            )

    # --- reads ----------------------------------------------------------------------

    def get_deployment(self, deployment_id: str) -> Deployment:
        return self._store.get_deployment(deployment_id)

    def list_deployments(self, *, include_stopped: bool = False) -> list[Deployment]:
        return self._store.list_deployments(include_stopped=include_stopped)

    def list_budgets(self) -> list[Budget]:
        return self._store.list_budgets()

    def budget_status(self) -> list[budgets.BudgetStatus]:
        """Each budget with its spend so far this window (for `gpu budget list`)."""
        now = _utcnow()
        return [
            budgets.evaluate(b, self._store.get_cost_records(b.deployment_id), now)
            for b in self._store.list_budgets()
        ]

    def list_models(self) -> list[ModelSpec]:
        return self._catalog.list_models()

    def events(
        self, deployment_id: str | None = None, *, since: datetime | None = None
    ) -> list[Event]:
        return self._events.query(deployment_id, since=since)

    def get_costs(self, deployment_id: str | None = None) -> list[CostRecord]:
        return self._store.get_cost_records(deployment_id)

    def record_proxy_usage(self, deployment_id: str, body: bytes) -> None:
        """Tally tokens for a forwarded proxy response (spec §11). Called by the OpenAI proxy on a
        successful metered response; a no-op when the body carries no usage."""
        usage.record(self._store, deployment_id, body)

    def get_usage(self, deployment_id: str | None = None) -> list[UsageSummary]:
        """Per-deployment token throughput + cost-per-token. Includes stopped deployments, so a
        torn-down deployment's totals stay visible after the fact."""
        deployments = self._store.list_deployments(include_stopped=True)
        if deployment_id is not None:
            deployments = [d for d in deployments if d.id == deployment_id]
        return [usage.summary(self._store, d) for d in deployments]

    async def run_batch(
        self,
        deployment_id: str,
        items: list[batch.BatchItem],
        *,
        concurrency: int = 64,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 3,
        on_done: Callable[[batch.BatchResult], None] | None = None,
    ) -> list[batch.BatchResult]:
        """Fan a list of prompts out over a READY deployment (spec §13). Throughput-bound batch
        work; results are metered like any proxy traffic so a run shows up in `gpu usage`."""
        deployment = self._store.get_deployment(deployment_id)  # raises if unknown
        if deployment.observed_state != DeploymentState.READY or not deployment.endpoint_url:
            raise ReconcileError(
                f"{deployment_id} is not READY (state: {deployment.observed_state.value})"
            )
        served = deployment.hf_repo or {m.id: m.hf_repo for m in self.list_models()}.get(
            deployment.model_id, deployment.model_id
        )
        return await batch.run(
            self._store,
            deployment,
            served,
            items,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            on_done=on_done,
        )

    async def get_health(self, deployment_id: str) -> HealthStatus:
        deployment = self._store.get_deployment(deployment_id)
        return await health.run_checks(
            deployment, self._provider(deployment.provider), self._runtime(deployment)
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

    async def list_volumes(self, *, provider: str = "runpod") -> list[VolumeInfo]:
        return await self._provider(provider).list_volumes()

    async def gpu_availability(
        self, *, model_id: str | None = None, gpu_type: str | None = None, provider: str = "runpod"
    ) -> list[GpuAvailability]:
        """Per-data-center GPU availability. Pass ``model_id`` to resolve to that model's GPU, or
        ``gpu_type`` (a catalog id or provider SKU) to check a specific GPU; ``gpu_type`` wins when
        both are given, matching a ``--gpu`` deploy override."""
        if gpu_type is None and model_id:
            gpu_type = self._catalog.get_profile(model_id).recommended_gpu
        if gpu_type is not None:
            caps = await self._provider(provider).capabilities()
            gpu_type = _match_gpu(caps.gpu_types, gpu_type).provider_sku
        return await self._provider(provider).gpu_availability(gpu_type)

    async def delete_volume(self, volume_id: str, *, provider: str = "runpod") -> None:
        await self._provider(provider).delete_volume(volume_id)

    async def estimate_cost(
        self,
        model_id: str,
        *,
        provider: str = "runpod",
        hours: float = 1.0,
        gpu: str | None = None,
    ) -> CostEstimate:
        """Price a model without deploying, in both metrics: hourly rate and cost per million
        tokens. Per-token is filled in only from throughput this install has actually measured for
        this model on this GPU, and is left absent otherwise (issue #25).

        Throughput precedence: this install's own traffic first, because that reflects the real
        workload; then the catalog baseline measured when the profile was validated, so a fresh
        install gets an answer before it has served anything; then nothing.

        Works for ad-hoc models with no catalog entry: an explicit ``gpu`` wins, then the catalog,
        then the GPU a previous deployment of this model used."""
        wanted = gpu or self._estimate_gpu(model_id)
        caps = await self._provider(provider).capabilities()
        matched = _match_gpu(caps.gpu_types, wanted)
        observed = usage.observed_throughput(
            self._store, model_id, gpu_names={matched.id, matched.provider_sku, wanted}
        ) or self._catalog_throughput(model_id, matched)
        return CostEstimate(
            model_id=model_id,
            provider=provider,
            gpu_type=matched.id,
            gpu_hourly_usd=matched.hourly_usd,
            hours=hours,
            estimated_usd=round(matched.hourly_usd * hours, 4),
            observed_tokens_per_sec=observed[0] if observed else None,
            throughput_basis=observed[1] if observed else None,
        )

    def _catalog_throughput(self, model_id: str, gpu: GPUType) -> tuple[float, str] | None:
        """The throughput recorded when this profile was validated, if it was measured on this GPU.

        Prefers the concurrent figure because it is semantically the same quantity the local-usage
        path reports (aggregate tokens over rented time); the single-stream figure is a different
        question and says so in its provenance. Refuses to reuse a measurement from other hardware,
        for the same reason the local path does.
        """
        try:
            validation = self._catalog.get_profile(model_id).validation
        except ModelNotFoundError:
            return None
        # The GPU the THROUGHPUT was measured on, which is not always the one the launch was
        # validated on. Falling back to validated_gpu keeps older entries working.
        measured_gpu = validation.throughput_gpu or validation.validated_gpu
        if measured_gpu not in (gpu.id, gpu.provider_sku):
            return None
        when = validation.throughput_measured_at or validation.validated_at
        stamp = f"measured {when} on {measured_gpu}"
        if validation.tokens_per_sec_concurrent:
            at = validation.measured_concurrency
            return (
                validation.tokens_per_sec_concurrent,
                f"catalog baseline at concurrency {at}, {stamp}",
            )
        if validation.tokens_per_sec:
            return validation.tokens_per_sec, f"catalog baseline, single stream, {stamp}"
        return None

    def _estimate_gpu(self, model_id: str) -> str:
        """The GPU to price ``model_id`` on: catalog first, then this install's own history so an
        ad-hoc deploy can be estimated too. Raises rather than defaulting to some arbitrary GPU,
        since a silently-wrong GPU makes the whole estimate wrong."""
        try:
            return self._catalog.get_profile(model_id).recommended_gpu
        except ModelNotFoundError:
            pass
        for deployment in self._store.list_deployments(include_stopped=True):
            if deployment.model_id == model_id:
                return deployment.profile.recommended_gpu
        raise ModelNotFoundError(
            f"No catalog entry or past deployment for {model_id!r}; pass a gpu to price it"
        )

    # --- internals ------------------------------------------------------------------

    async def _drive(self, deployment: Deployment, until: set[DeploymentState]) -> Deployment:
        """Inline reconcile loop for the ``wait=True`` path and for stop/delete/restart. Paced by
        ``config.reconcile_interval`` so it can follow a real provider (minutes to READY) without
        hammering the API; tests set the interval to 0. The daemon owns the loop for non-blocking
        deploys -- this is the caller-blocks path."""
        provider = self._provider(deployment.provider)
        runtime = self._runtime(deployment)
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

    async def _size_from_hub(self, hf_repo: str, provider: str) -> tuple[str, int]:
        """Pick a GPU (and count) for an uncurated model from its hub metadata.

        Raises with an actionable message when the model cannot be sized, rather than defaulting
        to some GPU: an under-sized guess OOMs minutes into a paid cold start, and an over-sized one
        silently overcharges. Neither is a good trade against one explicit flag."""
        token = self._config.hf_token.get_secret_value() if self._config.hf_token else None
        profile = await modelinfo.fetch_profile(hf_repo, token=token, transport=self._hf_transport)
        if profile is None:
            raise ModelNotFoundError(
                f"Could not read metadata for {hf_repo!r} (gated, missing, or unreadable). "
                "Pass an explicit gpu."
            )
        caps = await self._provider(provider).capabilities()
        chosen = modelinfo.select_gpu(profile, caps.gpu_types)
        if chosen is None:
            needed = modelinfo.required_vram_gb(profile)
            detail = f"needs ~{needed} GB" if needed else "has no published weight size"
            raise ModelNotFoundError(
                f"No GPU on {provider} fits {hf_repo!r} ({detail}). Pass an explicit gpu."
            )
        gpu, count = chosen
        _log.info(
            "sized model from hub metadata",
            extra={
                "hf_repo": hf_repo,
                "weight_gb": profile.weight_gb,
                "gpu": gpu.id,
                "gpu_count": count,
            },
        )
        return gpu.id, count

    def _provider(self, name: str) -> Provider:
        return self._injected_provider or build_provider(self._config, name)

    def _runtime(self, deployment: Deployment) -> Runtime:
        """The engine this deployment's profile asks for (spec §9). ``profile.runtime`` was inert
        while vLLM was the only implementation; honoring it is what makes runtime #2 reachable."""
        return self._injected_runtime or build_runtime(deployment.profile.runtime)

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


def _new_budget_id() -> str:
    return f"bud-{uuid4().hex[:6]}"


def _adhoc_model_id(hf_repo: str) -> str:
    """A display/routing id for an ad-hoc model, from the repo's last segment: Qwen/Qwen3-14B ->
    qwen3-14b. The deployment stores the full hf_repo separately; the proxy routes by both."""
    return hf_repo.rsplit("/", 1)[-1].lower()


def _apply_overrides(
    profile: RuntimeProfile, gpu: str | None, overrides: RuntimeOverrides | None
) -> RuntimeProfile:
    """Fold CLI overrides into a copy of the catalog profile. The profile decides by default; an
    explicit ``--gpu`` or ``overrides`` is the user overriding that decision (spec §7.1, §15)."""
    updates: dict[str, object] = {}
    chosen_gpu = gpu or (overrides.gpu if overrides else None)
    if chosen_gpu:
        updates["recommended_gpu"] = chosen_gpu
        updates["gpu_pinned"] = True  # a named GPU is a decision, not a recommendation
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
