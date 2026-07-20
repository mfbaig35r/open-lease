"""All Pydantic v2 domain models: the canonical contract shared by every interface.

This is the one file allowed to run long (spec §5). It is grouped, top to bottom, as:
enums, model/runtime catalog types, provider/compute types, deployment types, health types,
cost/event types, and the small facade DTOs used by the Orchestrator API (spec §7.1).

Persisted entities (`Deployment`, `Event`, `CostRecord`) carry ``schema_version`` so the store can
upgrade old documents and fail loudly on unknown versions (spec §6, §12).
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, computed_field, field_validator

SCHEMA_VERSION = 1


def _utcnow() -> datetime:
    return datetime.now(UTC)


# =====================================================================================
# Enums: the shared vocabulary (spec §6)
# =====================================================================================


class DeploymentState(StrEnum):
    """The lifecycle vocabulary used by events, CLI output, dashboards, and the timeline.

    This is a vocabulary, NOT a linear pipeline. The reconciler drives movement between states
    by comparing desired vs observed; it never "runs through" these in order (spec §7.2).
    """

    REQUESTED = "requested"
    PROVISIONING = "provisioning"  # provider creating instance
    BOOTING = "booting"  # instance up, container starting
    DOWNLOADING = "downloading_model"
    STARTING = "starting_server"
    READY = "ready"
    DEGRADED = "degraded"  # alive but unhealthy (spec §10)
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ReconcileAction(StrEnum):
    """The one action a reconcile tick decides on.

    ``next_step(deployment, observed) -> ReconcileAction`` is a PURE function so this enum is the
    boundary between "decide" (pure, exhaustively tested) and "execute" (the only side effects).
    """

    NONE = "none"
    CREATE_INSTANCE = "create_instance"
    DESTROY_INSTANCE = "destroy_instance"
    WAIT_FOR_PROVIDER = "wait_for_provider"
    WAIT_FOR_RUNTIME = "wait_for_runtime"
    ADOPT_INSTANCE = "adopt_instance"  # found by tag after partial failure (spec §7.5)
    MARK_READY = "mark_ready"
    MARK_DEGRADED = "mark_degraded"
    MARK_FAILED = "mark_failed"
    RETRY = "retry"


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    BOOTING = "booting"


class CloudType(StrEnum):
    """Provider capacity tier. Phase 1 defaults to on-demand; spot deferred (spec §8.1)."""

    ON_DEMAND = "on_demand"
    SPOT = "spot"


class Posture(StrEnum):
    """A scheduled deployment's desired running level at a point in time (capacity plan, Tier A1).

    Two levels in Phase 1: ON (full capacity) and OFF (torn down, config retained). WARM_STANDBY is
    deferred to replicas (Tier B1); with a single replica it would be identical to ON, so it earns a
    value of its own only once a deployment can hold more than one instance.
    """

    ON = "on"
    OFF = "off"


class BudgetWindow(StrEnum):
    """The recurring period a spend ceiling resets over (capacity plan, Tier A2)."""

    DAILY = "daily"
    MONTHLY = "monthly"


class BudgetAction(StrEnum):
    """What happens when a budget's ceiling is reached (Tier A2). ``warn`` only emits an event;
    ``stop`` tears down the in-scope deployment(s) for the rest of the window (a budget stop
    outranks a schedule); ``block_new`` refuses new deploys in scope while over budget but leaves
    running work alone."""

    WARN = "warn"
    STOP = "stop"
    BLOCK_NEW = "block_new"


class EventKind(StrEnum):
    DEPLOYMENT_REQUESTED = "deployment_requested"
    INSTANCE_CREATED = "instance_created"
    IMAGE_PULLED = "image_pulled"
    MODEL_DOWNLOAD_STARTED = "model_download_started"
    MODEL_DOWNLOAD_COMPLETED = "model_download_completed"
    SERVER_STARTED = "server_started"
    HEALTH_PASSED = "health_passed"
    DEPLOYMENT_READY = "deployment_ready"
    HEALTH_DEGRADED = "health_degraded"
    DEPLOYMENT_STOPPED = "deployment_stopped"
    DEPLOYMENT_DELETED = "deployment_deleted"
    DEPLOYMENT_FAILED = "deployment_failed"
    RECONCILE_ACTION = "reconcile_action"
    INSTANCE_ADOPTED = "instance_adopted"
    ORPHAN_DETECTED = "orphan_detected"
    ORPHAN_DESTROYED = "orphan_destroyed"
    COST_SNAPSHOT = "cost_snapshot"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_RELEASED = "budget_released"


# =====================================================================================
# Model catalog + runtime profiles (spec §6, §14)
# =====================================================================================


class ModelSpec(BaseModel):
    """What a model IS. Curated data, one per catalog entry (spec §14)."""

    id: str  # e.g. "qwen3-32b"
    hf_repo: str  # e.g. "Qwen/Qwen3-32B"
    family: str
    parameter_count: str  # human string, e.g. "32B"
    quantization: str | None = None
    min_gpu_memory_gb: int
    context_window: int
    license: str
    # capability flags
    chat: bool = True
    completion: bool = False
    embedding: bool = False
    vision: bool = False
    supports_tools: bool = False
    supports_reasoning: bool = False


class ValidationMetadata(BaseModel):
    """Proof a profile was actually launched. A profile without this does not ship (spec §14)."""

    validated_at: str  # ISO date, e.g. "2026-07-03"
    validated_provider: str
    validated_gpu: str
    validated_image: str
    startup_timeout_seconds: int  # overrides the default download-stage budget (§7.3)
    notes: str = ""


class RuntimeProfile(BaseModel):
    """How a model is served: image, GPU, launch args. The profile decides so users don't (§14)."""

    model_id: str
    runtime: str = "vllm"
    image: str
    launch_args: dict[str, str] = Field(default_factory=dict)
    tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.90
    recommended_gpu: str
    min_disk_gb: int
    env: dict[str, str] = Field(default_factory=dict)
    validation: ValidationMetadata


class RuntimeOverrides(BaseModel):
    """User overrides from the CLI (`--gpu`, `--set k=v`). Empty by default (spec §7.1, §15)."""

    gpu: str | None = None
    launch_args: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


# =====================================================================================
# Provider / compute types (spec §6, §8)
# =====================================================================================


class GPUType(BaseModel):
    id: str
    name: str
    memory_gb: int
    hourly_usd: float
    provider_sku: str


class VolumeSpec(BaseModel):
    size_gb: int
    mount_path: str
    persistent: bool = False


class InstanceRequest(BaseModel):
    """What the runtime asks the provider to create.

    ``name`` is the ``gpu-orch-{namespace}-{deployment_id}`` tag and is non-optional: it is the
    hook every idempotency, adoption, and orphan-sweep guarantee hangs on (spec §7.5).
    """

    name: str
    gpu_type: str
    # GPUs per pod. A tensor-parallel deploy (profile.tensor_parallel > 1) requests more than one;
    # it must equal vLLM's --tensor-parallel-size or the runtime fails asking for GPUs it lacks.
    gpu_count: int = 1
    image: str
    env: dict[str, str] = Field(default_factory=dict)
    disk_gb: int
    ports: list[int] = Field(default_factory=list)
    # Container command/args (e.g. the vLLM server invocation). The provider maps this to its
    # container-entrypoint mechanism (RunPod: dockerEntrypoint). Empty = use the image default.
    command: list[str] = Field(default_factory=list)
    cloud_type: CloudType = CloudType.ON_DEMAND
    volume: VolumeSpec | None = None
    # Persistent model cache (spec §14): attach a pre-created network volume by id, mounted at
    # ``volume_mount_path``. ``data_center_id`` pins the pod to the volume's region, since a network
    # volume is region-locked. All None unless caching is enabled.
    network_volume_id: str | None = None
    volume_mount_path: str | None = None
    data_center_id: str | None = None


class Instance(BaseModel):
    """A live (or recently live) provider instance. ``state`` is the provider-native string, stored
    verbatim; translation to DeploymentState happens only in map_to_observed_state (spec §8.1)."""

    provider_instance_id: str
    provider: str
    gpu_type: str
    gpu_count: int = 1  # GPUs backing this pod (for cost = rate x count and status display)
    state: str  # provider-native, verbatim
    public_url: str | None = None
    ports: list[int] = Field(default_factory=list)


class VolumeInfo(BaseModel):
    """A persistent network volume (the model cache). Returned by ``list_volumes`` (spec §14)."""

    id: str
    name: str
    size_gb: int
    data_center_id: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimated_monthly_usd(self) -> float:
        # RunPod standard storage: $0.07/GB/month for the first 1 TB (investigated 2026-07-04).
        return round(self.size_gb * 0.07, 2)


class GpuAvailability(BaseModel):
    """Per-data-center availability of a GPU type (spec §8). Read-only; used to pick a data center
    with capacity before pinning a cache volume, and to warn before a deploy that would fail."""

    data_center_id: str
    gpu_type_id: str  # provider-native sku, e.g. "NVIDIA A100 80GB PCIe"
    available: bool
    stock_status: str | None = None  # provider-native, e.g. "High" / "Medium" / "Low"


class ProviderCapabilities(BaseModel):
    gpu_types: list[GPUType] = Field(default_factory=list)
    supports_volumes: bool = False
    supports_snapshots: bool = False
    regions: list[str] = Field(default_factory=list)


class ProviderInfo(BaseModel):
    """Returned by ``list_providers`` (spec §7.1)."""

    name: str
    capabilities: ProviderCapabilities


# =====================================================================================
# Deployment types (spec §6)
# =====================================================================================


class StateTransition(BaseModel):
    from_state: DeploymentState
    to_state: DeploymentState
    at: datetime = Field(default_factory=_utcnow)
    reason: str = ""


class FailureInfo(BaseModel):
    stage: DeploymentState
    message: str
    retryable: bool
    attempts: int = 0
    last_attempt_at: datetime | None = None  # when the last attempt failed; drives retry backoff


class ScheduleRule(BaseModel):
    """One window of a schedule: on these weekdays, between ``start`` and ``end`` (wall-clock in the
    schedule's timezone), the deployment takes ``posture``. ``start`` later than ``end`` is an
    overnight window that wraps past midnight (e.g. 22:00-06:00)."""

    days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])  # 0=Mon .. 6=Sun
    start: str  # "HH:MM", inclusive
    end: str  # "HH:MM", exclusive
    posture: Posture

    @field_validator("start", "end")
    @classmethod
    def _valid_hhmm(cls, v: str) -> str:
        time.fromisoformat(v)  # raises ValueError on a malformed "HH:MM"
        return v

    @field_validator("days")
    @classmethod
    def _valid_days(cls, v: list[int]) -> list[int]:
        if any(d < 0 or d > 6 for d in v):
            raise ValueError("days must be 0 (Mon) through 6 (Sun)")
        return v


class Schedule(BaseModel):
    """A deployment's operating schedule (capacity plan, Tier A1). The daemon resolves the posture
    in force from the wall-clock each tick and drives ``desired_state`` to match.
    ``default_posture`` applies whenever no rule's window matches. Validators keep a malformed
    schedule from ever being persisted, so daemon resolution never raises on bad data."""

    timezone: str = "UTC"
    default_posture: Posture = Posture.OFF
    rules: list[ScheduleRule] = Field(default_factory=list)

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:  # KeyError subclass; convert so it reads as validation
            raise ValueError(f"unknown timezone {v!r}") from exc
        return v


class Deployment(BaseModel):
    """The record the reconcile loop operates on.

    The ``desired_state`` / ``observed_state`` pair is the whole point: the reconciler compares
    them each tick and takes one step to close the gap (spec §7.3).
    """

    schema_version: int = SCHEMA_VERSION
    id: str  # short, human-friendly: "dep-a1b2c3"
    model_id: str
    provider: str
    # hf_repo + context_window make a deployment self-contained: reconcile and the proxy can build
    # and route it without a catalog lookup, which lets an ad-hoc `--hf-repo` deploy run with no
    # catalog entry. Empty on records created before this existed; the catalog is the fallback.
    hf_repo: str = ""
    context_window: int = 0
    desired_state: DeploymentState
    observed_state: DeploymentState
    profile: RuntimeProfile
    instance: Instance | None = None
    endpoint_url: str | None = None
    # Best-effort model-download fraction (0..1) during bring-up, parsed from runtime logs when the
    # provider exposes them; None when unavailable (e.g. RunPod has no log API). Display-only.
    download_progress: float | None = None
    state_history: list[StateTransition] = Field(default_factory=list)
    failure: FailureInfo | None = None
    # Optional operating schedule (Tier A1). When set, the daemon resolves the posture in force at
    # each tick and drives ``desired_state`` to match, so capacity follows a plan (business hours,
    # overnight shutdown) rather than a variable meter. None = manual control (deploy/stop own it).
    schedule: Schedule | None = None
    # Held down by an exceeded stop-budget (Tier A2). Set by the daemon's budget loop; takes
    # precedence over the schedule (a ceiling outranks a plan), so a held deployment stays STOPPED
    # even inside an ON window. Cleared when the budget window resets or the budget is lifted.
    budget_hold: bool = False
    # Concurrency envelope (Tier A3), enforced by the OpenAI proxy. ``max_concurrency`` None means
    # unlimited (no gate). Up to ``max_queue`` requests wait up to ``queue_timeout_s`` for a slot;
    # anything beyond gets a 429. A slot is held for the whole request, streamed response included.
    max_concurrency: int | None = None
    max_queue: int = 0
    queue_timeout_s: float = 30.0
    # Count of unexpected runtime deaths (a created pod that vanished before reaching READY, e.g. an
    # OOM crash loop) since the last healthy READY. Distinct from ``failure.attempts``, which counts
    # only provider CREATE errors: a successful create clears ``failure`` next tick, so it can never
    # cap a runtime that keeps crashing after coming up. This survives that reset and trips
    # next_step to terminal FAILED at ``retry_max_attempts`` (cost-safety, spec §7.3).
    runtime_failures: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("max_concurrency")
    @classmethod
    def _positive_concurrency(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_concurrency must be at least 1 (or None for unlimited)")
        return v

    @field_validator("max_queue")
    @classmethod
    def _nonnegative_queue(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_queue cannot be negative")
        return v

    @field_validator("queue_timeout_s")
    @classmethod
    def _positive_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("queue_timeout_s must be greater than 0")
        return v


# =====================================================================================
# Health types (spec §6, §10)
# =====================================================================================


class CheckResult(BaseModel):
    ok: bool
    latency_ms: float | None = None
    detail: str = ""


class HealthStatus(BaseModel):
    status: HealthState
    checks: dict[str, CheckResult] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=_utcnow)


# =====================================================================================
# Cost + event types (spec §6, §11, §12)
# =====================================================================================


class Budget(BaseModel):
    """A spend ceiling over a recurring window (capacity plan, Tier A2). ``deployment_id`` None
    makes it account-wide (every deployment's cost); a value scopes it to one deployment. Spend is
    computed from ``CostRecord`` data (GPU time), so a budget is a policy layer with no new
    accounting. Window boundaries are UTC in Phase 1. ``on_exceed`` decides enforcement;
    ``warn_fraction`` is the soft threshold that emits an early warning event."""

    schema_version: int = SCHEMA_VERSION
    id: str  # short, human-friendly: "bud-a1b2c3"
    deployment_id: str | None = None  # None = account-wide
    window: BudgetWindow
    limit_usd: float
    on_exceed: BudgetAction = BudgetAction.WARN
    warn_fraction: float = 0.8

    @field_validator("limit_usd")
    @classmethod
    def _positive_limit(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("limit_usd must be greater than 0")
        return v

    @field_validator("warn_fraction")
    @classmethod
    def _fraction_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("warn_fraction must be in (0, 1]")
        return v


class CostRecord(BaseModel):
    """Simple Phase 1 cost: rate x elapsed. ``accrued_usd`` accrues until ``stopped_at`` is set."""

    schema_version: int = SCHEMA_VERSION
    deployment_id: str
    gpu_hourly_usd: float
    started_at: datetime
    stopped_at: datetime | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accrued_usd(self) -> float:
        end = self.stopped_at or _utcnow()
        elapsed_hours = max(0.0, (end - self.started_at).total_seconds() / 3600.0)
        return round(self.gpu_hourly_usd * elapsed_hours, 4)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimated_monthly_usd(self) -> float:
        return round(self.gpu_hourly_usd * 24 * 30, 2)


class CostEstimate(BaseModel):
    """Returned by ``estimate_cost`` without deploying (spec §7.1, §15)."""

    model_id: str
    provider: str
    gpu_type: str
    gpu_hourly_usd: float
    hours: float
    estimated_usd: float


class UsageSummary(BaseModel):
    """Token throughput and cost for a deployment (spec §11 extension). ``tokens_per_sec`` is the
    utilization signal (tokens served per second of rented time); ``cost_per_mtok`` is the crossover
    metric against per-token API pricing. Both fall to 0/None with no traffic yet."""

    deployment_id: str
    model_id: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    accrued_usd: float
    uptime_seconds: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tokens_per_sec(self) -> float:
        return round(self.total_tokens / self.uptime_seconds, 1) if self.uptime_seconds > 0 else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_per_mtok(self) -> float | None:
        if not self.total_tokens:
            return None
        return round(self.accrued_usd / self.total_tokens * 1_000_000, 2)


class Event(BaseModel):
    """Append-only. No subscribers, no bus (spec §12)."""

    schema_version: int = SCHEMA_VERSION
    id: str
    at: datetime = Field(default_factory=_utcnow)
    correlation_id: str
    deployment_id: str | None = None
    kind: EventKind
    payload: dict = Field(default_factory=dict)
