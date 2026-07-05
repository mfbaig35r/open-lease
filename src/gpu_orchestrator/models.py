"""All Pydantic v2 domain models: the canonical contract shared by every interface.

This is the one file allowed to run long (spec §5). It is grouped, top to bottom, as:
enums, model/runtime catalog types, provider/compute types, deployment types, health types,
cost/event types, and the small facade DTOs used by the Orchestrator API (spec §7.1).

Persisted entities (`Deployment`, `Event`, `CostRecord`) carry ``schema_version`` so the store can
upgrade old documents and fail loudly on unknown versions (spec §6, §12).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

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


class Deployment(BaseModel):
    """The record the reconcile loop operates on.

    The ``desired_state`` / ``observed_state`` pair is the whole point: the reconciler compares
    them each tick and takes one step to close the gap (spec §7.3).
    """

    schema_version: int = SCHEMA_VERSION
    id: str  # short, human-friendly: "dep-a1b2c3"
    model_id: str
    provider: str
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
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


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


class Event(BaseModel):
    """Append-only. No subscribers, no bus (spec §12)."""

    schema_version: int = SCHEMA_VERSION
    id: str
    at: datetime = Field(default_factory=_utcnow)
    correlation_id: str
    deployment_id: str | None = None
    kind: EventKind
    payload: dict = Field(default_factory=dict)
