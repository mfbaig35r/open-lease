# GPU Orchestrator — Implementation Requirements

**Version:** 1.2 (implementation-ready; hardened per external review — ReconcileAction, idempotency/orphan sweep, provider endpoint resolution, state mapping, schema versioning, catalog validation metadata)
**Status:** Approved for build
**Build tool:** Claude Code (Opus) with human review
**First provider:** RunPod (extracted from existing image-indexing integration)

**Changelog — 1.2 (2026-07-03):** second-pass review folded into the body. Resolved reconcile-loop ownership in the CLI (§7.3), namespaced instance tags so the orphan sweep is safe across installs sharing one RunPod key (§7.5, §8.1, gauntlet #3b), stated DEGRADED restart cost honestly (§10), split migrations into DDL vs JSON-payload layers (§12), and made the proxy route on both catalog id and `hf_repo` (§13).

---

## 1. Vision

A Python platform that makes GPU infrastructure programmable. It provisions GPUs, deploys open-source LLMs, manages their lifecycle, and exposes inference through a unified OpenAI-compatible API.

One orchestration core; many thin interfaces:

- **CLI** (Typer) — Phase 1
- **REST** (FastAPI) — Phase 2
- **MCP** (FastMCP) — Phase 3
- **Swamp Extension** — Phase 4
- **SDKs** (Python/TypeScript) — Future

RunPod is Provider #1, not the product. The product is the orchestration layer. Long-term: a control plane for open-source LLMs across any GPU infrastructure (Modal, Vast, Lambda, local, cloud).

### Definition of done (v1)

```bash
pip install gpu-orchestrator
gpu deploy qwen3-32b
# → 2–5 minutes later: an OpenAI-compatible URL
gpu status | gpu logs | gpu costs | gpu stop <id>
```

A user who has never touched RunPod goes from zero to chatting with a self-hosted model in under five minutes. A maintainer reading `core/orchestrator.py::deploy_model` understands the entire lifecycle in one sitting.

---

## 2. Design Ethos (binding, not aspirational)

> Craft with clarity. Code with purpose — clean, readable, maintainable by design. Simplicity is a discipline. Elegance is structure, not ornament. Pythonic, in service of understanding. Built for teams, future maintainers, and developers at every level.

This ethos is operationalized as **hard rules** for the build. They apply to every module and every Claude Code prompt (see §17):

| Rule | In practice |
|---|---|
| **E1. No frameworks where a data structure will do** | Provider registry = ABC + a `dict`. Event system = append-only log. No plugin loaders, no entry-points discovery, no message bus. |
| **E2. Readable top-to-bottom** | `deploy_model()` must read like the deployment flow diagram. No dispatch-through-indirection for its own sake. |
| **E3. Every feature exists exactly once** | Interfaces call core functions. Zero business logic in CLI/API/MCP layers. |
| **E4. Explicit over clever** | Pydantic models are the contract. Type hints everywhere. No metaclass magic, no dynamic attribute tricks. |
| **E5. Failure handling is structural, not scattered** | The deployment engine is a reconcile loop, not a linear pipeline with bolted-on retries (§7.3). |
| **E6. Extensible means "a clear seam exists"** | An abstract base class and a well-named module is extensibility. A plugin framework is deferred until Provider #3 demands it. |
| **E7. Debuggable at 2am without AI assistance** | If a maintainer can't trace a failure with logs + code reading alone, the design is wrong. |

---

## 3. Scope

### 3.1 Phase 1 — Core + RunPod + vLLM + CLI (the vertical slice)

Everything needed to make v1 "definition of done" true:

- Core orchestration engine with reconcile loop
- `Provider` abstract interface + **RunPod implementation extracted from existing image-indexing project code**
- `Runtime` abstract interface + **vLLM implementation**
- Curated model catalog (10–15 models) with runtime profiles
- Health engine (poll loop)
- Cost tracking (simple: rate × duration, per-deployment)
- Append-only event log
- SQLite state store
- Typer CLI
- OpenAI-compatible inference proxy (local, points at deployments)
- Config engine (.env + TOML, documented precedence)
- Structured JSON logging with correlation IDs
- Mock provider for offline development + tests

### 3.2 Phase 2 — FastAPI

Thin REST layer over core. Every endpoint ≤ ~10 lines: validate → call core → serialize.

### 3.3 Phase 3 — FastMCP

Thin MCP tool layer over core. Same functions, agent-facing.

### 3.4 Phase 4 — Swamp Extension

Consumes the FastAPI layer. Deployment explorer, cost dashboard, log viewer, health dashboard, model chat, deploy wizard.

### 3.5 Explicitly deferred (seams exist; no code)

Multi-provider scheduling · autoscaling · LoRA/fine-tuning · batch jobs · multi-model endpoints · Kubernetes backend · auth/multi-tenancy · billing/quotas · Terraform provider · TypeScript SDK · web console · plugin discovery framework · cost recommendations across providers · SGLang/TGI/Ollama runtimes · HuggingFace metadata auto-sync.

Deferral rule: each deferred item must be addable **without modifying core interfaces** — if a Phase 1 design decision would foreclose one, flag it, don't build it.

---

## 4. Architecture

```
 Swamp Ext (P4)   FastMCP (P3)   FastAPI (P2)   CLI (P1)
      │               │              │             │
      └───────────────┴──────┬───────┴─────────────┘
                             │  (all call the same functions)
              ┌──────────────▼───────────────┐
              │     gpu_orchestrator.core    │
              │                              │
              │  Orchestrator (facade)       │
              │  ├─ Deployment Engine        │  reconcile loop
              │  ├─ Health Engine            │  poll loop
              │  ├─ Cost Tracker             │
              │  ├─ Model Catalog            │  curated profiles
              │  ├─ Event Log                │  append-only
              │  ├─ State Store              │  SQLite
              │  └─ Config                   │
              └──────┬───────────────┬───────┘
                     │               │
             Provider ABC      Runtime ABC
                     │               │
              ┌──────┴─────┐   ┌─────┴────┐
              │  RunPod    │   │  vLLM    │
              │  Mock      │   │  (Mock)  │
              └────────────┘   └──────────┘
```

Two seams, both plain abstract base classes:

- **Provider** — provisions compute; knows nothing about LLMs.
- **Runtime** — serves models; knows nothing about providers.

The deployment engine is the only place that composes them.

---

## 5. Repository Layout

```
gpu-orchestrator/
├── pyproject.toml              # uv-managed; package name: gpu_orchestrator
├── README.md
├── docs/
│   ├── architecture.md
│   ├── configuration.md        # precedence documented here
│   └── adding-a-provider.md
├── src/gpu_orchestrator/
│   ├── __init__.py             # public API: Orchestrator + domain models
│   ├── models.py               # ALL Pydantic domain models (single file, §6)
│   ├── config.py               # settings, precedence, credential loading
│   ├── events.py               # EventLog (append-only)
│   ├── store.py                # SQLite state store
│   ├── logging.py              # structured JSON logging, correlation IDs
│   ├── errors.py               # exception hierarchy
│   ├── core/
│   │   ├── orchestrator.py     # Orchestrator facade — the file that must read like the spec
│   │   ├── reconciler.py       # reconcile loop (§7.3)
│   │   ├── health.py           # health poll loop
│   │   ├── costs.py            # cost tracking
│   │   └── catalog.py          # model catalog + runtime profiles
│   ├── providers/
│   │   ├── base.py             # Provider ABC + PROVIDERS: dict[str, type[Provider]]
│   │   ├── runpod.py           # extracted from existing project
│   │   └── mock.py             # in-memory fake for tests/offline dev
│   ├── runtimes/
│   │   ├── base.py             # Runtime ABC + RUNTIMES dict
│   │   └── vllm.py
│   ├── proxy/
│   │   └── openai_proxy.py     # OpenAI-compatible passthrough (§13)
│   └── cli/
│       ├── main.py             # Typer app
│       └── render.py           # rich tables/output formatting only
├── catalog/
│   └── models.toml             # curated model catalog (data, not code)
└── tests/
    ├── unit/
    ├── contract/               # provider contract tests (run against mock AND runpod)
    ├── integration/            # real RunPod, opt-in via env var
    └── cli/                    # snapshot tests
```

Layout rules: no file over ~400 lines; `models.py` may run to ~600. If `models.py` exceeds that or becomes hard to navigate, split **only by domain** (`models/deployment.py`, `models/provider.py`, `models/runtime.py`, `models/events.py`) and re-export everything from `models/__init__.py` so the import surface never changes. Any other module wanting to split is a design review moment, not an automatic action.

---

## 6. Domain Models (`models.py`)

All Pydantic v2. These are the canonical contract shared by every interface. Fields listed are required unless marked optional; add `created_at`/`updated_at` timestamps to persisted entities.

```python
class DeploymentState(str, Enum):
    REQUESTED    = "requested"
    PROVISIONING = "provisioning"      # provider creating instance
    BOOTING      = "booting"           # instance up, container starting
    DOWNLOADING  = "downloading_model"
    STARTING     = "starting_server"
    READY        = "ready"
    DEGRADED     = "degraded"          # alive but unhealthy (§10)
    STOPPING     = "stopping"
    STOPPED      = "stopped"
    FAILED       = "failed"


class ReconcileAction(str, Enum):
    NONE              = "none"
    CREATE_INSTANCE   = "create_instance"
    DESTROY_INSTANCE  = "destroy_instance"
    WAIT_FOR_PROVIDER = "wait_for_provider"
    WAIT_FOR_RUNTIME  = "wait_for_runtime"
    ADOPT_INSTANCE    = "adopt_instance"     # found by tag after partial failure (§7.5)
    MARK_READY        = "mark_ready"
    MARK_DEGRADED     = "mark_degraded"
    MARK_FAILED       = "mark_failed"
    RETRY             = "retry"
```

`ReconcileAction` is first-class so that `next_step(deployment, observed) -> ReconcileAction` is **pure logic** — no network, no side effects — and therefore the most-tested function in the repo (§18).

**Schema versioning:** every persisted model (`Deployment`, `Event`, `CostRecord`) carries `schema_version: int = 1`. Migrations upgrade stored JSON documents by version; a document with an unknown version fails loudly at load, never silently.

| Model | Key fields |
|---|---|
| `ModelSpec` | `id` (e.g. `qwen3-32b`), `hf_repo`, `family`, `parameter_count`, `quantization`, `min_gpu_memory_gb`, `context_window`, `license`, capability flags (`chat`, `completion`, `embedding`, `vision`, `supports_tools`, `supports_reasoning`) |
| `RuntimeProfile` | `model_id`, `runtime` (`"vllm"`), `image`, `launch_args: dict`, `tensor_parallel`, `gpu_memory_utilization`, `recommended_gpu`, `min_disk_gb`, `env: dict` |
| `GPUType` | `id`, `name`, `memory_gb`, `hourly_usd`, `provider_sku` |
| `InstanceRequest` | `gpu_type`, `image`, `env`, `disk_gb`, `ports`, `volume: VolumeSpec \| None` |
| `Instance` | `provider_instance_id`, `provider`, `gpu_type`, `state` (provider-native string), `public_url: str \| None`, `ports` |
| `Deployment` | `id` (short, human-friendly: `dep-a1b2c3`), `model_id`, `provider`, `desired_state: DeploymentState`, `observed_state: DeploymentState`, `instance: Instance \| None`, `endpoint_url: str \| None`, `profile: RuntimeProfile`, `state_history: list[StateTransition]`, `failure: FailureInfo \| None` |
| `StateTransition` | `from_state`, `to_state`, `at`, `reason` |
| `FailureInfo` | `stage`, `message`, `retryable: bool`, `attempts` |
| `HealthStatus` | `status` (`healthy/degraded/failed/booting`), `checks: dict[str, CheckResult]`, `checked_at` |
| `CheckResult` | `ok: bool`, `latency_ms`, `detail` |
| `CostRecord` | `deployment_id`, `gpu_hourly_usd`, `started_at`, `stopped_at \| None`, `accrued_usd` (computed), `estimated_monthly_usd` (computed) |
| `Event` | `id`, `at`, `correlation_id`, `deployment_id \| None`, `kind: EventKind`, `payload: dict` |
| `VolumeSpec` | `size_gb`, `mount_path`, `persistent: bool` |
| `ProviderCapabilities` | `gpu_types: list[GPUType]`, `supports_volumes`, `supports_snapshots`, `regions` |

**EventKind (enum):** `deployment_requested`, `instance_created`, `image_pulled`, `model_download_started`, `model_download_completed`, `server_started`, `health_passed`, `deployment_ready`, `health_degraded`, `deployment_stopped`, `deployment_deleted`, `deployment_failed`, `reconcile_action`, `instance_adopted`, `orphan_detected`, `orphan_destroyed`, `cost_snapshot`.

The `desired_state` / `observed_state` pair on `Deployment` is deliberate — it is the data structure the reconcile loop operates on (§7.3).

---

## 7. Core Engine

### 7.1 Orchestrator facade (`core/orchestrator.py`)

The single entry point every interface uses. Public API — these signatures are the contract:

```python
class Orchestrator:
    def __init__(self, config: Config | None = None): ...

    def deploy_model(
        self,
        model_id: str,
        *,
        provider: str = "runpod",
        gpu: str | None = None,          # override; profile decides by default
        wait: bool = False,               # block until READY or FAILED
        overrides: RuntimeOverrides | None = None,
    ) -> Deployment: ...

    def stop_deployment(self, deployment_id: str) -> Deployment: ...
    def delete_deployment(self, deployment_id: str) -> None: ...
    def restart_deployment(self, deployment_id: str) -> Deployment: ...
    def get_deployment(self, deployment_id: str) -> Deployment: ...
    def list_deployments(self, *, include_stopped: bool = False) -> list[Deployment]: ...
    def get_logs(self, deployment_id: str, *, tail: int = 100, follow: bool = False) -> Iterator[str]: ...
    def get_health(self, deployment_id: str) -> HealthStatus: ...
    def get_costs(self, deployment_id: str | None = None) -> list[CostRecord]: ...
    def list_models(self) -> list[ModelSpec]: ...
    def list_providers(self) -> list[ProviderInfo]: ...
    def estimate_cost(self, model_id: str, *, provider: str = "runpod", hours: float = 1.0) -> CostEstimate: ...
    def events(self, deployment_id: str | None = None, *, since: datetime | None = None) -> list[Event]: ...
```

**Readability requirement (E2):** `deploy_model` must read as: validate model → resolve profile → check provider capacity/capabilities → create Deployment record (desired=READY) → emit event → hand to reconciler → return (or wait). Anyone reading it sees the whole flow.

### 7.2 State machine = vocabulary; reconciler = engine

`DeploymentState` is the shared vocabulary for events, CLI output, dashboards, and the timeline. It is **not** implemented as a linear pipeline of steps.

### 7.3 Reconcile loop (`core/reconciler.py`)

The heart of the system. Deliberately small (~target: under 150 lines of logic).

```
loop (per active deployment, every RECONCILE_INTERVAL seconds, default 10s):
    observed = observe(deployment)             # ask provider + runtime what actually exists
    if observed == deployment.desired_state: continue
    action: ReconcileAction = next_step(deployment, observed)   # PURE — no I/O
    execute(action, deployment)                # the only place side effects happen
    record_transition(deployment, observed, action)
    emit_event(...)
```

The `next_step` / `execute` split is a hard boundary: `next_step` is a pure function of `(deployment, observed)` and is exhaustively unit-tested against every state combination; `execute` is a thin dispatcher mapping each `ReconcileAction` to one provider/runtime call.

Rules:

- **One step per tick.** No multi-stage sequences inside a single reconcile pass. This is what makes interruption/resume free.
- **`observe()` trusts reality, not our records.** It queries the provider for instance state and the runtime health endpoint. If RunPod says the pod is gone, observed state is gone — regardless of what SQLite says.
- **Retryable failures** (instance creation timeout, transient API errors, download stall): retry with capped exponential backoff (default: 3 attempts, 10s → 60s). Attempts recorded in `FailureInfo`.
- **Terminal failures** (quota exceeded, invalid image, OOM on correctly-sized GPU): transition to `FAILED`, stop the instance to prevent cost bleed, keep logs.
- **Cost safety invariant:** a deployment in `FAILED` or `STOPPED` must never have a running instance. The reconciler enforces this every tick — it is the guard against zombie pods burning money.
- **Timeout budget per stage** (configurable): provisioning 5m, booting 5m, download 30m, starting 5m. Budget exceeded → retryable failure.
- The same `reconcile_once()` function is callable directly (tests, future serverless); the long-running loop wraps it.

**Loop ownership (must be resolved before build step 5).** The reconcile loop needs an owner that outlives a single CLI command. In a `pip install` CLI, `gpu deploy` without `--wait` returns and its process exits, so nothing observes the pod through its 2 to 5 minute path to READY: no instance-id adoption, no timeout-budget enforcement, no catch of a mid-boot death. Gauntlet test #3 (Ctrl-C mid-provisioning, restart, resume) also requires a loop that runs after restart. Three candidate models:

- **Daemon (recommended).** A background process (natural fit alongside `gpu proxy`) owns the loop; CLI commands are thin clients over the shared store. Revise the wording above and the §4 picture if chosen.
- **`--wait` mandatory for correctness.** The loop runs only while a command blocks. Contradicts the non-blocking deploy in the DoD.
- **Reconcile-on-invocation.** Each CLI command runs a few ticks. Works, but progress is lurchy and depends on the user polling.

Pick one here explicitly; it shapes the `Orchestrator` lifecycle.

### 7.4 Concurrency

Phase 1: single-process. SQLite with WAL mode; one reconciler thread; per-deployment lock so a deployment is never reconciled by two ticks at once. No distributed locking — deferred with the multi-tenancy roadmap item.

### 7.5 Idempotency & crash recovery

The dangerous window: the process dies **after** the provider created a pod but **before** the instance ID was persisted. Without mitigation, that pod is invisible to us and burns money — a direct violation of the cost-safety invariant.

Mitigations (all Phase 1):

1. **Every instance is tagged/named with a namespaced deployment ID** at creation: `gpu-orch-{namespace}-{deployment_id}` (e.g. `gpu-orch-laptop-dep-a1b2c3`). `namespace` defaults to a stable per-install id (hostname or a generated install id in config) and is configurable. This is non-optional in `InstanceRequest`. **The namespace exists because the RunPod account is global to the API key while state.db is per-install:** two installs sharing `RUNPOD_API_KEY` (laptop plus CI, two developers) must not see or sweep each other's pods. Without it, the orphan sweep below would destroy another install's healthy pods.
2. **Adoption:** if `deployment.instance is None`, the reconciler's `observe()` first calls `provider.find_instance_by_deployment_id(deployment.id)`. Found → `ADOPT_INSTANCE`: persist the ID and continue as normal. This makes `deploy_model` safe to interrupt at any point.
3. **Orphan sweep:** on a slower cadence (default every 5 minutes, and once at startup), the reconciler lists provider instances **matching this install's `gpu-orch-{namespace}-` prefix only** and compares against known deployments. An instance matching no deployment (or matching a `STOPPED`/`FAILED`/deleted one) is logged, an `orphan_detected` event is emitted, and the instance is destroyed after a grace period (default 2 minutes, to avoid racing an in-flight creation). This upgrades the cost-safety invariant from per-deployment to **global within the namespace**: no pod owned by this install survives without a live deployment record, and no pod owned by another install is ever touched.
4. Instance creation itself is guarded: before creating, `observe()` has already checked for an adoptable instance, so a reconcile retry never double-creates.

---

## 8. Provider Interface (`providers/base.py`)

```python
class Provider(ABC):
    name: ClassVar[str]

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...
    @abstractmethod
    def create_instance(self, request: InstanceRequest) -> Instance: ...
    @abstractmethod
    def get_instance(self, provider_instance_id: str) -> Instance | None: ...  # None = gone
    @abstractmethod
    def destroy_instance(self, provider_instance_id: str) -> None: ...          # idempotent
    @abstractmethod
    def list_instances(self) -> list[Instance]: ...                              # includes gpu-orch-* filter support
    @abstractmethod
    def find_instance_by_deployment_id(self, deployment_id: str) -> Instance | None: ...  # tag/name lookup (§7.5)
    @abstractmethod
    def resolve_endpoint_url(self, instance: Instance, port: int) -> str | None: ...      # None until routable
    @abstractmethod
    def get_logs(self, provider_instance_id: str, tail: int = 100) -> list[str]: ...

PROVIDERS: dict[str, type[Provider]] = {"runpod": RunPodProvider, "mock": MockProvider}
```

**Endpoint resolution is provider-owned.** URL shape (RunPod proxy URLs, port mappings, future providers' load balancers) is infrastructure knowledge. Runtimes only declare which port they serve on (vLLM: 8000); the provider turns `(instance, port)` into a public URL. `resolve_endpoint_url` returns `None` until the endpoint is actually routable — the reconciler treats that as `WAIT_FOR_PROVIDER`.

Rules:

- This layer provisions **compute**. It never imports from `runtimes/` or `catalog.py`. It knows nothing about LLMs.
- `destroy_instance` is idempotent — destroying a nonexistent instance is a no-op, not an error.
- `get_instance` returning `None` is the canonical "it's gone" signal the reconciler relies on.
- Volumes/snapshots: interface methods exist (`create_volume`, `attach_volume`) but RunPod implementation may raise `NotSupportedError` per feature in Phase 1 if not needed for the vertical slice.

### 8.1 RunPod provider (`providers/runpod.py`)

**Step one of the build: extract and adapt the existing RunPod integration from the image-indexing project.** That code has already survived contact with RunPod's API — auth, pod lifecycle, quirks. Conforming it to the ABC is the first pressure test of the interface; where it doesn't fit cleanly, **change the ABC**, not the working code's semantics.

Known adaptation work (batch → long-running serving):

- Store RunPod pod states **verbatim** in `Instance.state`; do not collapse ambiguous states.
- **Hard rule: provider-native states are never stored as `DeploymentState`.** Translation happens in exactly one auditable pure function: `map_to_observed_state(instance: Instance | None, runtime_health: HealthStatus | None) -> DeploymentState`, which lives in the core (the provider doesn't know `DeploymentState` exists). Every weird RunPod semantic gets handled — and unit-tested — in this one place.
- Implement `resolve_endpoint_url` from the pod's proxy URL / TCP port mapping.
- Name every pod `gpu-orch-{namespace}-{deployment_id}` and implement `find_instance_by_deployment_id` (and the sweep's list filter) scoped to this install's namespace (§7.5).
- Handle spot vs on-demand as an `InstanceRequest` field (default: on-demand for Phase 1; spot deferred).

### 8.2 Mock provider (`providers/mock.py`)

In-memory. Simulates state transitions on a configurable clock, injectable failures (creation timeout, mid-boot death, API errors). This is what unit tests and offline development run against.

### 8.3 Provider contract tests (`tests/contract/`)

One parametrized test suite that runs against **every** provider in `PROVIDERS` (mock always; runpod when `GPU_ORCH_INTEGRATION=1`). Covers: create → observe → destroy roundtrip; idempotent destroy; `get_instance` on unknown id returns `None`; **`find_instance_by_deployment_id` finds a tagged instance and returns `None` for unknown ids**; **`resolve_endpoint_url` returns a reachable URL for a running instance and `None` before routability**; log retrieval. This suite *is* the provider spec — a new provider passes it or it isn't done, and any new provider/runtime behavior must be represented here first.

---

## 9. Runtime Interface & vLLM

```python
class Runtime(ABC):
    name: ClassVar[str]

    @abstractmethod
    def build_instance_request(self, profile: RuntimeProfile, gpu: GPUType) -> InstanceRequest: ...
    @abstractmethod
    def health_check(self, endpoint_url: str) -> CheckResult: ...       # HTTP-level
    @abstractmethod
    def model_ready(self, endpoint_url: str, model_id: str) -> CheckResult: ...  # /v1/models
    @abstractmethod
    def download_progress(self, logs: list[str]) -> float | None: ...   # parse from logs, 0..1

RUNTIMES: dict[str, type[Runtime]] = {"vllm": VLLMRuntime}
```

vLLM implementation:

- `build_instance_request` composes image (`vllm/vllm-openai:<pinned-version>`), launch args from profile (`--model`, `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`), HF token env, port 8000.
- Health: `GET /health` (HTTP alive), `GET /v1/models` (model loaded).
- Download progress parsed from vLLM/HF log lines; `None` when unparseable — the reconciler falls back to timeout budget alone.

---

## 10. Health Engine (`core/health.py`)

Poll loop (default 30s) for deployments in `READY`/`DEGRADED`:

| Check | Source | Failure meaning |
|---|---|---|
| `instance_alive` | provider `get_instance` | pod gone → observed state collapses, reconciler takes over |
| `http_alive` | runtime `/health` | server down |
| `model_loaded` | runtime `/v1/models` | server up, model not serving |
| `latency` | timed `model_loaded` call | degradation signal |

Status derivation: all ok → `healthy`; `instance_alive` ok but any other check failing → `degraded` (emits `health_degraded`, sets `observed_state=DEGRADED`); `instance_alive` failing → reconciler handles it. Consecutive-failure threshold (default 3) before declaring degraded, to absorb flapping. Phase 1 does **not** auto-restart degraded deployments — it reports; `restart` is a user action. (Auto-heal policy is a deferred roadmap item; the seam is the reconciler.)

**Cost of remediation, stated honestly:** because `restart` is stop + redeploy the same profile (§15) on an ephemeral disk that re-downloads the model (§14), restarting a large deployment is a full cold-start outage (up to `startup_timeout_seconds`, e.g. ~2400s for a 65GB model). So in Phase 1, DEGRADED is report-only and the user's only remediation is expensive. This is acceptable until persistent cache volumes land (§14), which is the first enhancement that makes restart cheap.

---

## 11. Cost Tracking (`core/costs.py`)

Phase 1 is deliberately simple (E1): `accrued = gpu_hourly_usd × elapsed_hours`, started at instance creation, stopped at instance destruction. Persisted as `CostRecord`; `cost_snapshot` event hourly. `estimate_cost()` = profile's recommended GPU rate × hours. `gpu costs` shows per-deployment accrued + projected monthly. Per-token cost, bandwidth, storage, cross-provider recommendations: deferred.

**The invariant that matters:** the reconciler's cost-safety rule (§7.3) is the real cost feature — no orphaned pods.

---

## 12. Events, Store, Config, Logging

**Event log (`events.py`):** append-only. `EventLog.emit(event)` writes to SQLite + structured log line. `EventLog.query(deployment_id, since, kind)`. No subscribers, no bus, no callbacks in Phase 1. (Swamp's timeline view reads this table via the API later.)

**State store (`store.py`):** SQLite (WAL), file at `~/.gpu-orchestrator/state.db` (configurable). Tables: `deployments` (JSON document column keyed by id — Pydantic in/out), `events`, `cost_records`. No ORM; `sqlite3` stdlib + thin typed helpers.

**Two-layer migrations, with clear ownership.** The `deployments` table is a JSON document column, so evolution happens at two levels and each owns exactly one thing: (1) **numbered SQL scripts applied on startup** own table/DDL shape — new tables, columns, indexes; (2) **per-document `schema_version` upgraders** (§6) own the JSON payload — a document read at version N is upgraded to the current version in code before use, and an unknown version fails loudly (§6). SQL migrations never rewrite JSON payloads; payload upgraders never touch DDL.

**Config (`config.py`):** Pydantic Settings. Precedence (highest wins): CLI flags → environment variables (`GPU_ORCH_*`, plus `RUNPOD_API_KEY`, `HF_TOKEN`) → `./gpu-orchestrator.toml` → `~/.gpu-orchestrator/config.toml` → defaults. Precedence documented in `docs/configuration.md`. Secrets never logged, never in state.db.

**Logging (`logging.py`):** structured JSON to stderr/file; human-readable rich output is a CLI rendering concern only. Every operation gets a `correlation_id` propagated through events and provider calls. OpenTelemetry: not integrated in Phase 1, but spans map 1:1 to reconcile actions and provider calls, so the hook points are the existing function boundaries — no restructuring needed later.

---

## 13. OpenAI-Compatible Inference Proxy (`proxy/openai_proxy.py`)

The highest-value user-facing feature — ships in Phase 1.

- `gpu proxy` starts a local server (default `localhost:8080`) exposing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`.
- Routes by request `model` field → matching READY deployment's endpoint. `/v1/models` lists all READY deployments. **Routing accepts both the catalog/deployment id and the profile's `hf_repo`** (e.g. `qwen3-32b` and `Qwen/Qwen3-32B`), because vLLM advertises the HF repo path at its own `/v1/models` and OpenAI clients commonly echo that string back; matching only the catalog id would miss those requests. This is exact-match on those two known keys per deployment — not fuzzy aliasing, which stays deferred.
- Pure passthrough: streams SSE, preserves status codes, adds `x-gpu-orch-deployment-id` header. **No payload normalization in Phase 1 — if vLLM supports the route, the proxy forwards it byte-for-byte; if not, the proxy returns the upstream error unchanged.** All four routes share one forwarding code path; the proxy's only intelligence is model-name → deployment routing. Tool-call normalization, request transformation, and model aliasing beyond the two exact keys above (catalog/deployment id and `hf_repo`) are explicitly deferred — this must not quietly become a compatibility layer.
- Also usable directly: `deployment.endpoint_url` is always shown in `gpu status` for users who want to skip the proxy.
- Implementation: small ASGI app (starlette + httpx streaming). This is infrastructure plumbing, not an interface layer — it lives in core's package, and Phase 2's FastAPI mounts it rather than reimplementing it.

---

## 14. Model Catalog (`catalog/models.toml` + `core/catalog.py`)

Curated data file — models are **data, not code**. Phase 1 ships 10–15 entries across Qwen3, Llama 3.x, Mistral/Ministral, DeepSeek (R1 distills), Gemma, Phi — each with `ModelSpec` + `RuntimeProfile` (image, GPU recommendation, tensor parallel, launch args) validated by Pydantic on load.

Selection rules per entry are pre-decided by the profile: users never choose GPU memory, image, CUDA version, or launch flags unless overriding (`--gpu`, `--set key=value`). Adding a model = adding a TOML block. HuggingFace metadata auto-sync: deferred.

**Validation metadata is part of every entry** — a profile without it doesn't ship:

```toml
[models.qwen3-32b.validation]
validated_at       = "2026-07-03"
validated_provider = "runpod"
validated_gpu      = "A100-80GB"
validated_image    = "vllm/vllm-openai:v0.9.1"
startup_timeout_seconds = 2400
notes = "First launch downloads ~65GB; expect slow cold start."
```

`validated_at` older than the pinned image = stale profile; bumping an image version requires re-validation (§18). `startup_timeout_seconds` overrides the default download-stage budget per model (§7.3).

**Phase 1 disk behavior — stated honestly:** Phase 1 uses ephemeral disks sized by `min_disk_gb`. **Every deployment re-downloads its model.** Cold starts for large models are slow and bandwidth-heavy; that is a known, accepted Phase 1 cost. Persistent model cache volumes are the first Phase 1.1/2 enhancement (the `VolumeSpec` seam already exists in the Provider interface for exactly this).

---

## 15. CLI (`cli/`)

Typer + rich. Should feel like `docker`/`kubectl`/`uv`: short commands, sensible defaults, `--json` on every read command for scripting.

| Command | Behavior |
|---|---|
| `gpu deploy <model> [--provider] [--gpu] [--wait] [--set k=v]` | Deploys; prints deployment id + live state ticker if `--wait` |
| `gpu status [<id>]` | Table: id, model, state, GPU, endpoint, uptime, accrued cost |
| `gpu stop <id>` / `gpu delete <id>` | Stop keeps record; delete removes it (confirms unless `--yes`) |
| `gpu restart <id>` | Stop + redeploy same profile |
| `gpu logs <id> [--tail N] [--follow]` | Provider/runtime logs |
| `gpu health <id>` | Check-by-check health table |
| `gpu models [--family]` | Catalog listing with GPU recommendation + est. $/hr |
| `gpu providers` | Configured providers + capability summary |
| `gpu costs [<id>]` | Accrued + projected monthly per deployment; total footer |
| `gpu events <id> [--since]` | Deployment timeline |
| `gpu estimate <model> [--hours]` | Cost estimate without deploying |
| `gpu proxy [--port]` | Start the OpenAI proxy |
| `gpu chat <id>` | Minimal REPL against a READY deployment (thin `httpx` loop) |
| `gpu config` | Show effective config + source of each value (secrets masked) |

CLI rules (E3): every command body is: parse args → call one `Orchestrator` method → render. Rendering lives in `cli/render.py`. Zero orchestration logic in the CLI. Exit codes: 0 success, 1 error, 2 usage. Errors print one clear human sentence + hint; stack traces only with `--debug`.

---

## 16. Phases 2–4 (specified now, built later)

### Phase 2 — FastAPI

- `create_app(orchestrator)` factory; routes mirror the Orchestrator API 1:1: `POST /deployments`, `GET /deployments`, `GET /deployments/{id}`, `DELETE /deployments/{id}`, `POST /deployments/{id}/restart`, `GET /deployments/{id}/logs|health|events`, `GET /models`, `GET /providers`, `GET /costs`, `POST /estimate`.
- Mounts the OpenAI proxy at `/v1/*`.
- Request/response bodies are the §6 Pydantic models — no parallel schema definitions.
- Auth: single static bearer token from config (multi-tenancy deferred).
- Acceptance: no route function exceeds ~10 lines.

### Phase 3 — FastMCP

Tools = thin wrappers over the same Orchestrator methods: `deploy_model`, `stop_deployment`, `restart_deployment`, `delete_deployment`, `list_models`, `list_deployments`, `get_deployment`, `deployment_logs`, `deployment_health`, `provider_status`, `estimate_cost`, `chat_completion` (routes through proxy logic). Tool descriptions written for agent consumption; destructive tools (`delete`) require explicit confirmation parameter.

### Phase 4 — Swamp Extension

Consumes Phase 2's REST API exclusively — no direct core imports. Views: Deployment Explorer, Model Catalog + Deploy Wizard, Cost Dashboard, Log Viewer, Health Dashboard, Deployment Timeline (reads events), Model Chat (via proxy). Requirements for Phase 4 get their own doc once Phases 1–2 are live; the API contract from Phase 2 is the dependency, and it's already fully specified by §7.1.

---

## 17. Building with Claude Code — Binding Prompt Constraints

The ethos (§2) must be enforced in generation, or the generator will happily produce the heavier version. Include these constraints verbatim in the project's `CLAUDE.md`:

```
ARCHITECTURE CONSTRAINTS (non-negotiable):
- No plugin frameworks, no entry-points discovery, no dynamic loading.
  Providers/runtimes are: an ABC + a module-level dict.
- No event bus, no pub/sub, no callbacks. Events are appended to a log.
- No ORM. sqlite3 stdlib with thin typed helpers.
- The reconciler takes ONE step per tick. Never chain stages in one pass.
- deploy_model() must read top-to-bottom like the deployment flow.
- No file over ~400 lines except models.py. No function over ~50 lines.
- Interfaces (CLI/API/MCP) contain zero business logic: parse → core call → render.
- Type hints everywhere. Pydantic v2 for all domain models.
- If a simpler structure serves, use it. Cleverness is a defect.
- When existing RunPod code conflicts with the Provider ABC, propose changing
  the ABC — the working code has authority over the speculative interface.

RELIABILITY CONSTRAINTS (non-negotiable):
- next_step() is a pure function: no network calls, no side effects, ever.
- Every function that calls a provider API emits an event or structured log
  carrying the correlation_id.
- Every code path that creates a provider instance has a corresponding
  cleanup path, and that cleanup path has a test.
- Every created instance is named gpu-orch-{deployment_id}. No exceptions.
- Never catch broad Exception without re-raising or converting to a typed
  OrchestratorError subclass.
- Every persisted Pydantic model round-trips through the store in a test,
  and carries schema_version.
- Provider-native states are never assigned to DeploymentState directly;
  all translation goes through map_to_observed_state().
- Any new provider/runtime behavior is represented in a contract test first.
```

Build order (each step reviewed before the next):

1. `models.py` + `errors.py` + `config.py` — the contract first.
   1b. **Immediately generate fixtures** (`tests/fixtures/deployments.py`, `catalog.py`, `events.py`) — canonical example objects in every state. These stabilize everything downstream and double as documentation.
2. `store.py` + `events.py` + `logging.py` — infrastructure (round-trip fixtures through the store here).
3. `providers/base.py` + **extract RunPod code** + `mock.py` + contract tests.
4. `runtimes/base.py` + `vllm.py` + catalog with 3 models.
5. `core/reconciler.py` + `core/orchestrator.py` — reviewed line-by-line; this is the heart.
6. `core/health.py` + `core/costs.py`.
7. CLI.
8. OpenAI proxy.
9. **Real-GPU validation gauntlet (§18)** — before expanding catalog or starting Phase 2.
10. Catalog to 10–15 models (each validated).
11. Phase 2 FastAPI → Phase 3 MCP → Phase 4 Swamp.

Human review is mandatory (E7) at steps 3, 5, and 9 minimum.

---

## 18. Testing & Validation

### Test suites

- **Unit** — `next_step()` (given desired+observed → expected `ReconcileAction`) is the most heavily tested code in the repo, exhaustively covering the state matrix including adoption and orphan cases; `map_to_observed_state()` against every known RunPod state; catalog validation (including required validation metadata); config precedence; cost math; state transitions; store round-trips for every persisted model.
- **Contract** — §8.3 provider suite, parametrized over mock + runpod.
- **Integration** (opt-in, `GPU_ORCH_INTEGRATION=1`, costs real money) — smallest catalog model, full lifecycle.
- **CLI snapshot** — rendered output of each command against the mock provider.
- **Failure injection** — mock provider scripted to die mid-boot, stall downloads, throw transient API errors; assert reconciler recovery and the cost-safety invariant.

### Real-GPU validation gauntlet (gate before Phase 2)

Run against real RunPod; all must pass:

1. Deploy smallest model → READY → chat via proxy → stop. Clean lifecycle, correct cost record.
2. Deploy → kill the pod from the RunPod console mid-download → reconciler detects, retries or fails cleanly, **no orphaned pod**.
3. Deploy → `Ctrl-C` the orchestrator process mid-provisioning → restart it → reconciler resumes from stored state, **adopting the tagged pod if the instance ID was never persisted**.
3b. Create a pod named `gpu-orch-{namespace}-dep-zzzzzz` in this install's namespace (no deployment record) → orphan sweep detects it, emits `orphan_detected`, destroys it after the grace period. Also create one in a different namespace → sweep leaves it untouched.
4. Deploy a model onto a deliberately undersized GPU (override) → OOM → terminal FAILED, instance destroyed, logs preserved.
5. Two concurrent deployments → both READY, proxy routes to each correctly by model name.
6. 24-hour soak: one deployment held READY; health loop stable, no flapping, cost accrual correct.

### Phase 1 acceptance criteria

- [ ] Zero-to-chat in under 5 minutes for a new user with only a RunPod key.
- [ ] Gauntlet passes.
- [ ] `deploy_model` + `reconciler.py` readable top-to-bottom by a mid-level Python dev with no repo context (the ethos test — actually hand it to someone).
- [ ] Every catalog entry deployed successfully at least once.
- [ ] Full test suite green offline against mock provider.
- [ ] `docs/adding-a-provider.md` written by implementing against the contract tests, not from imagination.

---

## 19. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Provider ABC wrong until Provider #2 | Extract working RunPod code first; ABC bends to it. Contract tests document actual semantics. Accept the ABC will change — keep it small. |
| Zombie pods burning money | Cost-safety invariant enforced every reconcile tick; gauntlet test #2 and #4 verify. |
| Generated code outpacing understanding | Build order with mandatory review gates; ethos constraints in CLAUDE.md; the "hand it to a mid-level dev" acceptance test. |
| vLLM image/version drift breaking profiles | Pin image versions per profile; catalog entries individually validated; version bump = re-validation. |
| Health check flapping | Consecutive-failure threshold; degraded ≠ auto-restart in Phase 1. |
| Scope regrowth ("just add SGLang real quick") | §3.5 deferral rule: seams exist, code doesn't, until a phase explicitly opens. |

---

## 20. Deferred Roadmap (unchanged from original vision)

Multi-provider scheduling & failover · automatic GPU selection (cost/latency/availability) · LoRA training & deployment · batch inference · scheduled deployments · autoscaling policies · multi-model endpoints · shared model caches · fine-tuning pipelines · distributed inference · Kubernetes backend · auth & multi-tenancy · billing & quotas · team workspaces · Terraform provider · Python/TS SDKs · web console · additional runtimes (SGLang, TGI, Ollama) · additional providers (Modal, Vast, Lambda, local) · HF metadata sync · cost recommendations.

Every item must remain addable without core interface changes. The architecture anticipates them through exactly two seams (Provider, Runtime), one facade (Orchestrator), one vocabulary (DeploymentState), and one contract (models.py).

---

*The platform earns its ambitions by making RunPod deployments boringly reliable first. Everything else is a thin door into the same room.*
