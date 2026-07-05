# Architecture

open-lease turns "run an open LLM on a GPU" into a small, auditable state machine. This is the map;
the authoritative detail is [`requirements/gpu-orchestrator-requirements.md`](../requirements/gpu-orchestrator-requirements.md).

## The shape

```
  gpu CLI  ─┐
  (FastAPI) ─┼─▶  Orchestrator (facade, §7.1)  ─▶  reconcile loop  ─▶  Provider  (compute)
  (MCP)    ─┘         │                                              └▶  Runtime   (model server)
                     store.py (SQLite)  +  events  +  costs  +  health
```

Two seams, both plain ABCs:

- **Provider** (`providers/base.py`) provisions compute and knows nothing about LLMs. It creates and
  destroys instances, lists them, resolves endpoint URLs, and reports GPU availability. RunPod is
  the real one; `mock.py` is an in-memory fake used by every test.
- **Runtime** (`runtimes/base.py`) serves a model on compute and knows nothing about providers. It
  turns a model + profile + GPU into a compute request and answers health questions about a URL.
  vLLM is the real one.

The **deployment engine** (`core/reconciler.py`) is the only place the two seams meet. Everything
else is a thin interface: parse input, call one `Orchestrator` method, render the result.

## The reconcile loop is the engine

A `Deployment` carries a `desired_state` and an `observed_state`. The loop closes the gap between
them, one step per tick:

```
observed = observe(deployment)            # ask the provider + runtime what actually exists
action   = next_step(deployment, observed)   # PURE: decide the one action to take
execute(action, deployment)               # the ONLY place side effects happen
record + emit + persist
```

Two properties fall out of this design:

- **Interruption and crash recovery are free.** Nothing is held in memory between ticks; state lives
  in SQLite. Kill the process mid-provision, restart, and the next tick re-observes reality and
  continues. If a pod was created but its id was never persisted, `observe` adopts it by its
  deterministic name (`gpu-orch-{namespace}-{deployment_id}`).
- **The decision is pure and exhaustively tested.** `next_step(deployment, observed) -> ReconcileAction`
  has no I/O and no clock, so it is unit-tested against the full desired × observed × failure matrix.
  `map_to_observed_state` is the single place a provider-native pod state becomes a `DeploymentState`.

`DeploymentState` (REQUESTED → PROVISIONING → BOOTING → DOWNLOADING → STARTING → READY, plus
DEGRADED / STOPPING / STOPPED / FAILED) is a shared vocabulary for events, the CLI, and the
timeline. It is not a linear pipeline; the reconciler moves between states by comparing desired vs
observed.

## Cost-safety is an invariant, not a feature

The scariest failure in GPU tooling is a pod that bills forever. Three mechanisms enforce that no pod
outlives its purpose:

1. **Terminal states destroy instances.** A deployment in FAILED or STOPPED must never have a running
   instance; the reconciler enforces this every tick.
2. **Adoption** recovers a pod whose id was never persisted, so a crash mid-create cannot orphan it.
3. **Orphan sweep** (a daemon loop) lists provider instances matching this install's
   `gpu-orch-{namespace}-` prefix, compares against known deployments, and destroys any with no live
   record after a grace period. This is scoped to the namespace, so it never touches another
   install's pods.

Retries use capped exponential backoff; per-stage timeout budgets convert a stalled boot or download
into a retryable failure rather than an endless bill.

## Who owns the loop

The reconcile loop needs an owner that outlives a single CLI command. That owner is a **daemon**
(`core/daemon.py`): a single asyncio process running four loops over the shared store —
`tick_reconcile`, `tick_health`, `tick_sweep` (orphans), `tick_retention` (event pruning) — plus an
hourly cost snapshot. Each loop is built on a callable core (`reconcile_once`,
`HealthMonitor.check_once`) so the cores stay unit-testable and the loops stay thin. `gpu deploy
--wait` drives the same `reconcile_once` inline; `gpu up`/`gpu down` manage the daemon and proxy as
detached processes.

Phase 1 is single-process (SQLite with a process-wide lock, one download lease per model). Multiple
daemon processes or a multi-tenant server is the Postgres seam, deferred.

## Persistence

`store.py` is SQLite (WAL, `sqlite3` stdlib, no ORM). Tables: `deployments` (the record the loop
operates on, a JSON document per row), `events` (append-only log), `cost_records`, and
`download_locks` (a per-model lease so two concurrent cold deploys cannot corrupt a shared cache).
Schema evolves in two layers: numbered DDL migrations (tracked by `PRAGMA user_version`) own table
shape; per-document `schema_version` upgraders own the JSON payload and fail loudly on an unknown
version.

## The inference proxy

`proxy/openai_proxy.py` is a small starlette + httpx app whose only intelligence is routing: it maps
a request's `model` field to a READY deployment's endpoint (matching both the catalog id and the HF
repo) and forwards the request with streaming, preserving status codes and adding an
`x-gpu-orch-deployment-id` header. It lives in the core package so a future FastAPI layer mounts it
rather than reimplementing it.
