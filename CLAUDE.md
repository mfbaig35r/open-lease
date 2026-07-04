# GPU Orchestrator: build guide for Claude Code

The authoritative spec is `requirements/gpu-orchestrator-requirements.md` (v1.2). Read the
relevant section before writing a module. This file carries the non-negotiable constraints and
the build order; the spec carries the detail.

## What this is

A Python platform that makes GPU infrastructure programmable: provision GPUs, deploy open-source
LLMs (vLLM), manage their lifecycle via a reconcile loop, and expose OpenAI-compatible inference.
One core (`gpu_orchestrator.core`), thin interfaces over it (CLI, REST, MCP, Swamp). RunPod is
Provider #1. Package: `gpu_orchestrator`, uv-managed.

Two seams, both plain ABCs: **Provider** (provisions compute, knows nothing about LLMs) and
**Runtime** (serves models, knows nothing about providers). The deployment engine is the only
place that composes them. The reconcile loop operates on a `desired_state` / `observed_state` pair.

## ARCHITECTURE CONSTRAINTS (non-negotiable)

- No plugin frameworks, no entry-points discovery, no dynamic loading.
  Providers/runtimes are: an ABC + a module-level dict.
- No event bus, no pub/sub, no callbacks. Events are appended to a log.
- No ORM. sqlite3 stdlib with thin typed helpers.
- The reconciler takes ONE step per tick. Never chain stages in one pass.
- deploy_model() must read top-to-bottom like the deployment flow.
- No file over ~400 lines except models.py. No function over ~50 lines.
- Interfaces (CLI/API/MCP) contain zero business logic: parse -> core call -> render.
- Type hints everywhere. Pydantic v2 for all domain models.
- If a simpler structure serves, use it. Cleverness is a defect.
- When existing RunPod code conflicts with the Provider ABC, propose changing
  the ABC: the working code has authority over the speculative interface.

## RELIABILITY CONSTRAINTS (non-negotiable)

- next_step() is a pure function: no network calls, no side effects, ever.
- Every function that calls a provider API emits an event or structured log
  carrying the correlation_id.
- Every code path that creates a provider instance has a corresponding
  cleanup path, and that cleanup path has a test.
- Every created instance is named gpu-orch-{namespace}-{deployment_id}. No exceptions.
- Never catch broad Exception without re-raising or converting to a typed
  OrchestratorError subclass.
- Every persisted Pydantic model round-trips through the store in a test,
  and carries schema_version.
- Provider-native states are never assigned to DeploymentState directly;
  all translation goes through map_to_observed_state().
- Any new provider/runtime behavior is represented in a contract test first.

## Build order (each step reviewed before the next)

1. models.py + errors.py + config.py: the contract first.
   1b. Immediately generate fixtures (tests/fixtures/deployments.py, catalog.py, events.py).
2. store.py + events.py + logging.py: infrastructure (round-trip fixtures through the store here).
3. providers/base.py + extract RunPod code + mock.py + contract tests.  *(human review gate)*
4. runtimes/base.py + vllm.py + catalog with 3 models.
5. core/reconciler.py + core/orchestrator.py: reviewed line-by-line; this is the heart.  *(human review gate)*
6. core/health.py + core/costs.py.
7. CLI.
8. OpenAI proxy.
9. Real-GPU validation gauntlet (spec §18): before Phase 2.  *(human review gate)*
10. Catalog to 10-15 models (each validated).
11. Phase 2 FastAPI -> Phase 3 MCP -> Phase 4 Swamp.

Human review is mandatory at steps 3, 5, and 9 minimum.

## Decisions

- **Async vs sync provider style: RESOLVED = async end-to-end** (2026-07-03). The Provider and
  Runtime ABCs, the Orchestrator facade (§7.1), the reconciler, the CLI commands, and the OpenAI
  proxy are all `async def`. The spec's sync signatures in §7.1/§8/§9 are read as their async
  equivalents. runpod-ephemeral's `httpx.AsyncClient` code transfers as-is. **The store stays sync**
  (SQLite is synchronous and fast); async core calls it directly, wrapping in `asyncio.to_thread`
  only if a call ever gets hot. Typer CLI commands cross the boundary with `asyncio.run` per command.

## Open decisions (resolve before the step that needs them)

- **Reconcile-loop ownership: RESOLVED = daemon** (2026-07-04). A background asyncio loop owns the
  reconciler; CLI commands are thin clients over the shared SQLite store. Deploy is non-blocking,
  survives restart, and satisfies gauntlet #3. Step 5 builds `reconcile_once()` as the callable core;
  the long-running daemon wrapper lands with the CLI (step 7). Supersedes the "open" wording in
  spec §7.3.

## Deferred from step 5 (tracked, not silently missing)

Step 5 (reconciler + orchestrator) landed the decision core, the observe/execute boundary,
`reconcile_once`, and the §7.1 facade. Consciously left for the step that owns them:

- **Exponential backoff spacing (10s -> 60s, spec §7.3).** The attempt cap and the per-stage
  timeout budget are in; the exponential *spacing* is not. It needs a real clock to space against,
  which is the daemon (step 7). Add a `last_attempt_at` to `FailureInfo` then, not before (no
  consumer exists yet). Baseline spacing today = the daemon's `reconcile_interval`.
- **Cost-record lifecycle.** Nothing writes a `CostRecord` on create or closes it on stop yet, so
  `Orchestrator.get_costs` currently returns only what is already stored. Wire create/close into
  the reconciler's execute path in step 6 (`core/costs.py`). `estimate_cost` (pure) is done.
- **Health thresholds / degraded-flap handling (spec §10).** The reconciler uses a minimal inline
  liveness+readiness probe (`_probe_health`); consecutive-failure thresholds and the degraded state
  machine are step 6 (`core/health.py`). `Orchestrator.get_health` is a thin live probe for now.
- **Provider dead-token handling.** `map_to_observed_state` folds provider "dead" tokens
  (EXITED/TERMINATED/...) to REQUESTED so next_step recreates or finishes teardown. Hardening
  (immediate destroy of a dead-but-present pod) is deferred to the real-GPU gauntlet (step 9).

## Step 5 deviations (from already-reviewed files)

- **`Runtime.serving_port: ClassVar[int]`** added to `runtimes/base.py` (vLLM = 8000). Spec §8 says
  runtimes declare their serving port; the step-4 ABC lacked it and `observe` needs it to resolve
  the endpoint URL. Provider still owns URL shape; runtime only declares the port.

## RunPod extraction source

`/Users/fbaig/Projects/runpod-ephemeral/src/runpod_ephemeral/runpod.py`: live-validated against the
real RunPod REST API. Covers ~4 of the 9 Provider-ABC methods; API mechanics (auth, create body,
`{podid}-{port}.proxy.runpod.net` URL, idempotent terminate) transfer cleanly. Naming/list/logs/
capabilities/get-None are net-new. (The spec's "image-indexing project" wording refers to this
shaderdex capture sandbox code.)

## House rules

- Before pushing: `ruff check` AND `ruff format --check` (CI runs both; they are different checks).
- No em dashes in code comments, docs, or copy. Use periods, commas, colons, or parens.
- Run tests: `uv run python -m pytest tests/ -v --tb=short`.
