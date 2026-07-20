# Plan: capacity envelope (schedules, budgets, concurrency, replicas, fallback)

Derived from the "Stop Buying Tokens, Start Operating AI Capacity" marketing outline. That outline
describes OpenLease as a governed enterprise capacity platform; today it is a GPU-capacity control
plane plus cost metering. This plan closes the gap between the two, and marks which gaps are worth
closing versus which drag OpenLease into a different (crowded) product category.

Two ways to close a gap: move the product toward the story (build), or move the story toward the
product (reframe the claim as roadmap). Each item below is tagged with which it deserves.

Ordered by leverage. Tier A is independently shippable and makes the post honest; Tiers B and C can
be published as clearly-labeled roadmap without being built first.

---

## Cross-cutting principle (governs every time-based or signal-based item)

`next_step()` stays a pure function (no clock, no network, no side effects; CLAUDE.md, reliability
constraint #1). Anything driven by wall-clock or metrics (a schedule, a budget check, a utilization
reading) is **resolved in the daemon and passed into the core as already-decided desired state**.
Time and metrics are inputs to the resolver, never reads inside the decision function.

How each item fits without violating the architecture:
- **Persistence:** new state is fields/tables in the existing sqlite store, each carrying
  `schema_version`, with a migration (`_MIGRATIONS` + `PRAGMA user_version`). No ORM.
- **Evaluation:** a daemon `tick_*` loop over the shared store, built on a callable core so the core
  stays unit-testable (the `tick_reconcile` / `tick_health` / `tick_sweep` / `tick_costs` pattern).
- **Enforcement:** dispatched through the existing `execute` / teardown paths. Cost-safety invariant
  is preserved (no FAILED/STOPPED deployment keeps a running instance).
- **Events:** appended to the log. No event bus, no pub/sub, no plugin framework, no dynamic loading.
- **Contract-test-first:** any new provider/runtime or reconciler behavior is represented in a
  contract or reconciler test before implementation (CLAUDE.md build discipline).

---

## Tier A: build. Cheap, on-thesis, makes the flagship post true.

These three deliver the outline's headline promises: predictable ceiling, capacity envelope,
controlled concurrency. None re-architects anything; each is an extension of an existing seam.

### A1. Operating schedules (scheduled availability): SHIPPED 2026-07-19

Makes true: "business-hours availability," "scheduled shutdown," predictable cost (outline Sections
4, 5, 8). Shipped as `gpu schedule <deployment> --on "<days> HH:MM-HH:MM" [--off ...] [--tz ...]
[--default on|off] [--clear]`. The daemon resolves the posture each reconcile tick and drives
`desired_state`; the pure decision core is unchanged. Two postures (ON/OFF); `WARM_STANDBY` deferred
to replicas (B1). Full: `models.Posture/Schedule/ScheduleRule`, `core/schedule.py` (pure resolver),
daemon `_reconcilable`/`_apply_schedule`, orchestrator `set_schedule`/`clear_schedule`,
`render.schedule_view`.

**Concept.** A deployment carries an optional `Schedule`: an ordered set of rules mapping a
recurring time window to a *posture*. Postures:
- `RUNNING`: full desired capacity (current behavior).
- `WARM_STANDBY`: minimum footprint kept alive (defined below; for now = 1 replica, the smallest
  viable pod for that model). Distinct from OFF so a cold start is not on the critical path at the
  next window.
- `OFF`: instances torn down, deployment config retained (not deleted).

**Data model.** `Schedule` Pydantic model on `Deployment`: `timezone: str`, `rules: list[Rule]`
where `Rule = {days, start, end, posture}`, plus a `default_posture` for uncovered time. Persisted
with `schema_version`; store migration adds the column.

**Daemon fit.** The daemon resolves the current posture from wall-clock at each tick (a pure
`resolve_posture(schedule, now) -> Posture` helper, unit-testable with injected `now`) and folds it
into the deployment's desired state before `reconcile_once`. `next_step` receives the resolved
posture; it never reads the clock. Option: fold into `tick_reconcile`, or a dedicated
`tick_schedule` that writes the desired posture the reconciler then drives.

**Invariants preserved.** `OFF` tears down through the existing terminal-teardown path (cost-safety
intact). Posture resolution is deterministic and pure. Timezone/DST handled in the resolver via a
fixed library, not ad hoc.

**Contract test first.** `resolve_posture` table test across window boundaries, DST edges, and
uncovered time falling back to `default_posture`; a reconciler test that a deployment in an `OFF`
window reconciles to zero instances and back to `RUNNING` at the boundary.

**Open sub-decision.** Define `WARM_STANDBY` precisely. Simplest v1: standby == 1 replica (needs no
new concept). Richer standby (a paused/checkpointed pod) depends on provider support and is deferred.

### A2. Budget ceilings (spend governance): SHIPPED 2026-07-19

Makes true: "hard monthly infrastructure budget," "daily infrastructure ceiling," "knowable cost
ceiling" (outline Sections 5, 9, 12). Shipped as `gpu budget set/list/rm` over `--limit`,
`--window daily|monthly`, `--on-exceed warn|stop|block_new`, `--deployment` (else account-wide). A
budget is a policy layer over existing `CostRecord` data, no new accounting. `stop` enforces via a
`Deployment.budget_hold` that outranks the schedule (precedence: budget > schedule > manual),
released when the window resets. `block_new` is checked at deploy admission. Full: `models.Budget`
+ enums + events, `core/budgets.py` (pure), store migration v4, daemon `tick_budget`, orchestrator
`set_budget`/`remove_budget`/`list_budgets`/`budget_status`, `render.budgets_table`. Window
boundaries are UTC in Phase 1 (a per-budget timezone is a cheap follow-up).

**Concept.** A `Budget` policy = `{scope, window: daily|monthly, limit_usd, on_exceed}` where
`on_exceed in {warn, stop, block_new}`. Scope is deployment-level and account-level (account = the
whole store) in v1; per-team/per-project scope depends on Tier C identity and is deferred.

**Data model.** `Budget` table keyed by scope; accrual is read from the existing `cost_records`
(hourly `cost_snapshot`) and `usage_records`. No new accounting; budgets are a policy layer over
data that already exists.

**Daemon fit.** New `tick_budget`: compute accrued spend for the active window, compare to
`limit_usd`, emit a `budget_threshold` event at a soft fraction (e.g. 80%), execute `on_exceed` at
100%. `stop` routes through the existing stop path; `block_new` sets a flag the deploy admission path
checks and refuses new deploys with a typed error.

**Invariants preserved.** Enforcement actions are existing safe operations (stop is already
cost-safe). Accrual source of truth stays `cost_records`; no parallel meter.

**Contract test first.** A test that seeds `cost_records` past a limit and asserts the correct event
+ action per `on_exceed`; a test that `block_new` refuses a deploy while an open deployment continues.

### A3. Concurrency limits + bounded queue: SHIPPED 2026-07-19

Makes true: "controlled concurrency envelope," "enforce concurrency limits," "queue excess work"
(outline Sections 5, core thesis). Shipped as `gpu limits <deployment> --max N [--queue M]
[--timeout S] [--clear]`. The streaming-slot lifecycle (the flagged hard part) is handled: a slot is
acquired before the upstream opens and released exactly once in the background task starlette runs
even on client disconnect, in a `finally` so a close/meter error cannot leak it; a send failure
before streaming also releases. Relies on `asyncio.Semaphore` cancellation-safety (3.11+). Full:
`Deployment.max_concurrency/max_queue/queue_timeout_s` (validated), `proxy/limiter.py`
(`DeploymentLimiter` + registry), proxy `_acquire_slot`/`_finish` wiring (`_route_table` shape
unchanged so the MCP chat tool is untouched), orchestrator `set_limits`, `render.limits_view`.

**With A3 shipped, Tier A is complete:** the flagship post's "predictable ceiling / capacity
envelope / controlled concurrency" claims are all true, not aspirational. Tiers B and C remain
roadmap.

**Concept.** Per-deployment in-flight request cap in the OpenAI proxy, with a bounded wait queue
(max depth + max wait); requests over the queue bound return `429`. This is the server-side twin of
what `gpu batch` already does client-side with an `asyncio.Semaphore`.

**Proxy fit.** The proxy is already in the request path and already rewrites the `model` field. Add a
per-deployment semaphore + bounded queue keyed by the routed deployment. Config lives on the
deployment (`max_concurrency`, `queue_depth`, `queue_timeout`).

**The hard part (call it out).** A *streaming* response holds its slot until the stream closes, so
slot acquisition must wrap the full stream lifecycle (acquire before upstream open, release on stream
teardown, including client disconnect and error). Get this wrong and slots leak. The proxy must stay
a thin router: this is a bounded semaphore, not a scheduler.

**Invariants preserved.** Byte-for-byte forwarding is unchanged; concurrency control wraps it, it
does not rewrite the body. The `Accept-Encoding: identity` and model-rewrite behavior stay intact.

**Contract test first.** A proxy test using the existing MockTransport that N+1 concurrent requests
with cap N sees the last queued then admitted (or 429 past the queue bound), and that a slot is
released when a streamed response completes and when a client disconnects mid-stream.

---

## Tier B: real horizontal capacity. Medium lift, on-model, publish as near roadmap.

### B1. Replicas + load balancing

Makes true: "multiple replicas," "add replicas during peak," "scale" (outline Section 5).

**Concept.** A deployment gains `desired_replicas: int` (today implicitly 1). The reconciler drives
observed instance count to N; the proxy load-balances across the READY pool for that model
(round-robin or least-in-flight). A replica set is N identical pods, each still one-model-per-pod
served under its HF repo id (the fact the live gauntlet proved).

**Reconciler fit.** `next_step` already reconciles instance existence; generalizing from a single
instance to a target count is real work but on-model (desired count vs observed count, create/destroy
one per tick per the one-step-per-tick rule). Instance naming needs a per-replica suffix on
`gpu-orch-{namespace}-{deployment_id}` (append `-{replica_index}`), preserving the deterministic-name
invariant.

**Proxy fit.** Routing today selects one endpoint per model; extend `_route_table` to hold the pool
and pick per request.

**Why it is medium-large.** Adoption, orphan sweep, and cost accrual all currently reason about one
instance per deployment; each generalizes to a set. Do this as its own reviewed step (heart-of-system
change, like the original reconciler), not folded into Tier A.

### B2. Autoscaling (demand-driven replicas): depends on A3 + B1

Makes true: "add capacity during peak demand," the dynamic half of the capacity dial.

**Concept.** `desired_replicas` tracks a demand signal (queue depth from A3, or utilization from
usage metering) within `{min, max}`, resolved in the daemon with hysteresis to avoid flapping. Reuse
the health engine's flap-absorption pattern (act only after a threshold of consecutive readings).

**Daemon fit.** A `tick_autoscale` reads the signal, computes a target within bounds, writes
`desired_replicas`; the B1 reconciler drives it. Scale-down must drain in-flight requests and respect
cost-safety on teardown.

**Prerequisite.** Needs A3's queue/concurrency metrics and B1's replica reconciliation. Not before
both land.

---

## Tier C: decide before spec. This is where scope escapes the wedge.

These are not "build later by default." Two of the three change what OpenLease *is* and should be a
deliberate choice, not a backlog inheritance.

### C1. Frontier fallback / multi-provider routing: build only the narrow case

**The scope trap.** The moment the proxy routes to an external commercial API, OpenLease stops being
a GPU-capacity control plane and becomes a general AI gateway, which is a crowded category with
entrenched incumbents. The outline's "burst to a commercial API" and "route approved requests to a
frontier API" imply that gateway.

**Recommendation.** Build only the narrow case the capacity story actually needs: a single configured
OpenAI-compatible upstream that the proxy bursts to when the local pool is saturated (A3 queue full)
or a budget's `on_exceed` degrades local capacity. Triggered deterministically, one upstream, off by
default, opt-in per deployment. Do **not** build general provider routing or a task-complexity
classifier (the latter also tends to want an LLM in the request path, which contradicts the "software
operates infrastructure" thesis). For anything richer than narrow overflow, document that OpenLease
sits behind an existing gateway rather than rebuilding one.

**Status:** conceptual only; needs an explicit yes on the narrow-overflow scope before it is specced.

### C2. Per-team / per-project quotas + identity: defer or integrate

**Why it is large.** Team/project quotas need a tenancy and identity model. The spec explicitly
defers this: sqlite single-process, a single static bearer token for the API, Postgres as the
multi-tenancy seam (spec Section 7.4). This is RBAC + attribution + an IdP story, a genuine platform
expansion.

**Recommendation.** Deliver account-level and deployment-level budgets (A2) now, which cover the
"hard ceiling" claim without identity. Defer team/project quotas to a "later, or integrate with your
identity provider" line in the post. Do not build identity to satisfy a marketing bullet.

**Status:** deferred. Reframe as roadmap in the post.

### C3. Deterministic complexity router (local vs frontier by task): defer

**Why.** Routing engineering tasks to local-vs-frontier by complexity needs classification logic in
the request path and multi-provider routing (C1). Heuristic versions are weak; model-based versions
put an LLM in the control path against the thesis. This belongs in the application layer, not the
runtime.

**Status:** out of near scope. Leave to the app that calls OpenLease.

---

## Publishing gate

The flagship post is honest to publish when **Tier A is shipped** (predictable ceiling, capacity
envelope, controlled concurrency are true, not aspirational) and **Tiers B and C are clearly labeled
as roadmap** in the OpenLease section. That closes the credibility gap without a long build. Tier A
is good product on its own merits, independent of the post.

## Suggested build order

1. **A1 schedules** (highest leverage; directly makes "capacity envelope / predictable cost" true).
2. **A2 budgets** (small, high-signal; makes the "hard ceiling" claim real).
3. **A3 concurrency + queue** (the streaming-slot lifecycle is the only fiddly part).
4. Decide **C1** scope (narrow overflow: yes/no). If yes, it is small and pairs with A2/A3 triggers.
5. **B1 replicas** as its own reviewed step (heart-of-system change).
6. **B2 autoscaling** once A3 + B1 exist.
7. **C2 / C3** stay roadmap unless a concrete engagement pulls them.
