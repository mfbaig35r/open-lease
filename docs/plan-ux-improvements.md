# Plan: top three UX improvements

Derived from [ux-feedback.md](ux-feedback.md). Ordered by leverage. Each is independently shippable.
None require re-architecting; they are last-mile UX on a working engine.

---

## 1. Download progress + ETA in `gpu status` — SHIPPED 2026-07-04

Layer B (the always-available ETA) shipped; Layer A wired as a seam. Investigation: RunPod exposes
no clean pod-log API (REST v1 has none; GraphQL introspection is disabled, no discoverable log
field, verified live), so real download percent is not available on RunPod. `observe` still calls
`provider.get_logs` + `runtime.download_progress` on every bring-up tick (a no-op for RunPod, real
for the mock and any future provider with logs) and stores `Deployment.download_progress`.
`gpu status` shows `starting_server 45%` when a percent is known, else `starting_server 8m/40m`
(elapsed in stage / the profile's `startup_timeout_seconds`), so a cold start never reads as stuck.

**Problem.** During `starting_server`, the user sees no progress and no ETA. `DOWNLOADING` exists in
`DeploymentState` but never surfaces because `providers/runpod.py:get_logs` returns `[]` (RunPod's
REST v1 has no pod-log endpoint), so `VLLMRuntime.download_progress` (which already parses a `NN%`
regex) has nothing to read.

**Approach: two layers, best-effort real progress + always-on honest ETA.**

- **Layer A (real %, best-effort): get pod logs so the existing parser has input.**
  - Investigate RunPod's GraphQL API for pod logs (`query { pod(input:{podId}) { ... } }` /
    container logs). The REST v1 lacks it; GraphQL historically exposed logs. This is an
    investigation task, not a certainty. If it works, `RunPodProvider.get_logs` fetches recent lines
    via GraphQL, and the reconciler's `observe` calls `runtime.download_progress(logs)` while in the
    booting/starting stages and stores the result.
  - If GraphQL logs are unavailable, a fallback is to have the launch command tee vLLM output to a
    file inside the container and serve it on a side port, but that is invasive and deferred; do not
    build it unless Layer B proves insufficient.
- **Layer B (always available): derive a coarse stage + budget-relative ETA from what is observable.**
  - We can already observe: pod running (RunPod desiredStatus), endpoint routable, `/health`
    reachability, `/v1/models` serving. Map these to a finer observed state so `DOWNLOADING` and
    `STARTING` actually separate (endpoint routable + `/health` failing for longer than the boot
    budget implies download-in-progress).
  - Show elapsed-in-stage against the model's expected size and the configured budget:
    "downloading (~65GB), 12m elapsed of ~50m budget". Honest, no false precision.

**Code touch-points.**
- `providers/runpod.py`: implement `get_logs` via GraphQL if viable (net-new; today it is a no-op).
- `core/reconciler.py` `observe` / `_probe_health`: when in a pre-READY serving-bound stage, fetch
  logs, call `runtime.download_progress`, and thread a `download_progress: float | None` onto the
  `Observation`; persist it on `Deployment` (new optional field, no migration since it is nested in
  the JSON doc).
- `models.py`: add `Deployment.download_progress: float | None = None` and optionally a
  `DeploymentState.DOWNLOADING` derivation rule in `map_to_observed_state` keyed on health + stage.
- `cli/render.py`: add progress/ETA to the `gpu status` state cell (e.g. `starting_server 34%` or
  `downloading ~12m/50m`), and to `gpu events`.

**Tradeoffs / risks.** Real % depends on RunPod GraphQL logs existing and being stable; treat Layer
A as best-effort and never regress if it returns nothing. `map_to_observed_state` must stay pure, so
the stage-timing that distinguishes DOWNLOADING lives in `observe`/reconcile, not in the pure map.

**Definition of done.** `gpu status` on a cold-starting deployment shows either a real percentage or
an honest elapsed/budget ETA, never a bare `starting_server` for minutes. Unit test the new
`download_progress` plumbing against a mock with scripted log lines.

---

## 2. Persistent model cache volume

**Problem.** Ephemeral disks (§14) mean every deploy re-downloads the model. 65GB per qwen3-32b
deploy. This is the single biggest felt-slowness. The `VolumeSpec` seam exists but
`RunPodProvider.create_instance` raises `NotSupportedError` on any volume today.

### Investigation findings (RunPod network volumes, 2026-07-04)

From the live REST OpenAPI spec and RunPod docs:

- **API surface.** `POST /v1/networkvolumes` create takes `{name, size (GB), dataCenterId}`; also
  `GET`/`PATCH`/`DELETE /v1/networkvolumes/{id}`. Pod create (`PodCreateInput`) takes a single
  `networkVolumeId` plus `volumeMountPath` (mount location). One volume per pod.
- **Multi-attach: YES.** "Multiple pods can now mount the same network volume simultaneously."
  So a shared per-namespace cache volume works even for concurrent deploys (the gauntlet #5 case).
  This is the crux and it is resolved in favor of a shared volume.
- **Concurrent-write corruption is the real risk, not attach.** "Writing to the same volume from
  multiple workers simultaneously may cause data corruption. Handle concurrent write access in your
  application logic." So two concurrent *cold* deploys of the *same* model, both writing weights to
  the same cache path, can corrupt. Cache *hits* (read-only) are safe to share concurrently.
- **Region locking: confirmed.** A volume is `dataCenterId`-scoped, and attaching it "constrains
  worker deployments to that volume's datacenter, which may limit GPU availability and reduce
  failover options." Attaching removes RunPod's automatic capacity spread.
- **Pricing.** $0.07/GB/month (first 1 TB), $0.05/GB beyond. NVMe, 200-400 MB/s (up to 10 GB/s).

### Live CRUD validation (2026-07-04)

Validated the volume API code against real RunPod (create/list/delete, idempotent-by-name), then
deleted every test volume. Findings: minimum volume size is **10GB** (5GB returns a 500); valid data
center ids include **US-KS-2, US-CA-2, EU-RO-1**; and RunPod's create is **flaky** (one call
returned a 500 "unexpected end of JSON" but still created the volume). The find-or-create-by-name
design self-heals that flakiness: a retry lists the orphaned volume and reuses it instead of leaking
a second one. Still to validate with spend: a pod actually booting with the volume mounted and a warm
redeploy reusing the cache (the speedup is only clearly measurable on a large model like qwen3-32b,
since a small model's download is dwarfed by the ~8GB vLLM image pull, which is NOT cached).

### Live pod validation (2026-07-05) — mechanism proven, warm blocked by capacity

Ran qwen3-32b cold with caching on: created the 100GB volume in a DC, downloaded 65GB of weights to
it, mounted at `/cache` with `HF_HOME` set, and served (chat returned "Hello."). **Cold
time-to-ready 17m20s.** So the cache mechanism works end to end on real RunPod. The warm redeploy
could NOT be measured: it correctly reused the same volume but the pod create failed because the
pinned data center ran out of A100 capacity between runs (empirically: US-KS-2 had no A100 at start,
EU-RO-1 had it for the cold deploy then went dry for the warm). This is the region-pinning downside,
demonstrated: a cached deploy is only as available as its one pinned DC's transient GPU stock, and it
cannot fall back. Cost-safe throughout (failed creates left 0 pods; warm accrued $0.00). Also fixed a
robustness bug found here: `ensure_cache_volume` now matches name AND data center, so a same-name
volume in a stale DC is not wrongly reused.

**Open follow-up:** the warm speedup number itself is still unmeasured (blocked by capacity, not
code). Retriable any time the chosen DC has A100 stock. Worth considering: a fallback that, if the
pinned DC is dry, warns and offers a no-cache deploy (full capacity spread) rather than failing.

### Resulting design (de-risked)

- **Shared per-namespace network volume** at a fixed `dataCenterId`, mounted at the HF cache dir,
  with `HF_HOME` / `HUGGINGFACE_HUB_CACHE` pointed at it. Multi-attach makes this safe for
  concurrency; per-model volumes are not needed.
- **Mitigate the concurrent-cold-download race** (the one real hazard): a per-model download lock so
  only the first deploy of an uncached model populates the cache while others wait for it, after
  which all reads are cache hits (safe to share). A pragmatic v1 is a small lock row in the store
  keyed by model_id, checked in the deploy path; HF's own `.incomplete` + lock files are a backstop
  but are not reliable across NFS hosts, so do not depend on them alone. Simplest possible v1:
  document the risk, rely on HF locking, and only build the store lock if corruption is observed.
- **Region pinning is a real tradeoff, made explicit.** A new `runpod_data_center_id` config pins
  both the volume and cache-enabled pods to one datacenter. Surface a clear "no capacity for <gpu>
  in <dc>" error, and keep caching opt-in (`cache_volume_enabled=false` default) so non-cache
  deploys keep RunPod's full capacity spread.
- **Cost.** Add storage accrual at the $0.07/GB rate (separate from GPU-hours) and a `gpu volumes`
  command to list/size/delete.

**Code touch-points.**
- `providers/runpod.py`: volume create/attach in `create_instance`; a `find_or_create_volume`
  helper; region handling.
- `providers/base.py`: promote volume support from optional to a first-class provider capability
  (already flagged via `ProviderCapabilities.supports_volumes`).
- `config.py`: `cache_volume_enabled` (default false), `cache_volume_size_gb`, `runpod_data_center_id`.
- `core/catalog.py` / profiles: optional per-model cache hint (size).
- `core/costs.py`: storage-cost accrual (new, separate from GPU-hours).
- `cli/main.py` + `render.py`: `gpu volumes`.

**Tradeoffs / risks.** This is the largest of the three; it touches the provider, catalog, config,
cost model, and CLI. Region pinning trades capacity flexibility for warm starts. Multi-attach
semantics may force per-model volumes. Ship behind an opt-in flag (`cache_volume_enabled=false`
default) and measure warm-start improvement before making it default.

**Definition of done.** With caching enabled, a second deploy of the same model reaches READY in the
warm-start time (single-digit minutes for a large model) instead of re-downloading, and `gpu costs`
reflects storage cost. Validated live on the same qwen3-32b that took ~17 minutes cold.

---

## 3. Make the daemon lifecycle invisible — SHIPPED 2026-07-04

Implemented as described below. `cli/process.py` (pidfile liveness + stale cleanup + detached
spawn); `gpu daemon [--detach|--stop|--status]`; `gpu proxy` pidfiled; `gpu up`/`gpu down`; and a
non-blocking `gpu deploy` warns (or `--auto-daemon` starts one) instead of silently stalling.
Verified end to end: `gpu up` -> both detached with pidfiles -> `gpu down` -> clean; deploy with no
daemon warns and creates no pod. `auto_daemon` config default is false (warn, do not surprise-spawn).

**Problem.** `gpu deploy` without `--wait` and no running daemon writes the record and never creates
a pod, silently. `gpu daemon` and `gpu proxy` both block a terminal. There is no `gpu up`.

**Approach: pidfile-managed detachable daemon, a loud guard on silent stalls, and a combined `up`.**

- **Detachable daemon.** `gpu daemon --detach` runs the daemon as a background process
  (`start_new_session`), writes `~/.gpu-orchestrator/daemon.pid`, and redirects logs to
  `~/.gpu-orchestrator/daemon.log`. Add `gpu daemon stop` (read pid, SIGTERM) and `gpu daemon
  status` (pid alive?). The pidfile also enforces the single-process invariant (§7.4): refuse to
  start a second daemon.
- **No silent stalls.** `gpu deploy` (non-blocking) checks the pidfile for a live daemon. If none
  and not `--wait`, print a clear warning: "No daemon running; dep-xxxx will not progress. Start one
  with `gpu daemon --detach`, or re-run with `--wait`." Default is warn, not auto-start, to avoid
  surprising background processes; an opt-in `--auto-daemon` (or a config default) can spawn one.
- **`gpu up`.** One command that starts the detached daemon and the proxy together (the spec already
  notes the daemon is a natural fit alongside `gpu proxy`, §7.3). `gpu down` stops both.

**Code touch-points.**
- `cli/main.py`: `--detach` on `daemon`; `daemon stop` / `daemon status` subcommands (or
  `gpu daemon --stop`); the deploy-time daemon-liveness check + warning; `gpu up` / `gpu down`.
- New small helper module (e.g. `cli/daemon_process.py` or a function in `core/daemon.py`) for
  pidfile read/write, liveness check, detached spawn. Keep detachment logic out of the core engine;
  it is an interface concern.
- `config.py`: `daemon_pid_file`, `daemon_log_file`, optional `auto_daemon`.

**Tradeoffs / risks.** Detaching a process is OS-specific (macOS/Linux via `os.setsid` /
`start_new_session`; Windows is out of Phase 1 scope). Auto-start is convenient but magic; keep the
default as a loud warning and make auto-start opt-in. The pidfile must handle stale pids (process
died without cleanup) by checking liveness before trusting it.

**Definition of done.** A new user can `gpu up`, then `gpu deploy qwen3-0.6b`, and the deployment
progresses to READY without a second terminal or any manual daemon management. A non-blocking deploy
with no daemon prints a clear warning instead of silently stalling.

---

## Suggested sequencing

1. **#3 first** (smallest, removes the worst footgun, unblocks a clean non-blocking workflow).
2. **#1 next** (medium, directly addresses the top complaint, no external dependencies beyond the
   RunPod-logs investigation).
3. **#2 last** (largest, most design risk, needs the RunPod multi-attach/region investigation; ship
   opt-in and measure).

A natural bundle after these is the "one-gesture chat" (`gpu deploy --chat`) noted in the feedback,
which becomes trivial once #3 makes the lifecycle seamless.
