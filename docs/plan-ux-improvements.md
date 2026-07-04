# Plan: top three UX improvements

Derived from [ux-feedback.md](ux-feedback.md). Ordered by leverage. Each is independently shippable.
None require re-architecting; they are last-mile UX on a working engine.

---

## 1. Download progress + ETA in `gpu status`

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

**Approach: a per-namespace RunPod network volume mounted at the HF cache dir.**

- On first deploy in a namespace, ensure a RunPod **network volume** exists (create if absent,
  region-pinned). Mount it at the HuggingFace cache path and set `HF_HOME` /
  `HUGGINGFACE_HUB_CACHE` env so vLLM downloads into and reads from it. Second deploy of the same
  model finds cached weights and skips the download.
- The `VolumeSpec` model already carries `size_gb`, `mount_path`, `persistent`. Extend the provider
  create body to attach a network volume by id.

**Key design decisions to resolve before building.**
- **Shared vs per-model volume.** A single shared cache volume per namespace is the most
  space-efficient, but RunPod network volumes are typically attachable to one pod at a time and are
  region-locked. That directly conflicts with the concurrent-deploy case just validated in gauntlet
  #5 (two pods at once). Options: (a) per-model volumes (concurrent-safe, more volumes to manage);
  (b) shared volume but serialize deploys that need it; (c) shared read-only mount if RunPod
  supports multi-attach read-only. Verify RunPod's actual multi-attach semantics first; this is the
  crux and determines the whole design.
- **Region pinning.** A volume is bound to a region, so the pod must launch in that region. This
  removes RunPod's automatic capacity spread and can cause "no capacity in region" failures. Need a
  region preference in config and a clear error when the pinned region is dry.
- **Lifecycle + cost.** Volumes persist beyond pods and cost storage per month. Needs `gpu volumes`
  (list/create/delete) and storage cost in the cost model (today cost is GPU-hours only, §11).

**Code touch-points.**
- `providers/runpod.py`: volume create/attach in `create_instance`; a `find_or_create_volume`
  helper; region handling.
- `providers/base.py`: promote volume support from optional to a first-class provider capability
  (already flagged via `ProviderCapabilities.supports_volumes`).
- `config.py`: `cache_volume_enabled`, `cache_volume_size_gb`, `region`.
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
