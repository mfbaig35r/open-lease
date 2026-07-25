# Changelog

All notable changes to open-lease are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scripts/release.py` cuts a release in one command, so no step depends on remembering it. It
  preflights (both repos clean and synced, the version untagged and not already on PyPI, since a PyPI
  version can never be re-uploaded), bumps the version and closes the CHANGELOG section, bundles the
  workbench, runs ruff + the suite + `uv build` + `twine check` + a scratch-venv install of the built
  wheel, then stops for confirmation before anything irreversible. Only then does it tag the UI repo,
  move `OPEN_LEASE_UI_REF` to that tag, and commit/tag/push here. `--dry-run` verifies and changes
  nothing; `--yes` skips the prompt.

### Changed
- The bundled workbench is pinned: the publish workflow builds it from the `OPEN_LEASE_UI_REF`
  repository variable (now an open-lease-ui tag) rather than that repo's `main`, so a backend tag
  identifies exactly one wheel. The workflow warns when the pin is unset instead of quietly falling
  back to `main`, and writes the workbench ref and commit it used into the run summary. Moving the pin
  is a step in `scripts/release.py`, because a pin nobody bumps silently ships a stale UI.

## [0.4.0] - 2026-07-25

### Added
- Multi-GPU (tensor-parallel) deploys. A model whose profile sets `tensor_parallel > 1` now
  provisions a pod with that many GPUs and shards vLLM across them. `InstanceRequest.gpu_count` is
  driven from `tensor_parallel`, the RunPod provider requests it (was hardcoded to 1), and cost
  accrues at the per-GPU rate times the count; `gpu status` shows `Nx <gpu>`. Single-pod only
  (NVLink/PCIe tensor parallelism); multi-node is out of scope.
- `gpu deploy --hf-repo <repo> --gpus N` requests a multi-GPU pod for an ad-hoc deploy, so a large
  model can be tensor-parallel without a catalog entry.
- Token-usage metering. The OpenAI proxy tallies tokens per deployment, and `gpu usage` (plus
  `GET /usage` and the `get_usage` MCP tool) reports requests, tokens, tokens/sec (utilization),
  accrued cost, and $/million-tokens, the crossover metric against per-token API pricing. Metering
  runs in the background so forwarding stays byte-for-byte; usage is pruned with events on retention.
- `gpu batch <deployment> <input.jsonl>` fans a file of prompts out over a READY deployment with
  bounded concurrency (`--concurrency`) and per-item retries, writing one result line per input.
  Throughput-bound batch work, the cheap way to parse thousands of documents; metered like proxy
  traffic so a run shows up in `gpu usage`. Accepts full chat requests, `{"prompt": ...}` objects,
  or bare strings, with an optional `--system` prompt.
- Operating schedules (capacity plan, Tier A1): `gpu schedule <deployment> --on "mon-fri 06:00-18:00"
  --tz America/New_York` makes capacity follow a plan instead of running until manually stopped. The
  daemon brings the deployment up during ON windows and tears it down otherwise, so overnight and
  weekend spend drop to zero on a known schedule. `--off` windows, `--default on|off`, `--clear` to
  return to manual control, and no-args to show the current schedule. The schedule only rewrites
  `desired_state`; the reconciler drives it, so the pure decision core is unchanged.
- Spend ceilings (capacity plan, Tier A2): `gpu budget set --limit 500 --window monthly --on-exceed
  stop` caps GPU spend over a daily or monthly window, account-wide or scoped to one deployment
  (`--deployment`). A budget is a policy layer over existing cost records, so there is no new
  accounting. `--on-exceed warn` emits an event, `stop` tears down the in-scope deployment(s) for
  the rest of the window (a budget stop outranks a schedule), `block_new` refuses new deploys while
  over budget. `gpu budget list` shows spend so far this window; `gpu budget rm <id>` removes one.
  The daemon evaluates budgets each tick and enforces `stop` via a hold that the reconciler drives.
- Concurrency limits (capacity plan, Tier A3): `gpu limits <deployment> --max 16 --queue 64
  --timeout 30` caps in-flight requests per deployment in the OpenAI proxy. It admits up to `--max`
  at once; up to `--queue` more wait up to `--timeout` seconds for a slot; anything beyond gets a
  429. A slot is held for the whole request including the streamed response, released exactly once
  when the stream ends (or the client disconnects), so slots never leak. `gpu limits <deployment>`
  shows the current limit; `--clear` removes it (unlimited). Off by default (no cap).
- Replicas + load balancing (capacity plan, Tier B1): the OpenAI proxy pools every READY deployment
  serving a model and round-robins across them, each keeping its own concurrency limiter so capacity
  adds up. `gpu deploy <model> --replicas N` deploys a pool; `gpu scale <model> N` resizes it (scale
  up clones an existing deployment's profile/limits/schedule, scale down stops the newest surplus).
  A replica is just another deployment of the same model, so the reconciler and the pure decision
  core are unchanged.
- Autoscaling (capacity plan, Tier B2): `gpu autoscale set <model> --max N --target <req/min>`
  keeps a model's replica count matched to its served request rate, between `--min` and `--max`,
  with hysteresis. The daemon adds or removes replicas declaratively (the reconcile loop drives
  them); `gpu autoscale list` / `rm` manage policies. The signal is served rate, so it tracks
  sustained load; a schedule-managed model is left to its schedule.

- Interface parity for the capacity envelope. Tiers A and B landed on the CLI first, which left the
  REST API and the MCP server able to spend but not to cap, the worst possible asymmetry for an agent
  driving GPUs. Both now reach the whole surface. REST: `PUT`/`DELETE /deployments/{id}/schedule`,
  `PUT`/`DELETE /deployments/{id}/limits`, `POST /scale`, `GET`/`POST /budgets` +
  `DELETE /budgets/{id}`, `GET /autoscale` + `PUT`/`DELETE /autoscale/{model_id}`, and
  `DELETE /volumes/{id}`. MCP: `set_schedule`/`clear_schedule`, `set_limits`/`clear_limits`,
  `scale_model`, `set_budget`/`list_budgets`/`remove_budget`,
  `set_autoscale`/`list_autoscale`/`remove_autoscale`, plus `deployment_events` and `list_volumes`.
  The MCP `set_schedule` tool takes the same human window spec as the CLI (`"mon-fri 06:00-18:00"`).
  `run_batch` stays CLI-only until it has a job resource to poll, and `delete_volume` stays off MCP.
- REST API: a domain validator rejection (a concurrency cap below 1, a budget limit of 0) now returns
  400 with the validator's message instead of a 500, and an unknown budget or autoscale policy id
  returns 404 (new `PolicyNotFoundError`).
- The bundled visual workbench (`gpu ui`) covers the capacity envelope: a Capacity page for spend
  ceilings, replica count, and autoscaling policies, plus a schedule and concurrency-limit editor on
  each deployment. A deployment whose state a policy explains now says so (scheduled off, budget
  hold), and a schedule that nothing is applying (no daemon running) is called out instead of looking
  broken. The release wheel builds it from open-lease-ui `main`.

### Fixed
- `GPU_ORCH_CORS_ORIGINS` works again. pydantic-settings JSON-decodes a list field's env value before
  any validator runs, so the documented comma-separated form raised `SettingsError` at startup and
  every command failed while it was set. The field is now `Annotated[list[str], NoDecode]` so the
  CSV validator sees the raw string, with a test that goes through the env var rather than an init
  kwarg (the kwarg path never had the bug, which is why the old test passed).

### Changed
- The schedule window-spec parser moved from `cli/main.py` to `core/schedule.py` (`build_schedule`),
  so the CLI and the MCP tool share one parser instead of each interface growing its own. Behavior is
  unchanged apart from the malformed-default message, which no longer names a CLI flag.
- `core.budgets.BudgetStatus` is a Pydantic model rather than a dataclass, so the REST API and MCP can
  return a budget's spend snapshot without a parallel schema. Attribute access is unchanged.
- Bumped the default ad-hoc deploy image from `vllm/vllm-openai:v0.9.1` (mid-2025) to `v0.25.1`. The
  old pin silently fails to load newer model architectures (a Dec-2025 model crash-looped until
  redeployed with a current image). Still a deliberate pin, not `latest`; override per deploy with
  `--image` when a model needs a different or nightly build.

## [0.3.0] - 2026-07-18

### Added
- Opt-in cross-origin for a hosted workbench: `gpu serve --cors-origin <origin>` (repeatable, or
  `GPU_ORCH_CORS_ORIGINS`) lets a UI at that exact origin call the API, including the Private Network
  Access preflight ack Chrome requires for a public HTTPS page to reach a loopback server. Off by
  default (same-origin only) and never wildcarded, so a running server is not exposed to other sites.
- `gpu ui` launches the local visual workbench (the open-lease-ui front end) served by the API at
  `/` and opens it in the browser; `gpu serve --ui <dir>` serves a built UI alongside the management
  API and the OpenAI proxy. The UI is auto-detected when bundled into the package.
- REST API: `POST /deployments` accepts `hf_repo` (+ `gpu`, `context`, `image`, `disk`) to deploy an
  ad-hoc model with no catalog entry, mirroring `gpu deploy --hf-repo`; `GET /availability` accepts a
  `gpu` query param to check a specific GPU. (Backs the open-lease-ui deploy wizard.)

### Fixed
- Adopted instances now open a cost record. A pod recovered by tag (spec §7.5) after a crash in the
  narrow create/persist window previously accrued nothing, so reported spend could silently drift
  below the provider's actual meter. `reconcile_once` opens a record on adoption when none is open
  (best-effort, never blocking the reconcile tick).

## [0.2.0] - 2026-07-06

### Added
- Ad-hoc model deploys: `gpu deploy --hf-repo <repo> --gpu <gpu>` (and the `deploy_hf_model` MCP
  tool) run any vLLM-servable Hugging Face model with no catalog entry. `--context` / `--image` /
  `--disk` tune the profile; the catalog now supplies curated recipes rather than gating what can
  run. Deployments are self-contained (they carry their own `hf_repo` and `context_window`), so
  reconcile and the OpenAI proxy no longer need a catalog lookup.

## [0.1.0] - 2026-07-06

Initial public release: the orchestration core plus three interfaces over it.

### Added
- Reconcile-loop engine over a desired/observed state pair. The decision core (`next_step`) is a
  pure function; `execute` is the only side-effecting dispatcher.
- Cost-safety invariant: no FAILED or STOPPED deployment ever keeps a running instance, enforced by
  terminal-state teardown and a namespaced orphan sweep. A persistently-crashing runtime is capped
  to terminal FAILED instead of recreating forever.
- Provider seam with RunPod (Provider #1) and an in-memory mock, verified by one contract suite.
- Runtime seam with vLLM.
- Interfaces over one Orchestrator facade: the `gpu` CLI, a REST API (`gpu serve`, optional `api`
  extra), an MCP server (`gpu-mcp`, optional `mcp` extra), and an OpenAI-compatible proxy that
  routes by model name to a READY deployment.
- Model catalog with validated qwen3-0.6b / qwen3-8b / qwen3-32b (llama-3.1-8b present but gated
  and unvalidated). Opt-in shared model-cache network volume, and GPU-availability polling.
- Background daemon (reconcile / health / orphan sweep / cost snapshot / event retention),
  per-deployment cost tracking, and download-progress reporting during bring-up.

[Unreleased]: https://github.com/mfbaig35r/open-lease/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/mfbaig35r/open-lease/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mfbaig35r/open-lease/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mfbaig35r/open-lease/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mfbaig35r/open-lease/releases/tag/v0.1.0
