# Changelog

All notable changes to open-lease are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `gpu ui` launches the local visual workbench (the open-lease-ui front end) served by the API at
  `/` and opens it in the browser; `gpu serve --ui <dir>` serves a built UI alongside the management
  API and the OpenAI proxy. The UI is auto-detected when bundled into the package.
- REST API: `POST /deployments` accepts `hf_repo` (+ `gpu`, `context`, `image`, `disk`) to deploy an
  ad-hoc model with no catalog entry, mirroring `gpu deploy --hf-repo`; `GET /availability` accepts a
  `gpu` query param to check a specific GPU. (Backs the open-lease-ui deploy wizard.)

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

[Unreleased]: https://github.com/mfbaig35r/open-lease/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mfbaig35r/open-lease/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mfbaig35r/open-lease/releases/tag/v0.1.0
