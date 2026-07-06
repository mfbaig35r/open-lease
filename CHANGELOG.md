# Changelog

All notable changes to open-lease are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/mfbaig35r/open-lease/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mfbaig35r/open-lease/releases/tag/v0.1.0
