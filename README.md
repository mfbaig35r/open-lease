# open-lease

Make GPU infrastructure programmable. Provision GPUs, deploy open-source LLMs, manage their
lifecycle with a reconcile loop, and serve inference through an OpenAI-compatible API. One
orchestration core, thin interfaces over it. [RunPod](https://runpod.io) is Provider #1.

The product is the orchestration layer, not the provider: two seams (Provider, Runtime), one facade
(`Orchestrator`), one vocabulary (`DeploymentState`), one contract (`models.py`). A deployment is
driven by a reconcile loop comparing desired vs observed state, so interruption and crash recovery
are free, and the cost-safety invariant (no orphaned pods burning money) holds by construction.

> Status: beta. The engine is validated against real RunPod (deploy, kill-and-recover, crash-resume,
> orphan sweep, concurrent deploys, runtime-crash cap). Not yet on PyPI, single provider, and the
> §18 24h soak remains. See [What's not done](#whats-not-done).

## Quickstart

The base install is the engine, CLI, and the OpenAI proxy. The REST API and MCP server are optional
extras (`open-lease[api]`, `open-lease[mcp]`, or `open-lease[all]`) so the core stays lean.

```bash
pip install open-lease                      # base: CLI + OpenAI proxy
# add the REST API and MCP server:  pip install 'open-lease[all]'

# credentials (see docs/configuration.md)
export RUNPOD_API_KEY=...                    # and HF_TOKEN for gated models; or use a .env

gpu models                                  # the model catalog
gpu availability qwen3-0.6b                 # which data centers can run it right now
gpu deploy qwen3-0.6b --wait                # provision + wait for READY
gpu status                                  # id, state, endpoint, uptime, accrued $
gpu stop <id>                               # tear down; verify with `gpu status`
```

First deploy of a model is download-bound: the vLLM image and the weights are pulled onto an
ephemeral disk, so a small model is ready in a few minutes and a large one takes longer. An opt-in
model cache (`cache_volume_enabled`) makes warm redeploys fast. `gpu status` shows a percent or an
elapsed/budget ETA so a cold start never looks stuck.

### Talk to a deployed model

```bash
gpu proxy                                   # OpenAI-compatible proxy on :8080 (another terminal)
curl localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hi"}]}'
```

The proxy routes by the request `model` field (catalog id or HF repo) to the matching READY
deployment. Or hit `deployment.endpoint_url` from `gpu status` directly.

### Run it in the background

```bash
gpu up            # start the daemon (reconcile/health/orphan sweep) and proxy, detached
gpu deploy qwen3-0.6b     # non-blocking; the daemon drives it to READY
gpu down          # stop both
```

## How it works

- **Core** (`core/`): `orchestrator.py` (the §7.1 facade), `reconciler.py` (the reconcile loop),
  `health.py`, `costs.py`, `catalog.py`, `daemon.py`.
- **Provider seam** (`providers/`): provisions compute, knows nothing about LLMs. RunPod + an
  in-memory mock; new providers are an ABC + a dict entry, verified by one contract suite.
- **Runtime seam** (`runtimes/`): serves a model on compute, knows nothing about providers. vLLM.
- **Interfaces**: the `gpu` CLI, a REST API (`gpu serve`, routes mirroring the Orchestrator, the
  OpenAI proxy mounted at `/v1/*`, auto-docs at `/docs`), and an MCP server (`gpu-mcp`, agent-facing
  tools over the same core). A Swamp extension is specified for later and consumes the REST API.

See [docs/architecture.md](docs/architecture.md) for the full picture, and
[requirements/gpu-orchestrator-requirements.md](requirements/gpu-orchestrator-requirements.md) for
the authoritative spec.

## Docs

- [Architecture](docs/architecture.md) — the seams, the reconcile loop, cost-safety.
- [Configuration](docs/configuration.md) — settings, precedence, credentials.
- [Adding a provider](docs/adding-a-provider.md) — implement the Provider ABC against the contract.
- [Phase 4 (Swamp extension)](docs/phase-4-swamp.md) — requirements for the front end that consumes
  the REST API.

## What's not done

- RunPod is the only real provider; the seam is proven but no second provider yet.
- The catalog is small: qwen3-0.6b / qwen3-8b / qwen3-32b are validated on real hardware;
  llama-3.1-8b is unvalidated (Meta gating, HF access pending).
- Gauntlet §18: the 24h soak is not yet run. OOM/terminal-failed is closed (a runtime-crash
  cap drives a persistently-failing deploy to terminal FAILED; covered by an offline test).
- The warm-cache speedup is proven mechanically but its timing is capacity-pending.

## Development

```bash
uv sync --extra dev
uv run python -m pytest tests/ -q
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

The `gpu` CLI and `gpu_orchestrator` import package keep their names for now; the distribution is
`open-lease`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, architecture constraints,
and how to add a provider; build order and non-negotiable rules are in [CLAUDE.md](CLAUDE.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
