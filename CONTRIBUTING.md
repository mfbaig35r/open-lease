# Contributing to open-lease

Thanks for looking. open-lease is the orchestration layer, not the provider: the goal is that a
new provider or runtime is a small, well-contained addition, and that the reconcile core stays
small and exhaustively tested.

## Dev setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mfbaig35r/open-lease && cd open-lease
uv sync --extra dev        # engine + CLI + proxy + REST + MCP + test/lint tools
```

## Before you push

CI runs both of these, and they are different checks:

```bash
uv run python -m pytest tests/ -q
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

The suite is offline: it runs against the in-memory mock provider and a mocked vLLM transport, so
it needs no credentials and spends nothing.

## Live-GPU testing (optional, costs real money)

The real-RunPod path is opt-in and never runs in CI. It needs `RUNPOD_API_KEY` (and `HF_TOKEN`
for gated models) in a local `.env`. Rules learned the expensive way:

- Stop a deployment the instant a check passes. A pod left READY bills the whole time.
- Tear down with `gpu stop <id>` then `gpu delete <id>`, never a raw provider-side pod delete: the
  latter leaves the deployment record wanting READY, and the reconciler correctly recreates it.
- Wrap any live test in a hard cleanup that force-deletes every `gpu-orch-*` pod on exit.

## Architecture rules (the short version)

The full, non-negotiable list is in [CLAUDE.md](CLAUDE.md). The ones that matter most in review:

- `next_step()` is a **pure** function: no network, no clock, no side effects. It is the most
  tested code in the repo. New reconcile behavior gets a pure test in the desired x observed matrix
  first.
- The reconciler takes **one step per tick**. Never chain stages in a single pass.
- **Cost-safety invariant**: no FAILED or STOPPED deployment ever keeps a running instance. Any
  code path that creates a provider instance has a matching cleanup path, and that path has a test.
- No plugin frameworks or dynamic loading. A provider or runtime is an ABC plus a module-level dict
  entry. Interfaces (CLI / REST / MCP) contain no business logic: parse, call the Orchestrator,
  render.
- Type hints everywhere; Pydantic v2 for all domain models. No file over ~400 lines except
  `models.py`; no function over ~50 lines.

## Adding a provider or model

- **Provider**: implement the Provider ABC against the contract suite. Walkthrough in
  [docs/adding-a-provider.md](docs/adding-a-provider.md).
- **Model**: add an entry to `catalog/models.toml`. Set `validated_at` only after a real deploy has
  reached READY and served a completion; leave it empty otherwise so the catalog stays honest.

## Pull requests

Branch off `main`, keep tests and both ruff checks green, and open a PR. Small, focused PRs review
fastest. If a change touches the reconcile core, call that out so it gets a careful read.

Writing style: no em dashes in code, comments, docs, or copy. Use a period, comma, colon, or
parentheses instead.
