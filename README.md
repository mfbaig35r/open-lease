# GPU Orchestrator

Make GPU infrastructure programmable. Provision GPUs, deploy open-source LLMs, manage their lifecycle, and serve inference through a unified OpenAI-compatible API.

```bash
pip install gpu-orchestrator
gpu deploy qwen3-32b
# a few minutes later: an OpenAI-compatible URL
gpu status | gpu logs | gpu costs | gpu stop <id>
```

One orchestration core; many thin interfaces (CLI, REST, MCP, Swamp extension). RunPod is Provider #1, not the product. The product is the orchestration layer: two seams (Provider, Runtime), one facade (Orchestrator), one vocabulary (DeploymentState), one contract (`models.py`).

## Status

Phase 1, early build. The implementation requirements live in
[`requirements/gpu-orchestrator-requirements.md`](requirements/gpu-orchestrator-requirements.md) (v1.2).
Build order and architecture constraints are in [`CLAUDE.md`](CLAUDE.md).

## Development

```bash
uv sync --extra dev
uv run python -m pytest tests/ -v
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```
