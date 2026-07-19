"""MCP server (spec §16, Phase 3). Agent-facing tools, each a thin wrapper over one Orchestrator
method. Same core as the CLI and REST API; the shape here is tools for an agent.

Tools return JSON-serializable dicts (the §6 models via ``model_dump``). Destructive tools require
an explicit ``confirm`` argument. ``chat_completion`` reuses the proxy's model-name routing.
"""

from __future__ import annotations

import httpx
from fastmcp import FastMCP

from ..core.orchestrator import Orchestrator
from ..proxy.openai_proxy import _route_table


def create_server(orchestrator: Orchestrator) -> FastMCP:
    mcp = FastMCP("open-lease")

    @mcp.tool
    async def deploy_model(
        model_id: str, provider: str = "runpod", gpu: str | None = None, wait: bool = False
    ) -> dict:
        """Deploy a catalog model by id (see list_models). Returns the deployment record. With
        wait=false it returns immediately (a daemon must reconcile it); wait=true blocks until READY
        or FAILED. To deploy a model not in the catalog, use deploy_hf_model."""
        dep = await orchestrator.deploy_model(model_id, provider=provider, gpu=gpu, wait=wait)
        return dep.model_dump(mode="json")

    @mcp.tool
    async def deploy_hf_model(
        hf_repo: str, gpu: str, provider: str = "runpod", context: int = 0, wait: bool = False
    ) -> dict:
        """Deploy ANY vLLM-servable Hugging Face repo with no catalog entry (e.g.
        hf_repo="Qwen/Qwen3-14B"). The engine is model-neutral. gpu is required (an ad-hoc model has
        no recommended GPU); context=0 lets vLLM auto-detect max length. Returns the deployment."""
        dep = await orchestrator.deploy_adhoc(
            hf_repo=hf_repo, gpu=gpu, provider=provider, context_window=context, wait=wait
        )
        return dep.model_dump(mode="json")

    @mcp.tool
    async def stop_deployment(deployment_id: str) -> dict:
        """Stop a deployment (destroys the pod, keeps the record so it can be restarted)."""
        return (await orchestrator.stop_deployment(deployment_id)).model_dump(mode="json")

    @mcp.tool
    async def restart_deployment(deployment_id: str) -> dict:
        """Restart a deployment: a full stop then redeploy of the same profile (a cold start)."""
        return (await orchestrator.restart_deployment(deployment_id)).model_dump(mode="json")

    @mcp.tool
    async def delete_deployment(deployment_id: str, confirm: bool = False) -> dict:
        """Destroy a deployment and permanently remove its record. DESTRUCTIVE: call with
        confirm=true to proceed."""
        if not confirm:
            return {"error": "delete is destructive; call again with confirm=true"}
        await orchestrator.delete_deployment(deployment_id)
        return {"deleted": deployment_id}

    @mcp.tool
    def list_models() -> list[dict]:
        """List the model catalog (ids, GPU needs, capabilities)."""
        return [m.model_dump(mode="json") for m in orchestrator.list_models()]

    @mcp.tool
    def list_deployments(include_stopped: bool = False) -> list[dict]:
        """List deployments and their current state."""
        return [
            d.model_dump(mode="json")
            for d in orchestrator.list_deployments(include_stopped=include_stopped)
        ]

    @mcp.tool
    def get_deployment(deployment_id: str) -> dict:
        """Get one deployment's full record (state, endpoint, instance, history)."""
        return orchestrator.get_deployment(deployment_id).model_dump(mode="json")

    @mcp.tool
    async def deployment_logs(deployment_id: str, tail: int = 100) -> list[str]:
        """Recent provider/runtime log lines for a deployment."""
        return list(await orchestrator.get_logs(deployment_id, tail=tail))

    @mcp.tool
    async def deployment_health(deployment_id: str) -> dict:
        """Check-by-check health of a deployment."""
        return (await orchestrator.get_health(deployment_id)).model_dump(mode="json")

    @mcp.tool
    async def provider_status() -> list[dict]:
        """Configured providers and their capabilities (GPU menu, regions)."""
        return [p.model_dump(mode="json") for p in await orchestrator.list_providers()]

    @mcp.tool
    async def gpu_availability(model_id: str | None = None) -> list[dict]:
        """Per-data-center GPU availability, optionally for a specific model's GPU."""
        rows = await orchestrator.gpu_availability(model_id=model_id)
        return [r.model_dump(mode="json") for r in rows]

    @mcp.tool
    async def estimate_cost(model_id: str, provider: str = "runpod", hours: float = 1.0) -> dict:
        """Estimate the cost of running a model for some hours, without deploying."""
        est = await orchestrator.estimate_cost(model_id, provider=provider, hours=hours)
        return est.model_dump(mode="json")

    @mcp.tool
    def get_costs(deployment_id: str | None = None) -> list[dict]:
        """Accrued cost records, optionally for one deployment."""
        return [c.model_dump(mode="json") for c in orchestrator.get_costs(deployment_id)]

    @mcp.tool
    def get_usage(deployment_id: str | None = None) -> list[dict]:
        """Token throughput and cost-per-token per deployment: requests, tokens, tokens/sec
        (utilization), accrued cost, and $/million-tokens (the crossover vs per-token API)."""
        return [u.model_dump(mode="json") for u in orchestrator.get_usage(deployment_id)]

    @mcp.tool
    async def chat_completion(model: str, messages: list[dict]) -> dict:
        """Chat with a READY deployment. ``model`` is the catalog id or the HF repo; ``messages`` is
        the OpenAI chat format. Routes to the matching deployment's endpoint."""
        route = _route_table(orchestrator).get(model)
        if route is None:
            return {"error": f"model {model!r} is not a READY deployment (try list_deployments)"}
        _deployment_id, endpoint, served = route
        async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
            resp = await client.post(
                f"{endpoint}/v1/chat/completions", json={"model": served, "messages": messages}
            )
            return resp.json()

    return mcp


def run() -> None:
    """Entry point (``gpu-mcp`` / ``gpu mcp``): serve over stdio for an MCP client."""
    from ..config import Config

    create_server(Orchestrator(Config())).run()
