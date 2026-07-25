"""MCP server (spec §16, Phase 3). Agent-facing tools, each a thin wrapper over one Orchestrator
method. Same core as the CLI and REST API; the shape here is tools for an agent.

Tools return JSON-serializable dicts (the §6 models via ``model_dump``). Destructive tools require
an explicit ``confirm`` argument. ``chat_completion`` reuses the proxy's model-name routing.

The capacity-envelope tools (plan tiers A + B) matter most here: an agent that can deploy GPUs but
cannot cap them is the dangerous shape. Schedules, concurrency limits, budgets, replicas, and
autoscaling are all reachable, and the schedule tool takes the same human window spec as the CLI
(``"mon-fri 06:00-18:00"``) via the shared parser in ``core.schedule``.

Two Orchestrator methods are deliberately not tools (tracked, not silently missing): ``run_batch``,
because a fan-out over thousands of prompts outlives a tool call and wants a job resource to poll,
and ``delete_volume``, because a shared model cache is not an agent's to destroy (the CLI keeps it).
"""

from __future__ import annotations

import httpx
from fastmcp import FastMCP

from ..core.orchestrator import Orchestrator
from ..core.schedule import build_schedule
from ..models import BudgetAction, BudgetWindow
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
    async def scale_model(model_id: str, replicas: int, wait: bool = False) -> list[dict]:
        """Set how many load-balanced replicas serve a model (Tier B1). The proxy round-robins over
        every READY deployment of the model, so capacity adds up. Scaling up clones an existing
        deployment (needs at least one active member); scaling down stops the newest surplus."""
        deps = await orchestrator.scale(model_id, replicas, wait=wait)
        return [d.model_dump(mode="json") for d in deps]

    @mcp.tool
    async def set_schedule(
        deployment_id: str,
        on: list[str] | None = None,
        off: list[str] | None = None,
        timezone: str = "UTC",
        default: str = "off",
    ) -> dict:
        """Make a deployment's capacity follow an operating schedule (Tier A1), so spend stops
        outside its windows instead of running until someone stops it. Each window is
        "<days> HH:MM-HH:MM", e.g. on=["mon-fri 06:00-18:00"]; days are mon..sun, a range
        (mon-fri), a list (mon,wed,fri), or "all". ``timezone`` is an IANA name
        (America/New_York) and windows are wall-clock in it. ``default`` is the posture when no
        window matches (on|off). Needs a running daemon to take effect."""
        schedule = build_schedule(on or [], off or [], timezone=timezone, default=default)
        dep = await orchestrator.set_schedule(deployment_id, schedule)
        return dep.model_dump(mode="json")

    @mcp.tool
    async def clear_schedule(deployment_id: str) -> dict:
        """Remove a deployment's schedule, returning it to manual start/stop control."""
        return (await orchestrator.clear_schedule(deployment_id)).model_dump(mode="json")

    @mcp.tool
    async def set_limits(
        deployment_id: str,
        max_concurrency: int,
        max_queue: int = 0,
        queue_timeout_s: float = 30.0,
    ) -> dict:
        """Cap in-flight requests to a deployment (Tier A3), enforced by the OpenAI proxy. It admits
        up to ``max_concurrency`` at once; up to ``max_queue`` more wait ``queue_timeout_s`` seconds
        for a slot, and anything beyond gets a 429. A running proxy picks the change up on its next
        restart."""
        dep = await orchestrator.set_limits(
            deployment_id,
            max_concurrency=max_concurrency,
            max_queue=max_queue,
            queue_timeout_s=queue_timeout_s,
        )
        return dep.model_dump(mode="json")

    @mcp.tool
    async def clear_limits(deployment_id: str) -> dict:
        """Remove a deployment's concurrency limit (unlimited)."""
        dep = await orchestrator.set_limits(deployment_id, max_concurrency=None)
        return dep.model_dump(mode="json")

    @mcp.tool
    async def set_budget(
        limit_usd: float,
        window: str = "monthly",
        on_exceed: str = "warn",
        deployment_id: str | None = None,
        warn_fraction: float = 0.8,
    ) -> dict:
        """Set a spend ceiling (Tier A2). ``window`` is daily or monthly; ``deployment_id`` None is
        account-wide. ``on_exceed``: warn (event only), stop (tear down the in-scope deployments for
        the rest of the window), or block_new (refuse new deploys while over). A hard ceiling needs
        the daemon running to enforce it."""
        budget = await orchestrator.set_budget(
            limit_usd=limit_usd,
            window=BudgetWindow(window),
            on_exceed=BudgetAction(on_exceed),
            deployment_id=deployment_id,
            warn_fraction=warn_fraction,
        )
        return budget.model_dump(mode="json")

    @mcp.tool
    def list_budgets() -> list[dict]:
        """Every budget with how much it has spent so far this window (and whether it is over)."""
        return [s.model_dump(mode="json") for s in orchestrator.budget_status()]

    @mcp.tool
    async def remove_budget(budget_id: str) -> dict:
        """Delete a budget by id. Any teardown hold it set clears on the daemon's next tick."""
        if not await orchestrator.remove_budget(budget_id):
            return {"error": f"no budget with id {budget_id!r}"}
        return {"removed": budget_id}

    @mcp.tool
    async def set_autoscale(
        model_id: str,
        max_replicas: int,
        target_rpm_per_replica: float,
        min_replicas: int = 1,
    ) -> dict:
        """Keep a model's replica count matched to its served request rate (Tier B2), between
        ``min_replicas`` and ``max_replicas``, at about ``target_rpm_per_replica`` requests per
        minute each. Rejected demand (a 429) is not part of the signal, so set the target below a
        replica's ceiling. Needs the daemon running."""
        policy = await orchestrator.set_autoscale(
            model_id=model_id,
            max_replicas=max_replicas,
            target_rpm_per_replica=target_rpm_per_replica,
            min_replicas=min_replicas,
        )
        return policy.model_dump(mode="json")

    @mcp.tool
    def list_autoscale() -> list[dict]:
        """Autoscaling policies by model."""
        return [p.model_dump(mode="json") for p in orchestrator.list_autoscale()]

    @mcp.tool
    async def remove_autoscale(model_id: str) -> dict:
        """Delete a model's autoscaling policy (its replica count stays where it is)."""
        if not await orchestrator.remove_autoscale(model_id):
            return {"error": f"no autoscaling policy for {model_id!r}"}
        return {"removed": model_id}

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
    def deployment_events(deployment_id: str | None = None) -> list[dict]:
        """The event log for a deployment (or all deployments when omitted): every lifecycle step,
        reconcile action, budget decision, and failure, in order. The first place to look when a
        deployment did not do what was asked."""
        return [e.model_dump(mode="json") for e in orchestrator.events(deployment_id)]

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
    async def list_volumes() -> list[dict]:
        """Persistent model-cache network volumes (a warm cache cuts cold-start download time)."""
        return [v.model_dump(mode="json") for v in await orchestrator.list_volumes()]

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
        pool = _route_table(orchestrator).get(model)
        if not pool:
            return {"error": f"model {model!r} is not a READY deployment (try list_deployments)"}
        _deployment_id, endpoint, served = pool[0]  # first ready replica (Tier B)
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
