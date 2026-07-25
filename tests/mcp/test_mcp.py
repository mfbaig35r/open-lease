"""Phase 3 MCP tools: thin wrappers over the Orchestrator, driven through FastMCP's in-memory
client against a mock-backed core. Destructive tools require confirm."""

from __future__ import annotations

import httpx
import pytest
from fastmcp import Client

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.orchestrator import Orchestrator
from gpu_orchestrator.mcp.server import create_server
from gpu_orchestrator.providers.mock import MockProvider
from gpu_orchestrator.runtimes.vllm import VLLMRuntime


def _runtime() -> VLLMRuntime:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3-0.6b"}]})
        return httpx.Response(404)

    return VLLMRuntime(transport=httpx.MockTransport(handler))


@pytest.fixture
def server(tmp_path):
    cfg = Config(namespace="test", state_db=tmp_path / "mcp.db", reconcile_interval=0)
    orch = Orchestrator(cfg, provider=MockProvider(namespace="test"), runtime=_runtime())
    return create_server(orch)


async def test_list_models(server):
    async with Client(server) as client:
        result = await client.call_tool("list_models", {})
        assert "qwen3-0.6b" in [m["id"] for m in result.data]


async def test_deploy_and_get(server):
    async with Client(server) as client:
        deployed = await client.call_tool(
            "deploy_model", {"model_id": "qwen3-0.6b", "provider": "mock", "wait": True}
        )
        assert deployed.data["observed_state"] == "ready"
        got = await client.call_tool("get_deployment", {"deployment_id": deployed.data["id"]})
        assert got.data["observed_state"] == "ready"


async def test_delete_requires_confirm(server):
    async with Client(server) as client:
        dep = await client.call_tool(
            "deploy_model", {"model_id": "qwen3-0.6b", "provider": "mock", "wait": True}
        )
        dep_id = dep.data["id"]
        unconfirmed = await client.call_tool("delete_deployment", {"deployment_id": dep_id})
        assert "error" in unconfirmed.data  # destructive: refused without confirm
        confirmed = await client.call_tool(
            "delete_deployment", {"deployment_id": dep_id, "confirm": True}
        )
        assert confirmed.data["deleted"] == dep_id


async def test_estimate_cost(server):
    async with Client(server) as client:
        result = await client.call_tool(
            "estimate_cost", {"model_id": "qwen3-0.6b", "provider": "mock"}
        )
        assert result.data["gpu_hourly_usd"] == 0.17


# --- capacity envelope (plan tiers A + B): an agent that can deploy can also cap -------


async def _deploy(client) -> str:
    dep = await client.call_tool(
        "deploy_model", {"model_id": "qwen3-0.6b", "provider": "mock", "wait": True}
    )
    return dep.data["id"]


async def test_capacity_tools_are_exposed(server):
    """Parity guard: the tools that make capacity bounded must exist, or an agent can spend without
    being able to set a ceiling."""
    async with Client(server) as client:
        names = {t.name for t in await client.list_tools()}
    assert {
        "scale_model",
        "set_schedule",
        "clear_schedule",
        "set_limits",
        "clear_limits",
        "set_budget",
        "list_budgets",
        "remove_budget",
        "set_autoscale",
        "list_autoscale",
        "remove_autoscale",
        "deployment_events",
    } <= names


async def test_schedule_from_window_spec(server):
    async with Client(server) as client:
        dep_id = await _deploy(client)
        result = await client.call_tool(
            "set_schedule",
            {
                "deployment_id": dep_id,
                "on": ["mon-fri 06:00-18:00"],
                "timezone": "America/New_York",
            },
        )
        schedule = result.data["schedule"]
        assert schedule["timezone"] == "America/New_York"
        assert schedule["rules"][0]["days"] == [0, 1, 2, 3, 4]  # mon-fri
        assert schedule["rules"][0]["posture"] == "on"
        assert schedule["default_posture"] == "off"  # nothing else runs

        cleared = await client.call_tool("clear_schedule", {"deployment_id": dep_id})
        assert cleared.data["schedule"] is None


async def test_schedule_rejects_a_malformed_window(server):
    async with Client(server) as client:
        dep_id = await _deploy(client)
        with pytest.raises(Exception, match="HH:MM"):
            await client.call_tool(
                "set_schedule", {"deployment_id": dep_id, "on": ["mon-fri mornings"]}
            )


async def test_limits_set_and_clear(server):
    async with Client(server) as client:
        dep_id = await _deploy(client)
        limited = await client.call_tool(
            "set_limits", {"deployment_id": dep_id, "max_concurrency": 8, "max_queue": 32}
        )
        assert limited.data["max_concurrency"] == 8
        assert limited.data["max_queue"] == 32
        cleared = await client.call_tool("clear_limits", {"deployment_id": dep_id})
        assert cleared.data["max_concurrency"] is None


async def test_scale_model(server):
    async with Client(server) as client:
        await _deploy(client)
        scaled = await client.call_tool(
            "scale_model", {"model_id": "qwen3-0.6b", "replicas": 2, "wait": True}
        )
        assert len(scaled.data) == 2
        listed = await client.call_tool("list_deployments", {})
        assert len(listed.data) == 2


async def test_budget_lifecycle(server):
    async with Client(server) as client:
        created = await client.call_tool(
            "set_budget", {"limit_usd": 250, "window": "daily", "on_exceed": "stop"}
        )
        budget_id = created.data["id"]
        assert created.data["on_exceed"] == "stop"

        listed = await client.call_tool("list_budgets", {})
        assert listed.data[0]["budget"]["id"] == budget_id
        assert listed.data[0]["spent_usd"] == 0.0  # nothing accrued yet

        removed = await client.call_tool("remove_budget", {"budget_id": budget_id})
        assert removed.data["removed"] == budget_id
        missing = await client.call_tool("remove_budget", {"budget_id": budget_id})
        assert "error" in missing.data


async def test_autoscale_lifecycle(server):
    async with Client(server) as client:
        policy = await client.call_tool(
            "set_autoscale",
            {"model_id": "qwen3-0.6b", "max_replicas": 4, "target_rpm_per_replica": 30},
        )
        assert policy.data["max_replicas"] == 4
        assert policy.data["min_replicas"] == 1
        listed = await client.call_tool("list_autoscale", {})
        assert listed.data[0]["model_id"] == "qwen3-0.6b"

        removed = await client.call_tool("remove_autoscale", {"model_id": "qwen3-0.6b"})
        assert removed.data["removed"] == "qwen3-0.6b"
        missing = await client.call_tool("remove_autoscale", {"model_id": "qwen3-0.6b"})
        assert "error" in missing.data


async def test_list_volumes(server):
    async with Client(server) as client:
        result = await client.call_tool("list_volumes", {})
        assert result.data == []  # no cache volume created in this run


async def test_deployment_events(server):
    async with Client(server) as client:
        dep_id = await _deploy(client)
        events = await client.call_tool("deployment_events", {"deployment_id": dep_id})
        kinds = [e["kind"] for e in events.data]
        assert "deployment_requested" in kinds
        assert "deployment_ready" in kinds
