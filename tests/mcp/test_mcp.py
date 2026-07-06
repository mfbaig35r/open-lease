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
