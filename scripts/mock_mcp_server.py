"""A zero-cost MCP server for exercising the open-lease tools without renting a GPU.

Same tool surface as ``gpu-mcp``, but wired to the mock provider so nothing hits RunPod:

- a tiny local "vLLM" HTTP server answers /health, /v1/models, and /v1/chat/completions,
- the mock provider points deployments at that local server (so the real runtime's readiness
  probes pass and chat_completion reaches a real endpoint),
- an in-process daemon drives non-blocking deploys to READY on its own.

Run via the `open-lease-mock` Claude Desktop entry, or directly:
    uv run --extra mcp python scripts/mock_mcp_server.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.daemon import Daemon
from gpu_orchestrator.core.orchestrator import Orchestrator
from gpu_orchestrator.mcp.server import create_server
from gpu_orchestrator.providers.mock import _RUNNING, MockProvider
from gpu_orchestrator.runtimes.vllm import VLLMRuntime

MOCK_PORT = 8199
STATE_DB = Path.home() / ".gpu-orchestrator" / "mock-mcp-state.db"


# --- the fake vLLM endpoint every mock deployment points at --------------------------


async def _health(_request) -> Response:
    return Response(status_code=200)


async def _models(_request) -> JSONResponse:
    return JSONResponse({"object": "list", "data": [{"id": "mock-model", "object": "model"}]})


async def _chat(request) -> JSONResponse:
    body = await request.json()
    messages = body.get("messages", [])
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    reply = f'Mock reply — no GPU was used. You said: "{last_user}"'
    return JSONResponse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": body.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


mock_vllm = Starlette(
    routes=[
        Route("/health", _health),
        Route("/v1/models", _models),
        Route("/v1/chat/completions", _chat, methods=["POST"]),
    ]
)


class MockLocalProvider(MockProvider):
    """Mock provider that resolves a running pod to the local fake-vLLM server, so the real runtime
    can probe it and chat_completion (which uses its own httpx client) can reach it."""

    async def resolve_endpoint_url(self, instance, port: int) -> str | None:
        pod = self._pods.get(instance.provider_instance_id)
        if pod is None or pod.state(self.steps_to_running) != _RUNNING:
            return None
        return f"http://127.0.0.1:{MOCK_PORT}"


def _fresh_state_db() -> None:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        f = STATE_DB.with_name(STATE_DB.name + suffix)
        if f.exists():
            f.unlink()


async def main() -> None:
    _fresh_state_db()  # clean slate each launch (mock pods are in-memory anyway)
    cfg = Config(
        namespace="mockmcp",
        state_db=STATE_DB,
        reconcile_interval=1,
        orphan_sweep_interval=3600,
    )
    provider = MockLocalProvider(namespace="mockmcp")
    runtime = VLLMRuntime()  # real runtime; probes the local mock server
    orchestrator = Orchestrator(cfg, provider=provider, runtime=runtime)
    daemon = Daemon(cfg, provider=provider, runtime=runtime)  # same instances -> shared pods
    mcp = create_server(orchestrator)

    server = uvicorn.Server(
        uvicorn.Config(mock_vllm, host="127.0.0.1", port=MOCK_PORT, log_level="critical")
    )
    background = [asyncio.create_task(server.serve()), asyncio.create_task(daemon.run())]
    try:
        await mcp.run_stdio_async()
    finally:
        for task in background:
            task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
