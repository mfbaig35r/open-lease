"""OpenAI-compatible inference proxy (spec §13). The highest-value user-facing feature.

A small starlette + httpx app whose ONLY intelligence is routing: it maps a request's ``model``
field to a READY deployment's endpoint and forwards the request byte-for-byte, streaming the
response back with status codes preserved and an ``x-gpu-orch-deployment-id`` header added. No
payload normalization, no aliasing beyond the two exact keys below -- this must not become a
compatibility layer (§13).

Routing accepts both keys a client might send per deployment: the catalog/deployment id
(``qwen3-32b``) and the profile's HF repo (``Qwen/Qwen3-32B``), because vLLM advertises the HF repo
at its own ``/v1/models`` and OpenAI clients echo it back. Exact match on those two, nothing fuzzy.

It lives in the core package (not an interface layer): ``gpu proxy`` serves it, and Phase 2's
FastAPI mounts it rather than reimplementing it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from ..core.orchestrator import Orchestrator

# Hop-by-hop headers that must not be forwarded in either direction (RFC 7230).
_HOP_BY_HOP = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
}

_FORWARDED_ROUTES = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
}


def create_proxy_app(
    orchestrator: Orchestrator, *, transport: httpx.AsyncBaseTransport | None = None
) -> Starlette:
    """Build the ASGI proxy over an Orchestrator's READY deployments. ``transport`` is the test seam
    (an httpx.MockTransport standing in for the upstream vLLM endpoints)."""

    async def models(_: Request) -> Response:
        return JSONResponse({"object": "list", "data": _ready_models(orchestrator)})

    async def forward(request: Request) -> Response:
        return await _forward(orchestrator, request, transport)

    routes = [Route("/v1/models", models, methods=["GET"])]
    routes += [Route(path, forward, methods=["POST"]) for path in sorted(_FORWARDED_ROUTES)]
    return Starlette(routes=routes)


def _ready_models(orchestrator: Orchestrator) -> list[dict]:
    """One entry per READY deployment, keyed by catalog id (what `gpu models` shows)."""
    return [
        {"id": d.model_id, "object": "model", "owned_by": "gpu-orchestrator", "deployment_id": d.id}
        for d in orchestrator.list_deployments()
        if d.observed_state.value == "ready" and d.endpoint_url
    ]


def _route_table(orchestrator: Orchestrator) -> dict[str, tuple[str, str]]:
    """model-name -> (deployment_id, endpoint_url) for every READY deployment, under both its
    catalog id and its HF repo. Rebuilt per request so routing always reflects live state."""
    hf_repo = {spec.id: spec.hf_repo for spec in orchestrator.list_models()}
    table: dict[str, tuple[str, str]] = {}
    for d in orchestrator.list_deployments():
        if d.observed_state.value != "ready" or not d.endpoint_url:
            continue
        table[d.model_id] = (d.id, d.endpoint_url)
        if d.model_id in hf_repo:
            table[hf_repo[d.model_id]] = (d.id, d.endpoint_url)
    return table


async def _forward(
    orchestrator: Orchestrator, request: Request, transport: httpx.AsyncBaseTransport | None
) -> Response:
    body = await request.body()
    model = _model_of(body)
    route = _route_table(orchestrator).get(model) if model else None
    if route is None:
        return JSONResponse(
            {
                "error": {
                    "message": f"model {model!r} is not a READY deployment (try `gpu status`)",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
            status_code=404,
        )

    deployment_id, endpoint = route
    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(None))
    upstream_request = client.build_request(
        "POST",
        f"{endpoint}{request.url.path}",
        content=body,
        headers=_forward_headers(request.headers),
    )
    upstream = await client.send(upstream_request, stream=True)

    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}
    headers["x-gpu-orch-deployment-id"] = deployment_id
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=headers,
        background=BackgroundTask(_aclose, upstream, client),
    )


async def _aclose(upstream: httpx.Response, client: httpx.AsyncClient) -> None:
    await upstream.aclose()
    await client.aclose()


def _model_of(body: bytes) -> str | None:
    try:
        return json.loads(body).get("model")
    except (json.JSONDecodeError, AttributeError):
        return None


def _forward_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}
