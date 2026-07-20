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

from ..core import usage
from .limiter import DeploymentLimiter, get_limiter

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

    # Per-deployment concurrency gates (Tier A3) and per-model round-robin counters (Tier B),
    # created lazily and kept for this proxy's lifetime.
    limiters: dict[str, DeploymentLimiter] = {}
    rr_state: dict[str, int] = {}

    async def models(_: Request) -> Response:
        return JSONResponse({"object": "list", "data": _ready_models(orchestrator)})

    async def forward(request: Request) -> Response:
        return await _forward(orchestrator, request, transport, limiters, rr_state)

    routes = [Route("/v1/models", models, methods=["GET"])]
    routes += [Route(path, forward, methods=["POST"]) for path in sorted(_FORWARDED_ROUTES)]
    return Starlette(routes=routes)


def _ready_models(orchestrator: Orchestrator) -> list[dict]:
    """One entry per READY model, deduped across replicas (Tier B): several deployments serving the
    same model appear once in `/v1/models`, keyed by catalog id (what `gpu models` shows)."""
    seen: dict[str, dict] = {}
    for d in orchestrator.list_deployments():
        if d.observed_state.value == "ready" and d.endpoint_url:
            seen.setdefault(
                d.model_id,
                {"id": d.model_id, "object": "model", "owned_by": "gpu-orchestrator"},
            )
    return list(seen.values())


def _route_table(orchestrator: Orchestrator) -> dict[str, list[tuple[str, str, str]]]:
    """model-name -> list of (deployment_id, endpoint_url, served_model) for every READY deployment
    serving it, keyed by both its catalog id and its HF repo. More than one entry means replicas of
    the same model (Tier B); the proxy load-balances across the list. ``served_model`` is what the
    backend actually serves (the HF repo), which the request's ``model`` field is rewritten to.
    Sorted by deployment id so the round-robin rotation is stable. Rebuilt per request so routing
    always reflects live state."""
    hf_repo = {spec.id: spec.hf_repo for spec in orchestrator.list_models()}
    table: dict[str, list[tuple[str, str, str]]] = {}
    for d in sorted(orchestrator.list_deployments(), key=lambda d: d.id):
        if d.observed_state.value != "ready" or not d.endpoint_url:
            continue
        served = d.hf_repo or hf_repo.get(d.model_id, d.model_id)
        entry = (d.id, d.endpoint_url, served)
        table.setdefault(d.model_id, []).append(entry)
        if served != d.model_id:
            table.setdefault(served, []).append(entry)
    return table


def _pick(
    pool: list[tuple[str, str, str]], model: str, rr_state: dict[str, int]
) -> tuple[str, str, str]:
    """Round-robin one entry from a model's replica pool. State is a per-model counter held for the
    proxy's lifetime, so successive requests for the same model spread across its replicas."""
    index = rr_state.get(model, 0)
    rr_state[model] = index + 1
    return pool[index % len(pool)]


async def _forward(
    orchestrator: Orchestrator,
    request: Request,
    transport: httpx.AsyncBaseTransport | None,
    limiters: dict[str, DeploymentLimiter],
    rr_state: dict[str, int],
) -> Response:
    body = await request.body()
    payload = _parse(body)
    model = payload.get("model") if payload else None
    pool = _route_table(orchestrator).get(model) if model else None
    if not pool:
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

    # Load-balance across replicas of this model (Tier B); each keeps its own concurrency limiter.
    deployment_id, endpoint, served_model = _pick(pool, model, rr_state)
    # Concurrency envelope (Tier A3): take a slot before opening the upstream connection. A full
    # queue or a timed-out wait is a 429. The slot is held until the streamed response finishes.
    admitted, limiter = await _acquire_slot(orchestrator, deployment_id, limiters)
    if not admitted:
        return _too_many_requests(model)

    # Rewrite ONLY the model field to the id the backend serves (its HF repo); vLLM 404s on our
    # catalog id. This is the minimum the dual-key routing requires -- everything else is forwarded
    # unchanged. (Found live: byte-for-byte + catalog-id routing is self-contradictory.)
    if payload is not None and payload.get("model") != served_model:
        payload["model"] = served_model
        body = json.dumps(payload).encode()

    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(None))
    upstream_request = client.build_request(
        "POST",
        f"{endpoint}{request.url.path}",
        content=body,
        headers=_forward_headers(request.headers),
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except Exception:  # never leak a held slot if the upstream never opened
        if limiter is not None:
            limiter.release()
        await client.aclose()
        raise

    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}
    headers["x-gpu-orch-deployment-id"] = deployment_id
    # Meter successful token endpoints: tee the raw bytes into a buffer (forwarding is unchanged),
    # then tally usage in the background after the stream drains. Non-metered paths pass straight.
    metered = upstream.status_code == 200 and usage.is_metered(request.url.path)
    buf: list[bytes] = []
    stream = _tee(upstream.aiter_raw(), buf) if metered else upstream.aiter_raw()
    return StreamingResponse(
        stream,
        status_code=upstream.status_code,
        headers=headers,
        background=BackgroundTask(
            _finish,
            upstream,
            client,
            orchestrator if metered else None,
            deployment_id,
            buf,
            limiter,
        ),
    )


async def _acquire_slot(
    orchestrator: Orchestrator, deployment_id: str, limiters: dict[str, DeploymentLimiter]
) -> tuple[bool, DeploymentLimiter | None]:
    """Take a concurrency slot for the routed deployment. Returns ``(admitted, limiter)``: the
    limiter must be released later when admitted with a limit; ``(True, None)`` when the deployment
    sets no limit; ``(False, None)`` when the request must be rejected (429)."""
    deployment = orchestrator.get_deployment(deployment_id)
    if deployment.max_concurrency is None:
        return True, None
    limiter = get_limiter(
        limiters,
        deployment_id,
        deployment.max_concurrency,
        deployment.max_queue,
        deployment.queue_timeout_s,
    )
    return (True, limiter) if await limiter.acquire() else (False, None)


def _too_many_requests(model: str | None) -> Response:
    return JSONResponse(
        {
            "error": {
                "message": f"{model!r} is at its concurrency limit; retry shortly",
                "type": "rate_limit_error",
                "code": "concurrency_limit",
            }
        },
        status_code=429,
    )


async def _tee(source, sink: list[bytes]):
    async for chunk in source:
        sink.append(chunk)
        yield chunk


async def _finish(
    upstream: httpx.Response,
    client: httpx.AsyncClient,
    orchestrator: Orchestrator | None,
    deployment_id: str,
    buf: list[bytes],
    limiter: DeploymentLimiter | None,
) -> None:
    """Runs after the response stream drains (starlette runs it even on client disconnect), so this
    is the one place a held concurrency slot is released. The release is in ``finally`` so a failure
    to close or meter can never leak the slot."""
    try:
        await upstream.aclose()
        await client.aclose()
        if orchestrator is not None and buf:
            orchestrator.record_proxy_usage(deployment_id, b"".join(buf))
    finally:
        if limiter is not None:
            limiter.release()


def _parse(body: bytes) -> dict | None:
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _forward_headers(headers) -> dict[str, str]:
    forwarded = {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}
    # Don't let httpx inject its own Accept-Encoding: if the client didn't ask for compression, tell
    # the backend not to compress, so we never hand a client a gzip body it didn't request.
    if "accept-encoding" not in {k.lower() for k in forwarded}:
        forwarded["accept-encoding"] = "identity"
    return forwarded
