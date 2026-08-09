"""Token-usage metering (spec §11 extension).

The OpenAI proxy forwards a response byte-for-byte; it also hands the response body and the routed
deployment id here so we can tally tokens without the proxy growing any accounting logic. This
module extracts usage from a response, records it, and derives the utilization + cost-per-token
summary that `gpu usage` shows. Parsing lives here (not the proxy) to keep the proxy a thin
forwarder.
"""

from __future__ import annotations

import json
from datetime import datetime

from ..models import Deployment, UsageSummary, _utcnow
from ..store import Store

# Only these endpoints carry an OpenAI `usage` block worth metering.
_METERED_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")


def is_metered(path: str) -> bool:
    return path in _METERED_PATHS


def extract_usage(body: bytes) -> tuple[int, int] | None:
    """(prompt_tokens, completion_tokens) from a response body, or None if absent. Handles the
    non-streaming JSON object and a streaming SSE body (usage rides the final `data:` chunk when the
    client sent `stream_options.include_usage`)."""
    text = body.decode("utf-8", "ignore").strip()
    if not text:
        return None
    if text.startswith("{"):
        return _from_obj(_loads(text))
    for line in reversed(text.splitlines()):  # SSE: last data: line carrying usage wins
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        got = _from_obj(_loads(payload))
        if got is not None:
            return got
    return None


def record(store: Store, deployment_id: str, body: bytes, now: datetime | None = None) -> None:
    """Extract usage from a forwarded response and record it. No-op when the body carries none (an
    error response, or a stream the client did not ask usage for)."""
    got = extract_usage(body)
    if got is None or got == (0, 0):
        return
    store.save_usage_record(deployment_id, got[0], got[1], now or _utcnow())


def summary(store: Store, deployment: Deployment, now: datetime | None = None) -> UsageSummary:
    """Combine token totals with the deployment's cost records into the utilization view."""
    now = now or _utcnow()
    requests, prompt, completion = store.get_usage_totals(deployment.id)
    records = store.get_cost_records(deployment.id)
    accrued = round(sum(r.accrued_usd for r in records), 4)
    uptime = sum(((r.stopped_at or now) - r.started_at).total_seconds() for r in records)
    return UsageSummary(
        deployment_id=deployment.id,
        model_id=deployment.model_id,
        requests=requests,
        prompt_tokens=prompt,
        completion_tokens=completion,
        accrued_usd=accrued,
        uptime_seconds=round(uptime, 1),
    )


def observed_throughput(
    store: Store,
    model_id: str,
    *,
    gpu_names: set[str] | None = None,
    now: datetime | None = None,
) -> tuple[float, str] | None:
    """Tokens/sec measured for ``model_id`` across this install's own past deployments, paired with
    a short provenance string. ``None`` when nothing has been measured.

    Throughput is a property of (model, GPU, runtime), not of the model alone, so a measurement is
    only reused when the deployment's GPU is one of ``gpu_names``. Callers pass every alias for the
    GPU they mean (catalog id and provider SKU), because profiles record either one.

    Deployments with cost records but no tokens are skipped rather than counted as zero throughput:
    a pod that was rented and never asked to serve says nothing about how fast the model runs.
    """
    now = now or _utcnow()
    tokens = 0
    uptime = 0.0
    matched = 0
    for deployment in store.list_deployments(include_stopped=True):
        if deployment.model_id != model_id:
            continue
        if gpu_names is not None and deployment.profile.recommended_gpu not in gpu_names:
            continue
        _, prompt, completion = store.get_usage_totals(deployment.id)
        if not (prompt + completion):
            continue
        records = store.get_cost_records(deployment.id)
        seconds = sum(((r.stopped_at or now) - r.started_at).total_seconds() for r in records)
        if seconds <= 0:
            continue
        tokens += prompt + completion
        uptime += seconds
        matched += 1
    if not tokens or uptime <= 0:
        return None
    plural = "" if matched == 1 else "s"
    return round(tokens / uptime, 1), f"measured over {matched} past deployment{plural}"


def _loads(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _from_obj(obj: dict | None) -> tuple[int, int] | None:
    usage = obj.get("usage") if obj else None
    if not isinstance(usage, dict):
        return None
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
