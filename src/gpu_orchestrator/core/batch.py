"""Batch inference (spec §13 extension): fan a file of prompts out over a READY deployment with
bounded concurrency and retries, then write the results.

This is throughput-bound work (parse N documents), not interactive: it saturates the GPU rather than
optimizing per-request latency, which is exactly where a self-hosted model is cheapest. It hits the
deployment endpoint directly (like `gpu chat`), records token usage so a batch shows up in
`gpu usage`, and turns one bad item into an error row rather than sinking the whole run.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..models import Deployment, _utcnow
from ..store import Store


@dataclass
class BatchItem:
    id: str
    messages: list[dict]


@dataclass
class BatchResult:
    id: str
    content: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_items(path: Path, *, system: str | None = None) -> list[BatchItem]:
    """Parse a JSONL file into batch items. Each non-blank line is one of: a full chat request
    ``{"id"?, "messages": [...]}``, a single turn ``{"id"?, "prompt": "..."}``, or a bare prompt
    string. A missing id defaults to the line index; ``system`` prepends a system turn to each item.
    """
    items: list[BatchItem] = []
    for i, raw in enumerate(path.read_text().splitlines()):
        line = raw.strip()
        if line:
            items.append(_parse_line(i, line, system))
    return items


def _parse_line(i: int, line: str, system: str | None) -> BatchItem:
    try:
        obj: object = json.loads(line)
    except json.JSONDecodeError:
        obj = line  # a bare prompt string, not JSON
    if isinstance(obj, dict) and "messages" in obj:
        item_id, messages = str(obj.get("id", i)), list(obj["messages"])
    elif isinstance(obj, dict):
        item_id = str(obj.get("id", i))
        messages = [{"role": "user", "content": str(obj.get("prompt", ""))}]
    else:
        item_id, messages = str(i), [{"role": "user", "content": str(obj)}]
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    return BatchItem(id=item_id, messages=messages)


async def run(
    store: Store,
    deployment: Deployment,
    served_model: str,
    items: list[BatchItem],
    *,
    concurrency: int = 64,
    max_tokens: int | None = None,
    temperature: float | None = None,
    retries: int = 3,
    transport: httpx.AsyncBaseTransport | None = None,
    on_done: Callable[[BatchResult], None] | None = None,
) -> list[BatchResult]:
    """Fan items out over the deployment endpoint with at most ``concurrency`` in flight. A
    semaphore bounds the window; every item is still scheduled, so passing 10k items is fine. Each
    success records its usage; failures (after retries) become error rows and never sink the run."""
    sem = asyncio.Semaphore(concurrency)
    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(300))

    async def one(item: BatchItem) -> BatchResult:
        async with sem:
            result = await _call_one(
                client,
                deployment.endpoint_url or "",
                served_model,
                item,
                max_tokens,
                temperature,
                retries,
            )
        if result.error is None:
            store.save_usage_record(
                deployment.id, result.prompt_tokens, result.completion_tokens, _utcnow()
            )
        if on_done is not None:
            on_done(result)
        return result

    try:
        return await asyncio.gather(*(one(item) for item in items))
    finally:
        await client.aclose()


async def _call_one(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    item: BatchItem,
    max_tokens: int | None,
    temperature: float | None,
    retries: int,
) -> BatchResult:
    body: dict = {"model": model, "messages": item.messages}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    last = "unknown error"
    for attempt in range(max(1, retries)):
        try:
            resp = await client.post(f"{endpoint}/v1/chat/completions", json=body)
            if (
                resp.status_code in (429,) or resp.status_code >= 500
            ):  # transient: back off and retry
                last = f"HTTP {resp.status_code}"
            else:
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                return BatchResult(
                    id=item.id,
                    content=data["choices"][0]["message"].get("content") or "",
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                )
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            last = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(min(30.0, 2.0**attempt))
    return BatchResult(id=item.id, error=last)


def write_results(path: Path, results: list[BatchResult]) -> None:
    with path.open("w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "id": r.id,
                        "content": r.content,
                        "error": r.error,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                    }
                )
                + "\n"
            )
