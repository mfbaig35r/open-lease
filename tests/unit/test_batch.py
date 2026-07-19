"""Batch inference: parse a prompts file, fan out over a deployment endpoint with bounded
concurrency + retries, record usage, and never let one bad item sink the run (§13 extension)."""

from __future__ import annotations

import json

import httpx

from gpu_orchestrator.core import batch
from gpu_orchestrator.models import DeploymentState
from gpu_orchestrator.store import Store
from tests.fixtures.deployments import make_deployment


def test_load_items_parses_three_line_formats(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(
        '{"id": "x", "messages": [{"role": "user", "content": "full"}]}\n'
        '{"prompt": "just a prompt"}\n'
        "a bare string prompt\n"
        "\n"  # blank lines skipped
    )
    items = batch.load_items(path, system="be terse")
    assert [i.id for i in items] == ["x", "1", "2"]  # explicit id, then line indices
    # every item gets the system turn prepended
    assert all(i.messages[0] == {"role": "system", "content": "be terse"} for i in items)
    assert items[0].messages[1]["content"] == "full"
    assert items[1].messages[1]["content"] == "just a prompt"
    assert items[2].messages[1]["content"] == "a bare string prompt"


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][-1]["content"]
        if content == "boom":
            return httpx.Response(500, json={"error": "kaboom"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f"echo:{content}"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    return httpx.MockTransport(handler)


async def test_run_collects_results_and_meters_only_successes(tmp_path):
    store = Store(tmp_path / "b.db")
    dep = make_deployment(DeploymentState.READY)
    dep.endpoint_url = "http://pod:8000"
    items = [
        batch.BatchItem("a", [{"role": "user", "content": "hi"}]),
        batch.BatchItem("b", [{"role": "user", "content": "boom"}]),  # forces a 500
    ]

    results = await batch.run(store, dep, "Qwen/X", items, retries=1, transport=_transport())

    by_id = {r.id: r for r in results}
    assert by_id["a"].content == "echo:hi" and by_id["a"].error is None
    assert by_id["b"].error is not None  # the 500 becomes an error row, not an exception
    # Only the successful call is metered.
    assert store.get_usage_totals(dep.id) == (1, 3, 2)


def test_write_results_round_trips(tmp_path):
    out = tmp_path / "out.jsonl"
    batch.write_results(
        out,
        [
            batch.BatchResult("a", content="hello", prompt_tokens=3, completion_tokens=2),
            batch.BatchResult("b", error="boom"),
        ],
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0] == {
        "id": "a",
        "content": "hello",
        "error": None,
        "prompt_tokens": 3,
        "completion_tokens": 2,
    }
    assert rows[1]["error"] == "boom" and rows[1]["content"] is None
