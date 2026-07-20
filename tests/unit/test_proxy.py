"""Step-8 OpenAI proxy (spec §13): routing by model name to a READY deployment's endpoint, byte
passthrough with the deployment-id header, and the catalog-id / hf_repo dual-key match that the
live gauntlet showed is necessary (vLLM advertises the HF repo, not our catalog id)."""

from __future__ import annotations

import asyncio
import json

import httpx
from starlette.testclient import TestClient

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.orchestrator import Orchestrator
from gpu_orchestrator.models import Deployment, DeploymentState, Instance
from gpu_orchestrator.proxy.openai_proxy import create_proxy_app
from gpu_orchestrator.store import Store
from tests.fixtures.catalog import QWEN3_06B_PROFILE

_ENDPOINT = "http://pod-xyz:8000"


async def _stream(payload: bytes):
    # An async body so the mock response is a real stream (not eagerly buffered), matching how the
    # proxy forwards vLLM's SSE; a buffered json= response trips httpx's StreamConsumed guard.
    yield payload


def _upstream() -> httpx.MockTransport:
    # Stands in for the vLLM endpoint the proxy forwards to.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            body = json.dumps(
                {
                    "id": "cmpl-1",
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                }
            ).encode()
            return httpx.Response(
                200,
                content=_stream(body),
                headers={"content-type": "application/json", "x-upstream": "vllm"},
            )
        return httpx.Response(404, content=_stream(b""))

    return httpx.MockTransport(handler)


def _client(tmp_path, **limits) -> TestClient:
    # Seed a READY deployment straight into the store, then point an Orchestrator at the same db.
    # ``limits`` (max_concurrency / max_queue / queue_timeout_s) exercise the Tier A3 gate.
    store = Store(tmp_path / "proxy.db")
    store.save_deployment(
        Deployment(
            id="dep-p1",
            model_id="qwen3-0.6b",
            provider="mock",
            desired_state=DeploymentState.READY,
            observed_state=DeploymentState.READY,
            profile=QWEN3_06B_PROFILE,
            instance=Instance(
                provider_instance_id="pod-xyz",
                provider="mock",
                gpu_type="RTX-A4000",
                state="RUNNING",
            ),
            endpoint_url=_ENDPOINT,
            **limits,
        )
    )
    orch = Orchestrator(Config(namespace="test", state_db=tmp_path / "proxy.db"))
    return TestClient(create_proxy_app(orch, transport=_upstream()))


def test_models_lists_ready_deployments(tmp_path):
    resp = _client(tmp_path).get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert ids == ["qwen3-0.6b"]


def test_routes_by_catalog_id(tmp_path):
    resp = _client(tmp_path).post(
        "/v1/chat/completions", json={"model": "qwen3-0.6b", "messages": []}
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi"
    assert resp.headers["x-gpu-orch-deployment-id"] == "dep-p1"
    assert resp.headers["x-upstream"] == "vllm"  # upstream headers preserved


def test_meters_token_usage_on_success(tmp_path):
    # A forwarded chat completion is metered: the background task tallies the response's usage to
    # the routed deployment, without changing the bytes the client received.
    client = _client(tmp_path)
    resp = client.post("/v1/chat/completions", json={"model": "qwen3-0.6b", "messages": []})
    assert resp.status_code == 200
    assert resp.json()["usage"]["total_tokens"] == 20  # forwarding still intact
    assert Store(tmp_path / "proxy.db").get_usage_totals("dep-p1") == (1, 12, 8)


def test_does_not_meter_unrouted_requests(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert resp.status_code == 404
    assert Store(tmp_path / "proxy.db").get_usage_totals("dep-p1") == (0, 0, 0)


def test_routes_by_hf_repo(tmp_path):
    # vLLM/OpenAI clients echo the HF repo id back; the proxy must accept it too.
    resp = _client(tmp_path).post(
        "/v1/chat/completions", json={"model": "Qwen/Qwen3-0.6B", "messages": []}
    )
    assert resp.status_code == 200
    assert resp.headers["x-gpu-orch-deployment-id"] == "dep-p1"


def test_routes_adhoc_model_by_its_own_hf_repo(tmp_path):
    # An ad-hoc deployment (model_id NOT in the catalog) is routed by both its derived id and its
    # stored hf_repo, without the proxy consulting the catalog for the served model.
    store = Store(tmp_path / "proxy.db")
    store.save_deployment(
        Deployment(
            id="dep-adhoc",
            model_id="qwen3-14b",
            hf_repo="Qwen/Qwen3-14B",
            provider="mock",
            desired_state=DeploymentState.READY,
            observed_state=DeploymentState.READY,
            profile=QWEN3_06B_PROFILE,
            instance=Instance(
                provider_instance_id="pod-a", provider="mock", gpu_type="X", state="RUNNING"
            ),
            endpoint_url=_ENDPOINT,
        )
    )
    client = TestClient(
        create_proxy_app(
            Orchestrator(Config(namespace="test", state_db=tmp_path / "proxy.db")),
            transport=_upstream(),
        )
    )
    for name in ("qwen3-14b", "Qwen/Qwen3-14B"):
        resp = client.post("/v1/chat/completions", json={"model": name, "messages": []})
        assert resp.status_code == 200
        assert resp.headers["x-gpu-orch-deployment-id"] == "dep-adhoc"


def test_rewrites_model_to_served_id(tmp_path):
    # The client sends our catalog id; vLLM only knows the HF repo, so the proxy must rewrite the
    # model field before forwarding (found live: byte-for-byte + catalog-id routing 404s at vLLM).
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(
            200, content=_stream(b'{"ok":true}'), headers={"content-type": "application/json"}
        )

    store = Store(tmp_path / "proxy.db")
    store.save_deployment(
        Deployment(
            id="dep-p1",
            model_id="qwen3-0.6b",
            provider="mock",
            desired_state=DeploymentState.READY,
            observed_state=DeploymentState.READY,
            profile=QWEN3_06B_PROFILE,
            endpoint_url=_ENDPOINT,
        )
    )
    orch = Orchestrator(Config(namespace="test", state_db=tmp_path / "proxy.db"))
    client = TestClient(create_proxy_app(orch, transport=httpx.MockTransport(handler)))

    client.post("/v1/chat/completions", json={"model": "qwen3-0.6b", "messages": []})
    assert seen["model"] == "Qwen/Qwen3-0.6B"  # rewritten to what the backend serves


def test_unknown_model_is_404(tmp_path):
    resp = _client(tmp_path).post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


def test_non_ready_deployment_not_routable(tmp_path):
    store = Store(tmp_path / "proxy.db")
    store.save_deployment(
        Deployment(
            id="dep-p2",
            model_id="qwen3-0.6b",
            provider="mock",
            desired_state=DeploymentState.READY,
            observed_state=DeploymentState.STARTING,  # not READY
            profile=QWEN3_06B_PROFILE,
            endpoint_url=_ENDPOINT,
        )
    )
    orch = Orchestrator(Config(namespace="test", state_db=tmp_path / "proxy.db"))
    client = TestClient(create_proxy_app(orch, transport=_upstream()))
    assert client.get("/v1/models").json()["data"] == []
    assert (
        client.post(
            "/v1/chat/completions", json={"model": "qwen3-0.6b", "messages": []}
        ).status_code
        == 404
    )


def test_limited_deployment_serves_and_releases_between_requests(tmp_path):
    # With a cap of 1 and no concurrency, two sequential requests both succeed: the slot from the
    # first must be released once its response finishes, or the second would 429.
    client = _client(tmp_path, max_concurrency=1, max_queue=0)
    for _ in range(2):
        resp = client.post("/v1/chat/completions", json={"model": "qwen3-0.6b", "messages": []})
        assert resp.status_code == 200


async def test_over_concurrency_limit_returns_429(tmp_path):
    gate = asyncio.Event()

    async def slow_body():
        yield b'{"choices":[{"message":{"content":"hi"}}],'
        await gate.wait()  # hold the slot open until the test releases it
        yield b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=slow_body(), headers={"content-type": "application/json"}
        )

    store = Store(tmp_path / "proxy.db")
    store.save_deployment(
        Deployment(
            id="dep-p1",
            model_id="qwen3-0.6b",
            provider="mock",
            desired_state=DeploymentState.READY,
            observed_state=DeploymentState.READY,
            profile=QWEN3_06B_PROFILE,
            endpoint_url=_ENDPOINT,
            max_concurrency=1,
            max_queue=0,
        )
    )
    orch = Orchestrator(Config(namespace="test", state_db=tmp_path / "proxy.db"))
    app = create_proxy_app(orch, transport=httpx.MockTransport(handler))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        body = {"model": "qwen3-0.6b", "messages": []}
        first = asyncio.create_task(client.post("/v1/chat/completions", json=body))
        await asyncio.sleep(0.05)  # let the first request take the only slot and block on the gate

        second = await client.post("/v1/chat/completions", json=body)
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "concurrency_limit"

        gate.set()  # release the first; its slot frees
        assert (await first).status_code == 200


def _seed_replica(store, dep_id, endpoint, model_id="qwen3-0.6b"):
    store.save_deployment(
        Deployment(
            id=dep_id,
            model_id=model_id,
            provider="mock",
            desired_state=DeploymentState.READY,
            observed_state=DeploymentState.READY,
            profile=QWEN3_06B_PROFILE,
            endpoint_url=endpoint,
        )
    )


def test_load_balances_across_replicas(tmp_path):
    # Two READY deployments serving the same model are a replica pool; the proxy round-robins across
    # them, so successive requests land on different pods (Tier B).
    store = Store(tmp_path / "proxy.db")
    _seed_replica(store, "dep-p1", "http://pod-a:8000")
    _seed_replica(store, "dep-p2", "http://pod-b:8000")
    orch = Orchestrator(Config(namespace="test", state_db=tmp_path / "proxy.db"))
    client = TestClient(create_proxy_app(orch, transport=_upstream()))

    seen = set()
    for _ in range(4):
        resp = client.post("/v1/chat/completions", json={"model": "qwen3-0.6b", "messages": []})
        assert resp.status_code == 200
        seen.add(resp.headers["x-gpu-orch-deployment-id"])
    assert seen == {"dep-p1", "dep-p2"}  # both replicas received traffic


def test_models_dedupes_replicas(tmp_path):
    store = Store(tmp_path / "proxy.db")
    _seed_replica(store, "dep-p1", "http://pod-a:8000")
    _seed_replica(store, "dep-p2", "http://pod-b:8000")
    orch = Orchestrator(Config(namespace="test", state_db=tmp_path / "proxy.db"))
    client = TestClient(create_proxy_app(orch, transport=_upstream()))

    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["qwen3-0.6b"]  # one entry, not one per replica
