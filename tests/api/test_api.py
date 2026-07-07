"""Phase 2 REST API: the FastAPI layer over the Orchestrator, driven against a mock-backed core.
Routes mirror the Orchestrator 1:1; the OpenAI proxy is mounted at /v1/*; auth is a bearer token."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from gpu_orchestrator.api import create_app
from gpu_orchestrator.config import Config
from gpu_orchestrator.core.orchestrator import Orchestrator
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


def _client(tmp_path, *, api_token: str | None = None) -> TestClient:
    cfg = Config(
        namespace="test", state_db=tmp_path / "api.db", reconcile_interval=0, api_token=api_token
    )
    orch = Orchestrator(cfg, provider=MockProvider(namespace="test"), runtime=_runtime())
    return TestClient(create_app(orch))


def test_deploy_list_get(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/deployments", json={"model_id": "qwen3-0.6b", "provider": "mock", "wait": True}
    )
    assert resp.status_code == 200
    dep = resp.json()
    assert dep["observed_state"] == "ready"

    assert client.get("/deployments").json()[0]["id"] == dep["id"]
    assert client.get(f"/deployments/{dep['id']}").json()["observed_state"] == "ready"


def test_deploy_adhoc_hf_repo(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/deployments",
        json={"hf_repo": "Qwen/Qwen3-14B", "gpu": "MOCK-GPU", "provider": "mock", "wait": True},
    )
    assert resp.status_code == 200
    dep = resp.json()
    assert dep["model_id"] == "qwen3-14b"  # derived; no catalog entry
    assert dep["hf_repo"] == "Qwen/Qwen3-14B"
    assert dep["observed_state"] == "ready"


def test_deploy_adhoc_requires_gpu(tmp_path):
    resp = _client(tmp_path).post(
        "/deployments", json={"hf_repo": "Qwen/Qwen3-14B", "provider": "mock"}
    )
    assert resp.status_code == 400
    assert "gpu" in resp.json()["error"]


def test_get_unknown_is_404(tmp_path):
    resp = _client(tmp_path).get("/deployments/nope")
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_models_and_estimate(tmp_path):
    client = _client(tmp_path)
    assert "qwen3-0.6b" in [m["id"] for m in client.get("/models").json()]
    est = client.post("/estimate", json={"model_id": "qwen3-0.6b", "provider": "mock"})
    assert est.status_code == 200
    assert est.json()["gpu_hourly_usd"] == 0.17  # RTX-A4000 rate


def test_stop_and_delete(tmp_path):
    client = _client(tmp_path)
    dep_id = client.post(
        "/deployments", json={"model_id": "qwen3-0.6b", "provider": "mock", "wait": True}
    ).json()["id"]

    assert client.post(f"/deployments/{dep_id}/stop").json()["observed_state"] == "stopped"
    assert client.delete(f"/deployments/{dep_id}").status_code == 204
    assert client.get(f"/deployments/{dep_id}").status_code == 404


def test_proxy_mounted_at_v1(tmp_path):
    client = _client(tmp_path)
    client.post("/deployments", json={"model_id": "qwen3-0.6b", "provider": "mock", "wait": True})
    body = client.get("/v1/models").json()  # served by the mounted proxy
    assert body["data"][0]["id"] == "qwen3-0.6b"


def test_bearer_auth(tmp_path):
    client = _client(tmp_path, api_token="s3cret")
    assert client.get("/models").status_code == 401  # no header
    ok = client.get("/models", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert client.get("/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
