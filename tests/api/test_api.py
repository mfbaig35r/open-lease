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


def test_deploy_adhoc_without_gpu_fails_clearly_when_the_model_cannot_be_sized(tmp_path):
    """gpu is optional now (issue #24), but an unreadable model must say so rather than guess."""
    cfg = Config(namespace="test", state_db=tmp_path / "api.db", reconcile_interval=0)
    orch = Orchestrator(
        cfg,
        provider=MockProvider(namespace="test"),
        runtime=_runtime(),
        hf_transport=httpx.MockTransport(lambda r: httpx.Response(401)),  # gated repo
    )
    resp = TestClient(create_app(orch)).post(
        "/deployments", json={"hf_repo": "meta-llama/Llama-3.1-8B-Instruct", "provider": "mock"}
    )
    assert resp.status_code in (400, 404)
    assert "gpu" in resp.json()["error"]


def test_serves_ui_when_ui_dir_set(tmp_path):
    ui = tmp_path / "web"
    ui.mkdir()
    (ui / "index.html").write_text("<html>workbench</html>")
    cfg = Config(namespace="test", state_db=tmp_path / "api.db", reconcile_interval=0)
    orch = Orchestrator(cfg, provider=MockProvider(namespace="test"), runtime=_runtime())
    client = TestClient(create_app(orch, ui_dir=ui))
    assert client.get("/").text == "<html>workbench</html>"  # UI at /
    assert client.get("/deployments").status_code == 200  # management API still works
    assert client.get("/v1/models").status_code == 200  # proxy /v1 still works alongside the UI


def test_ui_static_is_open_but_api_guarded_with_token(tmp_path):
    ui = tmp_path / "web"
    ui.mkdir()
    (ui / "index.html").write_text("ok")
    cfg = Config(
        namespace="test", state_db=tmp_path / "api.db", reconcile_interval=0, api_token="secret"
    )
    orch = Orchestrator(cfg, provider=MockProvider(namespace="test"), runtime=_runtime())
    client = TestClient(create_app(orch, ui_dir=ui))
    assert client.get("/").status_code == 200  # static assets load without a token
    assert client.get("/deployments").status_code == 401  # API still requires it


def _cors_app(tmp_path, origins):
    cfg = Config(namespace="test", state_db=tmp_path / "api.db", reconcile_interval=0)
    orch = Orchestrator(cfg, provider=MockProvider(namespace="test"), runtime=_runtime())
    return TestClient(create_app(orch, cors_origins=origins))


def test_cors_preflight_allows_configured_origin(tmp_path):
    origin = "https://openlease.canonicalresearch.dev"
    client = _cors_app(tmp_path, [origin])
    resp = client.options(
        "/deployments",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin
    # Chrome needs this ack to let a public HTTPS page reach a loopback server.
    assert resp.headers["access-control-allow-private-network"] == "true"
    # The real request echoes the origin too.
    got = client.get("/deployments", headers={"Origin": origin})
    assert got.headers["access-control-allow-origin"] == origin


def test_cors_rejects_unconfigured_origin(tmp_path):
    client = _cors_app(tmp_path, ["https://openlease.canonicalresearch.dev"])
    resp = client.get("/deployments", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers  # not the allowed origin


def test_cors_off_by_default(tmp_path):
    client = _client(tmp_path)  # no cors_origins
    resp = client.get("/deployments", headers={"Origin": "https://openlease.canonicalresearch.dev"})
    assert "access-control-allow-origin" not in resp.headers


def test_cors_origins_env_csv_parses():
    cfg = Config(cors_origins="https://a.example, https://b.example")
    assert cfg.cors_origins == ["https://a.example", "https://b.example"]


def test_cors_origins_from_the_actual_env_var(monkeypatch):
    """The documented way to set this is GPU_ORCH_CORS_ORIGINS, and the env path is not the same
    code as an init kwarg: pydantic-settings JSON-decodes a list field's env value before any
    validator runs, so without NoDecode a comma-separated value raised SettingsError at startup."""
    monkeypatch.setenv("GPU_ORCH_CORS_ORIGINS", "https://a.example, https://b.example")
    assert Config().cors_origins == ["https://a.example", "https://b.example"]


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


# --- capacity envelope (plan tiers A + B): parity with the CLI -------------------------


def _deploy(client) -> str:
    return client.post(
        "/deployments", json={"model_id": "qwen3-0.6b", "provider": "mock", "wait": True}
    ).json()["id"]


def test_schedule_set_and_clear(tmp_path):
    client = _client(tmp_path)
    dep_id = _deploy(client)
    resp = client.put(
        f"/deployments/{dep_id}/schedule",
        json={
            "timezone": "America/New_York",
            "default_posture": "off",
            "rules": [{"days": [0, 1, 2, 3, 4], "start": "06:00", "end": "18:00", "posture": "on"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["schedule"]["timezone"] == "America/New_York"
    assert client.get(f"/deployments/{dep_id}").json()["schedule"]["rules"][0]["start"] == "06:00"

    cleared = client.delete(f"/deployments/{dep_id}/schedule")
    assert cleared.status_code == 200
    assert cleared.json()["schedule"] is None


def test_schedule_rejects_bad_timezone(tmp_path):
    client = _client(tmp_path)
    dep_id = _deploy(client)
    resp = client.put(
        f"/deployments/{dep_id}/schedule", json={"timezone": "Mars/Olympus", "rules": []}
    )
    assert resp.status_code == 422  # the DTO is the domain model, so FastAPI rejects it up front


def test_limits_set_and_clear(tmp_path):
    client = _client(tmp_path)
    dep_id = _deploy(client)
    resp = client.put(
        f"/deployments/{dep_id}/limits",
        json={"max_concurrency": 16, "max_queue": 64, "queue_timeout_s": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["max_concurrency"] == 16
    assert resp.json()["max_queue"] == 64

    cleared = client.delete(f"/deployments/{dep_id}/limits")
    assert cleared.json()["max_concurrency"] is None


def test_limits_reject_bad_value_as_400(tmp_path):
    client = _client(tmp_path)
    dep_id = _deploy(client)
    resp = client.put(f"/deployments/{dep_id}/limits", json={"max_concurrency": 0})
    assert resp.status_code == 400  # the domain model's validator, surfaced as a clean error
    assert "error" in resp.json()


def test_scale_replicas(tmp_path):
    client = _client(tmp_path)
    _deploy(client)
    resp = client.post("/scale", json={"model_id": "qwen3-0.6b", "replicas": 3, "wait": True})
    assert resp.status_code == 200
    assert len(resp.json()) == 3
    assert len(client.get("/deployments").json()) == 3

    down = client.post("/scale", json={"model_id": "qwen3-0.6b", "replicas": 1})
    assert len(down.json()) == 1


def test_scale_without_a_member_is_400(tmp_path):
    resp = _client(tmp_path).post("/scale", json={"model_id": "qwen3-0.6b", "replicas": 2})
    assert resp.status_code == 400
    assert "deploy one first" in resp.json()["error"]


def test_budget_crud(tmp_path):
    client = _client(tmp_path)
    created = client.post(
        "/budgets", json={"limit_usd": 500, "window": "monthly", "on_exceed": "stop"}
    )
    assert created.status_code == 200
    budget_id = created.json()["id"]
    assert created.json()["deployment_id"] is None  # account-wide

    listed = client.get("/budgets").json()
    assert listed[0]["budget"]["id"] == budget_id
    assert listed[0]["exceeded"] is False
    assert "spent_usd" in listed[0]  # the status snapshot, not just the record

    assert client.delete(f"/budgets/{budget_id}").status_code == 204
    assert client.get("/budgets").json() == []
    assert client.delete(f"/budgets/{budget_id}").status_code == 404


def test_budget_rejects_non_positive_limit(tmp_path):
    resp = _client(tmp_path).post("/budgets", json={"limit_usd": 0, "window": "daily"})
    assert resp.status_code == 400
    assert "greater than 0" in resp.json()["error"]


def test_autoscale_crud(tmp_path):
    client = _client(tmp_path)
    resp = client.put(
        "/autoscale/qwen3-0.6b",
        json={"max_replicas": 4, "target_rpm_per_replica": 30, "min_replicas": 2},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "schema_version": resp.json()["schema_version"],
        "model_id": "qwen3-0.6b",
        "min_replicas": 2,
        "max_replicas": 4,
        "target_rpm_per_replica": 30.0,
    }
    assert client.get("/autoscale").json()[0]["model_id"] == "qwen3-0.6b"
    assert client.delete("/autoscale/qwen3-0.6b").status_code == 204
    assert client.get("/autoscale").json() == []
    assert client.delete("/autoscale/qwen3-0.6b").status_code == 404


def test_volume_delete(tmp_path):
    client = _client(tmp_path)
    assert client.get("/volumes").json() == []
    assert client.delete("/volumes/vol-1").status_code == 204  # idempotent, like the provider


def test_capacity_routes_are_token_guarded_behind_the_ui(tmp_path):
    """The UI-served build only guards the API prefixes, so a new prefix that is not listed would be
    reachable with no token. Every capacity route is under a guarded prefix."""
    ui = tmp_path / "web"
    ui.mkdir()
    (ui / "index.html").write_text("ok")
    cfg = Config(
        namespace="test", state_db=tmp_path / "api.db", reconcile_interval=0, api_token="secret"
    )
    orch = Orchestrator(cfg, provider=MockProvider(namespace="test"), runtime=_runtime())
    client = TestClient(create_app(orch, ui_dir=ui))
    for path in ("/budgets", "/autoscale", "/scale"):
        assert client.get(path).status_code == 401, path
