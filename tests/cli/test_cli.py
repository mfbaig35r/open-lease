"""Step-7 CLI snapshot/behaviour tests (spec §15, §18): the Typer app driven against a mock-backed
Orchestrator. The mock now offers the catalog's GPUs, so the real catalog + full deploy flow run
offline. The daemon path is not exercised here; ``--wait`` drives inline."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from gpu_orchestrator.cli import main as cli_main
from gpu_orchestrator.cli.main import app
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


@pytest.fixture
def cli(tmp_path, monkeypatch):
    cfg = Config(
        namespace="test",
        state_db=tmp_path / "cli.db",
        reconcile_interval=0,
        daemon_pid_file=tmp_path / "daemon.pid",
        proxy_pid_file=tmp_path / "proxy.pid",
        daemon_log_file=tmp_path / "daemon.log",
        proxy_log_file=tmp_path / "proxy.log",
    )
    orch = Orchestrator(cfg, provider=MockProvider(namespace="test"), runtime=_runtime())
    monkeypatch.setattr(cli_main, "_orchestrator", lambda: orch)
    monkeypatch.setattr(cli_main, "_config", lambda: cfg)
    return CliRunner(), orch


def test_deploy_wait_reaches_ready(cli):
    runner, _ = cli
    result = runner.invoke(app, ["deploy", "qwen3-0.6b", "--provider", "mock", "--wait"])
    assert result.exit_code == 0
    assert "ready" in result.output


def test_status_lists_deployment(cli):
    runner, orch = cli
    runner.invoke(app, ["deploy", "qwen3-0.6b", "--provider", "mock", "--wait"])
    dep_id = orch.list_deployments()[0].id
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert dep_id in result.output
    assert "ready" in result.output


def test_status_json_is_parseable(cli):
    runner, _ = cli
    runner.invoke(app, ["deploy", "qwen3-0.6b", "--provider", "mock", "--wait"])
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["model_id"] == "qwen3-0.6b"


def test_models_lists_catalog(cli):
    runner, _ = cli
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "qwen3-0.6b" in result.output


def test_estimate(cli):
    runner, _ = cli
    result = runner.invoke(app, ["estimate", "qwen3-0.6b", "--provider", "mock"])
    assert result.exit_code == 0
    assert "$" in result.output


def test_deploy_then_costs_and_health(cli):
    runner, orch = cli
    runner.invoke(app, ["deploy", "qwen3-0.6b", "--provider", "mock", "--wait"])
    dep_id = orch.list_deployments()[0].id

    costs = runner.invoke(app, ["costs"])
    assert costs.exit_code == 0
    assert dep_id in costs.output

    health = runner.invoke(app, ["health", dep_id])
    assert health.exit_code == 0
    assert "instance_alive" in health.output


def test_stop(cli):
    runner, orch = cli
    runner.invoke(app, ["deploy", "qwen3-0.6b", "--provider", "mock", "--wait"])
    dep_id = orch.list_deployments()[0].id
    result = runner.invoke(app, ["stop", dep_id])
    assert result.exit_code == 0
    assert "stopped" in result.output


def test_availability_lists_data_centers(cli):
    runner, _ = cli
    result = runner.invoke(app, ["availability", "qwen3-0.6b"])
    assert result.exit_code == 0
    assert "MOCK-DC-1" in result.output


def test_volumes_lists_empty(cli):
    runner, _ = cli
    result = runner.invoke(app, ["volumes"])
    assert result.exit_code == 0
    assert "No network volumes" in result.output


def test_config_and_providers(cli):
    runner, _ = cli
    assert runner.invoke(app, ["config"]).exit_code == 0
    providers = runner.invoke(app, ["providers"])
    assert providers.exit_code == 0
    assert "mock" in providers.output


def test_unknown_model_exits_1(cli):
    runner, _ = cli
    result = runner.invoke(app, ["deploy", "no-such-model", "--provider", "mock"])
    assert result.exit_code == 1
    assert "Error" in result.output


def test_deploy_warns_when_no_daemon(cli):
    # A non-blocking deploy with no daemon must warn loudly, not silently stall.
    runner, _ = cli
    result = runner.invoke(app, ["deploy", "qwen3-0.6b", "--provider", "mock"])
    assert result.exit_code == 0
    assert "no daemon running" in result.output


def test_daemon_status_when_not_running(cli):
    runner, _ = cli
    result = runner.invoke(app, ["daemon", "--status"])
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_daemon_stop_when_not_running(cli):
    runner, _ = cli
    result = runner.invoke(app, ["daemon", "--stop"])
    assert result.exit_code == 0
    assert "No daemon running" in result.output


def test_deploy_chat_reaches_repl(cli):
    # --chat deploys, waits for READY, then opens the REPL; "exit" quits cleanly (no network hit).
    runner, _ = cli
    result = runner.invoke(
        app, ["deploy", "qwen3-0.6b", "--provider", "mock", "--chat"], input="exit\n"
    )
    assert result.exit_code == 0
    assert "ready" in result.output
    assert "Chatting with qwen3-0.6b" in result.output


def test_chat_rejects_non_ready_deployment(cli):
    runner, orch = cli
    # A deployment that was never driven to READY: chat should refuse, not hang on a dead endpoint.
    runner.invoke(app, ["deploy", "qwen3-0.6b", "--provider", "mock"])  # non-blocking -> REQUESTED
    dep_id = orch.list_deployments()[0].id
    result = runner.invoke(app, ["chat", dep_id])
    assert result.exit_code == 1
    assert "not READY" in result.output
