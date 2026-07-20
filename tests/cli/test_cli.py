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


def test_deploy_hf_repo_requires_gpu(cli):
    runner, _ = cli
    result = runner.invoke(app, ["deploy", "--hf-repo", "Qwen/Qwen3-14B", "--provider", "mock"])
    assert result.exit_code != 0
    assert "--gpu" in result.output


def test_deploy_adhoc_hf_repo_reaches_ready(cli):
    runner, orch = cli
    result = runner.invoke(
        app,
        [
            "deploy",
            "--hf-repo",
            "Qwen/Qwen3-14B",
            "--gpu",
            "MOCK-GPU",
            "--provider",
            "mock",
            "--wait",
        ],
    )
    assert result.exit_code == 0
    assert "ready" in result.output
    dep = orch.list_deployments()[0]
    assert dep.hf_repo == "Qwen/Qwen3-14B"
    assert dep.model_id == "qwen3-14b"  # no catalog entry needed


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


def _deploy_id(runner, orch):
    result = runner.invoke(app, ["deploy", "qwen3-0.6b", "--provider", "mock", "--wait"])
    assert result.exit_code == 0
    return orch.list_deployments(include_stopped=True)[0].id


def test_schedule_set_shows_windows(cli):
    runner, orch = cli
    dep_id = _deploy_id(runner, orch)
    result = runner.invoke(
        app, ["schedule", dep_id, "--on", "mon-fri 06:00-18:00", "--tz", "America/New_York"]
    )
    assert result.exit_code == 0
    assert "America/New_York" in result.output
    assert "mon,tue,wed,thu,fri" in result.output
    assert orch.get_deployment(dep_id).schedule is not None


def test_schedule_clear_returns_to_manual(cli):
    runner, orch = cli
    dep_id = _deploy_id(runner, orch)
    runner.invoke(app, ["schedule", dep_id, "--on", "all 00:00-23:59"])
    result = runner.invoke(app, ["schedule", dep_id, "--clear"])
    assert result.exit_code == 0
    assert orch.get_deployment(dep_id).schedule is None


def test_schedule_bad_window_fails_cleanly(cli):
    runner, orch = cli
    dep_id = _deploy_id(runner, orch)
    result = runner.invoke(app, ["schedule", dep_id, "--on", "funday 06:00-18:00"])
    assert result.exit_code == 1
    assert orch.get_deployment(dep_id).schedule is None  # nothing persisted on bad input


def test_budget_set_and_list(cli):
    runner, orch = cli
    result = runner.invoke(
        app, ["budget", "set", "--limit", "100", "--window", "monthly", "--on-exceed", "warn"]
    )
    assert result.exit_code == 0
    assert len(orch.list_budgets()) == 1
    listed = runner.invoke(app, ["budget", "list"])
    assert listed.exit_code == 0
    assert "Spend ceilings" in listed.output


def test_budget_bad_window_fails_cleanly(cli):
    runner, orch = cli
    result = runner.invoke(app, ["budget", "set", "--limit", "100", "--window", "weekly"])
    assert result.exit_code == 1
    assert orch.list_budgets() == []  # nothing persisted


def test_budget_rm(cli):
    runner, orch = cli
    runner.invoke(app, ["budget", "set", "--limit", "50", "--window", "daily"])
    budget_id = orch.list_budgets()[0].id
    result = runner.invoke(app, ["budget", "rm", budget_id])
    assert result.exit_code == 0
    assert orch.list_budgets() == []


def test_limits_set_show_and_clear(cli):
    runner, orch = cli
    dep_id = _deploy_id(runner, orch)
    setr = runner.invoke(app, ["limits", dep_id, "--max", "8", "--queue", "32"])
    assert setr.exit_code == 0
    assert orch.get_deployment(dep_id).max_concurrency == 8
    shown = runner.invoke(app, ["limits", dep_id])
    assert shown.exit_code == 0 and "max=" in shown.output
    clr = runner.invoke(app, ["limits", dep_id, "--clear"])
    assert clr.exit_code == 0
    assert orch.get_deployment(dep_id).max_concurrency is None


def test_limits_bad_value_fails_cleanly(cli):
    runner, orch = cli
    dep_id = _deploy_id(runner, orch)
    result = runner.invoke(app, ["limits", dep_id, "--max", "0"])
    assert result.exit_code == 1
    assert orch.get_deployment(dep_id).max_concurrency is None  # nothing persisted


def test_deploy_replicas_creates_a_pool(cli):
    runner, orch = cli
    result = runner.invoke(
        app, ["deploy", "qwen3-0.6b", "--provider", "mock", "--wait", "--replicas", "2"]
    )
    assert result.exit_code == 0
    pool = [d for d in orch.list_deployments() if d.model_id == "qwen3-0.6b"]
    assert len(pool) == 2


def test_scale_command_resizes_the_pool(cli):
    runner, orch = cli
    _deploy_id(runner, orch)  # one replica, READY
    up = runner.invoke(app, ["scale", "qwen3-0.6b", "3"])
    assert up.exit_code == 0
    assert len([d for d in orch.list_deployments() if d.model_id == "qwen3-0.6b"]) == 3
