"""The ``gpu`` CLI (spec §15). Typer + rich, feels like docker/kubectl/uv.

Every command body is: parse args -> call one Orchestrator method -> render (E3). No orchestration
logic lives here. Async Orchestrator methods cross the sync CLI boundary via ``asyncio.run`` per
command (CLAUDE.md). Errors print one human sentence (+ hint) and exit 1; ``--debug`` shows the
traceback. Read commands take ``--json`` for scripting.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TypeVar

import typer

from ..config import Config
from ..core.orchestrator import Orchestrator
from ..errors import OrchestratorError
from ..logging import configure_logging
from ..models import RuntimeOverrides
from . import process, render

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Deploy and run open LLMs on GPUs."
)

_T = TypeVar("_T")
_state = {"debug": False}


@app.callback()
def _main(debug: bool = typer.Option(False, "--debug", help="Show tracebacks on error.")) -> None:
    _state["debug"] = debug
    configure_logging("DEBUG" if debug else "WARNING")


# --- error boundary + async bridge (the only shared plumbing) -------------------------


def _fail(exc: OrchestratorError) -> None:
    if _state["debug"]:
        raise exc
    render.error(str(exc))
    raise typer.Exit(1)


def _fail_msg(message: str) -> None:
    render.error(message)
    raise typer.Exit(1)


def _run(coro: Awaitable[_T]) -> _T:
    try:
        return asyncio.run(coro)  # type: ignore[arg-type]
    except OrchestratorError as exc:
        _fail(exc)
        raise  # unreachable; _fail always raises


def _call(fn: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    try:
        return fn(*args, **kwargs)
    except OrchestratorError as exc:
        _fail(exc)
        raise  # unreachable


def _config() -> Config:
    # Seam for tests: monkeypatch to point pidfiles/state at a tmp dir.
    return Config()


def _orchestrator() -> Orchestrator:
    # Seam for tests: monkeypatch this to inject a mock-backed Orchestrator.
    return Orchestrator(_config())


def _preflight_capacity(orch: Orchestrator, model: str, provider: str) -> None:
    """Best-effort: warn if no data center currently has capacity for the model's GPU, so the user
    is not left wondering why a deploy retries and fails. Never blocks the deploy."""
    try:
        rows = asyncio.run(orch.gpu_availability(model_id=model, provider=provider))
    except OrchestratorError:
        return  # unknown model / unsupported provider / probe failed: let deploy handle it
    if rows and not any(r.available for r in rows):
        render.warn(
            f"no data center currently has capacity for {model}'s GPU; the deploy may wait or fail",
            hint="check `gpu availability " + model + "`",
        )


def _overrides(sets: list[str] | None) -> RuntimeOverrides | None:
    if not sets:
        return None
    launch_args: dict[str, str] = {}
    env: dict[str, str] = {}
    for item in sets:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        (launch_args if key.startswith("--") else env)[key] = value
    return RuntimeOverrides(launch_args=launch_args, env=env)


# --- lifecycle commands ---------------------------------------------------------------


@app.command()
def deploy(
    model: str,
    provider: str = typer.Option("runpod", "--provider"),
    gpu: str | None = typer.Option(None, "--gpu", help="Override the profile's recommended GPU."),
    wait: bool = typer.Option(False, "--wait", help="Block until READY or FAILED."),
    chat: bool = typer.Option(False, "--chat", help="Wait for READY, then open a chat REPL."),
    set_: list[str] | None = typer.Option(None, "--set", help="Override key=value (repeatable)."),
    auto_daemon: bool = typer.Option(False, "--auto-daemon", help="Start a daemon if none is up."),
) -> None:
    """Deploy a model. Returns immediately unless --wait (or --chat, which implies it)."""
    orch = _orchestrator()
    _preflight_capacity(orch, model, provider)
    wait = wait or chat  # can't chat until it is READY
    dep = _run(
        orch.deploy_model(model, provider=provider, gpu=gpu, wait=wait, overrides=_overrides(set_))
    )
    render.console.print(f"Deployment [b]{dep.id}[/b] -> {dep.observed_state.value}")
    if chat:
        if dep.observed_state.value != "ready":
            _fail_msg(f"{dep.id} did not reach READY (state: {dep.observed_state.value}).")
        _run_chat(orch, dep)
        return
    # A non-blocking deploy only progresses if a daemon reconciles it. Never let it stall quietly.
    if not wait:
        cfg = orch.config
        if process.running_pid(cfg.daemon_pid_file) is None:
            if auto_daemon or cfg.auto_daemon:
                pid = process.spawn_detached(["daemon"], cfg.daemon_log_file)
                render.console.print(f"Started daemon (pid {pid}) to drive this deployment.")
            else:
                render.warn(
                    f"no daemon running, so {dep.id} will not progress",
                    hint="start one with `gpu daemon --detach` / `gpu up`, or deploy with --wait",
                )


@app.command()
def stop(deployment_id: str) -> None:
    """Stop a deployment (keeps the record)."""
    dep = _run(_orchestrator().stop_deployment(deployment_id))
    render.console.print(f"Stopped [b]{dep.id}[/b] -> {dep.observed_state.value}")


@app.command()
def delete(deployment_id: str, yes: bool = typer.Option(False, "--yes")) -> None:
    """Stop a deployment and remove its record."""
    if not yes:
        typer.confirm(f"Delete {deployment_id}? Its instance will be destroyed.", abort=True)
    _run(_orchestrator().delete_deployment(deployment_id))
    render.console.print(f"Deleted [b]{deployment_id}[/b]")


@app.command()
def restart(deployment_id: str) -> None:
    """Stop and redeploy the same profile (a full cold start)."""
    dep = _run(_orchestrator().restart_deployment(deployment_id))
    render.console.print(f"Restarted [b]{dep.id}[/b] -> {dep.observed_state.value}")


# --- read commands --------------------------------------------------------------------


@app.command()
def status(
    deployment_id: str | None = typer.Argument(None),
    all_: bool = typer.Option(False, "--all", help="Include stopped deployments."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show one deployment or all of them."""
    orch = _orchestrator()
    if deployment_id:
        deployments = [_call(orch.get_deployment, deployment_id)]
    else:
        deployments = _call(orch.list_deployments, include_stopped=all_)
    if json_:
        render.emit_json(deployments)
        return
    accrued = {
        d.id: round(sum(r.accrued_usd for r in _call(orch.get_costs, d.id)), 4) for d in deployments
    }
    render.deployments_table(deployments, accrued)


@app.command()
def health(deployment_id: str, json_: bool = typer.Option(False, "--json")) -> None:
    """Check-by-check health of a deployment."""
    status_ = _run(_orchestrator().get_health(deployment_id))
    if json_:
        render.emit_json(status_)
        return
    render.health_table(deployment_id, status_)


@app.command()
def logs(
    deployment_id: str,
    tail: int = typer.Option(100, "--tail"),
    follow: bool = typer.Option(False, "--follow"),
) -> None:
    """Print provider/runtime logs for a deployment."""
    for line in _run(_orchestrator().get_logs(deployment_id, tail=tail, follow=follow)):
        render.console.print(line)


@app.command()
def models(
    family: str | None = typer.Option(None, "--family"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List the model catalog."""
    specs = _call(_orchestrator().list_models)
    if family:
        specs = [s for s in specs if s.family == family]
    if json_:
        render.emit_json(specs)
        return
    render.models_table(specs)


@app.command()
def providers(json_: bool = typer.Option(False, "--json")) -> None:
    """List configured providers and their capabilities."""
    infos = _run(_orchestrator().list_providers())
    if json_:
        render.emit_json(infos)
        return
    render.providers_table(infos)


@app.command()
def costs(
    deployment_id: str | None = typer.Argument(None), json_: bool = typer.Option(False, "--json")
) -> None:
    """Accrued and projected cost per deployment."""
    records = _call(_orchestrator().get_costs, deployment_id)
    if json_:
        render.emit_json(records)
        return
    render.costs_table(records)


@app.command()
def events(
    deployment_id: str,
    since: str | None = typer.Option(None, "--since", help="ISO timestamp."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show a deployment's timeline."""
    since_dt = datetime.fromisoformat(since) if since else None
    evs = _call(_orchestrator().events, deployment_id, since=since_dt)
    if json_:
        render.emit_json(evs)
        return
    render.events_table(evs)


@app.command()
def estimate(
    model: str,
    provider: str = typer.Option("runpod", "--provider"),
    hours: float = typer.Option(1.0, "--hours"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Estimate cost without deploying."""
    est = _run(_orchestrator().estimate_cost(model, provider=provider, hours=hours))
    if json_:
        render.emit_json(est)
        return
    render.console.print(
        f"{est.model_id} on {est.provider} ({est.gpu_type}): "
        f"${est.estimated_usd:.4f} for {est.hours}h (${est.gpu_hourly_usd:.4f}/hr)"
    )


@app.command()
def availability(
    model: str | None = typer.Argument(None, help="Filter to a model's recommended GPU."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Per-data-center GPU availability (which data centers can run a model right now)."""
    rows = _run(_orchestrator().gpu_availability(model_id=model))
    if json_:
        render.emit_json(rows)
        return
    render.availability_table(rows)


@app.command()
def volumes(
    delete: str | None = typer.Option(None, "--delete", help="Delete a volume by id."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List (or delete) persistent model-cache network volumes."""
    orch = _orchestrator()
    if delete:
        _run(orch.delete_volume(delete))
        render.console.print(f"Deleted volume [b]{delete}[/b]")
        return
    vols = _run(orch.list_volumes())
    if json_:
        render.emit_json(vols)
        return
    render.volumes_table(vols)


@app.command()
def config(json_: bool = typer.Option(False, "--json")) -> None:
    """Show effective config with secrets masked."""
    effective = Config().effective()
    if json_:
        render.emit_json(effective)
        return
    render.config_table(effective)


# --- daemon (loop ownership) ----------------------------------------------------------


@app.command()
def daemon(
    detach: bool = typer.Option(False, "--detach", help="Run in the background."),
    stop: bool = typer.Option(False, "--stop", help="Stop a running daemon."),
    status: bool = typer.Option(False, "--status", help="Show whether a daemon is running."),
) -> None:
    """Run (or manage) the reconcile, health, and orphan-sweep loops. Foreground by default."""
    from ..core.daemon import Daemon

    cfg = _config()
    if stop:
        pid = process.stop(cfg.daemon_pid_file)
        render.console.print(f"Daemon stopped (pid {pid})." if pid else "No daemon running.")
        return
    if status:
        pid = process.running_pid(cfg.daemon_pid_file)
        render.console.print(
            f"Daemon running (pid {pid}). Logs: {cfg.daemon_log_file}"
            if pid
            else "Daemon not running."
        )
        return

    existing = process.running_pid(cfg.daemon_pid_file)
    if existing is not None:
        _fail_msg(f"daemon already running (pid {existing})")
    if detach:
        pid = process.spawn_detached(["daemon"], cfg.daemon_log_file)
        render.console.print(f"Daemon started (pid [b]{pid}[/b]). Logs: {cfg.daemon_log_file}")
        return

    process.write_pid(cfg.daemon_pid_file)
    render.console.print("gpu daemon running (reconcile / health / orphan sweep). Ctrl-C to stop.")
    try:
        asyncio.run(Daemon(cfg).run())
    except KeyboardInterrupt:
        render.console.print("Daemon stopped.")
    finally:
        process.clear_pid(cfg.daemon_pid_file)


# --- inference path (proxy) -----------------------------------------------------------


@app.command()
def proxy(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Start the OpenAI-compatible proxy: /v1/chat/completions, /v1/completions, /v1/models,
    /v1/embeddings, routed by model name to READY deployments."""
    import uvicorn

    from ..proxy.openai_proxy import create_proxy_app

    cfg = _config()
    existing = process.running_pid(cfg.proxy_pid_file)
    if existing is not None:
        _fail_msg(f"proxy already running (pid {existing})")
    host = host or cfg.proxy_host
    port = port or cfg.proxy_port
    render.console.print(
        f"OpenAI proxy on [b]http://{host}:{port}[/b] -> READY deployments. Ctrl-C to stop."
    )
    process.write_pid(cfg.proxy_pid_file)
    try:
        uvicorn.run(create_proxy_app(_orchestrator()), host=host, port=port, log_level="warning")
    finally:
        process.clear_pid(cfg.proxy_pid_file)


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Start the REST API (management routes + the OpenAI proxy at /v1/*)."""
    import uvicorn

    from ..api import create_app

    cfg = _config()
    host = host or cfg.api_host
    port = port or cfg.api_port
    authed = "token required" if cfg.api_token is not None else "OPEN (no api_token set)"
    render.console.print(
        f"open-lease API on [b]http://{host}:{port}[/b] ({authed}). Docs at /docs. Ctrl-C to stop."
    )
    uvicorn.run(create_app(_orchestrator()), host=host, port=port, log_level="warning")


@app.command()
def mcp() -> None:
    """Run the MCP server (agent-facing tools over the Orchestrator) on stdio."""
    from ..mcp.server import create_server

    create_server(_orchestrator()).run()


@app.command()
def chat(deployment_id: str) -> None:
    """Minimal REPL against a READY deployment (a thin httpx loop, not through the proxy)."""
    orch = _orchestrator()
    dep = _call(orch.get_deployment, deployment_id)
    if dep.observed_state.value != "ready" or not dep.endpoint_url:
        _fail_msg(f"{deployment_id} is not READY (state: {dep.observed_state.value}).")
    _run_chat(orch, dep)


def _run_chat(orch: Orchestrator, dep) -> None:
    """The chat REPL, shared by `gpu chat` and `gpu deploy --chat`. Hits the deployment endpoint
    directly with the served (HF repo) model id."""
    import httpx

    served = {m.id: m.hf_repo for m in orch.list_models()}
    model = served.get(dep.model_id, dep.model_id)
    render.console.print(f"Chatting with [b]{dep.model_id}[/b]. Type 'exit' or Ctrl-C to quit.")

    messages: list[dict] = []
    while True:
        try:
            content = typer.prompt("you")
        except (KeyboardInterrupt, EOFError, typer.Abort):
            break
        if content.strip().lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": content})
        try:
            resp = httpx.post(
                f"{dep.endpoint_url}/v1/chat/completions",
                json={"model": model, "messages": messages},
                timeout=120,
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            render.error(f"request failed: {exc}")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply.get("content") or ""})
        render.console.print(f"[green]{dep.model_id}[/green]: {reply.get('content') or ''}")


# --- combined lifecycle ---------------------------------------------------------------


@app.command()
def up(port: int | None = typer.Option(None, "--port")) -> None:
    """Start the daemon and the proxy in the background (idempotent)."""
    cfg = _config()
    port = port or cfg.proxy_port
    started: list[str] = []
    if process.running_pid(cfg.daemon_pid_file) is None:
        started.append(f"daemon (pid {process.spawn_detached(['daemon'], cfg.daemon_log_file)})")
    if process.running_pid(cfg.proxy_pid_file) is None:
        pid = process.spawn_detached(["proxy", "--port", str(port)], cfg.proxy_log_file)
        started.append(f"proxy (pid {pid}) on :{port}")
    render.console.print("Started " + ", ".join(started) if started else "Already up.")


@app.command()
def down() -> None:
    """Stop the background daemon and proxy."""
    cfg = _config()
    d = process.stop(cfg.daemon_pid_file)
    p = process.stop(cfg.proxy_pid_file)
    render.console.print(
        f"daemon: {'stopped ' + str(d) if d else 'not running'}; "
        f"proxy: {'stopped ' + str(p) if p else 'not running'}"
    )


if __name__ == "__main__":
    app()
