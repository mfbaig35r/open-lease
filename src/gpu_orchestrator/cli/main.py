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
from . import render

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


def _orchestrator() -> Orchestrator:
    # Seam for tests: monkeypatch this to inject a mock-backed Orchestrator.
    return Orchestrator(Config())


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
    set_: list[str] | None = typer.Option(None, "--set", help="Override key=value (repeatable)."),
) -> None:
    """Deploy a model. Returns immediately unless --wait."""
    orch = _orchestrator()
    dep = _run(
        orch.deploy_model(model, provider=provider, gpu=gpu, wait=wait, overrides=_overrides(set_))
    )
    render.console.print(f"Deployment [b]{dep.id}[/b] -> {dep.observed_state.value}")


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
def config(json_: bool = typer.Option(False, "--json")) -> None:
    """Show effective config with secrets masked."""
    effective = Config().effective()
    if json_:
        render.emit_json(effective)
        return
    render.config_table(effective)


# --- daemon (loop ownership) ----------------------------------------------------------


@app.command()
def daemon() -> None:
    """Run the reconcile, health, and orphan-sweep loops in the foreground."""
    from ..core.daemon import Daemon

    render.console.print(
        "[b]gpu daemon[/b] running (reconcile / health / orphan sweep). Ctrl-C to stop."
    )
    try:
        asyncio.run(Daemon(Config()).run())
    except KeyboardInterrupt:
        render.console.print("Daemon stopped.")


# --- inference path (arrives with the proxy, step 8) ----------------------------------


@app.command()
def proxy(port: int = typer.Option(8080, "--port")) -> None:
    """Start the OpenAI-compatible proxy (step 8)."""
    render.error("`gpu proxy` arrives in step 8 (the OpenAI proxy).")
    raise typer.Exit(1)


@app.command()
def chat(deployment_id: str) -> None:
    """Minimal REPL against a READY deployment (step 8)."""
    render.error("`gpu chat` arrives in step 8 (the OpenAI proxy).")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
