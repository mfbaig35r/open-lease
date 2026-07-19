"""CLI rendering: rich tables and JSON, nothing else (spec §15, E3).

Every function here takes already-fetched domain objects and prints them. No orchestration, no I/O
beyond stdout. ``--json`` on read commands routes through ``emit_json`` so output stays scriptable.
Colours map to state so ``gpu status`` reads at a glance.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..models import (
    CostRecord,
    Deployment,
    DeploymentState,
    Event,
    GpuAvailability,
    HealthStatus,
    ModelSpec,
    ProviderInfo,
    UsageSummary,
    VolumeInfo,
)

console = Console()
_err = Console(stderr=True)

_STATE_COLOR = {
    DeploymentState.READY: "green",
    DeploymentState.DEGRADED: "yellow",
    DeploymentState.FAILED: "red",
    DeploymentState.STOPPED: "dim",
    DeploymentState.STOPPING: "dim",
}


def error(message: str, hint: str | None = None) -> None:
    # escape: the message is data, not markup, so brackets (open-lease[api]) render literally.
    _err.print(f"[red]Error:[/red] {escape(message)}")
    if hint:
        _err.print(f"[dim]hint:[/dim] {escape(hint)}")


def warn(message: str, hint: str | None = None) -> None:
    _err.print(f"[yellow]warning:[/yellow] {escape(message)}")
    if hint:
        _err.print(f"[dim]hint:[/dim] {escape(hint)}")


def emit_json(payload: object) -> None:
    console.print_json(json.dumps(payload, default=_json_default))


def _json_default(obj: object) -> object:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _state(state: DeploymentState) -> str:
    return f"[{_STATE_COLOR.get(state, 'white')}]{state.value}[/]"


def _uptime(created_at: datetime) -> str:
    return _duration((datetime.now(UTC) - created_at).total_seconds())


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60}m"


_WAITING = {
    DeploymentState.PROVISIONING,
    DeploymentState.BOOTING,
    DeploymentState.DOWNLOADING,
    DeploymentState.STARTING,
}
_SERVING_BOUND = {DeploymentState.BOOTING, DeploymentState.DOWNLOADING, DeploymentState.STARTING}


def _entered_at(deployment: Deployment, state: DeploymentState) -> datetime | None:
    for transition in reversed(deployment.state_history):
        if transition.to_state == state:
            return transition.at
    return None


def progress_hint(deployment: Deployment) -> str | None:
    """A short progress annotation for a deployment still coming up: a real percent when the runtime
    exposed download progress, else elapsed-in-stage against the model's startup budget (e.g.
    ``12m/40m``) so the user can see it is making progress, not stuck. None once serving."""
    state = deployment.observed_state
    if state not in _WAITING:
        return None
    if deployment.download_progress is not None:
        return f"{int(deployment.download_progress * 100)}%"
    entered = _entered_at(deployment, state)
    if entered is None:
        return None
    elapsed = (datetime.now(UTC) - entered).total_seconds()
    if state in _SERVING_BOUND:
        budget = deployment.profile.validation.startup_timeout_seconds
        return f"{_duration(elapsed)}/{_duration(budget)}"
    return _duration(elapsed)


def deployments_table(deployments: list[Deployment], accrued: dict[str, float]) -> None:
    if not deployments:
        console.print("[dim]No deployments. Try `gpu deploy <model>`.[/dim]")
        return
    table = Table(title="Deployments")
    for col in ("ID", "MODEL", "STATE", "GPU", "ENDPOINT", "UPTIME", "ACCRUED $"):
        table.add_column(col)
    for dep in deployments:
        gpu = "-"
        if dep.instance:
            gpu = dep.instance.gpu_type
            if dep.instance.gpu_count > 1:
                gpu = f"{dep.instance.gpu_count}x {gpu}"
        hint = progress_hint(dep)
        state_cell = _state(dep.observed_state) + (f" [dim]{hint}[/]" if hint else "")
        table.add_row(
            dep.id,
            dep.model_id,
            state_cell,
            gpu,
            dep.endpoint_url or "-",
            _uptime(dep.created_at),
            f"{accrued.get(dep.id, 0.0):.4f}",
        )
    console.print(table)


def health_table(deployment_id: str, status: HealthStatus) -> None:
    table = Table(title=f"Health: {deployment_id} ({status.status.value})")
    table.add_column("CHECK")
    table.add_column("OK")
    table.add_column("DETAIL")
    for name, check in status.checks.items():
        mark = "[green]yes[/]" if check.ok else "[red]no[/]"
        latency = f" ({check.latency_ms:.0f}ms)" if check.latency_ms is not None else ""
        table.add_row(name, mark, f"{check.detail}{latency}")
    console.print(table)


def models_table(models: list[ModelSpec]) -> None:
    table = Table(title="Model catalog")
    for col in ("ID", "FAMILY", "PARAMS", "CONTEXT", "MIN GPU GB", "CAPABILITIES", "LICENSE"):
        table.add_column(col)
    for spec in models:
        caps = ", ".join(
            name
            for name, on in (
                ("chat", spec.chat),
                ("completion", spec.completion),
                ("embedding", spec.embedding),
                ("vision", spec.vision),
                ("tools", spec.supports_tools),
                ("reasoning", spec.supports_reasoning),
            )
            if on
        )
        table.add_row(
            spec.id,
            spec.family,
            spec.parameter_count,
            str(spec.context_window),
            str(spec.min_gpu_memory_gb),
            caps,
            spec.license,
        )
    console.print(table)


def providers_table(providers: list[ProviderInfo]) -> None:
    table = Table(title="Providers")
    for col in ("NAME", "GPUS", "VOLUMES", "REGIONS"):
        table.add_column(col)
    for info in providers:
        gpus = ", ".join(g.id for g in info.capabilities.gpu_types) or "-"
        table.add_row(
            info.name,
            gpus,
            "yes" if info.capabilities.supports_volumes else "no",
            ", ".join(info.capabilities.regions) or "-",
        )
    console.print(table)


def costs_table(records: list[CostRecord]) -> None:
    if not records:
        console.print("[dim]No cost records yet.[/dim]")
        return
    table = Table(title="Costs")
    for col in ("DEPLOYMENT", "$/HR", "ACCRUED $", "PROJECTED $/MO", "OPEN"):
        table.add_column(col)
    total = 0.0
    for record in records:
        total += record.accrued_usd
        table.add_row(
            record.deployment_id,
            f"{record.gpu_hourly_usd:.4f}",
            f"{record.accrued_usd:.4f}",
            f"{record.estimated_monthly_usd:.2f}",
            "yes" if record.stopped_at is None else "no",
        )
    table.add_section()
    table.add_row("[b]TOTAL[/b]", "", f"[b]{total:.4f}[/b]", "", "")
    console.print(table)


def usage_table(summaries: list[UsageSummary]) -> None:
    active = [s for s in summaries if s.requests > 0]
    if not active:
        console.print("[dim]No token usage yet (drive some requests through the proxy).[/dim]")
        return
    table = Table(title="Token usage + cost per model")
    for col in ("DEPLOYMENT", "MODEL", "REQS", "TOKENS", "TOK/S", "ACCRUED $", "$/M TOK"):
        table.add_column(col)
    for s in active:
        table.add_row(
            s.deployment_id,
            s.model_id,
            str(s.requests),
            f"{s.total_tokens:,}",
            f"{s.tokens_per_sec:.1f}",
            f"{s.accrued_usd:.4f}",
            "-" if s.cost_per_mtok is None else f"{s.cost_per_mtok:.2f}",
        )
    console.print(table)


def events_table(events: list[Event]) -> None:
    if not events:
        console.print("[dim]No events.[/dim]")
        return
    table = Table(title="Timeline")
    table.add_column("TIME")
    table.add_column("KIND")
    table.add_column("PAYLOAD")
    for event in events:
        table.add_row(event.at.isoformat(), event.kind.value, json.dumps(event.payload))
    console.print(table)


def volumes_table(volumes: list[VolumeInfo]) -> None:
    if not volumes:
        console.print(
            "[dim]No network volumes. Enable caching with cache_volume_enabled=true.[/dim]"
        )
        return
    table = Table(title="Network volumes (model cache)")
    for col in ("ID", "NAME", "SIZE GB", "DATA CENTER", "EST $/MO"):
        table.add_column(col)
    for v in volumes:
        table.add_row(
            v.id, v.name, str(v.size_gb), v.data_center_id or "-", f"{v.estimated_monthly_usd:.2f}"
        )
    console.print(table)


def availability_table(rows: list[GpuAvailability]) -> None:
    if not rows:
        console.print("[dim]No availability data (unsupported provider or unknown GPU).[/dim]")
        return
    table = Table(title="GPU availability by data center")
    for col in ("DATA CENTER", "GPU", "AVAILABLE", "STOCK"):
        table.add_column(col)
    # available first, then by stock
    order = {"High": 0, "Medium": 1, "Low": 2}
    for row in sorted(rows, key=lambda r: (not r.available, order.get(r.stock_status or "", 9))):
        mark = "[green]yes[/]" if row.available else "[dim]no[/]"
        table.add_row(row.data_center_id, row.gpu_type_id, mark, row.stock_status or "-")
    console.print(table)


def config_table(effective: dict[str, object]) -> None:
    table = Table(title="Effective config (secrets masked)")
    table.add_column("KEY")
    table.add_column("VALUE")
    for key, value in effective.items():
        table.add_row(key, str(value))
    console.print(table)
