"""REST API (spec §16, Phase 2). A thin FastAPI layer over the `Orchestrator`.

Every route is: take the request, call one Orchestrator method, return a §6 domain model (FastAPI
serializes it). No parallel schemas, no business logic here. The OpenAI proxy is mounted so `/v1/*`
serves inference alongside the management API. Auth is a single static bearer token from config
(multi-tenancy deferred); when no token is set the API is open, so bind to localhost.

The capacity envelope (plan tiers A + B: schedules, concurrency limits, budgets, replicas,
autoscaling) is here too, so an HTTP client can set a ceiling and not just spend. Interface parity
with the CLI is the rule: anything the Orchestrator exposes, every interface can reach.

Like the CLI and the proxy, this is an interface: the same core, a different shape.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from ..core.budgets import BudgetStatus
from ..core.orchestrator import Orchestrator
from ..errors import (
    DeploymentNotFoundError,
    ModelNotFoundError,
    OrchestratorError,
    PolicyNotFoundError,
)
from ..models import (
    AutoscalePolicy,
    Budget,
    BudgetAction,
    BudgetWindow,
    CostEstimate,
    CostRecord,
    Deployment,
    Event,
    GpuAvailability,
    HealthStatus,
    ModelSpec,
    ProviderInfo,
    RuntimeOverrides,
    Schedule,
    UsageSummary,
    VolumeInfo,
)


class DeployRequest(BaseModel):
    """Body for `POST /deployments` (an interface DTO, not a domain model). Provide ``model_id`` for
    a catalog model, or ``hf_repo`` (with ``gpu``) to deploy any vLLM-servable HF repo ad-hoc."""

    model_id: str | None = None
    hf_repo: str | None = None
    provider: str = "runpod"
    gpu: str | None = None
    context: int | None = None  # ad-hoc: max model length; omit to let vLLM auto-detect
    image: str | None = None  # ad-hoc: vLLM image
    disk: int | None = None  # ad-hoc: container disk GB
    wait: bool = False
    overrides: RuntimeOverrides | None = None


class EstimateRequest(BaseModel):
    model_id: str
    provider: str = "runpod"
    hours: float = 1.0


class LimitsRequest(BaseModel):
    """Body for `PUT /deployments/{id}/limits` (Tier A3). The domain model validates the values, so
    this carries them without re-stating the constraints."""

    max_concurrency: int
    max_queue: int = 0
    queue_timeout_s: float = 30.0


class ScaleRequest(BaseModel):
    """Body for `POST /scale` (Tier B1). Replicas are per model, not per deployment: the proxy
    load-balances every READY deployment of a model."""

    model_id: str
    replicas: int
    wait: bool = False


class BudgetRequest(BaseModel):
    """Body for `POST /budgets` (Tier A2). ``deployment_id`` None is account-wide."""

    limit_usd: float
    window: BudgetWindow = BudgetWindow.MONTHLY
    on_exceed: BudgetAction = BudgetAction.WARN
    deployment_id: str | None = None
    warn_fraction: float = 0.8


class AutoscaleRequest(BaseModel):
    """Body for `PUT /autoscale/{model_id}` (Tier B2)."""

    max_replicas: int
    target_rpm_per_replica: float
    min_replicas: int = 1


def create_app(
    orchestrator: Orchestrator,
    *,
    proxy_transport: httpx.AsyncBaseTransport | None = None,
    ui_dir: Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    app = FastAPI(title="open-lease", version="0.1.0")
    _install_auth(app, orchestrator, ui_served=ui_dir is not None)
    _install_error_handling(app)
    _install_cors(
        app, cors_origins if cors_origins is not None else orchestrator.config.cors_origins
    )

    @app.post("/deployments")
    async def deploy(body: DeployRequest) -> Deployment:
        if body.hf_repo:
            if not body.gpu:
                raise OrchestratorError("hf_repo requires a gpu (an ad-hoc model has no default)")
            return await orchestrator.deploy_adhoc(
                hf_repo=body.hf_repo,
                gpu=body.gpu,
                provider=body.provider,
                context_window=body.context or 0,
                image=body.image,
                disk_gb=body.disk,
                wait=body.wait,
                overrides=body.overrides,
            )
        if not body.model_id:
            raise OrchestratorError("provide model_id (a catalog model) or hf_repo")
        return await orchestrator.deploy_model(
            body.model_id,
            provider=body.provider,
            gpu=body.gpu,
            wait=body.wait,
            overrides=body.overrides,
        )

    @app.get("/deployments")
    def list_deployments(include_stopped: bool = False) -> list[Deployment]:
        return orchestrator.list_deployments(include_stopped=include_stopped)

    @app.get("/deployments/{deployment_id}")
    def get_deployment(deployment_id: str) -> Deployment:
        return orchestrator.get_deployment(deployment_id)

    @app.delete("/deployments/{deployment_id}", status_code=204)
    async def delete_deployment(deployment_id: str) -> None:
        await orchestrator.delete_deployment(deployment_id)

    @app.post("/deployments/{deployment_id}/stop")
    async def stop_deployment(deployment_id: str) -> Deployment:
        return await orchestrator.stop_deployment(deployment_id)

    @app.post("/deployments/{deployment_id}/restart")
    async def restart_deployment(deployment_id: str) -> Deployment:
        return await orchestrator.restart_deployment(deployment_id)

    @app.get("/deployments/{deployment_id}/logs")
    async def logs(deployment_id: str, tail: int = 100) -> list[str]:
        return list(await orchestrator.get_logs(deployment_id, tail=tail))

    @app.get("/deployments/{deployment_id}/health")
    async def health(deployment_id: str) -> HealthStatus:
        return await orchestrator.get_health(deployment_id)

    @app.get("/deployments/{deployment_id}/events")
    def events(deployment_id: str) -> list[Event]:
        return orchestrator.events(deployment_id)

    # --- capacity envelope (plan tiers A + B) -----------------------------------------
    # Schedules and limits are properties of one deployment, so they are sub-resources of it.
    # Budgets, autoscaling, and scale are account- or model-scoped, so they are their own resources.

    @app.put("/deployments/{deployment_id}/schedule")
    async def set_schedule(deployment_id: str, body: Schedule) -> Deployment:
        return await orchestrator.set_schedule(deployment_id, body)

    @app.delete("/deployments/{deployment_id}/schedule")
    async def clear_schedule(deployment_id: str) -> Deployment:
        return await orchestrator.clear_schedule(deployment_id)

    @app.put("/deployments/{deployment_id}/limits")
    async def set_limits(deployment_id: str, body: LimitsRequest) -> Deployment:
        return await orchestrator.set_limits(
            deployment_id,
            max_concurrency=body.max_concurrency,
            max_queue=body.max_queue,
            queue_timeout_s=body.queue_timeout_s,
        )

    @app.delete("/deployments/{deployment_id}/limits")
    async def clear_limits(deployment_id: str) -> Deployment:
        return await orchestrator.set_limits(deployment_id, max_concurrency=None)

    @app.post("/scale")
    async def scale(body: ScaleRequest) -> list[Deployment]:
        return await orchestrator.scale(body.model_id, body.replicas, wait=body.wait)

    @app.get("/budgets")
    def budgets() -> list[BudgetStatus]:
        return orchestrator.budget_status()

    @app.post("/budgets")
    async def create_budget(body: BudgetRequest) -> Budget:
        return await orchestrator.set_budget(
            limit_usd=body.limit_usd,
            window=body.window,
            on_exceed=body.on_exceed,
            deployment_id=body.deployment_id,
            warn_fraction=body.warn_fraction,
        )

    @app.delete("/budgets/{budget_id}", status_code=204)
    async def delete_budget(budget_id: str) -> None:
        if not await orchestrator.remove_budget(budget_id):
            raise PolicyNotFoundError(f"no budget with id {budget_id!r}")

    @app.get("/autoscale")
    def autoscale() -> list[AutoscalePolicy]:
        return orchestrator.list_autoscale()

    @app.put("/autoscale/{model_id}")
    async def set_autoscale(model_id: str, body: AutoscaleRequest) -> AutoscalePolicy:
        return await orchestrator.set_autoscale(
            model_id=model_id,
            max_replicas=body.max_replicas,
            target_rpm_per_replica=body.target_rpm_per_replica,
            min_replicas=body.min_replicas,
        )

    @app.delete("/autoscale/{model_id}", status_code=204)
    async def delete_autoscale(model_id: str) -> None:
        if not await orchestrator.remove_autoscale(model_id):
            raise PolicyNotFoundError(f"no autoscaling policy for {model_id!r}")

    @app.get("/models")
    def models() -> list[ModelSpec]:
        return orchestrator.list_models()

    @app.get("/providers")
    async def providers() -> list[ProviderInfo]:
        return await orchestrator.list_providers()

    @app.get("/availability")
    async def availability(
        model_id: str | None = None, gpu: str | None = None
    ) -> list[GpuAvailability]:
        return await orchestrator.gpu_availability(model_id=model_id, gpu_type=gpu)

    @app.get("/costs")
    def costs(deployment_id: str | None = None) -> list[CostRecord]:
        return orchestrator.get_costs(deployment_id)

    @app.get("/usage")
    def usage(deployment_id: str | None = None) -> list[UsageSummary]:
        return orchestrator.get_usage(deployment_id)

    @app.get("/volumes")
    async def volumes() -> list[VolumeInfo]:
        return await orchestrator.list_volumes()

    @app.delete("/volumes/{volume_id}", status_code=204)
    async def delete_volume(volume_id: str) -> None:
        await orchestrator.delete_volume(volume_id)

    @app.post("/estimate")
    async def estimate(body: EstimateRequest) -> CostEstimate:
        return await orchestrator.estimate_cost(
            body.model_id, provider=body.provider, hours=body.hours
        )

    from ..proxy.openai_proxy import create_proxy_app

    proxy = create_proxy_app(orchestrator, transport=proxy_transport)
    if ui_dir is not None:
        # Both the proxy and the UI want "/", so the proxy can't stay a "/" sub-mount here. Register
        # its /v1 routes directly (after the management routes), then serve the built UI as the
        # catch-all at "/". `gpu ui` / `gpu serve --ui` set this; without it, API-only as before.
        app.router.routes.extend(proxy.routes)
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    else:
        # The OpenAI proxy owns /v1/*; mount at root and let its own /v1 routes match.
        app.mount("/", proxy)
    return app


# --- plumbing -------------------------------------------------------------------------

_OPEN_PATHS = {"/docs", "/openapi.json", "/redoc"}
# The management + inference paths a token guards. When the UI is served, only these are guarded, so
# the static assets (which a browser cannot send a bearer header for) load; the UI's own API calls
# carry the token.
_API_PREFIXES = (
    "/deployments",
    "/models",
    "/providers",
    "/availability",
    "/costs",
    "/usage",
    "/volumes",
    "/estimate",
    "/scale",
    "/budgets",
    "/autoscale",
    "/v1",
)


def _install_auth(app: FastAPI, orchestrator: Orchestrator, *, ui_served: bool = False) -> None:
    token = orchestrator.config.api_token

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        path = request.url.path
        guarded = path not in _OPEN_PATHS and (not ui_served or path.startswith(_API_PREFIXES))
        if token is not None and guarded:
            if request.headers.get("authorization") != f"Bearer {token.get_secret_value()}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def _install_cors(app: FastAPI, origins: list[str]) -> None:
    """Opt-in cross-origin for the hosted workbench. No origins => untouched (same-origin only).

    An HTTPS page calling a local server needs two things: standard CORS, and a Private Network
    Access ack on the preflight (Chrome requires it for public -> loopback). Never wildcard: only
    the configured origins, so a running server is never exposed to arbitrary sites.
    """
    if not origins:
        return

    # Added after auth so it wraps (outer to) auth: CORS answers the preflight OPTIONS before auth
    # can 401 it (a preflight carries no Authorization header). allow_private_network sends the ack
    # Chrome requires for a public HTTPS page -> loopback server.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
        allow_private_network=True,
    )


def _install_error_handling(app: FastAPI) -> None:
    @app.exception_handler(OrchestratorError)
    async def _handle(request: Request, exc: OrchestratorError) -> JSONResponse:
        not_found = DeploymentNotFoundError | ModelNotFoundError | PolicyNotFoundError
        status = 404 if isinstance(exc, not_found) else 400
        return JSONResponse({"error": str(exc)}, status_code=status)

    @app.exception_handler(ValidationError)
    async def _handle_validation(request: Request, exc: ValidationError) -> JSONResponse:
        # A domain model rejected a value the request DTO could not judge (a concurrency cap below
        # 1, a non-positive budget limit). The constraint lives on the §6 model, so it surfaces here
        # rather than being re-stated as a parallel schema on the DTO.
        return JSONResponse({"error": str(exc.errors()[0]["msg"])}, status_code=400)
