"""REST API (spec §16, Phase 2). A thin FastAPI layer over the `Orchestrator`.

Every route is: take the request, call one Orchestrator method, return a §6 domain model (FastAPI
serializes it). No parallel schemas, no business logic here. The OpenAI proxy is mounted so `/v1/*`
serves inference alongside the management API. Auth is a single static bearer token from config
(multi-tenancy deferred); when no token is set the API is open, so bind to localhost.

Like the CLI and the proxy, this is an interface: the same core, a different shape.
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.orchestrator import Orchestrator
from ..errors import DeploymentNotFoundError, ModelNotFoundError, OrchestratorError
from ..models import (
    CostEstimate,
    CostRecord,
    Deployment,
    Event,
    GpuAvailability,
    HealthStatus,
    ModelSpec,
    ProviderInfo,
    RuntimeOverrides,
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


def create_app(
    orchestrator: Orchestrator, *, proxy_transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    app = FastAPI(title="open-lease", version="0.1.0")
    _install_auth(app, orchestrator)
    _install_error_handling(app)

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

    @app.get("/volumes")
    async def volumes() -> list[VolumeInfo]:
        return await orchestrator.list_volumes()

    @app.post("/estimate")
    async def estimate(body: EstimateRequest) -> CostEstimate:
        return await orchestrator.estimate_cost(
            body.model_id, provider=body.provider, hours=body.hours
        )

    # The OpenAI proxy owns /v1/*; mount at root and let its own /v1 routes match.
    from ..proxy.openai_proxy import create_proxy_app

    app.mount("/", create_proxy_app(orchestrator, transport=proxy_transport))
    return app


# --- plumbing -------------------------------------------------------------------------

_OPEN_PATHS = {"/docs", "/openapi.json", "/redoc"}


def _install_auth(app: FastAPI, orchestrator: Orchestrator) -> None:
    token = orchestrator.config.api_token

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if token is not None and request.url.path not in _OPEN_PATHS:
            if request.headers.get("authorization") != f"Bearer {token.get_secret_value()}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def _install_error_handling(app: FastAPI) -> None:
    @app.exception_handler(OrchestratorError)
    async def _handle(request: Request, exc: OrchestratorError) -> JSONResponse:
        status = 404 if isinstance(exc, DeploymentNotFoundError | ModelNotFoundError) else 400
        return JSONResponse({"error": str(exc)}, status_code=status)
