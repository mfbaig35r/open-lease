"""gpu_orchestrator.api: the REST interface (Phase 2). A thin FastAPI layer over the core."""

from .app import create_app

__all__ = ["create_app"]
