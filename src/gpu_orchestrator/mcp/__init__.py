"""gpu_orchestrator.mcp: the MCP interface (Phase 3). Agent-facing tools over the core."""

from .server import create_server, run

__all__ = ["create_server", "run"]
