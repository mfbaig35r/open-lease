"""Runtime seam package.

Importing ``base`` first guarantees the Runtime ABC is defined before the concrete runtimes load,
so ``RUNTIMES`` (built in ``base``) is safe regardless of which submodule a caller imports first.
"""

from __future__ import annotations

from .base import RUNTIMES, Runtime
from .vllm import VLLMRuntime

__all__ = ["RUNTIMES", "Runtime", "VLLMRuntime"]
