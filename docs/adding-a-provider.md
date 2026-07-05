# Adding a provider

A provider provisions compute and knows nothing about LLMs. Adding one is: subclass an ABC, register
it in a dict, and make the contract suite pass. No plugin framework, no entry-point discovery.

The contract suite (`tests/contract/test_provider_contract.py`) **is** the provider specification. It
runs against the mock always, and against your provider when `GPU_ORCH_INTEGRATION=1` (which costs
real money). If it passes, your provider is done; if it does not, it is not. Read `providers/mock.py`
as the reference implementation and `providers/runpod.py` as the real one.

## 1. Subclass `Provider`

```python
from ..models import GpuAvailability, Instance, InstanceRequest, ProviderCapabilities, VolumeInfo
from .base import Provider

class MyProvider(Provider):
    name = "mycloud"

    def __init__(self, *, namespace: str, api_key: str | None = None) -> None:
        super().__init__(namespace=namespace)
        ...
```

The required methods (all `async`):

| Method | Contract |
|---|---|
| `capabilities()` | Return `ProviderCapabilities` with a non-empty `gpu_types`. |
| `create_instance(request)` | Create a pod named `request.name`; return an `Instance`. |
| `get_instance(id)` | Return the `Instance`, or **`None` if it is gone**. This is the canonical "it no longer exists" signal the reconciler relies on. |
| `destroy_instance(id)` | Idempotent: destroying something absent is a no-op, not an error. |
| `list_instances()` | All instances owned by this install, **filtered to `self.instance_prefix()`**. |
| `find_instance_by_deployment_id(deployment_id)` | Look up by the exact name `self.instance_name(deployment_id)`; `None` if absent. |
| `resolve_endpoint_url(instance, port)` | A routable public URL for `port`, or **`None` until it is actually routable**. |
| `get_logs(id, tail)` | Recent log lines, or `[]` if the provider has no log API. |

Optional (the base returns unsupported/empty defaults, so implement only what your provider has):
`ensure_cache_volume`, `list_volumes`, `delete_volume` (persistent cache), and `gpu_availability`
(per-data-center GPU stock).

## 2. Register it

```python
# providers/base.py
from .mycloud import MyProvider
PROVIDERS: dict[str, type[Provider]] = {"runpod": RunPodProvider, "mock": MockProvider,
                                        "mycloud": MyProvider}
```

Then teach `core/orchestrator.py:build_provider` how to construct it (credentials, etc.), mirroring
the RunPod branch.

## 3. The obligations that keep the system honest

These are why the seam is safe, and the contract suite checks most of them:

- **Naming is the hook for everything.** Every instance is named
  `gpu-orch-{namespace}-{deployment_id}` via `self.instance_name(...)` (do not roll your own). The
  orphan sweep, adoption, and per-install isolation all hang on this. `list_instances` and
  `find_instance_by_deployment_id` must scope to `self.instance_prefix()` so you never see, or touch,
  another install's pods.
- **Store provider-native state verbatim.** `Instance.state` is the raw provider string (RunPod's
  `desiredStatus`, etc.). Do **not** translate it to a `DeploymentState`; that happens in exactly one
  core function, `map_to_observed_state`, so every weird provider semantic is handled and tested in
  one auditable place.
- **`None` means gone.** `get_instance` returning `None` is how the reconciler learns a pod died. If
  your API returns a tombstone record instead of a 404, map dead states to `None` (or leave them for
  `map_to_observed_state` to fold to REQUESTED, but prefer `None` for a truly gone pod).
- **Endpoint URLs are provider knowledge.** Runtimes only declare a port; you turn `(instance, port)`
  into a URL and return `None` until it is routable. The reconciler treats `None` as "keep waiting."
- **Never leak a raw exception.** Convert provider API errors to a typed `OrchestratorError`
  subclass: `ProviderAPIError` (unreachable/4xx/5xx), `InstanceCreationError` (accepted but never
  came up), `NotSupportedError` (an optional capability you do not implement). The reconciler catches
  these and drives retry/backoff/FAILED; a bare exception escaping the seam is a bug.
- **Cost-safety is shared.** You do not track cost, but you must make destruction reliable and
  idempotent, and `list_instances` complete, so the orphan sweep can guarantee no pod outlives its
  deployment.

## 4. Run the contract

```bash
# offline, against your provider's logic where it does not need the network:
uv run python -m pytest tests/contract -q

# against the real API (creates and destroys real instances; costs money):
GPU_ORCH_INTEGRATION=1 uv run python -m pytest tests/contract -q
```

Every test that creates an instance cleans it up in a `finally`. When the suite is green against
`GPU_ORCH_INTEGRATION=1`, the provider is real.
