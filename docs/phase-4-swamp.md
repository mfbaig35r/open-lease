# Phase 4: Swamp extension (requirements)

The visual front end for open-lease, built as a Swamp extension. Per the spec (§16), it **consumes
the Phase 2 REST API exclusively — no direct core imports.** open-lease is the backend; this is a
separate project that talks to it over HTTP. This doc pins the API contract and specifies each view;
Swamp-framework specifics (manifest, rendering, packaging) follow that project's conventions and are
out of scope here.

Status: not built. Phases 1 and 2 are live, so this is the spec-mandated next artifact (the API
contract it depends on is now real and stable).

## Boundary and dependencies

- **Backend**: an open-lease REST API reachable over HTTP (`gpu serve`, default `localhost:8000`).
- **No core imports.** The extension knows only the REST + `/v1/*` endpoints below. If a view needs
  something the API does not expose, that is a change to Phase 2, not a workaround in the extension.
- **Auth**: a single static bearer token (`api_token` on the backend). The extension takes a base
  URL and an optional token in its config and sends `Authorization: Bearer <token>` on every
  request. When the backend has no token set, requests are unauthenticated (localhost dev).
- **Errors**: non-2xx responses carry `{"error": "<one sentence>"}`. 404 = not found, 400 = bad
  request/operation, 401 = missing/wrong token. Surface the sentence to the user; never swallow it.

## The API contract this depends on

Management API (JSON, Pydantic §6 models):

| Endpoint | Used by | Returns |
|---|---|---|
| `POST /deployments` `{model_id, provider?, gpu?, wait?, overrides?}` | Deploy Wizard | `Deployment` |
| `GET /deployments?include_stopped=` | Deployment Explorer | `Deployment[]` |
| `GET /deployments/{id}` | Explorer detail (poll) | `Deployment` |
| `POST /deployments/{id}/stop` | Explorer | `Deployment` |
| `POST /deployments/{id}/restart` | Explorer | `Deployment` |
| `DELETE /deployments/{id}` | Explorer (confirm) | 204 |
| `GET /deployments/{id}/logs?tail=` | Log Viewer | `string[]` |
| `GET /deployments/{id}/health` | Health Dashboard | `HealthStatus` |
| `GET /deployments/{id}/events` | Timeline | `Event[]` |
| `GET /models?` | Model Catalog | `ModelSpec[]` |
| `GET /providers` | Catalog / status | `ProviderInfo[]` |
| `GET /availability?model_id=` | Deploy Wizard | `GpuAvailability[]` |
| `GET /costs?deployment_id=` | Cost Dashboard | `CostRecord[]` |
| `GET /volumes` | Cost Dashboard | `VolumeInfo[]` |
| `POST /estimate` `{model_id, provider?, hours?}` | Deploy Wizard | `CostEstimate` |

Inference (OpenAI-compatible, mounted at `/v1/*`):

| Endpoint | Used by |
|---|---|
| `GET /v1/models` | Model Chat (which models are live) |
| `POST /v1/chat/completions` (supports `stream: true`, SSE) | Model Chat |

Key model fields the views rely on:

- `Deployment`: `id`, `model_id`, `provider`, `desired_state`, `observed_state` (the enum:
  requested/provisioning/booting/downloading_model/starting_server/ready/degraded/stopping/stopped/failed),
  `instance` (`{provider_instance_id, gpu_type, state}` or null), `endpoint_url`,
  `download_progress` (0..1 or null), `state_history[]` (`{from_state, to_state, at, reason}`),
  `failure` (`{stage, message, retryable, attempts}` or null), `created_at`, `updated_at`.
- `HealthStatus`: `status`, `checks{name -> {ok, latency_ms, detail}}`.
- `CostRecord`: `deployment_id`, `gpu_hourly_usd`, `accrued_usd`, `estimated_monthly_usd`,
  `started_at`, `stopped_at`.
- `Event`: `id`, `at`, `kind`, `payload`.
- `GpuAvailability`: `data_center_id`, `gpu_type_id`, `available`, `stock_status`.

## Views

### 1. Deployment Explorer

The home view. A table of deployments (`GET /deployments`) with id, model, state (color by
observed_state), GPU, endpoint, uptime (from `created_at`), and accrued cost (sum of that
deployment's `GET /costs?deployment_id=`). A toggle includes stopped ones. Row actions: **stop**,
**restart**, **delete** (delete requires a confirm dialog — it is destructive and irreversible). A
selected row opens a detail panel that polls `GET /deployments/{id}` every ~3s while the deployment
is in a non-terminal state, showing the state, a progress hint (`download_progress` as a percent, or
elapsed-in-stage), the instance, the endpoint URL, and any `failure`.

### 2. Model Catalog + Deploy Wizard

`GET /models` renders the catalog (id, params, context, min GPU, capabilities, license). Selecting a
model opens the wizard: it calls `GET /availability?model_id=` (which data centers have the GPU right
now) and `POST /estimate` (projected $/hr), lets the user optionally override the GPU or launch args,
and submits `POST /deployments`. Warn clearly if no data center has capacity before submitting.
Default to non-blocking deploy (`wait: false`) and hand off to the Explorer detail panel to watch it
reach READY.

### 3. Cost Dashboard

`GET /costs` for accrued + projected-monthly per deployment, with a total. `GET /volumes` for the
persistent-cache storage line (`estimated_monthly_usd` per volume). Make clear that deployment cost is
GPU-hours and volume cost is storage — they are separate.

### 4. Log Viewer

`GET /deployments/{id}/logs?tail=` in a scrolling pane, with a tail-size control and a manual/auto
refresh. Note honestly that some providers (RunPod) expose no logs, so this may be empty; the
Explorer's progress hint is the fallback signal during bring-up.

### 5. Health Dashboard

`GET /deployments/{id}/health` rendered as a check-by-check panel (instance_alive, http_alive,
model_loaded, latency), each with ok/fail and detail. Poll while the deployment is serving. Health is
report-only in Phase 1 (DEGRADED is surfaced, not auto-healed), so the only remediation offered is
`restart`.

### 6. Deployment Timeline

`GET /deployments/{id}/events` rendered as a vertical timeline (requested → instance_created →
model_download → server_started → deployment_ready, plus health/reconcile/cost/orphan events). Each
node shows kind, timestamp, and payload. This is the audit trail; read-only.

### 7. Model Chat

`GET /v1/models` lists the READY deployments a user can chat with. A chat pane posts to
`POST /v1/chat/completions` with the selected model (the catalog id or HF repo both route), rendering
streamed tokens when `stream: true`. This is the "talk to my model" payoff, in the browser.

## Non-functional

- **Read-heavy polling, not push.** The backend has no websockets; views poll. Poll fast (~2–3s) only
  for deployments in a non-terminal state; back off to ~30s or on-demand once READY/STOPPED.
- **Destructive actions confirm.** `delete` (and arguably `restart`, a full cold start) require an
  explicit confirm, mirroring the CLI and MCP `confirm` semantics.
- **Cost visibility is a feature, not a footnote.** Accrued dollars should be visible wherever a
  running deployment is shown, since the core value proposition is not leaving pods billing.
- **Graceful degradation.** If the backend is unreachable or unauthorized, show a clear connection/
  auth banner, not a blank view.

## Out of scope for Phase 4

- Multi-tenant auth (the backend is single static token; multi-tenancy is a later roadmap item).
- Writing to the core directly, or any orchestration logic in the extension.
- Provider/model management beyond what the API exposes (adding catalog entries is a backend change).

## Open decisions for the Swamp side

- **Rendering + packaging**: follows the Swamp extension conventions (manifest, safety analyzer, the
  embedded-Deno gotchas from the other swamp packs); specify in that repo.
- **Streaming transport**: confirm Swamp's HTTP client can consume SSE for `/v1/chat/completions`;
  fall back to non-streaming completions if not.
- **Where the extension runs** vs where the backend runs (same host, or a configured base URL).
