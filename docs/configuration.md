# Configuration

Settings come from `config.py` (Pydantic Settings). Secrets are `SecretStr`, so they never appear in
logs, `repr`, or `state.db`.

## Precedence

Highest wins:

1. CLI flags (e.g. `--gpu`, `--port`)
2. Environment variables: `GPU_ORCH_*`, plus the unprefixed `RUNPOD_API_KEY` and `HF_TOKEN`
3. `./.env` in the working directory
4. `./gpu-orchestrator.toml`
5. `~/.gpu-orchestrator/config.toml`
6. Built-in defaults

`gpu config` prints the effective configuration with secrets masked.

## Credentials

| Variable | Required | Purpose |
|---|---|---|
| `RUNPOD_API_KEY` | yes (for the RunPod provider) | Create/destroy pods and volumes; read GPU availability. |
| `HF_TOKEN` | optional | Avoids Hugging Face rate limits on downloads; required for gated models (e.g. Llama). Injected into the pod env. |

The quickest setup is `cp .env.example .env` and fill in the two values. `.env` is gitignored;
`.env.example` documents every knob.

## Settings

All app settings use the `GPU_ORCH_` prefix as environment variables (e.g.
`GPU_ORCH_RECONCILE_INTERVAL=10`), or the unprefixed field name in a TOML file.

### Identity and storage

| Setting | Default | Notes |
|---|---|---|
| `namespace` | sanitized hostname | Scopes the orphan sweep so it never touches another install's pods. |
| `state_db` | `~/.gpu-orchestrator/state.db` | SQLite state file. |

### Loop cadences (seconds)

| Setting | Default |
|---|---|
| `reconcile_interval` | 10 |
| `health_poll_interval` | 30 |
| `health_failure_threshold` | 3 (consecutive failures before DEGRADED) |
| `orphan_sweep_interval` | 300 |
| `orphan_grace_period` | 120 |

### Retry and per-stage timeout budgets (seconds)

| Setting | Default |
|---|---|
| `retry_max_attempts` | 3 |
| `retry_backoff_min` / `retry_backoff_max` | 10 / 60 |
| `timeout_provisioning` | 300 |
| `timeout_booting` | 300 |
| `timeout_download` | 1800 |
| `timeout_starting` | 300 |

A model's profile may override the download/starting budget via `startup_timeout_seconds` (large
models download slowly; this is what `gpu status` shows the ETA against).

### Persistent model cache (opt-in)

| Setting | Default | Notes |
|---|---|---|
| `cache_volume_enabled` | `false` | A shared per-namespace network volume caches weights so a warm redeploy skips the download. |
| `cache_volume_size_gb` | 100 | RunPod minimum is 10GB. |
| `runpod_data_center_id` | none | RunPod DC id (e.g. `US-KS-2`, `EU-RO-1`). If unset, open-lease auto-picks a DC that currently has the GPU in stock. |

Caching pins the pod to one data center (a network volume is region-locked), which reduces GPU
availability spread. Leave it off unless you redeploy the same models often. Use `gpu availability
<model>` to see which data centers have capacity, and `gpu volumes` to manage volumes.

### Retention and process lifecycle

| Setting | Default | Notes |
|---|---|---|
| `event_retention_days` | 30 | The daemon prunes events older than this. |
| `auto_daemon` | `false` | If true, a non-blocking `gpu deploy` starts a daemon when none is running (default is to warn). |
| `daemon_pid_file` / `daemon_log_file` | `~/.gpu-orchestrator/daemon.{pid,log}` | |
| `proxy_pid_file` / `proxy_log_file` | `~/.gpu-orchestrator/proxy.{pid,log}` | |

### Proxy

| Setting | Default |
|---|---|
| `proxy_host` | `localhost` |
| `proxy_port` | 8080 |

## Example `gpu-orchestrator.toml`

```toml
namespace = "my-lab"
reconcile_interval = 10
cache_volume_enabled = true
runpod_data_center_id = "EU-RO-1"
event_retention_days = 14
```
