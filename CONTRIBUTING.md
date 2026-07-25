# Contributing to open-lease

Thanks for looking. open-lease is the orchestration layer, not the provider: the goal is that a
new provider or runtime is a small, well-contained addition, and that the reconcile core stays
small and exhaustively tested.

## Dev setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mfbaig35r/open-lease && cd open-lease
uv sync --extra dev        # engine + CLI + proxy + REST + MCP + test/lint tools
```

## Before you push

CI runs both of these, and they are different checks:

```bash
uv run python -m pytest tests/ -q
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

The suite is offline: it runs against the in-memory mock provider and a mocked vLLM transport, so
it needs no credentials and spends nothing.

## Live-GPU testing (optional, costs real money)

The real-RunPod path is opt-in and never runs in CI. It needs `RUNPOD_API_KEY` (and `HF_TOKEN`
for gated models) in a local `.env`. Rules learned the expensive way:

- Stop a deployment the instant a check passes. A pod left READY bills the whole time.
- Tear down with `gpu stop <id>` then `gpu delete <id>`, never a raw provider-side pod delete: the
  latter leaves the deployment record wanting READY, and the reconciler correctly recreates it.
- Wrap any live test in a hard cleanup that force-deletes every `gpu-orch-*` pod on exit.

## Architecture rules (the short version)

The full, non-negotiable list is in [CLAUDE.md](CLAUDE.md). The ones that matter most in review:

- `next_step()` is a **pure** function: no network, no clock, no side effects. It is the most
  tested code in the repo. New reconcile behavior gets a pure test in the desired x observed matrix
  first.
- The reconciler takes **one step per tick**. Never chain stages in a single pass.
- **Cost-safety invariant**: no FAILED or STOPPED deployment ever keeps a running instance. Any
  code path that creates a provider instance has a matching cleanup path, and that path has a test.
- No plugin frameworks or dynamic loading. A provider or runtime is an ABC plus a module-level dict
  entry. Interfaces (CLI / REST / MCP) contain no business logic: parse, call the Orchestrator,
  render.
- Type hints everywhere; Pydantic v2 for all domain models. No file over ~400 lines except
  `models.py`; no function over ~50 lines.

## Adding a provider or model

- **Provider**: implement the Provider ABC against the contract suite. Walkthrough in
  [docs/adding-a-provider.md](docs/adding-a-provider.md).
- **Model**: add an entry to `catalog/models.toml`. Set `validated_at` only after a real deploy has
  reached READY and served a completion; leave it empty otherwise so the catalog stays honest.

## Cutting a release

Run the script:

```bash
uv run python scripts/release.py 0.5.0 --dry-run   # preflight + verify, change nothing
uv run python scripts/release.py 0.5.0             # verify, then ask before publishing
```

Land and push the workbench work in [open-lease-ui](https://github.com/mfbaig35r/open-lease-ui)
first, and write the release notes under `## [Unreleased]` in the CHANGELOG. The script does the rest
and refuses to release an empty `[Unreleased]` section.

Publishing is tag-push, not a GitHub Release: a `v*` tag runs `.github/workflows/publish.yml`, which
builds the sdist and wheel and publishes to PyPI via Trusted Publishing. A version can never be
re-uploaded, which is why the script checks PyPI before it starts and stops for confirmation before
anything irreversible.

What it does, so you can do it by hand if it ever gets in the way:

1. **Preflight.** Both repos clean, on `main`, in sync with origin. The version is not already tagged
   in either repo and not already on PyPI. `git`, `gh`, `uv`, `pnpm` present and `gh` logged in.
2. **Version.** `version` in `pyproject.toml`, `version` in the UI's `package.json` (the workbench is
   versioned in lockstep, since it ships only inside this wheel), and the CHANGELOG's `[Unreleased]`
   section closed as the new version with today's date (UTC, like the tag) plus its compare link.
3. **Verify.** `pnpm bundle` for the workbench, then `ruff check`, `ruff format --check`, the suite,
   `uv build`, `twine check`, and an install of the built wheel into a throwaway venv to confirm it
   reports the new version and carries `gpu_orchestrator/web/index.html`.
4. **Confirm.** Everything to here is local and revertible (`git checkout -- pyproject.toml
   CHANGELOG.md`). It prints what it is about to do and waits.
5. **Publish.** Commit + push + tag the UI repo, then commit, tag, and push here. The UI tag goes
   first because pushing the release tag is what starts the workflow that resolves it.

### Which workbench a release bundles

The publish workflow **derives** the workbench ref from the release tag: `v0.5.0` here builds
open-lease-ui at `v0.5.0`. That is why the two version in lockstep, and why there is nothing to keep
in step by hand: the tag names the workbench. If open-lease-ui has no matching tag, the workflow fails
before it builds anything, with a message saying to tag the workbench first.

`OPEN_LEASE_UI_REF` remains as a **deliberate override** for the rare release that needs a workbench
other than its own version (a backend-only fix on an older UI, or a `main` build to exercise the
pipeline). It is normally unset. When it is set, the workflow warns, the run summary marks the
workbench as overridden rather than derived, and `scripts/release.py` reports it during preflight and
again at the confirmation prompt: an override is legitimate but should never be an accident. Clear one
with `gh variable delete OPEN_LEASE_UI_REF`.

Either way, the run summary records the workbench ref and commit that went into the wheel, so check
that it says what you expect, then install from PyPI into a scratch venv as a final check.

The script's file surgery is covered by `tests/unit/test_release_script.py`, so a change to the
CHANGELOG format that would break a release fails in CI first.

## Pull requests

Branch off `main`, keep tests and both ruff checks green, and open a PR. Small, focused PRs review
fastest. If a change touches the reconcile core, call that out so it gets a careful read.

Writing style: no em dashes in code, comments, docs, or copy. Use a period, comma, colon, or
parentheses instead.
