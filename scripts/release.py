"""Cut an open-lease release in one command, so no step depends on remembering it.

The release spans two repos: the published wheel bundles the visual workbench, which lives in
open-lease-ui. The publish workflow derives the workbench ref from the release tag (v0.5.0 here
builds open-lease-ui at v0.5.0), so the two version in lockstep and nothing needs keeping in step by
hand. What this script owes that arrangement is the UI tag itself, pushed before the release tag,
since pushing the release tag is what starts the run that looks for it.

What it does, in order:

1. **Preflight.** Both repos clean, on main, and in sync with origin. The target version is not
   already tagged here, tagged in the UI repo at a different commit, or on PyPI (a PyPI version can
   never be re-uploaded, so this is the one check worth being pedantic about). It also reports a
   leftover ``OPEN_LEASE_UI_REF`` override, which would send a workbench other than this tag's.
2. **Local edits.** ``version`` in pyproject.toml and in the UI's package.json, and the CHANGELOG's
   ``[Unreleased]`` section closed as the new version with today's date plus its compare link.
3. **Verify.** Bundle the workbench, then ruff, the test suite, ``uv build``, ``twine check``, and
   an install of the built wheel into a throwaway venv to confirm it reports the new version and
   carries the bundled UI.
4. **Confirm.** Everything above is local and revertible. Everything below is not, so it stops here
   and shows exactly what it is about to do to GitHub and PyPI.
5. **Publish.** Bump, commit, and tag the UI repo, then commit, tag, and push here. The UI tag goes
   first because the release tag push is what starts the workflow that resolves it.

Usage:
    uv run python scripts/release.py 0.5.0              # verify, then ask before publishing
    uv run python scripts/release.py 0.5.0 --dry-run    # preflight + verify only, no edits at all
    uv run python scripts/release.py 0.5.0 --yes        # no prompt (CI or a deliberate one-liner)

The UI repo defaults to ../open-lease-ui; override with --ui-repo.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = "open-lease"
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
# Files the version bump touches, so the failure hint and the release commit agree on the set.
TOUCHED = ("pyproject.toml", "CHANGELOG.md", "uv.lock")
# Set once the UI repo is resolved, so a failure can name the file to revert there too.
_UI_REPO: Path | None = None


class ReleaseError(Exception):
    """A precondition failed or a step did not do what it claimed. Always actionable."""


# --- shell ----------------------------------------------------------------------------


def run(*args: str, cwd: Path = REPO, quiet: bool = False) -> str:
    """Run a command, returning stdout. Raises ReleaseError with the command and stderr on failure,
    so a broken step never looks like a passing one."""
    if not quiet:
        print(f"  $ {' '.join(args)}")
    done = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()
        raise ReleaseError(f"`{' '.join(args)}` failed in {cwd}:\n{detail}")
    return done.stdout.strip()


def step(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def warn(message: str) -> None:
    print(f"  ! {message}")


# --- preflight ------------------------------------------------------------------------


def check_repo_ready(path: Path, label: str) -> str:
    """A repo is releasable when it is clean, on main, and matching origin. Returns its HEAD sha."""
    if not (path / ".git").is_dir():
        raise ReleaseError(f"{label}: {path} is not a git repo (pass --ui-repo?)")
    if run("git", "status", "--porcelain", cwd=path, quiet=True):
        raise ReleaseError(f"{label}: working tree is dirty; commit or revert first")
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=path, quiet=True)
    if branch != "main":
        raise ReleaseError(f"{label}: on branch {branch!r}, not main")
    run("git", "fetch", "origin", "main", "--tags", cwd=path, quiet=True)
    local = run("git", "rev-parse", "HEAD", cwd=path, quiet=True)
    remote = run("git", "rev-parse", "origin/main", cwd=path, quiet=True)
    if local != remote:
        raise ReleaseError(f"{label}: main and origin/main differ; push or pull first")
    return local


def check_tag_free(path: Path, tag: str, label: str, *, expect_sha: str | None = None) -> bool:
    """True when the tag already exists and points where we want it (idempotent re-run), False when
    it does not exist yet. Raises when it exists somewhere else."""
    existing = run("git", "tag", "--list", tag, cwd=path, quiet=True)
    if not existing:
        return False
    at = run("git", "rev-list", "-n", "1", tag, cwd=path, quiet=True)
    if expect_sha is None or at != expect_sha:
        raise ReleaseError(
            f"{label}: tag {tag} already exists (at {at[:9]}). Delete it or pick another version."
        )
    warn(f"{label}: tag {tag} already exists at the right commit; will not recreate it")
    return True


def check_pypi_free(version: str) -> None:
    url = f"https://pypi.org/pypi/{PACKAGE}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        warn(f"could not reach PyPI to check for a duplicate version ({exc}); continuing")
        return
    if version in data.get("releases", {}):
        raise ReleaseError(
            f"{PACKAGE} {version} is already on PyPI. A version can never be re-uploaded; "
            "bump to the next one."
        )
    print(f"  PyPI latest is {data['info']['version']}; {version} is free")


def check_tools() -> None:
    for tool in ("git", "gh", "uv", "pnpm"):
        if shutil.which(tool) is None:
            raise ReleaseError(f"{tool} is not on PATH; it is needed to cut a release")
    run("gh", "auth", "status", quiet=True)  # raises with gh's own message when not logged in


# --- local edits ----------------------------------------------------------------------


def bump_pyproject(version: str, *, write: bool) -> str:
    path = REPO / "pyproject.toml"
    text = path.read_text()
    match = re.search(r'^version = "(.+?)"$', text, flags=re.M)
    if match is None:
        raise ReleaseError('could not find `version = "..."` in pyproject.toml')
    previous = match.group(1)
    if previous == version:
        warn(f"pyproject.toml is already {version}")
        return previous
    if write:
        path.write_text(text[: match.start(1)] + version + text[match.end(1) :])
    print(f"  pyproject.toml: {previous} -> {version}")
    return previous


def close_changelog(version: str, *, write: bool) -> None:
    """Turn `## [Unreleased]` into the new version, open a fresh Unreleased, and add the compare
    link. Refuses to release an empty Unreleased section: a version with no notes is a mistake."""
    path = REPO / "CHANGELOG.md"
    text = path.read_text()
    if f"## [{version}]" in text:
        warn(f"CHANGELOG already has a [{version}] section")
        return
    marker = "## [Unreleased]\n"
    if marker not in text:
        raise ReleaseError("CHANGELOG.md has no `## [Unreleased]` section")
    head, _, rest = text.partition(marker)
    body = rest.split("\n## [", 1)[0]
    if not body.strip():
        raise ReleaseError("the CHANGELOG's [Unreleased] section is empty; nothing to release")

    previous = re.search(r"^## \[(\d+\.\d+\.\d+)\]", rest, flags=re.M)
    if previous is None:
        raise ReleaseError("could not find the previous version heading in CHANGELOG.md")
    prev = previous.group(1)

    # UTC, not local: the tag, the PyPI upload, and the workflow run are all stamped in UTC, so a
    # release cut late in the evening should not be dated the day before them.
    today = datetime.now(tz=UTC).date().isoformat()
    updated = f"{head}{marker}\n## [{version}] - {today}\n{rest}"
    old_link = f"[Unreleased]: https://github.com/mfbaig35r/{PACKAGE}/compare/v{prev}...HEAD"
    if old_link not in updated:
        raise ReleaseError(
            f"expected the compare link {old_link!r} at the bottom of CHANGELOG.md; fix it by hand"
        )
    new_links = (
        f"[Unreleased]: https://github.com/mfbaig35r/{PACKAGE}/compare/v{version}...HEAD\n"
        f"[{version}]: https://github.com/mfbaig35r/{PACKAGE}/compare/v{prev}...v{version}"
    )
    updated = updated.replace(old_link, new_links)
    if write:
        path.write_text(updated)
    print(f"  CHANGELOG.md: [Unreleased] closed as [{version}], compare link added")


def bump_ui_version(ui: Path, version: str, *, write: bool) -> bool:
    """Keep the workbench's package.json in lockstep (it ships only inside our wheel). True when a
    commit is needed."""
    path = ui / "package.json"
    text = path.read_text()
    match = re.search(r'^(\s*)"version": "(.+?)",$', text, flags=re.M)
    if match is None:
        raise ReleaseError("could not find a version field in the UI's package.json")
    if match.group(2) == version:
        return False
    if write:
        path.write_text(text[: match.start(2)] + version + text[match.end(2) :])
    print(f"  ui package.json: {match.group(2)} -> {version}")
    return True


# --- verification ---------------------------------------------------------------------


def bundle_ui(ui: Path) -> None:
    run("pnpm", "bundle", cwd=ui)
    index = REPO / "src" / "gpu_orchestrator" / "web" / "index.html"
    if not index.is_file():
        raise ReleaseError(f"`pnpm bundle` did not produce {index}")
    print(f"  bundled workbench: {sum(1 for _ in index.parent.rglob('*') if _.is_file())} files")


def verify_build(version: str) -> Path:
    run("uv", "run", "ruff", "check", "src/", "tests/")
    run("uv", "run", "ruff", "format", "--check", "src/", "tests/")
    run("uv", "run", "python", "-m", "pytest", "tests/", "-q")
    shutil.rmtree(REPO / "dist", ignore_errors=True)
    run("uv", "build")
    wheels = sorted((REPO / "dist").glob(f"*{version}*.whl"))
    if not wheels:
        raise ReleaseError(f"uv build produced no wheel for {version} (check the version bump)")
    wheel = wheels[0]
    names = zipfile.ZipFile(wheel).namelist()
    for required in ("gpu_orchestrator/web/index.html", "gpu_orchestrator/data/models.toml"):
        if not any(n.endswith(required) for n in names):
            raise ReleaseError(f"{wheel.name} is missing {required}")
    # Only the distributions: uv drops a .gitignore in dist/, and twine errors on anything it cannot
    # recognise as a distribution.
    dists = sorted(p for p in (REPO / "dist").glob("*") if p.suffix in {".whl", ".gz"})
    run("uvx", "twine", "check", *(str(p) for p in dists))
    print(f"  {wheel.name}: builds, bundles the UI, passes twine")
    return wheel


def verify_wheel_installs(wheel: Path, version: str) -> None:
    """Install the artifact we are about to publish into a throwaway venv. Catches the failures that
    only appear past the build: a wrong version string, or a wheel that omits the workbench."""
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        run("uv", "venv", str(venv), quiet=True)
        run(
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            str(venv / "bin" / "python"),
            f"{wheel}[all]",
        )
        reported = run(str(venv / "bin" / "gpu"), "--version", quiet=True)
        if reported != version:
            raise ReleaseError(
                f"installed wheel reports version {reported!r}, expected {version!r}"
            )
        run(
            str(venv / "bin" / "python"),
            "-c",
            "import importlib.resources as r, sys;"
            "base = r.files('gpu_orchestrator') / 'web';"
            "sys.exit(0 if (base / 'index.html').is_file() else 'wheel has no bundled UI')",
            quiet=True,
        )
        print(f"  clean install reports {reported} and carries the bundled workbench")


# --- publish --------------------------------------------------------------------------


def tag_ui(ui: Path, tag: str, version: str, *, needs_commit: bool) -> None:
    if needs_commit:
        run("git", "add", "package.json", cwd=ui)
        run("git", "commit", "-m", f"Version {version} (lockstep with open-lease)", cwd=ui)
        run("git", "push", "origin", "main", cwd=ui)
    run(
        "git",
        "tag",
        "-a",
        tag,
        "-m",
        f"open-lease-ui {version}: the workbench in open-lease {version}",
        cwd=ui,
    )
    run("git", "push", "origin", tag, cwd=ui)


def check_ui_override(tag: str) -> str | None:
    """The publish workflow derives the workbench ref from the release tag, so there is nothing to
    set here. What can still go wrong is a leftover OPEN_LEASE_UI_REF override from some one-off
    release, which would quietly ship a workbench other than the one the tag names. Report it; the
    confirmation prompt repeats it, because an override is legitimate but never accidental."""
    listed = run("gh", "variable", "list", quiet=True)
    for line in listed.splitlines():
        name, _, remainder = line.partition("\t")
        if name == "OPEN_LEASE_UI_REF":
            override = remainder.split("\t")[0].strip()
            warn(
                f"OPEN_LEASE_UI_REF is set to {override!r}, so the wheel bundles that workbench "
                f"and not {tag}. Run `gh variable delete OPEN_LEASE_UI_REF` unless you mean it."
            )
            return override
    print(f"  workbench ref derives from the tag: open-lease-ui will be built at {tag}")
    return None


def tag_release(version: str, tag: str, *, tag_exists: bool) -> None:
    # uv.lock records this package's own version, so `uv build` rewrites it during verification.
    # Leaving it out ends the release with a dirty tree, which then blocks the next one.
    run("git", "add", *TOUCHED)
    if run("git", "diff", "--cached", "--name-only", quiet=True):
        run("git", "commit", "-m", f"Release {version}")
    if not tag_exists:
        run("git", "tag", "-a", tag, "-m", f"{PACKAGE} {version}")
    run("git", "push", "origin", "main")
    run("git", "push", "origin", tag)


# --- driver ---------------------------------------------------------------------------


def confirm(version: str, tag: str, ui: Path, override: str | None) -> None:
    workbench = (
        f"  ! the wheel will bundle the workbench at {override!r}, not {tag}, because "
        "OPEN_LEASE_UI_REF is set\n"
        if override
        else ""
    )
    print(
        f"\n\033[1mReady to publish {PACKAGE} {version}.\033[0m All of the above was local. Next:\n"
        f"  1. commit + push {ui.name} package.json, then tag it {tag} and push the tag\n"
        f"  2. commit the version bump here, tag {tag}, and push both\n"
        f"  3. the tag push publishes {version} to PyPI, which can never be undone\n"
        f"{workbench}"
        f"\nThe workflow builds the workbench from open-lease-ui at this release's tag, which is\n"
        f"why the UI tag goes first: the tag push here starts the run that resolves it.\n"
    )
    if input("Type the version to continue: ").strip() != version:
        raise ReleaseError("aborted; nothing was pushed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut an open-lease release.")
    parser.add_argument("version", help="the version to release, e.g. 0.5.0")
    parser.add_argument("--ui-repo", default=str(REPO.parent / "open-lease-ui"))
    parser.add_argument("--dry-run", action="store_true", help="preflight + verify, change nothing")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    version = args.version.lstrip("v")
    if not SEMVER.match(version):
        raise ReleaseError(f"{args.version!r} is not a MAJOR.MINOR.PATCH version")
    tag = f"v{version}"
    ui = Path(args.ui_repo).resolve()
    write = not args.dry_run

    global _UI_REPO
    _UI_REPO = ui

    step(f"Preflight for {version}")
    check_tools()
    check_pypi_free(version)
    ui_head = check_repo_ready(ui, "open-lease-ui")
    check_repo_ready(REPO, "open-lease")
    release_tag_exists = check_tag_free(REPO, tag, "open-lease")
    ui_tag_exists = check_tag_free(ui, tag, "open-lease-ui", expect_sha=ui_head)
    override = check_ui_override(tag)

    step("Version + changelog")
    bump_pyproject(version, write=write)
    close_changelog(version, write=write)
    ui_needs_commit = bump_ui_version(ui, version, write=write)

    step("Verify")
    bundle_ui(ui)
    if args.dry_run:
        print("\n\033[1mDry run.\033[0m Nothing was edited, tagged, or pushed.")
        print("  Re-run without --dry-run to cut the release (the build check needs the bump).")
        return 0
    wheel = verify_build(version)
    verify_wheel_installs(wheel, version)

    if not args.yes:
        confirm(version, tag, ui, override)

    # The UI tag first: the workflow resolves the workbench from open-lease-ui at this tag, and the
    # release tag push below is what starts the workflow.
    step("Publish")
    if not ui_tag_exists:
        tag_ui(ui, tag, version, needs_commit=ui_needs_commit)
    tag_release(version, tag, tag_exists=release_tag_exists)

    print(
        f"\n\033[1m{PACKAGE} {version} is on its way.\033[0m\n"
        f"  Watch:  gh run watch $(gh run list --workflow Publish --limit 1 --json databaseId "
        f"--jq '.[0].databaseId')\n"
        f"  The run summary records the workbench it bundled: confirm that ref says {tag}.\n"
        f"  Then:   uv pip install --no-cache '{PACKAGE}[all]=={version}' in a scratch venv\n"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReleaseError as exc:
        print(f"\n\033[31mrelease failed:\033[0m {exc}", file=sys.stderr)
        # Name every file the version bump touches, in both repos. A half-reverted release leaves a
        # dirty tree that the next attempt refuses to start from.
        print("\nRevert any local edits with:", file=sys.stderr)
        print(f"  git checkout -- {' '.join(TOUCHED)}", file=sys.stderr)
        if _UI_REPO is not None:
            print(f"  git -C {_UI_REPO} checkout -- package.json", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted; nothing was pushed", file=sys.stderr)
        sys.exit(130)
