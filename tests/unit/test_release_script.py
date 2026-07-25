"""The release script's file surgery (scripts/release.py).

Not package code, but load-bearing: it rewrites pyproject.toml and CHANGELOG.md and then pushes
tags, so a bug here surfaces mid-release with a version already tagged. The text manipulation and
the guards that refuse a bad release are worth pinning down here instead of discovering by hand.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "release_script", Path(__file__).resolve().parents[2] / "scripts" / "release.py"
)
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


CHANGELOG = """\
# Changelog

Blurb.

## [Unreleased]

### Added
- A thing worth releasing.

## [0.4.0] - 2026-07-25

### Added
- The previous thing.

[Unreleased]: https://github.com/mfbaig35r/open-lease/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/mfbaig35r/open-lease/compare/v0.3.0...v0.4.0
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "open-lease"\nversion = "0.4.0"\n')
    monkeypatch.setattr(release, "REPO", tmp_path)
    return tmp_path


def test_bump_pyproject(repo):
    assert release.bump_pyproject("0.5.0", write=True) == "0.4.0"
    assert 'version = "0.5.0"' in (repo / "pyproject.toml").read_text()


def test_bump_pyproject_dry_run_leaves_the_file_alone(repo):
    release.bump_pyproject("0.5.0", write=False)
    assert 'version = "0.4.0"' in (repo / "pyproject.toml").read_text()


def test_close_changelog(repo):
    release.close_changelog("0.5.0", write=True)
    text = (repo / "CHANGELOG.md").read_text()

    # A fresh Unreleased stays on top, and the notes move under the new version.
    assert "## [Unreleased]\n\n## [0.5.0] - " in text
    assert text.index("## [0.5.0]") < text.index("- A thing worth releasing.")
    assert text.index("- A thing worth releasing.") < text.index("## [0.4.0]")

    # Compare links: Unreleased moves forward and the new version gets its own.
    assert "[Unreleased]: https://github.com/mfbaig35r/open-lease/compare/v0.5.0...HEAD" in text
    assert "[0.5.0]: https://github.com/mfbaig35r/open-lease/compare/v0.4.0...v0.5.0" in text
    assert "[0.4.0]: https://github.com/mfbaig35r/open-lease/compare/v0.3.0...v0.4.0" in text


def test_close_changelog_is_idempotent(repo):
    release.close_changelog("0.5.0", write=True)
    once = (repo / "CHANGELOG.md").read_text()
    release.close_changelog("0.5.0", write=True)  # a re-run after a failed publish must not double
    assert (repo / "CHANGELOG.md").read_text() == once


def test_close_changelog_refuses_an_empty_unreleased(repo):
    path = repo / "CHANGELOG.md"
    path.write_text(CHANGELOG.replace("### Added\n- A thing worth releasing.\n", ""))
    with pytest.raises(release.ReleaseError, match="empty"):
        release.close_changelog("0.5.0", write=True)


def test_close_changelog_refuses_a_missing_unreleased(repo):
    path = repo / "CHANGELOG.md"
    path.write_text(CHANGELOG.replace("## [Unreleased]\n", ""))
    with pytest.raises(release.ReleaseError, match="Unreleased"):
        release.close_changelog("0.5.0", write=True)


def test_close_changelog_refuses_a_broken_compare_link(repo):
    """The link block is edited by hand between releases, so a mismatch means stop, not guess."""
    path = repo / "CHANGELOG.md"
    path.write_text(CHANGELOG.replace("compare/v0.4.0...HEAD", "compare/v0.1.0...HEAD"))
    with pytest.raises(release.ReleaseError, match="compare link"):
        release.close_changelog("0.5.0", write=True)


def test_bump_ui_version(tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "package.json").write_text('{\n  "name": "open-lease-ui",\n  "version": "0.4.0",\n}\n')
    assert release.bump_ui_version(ui, "0.5.0", write=True) is True
    assert '"version": "0.5.0"' in (ui / "package.json").read_text()
    # Already in lockstep: nothing to commit in the UI repo.
    assert release.bump_ui_version(ui, "0.5.0", write=True) is False


def test_changelog_date_is_utc(repo):
    """The tag, the PyPI upload, and the workflow run are stamped in UTC. A release cut late in the
    evening local time was getting dated the previous day, out of step with all three."""
    release.close_changelog("0.5.0", write=True)
    expected = datetime.now(tz=UTC).date().isoformat()
    assert f"## [0.5.0] - {expected}" in (repo / "CHANGELOG.md").read_text()


def test_touched_files_include_the_lockfile():
    """uv.lock records this package's own version, so `uv build` rewrites it during verification. It
    has to be in the release commit, or the release ends dirty and blocks the next one."""
    assert "uv.lock" in release.TOUCHED
    assert "pyproject.toml" in release.TOUCHED
    assert "CHANGELOG.md" in release.TOUCHED


@pytest.mark.parametrize("bad", ["0.5", "v0.5.0.1", "1.0.0-rc1", "latest", ""])
def test_semver_guard_rejects(bad):
    assert release.SEMVER.match(bad) is None


@pytest.mark.parametrize("good", ["0.5.0", "1.0.0", "10.20.30"])
def test_semver_guard_accepts(good):
    assert release.SEMVER.match(good) is not None
