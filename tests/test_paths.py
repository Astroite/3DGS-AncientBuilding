import os
from pathlib import Path

import pytest

from gsdb import paths
from gsdb.paths import (
    SCRATCH_ROOT_ENV,
    ensure_within,
    ensure_work_dir,
    windows_to_wsl,
    wsl_to_windows,
)


def test_windows_wsl_round_trip() -> None:
    windows = r"D:\Project\3DGS\locations\site-001"
    wsl = windows_to_wsl(windows)
    assert wsl == "/mnt/d/Project/3DGS/locations/site-001"
    assert wsl_to_windows(wsl) == windows


def test_ensure_within_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert ensure_within(root / "child", root) == (root / "child").resolve()
    with pytest.raises(ValueError):
        ensure_within(root / ".." / "outside", root)


def _symlinks_available(tmp_path: Path) -> bool:
    try:
        (tmp_path / "probe-target").mkdir()
        os.symlink(tmp_path / "probe-target", tmp_path / "probe-link", target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    return True


def _scene(tmp_path: Path) -> Path:
    scene = tmp_path / "locations" / "site-001" / "scenes" / "scene-a"
    scene.mkdir(parents=True)
    return scene


def test_ensure_work_dir_stays_in_project_without_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SCRATCH_ROOT_ENV, raising=False)
    scene = _scene(tmp_path)
    work = ensure_work_dir(scene, "run-001")
    assert work == scene / "work" / "run-001"
    assert work.is_dir() and not work.is_symlink()
    assert ensure_work_dir(scene, "run-001") == work
    with pytest.raises(FileExistsError):
        ensure_work_dir(scene, "run-001", exist_ok=False)


def test_ensure_work_dir_redirects_bytes_to_the_scratch_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _symlinks_available(tmp_path):
        pytest.skip("This filesystem or account cannot create directory symlinks")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv(SCRATCH_ROOT_ENV, str(scratch))
    monkeypatch.setattr(paths, "windows_host", lambda: False)
    scene = _scene(tmp_path)
    work = ensure_work_dir(scene, "run-001", exist_ok=False)
    target = scratch / "site-001" / "scene-a" / "run-001"

    # The logical path still lives in the scene, so manifests keep scene-relative paths.
    assert work == scene / "work" / "run-001"
    assert work.is_symlink() and target.is_dir()
    assert ensure_within(work / "logs" / "mask.log", scene) == work / "logs" / "mask.log"
    (work / "images").mkdir()
    assert (target / "images").is_dir()

    # A second run of the same ID must not be silently reused or repointed.
    with pytest.raises(FileExistsError):
        ensure_work_dir(scene, "run-001", exist_ok=False)
    assert ensure_work_dir(scene, "run-001") == work
    monkeypatch.setenv(SCRATCH_ROOT_ENV, str(tmp_path / "other-scratch"))
    with pytest.raises(RuntimeError, match="refusing to repoint"):
        ensure_work_dir(scene, "run-001")


def test_ensure_work_dir_keeps_runs_that_predate_the_scratch_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _scene(tmp_path)
    existing = scene / "work" / "run-001"
    (existing / "images").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv(SCRATCH_ROOT_ENV, str(scratch))
    monkeypatch.setattr(paths, "windows_host", lambda: False)

    # An in-flight run must keep resuming against the data it already wrote.
    assert ensure_work_dir(scene, "run-001") == existing
    assert not existing.is_symlink()
    assert not (scratch / "site-001" / "scene-a" / "run-001").exists()

    # Two populated copies of the same run ID are ambiguous and must not be guessed at.
    conflicting = scratch / "site-001" / "scene-a" / "run-001" / "images"
    conflicting.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="keep one and remove the other"):
        ensure_work_dir(scene, "run-001")
