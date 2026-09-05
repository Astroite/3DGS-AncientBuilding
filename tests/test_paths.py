from pathlib import Path

import pytest

from gsdb.paths import (
    DATA_ROOT_ENV,
    ensure_run_dir,
    ensure_within,
    find_app_root,
    find_data_root,
    windows_to_wsl,
    wsl_to_windows,
)


def test_windows_wsl_round_trip() -> None:
    windows = r"D:\Project\3DGS\Data\site-001"
    wsl = windows_to_wsl(windows)
    assert wsl == "/mnt/d/Project/3DGS/Data/site-001"
    assert wsl_to_windows(wsl) == windows


def test_ensure_within_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert ensure_within(root / "child", root) == (root / "child").resolve()
    with pytest.raises(ValueError):
        ensure_within(root / ".." / "outside", root)


def _scene(tmp_path: Path) -> Path:
    scene = tmp_path / "site-001" / "scene-a"
    scene.mkdir(parents=True)
    return scene


def test_ensure_run_dir_creates_and_guards_against_collision(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    run = ensure_run_dir(scene, "run-001")
    assert run == scene / "run-001"
    assert run.is_dir()
    assert ensure_run_dir(scene, "run-001") == run
    with pytest.raises(FileExistsError):
        ensure_run_dir(scene, "run-001", exist_ok=False)


def test_find_app_root_walks_up_to_pyproject_toml(tmp_path: Path) -> None:
    app_root = tmp_path / "APP"
    (app_root / "src" / "gsdb").mkdir(parents=True)
    (app_root / "pyproject.toml").write_text("", encoding="utf-8")
    nested = app_root / "src" / "gsdb"
    assert find_app_root(nested) == app_root


def test_find_data_root_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configured = tmp_path / "elsewhere" / "Data"
    configured.mkdir(parents=True)
    monkeypatch.setenv(DATA_ROOT_ENV, str(configured))
    assert find_data_root(tmp_path) == configured


def test_find_data_root_defaults_to_data_sibling_of_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    app_root = tmp_path / "APP"
    app_root.mkdir()
    (app_root / "pyproject.toml").write_text("", encoding="utf-8")
    data_root = tmp_path / "Data"
    data_root.mkdir()
    assert find_data_root(app_root) == data_root


def test_find_data_root_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    app_root = tmp_path / "APP"
    app_root.mkdir()
    (app_root / "pyproject.toml").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="data root is missing"):
        find_data_root(app_root)
