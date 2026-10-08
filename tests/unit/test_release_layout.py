"""Portable release path, Data preference and command-contract checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gsstudio.infrastructure import paths
from gsstudio.infrastructure.adapters import gsplat
from gsstudio.infrastructure.runtime import gpu_lock, layout


def test_saved_data_root_is_shared_with_cli_and_override(tmp_path: Path, monkeypatch):
    data = tmp_path / "Data 中文"
    other = tmp_path / "Other"
    data.mkdir()
    other.mkdir()
    monkeypatch.setenv("GSSTUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.delenv("GSSTUDIO_DATA_ROOT", raising=False)
    assert paths.save_data_root(data) == data
    assert paths.find_data_root() == data
    assert json.loads((tmp_path / "settings" / "settings.json").read_text(encoding="utf-8"))["data_root"] == str(data)
    monkeypatch.setenv("GSSTUDIO_DATA_ROOT", str(other))
    assert paths.find_data_root() == other


def test_frozen_nested_executable_resolves_release_root(tmp_path: Path, monkeypatch):
    release = tmp_path / "GSStudio"
    worker = release / "worker"
    worker.mkdir(parents=True)
    (release / "release-manifest.json").write_text("{}", encoding="utf-8")
    executable = worker / "GSStudioWorker.exe"
    executable.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    assert paths.find_app_root() == release


def test_frozen_commands_use_bundled_executables(tmp_path: Path, monkeypatch):
    worker = tmp_path / "worker" / "GSStudioWorker.exe"
    trainer = tmp_path / "gpu-runtime" / "GSStudioTrainer.exe"
    worker.parent.mkdir()
    trainer.parent.mkdir()
    worker.touch()
    trainer.touch()
    monkeypatch.setattr(layout, "is_frozen", lambda: True)
    monkeypatch.setattr(layout, "find_app_root", lambda: tmp_path)
    monkeypatch.setattr(gsplat, "is_frozen", lambda: True)
    assert layout.worker_command() == [str(worker)]
    assert gsplat.native_training_command("--dataset", "D:/Data") == [
        str(trainer), "--dataset", "D:/Data"]


def test_gpu_lock_defaults_to_writable_user_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GSSTUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.delenv("GSSTUDIO_GPU_LOCK_PATH", raising=False)
    assert gpu_lock.gpu_lock_path() == tmp_path / "settings" / "gpu.lock"
    override = tmp_path / "shared.lock"
    monkeypatch.setenv("GSSTUDIO_GPU_LOCK_PATH", str(override))
    assert gpu_lock.gpu_lock_path() == override
    monkeypatch.setenv("GSSTUDIO_GPU_LOCK_PATH", "relative.lock")
    with pytest.raises(ValueError, match="absolute path"):
        gpu_lock.gpu_lock_path()


def test_release_command_missing_component_fails_explicitly(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(layout, "is_frozen", lambda: True)
    monkeypatch.setattr(layout, "find_app_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="worker is missing"):
        layout.worker_command()
