import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_windows_wrapper_has_no_wsl_tunnel() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "gsdb.ps1").read_text(encoding="utf-8")
    assert "wsl.exe" not in wrapper
    assert "WSLENV" not in wrapper
    assert ".venv" in wrapper


def test_windows_wrapper_runs_gsdb_from_the_venv() -> None:
    if os.name != "nt":
        pytest.skip("native Windows wrapper execution requires a Windows Python host")
    root = Path(__file__).resolve().parents[1]
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        pytest.skip("native venv is not bootstrapped; run scripts/bootstrap-windows.ps1")
    powershell = shutil.which("pwsh") or shutil.which("pwsh.exe") or shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    completed = subprocess.run(
        [powershell, "-NoProfile", "-File", str(root / "gsdb.ps1"), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0
    assert "gsdb" in completed.stdout.lower()
