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


@pytest.mark.skipif(os.name != 'nt', reason='Windows session initialization')
def test_session_preserves_environment_and_current_directory(tmp_path):
    import json
    root = Path(__file__).resolve().parents[1]
    powershell = shutil.which('pwsh') or shutil.which('powershell')
    if not powershell:
        pytest.skip('PowerShell unavailable')
    environment = os.environ.copy()
    environment.update(GSDB_DATA_ROOT=str(tmp_path), GSDB_MEDIA_HELPER='configured-helper.exe',
                       INSTA360_MEDIA_SDK_ROOT='configured-sdk', GSDB_GSPLAT_PYTHON='configured-python.exe')
    script = "$Before = (Get-Location).Path; . '" + str(root / 'scripts' / 'session.ps1').replace("'", "''") + "'; Invoke-Gsdb --help | Out-Null; @{before=$Before;after=(Get-Location).Path;data=$env:GSDB_DATA_ROOT;helper=$env:GSDB_MEDIA_HELPER;sdk=$env:INSTA360_MEDIA_SDK_ROOT;trainer=$env:GSDB_GSPLAT_PYTHON} | ConvertTo-Json"
    result = subprocess.run([powershell, '-NoProfile', '-Command', script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state['before'] == state['after'] == str(tmp_path)
    assert state['data'] == str(tmp_path)
    assert state['helper'] == 'configured-helper.exe'
    assert state['sdk'] == 'configured-sdk'
    assert state['trainer'] == 'configured-python.exe'
    wrapper = root / 'gsdb.ps1'
    result = subprocess.run([powershell, '-NoProfile', '-File', str(wrapper), '--help'],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([powershell, '-NoProfile', '-File', str(wrapper), 'unknown-command'],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode != 0
