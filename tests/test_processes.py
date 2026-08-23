import sys
from pathlib import Path

import pytest

from gsdb.processes import CommandError, run_logged


def test_logged_command_returns_timing_and_preserves_failure_metrics(tmp_path: Path) -> None:
    success_log = tmp_path / "success.log"
    metrics = run_logged([sys.executable, "-c", "print('ok')"], success_log)
    assert metrics["elapsed_seconds"] >= 0
    assert "ok" in success_log.read_text(encoding="utf-8")

    with pytest.raises(CommandError) as failure:
        run_logged([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failure.log")
    assert failure.value.returncode == 7
    assert failure.value.metrics["elapsed_seconds"] >= 0
