from pathlib import Path

import pytest

from gsdb.paths import ensure_within, windows_to_wsl, wsl_to_windows


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
