from pathlib import Path


def test_windows_wrapper_forwards_only_named_mimo_environment_variables() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "gsdb.ps1").read_text(encoding="utf-8")
    assert "MIMO_API_KEY" in wrapper
    assert "MIMO_BASE_URL" in wrapper
    assert "WSLENV" in wrapper
    assert "Bearer" not in wrapper
