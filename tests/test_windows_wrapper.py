from pathlib import Path


def test_windows_wrapper_forwards_only_named_deepseek_environment_variable() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "gsdb.ps1").read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY" in wrapper
    assert "MIMO_API_KEY" not in wrapper
    assert "MIMO_BASE_URL" not in wrapper
    assert "WSLENV" in wrapper
    assert "Bearer" not in wrapper
