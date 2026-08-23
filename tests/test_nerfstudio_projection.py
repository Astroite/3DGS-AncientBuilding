from pathlib import Path

import pytest


def test_reproducible_patch_contains_range_clamp() -> None:
    patch = (
        Path(__file__).resolve().parents[1]
        / "patches"
        / "nerfstudio-equirect-clamp.patch"
    ).read_text(encoding="utf-8")
    assert ".clamp(0.0, 1.0)" in patch


def test_installed_nerfstudio_projection_range() -> None:
    torch = pytest.importorskip("torch")
    equirect = pytest.importorskip("nerfstudio.process_data.equirect_utils")
    image = torch.zeros((1, 3, 16, 32), dtype=torch.float32)
    image[:, :, :, 16:] = 1.0
    result = equirect.equirect2persp(image, 120, 0, 0, 16, 16)
    assert result.shape == (1, 3, 16, 16)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
