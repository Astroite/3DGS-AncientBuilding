from pathlib import Path

import pytest


@pytest.mark.parametrize('kind', ['projection', 'nested_paths'])
def test_installer_patch_is_idempotent_and_rejects_unknown_source(tmp_path, monkeypatch, kind):
    import importlib.util
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / 'scripts' / 'apply_nerfstudio_patch.py'
    spec = importlib.util.spec_from_file_location('installer_patch', path)
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    source = tmp_path / 'dependency.py'
    monkeypatch.setattr(patcher.importlib.util, 'find_spec', lambda _: SimpleNamespace(origin=str(source)))
    if kind == 'projection':
        before = b'return torch.nn.functional.grid_sample(img, grid, mode="bicubic", padding_mode="zeros")'
        after = before + b'.clamp(0.0, 1.0)'
        needle, replacement = patcher.EQUIRECT_NEEDLE, patcher.EQUIRECT_REPLACEMENT
    else:
        before = b'            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name'
        after = b'            relative = Path(*filepath.parts[1:])\n            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / relative'
        needle, replacement = patcher.NESTED_DOWNSCALE_NEEDLE, patcher.NESTED_DOWNSCALE_REPLACEMENT
    source.write_bytes(before)
    patcher.patch_module('fixture', needle, replacement, kind)
    assert source.read_bytes() == after
    patcher.patch_module('fixture', needle, replacement, kind)
    assert source.read_bytes() == after
    source.write_bytes(b'unexpected upstream source')
    with pytest.raises(SystemExit, match='unexpected'):
        patcher.patch_module('fixture', needle, replacement, kind)
    assert source.read_bytes() == b'unexpected upstream source'


def test_installed_nerfstudio_projection_range() -> None:
    torch = pytest.importorskip("torch")
    equirect = pytest.importorskip("nerfstudio.process_data.equirect_utils")
    image = torch.zeros((1, 3, 16, 32), dtype=torch.float32)
    image[:, :, :, 16:] = 1.0
    result = equirect.equirect2persp(image, 120, 0, 0, 16, 16)
    assert result.shape == (1, 3, 16, 16)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
