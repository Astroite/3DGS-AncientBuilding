import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from gsdb import doctor as d


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    python = tmp_path / 'native-python.exe'
    python.write_bytes(b'fixture')
    xml = tmp_path / 'export.xml'
    xml.write_text('<export/>')
    monkeypatch.setattr(d, '_version', lambda _: (True, 'available'))
    monkeypatch.setattr(d, '_torch_cuda_check', lambda: (True, 'main CUDA'))
    monkeypatch.setattr(d, '_person_segmenter_check', lambda: (True, 'Mask R-CNN'))
    monkeypatch.setattr(d, '_training_python', lambda: python)
    monkeypatch.setattr(d, '_realityscan_resolvers', lambda: {'realityscan': lambda: python, 'realityscan_export': lambda: xml})
    monkeypatch.setattr(d, '_gsplat_runtime_check', lambda p: (True, 'independent CUDA'))
    from gsdb import sources, postshot
    monkeypatch.setattr(sources, 'media_capabilities', lambda _: dict(helper_version='test', sdk_version='test'))
    monkeypatch.setattr(postshot, 'postshot_version', lambda: ((1, 1, 69), '1.1.69'))
    return tmp_path


@pytest.mark.parametrize('backend', ['postshot', 'gsplat', 'all'])
def test_backend_selection_does_not_require_legacy_colmap(runtime, monkeypatch, backend):
    monkeypatch.setattr(d, '_colmap_cuda_build', lambda: pytest.fail('Unexpected legacy check'))
    result = d.run_doctor(runtime, minimum_free_gib=0, backend=backend)
    assert result['ok']
    assert ('gsplat_cuda' in result) == (backend in {'gsplat', 'all'})
    assert 'wsl_memory' not in result
    assert 'license not verified' in result['postshot'][1]


def test_missing_backend_and_optional_requirements(runtime, monkeypatch):
    from gsdb import postshot, sources
    def unavailable(*args):
        raise RuntimeError('missing')
    monkeypatch.setattr(postshot, 'postshot_version', unavailable)
    monkeypatch.setattr(sources, 'media_capabilities', unavailable)
    assert d.run_doctor(runtime, 0, backend='gsplat')['ok']
    assert not d.run_doctor(runtime, 0, backend='postshot')['ok']
    assert not d.run_doctor(runtime, 0, {'postshot'}, backend='gsplat')['ok']
    assert not d.run_doctor(runtime, 0, {'mediasdk'}, backend='gsplat')['ok']
    monkeypatch.setattr(d, '_colmap_cuda_build', lambda: (False, 'missing'))
    assert not d.run_doctor(runtime, 0, {'colmap'}, backend='gsplat')['ok']


def test_native_failure_and_export_configuration_block(runtime, monkeypatch):
    monkeypatch.setattr(d, '_gsplat_runtime_check', lambda p: (False, 'CUDA unavailable'))
    assert not d.run_doctor(runtime, 0, backend='gsplat')['ok']
    (runtime / 'export.xml').write_text('invalid XML')
    assert not d.run_doctor(runtime, 0)['realityscan_export'][0]


def test_native_check_uses_selected_interpreter_and_reports_failure(tmp_path, monkeypatch):
    python = tmp_path / 'native.exe'
    assert not d._gsplat_runtime_check(python)[0]
    python.touch()
    def failed(command, **kwargs):
        assert command[:2] == [str(python), '-c']
        assert 'gsplat.rasterization' in command[2]
        assert kwargs['timeout'] == 120
        return SimpleNamespace(returncode=1, stdout='', stderr='CUDA unavailable')
    monkeypatch.setattr(d.subprocess, 'run', failed)
    assert d._gsplat_runtime_check(python) == (False, 'CUDA unavailable')
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired('native', 120)
    monkeypatch.setattr(d.subprocess, 'run', timeout)
    assert not d._gsplat_runtime_check(python)[0]


def test_doctor_cli_forwards_backend_and_validates_unknown_options(tmp_path, monkeypatch):
    from gsdb import cli
    monkeypatch.setenv('GSDB_DATA_ROOT', str(tmp_path))
    def check(root, **kwargs):
        assert kwargs['backend'] == 'all'
        assert kwargs['require'] == {'mediasdk', 'colmap'}
        return {'ok': True, 'postshot': (True, 'version only')}
    monkeypatch.setattr(cli, 'run_doctor', check)
    assert CliRunner().invoke(cli.app, ['doctor', '--backend', 'all', '--require', 'mediasdk', '--require', 'colmap']).exit_code == 0
    with pytest.raises(ValueError, match='backend'):
        d.run_doctor(tmp_path, backend='unknown')
    with pytest.raises(ValueError, match='Unknown'):
        d.run_doctor(tmp_path, require={'unknown'})
