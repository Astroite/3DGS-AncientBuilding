import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare-postshot-experiment.py'
    spec = importlib.util.spec_from_file_location('postshot_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exception_requires_explicit_flag(experiment, monkeypatch):
    monkeypatch.setattr('sys.argv', ['test', '--run-dir', '.', '--attempt', 'fallback',
                                   '--label', 'trial', '--reason', 'test'])
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2


@pytest.mark.parametrize('fail', [False, True])
def test_experiment_audits_without_rewriting_source(experiment, monkeypatch, tmp_path, fail):
    work = tmp_path / 'scene' / 'run1'
    source = work / 'reconstruction-fallback'
    source.mkdir(parents=True)
    for p in [work / 'manifest.yaml', *[source / n for n in
              ('trajectory-qa.json', 'mask-filter.json', 'mask-final.json')]]:
        p.write_text('original')
    run = SimpleNamespace(id='run1', config=SimpleNamespace(schema_version=4),
                          stages={'reconstruct': SimpleNamespace(status='failed')})
    monkeypatch.setattr(experiment, 'load_model', lambda *a: SimpleNamespace(model_copy=lambda **k: run))
    original_validator = experiment.postshot.validate_trajectory_qa

    def prepare(scene, private_run, *, output, resume):
        assert experiment.postshot.validate_trajectory_qa() is None
        if fail:
            raise RuntimeError('integrity failure')
        output.mkdir()
        return {'output_path': str(output), 'counts': {'images': 1}}

    monkeypatch.setattr(experiment.postshot, 'prepare_postshot_dataset', prepare)
    monkeypatch.setattr('sys.argv', ['test', '--run-dir', str(work), '--attempt', 'fallback',
                                   '--label', 'trial', '--allow-failed-trajectory', '--reason', 'user test'])
    if fail:
        with pytest.raises(RuntimeError, match='integrity failure'):
            experiment.main()
    else:
        experiment.main()
    audit = json.loads(next((work / 'postshot-experiments' / 'trial').glob('*-audit.json')).read_text())
    assert audit['status'] == ('failed' if fail else 'prepared')
    assert audit['source_records_unchanged'] is True
    assert (work / 'manifest.yaml').read_text() == 'original'
    assert experiment.postshot.validate_trajectory_qa is original_validator
