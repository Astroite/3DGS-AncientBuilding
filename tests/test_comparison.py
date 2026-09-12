import json
from types import SimpleNamespace

import pytest

from gsdb import comparison as c
from gsdb.media import sha256_file


@pytest.fixture
def case(tmp_path, monkeypatch):
    run_dir = tmp_path / 'scene' / 'new-run'
    run_dir.mkdir(parents=True)
    (run_dir / 'selected-primary-metrics.jsonl').write_text('{"timestamp_seconds": 0}\n')
    run = SimpleNamespace(id='new-run', config=SimpleNamespace(schema_version=5,
        reconstruction=SimpleNamespace(primary='profile'), segment_qa='qa'),
        selected_dataset='new-run/reconstruction-primary',
        stages={'reconstruct': SimpleNamespace(status=SimpleNamespace(value='succeeded'))},
        metrics={'selected_attempt': 'primary'})
    monkeypatch.setattr(c, 'load_run', lambda scene, name: run)
    monkeypatch.setattr(c, 'save_run', lambda *a: None)
    def prepare(dataset, records, profile, qa, segment, output, **kwargs):
        assert dataset == run_dir / 'reconstruction-primary'
        assert records == [{'timestamp_seconds': 0}]
        output.mkdir(parents=True, exist_ok=True)
        (output / 'dataset.json').write_text('{}')
        return dict(images=[dict(split='train')], coverage_status='incomplete')
    monkeypatch.setattr(c, 'prepare_segment', prepare)
    calls = []
    def train(package, target, backend, steps, photo, **kwargs):
        calls.append((target, backend, photo, kwargs))
        target.mkdir(parents=True, exist_ok=True)
        record = dict(backend=backend, photo_comp=photo, requested_steps=steps or 30000,
                      package_sha256=sha256_file(package / 'dataset.json'), status='prepared')
        if not kwargs['dry_run']:
            (target / 'model.ply').write_bytes(b'model')
            (target / 'evaluation').mkdir(exist_ok=True)
            (target / 'evaluation' / 'metrics.json').write_text('{}')
            record.update(status='succeeded', model_sha256=sha256_file(target / 'model.ply'))
        (target / 'dispatch.json').write_text(json.dumps(record))
        return record
    monkeypatch.setattr(c, 'train_package', train)
    return run_dir, tmp_path / 'comparison', run, calls


def test_run_directory_replaces_old_case_file_and_dry_run(case):
    run_dir, output, run, calls = case
    result = c.compare_backends(run_dir, 'segment-001', output, dry_run=True)
    assert len(calls) == 4
    assert all(call[3]['dry_run'] for call in calls)
    assert result['coverage'] == 'incomplete'
    assert result['visual_acceptance'] == 'pending'
    assert not result['full_runs_authorized']
    assert not (run_dir / 'case.json').exists()
    assert (output / 'comparison.json').is_file()


def test_failed_reconstruction_cannot_train(case):
    run_dir, output, run, calls = case
    run.stages['reconstruct'].status.value = 'failed'
    with pytest.raises(RuntimeError, match='completed'):
        c.compare_backends(run_dir, 'segment-001', output)
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize('change', ['budget', 'model', 'package'])
def test_completed_result_identity_and_hash_guard(case, change):
    run_dir, output, run, calls = case
    c.compare_backends(run_dir, 'segment-001', output, backends=('gsplat',), photo_comp=(True,))
    target = output / 'gsplat-photo-on'
    if change == 'model':
        (target / 'model.ply').write_bytes(b'changed')
    if change == 'package':
        dispatch = json.loads((target / 'dispatch.json').read_text())
        dispatch['package_sha256'] = 'changed'
        (target / 'dispatch.json').write_text(json.dumps(dispatch))
    result = c.compare_backends(run_dir, 'segment-001', output, backends=('gsplat',),
                                photo_comp=(True,), steps=31000 if change == 'budget' else None)
    assert result['experiments'][0]['status'] == 'failed'
    assert len(calls) == 1


def test_native_checkpoint_resume_and_success_reuse(case):
    run_dir, output, run, calls = case
    target = output / 'gsplat-photo-off'
    target.mkdir(parents=True)
    (target / 'checkpoint.pt').write_bytes(b'checkpoint')
    c.compare_backends(run_dir, 'segment-001', output, backends=('gsplat',), photo_comp=(False,), resume=True)
    assert calls[0][3]['resume']
    result = c.compare_backends(run_dir, 'segment-001', output, backends=('gsplat',), photo_comp=(False,), resume=True)
    assert result['experiments'][0]['status'] == 'succeeded'
    assert len(calls) == 1


def test_license_block_preserves_gui_toggle_without_second_train(case, monkeypatch):
    run_dir, output, run, calls = case
    original = c.train_package
    def train(package, target, backend, steps, photo, **kwargs):
        if backend == 'postshot' and not photo:
            target.mkdir(parents=True)
            (target / 'train.log').write_text('Postshot Studio license required')
            raise RuntimeError('No trained model')
        return original(package, target, backend, steps, photo, **kwargs)
    monkeypatch.setattr(c, 'train_package', train)
    result = c.compare_backends(run_dir, 'segment-001', output, backends=('postshot',))
    assert [e['status'] for e in result['experiments']] == ['failed', 'blocked']
    assert calls[0][3]['dry_run']
