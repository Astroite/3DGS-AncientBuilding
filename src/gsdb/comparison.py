"""Backend comparisons on a directly selected, QA-approved Run segment."""
from pathlib import Path
import json
import math

from .media import sha256_file
from .runs import load_run, save_run
from .training import train_package
from .training_data import prepare_segment, json_write


def compare_backends(run_dir: Path, segment: str, output: Path, *,
                     backends=("postshot", "gsplat"), photo_comp=(False, True),
                     steps=None, duration_seconds=None, resume=False, dry_run=False):
    if not backends or len(set(backends)) != len(backends) or any(b not in {"postshot", "gsplat"} for b in backends):
        raise ValueError("Choose distinct backends: postshot, gsplat")
    if not photo_comp or any(type(p) is not bool for p in photo_comp) or len(set(photo_comp)) != len(photo_comp):
        raise ValueError("Choose distinct boolean compensation settings")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer")
    if duration_seconds is not None and (not math.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError("duration_seconds must be finite and positive")
    if not segment or any(c in segment for c in '/\\:') or segment in {'.', '..'}:
        raise ValueError('Invalid segment identifier')
    run_dir, output = run_dir.resolve(), output.resolve()
    scene = run_dir.parent
    run = load_run(scene, run_dir.name)
    if run.id != run_dir.name or run.config.schema_version != 5 or not run.selected_dataset or run.stages['reconstruct'].status.value != 'succeeded':
        raise RuntimeError('Requires a completed schema 5 reconstruction with segment QA')
    label = run.metrics['selected_attempt']
    records = [json.loads(line) for line in (run_dir / f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
    package_name = segment if duration_seconds is None else f'{segment}-first-{duration_seconds:g}s'
    package = run_dir / 'training-data' / package_name
    meta = prepare_segment(scene / run.selected_dataset, records, run.config.reconstruction.primary,
                           run.config.segment_qa, segment, package, duration_seconds=duration_seconds)
    output.mkdir(parents=True, exist_ok=True)
    expected_steps = max(30000, 30 * sum(r['split'] == 'train' for r in meta['images'])) if steps is None else steps
    package_hash = sha256_file(package / 'dataset.json')
    result = dict(run_dir=str(run_dir), package=str(package), segment=segment,
                  coverage=meta['coverage_status'], experiments=[], visual_acceptance='pending', full_runs_authorized=False)
    license_blocked = False
    for backend in backends:
        for photo in photo_comp:
            target = output / f'{backend}-photo-{"on" if photo else "off"}'
            entry = dict(backend=backend, photo_comp=photo, segment=segment, output=str(target))
            try:
                previous = json.loads((target / 'dispatch.json').read_text(encoding='utf-8')) if (target / 'dispatch.json').is_file() else {}
                if previous.get('status') == 'succeeded':
                    expected = dict(backend=backend, photo_comp=photo, requested_steps=expected_steps, package_sha256=package_hash)
                    if any(previous.get(k) != v for k, v in expected.items()):
                        raise RuntimeError('Completed comparison identity changed')
                    if sha256_file(target / 'model.ply') != previous.get('model_sha256') or not (target / 'evaluation' / 'metrics.json').is_file():
                        raise RuntimeError('Completed comparison artifacts changed or incomplete')
                    entry.update(previous)
                else:
                    blocked = backend == 'postshot' and license_blocked
                    entry.update(train_package(package, target, backend, steps, photo,
                        resume=resume and backend == 'gsplat' and (target / 'checkpoint.pt').exists(),
                        dry_run=dry_run or blocked))
                    if blocked:
                        entry.update(status='blocked', reason='Postshot Studio CLI license unavailable; GUI input prepared')
            except Exception as error:
                entry.update(status='failed', error=str(error))
                log = target / 'train.log'
                if backend == 'postshot' and log.is_file() and 'Studio license required' in log.read_text(encoding='utf-8', errors='replace'):
                    license_blocked = True
            result['experiments'].append(entry)
            run.metrics.setdefault('training_experiments', []).append(entry)
            save_run(scene, run)
            json_write(output / 'comparison.json', result)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
    return result
