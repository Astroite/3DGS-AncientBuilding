"""Backend dispatch; the dataset and validation cameras are backend independent."""
from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
from gsstudio.infrastructure.adapters.gsplat import native_training_command
from gsstudio.infrastructure.runtime.processes import run_logged
from gsstudio.pipeline.training.data import json_write, postshot_adapter, validate_package, validate_postshot_adapter


def _verify_attempt_history(output: Path, history: object) -> list[dict]:
    if not isinstance(history, list):
        raise RuntimeError('Experiment attempt history is invalid')
    for attempt in history:
        if not isinstance(attempt, dict):
            raise RuntimeError('Experiment attempt history is invalid')
        for kind in ('dispatch', 'log', 'training', 'failure'):
            evidence = attempt.get(kind)
            if evidence is None:
                continue
            if not isinstance(evidence, dict) or not isinstance(evidence.get('path'), str):
                raise RuntimeError('Experiment attempt evidence is invalid')
            name = evidence['path']
            if Path(name).name != name or name in {'.', '..'}:
                raise RuntimeError('Experiment attempt evidence path is unsafe')
            path = output / name
            if (not path.is_file() or path.stat().st_size != evidence.get('bytes') or
                    sha256_file(path) != evidence.get('sha256')):
                raise RuntimeError(f'Experiment attempt evidence changed: {name}')
    return history


def _archive_attempt(output: Path, previous: dict) -> list[dict]:
    history = list(_verify_attempt_history(output, previous.get('attempt_history', [])))
    index = 1
    while any((output / f'{name}-attempt-{index:03d}.{suffix}').exists()
              for name, suffix in (('dispatch', 'json'), ('train', 'log'),
                                   ('training', 'json'), ('failure', 'json'))):
        index += 1
    entry: dict = {'number': index, 'status': previous.get('status')}
    for kind, source, name in (
        ('dispatch', output / 'dispatch.json', f'dispatch-attempt-{index:03d}.json'),
        ('log', output / 'train.log', f'train-attempt-{index:03d}.log'),
        ('training', output / 'training.json', f'training-attempt-{index:03d}.json'),
        ('failure', output / 'failure.json', f'failure-attempt-{index:03d}.json'),
    ):
        if not source.is_file():
            continue
        destination = output / name
        temporary = output / f'.{name}.{uuid.uuid4().hex}.tmp'
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        entry[kind] = {'path': name, 'bytes': destination.stat().st_size,
                       'sha256': sha256_file(destination)}
    history.append(entry)
    return history


def verify_evaluation(output: Path, meta: dict) -> str:
    path = output / 'evaluation' / 'metrics.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    expected = sorted(row['image'] for row in meta['images'] if row['split'] == 'validation')
    rows = report.get('images') if isinstance(report, dict) else None
    if (not isinstance(report, dict) or report.get('schema_version') != 1 or
            not isinstance(rows, list) or len(rows) != len(expected) or
            any(not isinstance(row, dict) or not isinstance(row.get('image'), str) for row in rows) or
            sorted(row['image'] for row in rows) != expected):
        raise RuntimeError('Training evaluation evidence is incomplete or mismatched')
    return sha256_file(path)


def train_package(package: Path, output: Path, backend='postshot', steps=None,
                  photo_comp=True, resume=False, dry_run=False,
                  use_bilateral_grid=True, use_sparse_depth=True,
                  preview_every=0):
    from gsstudio.infrastructure.adapters.postshot import build_postshot_train_command, postshot_executable
    if backend not in ('gsplat','postshot'):
        raise ValueError('backend must be gsplat or postshot')
    package,output = package.resolve(),output.resolve()
    if package == output or package in output.parents:
        raise ValueError('Training outputs must be outside the immutable package')
    meta = validate_package(package)
    count = sum(r['split']=='train' for r in meta['images'])
    steps = max(30000,30*count) if steps is None else steps
    if steps<1:
        raise ValueError('steps must be positive')
    from gsstudio.infrastructure.paths import find_app_root
    app = find_app_root()
    previous = None
    if output.exists():
        if not (output/'dispatch.json').is_file():
            raise FileExistsError(f'Unrecognized experiment directory: {output}')
        previous = json.loads((output/'dispatch.json').read_text(encoding='utf-8'))
        if not resume and previous['status'] != 'prepared':
            raise FileExistsError(f'Experiment already executed: {output}')
    if resume and previous is None:
        raise FileNotFoundError(f'No experiment to resume: {output}')
    output.mkdir(parents=True,exist_ok=True)
    restart_from_step_zero = False
    if resume and backend == 'gsplat' and previous:
        if previous.get('status') == 'succeeded':
            raise RuntimeError('A completed experiment cannot be resumed')
        if not (output/'checkpoint.pt').is_file():
            summary_path = output/'training.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
            if (not isinstance(summary, dict) or summary.get('status') == 'succeeded' or
                    (summary.get('status') == 'stopped' and isinstance(summary.get('steps'), int)
                     and summary['steps'] > 0) or
                    (isinstance(previous.get('completed_steps'), int) and
                     previous['completed_steps'] > 0) or (output/'model.ply').exists()):
                raise RuntimeError('Checkpoint is missing after training produced results; inspect experiment evidence')
            restart_from_step_zero = (
                previous.get('status') in {'failed', 'prepared'} or
                (previous.get('status') == 'stopped' and
                 previous.get('stop_reason') == 'gpu_wait_cancelled')
            )
            if not restart_from_step_zero:
                raise RuntimeError('No recoverable checkpoint or verified step-zero restart')
    if backend=='gsplat':
        command = native_training_command('--dataset', str(package), '--output', str(output),
                                          '--steps', str(steps))
        if not photo_comp:
            command.append('--no-photo-comp')
        if not use_bilateral_grid:
            command.append('--no-use-bilateral-grid')
        if not use_sparse_depth:
            command.append('--no-use-sparse-depth')
        if restart_from_step_zero:
            command.append('--restart-from-step-zero')
        elif resume:
            command.append('--resume')
        if preview_every:
            command += ['--preview-every', str(preview_every)]
    else:
        if resume:
            raise ValueError('Postshot continuation uses its saved project; no implicit retraining')
        adapter = output/'postshot-input'
        if not adapter.exists():
            postshot_adapter(package,adapter)
        else:
            validate_postshot_adapter(package,adapter,meta)
        command = build_postshot_train_command(postshot_executable(),adapter,output/'model.psht',
            profile='Splat ADC',ksteps=math.ceil(steps/1000),export_ply=output/'model.ply')
        command += ['--photo-comp','true' if photo_comp else 'false','--max-sh-degree','3','--anti-aliasing','false','--no-recenter-points']
    record = dict(backend=backend,package_sha256=sha256_file(package/'dataset.json'),
                  command=command,requested_steps=steps,photo_comp=photo_comp,status='prepared',
                  use_bilateral_grid=bool(use_bilateral_grid) if backend=='gsplat' else None,
                  use_sparse_depth=bool(use_sparse_depth) if backend=='gsplat' else None)
    record['preview_every'] = preview_every if backend == 'gsplat' else None
    if restart_from_step_zero:
        record['restart_from_step_zero'] = True
    if previous:
        for key in ('backend','package_sha256','requested_steps','photo_comp',
                    'use_bilateral_grid','use_sparse_depth'):
            if previous.get(key) != record[key]:
                raise RuntimeError(f'Experiment identity changed: {key}')
    if dry_run:
        if not previous:
            json_write(output/'dispatch.json',record)
        return record
    if resume and previous:
        record['attempt_history'] = _archive_attempt(output, previous)
    json_write(output/'dispatch.json',record)
    try:
        if backend == 'postshot':
            with gpu_session():
                resources = run_logged(command,output/'train.log',cwd=app,monitor_gpu=True,gpu_index=0)
        else:
            # Native trainer owns the cross-process GPU lock, including direct launches.
            resources = run_logged(command,output/'train.log',cwd=app,monitor_gpu=True,gpu_index=0)
        if backend=='postshot':
            log = (output/'train.log').read_text(encoding='utf-8')
            if 'Studio license required' in log or not (output/'model.psht').is_file() or not (output/'model.ply').is_file():
                raise RuntimeError('Postshot did not produce trained project and PLY; inspect license/log. GUI-ready input is preserved.')
            evaluate = native_training_command('--dataset', str(package), '--output',
                                               str(output/'evaluation'), '--evaluate-ply',
                                               str(output/'model.ply'))
            run_logged(evaluate,output/'evaluate.log',cwd=app)
            evaluation_sha256 = verify_evaluation(output, meta)
        else:
            training = json.loads((output/'training.json').read_text(encoding='utf-8'))
            if training['status']=='stopped':
                checkpoint_path = output/'checkpoint.pt'
                waiting_cancelled = training.get('reason') == 'gpu_wait_cancelled'
                if not waiting_cancelled and (not checkpoint_path.is_file() or
                                              not isinstance(training.get('steps'), int) or
                                              training['steps'] < 0 or training['steps'] > steps or
                                              not isinstance(training.get('model_sha256'), str)):
                    raise RuntimeError('Stopped training lacks a completed checkpoint or model identity')
                if waiting_cancelled and not checkpoint_path.is_file() and (
                        training.get('steps') != 0 or (output/'model.ply').exists()):
                    raise RuntimeError('GPU-wait cancellation conflicts with prior training evidence')
                if checkpoint_path.is_file():
                    config = json.loads((output/'config.json').read_text(encoding='utf-8'))
                    if (config.get('package_sha256') != record['package_sha256'] or
                            config.get('steps') != steps or config.get('photo_comp') != photo_comp or
                            config.get('use_bilateral_grid') != use_bilateral_grid or
                            config.get('use_sparse_depth') != use_sparse_depth):
                        raise RuntimeError('Stopped checkpoint configuration differs from dispatch')
                if training.get('model_sha256') is not None and (
                        not (output/'model.ply').is_file() or
                        sha256_file(output/'model.ply') != training['model_sha256']):
                    raise RuntimeError('Stopped training PLY identity differs from its record')
                record.update(status='stopped',resources=resources,
                              completed_steps=training['steps'],
                              checkpoint=str(checkpoint_path) if checkpoint_path.is_file() else None,
                              stop_reason=training.get('reason'),
                              model_sha256=training.get('model_sha256'))
                json_write(output/'dispatch.json',record)
                return record
            if training['status']!='succeeded':
                raise RuntimeError('Native training/evaluation incomplete')
            config = json.loads((output/'config.json').read_text(encoding='utf-8'))
            if (training.get('steps') != steps or training.get('evaluation_status') != 'completed' or
                    config.get('package_sha256') != record['package_sha256'] or
                    config.get('steps') != steps or config.get('photo_comp') != photo_comp or
                    config.get('use_bilateral_grid') != use_bilateral_grid or
                    config.get('use_sparse_depth') != use_sparse_depth):
                raise RuntimeError('Native training identity or evaluation evidence differs from dispatch')
            evaluation_sha256 = verify_evaluation(output, meta)
            if training.get('model_sha256') != sha256_file(output/'model.ply'):
                raise RuntimeError('Native training PLY identity differs from its verified result')
        record.update(status='succeeded',resources=resources,
                      model_sha256=sha256_file(output/'model.ply'),
                      evaluation_sha256=evaluation_sha256)
    except Exception as error:
        record.update(status='failed',error=str(error))
        if getattr(error,'metrics',None):
            record['resources']=error.metrics
        json_write(output/'dispatch.json',record)
        raise
    json_write(output/'dispatch.json',record)
    return record
