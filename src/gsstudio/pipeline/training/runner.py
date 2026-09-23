"""Backend dispatch; the dataset and validation cameras are backend independent."""
from __future__ import annotations

import json
import math
from pathlib import Path

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
from gsstudio.infrastructure.adapters.gsplat import trainer_python
from gsstudio.infrastructure.runtime.processes import run_logged
from gsstudio.pipeline.training.data import json_write, postshot_adapter, validate_package, validate_postshot_adapter


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
    python = trainer_python()
    previous = None
    if output.exists():
        if not (output/'dispatch.json').is_file():
            raise FileExistsError(f'Unrecognized experiment directory: {output}')
        previous = json.loads((output/'dispatch.json').read_text(encoding='utf-8'))
        if not resume and previous['status'] != 'prepared':
            raise FileExistsError(f'Experiment already executed: {output}')
    output.mkdir(parents=True,exist_ok=True)
    if backend=='gsplat':
        command = [str(python),'-m','gsstudio.pipeline.training.native','--dataset',str(package),'--output',str(output),'--steps',str(steps)]
        if not photo_comp:
            command.append('--no-photo-comp')
        if not use_bilateral_grid:
            command.append('--no-use-bilateral-grid')
        if not use_sparse_depth:
            command.append('--no-use-sparse-depth')
        if resume:
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
    if previous:
        for key in ('backend','package_sha256','requested_steps','photo_comp',
                    'use_bilateral_grid','use_sparse_depth'):
            if previous.get(key) != record[key]:
                raise RuntimeError(f'Experiment identity changed: {key}')
    json_write(output/'dispatch.json',record)
    if dry_run:
        return record
    try:
        with gpu_session():
            resources = run_logged(command,output/'train.log',cwd=app,monitor_gpu=True,gpu_index=0)
        if backend=='postshot':
            log = (output/'train.log').read_text(encoding='utf-8')
            if 'Studio license required' in log or not (output/'model.psht').is_file() or not (output/'model.ply').is_file():
                raise RuntimeError('Postshot did not produce trained project and PLY; inspect license/log. GUI-ready input is preserved.')
            evaluate = [str(python),'-m','gsstudio.pipeline.training.native','--dataset',str(package),
                        '--output',str(output/'evaluation'),'--evaluate-ply',str(output/'model.ply')]
            with gpu_session():
                run_logged(evaluate,output/'evaluate.log',cwd=app)
        else:
            training = json.loads((output/'training.json').read_text(encoding='utf-8'))
            if training['status']!='succeeded':
                raise RuntimeError('Native training/evaluation incomplete')
        record.update(status='succeeded',resources=resources,model_sha256=sha256_file(output/'model.ply'))
    except Exception as error:
        record.update(status='failed',error=str(error))
        if getattr(error,'metrics',None):
            record['resources']=error.metrics
        json_write(output/'dispatch.json',record)
        raise
    json_write(output/'dispatch.json',record)
    return record
