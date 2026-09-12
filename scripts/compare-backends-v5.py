"""Reproducible backend experiments on one passed segment; never launches full Runs."""
from pathlib import Path
import argparse
import json

from gsdb.runs import load_run,save_run
from gsdb.training_data import prepare_segment,json_write
from gsdb.training import train_package
from gsdb.media import sha256_file

parser=argparse.ArgumentParser()
parser.add_argument('--case-file',required=True,type=Path)
parser.add_argument('--segment',required=True)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--steps',type=int)
parser.add_argument('--duration-seconds',type=float)
parser.add_argument('--dry-run',action='store_true')
parser.add_argument('--resume',action='store_true')
parser.add_argument('--backends',nargs='+',choices=['gsplat','postshot'],default=['postshot','gsplat'])
args=parser.parse_args()
state=json.loads(args.case_file.read_text(encoding='utf-8'))
scene=Path(state['scene'])
run=load_run(scene,state['run_id'])
if run.config.schema_version!=5 or not run.selected_dataset or run.stages['reconstruct'].status.value!='succeeded':
    raise RuntimeError('Requires completed schema 5 segment QA')
label=run.metrics['selected_attempt']
records=[json.loads(line) for line in (scene/run.id/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
package_name=args.segment if args.duration_seconds is None else f'{args.segment}-first-{args.duration_seconds:g}s'
package=scene/run.id/'training-data'/package_name
meta=prepare_segment(scene/run.selected_dataset,records,run.config.reconstruction.primary,
    run.config.segment_qa,args.segment,package,duration_seconds=args.duration_seconds)
args.output.mkdir(parents=True,exist_ok=True)
results=[]
for backend in args.backends:
    for photo in (False,True):
        output=args.output/f'{backend}-photo-{"on" if photo else "off"}'
        entry=dict(backend=backend,photo_comp=photo,segment=args.segment,output=str(output.resolve()))
        previous=json.loads((output/'dispatch.json').read_text()) if (output/'dispatch.json').exists() else {}
        if previous.get('status')=='succeeded':
            expected_steps=max(30000,30*sum(r['split']=='train' for r in meta['images'])) if args.steps is None else args.steps
            expected=dict(backend=backend,photo_comp=photo,requested_steps=expected_steps,
                          package_sha256=sha256_file(package/'dataset.json'))
            if any(previous.get(k)!=v for k,v in expected.items()):
                raise RuntimeError(f'Completed comparison identity changed: {output}')
            if sha256_file(output/'model.ply')!=previous.get('model_sha256') or not (output/'evaluation'/'metrics.json').is_file():
                raise RuntimeError(f'Completed comparison artifacts changed or incomplete: {output}')
            entry.update(previous)
        else:
            try:
                entry.update(train_package(package,output,backend,args.steps,photo,
                    resume=args.resume and backend=='gsplat' and (output/'checkpoint.pt').exists(),dry_run=args.dry_run))
            except Exception as error:
                entry.update(status='failed',error=str(error))
        results.append(entry)
        run.metrics.setdefault('training_experiments',[]).append(entry)
        save_run(scene,run)
        json_write(args.output/'comparison.json',dict(package=str(package.resolve()),segment=args.segment,
            coverage=meta['coverage_status'],experiments=results,visual_acceptance='pending',full_runs_authorized=False))
        print(json.dumps(entry,ensure_ascii=False),flush=True)
        if backend=='postshot' and entry['status']=='failed' and 'license/log' in entry.get('error',''):
            # The second toggle needs the same license. Preserve a reviewable GUI
            # input without issuing another known-invalid train operation.
            if not photo:
                target=args.output/'postshot-photo-on'
                prepared=train_package(package,target,'postshot',args.steps,True,dry_run=True)
                prepared.update(status='blocked',reason='Postshot CLI license unavailable; GUI input prepared')
                results.append(prepared)
                json_write(args.output/'comparison.json',dict(package=str(package.resolve()),segment=args.segment,
                    coverage=meta['coverage_status'],experiments=results,visual_acceptance='pending',full_runs_authorized=False))
            break
