"""Run remaining representative experiments serially; never start full-scene Runs."""
from pathlib import Path
import argparse
import json
import sys

from gsdb.processes import run_logged
from gsdb.runs import load_run
from gsdb.training_data import json_write,prepare_segment

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--train',action='store_true',help='Run full-budget native compensation off/on comparisons after segment checks')
args=parser.parse_args()
app=Path(__file__).resolve().parents[1]
root=args.output.resolve()
root.mkdir(parents=True,exist_ok=True)
progress_path=root/'validation-progress.json'
progress=json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.is_file() else dict(jobs=[],full_runs_started=False)

def command_job(key,command):
    record=dict(key=key,command=command,status='running')
    progress['jobs'].append(record)
    progress['active_job']=key
    json_write(progress_path,progress)
    print(f'Starting {key}',flush=True)
    try:
        record.update(status='completed',resources=run_logged(command,root/'queue-logs'/f'{key}.log',cwd=app))
    except Exception as error:
        record.update(status='failed',error=str(error))
    json_write(progress_path,progress)
    print(json.dumps(record,ensure_ascii=False),flush=True)
    return record

jobs=[('yunxiu-pose','v5'),('yanguan-stable-45','v5'),
      ('yanguan-stable','old-primary'),('yanguan-stable','old-fallback'),
      ('yunxiu-stable','old-primary'),('yunxiu-stable','old-fallback')]
for case,variant in jobs:
    command_job(f'{case}-{variant}',[sys.executable,str(app/'scripts'/'benchmark-v5.py'),
        '--case',case,'--variant',variant,'--output',str(root/'benchmark')])

# Confirm the old-profile controls reused each new Run's exact candidate pool.
controls=[]
for case in ('yanguan-stable','yunxiu-stable'):
    base=json.loads((root/'benchmark'/case/'case.json').read_text(encoding='utf-8'))
    scene=Path(base['scene'])
    base_run=load_run(scene,base['run_id'])
    rows=[]
    for variant in ('v5','old-primary','old-fallback'):
        state=base if variant=='v5' else json.loads((root/'benchmark'/case/variant/'case.json').read_text(encoding='utf-8'))
        run=load_run(scene,state['run_id'])
        if run.config.input_dataset_sha256!=base_run.config.input_dataset_sha256:
            raise RuntimeError(f'Candidate pool mismatch: {case}/{variant}')
        rows.append(dict(variant=variant,run=run.id,config_sha256=run.config_hash,
            reconstruction=run.metrics.get('reconstruction'),qa=run.metrics.get('segment_qa'),status=run.status.value))
    controls.append(dict(case=case,prepared_dataset_sha256=base_run.config.input_dataset_sha256,attempts=rows))
json_write(root/'projection-controls.json',controls)

if args.train:
    for case,duration in [('yanguan-stable-45',30.),('yunxiu-stable',None)]:
        state=json.loads((root/'benchmark'/case/'case.json').read_text(encoding='utf-8'))
        scene=Path(state['scene'])
        run=load_run(scene,state['run_id'])
        if run.stages['reconstruct'].status.value!='succeeded' or not run.selected_dataset:
            progress.setdefault('blocked',[]).append(dict(case=case,reason='No QA-approved reconstruction'))
            continue
        report=json.loads((scene/run.selected_dataset/'segments.json').read_text(encoding='utf-8'))
        candidates=[s for s in report['segments'] if s['status']=='passed' and
                    (duration is None or s['end_seconds']-s['start_seconds']>=duration)]
        if not candidates:
            progress.setdefault('blocked',[]).append(dict(case=case,reason='No complete 30-second segment'))
            continue
        segment=candidates[0]['id']
        command=[sys.executable,str(app/'scripts'/'compare-backends-v5.py'),
            '--case-file',str(root/'benchmark'/case/'case.json'),'--segment',segment,
            '--output',str(root/'benchmark'/case/'native-comparison'),'--backends','gsplat','--resume']
        if duration is not None:
            command+=['--duration-seconds',str(duration)]
        job=command_job(f'{case}-native-comparison',command)
        comparison=root/'benchmark'/case/'native-comparison'/'comparison.json'
        if comparison.is_file():
            result=json.loads(comparison.read_text(encoding='utf-8'))
            if any(e['status']!='succeeded' for e in result['experiments']):
                job.update(status='failed',error='Native comparison contains unsuccessful experiments')

progress['active_job']=None
progress['status']='experiments_finished_pending_review'
progress['postshot_status']='Studio CLI license blocked in the last observed attempt; GUI input prepared separately'
progress['full_runs_started']=False
json_write(progress_path,progress)
print(json.dumps(progress,ensure_ascii=False),flush=True)
