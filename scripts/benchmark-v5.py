"""Create isolated representative capture/Run pairs; never modify source Runs."""
from pathlib import Path
import argparse
import json

from gsdb.manifests import load_model,save_yaml
from gsdb.media import sha256_file
from gsdb.models import CaptureManifestV2,RunConfigV5,StageStatus,TimeSelection
from gsdb.pipeline import preprocess_run,mask_run,reconstruct_run
from gsdb.runs import create_run,load_run
from gsdb.sources import prepare_capture_input
from gsdb.training_data import json_write
from gsdb.benchmarks import reconstruct_profile_baseline

parser=argparse.ArgumentParser()
parser.add_argument('--case',required=True,choices=['yanguan-stable','yanguan-stable-45','yunxiu-stable','yanguan-gap','yunxiu-pose'])
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--through',choices=['preprocess','mask','reconstruct'],default='reconstruct')
parser.add_argument('--variant',choices=['v5','old-primary','old-fallback'],default='v5')
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
cases={
 'yanguan-stable':('yanguan-ancient-town-20260822','night-walk-4k','capture-009-4k',0.7007,30.7007),
 'yanguan-stable-45':('yanguan-ancient-town-20260822','night-walk-4k','capture-009-4k',0.7007,45.7007),
 'yunxiu-stable':('yunxiu-20260822','vid-20260823-104518-00-014','capture-014-full',104.7046,134.7046),
 'yanguan-gap':('yanguan-ancient-town-20260822','night-walk-4k','capture-009-4k',45.,95.),
 'yunxiu-pose':('yunxiu-20260822','vid-20260823-104518-00-014','capture-014-full',65.,110.),
}
location,scene_id,source_id,start,end=cases[args.case]
scene=root/'Data'/location/scene_id
output=args.output.resolve()/args.case
if args.variant!='v5':
    output=output/args.variant
output.mkdir(parents=True,exist_ok=True)
state_path=output/'case.json'
if state_path.exists():
    state=json.loads(state_path.read_text(encoding='utf-8'))
    run=load_run(scene,state['run_id'])
elif args.variant!='v5':
    base_state_path=output.parent/'case.json'
    if not base_state_path.is_file():
        raise RuntimeError('Prepare the v5 case first so all projection controls share identical candidates')
    base_state=json.loads(base_state_path.read_text(encoding='utf-8'))
    base=load_run(scene,base_state['run_id'])
    historical_id='20260906T164624Z-97ab61db' if location.startswith('yanguan') else '20260909T030258Z-2e7ae759'
    historical=load_run(scene,historical_id)
    profile=getattr(historical.config.reconstruction,args.variant.removeprefix('old-')).model_copy(deep=True)
    profile.use_rig=False
    values=base.config.model_dump(mode='json')
    values['reconstruction']['primary']=profile.model_dump(mode='json')
    values['reconstruction']['alignment_masks']=False
    values['masking']=historical.config.masking.model_dump(mode='json')
    run=create_run(scene,location,scene_id,RunConfigV5.model_validate(values))
    state=dict(case=args.case,variant=args.variant,run_id=run.id,scene=str(scene),
        source_capture_sha256=base_state['source_capture_sha256'],prepared_dataset_sha256=base.config.input_dataset_sha256,
        historical_profile_run=historical.id,historical_profile_config_sha256=historical.config_hash,
        start_seconds=start,end_seconds=end,status='prepared')
    json_write(state_path,state)
else:
    source_path=scene/'captures'/f'{source_id}.yaml'
    source_hash=sha256_file(source_path)
    capture=load_model(source_path,CaptureManifestV2).model_copy(deep=True)
    capture.id=f'v5-20260912-{args.case}'
    capture.selection=TimeSelection(start_seconds=start,end_seconds=end)
    capture.prepared_relative_path=None
    capture_path=scene/'captures'/f'{capture.id}.yaml'
    if capture_path.exists():
        existing=load_model(capture_path,CaptureManifestV2)
        if existing.selection != capture.selection or existing.source != capture.source:
            raise RuntimeError('Benchmark capture already exists with different inputs')
        capture=existing
    else:
        save_yaml(capture_path,capture)
    prepared=prepare_capture_input(scene,capture,candidate_fps=5.,resume=True)
    capture.prepared_relative_path=prepared.path.relative_to(scene).as_posix()
    save_yaml(capture_path,capture)
    config=RunConfigV5(capture_id=capture.id,input_dataset_sha256=prepared.dataset_sha256,
        prepared_relative_path=capture.prepared_relative_path,input=dict(source_kind=capture.source.kind,
            source_sha256=[r.sha256 for r in capture.source.files],source_probe=prepared.source_probe,
            normalization=capture.normalization,selection=capture.selection,candidate_frame_indices=list(prepared.frame_indices),
            candidate_fps=5.,helper_version=prepared.helper_version,sdk_version=prepared.sdk_version))
    run=create_run(scene,location,scene_id,config)
    state=dict(case=args.case,variant=args.variant,run_id=run.id,scene=str(scene),source_capture_sha256=source_hash,
               prepared_dataset_sha256=prepared.dataset_sha256,
               start_seconds=start,end_seconds=end,status='prepared')
    json_write(state_path,state)
    if sha256_file(source_path)!=source_hash:
        raise RuntimeError('Source capture changed')
try:
    reconstruction_operation=reconstruct_run if args.variant=='v5' else reconstruct_profile_baseline
    for name,operation in [('preprocess',preprocess_run),('mask',mask_run),('reconstruct',reconstruction_operation)]:
        if run.stages[name].status!=StageStatus.SUCCEEDED:
            print(f'{args.case}: {name} {run.id}',flush=True)
            run=operation(scene,run,resume=True)
        if name==args.through:
            break
    furthest=next(name for name in ('reconstruct','mask','preprocess') if run.stages[name].status==StageStatus.SUCCEEDED)
    state.update(status=f'{furthest}_complete',selected_dataset=run.selected_dataset,qa=run.metrics.get('segment_qa'))
    if 'error' in state and not any(stage.status==StageStatus.FAILED for stage in run.stages.values()):
        state.setdefault('recovered_errors',[]).append(state.pop('error'))
except Exception as error:
    state.update(status='failed',error=str(error))
    json_write(state_path,state)
    raise
json_write(state_path,state)
print(json.dumps(state,ensure_ascii=False),flush=True)
