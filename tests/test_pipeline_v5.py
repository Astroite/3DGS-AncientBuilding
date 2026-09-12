import json
from types import SimpleNamespace

from gsdb.models import ReconstructionConfigV5, SegmentQAConfig
from gsdb.pipeline_v5 import reconstruct_v5


def test_single_repair_retains_better_primary_segments(tmp_path,monkeypatch):
    from gsdb import pipeline_v5 as v5, pipeline, masking, sources, reconstruction_realityscan as rs
    work=tmp_path/'new-run'
    work.mkdir()
    records=[dict(file=f'frame_{i+1:06d}.jpg',source_file=f'source_{i}.jpg',timestamp_seconds=float(i)) for i in range(12)]
    candidates=records+[dict(file='extra.jpg',timestamp_seconds=5.5)]
    (work/'selected-primary-metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    (work/'frame-metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in candidates))
    prepared=tmp_path/'prepared'
    (prepared/'frames').mkdir(parents=True)
    for r in candidates:
        (prepared/'frames'/r.get('source_file',r['file'])).write_bytes(b'source')
    run=SimpleNamespace(id='new-run',config=SimpleNamespace(prepared_relative_path='prepared',input_dataset_sha256='same',
        reconstruction=ReconstructionConfigV5(),segment_qa=SegmentQAConfig()),metrics={},fallback_attempted=False)
    monkeypatch.setattr(v5,'begin_stage',lambda *a,**k:True)
    for name in ('save_run','complete_stage','fail_stage'):
        monkeypatch.setattr(v5,name,lambda *a,**k:None)
    monkeypatch.setattr(sources,'_dataset_from_manifest',lambda *a:SimpleNamespace(path=prepared,dataset_sha256='same'))
    monkeypatch.setattr(pipeline,'_prepare_masked_dataset',lambda *a,**k:{})
    included={'view_00/frame_000001.jpg','view_01/frame_000001.jpg'}
    monkeypatch.setattr(masking,'validate_mask_filter',lambda *a,**k:dict(accepted=[{'image':n} for n in included],original_image_count=2))
    monkeypatch.setattr(pipeline,'_v4_attempt_inventory',lambda *a:({},included))
    alignments=[]
    monkeypatch.setattr(rs,'run_realityscan_alignment',lambda dataset,*a,**k:alignments.append(dataset.name) or {})
    def report(dataset,*args):
        return dict(training_status='passed' if dataset.name.endswith('primary') else 'failed',coverage_status='incomplete',
            weak_input_frames=[6],segments=[dict(status='passed',start_seconds=0.,end_seconds=11.,images=list(included))])
    monkeypatch.setattr(v5,'write_segments',report)
    result=reconstruct_v5(tmp_path,run)
    assert alignments==['reconstruction-primary','reconstruction-repair']
    assert result.selected_dataset=='new-run/reconstruction-primary'
    assert result.metrics['selected_attempt']=='primary'
    decision=json.loads((work/'repair-plan.json').read_text())
    assert decision['attempt']==1 and decision['added_source_files']==['extra.jpg']


def test_quality_snapshot_reads_repair_instead_of_legacy_fallback(tmp_path):
    from gsdb.pipeline import quality_snapshot
    work=tmp_path/'new-run'
    work.mkdir()
    (work/'selected-repair-metrics.jsonl').write_text('{}\n'*12)
    run=SimpleNamespace(id='new-run',config=SimpleNamespace(schema_version=5,reconstruction=ReconstructionConfigV5()),
        fallback_attempted=True,selected_dataset=None,artifacts=[],metrics={'selected_attempt':'repair',
            'reconstruction':{'repair':{'registration_ratio':0.9}},
            'segment_qa':{'training_status':'passed','coverage_status':'incomplete'}},
        stages={name:SimpleNamespace(elapsed_seconds=0) for name in ('train','reconstruct')})
    result=quality_snapshot(tmp_path,run)
    assert result['registration_ratio']==0.9
    assert result['selected_frames']==12
    assert result['coverage_status']=='incomplete'
