import json
from types import SimpleNamespace

import pytest

from gsdb.models import ReconstructionConfigV5,SegmentQAConfig


@pytest.mark.parametrize('passes',[True,False])
def test_historical_control_runs_one_alignment_and_preserves_qa_failure(tmp_path,monkeypatch,passes):
    from gsdb import benchmarks as b
    work=tmp_path/'isolated-control'
    work.mkdir()
    (work/'selected-primary-metrics.jsonl').write_text(json.dumps(dict(timestamp_seconds=0.)))
    run=SimpleNamespace(id=work.name,config=SimpleNamespace(reconstruction=ReconstructionConfigV5(),segment_qa=SegmentQAConfig()),
                        metrics={},selected_dataset=None)
    events=[]
    monkeypatch.setattr(b,'begin_stage',lambda *a,**k:True)
    monkeypatch.setattr(b,'save_run',lambda *a:None)
    monkeypatch.setattr(b,'complete_stage',lambda *a,**k:events.append('completed'))
    monkeypatch.setattr(b,'fail_stage',lambda *a,**k:events.append('failed'))
    monkeypatch.setattr(b,'validate_mask_filter',lambda *a,**k:dict(accepted=[dict(image='view_00/frame_000001.jpg')],original_image_count=14))
    monkeypatch.setattr(b,'validate_mask_finalization',lambda *a,**k:{})
    def align(dataset,*a,**k):
        events.append(dataset.name)
        assert k['included_images']=={'view_00/frame_000001.jpg'}
        return dict(registered_images=1)
    monkeypatch.setattr(b,'run_realityscan_alignment',align)
    monkeypatch.setattr(b,'write_segments',lambda *a,**k:dict(training_status='passed' if passes else 'failed',coverage_status='incomplete'))
    # CPU-only orchestration test; actual GPU serialization has a separate process test.
    operation=b.reconstruct_profile_baseline.__wrapped__
    if passes:
        operation(tmp_path,run)
        assert run.selected_dataset=='isolated-control/reconstruction-primary'
        assert events==['reconstruction-primary','completed']
    else:
        with pytest.raises(RuntimeError,match='no passing'):
            operation(tmp_path,run)
        assert run.selected_dataset is None
        assert events==['reconstruction-primary','failed']
    assert run.metrics['segment_qa']['coverage_status']=='incomplete'
