import json
import numpy as np
import pytest

from gsdb.models import SegmentQAConfig, ReconstructionConfigV5
from gsdb.segments import analyze_segments, repair_records


def fixture(times, positions=None, views=3):
    records = [{'file':f'source_{i}.jpg','timestamp_seconds':t} for i,t in enumerate(times)]
    frames = []
    for i,t in enumerate(times,1):
        for v in range(views):
            m = np.diag([1.,-1.,-1.,1.])
            m[0,3] = positions[i-1] if positions is not None else t
            frames.append({'file_path':f'images/view_{v:02d}/frame_{i:06d}.jpg','transform_matrix':m.tolist()})
    included = {f['file_path'].removeprefix('images/') for f in frames}
    return frames,records,included


def test_long_gap_is_coverage_not_speed_failure():
    frames,records,included = fixture(list(range(12))+list(range(42,54)))
    report = analyze_segments(frames,records,included,SegmentQAConfig(),[np.eye(3)]*3)
    assert report['training_status']=='passed'
    assert report['coverage_status']=='incomplete'
    assert [i['type'] for i in report['issues']]==['coverage_gap']
    assert len(report['segments'])==2


def test_actual_spike_is_isolated_not_smoothed():
    positions = list(range(30))
    positions[15] = 1000
    frames,records,included = fixture(list(range(30)),positions)
    report = analyze_segments(frames,records,included,SegmentQAConfig(),[np.eye(3)]*3)
    assert sum(i['type']=='speed_jump' for i in report['issues'])==2
    assert 16 in report['excluded_frames']


def test_projection_rotation_removed():
    frames,records,included = fixture(list(range(15)),views=1)
    rot = np.diag([-1.,-1.,1.])
    for i,f in enumerate(frames):
        f['file_path'] = f['file_path'].replace('view_00',f'view_{i%2:02d}')
        f['transform_matrix'] = (np.array(f['transform_matrix']) @ np.block([[rot if i%2 else np.eye(3),np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]])).tolist()
    included = {f['file_path'].removeprefix('images/') for f in frames}
    report = analyze_segments(frames,records,included,SegmentQAConfig(),[np.eye(3),rot])
    assert not any(i['type']=='rotation_jump' for i in report['issues'])


def test_nonfinite_and_duplicate_are_integrity_errors():
    frames,records,included = fixture(list(range(12)))
    with pytest.raises(ValueError,match='Duplicate'):
        analyze_segments(frames+[frames[0]],records,included,SegmentQAConfig(),[np.eye(3)]*3)
    frames[0]['transform_matrix'][0][3]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        analyze_segments(frames,records,included,SegmentQAConfig(),[np.eye(3)]*3)


def test_registration_denominator_includes_unregistered_inputs():
    frames,records,included = fixture(list(range(12)))
    frames = [f for f in frames if 'view_00' in f['file_path']]
    report = analyze_segments(frames,records,included,SegmentQAConfig(),[np.eye(3)]*3)
    assert report['segments'][0]['registration_ratio']==pytest.approx(1/3)
    assert report['training_status']=='failed'


def test_repair_is_local_deduplicated_and_exhaustible():
    candidates = [{'file':f'{i}.jpg','timestamp_seconds':float(i)} for i in range(20)]
    selected = [{**candidates[i],'source_file':candidates[i]['file'],'file':f'frame_{i}.jpg'} for i in [0,5,10,15]]
    report = {'weak_input_frames':[2]}
    extra = repair_records(candidates,selected,report,2)
    assert [r['file'] for r in extra]==['3.jpg','4.jpg','6.jpg','7.jpg']
    assert repair_records(candidates,selected+extra,report,2)==[]


def test_v5_defaults():
    config = ReconstructionConfigV5()
    assert config.primary.images_per_equirect==14
    assert config.primary.temporal_rank_limit==2
    assert config.alignment_masks and not config.primary.use_rig
