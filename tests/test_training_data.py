import json
import cv2
import numpy as np
import pytest

from gsdb.media import sha256_file
from gsdb.training_data import json_write,postshot_adapter,validate_package,validate_postshot_adapter


def package(root):
    root.mkdir()
    for name in ('images','masks','colmap'):
        (root/name).mkdir()
    (root/'colmap'/'fixture.bin').write_bytes(b'initialization')
    rows=[]
    for i in range(2):
        image=f'images/{i}.jpg'
        mask=f'masks/{i}.jpg.png'
        cv2.imwrite(str(root/image),np.full((10,10,3),127,np.uint8))
        pixels=np.full((10,10),255,np.uint8)
        pixels[0,0]=0
        cv2.imwrite(str(root/mask),pixels)
        rows.append(dict(image=image,mask=mask,frame=i,timestamp_seconds=float(i),split='train' if i==0 else 'validation',
            K=np.eye(3).tolist(),world_to_camera=np.eye(4).tolist(),width=10,height=10))
    meta=dict(schema_version=1,validation='passed',images=rows,files={p.relative_to(root).as_posix():sha256_file(p) for p in root.rglob('*') if p.is_file()})
    json_write(root/'dataset.json',meta)
    return meta


def test_postshot_uses_inverse_mask_and_training_only(tmp_path):
    source=tmp_path/'package'
    package(source)
    before=sha256_file(source/'masks'/'0.jpg.png')
    result=postshot_adapter(source,tmp_path/'adapter')
    mask=cv2.imread(str(result/'masks'/'0.png'),0)
    assert mask[0,0]==255 and mask[1,1]==0
    assert [p.name for p in (result/'images').iterdir()]==['0.jpg']
    assert sha256_file(source/'masks'/'0.jpg.png')==before


def test_training_package_rejects_changed_files_and_temporal_leakage(tmp_path):
    source=tmp_path/'package'
    meta=package(source)
    meta['images'][1]['frame']=0
    json_write(source/'dataset.json',meta)
    with pytest.raises(RuntimeError,match='leaks'):
        validate_package(source)
    meta['images'][1]['frame']=1
    json_write(source/'dataset.json',meta)
    (source/'images'/'0.jpg').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='changed'):
        validate_package(source)


def test_postshot_prepared_input_rejects_wrong_mask_and_extra_views(tmp_path):
    source=tmp_path/'package'
    package(source)
    output=postshot_adapter(source,tmp_path/'adapter')
    original=cv2.imread(str(source/'masks'/'0.jpg.png'),0)
    cv2.imwrite(str(output/'masks'/'0.png'),original)
    with pytest.raises(RuntimeError,match='polarity'):
        validate_postshot_adapter(source,output)
    cv2.imwrite(str(output/'masks'/'0.png'),255-original)
    (output/'images'/'unexpected.jpg').write_bytes(b'unregistered')
    with pytest.raises(RuntimeError,match='inventory'):
        validate_postshot_adapter(source,output)


def test_segment_package_uses_training_only_colors_and_matching_poses(tmp_path,monkeypatch):
    from nerfstudio.data.utils.colmap_parsing_utils import Camera,Image,Point3D
    from gsdb.reconstruction import _write_colmap_binary_model
    from gsdb.models import ReconstructionConfigV5,SegmentQAConfig
    from gsdb import training_data as td
    dataset=tmp_path/'source'
    camera=Camera(1,'PINHOLE',20,20,np.array([10.,10.,10.,10.]))
    records=[dict(timestamp_seconds=float(i)) for i in range(12)]
    images={}
    frames=[]
    names=[]
    for i in range(12):
        name=f'view_00/frame_{i+1:06d}.jpg'
        names.append(name)
        for folder in ('images','masks'):
            (dataset/folder/'view_00').mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(dataset/'images'/name),np.full((20,20,3),255 if i==7 else 127,np.uint8))
        cv2.imwrite(str(dataset/'masks'/f'{name}.png'),np.full((20,20),255,np.uint8))
        images[i+1]=Image(i+1,np.array([1.,0.,0.,0.]),np.array([float(i),0.,0.]),1,name,
            np.array([[3.,3.],[5.,5.],[7.,7.]]),np.array([1,2,3]))
        m=np.diag([1.,-1.,-1.,1.]);m[0,3]=-i
        frames.append(dict(file_path=f'images/{name}',transform_matrix=m.tolist()))
    points={i:Point3D(i,np.array([float(i),0.,3.]),np.array([250,250,250]),0.,np.arange(1,13),np.full(12,i-1)) for i in (1,2,3)}
    _write_colmap_binary_model(dataset/'colmap',{1:camera},images,points)
    json_write(dataset/'transforms.json',dict(frames=frames,fl_x=10.,fl_y=10.,cx=10.,cy=10.,w=20,h=20))
    report=dict(training_status='passed',coverage_status='complete',segments=[dict(id='segment-001',status='passed',images=names,frames=list(range(1,13)))])
    monkeypatch.setattr(td,'validate_segments',lambda *a,**k:report)
    monkeypatch.setattr(td,'validate_mask_filter',lambda *a,**k:dict(accepted=[{'image':n} for n in names]))
    monkeypatch.setattr(td,'validate_mask_finalization',lambda *a,**k:{})
    result=td.prepare_segment(dataset,records,ReconstructionConfigV5().primary,SegmentQAConfig(),'segment-001',tmp_path/'package')
    assert result['initial_points']==3
    with np.load(tmp_path/'package'/'points.npz') as initialized:
        assert np.all(initialized['rgb']==127)
    assert not list(tmp_path.glob('.package.building-*'))
    frames[0]['transform_matrix'][0][3]=1234
    json_write(dataset/'transforms.json',dict(frames=frames,fl_x=10.,fl_y=10.,cx=10.,cy=10.,w=20,h=20))
    with pytest.raises(RuntimeError,match='pose/COLMAP mismatch'):
        td.prepare_segment(dataset,records,ReconstructionConfigV5().primary,SegmentQAConfig(),'segment-001',tmp_path/'bad-package')
    assert not (tmp_path/'bad-package').exists()


def test_earliest_window_rechecks_actual_input_registration_ratio():
    from gsdb.training_data import select_training_window
    from gsdb.models import SegmentQAConfig
    records=[dict(timestamp_seconds=float(i)) for i in range(45)]
    included={f'view_{v:02d}/frame_{i+1:06d}.jpg' for i in range(45) for v in range(3)}
    images=sorted(n for n in included if int(n.split('frame_')[1].split('.')[0])>15 or 'view_00' in n)
    segment=dict(id='segment-001',frames=list(range(1,46)),images=images,start_seconds=0.,end_seconds=44.,status='passed')
    report=dict(samples=[dict(frame=i+1,center=[float(i),0.,0.]) for i in range(45)])
    result=select_training_window(segment,report,records,included,SegmentQAConfig(),30.)
    assert result['start_seconds']==2.
    assert result['end_seconds']==32.
    assert result['input_images']==93
    assert result['registration_ratio']==pytest.approx(67/93)
    with pytest.raises(RuntimeError,match='longer independent capture'):
        select_training_window(segment,report,records,included,SegmentQAConfig(),60.)
