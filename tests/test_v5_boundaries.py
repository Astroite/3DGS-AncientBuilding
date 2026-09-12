import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gsdb.masking import filter_masked_images
from gsdb.models import ReconstructionConfigV5
from gsdb.reconstruction_realityscan import run_realityscan_alignment


def test_half_percent_is_inclusive(tmp_path):
    for folder in ('images','masks'):
        (tmp_path/folder).mkdir()
    for name,ignored in [('equal',50),('above',51),('below',49)]:
        cv2.imwrite(str(tmp_path/'images'/f'{name}.jpg'),np.full((100,100,3),127,np.uint8))
        mask=np.full((100,100),255,np.uint8)
        mask.flat[:ignored]=0
        cv2.imwrite(str(tmp_path/'masks'/f'{name}.jpg.png'),mask)
    records=[dict(image=f'{name}.jpg',mask=f'{name}.jpg.png',masked_fraction=ignored/10000)
             for name,ignored in [('equal',50),('above',51),('below',49)]]
    result=filter_masked_images(tmp_path,records,0.005)
    assert {r['image'] for r in result['accepted']}=={'equal.jpg','below.jpg'}
    assert {r['image'] for r in result['rejected']}=={'above.jpg'}


def test_realityscan_receives_binary_keep_masks_and_explicit_enable(tmp_path,monkeypatch):
    from gsdb import reconstruction_realityscan as rs
    for folder in ('images/view_00','masks/view_00'):
        (tmp_path/folder).mkdir(parents=True)
    name='view_00/frame_000001.jpg'
    cv2.imwrite(str(tmp_path/'images'/name),np.full((10,10,3),127,np.uint8))
    mask=np.full((10,10),255,np.uint8)
    mask[0,0]=0
    cv2.imwrite(str(tmp_path/'masks'/f'{name}.png'),mask)
    monkeypatch.setattr(rs,'realityscan_executable',lambda:Path('RealityScan.exe'))
    monkeypatch.setattr(rs,'_windows_argument',str)
    monkeypatch.setattr(rs,'export_params_file',lambda:tmp_path/'params.xml')
    def inspect(command,*args,**kwargs):
        assert 'inpMaskOpts=1' in command
        staged=tmp_path/'alignment-images'/'gsdb_000000.jpg.mask.png'
        assert np.array_equal(cv2.imread(str(staged),0),mask)
        audit=json.loads((tmp_path/'alignment-masks.json').read_text())
        assert audit['polarity']=='white_keep' and not audit['completed']
        assert audit['records'][0]['image']==name
        raise RuntimeError('inspected before external execution')
    monkeypatch.setattr(rs,'run_logged',inspect)
    config=ReconstructionConfigV5()
    with pytest.raises(RuntimeError,match='inspected before external execution'):
        run_realityscan_alignment(tmp_path,config.primary,tmp_path/'logs',config,included_images={name})
