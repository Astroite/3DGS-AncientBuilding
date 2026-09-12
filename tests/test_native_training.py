from pathlib import Path
import numpy as np
import pytest
import torch

from gsdb.native_train import PanoramaExposure, masked_losses, export_ply, load_ply
from gsdb.training_data import grouped_split


def test_masked_pixels_have_no_l1_or_ssim_gradient():
    pred = torch.rand(32,32,3,requires_grad=True)
    target = torch.rand(32,32,3)
    keep = torch.ones(32,32,dtype=torch.bool)
    keep[12:18,12:18]=False
    l1,ssim = masked_losses(pred,target,keep)
    (l1+ssim).backward()
    assert torch.count_nonzero(pred.grad[~keep])==0
    assert torch.count_nonzero(pred.grad[keep])>0


def test_exposure_reference_is_fixed_and_groups_share_parameters():
    model = PanoramaExposure([0.,0.5,1.])
    image = torch.ones(2,2,3)
    with torch.no_grad():
        model.values[:] = 0.2
    assert torch.equal(model(image,0),image)
    assert not torch.equal(model(image,1),image)
    (model(image,1).sum()+model.regularization()).backward()
    assert torch.count_nonzero(model.values.grad[0])==0


def test_time_groups_never_leak_across_split():
    split = grouped_split([i//14 for i in range(14*20)])
    assert split[7]=='validation' and split[15]=='validation'
    assert split[6]=='train'


def test_ply_roundtrip_preserves_sh_order_and_parameter_domains(tmp_path):
    n=5
    params = dict(means=torch.randn(n,3),quats=torch.tensor([[1.,0,0,0]]*n),
        scales=torch.randn(n,3),opacities=torch.randn(n),sh0=torch.randn(n,1,3),shN=torch.randn(n,15,3))
    path = tmp_path/'model.ply'
    export_ply(path,params)
    actual = load_ply(path,device='cpu')
    for name in params:
        assert torch.allclose(params[name],actual[name])
    export_ply(path,params,0)
    assert torch.count_nonzero(load_ply(path,'cpu')['shN'])==0
