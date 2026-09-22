"""Native Windows ADC trainer with masked losses and panorama exposure groups."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .manifests import canonical_hash
from .media import sha256_file
from .photometric import BilateralGrid
from .ply import FLOAT_PROPERTIES, read_ply_header
from .sparse_depth import load_sparse_depth
from .training_data import json_write, validate_package


def configure_windows_cuda():
    if os.name != 'nt':
        return
    try:
        from gsplat import csrc
        return
    except ImportError:
        pass
    cuda = Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8')
    if not (cuda/'bin'/'nvcc.exe').is_file():
        raise RuntimeError('Pinned native trainer requires CUDA Toolkit 11.8')
    locator = Path(r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe')
    installation = subprocess.check_output(
        [str(locator), '-latest', '-products', '*', '-requires',
         'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'],
        text=True,
        encoding='utf-8',
        errors='replace',
    ).strip()
    vcvars = Path(installation)/'VC'/'Auxiliary'/'Build'/'vcvars64.bat'
    if not vcvars.is_file():
        raise RuntimeError('MSVC build environment is unavailable')
    # cmd.exe writes OEM code-page bytes on Chinese Windows (often GBK), not UTF-8.
    raw_environment = subprocess.check_output(
        ['cmd.exe', '/d', '/s', '/c', f'""{vcvars}" -vcvars_ver=14.38 >nul && set"'],
    )
    for encoding in ('oem', 'mbcs', 'utf-8'):
        try:
            environment = raw_environment.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        environment = raw_environment.decode('utf-8', errors='replace')
    for line in environment.splitlines():
        if '=' in line and not line.startswith('='):
            key,value = line.split('=',1)
            os.environ[key]=value
    os.environ['CUDA_HOME']=str(cuda)
    os.environ['CUDA_PATH']=str(cuda)
    os.environ['PATH']=str(cuda/'bin')+os.pathsep+os.environ['PATH']
    os.environ['MAX_JOBS']='4'
    os.environ['TORCH_CUDA_ARCH_LIST']='8.9'


def masked_losses(prediction, target, keep):
    """HWC RGB, HW bool; SSIM windows never touch excluded observations."""
    weight = keep.float()
    l1 = ((prediction-target).abs()*weight[...,None]).sum()/(3*weight.sum().clamp_min(1))
    x,y = prediction.permute(2,0,1)[None],target.permute(2,0,1)[None]
    kernel = min(11, x.shape[-1], x.shape[-2])
    if kernel % 2 == 0:
        kernel -= 1
    avg = lambda z: F.avg_pool2d(z,kernel,stride=1)
    mx,my = avg(x),avg(y)
    vx,vy = avg(x*x)-mx*mx,avg(y*y)-my*my
    cov = avg(x*y)-mx*my
    ssim = ((2*mx*my+0.01**2)*(2*cov+0.03**2))/((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))
    support = (avg(weight[None,None]) >= 1-1e-6).float()
    ssim_loss = ((1-ssim)*support).sum()/(3*support.sum().clamp_min(1))
    return l1,ssim_loss


class PanoramaExposure(torch.nn.Module):
    def __init__(self, timestamps):
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(len(timestamps),6))
        self.register_buffer('anchor',torch.tensor([0.]+[1.]*(len(timestamps)-1))[:,None])
        self.register_buffer('dt',torch.tensor(np.diff(timestamps),dtype=torch.float32).clamp_min(0.1))

    def forward(self, rgb, group):
        value = (self.values*self.anchor)[group]
        return rgb*value[:3].exp()+value[3:]

    def regularization(self):
        v = self.values*self.anchor
        smooth = ((v[1:]-v[:-1]).square()/self.dt[:,None]).mean() if len(v)>1 else v.sum()*0
        return 1e-3*v.square().mean()+1e-3*smooth


def select_photometric(photo_comp, use_bilateral_grid, num_groups, group_times):
    """Three-way photometric routing: identity / BilateralGrid / PanoramaExposure."""
    if not photo_comp:
        return None
    if use_bilateral_grid:
        return BilateralGrid(num_groups)
    return PanoramaExposure(group_times)


def _historical_defaults(config: dict) -> dict:
    """Absent or null bilateral/sparse fields mean the pre-flag trainer behavior."""
    merged = dict(config)
    for key, fallback in (
        ('use_bilateral_grid', False),
        ('use_sparse_depth', False),
        ('sparse_depth_weight', 0.0),
    ):
        if merged.get(key) is None:
            merged[key] = fallback
    return merged


def _configs_match(previous: dict, current: dict) -> bool:
    keys = (
        'package_sha256', 'steps', 'photo_comp', 'antialiased', 'sh_degree',
        'seed', 'high_order_l2', 'checkpoint_every',
        'use_bilateral_grid', 'use_sparse_depth', 'sparse_depth_weight',
    )
    left = _historical_defaults(previous)
    right = _historical_defaults(current)
    return all(left.get(key) == right.get(key) for key in keys)


def export_ply(path: Path, params, sh_degree=3):
    count = len(params['means'])
    values = {k:v.detach().cpu().numpy() for k,v in params.items()}
    rest = values['shN'].copy()
    rest[:,(sh_degree+1)**2-1:,:] = 0
    quats = values['quats']/np.linalg.norm(values['quats'],axis=1,keepdims=True).clip(1e-12)
    data = np.concatenate([values['means'],np.zeros((count,3)),values['sh0'][:,0],
        rest.transpose(0,2,1).reshape(count,45),values['opacities'][:,None],values['scales'],quats],axis=1).astype('<f4')
    if not np.isfinite(data).all():
        raise RuntimeError('Cannot export nonfinite Gaussian parameters')
    header = 'ply\nformat binary_little_endian 1.0\ncomment coordinates COLMAP\n'
    header += f'element vertex {count}\n'+''.join(f'property float {p}\n' for p in FLOAT_PROPERTIES)+'end_header\n'
    temporary = path.with_suffix('.ply.tmp')
    with temporary.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(data.tobytes())
    temporary.replace(path)


def load_ply(path: Path, device='cuda'):
    header,count,props = read_ply_header(path)
    if not set(FLOAT_PROPERTIES) <= set(props):
        raise RuntimeError('Expected standard SH3 float PLY; export SH3 from Postshot')
    data = np.fromfile(path,dtype='<f4',offset=len(header)).reshape(count,len(props))
    if not np.isfinite(data).all():
        raise RuntimeError('Nonfinite PLY parameters')
    def tensor(names):
        return torch.tensor(data[:,[props.index(n) for n in names]],device=device)
    return dict(means=tensor(['x','y','z']),quats=tensor(['rot_0','rot_1','rot_2','rot_3']),
        scales=tensor(['scale_0','scale_1','scale_2']),opacities=tensor(['opacity'])[:,0],
        sh0=tensor(['f_dc_0','f_dc_1','f_dc_2'])[:,None],
        shN=tensor([f'f_rest_{i}' for i in range(45)]).reshape(count,3,15).transpose(1,2).contiguous())


def render(params, row, degree=3, antialiased=False, with_depth=False):
    from gsplat import rasterization
    device = params['means'].device
    render_mode = 'RGB+ED' if with_depth else 'RGB'
    return rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),
        opacities=params['opacities'].sigmoid(),colors=torch.cat([params['sh0'],params['shN']],dim=1),
        viewmats=torch.tensor(row['world_to_camera'],dtype=torch.float32,device=device)[None],
        Ks=torch.tensor(row['K'],dtype=torch.float32,device=device)[None],width=row['width'],height=row['height'],
        sh_degree=degree,packed=False,rasterize_mode='antialiased' if antialiased else 'classic',
        render_mode=render_mode)


def load_observation(package, row, device='cuda'):
    from .studio_data import read_image
    image = read_image(package/row['image'])
    mask = read_image(package/row['mask'],True)
    if image is None or mask is None or image.shape[:2] != mask.shape or mask.shape != (row['height'],row['width']):
        raise RuntimeError(f'Observation dimensions changed: {row["image"]}')
    return torch.tensor(image[...,::-1].copy(),device=device,dtype=torch.float32)/255, torch.tensor(mask==255,device=device)


def initialize(package, device='cuda'):
    from scipy.spatial import cKDTree
    with np.load(package/'points.npz') as points:
        xyz,rgb = points['xyz'],points['rgb'].astype(np.float32)/255
    distances,_ = cKDTree(xyz).query(xyz,k=min(4,len(xyz)),workers=-1)
    scales = np.sqrt(np.mean(distances[:,1:]**2,axis=1)).clip(1e-6)
    n = len(xyz)
    arrays = dict(means=xyz,scales=np.log(scales)[:,None].repeat(3,axis=1),
        quats=np.tile([1.,0.,0.,0.],(n,1)),opacities=np.full(n,math.log(0.1/0.9)),
        sh0=((rgb-0.5)/0.28209479177387814)[:,None],shN=np.zeros((n,15,3)))
    return torch.nn.ParameterDict({k:torch.nn.Parameter(torch.tensor(v,dtype=torch.float32,device=device)) for k,v in arrays.items()})


def evaluate(package, meta, params, output, antialiased=False):
    import lpips
    metric = lpips.LPIPS(net='alex',spatial=True).to(params['means'].device).eval()
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    with torch.no_grad():
        for index,row in enumerate(r for r in meta['images'] if r['split']=='validation'):
            target,keep = load_observation(package,row)
            raw,_,_ = render(params,row,antialiased=antialiased)
            raw = raw[0]
            pred = raw.clamp(0,1)
            mse = ((pred-target).square()*keep[...,None]).sum()/(3*keep.sum().clamp_min(1))
            _,ssim_loss = masked_losses(pred,target,keep)
            x = (pred*keep[...,None]).permute(2,0,1)[None]*2-1
            y = (target*keep[...,None]).permute(2,0,1)[None]*2-1
            lp = metric(x,y)
            # Exclude AlexNet receptive fields touching masks, including boundaries.
            valid = cv2.erode(keep.cpu().numpy().astype(np.uint8),np.ones((195,195),np.uint8),
                              borderType=cv2.BORDER_CONSTANT,borderValue=0)
            support = F.interpolate(torch.tensor(valid,device=keep.device,dtype=torch.float32)[None,None],size=lp.shape[-2:],mode='nearest')
            lpvalue = float((lp*support).sum()/support.sum()) if support.sum()>0 else None
            rows.append(dict(image=row['image'],frame=row['frame'],psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),
                ssim=float(1-ssim_loss),lpips=lpvalue,raw_max=float(raw.max()),raw_p999=float(torch.quantile(raw.flatten(),0.999)),
                over_one_fraction=float((raw>1).float().mean()),target_saturated_fraction=float((target>=254/255).float().mean())))
            cv2.imwrite(str(output/f'{index:05d}.png'),(pred.cpu().numpy()[...,::-1]*255).round().astype(np.uint8))
            cv2.imwrite(str(output/f'{index:05d}-reference.png'),(target.cpu().numpy()[...,::-1]*255).round().astype(np.uint8))
        # Rotate each of the first three validation cameras: same centers and intrinsics.
        scan = []
        for camera_index,row in enumerate([r for r in meta['images'] if r['split']=='validation'][:3]):
            for degrees in [-30,-15,0,15,30]:
                angle = math.radians(degrees)
                rotation = np.eye(4)
                rotation[:3,:3] = [[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]]
                scan_row = {**row,'world_to_camera':(rotation@np.array(row['world_to_camera'])).tolist()}
                item = dict(image=row['image'],yaw_delta=degrees)
                for degree in [0,3]:
                    raw,_,_ = render(params,scan_row,degree,antialiased)
                    basename=f'angle-{camera_index:02d}-{degrees:+03d}-sh{degree}'
                    values=raw[0].cpu().numpy()
                    cv2.imwrite(str(output/f'{basename}.png'),(values.clip(0,1)[...,::-1]*255).round().astype(np.uint8))
                    peaks=np.argsort(values.max(axis=2).reshape(-1))[-4096:]
                    np.savez_compressed(output/f'{basename}-raw-peaks.npz',pixel_indices=peaks,
                        rgb=values.reshape(-1,3)[peaks],shape=np.array(values.shape),world_to_camera=scan_row['world_to_camera'])
                    item[f'sh{degree}'] = dict(max=float(raw.max()),p999=float(torch.quantile(raw.flatten(),0.999)),over_one_fraction=float((raw>1).float().mean()),
                        render=f'{basename}.png',raw_peaks=f'{basename}-raw-peaks.npz')
                scan.append(item)
    result = {'schema_version':1,'color_space':'sRGB values; fixed display exposure 0 EV',
              'metrics_mode':'canonical appearance; no validation exposure fitting','images':rows,'angle_scan':scan,
              'averages':{k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in ['psnr','ssim','lpips']},
              'limitations':['Person residuals and real specular highlights require visual comparison.']}
    json_write(output/'metrics.json',result)
    return result


def sparse_depth_loss(rendered_depth, anchors, row_index):
    """L1 on SfM depth anchors belonging to one training view."""
    selected = anchors['row_index'] == row_index
    if not np.any(selected):
        return rendered_depth.sum() * 0.0, 0
    width = rendered_depth.shape[1]
    height = rendered_depth.shape[0]
    u = np.clip(np.rint(anchors['u'][selected]).astype(np.int64), 0, width - 1)
    v = np.clip(np.rint(anchors['v'][selected]).astype(np.int64), 0, height - 1)
    target = torch.tensor(anchors['depth'][selected], device=rendered_depth.device, dtype=rendered_depth.dtype)
    return (rendered_depth[v, u] - target).abs().mean(), int(selected.sum())


def train(package, output, steps=None, photo_comp=True, resume=False, antialiased=False, checkpoint_every=1000,
          observer=None, run_evaluation=True, sh_degree=3, use_bilateral_grid=True, use_sparse_depth=True,
          sparse_depth_weight=0.1):
    from gsplat import DefaultStrategy
    if not torch.cuda.is_available():
        raise RuntimeError('Native CUDA is unavailable')
    meta = validate_package(package)
    observations = [r for r in meta['images'] if r['split']=='train']
    steps = max(30000,30*len(observations)) if steps is None else steps
    if steps<1 or checkpoint_every<1 or sh_degree not in (0,1,2,3) or sparse_depth_weight<0:
        raise ValueError('Training and checkpoint budgets must be positive')
    anchors = load_sparse_depth(package) if use_sparse_depth else None
    sparse_depth_status = 'enabled' if anchors is not None and len(anchors['depth']) else (
        'unavailable' if use_sparse_depth else 'disabled')
    if anchors is not None and not len(anchors['depth']):
        sparse_depth_status = 'empty'
    config = dict(schema_version=2,package_sha256=sha256_file(package/'dataset.json'),steps=steps,
        photo_comp=photo_comp,antialiased=antialiased,sh_degree=sh_degree,seed=20260912,high_order_l2=1e-6,
        checkpoint_every=checkpoint_every,use_bilateral_grid=bool(use_bilateral_grid),
        use_sparse_depth=bool(use_sparse_depth),sparse_depth_weight=float(sparse_depth_weight),
        sparse_depth_status=sparse_depth_status)
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = output/'checkpoint.pt'
    if (output/'config.json').exists() and not resume:
        raise FileExistsError('Training output already exists; use --resume')
    if (output/'config.json').exists() and not _configs_match(json.loads((output/'config.json').read_text()), config):
        raise RuntimeError('Checkpoint training configuration changed')
    if resume and not checkpoint.exists():
        raise RuntimeError('No recoverable checkpoint')
    json_write(output/'config.json',config)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    saved = torch.load(checkpoint,map_location='cuda') if resume else None
    params = torch.nn.ParameterDict({k:torch.nn.Parameter(v) for k,v in saved['params'].items()}) if saved else initialize(package)
    centers = np.array([np.linalg.inv(np.array(r['world_to_camera']))[:3,3] for r in observations])
    extent = float(np.linalg.norm(np.ptp(centers,axis=0)))
    rates = dict(means=1.6e-4*max(extent,1e-6),scales=5e-3,quats=1e-3,opacities=5e-2,sh0=2.5e-3,shN=2.5e-3/20)
    optimizers = {k:torch.optim.Adam([params[k]],lr=lr,eps=1e-15) for k,lr in rates.items()}
    groups = sorted({r['frame'] for r in observations})
    group_ids = {f:i for i,f in enumerate(groups)}
    group_times = [next(r['timestamp_seconds'] for r in observations if r['frame']==f) for f in groups]
    photometric = select_photometric(photo_comp, use_bilateral_grid, len(groups), group_times)
    if photometric is not None:
        photometric = photometric.cuda()
    exposure_optimizer = (
        torch.optim.Adam(photometric.parameters(), lr=1e-3) if photometric is not None else None
    )
    # Observation identity for sparse-depth rows recorded at package prepare time.
    observation_row = {r['image']: index for index, r in enumerate(meta['images'])}
    scale = max(1.,len(observations)/1000)
    strategy = DefaultStrategy(refine_start_iter=500,refine_stop_iter=min(int(15000*scale),int(steps*0.75)),
        reset_every=max(3000,len(observations)+100),pause_refine_after_reset=len(observations))
    versions=dict(torch=torch.__version__,gsplat=version('gsplat'),cuda=torch.version.cuda,numpy=np.__version__,opencv=cv2.__version__)
    runtime=output/(f'runtime-resume-{time.time_ns()}.json' if resume else 'runtime.json')
    json_write(runtime,dict(versions=versions,device=torch.cuda.get_device_name(),strategy=asdict(strategy),
        initial_learning_rates=rates,training_images=len(observations),scene_scale=max(extent,1e-6)))
    strategy.check_sanity(params,optimizers)
    state = strategy.initialize_state(scene_scale=max(extent,1e-6))
    start = 0
    if saved:
        if saved['config'] != config:
            raise RuntimeError('Checkpoint identity mismatch')
        for k,opt in optimizers.items():
            opt.load_state_dict(saved['optimizers'][k])
        if photometric is not None:
            photometric.load_state_dict(saved.get('photometric', saved.get('exposure', {})))
            exposure_optimizer.load_state_dict(saved.get('photometric_optimizer', saved.get('exposure_optimizer', {})))
        state,start = saved['strategy'],saved['step']
        torch.set_rng_state(saved['rng_cpu'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in saved['rng_cuda']])
    started=time.monotonic()
    elapsed_before=float(saved.get('elapsed_seconds',0.)) if saved else 0.
    def save(step):
        temp = checkpoint.with_suffix('.pt.tmp')
        torch.save(dict(config=config,step=step,params={k:v.detach() for k,v in params.items()},
            optimizers={k:o.state_dict() for k,o in optimizers.items()},strategy=state,
            photometric=photometric.state_dict() if photometric is not None else {},
            photometric_optimizer=exposure_optimizer.state_dict() if exposure_optimizer is not None else {},
            exposure=photometric.state_dict() if photometric is not None else {},
            exposure_optimizer=exposure_optimizer.state_dict() if exposure_optimizer is not None else {},
            elapsed_seconds=elapsed_before+time.monotonic()-started,
            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),temp)
        temp.replace(checkpoint)
    if not saved:
        save(0)
    torch.cuda.reset_peak_memory_stats()
    completed_step = start
    try:
        def observe(loss=None):
            if observer is None:
                return None
            return observer(params, dict(step=completed_step,target=steps,loss=loss,splats=len(params['means']),
                elapsed_seconds=elapsed_before+time.monotonic()-started,
                vram_bytes=torch.cuda.memory_allocated()), lambda: save(completed_step))
        action = observe()
        for step in range(start,steps):
            if action == 'stop':
                save(completed_step)
                export_ply(output/'model.ply',params,sh_degree)
                summary = dict(status='stopped',steps=completed_step,target=steps,
                    model_sha256=sha256_file(output/'model.ply'))
                json_write(output/'training.json',summary)
                return summary
            row = observations[int(torch.randint(len(observations),(1,)))]
            target,keep = load_observation(package,row)
            degree = min(sh_degree,step//1000)
            need_depth = anchors is not None and sparse_depth_status == 'enabled'
            raw,alpha,info = render(params,row,degree,antialiased,with_depth=need_depth)
            strategy.step_pre_backward(params,optimizers,state,step,info)
            if need_depth:
                rgb_raw, depth_raw = raw[0][..., :3], raw[0][..., 3]
            else:
                rgb_raw, depth_raw = raw[0], None
            pred = photometric(rgb_raw,group_ids[row['frame']]) if photometric is not None else rgb_raw
            l1,ssim = masked_losses(pred,target,keep)
            loss = 0.8*l1+0.2*ssim+config['high_order_l2']*params['shN'].square().mean()
            if photometric is not None:
                loss = loss+photometric.regularization()
            depth_value = None
            if need_depth and depth_raw is not None:
                row_index = observation_row[row['image']]
                depth_value, depth_count = sparse_depth_loss(depth_raw, anchors, row_index)
                if depth_count:
                    loss = loss + sparse_depth_weight * depth_value
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizers['means'].param_groups[0]['lr'] = rates['means']*(0.01**(step/max(steps-1,1)))
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            if exposure_optimizer is not None:
                exposure_optimizer.step()
                exposure_optimizer.zero_grad(set_to_none=True)
            strategy.step_post_backward(params,optimizers,state,step,info)
            completed_step = step+1
            if completed_step % 100 == 0:
                print(json.dumps({'step':completed_step,'loss':float(loss.detach()),'splats':len(params['means'])}),flush=True)
            if completed_step % checkpoint_every == 0:
                save(completed_step)
            action = observe(float(loss.detach()))
        save(steps)
        # Unused SH bands are kept zero throughout training for lower degrees.
        export_ply(output/'model.ply',params,sh_degree)
        export_ply(output/'diagnostic-sh0.ply',params,0)
        reloaded = load_ply(output/'model.ply')
        with torch.no_grad():
            if any(not torch.equal(params[k],reloaded[k]) for k in params if k!='quats'):
                raise RuntimeError('PLY changed geometry, opacity or SH parameters')
            if not torch.allclose(F.normalize(params['quats'],dim=1),F.normalize(reloaded['quats'],dim=1),atol=1e-6,rtol=1e-6):
                raise RuntimeError('PLY changed quaternion orientation')
            a,_,_ = render(params,observations[0])
            b,_,_ = render(reloaded,observations[0])
            export_error = float((a-b).abs().max())
            export_rms = float((a-b).square().mean().sqrt())
        # Unit quaternions incur float32 renormalization rounding in CUDA.
        # Require exact remaining parameters, tiny RMS, and less than a quarter
        # of one 8-bit display level at every tested pixel.
        if export_error>1/(4*255) or export_rms>1e-6:
            raise RuntimeError(f'PLY roundtrip mismatch: max={export_error}, rms={export_rms}')
        del reloaded,a,b
        summary = dict(status='trained',steps=steps,mean_samples_per_image=steps/len(observations),
            elapsed_seconds=elapsed_before+time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(),
            timing_scope='cumulative_from_recorded_checkpoints' if not saved or 'elapsed_seconds' in saved else 'resume_only; earlier timing unavailable',
            splats=len(params['means']),export_roundtrip_max_error=export_error,
            export_roundtrip_rms_error=export_rms,
            model_sha256=sha256_file(output/'model.ply'),versions=versions)
        json_write(output/'training.json',summary)
        if run_evaluation:
            evaluate(package,meta,params,output/'evaluation',antialiased)
        summary['evaluation_status'] = 'completed' if run_evaluation else 'not_run'
        summary['status']='succeeded'
        json_write(output/'training.json',summary)
        if (output/'failure.json').exists():
            (output/'failure.json').rename(output/f'failure-before-success-{time.time_ns()}.json')
        return summary
    except Exception as error:
        json_write(output/'failure.json',dict(status='failed',error=str(error),completed_step=completed_step,
            recoverable_checkpoint=str(checkpoint),note='Resume only from the last atomically completed checkpoint; no automatic quality reduction.'))
        raise


def main():
    configure_windows_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--no-photo-comp',action='store_true')
    parser.add_argument('--use-bilateral-grid',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--use-sparse-depth',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--sparse-depth-weight',type=float,default=0.1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate-ply',type=Path)
    args = parser.parse_args()
    if args.steps is not None and args.steps<1:
        parser.error('steps must be positive')
    if args.sparse_depth_weight < 0:
        parser.error('sparse-depth-weight must be non-negative')
    if args.evaluate_ply:
        evaluate(args.dataset,validate_package(args.dataset),load_ply(args.evaluate_ply),args.output)
    else:
        try:
            train(args.dataset,args.output,args.steps,not args.no_photo_comp,args.resume,
                  use_bilateral_grid=args.use_bilateral_grid,use_sparse_depth=args.use_sparse_depth,
                  sparse_depth_weight=args.sparse_depth_weight)
        except Exception as error:
            # Initialization can fail before the first checkpoint (notably OOM).
            # Preserve the immutable input/config as the explicit restart point.
            if (args.output/'config.json').is_file() and not (args.output/'failure.json').exists():
                checkpoint=args.output/'checkpoint.pt'
                json_write(args.output/'failure.json',dict(status='failed',error=str(error),
                    recoverable_checkpoint=str(checkpoint) if checkpoint.is_file() else None,
                    recoverable_dataset=str(args.dataset.resolve()),
                    note='Free GPU memory, then resume a completed checkpoint or use a new output directory with the same input/config. No quality reduction applied.'))
            raise


if __name__=='__main__':
    main()
