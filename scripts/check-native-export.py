"""Measure native checkpoint/PLY invariants and unclipped render roundtrip error."""
from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
from gsdb.native_train import configure_windows_cuda,load_ply,render
from gsdb.training_data import json_write

parser=argparse.ArgumentParser()
parser.add_argument('--experiment',type=Path,required=True)
parser.add_argument('--dataset',type=Path,required=True)
args=parser.parse_args()
configure_windows_cuda()
saved=torch.load(args.experiment/'checkpoint.pt',map_location='cuda')['params']
loaded=load_ply(args.experiment/'model.ply')
meta=json.loads((args.dataset/'dataset.json').read_text(encoding='utf-8'))
row=next(r for r in meta['images'] if r['split']=='train')
with torch.no_grad():
    a,_,_=render(saved,row)
    b,_,_=render(loaded,row)
    diff=(a-b).abs()
    result=dict(max_error=float(diff.max()),mean_error=float(diff.mean()),rms_error=float(diff.square().mean().sqrt()),
        p999_error=float(torch.quantile(diff.flatten(),0.999)),fraction_over_quarter_8bit_level=float((diff>1/(4*255)).float().mean()),
        normalized_quaternion_max_error=float((F.normalize(saved['quats'],dim=1)-F.normalize(loaded['quats'],dim=1)).abs().max()),
        exact_parameter_equality={k:torch.equal(saved[k],loaded[k]) for k in saved if k!='quats'})
json_write(args.experiment/'export-diagnostic.json',result)
print(json.dumps(result,indent=2),flush=True)
