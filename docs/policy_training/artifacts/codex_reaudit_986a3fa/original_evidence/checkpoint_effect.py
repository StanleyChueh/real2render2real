import copy,json
from pathlib import Path
import torch
from run_fixtures import make_run_fixture
from r2r2r_dp import driver
from r2r2r_dp.protocol import resolve
out=Path('/tmp/r2r2r-codex-reaudit-986a3fa/checkpoint-effect');out.mkdir()
fx=make_run_fixture(out/'data',total_updates=4,warmup_updates=0)
def new(name):
 r=resolve(fx['protocol'],3,str(out/name));driver.write_run_files(r);return driver.Trainer(r,device='cpu')
t=new('prepare');t.train(stop_after=3);cp=torch.load(Path(t.out)/'ckpt_full_u0000003.pt',weights_only=False)
results=[]
for case in ['valid','missing_adam_state','bad_moment_shape','invalid_rng','nan_lr']:
 st=copy.deepcopy(cp)
 if case=='missing_adam_state':del st['optimizer']['state'][next(iter(st['optimizer']['state']))]
 elif case=='bad_moment_shape':next(iter(st['optimizer']['state'].values()))['exp_avg']=torch.zeros(2,3)
 elif case=='invalid_rng':st['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
 elif case=='nan_lr':st['optimizer']['param_groups'][0]['lr']=float('nan')
 p=out/(case+'.pt');torch.save(st,p);u=new(case);rec={'case':case,'preflight_problems':u.checkpoint_problems(st)}
 try:
  u.load_full(str(p));rec['load_accepted']=True;rec['summary']=u.train()
  rec['finite_model']=all(torch.isfinite(v).all().item() for v in u.model.state_dict().values());rec['final_checkpoint_problems']=u.checkpoint_problems(torch.load(Path(u.out)/'ckpt_full_u0000004.pt',weights_only=False))
 except Exception as e:rec['exception']=type(e).__name__+': '+str(e)
 results.append(rec)
(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
