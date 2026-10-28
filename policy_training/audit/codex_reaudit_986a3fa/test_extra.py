import copy,json
from pathlib import Path
import pytest,torch
from run_fixtures import make_run_fixture
from r2r2r_dp import driver
from r2r2r_dp.protocol import resolve

@pytest.fixture(scope='module')
def trained(tmp_path_factory):
 p=tmp_path_factory.mktemp('independent-v3');fx=make_run_fixture(p/'data',total_updates=3,warmup_updates=0)
 def new(out):
  r=resolve(fx['protocol'],3,str(out));driver.write_run_files(r);return driver.Trainer(r,device='cpu')
 t=new(p/'train');t.train();cp=torch.load(Path(t.out)/'ckpt_full_u0000003.pt',weights_only=False)
 return cp,new,p

@pytest.mark.parametrize('case',['optimizer_nan_lr','optimizer_missing_moment','optimizer_bad_moment_shape','scheduler_nan_lr','invalid_rng_payload'])
def test_corrupt_checkpoint_rejected(trained,case):
 cp,new,p=trained;st=copy.deepcopy(cp);t=new(p/case)
 if case=='optimizer_nan_lr': st['optimizer']['param_groups'][0]['lr']=float('nan')
 elif case=='optimizer_missing_moment':del next(iter(st['optimizer']['state'].values()))['exp_avg']
 elif case=='optimizer_bad_moment_shape':next(iter(st['optimizer']['state'].values()))['exp_avg']=torch.zeros(2,3)
 elif case=='scheduler_nan_lr':st['scheduler']['_last_lr']=[float('nan')]*len(st['scheduler']['_last_lr'])
 else:st['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
 path=p/(case+'.pt');torch.save(st,path)
 problems=t.checkpoint_problems(st)
 print(case,'preflight_problems=',problems)
 with pytest.raises((driver.CheckpointError,ValueError,RuntimeError,KeyError)):
  t.load_full(str(path))
