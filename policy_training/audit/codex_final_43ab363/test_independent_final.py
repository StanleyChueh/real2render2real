"""Independent tests against 43ab363. Writes synthetic fixtures/evidence only under /tmp.

Assertions express required behavior: a failing test is an observed audit defect,
not an expected-failure marker that makes the suite look green.
"""
import copy
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import types

import h5py
import numpy as np
import pytest
import torch
import zarr

import train_r2r2r_dp
from run_fixtures import make_run_fixture
from r2r2r_dp import driver, normalizer_v2 as nv, tinydp_bridge as tb
from r2r2r_dp.inference import R2R2RPolicy
from r2r2r_dp.launch_checks import check_launch, LaunchError
from r2r2r_dp.protocol import resolve, ProtocolError, config_from_dict
from tinydp.dataset.utils import default_vision_transform, rot_6d_to_rot_mat
from tinydp.util.ema import ModelEMA
from tinydp.util.misc import NativeScalerWithGradNormCount

EVIDENCE = Path('/tmp/r2r2r-codex-final-evidence')

def observe(name, **values):
    with (EVIDENCE/'observations.jsonl').open('a') as f:
        f.write(json.dumps({'name': name, **values}, allow_nan=True)+'\n')

@pytest.fixture
def fresh(tmp_path):
    return make_run_fixture(tmp_path/'fixture')

@pytest.fixture(scope='module')
def stable(tmp_path_factory):
    return make_run_fixture(tmp_path_factory.mktemp('independent_stable'), total_updates=3,
                            warmup_updates=0, full_val_every_updates=100, val_every_updates=100)

def run_of(fx, out):
    return resolve(fx['protocol'], 3, str(out))

def trainer(fx, out):
    r=run_of(fx,out);driver.write_run_files(r)
    return driver.Trainer(r,device='cpu')

class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__();self.w=torch.nn.Parameter(torch.tensor([[0.7]],dtype=torch.float64))
    def forward(self,b):return ((b['x']@self.w-b['y'])**2).mean()

class CountingAdam(torch.optim.AdamW):
    def __init__(self,p):super().__init__(p,lr=.01,weight_decay=.09);self.steps=0
    def step(self,*a,**kw):self.steps+=1;return super().step(*a,**kw)

def scalar_driver(out, batches, accum):
    t=driver.Trainer.__new__(driver.Trainer)
    t.out=str(out);out.mkdir();t.device=torch.device('cpu');t.amp=False
    t.sched={'total_updates':1,'accum_micro_batches':accum,'log_every_updates':1,'grad_clip':100.,
             'val_every_updates':100,'full_val_every_updates':100,'inference_export_every_updates':100,
             'full_ckpt_every_updates':100}
    t.model=Scalar();t.optimizer=CountingAdam(t.model.parameters())
    t.scheduler=torch.optim.lr_scheduler.LambdaLR(t.optimizer,lambda _:1.)
    t.scaler=NativeScalerWithGradNormCount();t.ema=ModelEMA(t.model,decay=.5)
    t.update=t.micro=t.epoch=t.batch_in_epoch=0;t.batches_per_epoch=100
    t.ds_train=None;t.sampler=None;t.val_subset=t.val_full=[]
    t._loader=lambda *a,**kw:iter(itertools.cycle(batches))
    t.evaluate=lambda *a,**kw:{'loss':0.,'windows':0,'weights':'ema'}
    t.save_full=lambda:'diagnostic-not-a-checkpoint';t.save_inference=lambda:'diagnostic-not-an-export'
    return t

def test_amp_skipped_step_does_not_advance_successful_update(tmp_path):
    b={'x':torch.ones((4,1),dtype=torch.float64),'y':torch.zeros((4,1),dtype=torch.float64)}
    t=scalar_driver(tmp_path/'amp',[b],4)
    t.scaler._scaler=torch.amp.GradScaler('cpu')  # real GradScaler, actual committed train() logic
    hook_calls=[0];ema_calls=[0];orig=t.ema.update
    def bad_first_group(g):
        hook_calls[0]+=1
        return torch.full_like(g,float('inf')) if hook_calls[0]<=4 else g
    t.model.w.register_hook(bad_first_group)
    def counted_ema(m):ema_calls[0]+=1;orig(m)
    t.ema.update=counted_ema
    result=t.train()
    observe('amp_skip',optimizer_steps=t.optimizer.steps,scheduler_steps=t.scheduler.last_epoch,
            ema_updates=ema_calls[0],successful_updates=t.update,micro_batches=t.micro,
            scaler_scale=t.scaler._scaler.get_scale(),summary=result)
    assert t.update==t.optimizer.steps==t.scheduler.last_epoch==ema_calls[0]==1
    assert t.micro==8  # one skipped four-microbatch attempt, then one successful attempt

def test_accumulated_adam_matches_large_batch(tmp_path):
    x=torch.arange(1,17,dtype=torch.float64).reshape(-1,1)/10;y=torch.sin(x)
    a=scalar_driver(tmp_path/'large',[{'x':x,'y':y}],1)
    b=scalar_driver(tmp_path/'micro',[{'x':x[i:i+4],'y':y[i:i+4]} for i in range(0,16,4)],4)
    a.train();b.train()
    for key in ['w']:
        assert torch.allclose(a.model.state_dict()[key],b.model.state_dict()[key],rtol=1e-10,atol=1e-10)
    for sa,sb in zip(a.optimizer.state.values(),b.optimizer.state.values()):
        for k in sa:assert torch.allclose(sa[k],sb[k],rtol=1e-10,atol=1e-10)
    observe('accumulation_equivalence',large_w=a.model.w.item(),micro_w=b.model.w.item(),
            large_steps=a.optimizer.steps,micro_steps=b.optimizer.steps)

def test_ema_scalar_oracle():
    m=Scalar();m.w.data.zero_();e=ModelEMA(m,decay=.5)
    m.w.data.fill_(10);e.update(m);assert e.ema['w'].item()==5
    m.w.data.fill_(20);e.update(m);assert e.ema['w'].item()==12.5
    observe('ema_scalar',ema=e.ema['w'].item())

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')],ids=['nan','inf','neg_inf'])
def test_normalization_rejects_nonfinite(bad):
    with pytest.raises((ValueError,FloatingPointError)):
        z=nv.normalized(np.array([[bad]]),np.array([0.]),np.array([1.]))
        observe('nonfinite_normalization',input=bad,output=z.tolist())

def test_bound_builder_cannot_certify_nan_data(fresh):
    root=Path(fresh['views'])/'train_subsets/n0003';ep=next(root.iterdir())
    z=ep/'robot_data/robot_data_proprio.zarr';a=zarr.load(str(z));a[21,7]=float('nan');zarr.save_array(str(z),a)
    with pytest.raises((ValueError,FloatingPointError)):
        st=nv.build_statistics(str(root),'n0003',fresh['manifest']['train_subsets']['3'],nv.file_sha256(fresh['manifest_path']),2,4,16)
        cov=nv.coverage(nv.view_episode_h5s(str(root)),st,2,4,16)
        observe('nan_bounds',has_nan_bounds=not np.isfinite(st['min_action']).all(),coverage=cov)

@pytest.mark.parametrize('field',['version','counts','salt'])
def test_manifest_metadata_rejected_even_when_re_pinned(fresh,tmp_path,field):
    m=json.load(open(fresh['manifest_path']))
    m[field]={'train':999,'val':999,'test':999} if field=='counts' else (999 if field=='version' else 'wrong-salt')
    Path(fresh['manifest_path']).write_text(json.dumps(m));sha=nv.file_sha256(fresh['manifest_path'])
    p=copy.deepcopy(fresh['protocol']);p['data']['manifest_sha256']=sha
    sp=Path(fresh['stats_dir'])/'action_statistics_v2_train_n0003.json'
    st=json.loads(sp.read_text());st['provenance']['manifest_sha256']=sha;sp.write_text(json.dumps(st))
    r=resolve(p,3,str(tmp_path/'out'))
    with pytest.raises((LaunchError,ValueError,AssertionError)):
        rep=check_launch(r);observe('accepted_manifest_metadata',field=field,counts=rep['counts'])

def dataset_order(r):
    cfg=config_from_dict(r['tinydp']);tb.install_ordered_glob()
    ds=tb.make_manifest_dataset_class(r['paths']['train_root'],r['paths']['eval_root'],r['paths']['action_statistics'])(
        cfg.dataset_cfg,cfg.shared_cfg,cfg.logging_cfg,default_vision_transform(32),split='train')
    return [(os.path.realpath(p),s,e) for p,s,e in ds.start_end]

def test_changed_view_order_rejected(fresh,tmp_path):
    r=run_of(fresh,tmp_path/'out');before=dataset_order(r)
    root=Path(r['paths']['train_root']);first=sorted(root.iterdir())[0];first.rename(root/'zz_renamed_episode')
    after=dataset_order(r);assert before!=after
    with pytest.raises((LaunchError,ValueError)):
        rep=check_launch(r);observe('accepted_order_change',order_changed=before!=after,
                                    same_config_sha256=run_of(fresh,tmp_path/'other')['config_sha256']==r['config_sha256'],counts=rep['counts'])

def test_corrupt_jpeg_rejected_before_model(fresh,tmp_path):
    root=Path(fresh['views'])/'train_subsets/n0003';ep=next(root.iterdir())
    (ep/'camera_0/rgb/0007.jpg').write_bytes(b'not a jpeg')  # only synthetic source fixture
    with pytest.raises((LaunchError,ValueError)):
        rep=check_launch(run_of(fresh,tmp_path/'out'));observe('accepted_corrupt_jpeg',counts=rep['counts'])

def test_wrong_action_layout_rejected(fresh,tmp_path):
    sp=Path(fresh['stats_dir'])/'action_statistics_v2_train_n0003.json'
    st=json.loads(sp.read_text());st['config']['layout']=list(reversed(st['config']['layout']));sp.write_text(json.dumps(st))
    with pytest.raises((LaunchError,ValueError)):
        rep=check_launch(run_of(fresh,tmp_path/'out'));observe('accepted_wrong_layout',counts=rep['counts'])

@pytest.mark.parametrize('case',['negative_clip','fractional_batch','nan_lr','bool_updates'])
def test_protocol_invalid_numerics_rejected(stable,tmp_path,case):
    p=copy.deepcopy(stable['protocol'])
    if case=='negative_clip':p['schedule']['grad_clip']=-1.
    elif case=='fractional_batch':p['tinydp']['shared_cfg.batch_size']=2.5
    elif case=='nan_lr':p['tinydp']['optimizer_cfg.lr']=float('nan')
    else:p['schedule']['total_updates']=True;p['schedule']['warmup_updates']=0
    with pytest.raises((ProtocolError,ValueError)):
        r=resolve(p,3,str(tmp_path/'out'));observe('accepted_protocol_numeric',case=case)

def test_validation_preserves_all_rng_and_raw_weights(stable,tmp_path):
    t=trainer(stable,tmp_path/'normal_val');t.model.train()
    raw={k:v.clone() for k,v in t.model.state_dict().items()}
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    rep=t.evaluate(t.val_subset)
    assert before[0]==random.getstate()
    after=np.random.get_state();assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]
    assert torch.equal(before[2],torch.get_rng_state()) and t.model.training
    assert all(torch.equal(v,t.model.state_dict()[k]) for k,v in raw.items())
    observe('normal_validation_rng_isolation',loss=rep['loss'],windows=rep['windows'])

def test_validation_exception_restores_weights_and_mode(stable,tmp_path):
    t=trainer(stable,tmp_path/'exception_val');t.model.train()
    with torch.no_grad():next(t.model.parameters()).add_(.5)
    raw={k:v.clone() for k,v in t.model.state_dict().items()}
    def explode(_):raise RuntimeError('injected validation failure')
    t.model.forward=explode
    with pytest.raises(RuntimeError,match='injected'):t.evaluate(t.val_subset)
    restored=all(torch.equal(v,t.model.state_dict()[k]) for k,v in raw.items())
    observe('validation_exception',raw_weights_restored=restored,training_mode=t.model.training)
    assert restored and t.model.training

def test_validation_nonfinite_loss_rejected(stable,tmp_path):
    t=trainer(stable,tmp_path/'nan_val');t.model.forward=lambda _:torch.tensor(float('nan'))
    with pytest.raises((FloatingPointError,ValueError,RuntimeError)):
        rep=t.evaluate(t.val_subset);observe('accepted_nan_validation',loss=rep['loss'])

def test_validation_partial_batch_weighting(stable,tmp_path):
    t=trainer(stable,tmp_path/'weighted_val')
    batches=driver.window_batches(t.ds_val,4,3,10)
    t.model.forward=lambda b:torch.tensor(1. if b['proprio'].shape[0]==3 else 9.)
    rep=t.evaluate(batches);observe('weighted_validation',**rep)
    assert rep['windows']==4 and rep['loss']==pytest.approx(3.)

@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root=tmp_path_factory.mktemp('independent_trained')
    fx=make_run_fixture(root,total_updates=3,warmup_updates=0,full_val_every_updates=100,val_every_updates=100)
    t=trainer(fx,root/'trained');t.train()
    return fx,t,root

def test_checkpoint_missing_ema_parameter_rejected(trained,tmp_path):
    fx,t,root=trained
    cp=torch.load(Path(t.out)/'ckpt_full_u0000003.pt',weights_only=False)
    missing=next(iter(cp['ema']));del cp['ema'][missing];path=tmp_path/'missing_ema.pt';torch.save(cp,path)
    u=trainer(fx,tmp_path/'ema_load')
    with pytest.raises((ValueError,KeyError,RuntimeError)):
        u.load_full(str(path));observe('accepted_missing_ema',missing_key=missing)

def test_inference_relocated_without_source_data(trained,tmp_path):
    fx,t,root=trained
    artifact=tmp_path/'relocated.pt';shutil.copyfile(Path(t.out)/'policy_ema_u0000003.pt',artifact)
    # Make every embedded external path unavailable; restore paths after the test.
    locations=[Path(fx[k]) for k in ['source','views','stats_dir']]+[Path(fx['manifest_path'])]
    moved=[]
    try:
        for p in locations:
            q=p.with_name(p.name+'.temporarily_unavailable');p.rename(q);moved.append((p,q))
        pol=R2R2RPolicy.load(str(artifact))
        obs=torch.zeros((1,4,2,3,32,32));prop=torch.zeros((1,4,20))
        a=pol.predict(obs,prop,4,seed=123);b=pol.predict(obs,prop,4,seed=123)
        assert torch.equal(a['normalized'],b['normalized']) and torch.isfinite(a['right_action']).all()
        observe('inference_portability',shape=list(a['right_action'].shape),all_external_paths_unavailable=True)
    finally:
        for p,q in reversed(moved):q.rename(p)

def test_inference_nonfinite_input_rejected(trained):
    fx,t,root=trained;pol=R2R2RPolicy.load(str(Path(t.out)/'policy_ema_u0000003.pt'))
    obs=torch.zeros((1,4,2,3,32,32));prop=torch.zeros((1,4,20));prop[0,0,0]=float('nan')
    with pytest.raises((ValueError,FloatingPointError,RuntimeError)):
        rep=pol.predict(obs,prop,4,seed=3);observe('accepted_nan_inference',normalized_finite=bool(torch.isfinite(rep['normalized']).all()),
                                                  physical_action_finite=bool(torch.isfinite(rep['right_action']).all()))

@pytest.mark.parametrize('v',[[2.,0,0,0,3.,0],[1.,0,0,1.,0,0],[0.,0,0,0,0,0]],ids=['nonunit','collinear','zero'])
def test_inherited_physical_rotation_conversion_is_valid_or_rejects(v):
    try:
        with np.errstate(divide='ignore',invalid='ignore'):R=rot_6d_to_rot_mat(np.array([v]))
    except (ValueError,FloatingPointError):return
    finite=bool(np.isfinite(R).all());det=float(np.linalg.det(R)[0]) if finite else None
    observe('inherited_rotation',input=v,finite=finite,determinant=det)
    assert finite and np.allclose(R@R.transpose(0,2,1),np.eye(3),atol=1e-6) and abs(det-1)<1e-6

def equal_nested(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b

def test_cpu_resume_across_epoch_boundary(tmp_path):
    fx=make_run_fixture(tmp_path/'resume_fixture',total_updates=9,warmup_updates=3,
                        full_ckpt_every_updates=3,full_val_every_updates=9,inference_export_every_updates=9)
    a,b=tmp_path/'continuous',tmp_path/'interrupted'
    def cli(out,*extra):
        return train_r2r2r_dp.main(['--protocol',fx['protocol_path'],'--train-size','3','--output-dir',str(out),
                                     '--device','cpu',*extra])
    cli(a);cli(b,'--stop-after-updates','7')
    (b/'ckpt_full_u0000007.pt').unlink()  # durable u6 is before the boundary at micro 30
    cli(b,'--resume','--num-workers','2')
    A=torch.load(a/'ckpt_full_u0000009.pt',weights_only=False);B=torch.load(b/'ckpt_full_u0000009.pt',weights_only=False)
    for k in ['model','ema','optimizer','scheduler','data_position']:assert equal_nested(A[k],B[k]),k
    assert A['micro_batches']==B['micro_batches']==36
    assert A['data_position']['epoch']==1 and A['data_position']['batch']==6
    observe('cross_epoch_cpu_resume',bitwise_equal=True,updates=9,micro_batches=36,data_position=A['data_position'])
