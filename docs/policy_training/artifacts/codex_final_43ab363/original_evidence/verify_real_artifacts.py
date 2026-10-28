import hashlib, json, os, subprocess, sys
from pathlib import Path
from collections import Counter
import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

TREE=Path('/tmp/r2r2r-codex-final-43ab363')
MAIN=Path('/home/stanley/Stanley_ws/real2render2real')
OUT=Path('/tmp/r2r2r-codex-final-evidence')
sys.path.insert(0,str(TREE/'policy_training'))
from r2r2r_dp.manifest import content_hash
from r2r2r_dp.verify import tree_fingerprint
from r2r2r_dp.launch_checks import check_launch
from r2r2r_dp.protocol import load_protocol, resolve

ref='2a0536689e81b1e01876c8cdf0550fa9fae20227'
candidate='43ab363058c4c9a5a45207b1414315320ef8438d'
def git(*args,cwd=MAIN):
 p=subprocess.run(['git',*args],cwd=cwd,capture_output=True,text=True,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
 return {'exit':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
rep={'candidate':git('rev-parse','HEAD',cwd=TREE),'reference':ref,
     'baseline_branch':git('rev-parse','repro/baseline-datagen'),
     'protected_diff':git('diff','--exit-code',ref,candidate,'--','scripts','real2render2real','dependencies','pyproject.toml','uv.lock','env_data_gen.sh','.gitmodules'),
     'recursive_submodules':git('submodule','status','--recursive'),
     'main_status_before':git('status','--short'),
     'trajgen_metadata':git('diff','--stat',cwd=MAIN/'dependencies/trajgen')}
art=TREE/'docs/policy_training/artifacts'
m=json.loads((art/'manifest_v1.json').read_text());by={e['uid']:e for e in m['episodes']}
rep['manifest_sha256']=hashlib.sha256((art/'manifest_v1.json').read_bytes()).hexdigest()
rep['source_fingerprint']=tree_fingerprint([str(Path(m['source_root'])/e['rel_path']) for e in m['episodes']])
rep['fingerprint_matches_preparation']=rep['source_fingerprint']=='dd23bd44ce626485a9f3ec8f78ce3c75cfeadb90785efdcf456c8f997c2f2bcd'
errors=[];arrays={};actions={};counts=Counter()
for e in m['episodes']:
 ep=Path(m['source_root'])/e['rel_path']
 with h5py.File(ep/'robot_data/robot_data.h5','r') as f:a={k:f[k][:] for k in ('ee_poses','joint_angles','gripper_binary_cmd')}
 if content_hash(a)!=e['content_sha256'] or not all(np.isfinite(v).all() for v in a.values()):errors.append([e['uid'],'content/nonfinite'])
 arrays[e['uid']]=a;counts[e['split']]+=1;parts=[]
 for side in [0,1]:
  ee=a['ee_poses'][:,7*side:7*(side+1)]
  r6=Rotation.from_quat(ee[:,[4,5,6,3]]).as_matrix()[:,:2,:].reshape(-1,6)
  parts.append(np.c_[ee[:,:3],r6,a['gripper_binary_cmd'][:,side,None]])
 actions[e['uid']]=np.concatenate(parts,1)
rep['source_counts']=dict(counts)
views=MAIN.parent/'r2r2r_policy_runs/dataset_v1/views';nv=0;ns=0
for label,uids in [('val',[e['uid'] for e in m['episodes'] if e['split']=='val']),('test',[e['uid'] for e in m['episodes'] if e['split']=='test'])]+list(m['train_subsets'].items()):
 root=views/'splits'/label if label in ('val','test') else views/'train_subsets'/f'n{int(label):04d}'
 wanted={f"ep_{by[u]['split']}_{by[u]['split_rank']:04d}" for u in uids}
 if {p.name for p in root.iterdir()}!=wanted:errors.append([label,'view membership'])
 for uid in uids:
  e=by[uid];ep=root/f"ep_{e['split']}_{e['split_rank']:04d}";src=Path(m['source_root'])/e['rel_path'];nv+=1
  for rel in ['robot_data/robot_data.h5','camera_0','camera_1']:
   if not (ep/rel).is_symlink() or (ep/rel).resolve()!=(src/rel).resolve():errors.append([label,uid,rel])
  for key,suffix in [('ee_poses','proprio'),('joint_angles','joint'),('gripper_binary_cmd','gripper_cmd')]:
   z=zarr.load(str(ep/f'robot_data/robot_data_{suffix}.zarr'));ns+=1
   if z.dtype!=arrays[uid][key].dtype or not np.array_equal(z,arrays[uid][key]):errors.append([label,uid,key])
rep['views_checked']=nv;rep['sidecars_checked']=ns
stats=[]
for n,uids in sorted(m['train_subsets'].items(),key=lambda x:int(x[0])):
 st=json.loads((art/f'normalizer_v2/action_statistics_v2_train_n{int(n):04d}.json').read_text());lo=np.array(st['min_action']);hi=np.array(st['max_action'])
 raw=np.concatenate([actions[u][2:159] for u in uids]);supervised=np.concatenate([actions[u][8:159] for u in uids])
 expected_lo=raw.min(0);expected_hi=raw.max(0)
 z=2*(supervised[:,10:20]-lo[10:20])/(hi[10:20]-lo[10:20])-1
 row={'n':int(n),'min_bound_error':float(np.abs(expected_lo-lo).max()),'max_bound_error':float(np.abs(expected_hi-hi).max()),'supervised_rows':len(z),
      'supervised_rows_outside':int((np.abs(z)>1+1e-6).any(1).sum()),'max_normalized_abs':float(np.abs(z).max())}
 v1=json.loads((art/f'action_statistics_train_n{int(n):04d}.json').read_text())
 if int(n)==150:
  uid='run06_env2_gpu0_noprealloc/env_1_2026_10_22_20_06_53';value=actions[uid][15,13]
  row['original_row15_v1']=float(2*(value-v1['min_action'][13])/(v1['max_action'][13]-v1['min_action'][13])-1)
  row['original_row15_v2']=float(2*(value-lo[13])/(hi[13]-lo[13])-1)
 stats.append(row)
rep['independent_normalizers']=stats
rep['original_artifacts_diff']=git('diff','--exit-code',ref,candidate,'--','docs/policy_training/artifacts/manifest_v1.json',*[f'docs/policy_training/artifacts/action_statistics_train_n{n:04d}.json' for n in (50,100,150,1000,1461)])
proto=load_protocol(str(TREE/'policy_training/configs/baseline_v2.json'))
rep['real_launch_checks']={}
for n in [50,100,150,1000,1461]:
 try:rep['real_launch_checks'][n]=check_launch(resolve(proto,n,f'/tmp/r2r2r-codex-final-evidence/uncreated-real-n{n}'))
 except Exception as e:rep['real_launch_checks'][n]={'error':repr(e)}
rep['errors']=errors
rep['main_status_after']=git('status','--short')
(OUT/'real_artifact_verification.json').write_text(json.dumps(rep,indent=2))
print(json.dumps(rep,indent=2))
