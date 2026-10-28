import importlib.util,os,sys,subprocess,tempfile,shutil,json
from pathlib import Path
root=Path('/tmp/r2r2r-v3-audit-worktree');out=Path('/tmp/r2r2r-codex-reaudit-986a3fa');sys.path.insert(0,str(root/'policy_training/tools'))
import mutation_check as m
py='/home/stanley/Stanley_ws/real2render2real/.venv-tinydp/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(root/'dependencies/tinydp'))
rows=[]
for i,mut in enumerate([('CONTROL',None,None,None,None)]+m.MUTATIONS):
 name,rel,old,new,test=mut
 with tempfile.TemporaryDirectory(prefix='v3mut-') as d:
  w=Path(d)/'policy_training';shutil.copytree(root/'policy_training',w,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  for n in ['dependencies','docs']: (Path(d)/n).symlink_to(root/n,target_is_directory=True)
  if rel:
   p=w/rel;s=p.read_text();assert s.count(old)==1,(name,s.count(old));p.write_text(s.replace(old,new))
  tests=[test] if test else sorted({x[4] for x in m.MUTATIONS})
  r=subprocess.run([py,'-B','-m','pytest','-q','-x','-p','no:cacheprovider',*tests],cwd=w,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
  (out/f'mutation-{i:02d}.log').write_text(r.stdout)
  rec={'i':i,'name':name,'exit':r.returncode,'tail':r.stdout.splitlines()[-1:]};rows.append(rec);print(json.dumps(rec),flush=True)
  (out/'mutation_results.json').write_text(json.dumps(rows,indent=2))
