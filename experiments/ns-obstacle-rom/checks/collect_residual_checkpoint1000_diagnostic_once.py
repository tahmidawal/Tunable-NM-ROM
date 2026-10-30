"""Root-authorized closed checkpoint copy, not model selection or evaluation."""
from pathlib import Path
import subprocess,json,hashlib,shutil,time
B=Path(__file__).resolve().parents[1];out=B/'checks/residual-checkpoint1000-diagnostic';remote='/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/neural-pod-guided-residual01'
expected='ea4b686d863820891e01392dd16b9443bae87eeb1c40df4c85e99e7b4141d54a';limit=8*2**20
assert not out.exists() and shutil.disk_usage(B).free>=10*2**30+256*2**20+2*limit
code=f'''from pathlib import Path
import json,hashlib
p=Path({remote!r});names=['results/fit/state-01000-periodic.npz','results/fit/state-01000-periodic.json','results/fit/CONFIG.json','results/fit/RUNTIME.json','TRAINING-RELEASE.json','SOURCE_COMMIT.txt','SOURCE-SHA256SUMS'];rows=[]
for n in names:
 x=p/n;assert x.is_file() and not x.is_symlink()
 with x.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 rows.append({{'path':n,'bytes':x.stat().st_size,'sha256':h}})
r=json.loads((p/names[1]).read_text());assert r['iteration']==1000 and r['label']=='periodic' and r['array_file']=='state-01000-periodic.npz' and r['array_sha256']=={expected!r} and rows[0]['sha256']=={expected!r}
assert sum(x['bytes'] for x in rows)<={limit}
print(json.dumps({{'files':rows,'iteration':1000,'checkpoint_sha256':{expected!r},'closed_published_checkpoint_only':True,'read_scope':'immutable periodic checkpoint, not latest/current optimizer state'}}))
'''
raw=subprocess.check_output(['ssh','tufts-login','/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python','-'],input=code.encode());binding=json.loads(raw);out.mkdir();binding.update(local_free_before_bytes=shutil.disk_usage(B).free,maximum_collection_bytes=limit,root_authorized=True,purpose='provisional per-layer parameter/gradient norm diagnosis only; no fit/NN/solver evaluation or intervention')
for row in binding['files']:
 dest=out/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['scp','-q','tufts-login:'+remote+'/'+row['path'],str(dest)],check=True)
 with dest.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert dest.stat().st_size==row['bytes'] and h==row['sha256']
# Re-read the published remote hashes after transfer to exclude changed inputs.
assert subprocess.check_output(['ssh','tufts-login','/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python','-'],input=code.encode())==raw
binding.update(status='checksum_verified_closed_checkpoint',completed_epoch=time.time(),logical_bytes=sum(x['bytes'] for x in binding['files']));(out/'BINDING.json').write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(binding,indent=2))
