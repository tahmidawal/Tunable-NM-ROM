"""Provisional algebra on one owner-verified closed checkpoint, no network evaluation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from audit_saved_blocks import summarize,digest
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--binding-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert sha(a.input/'BINDING.json')==a.binding_sha256
b=json.loads((a.input/'BINDING.json').read_text());assert b['closed_published_checkpoint_only'] is True and b['iteration']==1000 and b['status']=='checksum_verified_closed_checkpoint'
assert b['checkpoint_sha256']=='ea4b686d863820891e01392dd16b9443bae87eeb1c40df4c85e99e7b4141d54a'
paths=[r['path'] for r in b['files']];assert len(paths)==len(set(paths)) and {'results/fit/state-01000-periodic.json','results/fit/state-01000-periodic.npz'}<=set(paths)
provenance={}
for row in b['files']:
 q=a.input/row['path'];assert q.resolve().is_relative_to(a.input.resolve()) and not q.is_symlink();assert q.stat().st_size==row['bytes'] and sha(q)==row['sha256'];provenance[row['path']]=row['sha256']
r=json.loads((a.input/'results/fit/state-01000-periodic.json').read_text());assert r['iteration']==1000 and r['label']=='periodic' and r['array_sha256']==b['checkpoint_sha256'] and r['array_file']=='state-01000-periodic.npz'
assert sha(a.input/'results/fit'/r['array_file'])==r['array_sha256']==b['checkpoint_sha256']
with np.load(a.input/'results/fit'/r['array_file'],allow_pickle=False) as z:t=z['theta'].copy();g=z['gradient'].copy()
assert digest(t)==r['theta_sha256'] and digest(g)==r['gradient_sha256'];assert np.isclose(np.max(abs(g)),r['gradient_inf'],rtol=1e-13,atol=1e-16)
result={'schema':'ns-obstacle-provisional-closed-checkpoint-parameter-blocks-v1','passed':True,'terminal_model_audit':False,'physical_admission':False,'iteration':1000,'binding_sha256':a.binding_sha256,'provenance':provenance,'source_sha256':sha(Path(__file__)),'algebra_source_sha256':sha(Path(__file__).with_name('audit_saved_blocks.py')),'theta_sha256':digest(t),'gradient_sha256':digest(g),'child_parameters':22792,'parent_scope':'Frozen parent excluded from this child checkpoint; parent gradients and norms not inferred.','diagnostic':summarize(t,g),'limitations':['One immutable completed checkpoint only; running optimizer state was not accessed.','No initial/final comparison or terminal model acceptance.','Parameter/gradient norms are coordinate dependent, not field amplitudes or Hessian condition estimates.','Exact-zero support is not a smallness threshold.','No neural evaluation, training, solver, validation, scale choice or intervention.']}
assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'result_sha256':sha(a.output),'passed':True}))
