"""Independent reference and checkpoint checks for an incomplete attempt."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from audit import stencil,head

p=argparse.ArgumentParser();p.add_argument('collected');a=p.parse_args();root=Path(a.collected)
r=json.loads((root/'out/result.json').read_text());cfg=r['config']
assert r['backend']=='gpu' and r['x64'] and r['precision']=='highest'
assert not r['complete'] and r['stage']=='directions' and not r['invocations']
checkpoint=root/'training/checkpoint.pkl'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
ck=pickle.loads(checkpoint.read_bytes());bank=pickle.loads((root/'training/bank.pkl').read_bytes())
params=ck['params'];coords=np.random.default_rng(12).uniform(.05,.95,(19,3));angles=2*np.pi*coords@params['B']
x=np.concatenate([np.sin(angles),np.cos(angles)],axis=-1)
for w,b in params['g'][:-1]:
    x=x@w+b;x=x/(1+np.exp(-x))
w,b=params['g'][-1];features=(x@w+b)*(64*np.prod(coords*(1-coords),axis=1)*params['out_scale'])[:,None]
assert np.isfinite(features).all() and np.isfinite(head(params,ck['Z_tr'][:19])).all()
q=bank['Q'];orth=float(np.linalg.norm(q.T@q-np.eye(q.shape[1])))
assert orth<1e-10
maximum=0.
for case in range(len(cfg['validation_rows'])):
    data=np.load(root/f'out/reference_case{case}.npz');fields=data['fields'];nu=float(data['nu'])
    assert np.array_equal(fields[0],data['u0'])
    for i in range(1,len(fields)):
        adv,lap=stencil(fields[i],cfg['nodes'])
        defect=fields[i]-fields[i-1]+cfg['dt']*(adv-nu*lap)
        maximum=max(maximum,float(np.linalg.norm(defect)/np.linalg.norm(fields[i-1])))
assert maximum<2e-9
record=dict(partial_checks_passed=True,complete_panel=False,reference_cases=len(cfg['validation_rows']),
    max_reference_defect=maximum,bank_orthogonality_frobenius=orth,
    retained_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    scope='Saved reference defects, bank orthogonality and finite independent network evaluation only. No reduced trajectory or complete panel result exists.')
(root/'out/audit-partial-local.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
