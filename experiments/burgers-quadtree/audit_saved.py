"""Independent NumPy rescoring of saved projection predictions and reference residuals."""
from pathlib import Path
import json,numpy as np,hashlib
H=Path(__file__).resolve().parent;root=H/'results/representation01/output'
U=np.load(root/'snapshots.npz')['fields'];raw=U[:,:,1:-1,1:-1].reshape(40,11,-1);norm=np.linalg.norm(raw[:,0],axis=1)
y=(raw[32:]/norm[32:,None,None]).reshape(88,-1).T
result=json.loads((root/'result.json').read_text());d=[]
for row in result['rows']:
 p=np.load(root/f"{row['kind']}_d{row['depth']}_k{row['k']}.npz")['prediction']
 err=np.linalg.norm(p-y,axis=0).reshape(8,11)
 d.append(float(np.max(np.abs(err-np.asarray(row['per_time'])))))
 # Deliberately wrong trajectory normalization must change reported error.
 bad=np.linalg.norm(p-2*y,axis=0)
 assert np.max(np.abs(bad-err.ravel()))>.01
r=json.loads((H/'results/reference01/output/result.json').read_text())
res=max(x.get('residual',0) for x in r['cases']);assert res<=1e-11
assert max(d)<1e-12
out=dict(independent_projection_rescore_max_abs=max(d),records=len(d),deliberate_wrong_normalization_detected=True,maximum_reference_residual=res,passed=True)
(H/'evidence/independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
