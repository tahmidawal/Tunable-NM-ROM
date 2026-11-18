"""Independent NumPy rescoring; no training imports and no final fields."""
from pathlib import Path
import hashlib,json,numpy as np
H=Path(__file__).resolve().parent
r=json.loads((H/'results/audit01/output/result.json').read_text())
p=H/'results/audit01/output/actual4096.npz';v=np.load(p)
u=v['prediction'];truth=v['reference'];assert u.shape==truth.shape==(4096,4096)
relative=float(np.sqrt(np.sum((u-truth)**2,dtype=np.float64)/np.sum(truth**2,dtype=np.float64)))
expected=r['dense_witness']['grids']['4096']['relative_l2']
assert abs(relative-expected)<1e-12,(relative,expected)
pointwise=float(np.max(np.abs(u-truth))/np.max(np.abs(truth)))
assert abs(pointwise-r['dense_witness']['grids']['4096']['relative_linf'])<1e-12
assert np.isfinite(u).all() and np.isfinite(truth).all()
boundary=float(max(abs(u[0]).max(),abs(u[-1]).max(),abs(u[:,0]).max(),abs(u[:,-1]).max()))
assert boundary<1e-12
# Explicitly reject the old initial-norm convention on a constructed decaying field.
initial_norm=np.linalg.norm(truth);current=.5*truth;prediction=current+.01*truth
correct=np.linalg.norm(prediction-current)/np.linalg.norm(current)
wrong=np.linalg.norm(prediction-current)/initial_norm
assert abs(correct-.02)<1e-12 and abs(wrong-.01)<1e-12
for name in ('refsdev01','refstrain01'):
    ref=json.loads((H/f'results/{name}/output/result.json').read_text())
    for row in ref['cases']:
        allowance=np.asarray(row['spatial_last'])+np.asarray(row['temporal_last'])+np.asarray(row['scoring_change'])
        np.testing.assert_allclose(allowance,row['allowance'],rtol=0,atol=1e-18)
        assert allowance.max()<=.001
result=dict(status='passed_independent_saved_array_audit',relative_l2=relative,relative_linf=pointwise,
            shape=list(u.shape),boundary_max=boundary,dense_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            current_time_normalization_fixture=float(correct),deliberately_wrong_initial_norm=float(wrong),
            reference_allowance_arithmetic_pass=True,final_fields_opened=False)
(H/'manifests/independent-array-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
