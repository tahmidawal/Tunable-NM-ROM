"""Independent saved-field residual and SciPy CG counter replay (no JAX)."""
import json
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def stencil(u,n):
    out=6*u.copy()
    for axis in range(3):
        lo=[slice(None)]*3;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        out[tuple(lo)]-=u[tuple(hi)];out[tuple(hi)]-=u[tuple(lo)]
    return n*n*out


def audit(out,record):
    checked=0;replays=0;maxdefect=0.;failed=0
    spec=record['config']['iterative_cg']
    assert spec['preconditioner']=='identity' and spec['relative_tolerances']==[1e-2,1e-4,1e-6]
    for mesh in record['meshes']:
        expected={f'cg_identity_rtol{tol:.0e}' for tol in spec['relative_tolerances']}
        assert expected=={name for name in mesh['methods'] if name.startswith('cg_')}
    for row in record['invocations']:
        if not row['method'].startswith('cg_'):continue
        n=row['intervals'];tol=row['cg_relative_tolerance'];cap=row['cg_max_iterations']
        assert tol in spec['relative_tolerances'] and cap==n*spec['max_iterations_per_interval']
        assert row['cg_preconditioner']=='identity'
        k=row['iterations'];assert 0<=k<=cap and row['cg_matvecs']==k+1
        assert row['cg_hit_iteration_cap']==(k>=cap)
        assert 'field_file' in row,'CG must retain every timed invocation field'
        pred=np.load(out/row['field_file'])['prediction']
        f=np.load(out/'fields'/f'N{n}_case{row["case"]}_reference.npz')['forcing']
        true=float(np.linalg.norm(f-stencil(pred,n))/max(np.linalg.norm(f),np.finfo(float).tiny))
        defect=abs(true-row['cg_true_relative_residual']);maxdefect=max(maxdefect,defect)
        assert defect<1e-11
        passed=bool(row['cg_healthy'] and np.isfinite(true) and true<=tol)
        assert row['cg_converged']==passed and row['stationary']==passed
        reason='true_residual_pass' if passed else ('breakdown' if not row['cg_healthy'] else
            ('iteration_cap' if k>=cap else 'true_residual_failed'))
        assert row['cg_stopping_reason']==reason
        failed+=not passed
        if row['repetition']==0:
            count=[0]
            def callback(x):count[0]+=1
            op=LinearOperator((f.size,f.size),matvec=lambda x:stencil(x.reshape(f.shape),n).ravel(),dtype=np.float64)
            reference,info=cg(op,f.ravel(),rtol=tol,atol=0.,maxiter=cap,callback=callback)
            assert count[0]==k,(count[0],k)
            assert np.linalg.norm(reference.reshape(f.shape)-pred)/max(np.linalg.norm(pred),1e-300)<1e-9
            assert (info==0)==(row['cg_recursive_relative_residual']<=tol)
            replays+=1
        checked+=1
    for row in json.loads((out/'summary.json').read_text())['rows']:
        if not row['method'].startswith('cg_'):continue
        group=[r for r in record['invocations'] if r['method']==row['method'] and r['intervals']==row['intervals']]
        assert row['cg_failed_invocations']==sum(not r['cg_converged'] for r in group)
        assert row['cg_iterations_repetitions']==[r['iterations'] for r in group]
        assert row['cg_iterations_median']==np.median([r['iterations'] for r in group])
        assert row['cg_true_relative_residual_worst']==max(r['cg_true_relative_residual'] for r in group)
    return dict(passed=True,checked_invocation_fields=checked,independent_scipy_replays=replays,
        failed_invocations=failed,maximum_true_residual_absolute_defect=maxdefect,
        preconditioner='identity; constant diagonal Jacobi changes only scaling for this constant-coefficient stencil')
