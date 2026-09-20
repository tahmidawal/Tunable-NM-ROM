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



def replay_history(f,pred,history,n,tol,cap):
    history=np.asarray(history,dtype=float).reshape(-1,5)
    k=len(history);threshold=tol**2*np.vdot(f,f)
    x=np.zeros_like(f);r=f.copy();p=f.copy();rho=float(np.vdot(r,r));maximum=0.
    for index,(alpha,beta,oldrho,pap,newrho) in enumerate(history):
        assert oldrho>threshold,('iteration after tolerance satisfied',index)
        ap=stencil(p,n);observed_pap=float(np.vdot(p,ap))
        defect=abs(observed_pap-pap)/max(abs(pap),1e-300);maximum=max(maximum,defect)
        assert defect<1e-7,('matrix-vector/history mismatch',index,defect)
        assert abs(rho-oldrho)/max(abs(oldrho),1e-300)<1e-7
        if pap>0:
            assert abs(alpha-oldrho/pap)<1e-12*max(abs(alpha),1e-300)
        else:assert alpha==0.
        assert abs(beta-newrho/oldrho)<1e-12*max(abs(beta),1e-300)
        x=x+alpha*p;r=r-alpha*ap
        rho=float(np.vdot(r,r));assert abs(rho-newrho)/max(abs(newrho),1e-300)<1e-7
        p=r+beta*p
    defect=float(np.linalg.norm(x-pred)/max(np.linalg.norm(pred),1e-300))
    assert defect<1e-9,('saved-coefficient trajectory replay',defect)
    if k and k<cap:assert history[-1,4]<=threshold
    return dict(field_relative_defect=defect,maximum_history_pap_relative_defect=maximum)


def audit(out,record):
    checked=0;replays=0;maxdefect=0.;failed=0;history_checks=[];scipy_checks=[]
    spec=record['config']['iterative_cg']
    assert spec['preconditioner']=='identity' and spec['relative_tolerances']==[1e-2,1e-4,1e-6]
    for mesh in record['meshes']:
        expected={f'cg_identity_rtol{tol:.0e}' for tol in spec['relative_tolerances']}
        if record['config'].get('plain_cg_control',False):expected|={name.replace('cg_identity_','cg_identity_plain_') for name in list(expected)}
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
        traced='_plain_' not in row['method']
        assert ('cg_iteration_history' in row)==traced
        history=np.asarray(row.get('cg_iteration_history',[])).reshape(-1,5)
        if traced:assert len(history)==k
        if traced and k:
            assert np.all(history[:,2]>tol**2*np.vdot(f,f))
            assert abs(np.sqrt(history[-1,4]/np.vdot(f,f))-row['cg_recursive_relative_residual'])<1e-12
        if row['repetition']==0:
            if traced:history_checks.append(replay_history(f,pred,history,n,tol,cap))
            count=[0]
            def callback(x):count[0]+=1
            op=LinearOperator((f.size,f.size),matvec=lambda x:stencil(x.reshape(f.shape),n).ravel(),dtype=np.float64)
            reference,info=cg(op,f.ravel(),rtol=tol,atol=0.,maxiter=cap,callback=callback)
            cpu=reference.reshape(f.shape);cpu_residual=f-stencil(cpu,n)
            difference=float(np.linalg.norm(cpu-pred)/max(np.linalg.norm(pred),1e-300))
            lambda_min=12*n*n*np.sin(np.pi/(2*n))**2
            bound=float((np.linalg.norm(f-stencil(pred,n))+np.linalg.norm(cpu_residual))/lambda_min/max(np.linalg.norm(pred),1e-300))
            assert difference<=bound+1e-12
            assert info>=0
            scipy_checks.append(dict(intervals=n,case=row['case'],method=row['method'],iterations=count[0],
                saved_iterations=k,iteration_count_matches=(count[0]==k),timed_iteration_history=traced,field_relative_difference=difference,spd_residual_difference_bound=bound))
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
        independent_saved_coefficient_history_replays=history_checks,scipy_comparisons=scipy_checks,
        iteration_scope='Traced control: saved coefficient/recurrence trajectories independently replayed for repetition zero. Efficient untraced CG: true residuals checked on every measured field; independent SciPy iteration counts and SPD residual-based field bounds checked per case. Counts may differ between finite-precision tolerance-stopped trajectories; every difference is reported.',
        preconditioner='identity; constant diagonal Jacobi changes only scaling for this constant-coefficient stencil')
