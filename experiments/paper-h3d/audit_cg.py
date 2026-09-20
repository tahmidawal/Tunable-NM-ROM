"""Independent NumPy stencil/residual audit of saved CG trajectory invocations."""
import json
from pathlib import Path
import numpy as np


def laplacian(u, n):
    result = 6*u.copy()
    for axis in range(3):
        lower=[slice(None)]*3; upper=lower.copy()
        lower[axis]=slice(None,-1); upper[axis]=slice(1,None)
        result[tuple(lower)] -= u[tuple(upper)]
        result[tuple(upper)] -= u[tuple(lower)]
    return n*n*result


def audit(out, record):
    out=Path(out);checks=[]
    metadata={(mesh['intervals'],name):value for mesh in record['meshes']
        for name,value in mesh['methods'].items() if name.startswith('fom_cn_cg_')}
    if 'iterative_cg_controls' in record['config']:
        requested = {f"fom_cn_cg_dt{float(setting['dt']):g}_rtol{float(setting['relative_tolerance']):.0e}"
            for setting in record['config']['iterative_cg_controls']} if record['config'].get('include_linear_controls',True) else set()
        assert len(requested)==(len(record['config']['iterative_cg_controls']) if record['config'].get('include_linear_controls',True) else 0)
        for mesh in record['meshes']:
            assert {name for n,name in metadata if n==mesh['intervals']}==requested
    for row in record['invocations']:
        key=(row['intervals'],row['method'])
        if key not in metadata:continue
        meta=metadata[key];stats=np.asarray(row['cg_stats']);tol=meta['relative_tolerance']
        assert stats.shape[1]==6 and np.isfinite(stats).all()
        np.testing.assert_array_equal(stats[:,0],stats[:,0].astype(int))
        assert np.all((stats[:,0]>=0)&(stats[:,0]<=meta['max_iterations']))
        assert np.all(np.isin(stats[:,2:5],[0.,1.]))
        expected=(stats[:,1]<=tol)&(stats[:,3]==0)
        np.testing.assert_array_equal(stats[:,2],expected)
        assert row['cg_iterations']==int(sum(stats[:,0]))
        assert row['cg_failed_steps']==row['nonstationary_solves']==int(np.count_nonzero(~expected))
        np.testing.assert_array_equal(stats[:,4],(stats[:,0]>=meta['max_iterations'])&~expected)
        if 'field_file' not in row:continue
        assert row['cg_untimed_trace_exact_parity'] is True and row['cg_untimed_trace_seconds']>=0
        payload=np.load(out/row['field_file']);states=payload['cg_states'];n=row['intervals']
        np.testing.assert_array_equal(stats,payload['cg_stats'])
        initial=np.load(out/'fields'/f'N{n}_case{row["case"]}_reference.npz')['same_grid'][0]
        np.testing.assert_array_equal(states[0],initial)
        ticks=np.rint(np.asarray(record['config']['times'])/meta['dt']).astype(int)
        assert states.shape[0]==ticks[-1]+1 and len(stats)==ticks[-1]
        np.testing.assert_array_equal(payload['prediction'],states[ticks])
        alpha=record['config']['diffusivity']*meta['dt']/2
        residuals=[]
        for previous,current in zip(states[:-1],states[1:]):
            rhs=previous-alpha*laplacian(previous,n)
            residual=rhs-current-alpha*laplacian(current,n)
            residuals.append(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1e-300))
        np.testing.assert_allclose(stats[:,1],residuals,rtol=2e-7,atol=3e-14)
        checks.append(dict(intervals=n,case=row['case'],method=row['method'],
            steps=len(stats),untimed_trace_exact_parity_verified=True,max_residual_difference=float(np.max(np.abs(stats[:,1]-residuals))),
            failed_steps=row['cg_failed_steps']))
    return dict(passed=True,independent_saved_trajectory_checks=checks,
        scope='NumPy stencil, every retained time-step true residual, declared stops/caps and all invocation counters; failed solves remain explicitly counted')
