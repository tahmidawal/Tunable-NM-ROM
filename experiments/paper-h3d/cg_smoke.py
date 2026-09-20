"""Sub-minute GPU verification against independent SciPy modal CN propagation."""
import os, tempfile
from pathlib import Path
import numpy as np
from scipy.fft import dstn,idstn
import jax
import common as C
import iterative_cg as I
from audit_cg import laplacian
assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
assert os.environ['JAX_DEFAULT_MATMUL_PRECISION']=='highest'
cfg=dict(times=[0.,.05,.1],diffusivity=.02)
n=8;rng=np.random.default_rng(81);u=rng.normal(size=(n-1,)*3)
k=np.arange(1,n);v=4*n*n*np.sin(np.pi*k/(2*n))**2
lam=v[:,None,None]+v[None,:,None]+v[None,None,:]
checks=[]
fixture=tempfile.TemporaryDirectory();out=Path(fixture.name);(out/'fields').mkdir()
record=dict(config=cfg,meshes=[dict(intervals=n,methods={})],invocations=[])
for dt,tol,cap in [(.025,1e-12,100),(.05,1e-2,100),(.025,1e-12,1)]:
    timed=jax.device_get(I.engine(n,cfg,dt,tol,cap)(u))
    assert set(timed)=={'prediction','cg_stats'}
    result=jax.device_get(I.engine(n,cfg,dt,tol,cap,retain_trace=True)(u));states=result['cg_states'];stats=result['cg_stats']
    np.testing.assert_array_equal(timed['prediction'],result['prediction'])
    np.testing.assert_array_equal(timed['cg_stats'],result['cg_stats'])
    alpha=cfg['diffusivity']*dt/2;res=[]
    for old,new in zip(states[:-1],states[1:]):
        rhs=old-alpha*laplacian(old,n)
        res.append(np.linalg.norm(rhs-new-alpha*laplacian(new,n))/np.linalg.norm(rhs))
    np.testing.assert_allclose(res,stats[:,1],rtol=2e-4,atol=3e-15)
    np.testing.assert_array_equal(stats[:,2],(stats[:,1]<=tol)&(stats[:,3]==0))
    modal=idstn(dstn(u,type=1,norm='ortho')*((1-alpha*lam)/(1+alpha*lam))**int(.1/dt),type=1,norm='ortho')
    error=float(np.linalg.norm(states[-1]-modal)/np.linalg.norm(modal))
    if tol==1e-12 and cap==100:assert error<2e-11 and np.all(stats[:,2]==1)
    if cap==1:assert np.all(stats[:,4]==1) and np.all(stats[:,2]==0)
    name=f'fom_cn_cg_dt{dt:g}_rtol{tol:.0e}_cap{cap}'
    record['meshes'][0]['methods'][name]=dict(dt=dt,relative_tolerance=tol,max_iterations=cap)
    np.savez(out/'fields'/'N8_case0_reference.npz',same_grid=np.stack([u]*3))
    field='fields/'+name+'.npz';np.savez(out/field,**result)
    counters=dict(intervals=n,case=0,method=name,cg_stats=stats.tolist(),cg_iterations=int(sum(stats[:,0])),
        cg_failed_steps=int(sum(stats[:,2]!=1)),nonstationary_solves=int(sum(stats[:,2]!=1)))
    record['invocations'].extend([dict(**counters,field_file=field,cg_untimed_trace_exact_parity=True,cg_untimed_trace_seconds=0.),dict(**counters)])
    checks.append(dict(dt=dt,tolerance=tol,cap=cap,modal_relative_error=error,stats=stats.tolist()))
zero=jax.device_get(I.engine(n,cfg,.025,1e-6,100)(np.zeros_like(u)))
assert np.all(zero['cg_stats'][:,0]==0) and np.all(zero['cg_stats'][:,2]==1)
from audit_cg import audit
residual_audit=audit(out,record)
fixture.cleanup()
C.dump(Path(__file__).with_name('smokes')/'iterative-cg.json',dict(passed=True,backend=jax.default_backend(),x64=True,precision='highest',checks=checks,zero_rhs_passed=True,independent_audit=residual_audit))
print('CG_SMOKE_PASS',checks[0]['modal_relative_error'])
