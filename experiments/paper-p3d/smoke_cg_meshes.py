"""Bounded solver-only native/transfer mesh counter and residual smoke."""
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator,cg
import iterative_cg as CG
from audit_cg import stencil,replay_history
from audit import forcing

begin=time.perf_counter();assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
rows=[]
for n in [32,64]:
    f=forcing(n,np.array([.41,.52,.63,.11,.9]))
    operator=LinearOperator((f.size,f.size),matvec=lambda x:stencil(x.reshape(f.shape),n).ravel(),dtype=np.float64)
    for tol in [1e-2,1e-4,1e-6]:
        field,stats,history=jax.device_get(CG.engine(n,tol,4*n)(jnp.asarray(f)))
        count=[0]
        def callback(x):count[0]+=1
        independent,info=cg(operator,f.ravel(),rtol=tol,atol=0.,maxiter=4*n,callback=callback)
        defect=float(np.linalg.norm(field.ravel()-independent)/np.linalg.norm(independent))
        true=float(np.linalg.norm(f-stencil(field,n))/np.linalg.norm(f))
        print(n,tol,stats.tolist(),info,count[0],defect,flush=True)
        replay=replay_history(f,field,history[:int(stats[0])],n,tol,4*n)
        assert stats[3]==1 and info==0
        bound=(np.linalg.norm(f-stencil(field,n))+np.linalg.norm(f-stencil(independent.reshape(f.shape),n)))/(12*n*n*np.sin(np.pi/(2*n))**2*np.linalg.norm(field))
        assert defect<=bound+1e-12
        assert abs(true-float(stats[1]))<1e-11
        rows.append(dict(intervals=n,tolerance=tol,cap=4*n,iterations=int(stats[0]),
            independent_field_relative_defect=defect,true_relative_residual=true,history_replay=replay,scipy_iterations=count[0],spd_residual_difference_bound=bound))
output=Path(__file__).parent/'runs/cg-mesh-smoke.json'
output.write_text(json.dumps(dict(passed=True,backend='gpu',x64=True,rows=rows,
    smoke_seconds=time.perf_counter()-begin,timing_claim=False),indent=2)+'\n')
print(output.read_text())
