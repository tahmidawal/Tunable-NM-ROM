"""Bounded production-mesh checks of the untraced CG query."""
import json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator,cg
from audit import forcing
from audit_cg import stencil
import iterative_cg as CG

begin=time.perf_counter();assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
rows=[]
for n in [32,64]:
 f=forcing(n,np.array([.41,.52,.63,.11,.9]));op=LinearOperator((f.size,f.size),matvec=lambda x:stencil(x.reshape(f.shape),n).ravel(),dtype=np.float64)
 for tol in [1e-2,1e-4,1e-6]:
  pred,stats=jax.device_get(CG.engine(n,tol,4*n,retain_history=False)(jnp.asarray(f)))
  count=[0]
  def callback(x):count[0]+=1
  other,status=cg(op,f.ravel(),rtol=tol,atol=0.,maxiter=4*n,callback=callback);other=other.reshape(f.shape)
  true=float(np.linalg.norm(f-stencil(pred,n))/np.linalg.norm(f))
  difference=float(np.linalg.norm(pred-other)/np.linalg.norm(pred))
  bound=float((np.linalg.norm(f-stencil(pred,n))+np.linalg.norm(f-stencil(other,n)))/(12*n*n*np.sin(np.pi/(2*n))**2*np.linalg.norm(pred)))
  assert pred.dtype==np.float64 and stats[3]==1 and status==0 and true<=tol
  assert abs(true-float(stats[1]))<1e-11 and difference<=bound+1e-12
  rows.append(dict(intervals=n,tolerance=tol,iterations=int(stats[0]),scipy_iterations=count[0],
   true_relative_residual=true,field_relative_difference=difference,spd_residual_bound=bound))
Path(__file__).with_name('runs').joinpath('cg-plain-mesh-smoke.json').write_text(json.dumps(dict(passed=True,
 backend='gpu',x64=True,rows=rows,smoke_seconds=time.perf_counter()-begin,timing_claim=False),indent=2)+'\n')
