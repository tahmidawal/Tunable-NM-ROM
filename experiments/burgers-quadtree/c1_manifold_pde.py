"""Actual overdetermined continuous weak nonlinear manifold solve diagnostic."""
from campaign import *
from owned_inputs import verified
from spectral_cost import PJ
import continuous as C,manifold_core as M,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1manifold','best.pkl'),'rb')));V=jnp.asarray(np.load(verified('c1manifold','lift.npz'))['basis']);K=V.T@(C.eigenvalues(128)[:,None]*V);_,dev=draw();rows=[]
for q,dt in [(192,.005),(256,.005),(256,.0025)]:
 G,H=C.flux_tables(V,128,q)
 @jax.jit
 def step(z,nu,norm,h,theta,G,H,K):
  old=M.decode(p[1],z)
  def residual(a):
   new=M.decode(p[1],a);mid=theta*new+(1-theta)*old
   return new-old+h*(nu*(K@mid)-norm*(H.T@(G@mid)**2))
  return M.lm(residual,z,60)
 for i in (0,9):
  d=np.load(verified('c1data',f'case{128+i}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(d['initial_norm']);c0=V.T@truth[0].ravel()/(2*norm)
  z,g,e=jax.jit(lambda c:M.lm(lambda z:M.decode(p[1],z)-c,M.encode(p[0],c,128),60))(c0)
  states=[M.decode(p[1],z)];diagnostics=[];t=0.;start=time.perf_counter()
  schedule=[(dt/2,1.),(dt/2,1.)]+[(dt,.5)]*(round(.25/dt)-1)
  for n,(h,theta) in enumerate(schedule):
   z,g,e=step(z,dev[i,4],norm,h,theta,G,H,K);jax.block_until_ready(z);t+=h
   diagnostics.append(dict(time=t,stationarity=float(g),residual=float(e)))
   if abs(t/.025-round(t/.025))<1e-8:states.append(M.decode(p[1],z))
  seconds=time.perf_counter()-start;cs=2*norm*(jnp.stack(states)@V.T).reshape(11,127,127);err=np.asarray(jnp.linalg.norm((cs-truth).reshape(11,-1),axis=1))/(2*norm)
  final=PJ.synth_mesh(cs[-1],4095);assert final.shape==(4096,4096)
  np.savez_compressed(o/f'q{q}_dt{dt}_case{i}.npz',coefficients=cs,error=err,final4096=final)
  rows.append(dict(case=i,q=q,dt=dt,errors=err,solve_seconds=seconds,diagnostics=diagnostics,output_shape=list(final.shape)))
  save(o/'result.json',dict(meta=meta(),rows=rows));print(q,dt,i,float(err[[2,4,6,8,10]].max()),flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='actual weak midpoint/Rannacher latent PDE solve,1024 tests for128 coordinates; prepared input diagnostic, not complete-query timing'))
