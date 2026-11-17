from campaign import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
rep=dict(meta=meta(),status='running',fixtures=[],cases=[]);save(o/'result.json',rep)
# Smooth manufactured solution u=exp(-t) sin(pi*x) sin(pi*y).
nu=.03
def exact(t,X): return jnp.exp(-t)*jnp.sin(jnp.pi*X[0])*jnp.sin(jnp.pi*X[1])
def forcing(t,X):
 u=exact(t,X)
 ux=jnp.exp(-t)*jnp.pi*jnp.cos(jnp.pi*X[0])*jnp.sin(jnp.pi*X[1])
 uy=jnp.exp(-t)*jnp.pi*jnp.sin(jnp.pi*X[0])*jnp.cos(jnp.pi*X[1])
 return -u+u*(ux+uy)+2*nu*jnp.pi**2*u
for L in ([8,16] if a.smoke else [32,64,128]):
 dt=.0025 if a.smoke else .000625
 f=R.make_solver(2,L,forcing=forcing)
 v,sec=timer(lambda:f(exact(0,R.node_coords(L,2)),nu,dt,'cnr',1e-11,1e-10))
 out=np.asarray(v['out']);truth=np.stack([np.asarray(exact(t,R.node_coords(L,2))).ravel() for t in R.TIMES])
 err=np.linalg.norm(out-truth,axis=1)/np.linalg.norm(truth[0])
 np.savez_compressed(o/f'fixture{L}.npz',**v,truth=truth)
 rep['fixtures'].append(dict(L=L,dt=dt,error=err,seconds=sec));save(o/'result.json',rep)
if a.smoke:
 assert rep['fixtures'][-1]['error'][-1]<rep['fixtures'][0]['error'][-1]
 rep['status']='smoke_pass';save(o/'result.json',rep);raise SystemExit
train,dev=draw();np.savez(o/'parameters.npz',train=train,dev=dev)
# Separate space, time, joint refinements; all raw states retained.
settings=[(128,.000625),(256,.000625),(512,.000625),(512,.0025),(512,.00125),(128,.0025),(256,.00125)]
solvers={L:R.make_solver(2,L) for L in (128,256,512)}
for i,ph in enumerate(dev[:8]):
 for L,dt in settings:
  v,sec=timer(lambda:solvers[L](E.initial(L,ph)[1:-1,1:-1],ph[4],dt,'cnr',1e-11,1e-10))
  st=int(v['stages']);res=float(np.max(np.asarray(v['res'])[:st]));name=f'dev{i}_fd{L}_dt{dt}.npz'
  np.savez_compressed(o/name,**v,parameters=ph)
  rep['cases'].append(dict(case=i,L=L,dt=dt,seconds=sec,residual=res,finite=bool(np.isfinite(v['out']).all()),file=name));save(o/'result.json',rep)
 for N in (64,128,256):
  f=P.make_ps(N,2);b=P.ic_coefs_2d(ph,N,256)
  t=time.perf_counter();v=f(b,ph[4],.000625,P.lattice_S(N,256));sec=time.perf_counter()-t
  np.savez_compressed(o/f'dev{i}_ps{N}.npz',**v,parameters=ph)
  rep['cases'].append(dict(case=i,N=N,dt=.000625,seconds=sec,method='spectral'));save(o/'result.json',rep)
rep['status']='computed_pending_independent_qualification';save(o/'result.json',rep)
