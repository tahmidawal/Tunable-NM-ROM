"""New global smooth-space compression hypothesis; frozen TRAIN-learned banks."""
from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--ranks',default='64,128,192,256');ap.add_argument('--orders',default='32,48,64,96,192');a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();rows=[];solvers={dt:C.flux_trajectory(dt) for dt in (.0125,.00625)}
for name in ('neural','POD'):
 d=np.load(verified('c1globalneural',f'A/{name}_physics_basis.npz'));W=jnp.asarray(d['basis']);e=jnp.asarray(d['eigenvalues']);np.savez_compressed(o/f'{name}_source_basis.npz',basis=W,eigenvalues=e)
 for rank in map(int,a.ranks.split(',')):
  w=W[:,:rank];eigs=e[:rank]
  for q in map(int,a.orders.split(',')):
   G,H=C.flux_tables(w,128,q)
   for case,ph in enumerate(dev[:32]):
    d=np.load(verified('c1data',f'case{128+case}.npz'));truth=jnp.asarray(d['coefficients']).reshape(11,-1)/2;norm=float(d['initial_norm']);a0=w.T@truth[0];proj=np.asarray(jnp.linalg.norm(truth-(truth@w)@w.T,axis=1))/norm
    for dt in (.0125,.00625):
     states,sec=timer(lambda:solvers[dt](a0,ph[4],G,H,eigs));err=np.asarray(jnp.linalg.norm(states@w.T-truth,axis=1))/norm;np.savez_compressed(o/f'{name}_r{rank}_q{q}_dt{dt}_case{case}.npz',states=states,error=err)
     rows.append(dict(name=name,rank=rank,q=q,dt=dt,case=case,projection=proj,error=err,evolved=float(err[[2,4,6,8,10]].max()),seconds=sec,finite=bool(np.isfinite(states).all())))
   save(o/'result.json',dict(meta=meta(),rows=rows));print(name,rank,q,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='new constrained globally supported neural/POD trial spaces; same initial data, actual weak PDE with independent quadrature/time controls; not local-refinement success or complete cost'))
