from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();rows=[]
for name in ['neural_r256_d2_k32','poly_r128_d2_k64']:
 V=jnp.asarray(np.load(verified('c1local',name+'.npz'))['basis']);W,eigs,rot=C.diffusion_basis(V,128)
 np.savez_compressed(o/(name+'_basis.npz'),basis=W,eigenvalues=eigs,rotation=rot)
 for q,dt in [(96,.005),(128,.005),(128,.0025)]:
  G,H=C.flux_tables(W,128,q);f=C.flux_trajectory(dt)
  for i,ph in enumerate(dev[:32]):
   d=np.load(verified('c1data',f'case{128+i}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(d['initial_norm']);a0=W.T@truth[0].ravel()/2
   states,sec=timer(lambda:f(a0,ph[4],G,H,eigs));pred=(2*(states@W.T)).reshape(11,127,127)
   error=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm)
   np.savez_compressed(o/f'{name}_q{q}_dt{dt}_case{i}.npz',states=states,error=error)
   rows.append(dict(name=name,q=q,dt=dt,case=i,evolved=float(error[[2,4,6,8,10]].max()),error=error,seconds=sec,finite=bool(np.isfinite(states).all())))
  save(o/'result.json',dict(meta=meta(),rows=rows));print(name,q,dt,max(x['evolved'] for x in rows if x['name']==name and x['q']==q and x['dt']==dt),flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='prepared initial coefficient diagnostic; complete query timing follows numerical qualification'))
