"""Quadrature qualification and initial-input localized continuous PDE control."""
from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();tiles=json.loads(verified('c1tiles','result.json').read_text())['rows'];rows=[]
models=[('poly','c1local','poly_r128_d2_k64.npz',list(range(32)))]
for i in range(3):
 for j in range(3):
  cases=[r['case'] for r in tiles if r['rank']==256 and r['tile']==[i,j]]
  if cases:models.append((f'tile{i}{j}','c1tiles',f'r256_tile{i}{j}.npz',cases))
for name,job,file,cases in models:
 V=jnp.asarray(np.load(verified(job,file))['basis']);W,eigs,rot=C.diffusion_basis(V,128)
 np.savez_compressed(o/(name+'_basis.npz'),basis=W,eigenvalues=eigs,rotation=rot)
 previous={}
 for q,dt in [(128,.005),(192,.005),(256,.005),(256,.0025)]:
  G,H=C.flux_tables(W,128,q);f=C.flux_trajectory(dt)
  for i in cases:
   d=np.load(verified('c1data',f'case{128+i}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(d['initial_norm']);a0=W.T@truth[0].ravel()/2
   states,sec=timer(lambda:f(a0,dev[i,4],G,H,eigs));pred=(2*(states@W.T)).reshape(11,127,127)
   error=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm)
   delta=None if i not in previous else float(jnp.linalg.norm(states-previous[i],axis=1)[jnp.array([2,4,6,8,10])].max()/norm)
   previous[i]=states
   np.savez_compressed(o/f'{name}_q{q}_dt{dt}_case{i}.npz',states=states,error=error)
   rows.append(dict(name=name,q=q,dt=dt,case=i,evolved=float(error[[2,4,6,8,10]].max()),error=error,previous_setting_change=delta,seconds=sec,finite=bool(np.isfinite(states).all())))
  save(o/'result.json',dict(meta=meta(),rows=rows));print(name,q,dt,max(x['evolved'] for x in rows if x['name']==name and x['q']==q and x['dt']==dt),flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='prepared coefficient accuracy diagnostic; no total query timing or output claim'))
