"""Operator-ordered trial-space compression; separate projection/PDE evidence."""
from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();tiles=json.loads(verified('c1tiles','result.json').read_text())['rows'];rows=[];solvers={dt:C.flux_trajectory(dt) for dt in (.025,.0125,.00625)}
for i in range(3):
 for j in range(3):
  cases=[r['case'] for r in tiles if r['rank']==256 and r['tile']==[i,j]]
  if cases:
   b=np.load(verified('c1qualify',f'tile{i}{j}_basis.npz'));W=jnp.asarray(b['basis']);eigs=jnp.asarray(b['eigenvalues'])
  else:
   V=jnp.asarray(np.load(verified('c1tiles',f'r256_tile{i}{j}.npz'))['basis']);W,eigs,_=C.diffusion_basis(V,128)
  np.savez_compressed(o/f'tile{i}{j}_basis.npz',basis=W,eigenvalues=eigs)
  if not cases:continue
  for q in (64,96,128,192):
   G,H=C.flux_tables(W,128,q)
   for rank in (128,192,256,384):
    w=W[:,:rank];g=G[:,:rank];h=H[:,:rank];e=eigs[:rank]
    for case in cases:
     d=np.load(verified('c1data',f'case{128+case}.npz'));truth=jnp.asarray(d['coefficients']).reshape(11,-1)/2;norm=float(d['initial_norm']);a0=w.T@truth[0];proj=np.asarray(jnp.linalg.norm(truth-(truth@w)@w.T,axis=1))/norm
     for dt in (.025,.0125,.00625):
      states,sec=timer(lambda:solvers[dt](a0,dev[case,4],g,h,e));err=np.asarray(jnp.linalg.norm(states@w.T-truth,axis=1))/norm
      np.savez_compressed(o/f'r{rank}_q{q}_dt{dt}_case{case}.npz',states=states,error=err)
      rows.append(dict(case=case,rank=rank,q=q,dt=dt,projection=proj,error=err,evolved=float(err[[2,4,6,8,10]].max()),seconds=sec,finite=bool(np.isfinite(states).all())))
  save(o/'result.json',dict(meta=meta(),rows=rows));print(i,j,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='lowest projected-diffusion eigenmodes of frozen physical local bank; new constrained trial space, not unchanged decoder accuracy or speed claim'))
