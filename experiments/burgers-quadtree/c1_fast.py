"""Finite reduced-integration/rank screen; prepared-coefficient PDE evidence."""
from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();tiles=json.loads(verified('c1tiles','result.json').read_text())['rows'];rows=[];solvers={h:C.flux_trajectory(h) for h in (.0125,.00625,.005,.0025)};widths=[]
for ph in dev[:32]:
 u=E.initial(4095,ph)[::16,::16];x=np.arange(256)*16/4095;s=u.sum();mx=(u.sum(1)*x).sum()/s;my=(u.sum(0)*x).sum()/s
 widths.append(float(np.sqrt(((u.sum(1)*(x-mx)**2).sum()+(u.sum(0)*(x-my)**2).sum())/(2*s))))
for rank in (128,256):
 for i in range(3):
  for j in range(3):
   cases=[r['case'] for r in tiles if r['rank']==rank and r['tile']==[i,j]]
   if not cases:continue
   V=jnp.asarray(np.load(verified('c1tiles',f'r{rank}_tile{i}{j}.npz'))['basis']);W,eigs,rot=C.diffusion_basis(V,128)
   np.savez_compressed(o/f'r{rank}_tile{i}{j}_basis.npz',basis=W,eigenvalues=eigs,rotation=rot)
   for q in (48,64,80,96,128,192):
    G,H=C.flux_tables(W,128,q)
    for dt in (.0125,.00625,.005,.0025):
     for case in cases:
      d=np.load(verified('c1data',f'case{128+case}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(d['initial_norm']);a0=W.T@truth[0].ravel()/2
      states,sec=timer(lambda:solvers[dt](a0,dev[case,4],G,H,eigs));pred=(2*(states@W.T)).reshape(11,127,127);err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm)
      np.savez_compressed(o/f'r{rank}_q{q}_dt{dt}_case{case}.npz',states=states,error=err)
      rows.append(dict(rank=rank,q=q,dt=dt,case=case,estimated_width=widths[case],error=err,evolved=float(err[[2,4,6,8,10]].max()),seconds=sec,finite=bool(np.isfinite(states).all())))
   save(o/'result.json',dict(meta=meta(),rows=rows));print(rank,i,j,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='finite reduced integration/coarse capacity diagnostic; selector thresholds .06/.08/.10 fixed prospectively, no future input at query'))
