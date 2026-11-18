"""Physics solves on bounded quadtree configurations and matched uniform control."""
from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
rows=[];solvers={dt:C.flux_trajectory(dt) for dt in (.0125,.00625,.005)}
for family in ('A','B'):
 source=json.loads(verified('c1trees',f'{family}/result.json').read_text())['rows'];params=draw()[1] if family=='A' else json.loads(verified('c1familyb','parameters.json').read_text())['dev']
 selected=[x for x in source if x['coarse']==128 and x['refined_parents'] in (2,4) and x['k']==32]
 for model in sorted(set(x['model'] for x in selected)):
  cases=[x['case'] for x in selected if x['model']==model];V=jnp.asarray(np.load(verified('c1trees',f'{family}/{model}.npz'))['basis']);W,eigs,rot=C.diffusion_basis(V,128)
  np.savez_compressed(o/f'{family}_{model}_basis.npz',basis=W,eigenvalues=eigs,rotation=rot)
  for q,dt in [(128,.0125),(192,.0125),(256,.0125),(256,.00625),(256,.005)]:
   G,H=C.flux_tables(W,128,q)
   for case in cases:
    d=np.load(verified('c1data' if family=='A' else 'c1familyb',f'case{128+case if family=="A" else 32+case}.npz'));truth=np.asarray(d['coefficients']);norm=float(d['initial_norm']);low=jnp.asarray(truth[:,:127,:127]).reshape(11,-1)/2;tail=truth.copy();tail[:,:127,:127]=0;tailnorm=np.linalg.norm(tail.reshape(11,-1),axis=1)/(2*norm);a0=W.T@low[0];nu=params[case][4] if family=='A' else params[case]['nu']
    states,sec=timer(lambda:solvers[dt](a0,nu,G,H,eigs));err=np.sqrt((np.asarray(jnp.linalg.norm(states@W.T-low,axis=1))/norm)**2+tailnorm**2)
    np.savez_compressed(o/f'{family}_{model}_q{q}_dt{dt}_case{case}.npz',states=states,error=err)
    rows.append(dict(family=family,model=model,case=case,q=q,dt=dt,error=err,evolved=float(err[[2,4,6,8,10]].max()),seconds=sec,finite=bool(np.isfinite(states).all())))
   save(o/'result.json',dict(meta=meta(),rows=rows));print(family,model,q,dt,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='initial-only bounded quadtree and uniform controls; same learned features, actual continuous weak PDE; no complete-output timing claim'))
