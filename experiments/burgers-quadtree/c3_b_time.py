from campaign import *
from owned_inputs import verified
import continuous as C,family_b as FB,quadtree as QT,itertools,argparse
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);from time_fixture import check;check(o/'time-fixture.json',require_gpu=True);params=json.loads(verified('c1familyb','parameters.json').read_text())['dev'][:8];xy=jnp.asarray(E.coords(512,True));inputs=[];orders=[];truths=[];norms=[];rows=[];solvers={dt:C.flux_trajectory(dt,interval=.05) for dt in (.01,.005)}
for case,ph in enumerate(params):
 u=jnp.asarray(FB.initial(4095,ph));orders.append(tuple(sorted(QT.initial_order(u)[:2])));inputs.append(P.analysis(map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511),512));d=np.load(verified('c1familyb',f'case{32+case}.npz'));truths.append(jnp.asarray(d['coefficients'])[::2]);norms.append(float(d['initial_norm']))
for split in list(itertools.combinations(range(4),2))+[(0,1,2,3)]:
 cases=list(range(8)) if len(split)==4 else [i for i,x in enumerate(orders) if x==split]
 if not cases:continue
 name='uniform1152' if len(split)==4 else 'tree768';d=np.load(verified('c1bcomplete','split'+''.join(map(str,split))+'_basis.npz'));W=jnp.asarray(d['basis']);e=jnp.asarray(d['eigenvalues'])
 for q in (128,192,256):
  G,Hq=C.flux_tables(W,128,q)
  for case in cases:
   for dt in (.01,.005):
    states,sec=timer(lambda:solvers[dt](W.T@inputs[case][:127,:127].ravel()/2,params[case]['nu'],G,Hq,e));pred=jnp.pad((2*(states@W.T)).reshape(6,127,127),((0,0),(0,128),(0,128)));err=np.asarray(jnp.linalg.norm((pred-truths[case]).reshape(6,-1),axis=1))/(2*norms[case]);np.savez_compressed(o/f'{name}_q{q}_dt{dt}_case{case}.npz',states=states,error=err);rows.append(dict(name=name,case=case,q=q,dt=dt,error=err,evolved=float(err[1:].max()),seconds=sec))
  save(o/'result.json',dict(meta=meta(),rows=rows));print(name,split,q,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,saved_times=np.linspace(0,.25,6),contract='same-input intermediate timestep, exact0.05 output intervals; independent halfstep and quadrature controls; no cost claim'))
