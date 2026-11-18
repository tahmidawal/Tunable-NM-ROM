"""Same-input time/quadrature controls for actual family-B queries."""
from campaign import *
from owned_inputs import verified
import continuous as C,family_b as FB,quadtree as QT,argparse
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
params=json.loads(verified('c1familyb','parameters.json').read_text())['dev'][:8];xy=jnp.asarray(E.coords(512,True));rows=[]
for case,ph in enumerate(params):
 u=jnp.asarray(FB.initial(4095,ph));initial=P.analysis(map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511),512)[:127,:127];norm=float(np.load(verified('c1familyb',f'case{32+case}.npz'))['initial_norm'])
 for name,split in [('tree768',tuple(sorted(QT.initial_order(u)[:2]))),('uniform1152',(0,1,2,3))]:
  d=np.load(verified('c1bcomplete','split'+''.join(map(str,split))+'_basis.npz'));W=jnp.asarray(d['basis']);e=jnp.asarray(d['eigenvalues']);base=jnp.asarray(np.load(verified('c1bcomplete',f'{name}_case{case}_block0_rep0.npz'))['coefficients']);a0=W.T@initial.ravel()/2
  for q,dt in [(192,.00625),(384,.00625),(256,.003125)]:
   G,H=C.flux_tables(W,128,q);f=C.flux_trajectory(dt);states,sec=timer(lambda:f(a0,ph['nu'],G,H,e));pred=(2*(states@W.T)).reshape(11,127,127);delta=np.asarray(jnp.linalg.norm((pred-base).reshape(11,-1),axis=1))/(2*norm)
   np.savez_compressed(o/f'{name}_q{q}_dt{dt}_case{case}.npz',coefficients=pred,change=delta);rows.append(dict(case=case,name=name,q=q,dt=dt,change=delta,seconds=sec))
 save(o/'result.json',dict(meta=meta(),rows=rows));print(case,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='same resident-array restriction as timed familyB query, independent halfstep and nonlinear quadrature checks; no retuning'))
