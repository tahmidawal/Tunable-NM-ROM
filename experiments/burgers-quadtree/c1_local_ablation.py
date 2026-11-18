"""Matched512-coordinate local polynomial/neural/local-POD controls."""
from campaign import *
from owned_inputs import verified
import continuous as C,local_bank as B,specialized as S,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1local','network.pkl'),'rb')));teacher=jnp.asarray(np.load(verified('c1local','teacher.npz'))['features'].reshape(33,33,64));V=jnp.asarray(np.load(verified('c1data','pod.npz'))['basis'])[:,:256];xy=jnp.asarray(E.coords(128,True));_,dev=draw();tiles=json.loads(verified('c1tiles','result.json').read_text())['rows'];rows=[];solvers={h:C.flux_trajectory(h) for h in (.005,.0025)}
for kind in ('poly','neural','local_pod'):
 blocks={}
 for i in range(4):
  for j in range(4):
   patches=[(i/4,j/4,.25)]
   F=S.features(p,xy,patches,64) if kind=='neural' else B.features(xy,patches,64,'poly' if kind=='poly' else 'pod',pod=teacher)
   blocks[i,j]=jax.vmap(lambda f:P.analysis(f.reshape(127,127),128).ravel()/2,in_axes=1,out_axes=1)(F)
 for i in range(3):
  for j in range(3):
   cases=[r['case'] for r in tiles if r['rank']==256 and r['tile']==[i,j]]
   if not cases:continue
   A=jnp.concatenate([V]+[blocks[ii,jj] for ii in (i,i+1) for jj in (j,j+1)],axis=1);Q,s,vh=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;Q=Q[:,keep];W,eigs,rot=C.diffusion_basis(Q,128)
   np.savez_compressed(o/f'{kind}_tile{i}{j}.npz',basis=W,eigenvalues=eigs,physical_to_orthogonal=(vh[keep].T/s[keep])@rot,singular_values=s)
   for q,dt in [(128,.005),(192,.005),(192,.0025)]:
    G,H=C.flux_tables(W,128,q)
    for case in cases:
     d=np.load(verified('c1data',f'case{128+case}.npz'));truth=jnp.asarray(d['coefficients']).reshape(11,-1)/2;norm=float(d['initial_norm']);a0=W.T@truth[0]
     states,sec=timer(lambda:solvers[dt](a0,dev[case,4],G,H,eigs));err=np.asarray(jnp.linalg.norm(states@W.T-truth,axis=1))/norm;proj=np.asarray(jnp.linalg.norm(truth-(truth@W)@W.T,axis=1))/norm
     np.savez_compressed(o/f'{kind}_q{q}_dt{dt}_case{case}.npz',states=states,error=err)
     rows.append(dict(kind=kind,case=case,q=q,dt=dt,error=err,projection=proj,seconds=sec,evolved=float(err[[2,4,6,8,10]].max())))
   save(o/'result.json',dict(meta=meta(),rows=rows));print(kind,i,j,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='same coarse256 and initial-only four-patch policy,64local columns per patch; local POD is explicitly bilinear canonical teacher interpolation, not claimed optimal/C2'))
