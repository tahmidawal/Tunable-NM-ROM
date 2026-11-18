"""Double feature integration N256, freeze initial-field tile selection."""
from campaign import *
from owned_inputs import verified
import continuous as C,specialized as S,local_bank as B,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();rows=[];tiles=json.loads(verified('c1tiles','result.json').read_text())['rows'];N=256
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1local','network.pkl'),'rb')))
V=np.load(verified('c1data','pod.npz'))['basis'][:,:256].reshape(127,127,256);V=jnp.asarray(np.pad(V,((0,128),(0,128),(0,0))).reshape(255*255,256));xy=jnp.asarray(E.coords(N,True))
def coeff(F):return jax.vmap(lambda f:P.analysis(f.reshape(N-1,N-1),N).ravel()/2,in_axes=1,out_axes=1)(F)
for i in range(3):
 for j in range(3):
  cases=[r['case'] for r in tiles if r['rank']==256 and r['tile']==[i,j]]
  if not cases:continue
  patches=[(ii/4,jj/4,.25) for ii in (i,i+1) for jj in (j,j+1)]
  A=jnp.concatenate([V,coeff(B.features(xy,patches,32,'poly')),coeff(S.features(p,xy,patches,32))],axis=1)
  Q,s,vh=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;Q=Q[:,keep];W,eigs,rot=C.diffusion_basis(Q,N)
  np.savez_compressed(o/f'tile{i}{j}_basis.npz',basis=W,eigenvalues=eigs,physical_to_orthogonal=(vh[keep].T/s[keep])@rot,singular_values=s)
  G,H=C.flux_tables(W,N,384)
  for dt in (.005,.0025):
   f=C.flux_trajectory(dt)
   for case in cases:
    d=np.load(verified('c1data',f'check{128+case}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(np.load(verified('c1data',f'case{128+case}.npz'))['initial_norm']);a0=W.T@truth[0].ravel()/2
    states,sec=timer(lambda:f(a0,dev[case,4],G,H,eigs));pred=(2*(states@W.T)).reshape(11,255,255);err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm)
    old=np.load(verified('c1qualify',f'tile{i}{j}_q256_dt{dt}_case{case}.npz'))['states'];oldW=np.load(verified('c1qualify',f'tile{i}{j}_basis.npz'))['basis'];oldpred=np.pad((2*(old@oldW.T)).reshape(11,127,127),((0,0),(0,128),(0,128)))
    difference=np.asarray(jnp.linalg.norm((pred-oldpred).reshape(11,-1),axis=1))/(2*norm)
    projection=np.asarray(jnp.linalg.norm(truth.reshape(11,-1)/2-(truth.reshape(11,-1)/2@W)@W.T,axis=1))/norm
    np.savez_compressed(o/f'tile{i}{j}_dt{dt}_case{case}.npz',states=states,coefficients=pred,error=err)
    rows.append(dict(case=case,tile=[i,j],dt=dt,error=err,projection=projection,integration_change=difference,seconds=sec,evolved=float(err[[2,4,6,8,10]].max())))
   save(o/'result.json',dict(meta=meta(),rows=rows));print(i,j,dt,max(x['evolved'] for x in rows if x['tile']==[i,j] and x['dt']==dt),flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='doubled feature-integration PDE accuracy; prepared coefficients, not query timing'))
