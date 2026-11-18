from campaign import *
from owned_inputs import verified
import specialized as S,local_bank as B,continuous as C,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1local','network.pkl'),'rb')));V=jnp.asarray(np.load(verified('c1data','pod.npz'))['basis']);_,params=draw()
points=jnp.asarray(E.coords(128,True));blocks={}
def coeff(F):return jax.vmap(lambda f:P.analysis(f.reshape(127,127),128).ravel()/2,in_axes=1,out_axes=1)(F)
for i in range(4):
 for j in range(4):
  patch=[(i/4,j/4,.25)];blocks[i,j]=jnp.concatenate([coeff(B.features(points,patch,32,'poly')),coeff(S.features(p,points,patch,32))],axis=1)
Ds=[];tiles=[]
for i,ph in enumerate(params[:32]):
 d=np.load(verified('c1data',f'case{128+i}.npz'));Ds.append(jnp.asarray(d['coefficients']).reshape(11,-1).T/(2*float(d['initial_norm'])))
 # Fixed policy from the supplied physical initial array, not future fields or descriptor access.
 initial=E.initial(4095,ph);peak=np.unravel_index(np.argmax(initial),initial.shape);tiles.append(tuple(min(2,int(x*4/4096)) for x in peak))
rows=[]
for rank in (128,256):
 models={}
 for i in range(3):
  for j in range(3):
   A=jnp.concatenate([V[:,:rank]]+[blocks[ii,jj] for ii in (i,i+1) for jj in (j,j+1)],axis=1)
   Q,s,vh=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;Q=Q[:,keep];models[i,j]=Q
   np.savez_compressed(o/f'r{rank}_tile{i}{j}.npz',basis=Q,singular_values=s,physical_to_orthogonal=vh[keep].T/s[keep])
 for case,(D,tile) in enumerate(zip(Ds,tiles)):
  Q=models[tile];err=np.asarray(jnp.linalg.norm(D-Q@(Q.T@D),axis=0));rows.append(dict(rank=rank,case=case,tile=tile,columns=rank+256,per_time=err,evolved=float(err[[2,4,6,8,10]].max())))
 save(o/'result.json',dict(meta=meta(),rows=rows));print(rank,max(x['evolved'] for x in rows if x['rank']==rank),flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,policy='forward two-by-two depth2 patch tile from initial-array peak, clipped at walls; not future-guided'))
