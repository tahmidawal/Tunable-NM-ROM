from campaign import *
from owned_inputs import verified
import specialized as S,local_bank as B,optax,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1local','network.pkl'),'rb')));V=jnp.asarray(np.load(verified('c1data','pod.npz'))['basis']);Ys=[]
for i in range(160):
 d=np.load(verified('c1data',f'case{i}.npz'));Ys.append(np.asarray(d['coefficients']).reshape(11,-1).T/(2*float(d['initial_norm'])))
U=jnp.asarray(np.concatenate(Ys[:128],axis=1));D=jnp.asarray(np.concatenate(Ys[128:],axis=1));patches=B.uniform(2)
def coeff(F,N):return jax.vmap(lambda f:P.analysis(f.reshape(N-1,N-1),N).reshape(-1)/2,in_axes=1,out_axes=1)(F)
points=jnp.asarray(E.coords(128,True));rows=[]
for r,kp,kn in [(128,16,16),(128,32,16),(128,32,32),(256,16,16),(256,32,16),(256,32,32)]:
 A=jnp.concatenate([V[:,:r],coeff(B.features(points,patches,kp,'poly'),128),coeff(S.features(p,points,patches,kn),128)],axis=1)
 uu,sv,vh=jnp.linalg.svd(A,full_matrices=False);keep=sv>sv[0]*1e-10;Q=uu[:,keep]
 er=np.asarray(jnp.linalg.norm(D-Q@(Q.T@D),axis=0)).reshape(32,11)
 row=dict(r=r,kp=kp,kn=kn,columns=A.shape[1],rank=int(keep.sum()),per_time=er,worst=float(er[:,[2,4,6,8,10]].max()));rows.append(row)
 np.savez_compressed(o/f'r{r}_p{kp}_n{kn}.npz',basis=Q,physical_to_orthogonal=vh[keep].T/sv[keep],singular_values=sv)
 save(o/'result.json',dict(meta=meta(),screen=rows));print('hybrid',r,kp,kn,row['worst'],flush=True)
# Choose one finite recipe on development representation; then optimize the ASSEMBLED objective.
best=min(rows,key=lambda x:x['worst']);r,kp,kn=best['r'],best['kp'],best['kn'];N=64;xy=jnp.asarray(E.coords(N,True))
cut=lambda x:x.reshape(127,127,-1)[:63,:63].reshape(63*63,-1)
coarse=cut(V[:,:r]);poly=coeff(B.features(xy,patches,kp,'poly'),N);target=cut(U)
opt=optax.adam(optax.cosine_decay_schedule(1e-4,2000,alpha=.1));state=opt.init(p)
def loss(p,Y,xy,coarse,poly):
 A=jnp.concatenate([coarse,poly,coeff(S.features(p,xy,patches,kn),N)],axis=1)
 c,sv,rank=B.fit(A,Y);return jnp.mean(jnp.sum((A@jax.lax.stop_gradient(c)-Y)**2,axis=0))
@jax.jit
def update(p,state,Y,xy,coarse,poly):
 val,g=jax.value_and_grad(loss)(p,Y,xy,coarse,poly);u,state=opt.update(g,state,p);return optax.apply_updates(p,u),state,val
rng=np.random.default_rng(2026111802);history=[];bestscore=np.inf
for it in range(2000):
 idx=rng.integers(128,size=8)*11+rng.integers(11,size=8)
 p,state,val=update(p,state,target[:,idx],xy,coarse,poly);assert np.isfinite(float(val))
 if (it+1)%200==0:
  A=jnp.concatenate([V[:,:r],coeff(B.features(points,patches,kp,'poly'),128),coeff(S.features(p,points,patches,kn),128)],axis=1)
  uu,sv,vh=jnp.linalg.svd(A,full_matrices=False);keep=sv>sv[0]*1e-10;Q=uu[:,keep]
  er=np.asarray(jnp.linalg.norm(D-Q@(Q.T@D),axis=0)).reshape(32,11);score=float(er[:,[2,4,6,8,10]].max())
  history.append(dict(step=it+1,loss=float(val),per_time=er,worst=score,rank=int(keep.sum())))
  with open(o/f'step{it+1}.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
  if score<bestscore:
   bestscore=score
   np.savez_compressed(o/'best_basis.npz',basis=Q,physical_to_orthogonal=vh[keep].T/sv[keep],singular_values=sv)
   with open(o/'best.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
  save(o/'result.json',dict(meta=meta(),screen=rows,selected=best,history=history));print('assembled',it+1,score,flush=True)
save(o/'result.json',dict(meta=meta(),screen=rows,selected=best,history=history,complete=True,limitation='training integration N64; checkpoint selection N128; doubled integration qualification required'))
