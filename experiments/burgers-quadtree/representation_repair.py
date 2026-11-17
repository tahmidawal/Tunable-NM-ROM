from campaign import *
from owned_inputs import verified
import local_bank as B
import pickle,optax,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
U=np.load(verified('representation01','snapshots.npz'))['fields'];raw=U[:,:,1:-1,1:-1].reshape(40,11,-1)
norm=np.linalg.norm(raw[:,0],axis=1);Y=(raw/norm[:,None,None]).reshape(440,-1).T
Ut=jnp.asarray(Y[:,:352]);Ud=jnp.asarray(Y[:,352:]);V=jnp.asarray(np.load(verified('representation01','global_pod.npz'))['basis'])
xy=jnp.asarray(E.coords(64,True));params=pickle.load(open(verified('representation01','shared_neural.pkl'),'rb'));params=jax.tree_util.tree_map(jnp.asarray,params)
rep=dict(meta=meta(),rows=[],training=[],status='development representation only')
def assess(tag,params):
 for r in (16,32,64,128):
  A=V[:,:r];c=A.T@Ud;er=np.asarray(jnp.linalg.norm(A@c-Ud,axis=0)).reshape(8,11)
  rep['rows'].append(dict(tag=tag,kind='global_POD',r=r,evolved=er[:,[2,4,6,8,10]].max(1),initial=er[:,0]))
 for r in (32,64):
  for k in (8,16,32):
   for depth in (1,2):
    if r+4**depth*k>512:continue
    A=jnp.concatenate([V[:,:r],B.features(xy,B.uniform(depth),k,'neural',params)],axis=1)
    c,sv,rank=B.fit(A,Ud);er=np.asarray(jnp.linalg.norm(A@c-Ud,axis=0)).reshape(8,11)
    rep['rows'].append(dict(tag=tag,kind='neural',r=r,k=k,depth=depth,columns=A.shape[1],rank=int(rank),evolved=er[:,[2,4,6,8,10]].max(1),initial=er[:,0]))
 save(o/'result.json',rep)
assess('coarse_repair',params)
# Fixed finite configuration pool with equal class weight; guided patch picked from TRAIN residual only.
base=B.uniform(1);configs=[B.uniform(1),B.uniform(2)]
rng=np.random.default_rng(2026111705)
for j in range(4):configs.append(base+B.children(base[int(rng.integers(4))]))
energy=np.sum(np.asarray(Ut-V[:,:32]@(V[:,:32].T@Ut))**2,axis=1)
for j in np.argsort([np.sum(energy*np.asarray(B.window(xy,p))**2) for p in base])[-2:]:configs.append(base+B.children(base[j]))
# Each update samples one configuration class uniformly and refits a sampled prefix.
opt=optax.adam(optax.cosine_decay_schedule(1e-3,20000,alpha=.01));state=opt.init(params)
updates={}
for ci,patches in enumerate(configs):
 for k in (4,8,16,32):
  if 32+len(patches)*k>512:continue
  def make_update(patches,k):
   def loss(p,target,coarse,xy):
    A=jnp.concatenate([coarse,B.features(xy,patches,k,'neural',p)],axis=1);c,sv,rank=B.fit(A,target)
    return jnp.mean(jnp.sum((A@jax.lax.stop_gradient(c)-target)**2,axis=0))
   @jax.jit
   def step(p,state,target,coarse,xy):
    val,g=jax.value_and_grad(loss)(p,target,coarse,xy);u,state=opt.update(g,state,p);return optax.apply_updates(p,u),state,val
   return step
  updates[ci,k]=make_update(patches,k)
start=time.perf_counter();best=np.inf
for it in range(20000):
 cls=int(rng.integers(3));ci=int(rng.choice(([0,1],list(range(2,6)),[6,7])[cls]));ks=[k for k in (4,8,16,32) if (ci,k) in updates];k=int(rng.choice(ks))
 idx=rng.integers(32,size=8)*11+rng.integers(11,size=8)
 params,state,loss=updates[ci,k](params,state,Ut[:,idx],V[:,:32],xy);loss=float(loss);assert np.isfinite(loss)
 if (it+1)%1000==0:
  # Select by fixed development projection, not training loss.
  A=jnp.concatenate([V[:,:32],B.features(xy,base,32,'neural',params)],axis=1);c,_,_=B.fit(A,Ud)
  score=float(jnp.mean(jnp.sum((A@c-Ud)**2,axis=0)));rep['training'].append(dict(step=it+1,loss=loss,development=score,seconds=time.perf_counter()-start))
  with open(o/f'step{it+1}.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,params),f)
  if score<best:
   best=score
   with open(o/'best.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,params),f)
  save(o/'result.json',rep);print('step',it+1,score,flush=True)
params=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(o/'best.pkl','rb')));assess('mixed_training_repair',params)
rep['complete']=True;save(o/'result.json',rep)
