"""Expanded matched-data nonlinear POD AE: encoder vs optimized projection."""
from campaign import *
from owned_inputs import verified
import manifold_core as M,optax,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
Y=[]
for i in range(160):
 d=np.load(verified('c1data',f'case{i}.npz'));Y.append(np.asarray(d['coefficients']).reshape(11,-1).T/(2*float(d['initial_norm'])))
U=jnp.asarray(np.concatenate(Y[:128],axis=1));D=jnp.asarray(np.concatenate(Y[128:],axis=1));V=jnp.asarray(np.load(verified('c1manifold','lift.npz'))['basis']);s=jnp.asarray(np.load(verified('c1manifold','lift.npz'))['singular_values'])
np.savez_compressed(o/'lift.npz',basis=V,singular_values=s)
tr=(V.T@U).T;de=(V.T@D).T;trunc=np.asarray(jnp.linalg.norm(D-V@de.T,axis=0)).reshape(32,11)
mean=jnp.mean(tr,axis=0);scale=jnp.maximum(jnp.std(tr,axis=0),1e-5);np.savez_compressed(o/'normalization.npz',mean=mean,scale=scale)
def enc(p,x,latent):return M.encode(p,(x-mean)/scale,latent)
def dec(p,z):return M.decode(p,z)*scale+mean
latent=128;ke,kd=jax.random.split(jax.random.key(2026111808));p=(M.init(ke,[1024,512,512,latent]),M.init(kd,[latent,512,512,1024]))
opt=optax.adam(optax.cosine_decay_schedule(3e-4,50000,alpha=.01));state=opt.init(p)
def loss(p,x):return jnp.mean(jnp.sum((dec(p[1],enc(p[0],x,latent))-x)**2,axis=1))
@jax.jit
def update(p,state,x):
 val,g=jax.value_and_grad(loss)(p,x);u,state=opt.update(g,state,p);return optax.apply_updates(p,u),state,val
rng=np.random.default_rng(2026111808);history=[];best=np.inf;t=time.perf_counter()
for it in range(50000):
 p,state,val=update(p,state,tr[rng.integers(len(tr),size=64)])
 if (it+1)%5000==0:
  er=np.asarray(jnp.linalg.norm(dec(p[1],enc(p[0],de,latent))-de,axis=1)).reshape(32,11);score=float(np.mean(er[:,[2,4,6,8,10]]**2))
  history.append(dict(step=it+1,loss=float(val),score=score))
  with open(o/f'step{it+1}.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
  if score<best:
   best=score
   with open(o/'best.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
  print('train',it+1,score,flush=True)
train_seconds=time.perf_counter()-t;p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(o/'best.pkl','rb')));codes=enc(p[0],de,latent)
encoder=np.sqrt(np.asarray(jnp.sum((dec(p[1],codes)-de)**2,axis=1)).reshape(32,11)+trunc**2)
@jax.jit
def project(x,z):return M.lm(lambda a:dec(p[1],a)-x,z,60)
# Two prescribed starts, save both. Optimization is a representation diagnostic only.
zs=[];grad=[];errs=[];seconds=[]
for i in range(len(de)):
 runs=[];t=time.perf_counter()
 for z0 in (codes[i],(de[i,:latent]-mean[:latent])/scale[:latent]):
  z,g,e=project(de[i],z0);jax.block_until_ready(z);runs.append((z,float(g),float(e)))
 j=min(range(2),key=lambda j:runs[j][2]);zs.append(runs[j][0]);grad.append([r[1] for r in runs]);errs.append([r[2] for r in runs]);seconds.append(time.perf_counter()-t)
 if i%44==0:print('projection',i,errs[-1],flush=True)
optimized=np.sqrt(np.min(np.asarray(errs),axis=1).reshape(32,11)**2+trunc**2)
np.savez_compressed(o/'projection.npz',codes=jnp.stack(zs),encoder_error=encoder,optimized_error=optimized,truncation_error=trunc,stationarity=grad,start_errors=errs,seconds=seconds)
save(o/'result.json',dict(meta=meta(),complete=True,training=history,training_seconds=train_seconds,lift=1024,latent=latent,encoder_worst=float(encoder[:,[2,4,6,8,10]].max()),optimized_worst=float(optimized[:,[2,4,6,8,10]].max()),truncation_worst=float(trunc[:,[2,4,6,8,10]].max()),optimized_cases_below_half_percent=int((optimized[:,[2,4,6,8,10]].max(1)<=.005).sum()),max_stationarity=float(np.max(grad)),contract='TRAIN-whitened wider POD AE, two-start optimized projection; not PDE solve; normalization and stationarity saved'))
