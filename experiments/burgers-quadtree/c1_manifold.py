"""Expanded matched-data nonlinear POD AE: encoder vs optimized projection."""
from campaign import *
from owned_inputs import verified
import manifold_core as M,optax,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
Y=[]
for i in range(160):
 d=np.load(verified('c1data',f'case{i}.npz'));Y.append(np.asarray(d['coefficients']).reshape(11,-1).T/(2*float(d['initial_norm'])))
U=jnp.asarray(np.concatenate(Y[:128],axis=1));D=jnp.asarray(np.concatenate(Y[128:],axis=1));V,s,_=jnp.linalg.svd(U,full_matrices=False);V=V[:,:1024]
np.savez_compressed(o/'lift.npz',basis=V,singular_values=s)
tr=(V.T@U).T;de=(V.T@D).T;trunc=np.asarray(jnp.linalg.norm(D-V@de.T,axis=0)).reshape(32,11)
latent=128;ke,kd=jax.random.split(jax.random.key(2026111803));p=(M.init(ke,[1024,256,256,latent]),M.init(kd,[latent,256,256,1024]))
opt=optax.adam(optax.cosine_decay_schedule(3e-4,30000,alpha=.01));state=opt.init(p)
def loss(p,x):return jnp.mean(jnp.sum((M.decode(p[1],M.encode(p[0],x,latent))-x)**2,axis=1))
@jax.jit
def update(p,state,x):
 val,g=jax.value_and_grad(loss)(p,x);u,state=opt.update(g,state,p);return optax.apply_updates(p,u),state,val
rng=np.random.default_rng(2026111803);history=[];best=np.inf;t=time.perf_counter()
for it in range(30000):
 p,state,val=update(p,state,tr[rng.integers(len(tr),size=64)])
 if (it+1)%3000==0:
  er=np.asarray(jnp.linalg.norm(M.decode(p[1],M.encode(p[0],de,latent))-de,axis=1)).reshape(32,11);score=float(np.mean(er[:,[2,4,6,8,10]]**2))
  history.append(dict(step=it+1,loss=float(val),score=score))
  with open(o/f'step{it+1}.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
  if score<best:
   best=score
   with open(o/'best.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
  print('train',it+1,score,flush=True)
train_seconds=time.perf_counter()-t;p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(o/'best.pkl','rb')));codes=M.encode(p[0],de,latent)
encoder=np.sqrt(np.asarray(jnp.sum((M.decode(p[1],codes)-de)**2,axis=1)).reshape(32,11)+trunc**2)
@jax.jit
def project(x,z):return M.lm(lambda a:M.decode(p[1],a)-x,z,60)
# Two prescribed starts, save both. Optimization is a representation diagnostic only.
zs=[];grad=[];errs=[];seconds=[]
for i in range(len(de)):
 runs=[];t=time.perf_counter()
 for z0 in (codes[i],de[i,:latent]):
  z,g,e=project(de[i],z0);jax.block_until_ready(z);runs.append((z,float(g),float(e)))
 j=min(range(2),key=lambda j:runs[j][2]);zs.append(runs[j][0]);grad.append([r[1] for r in runs]);errs.append([r[2] for r in runs]);seconds.append(time.perf_counter()-t)
 if i%44==0:print('projection',i,errs[-1],flush=True)
optimized=np.sqrt(np.min(np.asarray(errs),axis=1).reshape(32,11)**2+trunc**2)
np.savez_compressed(o/'projection.npz',codes=jnp.stack(zs),encoder_error=encoder,optimized_error=optimized,truncation_error=trunc,stationarity=grad,start_errors=errs,seconds=seconds)
save(o/'result.json',dict(meta=meta(),complete=True,training=history,training_seconds=train_seconds,lift=1024,latent=latent,encoder_worst=float(encoder[:,[2,4,6,8,10]].max()),optimized_worst=float(optimized[:,[2,4,6,8,10]].max()),truncation_worst=float(trunc[:,[2,4,6,8,10]].max()),optimized_cases_below_half_percent=int((optimized[:,[2,4,6,8,10]].max(1)<=.005).sum()),max_stationarity=float(np.max(grad)),contract='two-start optimized manifold projection, not a PDE solve; convergence diagnostics retained'))
