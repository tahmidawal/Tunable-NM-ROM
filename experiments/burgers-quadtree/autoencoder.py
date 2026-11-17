"""Matched-data nonlinear POD autoencoder, full weak solve control (no hyper-reduction claim)."""
from campaign import *
from owned_inputs import verified
import local_bank as B
import optax,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
U=np.load(verified('representation01','snapshots.npz'))['fields'];raw=U[:,:,1:-1,1:-1].reshape(40,11,-1)
norm=np.linalg.norm(raw[:,0],axis=1);Y=jnp.asarray((raw/norm[:,None,None]).reshape(440,-1).T)
V=jnp.asarray(np.load(verified('representation01','global_pod.npz'))['basis']);rng=np.random.default_rng(2026111706)
rep=dict(meta=meta(),rows=[],training=[],status='projection comparator, PDE manifold path next')
def init_mlp(key,dims):
 return [(jax.random.normal(q,(a,b),dtype=jnp.float64)*np.sqrt(2/(a+b)),jnp.zeros(b)) for q,a,b in zip(jax.random.split(key,len(dims)-1),dims[:-1],dims[1:])]
def run(p,x):
 for w,b in p[:-1]:x=jnp.tanh(x@w+b)
 return x@p[-1][0]+p[-1][1]
for lift in (128,256):
 coeff=(V[:,:lift].T@Y).T;tr=coeff[:352];de=coeff[352:]
 trunc=np.asarray(jnp.linalg.norm(V[:,:lift]@de.T-Y[:,352:],axis=0)).reshape(8,11)
 for latent in (16,32,64):
  key=jax.random.key(2026111706+lift+latent);ke,kd=jax.random.split(key)
  params=(init_mlp(ke,[lift,128,128,latent]),init_mlp(kd,[latent,128,128,lift]))
  opt=optax.adam(optax.cosine_decay_schedule(1e-3,10000,alpha=.01));state=opt.init(params)
  def loss(p,x):return jnp.mean(jnp.sum((run(p[1],run(p[0],x))-x)**2,axis=1))
  @jax.jit
  def step(p,s,x):
   val,g=jax.value_and_grad(loss)(p,x);u,s=opt.update(g,s,p);return optax.apply_updates(p,u),s,val
  best=np.inf;t=time.perf_counter()
  for it in range(10000):
   idx=rng.integers(0,352,64);params,state,value=step(params,state,tr[idx])
   if (it+1)%1000==0:
    er=np.asarray(jnp.linalg.norm(V[:,:lift]@run(params[1],run(params[0],de)).T-Y[:,352:],axis=0)).reshape(8,11);score=float(np.mean(er**2))
    rep['training'].append(dict(lift=lift,latent=latent,step=it+1,loss=float(value),development=score))
    if score<best:
     best=score
     with open(o/f'ae_lift{lift}_latent{latent}.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,params),f)
  params=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(o/f'ae_lift{lift}_latent{latent}.pkl','rb')))
  codes=run(params[0],de);pred=V[:,:lift]@run(params[1],codes).T
  er=np.asarray(jnp.linalg.norm(pred-Y[:,352:],axis=0)).reshape(8,11)
  rep['rows'].append(dict(lift=lift,latent=latent,seconds=time.perf_counter()-t,projection_error=er,truncation_error=trunc,evolved_max=er[:,[2,4,6,8,10]].max(1)))
  np.savez_compressed(o/f'ae_lift{lift}_latent{latent}.npz',codes=codes,prediction=pred)
  save(o/'result.json',rep);print(lift,latent,er[:,[2,4,6,8,10]].max(),flush=True)
rep['complete']=True;save(o/'result.json',rep)
