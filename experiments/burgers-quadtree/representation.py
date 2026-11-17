from campaign import *
import local_bank as B
import snapshots
import optax,argparse,pickle
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
rep=dict(meta=meta(),status='provisional representation only; no PDE or qualified reference claim',rows=[],losses=[])
train,dev=draw();np.savez(o/'parameters.npz',train=train,dev=dev)
f=snapshots.make();arrays=[]
for i,ph in enumerate(np.concatenate([train[:32],dev[:8]])):
 b=jnp.asarray(P.ic_coefs_2d(ph,128,256));v,sec=timer(lambda:f(b,ph[4]));v=np.array(v);v[0]=E.initial(64,ph)
 assert np.isfinite(v).all();arrays.append(v);print('data',i,sec,flush=True)
U=np.stack(arrays);np.savez_compressed(o/'snapshots.npz',fields=U,parameters=np.concatenate([train[:32],dev[:8]]))
xy=jnp.asarray(E.coords(64,True));raw=U[:,:,1:-1,1:-1].reshape(40,11,-1)
norm=np.linalg.norm(raw[:,0],axis=1);Y=(raw/norm[:,None,None]).reshape(440,-1).T
# Physical interior nodal quadrature is uniform; boundary fields vanish exactly.
Ut=jnp.asarray(Y[:,:352]);Ud=jnp.asarray(Y[:,352:]);left,s,_=jnp.linalg.svd(Ut,full_matrices=False)
coarse=left[:,:16];np.savez(o/'global_pod.npz',basis=left,singular_values=s)
patches=B.uniform(1)
# Shared local POD, trained only on missing detail, on a canonical halo grid.
res=np.asarray(Ut-coarse@(coarse.T@Ut)).reshape(63,63,-1)
from scipy.ndimage import map_coordinates
z=np.linspace(-.25,1.25,17);zz=np.stack(np.meshgrid(z,z,indexing='ij'),axis=0)
patchdata=[]
for p in patches:
 coords=(np.asarray(p[:2])[:,None,None]+p[2]*zz)*64-1
 patchdata.append(np.stack([map_coordinates(res[:,:,i],coords,order=1,mode='constant') for i in range(res.shape[-1])],axis=-1).reshape(289,-1))
pu,_,_=np.linalg.svd(np.concatenate(patchdata,axis=1),full_matrices=False);pod=jnp.asarray(pu[:,:32].reshape(17,17,32));np.save(o/'local_pod.npy',pod)

def evaluate(kind,params=None):
 for depth in (1,2):
  for k in (4,8,16,32):
   if 16+4**depth*k>512:continue
   t=time.perf_counter();A=jnp.concatenate([coarse,B.features(xy,B.uniform(depth),k,kind,params,pod)],axis=1)
   c,sv,r=B.fit(A,Ud);pred=A@c;err=np.asarray(jnp.linalg.norm(pred-Ud,axis=0)).reshape(8,11)
   rep['rows'].append(dict(kind=kind,depth=depth,k=k,rank=int(r),columns=A.shape[1],min_sv=float(sv[-1]),max_error_by_case=err.max(1),per_time=err,assembly_fit_seconds=time.perf_counter()-t))
   np.savez_compressed(o/f'{kind}_d{depth}_k{k}.npz',coefficients=c,singular_values=sv,prediction=pred)
   save(o/'result.json',rep)
evaluate('poly');evaluate('pod')
# First finite throughput recipe: uniform depth1 assembled objective, 8 equally sampled trajectory-time pairs.
# Other configuration classes are a registered follow-up, not claimed trained here.
params=B.init(jax.random.key(2026111704));opt=optax.adam(1e-3);state=opt.init(params)
def loss(params,target):
 A=jnp.concatenate([coarse,B.features(xy,patches,8,'neural',params)],axis=1)
 c,sv,r=B.fit(A,target);c=jax.lax.stop_gradient(c)
 return jnp.mean(jnp.sum((A@c-target)**2,axis=0))
@jax.jit
def update(params,state,target):
 value,grad=jax.value_and_grad(loss)(params,target);upd,state=opt.update(grad,state,params)
 return optax.apply_updates(params,upd),state,value
rng=np.random.default_rng(2026111704);start=time.perf_counter()
for it in range(200):
 idx=rng.integers(0,32,8)*11+rng.integers(0,11,8)
 params,state,value=update(params,state,Ut[:,idx]);value=float(value)
 assert np.isfinite(value),f'nonfinite loss at {it}'
 rep['losses'].append(value)
 if it%20==0:print('train',it,value,flush=True);save(o/'result.json',rep)
rep['training_seconds']=time.perf_counter()-start
with open(o/'shared_neural.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,params),f)
evaluate('neural',params)
rep['complete']=True;save(o/'result.json',rep)
