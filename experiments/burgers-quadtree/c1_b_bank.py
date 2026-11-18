"""Broader-coverage specialized local neural features, with independent span assessment."""
from campaign import *
from owned_inputs import verified
import local_bank as B,continuous as C,argparse,optax,pickle
from scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
train=[];dev=[]
for i in range(40):
 d=np.load(verified('c1familyb',f'case{i}.npz'));Y=np.asarray(d['coefficients'])[:,:127,:127].reshape(11,-1).T/(2*float(d['initial_norm']))
 (train if i<32 else dev).append(Y)
U=jnp.asarray(np.concatenate(train,axis=1));D=jnp.asarray(np.concatenate(dev,axis=1));V,sv,_=jnp.linalg.svd(U,full_matrices=False);np.savez_compressed(o/'pod.npz',basis=V,singular_values=sv);coarse=V[:,:64]
res=U-coarse@(coarse.T@U)
# Teacher is TRAIN-only; local network is subsequently assessed in assembled physical norm.
fields=np.asarray(jax.vmap(lambda b:P.synth_lattice(2*b.reshape(127,127),P.lattice_S(128,128),2),in_axes=1)(res))
z=np.linspace(-.25,1.25,33);zz=np.stack(np.meshgrid(z,z,indexing='ij'));patches=B.uniform(2)
cov=np.zeros((1089,1089))
for p in patches:
 coords=(np.asarray(p[:2])[:,None,None]+p[2]*zz)*128
 Y=np.stack([map_coordinates(f,coords,order=1,mode='constant') for f in fields],axis=-1).reshape(1089,-1)
 cov+=Y@Y.T/Y.shape[1]
s,teacher=np.linalg.eigh(cov);teacher=teacher[:,::-1][:,:64];s=s[::-1]
# Equal feature fitting errors; retain teacher eigenvalues for later weighting diagnosis.
np.savez_compressed(o/'teacher.npz',features=teacher,eigenvalues=s)
xy=jnp.asarray(zz.reshape(2,-1).T);target=jnp.asarray(teacher)
def encode(x):
 f=jnp.array([1.,2.,4.,8.]);return jnp.concatenate([x,jnp.sin(jnp.pi*x[:,:,None]*f).reshape(len(x),-1),jnp.cos(jnp.pi*x[:,:,None]*f).reshape(len(x),-1)],axis=1)
def init():
 dims=[18,256,256,64];return [(jax.random.normal(k,(a,b),dtype=jnp.float64)*np.sqrt(2/(a+b)),jnp.zeros(b)) for k,a,b in zip(jax.random.split(jax.random.key(2026111811),3),dims[:-1],dims[1:])]
def net(p,x):
 x=encode(x)
 for w,b in p[:-1]:x=jnp.tanh(x@w+b)
 return x@p[-1][0]+p[-1][1]
p=init();opt=optax.adam(optax.cosine_decay_schedule(.001,20000,alpha=.01));state=opt.init(p)
@jax.jit
def update(p,state,x,y):
 def loss(p):return jnp.mean((net(p,x)-y)**2)
 loss,g=jax.value_and_grad(loss)(p);updates,state=opt.update(g,state,p);return optax.apply_updates(p,updates),state,loss
rng=np.random.default_rng(2026111811);losses=[]
for it in range(20000):
 idx=rng.integers(0,1089,512);p,state,l=update(p,state,xy[idx],target[idx])
 if (it+1)%1000==0:losses.append(dict(step=it+1,loss=float(l)));print('fit',it+1,float(l),flush=True)
with open(o/'network.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
rows=[];points=jnp.asarray(E.coords(128,True))
for kind in ('poly','neural','hybrid'):
 for r,depth,k in [(64,2,16),(64,2,32),(128,2,16),(128,2,32)]:
  blocks=[]
  for patch in B.uniform(depth):
   zz=(points-jnp.asarray(patch[:2]))/patch[2]
   if kind=='neural':f=net(p,zz)[:,:k]*B.window(points,patch)[:,None]
   elif kind=='hybrid':f=jnp.concatenate([B.features(points,[patch],k,'poly'),net(p,zz)[:,:k]*B.window(points,patch)[:,None]],axis=1)
   else:f=B.features(points,[patch],k,'poly')
   coef=jax.vmap(lambda f:P.analysis(f.reshape(127,127),128).reshape(-1)/2,in_axes=1,out_axes=1)(f)
   blocks.append(coef)
  A=jnp.concatenate([V[:,:r]]+blocks,axis=1);u,s,vh=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;Q=u[:,keep]
  err=np.asarray(jnp.linalg.norm(D-Q@(Q.T@D),axis=0)).reshape(8,11)
  row=dict(kind=kind,r=r,depth=depth,k=k,columns=A.shape[1],rank=int(keep.sum()),per_time=err,evolved=err[:,[2,4,6,8,10]].max(1),initial=err[:,0]);rows.append(row)
  np.savez_compressed(o/f'{kind}_r{r}_d{depth}_k{k}.npz',basis=Q,singular_values=s,physical_to_orthogonal=vh[keep].T/s[keep])
  save(o/'result.json',dict(meta=meta(),rows=rows,training=losses));print(kind,r,depth,k,max(row['evolved']),flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,training=losses,complete=True,status='family B matched-data train32/dev8 span; no PDE or output claim'))
