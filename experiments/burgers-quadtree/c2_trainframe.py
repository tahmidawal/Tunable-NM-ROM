"""Separate geometry transfer from TRAIN-only aligned teacher/neural learning."""
from campaign import *
from owned_inputs import verified
import frame_geometry as FG,frame_features as FF,local_bank as B,optax,pickle,argparse,subprocess
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'c2_frame.py','--geometry','initial','--out',str(o/'transfer')],check=True)
train,dev=draw();V=jnp.asarray(np.load(verified('c1data','pod.npz'))['basis'])[:,:64];z=np.linspace(-.25,1.25,65);zz=np.stack(np.meshgrid(z,z,indexing='ij'),axis=-1).reshape(-1,2);Y=[];weights=[];geometries=[];k=jnp.arange(1,128);start=time.perf_counter()
for i,ph in enumerate(train[:128]):
 patch=FG.geometry(E.initial(4095,ph),ph[4],mode='initial');geometries.append(patch);d=np.load(verified('c1data',f'case{i}.npz'));b=jnp.asarray(d['coefficients']).reshape(11,-1).T/float(d['initial_norm']);b=b-V@(V.T@b);b=b.T.reshape(11,127,127)
 xx=patch[0]+patch[2]*jnp.asarray(z);yy=patch[1]+patch[2]*jnp.asarray(z);sx=jnp.sin(jnp.pi*xx[:,None]*k);sy=jnp.sin(jnp.pi*yy[:,None]*k);field=jnp.einsum('xi,tij,yj->txy',sx,b,sy).reshape(11,-1).T
 xy=jnp.asarray(patch[:2])+patch[2]*jnp.asarray(zz);win=B.window(xy,patch);valid=jnp.all((xy>=0)&(xy<=1),axis=1);w=win**2*valid*(patch[2]*1.5/64)**2
 target=field*win[:,None]/jnp.maximum(win[:,None]**2,1e-12);Y.append(target);weights.append(w)
Y=jnp.stack(Y);weights=jnp.stack(weights);teacher,eigenvalues,als=FF.weighted_teacher(Y,weights,128,10);teacher_seconds=time.perf_counter()-start;meanw=np.asarray(jnp.mean(weights,axis=0));np.savez_compressed(o/'teacher.npz',features=teacher,eigenvalues=eigenvalues,mean_weights=meanw,z=z,geometries=geometries)
p=FF.init(jax.random.key(2026111812));opt=optax.adam(optax.cosine_decay_schedule(.001,30000,alpha=.01));state=opt.init(p);xy=jnp.asarray(zz);rng=np.random.default_rng(2026111812);prob=meanw/meanw.sum();history=[];start=time.perf_counter()
@jax.jit
def update(p,state,x,y):
 loss,g=jax.value_and_grad(lambda p:jnp.mean((FF.apply(p,x)-y)**2))(p);updates,state=opt.update(g,state,p);return optax.apply_updates(p,updates),state,loss
for it in range(30000):
 idx=rng.choice(len(zz),512,p=prob);p,state,loss=update(p,state,xy[idx],teacher[idx])
 if (it+1)%3000==0:
  history.append(dict(step=it+1,loss=float(loss)));print('teacher neural',it+1,float(loss),flush=True)
  with open(o/f'step{it+1}.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
with open(o/'network.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
training_seconds=time.perf_counter()-start;rows=[];coarse={};points={}
for N in (128,256):
 points[N]=jnp.asarray(E.coords(N,True));ls=P.lattice_S(128,N);coarse[N]=jax.vmap(lambda v:P.synth_lattice(2*v.reshape(127,127),ls,2),in_axes=1)(V)[:,1:-1,1:-1].reshape(64,-1).T
pod=teacher.reshape(65,65,128)
for i,ph in enumerate(dev[:32]):
 patch=FG.geometry(E.initial(4095,ph),ph[4],mode='initial');blocks={};truth={};norm=float(np.load(verified('c1data',f'case{128+i}.npz'))['initial_norm'])
 for N in (128,256):
  d=np.load(verified('c1data',f'{"case" if N==128 else "check"}{128+i}.npz'));ls=P.lattice_S(N,N);truth[N]=jax.vmap(lambda b:P.synth_lattice(b,ls,2))(jnp.asarray(d['coefficients']))[:,1:-1,1:-1].reshape(11,-1).T/norm
  blocks[N]={'neural':FF.features(p,points[N],patch,128),'local_pod':B.features(points[N],[patch],128,'pod',pod=pod),'poly':B.features(points[N],[patch],64,'poly')}
 for r in (32,64):
  for kind in ('neural','local_pod','hybrid'):
   for n in ((32,64,128) if kind!='hybrid' else (64,128)):
    def matrix(N):
     f=jnp.concatenate([blocks[N]['poly'][:,:n//2],blocks[N]['neural'][:,:n//2]],axis=1) if kind=='hybrid' else blocks[N][kind][:,:n]
     return jnp.concatenate([coarse[N][:,:r],f],axis=1)
    c,s,rank=B.fit(matrix(128)/128,truth[128]/128);errs={N:np.asarray(jnp.linalg.norm(matrix(N)@c-truth[N],axis=0))/N for N in (128,256)}
    np.savez_compressed(o/f'case{i}_r{r}_k{n}_{kind}.npz',coefficients=c,singular_values=s,patch=patch,error128=errs[128],error256=errs[256])
    rows.append(dict(case=i,r=r,k=n,kind=kind,rank=int(rank),error128=errs[128],error256=errs[256],evolved=float(errs[256][[2,4,6,8,10]].max()),evolved_score_change=float(np.max(np.abs(errs[128][[2,4,6,8,10]]-errs[256][[2,4,6,8,10]])))))
 save(o/'result.json',dict(meta=meta(),rows=rows,als=als,training=history,teacher_seconds=teacher_seconds,training_seconds=training_seconds));print('frame projection',i,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,als=als,training=history,teacher_seconds=teacher_seconds,training_seconds=training_seconds,complete=True,contract='TRAIN-only geometry-aligned weighted local POD and neural teacher; independent finer-grid same-coefficient rescore; no PDE claim'))
