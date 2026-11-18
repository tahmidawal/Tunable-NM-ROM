"""Conditional local-frame architecture screen with independent finer-grid rescore."""
from campaign import *
from owned_inputs import verified
import frame_geometry as FG,specialized as S,local_bank as B,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--geometry',choices=['diffusive','initial'],default='diffusive');a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1local','network.pkl'),'rb')));teacher=jnp.asarray(np.load(verified('c1local','teacher.npz'))['features'].reshape(33,33,64));V=jnp.asarray(np.load(verified('c1data','pod.npz'))['basis'])[:,:64];_,dev=draw();grids={};coarse={};rows=[]
dims=[18,256,256,64];randomp=[(jax.random.normal(key,(a,b),dtype=jnp.float64)*np.sqrt(2/(a+b)),jnp.zeros(b)) for key,a,b in zip(jax.random.split(jax.random.key(2026111810),3),dims[:-1],dims[1:])]
for N in (128,256):
 grids[N]=jnp.asarray(E.coords(N,True));ls=P.lattice_S(128,N)
 fields=jax.vmap(lambda v:P.synth_lattice(2*v.reshape(127,127),ls,2),in_axes=1)(V);coarse[N]=fields[:,1:-1,1:-1].reshape(64,-1).T
recipes=[(r,k,kind) for r in (32,64) for k in (32,64) for kind in ('poly','neural','hybrid')]+[(64,64,'local_pod'),(64,64,'random')]
for case,ph in enumerate(dev[:32]):
 patch=FG.geometry(E.initial(4095,ph),ph[4],mode=a.geometry);blocks={};truth={};d=np.load(verified('c1data',f'case{128+case}.npz'));norm=float(d['initial_norm'])
 for N in (128,256):
  data=d if N==128 else np.load(verified('c1data',f'check{128+case}.npz'));ls=P.lattice_S(N,N);field=jax.vmap(lambda b:P.synth_lattice(b,ls,2))(jnp.asarray(data['coefficients']));truth[N]=field[:,1:-1,1:-1].reshape(11,-1).T/norm
  xy=grids[N];blocks[N]={kind:(S.features(p if kind=='neural' else randomp,xy,[patch],64) if kind in ('neural','random') else B.features(xy,[patch],64,'poly' if kind=='poly' else 'pod',pod=teacher)) for kind in ('poly','neural','random','local_pod')}
 for r,k,kind in recipes:
  def matrix(N):
   local=jnp.concatenate([blocks[N]['poly'][:,:k//2],blocks[N]['neural'][:,:k//2]],axis=1) if kind=='hybrid' else blocks[N][kind][:,:k]
   return jnp.concatenate([coarse[N][:,:r],local],axis=1)
  low=matrix(128);start=time.perf_counter();coeff,s,rank=B.fit(low/128,truth[128]/128);jax.block_until_ready(coeff);elapsed=time.perf_counter()-start
  errors={N:np.asarray(jnp.linalg.norm(matrix(N)@coeff-truth[N],axis=0))/N for N in (128,256)}
  np.savez_compressed(o/f'case{case}_r{r}_k{k}_{kind}.npz',coefficients=coeff,singular_values=s,patch=patch,error128=errors[128],error256=errors[256])
  rows.append(dict(case=case,r=r,k=k,kind=kind,patch=patch,rank=int(rank),columns=r+k,error128=errors[128],error256=errors[256],evolved=float(errors[256][[2,4,6,8,10]].max()),score_change=float(np.max(np.abs(errors[128]-errors[256]))),fit_seconds=elapsed))
 save(o/'result.json',dict(meta=meta(),rows=rows));print(case,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,geometry=a.geometry,contract='geometry inferred only from supplied initial array and viscosity; same TRAIN-frozen features; finer-grid score uses unchanged fitted coefficients; not PDE or cost claim'))
