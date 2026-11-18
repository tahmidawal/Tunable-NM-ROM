"""Additional output-heavy and transfer-inclusive development contracts, no test data."""
from campaign import *
from owned_inputs import verified
from spectral_cost import PJ,make
import continuous as C,family_b as FB,quadtree as QT,argparse
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
xy=jnp.asarray(E.coords(512,True));x=jnp.arange(4096,dtype=jnp.float64)*256/4095;coords=jnp.stack(jnp.meshgrid(x,x,indexing='ij')).reshape(2,-1)
@jax.jit
def extend(c):return map_coordinates(PJ.synth_mesh(c,256),coords,order=1,mode='nearest').reshape(4096,4096)
@jax.jit
def four(c):return jax.vmap(extend)(c)
if a.smoke:
 c=jnp.zeros((4,3,3)).at[:,0,0].set(jnp.arange(1,5,dtype=jnp.float64));v=four(c);jax.block_until_ready(v);assert v.shape==(4,4096,4096);norms=np.asarray(jnp.linalg.norm(v.reshape(4,-1),axis=1));assert np.allclose(norms/norms[0],[1,2,3,4],rtol=1e-12);save(o/'smoke.json',dict(meta=meta(),shape=list(v.shape),norms=norms));print('four-output materialization passed',flush=True);raise SystemExit
peak=jax.jit(lambda u:jnp.argmax(u));rows=[];rng=np.random.default_rng(2026111818)
for family in ('A','B'):
 models={};setup=time.perf_counter();dt=.0125 if family=='A' else .00625;f=C.flux_trajectory(dt)
 if family=='A':
  for i in range(3):
   for j in range(3):
    d=np.load(verified('c1compress',f'tile{i}{j}_basis.npz'));W=jnp.asarray(d['basis'])[:,:384];e=jnp.asarray(d['eigenvalues'])[:384];G,H=C.flux_tables(W,128,128);models[i,j]=(W,e,G,H)
 else:
  import itertools
  for split in itertools.combinations(range(4),2):
   d=np.load(verified('c1bcomplete','split'+''.join(map(str,split))+'_basis.npz'));W=jnp.asarray(d['basis']);e=jnp.asarray(d['eigenvalues']);G,H=C.flux_tables(W,128,256);models[split]=(W,e,G,H)
 jax.block_until_ready(models);setup_seconds=time.perf_counter()-setup
 @jax.jit
 def solve(u,nu,W,e,G,H):
  b=P.analysis(map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511),512)[:127,:127];states=f(W.T@b.ravel()/2,nu,G,H,e);coeff=(2*(states@W.T)).reshape(11,127,127);return coeff,extend(coeff[-1])
 def local(u,nu):
  if family=='A':
   index=int(peak(u));key=tuple(min(2,int(v*4/4096)) for v in divmod(index,4096))
  else:key=tuple(sorted(QT.initial_order(u)[:2]))
  return solve(u,nu,*models[key])
 # Fixed classical settings registered before this contract; no per-case winner selection.
 queries={'local':local,'spectral32':make(32,dt,512,'bilinear256'),'spectral48':make(48,dt,512,'bilinear256'),'spectral24':make(24,dt,512,'bilinear256')}
 params=draw()[1][:32] if family=='A' else json.loads(verified('c1familyb','parameters.json').read_text())['dev'][:8]
 for case,ph in enumerate(params):
  host=E.initial(4095,ph) if family=='A' else FB.initial(4095,ph);u=jnp.asarray(host);jax.block_until_ready(u);nu=ph[4] if family=='A' else ph['nu'];times={};cold={};last={}
  for contract in ('five_resident','final_host','five_host'):
   def run(name):
    inp=jnp.asarray(host) if contract.endswith('host') else u;c,final=queries[name](inp,nu)
    if contract.startswith('five'):
     first=c[jnp.array([2,4,6,8])] if name=='local' else c[1:5];fields=(four(first),final)
    else:fields=(final,)
    if contract.endswith('host'):fields=jax.tree_util.tree_map(np.asarray,fields)
    return c,fields
   for name in queries:last[name],cold[contract,name]=timer(lambda:run(name))
   times[contract]={name:[[],[]] for name in queries}
   for block in range(2):
    for rep in range(3):
     for name in rng.permutation(list(queries)):
      E.burn(.2);last[name],sec=timer(lambda:run(name));times[contract][name][block].append(sec)
   for name,(c,fields) in last.items():np.savez_compressed(o/f'{family}_{contract}_{name}_case{case}.npz',coefficients=c,output_norms=[float(np.linalg.norm(np.asarray(v))) for v in fields],output_shapes_json=json.dumps([list(v.shape) for v in fields]))
  rows.append(dict(family=family,case=case,times=times,cold={f'{k[0]}:{k[1]}':v for k,v in cold.items()},setup_seconds=setup_seconds));save(o/'result.json',dict(meta=meta(),rows=rows));print(family,case,flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='development supplemental output-heavy and host-transfer contracts; all five actual4096 fields materialized, common inputs; three repetitions in two blocks; no superiority claim without paired per-contract accuracy qualification'))
