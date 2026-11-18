"""Qualify the same compressed physical space, then complete matched-output costs."""
from campaign import *
from owned_inputs import verified
from spectral_cost import PJ,make
import continuous as C,specialized as S,local_bank as B,pickle,argparse
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();tiles=json.loads(verified('c1tiles','result.json').read_text())['rows'];models={};resolution=[];setup=time.perf_counter()
# Complete nine-branch cache, including branches absent from development.
for i in range(3):
 for j in range(3):
  d=np.load(verified('c1compress',f'tile{i}{j}_basis.npz'));W=jnp.asarray(d['basis'])[:,:384];e=jnp.asarray(d['eigenvalues'])[:384];G,H=C.flux_tables(W,128,128);jax.block_until_ready((W,e,G,H));models[i,j]=(W,e,G,H)
setup_seconds=time.perf_counter()-setup
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1local','network.pkl'),'rb')));V=np.load(verified('c1data','pod.npz'))['basis'][:,:256].reshape(127,127,256);V=jnp.asarray(np.pad(V,((0,128),(0,128),(0,0))).reshape(255**2,256));points=jnp.asarray(E.coords(256,True))
def analysis(F):return jax.vmap(lambda f:P.analysis(f.reshape(255,255),256).ravel()/2,in_axes=1,out_axes=1)(F)
for (i,j),(oldW,_,_,_) in models.items():
 cases=[r['case'] for r in tiles if r['rank']==256 and r['tile']==[i,j]]
 if not cases:continue
 blocks=[V]
 for ii in (i,i+1):
  for jj in (j,j+1):
   patch=[(ii/4,jj/4,.25)];blocks.append(jnp.concatenate([analysis(B.features(points,patch,32,'poly')),analysis(S.features(p,points,patch,32))],axis=1))
 A=jnp.concatenate(blocks,axis=1);d=np.load(verified('c1tiles',f'r256_tile{i}{j}.npz'));transform=jnp.asarray(d['physical_to_orthogonal'])@(jnp.asarray(d['basis']).T@oldW)
 U,s,vh=jnp.linalg.svd(A@transform,full_matrices=False);assert bool(jnp.all(s>s[0]*1e-10));W,e,_=C.diffusion_basis(U,256);G,H=C.flux_tables(W,256,384)
 np.savez_compressed(o/f'tile{i}{j}_resolved.npz',basis=W,eigenvalues=e,singular_values=s,physical_transform=transform)
 for dt in (.0125,.00625):
  solver=C.flux_trajectory(dt)
  for case in cases:
   d=np.load(verified('c1data',f'check{128+case}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(np.load(verified('c1data',f'case{128+case}.npz'))['initial_norm']);states,sec=timer(lambda:solver(W.T@truth[0].ravel()/2,dev[case,4],G,H,e));pred=(2*(states@W.T)).reshape(11,255,255)
   old=np.load(verified('c1compress',f'r384_q192_dt{dt}_case{case}.npz'))['states'];oldpred=jnp.pad((2*(old@oldW.T)).reshape(11,127,127),((0,0),(0,128),(0,128)))
   err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm);delta=np.asarray(jnp.linalg.norm((pred-oldpred).reshape(11,-1),axis=1))/(2*norm)
   np.savez_compressed(o/f'resolved_dt{dt}_case{case}.npz',coefficients=pred,error=err)
   resolution.append(dict(case=case,dt=dt,error=err,feature_change=delta,seconds=sec))
 save(o/'resolution.json',dict(rows=resolution));print('resolved',i,j,flush=True)
xy=jnp.asarray(E.coords(512,True));x=jnp.arange(4096,dtype=jnp.float64)*256/4095;coords=jnp.stack(jnp.meshgrid(x,x,indexing='ij')).reshape(2,-1);f=C.flux_trajectory(.0125)
@jax.jit
def extend(c):return map_coordinates(PJ.synth_mesh(c,256),coords,order=1,mode='nearest').reshape(4096,4096)
@jax.jit
def solve(u,nu,W,e,G,H):
 b=P.analysis(map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511),512)[:127,:127];states=f(W.T@b.ravel()/2,nu,G,H,e);coeff=(2*(states@W.T)).reshape(11,127,127);return coeff,extend(coeff[-1])
peak=jax.jit(lambda u:jnp.argmax(u))
def rom(u,nu):
 index=int(peak(u));tile=tuple(min(2,int(v*4/4096)) for v in divmod(index,4096));return solve(u,nu,*models[tile])
queries={'compressed384':rom,**{f'spectral{n}':make(n,.0125,512,'bilinear256') for n in (24,32,48)}};rows=[];rng=np.random.default_rng(2026111813)
for i,ph in enumerate(dev[:32]):
 u=jnp.asarray(E.initial(4095,ph));jax.block_until_ready(u);norm=float(jnp.linalg.norm(u));times={name:[[],[]] for name in queries};cold={};outputs={}
 for name,query in queries.items():outputs[name],cold[name]=timer(lambda:query(u,ph[4]))
 for block in range(2):
  for rep in range(5):
   for name in rng.permutation(list(queries)):
    E.burn(.4);outputs[name],sec=timer(lambda:queries[name](u,ph[4]));times[name][block].append(sec)
    np.savez_compressed(o/f'{name}_case{i}_block{block}_rep{rep}.npz',coefficients=outputs[name][0],final_norm=float(jnp.linalg.norm(outputs[name][1])))
 refjob='reference01' if i<8 else 'c1ref32';fd=np.load(verified(refjob,f'dev{i}_fd512_dt0.000625.npz'))['out'].reshape(6,511,511);bc=jax.vmap(lambda u:P.analysis(u,512))(jnp.asarray(fd[1:]));errors={name:[] for name in queries};extension={name:[] for name in queries}
 for t in range(5):
  ref=PJ.synth_mesh(bc[t],4095)
  for name,(coeff,field) in outputs.items():
   c=coeff[2*(t+1)] if name=='compressed384' else coeff[t+1];dense=extend(c);exact=PJ.synth_mesh(c,4095);errors[name].append(float(jnp.linalg.norm(dense-ref))/norm);extension[name].append(float(jnp.linalg.norm(dense-exact))/norm)
 if i==9:np.savez_compressed(o/'final4096_case9.npz',reference=ref,**{name:pair[1] for name,pair in outputs.items()})
 rows.append(dict(case=i,times=times,cold_query_seconds=cold,errors=errors,extension_errors=extension,output_shape=[4096,4096]));save(o/'result.json',dict(meta=meta(),rows=rows,resolution=resolution,setup_seconds=setup_seconds));print('cost',i,{k:float(np.median(v)) for k,v in times.items()},flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,resolution=resolution,setup_seconds=setup_seconds,complete=True,contract='common resident4096 input and actual4096 output; selection restriction solve decode timed, complete nine-model setup separate; inherited verified same-case dense reference qualification; development only'))
