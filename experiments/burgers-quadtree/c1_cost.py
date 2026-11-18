"""Paired common-array actual4096 development timings; all online stages charged."""
from campaign import *
from owned_inputs import verified
from spectral_cost import PJ,make
import continuous as C,argparse
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();setup=time.perf_counter();models={};q=128
for i in range(3):
 for j in range(3):
  # Empty development tiles are still prepared: selector must be defined for every input.
  V=jnp.asarray(np.load(verified('c1tiles',f'r256_tile{i}{j}.npz'))['basis']);W,eigs,_=C.diffusion_basis(V,128);G,H=C.flux_tables(W,128,q);jax.block_until_ready((W,eigs,G,H));models[i,j]=(W,eigs,G,H)
setup_seconds=time.perf_counter()-setup;xy=jnp.asarray(E.coords(512,True));f=C.flux_trajectory(.005)
x=jnp.arange(4096,dtype=jnp.float64)*256/4095;coords=jnp.stack(jnp.meshgrid(x,x,indexing='ij')).reshape(2,-1)
@jax.jit
def extend(c):return map_coordinates(PJ.synth_mesh(c,256),coords,order=1,mode='nearest').reshape(4096,4096)
@jax.jit
def solve(u,nu,W,eigs,G,H):
 initial=map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511);b=P.analysis(initial,512)[:127,:127]
 a0=W.T@b.ravel()/2;states=f(a0,nu,G,H,eigs);coeff=(2*(states@W.T)).reshape(11,127,127)
 return coeff,extend(coeff[-1])
peak=jax.jit(lambda u:jnp.argmax(u))
def rom(u,nu):
 index=int(peak(u));tile=tuple(min(2,int(v*4/4096)) for v in divmod(index,4096))
 return solve(u,nu,*models[tile])
queries={'local_hybrid':rom,'spectral16_dt005':make(16,.005,512,'bilinear256'),'spectral32_dt01':make(32,.01,512,'bilinear256'),'spectral32_dt005':make(32,.005,512,'bilinear256'),'spectral64_dt005':make(64,.005,512,'bilinear256'),'spectral64_dt00125':make(64,.00125,512,'bilinear256')}
rows=[];rng=np.random.default_rng(2026111805)
for i,ph in enumerate(dev[:32]):
 u=jnp.asarray(E.initial(4095,ph));jax.block_until_ready(u);norm=float(jnp.linalg.norm(u));cold={};times={name:[[],[]] for name in queries};outputs={}
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
   c=coeff[2*(t+1)] if name=='local_hybrid' else coeff[t+1];dense=extend(c);exact=PJ.synth_mesh(c,4095)
   errors[name].append(float(jnp.linalg.norm(dense-ref))/norm);extension[name].append(float(jnp.linalg.norm(dense-exact))/norm)
 np.savez_compressed(o/f'final4096_case{i}.npz',reference=ref,local_hybrid=outputs['local_hybrid'][1],spectral=outputs['spectral64_dt005'][1])
 rows.append(dict(case=i,cold_query_seconds=cold,times=times,errors=errors,extension_errors=extension,output_shape=[4096,4096]));save(o/'result.json',dict(meta=meta(),setup_seconds=setup_seconds,rows=rows));print(i,{k:max(v) for k,v in errors.items()},{k:float(np.median(v)) for k,v in times.items()},flush=True)
save(o/'result.json',dict(meta=meta(),setup_seconds=setup_seconds,rows=rows,complete=True,contract='resident common4096 input to resident4096 final output; cache setup separate, initialization selection restriction PDE and reconstruction all timed; development only'))
