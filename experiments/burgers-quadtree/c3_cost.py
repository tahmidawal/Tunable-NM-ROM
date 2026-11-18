"""Frozen global-space finalists: finer features and complete common-array query costs."""
from campaign import *
from owned_inputs import verified
from spectral_cost import PJ,make
import continuous as C,frame_features as FF,pickle,argparse
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);config=json.loads((H/'configs/c3-finalists.json').read_text());assert config['registered_before_submission'] and config['finalists'];_,dev=draw();models={};resolution=[];setup=time.perf_counter()
for row in config['finalists']:
 name=row['name'];rank=row['rank'];q=row['q'];d=np.load(verified('c1globalneural',f'A/{name}_physics_basis.npz'));W=jnp.asarray(d['basis'])[:,:rank];e=jnp.asarray(d['eigenvalues'])[:rank];G,Hq=C.flux_tables(W,128,q);jax.block_until_ready((W,e,G,Hq));models[name]=(W,e,G,Hq);np.savez_compressed(o/f'{name}_query_basis.npz',basis=W,eigenvalues=e)
setup_seconds=time.perf_counter()-setup;cache_bytes=sum(int(z.size*z.dtype.itemsize) for model in models.values() for z in model)
for row in config['finalists']:
 name=row['name'];oldW,olde,_,_=models[name];rank=row['rank'];controlG,controlH=C.flux_tables(oldW,128,256)
 if name=='neural':
  d=np.load(verified('c1globalneural','A/neural_basis.npz'));transform=jnp.asarray(d['physical_transform'])@(jnp.asarray(d['basis']).T@oldW);p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1globalneural','A/network.pkl'),'rb')));xy=jnp.asarray(E.coords(256,True));F=FF.apply(p,xy)*jnp.prod(jnp.sin(jnp.pi*xy),axis=1)[:,None];A=jax.vmap(lambda f:P.analysis(f.reshape(255,255),256).ravel()/2,in_axes=1,out_axes=1)(F)@transform
 else:A=jnp.pad(oldW.reshape(127,127,rank),((0,128),(0,128),(0,0))).reshape(255**2,rank)
 U,s,_=jnp.linalg.svd(A,full_matrices=False);assert bool(jnp.all(s>s[0]*1e-10));W,e,_=C.diffusion_basis(U,256);G,Hq=C.flux_tables(W,256,256);np.savez_compressed(o/f'{name}_fine_basis.npz',basis=W,eigenvalues=e,singular_values=s)
 for dt in (.0125,.00625):
  solver=C.flux_trajectory(dt)
  for case,ph in enumerate(dev[:32]):
   truth=jnp.asarray(np.load(verified('c1data',f'check{128+case}.npz'))['coefficients']);norm=float(np.load(verified('c1data',f'case{128+case}.npz'))['initial_norm']);states,sec=timer(lambda:solver(W.T@truth[0].ravel()/2,ph[4],G,Hq,e));pred=(2*(states@W.T)).reshape(11,255,255);old=np.load(verified(row['source_job'],f'{name}_r{rank}_q192_dt{dt}_case{case}.npz'))['states'];initial128=jnp.asarray(np.load(verified('c1data',f'case{128+case}.npz'))['coefficients'][0]);control=solver(oldW.T@initial128.ravel()/2,ph[4],controlG,controlH,olde);quad=np.asarray(jnp.linalg.norm(control-old,axis=1))/norm;oldpred=jnp.pad((2*(control@oldW.T)).reshape(11,127,127),((0,0),(0,128),(0,128)));err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm);delta=np.asarray(jnp.linalg.norm((pred-oldpred).reshape(11,-1),axis=1))/(2*norm);np.savez_compressed(o/f'{name}_resolved_dt{dt}_case{case}.npz',coefficients=pred,control_states=control,error=err);resolution.append(dict(name=name,case=case,dt=dt,error=err,feature_and_initial_resolution_change=delta,quadrature192_to256=quad,seconds=sec))
  save(o/'resolution.json',dict(rows=resolution));print('resolved',name,dt,flush=True)
xy=jnp.asarray(E.coords(512,True));x=jnp.arange(4096,dtype=jnp.float64)*256/4095;coords=jnp.stack(jnp.meshgrid(x,x,indexing='ij')).reshape(2,-1);solver=C.flux_trajectory(.0125)
@jax.jit
def extend(c):return map_coordinates(PJ.synth_mesh(c,256),coords,order=1,mode='nearest').reshape(4096,4096)
@jax.jit
def solve(u,nu,W,e,G,Hq):
 b=P.analysis(map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511),512)[:127,:127];states=solver(W.T@b.ravel()/2,nu,G,Hq,e);return states,extend((2*(W@states[-1])).reshape(127,127))
def query(model):return lambda u,nu:solve(u,nu,*model)
queries={name:query(model) for name,model in models.items()};queries.update({f'spectral{n}':make(n,.0125,512,'bilinear256') for n in (24,32,48)});rows=[];rng=np.random.default_rng(2026111820)
for case,ph in enumerate(dev[:32]):
 u=jnp.asarray(E.initial(4095,ph));jax.block_until_ready(u);norm=float(jnp.linalg.norm(u));times={name:[[],[]] for name in queries};cold={};outputs={}
 for name,f in queries.items():outputs[name],cold[name]=timer(lambda:f(u,ph[4]))
 for block in range(2):
  for rep in range(5):
   for name in rng.permutation(list(queries)):
    E.burn(.2);outputs[name],sec=timer(lambda:queries[name](u,ph[4]));times[name][block].append(sec);np.savez_compressed(o/f'{name}_case{case}_block{block}_rep{rep}.npz',states=outputs[name][0],final_norm=float(jnp.linalg.norm(outputs[name][1])))
 refjob='reference01' if case<8 else 'c1ref32';fd=np.load(verified(refjob,f'dev{case}_fd512_dt0.000625.npz'))['out'].reshape(6,511,511);bc=jax.vmap(lambda u:P.analysis(u,512))(jnp.asarray(fd[1:]));coeff={name:(2*(states@models[name][0].T)).reshape(11,127,127) if name in models else states for name,(states,field) in outputs.items()};errors={name:[] for name in queries};extensions={name:[] for name in queries}
 for t in range(5):
  ref=PJ.synth_mesh(bc[t],4095)
  for name in queries:
   c=coeff[name][2*(t+1)] if name in models else coeff[name][t+1];field=extend(c);errors[name].append(float(jnp.linalg.norm(field-ref))/norm);extensions[name].append(float(jnp.linalg.norm(field-PJ.synth_mesh(c,4095)))/norm)
 if case==9:np.savez_compressed(o/'final4096_case9.npz',reference=ref,**{name:pair[1] for name,pair in outputs.items()})
 rows.append(dict(case=case,times=times,cold_query_seconds=cold,errors=errors,extension_errors=extensions,output_shape=[4096,4096]));save(o/'result.json',dict(meta=meta(),config=config,rows=rows,resolution=resolution,setup_seconds=setup_seconds,cache_array_bytes=cache_bytes));print('cost',case,flush=True)
save(o/'result.json',dict(meta=meta(),config=config,rows=rows,resolution=resolution,setup_seconds=setup_seconds,cache_array_bytes=cache_bytes,complete=True,contract='same resident4096 initial field and final output; reduced states retained for offline scoring, only final physical decode charged to final-output query; no spatial-refinement or unseen-confirmation claim'))
