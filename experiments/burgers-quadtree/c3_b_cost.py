"""Frozen cheaper B operators, qualified fine trajectories, complete output costs."""
from campaign import *
from owned_inputs import verified
from spectral_cost import PJ,make
import continuous as C,family_b as FB,quadtree as QT,argparse,itertools
from jax.scipy.ndimage import map_coordinates
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);config=json.loads((H/'configs/b-efficiency-finalists.json').read_text());assert config['registered_before_submission'];recipes={x['name']:x for x in config['finalists']};params=json.loads(verified('c1familyb','parameters.json').read_text())['dev'][:8];models={};setup={name:0. for name in recipes};cache={name:0 for name in recipes}
for split in list(itertools.combinations(range(4),2))+[(0,1,2,3)]:
 name='uniform1152' if len(split)==4 else 'tree768'
 if name not in recipes:continue
 start=time.perf_counter();d=np.load(verified('c1bcomplete','split'+''.join(map(str,split))+'_basis.npz'));W=jnp.asarray(d['basis']);e=jnp.asarray(d['eigenvalues']);G,Hq=C.flux_tables(W,128,recipes[name]['q']);jax.block_until_ready((W,e,G,Hq));models[split]=(W,e,G,Hq);setup[name]+=time.perf_counter()-start;cache[name]+=sum(int(z.size*z.dtype.itemsize) for z in models[split])
xy=jnp.asarray(E.coords(512,True));x=jnp.arange(4096,dtype=jnp.float64)*256/4095;coords=jnp.stack(jnp.meshgrid(x,x,indexing='ij')).reshape(2,-1)
@jax.jit
def extend(c):return map_coordinates(PJ.synth_mesh(c,256),coords,order=1,mode='nearest').reshape(4096,4096)
def solver(dt):
 f=C.flux_trajectory(dt)
 @jax.jit
 def run(u,nu,W,e,G,Hq):
  b=P.analysis(map_coordinates(u,(xy*4095).T,order=1,mode='nearest').reshape(511,511),512)[:127,:127];states=f(W.T@b.ravel()/2,nu,G,Hq,e);return states,extend((2*(W@states[-1])).reshape(127,127))
 return run
solvers={name:solver(x['dt']) for name,x in recipes.items()}
def local_query(name):
 def run(u,nu):
  split=(0,1,2,3) if name=='uniform1152' else tuple(sorted(QT.initial_order(u)[:2]));return solvers[name](u,nu,*models[split])
 return run
queries={name:local_query(name) for name in recipes};queries.update({f'spectral{x["N"]}_dt{x["dt"]}':make(x['N'],x['dt'],512,'bilinear256') for x in config['classical']});rows=[];rng=np.random.default_rng(2026111821)
for case,ph in enumerate(params):
 u=jnp.asarray(FB.initial(4095,ph));jax.block_until_ready(u);norm=float(jnp.linalg.norm(u));normphys=float(np.load(verified('c1familyb',f'case{32+case}.npz'))['initial_norm']);order=tuple(sorted(QT.initial_order(u)[:2]));times={name:[[],[]] for name in queries};cold={};outputs={}
 for name,f in queries.items():outputs[name],cold[name]=timer(lambda:f(u,ph['nu']))
 for block in range(2):
  for rep in range(5):
   for name in rng.permutation(list(queries)):
    E.burn(.2);outputs[name],sec=timer(lambda:queries[name](u,ph['nu']));times[name][block].append(sec);np.savez_compressed(o/f'{name}_case{case}_block{block}_rep{rep}.npz',states=outputs[name][0],final_norm=float(jnp.linalg.norm(outputs[name][1])))
 coeff={};checks={}
 for name,(states,field) in outputs.items():
  if name in recipes:
   split=(0,1,2,3) if name=='uniform1152' else order;W=models[split][0];coeff[name]=(2*(states@W.T)).reshape(11,127,127);fine=jnp.asarray(np.load(verified('c1bcomplete',f'resolved_split{"".join(map(str,split))}_dt0.00625_case{case}.npz'))['coefficients']);delta=np.asarray(jnp.linalg.norm((jnp.pad(coeff[name],((0,0),(0,128),(0,128)))-fine).reshape(11,-1),axis=1))/(2*normphys);checks[name]=dict(combined_fine_space_time_change=delta,control='existing N256/Gauss384/dt0.00625 trajectory; independent finer0.003125 and same-input quadrature/halfstep checks retained')
  else:coeff[name]=states
 fd=np.load(verified('c1familyb',f'dev{case}_fd512_dt0.0003125.npz'))['out'].reshape(6,511,511);bc=jax.vmap(lambda u:P.analysis(u,512))(jnp.asarray(fd[1:]));errors={name:[] for name in queries};extension={name:[] for name in queries}
 for t in range(5):
  ref=PJ.synth_mesh(bc[t],4095)
  for name in queries:
   c=coeff[name][2*(t+1)] if name in recipes else coeff[name][t+1];field=extend(c);errors[name].append(float(jnp.linalg.norm(field-ref))/norm);extension[name].append(float(jnp.linalg.norm(field-PJ.synth_mesh(c,4095)))/norm)
 if case==0:np.savez_compressed(o/'final4096_case0.npz',reference=ref,**{name:pair[1] for name,pair in outputs.items()})
 rows.append(dict(case=case,times=times,cold_query_seconds=cold,errors=errors,extension_errors=extension,refinement_checks=checks,output_shape=[4096,4096]));save(o/'result.json',dict(meta=meta(),config=config,rows=rows,setup_seconds_by_policy=setup,cache_array_bytes_by_policy=cache));print(case,flush=True)
save(o/'result.json',dict(meta=meta(),config=config,rows=rows,setup_seconds_by_policy=setup,cache_array_bytes_by_policy=cache,complete=True,contract='all-case familyB common resident4096 input to actual final4096 output; all finite tree caches prepared; reduced-state retention, final-only decode; unchanged qualified finer physical trajectories and independent operator controls'))
