"""Matched-data coordinate-neural banks and exact POD span/PDE controls."""
from campaign import *
from owned_inputs import verified
import frame_features as FF,continuous as C,optax,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--family',choices=['A','B'],required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True);family=a.family
source=np.load(verified('c1manifold','lift.npz') if family=='A' else verified('c1bbank','pod.npz'));k=1024 if family=='A' else 256;width=1024 if family=='A' else 512;V=jnp.asarray(source['basis'][:,:k]);singular=jnp.asarray(source['singular_values'][:k]);points=jnp.asarray(E.coords(128,True));ls=P.lattice_S(128,128);teacher=jax.vmap(lambda v:P.synth_lattice(2*v.reshape(127,127),ls,2),in_axes=1)(V)[:,1:-1,1:-1].reshape(k,-1).T
key=jax.random.key(2026111816 if family=='A' else 2026111817);dims=[22,width,width,k];p=[(jax.random.normal(q,(m,n),dtype=jnp.float64)*np.sqrt(2/(m+n)),jnp.zeros(n)) for q,m,n in zip(jax.random.split(key,3),dims[:-1],dims[1:])]
def features(p,x):return FF.apply(p,x)*jnp.prod(jnp.sin(jnp.pi*x),axis=1)[:,None]
weight=(singular/singular[0])**2;opt=optax.adam(optax.cosine_decay_schedule(.001,30000,alpha=.01));state=opt.init(p);rng=np.random.default_rng(2026111816 if family=='A' else 2026111817);history=[];start=time.perf_counter()
@jax.jit
def step(p,state,x,y):
 loss,g=jax.value_and_grad(lambda p:jnp.mean(jnp.sum((features(p,x)-y)**2*weight,axis=1)))(p);updates,state=opt.update(g,state,p);return optax.apply_updates(p,updates),state,loss
for it in range(30000):
 idx=rng.integers(0,len(points),256);p,state,loss=step(p,state,points[idx],teacher[idx])
 if (it+1)%5000==0:
  history.append(dict(step=it+1,loss=float(loss)));print(family,'train',it+1,float(loss),flush=True)
  with open(o/f'step{it+1}.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
training_seconds=time.perf_counter()-start
with open(o/'network.pkl','wb') as f:pickle.dump(jax.tree_util.tree_map(np.asarray,p),f)
F=features(p,points);A=jax.vmap(lambda f:P.analysis(f.reshape(127,127),128).ravel()/2,in_axes=1,out_axes=1)(F);Q,s,vh=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;Q=Q[:,keep];transform=vh[keep].T/s[keep];points256=jnp.asarray(E.coords(256,True));F256=features(p,points256);A256=jax.vmap(lambda f:P.analysis(f.reshape(255,255),256).ravel()/2,in_axes=1,out_axes=1)(F256);Q256=A256@transform
np.savez_compressed(o/'neural_basis.npz',basis=Q,physical_transform=transform,singular_values=s)
count=32 if family=='A' else 8;params=draw()[1][:count] if family=='A' else json.loads(verified('c1familyb','parameters.json').read_text())['dev'][:count];rows=[];truths=[];norms=[]
for case in range(count):
 d=np.load(verified('c1data' if family=='A' else 'c1familyb',f'{"check" if family=="A" else "case"}{case+(128 if family=="A" else 32)}.npz'));truth=jnp.asarray(d['coefficients']).reshape(11,255,255)/2;norm=float(np.load(verified('c1data' if family=='A' else 'c1familyb',f'case{case+(128 if family=="A" else 32)}.npz'))['initial_norm']);low=truth[:,:127,:127].reshape(11,-1);truths.append(truth);norms.append(norm)
 for name,basis,fine in [('neural',Q,Q256),('POD',V,jnp.pad(V.reshape(127,127,k),((0,128),(0,128),(0,0))).reshape(255**2,k))]:
  coeff=low@basis;pred=coeff@fine.T;error=np.asarray(jnp.linalg.norm(pred-truth.reshape(11,-1),axis=1))/norm;rows.append(dict(case=case,name=name,columns=basis.shape[1],error=error,evolved=float(error[[2,4,6,8,10]].max())));np.savez_compressed(o/f'{name}_projection_case{case}.npz',coefficients=coeff,error=error)
solve_rows=[]
for name,basis in [('neural',Q),('POD',V)]:
 passing=max(x['evolved'] for x in rows if x['name']==name)<=.01;cases=list(range(count)) if passing else [0,9 if family=='A' else 2];W,e,_=C.diffusion_basis(basis,128);np.savez_compressed(o/f'{name}_physics_basis.npz',basis=W,eigenvalues=e)
 for q,dt in [(192,.005),(256,.005),(256,.0025)]:
  G,H=C.flux_tables(W,128,q);solver=C.flux_trajectory(dt)
  for case in cases:
   truth=truths[case];nu=params[case][4] if family=='A' else params[case]['nu'];states,sec=timer(lambda:solver(W.T@truth[0,:127,:127].ravel(),nu,G,H,e));pred=jnp.pad((states@W.T).reshape(11,127,127),((0,0),(0,128),(0,128)));err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/norms[case];solve_rows.append(dict(case=case,name=name,q=q,dt=dt,error=err,evolved=float(err[[2,4,6,8,10]].max()),seconds=sec,all_case_projection_pass=passing));np.savez_compressed(o/f'{name}_q{q}_dt{dt}_case{case}.npz',states=states,error=err)
  save(o/'result.json',dict(meta=meta(),family=family,training=history,training_seconds=training_seconds,projection=rows,pde=solve_rows));print(family,name,q,dt,flush=True)
save(o/'result.json',dict(meta=meta(),family=family,training=history,training_seconds=training_seconds,projection=rows,pde=solve_rows,complete=True,contract='matched training trajectories, TRAIN-only POD teacher, new globally supported neural bank; unchanged coefficients rescored N256; actual weak PDE all cases if projection passes else specified diagnostics; no complete-query timing claim'))
