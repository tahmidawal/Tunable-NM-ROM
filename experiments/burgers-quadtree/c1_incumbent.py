"""Frozen historical decoder with separately labelled continuous weak dynamics."""
from campaign import *
from owned_inputs import verified
import continuous as C,manifold_core as M,qcore as Q,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
mdl=Q.Model();xy=jnp.asarray(E.coords(128,True));F=jax.jit(lambda p,x:Q.sc.features(p,x))(mdl.params,xy)@jnp.asarray(mdl.T)
A=jax.vmap(lambda f:P.analysis(f.reshape(127,127),128).ravel()/2,in_axes=1,out_axes=1)(F);_,dev=draw();truths=[];norms=[]
for i in range(32):
 d=np.load(verified('c1data',f'case{128+i}.npz'));truths.append(jnp.asarray(d['coefficients']).reshape(11,-1)/2);norms.append(float(d['initial_norm']))
rows=[];solvers={h:C.flux_trajectory(h) for h in (.005,.0025)}
for rank in (128,384,512):
 U,s,vh=jnp.linalg.svd(A[:,:rank],full_matrices=False);keep=s>s[0]*1e-10;U=U[:,keep];W,eigs,rotation=C.diffusion_basis(U,128)
 np.savez_compressed(o/f'bank{rank}.npz',basis=W,eigenvalues=eigs,physical_to_orthogonal=(vh[keep].T/s[keep])@rotation,singular_values=s)
 for q,dt in [(128,.005),(192,.005),(192,.0025)]:
  G,H=C.flux_tables(W,128,q)
  for i in range(32):
   a0=W.T@truths[i][0];states,sec=timer(lambda:solvers[dt](a0,dev[i,4],G,H,eigs));err=np.asarray(jnp.linalg.norm(states@W.T-truths[i],axis=1))/norms[i];proj=np.asarray(jnp.linalg.norm(truths[i]-(truths[i]@W)@W.T,axis=1))/norms[i]
   np.savez_compressed(o/f'bank{rank}_q{q}_dt{dt}_case{i}.npz',states=states,error=err)
   rows.append(dict(rank=rank,q=q,dt=dt,case=i,error=err,projection=proj,seconds=sec,evolved=float(err[[2,4,6,8,10]].max())))
  save(o/'result.json',dict(meta=meta(),bank=rows));print('bank',rank,q,dt,max(x['evolved'] for x in rows if x['rank']==rank and x['q']==q and x['dt']==dt),flush=True)
# Full frozen head in the same continuous bank coordinates; physical, not normalized coefficients.
U,s,vh=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;U=U[:,keep];Tmap=U.T@A;hp=mdl.head_params(512)
def decode(z):return Tmap@Q.sc.head(hp,z)
Z=jnp.asarray(mdl.Zold[::max(1,len(mdl.Zold)//8192)]);train_codes=jax.jit(jax.vmap(decode))(Z);projections=[];codes=[]
@jax.jit
def fit(c,z):return M.lm(lambda x:decode(x)-c,z,100)
for i in range(32):
 target=truths[i]@U;bank_error=jnp.linalg.norm(truths[i]-target@U.T,axis=1);nearest=np.asarray(jnp.argsort(jnp.sum((target[:,None,:]-train_codes[None,:,:])**2,axis=2),axis=1)[:,:2]);zs=[];errs=[];grads=[]
 for t in range(11):
  trials=[fit(target[t],z0) for z0 in (Z[nearest[t,0]],Z[nearest[t,1]],jnp.zeros(mdl.K))];pick=min(range(3),key=lambda k:float(trials[k][2]));z,g,e=trials[pick];zs.append(z);errs.append(float(jnp.sqrt(e**2+bank_error[t]**2))/norms[i]);grads.append([float(v[1]) for v in trials])
 codes.append(jnp.stack(zs));projections.append(dict(case=i,error=errs,stationarity=grads));print('head projection',i,max(errs[k] for k in [2,4,6,8,10]),flush=True)
np.savez_compressed(o/'head_projection.npz',codes=jnp.stack(codes));K=U.T@(C.eigenvalues(128)[:,None]*U);headrows=[]
for q,dt in [(128,.005),(192,.005),(192,.0025)]:
 G,H=C.flux_tables(U,128,q)
 @jax.jit
 def step(z,nu,h,theta,G,H,K):
  old=decode(z)
  def residual(a):
   new=decode(a);mid=theta*new+(1-theta)*old;return new-old+h*(nu*(K@mid)-H.T@(G@mid)**2)
  return M.lm(residual,z,60)
 for i in (0,9):
  z=codes[i][0];states=[decode(z)];stats=[];t=0.;start=time.perf_counter()
  for h,theta in [(dt/2,1.),(dt/2,1.)]+[(dt,.5)]*(round(.25/dt)-1):
   z,g,e=step(z,dev[i,4],h,theta,G,H,K);t+=h;stats.append(dict(time=t,stationarity=float(g),residual=float(e)))
   if abs(t/.025-round(t/.025))<1e-8:states.append(decode(z))
  sec=time.perf_counter()-start;states=jnp.stack(states);err=np.asarray(jnp.linalg.norm(states@U.T-truths[i],axis=1))/norms[i]
  np.savez_compressed(o/f'head_q{q}_dt{dt}_case{i}.npz',states=states,error=err)
  headrows.append(dict(case=i,q=q,dt=dt,error=err,diagnostics=stats,seconds=sec));save(o/'result.json',dict(meta=meta(),bank=rows,head_projection=projections,head_pde=headrows));print('head PDE',i,q,dt,float(err[[2,4,6,8,10]].max()),flush=True)
save(o/'result.json',dict(meta=meta(),bank=rows,head_projection=projections,head_pde=headrows,complete=True,contract='same frozen historical decoder, different explicitly qualified continuous weak operator; not matched-data retraining or historical solver reproduction; prepared-input diagnostics'))
