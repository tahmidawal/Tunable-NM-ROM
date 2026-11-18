from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
params=np.load(verified('c1data','parameters.npz'))['dev'][:32];V=jnp.asarray(np.load(verified('c1data','pod.npz'))['basis']);rows=[];setups=[]
for rank in (128,256,384,512):
 t=time.perf_counter();W,eigs,rotation=C.diffusion_basis(V[:,:rank],128);jax.block_until_ready(W)
 np.savez_compressed(o/f'basis{rank}.npz',basis=W,eigenvalues=eigs,rotation=rotation)
 setups.append(dict(rank=rank,diffusion_setup_seconds=time.perf_counter()-t))
 for q,dt in [(48,.005),(64,.005),(96,.005),(96,.0025)]:
  t=time.perf_counter();G,H=C.flux_tables(W,128,q);jax.block_until_ready(G);setup=time.perf_counter()-t;solve=C.flux_trajectory(dt)
  for i,ph in enumerate(params):
   data=np.load(verified('c1data',f'case{128+i}.npz'));truth=jnp.asarray(data['coefficients']);norm=float(data['initial_norm']);a0=W.T@truth[0].ravel()/2
   states,sec=timer(lambda:solve(a0,ph[4],G,H,eigs));b=(2*(states@W.T)).reshape(11,127,127)
   error=np.asarray(jnp.linalg.norm((b-truth).reshape(11,-1),axis=1))/(2*norm)
   np.savez_compressed(o/f'r{rank}_q{q}_dt{dt}_case{i}.npz',states=states,error=error)
   rows.append(dict(rank=rank,q=q,dt=dt,case=i,errors=error,evolved=float(error[[2,4,6,8,10]].max()),seconds=sec,setup_seconds=setup,finite=bool(np.isfinite(states).all())))
  save(o/'result.json',dict(meta=meta(),rows=rows,setups=setups));print(rank,q,dt,max(x['evolved'] for x in rows if x['rank']==rank and x['q']==q and x['dt']==dt),flush=True)
# Full-evaluation small panel isolates quadrature without changing the model/operator.
rank=256;W,eigs,_=C.diffusion_basis(V[:,:rank],128);full=C.reduced_trajectory(128,.0025)
for i,ph in enumerate(params[:4]):
 data=np.load(verified('c1data',f'case{128+i}.npz'));truth=jnp.asarray(data['coefficients']);a0=W.T@truth[0].ravel()/2
 states,sec=timer(lambda:full(a0,ph[4],W,eigs));np.savez_compressed(o/f'full_r256_case{i}.npz',states=states)
rowsmeta=dict(meta=meta(),rows=rows,setups=setups,complete=True,contract='coefficient-input operator screening; not complete query timing')
save(o/'result.json',rowsmeta)
