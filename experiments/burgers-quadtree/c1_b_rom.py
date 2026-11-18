"""Family B initial-gradient patch allocation and continuous weak PDE screen."""
from campaign import *
from owned_inputs import verified
import continuous as C,specialized as S,local_bank as B,family_b as FB,pickle,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open(verified('c1bbank','network.pkl'),'rb')));V=jnp.asarray(np.load(verified('c1bbank','pod.npz'))['basis'])[:,:128];params=json.loads(verified('c1familyb','parameters.json').read_text())['dev'];points=jnp.asarray(E.coords(128,True));blocks=[]
for patch in B.uniform(2):
 F=jnp.concatenate([B.features(points,[patch],32,'poly'),S.features(p,points,[patch],32)],axis=1)
 blocks.append(jax.vmap(lambda f:P.analysis(f.reshape(127,127),128).ravel()/2,in_axes=1,out_axes=1)(F))
rows=[]
for i,ph in enumerate(params[:8]):
 initial=FB.initial(4095,ph);small=initial[::16,::16];gx,gy=np.gradient(small);energy=(gx*gx+gy*gy).reshape(4,64,4,64).sum((1,3))
 # Anticipate only prescribed positive transport, never future fields.
 score=energy.copy();score[1:]+=energy[:-1]/2;score[:,1:]+=energy[:,:-1]/2;order=np.argsort(-score.ravel(),kind='stable')
 d=np.load(verified('c1familyb',f'case{32+i}.npz'));truth=np.asarray(d['coefficients']);norm=float(d['initial_norm']);low=jnp.asarray(truth[:,:127,:127]);tail=truth.copy();tail[:,:127,:127]=0;tailnorm=np.linalg.norm(tail.reshape(11,-1),axis=1)/(2*norm)
 for active in (8,12,16):
  ids=order[:active];A=jnp.concatenate([V]+[blocks[k] for k in ids],axis=1);Q,s,vh=jnp.linalg.svd(A,full_matrices=False);keep=s>s[0]*1e-10;Q=Q[:,keep];W,eigs,rot=C.diffusion_basis(Q,128)
  np.savez_compressed(o/f'case{i}_active{active}_basis.npz',basis=W,eigenvalues=eigs,patch_ids=ids,physical_to_orthogonal=(vh[keep].T/s[keep])@rot)
  Y=low.reshape(11,-1)/(2*norm);proj=np.sqrt(np.asarray(jnp.linalg.norm(Y-(Y@W)@W.T,axis=1))**2+tailnorm**2)
  previous=None
  for q,dt in [(192,.005),(256,.005),(256,.0025)]:
   G,H=C.flux_tables(W,128,q);f=C.flux_trajectory(dt);a0=W.T@low[0].ravel()/2
   states,sec=timer(lambda:f(a0,ph['nu'],G,H,eigs));pred=(2*(states@W.T)).reshape(11,127,127);err=np.sqrt((np.asarray(jnp.linalg.norm((pred-low).reshape(11,-1),axis=1))/(2*norm))**2+tailnorm**2)
   change=None if previous is None else np.asarray(jnp.linalg.norm(states-previous,axis=1))/norm;previous=states
   np.savez_compressed(o/f'case{i}_active{active}_q{q}_dt{dt}.npz',states=states,error=err)
   rows.append(dict(case=i,active=active,patch_ids=ids,q=q,dt=dt,projection=proj,error=err,integration_change=change,seconds=sec,evolved=float(err[[2,4,6,8,10]].max())))
  save(o/'result.json',dict(meta=meta(),rows=rows));print(i,active,rows[-1]['evolved'],flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='familyB initial-gradient-only allocation, fixed-depth weak PDE diagnostic; prepared coefficients, no speed claim'))
