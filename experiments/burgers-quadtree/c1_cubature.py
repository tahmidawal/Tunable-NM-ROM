"""TRAIN-only positive empirical flux quadrature; exact projected linear terms."""
from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();tiles=json.loads(verified('c1tiles','result.json').read_text())['rows'];Ys=[]
for i in range(128):
 d=np.load(verified('c1data',f'case{i}.npz'));Ys.append(np.asarray(d['coefficients']).reshape(11,-1).T/(2*float(d['initial_norm'])))
Y=jnp.asarray(np.concatenate(Ys,axis=1));rng=np.random.default_rng(2026111807);rows=[];fits=[];f=C.flux_trajectory(.005);fh=C.flux_trajectory(.0025)
for i in range(3):
 for j in range(3):
  t=time.perf_counter();V=jnp.asarray(np.load(verified('c1tiles',f'r256_tile{i}{j}.npz'))['basis']);W,eigs,_=C.diffusion_basis(V,128);G,H=C.flux_tables(W,128,128)
  train=W.T@Y;values=(G@train)**2
  # Structured randomized sketch of the full test-by-training-state flux tensor.
  rv=jnp.asarray(rng.choice([-1.,1.],(W.shape[1],32))/np.sqrt(32));rt=jnp.asarray(rng.choice([-1.,1.],(Y.shape[1],32))/np.sqrt(32))
  A=(H@rv)[:,:,None]*(values@rt)[:,None,:];A=np.asarray(A.reshape(len(G),1024).T);b=A.sum(1)
  prob=np.linalg.norm(A,axis=0);prob=.9*prob/prob.sum()+.1/len(prob);cp=np.sort(rng.choice(len(G),4096,replace=False,p=prob))
  D=A[:,cp];scale=np.linalg.norm(A,axis=1)+1e-30;D=D/scale[:,None];target=b/scale;cn=np.linalg.norm(D,axis=0)+1e-30;D/=cn[None,:]
  full=H.T@values
  np.savez_compressed(o/f'tile{i}{j}_basis.npz',basis=W,eigenvalues=eigs)
  for cap in (128,256):
   fitstart=time.perf_counter();selected,weights=E.nnls_capped(D,target,cap);multipliers=weights/cn[selected];pos=cp[selected];g=G[pos];h=H[pos]*jnp.asarray(multipliers)[:,None]
   approx=h.T@values[pos];delta=np.asarray(jnp.linalg.norm(full-approx,axis=0));relative=delta/np.maximum(np.asarray(jnp.linalg.norm(full,axis=0)),1e-12)
   np.savez_compressed(o/f'tile{i}{j}_cap{cap}.npz',positions=pos,multipliers=multipliers,G=g,H=h,training_flux_absolute_error=delta,training_flux_relative_error=relative)
   fits.append(dict(tile=[i,j],cap=cap,support=len(pos),positive=bool(np.all(multipliers>=0)),sketch_relative_fit=float(np.linalg.norm(D[:,selected]@weights-target)/np.linalg.norm(target)),max_train_flux_error=float(delta.max()),max_train_flux_relative=float(relative.max()),fit_seconds=time.perf_counter()-fitstart,setup_elapsed=time.perf_counter()-t))
   cases=[r['case'] for r in tiles if r['rank']==256 and r['tile']==[i,j]]
   for case in cases:
    d=np.load(verified('c1data',f'case{128+case}.npz'));truth=jnp.asarray(d['coefficients']);norm=float(d['initial_norm']);a0=W.T@truth[0].ravel()/2
    for dt,solver in [(.005,f),(.0025,fh)]:
     states,sec=timer(lambda:solver(a0,dev[case,4],g,h,eigs));pred=(2*(states@W.T)).reshape(11,127,127);err=np.asarray(jnp.linalg.norm((pred-truth).reshape(11,-1),axis=1))/(2*norm)
     base=np.load(verified('c1qualify',f'tile{i}{j}_q128_dt0.005_case{case}.npz'))['states'];baseW=jnp.asarray(np.load(verified('c1qualify',f'tile{i}{j}_basis.npz'))['basis']);difference=np.asarray(jnp.linalg.norm(states@W.T-jnp.asarray(base)@baseW.T,axis=1))/norm
     np.savez_compressed(o/f'cap{cap}_dt{dt}_case{case}.npz',states=states,error=err)
     rows.append(dict(case=case,cap=cap,dt=dt,error=err,full_quadrature_change=difference,seconds=sec,finite=bool(np.isfinite(states).all()),evolved=float(err[[2,4,6,8,10]].max())))
   save(o/'result.json',dict(meta=meta(),fits=fits,rows=rows));print(i,j,cap,fits[-1],flush=True)
save(o/'result.json',dict(meta=meta(),fits=fits,rows=rows,complete=True,contract='positive empirical nonlinear-flux cubature trained on TRAIN only; mass/diffusion exact; full quadrature and actual PDE qualification retained'))
