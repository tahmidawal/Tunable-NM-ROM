from campaign import *
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
train,dev=draw();params=np.concatenate([train[:128],dev[:32]]);np.savez(o/'parameters.npz',train=train,dev=dev)
r=dict(meta=meta(),cross_checks=[],projection=[],status='data/spans; not PDE ROM')
sol=C.full_trajectory(128,.000625);check=C.full_trajectory(256,.0003125);cols=[];norms=[]
for i,ph in enumerate(params):
 b=jnp.asarray(P.ic_coefs_2d(ph,128,256));v,sec=timer(lambda:sol(b,ph[4]));norm=np.linalg.norm(E.initial(512,ph))/512;norms.append(norm)
 np.savez_compressed(o/f'case{i}.npz',coefficients=v,parameters=ph,initial_norm=norm)
 cols.append(np.asarray(v).reshape(11,-1).T/(2*norm))
 if i>=128 or i%16==0:
  hi=check(jnp.asarray(P.ic_coefs_2d(ph,256,512)),ph[4]);lo=np.asarray(v);hd=np.asarray(hi)
  diff=np.sqrt(np.sum((hd[:,:127,:127]-lo)**2,axis=(1,2))+np.sum(hd[:,127:,:]**2,axis=(1,2))+np.sum(hd[:,:127,127:]**2,axis=(1,2)))/(2*norm)
  r['cross_checks'].append(dict(case=i,errors=diff,max_evolved=float(diff[1:].max()),passed=bool(diff[1:].max()<=.001)))
  np.savez_compressed(o/f'check{i}.npz',coefficients=hi)
 save(o/'result.json',r);print('data',i,sec,flush=True)
Y=jnp.asarray(np.concatenate(cols[:128],axis=1));D=jnp.asarray(np.concatenate(cols[128:],axis=1));t=time.perf_counter()
# Exact thin SVD on the GPU, with large training array explicitly supplied.
U,s,vh=jax.jit(lambda x:jnp.linalg.svd(x,full_matrices=False))(Y);jax.block_until_ready(U)
np.savez_compressed(o/'pod.npz',basis=U[:,:768],singular_values=s,norms=norms)
r['pod_seconds']=time.perf_counter()-t
for rank in (64,128,256,384,512,768):
 err=np.asarray(jnp.linalg.norm(D-U[:,:rank]@(U[:,:rank].T@D),axis=0)).reshape(32,11)
 r['projection'].append(dict(rank=rank,per_time=err,evolved_max=err[:,[2,4,6,8,10]].max(1),initial=err[:,0]))
r['complete']=True;save(o/'result.json',r)
