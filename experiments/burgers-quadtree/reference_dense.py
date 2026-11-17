from campaign import *
from owned_inputs import verified
import argparse
sys.path.insert(0,str(T/'jcp-fom2d'));import psjax2d as PJ
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
rep=dict(meta=meta(),rows=[],contract='4096 nodes including walls; final field saved, all five evolved fields scored')
train,dev=draw()
for i,ph in enumerate(dev[:8]):
 bps=jnp.asarray(np.load(verified('reference01',f'dev{i}_ps256.npz'))['coef_final'])
 fds=[]
 for L in (256,512):
  arr=np.load(verified('reference01',f'dev{i}_fd{L}_dt0.000625.npz'))['out'].reshape(6,L-1,L-1)
  fds.append(jax.vmap(lambda u:P.analysis(u,L))(jnp.asarray(arr[1:])))
 row=dict(case=i,grids=[])
 for nodes in (512,1024,2048,4096):
  M=nodes-1
  # 512-node grid cannot contain 511 sine modes with M=511; direct synthesis handles it.
  x=jnp.arange(nodes,dtype=jnp.float64)/M
  def synth(b):
   if b.shape[0]>=M:
    S=jnp.sin(jnp.pi*x[:,None]*jnp.arange(1,b.shape[0]+1)[None,:]);return S@b@S.T
   return PJ.synth_mesh(b,M)
  norm=float(jnp.linalg.norm(jnp.asarray(E.initial(M,ph))))
  errs=[]
  for t in range(5):
   lo=synth(fds[0][t]);hi=synth(fds[1][t]);errs.append(float(jnp.linalg.norm(hi-lo))/norm)
  ps=synth(bps);hi=synth(fds[1][-1]);cross=float(jnp.linalg.norm(ps-hi))/norm
  row['grids'].append(dict(nodes=nodes,fd_space_deltas=errs,final_ps_fd=cross))
  if nodes==4096:
   assert hi.shape==(4096,4096);np.savez_compressed(o/f'dev{i}_final4096.npz',field=np.asarray(hi),parameters=ph)
 row['score_grid_change']=float(np.max(np.abs(np.asarray(row['grids'][-1]['fd_space_deltas'])-np.asarray(row['grids'][-2]['fd_space_deltas']))))
 row['grid_change_pass']=row['score_grid_change']<=.0002
 rep['rows'].append(row);save(o/'result.json',rep);print(i,row['score_grid_change'],flush=True)
rep['complete']=True;save(o/'result.json',rep)
