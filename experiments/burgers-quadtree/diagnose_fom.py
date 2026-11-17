from campaign import *
from owned_inputs import verified
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();rows=[]
for L,dt in [(128,.005),(128,.00125),(256,.00125),(512,.00125),(1024,.00125)]:
 solver=R.make_solver(2,L,adv='upwind')
 for i,ph in enumerate(dev[:8]):
  u0=E.initial(L,ph)[1:-1,1:-1];v,sec=timer(lambda:solver(u0,ph[4],dt,'be',1e-11,1e-10))
  pred=np.pad(np.asarray(v['out']).reshape(6,L-1,L-1),((0,0),(1,1),(1,1)))[:,::L//128,::L//128]
  ref=np.pad(np.load(verified('reference01',f'dev{i}_fd512_dt0.000625.npz'))['out'].reshape(6,511,511),((0,0),(1,1),(1,1)))[:,::4,::4]
  error=np.linalg.norm((pred-ref).reshape(6,-1),axis=1)/np.linalg.norm(ref[0]);res=float(np.max(np.asarray(v['res'])[:int(v['stages'])]))
  np.savez_compressed(o/f'L{L}_dt{dt}_case{i}.npz',fields129=pred,res=v['res'],newton=v['nwt'])
  rows.append(dict(L=L,dt=dt,case=i,error=error,max_residual=res,seconds=sec));save(o/'result.json',dict(meta=meta(),rows=rows));print(L,dt,i,error[1:].max(),flush=True)
