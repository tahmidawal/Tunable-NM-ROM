from campaign import *
from owned_inputs import verified
import continuous as C,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
_,dev=draw();rows=[]
for L,dt in [(128,.000625),(256,.000625),(512,.000625),(512,.0025),(512,.00125)]:
 sol=R.make_solver(2,L)
 for i,ph in enumerate(dev[8:32],8):
  v,sec=timer(lambda:sol(E.initial(L,ph)[1:-1,1:-1],ph[4],dt,'cnr',1e-11,1e-10))
  np.savez_compressed(o/f'dev{i}_fd{L}_dt{dt}.npz',**v)
  rows.append(dict(case=i,L=L,dt=dt,seconds=sec,residual=float(np.nanmax(v['res']))));save(o/'result.json',dict(meta=meta(),rows=rows));print(i,L,dt,flush=True)
# Strongest spatial/temporal spectral level on the pre-identified hardest representation case.
i=9;ph=dev[i];v=C.full_trajectory(512,.00015625)(jnp.asarray(P.ic_coefs_2d(ph,512,1024)),ph[4]);np.savez_compressed(o/'dev9_ps512.npz',coefficients=v)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True))
