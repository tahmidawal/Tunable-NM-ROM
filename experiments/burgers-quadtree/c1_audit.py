"""Independent FD convergence audit; no new reference solve or unseen cases."""
from campaign import *
from owned_inputs import verified
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();o=Path(args.out);o.mkdir(parents=True,exist_ok=True)
rows=[]
for family,cases,job,fine in [('A',range(8,32),'c1ref32',.000625),('B',range(8),'c1familyb',.0003125)]:
 for i in cases:
  def fd(L,dt):
   a=np.load(verified(job,f'dev{i}_fd{L}_dt{dt}.npz'))['out'].reshape(6,L-1,L-1)
   return np.pad(a,((0,0),(1,1),(1,1)))[:,::L//128,::L//128]
  levels=[fd(L,fine) for L in (128,256,512)];norm=np.linalg.norm(levels[0][0])
  def delta(a,b):return float(np.linalg.norm((a-b)[1:].reshape(5,-1),axis=1).max()/norm)
  ds=[delta(levels[k],levels[k+1]) for k in (0,1)];ts=[fd(512,t) for t in (4*fine,2*fine,fine)];dt=[delta(ts[k],ts[k+1]) for k in (0,1)]
  data=np.load(verified('c1data' if family=='A' else job,f'case{128+i if family=="A" else 32+i}.npz'));b=data['coefficients'][::2]
  k=np.arange(1,b.shape[1]+1);S=np.sin(np.pi*np.arange(129)[:,None]*k/128)
  ps=np.einsum('xi,tij,yj->txy',S,b,S,optimize=True)
  cross=delta(ps,levels[-1]);regular=2.5<ds[0]/ds[1]<5.5 and 2.5<dt[0]/dt[1]<5.5
  est=(ds[-1]+dt[-1])/3 if regular else None
  rows.append(dict(family=family,case=i,space_deltas=ds,time_deltas=dt,space_ratio=ds[0]/ds[1],time_ratio=dt[0]/dt[1],cross_difference=cross,conditional_uncertainty=est,preliminary_pass=bool(regular and est<=.001 and cross<=.001),dense_extension_qualified=False))
 save(o/'result.json',dict(meta=meta(),rows=rows));print(family,all(r['preliminary_pass'] for r in rows if r['family']==family),flush=True)
save(o/'result.json',dict(meta=meta(),rows=rows,complete=True,contract='conditional convergence audit, not rigorous bound; dense4096 scoring remains'))
