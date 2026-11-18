from campaign import *
import continuous as C,family_b as B,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
tr=B.draw(2026111711,256);dv=B.draw(2026111712,64);save(o/'parameters.json',dict(train=tr,dev=dv));r=dict(meta=meta(),cross_checks=[],fd=[],family='B')
f=C.full_trajectory(256,.0003125);hi=C.full_trajectory(512,.00015625)
for i,p in enumerate(tr[:32]+dv[:8]):
 b=P.analysis(jnp.asarray(B.initial(1024,p)[1:-1,1:-1]),1024)[:255,:255]
 v,sec=timer(lambda:f(b,p['nu']));norm=np.linalg.norm(B.initial(512,p))/512
 np.savez_compressed(o/f'case{i}.npz',coefficients=v,initial_norm=norm)
 if i>=32:
  hb=P.analysis(jnp.asarray(B.initial(2048,p)[1:-1,1:-1]),2048)[:511,:511];hv=hi(hb,p['nu']);lo=np.asarray(v);h=np.asarray(hv)
  diff=np.sqrt(np.sum((h[:,:255,:255]-lo)**2,axis=(1,2))+np.sum(h[:,255:,:]**2,axis=(1,2))+np.sum(h[:,:255,255:]**2,axis=(1,2)))/(2*norm)
  np.savez_compressed(o/f'check{i}.npz',coefficients=hv)
  r['cross_checks'].append(dict(case=i-32,error=diff,passed=bool(diff[1:].max()<=.001)))
 save(o/'result.json',r);print('familyB',i,sec,flush=True)
for L,dt in [(128,.0003125),(256,.0003125),(512,.0003125),(512,.00125),(512,.000625)]:
 solver=R.make_solver(2,L)
 for i,p in enumerate(dv[:8]):
  v,sec=timer(lambda:solver(B.initial(L,p)[1:-1,1:-1],p['nu'],dt,'cnr',1e-11,1e-10))
  np.savez_compressed(o/f'dev{i}_fd{L}_dt{dt}.npz',**v)
  r['fd'].append(dict(case=i,L=L,dt=dt,seconds=sec,max_residual=float(np.nanmax(v['res']))));save(o/'result.json',r)
r['complete']=True;save(o/'result.json',r)
