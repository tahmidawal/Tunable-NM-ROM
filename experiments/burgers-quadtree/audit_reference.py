from pathlib import Path
import numpy as np,json
H=Path(__file__).resolve().parent;o=H/'results/reference01/output'
meta=json.loads((o/'result.json').read_text());rows=[]
def fd(i,L,dt,M=128):
 a=np.load(o/f'dev{i}_fd{L}_dt{dt}.npz')['out'].reshape(6,L-1,L-1)
 return np.pad(a,((0,0),(1,1),(1,1)))[:,::L//M,::L//M]
for i in range(8):
 levels=[fd(i,L,.000625) for L in (128,256,512)];norm=np.linalg.norm(levels[0][0])
 def delta(a,b):return float(np.linalg.norm((a-b)[1:].reshape(5,-1),axis=1).max()/norm)
 ds=[delta(levels[k],levels[k+1]) for k in (0,1)]
 ts=[fd(i,512,dt) for dt in (.0025,.00125,.000625)];dt=[delta(ts[k],ts[k+1]) for k in (0,1)]
 ps=[np.load(o/f'dev{i}_ps{N}.npz')['lat'][:,::2,::2] for N in (64,128,256)]
 dp=[delta(ps[k],ps[k+1]) for k in (0,1)]
 cross=delta(ps[-1],levels[-1]);regular=2.5<ds[0]/ds[1]<5.5 and 2.5<dt[0]/dt[1]<5.5
 estimate=(ds[-1]+dt[-1])/3 if regular else None
 rows.append(dict(case=i,space_deltas=ds,time_deltas=dt,ps_deltas=dp,cross_difference=cross,regular_second_order=regular,conditional_uncertainty=estimate,preliminary_pass=bool(regular and estimate<=.001 and cross<=.001),dense_extension_qualified=False))
r=dict(status='preliminary; independent scoring extension to 4096 not yet qualified',rows=rows)
(H/'reference-assessment.json').write_text(json.dumps(r,indent=2)+'\n')
for x in rows:print(x)
