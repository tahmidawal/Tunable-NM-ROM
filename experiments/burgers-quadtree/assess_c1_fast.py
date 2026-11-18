"""Select only prospectively registered development policies, retain every failure."""
from pathlib import Path
import numpy as np,json
H=Path(__file__).resolve().parent;o=H/'results/c1fast/output';r=json.loads((o/'result.json').read_text());assert r['complete'];lookup={(x['rank'],x['q'],x['dt'],x['case']):x for x in r['rows']};rows=[]
refs={x['case']:x['conditional_uncertainty'] for x in json.loads((H/'reference-assessment.json').read_text())['rows']}
refs.update({x['case']:x['conditional_uncertainty'] for x in json.loads((H/'evidence/c1audit-result.json').read_text())['rows'] if x['family']=='A'})
def state(rank,q,dt,case):return np.load(o/f'r{rank}_q{q}_dt{dt}_case{case}.npz')['states']
for threshold in (0.,.06,.08,.10,1.):
 for q in (48,64,80,96,128):
  for dt in (.025,.0125,.005):
   errors=[];quad=[];time=[];seconds=[];ranks=[]
   for case in range(32):
    rank=256 if threshold==1 or (threshold>0 and lookup[256,128,.005,case]['estimated_width']<threshold) else 128;ranks.append(rank)
    x=lookup[rank,q,dt,case];norm=float(np.load(H/f'results/c1data/output/case{128+case}.npz')['initial_norm']);a=state(rank,q,dt,case);b=state(rank,192,dt,case);c=state(rank,q,dt/2,case)
    errors.append(x['evolved']+refs[case]);quad.append(float(np.linalg.norm(a-b,axis=1)[[2,4,6,8,10]].max()/norm));time.append(float(np.linalg.norm(a-c,axis=1)[[2,4,6,8,10]].max()/norm));seconds.append(x['seconds'])
   passed=bool(np.isfinite(errors+quad+time).all() and max(errors)<=.01 and max(quad)<=.001 and max(time)<=.001)
   rows.append(dict(threshold=threshold,q=q,dt=dt,ranks=ranks,worst_error_plus_uncertainty=max(errors),worst_quadrature_change=max(quad),worst_time_change=max(time),passed=passed,median_prepared_seconds=float(np.median(seconds))))
(H/'evidence/c1fast-assessment.json').write_text(json.dumps(dict(rows=rows,contract='development prepared-input screen; complete cost and doubled representation still required'),indent=2)+'\n')
print(json.dumps(sorted([x for x in rows if x['passed']],key=lambda x:x['median_prepared_seconds']),indent=2))
