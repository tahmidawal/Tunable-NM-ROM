from pathlib import Path
import numpy as np,json,sys
H=Path(__file__).resolve().parent;job=sys.argv[1] if len(sys.argv)>1 else 'c3globalcompress';assert job in ('c3globalcompress','c3globalcapacity');o=H/f'results/{job}/output';r=json.loads((o/'result.json').read_text());assert r['complete'];lookup={(x['name'],x['rank'],x['q'],x['dt'],x['case']):x for x in r['rows']};refs={x['case']:x['uncertainty'] for x in json.loads((H/'evidence/c1cost-assessment.json').read_text())['reference_audit']};rows=[]
def state(name,rank,q,dt,case):return np.load(o/f'{name}_r{rank}_q{q}_dt{dt}_case{case}.npz')['states']
for name in ('neural','POD'):
 for rank in sorted({x['rank'] for x in r['rows']}):
  for q in sorted({x['q'] for x in r['rows']} - {192}):
   errors=[];quad=[];time=[];seconds=[];projection=[]
   for case in range(32):
    x=lookup[name,rank,q,.0125,case];norm=float(np.load(H/f'results/c1data/output/case{128+case}.npz')['initial_norm']);a=state(name,rank,q,.0125,case);b=state(name,rank,192,.0125,case);c=state(name,rank,q,.00625,case);errors.append(x['evolved']+refs[case]);quad.append(float(np.linalg.norm(a-b,axis=1)[[2,4,6,8,10]].max()/norm));time.append(float(np.linalg.norm(a-c,axis=1)[[2,4,6,8,10]].max()/norm));seconds.append(x['seconds']);projection.append(max(np.asarray(x['projection'])[[2,4,6,8,10]]))
   passed=bool(np.isfinite(errors+quad+time).all() and max(errors)<=.01 and max(quad)<=.001 and max(time)<=.001);rows.append(dict(name=name,rank=rank,q=q,dt=.0125,worst_error_plus_uncertainty=max(errors),worst_quadrature_change=max(quad),worst_time_change=max(time),worst_projection=max(projection),passed=passed,median_prepared_seconds=float(np.median(seconds))))
(H/f'evidence/{job}-assessment.json').write_text(json.dumps(dict(rows=rows,contract='development prepared-input screen; finer physical features and complete-output cost still required'),indent=2)+'\n');print(json.dumps(sorted([x for x in rows if x['passed']],key=lambda x:x['median_prepared_seconds']),indent=2))
