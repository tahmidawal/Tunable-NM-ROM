from pathlib import Path
import numpy as np,json
H=Path(__file__).resolve().parent;o=H/'results/c1classical/output';r=json.loads((o/'result.json').read_text());assert r['complete'];rows=r['rows'];out=[]
refs={v['case']:v['uncertainty'] for v in json.loads((H/'evidence/c1cost-assessment.json').read_text())['reference_audit']}
for N in (24,32,48):
 for dt in (.025,.0125):
  checks=[]
  for i in range(32):
   norm=float(np.load(H/f'results/c1data/output/case{128+i}.npz')['initial_norm'])
   def b(h,over=1):return np.load(o/f'N{N}_dt{h}_over{over}_case{i}.npz')['coefficients']
   def delta(x,y):return float(np.linalg.norm((x-y).reshape(11,-1),axis=1)[[2,4,6,8,10]].max()/(2*norm))
   row=next(v for v in rows if v['N']==N and v['dt']==dt and v['over']==1 and v['case']==i)
   checks.append(dict(case=i,error_plus_uncertainty=row['evolved']+refs[i],time_change=delta(b(dt),b(dt/2)),integration_change=delta(b(.0125),b(.0125,2)),seconds=row['seconds']))
  passed=all(v['error_plus_uncertainty']<=.01 and v['time_change']<=.001 and v['integration_change']<=.001 for v in checks)
  out.append(dict(N=N,dt=dt,passed=passed,worst_error_plus_uncertainty=max(v['error_plus_uncertainty'] for v in checks),worst_time_change=max(v['time_change'] for v in checks),worst_integration_change=max(v['integration_change'] for v in checks),median_prepared_seconds=float(np.median([v['seconds'] for v in checks])),cases=checks))
(H/'evidence/c1classical-assessment.json').write_text(json.dumps(dict(rows=out),indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k!='cases'} for x in out],indent=2))
