from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;o=H/'results/c3befficiency/output';r=json.loads((o/'result.json').read_text());assert r['complete'];refs={x['case']:x['uncertainty'] for x in json.loads((H/'evidence/c1bcomplete-assessment.json').read_text())['reference_audit']};lookup={(x['name'],x['q'],x['dt'],x['case']):x for x in r['rows']};rows=[]
for name in ('tree768','uniform1152'):
 for q in (64,96,128,192):
  for dt in (.025,.0125):
   err=[];quad=[];time=[];seconds=[]
   for case in range(8):
    norm=float(np.load(H/f'results/c1familyb/output/case{32+case}.npz')['initial_norm']);state=lambda q,dt:np.load(o/f'{name}_q{q}_dt{dt}_case{case}.npz')['states'];a=state(q,dt);b=state(256,dt);c=state(q,dt/2);x=lookup[name,q,dt,case];err.append(x['evolved']+refs[case]);quad.append(float(np.linalg.norm(a-b,axis=1)[[2,4,6,8,10]].max()/norm));time.append(float(np.linalg.norm(a-c,axis=1)[[2,4,6,8,10]].max()/norm));seconds.append(x['seconds'])
   rows.append(dict(name=name,q=q,dt=dt,worst_error_plus_uncertainty=max(err),worst_quadrature_change=max(quad),worst_time_change=max(time),median_prepared_seconds=float(np.median(seconds)),passed=bool(np.isfinite(err+quad+time).all() and max(err)<=.01 and max(quad)<=.001 and max(time)<=.001)))
classic=[]
for N in (32,48,64):
 errors=[];time=[];quad=[]
 for case in range(8):
  get=lambda dt,over:np.load(o/f'classical{N}_dt{dt}_over{over}_case{case}.npz');norm=float(np.load(H/f'results/c1familyb/output/case{32+case}.npz')['initial_norm']);base=get(.01,1);a=base['coefficients'];errors.append(float(max(base['error'][1:]))+refs[case]);time.append(float(np.linalg.norm((a-get(.005,1)['coefficients'])[1:].reshape(5,-1),axis=1).max()/(2*norm)));quad.append(float(np.linalg.norm((a-get(.01,2)['coefficients'])[1:].reshape(5,-1),axis=1).max()/(2*norm)))
 classic.append(dict(N=N,dt=.01,worst_error_plus_uncertainty=max(errors),worst_time_change=max(time),worst_integration_change=max(quad),passed=bool(np.isfinite(errors+time+quad).all() and max(errors)<=.01 and max(time)<=.001 and max(quad)<=.001)))
result=dict(rows=rows,classical=classic,contract='development same-input PDE/operator screen; no complete-cost claim');(H/'evidence/c3befficiency-assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passing=sorted([x for x in rows if x['passed']],key=lambda x:x['median_prepared_seconds']),classical=classic),indent=2))
