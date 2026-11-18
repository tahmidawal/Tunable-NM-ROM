"""Independent NumPy cost/refinement assessment of terminal owned panels."""
from pathlib import Path
import json,numpy as np,subprocess,sys
H=Path(__file__).resolve().parent;job=sys.argv[1];assert job in ('c1compresscost','c1bcomplete');r=json.loads((H/f'results/{job}/output/result.json').read_text());assert r['complete'];B=job=='c1bcomplete';count=8 if B else 32;assert len(r['rows'])==count
# Read-only remote CPU norms avoid downloading hundreds of MB of derived arrays.
script=r'''
from pathlib import Path
import json,numpy as np,hashlib
job=JOB;root=Path('/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117')/job
manifest=dict((name,digest) for digest,name in (x.split('  ',1) for x in (root/'OUTPUTS.sha256').read_text().splitlines()))
def read(name):
 p=root/'output'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['output/'+name];return np.load(p)['coefficients']
r=json.loads((root/'output/result.json').read_text());rows=[]
for x in r['resolution']:
 dt=.00625 if job=='c1bcomplete' else .0125
 if x['dt']!=dt:continue
 case=x['case'];prefix='resolved_split'+''.join(map(str,x['split'])) if job=='c1bcomplete' else 'resolved'
 a=read(f'{prefix}_dt{dt}_case{case}.npz');b=read(f'{prefix}_dt{dt/2}_case{case}.npz')
 data=root.parent/('c1familyb' if job=='c1bcomplete' else 'c1data')/'output'/f'case{case+(32 if job=="c1bcomplete" else 128)}.npz'
 norm=float(np.load(data)['initial_norm']);rows.append(dict(case=case,split=x.get('split'),fine_time_change=(np.linalg.norm((a-b).reshape(11,-1),axis=1)/(2*norm)).tolist()))
print(json.dumps(rows))
'''.replace('JOB',repr(job))
remote=json.loads(subprocess.check_output(['ssh','tufts-login','/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python -'],input=script.encode()));(H/f'evidence/{job}-fine-time.json').write_text(json.dumps(remote,indent=2)+'\n')
if B:
 refs=[]
 for x in r['rows']:
  c=x['reference_checks'];ds=np.max([z['space_deltas'] for z in c],axis=0);dt=np.max([z['time_deltas'] for z in c],axis=0);u=(ds[-1]+dt[-1])/3
  refs.append(dict(case=x['case'],uncertainty=u,space_ratio=ds[0]/ds[1],time_ratio=dt[0]/dt[1],cross=max(z['cross_difference'] for z in c),score_change=max(z['score_grid_change'] for z in c),passed=bool(2.5<ds[0]/ds[1]<5.5 and 2.5<dt[0]/dt[1]<5.5 and u<=.001 and max(z['cross_difference'] for z in c)<=.001 and max(z['score_grid_change'] for z in c)<=.0002)))
else:refs=json.loads((H/'evidence/c1cost-assessment.json').read_text())['reference_audit']
uncert=np.array([x['uncertainty'] for x in refs]);romnames=['tree768','uniform1152'] if B else ['compressed384'];methods=[]
for name in r['rows'][0]['times']:
 err=np.array([max(x['errors'][name]) for x in r['rows']]);timings=np.array([x['times'][name] for x in r['rows']]);checks={};qualified=bool(all(x['passed'] for x in refs) and np.all(err+uncert<=.01))
 if name in romnames:
  size=2 if name=='tree768' else 4
  fine=[x for x in r['resolution'] if not B or len(x['split'])==size];time=[x for x in remote if not B or len(x['split'])==size]
  checks['feature_change']=max(max(np.asarray(x['feature_change'])[[2,4,6,8,10]]) for x in fine);checks['fine_time_change']=max(max(np.asarray(x['fine_time_change'])[[2,4,6,8,10]]) for x in time)
  if B:
   path=H/'evidence/c1bcontrols-result.json'
   if path.exists():
    control=[x for x in json.loads(path.read_text())['rows'] if x['name']==name];assert len(control)==24
    checks['query_time_change']=max(max(np.asarray(x['change'])[[2,4,6,8,10]]) for x in control if x['dt']==.003125);checks['query_integration_change']=max(max(np.asarray(x['change'])[[2,4,6,8,10]]) for x in control if x['dt']==.00625)
   else:checks['query_controls_pending']=True;qualified=False
  else:
   x=next(x for x in json.loads((H/'evidence/c1compress-assessment.json').read_text())['rows'] if x['rank']==384 and x['q']==128 and x['dt']==.0125);checks['query_time_change']=x['worst_time_change'];checks['query_integration_change']=x['worst_quadrature_change']
  qualified=qualified and all(v<=.001 for k,v in checks.items() if k!='query_controls_pending')
 else:
  if B:
   checks['query_time_change']=max(max(x['classical_checks'][name]['time_change'][1:]) for x in r['rows']);checks['query_integration_change']=max(max(x['classical_checks'][name]['integration_change'][1:]) for x in r['rows'])
  else:
   n=int(name.removeprefix('spectral'));x=next(x for x in json.loads((H/'evidence/c1classical-assessment.json').read_text())['rows'] if x['N']==n and x['dt']==.0125);checks['query_time_change']=x['worst_time_change'];checks['query_integration_change']=x['worst_integration_change']
  qualified=qualified and all(v<=.001 for v in checks.values())
 methods.append(dict(name=name,worst_error=float(err.max()),worst_error_plus_uncertainty=float((err+uncert).max()),qualified=bool(qualified),checks=checks,median_seconds=float(np.median(timings)),per_case_block_medians=np.median(timings,axis=2).tolist()))
classical=[x for x in methods if x['name'] not in romnames and x['qualified']];speed=[]
if classical:
 best=min(classical,key=lambda x:x['median_seconds']);rng=np.random.default_rng(2026111815);sample=rng.integers(0,count,(10000,count))
 for rom in [x for x in methods if x['name'] in romnames]:
  ratio=np.asarray(rom['per_case_block_medians'])/np.asarray(best['per_case_block_medians']);ci=np.quantile(np.median(ratio[sample],axis=1),[.025,.975],axis=0);fraction=np.mean(ratio<1,axis=0);q90=np.quantile(ratio,.9,axis=0);med=np.median(ratio,axis=0)
  speed.append(dict(name=rom['name'],baseline=best['name'],primary_median_time_ratio=med.tolist(),paired_case_bootstrap_95=ci.tolist(),fraction_faster=fraction.tolist(),ratio_q90=q90.tolist(),useful_speed_pass=bool(rom['qualified'] and np.all(ci[1]<1) and np.all(fraction>=.75) and np.all(q90<=1.1)),substantial_speed_pass=bool(rom['qualified'] and np.all(med<=.5)),rom_accuracy_qualified=rom['qualified']))
result=dict(reference_audit=refs,methods=methods,speed=speed,setup_seconds=r['setup_seconds'],contract='development matched-output costs; independent numerical controls; offline/cache setup separately charged, no final-test claim')
(H/f'evidence/{job}-assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({**result,'reference_audit':'retained','methods':[{k:v for k,v in x.items() if k!='per_case_block_medians'} for x in methods]},indent=2))
