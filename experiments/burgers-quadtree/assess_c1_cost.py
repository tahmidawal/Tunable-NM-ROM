"""Pure NumPy audit of completed paired cost panel; no experiment or test access."""
from pathlib import Path
import numpy as np,json
H=Path(__file__).resolve().parent;o=H/'results/c1cost/output';r=json.loads((o/'result.json').read_text());assert r.get('complete') and len(r['rows'])==32
rows=r['rows'];methods=list(rows[0]['times']);uncertainty=[];refs=[]
for x in rows:
 c=x['reference_checks'];ds=np.max([z['space_deltas'] for z in c],axis=0);dt=np.max([z['time_deltas'] for z in c],axis=0);regular=2.5<ds[0]/ds[1]<5.5 and 2.5<dt[0]/dt[1]<5.5;u=(ds[-1]+dt[-1])/3
 refs.append(dict(case=x['case'],space_ratio=ds[0]/ds[1],time_ratio=dt[0]/dt[1],uncertainty=u,cross=max(z['cross_difference'] for z in c),score_change=max(z['score_grid_change'] for z in c),passed=bool(regular and u<=.001 and max(z['cross_difference'] for z in c)<=.001 and max(z['score_grid_change'] for z in c)<=.0002)));uncertainty.append(u)
def coefficient_difference(case,a,b):
 norm=float(np.load(H/f'results/c1data/output/case{128+case}.npz')['initial_norm'])
 A=np.load(o/f'{a}_case{case}_block0_rep0.npz')['coefficients'];B=np.load(o/f'{b}_case{case}_block0_rep0.npz')['coefficients'];n=max(A.shape[1],B.shape[1]);A=np.pad(A,((0,0),(0,n-A.shape[1]),(0,n-A.shape[2])));B=np.pad(B,((0,0),(0,n-B.shape[1]),(0,n-B.shape[2])))
 return float(np.linalg.norm((A-B)[1:].reshape(5,-1),axis=1).max()/(2*norm))
checks={}
for name,other in [('spectral16_dt005','spectral32_dt005'),('spectral32_dt005','spectral64_dt005'),('spectral32_dt01','spectral32_dt005'),('spectral64_dt005','spectral64_dt00125')]:checks[name]=[coefficient_difference(i,name,other) for i in range(32)]
metrics=[]
for name in methods:
 errors=np.array([max(x['errors'][name]) for x in rows]);timings=np.array([x['times'][name] for x in rows]);qualified=all(x['passed'] for x in refs) and np.all(errors+uncertainty<=.01)
 if name in checks:qualified=qualified and max(checks[name])<=.001
 if name=='spectral32_dt01':qualified=qualified and max(checks['spectral32_dt005'])<=.001
 metrics.append(dict(name=name,worst_error=float(errors.max()),worst_error_plus_uncertainty=float(np.max(errors+uncertainty)),qualified=bool(qualified),median_seconds=float(np.median(timings)),per_case_block_medians=np.median(timings,axis=2).tolist()))
classical=[x for x in metrics if x['qualified'] and x['name']!='local_hybrid'];speed=None
if classical:
 best=min(classical,key=lambda x:x['median_seconds']);rom=next(x for x in metrics if x['name']=='local_hybrid');ratios=np.asarray(best['per_case_block_medians'])/np.asarray(rom['per_case_block_medians']);rng=np.random.default_rng(2026111806);sample=rng.integers(0,32,(10000,32));ci=np.quantile(np.median(ratios[sample],axis=1),[.025,.975],axis=0)
 speed=dict(baseline=best['name'],per_block_median_speedup=np.median(ratios,axis=0).tolist(),paired_trajectory_bootstrap_95=ci.tolist(),complete_accuracy_pass=rom['qualified'],setup_seconds=r['setup_seconds'],note='development ratios; no final test claim; setup must be amortized separately')
result=dict(reference_audit=refs,methods=metrics,coefficient_refinement=checks,speed=speed)
(H/'evidence/c1cost-assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(methods=[{k:v for k,v in x.items() if k!='per_case_block_medians'} for x in metrics],speed=speed),indent=2))
