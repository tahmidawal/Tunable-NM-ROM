"""Independent NumPy audit of all new frozen-width trajectories and qualification."""
from pathlib import Path
import json,hashlib
import numpy as np
from analyze_width_results import load_bank,compare_metrics
L=Path(__file__).resolve().parent;D=L/'checks/fresh';D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
report={'scope':'untouched frozen-choice same-family confirmation; same-grid same-step discrete target','case_count':8,'source':{},'arms':{},'all_qualified':True,'time':(np.arange(51)*.005).tolist(),'no_timing_claim':True}
reference0={};Gbase=None
for attempt,R,M in [('f128',128,2048),('f512',512,8192)]:
 folder=L/'runs'/attempt;out=folder/'output';env=read(out/'environment.json');cfg=env['config'];s=read(out/'summary.json');collect=read(folder/'collection.json');cohort=read(out/'cohort-drawing.json')
 assert collect['scientifically_qualified'] and cohort['guard_passed'];assert len(s['cases'])==8
 assert cfg['seed_first_draw']==2026111902 and cfg['case_ids']==list(range(8)) and cfg['solve_rows']==[M]
 G=load_bank(out);assert G.dtype==np.float64
 if Gbase is None:Gbase=G.copy()
 bankgap=float(np.linalg.norm(G[:,:128]-Gbase)/np.linalg.norm(Gbase));assert bankgap<=1e-10
 report['source'][str(R)]={'environment_sha256':sha(out/'environment.json'),'summary_sha256':sha(out/'summary.json'),'collection_sha256':sha(folder/'collection.json'),'cohort':cohort,'bank_prefix_gap':bankgap,'core_sha256':env['source_sha256']['core.py'],'job_id':env['job_id']}
 cases=[]
 for case in range(8):
  p=out/f'case-{case:02d}';rec=s['cases'][str(case)];arm=rec['arms'][str(M)];ref=np.load(p/'reference-1.npz')['fields'];norm0=np.linalg.norm(ref[0]);projection=np.load(p/'projection.npz');coeff=projection['coeff'];proj=coeff@G.T
  if R==128:reference0[case]=ref.copy()
  refgap=float(np.linalg.norm(ref-reference0[case],axis=1).max()/norm0)
  z=np.load(p/f'm{M}-tighter-rep0.npz');field=z['coeff']@G.T;metrics=compare_metrics(field,ref,256);e=metrics[:,0]
  floor=np.linalg.norm(proj-ref,axis=1)/norm0;departure=np.linalg.norm(field-proj,axis=1)/norm0
  metricgap=float(np.abs(metrics-z['metrics']).max());pyth=float(np.abs(e*e-floor*floor-departure*departure).max())
  levels={}
  for name,tol in cfg['solver_gtols'].items():
   arr=np.load(p/f'm{M}-{name}-rep0.npz');stats=arr['stats'];levels[name]={'stationary':bool(np.all(stats[:,1]==1)&np.all(stats[:,2]<=tol)),'max_stationarity':float(stats[:,2].max()),'finite':bool(np.isfinite(arr['coeff']).all()),'iterations':int(stats[:,0].sum())}
  tight=np.load(p/f'm{M}-tight-rep0.npz');tg=float(np.linalg.norm((z['coeff']-tight['coeff'])@G.T,axis=1).max()/norm0)
  threshold=min(1e-6,.01*float(e.max())) if e.max()>=1e-4 else 1e-6
  passed=bool(arm['numerically_eligible'] and rec['reference_eligible'] and refgap<=1e-8 and metricgap<=1e-9 and pyth<=1e-10 and tg<=threshold and all(v['finite'] and v['stationary'] for k,v in levels.items() if k!='prod'))
  row=dict(case=case,physical=cohort['physical_parameters'][case],qualified=passed,error=e.tolist(),projection_error=floor.tolist(),evolution_distance=departure.tolist(),gradient_error=metrics[:,1].tolist(),upwind_advection_error=metrics[:,2].tolist(),laplacian_error=metrics[:,3].tolist(),max_error=float(e.max()),initial_error=float(e[0]),max_evolved_error=float(e[1:].max()),max_error_time=float(.005*np.argmax(e)),metric_disagreement=metricgap,pythagorean_defect=pyth,common_reference_disagreement=refgap,tight_tighter_difference=tg,tight_tighter_threshold=threshold,levels=levels)
  cases.append(row);print(f'R{R} case{case} qualified={passed} error={e.max():.9g} metricdiff={metricgap:.2g}',flush=True)
 report['arms'][f'R{R}-M{M}']={'R':R,'M':M,'cases':cases,'median_max_error':float(np.median([r['max_error'] for r in cases])),'worst_max_error':max(r['max_error'] for r in cases),'qualified_count':sum(r['qualified'] for r in cases)}
 report['all_qualified'] &= all(r['qualified'] for r in cases)
 write(D/'analysis.json',report)
base=report['arms']['R128-M2048'];selected=report['arms']['R512-M8192'];ratios=np.asarray([s['max_error']/b['max_error'] for s,b in zip(selected['cases'],base['cases'])])
report['primary_confirmation']={'casewise_selected_over_baseline':ratios.tolist(),'median_paired_ratio':float(np.median(ratios)),'ratio_of_worst_errors':selected['worst_max_error']/base['worst_max_error'],'max_casewise_ratio':float(ratios.max()),'all_eight_improved':bool(np.all(ratios<1)),'gate_passed':bool(report['all_qualified'] and selected['worst_max_error']/base['worst_max_error']<=.8 and ratios.max()<=1.02)}
report['all_sources_same_cases']=report['source']['128']['cohort']['parameter_sha256']==report['source']['512']['cohort']['parameter_sha256'];assert report['all_sources_same_cases']
write(D/'analysis.json',report)
print(json.dumps(report['primary_confirmation'],indent=2))
