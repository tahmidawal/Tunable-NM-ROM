"""Saved-array audit and plots for bounded true16 proposal screen; no model execution."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--source',default=str(HERE/'evidence'));a=ap.parse_args();src=Path(a.source)
x=json.loads((src/'result.json').read_text());assert x['complete'] and x['backend']=='gpu' and x['x64'] and x['precision']=='highest'
assert len(x['controls'])==2 and all(q['passed'] for q in x['controls'])
for name,q in x['qualification'].items():
 with np.load(src/f'quadrature_{name}.npz') as z:
  errors=[]
  for metric in ['r','J','N','NJ']:
   v,w=z[f'candidate_{metric}'],z[f'reference_{metric}']
   axes=tuple(range(1,v.ndim));rat=np.sqrt(np.sum((v-w)**2,axis=axes))/np.maximum(np.sqrt(np.sum(w*w,axis=axes)),1e-12)
   errors.append(float(np.max(rat)))
  np.testing.assert_allclose(errors,q['max_relative_errors'],rtol=1e-10,atol=1e-13)
  assert (max(errors)<=1e-3)==q['passed']
records={}
for r in x['records']:
 arm,ci=r['arm'],r['case']
 with np.load(src/f'{arm}_{ci}.npz') as z:
  assert int(z['it_sum'])==r['fine_updates'];np.testing.assert_allclose(z['gamma_fresh'],z['step_gn'],rtol=1e-7,atol=1e-12)
  valid=bool(np.isfinite(z['gamma_fresh']).all() and np.all((z['gamma_fresh']<=1e-3)|(z['residual_norm_fresh']<=r['initial_scale']*1e-9)) and z['nfail']==0)
  assert valid==r['valid']
  if arm!='direct':
   si=z['seed_info'];take=si[:,0].astype(bool)
   assert si[:,1].sum()==r['coarse_updates'] and take.sum()==r['accepted']
   assert np.all(z['seed_codes'][:,80:]==0)
   assert np.all(si[take,8]<(1-1e-8)*si[take,7]) and np.all(si[take,14]==1)
   ratios=(si[:,15]/np.maximum(si[:,7],1e-300)).tolist()
   proposals=(si[:,8]/np.maximum(si[:,7],1e-300)).tolist()
  else:ratios=[];proposals=[]
  records[arm,ci]=dict(**r,truncation_loss_ratios=ratios,proposal_loss_ratios=proposals)
lookup={(t['arm'],t['case'],t['block']):t for t in x['timings']};arms=sorted({k[0] for k in lookup},key=lambda a:(a!='direct',a))
assert len(lookup)==len(arms)*42==len(x['timings'])
summary=dict(job_id=x['job_id'],source_sha256=hashlib.sha256((src/'result.json').read_bytes()).hexdigest(),qualification=x['qualification'],components=[],arms={},excluded_repetitions=0)
for comp in x['components']:
 vals=np.array(comp['seconds'])*1e6;q1,q3=np.quantile(vals,[.25,.75]);iqr=q3-q1
 summary['components'].append(dict(config=comp['config'],K=comp['K'],operation=comp['operation'],median_us=float(np.median(vals)),outliers=int(np.sum((vals<q1-1.5*iqr)|(vals>q3+1.5*iqr))),samples_us=vals.tolist()))
for arm in arms:
 secs=np.array([[lookup[arm,c,b]['seconds'] for b in range(7)] for c in range(6)])
 base=np.array([[lookup['direct',c,b]['seconds'] for b in range(7)] for c in range(6)])
 ratios=secs/base;med=np.median(ratios,axis=1)
 rr=[records[arm,c] for c in range(6)]
 entry=dict(seconds=secs.tolist(),paired_ratios=ratios.tolist(),median_ratio=float(np.median(med)),worst_case_ratio=float(np.max(med)),slower_pairs=int(np.sum(ratios>1)),
  fine_updates=sum(r['fine_updates'] for r in rr),coarse_updates=sum(r['coarse_updates'] for r in rr),accepted=sum(r['accepted'] for r in rr),
  production_field_gap_max=max(r['field_gap_to_production'] for r in rr),all_stationary=all(r['valid'] for r in rr))
 if arm!='direct':
  tr=np.array(sum([r['truncation_loss_ratios'] for r in rr],[]));pr=np.array(sum([r['proposal_loss_ratios'] for r in rr],[]))
  entry.update(truncation_loss_ratio_median=float(np.median(tr)),proposal_loss_ratio_median=float(np.median(pr)))
  assert x['gates'][arm]['passed']==bool(entry['all_stationary'] and np.median(med)<=.85 and np.max(med)<=1.10)
  np.testing.assert_allclose(ratios,x['gates'][arm]['per_case_ratios'],rtol=1e-13)
  entry['cost_gate']=x['gates'][arm]['passed']
 q1,q3=np.quantile(secs,[.25,.75],axis=1);iqr=q3-q1
 entry['timing_outliers']=int(np.sum((secs<(q1-1.5*iqr)[:,None])|(secs>(q3+1.5*iqr)[:,None])))
 summary['arms'][arm]=entry
out=HERE.parent/'plots/cheap';out.mkdir(exist_ok=True,parents=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,5),layout='constrained')
configs=['fine64','full','q64','q32m64'];labels=['Fine64\n16,384 pts','Coarse16\n16,384 pts','Coarse16\n4,096 pts','Coarse16\n1,024 pts; M64']
values=[next(c['median_us'] for c in summary['components'] if c['config']==name and c['operation']=='full_residual_jacobian') for name in configs]
colors=['#2265a8']+['#21835b' if summary['qualification'][n]['passed'] else '#bdbdbd' for n in configs[1:]]
ax.bar(range(4),values,color=colors)
for i,v in enumerate(values):ax.text(i,v+max(values)*.03,f'{v:.1f}',ha='center')
ax.set(xticks=range(4),xticklabels=labels,ylabel='Microseconds per residual + Jacobian evaluation',title='Can the coarse calculation be cheaper?',ylim=(0,max(values)*1.22))
nonbase=[a for a in arms if a!='direct'];ratios=[summary['arms'][a]['median_ratio'] for a in nonbase]
bx.bar(range(len(nonbase)),ratios,color='#d87916');bx.axhline(1,color='#2265a8',label='Direct64 baseline')
for i,v in enumerate(ratios):bx.text(i,v+.02,f'{v:.2f}×',ha='center')
bx.set(xticks=range(len(nonbase)),xticklabels=[a.replace('_b','\n').replace('q32m64','32² / M64').replace('q64','64²').replace('full','128²')+' updates' for a in nonbase],ylabel='Full-query time / direct64 time',title='Full-quadrature control: complete solver',ylim=(0,max(ratios+[1])*1.23))
bx.legend(frameon=False)
fig.suptitle('Cheap coarse calculations versus actual solver savings',fontsize=17)
fig.supxlabel(f"Job {x['job_id']} · left: isolated synchronized calls at case1 initial state (21 repetitions)\nRight: full-quadrature proposals only; 6 cases × 7 paired repetitions.\nGrey bars: quadrature panel gate failed; these settings were not tested end-to-end.",fontsize=10)
fig.savefig(out/'coarse-feasibility.png',dpi=170);fig.savefig(out/'coarse-feasibility.pdf')
print(json.dumps(summary,indent=2))
