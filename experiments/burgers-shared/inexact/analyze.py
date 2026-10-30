"""Audit same-invocation timing/loss witnesses and render their comparison."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--source',default=str(HERE/'evidence'));a=ap.parse_args();src=Path(a.source)
x=json.loads((src/'result.json').read_text());assert x['complete'] and x['backend']=='gpu' and x['x64'] and x['precision']=='highest'
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
arms=['direct','q32m64_b1','q32m64_b2'];records={(r['arm'],r['case']):r for r in x['records']}
assert len(x['controls'])==2 and all(c['passed'] for c in x['controls'])
certs={};aggregate={}
for arm in arms:
 accepted=coarse=fine=0
 for ci in range(6):
  with np.load(src/f'{arm}_{ci}.npz') as z:
   r=records[arm,ci];rn=z['residual_norm_fresh'];gn=z['gamma_fresh'];scale=r['initial_scale']
   assert np.all((gn<=1e-3)|(rn<=scale*1e-9)) and z['nfail']==0 and int(z['it_sum'])==r['fine_updates']
   np.testing.assert_allclose(gn,z['step_gn'],rtol=1e-7,atol=1e-12)
   np.testing.assert_array_equal(z['W'],z['states'][::10])
   certs[arm,ci]=(rn,gn,sha(z['W']),np.array(z['step_it']))
   fine+=int(z['it_sum'])
   if arm!='direct':
    info=z['seed_info'];take=info[:,0].astype(bool)
    np.testing.assert_array_equal(take,(info[:,14]==1)&np.isfinite(info[:,7])&np.isfinite(info[:,8])&(info[:,8]<info[:,7])&(info[:,8]<=(1-1e-8)*info[:,7]))
    assert np.all(z['seed_codes'][:,80:]==0)
    coarse+=int(info[:,1].sum());accepted+=int(take.sum())
 aggregate[arm]=dict(fine_updates=fine,coarse_updates=coarse,accepted=accepted)
for ci in range(6):
 with np.load(src/f'tight_{ci}.npz') as z,np.load(src/f'direct_{ci}.npz') as d:
  assert np.all(z['gamma_fresh']<=1e-7) and z['nfail']==0
  np.testing.assert_array_equal(z['states'][0],d['states'][0])
  np.testing.assert_allclose(z['gamma_fresh'],z['step_gn'],rtol=1e-6,atol=1e-12)
lookup={}
for t in x['timings']:
 key=t['arm'],t['case'],t['block'];assert key not in lookup;lookup[key]=t
 with np.load(src/t['witness']) as z:
  assert sha(z['W'])==t['W_sha256']==certs[key[:2]][2]
  np.testing.assert_allclose(z['residual_norm_timed'],z['step_tolratio']*z['residual_denominator'],rtol=1e-14)
  assert float(z['residual_denominator'])==float(z['initial_scale'])*1e-9+1e-300
  np.testing.assert_allclose(z['residual_norm_timed'],certs[key[:2]][0],rtol=1e-7,atol=1e-12)
  np.testing.assert_allclose(z['step_gn'],certs[key[:2]][1],rtol=1e-7,atol=1e-12)
  np.testing.assert_array_equal(z['step_it'],certs[key[:2]][3])
  assert z['nfail']==0 and int(z['it_sum'])==t['fine_updates']
  np.testing.assert_allclose(float(z['residual_norm_timed'][-1]**2),t['final_loss'],rtol=1e-14)
assert set(lookup)=={(a,c,b) for a in arms for c in range(6) for b in range(7)}
summary=dict(job_id=x['job_id'],source_sha256=hashlib.sha256((src/'result.json').read_bytes()).hexdigest(),arms={},qualification=x['qualification'],tight_comparator=x['tight_records'],excluded_repetitions=0,loss_tolerance=x['loss_agreement_tolerance'],field_tolerance=x['field_agreement_tolerance'],scope='Final losses and field gaps from the actual timed invocations; losses use own accepted histories. Tight comparator is a fine ROM, not physical truth.')
for arm in arms:
 ms=np.array([[lookup[arm,c,b]['seconds']*1000 for b in range(7)] for c in range(6)])
 loss=np.array([[lookup[arm,c,b]['final_loss'] for b in range(7)] for c in range(6)])
 base_ms=np.array([[lookup['direct',c,b]['seconds']*1000 for b in range(7)] for c in range(6)])
 base_loss=np.array([[lookup['direct',c,b]['final_loss'] for b in range(7)] for c in range(6)])
 ratios=ms/base_ms;rel=np.abs(loss-base_loss)/base_loss
 gap=np.array([[lookup[arm,c,b]['field_gap_tight'] for b in range(7)] for c in range(6)])
 q1,q3=np.quantile(ms,[.25,.75],axis=1);iqr=q3-q1
 entry=dict(**aggregate[arm],ms=ms.tolist(),loss=loss.tolist(),paired_runtime_ratio=ratios.tolist(),median_ratio=float(np.median(np.median(ratios,axis=1))),worst_case_median_ratio=float(np.max(np.median(ratios,axis=1))),slower_pairs=int(np.sum(ratios>1)),
  final_loss_relative_difference=rel.tolist(),worst_final_loss_relative_difference=float(rel.max()),same_loss_passed=bool(np.max(rel)<=x['loss_agreement_tolerance']),field_gap_tight=gap.tolist(),worst_field_gap_tight=float(gap.max()),tight_field_agreement_passed=bool(np.max(gap)<=x['field_agreement_tolerance']),timing_outliers=int(np.sum((ms<(q1-1.5*iqr)[:,None])|(ms>(q3+1.5*iqr)[:,None]))))
 if arm!='direct':
  gate=x['gates'][arm];np.testing.assert_allclose(ratios,gate['per_case_ratios'],rtol=1e-13)
  assert gate['passed']==bool(entry['median_ratio']<=.85 and entry['worst_case_median_ratio']<=1.10)
  entry['cost_gate_passed']=gate['passed']
 summary['arms'][arm]=entry
out=HERE.parent/'plots/inexact';out.mkdir(parents=True,exist_ok=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
labels=['Direct K=64','K16 / 1,024 pts: 1 update → K64','K16 / 1,024 pts: 2 updates → K64']
for i,(arm,color,marker,label) in enumerate(zip(arms,['#2265a8','#d87916','#21835b'],['o','s','^'],labels)):
 v=summary['arms'][arm];pos=np.arange(1,7)+(i-1)*.23;ms=np.array(v['ms']);med=np.median(ms,axis=1);loss=np.array(v['loss'])
 ax.bar(pos,med,width=.22,color=color,label=label)
 ax.errorbar(pos,med,yerr=[med-ms.min(1),ms.max(1)-med],fmt='none',capsize=2,ecolor='#444444')
 bx.scatter(pos,np.median(loss,axis=1),s=65,c=color,marker=marker)
ax.set(title='Complete-solve runtime',ylabel='Milliseconds (median; whiskers min–max)',xlabel='Development case',xticks=range(1,7))
ax.set_ylim(0,max(max(np.array(summary['arms'][a]['ms']).max() for a in arms)*1.25,130));ax.legend(frameon=False,fontsize=9)
bx.set(title='Returned fine loss after K=64 refinement',ylabel=r'$\|r_{50}\|_2^2$ at t = 0.25 (log scale)',xlabel='Development case',xticks=range(1,7),yscale='log')
bx.grid(axis='y',alpha=.15,which='both')
for c in range(6):
 vals=[np.median(summary['arms'][a]['loss'][c]) for a in arms]
 bx.text(c+1,max(vals)*1.25,f'≈ {np.median(vals):.4g}',ha='center',fontsize=9)
bx.set_ylim(.00025,.065)
fig.suptitle('1,024-point coarse proposals → original fine solve: time and loss',fontsize=17)
fig.supxlabel(f"Job {x['job_id']} · 6 cases × 7 paired repetitions · same H200 GPU · actual timed-output losses\nFine target unchanged: K64, 16,384 quadrature points, 256 test modes. Coarse: K16, 1,024 points, 64 modes.\nLoss is the weighted equation mismatch on each trajectory own history, not physical-reference error. Prior coarse-quadrature discrepancy remains.",fontsize=10)
fig.savefig(out/'time-and-loss.png',dpi=170);fig.savefig(out/'time-and-loss.pdf')
print(json.dumps({a:{k:v for k,v in d.items() if not isinstance(v,list)} for a,d in summary['arms'].items()},indent=2))
