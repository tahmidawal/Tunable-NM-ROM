"""Compare observed richer-test outcomes with the same two cases and dt in admitted dev16."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('baseline',type=Path);ap.add_argument('enrichment',type=Path);a=ap.parse_args()
    bs=json.loads((a.baseline/'summary.json').read_text());es=json.loads((a.enrichment/'summary.json').read_text());ids=[c['index'] for c in es['cases']]
    records=[];geometry=[]
    for root,s in [(a.baseline,bs),(a.enrichment,es)]:
        for c in s['cases']:
            if c['index'] not in ids:continue
            for r in c['runs']:
                if r['dt']!=.01 or r['gtol']!=min(s['config']['solver']['gtols']):continue
                records.append({'case':c['index'],'rows':r['rows'],'arm':r['arm'],'error':r['evolved_worst'],'source_job':s['job_id']})
            for setting in s['config']['test_settings']:
                if setting['dt']!=.01:continue
                probes=json.loads((root/f"case_{c['index']:03d}"/f"reset_m{setting['rows']}_dt0.01.json").read_text())
                for q in probes:
                    g=q.get('shared_state_geometry')
                    if g:geometry.append({'case':c['index'],'rows':setting['rows'],'head_smin':g['head_physical_singular_values'][-1],'span_smin':g['span_physical_singular_values'][-1],'span_outside':g['least_visible_span_outside_head_fraction'][0]})
    out=a.enrichment/'figures';out.mkdir(exist_ok=True)
    measured={'case_indices':ids,'common_dt':.01,'records':records,'same_state_geometry':geometry,'scope':'Two fixed development cases only; finite test-space robustness, not convergence or a uniform stability result.'}
    (a.enrichment/'enrichment_comparison.json').write_text(json.dumps(measured,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.5));ms=sorted(set(x['rows'] for x in records))
    colors={'head_common':'#137c73','span_common':'#c76b16','span_native':'#7c77a5'}
    for arm,label in [('head_common','Head, common initial state'),('span_common','Span, common initial state'),('span_native','Span, native projection')]:
        vals=np.asarray([[x['error']*100 for x in records if x['rows']==m and x['arm']==arm] for m in ms])
        axes[0].plot(ms,np.median(vals,axis=1),'-o',color=colors[arm],label=label,lw=2,ms=4)
        axes[0].fill_between(ms,vals.min(1),vals.max(1),color=colors[arm],alpha=.12)
    ratios=[]
    for m in ms:
        rr=[]
        for c in ids:
            hh=next(x['error'] for x in records if x['rows']==m and x['case']==c and x['arm']=='head_common')
            ss=next(x['error'] for x in records if x['rows']==m and x['case']==c and x['arm']=='span_common')
            rr.append(ss/hh)
        ratios.append(rr)
    ratios=np.asarray(ratios);axes[1].plot(ms,np.median(ratios,axis=1),'-o',color='#3a5369',lw=2,ms=4);axes[1].fill_between(ms,ratios.min(1),ratios.max(1),color='#3a5369',alpha=.15);axes[1].axhline(1,ls='--',color='#555555',lw=1)
    for key,color,label in [('head_smin','#137c73','Head tangent'),('span_smin','#c76b16','Span tangent')]:
        vals=[np.asarray([x[key] for x in geometry if x['rows']==m]) for m in ms]
        axes[2].plot(ms,[np.median(v) for v in vals],'-o',color=color,label=label,lw=2,ms=4)
        axes[2].fill_between(ms,[v.min() for v in vals],[v.max() for v in vals],color=color,alpha=.12)
    axes[0].set(title='Matched physical error',ylabel='Maximum evolved error (% initial norm)');axes[0].legend(fontsize=7)
    axes[1].set(title='Does the span catch up?',ylabel='Span / head physical error')
    axes[2].set(title='Local physical residual visibility',ylabel='Smallest physically normalized singular value');axes[2].legend(fontsize=8);axes[2].set_yscale('log')
    for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks(ms,[str(m) for m in ms]);ax.set_xlabel('Solved scalar weak rows');ax.grid(alpha=.18)
    qualified=es['qualification'].get('passed',False) and bs['qualification'].get('passed',False)
    fig.suptitle('Measured richer-test sensitivity: same two development cases, dt=0.01\n'+('Numerically qualified' if qualified else 'PROVISIONAL — inspect qualification'),fontsize=12)
    fig.text(.5,.01,'Lines: pointwise medians; shading: observed range, not a confidence interval. Right panel uses local same-state tangents.\nThis finite test-space check does not establish convergence to a full residual or a uniform stability bound.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.09,1,.92]);fig.savefig(out/'richer_test_sensitivity.png',dpi=170);fig.savefig(out/'richer_test_sensitivity.pdf');print(out/'richer_test_sensitivity.png')
if __name__=='__main__':main()
