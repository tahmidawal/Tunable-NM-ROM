"""Measured same-state mechanisms, with all qualified probes shown."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();s=json.loads((a.run/'summary.json').read_text())
    rows=[];case_geometry=[];excluded=0
    for c in s['cases']:
        for setting in s['config']['test_settings']:
            p=a.run/f"case_{c['index']:03d}"/f"reset_m{setting['rows']}_dt{setting['dt']:g}.json"
            if not p.exists():continue
            records=json.loads(p.read_text());geo=[]
            for q in records:
                if q['span_result']['reason']!='stationarity':excluded+=1;continue
                rows.append([c['index'],setting['rows'],setting['dt'],q['span_result']['residual_norm']/q['head_seed_residual_norm'],q['span_physical_error']/q['head_physical_error']])
                if 'shared_state_geometry' in q:
                    g=q['shared_state_geometry'];geo.append([g['least_visible_span_outside_head_fraction'][0],g['actual_step_linearized_outside_head_fraction']])
            if geo:case_geometry.append([c['index'],setting['rows'],setting['dt'],*np.median(geo,axis=0)])
    data=np.asarray(rows);gdata=np.asarray(case_geometry)
    out=a.run/'figures';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'mechanism_plot_data.npz',probes=data,case_setting_medians=gdata)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    colors=['#2b8e88','#ce7e29','#755a9f']
    for color,setting in zip(colors,s['config']['test_settings']):
        selected=data[(data[:,1]==setting['rows'])&(data[:,2]==setting['dt'])]
        axes[0].scatter(selected[:,3],selected[:,4],s=13,alpha=.45,color=color,label=f"{setting['rows']} rows, dt={setting['dt']:g}",edgecolors='none')
    axes[0].axvline(1,color='#555555',ls='--',lw=1);axes[0].axhline(1,color='#555555',ls='--',lw=1)
    axes[0].set(xlabel='Span / head weak residual norm',ylabel='Span / head physical error',title='Same previous state: what lowering the objective does')
    axes[0].set_xscale('log');axes[0].set_yscale('log');axes[0].yaxis.set_major_locator(mticker.LogLocator(base=2));axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:g}'));axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    if len(gdata):
        rng=np.random.default_rng(109)
        for i,label in enumerate(['Least-visible span direction','Actual span departure']):
            values=100*gdata[:,i+3]
            axes[1].scatter(np.full(len(values),i)+rng.uniform(-.11,.11,len(values)),values,s=18,color=('#247d78','#c27824')[i],alpha=.6)
            axes[1].plot([i-.17,i+.17],[np.median(values)]*2,color='black',lw=2)
        axes[1].set_xticks([0,1],['Least-visible\nspan direction','Actual span departure\n(tangent-linearized)'])
        axes[1].set(ylabel='Physical norm outside head + frame tangent (%)',title='How much the head excludes each direction')
        axes[1].grid(axis='y',alpha=.15)
    else:axes[1].text(.5,.5,'Same-state geometry not saved in this pilot',ha='center',va='center');axes[1].set_axis_off()
    status='Numerically qualified' if s['qualification'].get('passed',False) else 'PROVISIONAL — inspect qualification failures'
    fig.suptitle(f'Measured development mechanism — {status}',fontsize=13)
    fig.text(.5,.015,f'Left: all {len(rows)} stationary step probes; {excluded} unqualified probes excluded explicitly. Right: each point is one case/setting median.\nLocal tangent diagnostics do not establish global manifold distance or a uniform stability bound.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.09,1,.94]);fig.savefig(out/'same_state_mechanism.png',dpi=170);fig.savefig(out/'same_state_mechanism.pdf')
    print(out/'same_state_mechanism.png')
if __name__=='__main__':main()
