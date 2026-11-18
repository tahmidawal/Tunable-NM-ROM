"""Measured-only figures from complete or provisional experiment JSON."""
from pathlib import Path
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'head_common':'#137c73','span_common':'#c76b16','span_native':'#7c77a5'}
LABELS={'head_common':'Head, shared initial state','span_common':'Span, shared initial state','span_native':'Span, native projection'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    report=json.loads((args.run/'summary.json').read_text());figdir=args.run/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    cfg=report['config'];gtol=min(cfg['solver']['gtols']);cases=report['cases'];settings=cfg['test_settings']
    qualified=bool(report.get('qualification',{}).get('passed',False));status='Numerically qualified' if qualified else 'PROVISIONAL — numerical qualification incomplete or failed'
    fig,axes=plt.subplots(2,len(settings),figsize=(13,7),squeeze=False)
    for col,setting in enumerate(settings):
        rows,dt=setting['rows'],setting['dt'];times=np.arange(round(cfg['horizon']/dt)+1)*dt
        for arm in cfg['arms']:
            selected=[(c,next((r for r in c['runs'] if r['arm']==arm and r['rows']==rows and r['dt']==dt and r['gtol']==gtol),None)) for c in cases]
            selected=[(c,r) for c,r in selected if r is not None]
            if not selected:continue
            errors=np.asarray([[m['relative_error'] for m in r['metrics']] for _,r in selected])*100
            ax=axes[0,col];color=COLORS[arm];style='-' if arm!='span_native' else '--'
            ax.plot(times,np.median(errors,axis=0),color=color,ls=style,label=LABELS[arm],lw=2)
            ax.fill_between(times,errors.min(0),errors.max(0),color=color,alpha=.11)
            expanded=[]
            for c,r in selected:
                rec=json.loads((args.run/f"case_{c['index']:03d}"/r['records_file']).read_text())['records']
                expanded.append([x['expanded_only_residual_norm']/c['norm_u0'] for x in rec])
            arr=np.asarray(expanded)
            axes[1,col].semilogy(times[1:],np.median(arr,axis=0),color=color,ls=style,lw=2)
            axes[1,col].fill_between(times[1:],arr.min(0),arr.max(0),color=color,alpha=.11)
        axes[0,col].set_title(f'{rows} solved weak rows; dt={dt:g}')
        axes[0,col].set_ylabel('Physical velocity error (% initial norm)')
        axes[1,col].set_ylabel('Extra-row weak residual / initial norm')
        axes[1,col].set_xlabel('Time')
        for row in range(2): axes[row,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'Matched head/span development trajectories — {len(cases)} cases\n{status}',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(figdir/'matched_trajectories.png');fig.savefig(figdir/'matched_trajectories.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    setting=settings[-1];rows,dt=setting['rows'],setting['dt'];times=np.arange(round(cfg['horizon']/dt)+1)*dt
    for arm in cfg['arms']:
        selected=[(c,next((r for r in c['runs'] if r['arm']==arm and r['rows']==rows and r['dt']==dt and r['gtol']==gtol),None)) for c in cases]
        selected=[(c,r) for c,r in selected if r]
        if not selected:continue
        color=COLORS[arm]
        phase=np.asarray([[np.linalg.norm(m['phase_displacement']) for m in r['metrics']] for _,r in selected])
        shape=np.asarray([[m['shape_aligned_relative_error']*100 for m in r['metrics']] for _,r in selected])
        for ax,arr in zip(axes[:2],(phase,shape)):
            ax.plot(times,np.median(arr,0),color=color,lw=2,label=LABELS[arm]);ax.fill_between(times,arr.min(0),arr.max(0),color=color,alpha=.12)
        if arm!='head_common':
            dist=np.asarray([r['head_fit_distance'] for _,r in selected])*100
            axes[2].plot(times,np.median(dist,0),color=color,lw=2,label=LABELS[arm]);axes[2].fill_between(times,dist.min(0),dist.max(0),color=color,alpha=.12)
    for ax,title,ylabel in zip(axes,('Physical phase drift','Shape error after centroid alignment','Best-found distance to head, fixed reached frame'),('Centroid displacement (periodic domain)','Error (% initial norm)','Distance (% initial norm)')):
        ax.set(title=title,xlabel='Time',ylabel=ylabel);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle(f'{rows} rows; dt={dt:g} — {status}',fontsize=11);fig.tight_layout();fig.savefig(figdir/'drift_diagnostics.png');fig.savefig(figdir/'drift_diagnostics.pdf')
    print(figdir)
if __name__=='__main__':main()
