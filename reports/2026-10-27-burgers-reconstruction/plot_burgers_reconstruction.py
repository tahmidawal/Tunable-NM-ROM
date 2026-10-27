"""Plot the primary same-input comparisons from the generated report's JSON."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    p=argparse.ArgumentParser()
    p.add_argument('report_json',type=Path)
    a=p.parse_args()
    d=json.loads(a.report_json.read_text())
    colors={'win':'#1976a3','tie':'#8c959f','loss':'#c75b28'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'burgers-same-sample-screen'})
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.8),layout='constrained')
    plotted=[]
    for dim,ax in zip((2,3),axes):
        c=next(x for x in d['primary'] if x['dimension']==dim)
        rr=[r for r in d['selected_cases'] if r['dimension']==dim and r['Lc']==c['Lc'] and r['source']=='B']
        bank={r['case']:r for r in rr if r['config']==c.get('method')}
        ordinary={r['case']:r for r in rr if r['config']==c.get('baseline')}
        pts=[]
        for pair in c.get('pairs',[]):
            if not pair['finite']:
                continue
            x,y=100*ordinary[pair['case']]['error'],100*bank[pair['case']]['error']
            kind='win' if pair['win'] else 'loss' if pair['loss'] else 'tie'
            pts.append(dict(case=pair['case'],classical_error_percent=x,bank_error_percent=y,kind=kind))
        if pts:
            lo=min(min(q['classical_error_percent'],q['bank_error_percent']) for q in pts)
            hi=max(max(q['classical_error_percent'],q['bank_error_percent']) for q in pts)
            if lo>0:
                ax.set_xscale('log');ax.set_yscale('log');bounds=(lo/1.35,hi*1.35)
            else:
                bounds=(0,hi*1.12 if hi else 1)
            ax.plot(bounds,bounds,color='#495057',lw=1,ls='--',zorder=1)
            for kind in colors:
                qq=[q for q in pts if q['kind']==kind]
                ax.scatter([q['classical_error_percent'] for q in qq],[q['bank_error_percent'] for q in qq],
                           s=40,c=colors[kind],edgecolors='white',linewidths=.45,alpha=.8,zorder=3)
            ax.set_xlim(bounds);ax.set_ylim(bounds);ax.set_aspect('equal',adjustable='box')
            fine=next(iter(bank.values()))['Lf']
            ax.set_title(f"{dim}D · {c['Lc']} → {fine} intervals per axis")
            ax.text(.035,.965,f"{len(pts)}/{c['cases']} cases plotted\n{c['failed_pairs']} failed pairs",
                    transform=ax.transAxes,va='top',fontsize=9,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
            ax.text(.97,.04,'Below diagonal: lower decoder error',transform=ax.transAxes,
                    ha='right',fontsize=8,color='#495057')
        else:
            ax.text(.5,.5,'Frozen comparison unavailable',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(f'{dim}D')
        ax.set_xlabel('Selected classical reconstruction error (%)')
        ax.set_ylabel('Learned decoder error (%)')
        ax.grid(True,which='major',color='#e9ecef',lw=.7)
        plotted.append(dict(dimension=dim,coarse_intervals=c['Lc'],method=c.get('method'),
                            baseline=c.get('baseline'),declared_cases=c['cases'],points=pts))
    handles=[Line2D([],[],marker='o',linestyle='',color=colors[k],label=label)
             for k,label in [('win','Decoder lower'),('tie','Unresolved difference'),('loss','Decoder higher')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Actual coarse fields: the same samples, frozen reconstruction settings',fontsize=12)
    stem=a.report_json.with_suffix('')
    fig.savefig(a.report_json.with_suffix('.png'),dpi=220,metadata={'Description':'Case maximum relative L2 errors; conditional reference-resolution classification.'})
    fig.savefig(a.report_json.with_suffix('.svg'),metadata={'Date':None})
    plt.close(fig)
    stem.with_name(stem.name+'.figure.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(a.report_json.read_bytes()).hexdigest(),
        classification='Paired difference exceeds twice the case reference uncertainty; conditional numerical resolution, not statistical significance.',
        panels=plotted),indent=2,allow_nan=False)+'\n')
    print(a.report_json.with_suffix('.png'))


if __name__=='__main__':main()
