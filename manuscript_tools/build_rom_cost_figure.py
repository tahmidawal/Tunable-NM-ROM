"""Re-render the saved cost figure with ROMs/classical solvers only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'manuscript_tools/evidence/F1_tunability_data.json').read_text())
plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),layout='constrained')
limits={'poisson':(.0002,80),'heat':(.006,20),'burgers':(.015,80),'ns':(.006,400)}
for ax,(key,q) in zip(axes.flat,data.items()):
    ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(*limits[key])
    ax.set_title(q['title'],loc='left')
    for name,label,color,marker in [('span','ROM span','#2463a2','o'),('head','ROM head','#cd6432','s')]:
        points=q.get(name,[])
        if points:ax.plot([p[1] for p in points],[p[2] for p in points],'-'+marker,color=color,label=label,markersize=4)
    points=q.get('fom',[])
    ax.scatter([p[0] for p in points],[p[1] for p in points],marker='D',color='#777777',s=12,label='Classical solver')
    points=q.get('spec',[])
    if points:ax.scatter([p[0] for p in points],[p[1] for p in points],marker='P',color='#8653a5',s=22,label='Spectral / transform')
    if 'spec_exact' in q:
        x,_=q['spec_exact'];y=limits[key][0]*1.4
        ax.scatter([x],[y],marker='v',color='#8653a5',s=25,label='Spectral / transform')
        ax.annotate('exact transform',xy=(x,y),xytext=(5,6),textcoords='offset points',fontsize=7)
    unstable=q.get('fom_unstable',[])
    if unstable:
        y=limits[key][1]/1.4
        ax.scatter([p[0] for p in unstable],[y]*len(unstable),marker='x',color='#777777',s=20)
        ax.text(.98,.97,'unstable solver rungs',transform=ax.transAxes,ha='right',va='top',fontsize=7)
    ax.grid(which='major',color='#dddddd',linewidth=.5)
    ax.set_xlabel('Online query time (GPU ms)');ax.set_ylabel('Worst-case relative error (%)')
    ax.legend(frameon=False,fontsize=6.5,loc='best')
out=ROOT/'paper_revision/figs/F1_tunability.pdf'
fig.savefig(out,metadata={'CreationDate':None,'ModDate':None})
fig.savefig(ROOT/'manuscript_tools/checks/F1_accuracy_scope.png',dpi=160)
print(out)
