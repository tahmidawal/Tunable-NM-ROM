"""Deterministic publication plots from saved scalar records; no simulation imports."""
from pathlib import Path
import argparse,hashlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ARMS=('head_common','span_common')
LABELS={'head_common':'Nonlinear head','span_common':'Common-IC span'}
COLORS={'head_common':'#0072B2','span_common':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'svg.hashsalt':'jcp-phase-confirm-20261119','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def save(fig,path):
 fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight',metadata={'Software':'saved-record phase-shape builder'})
 fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',metadata={'Date':None})
 fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
 plt.close(fig)

def plot_one(summary_path,kind,out,replay_path=None):
 s=json.loads(summary_path.read_text());cases=s['cases'];n=len(cases)
 series={arm:{'field':[],'shape':[],'phase':[],'pred_concentration':[],'truth_concentration':[]} for arm in ARMS}
 indices=[];csvrows=[]
 replay=json.loads(replay_path.read_text()) if replay_path and replay_path.exists() else None
 reliability_known=True
 for case in cases:
  ci=case['index'];indices.append(ci)
  for arm in ARMS:
   row=next(r for r in case['runs'] if r['arm']==arm and r['rows']==584 and r['dt']==.01 and r['gtol']==1e-9)
   met=row['metrics'];times=np.arange(len(met))*.01
   rel=None
   if replay:
    rel=next(q for q in replay['cases'] if q['index']==ci)
    rel=next(q for q in rel['runs'] if q['arm']==arm)['metrics']
   for t,m in enumerate(met):
    conc=(rel[t] if rel else m)
    if 'pred_centroid_concentration' not in conc:reliability_known=False
    csvrows.append({'cohort':kind,'case':ci,'arm':arm,'time':float(times[t]),'field_error':m['relative_error'],'centroid_aligned_error':m['shape_aligned_relative_error'],'phase_norm_domain_fraction':float(np.linalg.norm(m['phase_displacement'])),'centroid_concentration_min_pred':min(conc['pred_centroid_concentration']) if 'pred_centroid_concentration' in conc else None,'centroid_concentration_min_truth':min(conc['truth_centroid_concentration']) if 'truth_centroid_concentration' in conc else None})
   series[arm]['field'].append([m['relative_error'] for m in met]);series[arm]['shape'].append([m['shape_aligned_relative_error'] for m in met]);series[arm]['phase'].append([np.linalg.norm(m['phase_displacement']) for m in met])
   cmet=rel if rel else met
   if all('pred_centroid_concentration' in m for m in cmet):
    series[arm]['pred_concentration'].append([m['pred_centroid_concentration'] for m in cmet]);series[arm]['truth_concentration'].append([m['truth_centroid_concentration'] for m in cmet])
 for arm in ARMS:
  for key in series[arm]:series[arm][key]=np.asarray(series[arm][key],dtype=float)
 fig,axes=plt.subplots(1,3,figsize=(12.0,3.75),layout='constrained')
 specs=[('field','(a) Ordinary field error','Error / initial-field norm (%)'),('phase','(b) Physical position error','Centroid displacement\n(% of domain length)'),('shape','(c) Centroid-aligned field error','Error / initial-field norm (%)')]
 for ax,(key,title,ylabel) in zip(axes,specs):
  for arm in ARMS:
   arr=100*series[arm][key]
   ax.fill_between(times,arr.min(axis=0),arr.max(axis=0),color=COLORS[arm],alpha=.12,lw=0)
   ax.plot(times,np.median(arr,axis=0),color=COLORS[arm],lw=2.2,label=LABELS[arm])
  ax.set(title=title,xlabel='Time',ylabel=ylabel,xlim=(0,times[-1]));ax.grid(axis='y',color='#dddddd',lw=.6);ax.set_ylim(bottom=0)
 commonmax=1.06*max(100*series[a][key].max() for a in ARMS for key in ('field','shape'))
 axes[0].set_ylim(0,commonmax);axes[2].set_ylim(0,commonmax);axes[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
 axes[0].legend(loc='upper left',frameon=False)
 cohort='Development' if kind=='development' else 'Fresh frozen-choice confirmation'
 fig.suptitle(f'{cohort}: position and aligned shape in matched dynamics\n{n} cases · R=64, k=8, M=584, Δt=0.01 · line: median; shade: full case range',fontsize=12)
 save(fig,out/f'phase_shape_{kind}')
 fig,axes=plt.subplots(4,4,figsize=(11.5,10),sharex=True,sharey=True,layout='constrained')
 for ax,ci in zip(axes.flat,range(n)):
  for arm in ARMS:
   ax.plot(times,100*series[arm]['field'][ci],color=COLORS[arm],lw=1.4)
   ax.plot(times,100*series[arm]['shape'][ci],color=COLORS[arm],lw=1.2,ls='--')
  ax.set_title(f'Case {indices[ci]:02d}');ax.set_ylim(0,commonmax);ax.grid(axis='y',alpha=.25)
 for ax in axes[-1]:ax.set_xlabel('Time')
 for ax in axes[:,0]:ax.set_ylabel('Initial-normalized error (%)')
 fig.suptitle(f'{cohort}: every paired case\nBlue: head; orange: common-IC span · solid: ordinary; dashed: centroid-aligned',fontsize=13)
 save(fig,out/f'phase_shape_all_cases_{kind}')
 maxima={a:{key:series[a][key][:,1:].max(axis=1) for key in ('field','shape','phase')} for a in ARMS}
 stats={'cohort':kind,'cases':n,'case_indices':indices,'metric_time_scope':'maximum over evolved times, t>0','field_denominator':'original physical initial-field norm','phase_units':'fraction of unit domain length; Euclidean norm of three wrapped coordinates','alignment':'Fourier translation by negative physical-centroid difference; reference-assisted diagnosis, not optimal registration','qualification':s.get('qualification'), 'parameters':[c['parameters'] for c in cases],'parameter_sha256':s.get('parameter_sha256'),'by_arm':{},'reliability_known':reliability_known,'centroid_replay_qualification':replay.get('qualified') if replay else None,'sources':{str(summary_path):sha(summary_path)}}
 if replay:stats['sources'][str(replay_path)]=sha(replay_path)
 for arm in ARMS:
  stats['by_arm'][arm]={key:{'casewise_maxima':vals.tolist(),'median_of_casewise_maxima':float(np.median(vals)),'worst':float(vals.max())} for key,vals in maxima[arm].items()}
  if series[arm]['pred_concentration'].size:
   for key in ('pred_concentration','truth_concentration'):
    arr=series[arm][key];stats['by_arm'][arm][key]={'minimum':float(arr.min()),'below_0p1':int((arr<.1).sum()),'coordinate_time_count':int(arr.size)}
 stats['paired']={key:{'span_over_head_max_error_ratios':(maxima['span_common'][key]/maxima['head_common'][key]).tolist(),'median_ratio':float(np.median(maxima['span_common'][key]/maxima['head_common'][key])),'head_lower_count':int((maxima['head_common'][key]<maxima['span_common'][key]).sum())} for key in ('field','shape','phase')}
 if reliability_known:
  fig,ax=plt.subplots(figsize=(6.2,3.4),layout='constrained')
  for arm in ARMS:
   arr=series[arm]['pred_concentration'];mins=arr.min(axis=(0,2));ax.plot(times,mins,color=COLORS[arm],label=LABELS[arm],lw=2)
  truth=series['head_common']['truth_concentration'].min(axis=(0,2));ax.plot(times,truth,color='#333333',ls=':',lw=2,label='Reference')
  ax.axhline(.1,color='#888888',lw=1,ls='--',label='Declared ambiguity flag')
  ax.set(xlabel='Time',ylabel='Minimum concentration across cases and coordinates',ylim=(0,1),xlim=(0,times[-1]),title=f'{cohort}: circular-centroid reliability');ax.legend(frameon=False,ncol=2)
  save(fig,out/f'centroid_reliability_{kind}')
 with (out/f'phase_shape_{kind}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 dump(out/f'phase_shape_{kind}.json',stats)
 text=f'''# {cohort}: physical position and centroid-aligned shape\n\nMeasured diagnostics from saved accepted trajectories. Qualification and sources are retained in the adjacent JSON; the three-panel plot shows median histories and full case ranges.\n\nBoth field errors use the original physical initial-field norm. Phase is the Euclidean norm of wrapped energy-weighted physical-centroid differences, shown as percent domain length. Alignment uses reference information and is diagnostic only; it is not an online correction or a globally optimal translation. The head and span start from the same reconstructed field and have the same three frame unknowns.\n\n| Diagnostic, maximum over evolved times | Head median | Span median | Head worst | Span worst | Median paired span/head ratio | Head lower cases |\n|---|---:|---:|---:|---:|---:|---:|\n'''
 for key,label in [('field','Ordinary field error (%)'),('shape','Centroid-aligned field error (%)'),('phase','Physical displacement (% domain length)')]:
  h=stats['by_arm']['head_common'][key];sp=stats['by_arm']['span_common'][key];pa=stats['paired'][key]
  text+=f"| {label} | {100*h['median_of_casewise_maxima']:.8g} | {100*sp['median_of_casewise_maxima']:.8g} | {100*h['worst']:.8g} | {100*sp['worst']:.8g} | {pa['median_ratio']:.6g} | {pa['head_lower_count']}/{n} |\n"
 if reliability_known:
  cmin=min(stats['by_arm'][arm][key]['minimum'] for arm in ARMS for key in ('pred_concentration','truth_concentration'));nbad=sum(stats['by_arm'][arm]['pred_concentration']['below_0p1'] for arm in ARMS)+stats['by_arm']['head_common']['truth_concentration']['below_0p1']
  text+=f'\nThe minimum circular-moment concentration is {cmin:.8g}; {nbad} coordinate/time entries fall below the prospectively declared descriptive flag. This assesses centroid definition, not uniqueness of the underlying physical structure.\n'
 else:text+='\nCircular-moment concentration is pending; phase interpretation remains provisional until the GPU replay is qualified.\n'
 text+='\nThe aligned-error comparison retains its own measured advantage or disadvantage. It cannot be obtained by subtracting phase error from ordinary error, and cannot prove a unique deformation mechanism. No speed or uniform-stability conclusion follows from this plot.\n\n## Glossary\n\nA head is a nonlinear restriction on spatial coefficients; a span permits arbitrary coefficients. Common IC means identical initial reconstructed fields. Median is the middle case statistic; worst is the largest. A paired ratio divides span by head on the same case before summarizing. Circular concentration measures how well a periodic centroid is defined. The full case range is variability across physical inputs, not a statistical confidence interval.\n'
 (out/f'caption_{kind}.md').write_text(text)
 return stats

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'plots');ap.add_argument('--run',type=Path,default=HERE/'runs/c01/output');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 stats=[plot_one(HERE/'saved_development/summary.json','development',a.out,a.run/'centroid_replay.json')]
 if (a.run/'summary.json').exists():stats.append(plot_one(a.run/'summary.json','fresh',a.out))
 dump(a.out/'plot_manifest.json',{'builder_sha256':sha(__file__),'data_only':True,'cohorts':[s['cohort'] for s in stats],'sources':{k:v for s in stats for k,v in s['sources'].items()},'outputs':{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file() and p.name!='plot_manifest.json'}})
 print(json.dumps({s['cohort']:s['paired'] for s in stats},indent=2))
if __name__=='__main__':main()
