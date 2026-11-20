"""Build honest manuscript panels from previously verified saved scalars and references."""
from pathlib import Path
import json,hashlib,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
L=Path(__file__).resolve().parent
ROOT=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
OLD=ROOT/'worktrees/2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility'
D=L/'figures/saved';D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(OLD/'checks/width-results/analysis.json');table=read(OLD/'checks/width-results/table.json')
assert a['all_saved_array_checks_passed'] and a['all_arms_numerically_eligible']
arms=[(128,2048),(256,2048),(256,4096),(512,2048),(512,8192)]
times=np.arange(51)*.005
metrics=['error','gradient_error','upwind_advection_error','laplacian_error']
labels=['Field error / initial reference norm (%)','Gradient error / current reference gradient (%)','Upwind-advection error / current reference advection (%)','Laplacian error / current reference Laplacian (%)']
colors={(128,2048):'#59646c',(256,2048):'#278696',(256,4096):'#278696',(512,2048):'#a43d69',(512,8192):'#a43d69'}
styles={(128,2048):'-',(256,2048):'-',(256,4096):'--',(512,2048):'-',(512,8192):'--'}
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'jcp-width-saved-20261119','pdf.fonttype':42,'savefig.bbox':'tight'})
def save(fig,stem):
 for ext in ['png','svg','pdf']:
  md={'Creator':'JCP saved-result plot builder'}
  if ext=='pdf':md.update(CreationDate=None,ModDate=None)
  if ext=='svg':md['Date']=None
  fig.savefig(D/f'{stem}.{ext}',dpi=210,metadata=md)
  if ext=='svg':
   path=D/f'{stem}.svg';path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
 plt.close(fig)
def series(r,m,k):return np.asarray([a['cases'][f'R{r}-case{i}']['arms'][str(m)][k] for i in range(6)])
plotdata={'scope':a['scope'],'times':times.tolist(),'normalization':dict(error='original initial-field norm',gradient_error='instantaneous reference gradient norm',upwind_advection_error='instantaneous reference advection norm',laplacian_error='instantaneous reference Laplacian norm'),'arms':{},'source_sha256':sha(OLD/'checks/width-results/analysis.json')}
for r,m in arms:plotdata['arms'][f'R{r}-M{m}']={k:series(r,m,k).tolist() for k in metrics}
fig,axs=plt.subplots(2,2,figsize=(12,7.4),sharex=True)
shown=[(128,2048),(512,2048),(512,8192)]
for ax,k,label in zip(axs.flat,metrics,labels):
 for r,m in shown:
  v=100*series(r,m,k);ax.fill_between(times,v.min(0),v.max(0),color=colors[r,m],alpha=.08)
  ax.plot(times,np.median(v,axis=0),color=colors[r,m],ls=styles[r,m],lw=2,label=f'R′={r}, M={m}')
 ax.set_yscale('log');ax.set_title(label);ax.grid(alpha=.2,which='both');ax.set_xlabel('Time')
 if k=='laplacian_error':ax.axhline(100,color='#9c6728',lw=1,ls=':',label='100% derivative error')
handles,leg=axs[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',ncol=3,bbox_to_anchor=(.5,.96),frameon=False)
fig.suptitle('Burgers: small field error does not ensure accurate derivatives',y=1.01,fontsize=15)
fig.text(.5,.015,'Median across all six development cases; faint bands show the full case range. Discrete operators, same-grid reference.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.91));save(fig,'derivative-error-time')
fig,axs=plt.subplots(2,2,figsize=(12,7.6))
for ax,k,label in zip(axs.flat,metrics,labels):
 for x,(r,m) in enumerate(arms):
  maxima=100*series(r,m,k).max(1);ax.scatter(x+np.linspace(-.13,.13,6),maxima,s=24,color=colors[r,m],alpha=.8)
  ax.plot([x-.23,x+.23],[np.median(maxima)]*2,color='black',lw=2)
 ax.set_xticks(range(len(arms)),[f'{r}\n{m}' for r,m in arms]);ax.set_xlabel('Active width R′ / weak tests M');ax.set_yscale('log');ax.set_title(label);ax.grid(axis='y',alpha=.2,which='both')
 if k=='laplacian_error':ax.axhline(100,color='#9c6728',ls=':',lw=1)
fig.suptitle('Casewise maxima over every saved time, including initialization',fontsize=14)
fig.text(.5,.005,'Dots: individual cases; black bars: median. The four panels have different error denominators.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.025,1,.98));save(fig,'derivative-error-case-maxima')
# Exactly the five registered cells with a common independent residual observer.
widths=[128,256,512]; tests=[2048,4096,8192]
map_defs=[('max_discrete_error','Worst maximum field error (%)',100),('initial_error','Worst initial projection error (%)',100),('evolution_distance_max','Worst distance from reference projection (%)',100),('shell_max','Worst independent scaled residual norm',1),('unscaled_shell_max','Worst independent unscaled residual norm',1),('eligible','Qualified cases / six',1)]
fig,axs=plt.subplots(2,3,figsize=(13.5,7.8))
mapvals={}
for ax,(k,title,mult) in zip(axs.flat,map_defs):
 arr=np.full((3,3),np.nan)
 for r,m in arms:
  rows=[t for t in table if t['R']==r and t['M']==m]
  arr[tests.index(m),widths.index(r)]=sum(t[k] for t in rows) if k=='eligible' else mult*max(t[k] for t in rows)
 mapvals[k]=[[None if not np.isfinite(v) else float(v) for v in row] for row in arr]
 masked=np.ma.masked_invalid(arr);cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#f1f2f3')
 norm=None if k=='eligible' else LogNorm(vmin=np.nanmin(arr)*.99,vmax=np.nanmax(arr)*1.01)
 im=ax.imshow(masked,origin='lower',cmap=cmap,norm=norm,aspect='auto',vmin=0 if k=='eligible' else None,vmax=6 if k=='eligible' else None)
 for iy in range(3):
  for ix in range(3):
   v=arr[iy,ix];txt='not run' if not np.isfinite(v) else (f'{int(v)}/6' if k=='eligible' else f'{v:.3g}')
   color='#697078' if not np.isfinite(v) else ('black' if (im.norm(v)>.6) else 'white')
   ax.text(ix,iy,txt,ha='center',va='center',color=color,fontsize=11)
 ax.set_xticks(range(3),widths);ax.set_yticks(range(3),tests);ax.set_xlabel('Active width R′');ax.set_ylabel('Weak tests M');ax.set_title(title)
 if k!='eligible':fig.colorbar(im,ax=ax,shrink=.75)
fig.suptitle('Width–test interaction: measured configurations only',fontsize=15)
fig.text(.5,.002,'Dense discrete upwind model; six development cases. Common independent observer: rows 8193–9216. Blank cells were not measured.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.03,1,.97));save(fig,'width-test-map')
# Main accuracy map adds earlier measured R128 tests; observer maps remain separate.
full=read(OLD/'checks/full6/analysis.json')
assert full['saved_array_audit_passed']
for attempt in ['p03','p04b']:
 collect=read(OLD/f'runs/{attempt}/collection.json')
 assert collect['checksum_verified'] and collect['gpu_log_valid'] and collect['no_environmental_warnings'] and collect['scheduler_state']=='COMPLETED'
expanded=list(table)
for case in range(6):
 for m in [256,512,1024]:
  x=full['cases'][str(case)]['arms'][str(m)]
  assert x['numerically_eligible']
  expanded.append(dict(R=128,M=m,case=case,max_discrete_error=max(x['error']),initial_error=x['initial_error'],evolution_distance_max=max(x['evolution_distance']),eligible=x['numerically_eligible']))
alltests=[256,512,1024,2048,4096,8192]
fig,axs=plt.subplots(1,3,figsize=(12.5,6.7))
fullmap={}
for ax,(key,title) in zip(axs,[('max_discrete_error','Worst maximum field error (%)'),('initial_error','Worst initial projection error (%)'),('evolution_distance_max','Worst evolution distance (%)')]):
 arr=np.full((len(alltests),3),np.nan)
 for m in alltests:
  for r in widths:
   rows=[z for z in expanded if z['R']==r and z['M']==m]
   if rows:arr[alltests.index(m),widths.index(r)]=100*max(z[key] for z in rows)
 fullmap[key]=[[None if not np.isfinite(v) else float(v) for v in row] for row in arr]
 cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#f1f2f3');im=ax.imshow(np.ma.masked_invalid(arr),origin='lower',aspect='auto',cmap=cmap,norm=LogNorm(vmin=np.nanmin(arr)*.99,vmax=np.nanmax(arr)*1.01))
 for y,row in enumerate(arr):
  for x,v in enumerate(row):
   ax.text(x,y,'not run' if not np.isfinite(v) else f'{v:.3g}',ha='center',va='center',fontsize=10,color='#697078' if not np.isfinite(v) else ('black' if im.norm(v)>.6 else 'white'))
 ax.set_xticks(range(3),widths);ax.set_yticks(range(len(alltests)),alltests);ax.set_xlabel('Active width R′');ax.set_ylabel('Weak tests M');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle('Accuracy across all measured width–test configurations',fontsize=14)
fig.text(.5,.012,'Same six development cases and initial-field norm. No interpolation between cells. Independent-observer comparisons are shown separately.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.97));save(fig,'width-test-accuracy-map')
plotdata['expanded_accuracy_map']={'test_counts':alltests,'widths':widths,'cells':fullmap,'additional_source_sha256':sha(OLD/'checks/full6/analysis.json')}

# Retain every fixed-M transient counterexample, rather than selecting one.
fig,axs=plt.subplots(2,3,figsize=(12,6.3),sharex=True,sharey=True)
reg=[]
for i,ax in enumerate(axs.flat):
 diff=100*(series(512,2048,'error')[i]-series(256,2048,'error')[i]);reg.append(dict(case=i,maximum_increase_percentage_points=float(diff.max()),time=float(times[diff.argmax()]),increased_time_count=int(np.sum(diff>1e-8))))
 ax.axhline(0,color='#6a7177',lw=1);ax.fill_between(times,0,diff,where=diff>0,color='#b64759',alpha=.3)
 ax.plot(times,diff,color='#a23a50');ax.set_title(f'Development case {i}');ax.grid(alpha=.2);ax.set_xlabel('Time');ax.set_ylabel('Field-error change (percentage points)')
fig.suptitle('More width at fixed tests can worsen parts of the trajectory',fontsize=14)
fig.text(.5,.005,'R′=512 minus R′=256, both M=2048. Positive regions are retained regressions; each arm starts from its own native projection.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.025,1,.97));save(fig,'width-test-transient-regressions')
# Denominator audit from existing reference fields, one case at a time.
denoms={}
for i in range(6):
 p=OLD/f'runs/w128/output/case-{i:02d}/reference-1.npz';u=np.load(p)['fields'];n=256
 padded=np.pad(u.reshape(-1,n-1,n-1),((0,0),(1,1),(1,1)));c=padded[:,1:-1,1:-1];xm=padded[:,:-2,1:-1];xp=padded[:,2:,1:-1];ym=padded[:,1:-1,:-2];yp=padded[:,1:-1,2:]
 adv=(c*n*(np.where(c>0,c-xm,xp-c)+np.where(c>0,c-ym,yp-c))).reshape(len(u),-1)
 lap=(n*n*(xm+xp+ym+yp-4*c)).reshape(len(u),-1)
 grad=np.stack(np.gradient(padded,1/n,axis=(1,2)),axis=1).reshape(len(u),-1)
 norms={'error':np.repeat(np.linalg.norm(u[0]),len(u)),'gradient_error':np.linalg.norm(grad,axis=1),'upwind_advection_error':np.linalg.norm(adv,axis=1),'laplacian_error':np.linalg.norm(lap,axis=1)}
 denoms[str(i)]={'reference_sha256':sha(p),'denominator_euclidean_norms':{k:v.tolist() for k,v in norms.items()},'range':{k:{'min':float(v.min()),'max':float(v.max()),'min_over_initial':float(v.min()/v[0]),'nonzero':bool(np.all(v>0))} for k,v in norms.items()},'absolute_numerator_ranges':{f'R{r}-M{m}':{k:{'min':float((series(r,m,k)[i]*norms[k]).min()),'max':float((series(r,m,k)[i]*norms[k]).max())} for k in metrics} for r,m in arms}}
plotdata['width_test_map']=mapvals;plotdata['regressions']=reg
plotdata['denominator_audit']=denoms
dump(D/'plotted-data.json',plotdata)
summary={'derivative_case_maxima':{f'R{r}-M{m}':{k:{'median_percent':float(100*np.median(series(r,m,k).max(1))),'worst_percent':float(100*series(r,m,k).max())} for k in metrics} for r,m in arms},'denominator_ranges':{k:{'minimum':min(d['range'][k]['min'] for d in denoms.values()),'maximum':max(d['range'][k]['max'] for d in denoms.values()),'smallest_fraction_of_initial':min(d['range'][k]['min_over_initial'] for d in denoms.values())} for k in metrics},'regressions':reg}
dump(D/'summary.json',summary)
(D/'CAPTIONS.md').write_text('''# Saved Burgers manuscript panels\n\nThese plots are deterministic postprocessing of the accepted development results. They introduce no new PDE solve. All are a linear-span, dense discrete-upwind study; they do not establish sparse speed.\n\n**Derivative error through time.** Median and full range across all six development cases for the baseline, wider fixed-test model and wider enriched-test model. The field denominator is the original initial-field norm, whereas derivative and operator denominators are instantaneous corresponding reference norms. Gradient includes the original padded-grid convention. All derivatives are the implemented discrete operators; no analytic/continuum certificate is implied. The Laplacian panel retains errors above unity.\n\n**Casewise derivative maxima.** Each point is one physical case's maximum over all saved times including initialization. Black bars give medians; timing repetitions and time samples are not independent cases. The near-zero-denominator audit and absolute numerator ranges accompany the plot in plotted-data.json.\n\n**Expanded accuracy map.** The earlier measured small-test R128 arms are combined with the accepted width follow-up under the same physical cases, target and normalization. No independent residual observers are pooled.\n\n**Measured width–test observer map.** Only the five registered width/test cells with a common independent observer are shown; untested cells are grey. Entries are worst casewise temporal maxima, except the initial projection panel and qualification counts. The reported evolution distance is computed from fields relative to the reference's mass projection, not by subtracting scalar errors.\n\n**Transient regressions.** For each existing case, plot the actual field-error change when active width increases at fixed tests. Positive regions retain adverse transient behavior. The native initial projection differs with width; this comparison does not attribute the entire change to time integration alone.\n\n## Glossary\n\n**R′:** active linear-span width. **M:** actual scalar weak tests. **Weak residual:** equation error integrated against test functions. **Independent observer:** unused test rows assessing the residual. **Projection:** the least-squares representation of the reference in a fixed span. **Qualified:** passed registered numerical checks. **Derivative denominator:** norm of the corresponding reference operator at that time. **Percentage point:** absolute difference between two percentage errors. **Case range:** smallest through largest physical-case value.\n''')
manifest={'sources':{str(p):sha(p) for p in [OLD/'checks/width-results/analysis.json',OLD/'checks/width-results/table.json',OLD/'checks/full6/analysis.json',OLD/'runs/p03/collection.json',OLD/'runs/p04b/collection.json',L/'build_saved_plots.py']},'outputs':{p.name:sha(p) for p in D.iterdir() if p.is_file() and p.name!='MANIFEST.json'}}
dump(D/'MANIFEST.json',manifest)
print(json.dumps(summary,indent=2))
