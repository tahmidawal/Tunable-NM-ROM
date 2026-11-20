"""Figures and a source-generated lane note for the frozen closure experiments."""
from pathlib import Path
import json,hashlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
L=Path(__file__).resolve().parent;D=L/'figures/closure';D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=read(L/'checks/fresh/analysis.json');b=read(L/'checks/bridge/analysis.json');saved=read(L/'figures/saved/summary.json')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'jcp-width-closure-20261119','pdf.fonttype':42,'savefig.bbox':'tight'})
def save(fig,name):
 for ext in ['png','svg','pdf']:
  md={'Creator':'JCP width closure'}
  if ext=='pdf':md.update(CreationDate=None,ModDate=None)
  if ext=='svg':md['Date']=None
  fig.savefig(D/f'{name}.{ext}',dpi=210,metadata=md)
  if ext=='svg':
   path=D/f'{name}.svg';path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
 plt.close(fig)
base=f['arms']['R128-M2048'];sel=f['arms']['R512-M8192'];times=np.asarray(f['time'])
fig,axs=plt.subplots(1,3,figsize=(13,4.1))
for arm,color,label in [(base,'#59646c','R′=128, M=2048'),(sel,'#a43d69','R′=512, M=8192')]:
 curves=np.asarray([c['error'] for c in arm['cases']])*100
 axs[0].fill_between(times,curves.min(0),curves.max(0),color=color,alpha=.1)
 axs[0].plot(times,np.median(curves,axis=0),color=color,lw=2,label=label)
axs[0].set_yscale('log');axs[0].set_xlabel('Time');axs[0].set_ylabel('Field error / initial reference norm (%)');axs[0].set_title('New cases: median and full range');axs[0].legend(frameon=False,fontsize=9)
x=np.asarray([c['max_error'] for c in base['cases']])*100;y=np.asarray([c['max_error'] for c in sel['cases']])*100
lo=min(x.min(),y.min())*.7;hi=max(x.max(),y.max())*1.4
axs[1].plot([lo,hi],[lo,hi],color='#8d9398',ls='--',lw=1);axs[1].scatter(x,y,color='#a43d69',s=40)
for i,(xx,yy) in enumerate(zip(x,y)):axs[1].annotate(str(i),(xx,yy),xytext=((-12,-12) if i==0 else (4,11) if i==5 else (4,4)),textcoords='offset points',fontsize=8)
axs[1].set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),xlabel='Baseline casewise maximum error (%)',ylabel='Selected casewise maximum error (%)',title='Every frozen confirmation case')
rat=np.asarray(f['primary_confirmation']['casewise_selected_over_baseline']);axs[2].scatter(range(len(rat)),rat,color='#a43d69');axs[2].axhline(1,color='#8d9398',ls='--');axs[2].axhline(np.median(rat),color='#a43d69',lw=1,ls=':');axs[2].set(yscale='log',xlabel='Fresh case',ylabel='Selected / baseline maximum error',title='Paired error ratios (lower is better)',xticks=range(8))
for ax in axs:ax.grid(alpha=.2,which='both')
fig.suptitle('Frozen wider model confirms improved field accuracy on new cases',fontsize=14,y=1.03)
fig.text(.5,-.015,'Same-grid, same-step discrete reference; primary maximum includes initialization. Both width and tests change; no runtime claim.',ha='center',fontsize=9)
fig.tight_layout();save(fig,'fresh-width-confirmation')
fig,axs=plt.subplots(1,3,figsize=(13,4.3))
names=list(b['rules']);colors=['#278696','#a43d69'];xx=np.arange(4);gatefields=['heldout_moment_max','residual_initial_norm_max','jacobian_action_relative_max','cold_start_field_difference_max'];limits=np.asarray([1e-2,1e-5,1e-3,1e-4])
for i,(name,color) in enumerate(zip(names,colors)):
 r=b['rules'][name];values=np.asarray([r[k] for k in gatefields])/limits
 axs[0].bar(xx+(i-.5)*.32,values,width=.3,color=color,label='Checkerboard' if i==0 else 'Three-quarter grid')
axs[0].axhline(1,color='black',ls='--',lw=1);axs[0].set_yscale('log');axs[0].set_xticks(xx,['Moments','Residual','Jacobian\naction','Initial fit']);axs[0].set_ylabel('Error / registered acceptance limit');axs[0].set_title('Only values at or below one pass');axs[0].legend(frameon=False,fontsize=8)
for i,(name,color) in enumerate(zip(names,colors)):
 r=b['rules'][name];v=np.asarray([q['field_difference_initial_norm'] for q in r['cold_start']])*100
 axs[1].scatter(np.arange(3)+(i-.5)*.1,v,color=color,s=35)
axs[1].axhline(.01,color='black',ls='--',lw=1);axs[1].set_xticks(range(3),['Case 3','Case 4','Case 5']);axs[1].set_ylabel('Sparse / dense initial-fit field difference (%)');axs[1].set_title('Cold-start fidelity remains limiting')
for i,(name,color) in enumerate(zip(names,colors)):
 r=b['rules'][name];total=r['dense_count'];axs[2].bar(i-.17,100*r['actual_nonzero_support_count']/total,width=.32,color=color,alpha=.5);axs[2].bar(i+.17,100*r['actual_nonzero_stencil_count']/total,width=.32,color=color)
axs[2].axhline(80,color='black',ls='--',lw=1);axs[2].set_xticks([0,1],['Checkerboard','Three-quarter grid']);axs[2].set_ylim(0,135);axs[2].set_ylabel('Fraction of dense interior sites (%)');axs[2].set_title('Fitted support / full upwind-site closure')
from matplotlib.patches import Patch
axs[2].legend(handles=[Patch(facecolor='#71777d',alpha=.5,label='Fitted samples'),Patch(facecolor='#71777d',label='Upwind sites')],loc='upper center',ncol=2,frameon=False,fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.2)
fig.suptitle('The bounded sparse bridge fails its deployment gates',fontsize=14,y=1.03)
fig.text(.5,-.02,'Development data only. No pruning: every fitted weight is positive. Three-quarter fit reaches its iteration cap; no sparse-rollout or speed claim.',ha='center',fontsize=9)
fig.tight_layout();save(fig,'sparse-bridge-qualification')
# Parameter coverage for the freshly generated cases: axes are physical, not latent distance.
phys=np.asarray([x['physical'] for x in sel['cases']]);fig,axs=plt.subplots(1,2,figsize=(9.3,4))
sc=axs[0].scatter(phys[:,0],phys[:,1],c=100*np.asarray([x['max_error'] for x in sel['cases']]),cmap='viridis',s=60);fig.colorbar(sc,ax=axs[0],label='Selected maximum field error (%)')
for i in range(len(phys)):axs[0].annotate(str(i),phys[i,:2],xytext=(4,4),textcoords='offset points',fontsize=8)
axs[0].set(xlabel='Initial pulse x-center',ylabel='Initial pulse y-center',title='Fresh in-family physical cases',xlim=(.15,.85),ylim=(.15,.85))
axs[1].scatter(phys[:,-1],100*np.asarray([x['max_error'] for x in sel['cases']]),color='#a43d69',label='Selected model');axs[1].scatter(phys[:,-1],100*np.asarray([x['max_error'] for x in base['cases']]),color='#59646c',label='Baseline');axs[1].set(xscale='log',yscale='log',xlabel='Viscosity',ylabel='Casewise maximum field error (%)',title='Case variation, not extrapolation');axs[1].legend(frameon=False,fontsize=9)
fig.tight_layout();save(fig,'fresh-case-coverage')
lines=['# Burgers width confirmation and sparse deployment closure','', 'Final numerical results for the registered frozen-choice same-family confirmation and bounded developmental sparse bridge. The dense confirmation passes; neither sparse rule is admitted for deployment.','', '## Frozen confirmation','', '| Configuration | Qualified cases | Median primary error (%) | Worst primary error (%) |','|---|---:|---:|---:|']
for label,arm in [('Baseline R128/M2048',base),('Selected R512/M8192',sel)]:lines.append(f"| {label} | {arm['qualified_count']}/{len(arm['cases'])} | {100*arm['median_max_error']:.9f} | {100*arm['worst_max_error']:.9f} |")
q=f['primary_confirmation'];lines+=['',f"All fresh cases improve: {q['all_eight_improved']}. Median paired selected/baseline error ratio: {q['median_paired_ratio']:.9f}. Ratio of worst errors: {q['ratio_of_worst_errors']:.9f}. Registered confirmation gate: {q['gate_passed']}.",'','The primary metric is the maximum over every stored state including initialization, normalized by the original initial-field norm. The reference is the same-grid, same-step discrete FOM. Width and test count both change, so the joint gain is not attributed to width alone. This is an accuracy confirmation, with no cross-job cost claim.','', '## Sparse bridge','', '| Rule | Nonzero samples | Actual stencil sites | Held-out moment error | Residual error / initial norm | Relative Jacobian-action error | Initial-fit field difference | Admitted |','|---|---:|---:|---:|---:|---:|---:|---|']
for name,r in b['rules'].items():lines.append(f"| {name} | {r['actual_nonzero_support_count']} | {r['actual_nonzero_stencil_count']} | {r['heldout_moment_max']:.9g} | {r['residual_initial_norm_max']:.9g} | {r['jacobian_action_relative_max']:.9g} | {r['cold_start_field_difference_max']:.9g} | {r['admitted']} |")
lines+=['','Both rules fail the initial-fit and site-economy gates. The larger rule also fails residual and Jacobian-action gates. Its bounded NNLS fit reaches the iteration cap; its optimizer does not report convergence even though the separately registered projected-gradient threshold passes. The checkerboard fit has a successful optimizer exit, but that does not qualify its initialization or physical evaluation cost. Exact nonzero-weight closure equals proposed-support closure; no pruning threshold is used.','', 'The gate closes the registered continuation: no sparse rollout, time-step interaction or sparse speed claim is admitted. This is a negative result for these frozen supports and fit recipe, not an impossibility theorem for every quadrature. The successful dense wider model remains grid-bound.','', '## Derivative interpretation','', 'The saved derivative panels retain large discrete Laplacian discrepancies despite small field error. These are not caused by zero reference denominators: the saved denominator audit records every reference norm and absolute error numerator. The derivative targets are the implemented discrete operators, not analytic network derivatives or continuum certificates.','', '## Evidence and plots','', '- [Fresh numerical audit](checks/fresh/analysis.json)', '- [Sparse independent audit](checks/bridge/analysis.json)', '- [Saved derivative and width plot data](figures/saved/plotted-data.json)', '- [Fresh confirmation plot](figures/closure/fresh-width-confirmation.png)', '- [Sparse bridge plot](figures/closure/sparse-bridge-qualification.png)','', '## Glossary','', '**R:** active bank width. **M:** actual scalar weak-test count. **Qualified:** passed registered reference, solve, array and environment checks. **Primary error:** largest original-initial-norm field error over the registered trajectory. **FOM:** full-order discrete model. **Stencil closure:** every distinct physical site needed for the exact possible upwind neighbors. **Jacobian action:** response of the residual to a physically normalized perturbation. **NNLS:** least-squares integration-weight fit constrained to nonnegative weights. **Cold start:** finding coefficients from the supplied initial physical field. **Development:** previously exposed cases used for the sparse audit. **Fresh confirmation:** new physical cases generated only after the method settings were frozen.']
(L/'RESULTS.md').write_text('\n'.join(lines)+'\n')
dump(D/'plotted-data.json',{'fresh':f,'sparse':b})
dump(D/'MANIFEST.json',{'sources':{str(p):sha(p) for p in [L/'checks/fresh/analysis.json',L/'checks/bridge/analysis.json',L/'build_closure_plots.py']},'outputs':{p.name:sha(p) for p in D.iterdir() if p.is_file() and p.name!='MANIFEST.json'},'result_note_sha256':sha(L/'RESULTS.md')})
