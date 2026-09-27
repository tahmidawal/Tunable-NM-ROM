"""Plot actual saved validation fields and the independently audited CP span fit."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ap=argparse.ArgumentParser();ap.add_argument('--diagnostic',required=True);ap.add_argument('--fields',required=True)
args=ap.parse_args();out=Path(args.diagnostic);fields=Path(args.fields)
report=json.loads((out/'diagnosis.json').read_text())
case=report['rows'][0]['case'];images=[];truth=None
for method in ('cp','modcp','film'):
    row=next(r for r in report['rows'] if r['case']==case and r['method']==method)
    file=fields/f'validation_{method}_cap10_dt0.00125_q4_tol0.0001_L256_case{case}.npz'
    with np.load(file) as data:
        images.append(data['u'][0]);truth=data['truth_u'][0]
with np.load(out/'cp_case3_free_span_projection.npz') as data:span=truth+data['residual'][0]
images=[truth]+images+[span]
names=['Supplied initial field','CP','Modified CP','FiLM INR','CP free coefficients\n(offline projection)']
norm=np.linalg.norm(truth);errors=[np.linalg.norm(x-truth)/norm for x in images]
minimum=min(x.min() for x in images);maximum=max(x.max() for x in images)
error_limit=max(np.abs(x-truth).max() for x in images)
fig,axes=plt.subplots(2,5,figsize=(15,6),constrained_layout=True)
for column,(name,field,error) in enumerate(zip(names,images,errors)):
    top=axes[0,column].imshow(field.T,origin='lower',extent=(0,1,0,1),vmin=minimum,vmax=maximum,cmap='viridis')
    bottom=axes[1,column].imshow((field-truth).T,origin='lower',extent=(0,1,0,1),vmin=-error_limit,vmax=error_limit,cmap='RdBu_r')
    axes[0,column].set_title(name,fontsize=11)
    axes[1,column].set_title(f'Initial relative L2: {error:.2%}',fontsize=10)
    for row in range(2):
        axes[row,column].set_xlabel('x');axes[row,column].set_ylabel('y');axes[row,column].set_aspect('equal')
fig.colorbar(top,ax=axes[0,:],shrink=.8,label='Field value')
fig.colorbar(bottom,ax=axes[1,:],shrink=.8,label='Prediction minus supplied field')
fig.suptitle(f'Validation case {case}, {truth.shape[0]-1} intervals: initial fit before PDE evolution\n'
             'The free-coefficient CP projection bounds this frozen checkpoint; it is excluded from online comparisons.',fontsize=12)
for suffix in ('png','pdf'):fig.savefig(out/f'validation_initial_case{case}.{suffix}',dpi=180)
plt.close(fig)
