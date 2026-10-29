"""Plot recorded full-training metrics; never interpret modal errors as flow errors."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--metadata',type=Path,required=True)
p.add_argument('--review',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();review=json.loads(a.review.read_text());cfg=json.loads((a.metadata/'CONFIG.json').read_text())
paths=sorted(a.metadata.glob('checkpoint-*.json'));rows=[json.loads(x.read_text()) for x in paths]
assert len(rows)==review['checkpoint_count'] and review['passed']
t=np.array([x['elapsed_seconds'] for x in rows]);f=np.array([x['full_post_update_objective'] for x in rows]);v=np.sqrt(np.array([x['metrics']['value_relative_squared_by_mode'] for x in rows]));g=np.sqrt(np.array([x['metrics']['gradient_relative_squared_by_mode'] for x in rows]))
assert np.isfinite(f).all() and np.isfinite(v).all() and np.isfinite(g).all()
selected=next(i for i,x in enumerate(paths) if x.stem==review['selected']['checkpoint']);assert f[selected]==review['selected']['objective']
fig,ax=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
ax[0].semilogy(t,f,color='#244870');ax[0].scatter(t[selected],f[selected],color='#d46332',zorder=5,label='Selected checkpoint');ax[0].legend();ax[0].set(title='Combined training objective',ylabel='Weighted relative squared errors')
for i in range(v.shape[1]):
 label=f'Mode {i+1}'
 ax[1].semilogy(t,100*v[:,i],label=label)
 ax[2].semilogy(t,100*g[:,i],label=label)
ax[1].set(title='Velocity-mode fitting error',ylabel='Relative mode error [%]');ax[2].set(title='Gradient-mode fitting error',ylabel='Relative mode error [%]');ax[2].legend(ncol=2)
switch=next(x['elapsed_seconds'] for x in rows if x['phase']=='lbfgs')
for z in ax:
 z.axvline(switch,color='#888',linestyle='--',linewidth=.8)
 z.set_xlabel('Optimizer elapsed [s], excluding setup');z.grid(alpha=.2)
fig.suptitle('First POD-guided neural bank: recorded training diagnostics',fontsize=15)
fig.supxlabel(f"{review['adam_steps_recorded']:,} Adam updates and {review['accepted_lbfgs_steps']:,} accepted L-BFGS steps; dashed line marks optimizer change.\nTermination: {review['optimizer_status']}. These normalized training-mode errors are not actual ROM flow errors.",fontsize=10)
a.output.mkdir(parents=True,exist_ok=True)
for ext in ('png','pdf'):fig.savefig(a.output/('training-progress.'+ext),dpi=150)
plt.close(fig)
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
manifest={'generator_sha256':sha(Path(__file__)),'review_sha256':sha(a.review),'config_sha256':sha(a.metadata/'CONFIG.json'),'checkpoint_metadata_sha256':{x.name:sha(x) for x in paths},'selected':review['selected'],'model_sha256':review['model_sha256'],'mode_weights':cfg['mode_weights'],'interpretation':'Recorded normalized modal training fit, not PDE flow accuracy. Single-job optimizer elapsed excludes setup; no cross-job timing comparison.'}
(a.output/'PLOT-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
