"""Plot saved physical error trajectories, with no new model evaluation or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'mlp':'#315f9e','quadratic':'#bc4b30','mlp_velocity':'#13816c','quadratic_velocity':'#8b4da6'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result',action='append',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    datasets=[]
    for name in args.result:
        path=Path(name)
        result=json.loads(path.read_text())
        for bc,entry in result['boundary_results'].items():
            source={'result_path':str(path),'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_commit':result['provenance']['source_commit'],'job_id':result['provenance']['job_id']}
            datasets.append((path.parent,bc,entry,result['config'],source))
    figure,axes=plt.subplots(len(datasets),3,figsize=(13,4*len(datasets)),squeeze=False,sharex=True)
    metrics=[('displacement_error','Displacement error'),('velocity_error','Physical velocity error'),('energy_state_error','Error-state energy norm')]
    metadata=[]
    for row,(base,bc,entry,config,source) in enumerate(datasets):
        t=np.arange(int(round(config['end_time']/config['observation_dt']))+1)*config['observation_dt']
        for name,color in COLORS.items():
            arms=[arm for arm in entry['arms'] if arm['name']==name]
            temporal_unresolved=any(not arm['rollout']['refinement_passed'] for arm in arms)
            curve_label=name+(' [time unresolved]' if temporal_unresolved else '')
            collected={metric:[] for metric,_ in metrics}
            failures=[]
            for arm in arms:
                dt=arm['rollout']['primary_dt']
                path=base/bc/f"{name}_{arm['optimizer_seed']}"/'rollouts.npz'
                with np.load(path) as raw:
                    cases=[case for case in arm['rollout']['cases'] if case['dt']==dt]
                    for case in cases:
                        if not case['completed']:
                            failures.append({'seed':arm['optimizer_seed'],'case':case['case']})
                            continue
                        prefix=f"dt{dt}_case{case['case']}_"
                        for metric,_ in metrics:
                            collected[metric].append(raw[prefix+metric])
            for col,(metric,label) in enumerate(metrics):
                ax=axes[row,col]
                if collected[metric]:
                    values=np.stack(collected[metric])
                    middle=np.median(values,axis=0)
                    lower,upper=np.quantile(values,[.25,.75],axis=0)
                    ax.plot(t,middle,label=curve_label,color=color,lw=1.8)
                    ax.fill_between(t,lower,upper,color=color,alpha=.12)
                ax.axhline(config['accuracy_target'],color='#777777',linestyle='--',linewidth=.8)
                ax.set_title(f'{bc}: {label}')
                ax.set_ylabel('Initial-state-normalized error')
                ax.set_xlabel('Time')
                ax.grid(alpha=.2)
            metadata.append({'source':source,'boundary':bc,'head':name,'optimizer_seeds':[arm['optimizer_seed'] for arm in arms],'original_time_refinement_unresolved':temporal_unresolved,'finite_completed_case_repeat_count':len(collected['displacement_error']),'failed_cases':failures,'interpretation':'Median and interquartile range pooling completed case records from optimizer repeats on the same data; descriptive spread, not independent-data replicates or confidence intervals. Failed trajectories excluded from curves and counted explicitly here and in the report.'})
        axes[row,0].legend(fontsize=8)
    figure.suptitle('Fresh wave rollouts: predeclared primary time step\nMedians and IQR pool case records and optimizer repeats on the same data\nReflective quadratic curves fail the original timestep check; labels mark unresolved temporal accuracy',fontsize=11)
    figure.tight_layout(rect=(0,0,1,.94))
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    figure.savefig(out/'fresh-wave-error-trajectories.png',dpi=180)
    figure.savefig(out/'fresh-wave-error-trajectories.pdf')
    (out/'plot-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
