"""Saved-field diagnostics separating initial error, evolution and near-wall loss."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from generate_rollout import summarize
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run);out=run/'out'
d=json.loads((out/'pilot.json').read_text());audit=json.loads((run/'AUDIT.json').read_text())
assert audit['source_sha256']==hashlib.sha256((out/'pilot.json').read_bytes()).hexdigest()
summary,groups=summarize(d);result=dict(source_sha256=audit['source_sha256'],scope='postprocessing of actual archived full-query fields; no new fitting or solve',cases=[],controls=[])
for L in sorted({r['intervals'] for r in summary}):
    x=np.arange(L+1)/L;xx,yy=np.meshgrid(x,x,indexing='ij');distance=np.minimum.reduce([xx,yy,1-xx,1-yy])
    band=(distance>0)&(distance<1/d['checkpoint_training_intervals'])
    metric=d['reference_metrics_by_output'][str(L)];margins={r['case']:r['conservative_difference_sum'] for r in metric['uncertainty']}
    for r in metric['order_audit']:
        if r['empirical_richardson_estimate'] is not None:margins[r['case']]=max(margins[r['case']],r['empirical_richardson_estimate'])
    for case in range(d['config']['cases']):
        ref=np.load(out/f"ref_L{d['config']['reference_mesh']}_dt{d['config']['reference_dt']}_case{case}.npz")['dense_fields']
        stride=d['dense_observation_intervals']//L;ref=ref[:,::stride,::stride];n0=np.linalg.norm(ref[0])
        fields={}
        for arm in d['arm_grid']:
            rule,budget,dt,stall=arm['cold_rule'],arm['ic_budget'],arm['dt'],arm['stall']
            name=f'rom_L{L}_{rule}_ic{budget}_dt{dt}_stall{stall}_starts1'
            row=next(r for r in groups[name] if r['case']==case and r['rep']==0)
            f=np.load(out/row['dense_artifact'])['fields'];assert hashlib.sha256(f.tobytes()).hexdigest()==row['field_sha256']
            fields[rule,budget,dt,stall]=f
        case_arms=[('edge',60,.005,.01),('fixed_gauss',180,.005,.01)] if d['experiment']=='fixed_gauss_rollout' else list(fields)
        for rule,budget,dt,stall in case_arms:
            f=fields[rule,budget,dt,stall];delta=f-ref
            err=np.linalg.norm(delta.reshape(len(f),-1),axis=1)/n0
            band_err=np.linalg.norm(delta[:,band],axis=1)/n0
            band_share=np.sum(delta[:,band]**2,axis=1)/np.maximum(np.sum(delta**2,axis=(1,2)),1e-300)
            target=.05;label='initial field already exceeds target' if err[0]+margins[case]>target else ('later trajectory exceeds target' if max(err[1:])+margins[case]>target else 'measured error plus margin meets target')
            result['cases'].append(dict(intervals=L,case=case,cold_rule=rule,ic_budget=budget,dt=dt,stall=stall,target=target,reference_margin=margins[case],target_diagnostic=label,
                complete_grid_error_per_time=err.tolist(),training_unseen_boundary_band_nodes=int(band.sum()),
                training_unseen_boundary_band_threshold=1/d['checkpoint_training_intervals'],
                boundary_band_error_per_time=band_err.tolist(),boundary_band_squared_error_fraction_per_time=band_share.tolist()))
        primary=fields['fixed_gauss',180,.005,.01]
        controls=[('fit_budget',('fixed_gauss',60,.005,.01)),('time_step',('fixed_gauss',180,.0025,.01)),('evolution_stopping',('fixed_gauss',180,.005,.001))] if d['experiment']=='fixed_gauss_rollout' else [('time_step',key) for key in fields if key[2]!=.005]
        for control,key in controls:
            f=fields[key]
            if d['experiment']=='gauss_timestep_study':assert np.array_equal(f[0],primary[0])
            diff=np.linalg.norm((f-primary).reshape(len(f),-1),axis=1)/n0
            result['controls'].append(dict(intervals=L,case=case,control=control,dt=key[2],initial_field_max_difference=float(np.max(np.abs(f[0]-primary[0]))),relative_field_difference_per_time=diff.tolist(),
                primary_worst=float(max(np.linalg.norm((primary-ref).reshape(len(f),-1),axis=1)/n0)),
                control_worst=float(max(np.linalg.norm((f-ref).reshape(len(f),-1),axis=1)/n0))))
result['limitation']='Boundary-band errors locate returned-field loss at points closer to the wall than the training mesh first interior node; they do not independently identify a bank projection floor or prove causality. Evolution controls distinguish measured effects without proving an optimizer is stationary.'
(run/'DIAGNOSTICS.json').write_text(json.dumps(result,indent=2)+'\n');print(run/'DIAGNOSTICS.json')
