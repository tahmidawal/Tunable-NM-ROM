"""Generate result panels exclusively from audited invocation JSONs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(record):
    source=record/'cluster/out/pilot/result.json';data=json.loads(source.read_text())
    assert json.loads((record/'audit.json').read_text())['passed']
    groups=defaultdict(list)
    for row in data['invocations']:groups[row['intervals'],row.get('cohort','opened'),row['method'],row['setting']].append(row)
    panel=[]
    for (n,cohort,method,setting),rows in groups.items():
        times=np.array([r['seconds']['complete_device_query']*1e3 for r in rows]);median=float(np.median(times))
        initial={name:100*max(r['same_grid_discrepancy'][name]['max_initial_normalized'] for r in rows) for name in ('displacement','velocity','energy_state')}
        current={name:100*max(r['same_grid_discrepancy'][name]['max_current_relative'] for r in rows) for name in ('displacement','velocity','energy_state')}
        phase=max(float(np.nanmax(np.asarray(r['same_grid_discrepancy']['phase']['error'],dtype=float))) for r in rows)
        energy_drift=max(float(np.max(abs(np.asarray(r['same_grid_discrepancy']['energy_fraction'])/r['same_grid_discrepancy']['energy_fraction'][0]-1))) for r in rows)
        numerical=all(r['completed'] and r.get('fit_stationary',True) and r.get('cg_all_converged',True)
            and r.get('projection_kinematic_eligible',True) and r.get('maximum_normal_backward_error',0.)<1e-10 for r in rows)
        related_refinements=[x for x in data['time_refinement'] if x['intervals']==n and x['method']==method and x['dt']==setting and x['case'].startswith(cohort+'_')]
        related_kinematics=[x for x in data.get('projection_kinematics',[]) if x['intervals']==n and x['method']==method and x['case'].startswith(cohort+'_')]
        time_pass=all(x['passed'] for x in related_refinements) if related_refinements else None
        kinematic_pass=all(x['passed'] for x in related_kinematics) if related_kinematics else None
        panel.append(dict(intervals=n,cohort=cohort,method=method,setting=setting,median_gpu_ms=median,
            timing_ms=times.tolist(),outliers_above_twice_median=int(np.sum(times>2*median)),
            worst_initial_errors_percent=initial,worst_current_errors_percent=current,
            worst_defined_sine_projection_phase_radians=phase,max_energy_drift_from_returned_initial_percent=100*energy_drift,
            all_state_5percent_pass=max(initial.values())<=5,all_completed=all(r['completed'] for r in rows),
            all_initial_fits_stationary=all(r.get('fit_stationary',True) for r in rows),
            all_projection_fits_stationary=all(r.get('all_output_fits_stationary',True) for r in rows),
            all_cg_converged=all(r.get('cg_all_converged',True) for r in rows),numerical_gate_pass=numerical,
            time_refinement_pass=time_pass,projection_kinematic_pass=kinematic_pass,
            physical_and_numerical_pass=max(initial.values())<=5 and numerical and time_pass is not False and kinematic_pass is not False,
            max_normal_backward=max(r.get('maximum_normal_backward_error',0.) for r in rows)))
    for row in panel:
        fom=[x for x in panel if (x['intervals'],x['cohort'])==(row['intervals'],row['cohort']) and x['method'].startswith('cg') and x['physical_and_numerical_pass']]
        if fom:
            best=min(fom,key=lambda x:x['median_gpu_ms'])
            row['fastest_tested_passing_cg']=dict(method=best['method'],tolerance=best['setting'],median_gpu_ms=best['median_gpu_ms'])
            row['fastest_passing_cg_over_method']=best['median_gpu_ms']/row['median_gpu_ms']
    payload=dict(source_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),panels=panel,
        parity=data['parity'],time_refinement=data['time_refinement'],projection_kinematics=data.get('projection_kinematics',[]),
        standalone_profile_label='svd_r includes QR followed by SVD(R); standalone component times are not additive stage fractions.')
    (record/'panel.json').write_text(json.dumps(payload,indent=2)+'\n')
    for p in panel:print(p['intervals'],p['cohort'],p['method'],p['setting'],p['median_gpu_ms'],p['worst_initial_errors_percent'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);analyze(ap.parse_args().record)
