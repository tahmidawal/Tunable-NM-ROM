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
        panel.append(dict(intervals=n,cohort=cohort,method=method,setting=setting,median_gpu_ms=median,
            timing_ms=times.tolist(),outliers_above_twice_median=int(np.sum(times>2*median)),
            worst_initial_errors_percent=initial,worst_current_errors_percent=current,
            all_state_5percent_pass=max(initial.values())<=5,all_completed=all(r['completed'] for r in rows),
            all_initial_fits_stationary=all(r.get('fit_stationary',True) for r in rows),
            all_projection_fits_stationary=all(r.get('all_output_fits_stationary',True) for r in rows),
            max_normal_backward=max(r.get('maximum_normal_backward_error',0.) for r in rows)))
    payload=dict(source_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),panels=panel,
        parity=data['parity'],time_refinement=data['time_refinement'],projection_kinematics=data.get('projection_kinematics',[]),
        standalone_profile_label='svd_r includes QR followed by SVD(R); standalone component times are not additive stage fractions.')
    (record/'panel.json').write_text(json.dumps(payload,indent=2)+'\n')
    for p in panel:print(p['intervals'],p['cohort'],p['method'],p['setting'],p['median_gpu_ms'],p['worst_initial_errors_percent'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);analyze(ap.parse_args().record)
