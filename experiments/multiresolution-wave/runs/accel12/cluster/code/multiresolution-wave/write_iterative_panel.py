"""Export concise machine-readable wave rows from the independently audited run."""
import argparse
import hashlib
import json
from pathlib import Path


def main(record):
    path=record/'analysis/summary.json';summary=json.loads(path.read_text())
    assert summary['integrity_audit_passed'] and summary['extra_independent_audit']['passed']
    rows=[]
    for g in summary['groups']:
        rows.append(dict(boundary=g['boundary'],intervals=g['intervals'],method=g['method'],gpu_median_ms=1000*g['query_median'],gpu_plus_output_transfer_median_ms=1000*g['device_plus_output_transfer_median'],case_gpu_ms={str(c['case']):[1000*t for t in c['raw_query_seconds']] for c in g['cases']},worst_current_relative={name:x['current_max_worst'] for name,x in g['errors'].items()},worst_initial_normalized={name:x['initial_max_worst'] for name,x in g['errors'].items()},components_at_5_percent=g['component_accuracy_by_target']['0.05'],all_current_qualified_at_5_percent=g['accuracy_qualification_by_target']['0.05']['all_current_qualified'],all_initial_qualified_at_5_percent=g['accuracy_qualification_by_target']['0.05']['all_initial_qualified'],failed_cases=g['failed_cases'],nonstationary_cases=g['nonstationary_cases'],cg_failed_cases=g['cg_failed_cases'],time_refinement_failed_cases=g['time_refinement_failed_cases'],reference_refinement_failed_cases=g['reference_refinement_failed_cases'],maximum_cg_true_relative=g['maximum_cg_true_relative'],cg_max_step_iterations=g['cg_max_step_iterations'],cg_total_iterations=g['cg_total_iterations'],timing_outliers=g['timing_outliers'],cg_1e_minus6_over_method_ratio_of_cohort_medians=g['fom_over_method_ratio_of_cohort_medians'],paired_cg_1e_minus6_over_method_median=g['paired_fom_over_method_median']))
    result=dict(status='audited development results',provenance=summary['provenance'],config=summary['config'],audit_path=str(path),audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),audited_invocations=summary['audited_invocations'],timed_invocations=summary['config']['expected_timed_invocations'],rows=rows,definitions=dict(median='Median of per-case repetition medians; complete supplied-GPU-field to requested-GPU-field query.',host='GPU query plus both output transfers only; input transfers are excluded, so this is not host-to-host time.',time_grid='Both methods share dt=0.0025 and 49 outputs through t=2.4; FOM is new implicit midpoint, ROM retains original RK4.',fom='Warm-started unpreconditioned CG, compiled solve and rollout, charged true residual at every step; a new matched-time-step iterative control, not a historical wave implementation.',reference='Verified fresh same-grid semidiscrete DST for reflective waves; independently checked two-level RK4 temporal refinement for absorbing waves. No continuum bound.',current='Error divided by the corresponding reference norm at each time; vanishing-reference flags are retained.',initial='Displacement divided by initial displacement L2; velocity and joint energy-state divided by initial physical energy scale.',qualification='Every case and all displacement, velocity and energy-state components must pass the target plus completed/rank/fit/CG/reference/time-refinement gates. Current and initial-normalized qualification are distinct.',outlier='Repetition exceeding twice its own case median; all repetitions retained.',scope='Two exposed development cases per boundary, one frozen post-reset MLP32 learned-bank checkpoint per boundary; no retraining, final cohort, or pre-reset evidence.'))
    (record/'analysis/panel.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in rows:
        if row['method'] in (result['config']['primary_method'],'cg_1e-06','cg_0.01'):
            print(row['boundary'],row['intervals'],row['method'],f"{row['gpu_median_ms']:.6f} ms",f"u={100*row['worst_current_relative']['displacement']:.6f}%",f"v={100*row['worst_current_relative']['velocity']:.6f}%",f"energy={100*row['worst_current_relative']['energy_state']:.6f}%",'current5=',row['all_current_qualified_at_5_percent'],'initial5=',row['all_initial_qualified_at_5_percent'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);main(ap.parse_args().record)
