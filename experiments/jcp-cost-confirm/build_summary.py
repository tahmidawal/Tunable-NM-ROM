"""Generate the lane's numerical closure from retained JSON, never typed metrics."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/p01'
OUT=RUN/'output'
r=json.loads((OUT/'paired/result.json').read_text())
p=json.loads((OUT/'provenance.json').read_text())
q=json.loads((OUT/'numerical_qualification.json').read_text())
refs=json.loads((OUT/'references.json').read_text())
audit=json.loads((RUN/'audit.json').read_text())
sacct=[line.split('|') for line in (RUN/'sacct.txt').read_text().splitlines()]
allocation=next(a for a in sacct if a[0]==p['job_id'])
seconds=int(allocation[3])
rows=[]
for name,v in r['confirmation'].items():
    qq=[a for a in q if a['subject']==name]
    if qq:
        ee=[a['nominal_metrics']['evolved_max_error'] for a in qq]
        nominal_invalid=[a['case'] for a in qq if not a['nominal_valid']]
        stricter_invalid=[a['case'] for a in qq if not a['tighter_valid']]
        failed_refs=[a['case'] for a in qq if a['reference_refinement_difference']>a['reference_refinement_bound']]
    else:
        ee=[v['cases'][str(i)]['evolved_max_error'] for i in range(16)]
        nominal_invalid=[];stricter_invalid=[]
        failed_refs=[a['case'] for a in refs if a['dt_refinement_max']>min(p['config']['reference_absolute_gate'],p['config']['reference_fraction_of_target_error']*v['cases'][str(a['case'])]['evolved_max_error'])]
    rows.append({'subject':name,'nominal_all_case_count':len(ee),'nominal_all_case_median_error':float(np.median(ee)),
        'nominal_all_case_worst_error':max(ee),'nominal_invalid_cases':nominal_invalid,
        'tighter_invalid_cases':stricter_invalid,'fractional_reference_failed_cases':failed_refs,
        'failed_timed_invocations':v['failure_count'],'nominal_valid_timed_cases':len(v['cases']),
        'valid_subset_median_ms':v['median_case_ms'],'time_is_partial_cohort':not v['complete'],
        'numerical_pass':r['numerical_pass'][name],'timing_drift':r['timing_drifts'][name],
        'timing_outlier_count':v['outlier_count_tukey']})
summary={'schema':'jcp-cost-confirm-summary-v1','status':'completed scoped-negative frozen confirmation',
    'job_id':p['job_id'],'gpu':p['gpu'],'allocated_gpu_seconds':seconds,'allocated_gpu_hours':seconds/3600,
    'case_count':len(refs),'config':p['config'],'methods':rows,'comparison_decisions':r['comparisons'],
    'fixed_threshold_eligibility':r['frozen_threshold_eligibility'],'audit':audit,
    'no_new_speed_claim':True,'fresh_reselection':False,'held_out_cases_dropped':False,
    'metric_note':'all-case nominal accuracy includes returned but rejected solves as diagnostics; valid timing-subset medians are separate and not accuracy-cost evidence',
    'remaining_experiments':['A development-only initializer robustness repair and then a different untouched confirmation cohort are needed before an all-case production claim.',
      'If pursuing the coarse-classical comparison, refine the physical reference sufficiently and repeat a separately registered same-allocation timing block with the same frozen comparator; current thresholds and nominal head failure still preclude the current claim.',
      'No more timing candidate search or fresh-cohort tuning belongs to this closed batch.']}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
text=f'''# Frozen-choice cost confirmation and accuracy–cost plots

The experiment is complete with a scoped negative confirmation result. Saved-development plots are final; fresh nominal error diagnostics include explicitly rejected solves and do not establish a new speed claim. All numbers below are generated from retained run JSONs.

The fixed learned-head configurations and their prior selected comparators were evaluated on {len(refs)} new physical cases, without reselection. No fresh comparison passes all fixed accuracy, numerical and timing criteria. Each method's worst fresh error exceeds the corresponding old head worst-error limit; therefore this failure is substantive even before considering numerical qualification.

## Nominal accuracy, including rejected returned fields

These values cover every case. They are diagnostics of returned fields, not evidence that rejected solves satisfy the registered stopping condition. They are not combined with separately timed calls into an accuracy–cost point.

| Method | Cases | Median evolved error (%) | Worst evolved error (%) | Nominal invalid cases | Tightened invalid cases | Fractional reference failures |
|---|---:|---:|---:|---|---|---|
'''
for a in rows:text+=f"| `{a['subject']}` | {a['nominal_all_case_count']} | {100*a['nominal_all_case_median_error']:.9f} | {100*a['nominal_all_case_worst_error']:.9f} | {a['nominal_invalid_cases']} | {a['tighter_invalid_cases']} | {a['fractional_reference_failed_cases']} |\n"
failed=sum(a['failed_timed_invocations'] for a in rows)
maxgap=audit['max_solver_refinement'];maxref=audit['reference_max_refinement']
text+=f'''
Both learned heads fail nominal initialization on the same case. The returned fields remain finite, but the initial fit misses the frozen stationarity threshold. Tightening additionally exposes fit/stopping failures, despite maximum nominal–tighter field discrepancy of {maxgap:.12g}. A small field discrepancy does not retrospectively waive the registered stopping criterion. All reference absolute checks pass, with maximum refinement discrepancy {maxref:.12g}; the coarse comparator's smaller errors fail the relative reference bound on the cases listed above.

## Timing evidence remains separate

All {audit['invocations']} timed invocation records are retained, including {failed} failures. The two head timing summaries contain only their nominally successful subset and therefore are not reported as complete-cohort cost evidence here. The fresh figure displays those paired observations as explicitly unqualified diagnostics. The coarse comparator also fails the unchanged drift threshold. No old absolute time is pooled with this allocation.

The saved selection plot preserves all complete finite configurations, shows off-scale inaccurate settings at its upper boundary, and provides a full-range companion. Incomplete configurations have a separate failure table. Its marginal envelopes are visual summaries for one error statistic; qualification still requires both median and worst errors. Saved confirmation ratios and timing-repetition variation are separate figures.

## Completion and remaining scope

Job {p['job_id']} completed in {seconds} allocated GPU-seconds ({seconds/3600:.9f} GPU-hours) on {p['gpu'].split(',')[0]}. The independent evidence audit passes: raw repetitions, case/cohort summaries, paired ratios, source hashes and precision/backend records agree. This audit validates the retained evidence, not the scientific qualification. The result archive and every listed file were checksum-verified locally before the owned remote job directory was removed.

An all-case production claim requires an initializer robustness correction developed outside this now-exposed confirmation cohort, followed by a different untouched confirmation. Further strict comparison to the coarse classical solver also needs a more accurate reference and a clean registered timing repeat; these alone cannot erase the current head initialization and worst-error failures. No change to the frozen method, comparator search, new cohort or further job is performed in this batch. The previous development evidence remains valid within its stated scope.

Builders, all plotted values, source hashes and PNG/PDF/SVG exports are in `build_plots.py` and `plots/`. The lane remains separate and unmerged pending the user's decision. Root records the canonical lab-log closure and publishes the combined main report.

## Glossary

**Nominal:** the original fixed numerical settings. **Tightened:** the same method with stricter stopping tolerances, used as a sensitivity check. **Stationarity:** the gradient stopping condition for the least-squares solve. **Initial fit:** finding the starting reduced coordinates from the physical field. **Fractional reference bound:** requiring reference uncertainty to be small relative to the measured error. **Case:** one physical initial condition and viscosity. **Median/worst:** the middle/largest casewise evolved error. **Invocation:** one complete input-to-output execution. **Partial timing subset:** cases whose nominal solver termination passed; it is not the full cohort. **Drift:** change in repeated measured time through the measurement block. **Frozen:** fixed before fresh truth solutions were generated. **Qualification:** the combined numerical, accuracy and timing criteria. **POD:** a snapshot-derived spatial basis. **Scoped negative:** the fixed experiment failed its claim without disproving every possible version of the method.
'''
(HERE/'SUMMARY.md').write_text(text)
print(json.dumps({'summary':str(HERE/'summary.json'),'allocated_gpu_seconds':seconds,'failed_timed_invocations':failed,'comparisons_qualified':sum(c['qualified_fixed_threshold'] for v in r['comparisons'].values() for c in v.values())}))
