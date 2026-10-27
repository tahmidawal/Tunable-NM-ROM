"""Generate A3's scoped table and final handoff from audited saved evidence."""
from pathlib import Path
import csv
import json
from analyse import sha


def main():
    h=Path(__file__).resolve().parent;d=h/'runs/k16fine01/collected';rp=d/'output/result.json'
    r=json.loads(rp.read_text());sp=h/'checks/k16fine01/SUMMARY.json';s=json.loads(sp.read_text())
    assert r['complete'] and s['offline_checks_pass'] and not r['admission']['full64_adapter_release']
    audit=h/'audits/results-A3-01.md';assert audit.exists()
    rows=list(csv.DictReader((d/'sacct.txt').read_text().splitlines(),delimiter='|'))
    acct=next(x for x in rows if x['JobID']==str(r['job_id']));assert acct['State']=='COMPLETED' and acct['ExitCode']=='0:0'
    elapsed=int(acct['ElapsedRaw']);hours=elapsed/3600
    low={i:max(row['k16_to_none']-row['k16']['tight_distance']-row['none']['tight_distance'],0.) for i,row in r['cases'].items()}
    gradient_bad=sum(not flag for flags in r['admission']['extra_gradient_step_mask'].values() for flag in flags)
    lines=['# Archived POD support at the selected no-HR endpoint','',
        'Final four-case diagnostic evidence at the independently selected no-HR endpoint. Both own-objective nonlinear checks pass, while shared field consistency and the separate additional POD gradient checks fail; no complete-cohort HR release follows.','',
        f"The frozen endpoint is `{r['selection']['selected']['subject_id']}`. Original support failures and the stopped constructor remain preserved. These are exposed validation cases, not sealed confirmation or a complete accuracy/cost frontier.",'',
        '| arm | own-objective Q1 | nonstationary steps | median E0 error | worst E0 error |',
        '|---|---|---:|---:|---:|']
    for arm in ('k16','none'):
        q=s['Q1'][arm];label='k16 (sampled objective)' if arm=='k16' else 'none (full residual)'
        lines.append(f"| {label} | {q['pass_']} | {q['nonstationary_steps']} / {q['steps']} | {q['median_error']:.6%} | {q['worst_error']:.6%} |")
    lines+=['','Errors are nominal relative to approved E0 fields with conditional reference uncertainty; their small inter-arm differences do not establish an accuracy ranking.','',
        '| case | k16 to matching-none field distance | common bar | common pass | native-tight distance lower bound (diagnostic) |',
        '|---|---:|---:|---|---:|']
    for i,row in r['cases'].items():
        lines.append(f"| {i} | {row['k16_to_none']:.10g} | {r['admission']['shared_field_bar']:.6g} | {r['admission']['shared_field_case_mask'][i]} | {low[i]:.10g} |")
    lines+=['','The native-tight lower bound is a post-processing diagnostic from the triangle inequality and the two saved production-to-tight distances; it is not an additional admission predicate.','',
        '| case | maximum gradient discrepancy | maximum full stationarity | gradient misses | full-stationarity misses |',
        '|---|---:|---:|---:|---:|']
    for i,m in s['whole_residual'].items():
        lines.append(f"| {i} | {max(m['delta_g']):.10g} | {max(m['full_gamma']):.10g} | {sum(not x for x in m['gradient_step_mask'])} | {sum(m['full_stationarity_bad_step_mask'])} |")
    a=r['admission'];lines+=['',
        f"The extra gradient and full-stationarity bars are {a['extra_gradient_bar']:g}. The gradient bar applies at every live step. There are {gradient_bad} gradient misses and {a['extra_full_stationarity_bad_steps']} / {a['extra_full_stationarity_steps']} full-stationarity misses; the allowed pooled fraction is {a['extra_full_stationarity_fraction_limit']:.0%}.",'',
        f"Matching-none endpoint reproduction passed: {r['endpoint_reproduction_pass']}. All actual timed payloads remain exactly hash-linked or separately saved; {len(r['timing'])} ABA panels are retained and {sum(t['timing_admitted'] for t in r['timing'])} pass diagnostic drift admission. No rejected row enters an admitted cost comparison.",'',
        f"Job `{r['job_id']}` completed with exit `{acct['ExitCode']}` in {elapsed} allocated seconds ({hours:.9g} GPU-hours). The full-residual fallback remains qualified in its original complete-cohort scope. A qualified HR comparison remains unresolved; this result does not show POD loses to the bank.",'',
        '## Glossary','',
        '- arm: one frozen numerical model; k16 uses the archived weighted support, none uses every residual row.',
        '- own-objective Q1: trajectory validity, fresh stationarity and agreement with stricter solves of that same objective.',
        '- nonstationary step: a returned state whose fresh normalized objective gradient exceeds its frozen bar.',
        '- E0: the approved refined reference, with conditional numerical uncertainty.',
        '- median / worst error: middle / largest error among the diagnostic cases.',
        '- field distance: largest output-field difference divided by the common initial-field norm.',
        '- native-tight: the stricter native solver; the lower bound uses its saved distance from production.',
        '- gradient discrepancy: sampled versus full-residual gradients in common full-grid objective units.',
        '- full stationarity: the full-residual gradient norm normalized by full Jacobian and residual norms.',
        '- miss: a step that fails the corresponding frozen numerical bar.',
        '- ABA panel: k16, matching-none, k16 timed successively on one GPU to check drift.',
        '- HR: hyper-reduction, using weighted selected residual rows.',
        '- GPU-hours: scheduler-allocated GPU time including setup and diagnostics.','']
    (h/'checks/k16fine01/report.md').write_text('\n'.join(lines))
    lp=h/'runs/JOB-LEDGER.json';ledger=json.loads(lp.read_text());job=next(x for x in ledger['jobs'] if x['attempt']=='k16fine01')
    job.update(state='COMPLETED_FAILED_HR_ADMISSION',exit_code=acct['ExitCode'],elapsed_seconds=elapsed,
        spent_gpu_hours=hours,complete_checkpoint=True,checksums_verified=True,results_audit='READY',full64_adapter_release=False)
    ledger['spent_gpu_hours']=sum(x['spent_gpu_hours'] for x in ledger['jobs'])
    ledger['A3_replay_authorization'].update(state='COMPLETED_FAILED_HR_ADMISSION',results_audit='READY')
    lp.write_text(json.dumps(ledger,indent=2)+'\n')
    p=h/'runs/EXPORT.json';e=json.loads(p.read_text());old=e['a3_replay']
    old.update(state='COMPLETED_FAILED_HR_ADMISSION',results_audit='READY_SCOPED_FAILED_HR_ADMISSION',
        results_audit_path=str(audit),results_audit_sha256=sha(audit),no_rollout_yet=False,
        result_path=str(rp),result_sha256=sha(rp),summary_path=str(sp),summary_sha256=sha(sp),
        report_path=str(h/'checks/k16fine01/report.md'),report_sha256=sha(h/'checks/k16fine01/report.md'),
        own_objective_Q1=r['Q1'],admission=r['admission'],
        common_field_distances={i:row['k16_to_none'] for i,row in r['cases'].items()},
        diagnostic_native_tight_lower_bounds=low,
        matching_none_reproduction={i:row['none']['endpoint_reproduction'] for i,row in r['cases'].items()},
        timing_coverage_complete=r['timing_coverage_complete'],timing_repetitions=r['timing'],
        collection_manifest_path=str(d/'COLLECTION.sha256'),collection_manifest_sha256=sha(d/'COLLECTION.sha256'),
        resource=dict(job_id=r['job_id'],slurm_state=acct['State'],exit_code=acct['ExitCode'],elapsed_seconds=elapsed,
            gpu_hours=hours,gpu=r['gpu'],peak_device_bytes=r['peak_device_bytes'],peak_host_bytes=r['peak_host_bytes'],
            peak_scope='cumulative whole A3 job including both models, anchors and metrics; not standalone model memory'),
        scope='four preselected exposed cases only; no full64 HR release',full64_adapter_release=False,
        interpretation='Both common fields and additional POD-only gates fail; no POD accuracy loss or bank superiority follows.')
    e['lane_final_scope'].update(all_lane_jobs_terminal=True,all_authorized_work_complete=True,
        qualified_HR_comparison='unresolved; archived supports fail common fields at original and selected stronger endpoint; single larger constructor failed',
        actual_gpu_hours=ledger['spent_gpu_hours'],further_support_search_stopped=True,new_HR_adapter_released=False)
    p.write_text(json.dumps(e,indent=2)+'\n')
    print(json.dumps(dict(resource=old['resource'],scope=e['lane_final_scope']),indent=2))


if __name__=='__main__':main()
