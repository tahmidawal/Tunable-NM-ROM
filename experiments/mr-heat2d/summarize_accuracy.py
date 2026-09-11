"""Generate exact cohort panels exclusively from the audited invocation records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(record):
    out=record/'archive/outputs'; result_path=out/'results.json'
    result=json.loads(result_path.read_text()); audit_path=record/'analysis/audit.json'
    audit=json.loads(audit_path.read_text())
    assert audit['passed'] and audit['result_sha256']==hashlib.sha256(result_path.read_bytes()).hexdigest()
    scopes={'all_development':lambda c:True,'opened_development':lambda c:c!='confirmation_development',
            'original':lambda c:c=='original','fresh':lambda c:c=='fresh',
            'confirmation_development':lambda c:c=='confirmation_development'}
    rows=[]; lookup={(r['intervals'],r['case'],r['method']):r for r in result['rows']}
    for scope,predicate in scopes.items():
        for n in result['settings']['requested_intervals']:
            for method in result['settings']['methods']:
                native=[r for r in result['rows'] if r['intervals']==n and r['method']==method and predicate(r['cohort'])]
                reps=[p for r in native for p in r['repetitions']]
                case_errors=[max(max(p['vs_physical']['relative_current']) for p in r['repetitions']) for r in native]
                row=dict(scope=scope,intervals=n,method=method,cases=len(native),invocations=len(reps),
                    case_ids=[r['case'] for r in native],case_worst_relative_errors=case_errors,
                    physical_error_median=float(np.median(case_errors)),physical_error_worst=max(case_errors),
                    initial_error_worst=max(p['vs_physical']['relative_current'][0] for p in reps),
                    later_error_worst=max(max(p['vs_physical']['relative_current'][1:]) for p in reps),
                    initial_nonstationary_fits=0,step_nonstationary_fits=0,total_initial_attempts=0,total_step_attempts=0,
                    initial_exit_counts={},step_exit_counts={},cg_nonconverged_steps=0,
                    empirical_gate_failed_cases=[],numerical_gate_failed_cases=[])
                for r in native:
                    empirical_fail=False; numerical_fail=False
                    delta=max(result['cases'][r['case']]['reference_refinement']['relative_current'])
                    for p in r['repetitions']:
                        empirical_fail|=(max(p['vs_physical']['relative_current'])+delta)/(1-delta)>result['settings']['accuracy_target']
                        if p['solver']:
                            for prefix,key in [('initial','initial_fits'),('step','steps')]:
                                infos=np.asarray(p['solver'][key]); failed=int(np.sum(infos[:,2]!=1))
                                row[prefix+'_nonstationary_fits']+=failed; numerical_fail|=failed>0
                                row['total_'+prefix+'_attempts']+=int(np.sum(infos[:,0]))
                                for reason in infos[:,2]:
                                    reason=str(int(reason)); counts=row[prefix+'_exit_counts']; counts[reason]=counts.get(reason,0)+1
                        if 'cg_steps' in p:
                            failed=int(np.sum(np.asarray(p['cg_steps'])[:,3]!=1)); row['cg_nonconverged_steps']+=failed; numerical_fail|=failed>0
                    if empirical_fail:row['empirical_gate_failed_cases'].append(r['case'])
                    if numerical_fail:row['numerical_gate_failed_cases'].append(r['case'])
                row['physical_gate_passed']=not row['empirical_gate_failed_cases']
                row['numerical_gate_passed']=not row['numerical_gate_failed_cases']
                row['qualified']=row['physical_gate_passed'] and row['numerical_gate_passed']
                for phase in ['device','host']:
                    key=phase+'_seconds'; values=[p['phases'][key] for p in reps]
                    row[phase+'_median_ms']=float(np.median(values)*1000)
                    row[phase+'_median_case_medians_ms']=float(np.median([np.median([p['phases'][key] for p in r['repetitions']]) for r in native])*1000)
                    row[phase+'_outliers']=sum(sum(p['phases'][key]>1.5*np.median([q['phases'][key] for q in r['repetitions']]) for p in r['repetitions']) for r in native)
                    for comparator in ['nmrom_frozen','fom_cg_cn','fom_cg_cn_tol1e2','fom_same_grid']:
                        other=[lookup[n,r['case'],comparator] for r in native]
                        med=float(np.median([p['phases'][key] for r in other for p in r['repetitions']]))
                        row[phase+'_ratio_'+comparator]=med/np.median(values)
                        paired=[lookup[n,r['case'],comparator]['repetitions'][i]['phases'][key]/p['phases'][key] for r in native for i,p in enumerate(r['repetitions'])]
                        row[phase+'_median_paired_ratio_'+comparator]=float(np.median(paired))
                rows.append(row)
    panel=dict(schema='heat-accuracy-panel-v1',status='audited_development_single_training_seed',
        source_commit=result['source_manifest']['source_commit'],metadata=result['metadata'],
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        primary_method=result['settings']['primary_method'],timed_invocations=result['timed_invocations'],
        rows=rows,models=result['models'],training_baseline=result['training_baseline'],
        bank_gate=result['bank_gate'],representation_summaries=audit['representation_summaries'],
        fine_representation_summaries=audit['fine_representation_summaries'],final_cohort_opened=False,
        limitations=['One training/minibatch seed; restricted single-bump family; no final paper cases.',
                    'New development draws were pinned before training, with all scheduled endpoints retained.',
                    'Initial-plus-tail is the predeclared primary; initial-only is a separately reported ablation.',
                    'Numerical gate requires every attempted online initial fit and step to be stationary.',
                    'CG speed ratios name their tolerances; direct sine-transform FOM is retained.',
                    'Physical uncertainty is an empirical spectral-refinement estimate, not a rigorous bound.'])
    (record/'analysis/panel.json').write_text(json.dumps(panel,indent=2)+'\n')
    notes=['# Heat accuracy from fixed-bank initial and tail training','',
           'These audited development results compare matched training procedures while retaining the nonlinear decoder and weak heat solver. They are provisional for publication because only one training seed and a restricted family have been tested; final cases remain unopened.','',
           f"Source `{panel['source_commit']}`, GPU job `{panel['metadata']['job_id']}`, {panel['metadata']['gpu']}. All {panel['timed_invocations']} timed invocations are retained, with cost and field error from the same invocation.",'',
           'The initial-plus-tail arm is the predeclared primary. The initial-only arm isolates the effect of emphasizing the initial field; it is not a newly selected primary. The frozen model on the older opened cohort must be compared separately from its newly added development cases.','']
    for scope in ['opened_development','confirmation_development','all_development']:
        notes.extend([f'## {scope.replace("_"," ").capitalize()}','',
                      '| N | Method | GPU ms | Host ms | Median / worst error % | Tight / loose CG GPU ratio | Nonstationary initial / steps | Qualifies at 5% |',
                      '|---|---|---|---|---|---|---|---|'])
        for r in rows:
            if r['scope']!=scope:continue
            notes.append(f"| {r['intervals']} | {r['method']} | {r['device_median_ms']:.6f} | {r['host_median_ms']:.6f} | {100*r['physical_error_median']:.6f} / {100*r['physical_error_worst']:.6f} | {r['device_ratio_fom_cg_cn']:.6f} / {r['device_ratio_fom_cg_cn_tol1e2']:.6f} | {r['initial_nonstationary_fits']} / {r['step_nonstationary_fits']} | {r['qualified']} |")
        notes.append('')
    notes.extend(['## Training and representation','',
                  '| Model | Full / initial / later training relative MSE | Worst initial / later diagnostic error % | Selected diagnostic nonstationary fits |',
                  '|---|---|---|---|'])
    for t,r in zip(audit['training_metric_checks'],audit['representation_summaries']):
        assert t['model']==r['model']
        notes.append(f"| {t['model']} | {t['full_relative_mse']:.12g} / {t['initial_relative_mse']:.12g} / {t['later_relative_mse']:.12g} | {100*r['worst_initial_error']:.6f} / {100*r['worst_later_error']:.6f} | {r['nonstationary_selected']} |")
    notes.extend(['',f"The unchanged bank's worst training-mesh development projection error is {100*panel['bank_gate']['worst_error']:.6f}%. Fine-mesh per-case full-field projection and every attempted fit's gradient are retained in `analysis/panel.json` and the raw result. Diagnostic best-found fits are not certified global minima.",'',
                  'GPU times are pooled repetition medians. Tight CG uses tolerance 1e-6; loose CG uses 1e-2, both with the matched Crank–Nicolson time step. Ratios above one favor the indicated method. Every repetition is retained; an outlier exceeds 1.5 times the median of its own case/method/mesh group. Exact outlier counts, solver exits and paired ratios are in `analysis/panel.json`.','',
                  '## Plain-language glossary','',
                  '- **N:** intervals along each spatial axis; full output contains the interior grid.',
                  '- **GPU / host ms:** blocked full GPU input-to-output query time / the same invocation including input and output transfers.',
                  '- **Median / worst error:** median or largest case error, where each case uses its worst output time and all retained repetitions.',
                  '- **CG ratio:** the named full-order conjugate-gradient solver time divided by this method time.',
                  '- **Stationarity / qualifies:** sufficiently small reduced objective gradient / all attempted solves stationary and every field meeting the empirical physical target.',
                  '- **Bank / head / MSE:** learned spatial functions / nonlinear coefficient map / mean squared relative reconstruction error.',
                  '- **Opened / confirmation development:** previous development cases / new draws fixed before this training round; neither is the final paper cohort.',
                  '- **Primary / ablation:** setting declared before results / controlled alternative isolating a training change.',''])
    (record/'analysis/summary.md').write_text('\n'.join(notes))
    print(json.dumps(dict(rows=len(rows),timed_invocations=panel['timed_invocations'],panel=str(record/'analysis/panel.json'))))
    return panel

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('record',type=Path);summarize(p.parse_args().record)
