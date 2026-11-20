"""Independently reconstruct retained timing/accuracy statistics and qualification counts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    run=args.run
    out=run/'output'
    result=json.loads((out/'paired/result.json').read_text())
    provenance=json.loads((out/'provenance.json').read_text())
    cfg=provenance['config']
    refs=json.loads((out/'references.json').read_text())
    q=json.loads((out/'numerical_qualification.json').read_text())
    rows=[json.loads(s) for s in (out/'paired/invocations.jsonl').read_text().splitlines()]
    errors=[]
    names=set(result['confirmation'])
    if len(names)!=6 or len(rows)!=516 or len(refs)!=16 or len(q)!=64:
        errors.append('Unexpected subject/invocation/reference/qualification count')
    for block,count in [('sentinel_before',18),('fresh_confirmation',480),('sentinel_after',18)]:
        if sum(r['block']==block for r in rows)!=count:errors.append('Block count '+block)
    if not all(r['numerical_valid'] if 'numerical_valid' in r else True for r in rows):
        errors.append('Unexpected unclassified numerical validity field')
    checks={}
    for name in names:
        method=result['confirmation'][name]
        retained=[r for r in rows if r['subject']==name and r['block']=='fresh_confirmation']
        failures=[r for r in retained if r['status']!='ok']
        if len(retained)!=80 or method['failure_count']!=len(failures):errors.append('Count mismatch '+name)
        case_times=[];case_errors=[]
        for i in range(16):
            rr=[r for r in retained if r['case_id']==i and r['status']=='ok']
            if not rr:continue
            stored=method['cases'][str(i)]
            times=np.array([r['elapsed_ms'] for r in rr])
            ee=max(r['evolved_max_error'] for r in rr)
            if not np.array_equal(times,stored['repetitions_ms']):errors.append('Lost/reordered repetitions '+name)
            if float(np.median(times))!=stored['median_ms'] or ee!=stored['evolved_max_error']:errors.append('Case summary mismatch '+name)
            for r in rr:
                if max(r['relative_errors'][1:])!=r['evolved_max_error']:errors.append('Accuracy aggregation mismatch '+name)
                if r['output_dtype']!='float64' or r['output_shape']!=[6,3,64,64,64]:errors.append('Output contract mismatch '+name)
            case_times.append(float(np.median(times)));case_errors.append(ee)
        if case_times and (float(np.median(case_times))!=method['median_case_ms'] or float(np.median(case_errors))!=method['median_case_evolved_max_error'] or max(case_errors)!=method['worst_evolved_error']):errors.append('Cohort summary mismatch '+name)
        method_q=[r for r in q if r['subject']==name]
        checks[name]={'invocations':len(retained),'failures':len(failures),
                      'numerical_checks':len(method_q),'numerical_failures':sum(not r['passed'] for r in method_q),
                      'fractional_reference_failures':sum(not (r['dt_refinement_max'] <= min(cfg['reference_absolute_gate'],cfg['reference_fraction_of_target_error']*method['cases'][str(r['case'])]['evolved_max_error'])) for r in refs if str(r['case']) in method['cases']),
                      'timing_outliers':method['outlier_count_tukey']}
    for target,groups in result['comparisons'].items():
        for group,row in groups.items():
            base=result['confirmation'][row['comparator']]
            tar=result['confirmation'][target]
            paired=[base['cases'][str(i)]['median_ms']/tar['cases'][str(i)]['median_ms'] for i in row['paired_case_ids']]
            if paired!=row['paired_case_ratios'] or float(np.median(paired))!=row['median_paired_case_ratio']:errors.append('Paired ratio mismatch')
            for arm in [target,row['comparator']]:
                a=result['confirmation'][arm];lim=cfg['frozen_thresholds'][target]
                expected=bool(a['complete'] and result['numerical_pass'][arm] and a['median_case_evolved_max_error']<=lim['median_case_evolved_max_error'] and a['worst_evolved_error']<=lim['worst_evolved_error'])
                if expected!=result['frozen_threshold_eligibility'][target][arm]:errors.append('Fixed threshold mismatch')
    log=(run/'job.log').read_text()
    if 'jax_backend=gpu' not in log or 'RUN_EXIT=0' not in log:errors.append('GPU/completion log missing')
    bad_phrases=['captured-large-constant','failed to allocate','out of memory','No space left on device']
    if any(p.lower() in log.lower() for p in bad_phrases):errors.append('Invalid runtime warning')
    if provenance['backend']!='gpu' or not provenance['x64'] or provenance['precision']!='highest':errors.append('Precision/backend failed')
    for rel,rec in provenance['source_provenance']['files'].items():
        p=run/'code/experiments/jcp-cost-confirm'/rel
        if hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256']:errors.append('Copied source hash mismatch '+rel)
    source_hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'paired/result.json',out/'paired/invocations.jsonl',out/'provenance.json',out/'numerical_qualification.json',out/'references.json',run/'job.log']}
    record={'schema':'fresh-cost-audit-v1','passed':not errors,'errors':errors,'subjects':checks,
            'invocations':len(rows),'reference_cases':len(refs),'numerical_checks':len(q),
            'all_numerical_checks_pass':result['all_numerical_checks_pass'],
            'reference_max_refinement':max(r['dt_refinement_max'] for r in refs),
            'max_solver_refinement':max(r['solver_refinement_difference'] for r in q),
            'source_hashes':source_hashes}
    (run/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
