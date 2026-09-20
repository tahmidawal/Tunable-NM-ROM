"""Generate same-job comparison rows from an independently audited panel."""
import argparse
import hashlib
import json
from pathlib import Path


def generate(attempt):
    audit=json.loads((attempt/'audit-local.json').read_text());assert audit['passed']
    out=attempt/'archive/out'
    record=json.loads((out/'result.json').read_text())
    summary=json.loads((out/'summary.json').read_text());assert record['complete'] and summary['complete']
    rows=[]
    for row in summary['rows']:
        if row['method'].startswith('cg_'):continue
        n=row['intervals'];same=[x for x in summary['rows'] if x['intervals']==n]
        dst=next(x for x in same if x['method']=='dst_exact')
        candidates=[x for x in same if x['method'].startswith('cg_identity_plain_')
                    and x['cg_failed_invocations']==0 and x['nonfinite_cases']==0
                    and x['same_grid_error_worst']<=row['same_grid_error_worst']]
        cg=min(candidates,key=lambda x:x['device_ms_median']) if candidates else None
        result={k:row[k] for k in ['intervals','method','cases','device_ms_median','total_ms_median',
            'same_grid_error_mean','same_grid_error_median','same_grid_error_worst','physical_error_worst',
            'nonfinite_cases','nonstationary_cases','timing_outliers_above_1p5_median']}
        result.update(dst_device_ms=dst['device_ms_median'],
            speedup_vs_dst_device=dst['device_ms_median']/row['device_ms_median'],
            speedup_vs_dst_total=dst['total_ms_median']/row['total_ms_median'],
            matched_cg_method=cg['method'] if cg else None,
            matched_cg_error_worst=cg['same_grid_error_worst'] if cg else None,
            matched_cg_device_ms=cg['device_ms_median'] if cg else None,
            speedup_vs_matched_cg_device=cg['device_ms_median']/row['device_ms_median'] if cg else None,
            speedup_vs_matched_cg_total=cg['total_ms_median']/row['total_ms_median'] if cg else None)
        rows.append(result)
    return dict(schema='poisson3d-paper-comparisons-v1',source_commit=record['source_commit'],
        job_id=record['job_id'],gpu=record['gpu'],final_cohort_opened=record['final_cohort_opened'],
        result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest(),
        comparison_policy='Fastest device-median efficient CG setting with zero failed invocations and worst same-grid field error no larger than the method; total-time ratio uses that same setting. Null means no measured CG setting qualifies. Traced CG excluded from efficient baseline selection. Ratios are baseline/method within one job.',
        error_policy='Relative L2, maximum over held-out cases; CG uses worst repetition per case. Mean and median are also retained.',
        rows=rows,cg_controls=[x for x in summary['rows'] if x['method'].startswith('cg_')])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(generate(Path(a.attempt)),indent=2)+'\n')
