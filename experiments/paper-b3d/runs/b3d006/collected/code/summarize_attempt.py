"""Generate compact summaries from the complete raw timed invocation records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def panel(path):
    r=json.loads((path/'result.json').read_text());raw=r['invocations']+r.get('operator_invocations',[])
    rows=[]
    for method in sorted({v['method'] for v in raw}):
        selected=[v for v in raw if v['method']==method];times=np.asarray([v['gpu_ms'] for v in selected]);median=float(np.median(times))
        accepted=[v.get('stationary',v.get('converged')) for v in selected]
        case_errors=[max(v['worst_evolved'] for v in selected if v['case']==case) for case in sorted({v['case'] for v in selected})]
        rows.append(dict(method=method,n=len(selected),cases=len(case_errors),median_ms=median,
            timing_outliers_above_1p5_median=int(np.sum(times>1.5*median)),timing_repetitions_ms=times.tolist(),
            worst_all=max(v['worst_all'] for v in selected),worst_evolved=max(case_errors),median_evolved=float(np.median(case_errors)),
            stationary_or_converged=None if all(v is None for v in accepted) else sum(bool(v) for v in accepted)))
    return dict(scope=r['comparison_scope'],rows=rows,source_result_sha256=hashlib.sha256((path/'result.json').read_bytes()).hexdigest(),
        source=r['commit'],job_id=r['job_id'],backend=r['backend'],gpu=r['gpu'],bank_floor=r.get('bank_floor'),
        representation=r.get('representation'),local_audit=json.loads((path/'audit-local.json').read_text()),
        stationarity_audit=json.loads((path/'audit-stationarity-local.json').read_text()),final_cohort_unopened=r['final_cohort_unopened'])


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();root=Path(__file__).parent/'runs'/a.attempt
    collected=json.loads((root/'COLLECTED.json').read_text());assert collected['checksums_passed'] and collected['independent_audit_passed']
    out=root/'collected/out'
    result=panel(out if (out/'result.json').exists() else out/'seed0')
    result.update(checksums_passed=collected['checksums_passed'],remote_directory_removed=collected['remote_directory_removed'])
    if (out/'head64/result.json').exists():result['head64']=panel(out/'head64')
    if (out/'seed1/result.json').exists():
        result['seed1']=panel(out/'seed1')
        result['seed_reporting']='seed0 is the original-bank primary recipe; seed1 is an independent initialization confirmation, not a selected best seed'
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['job_id','source','checksums_passed','remote_directory_removed']},indent=2))


if __name__=='__main__':main()
