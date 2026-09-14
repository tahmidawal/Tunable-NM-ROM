"""Shared deterministic cohort-wide summaries and predeclared selectors."""
from collections import Counter
import numpy as np


def summarize(rows,cfg):
    out={}
    for method in sorted({r['method'] for r in rows}):
        rs=[r for r in rows if r['method']==method];cases=sorted({r['case'] for r in rs})
        groups={c:[r for r in rs if r['case']==c] for c in cases}
        passed={c:all(r['solver_valid'] and r['conservative_physical_error']<=cfg['development_target'] and r['reference_delta']<=cfg['development_target']*cfg['reference_fraction'] for r in g) for c,g in groups.items()}
        out[method]=dict(gpu_median_ms=float(np.median([r['fused_device_seconds'] for r in rs])*1000),host_median_ms=float(np.median([r['total_seconds'] for r in rs])*1000),
            gpu_repetitions_ms=[r['fused_device_seconds']*1000 for r in rs],host_repetitions_ms=[r['total_seconds']*1000 for r in rs],
            worst_relative_error=max(r['physical_error'] for r in rs),median_relative_error=float(np.median([r['physical_error'] for r in rs])),
            worst_adjusted_relative_error=max(r['conservative_physical_error'] for r in rs),worst_error_case=max(rs,key=lambda r:r['physical_error'])['case'],
            all_cases_pass_target=all(passed.values()),failed_target_cases=sum(not x for x in passed.values()),invalid_invocations=sum(not r['solver_valid'] for r in rs),
            all_finite=all(r['finite'] for r in rs),invocations=len(rs),cases=len(cases),stop_reason_counts=dict(Counter(str(r['reason']) for r in rs)),
            gpu_outlier_count=int(sum(r['fused_device_seconds']>1.5*np.median([q['fused_device_seconds'] for q in groups[r['case']]]) for r in rs)),
            host_outlier_count=int(sum(r['total_seconds']>1.5*np.median([q['total_seconds'] for q in groups[r['case']]]) for r in rs)))
        if 'stationary' in rs[0]:
            out[method].update(stationary_invocations=sum(r['stationary'] for r in rs),all_stationary=all(r['stationary'] and r['all_start_linear_valid'] for r in rs),
                selected_attempts_median=float(np.median([r['attempts'] for r in rs])),total_attempts_median=float(np.median([r['total_attempts'] for r in rs])),
                maximum_stationarity=max(r['stationarity'] for r in rs),all_start_stop_reason_counts=dict(Counter(str(s['reason']) for r in rs for s in r['starts'])))
    return out


def select(methods,preset_ids):
    candidates=[m for m in preset_ids if methods[m]['all_finite']]
    stationary=[m for m in candidates if methods[m]['all_stationary']]
    def fastest(ms):return min(ms,key=lambda m:(methods[m]['gpu_median_ms'],m)) if ms else None
    def accurate(ms):return min(ms,key=lambda m:(methods[m]['worst_relative_error'],methods[m]['gpu_median_ms'],m)) if ms else None
    return dict(fastest_gpu=fastest(candidates),most_accurate=accurate(stationary),unrestricted_accuracy=accurate(candidates),
        fastest_passing=fastest([m for m in candidates if methods[m]['all_cases_pass_target']]),
        fastest_stationary=fastest(stationary))
