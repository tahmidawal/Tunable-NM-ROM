"""Generate the two-lane comparison from exact retained run records."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    result = dict(status='complete: independent numerical and retention audits accepted; owned remote duplicates removed', lanes={}, scope=(
        'Six exposed development cases on one shared model; production stationarity only. '
        'Tight fine-baseline agreement, physical PDE accuracy and successor quadrature were not run after the cost stop.'))
    ledger = json.loads((HERE/'GPU-LEDGER.json').read_text())
    result['gpu_seconds'] = sum(e['actual_gpu_seconds'] for e in ledger['entries'])
    result['gpu_cap_seconds'] = ledger['cap_seconds']
    for name, arm in (('step', 'affine'), ('replay', 'replay')):
        path = HERE/f'checks/source-results/{name}.json'
        data = json.loads(path.read_text())
        assert data['complete'] and not data['gate']['passed'] and not data['continue_accuracy']
        timings = {(t['arm'], t['case'], t['block']): t for t in data['timings']}
        assert len(timings) == len(data['timings']) == 84
        ratios = []; per_case = []
        for case in range(6):
            paired = [timings[arm, case, b]['seconds']/timings['direct', case, b]['seconds'] for b in range(7)]
            assert np.isfinite(paired).all() and min(paired) > 0
            ratios.extend(paired); per_case.append(float(np.median(paired)))
        np.testing.assert_allclose(per_case, data['gate']['per_case_paired_median_ratios'], rtol=1e-13)
        totals = {}
        for method in ('direct', arm):
            rows = [r for r in data['records'] if r['arm'] == method]
            assert len(rows) == 6 and all(r['valid'] for r in rows)
            totals[method] = dict(fine_corrections=sum(r['fine_corrections'] for r in rows),
                fine_jacobians=sum(r['fine_J' if name == 'step' else 'fine_physics_J'] for r in rows))
            if method != 'direct':
                totals[method].update(
                    coarse_corrections=int(sum(r['auxiliary_sums']['aux_it'] if name == 'step' else r['coarse_corrections'] for r in rows)),
                    coarse_jacobians=sum(r['coarse_J' if name == 'step' else 'coarse_physics_J'] for r in rows),
                    accepted_guesses=int(sum(r['auxiliary_sums']['accepted'] if name == 'step' else r['accepted_candidates'] for r in rows)),
                    guard_residual_calls=sum(r['fine_guard_residuals' if name == 'step' else 'guard_residual_calls'] for r in rows))
        result['lanes'][name] = dict(source_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            job_id=data['job_id'], gpu=data['gpu'], settings=data['settings'], setup=data['setup'],
            compile_warm_seconds=data['compile_warm_seconds'], per_case_paired_median_ratios=per_case,
            median_ratio=float(np.median(per_case)), worst_case_ratio=max(per_case),
            slower_pairs=sum(x > 1 for x in ratios), pairs=len(ratios), counts=totals,
            controls_passed=all(t['identity'] and t['online_valid'] for t in data['timings']),
            strong_accuracy_checks=data['strong_accuracy_checks'])
    (HERE/'checks/comparison-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:dict(ratio=v['median_ratio'],counts=v['counts']) for k,v in result['lanes'].items()},indent=2))


if __name__ == '__main__': main()
