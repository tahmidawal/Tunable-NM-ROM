"""Independent NumPy audit of a completed Poisson head-ablation result. No JAX."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', default=None, help='directory holding the saved output .npz files')
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    checks, fail = {}, []

    def gate(name, ok, detail=None):
        checks[name] = dict(passed=bool(ok), detail=detail)
        if not ok:
            fail.append(name)

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)

    inv = r['invocations']
    gate('every_invocation_paired',
         all('physical_error' in x and 'total_seconds' in x and x['finite'] for x in inv))
    reps = r['config']['repetitions']
    counts = {}
    for x in inv:
        counts.setdefault((x['intervals'], x['name'], x['case']), 0)
        counts[(x['intervals'], x['name'], x['case'])] += 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    hashes = {}
    for x in inv:
        hashes.setdefault((x['intervals'], x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])

    setup = {(s['intervals'], s['arm']): s for s in r['arm_setup'] if 'arm' in s}
    roms = [x for x in inv if x['kind'] in ('rom', 'retained')]
    gate('overdetermined_weak_system',
         all(x['M'] > x['k'] for x in roms if x['M'] and x['k']),
         [(x['name'], x['M'], x['k']) for x in roms if x['M'] and x['k'] and x['M'] <= x['k']][:5])
    gate('retained_solver_valid',
         all(x.get('solver_valid', True) for x in inv if x['kind'] == 'retained'))
    gate('rom_stationary',
         all(x['stationarity'] <= r['config']['stationarity_tolerance'] * (1 + 1e-7)
             for x in inv if x['kind'] == 'rom'),
         sorted({x['name'] for x in inv if x['kind'] == 'rom'
                 and x['stationarity'] > r['config']['stationarity_tolerance'] * (1 + 1e-7)}))

    if a.fields:
        d = Path(a.fields)
        bad = [x['artifact'] for x in inv if not (d / x['artifact']).exists()]
        gate('artifacts_present', not bad, bad[:5])
        if not bad:
            cache = {}

            def field_of(name):
                if name not in cache:
                    cache[name] = np.load(d / name)['field']
                return cache[name]

            base = {(x['intervals'], x['case']): x['artifact'] for x in inv if x['name'] == 'dst_direct'}
            for x in inv:
                b = base.get((x['intervals'], x['case']))
                if b is not None:
                    f, g = field_of(x['artifact']), field_of(b)
                    # The direct transform solve IS the exact discrete solution on this
                    # mesh, so this isolates reduction error from discretization error.
                    x['same_grid_error'] = float(np.linalg.norm(f - g) / max(np.linalg.norm(g), 1e-300))

    table = {}
    for x in inv:
        key = (x['intervals'], x['name'])
        t = table.setdefault(key, dict(intervals=x['intervals'], arm=x['name'], kind=x['kind'],
                                       k=x.get('k'), family=x.get('family'), ms=[], case_error={},
                                       iterations=[], stationarity=[], reasons=[]))
        t['ms'].append(x['total_seconds'] * 1e3)
        t['case_error'][x['case']] = x['physical_error']
        if 'same_grid_error' in x:
            t.setdefault('case_same_grid', {})[x['case']] = x['same_grid_error']
        if x['kind'] == 'rom':
            t['iterations'].append(x['iterations'])
            t['stationarity'].append(x['stationarity'])
            t['reasons'].append(x['reason'])
        elif x['kind'] == 'retained':
            t['iterations'].append(x['jacobians'])
            t['stationarity'].append(x['stationarity'])
            t['reasons'].append(x['reason'])
    recon = {(x['intervals'], x['arm']): x for x in r['reconstruction']}
    rows = []
    for key, t in sorted(table.items()):
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        rc = recon.get(key)
        sg = t.get('case_same_grid')
        sgv = np.array([sg[c] for c in sorted(sg)]) if sg else None
        rows.append(dict(intervals=t['intervals'], arm=t['arm'], kind=t['kind'], k=t['k'],
                         family=t['family'],
                         worst_error_percent=float(np.max(errs) * 100),
                         median_error_percent=float(np.median(errs) * 100),
                         worst_same_grid_percent=(float(np.max(sgv) * 100) if sgv is not None else None),
                         median_same_grid_percent=(float(np.median(sgv) * 100) if sgv is not None else None),
                         worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100) if rc else None),
                         worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
                         median_query_ms=median(t['ms']), query_repetitions=len(t['ms']),
                         median_iterations=(median(t['iterations']) if t['iterations'] else None),
                         max_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
                         stop_reasons=sorted(set(t['reasons'])) if t['reasons'] else None,
                         M=(setup.get(key, {}) or {}).get('M')))
    checks['arm_table'] = rows

    out = dict(result=str(a.result), result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               failed=fail, passed=not fail, checks=checks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, indent=2))
    for name, v in checks.items():
        if isinstance(v, dict) and 'passed' in v:
            print(('PASS ' if v['passed'] else 'FAIL '), name, v['detail'])
    assert not fail, fail


if __name__ == '__main__':
    main()
