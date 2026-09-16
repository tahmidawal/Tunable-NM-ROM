"""Independent NumPy audit of the Poisson prior dial. No JAX, no GPU.

`dst_direct` is the exact discrete solution on the working mesh, so it defines the
same-grid metric; the physical error is against the refinement chain. Gate (iii)
compares lambda = infinity with the pabl01 head ablation's arm (a) on the same
sources, and reports the lambda -> 0 end against its free-bank arm (d).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / 'experiments/head-ablation/artifacts/pabl01/result.json'
GATE_TOLERANCE = 1e-9


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', default=None)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    checks, fail = {}, []

    def gate(name, ok, detail=None):
        checks[name] = dict(passed=bool(ok), detail=detail)
        if not ok:
            fail.append(name)

    def note(name, detail):
        checks[name] = dict(passed=None, detail=detail)

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('bank_factorization',
         all(b['factorization_relative'] < 1e-12 and b['orthonormality_deviation'] < 1e-12
             for b in r['bank'].values()),
         {k: dict(factorization=b['factorization_relative'],
                  orthonormality=b['orthonormality_deviation'],
                  By_condition=b['By_condition'], overdetermined=b['overdetermined'])
          for k, b in r['bank'].items()})

    inv = r['invocations']
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
    gate('every_invocation_paired',
         all('physical_error' in x and 'total_seconds' in x and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['K'] for x in roms))
    gate('infinite_lambda_has_no_correction',
         all(x['correction_norm'] == 0.0 for x in roms if x['lambda_rel'] is None))
    gate('lambda_scaling_consistent',
         all(x['lambda_absolute'] is None
             or abs(x['lambda_absolute'] - x['lambda_rel'] * x['sigma'] ** 2)
             <= 1e-12 * max(abs(x['lambda_absolute']), 1e-300) for x in roms))
    note('free_bank_limit_reachable',
         {k: b['overdetermined'] for k, b in r['bank'].items()})
    note('rom_stationarity_beyond_tolerance',
         sorted({(x['intervals'], x['name']) for x in roms
                 if x['stationarity'] > r['config']['stationarity_tolerance'] * (1 + 1e-7)}))

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

            base = {(x['intervals'], x['case']): x['artifact'] for x in inv
                    if x['name'] == 'dst_direct'}
            for x in inv:
                g = field_of(base[(x['intervals'], x['case'])])
                f = field_of(x['artifact'])
                x['same_grid_error'] = float(np.linalg.norm(f - g) / max(np.linalg.norm(g), 1e-300))

    table = {}
    for x in inv:
        key = (x['intervals'], x['name'])
        t = table.setdefault(key, dict(
            intervals=x['intervals'], arm=x['name'], kind=x['kind'],
            lambda_rel=x.get('lambda_rel'), lambda_absolute=x.get('lambda_absolute'),
            sigma=x.get('sigma'), M=x.get('M'), K=x.get('K'), ms=[], device_ms=[],
            case_error={}, case_same_grid={}, iterations=[], stationarity=[], reasons=[],
            correction=[]))
        t['ms'].append(x['total_seconds'] * 1e3)
        if 'fused_device_seconds' in x:
            t['device_ms'].append(x['fused_device_seconds'] * 1e3)
        t['case_error'][x['case']] = x['physical_error']
        if 'same_grid_error' in x:
            t['case_same_grid'][x['case']] = x['same_grid_error']
        if x['kind'] == 'rom':
            t['iterations'].append(x['iterations'])
            t['stationarity'].append(x['stationarity'])
            t['reasons'].append(x['reason'])
            t['correction'].append(x['relative_correction'])
    recon = {x['intervals']: x for x in r['reconstruction']}
    rows = []
    for key, t in sorted(table.items()):
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        sg = t['case_same_grid']
        sgv = np.array([sg[c] for c in sorted(sg)]) if sg else None
        rc = recon.get(t['intervals'])
        if t['kind'] != 'rom':
            floor = best = None
        elif t['lambda_rel'] is None:
            floor, best = rc['worst_bank_projection'], rc['worst_best_found']
        else:
            floor = best = rc['worst_bank_projection']
        rows.append(dict(
            intervals=t['intervals'], arm=t['arm'], kind=t['kind'], lambda_rel=t['lambda_rel'],
            lambda_absolute=t['lambda_absolute'], sigma=t['sigma'], M=t['M'], K=t['K'],
            worst_error_percent=float(np.max(errs) * 100),
            median_error_percent=float(np.median(errs) * 100),
            worst_same_grid_percent=(float(np.max(sgv) * 100) if sgv is not None else None),
            median_same_grid_percent=(float(np.median(sgv) * 100) if sgv is not None else None),
            worst_bank_projection_percent=(None if floor is None else float(floor * 100)),
            worst_best_found_percent=(None if best is None else float(best * 100)),
            max_relative_correction_percent=(float(np.max(t['correction']) * 100)
                                             if t['correction'] else None),
            median_query_ms=median(t['ms']),
            median_device_ms=(median(t['device_ms']) if t['device_ms'] else None),
            query_repetitions=len(t['ms']),
            median_iterations=(median(t['iterations']) if t['iterations'] else None),
            max_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
            all_completed=(bool(all(x in (1, 2, 4) for x in t['reasons'])) if t['reasons'] else None),
            stop_reasons=(sorted(set(t['reasons'])) if t['reasons'] else None)))
    rows.sort(key=lambda x: (x['intervals'], x['kind'] != 'rom',
                             -(np.inf if x['lambda_rel'] is None else x['lambda_rel'])))
    checks['arm_table'] = rows

    # ------------- gate (iii): lambda = infinity against pabl01 arm (a) -------
    if ABLATION.exists():
        ab = json.loads(ABLATION.read_text())
        detail = {}
        worst = 0.
        for n in r['config']['intervals']:
            want = {x['case']: x['physical_error'] for x in ab['invocations']
                    if x['name'] == 'a_neural' and x['intervals'] == n}
            got = {x['case']: x['physical_error'] for x in inv
                   if x['name'] == 'laminf' and x['intervals'] == n}
            wh = {x['case']: x['field_sha256'] for x in ab['invocations']
                  if x['name'] == 'a_neural' and x['intervals'] == n}
            gh = {x['case']: x['field_sha256'] for x in inv
                  if x['name'] == 'laminf' and x['intervals'] == n}
            shared = sorted(set(want) & set(got))
            deltas = [abs(got[c] - want[c]) / max(want[c], 1e-300) for c in shared]
            worst = max(worst, max(deltas) if deltas else np.inf)
            detail[str(n)] = dict(compared=len(shared),
                                  worst_relative_delta=(max(deltas) if deltas else None),
                                  bitwise_identical_fields=sum(1 for c in shared if wh[c] == gh[c]))
        gate('laminf_reproduces_pabl01_arm_a', worst < GATE_TOLERANCE,
             dict(worst_relative_delta=worst, tolerance=GATE_TOLERANCE, per_mesh=detail,
                  ablation_gpu=ab.get('gpu'), dial_gpu=r.get('gpu')))
        free = {}
        for n in r['config']['intervals']:
            want = {x['case']: x['physical_error'] for x in ab['invocations']
                    if x['name'] == 'd_freebank' and x['intervals'] == n}
            small = min([x for x in r['config']['lambda_rel'] if x is not None])
            got = {x['case']: x['physical_error'] for x in inv
                   if x['lambda_rel'] == small and x['intervals'] == n}
            shared = sorted(set(want) & set(got))
            free[str(n)] = dict(lambda_rel=small, compared=len(shared),
                                worst_relative_delta=(max(abs(got[c] - want[c]) / max(want[c], 1e-300)
                                                          for c in shared) if shared else None))
        note('smallest_lambda_versus_pabl01_free_bank', free)
        gate('cohort_identical',
             bool(np.allclose(np.asarray(ab['cohort']['parameters']),
                              np.asarray(r['cohort']['parameters']))))

    out = dict(result=str(a.result), result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               failed=fail, passed=not fail, checks=checks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, indent=2))
    for name, v in checks.items():
        if isinstance(v, dict) and 'passed' in v:
            print({True: 'PASS ', False: 'FAIL ', None: 'NOTE '}[v['passed']], name, v['detail'])
    assert not fail, fail


if __name__ == '__main__':
    main()
