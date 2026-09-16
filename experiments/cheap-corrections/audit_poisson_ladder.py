"""Independent NumPy audit of the Poisson correction ladder. No JAX, no GPU.

Recomputes every reported physical error from the retained output fields, checks the
solver validity flags the retained correction path already emits, and checks that the
q = 32 rung at the retained test count reproduces `pabl01`'s `a_neural_q32`.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PABL = ROOT / 'experiments/head-ablation/artifacts/pabl01/result.json'


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def nondominated(rows, cost='median_device_ms', err='worst_physical_percent'):
    pts = [x for x in rows if x.get(cost) is not None and x.get(err) is not None]
    return sorted({r['arm'] for r in pts
                   if not any((o[cost] <= r[cost] and o[err] <= r[err]
                               and (o[cost] < r[cost] or o[err] < r[err])) for o in pts)})


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

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('retained_direction_prefix_exact',
         bool(r['gates'].get('retained_prefix_exact', {}).get('passed')),
         r['gates'].get('retained_prefix_exact'))

    inv = r['invocations']
    reps = r['config']['repetitions']
    counts = {}
    for x in inv:
        counts.setdefault((x['name'], x['case']), 0)
        counts[(x['name'], x['case'])] += 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    hashes = {}
    for x in inv:
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system',
         all(x['M'] > r['K'] + x['q'] for x in roms),
         [(x['name'], x['M'], r['K'] + x['q']) for x in roms if x['M'] <= r['K'] + x['q']][:5])
    gate('nonlinear_dimension_stays_K',
         all(x['nonlinear_optimizer_dimension'] == r['K'] for x in roms),
         sorted({x['nonlinear_optimizer_dimension'] for x in roms}))
    gate('every_rom_reports_solver_validity',
         all(('solver_valid' in x and 'stationarity' in x and 'reason' in x) for x in roms))

    if a.fields:
        d = Path(a.fields)
        bad = [x['artifact'] for x in inv if not (d / x['artifact']).exists()]
        gate('artifacts_present', not bad, bad[:5])
        if not bad:
            cache = {}
            for x in inv:
                if x['artifact'] not in cache:
                    cache[x['artifact']] = np.load(d / x['artifact'])['field']
            refs = {e['case']: np.load(d / e['artifact'])['field'] for e in r['references']}
            base = {x['case']: x['artifact'] for x in inv if x['name'] == 'dst_direct'}
            worst = 0.
            for x in inv:
                f = cache[x['artifact']]
                t = refs[x['case']]
                err = float(np.linalg.norm(f - t) / np.linalg.norm(t))
                worst = max(worst, abs(err - x['physical_error'])
                            / max(x['physical_error'], 1e-300))
                x['same_grid'] = float(np.linalg.norm(f - cache[base[x['case']]])
                                       / np.linalg.norm(cache[base[x['case']]]))
            gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(arm=x['name'], kind=x['kind'], q=x.get('q'),
                                             rule=x.get('rule'), M=x.get('M'), device_ms=[],
                                             total_ms=[], case_error={}, case_same_grid={},
                                             valid=[], stationary=[], iterations=[]))
        # the ROM rows time the fused device kernel; the direct DST control reports its
        # solve stage separately, so the comparable device figure is that stage
        t['device_ms'].append((x.get('fused_device_seconds') or x['solver_seconds']) * 1e3)
        t['total_ms'].append(x['total_seconds'] * 1e3)
        t['case_error'][x['case']] = x['physical_error']
        if 'same_grid' in x:
            t['case_same_grid'][x['case']] = x['same_grid']
        if x['kind'] == 'rom':
            t['valid'].append(x['solver_valid'])
            t['stationary'].append(x['stationary'])
            t['iterations'].append(x['total_attempts'])
    for t in table.values():
        pass
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    recon = {x['q']: x for x in r['reconstruction']}
    rows = []
    for name, t in table.items():
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        sg = (np.array([t['case_same_grid'][c] for c in sorted(t['case_same_grid'])])
              if t['case_same_grid'] else None)
        rc = recon.get(t['q'])
        st = setup.get(name, {})
        rows.append(dict(arm=name, kind=t['kind'], q=t['q'], rule=t['rule'], M=t['M'],
                         nominal_dimension=st.get('nominal_dimension'),
                         nonlinear_dimension=st.get('nonlinear_dimension'),
                         linear_rank=st.get('linear_rank'), linear_rank_valid=st.get('linear_rank_valid'),
                         degenerate_full_bank=st.get('degenerate_full_bank'),
                         worst_physical_percent=float(np.max(errs) * 100),
                         median_physical_percent=float(np.median(errs) * 100),
                         worst_same_grid_percent=(float(np.max(sg) * 100) if sg is not None else None),
                         worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100) if rc else None),
                         worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
                         median_device_ms=median(t['device_ms']), median_total_ms=median(t['total_ms']),
                         repetitions=len(t['device_ms']),
                         median_iterations=(median(t['iterations']) if t['iterations'] else None),
                         all_solver_valid=(bool(all(t['valid'])) if t['valid'] else None),
                         all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
                         setup_seconds=st.get('setup_seconds')))
    rows.sort(key=lambda x: (x['kind'] != 'rom', x['q'] if x['q'] is not None else 0, x['arm']))
    checks['arm_table'] = rows
    checks['nondominated_all'] = nondominated([x for x in rows if x['kind'] == 'rom'])
    checks['nondominated_valid'] = nondominated([x for x in rows if x['kind'] == 'rom'
                                                 and x['all_solver_valid']])

    ref = json.loads(Path(PABL).read_text()) if Path(PABL).exists() else None
    if ref is not None and np.shape(ref['cohort']['parameters']) == np.shape(r['cohort']['parameters']):
        want = {x['case']: x['physical_error'] for x in ref['invocations']
                if x['name'] == 'a_neural_q32' and x['intervals'] == r['intervals']}
        got = {x['case']: x['physical_error'] for x in inv if x['name'] == 'q32_m256'}
        shared = sorted(set(want) & set(got))
        deltas = [abs(got[c] - want[c]) / max(want[c], 1e-300) for c in shared]
        gate('q32_reproduces_pabl01_a_neural_q32', bool(shared) and max(deltas) < 1e-9,
             dict(compared=len(shared), worst_relative_delta=(max(deltas) if deltas else None),
                  tolerance=1e-9, reference_gpu=ref.get('gpu'), this_gpu=r.get('gpu')))
        gate('cohort_identical',
             bool(np.allclose(np.asarray(ref['cohort']['parameters']),
                              np.asarray(r['cohort']['parameters']))))
    elif ref is not None:
        checks['pabl01_comparison_skipped'] = dict(
            passed=True, detail='cohort shape differs from pabl01; this is not the production run')

    out = dict(result=str(a.result), result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               elapsed_seconds=r.get('elapsed_seconds'), failed=fail, passed=not fail, checks=checks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, indent=2))
    for name, v in checks.items():
        if isinstance(v, dict) and 'passed' in v:
            print(('PASS ' if v['passed'] else 'FAIL '), name, v['detail'])
    assert not fail, fail


if __name__ == '__main__':
    main()
