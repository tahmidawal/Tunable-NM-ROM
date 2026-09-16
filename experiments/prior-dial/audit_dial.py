"""Independent NumPy audit of the Burgers prior dial. No JAX, no GPU.

Recomputes every reported error from the retained output fields, measures each
arm against the converged same-mesh full-order solve at every output time, checks
lambda = infinity against the head-ablation job's arm (a) on the same cases, and
builds the arm table the report reads. Nothing here imports the job's own code.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / 'experiments/head-ablation/artifacts/abl01/result.json'
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
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r['checkpoint_sha256_after'])
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    gate('bank_factorization', r['bank']['factorization_relative'] < 1e-12
         and r['bank']['orthonormality_deviation'] < 1e-12,
         dict(factorization=r['bank']['factorization_relative'],
              orthonormality=r['bank']['orthonormality_deviation'],
              R_condition=r['bank']['R_condition']))

    inv = r['invocations']
    reps = r['config']['reps']
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
    gate('every_invocation_paired', all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))

    roms = [x for x in inv if x['kind'] == 'rom']
    gate('lambda_scaling_consistent',
         all(x['lambda_absolute'] is None
             or abs(x['lambda_absolute'] - x['lambda_rel'] * x['sigma'] ** 2)
             <= 1e-12 * max(abs(x['lambda_absolute']), 1e-300) for x in roms))
    gate('infinite_lambda_has_no_correction',
         all(x['max_correction_norm'] == 0.0 for x in roms if x['lambda_rel'] is None),
         sorted({x['name'] for x in roms
                 if x['lambda_rel'] is None and x['max_correction_norm'] != 0.0}))
    gate('infinite_lambda_solves_latent_only',
         all(x['solved_dimension'] == r['K'] for x in roms if x['lambda_rel'] is None))
    gate('finite_lambda_solves_full_bank',
         all(x['solved_dimension'] == r['K'] + r['R'] for x in roms if x['lambda_rel'] is not None))
    note('underdetermined_blocks',
         sorted({(x['M'], x['name']) for x in roms
                 if x['lambda_rel'] is not None and x['M'] < r['R']})[:8])
    note('eq_quadrature_ratio',
         sorted({(x['M'], x['m']) for x in roms if x['quadrature'] == 'eq'}))

    # ------------------------------ errors recomputed from the saved fields ---
    if a.fields:
        d = Path(a.fields)
        bad = [x['artifact'] for x in inv if not (d / x['artifact']).exists()]
        gate('artifacts_present', not bad, bad[:5])
        if not bad:
            refs = {e['case']: np.load(d / e['artifact'])['fields'] for e in r['reference']}
            base = {x['case']: x['artifact'] for x in inv if x['name'] == 'fft_tight'}
            cache = {}

            def fields_of(name):
                if name not in cache:
                    cache[name] = np.load(d / name)['fields']
                return cache[name]

            worst = 0.
            for x in inv:
                truth = refs[x['case']]
                f = fields_of(x['artifact'])
                n0 = np.linalg.norm(truth[0])
                err = float(np.max(np.linalg.norm((f - truth).reshape(len(f), -1), axis=1)) / n0)
                worst = max(worst, abs(err - x['error']['fixed_initial_max'])
                            / max(x['error']['fixed_initial_max'], 1e-300))
                g = fields_of(base[x['case']])
                per = np.linalg.norm((f - g).reshape(len(f), -1), axis=1) / n0
                x['same_grid_per_time'] = per.tolist()
                x['same_grid_max'] = float(np.max(per))
                x['same_grid_evolved_max'] = float(np.max(per[1:]))
                x['same_grid_argmax_time'] = int(np.argmax(per))
            gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

            # The t = 0 output is the head's compression of the supplied field and
            # cannot depend on lambda, because the initializer is arm (a)'s and
            # starts the correction at y = 0. Assert that, since the whole reading
            # of the all-times metric rests on it. lambda = infinity and the finite
            # path are separately compiled graphs, so the bar is round-off, not
            # bitwise equality.
            t0 = {}
            for x in inv:
                if x['kind'] == 'rom':
                    t0.setdefault(x['case'], []).append(x['same_grid_per_time'][0])
            spread = {c: (max(v) - min(v)) / max(abs(np.mean(v)), 1e-300) for c, v in t0.items()}
            gate('initial_output_is_lambda_independent',
                 max(spread.values()) < 1e-12,
                 dict(worst_relative_spread=max(spread.values()), tolerance=1e-12,
                      per_case=spread,
                      note=('the t = 0 same-grid error over every lambda, every test count and '
                            'every quadrature, on each case; arm (a) initializer means y = 0 '
                            'there')))
            note('all_times_worst_is_at_t0',
                 sorted({(x['name'], x['same_grid_argmax_time']) for x in inv
                         if x['kind'] == 'rom' and x['same_grid_argmax_time'] != 0})[:10])

    # --------------------------------------------------- per-arm aggregates ---
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], lambda_rel=x.get('lambda_rel'),
            lambda_absolute=x.get('lambda_absolute'), sigma=x.get('sigma'),
            solved_dimension=x.get('solved_dimension'), M=x.get('M'), m=x.get('m'),
            quadrature=x.get('quadrature'), dt=x.get('dt'), role=x.get('role'),
            trust_y_mode=x.get('trust_y_mode'), gpu_ms=[], host_ms=[], case_error={},
            case_same_grid={}, case_same_grid_evolved={}, iterations=[], stationarity=[],
            stationary=[], completed=[], budget_exits=[], correction=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_error'][x['case']] = x['error']['fixed_initial_max']
        if 'same_grid_max' in x:
            t['case_same_grid'][x['case']] = x['same_grid_max']
            t['case_same_grid_evolved'][x['case']] = x['same_grid_evolved_max']
        t['iterations'] += x['iterations']
        if x['kind'] == 'rom':
            t['stationarity'].append(max(max(x['step_stationarity']), x['ic_stationarity']))
            t['stationary'].append(x['stationary'])
            t['completed'].append(x['completed'])
            t['budget_exits'].append(x['budget_exits'])
            t['correction'].append(x['max_relative_correction'])
    rc = r['reconstruction'][0]
    rows = []
    for name, t in table.items():
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        sg = (np.array([t['case_same_grid'][c] for c in sorted(t['case_same_grid'])])
              if t['case_same_grid'] else None)
        sge = (np.array([t['case_same_grid_evolved'][c] for c in sorted(t['case_same_grid_evolved'])])
               if t['case_same_grid_evolved'] else None)
        # Layer 2: the head manifold for lambda = infinity, the whole bank (= layer 1)
        # for every finite lambda, because the penalty restricts the solver and not
        # the reachable set.
        if t['kind'] != 'rom':
            floor = best = None
        elif t['lambda_rel'] is None:
            floor, best = rc['worst_bank_projection'], rc['worst_best_found']
        else:
            floor = best = rc['worst_bank_projection']
        rows.append(dict(
            arm=name, kind=t['kind'], lambda_rel=t['lambda_rel'],
            lambda_absolute=t['lambda_absolute'], sigma=t['sigma'],
            solved_dimension=t['solved_dimension'], M=t['M'], m=t['m'],
            quadrature=t['quadrature'], dt=t['dt'], role=t['role'],
            trust_y_mode=t['trust_y_mode'],
            worst_reference_percent=float(np.max(errs) * 100),
            median_reference_percent=float(np.median(errs) * 100),
            worst_same_grid_percent=(float(np.max(sg) * 100) if sg is not None else None),
            median_same_grid_percent=(float(np.median(sg) * 100) if sg is not None else None),
            worst_same_grid_evolved_percent=(float(np.max(sge) * 100) if sge is not None else None),
            median_same_grid_evolved_percent=(float(np.median(sge) * 100) if sge is not None else None),
            worst_bank_projection_percent=(None if floor is None else float(floor * 100)),
            worst_best_found_percent=(None if best is None else float(best * 100)),
            max_relative_correction_percent=(float(np.max(t['correction']) * 100)
                                             if t['correction'] else None),
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            gpu_ms_repetitions=len(t['gpu_ms']),
            median_iterations=median(t['iterations']), max_iterations=float(np.max(t['iterations'])),
            max_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
            all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
            all_completed=(bool(all(t['completed'])) if t['completed'] else None),
            total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None)))
    rows.sort(key=lambda x: (x['kind'] != 'rom', x['M'] or 0, x['quadrature'] or '',
                             x['trust_y_mode'] or '',
                             -(np.inf if x['lambda_rel'] is None else x['lambda_rel'])))
    checks['arm_table'] = rows

    # ------------------- gate (ii): lambda = infinity against abl01 arm (a) ---
    if ABLATION.exists():
        ab = json.loads(ABLATION.read_text())
        want = {x['case']: x['error']['fixed_initial_max'] for x in ab['invocations']
                if x['name'] == 'a_neural_eq' and x['intervals'] == r['intervals']}
        got = {x['case']: x['error']['fixed_initial_max'] for x in inv
               if x['name'] == 'M64_eq_laminf'}
        wh = {x['case']: x['field_sha256'] for x in ab['invocations']
              if x['name'] == 'a_neural_eq' and x['intervals'] == r['intervals']}
        gh = {x['case']: x['field_sha256'] for x in inv if x['name'] == 'M64_eq_laminf'}
        shared = sorted(set(want) & set(got))
        deltas = [abs(got[c] - want[c]) / max(want[c], 1e-300) for c in shared]
        identical = sum(1 for c in shared if wh[c] == gh[c])
        gate('laminf_reproduces_head_ablation_arm_a', bool(shared) and max(deltas) < GATE_TOLERANCE,
             dict(compared=len(shared), worst_relative_delta=(max(deltas) if deltas else None),
                  bitwise_identical_fields=identical, tolerance=GATE_TOLERANCE,
                  ablation_gpu=ab.get('gpu'), dial_gpu=r.get('gpu'),
                  note=('prior-dial M64_eq_laminf versus head-ablation arm a_neural_eq, same cases '
                        'and mesh; bitwise code equivalence is established separately by the local '
                        'smoke, which is bit-identical to the incumbent solver')))
        gate('cases_identical',
             bool(np.allclose(np.asarray(ab['physical_cases']), np.asarray(r['physical_cases']))))

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
