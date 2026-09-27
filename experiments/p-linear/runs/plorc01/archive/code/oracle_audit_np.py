"""Independent NumPy audit of saved coefficient-metric oracle fits and gradients.

No JAX or driver import. Decoder derivatives use an explicit SiLU chain rule.
The optional full-bank audit streams regenerated fields at every original query mesh.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'p-bank-head'))
import pbh_audit_np as N


def head_jacobian(params, z):
    x = np.asarray(z)
    derivative = np.eye(len(z))
    for w, b in params['h'][:-1]:
        y = x @ np.asarray(w) + np.asarray(b)
        dy = np.asarray(w).T @ derivative
        sigmoid = 1. / (1. + np.exp(-y))
        derivative = (sigmoid + y * sigmoid * (1. - sigmoid))[:, None] * dy
        x = y * sigmoid
    w, b = params['h'][-1]
    return (x @ w + b + z @ params['h_lin'],
            np.asarray(w).T @ derivative + np.asarray(params['h_lin']).T)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--full-bank', action='store_true')
    a = ap.parse_args()
    started = time.monotonic()
    result_path = a.archive / 'output/result.json'
    result = json.loads(result_path.read_text())
    code = a.archive / 'code'
    history = json.loads((code / 'oracle-history.json').read_text())
    checks = []
    def check(name, ok, **facts):
        checks.append(dict(name=name, passed=bool(ok), **facts))
    check('complete', result['complete'])
    check('precision', result['backend'] == 'gpu' and result['x64'] and result['matmul_precision'] == 'highest')
    if not result['smoke']:
        check('full_panel_grid', {(x['intervals'], x['model']) for x in result['panels']} ==
              {(n, mid) for n in (256, 1024) for mid in ('new_K32', 'incumbent')})
    maxima = dict(objective=0., jacobian=0., gradient=0., full_field=0.)
    count = 0
    for panel in result['panels']:
        n, mid, R = panel['intervals'], panel['model'], panel['R']
        model = next(x for x in history['256']['config']['models'] if x['id'] == mid)
        ck = code / Path(model['checkpoint']).name
        p, _, _ = N.load(ck)
        check(f'{n}:{mid}:checkpoint', hashlib.sha256(ck.read_bytes()).hexdigest() == panel['checkpoint_sha256'])
        metric_path = a.archive / 'output' / panel['metric_artifact']
        check(f'{n}:{mid}:metric_hash', hashlib.sha256(metric_path.read_bytes()).hexdigest() == panel['metric_sha256'])
        metric = np.load(metric_path)
        Rg, T, perp2, nu2, Cfull = [metric[k] for k in ['Rg', 'T', 'perp2', 'nu2', 'Cfull']]
        gram_defect = np.linalg.norm(Rg.T @ Rg - metric['gram']) / np.linalg.norm(metric['gram'])
        target_defect = np.linalg.norm(T @ Rg - metric['gt_u']) / np.linalg.norm(metric['gt_u'])
        check(f'{n}:{mid}:metric_equations', gram_defect < 1e-12 and target_defect < 1e-12,
              gram_relative_defect=float(gram_defect), target_relative_defect=float(target_defect))
        floor = np.sqrt(perp2 / nu2)
        if not result['smoke']:
            original_cfg = history[str(n)]['config']
            expected_dev = np.concatenate((N.source_params(original_cfg['eval_seed'], original_cfg['eval_count']),
                                           N.source_params(original_cfg['fresh_seed'], original_cfg['fresh_count'])))
            check(f'{n}:{mid}:original_cohort', np.allclose(panel['cohort_parameters'], expected_dev, rtol=0, atol=1e-15))
            expected_q = [q for q in original_cfg['ladder_q'] if q <= R and
                          (mid == 'new_K32' or q in original_cfg['control_q'])]
            check(f'{n}:{mid}:all_original_ranks', [r['q'] for r in panel['rungs']] == expected_q and
                  all(len(r['cases']) == len(expected_dev) for r in panel['rungs']))
            oldrec = next(x for x in history[str(n)]['reconstruction'] if x['model'] == mid)
            oldfloor = oldrec['bank_projection']['per_case']
            defect = float(np.max(np.abs(floor - oldfloor)))
            check(f'{n}:{mid}:old_floor', defect < 2e-7, maximum_absolute_defect=defect)
        previous = None
        all_coefficients, expected_errors = [], []
        for rung in panel['rungs']:
            q = rung['q']
            Q = np.linalg.qr(Rg @ Cfull[:, :q], mode='reduced')[0] if q < R else np.eye(R)
            if q == 0:
                Q = np.empty((R, 0))
            orth = float(np.linalg.norm(Q.T @ Q - np.eye(q)))
            check(f'{n}:{mid}:q{q}:projector', orth < 1e-8)
            replay = rung.get('retained_solve_replay', [])
            if replay:
                defect = max(abs(x['original_error'] - x['regenerated_error']) for x in replay)
                check(f'{n}:{mid}:q{q}:original_direction_replay', defect < 2e-7,
                      maximum_error_defect=defect, replayed_states=len(replay))
            errors = []
            for row in rung['cases']:
                c = row['case']
                coefficient = np.asarray(row['selected_coefficient'])
                projected = Rg @ coefficient - T[c]
                error = float(np.sqrt(np.dot(projected, projected) + perp2[c]) / np.sqrt(nu2[c]))
                maxima['objective'] = max(maxima['objective'], abs(error - row['error']))
                count += 1
                check(f'{n}:{mid}:q{q}:case{c}:error', abs(error - row['error']) < 2e-7 and error >= floor[c] - 1e-10)
                if q < R:
                    z = np.asarray(row['selected_latent'])
                    h, J = head_jacobian(p, z)
                    raw = Rg @ h - T[c]
                    residual = raw - Q @ (Q.T @ raw)
                    Jr = Rg @ J
                    Jr -= Q @ (Q.T @ Jr)
                    g = np.linalg.norm(Jr.T @ residual) / (np.linalg.norm(Jr) * np.linalg.norm(residual) + 1e-300)
                    jd = np.linalg.norm(Jr - np.asarray(row['selected_jacobian'])) / max(np.linalg.norm(Jr), 1e-300)
                    gd = abs(float(g) - row['normalized_gradients'][row['selected']])
                    maxima['jacobian'] = max(maxima['jacobian'], float(jd))
                    maxima['gradient'] = max(maxima['gradient'], gd)
                    check(f'{n}:{mid}:q{q}:case{c}:gradient', jd < 2e-7 and gd < 2e-7,
                          normalized_gradient=float(g), jacobian_relative_defect=float(jd))
                    check(f'{n}:{mid}:q{q}:case{c}:start_selection',
                          row['selected_residual_norm'] <= min(row['residual_norms']) + 1e-10)
                    if row['reasons'][row['selected']] == 4:
                        check(f'{n}:{mid}:q{q}:case{c}:stationarity', g <= 1e-6 + 2e-7)
                else:
                    check(f'{n}:{mid}:endpoint:case{c}', abs(error - floor[c]) < 1e-10)
                if not result['smoke']:
                    solved = [x for x in history[str(n)]['solved_states']
                              if x['model'] == mid and x['q'] == q and x['case'] == c]
                    if solved:
                        check(f'{n}:{mid}:q{q}:case{c}:solved_bracket',
                              error <= min(x['same_grid_error'] for x in solved) + 2e-7)
                errors.append(error)
            if previous is not None:
                check(f'{n}:{mid}:q{q}:nested_best_found', np.max(np.asarray(errors) - previous) < 2e-7)
            previous = np.asarray(errors)
            check(f'{n}:{mid}:q{q}:summary', abs(max(errors) - rung['worst']) < 2e-7)
            if q == 0 and not result['smoke']:
                old = next(x for x in oldrec['augmented'] if x['q'] == 0)['best_found']['per_case']
                defect = max(abs(row['original_starts_error'] - old[row['case']]) for row in rung['cases'])
                check(f'{n}:{mid}:q0:unchanged_original_starts', defect < 2e-7, maximum_absolute_defect=defect)
            all_coefficients.append(np.asarray([x['selected_coefficient'] for x in rung['cases']]))
            expected_errors.append(np.asarray([x['error'] for x in rung['cases']]))
        if a.full_bank:
            dev = np.asarray(panel['cohort_parameters'])
            truth = np.stack([N.solve(n, x)[1:-1, 1:-1].ravel() for x in dev])
            xy = N.coords(n)
            coeff = np.concatenate(all_coefficients)
            summed = np.zeros(len(coeff))
            for offset in range(0, len(xy), 16384):
                bank = N.features(p, xy[offset:offset + 16384])
                pred = coeff @ bank.T
                target = np.tile(truth[:, offset:offset + 16384], (len(all_coefficients), 1))
                summed += np.sum((pred - target) ** 2, axis=1)
            actual = np.sqrt(summed / np.tile(np.sum(truth ** 2, axis=1), len(all_coefficients)))
            defect = float(np.max(np.abs(actual - np.concatenate(expected_errors))))
            maxima['full_field'] = max(maxima['full_field'], defect)
            check(f'{n}:{mid}:independent_full_field', defect < 2e-7, maximum_absolute_defect=defect)
    report = dict(passed=all(x['passed'] for x in checks), checks=checks, cases=count,
                  maximum_defects=maxima, full_bank_audit=a.full_bank,
                  result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                  seconds=time.monotonic() - started,
                  scope='Independent NumPy decoder, SiLU derivatives, saved coefficient objective, endpoint, brackets and optional regenerated full fields. Finite best-found fit; no global optimum guarantee.')
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))
    if not report['passed']:
        print(json.dumps([x for x in checks if not x['passed']], indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
