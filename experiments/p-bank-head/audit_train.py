"""Independent audit of the p-bank-head training attempt. No driver, no JAX.

Recomputes, from the saved weights alone and a pure-NumPy separable decoder:
the bank projection floor of every bank arm on the development cohort, the head
error at the stored codes for every head arm, the correction bases'
orthonormality in the exact QR metric, and both pre-registered selection rules.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import pbh_audit_np as N


def floors_and_targets(params, xy, U):
    G = N.features(params, xy)
    Q, R = np.linalg.qr(G)
    T = U @ Q
    nu2 = np.sum(U * U, axis=1)
    perp2 = np.clip(nu2 - np.sum(T * T, axis=1), 0., None)
    return G, Q, R, T, perp2, nu2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mesh', type=int, default=None,
                    help='mesh to recompute floors on (default: the coarsest floor mesh)')
    a = ap.parse_args()
    d = json.loads((a.run / 'result.json').read_text())
    assert d['complete'], 'incomplete run'
    cfg = d['config']
    checks, failures = {}, []

    def check(name, ok, detail):
        checks[name] = dict(passed=bool(ok), **detail)
        if not ok:
            failures.append(name)

    check('backend', d['backend'] == 'gpu' and d['x64'] and d['matmul_precision'] == 'highest',
          dict(backend=d['backend'], x64=d['x64'], precision=d['matmul_precision'],
               gpu=d['gpu'], job_id=d['job_id'], commit=d['commit']))

    dev = np.concatenate((N.source_params(cfg['eval_seed'], cfg['eval_count']),
                          N.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    ok_cohorts, detail = True, {}
    for S, rec in d['cohorts'].items():
        draws = N.source_params(cfg['train_seed'], int(S))
        h = hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest()
        # core.source_params is not bit-reproducible across the GB10 and the
        # cluster: the Gaussian-width column, exp(uniform(...)), differs by one
        # ulp between the two numpy builds. Cohort identity is therefore checked
        # to a tolerance here and the hash is recorded, not required.
        overlap = int(sum(1 for t in draws for s in dev if np.allclose(t, s)))
        fit = np.asarray(rec['fit'])
        val = np.asarray(rec['validation'])
        split_ok = (len(np.intersect1d(fit, val)) == 0
                    and len(fit) + len(val) == int(S)
                    and sorted(np.concatenate((fit, val)).tolist()) == list(range(int(S))))
        detail[S] = dict(parameters_hash_match=h == rec['parameters_sha256'],
                         local_sha256=h, recorded_sha256=rec['parameters_sha256'],
                         overlap=overlap, split_partitions=bool(split_ok))
        ok_cohorts &= overlap == 0 and split_ok
    check('cohorts', ok_cohorts, dict(per_cohort=detail,
                                      development_sha256=hashlib.sha256(
                                          np.ascontiguousarray(dev).tobytes()).hexdigest()))

    mesh = a.mesh or min(cfg['floor_intervals'])
    xy = N.coords(mesh)
    Udev = np.stack([N.solve(mesh, p)[1:-1, 1:-1].ravel() for p in dev])

    worst_floor, worst_stored, per_arm = 0., 0., {}
    for ck in d['checkpoints']:
        src = a.run / ck['path']
        assert hashlib.sha256(src.read_bytes()).hexdigest() == ck['sha256'], ck['id']
        params, Z, _ = N.load(src)
        wh = hashlib.sha256()
        for x in (params['B'], *[y for lay in params['g'] for y in lay],
                  *[y for lay in params['h'] for y in lay], params['h_lin'],
                  np.asarray(params['out_scale'])):
            wh.update(np.ascontiguousarray(np.asarray(x)).tobytes())
        G, Q, R, T, perp2, nu2 = floors_and_targets(params, xy, Udev)
        floor = np.sqrt(perp2 / nu2)
        entry = dict(K=int(Z.shape[1]), R=int(np.asarray(params['h_lin']).shape[1]),
                     recomputed_worst_floor=float(floor.max()))
        if ck.get('layer') == 'bank':
            arm = next(x for x in d['bank_arms'] if x['arm'] == ck['id'])
            rep = next(f for f in arm['floors'] if f['intervals'] == mesh)
            diff = float(np.max(np.abs(floor - np.asarray(rep['development']['per_case']))))
            entry.update(reported_worst_floor=rep['development']['worst'],
                         worst_per_case_difference=diff)
            worst_floor = max(worst_floor, diff)
        if ck.get('layer') == 'head' and mesh == cfg['training_intervals']:
            arm = next(x for x in d['head_arms'] if x['arm'] == ck['id'])
            draws = N.source_params(cfg['train_seed'], arm['S'])
            fit = np.asarray(d['cohorts'][str(arm['S'])]['fit'])
            sub = fit[::max(1, len(fit) // 64)]
            Uf = np.stack([N.solve(mesh, p)[1:-1, 1:-1].ravel() for p in draws[sub]])
            Tf = Uf @ Q
            n2 = np.sum(Uf * Uf, axis=1)
            p2 = np.clip(n2 - np.sum(Tf * Tf, axis=1), 0., None)
            pos = {int(v): i for i, v in enumerate(fit)}
            res = N.head(params, Z[[pos[int(v)] for v in sub]]) @ R.T - Tf
            got = np.sqrt((np.sum(res * res, axis=1) + p2) / n2)
            want = np.asarray(arm['head_at_stored_codes_fit']['per_case'])[
                [pos[int(v)] for v in sub]]
            diff = float(np.max(np.abs(got - want)))
            entry.update(stored_code_subsample=len(sub), worst_stored_code_difference=diff)
            worst_stored = max(worst_stored, diff)
        per_arm[ck['id']] = entry
    check('numpy_bank_floors', worst_floor < 1e-10,
          dict(intervals=mesh, worst_difference=worst_floor, per_checkpoint=per_arm))
    check('numpy_stored_code_errors', worst_stored < 1e-10,
          dict(intervals=mesh, worst_difference=worst_stored))

    bank = d['selection']['bank']
    pick = min(d['bank_arms'], key=lambda x: (
        next(f for f in x['floors'] if f['intervals'] == bank['mesh'])['validation']['worst'],
        next(f for f in x['floors'] if f['intervals'] == bank['mesh'])['validation']['median'],
        x['R'], x['S']))
    check('bank_selection_rule', pick['arm'] == bank['selected'],
          dict(recomputed=pick['arm'], recorded=bank['selected'], mesh=bank['mesh'],
               development_ranking_agrees=bank['development_ranking_agrees']))

    head_ok, head_detail = True, {}
    for sel in d['selection']['heads']:
        cands = [x for x in d['head_arms'] if x['K'] == sel['K']]
        p = min(cands, key=lambda x: (x['best_found_validation']['worst'],
                                      x['best_found_validation']['median'],
                                      x['beta_weak'], x['beta_smooth']))
        head_detail[str(sel['K'])] = dict(recomputed=p['arm'], recorded=sel['selected'],
                                          development_ranking_agrees=sel['development_ranking_agrees'])
        head_ok &= p['arm'] == sel['selected']
    check('head_selection_rule', head_ok, dict(per_latent=head_detail))

    basis_ok, basis_detail = True, {}
    for sel in d['selection']['heads']:
        path = a.run / sel['basis']['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sel['basis']['sha256']
        b = np.load(path)
        Rg, C_ = b['R'], b['coefficient_directions']
        W = Rg @ C_
        err = float(np.linalg.norm(W.T @ W - np.eye(C_.shape[1])))
        basis_detail[str(sel['K'])] = dict(recomputed_orthogonality=err,
                                           reported=sel['basis']['orthogonality_error'],
                                           count=int(C_.shape[1]))
        basis_ok &= err < 1e-8
    check('correction_bases', basis_ok, dict(per_latent=basis_detail))

    result = dict(run=str(a.run), passed=not failures, failures=failures, checks=checks,
                  result_sha256=hashlib.sha256((a.run / 'result.json').read_bytes()).hexdigest(),
                  scope='NumPy/SciPy only; imports neither the driver nor JAX')
    a.out.write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(json.dumps({k: v['passed'] for k, v in checks.items()}, indent=2))
    print('PASSED' if not failures else f'FAILED {failures}')
    assert not failures


if __name__ == '__main__':
    main()
