"""The prior dial on Burgers 2D at one mesh, one frozen checkpoint.

Only lambda moves. Every network weight, the bank G, K = 16, the initializer
policy, dt, the stopping rule and the output contract are the head-ablation
arm (a) contract. See DESIGN.md for the objective and the declared scaling of
lambda.

The correction ladder's two expensive offline stages are deliberately absent:
the correction directions here are the WHOLE bank in the field metric
(C = R_G^{-1}), fixed by the checkpoint, so no snapshots and no residual POD are
needed and no reported quantity depends on them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import prior as PR


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def lam_tag(lr):
    return 'inf' if lr is None else f'{lr:g}'.replace('.', 'p').replace('-', 'm').replace('+', '')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L = int(cfg['intervals'])
    dt = cfg['dt']
    strict = cfg['strict']

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
                  checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R, intervals=L,
                  spatial_bank_frozen=True, network_weights_frozen=True, final_cohort_unopened=True,
                  output_times=[0, .05, .1, .15, .2, .25],
                  objective=('min_{z,c} ||r_w(c)||^2 + lambda ||R_G (c - h_theta(z))||^2, '
                             'lambda = lambda_rel * sigma^2, sigma = ||A R_G^{-1}||_2 = ||Phi^T Q_G||_2'),
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output fields; '
                                   'same-invocation host transfers also measured; identical for every arm'),
                  reference=[], bank={}, arm_setup=[], reconstruction=[],
                  invocations=[], declared_subjects=[], verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    report['cohort_note'] = ('exactly the six opened development cases of the head-ablation job and the '
                             'correction ladder; no new case is opened')
    save()

    # ---------------------------------------------------------- references ---
    rf, rt = cfg['reference_mesh'], cfg['reference_dt']
    refs = {}
    q, _ = e.make_fom(rf, rt)
    for case, phys in enumerate(physical):
        t = time.perf_counter()
        f, it, rn = host(q(jnp.asarray(e.initial(rf, phys)), float(phys[4]), 1e-11, 1e-9))
        assert np.isfinite(f).all() and np.max(rn) < 2e-11
        f = np.array(f[:, ::rf // L, ::rf // L], copy=True)
        refs[case] = f
        name = f'ref_L{rf}_dt{rt}_case{case}.npz'
        np.savez_compressed(out / name, fields=f, iterations=it, residuals=rn)
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=L,
                                        field_sha256=sha_array(f), seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q
    jax.clear_caches()

    # ------------------------------------------------------------- the bank --
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qg, Rg = A.whiten(G)
    jax.block_until_ready(Rg)
    svR = np.asarray(jnp.linalg.svd(Rg, compute_uv=False))
    report['bank'] = dict(rank=R, orthonormality_deviation=float(
        jnp.linalg.norm(Qg.T @ Qg - jnp.eye(R)) / np.sqrt(R)),
        factorization_relative=float(jnp.linalg.norm(Qg @ Rg - G) / jnp.linalg.norm(G)),
        R_condition=float(svR[0] / max(svR[-1], 1e-300)),
        R_singular_values=svR[:min(len(svR), 600)].tolist(),
        note=('C = R_G^{-1} is never applied to a solved vector: the online state is formed as '
              'G h(z) + Q_G y, so the correction is conditioned by Q_G alone'))
    save()

    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    report['trust_radius'] = trust
    report['code_stride'] = int(stride)

    # The initializer is arm (a)'s, shared identically by every arm at every lambda.
    cold, cinfo = A.build_cold(bank, A.neural_head(params), Zsub, cfg['cold_axis_points'])
    report['initializer'] = cinfo
    save()
    print('BANK', round(time.perf_counter() - begin, 1), 'condR', report['bank']['R_condition'], flush=True)

    # ------------------------------------------------ operators and queries --
    groups = {}
    for blk in cfg['blocks']:
        key = (int(blk['M']), blk['quadrature'])
        if key in groups:
            continue
        M, quadrature = key
        m = int(cfg['quadrature_multiplier'] * M) if quadrature == 'eq' else None
        t0 = time.perf_counter()
        data, info = PR.build_prior_operators(bank, L, M, quadrature, R, Qg, Rg, Zcoef=Zold, m=m,
                                              params=params, eq_seed=cfg['eq_seed'],
                                              candidate_cap=cfg['candidate_cap'],
                                              fit_states=cfg['fit_states'])
        # lambda = infinity is arm (a): the y block has zero width and the y-space
        # operators are not even handed to it.
        data0 = {k: v for k, v in data.items() if k not in ('Qg', 'Ay', 'G5y')}
        linear_fin = 'gj' if (K + R) <= cfg['gauss_jordan_max'] else 'lu'
        qi = PR.make_prior_query(params, K, 0, L, dt, quadrature, linear='gj', **strict)
        qf = PR.make_prior_query(params, K, R, L, dt, quadrature, linear=linear_fin, **strict)
        groups[key] = dict(M=M, quadrature=quadrature, m=m, data=data, data0=data0,
                           query_inf=qi, query_fin=qf, sigma=info['sigma'],
                           linear_finite=linear_fin)
        info.update(group=f'M{M}_{quadrature}', linear_solve_finite=linear_fin,
                    linear_solve_infinite='gj', total_setup_seconds=time.perf_counter() - t0)
        report['arm_setup'].append(info)
        print('GROUP', key, round(info['total_setup_seconds'], 1), 'sigma', info['sigma'], flush=True)
        save()

    subjects = []
    for blk in cfg['blocks']:
        key = (int(blk['M']), blk['quadrature'])
        g = groups[key]
        lams = cfg['lambda_rel'] if blk['lambdas'] == 'all' else blk['lambdas']
        tag = blk.get('tag')
        for lr in lams:
            name = f"M{key[0]}_{key[1]}_lam{lam_tag(lr)}" + (f'_{tag}' if tag else '')
            ty = float('inf') if blk['trust_y'] == 'free' else trust
            solved = K if lr is None else K + R
            assert key[0] > K or lr is not None, (name, key[0], K)
            subjects.append(dict(kind='rom', name=name, key=key, lambda_rel=lr, trust_y=ty,
                                 role=blk['role'], trust_y_mode=blk['trust_y'],
                                 solved_dimension=solved))
            report['declared_subjects'].append(dict(
                name=name, method='rom', M=key[0], m=g['m'], quadrature=key[1], dt=dt,
                lambda_rel=lr, lambda_absolute=(None if lr is None else lr * g['sigma'] ** 2),
                sigma=g['sigma'], solved_dimension=solved, role=blk['role'],
                trust_z=trust, trust_y=(None if np.isinf(ty) else ty),
                underdetermined_at_small_lambda=bool(key[0] < R),
                linear_solve=('gj' if lr is None else g['linear_finite'])))
    names = [s['name'] for s in subjects]
    assert len(set(names)) == len(names), names
    save()

    foms = {'fft': ip.make_fom(L, dt, 'fft')}
    for fs in cfg['fom_settings']:
        report['declared_subjects'].append(dict(method='fom', dt=dt, **fs))
    save()

    # ---------------------------------------------- untimed diagnostics ------
    # Layer 1 (bank projection floor) and layer 2 (best-found reconstruction).
    # For any FINITE lambda the reachable set is the whole bank, so layer 2 equals
    # layer 1 there: the penalty is a solver-side prior, not a restriction of the
    # manifold. Only the lambda = infinity row has a manifold of its own.
    recon = A.make_reconstruction(A.neural_head(params), K, L, cfg['recon_budget'], linear='gj')
    Hc = jax.jit(jax.vmap(lambda z: A.sc.head(params, z)))(jnp.asarray(Zsub))
    Hn = jnp.sum((Hc @ Rg.T) ** 2, 1)
    rows = []
    for case in refs:
        ref = refs[case]
        n0 = float(np.linalg.norm(ref[0]))
        bank_err, man_err, iters = [], [], []
        for ti in range(ref.shape[0]):
            target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
            bank_err.append(float(jnp.linalg.norm(Qg @ (Qg.T @ target) - target)) / n0)
            score = Hn - 2 * (Hc @ (G.T @ target))
            starts = jnp.asarray(Zsub)[jnp.argsort(score)[:cfg['recon_starts']]]
            z, rn, it, reason = host(recon(starts, target, G))
            man_err.append(float(rn) / n0)
            iters.append(int(it))
        rows.append(dict(case=case, bank_projection_per_time=bank_err,
                         bank_projection_max=float(np.max(bank_err)),
                         best_found_per_time=man_err, best_found_max=float(np.max(man_err)),
                         reconstruction_iterations=iters))
    report['reconstruction'] = [dict(
        manifold='head', solved_dimension=K, cases=rows,
        worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
        worst_best_found=float(max(r['best_found_max'] for r in rows)),
        note=('the head manifold is the lambda = infinity reachable set; every finite lambda '
              'reaches the whole bank, so its best-found reconstruction is the bank projection '
              'floor in this same row'))]
    save()
    print('RECON', round(time.perf_counter() - begin, 1),
          'bank', report['reconstruction'][0]['worst_bank_projection'],
          'head', report['reconstruction'][0]['worst_best_found'], flush=True)

    # ------------------------------------------------------ timed queries ----
    all_subjects = subjects + [dict(kind='fom', name=fs['name'], setting=fs)
                               for fs in cfg['fom_settings']]
    inputs = [e.initial(L, phys) for phys in physical]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, u, case):
        nu = float(physical[case, 4])
        if sub['kind'] == 'rom':
            g = groups[sub['key']]
            if sub['lambda_rel'] is None:
                return g['query_inf'](u, nu, jnp.asarray(0.), jnp.asarray(trust),
                                      jnp.asarray(np.inf), g['data0'], cold)
            sl = float(np.sqrt(sub['lambda_rel']) * g['sigma'])
            return g['query_fin'](u, nu, jnp.asarray(sl), jnp.asarray(trust),
                                  jnp.asarray(sub['trust_y']), g['data'], cold)
        fs = sub['setting']
        fn, pre = foms[fs['preconditioner']]
        return fn(u, nu, fs['ntol'], fs['ltol'], *pre)

    t = time.perf_counter()
    for sub in all_subjects:
        jax.block_until_ready(invoke(sub, jnp.asarray(inputs[0]), 0))
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(all_subjects))
    print('WARMUP', round(time.perf_counter() - t, 1), flush=True)
    save()

    artifacts = {}
    for rep in range(cfg['reps']):
        for case in range(len(physical)):
            for i in order_rng.permutation(len(all_subjects)):
                sub = all_subjects[int(i)]
                e.burn(cfg['burn_seconds'])
                ht = time.perf_counter()
                u = jax.device_put(np.array(inputs[case], copy=True))
                jax.block_until_ready(u)
                gt = time.perf_counter()
                value = invoke(sub, u, case)
                jax.block_until_ready(value)
                gs = time.perf_counter() - gt
                f = np.asarray(value[0])
                hs = time.perf_counter() - ht
                v = host(value)
                assert np.isfinite(f).all()
                h = sha_array(f)
                key = (sub['name'], case, h)
                if key not in artifacts:
                    fn = f"L{L}_{sub['name']}_case{case}_rep{rep}.npz"
                    extra = (dict(internal_latents=v[7], correction_norms=v[12],
                                  output_state_norms=v[13]) if sub['kind'] == 'rom' else {})
                    np.savez_compressed(out / fn, fields=f, **extra)
                    artifacts[key] = fn
                row = dict(intervals=L, case=case, cohort=report['cohort_roles'][case], rep=rep,
                           kind=sub['kind'], name=sub['name'], gpu_seconds=gs, host_seconds=hs,
                           output_bytes=int(f.nbytes), field_sha256=h, artifact=artifacts[key],
                           error=e.errors(f, refs[case], L), iterations=v[1].tolist(),
                           residuals=v[2].tolist(), finite=True)
                if sub['kind'] == 'rom':
                    g = groups[sub['key']]
                    reasons = v[3].tolist()
                    ynorm = np.asarray(v[12])
                    unorm = np.asarray(v[13])
                    row.update(M=g['M'], m=g['m'], quadrature=g['quadrature'], dt=dt,
                               lambda_rel=sub['lambda_rel'], sigma=g['sigma'],
                               lambda_absolute=(None if sub['lambda_rel'] is None
                                                else sub['lambda_rel'] * g['sigma'] ** 2),
                               solved_dimension=sub['solved_dimension'], role=sub['role'],
                               trust_y_mode=sub['trust_y_mode'],
                               linear_solve=('gj' if sub['lambda_rel'] is None else g['linear_finite']),
                               stop_reasons=reasons, ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=v[8].tolist(), ic_stationarity=float(v[9]),
                               ic_residual=float(v[10]), ic_input_norm=float(v[11]),
                               ic_relative_residual=float(v[10]) / max(float(v[11]), 1e-300),
                               max_correction_norm=float(np.max(ynorm)),
                               max_relative_correction=float(np.max(ynorm) / max(np.max(unorm), 1e-300)),
                               budget_exits=int(sum(1 for r in reasons if r == 0)),
                               rejected_exits=int(sum(1 for r in reasons if r == 3)),
                               reason_counts={str(k): int(c) for k, c in
                                              zip(*np.unique(np.asarray(reasons), return_counts=True))},
                               stationary=bool(max(float(np.max(v[8])), float(v[9]))
                                               <= strict['gtol'] * (1 + 1e-7)),
                               completed=bool(all(r in (1, 2, 4) for r in reasons)
                                              and int(v[6]) in (1, 2, 4)))
                else:
                    row.update(nonlinear_converged=bool(np.max(v[2]) <= sub['setting']['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('PRIOR DIAL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
