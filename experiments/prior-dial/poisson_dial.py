"""The prior dial on Poisson 2D, in the frozen pabl01 bank and head.

The Poisson weak residual is exactly r(c) = B c - f_m, LINEAR in the bank
coefficients, so the penalized problem

    min_{z, c}  || B c - f_m ||^2 + lambda || R_G (c - h_theta(z)) ||^2

eliminates c in closed form. With y = R_G (c - h_theta(z)) and B_y = B R_G^{-1},

    ( B_y^T B_y + lambda I ) y = B_y^T ( f_m - B h_theta(z) ) ,

which is the stated ( B^T B + lambda R_G^T R_G ) c = B^T f_m + lambda R_G^T R_G h(z).
It is evaluated from ONE thin SVD of B_y built at setup - never from the Gram,
which would square the condition number - so the whole lambda sweep shares one
factorization and the inner solve is two R x R matvecs inside the timed query.

The outer Levenberg-Marquardt therefore runs in z only and differentiates THROUGH
the closed form (Golub-Pereyra), so the solved dimension on Poisson is K, the
trust radius keeps its original latent meaning, and lambda = infinity (y == 0) is
pabl01's arm (a) verbatim.

Staged flat: every module sits beside this file on the cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import pilot as P
import arms as A
import sep_common as sc


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def lam_tag(lr):
    return 'inf' if lr is None else f'{lr:g}'.replace('.', 'p').replace('-', 'm').replace('+', '')


def reduce_bank(V, S, I, J, n):
    """Scaled sine coefficients of every column: core.assemble's own einsum."""
    cubes = jnp.asarray(V).reshape((n - 1, n - 1, V.shape[-1]))
    c = jnp.einsum('xa,xyr,yb->abr', S, cubes, S)
    return c[I, J]


def make_prior_query(params, K, Ry, n, trust, budget, gtol, linear):
    """One matched complete query: host source in, dense nodal field out.

    Ry = 0 drops every y term at trace time, leaving pabl01 arm (a) exactly.
    """
    def correction(z, fm, B, By, Ub, sv, Vb, lam):
        d = fm - B @ sc.head(params, z)
        return Vb @ ((sv / (sv ** 2 + lam)) * (Ub.T @ d)), d

    if Ry:
        def residual(z, fm, B, By, Ub, sv, Vb, lam, sl):
            y, d = correction(z, fm, B, By, Ub, sv, Vb, lam)
            return jnp.concatenate((By @ y - d, sl * y))
    else:
        def residual(z, fm, B, By, Ub, sv, Vb, lam, sl):
            return B @ sc.head(params, z) - fm

    lm = A.make_stationary_lm(residual, budget, trust, gtol, linear)

    @jax.jit
    def kernel(source, S, I, J, W, B, By, Ub, sv, Vb, bank, Qg, lam, sl,
               predictions, codes):
        # The incumbent skinny sine-product projection, charged inside the query.
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        index = jnp.argmin(jnp.sum((predictions - fm[None, :]) ** 2, axis=1))
        z, rn, it, reason, gn = lm(codes[index], (fm, B, By, Ub, sv, Vb, lam, sl), 0.)
        u = bank @ sc.head(params, z)
        if Ry:
            y, _ = correction(z, fm, B, By, Ub, sv, Vb, lam)
            u = u + Qg @ y
            ynorm = jnp.linalg.norm(y)
        else:
            ynorm = jnp.asarray(0.)
        field = jnp.pad(u.reshape(n - 1, n - 1), 1)
        return (field, z, rn, it, reason, gn, index, jnp.linalg.norm(fm), ynorm,
                jnp.linalg.norm(u))
    return kernel


def query_once(kernel, source, ops, By, Ub, sv, Vb, Qg, lam, sl, predictions, codes):
    start = time.perf_counter()
    src = jax.device_put(source)
    src.block_until_ready()
    input_end = time.perf_counter()
    outv = kernel(src, ops['S'], ops['I'], ops['J'], ops['W'], ops['B'], By, Ub, sv, Vb,
                  ops['bank'], Qg, lam, sl, predictions, codes)
    jax.block_until_ready(outv)
    device_end = time.perf_counter()
    field, z, rn, it, reason, gn, index, fmn, ynorm, unorm = jax.device_get(outv)
    end = time.perf_counter()
    return np.asarray(field), dict(
        total_seconds=end - start, input_seconds=input_end - start,
        fused_device_seconds=device_end - input_end, output_seconds=end - device_end,
        residual=float(rn), iterations=int(it), reason=int(reason), stationarity=float(gn),
        relative_residual=float(rn) / max(float(fmn), 1e-300), selected_code_index=int(index),
        correction_norm=float(ynorm),
        relative_correction=float(ynorm) / max(float(unorm), 1e-300),
        latent=np.asarray(z).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--basis', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    params, codes, ckcfg = sc.load_pkl(a.checkpoint)
    basis = np.load(a.basis)
    np.testing.assert_array_equal(np.asarray(codes), basis['training_latents'])
    K = int(np.asarray(codes).shape[1])
    R = int(np.asarray(params['h_lin']).shape[1])
    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
                  checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                  basis_sha256=hashlib.sha256(Path(a.basis).read_bytes()).hexdigest(), K=K, R=R,
                  spatial_bank_frozen=True, network_weights_frozen=True, final_cohort_unopened=True,
                  objective=('min_{z,c} ||B c - f_m||^2 + lambda ||R_G (c - h_theta(z))||^2, '
                             'lambda = lambda_rel * sigma^2, sigma = ||B R_G^{-1}||_2; c eliminated '
                             'in closed form from one thin SVD of B R_G^{-1}'),
                  timing_contract=('host source array in to dense nodal field out, including projection, '
                                   'initialization, nonlinear solve, the inner Tikhonov solve and decode; '
                                   'identical for every arm'),
                  references=[], bank={}, arm_setup=[], reconstruction=[],
                  invocations=[], declared_subjects=[], complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    draws = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                            C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    report['cohort'] = dict(parameters=draws.tolist(), sha256=sha_array(draws),
                            groups=['existing development'] * cfg['eval_count']
                                   + ['fresh development'] * cfg['fresh_count'],
                            note='exactly the twelve development sources of the pabl01 head ablation')
    save()

    order_rng = np.random.default_rng(cfg['order_seed'])
    nlo, nhi = cfg['reference_intervals'][0], cfg['reference_intervals'][-1]
    fine_ref = {}
    for case, param in enumerate(draws):
        t0 = time.perf_counter()
        fine_ref[case] = P.reference(param, nhi)
        coarse = P.reference(param, nlo)
        report['references'].append(dict(
            case=case, fine_intervals=nhi, coarse_intervals=nlo,
            reference_delta=C.relative(coarse, fine_ref[case][::nhi // nlo, ::nhi // nlo]),
            fine_sha256=sha_array(fine_ref[case]), seconds=time.perf_counter() - t0))
    save()
    print('REFERENCES done', round(time.perf_counter() - begin, 1), flush=True)

    for n in cfg['intervals']:
        print('MESH', n, flush=True)
        lam_eig = jnp.asarray(C.eigenvalues(n))
        chain = {case: np.array(fine_ref[case][::nhi // n, ::nhi // n], copy=True) for case in fine_ref}

        ops = C.assemble(params, np.asarray(codes), n, cfg['requested_modes'], cfg['lm_budget'])
        bank, B, S, I, J = ops['bank'], ops['B'], ops['S'], ops['I'], ops['J']
        M = int(B.shape[0])
        trust = float(ops['info']['trust_delta'])
        report['arm_setup'].append(dict(intervals=n, arm='shared_assembly', M=M, trust_radius=trust,
                                        **{k: ops['info'][k] for k in
                                           ('retained_modes', 'stored_features', 'retained_bank_rank',
                                            'trust_delta', 'operator_sha256', 'bank_sha256')}))
        save()

        t0 = time.perf_counter()
        Qg, Rg = A.whiten(bank)
        jax.block_until_ready(Rg)
        By = reduce_bank(Qg, S, I, J, n)
        Ub, sv, Vbt = jnp.linalg.svd(By, full_matrices=False)
        Vb = Vbt.T
        sigma = float(sv[0])
        svR = np.asarray(jnp.linalg.svd(Rg, compute_uv=False))
        report['bank'][str(n)] = dict(
            rank=R, M=M, sigma=sigma, seconds=time.perf_counter() - t0,
            By_singular_values=np.asarray(sv).tolist(),
            By_condition=float(sv[0] / max(float(sv[-1]), 1e-300)),
            R_condition=float(svR[0] / max(svR[-1], 1e-300)),
            orthonormality_deviation=float(jnp.linalg.norm(Qg.T @ Qg - jnp.eye(R)) / np.sqrt(R)),
            factorization_relative=float(jnp.linalg.norm(Qg @ Rg - bank) / jnp.linalg.norm(bank)),
            overdetermined=bool(M > R))
        save()
        print('BANK', n, 'sigma', sigma, 'condBy', report['bank'][str(n)]['By_condition'], flush=True)

        linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
        kern_inf = make_prior_query(params, K, 0, n, trust, cfg['lm_budget'],
                                    cfg['stationarity_tolerance'], linear)
        kern_fin = make_prior_query(params, K, R, n, trust, cfg['lm_budget'],
                                    cfg['stationarity_tolerance'], linear)
        predictions = jax.jit(jax.vmap(lambda z: B @ sc.head(params, z)))(jnp.asarray(codes))
        codes_d = jnp.asarray(codes)

        subjects = []
        for lr in cfg['lambda_rel']:
            name = f'lam{lam_tag(lr)}'
            subjects.append(dict(kind='rom', name=name, lambda_rel=lr))
            report['declared_subjects'].append(dict(
                intervals=n, name=name, method='rom', M=M, K=K, R=R, lambda_rel=lr,
                lambda_absolute=(None if lr is None else lr * sigma ** 2), sigma=sigma,
                solved_dimension=K, inner='exact Tikhonov elimination of c' if lr is not None else 'none',
                trust_radius=trust, linear_solve=linear,
                free_bank_limit_reachable=bool(M > R)))
        subjects.append(dict(kind='fom', name='dst_direct'))
        report['declared_subjects'].append(dict(intervals=n, name='dst_direct', method='fom'))
        save()

        # ------------------------------------------------ untimed diagnostics ----
        # Layer 1 is the bank projection floor; layer 2 is the reachable-set
        # best-found fit. For any FINITE lambda the reachable set is the whole bank,
        # so layer 2 coincides with layer 1 there; only lambda = infinity has a
        # manifold of its own.
        recon = A.make_reconstruction(A.neural_head(params), K, n, cfg['recon_budget'], linear=linear)
        Hc = jax.jit(jax.vmap(lambda z: sc.head(params, z)))(codes_d)
        Hn = jnp.sum((Hc @ Rg.T) ** 2, 1)
        rows = []
        for case in chain:
            target = jnp.asarray(chain[case][1:-1, 1:-1].ravel())
            nref = float(np.linalg.norm(chain[case]))
            bank_err = float(jnp.linalg.norm(Qg @ (Qg.T @ target) - target)) / nref
            score = Hn - 2 * (Hc @ (bank.T @ target))
            starts = codes_d[jnp.argsort(score)[:cfg['recon_starts']]]
            z, rn, it, reason = jax.device_get(recon(starts, target, bank))
            rows.append(dict(case=case, bank_projection=bank_err, best_found=float(rn) / nref,
                             reconstruction_iterations=int(it)))
        report['reconstruction'].append(dict(
            intervals=n, manifold='head', k=K, cases=rows,
            worst_bank_projection=float(max(r['bank_projection'] for r in rows)),
            worst_best_found=float(max(r['best_found'] for r in rows)),
            note=('head manifold = the lambda = infinity reachable set; every finite lambda '
                  'reaches the whole bank, so its best-found reconstruction is the bank '
                  'projection floor in this same row')))
        save()
        print('RECON', n, round(time.perf_counter() - begin, 1), flush=True)

        sources = [C.full_source(n, q) for q in draws]

        def invoke(sub, source, case):
            if sub['kind'] == 'fom':
                return C.fom_query(source, lam_eig)
            lr = sub['lambda_rel']
            if lr is None:
                return query_once(kern_inf, source, ops, By, Ub, sv, Vb, Qg,
                                  jnp.asarray(0.), jnp.asarray(0.), predictions, codes_d)
            return query_once(kern_fin, source, ops, By, Ub, sv, Vb, Qg,
                              jnp.asarray(lr * sigma ** 2), jnp.asarray(float(np.sqrt(lr)) * sigma),
                              predictions, codes_d)

        t = time.perf_counter()
        for sub in subjects:
            invoke(sub, sources[0], 0)
        report.setdefault('compile_warmup', []).append(
            dict(intervals=n, seconds=time.perf_counter() - t, subjects=len(subjects)))
        save()
        print('WARMUP', n, round(time.perf_counter() - t, 1), flush=True)

        artifacts = {}
        for rep in range(cfg['repetitions']):
            for case in range(len(draws)):
                for i in order_rng.permutation(len(subjects)):
                    sub = subjects[int(i)]
                    C.burn(cfg['burn_seconds'])
                    field, row = invoke(sub, sources[case], case)
                    assert np.isfinite(field).all()
                    h = sha_array(field)
                    key = (sub['name'], case, h)
                    if key not in artifacts:
                        fn = f"n{n}_{sub['name']}_case{case}_rep{rep}.npz"
                        np.savez_compressed(out / fn, field=field)
                        artifacts[key] = fn
                    report['invocations'].append(dict(
                        intervals=n, case=case, group=report['cohort']['groups'][case], rep=rep,
                        kind=sub['kind'], name=sub['name'], field_sha256=h, artifact=artifacts[key],
                        physical_error=C.relative(field, chain[case]),
                        source_sha256=C.sha(sources[case]), finite=True,
                        lambda_rel=sub.get('lambda_rel'),
                        lambda_absolute=(None if sub.get('lambda_rel') is None
                                         else sub['lambda_rel'] * sigma ** 2),
                        sigma=(sigma if sub['kind'] == 'rom' else None),
                        K=(K if sub['kind'] == 'rom' else None),
                        M=(M if sub['kind'] == 'rom' else None),
                        trust_radius=(trust if sub['kind'] == 'rom' else None),
                        linear_solve=(linear if sub['kind'] == 'rom' else None), **row))
            print('TIMED', n, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()
        del ops, bank, B, Qg, Rg, By, Ub, sv, Vb, predictions, kern_inf, kern_fin
        jax.clear_caches()

    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('POISSON PRIOR DIAL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
