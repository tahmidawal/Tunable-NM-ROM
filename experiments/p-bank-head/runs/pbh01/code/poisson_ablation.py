"""Head ablation at matched latent dimension in the frozen Poisson spatial bank.

The Poisson weak residual is exactly

    r(z) = B h(z) - f_m,     B = Phi^T(bank),  f_m = lambda^{-1} Phi^T f,

with no time stepping and no quadrature approximation, so the arms differ only in
the coefficient map h. Arm (a) also runs the retained r128_q32 method, which adds
32 linear correction directions with exact analytic elimination.

Staged flat: every module sits beside this file on the cluster.
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

import core as C
import correction_core as CC
import pilot as P
import arms as A
import sep_common as sc


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def reduce_bank(V, S, I, J, n):
    """Scaled sine coefficients of every bank column: the same einsum core.assemble
    uses for the neural bank, so the reduced operator is built identically."""
    cubes = jnp.asarray(V).reshape((n - 1, n - 1, V.shape[-1]))
    c = jnp.einsum('xa,xyr,yb->abr', S, cubes, S)
    return c[I, J]


def radius(Z):
    Z = np.asarray(Z)
    return float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))


def make_query(head, B, bank, n, trust, budget, gtol, linear):
    """One matched complete query: host source in, dense nodal field out."""
    lm = A.make_stationary_lm(lambda z, fm: B @ head(z) - fm, budget, trust, gtol, linear)

    @jax.jit
    def kernel(source, S, I, J, W, B, bank, predictions, codes):
        # The incumbent skinny sine-product projection, charged inside the query.
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        index = jnp.argmin(jnp.sum((predictions - fm[None, :]) ** 2, axis=1))
        z, rn, it, reason, gn = lm(codes[index], (fm,), 0.)
        field = jnp.pad((bank @ head(z)).reshape(n - 1, n - 1), 1)
        return field, z, rn, it, reason, gn, index, jnp.linalg.norm(fm)
    return kernel


def query_once(kernel, source, ops, predictions, codes, B, bank):
    start = time.perf_counter()
    src = jax.device_put(source)
    src.block_until_ready()
    input_end = time.perf_counter()
    out = kernel(src, ops['S'], ops['I'], ops['J'], ops['W'], B, bank, predictions, codes)
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    field, z, rn, it, reason, gn, index, fmn = jax.device_get(out)
    end = time.perf_counter()
    return np.asarray(field), dict(total_seconds=end - start, input_seconds=input_end - start,
                                   fused_device_seconds=device_end - input_end,
                                   output_seconds=end - device_end, residual=float(rn),
                                   iterations=int(it), reason=int(reason), stationarity=float(gn),
                                   relative_residual=float(rn) / max(float(fmn), 1e-300),
                                   selected_code_index=int(index), latent=np.asarray(z).tolist())


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
                  timing_contract=('host source array in to dense nodal field out, including projection, '
                                   'initialization, nonlinear solve and decode; identical for every arm'),
                  references=[], snapshots={}, fits={}, arm_setup=[], reconstruction=[],
                  invocations=[], declared_subjects=[], complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    draws = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                            C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    report['cohort'] = dict(parameters=draws.tolist(), sha256=sha_array(draws),
                            groups=['existing development'] * cfg['eval_count']
                                   + ['fresh development'] * cfg['fresh_count'])
    train_draws = C.source_params(cfg['train_seed'], cfg['train_count'])
    assert not any(np.allclose(t, s) for t in train_draws for s in draws), 'training/eval overlap'
    report['train_parameters_sha256'] = sha_array(train_draws)
    save()

    order_rng = np.random.default_rng(cfg['order_seed'])
    ranks = sorted(set(cfg['pod_ranks'] + [K]))

    # References are mesh-independent solutions of the same continuous problem;
    # compute the refinement chain once and restrict it onto each working mesh.
    nlo, nhi = cfg['reference_intervals'][0], cfg['reference_intervals'][-1]
    fine_ref, coarse_ref = {}, {}
    for case, param in enumerate(draws):
        t0 = time.perf_counter()
        fine_ref[case] = P.reference(param, nhi)
        coarse_ref[case] = P.reference(param, nlo)
        report['references'].append(dict(case=case, fine_intervals=nhi, coarse_intervals=nlo,
            reference_delta=C.relative(coarse_ref[case][::nlo // nlo, ::nlo // nlo],
                                       fine_ref[case][::nhi // nlo, ::nhi // nlo]),
            fine_sha256=sha_array(fine_ref[case]), seconds=time.perf_counter() - t0))
    save()
    print('REFERENCES done', round(time.perf_counter() - begin, 1), flush=True)

    for n in cfg['intervals']:
        print('MESH', n, flush=True)
        lam = jnp.asarray(C.eigenvalues(n))
        chain = {case: np.array(fine_ref[case][::nhi // n, ::nhi // n], copy=True) for case in fine_ref}

        ops = C.assemble(params, np.asarray(codes), n, cfg['requested_modes'], cfg['lm_budget'])
        report['arm_setup'].append(dict(intervals=n, arm='shared_assembly', **{
            k: ops['info'][k] for k in ('retained_modes', 'stored_features', 'retained_bank_rank',
                                        'trust_delta', 'operator_sha256', 'bank_sha256')}))
        bank, B, S, I, J = ops['bank'], ops['B'], ops['S'], ops['I'], ops['J']
        M = int(B.shape[0])

        t0 = time.perf_counter()
        U = np.stack([np.asarray(C.dst_solve(jnp.asarray(C.full_source(n, q)), lam))[1:-1, 1:-1].ravel()
                      for q in train_draws])
        Ut = jnp.asarray(U.T)
        Qb, Rb = A.whiten(bank)
        coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
        bank_proj = float(jnp.linalg.norm(Ut - Qb @ (Qb.T @ Ut)) / jnp.linalg.norm(Ut))
        gram = Ut.T @ Ut
        w, V = jnp.linalg.eigh(gram)
        w, V = w[::-1], V[:, ::-1]
        energy = float(jnp.sum(jnp.clip(w, 0., None)))
        Vmodes = Ut @ (V[:, :max(ranks)] / jnp.sqrt(jnp.clip(w[:max(ranks)], 1e-300, None))[None, :])
        coords_full = np.asarray(Ut.T @ Vmodes)
        del Ut
        report['snapshots'][str(n)] = dict(sources=len(train_draws), seconds=time.perf_counter() - t0,
                                           snapshot_sha256=sha_array(U),
                                           bank_projection_relative_rms=bank_proj,
                                           pod_total_energy=energy,
                                           pod_eigenvalues=np.asarray(w[:max(ranks)]).tolist(),
                                           pod_tail_fraction={str(k): float(max(energy - float(np.sum(np.asarray(w[:k]))), 0.) / max(energy, 1e-300)) for k in ranks})
        del U
        save()

        Hdec = sc.head(params, jnp.asarray(codes))
        lin, Zlin, Wt, ilin = A.fit_linear_map(Hdec, Rb, K, seed=cfg['fit_seed'])
        quad, iquad = A.fit_quadratic_map(Hdec, Rb, K, Zlin, Wt, seed=cfg['fit_seed'])
        lint, Zlint, Wtt, ilint = A.fit_linear_map(coef_truth, Rb, K, seed=cfg['fit_seed'])
        report['fits'][str(n)] = dict(linear_decoder_output=ilin, quadratic_decoder_output=iquad,
                                      linear_truth_projection=ilint)
        save()

        specs = [dict(name='a_neural', family='neural', k=K, head=lambda z: sc.head(params, z),
                      B=B, bank=bank, Zcand=np.asarray(codes), linear_manifold=False,
                      fit='retained checkpoint, not refitted'),
                 dict(name='b_linear_dec', family='linear', k=K, head=A.linear_head(lin), B=B, bank=bank,
                      Zcand=Zlin, linear_manifold=True,
                      fit='field-metric POD of decoder-output coefficients'),
                 dict(name='b_linear_truth', family='linear', k=K, head=A.linear_head(lint), B=B, bank=bank,
                      Zcand=Zlint, linear_manifold=True,
                      fit='field-metric POD of bank-projected discrete truth'),
                 dict(name='c_quad_dec', family='quadratic', k=K, head=A.quadratic_head(quad), B=B, bank=bank,
                      Zcand=Zlin, linear_manifold=False,
                      fit='quadratic manifold on the same rank-k POD coordinates')]
        for k in ranks:
            Vk = Vmodes[:, :k]
            specs.append(dict(name=f'e_pod{k}', family='pod', k=k, head=A.identity_head(),
                              B=reduce_bank(Vk, S, I, J, n), bank=Vk, Zcand=coords_full[:, :k],
                              linear_manifold=True, fit='classical POD of the same discrete truth'))
        specs.append(dict(name='d_freebank', family='free', k=R, head=A.identity_head(), B=B, bank=bank,
                          Zcand=np.asarray(Hdec), linear_manifold=True,
                          fit='unrestricted bank coefficients'))

        built = []
        for s in specs:
            k = s['k']
            assert M > k, (s['name'], M, k)
            linear = 'gj' if k <= cfg['gauss_jordan_max'] else 'lu'
            trust = radius(s['Zcand'])
            pred = jax.jit(jax.vmap(lambda z: s['B'] @ s['head'](z)))(jnp.asarray(s['Zcand']))
            kern = make_query(s['head'], s['B'], s['bank'], n, trust, cfg['lm_budget'],
                              cfg['stationarity_tolerance'], linear)
            built.append(dict(name=s['name'], spec=s, kernel=kern, predictions=pred,
                              codes=jnp.asarray(s['Zcand']), k=k, linear_solve=linear))
            report['arm_setup'].append(dict(intervals=n, arm=s['name'], family=s['family'], k=k, M=M,
                                            trust_radius=trust, linear_solve=linear, fit=s['fit'],
                                            candidates=int(len(s['Zcand'])),
                                            linear_manifold=s['linear_manifold'],
                                            operator_sha256=C.sha(np.asarray(s['B']))))
            print('ARM', n, s['name'], flush=True)
            save()

        # ---- retained r128_q32: neural head plus exactly eliminated corrections ----
        engine = CC.prepare_correction(ops, np.asarray(codes), basis['coefficient_directions'],
                                       cfg['correction_count'], cfg['retained'])
        report['arm_setup'].append(dict(intervals=n, arm='a_neural_q32', family='neural+linear', k=K, M=M,
                                        **{kk: engine['info'][kk] for kk in
                                           ('correction_count', 'nominal_latent_dimension',
                                            'nonlinear_optimizer_dimension', 'linear_rank',
                                            'linear_rank_valid', 'projected_operator_sha256')}))
        save()

        # ------------------------------------------------ untimed diagnostics ----
        whit = {}
        for b in built:
            s = b['spec']
            Bn = s['bank']
            key = id(Bn)
            if key not in whit:
                whit[key] = A.whiten(Bn)
            Bq, Br = whit[key]
            if s['linear_manifold']:
                c0 = s['head'](jnp.zeros(b['k']))
                Wc = jax.jacfwd(s['head'])(jnp.zeros(b['k']))
                span, _ = jnp.linalg.qr(Bn @ Wc, mode='reduced')
                off = Bn @ c0
            else:
                Hc = jax.jit(jax.vmap(s['head']))(jnp.asarray(s['Zcand']))
                Hn = jnp.sum((Hc @ Br.T) ** 2, 1)
                recon = A.make_reconstruction(s['head'], b['k'], n, cfg['recon_budget'],
                                              linear=b['linear_solve'])
            rows = []
            for case in chain:
                target = jnp.asarray(chain[case][1:-1, 1:-1].ravel())
                nref = float(np.linalg.norm(chain[case]))
                proj = Bq @ (Bq.T @ target)
                bank_err = float(jnp.linalg.norm(proj - target)) / nref
                if s['linear_manifold']:
                    fit = off + span @ (span.T @ (target - off))
                    man = float(jnp.linalg.norm(fit - target)) / nref
                    its = 0
                else:
                    score = Hn - 2 * (Hc @ (Bn.T @ target))
                    starts = jnp.asarray(s['Zcand'])[jnp.argsort(score)[:cfg['recon_starts']]]
                    z, rn, it, reason = jax.device_get(recon(starts, target, Bn))
                    man = float(rn) / nref
                    its = int(it)
                rows.append(dict(case=case, bank_projection=bank_err, best_found=man,
                                 reconstruction_iterations=its))
            report['reconstruction'].append(dict(intervals=n, arm=b['name'], k=b['k'],
                linear_manifold=s['linear_manifold'], cases=rows,
                worst_bank_projection=float(max(r['bank_projection'] for r in rows)),
                worst_best_found=float(max(r['best_found'] for r in rows))))
            print('RECON', n, b['name'], flush=True)
            save()

        # ------------------------------------------------------ timed queries ----
        subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
        subjects.append(dict(kind='retained', name='a_neural_q32'))
        subjects.append(dict(kind='fom', name='dst_direct'))
        report['declared_subjects'] += [dict(intervals=n, **{k: v for k, v in s.items() if k != 'index'})
                                        for s in subjects]
        sources = [C.full_source(n, q) for q in draws]

        def invoke(sub, source, case):
            if sub['kind'] == 'rom':
                b = built[sub['index']]
                return query_once(b['kernel'], source, ops, b['predictions'], b['codes'],
                                  b['spec']['B'], b['spec']['bank'])
            if sub['kind'] == 'retained':
                return CC.correction_query(source, ops, engine, cfg['retained'])
            return C.fom_query(source, lam)

        t = time.perf_counter()
        for sub in subjects:
            invoke(sub, sources[0], 0)
        report.setdefault('compile_warmup', []).append(dict(intervals=n, seconds=time.perf_counter() - t))
        save()

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
                    b = built[sub['index']] if sub['kind'] == 'rom' else None
                    report['invocations'].append(dict(
                        intervals=n, case=case, group=report['cohort']['groups'][case], rep=rep,
                        kind=sub['kind'], name=sub['name'], field_sha256=h, artifact=artifacts[key],
                        physical_error=C.relative(field, chain[case]),
                        source_sha256=C.sha(sources[case]), finite=True,
                        k=(b['k'] if b else (K if sub['kind'] == 'retained' else None)),
                        M=(M if sub['kind'] != 'fom' else None),
                        family=(b['spec']['family'] if b else
                                ('neural+linear' if sub['kind'] == 'retained' else 'fom')),
                        linear_solve=(b['linear_solve'] if b else None), **row))
            print('TIMED', n, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()
        del built, specs, ops, engine, bank, B, Vmodes, coords_full, coef_truth, Hdec
        jax.clear_caches()

    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('POISSON HEAD ABLATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
