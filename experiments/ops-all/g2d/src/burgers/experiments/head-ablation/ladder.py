"""Fixed-weight correction ladder on Burgers 2D at one mesh.

The frozen checkpoint, weak objective, test-mode family, initializer policy,
stopping rule and output contract are the head-ablation arm (a) contract. The
only online knob is the number q of extra linear bank directions solved jointly
with the latent coordinates:

    u(z, y) = G ( h_theta(z) + C_q y ),      w = (z, y) in R^{K+q}.

q = 0 is arm (a) exactly: C_0 has no columns, so h(w) = h_theta(w). Because the
Burgers weak residual is quadratic in the coefficients through the upwind
advection term, y cannot be eliminated analytically as it is on Poisson; the
whole augmented vector is solved by the same Levenberg-Marquardt iteration.

The directions C_q are nested and fixed offline: a field-metric proper orthogonal
decomposition of the decoder-output residual, that is, of what the frozen head
cannot represent on its own training family. A second knob, the time step, is
varied at q = 0 only.
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


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def corrected_head(params, C, K):
    """h(w) = h_theta(w[:K]) + C w[K:]. At q = 0 this is h_theta verbatim."""
    def head(w):
        return A.sc.head(params, w[:K]) + C @ w[K:]
    return head


def generate_snapshots(L, dt, physical, stride, ntol, ltol):
    q, _ = e.make_fom(L, dt, None, .25, dt)
    keep = np.arange(0, int(round(.25 / dt)) + 1, stride)
    out, worst = [], 0.
    t0 = time.perf_counter()
    for phys in physical:
        f, it, rn = host(q(jnp.asarray(e.initial(L, phys)), float(phys[4]), ntol, ltol))
        assert np.isfinite(f).all()
        worst = max(worst, float(np.max(rn)))
        out.append(np.array(f[keep][:, 1:-1, 1:-1].reshape(len(keep), -1), copy=True))
    del q
    jax.clear_caches()
    U = np.concatenate(out, axis=0)
    return U, dict(trajectories=int(len(physical)), states_per_trajectory=int(len(keep)),
                   snapshots=int(U.shape[0]), state_stride=int(stride), newton_tolerance=ntol,
                   linear_tolerance=ltol, max_relative_residual=worst,
                   seconds=time.perf_counter() - t0, snapshot_sha256=sha_array(U))


def residual_directions(params, Rb, coef, Zcand, K, cfg):
    """Field-metric POD of what the frozen head cannot represent.

    For each snapshot's bank coefficients eta, the best-found head code z is found
    with the shared Levenberg-Marquardt rule in the field metric; the residual
    eta - h_theta(z) is then decomposed. Nested by construction, so C_q is the
    first q columns for every q on the ladder.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg['residual_seed'])
    idx = np.sort(rng.choice(len(coef), min(cfg['residual_snapshots'], len(coef)), replace=False))
    target = jnp.asarray(np.asarray(coef)[idx])                       # (S, R)
    head = lambda z: A.sc.head(params, z)
    Hc = jax.jit(jax.vmap(head))(jnp.asarray(Zcand))                  # (C, R)
    Hrot = Hc @ Rb.T
    Hn = jnp.sum(Hrot * Hrot, 1)
    lm = A.make_stationary_lm(lambda z, t: Rb @ (head(z) - t), cfg['residual_budget'],
                              gtol=cfg['strict']['gtol'], linear='gj')

    @jax.jit
    def fit(t):
        score = Hn - 2 * (Hrot @ (Rb @ t))
        starts = jnp.asarray(Zcand)[jnp.argsort(score)[:cfg['residual_starts']]]
        outs = jax.vmap(lambda z0: lm(z0, (t,), 0.))(starts)
        best = jnp.argmin(outs[1])
        return outs[0][best], outs[1][best], outs[2][best], outs[3][best]

    zs, rns, its, reasons = [], [], [], []
    for s in range(0, len(target), 64):
        v = host(jax.vmap(fit)(target[s:s + 64]))
        zs.append(v[0]); rns.append(v[1]); its.append(v[2]); reasons.append(v[3])
    Z = jnp.asarray(np.concatenate(zs))
    rho = target - jax.jit(jax.vmap(head))(Z)                         # (S, R)
    tilde = rho @ Rb.T
    _, sv, Vt = jnp.linalg.svd(tilde, full_matrices=False)
    rank = int(min(Vt.shape[0], coef.shape[1]))
    Ct = Vt[:rank].T                                                  # (R, rank), field-orthonormal
    C = jnp.linalg.solve(Rb, Ct)
    total = float(jnp.sum(sv ** 2))
    fitted = np.sqrt(np.asarray(jnp.sum(tilde * tilde, 1) / jnp.maximum(jnp.sum((target @ Rb.T) ** 2, 1), 1e-300)))
    info = dict(rule=('field-metric POD of the decoder-output residual eta - h_theta(z*), with z* the '
                      'best-found head code under the shared LM rule; nested in q'),
                snapshots=int(len(idx)), snapshot_indices_sha256=sha_array(idx),
                starts=cfg['residual_starts'], budget=cfg['residual_budget'],
                seed=cfg['residual_seed'], available_rank=rank,
                head_fit_relative_median=float(np.median(fitted)),
                head_fit_relative_worst=float(np.max(fitted)),
                head_fit_iterations_median=float(np.median(np.concatenate(its))),
                head_fit_reason_counts={str(k): int(v) for k, v in
                                        zip(*np.unique(np.concatenate(reasons), return_counts=True))},
                singular_values=np.asarray(sv[:min(len(sv), 600)]).tolist(),
                residual_energy_captured={},
                seconds=time.perf_counter() - t0, directions_sha256=sha_array(np.asarray(C)))
    for q in cfg['q_ladder']:
        kept = float(np.sum(np.asarray(sv[:q]) ** 2))
        info['residual_energy_captured'][str(q)] = kept / max(total, 1e-300)
    return C, info


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
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output fields; '
                                   'same-invocation host transfers also measured; identical for every arm'),
                  reference=[], snapshots={}, directions={}, arm_setup=[], reconstruction=[],
                  invocations=[], declared_subjects=[], verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    report['cohort_note'] = ('exactly the six opened development cases of the head-ablation job, which is '
                             'what arm (a) used; the eight refinement calibration cases belong to a '
                             'different lane and are not used here')
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    report['train_physical_sha256'] = sha_array(train_physical)
    save()

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

    U, sinfo = generate_snapshots(L, dt, train_physical, cfg['train_state_stride'],
                                  cfg['snapshot_ntol'], cfg['snapshot_ltol'])
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)
    jax.block_until_ready(Rb)
    Ut = jnp.asarray(U.T)
    coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
    sinfo['bank_projection_relative_rms'] = float(
        jnp.linalg.norm(Ut - Qb @ (Qb.T @ Ut)) / jnp.linalg.norm(Ut))
    del Ut, U
    jax.clear_caches()
    report['snapshots'][str(L)] = sinfo
    save()
    print('SNAPSHOTS', round(sinfo['seconds'], 1), flush=True)

    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    Cfull, dinfo = residual_directions(params, Rb, coef_truth, Zsub, K, {**cfg, 'strict': strict})
    report['directions'] = dinfo
    save()
    print('DIRECTIONS', round(dinfo['seconds'], 1), 'rank', dinfo['available_rank'], flush=True)

    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    Mmax = int(cfg['test_multiplier'] * (K + max(cfg['q_ladder'])))

    # ---------------------------------------------------------------- subjects
    specs = []
    for q in cfg['q_ladder']:
        M = int(cfg['test_multiplier'] * (K + q))
        quads = ['dense'] + (['eq'] if q in cfg['eq_q'] else [])
        for quadrature in quads:
            for step in ([dt] + ([cfg['dt_alt']] if q == 0 else [])):
                tag = '' if step == dt else f'_dt{step:g}'.replace('.', 'p')
                specs.append(dict(name=f'q{q}_{quadrature}{tag}', q=q, M=M, quadrature=quadrature, dt=step))
    # A q = 0 control at the ladder's largest test count, so the cost curve can be
    # read with the growth of M separated from the growth of q.
    specs.append(dict(name='q0_dense_Mmax', q=0, M=Mmax, quadrature='dense', dt=dt))

    built = []
    operators = {}
    for s in specs:
        q, M = s['q'], s['M']
        assert M > K + q, (s['name'], M, K + q)
        C = Cfull[:, :q]
        head = corrected_head(params, C, K)
        Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        if s['quadrature'] == 'eq':
            m = int(cfg['quadrature_multiplier'] * M)
            # q = 0 must reuse arm (a)'s exact rule: same full code table, same seed.
            eq_codes = (np.concatenate((Zold, np.zeros((len(Zold), q))), axis=1) if q == 0
                        else Zaug)
        else:
            m, eq_codes = None, None
        t0 = time.perf_counter()
        # Reduced operators and the initializer do not depend on the time step, so
        # the two time-step arms at q = 0 share one build and one quadrature fit.
        key = (q, M, s['quadrature'])
        if key not in operators:
            operators[key] = (A.build_operators(bank, L, M, s['quadrature'], Zcoef=eq_codes, m=m,
                                                eq_seed=cfg['eq_seed'], candidate_cap=cfg['candidate_cap'],
                                                fit_states=cfg['fit_states'], head=head),
                              A.build_cold(bank, head, Zaug, cfg['cold_axis_points']))
        (data, info), (cold, cinfo) = operators[key]
        info = dict(info)
        linear = 'gj' if (K + q) <= cfg['gauss_jordan_max'] else 'lu'
        query = A.make_query(head, K + q, L, s['dt'], trust, s['quadrature'], linear=linear, **strict)
        info.update(arm=s['name'], q=q, solved_dimension=K + q, dt=s['dt'], trust_radius=trust,
                    linear_solve=linear, cold=cinfo, total_setup_seconds=time.perf_counter() - t0,
                    correction_directions_used=q)
        report['arm_setup'].append(info)
        report['declared_subjects'].append(dict(name=s['name'], method='rom', q=q, M=M, m=m,
                                                quadrature=s['quadrature'], dt=s['dt'],
                                                solved_dimension=K + q, linear_solve=linear))
        built.append(dict(**s, data=data, cold=cold, query=query, head=head, C=C,
                          linear_solve=linear, m=m))
        print('ARM', s['name'], round(info['total_setup_seconds'], 1), flush=True)
        save()

    foms = {'fft': ip.make_fom(L, dt, 'fft')}
    for fs in cfg['fom_settings']:
        report['declared_subjects'].append(dict(method='fom', dt=dt, **fs))

    # -------------------------------------------------- untimed diagnostics ---
    Bq, Br = Qb, Rb
    recon_cache = {}
    for b in built:
        if (b['q'], 'recon') in recon_cache:
            continue
        head = b['head']
        Hc = jax.jit(jax.vmap(head))(jnp.asarray(np.concatenate(
            (Zsub, np.zeros((len(Zsub), b['q']))), axis=1)))
        Hn = jnp.sum((Hc @ Br.T) ** 2, 1)
        recon = A.make_reconstruction(head, K + b['q'], L, cfg['recon_budget'], linear=b['linear_solve'])
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            bank_err, man_err, iters = [], [], []
            for ti in range(ref.shape[0]):
                target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                bank_err.append(float(jnp.linalg.norm(Bq @ (Bq.T @ target) - target)) / n0)
                score = Hn - 2 * (Hc @ (G.T @ target))
                starts = jnp.asarray(np.concatenate((Zsub, np.zeros((len(Zsub), b['q']))), axis=1))[
                    jnp.argsort(score)[:cfg['recon_starts']]]
                z, rn, it, reason = host(recon(starts, target, G))
                man_err.append(float(rn) / n0)
                iters.append(int(it))
            rows.append(dict(case=case, bank_projection_per_time=bank_err,
                             bank_projection_max=float(np.max(bank_err)),
                             best_found_per_time=man_err, best_found_max=float(np.max(man_err)),
                             reconstruction_iterations=iters))
        entry = dict(q=b['q'], solved_dimension=K + b['q'], cases=rows,
                     worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
                     worst_best_found=float(max(r['best_found_max'] for r in rows)))
        recon_cache[(b['q'], 'recon')] = entry
        report['reconstruction'].append(entry)
        print('RECON q', b['q'], round(entry['worst_best_found'] * 100, 5), flush=True)
        save()

    # ------------------------------------------------------- timed queries ----
    subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
    subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]
    inputs = [e.initial(L, phys) for phys in physical]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, u, case):
        nu = float(physical[case, 4])
        if sub['kind'] == 'rom':
            b = built[sub['index']]
            return b['query'](u, nu, b['data'], b['cold'])
        fs = sub['setting']
        fn, pre = foms[fs['preconditioner']]
        return fn(u, nu, fs['ntol'], fs['ltol'], *pre)

    t = time.perf_counter()
    for sub in subjects:
        jax.block_until_ready(invoke(sub, jnp.asarray(inputs[0]), 0))
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
    print('WARMUP', round(time.perf_counter() - t, 1), flush=True)
    save()

    artifacts = {}
    for rep in range(cfg['reps']):
        for case in range(len(physical)):
            for i in order_rng.permutation(len(subjects)):
                sub = subjects[int(i)]
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
                    extra = dict(internal_latents=v[7]) if sub['kind'] == 'rom' else {}
                    np.savez_compressed(out / fn, fields=f, **extra)
                    artifacts[key] = fn
                row = dict(intervals=L, case=case, cohort=report['cohort_roles'][case], rep=rep,
                           kind=sub['kind'], name=sub['name'], gpu_seconds=gs, host_seconds=hs,
                           output_bytes=int(f.nbytes), field_sha256=h, artifact=artifacts[key],
                           error=e.errors(f, refs[case], L), iterations=v[1].tolist(),
                           residuals=v[2].tolist(), finite=True)
                if sub['kind'] == 'rom':
                    b = built[sub['index']]
                    reasons = v[3].tolist()
                    row.update(q=b['q'], solved_dimension=K + b['q'], M=b['M'], m=b['m'],
                               quadrature=b['quadrature'], dt=b['dt'], linear_solve=b['linear_solve'],
                               stop_reasons=reasons, ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=v[8].tolist(), ic_stationarity=float(v[9]),
                               ic_residual=float(v[10]), ic_input_norm=float(v[11]),
                               ic_relative_residual=float(v[10]) / max(float(v[11]), 1e-300),
                               budget_exits=int(sum(1 for r in reasons if r == 0)),
                               rejected_exits=int(sum(1 for r in reasons if r == 3)),
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
    print('CORRECTION LADDER COMPLETE', flush=True)


if __name__ == '__main__':
    main()
