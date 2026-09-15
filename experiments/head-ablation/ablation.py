"""Head ablation at matched latent dimension in one frozen Burgers spatial bank.

Arms (a) neural, (b) linear, (c) quadratic, (d) free bank coefficients and
(e) classical POD-LSPG run through the same weak objective, test modes, time
discretization, initializer policy, stopping rule and output contract. Every
timed invocation pairs its own cost and accuracy; all repetitions are retained.
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


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


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


def pod_basis(Ut, kmax):
    """Method of snapshots on the plain grid inner product. Ut is (n, Ns)."""
    gram = Ut.T @ Ut
    w, V = jnp.linalg.eigh(gram)
    w = w[::-1]
    V = V[:, ::-1]
    total = float(jnp.sum(jnp.clip(w, 0., None)))
    wk = jnp.clip(w[:kmax], 1e-300, None)
    modes = Ut @ (V[:, :kmax] / jnp.sqrt(wk)[None, :])
    return jax.block_until_ready(modes), np.asarray(w[:kmax]), total


def radius(Z):
    Z = np.asarray(Z)
    return .01 * float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))


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
    dt = cfg['dt']
    strict = cfg['strict']

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
                  checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R,
                  spatial_bank_frozen=True, network_weights_frozen=True, final_cohort_unopened=True,
                  output_times=[0, .05, .1, .15, .2, .25],
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output fields; '
                                   'same-invocation host transfers also measured; identical for every arm'),
                  reference=[], snapshots={}, fits={}, arm_setup=[], reconstruction=[],
                  invocations=[], declared_subjects=[], verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    report['train_physical_sha256'] = sha_array(train_physical)
    save()

    meshes = cfg['meshes']
    largest = max(meshes)
    rf, rt = cfg['reference_mesh'], cfg['reference_dt']
    refs = {}
    q, _ = e.make_fom(rf, rt)
    for case, phys in enumerate(physical):
        t = time.perf_counter()
        f, it, rn = host(q(jnp.asarray(e.initial(rf, phys)), float(phys[4]), 1e-11, 1e-9))
        assert np.isfinite(f).all() and np.max(rn) < 2e-11
        f = np.array(f[:, ::rf // largest, ::rf // largest], copy=True)
        refs[case] = f
        name = f'ref_L{rf}_dt{rt}_case{case}.npz'
        np.savez_compressed(out / name, fields=f, iterations=it, residuals=rn)
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=largest,
                                        field_sha256=sha_array(f), seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q
    jax.clear_caches()

    order_rng = np.random.default_rng(cfg['order_seed'])

    for L in meshes:
        print('MESH', L, flush=True)
        truth = {c: refs[c][:, ::largest // L, ::largest // L] for c in refs}

        U, sinfo = generate_snapshots(L, dt, train_physical, cfg['train_state_stride'],
                                      cfg['snapshot_ntol'], cfg['snapshot_ltol'])
        report['snapshots'][str(L)] = sinfo
        save()
        print('SNAPSHOTS', L, round(sinfo['seconds'], 1), flush=True)

        bank = A.CoordBank(params, K, R)
        G = bank.on_grid(L)
        Qb, Rb = A.whiten(G)
        jax.block_until_ready(Rb)
        stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
        Zsub = jnp.asarray(Zold[::stride])
        Hdec = jax.jit(lambda z: A.sc.head(params, z))(Zsub)

        Ut = jnp.asarray(U.T)
        coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
        bank_proj = float(jnp.linalg.norm(Ut - Qb @ (Qb.T @ Ut)) / jnp.linalg.norm(Ut))
        ranks = sorted(set(cfg['pod_ranks'] + [K]))
        Vmodes, eigen, energy = pod_basis(Ut, max(ranks))
        coords_full = np.asarray(Ut.T @ Vmodes)
        del Ut
        U = None
        jax.clear_caches()
        report['snapshots'][str(L)].update(
            bank_projection_relative_rms=bank_proj, pod_total_energy=energy,
            pod_eigenvalues=np.asarray(eigen).tolist(),
            pod_tail_fraction={str(k): float(max(energy - float(np.sum(eigen[:k])), 0.) / max(energy, 1e-300))
                               for k in ranks})

        lin_dec, Zlin_dec, Wt_dec, info_lin_dec = A.fit_linear_map(Hdec, Rb, K, seed=cfg['fit_seed'])
        quad_dec, info_quad_dec = A.fit_quadratic_map(Hdec, Rb, K, Zlin_dec, Wt_dec, seed=cfg['fit_seed'])
        lin_tru, Zlin_tru, Wt_tru, info_lin_tru = A.fit_linear_map(coef_truth, Rb, K, seed=cfg['fit_seed'])
        report['fits'][str(L)] = dict(linear_decoder_output=info_lin_dec, quadratic_decoder_output=info_quad_dec,
                                      linear_truth_projection=info_lin_tru,
                                      decoder_codes=int(len(Zsub)), code_stride=int(stride))
        save()

        specs = [
            dict(name='a_neural', family='neural', k=K, head=A.neural_head(params), bank=bank,
                 Zcand=np.asarray(Zsub), trust=radius(Zold), linear_manifold=False, eq_codes=Zold,
                 fit='retained checkpoint, not refitted'),
            dict(name='b_linear_dec', family='linear', k=K, head=A.linear_head(lin_dec), bank=bank,
                 Zcand=Zlin_dec, trust=radius(Zlin_dec), linear_manifold=True, eq_codes=Zlin_dec,
                 fit='field-metric POD of decoder-output coefficients h(Z_train)'),
            dict(name='b_linear_truth', family='linear', k=K, head=A.linear_head(lin_tru), bank=bank,
                 Zcand=Zlin_tru, trust=radius(Zlin_tru), linear_manifold=True, eq_codes=Zlin_tru,
                 fit='field-metric POD of bank-projected truth snapshots'),
            dict(name='c_quad_dec', family='quadratic', k=K, head=A.quadratic_head(quad_dec), bank=bank,
                 Zcand=Zlin_dec, trust=radius(Zlin_dec), linear_manifold=False, eq_codes=Zlin_dec,
                 fit='quadratic manifold on the same rank-k POD coordinates'),
        ]
        for k in ranks:
            co = coords_full[:, :k]
            specs.append(dict(name=f'e_pod{k}', family='pod', k=k, head=A.identity_head(),
                              bank=A.GridBank(Vmodes[:, :k], L), Zcand=co,
                              trust=radius(co), linear_manifold=True, eq_codes=co,
                              fit='classical POD of the same truth snapshots'))
        specs.append(dict(name='d_freebank', family='free', k=R, head=A.identity_head(), bank=bank,
                          Zcand=np.asarray(Hdec), trust=radius(Hdec), linear_manifold=True,
                          eq_codes=np.asarray(Hdec), fit='unrestricted bank coefficients',
                          force_M=int(cfg['free_bank_M'])))

        built = []
        for s in specs:
            for quadrature in cfg['quadratures'].get(s['name'], ['dense']):
                k = s['k']
                M = int(s.get('force_M', cfg['test_multiplier'] * k))
                m = int(cfg['quadrature_multiplier'] * M) if quadrature == 'eq' else None
                linear = 'gj' if k <= cfg['gauss_jordan_max'] else 'lu'
                t0 = time.perf_counter()
                data, info = A.build_operators(s['bank'], L, M, quadrature, Zcoef=s['eq_codes'], m=m,
                                               eq_seed=cfg['eq_seed'], candidate_cap=cfg['candidate_cap'],
                                               fit_states=cfg['fit_states'], head=s['head'])
                cold, cinfo = A.build_cold(s['bank'], s['head'], s['Zcand'], cfg['cold_axis_points'])
                name = f"{s['name']}_{quadrature}"
                info.update(arm=name, family=s['family'], k=k, trust_radius=s['trust'], linear_solve=linear,
                            fit=s['fit'], cold=cinfo, total_setup_seconds=time.perf_counter() - t0,
                            linear_manifold=s['linear_manifold'])
                report['arm_setup'].append(info)
                query = A.make_query(s['head'], k, L, dt, s['trust'], quadrature, linear=linear, **strict)
                built.append(dict(name=name, spec=s, data=data, cold=cold, query=query, k=k, M=M,
                                  m=m, quadrature=quadrature, linear_solve=linear))
                report['declared_subjects'].append(dict(intervals=L, name=name, method='rom', family=s['family'],
                                                        k=k, M=M, m=m, quadrature=quadrature, dt=dt,
                                                        linear_solve=linear))
                print('ARM', L, name, round(info['total_setup_seconds'], 1), flush=True)
                save()

        foms = {'fft': ip.make_fom(L, dt, 'fft')}
        for fs in cfg['fom_settings']:
            report['declared_subjects'].append(dict(intervals=L, method='fom', dt=dt, **fs))

        # ------------------------------------------------- untimed diagnostics
        for b in built:
            s = b['spec']
            B = b['data']['G']
            Bq, Br = A.whiten(B)
            hp_cols = None
            if s['linear_manifold']:
                if s['family'] == 'linear':
                    c0 = s['head'](jnp.zeros(b['k']))
                    Wc = jax.jacfwd(s['head'])(jnp.zeros(b['k']))
                    span, _ = jnp.linalg.qr(B @ Wc, mode='reduced')
                    hp_cols = (span, B @ c0)
                else:
                    span, _ = jnp.linalg.qr(B, mode='reduced')
                    hp_cols = (span, jnp.zeros(B.shape[0]))
            else:
                Hc = jax.jit(jax.vmap(s['head']))(jnp.asarray(s['Zcand']))
                Hrot = Hc @ Br.T
                Hn = jnp.sum(Hrot * Hrot, 1)
                recon = A.make_reconstruction(s['head'], b['k'], L, cfg['recon_budget'],
                                              linear=b['linear_solve'])
            rows = []
            for case in refs:
                ref = truth[case]
                n0 = float(np.linalg.norm(ref[0]))
                bank_err, man_err, iters = [], [], []
                for ti in range(ref.shape[0]):
                    target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                    proj = Bq @ (Bq.T @ target)
                    bank_err.append(float(jnp.linalg.norm(proj - target)) / n0)
                    if s['linear_manifold']:
                        span, off = hp_cols
                        d = target - off
                        fit = off + span @ (span.T @ d)
                        man_err.append(float(jnp.linalg.norm(fit - target)) / n0)
                        iters.append(0)
                    else:
                        bt = B.T @ target
                        score = Hn - 2 * (Hc @ bt)
                        starts = jnp.asarray(s['Zcand'])[jnp.argsort(score)[:cfg['recon_starts']]]
                        z, rn, it, reason = host(recon(starts, target, B))
                        man_err.append(float(rn) / n0)
                        iters.append(int(it))
                rows.append(dict(case=case, bank_projection_per_time=bank_err,
                                 bank_projection_max=float(np.max(bank_err)),
                                 best_found_per_time=man_err, best_found_max=float(np.max(man_err)),
                                 reconstruction_iterations=iters))
            report['reconstruction'].append(dict(intervals=L, arm=b['name'], k=b['k'],
                                                 linear_manifold=s['linear_manifold'], cases=rows,
                                                 worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
                                                 worst_best_found=float(max(r['best_found_max'] for r in rows))))
            print('RECON', L, b['name'], round(max(r['best_found_max'] for r in rows) * 100, 5), flush=True)
            save()

        # ------------------------------------------------------ timed queries
        subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
        subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]
        inputs = [e.initial(L, phys) for phys in physical]

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
        report.setdefault('compile_warmup', []).append(dict(intervals=L, seconds=time.perf_counter() - t,
                                                            subjects=len(subjects)))
        print('WARMUP', L, round(time.perf_counter() - t, 1), flush=True)
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
                               error=e.errors(f, truth[case], L), iterations=v[1].tolist(),
                               residuals=v[2].tolist(), finite=True)
                    if sub['kind'] == 'rom':
                        b = built[sub['index']]
                        row.update(k=b['k'], M=b['M'], m=b['m'], quadrature=b['quadrature'],
                                   family=b['spec']['family'], linear_solve=b['linear_solve'],
                                   stop_reasons=v[3].tolist(), ic_iterations=int(v[5]), ic_reason=int(v[6]),
                                   step_stationarity=v[8].tolist(), ic_stationarity=float(v[9]),
                                   stationary=bool(max(float(np.max(v[8])), float(v[9]))
                                                   <= strict['gtol'] * (1 + 1e-7)))
                    else:
                        row.update(nonlinear_converged=bool(np.max(v[2]) <= sub['setting']['ntol'] * (1 + 1e-9)))
                    report['invocations'].append(row)
            print('TIMED', L, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()
        del built, foms, specs, Vmodes, coords_full, coef_truth, Hdec, G, Qb, Rb
        jax.clear_caches()

    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HEAD ABLATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
