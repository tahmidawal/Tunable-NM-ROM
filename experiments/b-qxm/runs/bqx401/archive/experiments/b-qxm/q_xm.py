"""The crossed (q, M) grid: is the ladder's accuracy control the rank q, the test count M, or both?

`q_ridge.py` (job qrg201) with the sweep replaced by explicit cells and NOTHING else moved.
Same frozen checkpoint, same OLD nested direction rule and seed, same reachable set, same
weak objective (dense quadrature), same initializer policy, same stopping rule, same
per-step budget 600, same mesh, same time step, the same six opened development cases, and
the same lambda = 0 retained solver path (`ridge.make_query` at lambda = 0 is a Python
branch to `topfix.make_query('base')` / `varpro.make_query`, the audited b-ladder-top
contract). Each cell declares two numbers:

  q   the number of correction directions (the first q columns of the nested C)
  M   the number of weak test modes, an explicit integer rather than a multiplier

Three mechanics differ from qrg201, none of which touches the arithmetic of any arm:

  * dense operators depend on M only, so they are built once per distinct M and shared;
  * the bank G on the grid is a deterministic function of the checkpoint, so it is built
    once and shared by every arm after an in-job bitwise equality check against a fresh
    build (gate `operators_shared_bitwise`);
  * the in-job gate is `t0_field_invariant_in_M`: the initial fit uses only the cold Gauss
    space and C_q, never the test modes, so at fixed q the t = 0 output field must not
    depend on M.

`bank_G.npz` is not written (no R3 in this lane); per-step bank coefficients still are,
untimed. Everything is written incrementally to `result.json` so a truncated job is
collectable.
"""
from __future__ import annotations

import argparse
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
import ladder as LD
import directions as DIR
import topfix as TF
import ridge as RG

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump


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
    assert cfg['quadrature'] == 'dense'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    assert K == cfg['K']
    L = int(cfg['intervals'])
    dt = cfg['dt']
    strict = cfg['strict']

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  jax_version=jax.__version__, checkpoint_sha256=sha_file(a.checkpoint),
                  K=K, R=R, intervals=L, dt=dt,
                  spatial_bank_frozen=True, network_weights_frozen=True,
                  final_cohort_unopened=True,
                  output_times=[0, .05, .1, .15, .2, .25],
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output '
                                   'fields; identical for every arm. The per-step bank '
                                   'coefficients are written OUTSIDE the timed query and are '
                                   'never charged to any arm.'),
                  reference=[], snapshots={}, directions={}, arm_setup=[], reconstruction=[],
                  invocations=[], declared_subjects=[], gates={}, verification=ip.verify(),
                  complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['physical_sha256'] = sha_array(physical)
    want_phys = cfg.get('expected_physical_sha256')
    report['gates']['evaluation_cohort_bitwise_abl01'] = dict(
        expected=want_phys, got=report['physical_sha256'],
        passed=(None if want_phys is None else report['physical_sha256'] == want_phys))
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    report['cohort_note'] = ('exactly the six opened development cases of the audited ladder, of '
                             'cheap-corrections, of b-ladder-top, of q-trajdirs and of q-ridge; '
                             'no new case is opened')
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'train/eval overlap'
    report['train_physical_sha256'] = sha_array(train_physical)
    save()

    # ------------------------------------------------------------- references --
    rf, rt = cfg['reference_mesh'], cfg['reference_dt']
    refs = {}
    q_fom, _ = e.make_fom(rf, rt)
    for case, phys in enumerate(physical):
        t = time.perf_counter()
        f, it, rn = host(q_fom(jnp.asarray(e.initial(rf, phys)), float(phys[4]), 1e-11, 1e-9))
        assert np.isfinite(f).all() and np.max(rn) < 2e-11
        f = np.array(f[:, ::rf // L, ::rf // L], copy=True)
        refs[case] = f
        name = f'ref_L{rf}_dt{rt}_case{case}.npz'
        np.savez_compressed(out / name, fields=f, iterations=it, residuals=rn)
        got = sha_array(f)
        srcs = cfg.get('source_reference_sha256') or {}
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=L,
                                        field_sha256=got,
                                        bitwise_matches={k: (v.get(str(case)) == got)
                                                         for k, v in srcs.items()},
                                        seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q_fom
    jax.clear_caches()

    # ------------------------------------------------------------- snapshots ---
    U, sinfo = LD.generate_snapshots(L, dt, train_physical, cfg['train_state_stride'],
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
    report['bank_G'] = dict(sha256=sha_array(np.asarray(G)), shape=list(G.shape),
                            written=False,
                            note='the frozen coordinate bank on the query grid; hashed, not written')
    save()
    print('SNAPSHOTS', round(sinfo['seconds'], 1), flush=True)

    # ------------------------------------------------------------- directions --
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    Cfull, Ct, Zstar, rho, dinfo = DIR.audited(params, Rb, coef_truth, Zsub, K,
                                               {**cfg, 'strict': strict})
    want = cfg.get('expected_directions_sha256')
    dinfo['expected_sha256'] = want
    dinfo['bitwise_matches_qlad01'] = (None if want is None else dinfo['directions_sha256'] == want)
    dinfo['bitwise_matches_sources'] = {k: (v == dinfo['directions_sha256']) for k, v in
                                        (cfg.get('source_directions_sha256') or {}).items()}
    report['directions'] = dinfo
    report['gates']['directions_hash'] = dict(expected=want, got=dinfo['directions_sha256'],
                                              passed=dinfo['bitwise_matches_qlad01'])
    np.savez_compressed(out / 'directions.npz', C=np.asarray(Cfull))
    report['directions']['artifact'] = 'directions.npz'
    save()
    print('DIRECTIONS', round(dinfo['seconds'], 1), 'rank', dinfo['available_rank'],
          'hash', dinfo['directions_sha256'][:12], 'matches', dinfo['bitwise_matches_sources'],
          flush=True)

    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    report['trust_radius'] = trust

    # ---------------------------------------------------------------- subjects -
    specs, seen = [], set()
    for c in cfg['cells']:
        q, M = int(c['q']), int(c['M'])
        assert M > K + q, (q, M)
        linear_override = c.get('linear')
        name = f'q{q}_M{M}_dense' + (f'_{linear_override}' if linear_override else '')
        assert name not in seen, name
        seen.add(name)
        specs.append(dict(name=name, q=q, M=M, labels=list(c['labels']), quadrature='dense',
                          rule='+'.join(c['labels']), lam_rel=0., fitter='bounded',
                          linear_override=linear_override))
    assert dinfo['available_rank'] >= max(s['q'] for s in specs)
    save()

    # ---------------------------------------------------------------- build ----
    # Dense operators are a function of M alone (`topfix.build_operators` ignores the head
    # for quadrature='dense'); the bank G is a function of the checkpoint alone. Both are
    # built once and shared. The first fresh build is compared bitwise to the shared copy.
    built, ops, colds = {}, {}, {}
    Gshared = G
    share_checks = []
    for s in sorted(specs, key=lambda s: (s['q'], s['M'])):
        q, M = s['q'], s['M']
        C = Cfull[:, :q]
        head = RG.corrected_head(params, C, K)
        if q not in colds:
            Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
            colds[q] = A.build_cold(bank, head, Zaug, cfg['cold_axis_points'])
        cold, cinfo = colds[q]
        t0 = time.perf_counter()
        if M not in ops:
            data, info = TF.build_operators(bank, L, M, 'dense')
            same = bool(np.array_equal(np.asarray(data['G']), np.asarray(Gshared)))
            share_checks.append(dict(M=M, fresh_bank_bitwise_equals_shared=same))
            assert same, ('fresh bank differs from the shared bank; refusing to share', M)
            data['G'] = Gshared
            ops[M] = (data, info)
        data, info = ops[M]
        info = dict(info)
        dim = K + q
        linear = 'gj' if dim <= cfg['gauss_jordan_max'] else 'lu'
        if s.get('linear_override'):
            # The solver control (DESIGN.md §3): the same cell with the other inner linear
            # solve, so the gj -> lu switch at K + q > 64 is not confounded with the rank.
            linear = s['linear_override']
        sig = RG.sigma_q(data, C)
        query = RG.make_query(params, C, K, q, L, dt, trust, 'dense', 0.,
                              ic_budget=strict['ic_budget'], step_budget=strict['step_budget'],
                              gtol=strict['gtol'], ic_gtol=cfg['ic_gtol'], linear=linear,
                              inner_damping=cfg['inner_damping'])
        info.update(arm=s['name'], q=q, M=M, labels=s['labels'], rule=s['rule'], lam_rel=0.,
                    lam_abs=0., sigma_q=sig, fitter=s['fitter'],
                    step_budget=strict['step_budget'], solved_dimension=dim, dt=dt,
                    trust_radius=trust, linear_solve=linear, linear_override=s.get('linear_override'),
                    cold=cinfo,
                    tests_per_unknown=float(M) / float(dim), operators_shared_by_M=True,
                    total_setup_seconds=time.perf_counter() - t0, correction_directions_used=q)
        report['arm_setup'].append(info)
        report['declared_subjects'].append(dict(
            name=s['name'], method='rom', q=q, M=M, m=None, rule=s['rule'], labels=s['labels'],
            quadrature='dense', lam_rel=0., lam_abs=0., fitter=s['fitter'],
            solved_dimension=dim, linear_solve=linear, dt=dt))
        built[s['name']] = dict(**s, data=data, cold=cold, query=query, C=C, m=None,
                                linear_solve=linear, lam_abs=0., sigma_q=sig,
                                coef_fn=RG.coefficients(params, C, K))
        print('ARM', s['name'], 'M', M, 'sigma', sig, round(info['total_setup_seconds'], 1),
              flush=True)
        save()
    report['gates']['operators_shared_bitwise'] = dict(
        checks=share_checks, passed=bool(all(c['fresh_bank_bitwise_equals_shared'] for c in share_checks)))
    save()

    foms = {}
    for fs in cfg['fom_settings']:
        key = (fs['preconditioner'], fs['dt'])
        if key not in foms:
            foms[key] = ip.make_fom(L, fs['dt'], fs['preconditioner'])
        report['declared_subjects'].append(dict(method='fom', **fs))
    inputs = [e.initial(L, phys) for phys in physical]

    # ------------------------------------------------- untimed diagnostics -----
    for q in sorted({s['q'] for s in specs}):
        C = Cfull[:, :q]
        head = RG.corrected_head(params, C, K)
        Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        Hc = jax.jit(jax.vmap(head))(jnp.asarray(Zaug))
        Hn = jnp.sum((Hc @ Rb.T) ** 2, 1)
        recon = A.make_reconstruction(head, K + q, L, cfg['recon_budget'],
                                      linear=('gj' if K + q <= cfg['gauss_jordan_max'] else 'lu'))
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            bank_err, man_err, iters = [], [], []
            for ti in range(ref.shape[0]):
                target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                bank_err.append(float(jnp.linalg.norm(Qb @ (Qb.T @ target) - target)) / n0)
                score = Hn - 2 * (Hc @ (G.T @ target))
                starts = jnp.asarray(Zaug)[jnp.argsort(score)[:cfg['recon_starts']]]
                z, rn, it, reason = host(recon(starts, target, G))
                man_err.append(float(rn) / n0)
                iters.append(int(it))
            rows.append(dict(case=case, bank_projection_per_time=bank_err,
                             bank_projection_max=float(np.max(bank_err)),
                             best_found_per_time=man_err, best_found_max=float(np.max(man_err)),
                             reconstruction_iterations=iters))
        entry = dict(q=q, solved_dimension=K + q, cases=rows,
                     worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
                     worst_best_found=float(max(r['best_found_max'] for r in rows)))
        report['reconstruction'].append(entry)
        print('RECON q', q, round(entry['worst_best_found'] * 100, 5), flush=True)
        save()
    jax.clear_caches()

    # ---------------------------------------------------------- timed queries --
    subjects = [dict(kind='rom', name=s['name']) for s in specs]
    subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, u, case):
        nu = float(physical[case, 4])
        if sub['kind'] == 'rom':
            b = built[sub['name']]
            return b['query'](u, nu, b['data'], b['cold'])
        fs = sub['setting']
        fn, pre = foms[(fs['preconditioner'], fs['dt'])]
        return fn(u, nu, fs['ntol'], fs['ltol'], *pre)

    t = time.perf_counter()
    mem = []
    for sub in subjects:
        jax.block_until_ready(invoke(sub, jnp.asarray(inputs[0]), 0))
        st = jax.devices()[0].memory_stats() or {}
        mem.append(dict(subject=sub['name'], bytes_in_use=st.get('bytes_in_use'),
                        peak_bytes_in_use=st.get('peak_bytes_in_use')))
        print('WARM', sub['name'], round(time.perf_counter() - t, 1),
              'device_MB', (st.get('bytes_in_use') or 0) // 2**20, flush=True)
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects),
                                    device_memory_after_each=mem)
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
                    extra = {}
                    if sub['kind'] == 'rom':
                        # UNTIMED: the per-step bank coefficients, for any later audit.
                        coef = np.asarray(built[sub['name']]['coef_fn'](value[7]))
                        extra = dict(internal_latents=v[7], coefficients=coef)
                    np.savez_compressed(out / fn, fields=f, **extra)
                    artifacts[key] = fn
                row = dict(intervals=L, case=case, cohort=report['cohort_roles'][case], rep=rep,
                           kind=sub['kind'], name=sub['name'], gpu_seconds=gs, host_seconds=hs,
                           output_bytes=int(f.nbytes), field_sha256=h,
                           t0_field_sha256=sha_array(f[0]), artifact=artifacts[key],
                           error=e.errors(f, refs[case], L), iterations=v[1].tolist(),
                           residuals=v[2].tolist(), finite=True,
                           family=('rom' if sub['kind'] == 'rom' else 'fom'))
                if sub['kind'] == 'rom':
                    b = built[sub['name']]
                    reasons = v[3].tolist()
                    gj = np.asarray(v[12], dtype=float)
                    worst = max(float(np.max(gj)), float(v[13]))
                    scale = float(v[11])
                    row.update(q=b['q'], solved_dimension=K + b['q'], M=b['M'], m=None,
                               rule=b['rule'], labels=b['labels'], lam_rel=0., lam_abs=0.,
                               sigma_q=b['sigma_q'], fitter=b['fitter'],
                               step_budget=strict['step_budget'], quadrature='dense',
                               dt=dt, linear_solve=b['linear_solve'], stop_reasons=reasons,
                               ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=v[8].tolist(), ic_stationarity=float(v[9]),
                               step_joint_stationarity=gj.tolist(),
                               ic_joint_stationarity=float(v[13]),
                               step_inner_stationarity=np.asarray(v[14], dtype=float).tolist(),
                               step_residual_over_tolerance=(
                                   np.asarray(v[2], dtype=float) / max(1e-9 * scale, 1e-300)).tolist(),
                               ic_residual=float(v[10]), ic_input_norm=scale,
                               ic_relative_residual=float(v[10]) / max(scale, 1e-300),
                               budget_exits=int(sum(1 for r in reasons if r == 0)),
                               rejected_exits=int(sum(1 for r in reasons if r == 3)),
                               residual_exits=int(sum(1 for r in reasons if r == 1)),
                               gradient_exits=int(sum(1 for r in reasons if r == 4)),
                               tiny_step_exits=int(sum(1 for r in reasons if r == 2)),
                               worst_joint_stationarity=worst,
                               stationary=bool(worst <= strict['gtol'] * (1 + 1e-7)),
                               completed=bool(all(r in (1, 2, 4) for r in reasons)
                                              and int(v[6]) in (1, 2, 4)),
                               converged=bool(worst <= strict['gtol'] * (1 + 1e-7)
                                              and all(r in (1, 2, 4) for r in reasons)
                                              and int(v[6]) in (1, 2, 4)
                                              and sum(1 for r in reasons if r == 0) == 0))
                else:
                    row.update(dt=sub['setting']['dt'],
                               nonlinear_converged=bool(
                                   np.max(v[2]) <= sub['setting']['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # ------------------------------------------------------------- in-job gates
    # The initial fit uses only the cold Gauss space and C_q, never the M test modes, so at
    # fixed (q, case) the t = 0 output field must be identical across M in exact arithmetic.
    # Each M is its own jitted program, and an ulp-level difference in a GEMM inside the IC
    # Levenberg-Marquardt loop can change its iterate path within the 1e-6 stationarity
    # tolerance (seen on the GB10, DESIGN.md §9 A0), so the BLOCKING gate is 1e-6 relative;
    # 1e-12 and bitwise are recorded as probes.
    groups, fields0 = {}, {}
    for x in report['invocations']:
        if x['kind'] != 'rom':
            continue
        groups.setdefault((x['q'], x['case']), set()).add(x['t0_field_sha256'])
        fields0.setdefault((x['q'], x['case']), {})[x['M']] = x['artifact']
    worst_rel = 0.
    for (q, case), arts in fields0.items():
        Ms = sorted(arts)
        f0 = np.load(out / arts[Ms[0]])['fields'][0]
        for M in Ms[1:]:
            f1 = np.load(out / arts[M])['fields'][0]
            worst_rel = max(worst_rel, float(np.linalg.norm(f1 - f0) / np.linalg.norm(f0)))
    families = {f'q{k[0]}_case{k[1]}': sorted(v) for k, v in groups.items()}
    report['gates']['t0_field_invariant_in_M'] = dict(
        families=len(families), bitwise_violations={k: v for k, v in families.items() if len(v) > 1},
        bitwise=bool(all(len(v) == 1 for v in families.values())),
        worst_relative_difference=worst_rel, tolerance=1e-6, passed=bool(worst_rel <= 1e-6),
        within_1e12=bool(worst_rel <= 1e-12))
    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('Q-XM COMPLETE', flush=True)


if __name__ == '__main__':
    main()
