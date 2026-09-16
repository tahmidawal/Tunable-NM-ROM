"""Q1: make the top of the Burgers correction ladder converge.

Same frozen checkpoint, same nested direction rule and seed, same reachable set, same
weak objective, same initializer policy, same stopping tolerance and budgets, same mesh,
same time step and the same six opened development cases as the audited ladder
(job 3713867) and the cheap-corrections cell (job 3734098). Three fixes, each isolated:

  (a) cascade warm start from the converged q/2 rung, padded with zeros
  (b) column equilibration of the augmented Jacobian before the normal equations
  (c) a Levenberg schedule and a trust radius for the y block, decoupled from z

plus one empirical quadrature rule per rung fitted with a real walltime budget, and a
validity gate (`truncated == false`) that disqualifies a rule the fitter could not
finish.

Everything is written incrementally to `result.json` so a truncated job is collectable.
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
import varpro as VP
import directions as DIR
import topfix as TF

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump


def test_count(rule, K, q, fixed):
    if rule == 'm4':
        return 4 * (K + q)
    if rule == 'm2':
        return 2 * (K + q)
    if rule == 'm256':
        return int(fixed)
    raise ValueError(rule)


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
                                   'a cascade arm also receives its coarse trajectory on the GPU before '
                                   'the timer starts, exactly as the initial field is; same-invocation '
                                   'host transfers also measured; identical for every arm'),
                  reference=[], snapshots={}, directions={}, arm_setup=[], conditioning=[],
                  reconstruction=[], invocations=[], declared_subjects=[], cascade_sources={},
                  gates={}, verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    report['cohort_note'] = ('exactly the six opened development cases of the audited ladder, of the '
                             'cheap-corrections cell and of head-ablation arm (a); no new case is opened')
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
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
        want = cfg.get('expected_reference_sha256', {}).get(str(case))
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=L,
                                        field_sha256=got, cclad01_field_sha256=want,
                                        bitwise_matches_cclad01=(None if want is None else got == want),
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
    save()
    print('SNAPSHOTS', round(sinfo['seconds'], 1), flush=True)

    # ------------------------------------------------------------- directions --
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    dcfg = {**cfg, 'strict': strict}
    Cfull, Ct, Zstar, rho, dinfo = DIR.audited(params, Rb, coef_truth, Zsub, K, dcfg)
    want = cfg.get('expected_directions_sha256')
    dinfo['expected_sha256'] = want
    dinfo['bitwise_matches_cclad01'] = (None if want is None else dinfo['directions_sha256'] == want)
    report['directions'] = dinfo
    report['gates']['directions_hash'] = dict(expected=want, got=dinfo['directions_sha256'],
                                              passed=dinfo['bitwise_matches_cclad01'])
    save()
    print('DIRECTIONS', round(dinfo['seconds'], 1), 'rank', dinfo['available_rank'],
          'hash_match', dinfo['bitwise_matches_cclad01'], flush=True)

    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    report['trust_radius'] = trust

    # ---------------------------------------------------------------- subjects -
    specs, seen = [], set()

    def add(q, rule, quadrature, arm, fitter='bounded', timed=True, step_budget=None,
            trust_scale=None, tag=''):
        M = test_count(rule, K, q, cfg['fixed_test_count'])
        if M <= K + q:
            return None
        name = f'q{q}_{rule}_{quadrature}_{arm}' + tag
        if name in seen:
            return name
        seen.add(name)
        specs.append(dict(name=name, q=q, M=M, rule=rule, quadrature=quadrature, arm=arm,
                          fitter=fitter, timed=timed,
                          step_budget=int(step_budget or strict['step_budget']),
                          trust_scale=float(trust_scale or 1.)))
        return name

    for g in cfg['gate_arms']:
        add(g['q'], g['rule'], g['quadrature'], g['arm'], g.get('fitter', 'bounded'))
    for blk in cfg['sweep']:
        for arm in blk['arms']:
            add(blk['q'], blk['rule'], blk['quadrature'], arm, blk.get('fitter', 'bounded'),
                step_budget=blk.get('step_budget'), trust_scale=blk.get('trust_scale'),
                tag=blk.get('tag', ''))
    for src in cfg['cascade_setup']:
        add(src['q'], src['rule'], src['quadrature'], src['arm'], timed=False)

    by_name = {s['name']: s for s in specs}
    cascade_source = {}
    for s in specs:
        if s['arm'] == 'casc':
            src = f"q{s['q'] // 2}_{s['rule']}_{s['quadrature']}_{cfg['cascade_source_arm']}"
            assert src in by_name, (s['name'], src)
            cascade_source[s['name']] = src
    report['cascade_sources'] = cascade_source
    save()

    # ---------------------------------------------------------------- build ----
    built, datas, colds, recon_done = {}, {}, {}, {}
    order = sorted(specs, key=lambda s: (s['q'], s['arm'] == 'casc'))
    for s in order:
        q, M = s['q'], s['M']
        assert M > K + q, (s['name'], M, K + q)
        C = Cfull[:, :q]
        head = TF.corrected_head(params, C, K)
        if q not in colds:
            Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
            colds[q] = A.build_cold(bank, head, Zaug, cfg['cold_axis_points'])
        cold, cinfo = colds[q]
        key = (q, M, s['quadrature'], s['fitter'])
        t0 = time.perf_counter()
        if key not in datas:
            if s['quadrature'] == 'eq':
                m = int(min(cfg['quadrature_multiplier'] * M, cfg['quadrature_cap']))
                if q == 0:
                    W = np.concatenate((Zold, np.zeros((len(Zold), 0))), axis=1)
                else:
                    W = TF.enriched_codes(Zstar, rho, Rb, Ct, q)
            else:
                m, W = None, None
            datas[key] = TF.build_operators(
                bank, L, M, s['quadrature'], head=head, Wcodes=W, m=m, fitter=s['fitter'],
                eq_seed=cfg['eq_seed'], candidate_cap=cfg['candidate_cap'],
                fit_states=cfg['fit_states'], max_fit_rows=cfg['max_fit_rows'],
                eq_seconds=cfg['eq_seconds'], blocks_wanted=cfg['eq_blocks'])
            if s['quadrature'] == 'eq':
                print('EQFIT', s['name'], 'm', datas[key][1].get('m'), 'valid',
                      datas[key][1].get('eq_rule_valid'), round(
                          datas[key][1]['eq_fit']['seconds'], 1), flush=True)
        data, info = datas[key]
        info = dict(info)
        dim = K + q
        linear = 'gj' if dim <= cfg['gauss_jordan_max'] else 'lu'
        cascade = s['arm'] == 'casc'
        arm_strict = dict(strict, step_budget=s['step_budget'])
        arm_trust = trust * s['trust_scale']
        query = TF.make_query(params, C, K, q, L, dt, arm_trust, s['quadrature'], s['arm'],
                              Rb=Rb, linear=linear, inner_damping=cfg['inner_damping'],
                              tau_y=cfg['tau_y'], cascade=cascade, ic_gtol=cfg['ic_gtol'],
                              **arm_strict)
        info.update(arm=s['name'], q=q, rule=s['rule'], fix=s['arm'], fitter=s['fitter'],
                    step_budget=s['step_budget'], trust_scale=s['trust_scale'],
                    solved_dimension=dim, dt=dt, trust_radius=arm_trust, linear_solve=linear,
                    cold=cinfo, timed=s['timed'], cascade_source=cascade_source.get(s['name']),
                    fixes=TF.ARMS[s['arm']], tau_y=(cfg['tau_y'] if TF.ARMS[s['arm']]['adaptive_y'] else None),
                    total_setup_seconds=time.perf_counter() - t0, correction_directions_used=q)
        report['arm_setup'].append(info)
        if s['timed']:
            report['declared_subjects'].append(dict(
                name=s['name'], method='rom', q=q, M=M, m=info.get('m'), rule=s['rule'],
                quadrature=s['quadrature'], fix=s['arm'], fitter=s['fitter'],
                solved_dimension=dim, linear_solve=linear, dt=dt))
        built[s['name']] = dict(**s, data=data, cold=cold, query=query, head=head, C=C,
                                linear_solve=linear, m=info.get('m'), cascade=cascade)
        print('ARM', s['name'], 'M', M, 'm', info.get('m'),
              round(info['total_setup_seconds'], 1), flush=True)
        save()

    foms = {'fft': ip.make_fom(L, dt, 'fft')}
    for fs in cfg['fom_settings']:
        report['declared_subjects'].append(dict(method='fom', dt=dt, **fs))

    inputs = [e.initial(L, phys) for phys in physical]

    # ------------------------------------------- cascade warm trajectories -----
    warms = {}
    for name, src in cascade_source.items():
        b = built[src]
        q = built[name]['q']
        rows = []
        t0 = time.perf_counter()
        for case in range(len(physical)):
            v = b['query'](jnp.asarray(inputs[case]), float(physical[case, 4]), b['data'], b['cold'])
            w = np.asarray(v[7])
            rows.append(np.concatenate((w, np.zeros((len(w), K + q - w.shape[1]))), axis=1))
        warms[name] = [jnp.asarray(r) for r in rows]
        print('CASCADE', name, 'from', src, round(time.perf_counter() - t0, 1), flush=True)
    save()

    # ------------------------------------------------ conditioning probe -------
    for q in cfg['conditioning_q']:
        M = test_count(cfg['conditioning_rule'], K, q, cfg['fixed_test_count'])
        key = (q, M, 'dense', 'bounded')
        if key not in datas:
            C = Cfull[:, :q]
            datas[key] = TF.build_operators(bank, L, M, 'dense')
            if q not in colds:
                Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
                colds[q] = A.build_cold(bank, TF.corrected_head(params, Cfull[:, :q], K),
                                        Zaug, cfg['cold_axis_points'])
        data, _ = datas[key]
        cond = TF.conditioning(params, Cfull[:, :q], K, q, L, dt, trust, 'dense', data,
                               jnp.asarray(inputs[0]), float(physical[0, 4]), colds[q][0],
                               step_index=cfg['conditioning_step'])
        cond['rule'] = cfg['conditioning_rule']
        report['conditioning'].append(cond)
        print('COND q', q, 'kappa', cond['lambdas']['1e-06']['unscaled_condition'],
              '->', cond['lambdas']['1e-06']['equilibrated_condition'], flush=True)
        save()
    jax.clear_caches()

    # ------------------------------------------------- untimed diagnostics -----
    for s in specs:
        q = s['q']
        if q in recon_done or not s['timed']:
            continue
        head = built[s['name']]['head']
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
        recon_done[q] = entry
        report['reconstruction'].append(entry)
        print('RECON q', q, round(entry['worst_best_found'] * 100, 5), flush=True)
        save()

    # ---------------------------------------------------------- timed queries --
    subjects = [dict(kind='rom', name=s['name']) for s in specs if s['timed']]
    subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, u, case, warm=None):
        nu = float(physical[case, 4])
        if sub['kind'] == 'rom':
            b = built[sub['name']]
            if b['cascade']:
                return b['query'](u, nu, b['data'], b['cold'], warm)
            return b['query'](u, nu, b['data'], b['cold'])
        fs = sub['setting']
        fn, pre = foms[fs['preconditioner']]
        return fn(u, nu, fs['ntol'], fs['ltol'], *pre)

    t = time.perf_counter()
    for sub in subjects:
        w0 = warms[sub['name']][0] if (sub['kind'] == 'rom' and built[sub['name']]['cascade']) else None
        jax.block_until_ready(invoke(sub, jnp.asarray(inputs[0]), 0, w0))
        print('WARM', sub['name'], round(time.perf_counter() - t, 1), flush=True)
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
    save()

    artifacts = {}
    for rep in range(cfg['reps']):
        for case in range(len(physical)):
            for i in order_rng.permutation(len(subjects)):
                sub = subjects[int(i)]
                cascade = sub['kind'] == 'rom' and built[sub['name']]['cascade']
                e.burn(cfg['burn_seconds'])
                ht = time.perf_counter()
                u = jax.device_put(np.array(inputs[case], copy=True))
                warm = jax.device_put(np.asarray(warms[sub['name']][case])) if cascade else None
                jax.block_until_ready(u if warm is None else (u, warm))
                gt = time.perf_counter()
                value = invoke(sub, u, case, warm)
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
                    b = built[sub['name']]
                    reasons = v[3].tolist()
                    gj = np.asarray(v[12], dtype=float)
                    worst = max(float(np.max(gj)), float(v[13]))
                    scale = float(v[11])
                    row.update(q=b['q'], solved_dimension=K + b['q'], M=b['M'], m=b['m'],
                               rule=b['rule'], fix=b['arm'], fitter=b['fitter'],
                               step_budget=b['step_budget'], trust_scale=b['trust_scale'],
                               quadrature=b['quadrature'], dt=dt, linear_solve=b['linear_solve'],
                               cascade=bool(b['cascade']),
                               cascade_source=cascade_source.get(sub['name']),
                               stop_reasons=reasons, ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=v[8].tolist(), ic_stationarity=float(v[9]),
                               step_joint_stationarity=gj.tolist(), ic_joint_stationarity=float(v[13]),
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
                    row.update(nonlinear_converged=bool(np.max(v[2]) <= sub['setting']['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # ------------------------------------------------------------- in-job gates
    def fields_by(name):
        return {x['case']: x['field_sha256'] for x in report['invocations'] if x['name'] == name}

    base0 = fields_by('q0_m4_dense_base')
    same = {}
    for nm in sorted({s['name'] for s in specs
                      if s['q'] == 0 and s['timed'] and s['quadrature'] == 'dense'}):
        if nm == 'q0_m4_dense_base':
            continue
        got = fields_by(nm)
        same[nm] = dict(compared=len(set(base0) & set(got)),
                        identical=sum(1 for c in set(base0) & set(got) if base0[c] == got[c]))
    report['gates']['q0_arms_bitwise'] = dict(
        reference='q0_m4_dense_base', arms=same,
        passed=(None if not same else
                all(v['compared'] > 0 and v['identical'] == v['compared'] for v in same.values())))
    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('Q1 TOP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
