"""EQ rule certification and the rebuilt ladder (DESIGN.md §A3).

Same frozen checkpoint, same OLD nested direction rule and seed, same reachable set, same
weak objective, same initializer policy, same stopping rule, same per-step budget 600, same
mesh, same time step and the same six opened development cases as the audited ladder, the
cheap-corrections cell and b-ladder-top. Only the empirical-quadrature RULE changes, and it
changes in exactly three declared ways: the population it is fitted on (static decoder
outputs vs states the ROM actually reaches), its size m, and the fitter.

Phases, each written incrementally to `result.json` so a truncated job is collectable:

  1 references, snapshots, the frozen bank, the audited directions
  2 reachable-state collection: dense-quadrature rollouts on DISJOINT training-family
    trajectories, every Levenberg-Marquardt iterate retained (untimed)
  3 rule fitting: static and reachable populations, m in {1024, 2048, 4096, 8192}
  4 certification: rho on the held-out reachable states, against a bar declared in DESIGN.md
  5 the rebuilt ladder: the cheapest certified rule per rung, dense twins, old-rule arms,
    same-job full-order controls, three timed repetitions
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
import ridge as RG
import eqcert as EC

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
    rho_bar = float(cfg['rho_bar'])

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  jax_version=jax.__version__, checkpoint_sha256=sha_file(a.checkpoint),
                  K=K, R=R, intervals=L, dt=dt, rho_bar=rho_bar,
                  spatial_bank_frozen=True, network_weights_frozen=True,
                  final_cohort_unopened=True,
                  output_times=[0, .05, .1, .15, .2, .25],
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output '
                                   'fields; identical for every arm. Reachable-state collection, '
                                   'rule fitting and certification are untimed and never enter a '
                                   'timed query.'),
                  rho_definition=('|| sum_j w_j Phi(x_j) a(u)(x_j) - Phi^T a(u) || / '
                                  '|| Phi^T a(u) ||, q-diag 2026-09-16, evaluated on HELD-OUT '
                                  'reachable states'),
                  reference=[], snapshots={}, directions={}, collection=[], rules=[],
                  rule_choice=[], arm_setup=[], reconstruction=[], invocations=[],
                  declared_subjects=[], gates={}, verification=ip.verify(), complete=False)
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
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'train/eval overlap'
    report['train_physical_sha256'] = sha_array(train_physical)

    nfit, ncert = cfg['fit_trajectories'], cfg['cert_trajectories']
    fit_idx = np.arange(nfit)
    cert_idx = np.arange(nfit, nfit + ncert)
    assert not set(fit_idx) & set(cert_idx)
    report['trajectory_split'] = dict(
        fit=fit_idx.tolist(), certification=cert_idx.tolist(),
        note=('disjoint training-family trajectories; neither overlaps the six evaluation '
              'cases, which are asserted above to be disjoint from the whole training draw'))
    report['gates']['fit_and_certification_trajectories_disjoint'] = dict(
        passed=bool(not set(fit_idx.tolist()) & set(cert_idx.tolist())))
    save()

    # ------------------------------------------------------------ references --
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
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=L,
                                        field_sha256=got, seconds=time.perf_counter() - t))
        print('REFERENCE', case, round(time.perf_counter() - begin, 1), flush=True)
        save()
    del q_fom
    jax.clear_caches()

    # ------------------------------------------------------------ snapshots ---
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
    Gh = np.asarray(G)
    np.savez(out / 'bank_G.npz', G=Gh)
    report['bank_G'] = dict(artifact='bank_G.npz', sha256=sha_array(Gh), shape=list(Gh.shape),
                            dtype=str(Gh.dtype))
    del Gh
    save()
    print('SNAPSHOTS', round(sinfo['seconds'], 1), flush=True)

    # ------------------------------------------------------------ directions --
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    Cfull, Ct, Zstar, rho_head, dinfo = DIR.audited(params, Rb, coef_truth, Zsub, K,
                                                    {**cfg, 'strict': strict})
    want = cfg.get('expected_directions_sha256')
    dinfo['expected_sha256'] = want
    report['directions'] = dinfo
    report['gates']['directions_hash'] = dict(expected=want, got=dinfo['directions_sha256'],
                                              passed=(None if want is None
                                                      else dinfo['directions_sha256'] == want))
    save()
    print('DIRECTIONS', round(dinfo['seconds'], 1), 'rank', dinfo['available_rank'], flush=True)

    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    report['trust_radius'] = trust
    Phi, lam_modes = {}, {}
    QS = list(cfg['q_ladder'])

    def modes(M):
        if M not in Phi:
            ph, lm, _ = e.modes(L, M)
            Phi[M], lam_modes[M] = ph, lm
        return Phi[M], lam_modes[M]

    def dense_data(M):
        ph, lm = modes(M)
        P = jnp.asarray(ph)
        return dict(A=P.T @ G, lam=jnp.asarray(lm), G=G, Phi=P)

    def eq_data(M, rule):
        ph, lm = modes(M)
        P = jnp.asarray(ph)
        return dict(A=P.T @ G, lam=jnp.asarray(lm), G=G, G5=rule['G5'], Pq=rule['Pq'])

    # ------------------------------------------- phase 2: reachable states ----
    colds, dense_op, pools = {}, {}, {}
    rng = np.random.default_rng(cfg['collect_seed'])
    for q in QS:
        M = 4 * (K + q)
        C = Cfull[:, :q]
        head = RG.corrected_head(params, C, K)
        Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        colds[q] = A.build_cold(bank, head, Zaug, cfg['cold_axis_points'])
        dense_op[q] = dense_data(M)
        t0 = time.perf_counter()
        collect = EC.make_collect_query(
            params, C, K, q, L, dt, trust, iters=cfg['collect_iters'],
            ic_budget=strict['ic_budget'], gtol=strict['gtol'],
            linear=('gj' if K + q <= cfg['gauss_jordan_max'] else 'lu'),
            inner_damping=cfg['inner_damping'])
        cf = jax.jit(jax.vmap(head))
        got = {}
        for tag, idx in (('fit', fit_idx), ('cert', cert_idx)):
            rows = []
            for i in idx:
                ph = train_physical[i]
                seen, _ = collect(jnp.asarray(e.initial(L, ph)), float(ph[4]), dense_op[q],
                                  colds[q][0])
                w = np.asarray(seen).reshape(-1, K + q)
                rows.append(np.asarray(cf(jnp.asarray(w))))
            got[tag] = np.concatenate(rows)
        pools[q] = got
        entry = dict(q=q, M=M, iterates_per_step=cfg['collect_iters'] + 1,
                     fit_states_available=int(len(got['fit'])),
                     certification_states_available=int(len(got['cert'])),
                     fit_trajectories=fit_idx.tolist(),
                     certification_trajectories=cert_idx.tolist(),
                     fit_pool_sha256=sha_array(got['fit']),
                     certification_pool_sha256=sha_array(got['cert']),
                     seconds=time.perf_counter() - t0)
        report['collection'].append(entry)
        print('COLLECT q', q, entry['fit_states_available'], entry['certification_states_available'],
              round(entry['seconds'], 1), flush=True)
        save()
        del collect
        jax.clear_caches()

    cert_sel = {q: pools[q]['cert'][rng.choice(len(pools[q]['cert']),
                                               min(cfg['cert_states'], len(pools[q]['cert'])),
                                               replace=False)] for q in QS}

    # ------------------------------------------------- phase 3+4: the rules ---
    def pick_fit(q, m, M):
        n = int(np.clip(int(np.ceil(4 * m / M)), cfg['fit_states_min'], cfg['fit_states_max']))
        pool = pools[q]['fit']
        sel = rng.choice(len(pool), min(n, len(pool)), replace=False)
        return pool[np.sort(sel)]

    rules = {}
    for q in QS:
        M = 4 * (K + q)
        ph, _ = modes(M)
        cand = np.sort(rng.choice((L - 1) ** 2, cfg['candidate_cap'], replace=False))
        cand_small = np.sort(np.random.default_rng(cfg['eq_seed']).choice(
            (L - 1) ** 2, cfg['incumbent_candidate_cap'], replace=False))
        specs = [dict(population='static', m=int(min(4 * M, cfg['incumbent_cap'])),
                      fitter='gpu', candidates=cand_small)]
        specs += [dict(population='reachable', m=int(mm), fitter='gpu', candidates=cand)
                  for mm in cfg['m_grid']]
        for s in specs:
            key = (q, s['population'], s['m'], s['fitter'])
            if s['population'] == 'static':
                codes = EC.static_codes(Zstar, rho_head, Rb, Ct, q, Zold)
                n = int(np.clip(int(np.ceil(4 * s['m'] / M)), cfg['fit_states_min'],
                                cfg['fit_states_max']))
                sel = np.sort(rng.choice(len(codes), min(n, len(codes)), replace=False))
                coefs = np.asarray(jax.jit(jax.vmap(RG.corrected_head(params, Cfull[:, :q], K)))(
                    jnp.asarray(codes[sel])))
            else:
                coefs = pick_fit(q, s['m'], M)
            rule, info = EC.fit_rule(bank, G, ph, L, M, s['m'], coefs, s['candidates'],
                                     fitter=s['fitter'], blocks=cfg['fit_blocks'],
                                     inner_iters=cfg['active_set_iterations'],
                                     seconds=cfg['eq_seconds'])
            cert = EC.certify(G, ph, L, rule, cert_sel[q], chunk=cfg['certify_chunk'])
            info.update(q=q, population=s['population'], certification=cert,
                        certified_primary=bool(cert['rho_max'] <= rho_bar and info['eq_rule_valid']),
                        certified_secondary=bool(cert['rho_p95'] <= rho_bar and info['eq_rule_valid']),
                        rho_bar=rho_bar)
            rules[key] = (rule, info)
            report['rules'].append({k: v for k, v in info.items()
                                    if k not in ('nodes', 'weights')} |
                                   dict(nodes_sha256=sha_array(np.asarray(info['nodes'])),
                                        weights_sha256=sha_array(np.asarray(info['weights']))))
            np.savez_compressed(out / f"rule_q{q}_{s['population']}_m{s['m']}.npz",
                                nodes=np.asarray(info['nodes']),
                                weights=np.asarray(info['weights']))
            print('RULE q', q, s['population'], 'm', info['m'], 'fit', f"{info['relative_fit']:.3e}",
                  'rho_max', f"{cert['rho_max']:.4f}", 'p95', f"{cert['rho_p95']:.4f}",
                  'certified', info['certified_primary'], round(info['total_seconds'], 1), flush=True)
            save()

    # ------------------------------------------------- the per-rung choice ----
    chosen = {}
    for q in QS:
        M = 4 * (K + q)
        cands = [(k, v[1]) for k, v in rules.items()
                 if k[0] == q and k[1] == 'reachable']
        cands.sort(key=lambda kv: kv[0][2])
        pick = next((k for k, i in cands if i['certified_primary']), None)
        basis = 'primary'
        if pick is None:
            pick = next((k for k, i in cands if i['certified_secondary']), None)
            basis = 'secondary'
        if pick is None:
            pick = cands[-1][0]
            basis = 'none (largest m, uncertified)'
        chosen[q] = pick
        info = rules[pick][1]
        report['rule_choice'].append(dict(
            q=q, M=M, chosen_m=info['m'], basis=basis, rho_max=info['certification']['rho_max'],
            rho_p95=info['certification']['rho_p95'], relative_fit=info['relative_fit'],
            rho_bar=rho_bar,
            considered=[dict(m=i['m'], rho_max=i['certification']['rho_max'],
                             rho_p95=i['certification']['rho_p95'],
                             certified_primary=i['certified_primary'],
                             certified_secondary=i['certified_secondary']) for _, i in cands]))
        print('CHOICE q', q, 'm', info['m'], basis, flush=True)
    save()

    # ------------------------------------------------------- phase 5: arms ----
    built, subjects = {}, []

    def add(name, q, M, quadrature, data, m=None, rule_kind=None, extra=None):
        dim = K + q
        linear = 'gj' if dim <= cfg['gauss_jordan_max'] else 'lu'
        query = RG.make_query(params, Cfull[:, :q], K, q, L, dt, trust, quadrature, 0.,
                              ic_budget=strict['ic_budget'], step_budget=strict['step_budget'],
                              gtol=strict['gtol'], ic_gtol=cfg['ic_gtol'], linear=linear,
                              inner_damping=cfg['inner_damping'])
        built[name] = dict(name=name, q=q, M=M, m=m, quadrature=quadrature, data=data,
                           cold=colds[q][0], query=query, linear_solve=linear,
                           rule_kind=rule_kind, coef_fn=RG.coefficients(params, Cfull[:, :q], K))
        info = dict(arm=name, q=q, M=M, m=m, quadrature=quadrature, rule_kind=rule_kind,
                    solved_dimension=dim, linear_solve=linear, dt=dt, trust_radius=trust,
                    step_budget=strict['step_budget'], tests_per_unknown=M / dim)
        info.update(extra or {})
        report['arm_setup'].append(info)
        report['declared_subjects'].append(dict(name=name, method='rom', q=q, M=M, m=m,
                                                quadrature=quadrature, rule_kind=rule_kind))
        subjects.append(dict(kind='rom', name=name))

    for q in QS:
        M = 4 * (K + q)
        rk, ri = rules[chosen[q]]
        add(f'q{q}_m4_eqcert', q, M, 'eq', eq_data(M, rk), m=ri['m'], rule_kind='reachable',
            extra=dict(rho_max=ri['certification']['rho_max'],
                       rho_p95=ri['certification']['rho_p95'],
                       certified_primary=ri['certified_primary'],
                       certified_secondary=ri['certified_secondary'],
                       relative_fit=ri['relative_fit']))
        sk = (q, 'static', int(min(4 * M, cfg['incumbent_cap'])), 'gpu')
        rk2, ri2 = rules[sk]
        add(f'q{q}_m4_eqstatic', q, M, 'eq', eq_data(M, rk2), m=ri2['m'], rule_kind='static',
            extra=dict(rho_max=ri2['certification']['rho_max'],
                       rho_p95=ri2['certification']['rho_p95'],
                       certified_primary=ri2['certified_primary'],
                       certified_secondary=ri2['certified_secondary'],
                       relative_fit=ri2['relative_fit']))
    for q in cfg['dense_twins']:
        add(f'q{q}_m4_dense', q, 4 * (K + q), 'dense', dense_op[q], rule_kind='dense')

    # ---- the two b-ladder-top reproduction arms, with the RETAINED fitters ----
    for g in cfg['reproduction_arms']:
        q, mult = g['q'], g['test_multiplier']
        M = mult * (K + q)
        C = Cfull[:, :q]
        head = RG.corrected_head(params, C, K)
        W = (np.concatenate((Zold, np.zeros((len(Zold), 0))), axis=1) if q == 0
             else RG.enriched_codes(Zstar, rho_head, Rb, Ct, q))
        t0 = time.perf_counter()
        data, info = TF.build_operators(
            bank, L, M, 'eq', head=head, Wcodes=W, m=int(min(4 * M, cfg['incumbent_cap'])),
            fitter=g['fitter'], eq_seed=cfg['eq_seed'], candidate_cap=cfg['incumbent_candidate_cap'],
            fit_states=cfg['incumbent_fit_states'], max_fit_rows=cfg['incumbent_max_fit_rows'],
            eq_seconds=cfg['eq_seconds'], blocks_wanted=cfg['eq_blocks'])
        if q not in colds:
            Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
            colds[q] = A.build_cold(bank, head, Zaug, cfg['cold_axis_points'])
        ph, _ = modes(M)
        cpool = cert_sel.get(q)
        cert = (EC.certify(G, ph, L, dict(G5=data['G5'], Pq=data['Pq']), cpool,
                           chunk=cfg['certify_chunk']) if cpool is not None else None)
        add(g['name'], q, M, 'eq', data, m=info.get('m'), rule_kind=f"incumbent_{g['fitter']}",
            extra=dict(relative_fit=info.get('eq_relative_fit'),
                       eq_rule_valid=info.get('eq_rule_valid'),
                       eq_fit=info.get('eq_fit'), certification=cert,
                       setup_seconds=time.perf_counter() - t0))
        print('REPRO', g['name'], 'M', M, 'm', info.get('m'),
              round(time.perf_counter() - t0, 1), flush=True)
        save()

    foms = {}
    for fs in cfg['fom_settings']:
        key = (fs['preconditioner'], fs['dt'])
        if key not in foms:
            foms[key] = ip.make_fom(L, fs['dt'], fs['preconditioner'])
        report['declared_subjects'].append(dict(method='fom', **fs))
        subjects.append(dict(kind='fom', name=fs['name'], setting=fs))
    inputs = [e.initial(L, phys) for phys in physical]
    save()

    # ------------------------------------------------ untimed diagnostics -----
    for q in QS:
        head = RG.corrected_head(params, Cfull[:, :q], K)
        Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
        Hc = jax.jit(jax.vmap(head))(jnp.asarray(Zaug))
        Hn = jnp.sum((Hc @ Rb.T) ** 2, 1)
        recon = A.make_reconstruction(head, K + q, L, cfg['recon_budget'],
                                      linear=('gj' if K + q <= cfg['gauss_jordan_max'] else 'lu'))
        rows = []
        for case in refs:
            ref = refs[case]
            n0 = float(np.linalg.norm(ref[0]))
            bank_err, man_err = [], []
            for ti in range(ref.shape[0]):
                target = jnp.asarray(ref[ti][1:-1, 1:-1].ravel())
                bank_err.append(float(jnp.linalg.norm(Qb @ (Qb.T @ target) - target)) / n0)
                score = Hn - 2 * (Hc @ (G.T @ target))
                starts = jnp.asarray(Zaug)[jnp.argsort(score)[:cfg['recon_starts']]]
                z, rn, it, reason = host(recon(starts, target, G))
                man_err.append(float(rn) / n0)
            rows.append(dict(case=case, bank_projection_max=float(np.max(bank_err)),
                             best_found_max=float(np.max(man_err)),
                             bank_projection_per_time=bank_err, best_found_per_time=man_err))
        report['reconstruction'].append(dict(
            q=q, solved_dimension=K + q, cases=rows,
            worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
            worst_best_found=float(max(r['best_found_max'] for r in rows))))
        print('RECON q', q, round(report['reconstruction'][-1]['worst_best_found'] * 100, 5),
              flush=True)
        save()
    jax.clear_caches()

    # ---------------------------------------------------------- timed queries -
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
    for sub in subjects:
        jax.block_until_ready(invoke(sub, jnp.asarray(inputs[0]), 0))
        print('WARM', sub['name'], round(time.perf_counter() - t, 1), flush=True)
    report['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
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
                        extra = dict(internal_latents=v[7],
                                     coefficients=np.asarray(
                                         built[sub['name']]['coef_fn'](value[7])))
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
                    row.update(q=b['q'], solved_dimension=K + b['q'], M=b['M'], m=b['m'],
                               rule=('m4' if b['M'] == 4 * (K + b['q']) else 'other'),
                               rule_kind=b['rule_kind'], quadrature=b['quadrature'], dt=dt,
                               lam_rel=0., step_budget=strict['step_budget'],
                               linear_solve=b['linear_solve'], stop_reasons=reasons,
                               ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=v[8].tolist(), ic_stationarity=float(v[9]),
                               step_joint_stationarity=gj.tolist(),
                               ic_joint_stationarity=float(v[13]),
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

    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('Q-EQCERT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
