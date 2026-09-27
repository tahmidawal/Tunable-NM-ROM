"""b-eqtop driver: certify the top rungs of the EQ ladder and rebuild it (DESIGN.md).

Phases 1 and 2 are `q-ridge/q_eqcert.py` verbatim (references, snapshots, bank,
directions, the reachable-state collection on disjoint fit / certification trajectories,
the same seeds), so the held-out certification states are the ones qrg304 used. Then:

  3 the archived qrg304 rules are re-certified on this job's held-out states (a gate);
  4 new rules at the declared rungs are fitted in ascending-m chains by CPU worker
    processes running `varpro.bounded_nnls` unchanged, one chain per (rung, arm), each rule
    certified on the GPU as it lands, chains stopping by the declared rule;
  5 the cheapest rule per rung and per bar is chosen, and the ladder is rebuilt with dense
    twins, the two qrg304 reproduction arms and the same-job full-order controls, three
    timed repetitions, under the budget-600 block-damped variable-projection contract.

Everything is written incrementally to `result.json`; every rule is saved as it is fitted.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
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
import ridge as RG
import eqcert as EC
import eqtop as ET

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rules', required=True, help='directory of archived qrg304 rules')
    p.add_argument('--tmp', required=True, help='scratch directory for designs and fits')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
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
    bars = {k: float(v) for k, v in cfg['bars'].items()}          # primary, tight
    rho_bar = bars['primary']

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  jax_version=jax.__version__, checkpoint_sha256=sha_file(a.checkpoint),
                  K=K, R=R, intervals=L, dt=dt, rho_bar=rho_bar, bars=bars,
                  spatial_bank_frozen=True, network_weights_frozen=True,
                  final_cohort_unopened=True, output_times=[0, .05, .1, .15, .2, .25],
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output '
                                   'fields; identical for every arm. Collection, design '
                                   'construction, fitting and certification are untimed and '
                                   'never enter a timed query; no worker process runs during '
                                   'the timed phase.'),
                  rho_definition=('|| sum_j w_j Phi(x_j) a(u)(x_j) - Phi^T a(u) || / '
                                  '|| Phi^T a(u) ||, q-diag 2026-09-16, on HELD-OUT reachable '
                                  'states'),
                  reference=[], snapshots={}, directions={}, collection=[], archived_rules=[],
                  rules=[], chains={}, rule_choice=[], arm_setup=[], invocations=[],
                  declared_subjects=[], gates={}, verification=ip.verify(), complete=False,
                  fit_phase_complete=False)
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
    report['trajectory_split'] = dict(fit=fit_idx.tolist(), certification=cert_idx.tolist())
    report['gates']['fit_and_certification_trajectories_disjoint'] = dict(
        passed=bool(not set(fit_idx.tolist()) & set(cert_idx.tolist())))
    save()

    # ------------------------------------------------------------ references --
    rf, rt = cfg['reference_mesh'], cfg['reference_dt']
    refs = {}
    if cfg['timed_phase']:
        q_fom, _ = e.make_fom(rf, rt)
        for case, phys in enumerate(physical):
            t = time.perf_counter()
            f, it, rn = host(q_fom(jnp.asarray(e.initial(rf, phys)), float(phys[4]), 1e-11, 1e-9))
            assert np.isfinite(f).all() and np.max(rn) < 2e-11
            f = np.array(f[:, ::rf // L, ::rf // L], copy=True)
            refs[case] = f
            name = f'ref_L{rf}_dt{rt}_case{case}.npz'
            np.savez_compressed(out / name, fields=f, iterations=it, residuals=rn)
            report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                            max_relative_residual=float(np.max(rn)),
                                            downsampled_to=L, field_sha256=sha_array(f),
                                            seconds=time.perf_counter() - t))
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
    report['gates']['directions_hash_matches_qrg304'] = dict(
        expected=want, got=dinfo['directions_sha256'],
        passed=(None if want is None else dinfo['directions_sha256'] == want),
        note='informational: GPU-model dependent; sets the fidelity tolerance tier')
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
    # Verbatim qrg304: same seeds, same rung order, so the pools and the held-out draw are
    # the ones that job certified on.
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
        exp_pool = (cfg.get('expected_pool_sha256') or {}).get(str(q)) or {}
        entry = dict(q=q, M=M, iterates_per_step=cfg['collect_iters'] + 1,
                     fit_states_available=int(len(got['fit'])),
                     certification_states_available=int(len(got['cert'])),
                     fit_pool_sha256=sha_array(got['fit']),
                     certification_pool_sha256=sha_array(got['cert']),
                     matches_qrg304=dict(
                         fit=(None if not exp_pool else exp_pool.get('fit') == sha_array(got['fit'])),
                         cert=(None if not exp_pool else exp_pool.get('cert') == sha_array(got['cert']))),
                     seconds=time.perf_counter() - t0)
        report['collection'].append(entry)
        print('COLLECT q', q, entry['fit_states_available'], entry['certification_states_available'],
              entry['matches_qrg304'], round(entry['seconds'], 1), flush=True)
        save()
        del collect
        jax.clear_caches()

    cert_sel = {q: pools[q]['cert'][rng.choice(len(pools[q]['cert']),
                                               min(cfg['cert_states'], len(pools[q]['cert'])),
                                               replace=False)] for q in QS}
    report['gates']['collection_pools_bitwise_qrg304'] = dict(
        passed=all(x['matches_qrg304']['cert'] for x in report['collection'])
        if cfg.get('expected_pool_sha256') else None,
        note='informational: GPU-model dependent')

    # ----------------------------------------- phase 3: archived rules ----------
    manifest = json.loads((Path(a.rules) / 'MANIFEST.json').read_text())
    archived = {}
    for x in manifest['rules']:
        q = x['q']
        if q not in cert_sel:
            continue
        nodes, weights = ET.load_archived_rule(Path(a.rules) / x['file'])
        assert sha_array(nodes) == x['nodes_sha256'] and sha_array(weights) == x['weights_sha256']
        ph, _ = modes(4 * (K + q))
        rule, cert = ET.certify_rule(bank, G, ph, L, cert_sel[q], nodes, weights,
                                     chunk=cfg['certify_chunk'])
        key = (q, x['population'], x['m_target'])
        archived[key] = (rule, dict(x, certification=cert, m=int(len(nodes)),
                                    rho_max_qrg304=x['rho_max'],
                                    rho_max_rel_diff=abs(cert['rho_max'] - x['rho_max'])
                                    / max(x['rho_max'], 1e-300),
                                    certified_primary=bool(cert['rho_max'] <= bars['primary']),
                                    certified_tight=bool(cert['rho_max'] <= bars['tight']),
                                    certified_secondary=bool(cert['rho_p95'] <= bars['primary'])))
        report['archived_rules'].append({k: v for k, v in archived[key][1].items()})
        print('ARCHIVED q', q, x['population'], 'm', x['m'], 'rho_max', f"{cert['rho_max']:.4f}",
              'qrg304', f"{x['rho_max']:.4f}", flush=True)
    worst = max(x['rho_max_rel_diff'] for x in report['archived_rules'])
    report['gates']['archived_rules_recertify'] = dict(
        passed=bool(worst <= cfg['recertify_tolerance']), worst_relative_difference=worst,
        tolerance=cfg['recertify_tolerance'])
    save()

    # ----------------------------------------- phase 4: the new rules ----------
    def incumbent_states(M):
        return int(np.clip(cfg['incumbent_max_fit_rows'] // int(M), 8, cfg['incumbent_fit_states']))

    chains, designs = [], []
    PY = sys.executable
    for q in cfg['new_rungs']:
        M = 4 * (K + q)
        ph, _ = modes(M)
        cand = ET.candidate_pool(L, cfg['candidate_pool'], cfg['pool_seed'] + q)
        for arm in cfg['arms']:
            n = incumbent_states(M) if arm['fit_states'] == 'incumbent' else int(arm['fit_states'])
            pool = pools[q]['fit']
            sel = np.sort(rng.choice(len(pool), min(n, len(pool)), replace=False))
            t0 = time.perf_counter()
            D, b, dinfo_ = ET.build_design(G, ph, L, pool[sel], cand, scaling=arm['scaling'])
            if arm['compress']:
                Dw, bw, cinfo = ET.compress(D, b, where='gpu')
            else:
                Dw, bw, cinfo = D, b, dict(compressed=False, rows_in=int(D.shape[0]),
                                           rows_out=int(D.shape[0]), unreachable_residual=0.,
                                           b_norm=float(np.linalg.norm(b)))
            key = f"q{q}_{arm['name']}"
            dpath, bpath, spath = tmp / f'{key}_D.npy', tmp / f'{key}_b.npy', tmp / f'{key}_side.json'
            np.save(dpath, np.ascontiguousarray(Dw))
            np.save(bpath, np.ascontiguousarray(bw))
            spath.write_text(json.dumps(dict(unreachable_residual=cinfo['unreachable_residual'],
                                             b_norm=cinfo['b_norm'])))
            del D, b, Dw, bw
            rec = dict(key=key, q=q, M=M, arm=arm['name'], fit_states=int(len(sel)),
                       fit_state_indices_sha256=sha_array(sel), scaling=arm['scaling'],
                       candidates=int(len(cand)), candidates_sha256=sha_array(cand),
                       design=dinfo_ | dict(target_norms=None), compression=cinfo,
                       build_seconds=time.perf_counter() - t0)
            designs.append(rec)
            chains.append(dict(key=key, q=q, M=M, arm=arm['name'], grid=list(arm['m_grid']),
                               adaptive=bool(arm['adaptive']), confirm=int(arm.get('confirm', 1)),
                               design=dpath, b=bpath, sidecar=spath, tmpdir=tmp, cand=cand,
                               Phi=ph))
            print('DESIGN', key, dinfo_['rows'], '->', cinfo['rows_out'], 'rows',
                  round(rec['build_seconds'], 1), flush=True)
    report['designs'] = designs
    save()
    jax.clear_caches()

    def certify_fn(chain, m, support, weights, info):
        q = chain['q']
        pos = chain['cand'][support]
        rule, cert = ET.certify_rule(bank, G, chain['Phi'], L, cert_sel[q], pos, weights,
                                     chunk=cfg['certify_chunk'])
        valid = bool(not info['truncated'])
        rec = dict(key=chain['key'], q=q, M=chain['M'], arm=chain['arm'], population='reachable',
                   m=int(len(pos)), m_target=int(m), certification=cert,
                   certified_primary=bool(cert['rho_max'] <= bars['primary'] and valid),
                   certified_tight=bool(cert['rho_max'] <= bars['tight'] and valid),
                   certified_secondary=bool(cert['rho_p95'] <= bars['primary'] and valid),
                   eq_rule_valid=valid, bars=bars, fit=info,
                   nodes_sha256=sha_array(np.asarray(pos)), weights_sha256=sha_array(np.asarray(weights)))
        np.savez_compressed(out / f"rule_q{q}_{chain['arm']}_m{m}.npz",
                            nodes=np.asarray(pos), weights=np.asarray(weights))
        new_rules[(q, chain['arm'], int(m))] = (rule, rec)
        print('RULE', chain['key'], 'm', rec['m'], 'fit', f"{info['relative_fit']:.3e}",
              'rho_max', f"{cert['rho_max']:.4f}", 'p95', f"{cert['rho_p95']:.4f}",
              'primary', rec['certified_primary'], 'tight', rec['certified_tight'],
              round(info['seconds'], 1), 's', flush=True)
        return rec

    def on_rule(rec):
        report['rules'].append({k: v for k, v in rec.items()})
        save()

    new_rules = {}
    t_fit = time.perf_counter()
    deadline = begin + float(cfg['fit_submission_deadline_seconds'])
    report['chains'] = ET.run_chains(chains, certify_fn, PY, cfg['fit_workers'],
                                     cfg['fit_threads'], cfg['eq_seconds'], cfg['eq_blocks'],
                                     on_rule, submit_deadline=deadline)
    report['fit_phase_seconds'] = time.perf_counter() - t_fit
    report['fit_phase_complete'] = True
    save()
    print('FITS DONE', round(report['fit_phase_seconds'], 1), report['chains'], flush=True)
    for pth in tmp.glob('*_D.npy'):
        pth.unlink()

    # ------------------------------------------------- the per-rung choice ----
    # Cheapest (smallest m) rule meeting the bar; ties broken by the declared arm order.
    order = {arm['name']: i for i, arm in enumerate(cfg['arms'])}
    chosen = {}
    for bar in ('primary', 'tight'):
        for q in QS:
            cands = []
            for (qq, pop, mt), (rule, info) in archived.items():
                if qq == q and pop == 'reachable':
                    cands.append((info['m'], -1, ('archived', qq, pop, mt), info))
            for (qq, arm, mt), (rule, info) in new_rules.items():
                if qq == q:
                    cands.append((info['m'], order[arm], ('new', qq, arm, mt), info))
            cands.sort(key=lambda t: (t[0], t[1]))
            flag = 'certified_primary' if bar == 'primary' else 'certified_tight'
            pick = next((c for c in cands if c[3][flag]), None)
            basis = bar
            if pick is None and bar == 'primary':
                pick = next((c for c in cands if c[3]['certified_secondary']), None)
                basis = 'secondary'
            if pick is None:
                basis = 'none'
            chosen[(bar, q)] = (pick, basis)
            report['rule_choice'].append(dict(
                bar=bar, q=q, M=4 * (K + q), basis=basis,
                chosen=(None if pick is None else dict(source=pick[2][0], m=pick[0],
                                                       arm=pick[2][2], m_target=pick[2][3],
                                                       rho_max=pick[3]['certification']['rho_max'],
                                                       rho_p95=pick[3]['certification']['rho_p95'])),
                considered=[dict(source=c[2][0], arm=c[2][2], m=c[0], m_target=c[2][3],
                                 rho_max=c[3]['certification']['rho_max'],
                                 rho_p95=c[3]['certification']['rho_p95'],
                                 certified_primary=c[3]['certified_primary'],
                                 certified_tight=c[3]['certified_tight'],
                                 certified_secondary=c[3]['certified_secondary']) for c in cands]))
            print('CHOICE', bar, 'q', q, basis, None if pick is None else (pick[2], pick[0]), flush=True)
    save()
    if not cfg['timed_phase']:
        report['elapsed_seconds'] = time.perf_counter() - begin
        report['complete'] = True
        save()
        (out / 'COMPLETE').write_text('complete\n')
        print('Q-EQTOP COMPLETE (no timed phase)', flush=True)
        return

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

    def rule_of(pick):
        src, q, arm, mt = pick[2]
        return (archived if src == 'archived' else new_rules)[(q, arm, mt)]

    def extra_of(info, src):
        c = info['certification']
        return dict(rho_max=c['rho_max'], rho_p95=c['rho_p95'], rho_median=c['rho_median'],
                    certified_primary=info['certified_primary'],
                    certified_tight=info['certified_tight'],
                    certified_secondary=info['certified_secondary'],
                    relative_fit=(info.get('relative_fit') if src == 'archived'
                                  else info['fit']['relative_fit']),
                    rule_source=src, rule_arm=info.get('arm', info.get('population')))

    # Arms in declared priority order; the list is cut at `max_rom_arms` (never asserted
    # after ten hours of fitting) and every dropped or aliased arm is recorded.
    wanted = []                       # (name, tag, builder)
    for q in QS:
        M = 4 * (K + q)
        pick, basis = chosen[('primary', q)]
        if pick is not None:
            wanted.append((f'q{q}_eq_primary', pick[2], ('rule', q, M, pick, basis)))
    for q in cfg['dense_twins']:
        wanted.append((f'q{q}_dense', ('dense', q), ('dense', q)))
    for g in cfg['reproduction_arms']:
        tag = ('archived', g['q'], 'reachable', g['m'])
        if (g['q'], 'reachable', g['m']) in archived:
            wanted.append((g['name'], tag, ('rule', g['q'], 4 * (K + g['q']),
                                            (archived[tag[1:]][1]['m'], -1, tag, archived[tag[1:]][1]),
                                            'reproduction')))
    for q in QS:
        M = 4 * (K + q)
        pick, basis = chosen[('tight', q)]
        if pick is not None:
            wanted.append((f'q{q}_eq_tight', pick[2], ('rule', q, M, pick, basis)))
    report['arm_aliases'], report['arms_dropped'] = {}, []
    used = {}
    for name, tag, spec in wanted:
        if tag in used:
            report['arm_aliases'][name] = used[tag]      # the same rule already runs under that name
            continue
        if len(subjects) >= cfg['max_rom_arms']:
            report['arms_dropped'].append(name)
            continue
        if spec[0] == 'dense':
            add(name, spec[1], 4 * (K + spec[1]), 'dense', dense_op[spec[1]], rule_kind='dense')
        else:
            _, q, M, pick, basis = spec
            rule, info = rule_of(pick)
            add(name, q, M, 'eq', eq_data(M, rule), m=info['m'],
                rule_kind=f'{pick[2][0]}:{pick[2][2]}',
                extra=extra_of(info, pick[2][0]) | dict(basis=basis))
        used[tag] = name
    print('ARMS', [s['name'] for s in subjects], 'aliases', report['arm_aliases'],
          'dropped', report['arms_dropped'], flush=True)

    foms = {}
    for fs in cfg['fom_settings']:
        key = (fs['preconditioner'], fs['dt'])
        if key not in foms:
            foms[key] = ip.make_fom(L, fs['dt'], fs['preconditioner'])
        report['declared_subjects'].append(dict(method='fom', **fs))
        subjects.append(dict(kind='fom', name=fs['name'], setting=fs))
    inputs = [e.initial(L, phys) for phys in physical]
    save()

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
    print('Q-EQTOP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
