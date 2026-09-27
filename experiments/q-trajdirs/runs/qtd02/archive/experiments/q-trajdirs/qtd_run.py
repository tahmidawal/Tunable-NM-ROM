"""Trajectory-fitted correction directions against the incumbent static rule.

One job. The frozen Burgers checkpoint, the frozen bank and head, the weak objective,
the initializer policy, the block-damped variable-projection solver at per-step budget
600, the output contract, the mesh, the time step and the six opened development cases
are the `b-ladder-top` Q1-B contract verbatim. The ONLY thing that changes between
direction sets is the matrix C_q:

  old   field-metric POD of the head's STATIC reconstruction residual eta - h(z*)
  traj  field-metric POD of the q = 0 ROM's TRAJECTORY error c*_t - h(z_t) over 32
        seeded training trajectories, every internal step
  prac  the same quantity over 6 training-family trajectories (the practitioner's rule)

Everything is written incrementally to `result.json` so a truncated job is collectable.
See `DESIGN.md` for the equations, the cohorts, the gates and the pre-registered pass.
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
import trajdirs as TD

sha_array, sha_file, host, dump = LD.sha_array, LD.sha_file, LD.host, LD.dump
TIMES = [0., .05, .1, .15, .2, .25]


def test_count(rule, K, q, fixed):
    if rule == 'm4':
        return 4 * (K + q)
    if rule == 'mfix':
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
    qlad = list(cfg['q_ladder'])

    report = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
                  job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
                  gpu=jax.devices()[0].device_kind, x64=True,
                  matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                  jax_version=jax.__version__, checkpoint_sha256=sha_file(a.checkpoint),
                  K=K, R=R, intervals=L, output_times=TIMES,
                  spatial_bank_frozen=True, network_weights_frozen=True,
                  final_cohort_unopened=True,
                  timing_contract=('supplied dense initial field on GPU to six dense GPU output '
                                   'fields; same-invocation host transfers also measured; '
                                   'identical for every subject'),
                  reference=[], snapshots={}, direction_sets={}, direction_comparison={},
                  arm_setup=[], reconstruction=[], invocations=[], declared_subjects=[],
                  gates={}, verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    # ------------------------------------------------------------- the cohort --
    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    report['cohort_note'] = ('exactly the six opened development cases of abl01, qlad01, '
                             'cclad01 and the b-ladder-top jobs; no new case is opened')
    got = sha_array(physical)
    want = cfg.get('expected_cohort_sha256')
    report['gates']['evaluation_cohort_bitwise_abl01'] = dict(
        expected=want, got=got, passed=(None if want is None else got == want))

    # ONE draw of the incumbent training family; every direction cohort is a set of
    # INDICES into it. `params_draw` is column-wise, so `params_draw(s, 32)` is NOT a
    # prefix of `params_draw(s, 128)` (b-speed deviation D3).
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    assert not any(np.allclose(t, s) for t in train_physical for s in physical), 'training/eval overlap'
    report['train_physical_sha256'] = sha_array(train_physical)

    rngA = np.random.default_rng(cfg['direction_traj_seed'])
    idxA = np.sort(rngA.choice(len(train_physical), cfg['direction_traj_count'], replace=False))
    rest = np.setdiff1d(np.arange(len(train_physical)), idxA)
    rngB = np.random.default_rng(cfg['direction_prac_seed'])
    idxB = np.sort(rngB.choice(rest, cfg['direction_prac_count'], replace=False))
    cohorts = {'traj': idxA, 'prac': idxB}
    cohorts = {k: v for k, v in cohorts.items()
               if k in cfg.get('direction_sets_to_build', ['traj', 'prac'])}
    dj = {}
    for name, idx in cohorts.items():
        sub = train_physical[idx]
        d = float(np.min(np.linalg.norm(sub[:, None] - physical[None], axis=2)))
        dj[name] = dict(count=int(len(idx)), indices=idx.tolist(),
                        indices_sha256=sha_array(idx), parameters_sha256=sha_array(sub),
                        min_distance_to_evaluation_cases=d, disjoint=bool(d > 1e-12))
    dj['traj_prac_disjoint'] = bool(len(np.intersect1d(idxA, idxB)) == 0)
    dj['built'] = sorted(cohorts)
    report['gates']['direction_cohorts_disjoint_from_evaluation'] = dict(
        passed=bool(all(v['disjoint'] for k, v in dj.items() if isinstance(v, dict))
                    and dj['traj_prac_disjoint']), detail=dj,
        note=('cohorts are built only for the direction sets this job constructs; a job '
              'that builds none still records the seeded index sets it would have used'))
    report['direction_cohorts'] = dj
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
        h = sha_array(f)
        w = cfg.get('expected_reference_sha256', {}).get(str(case))
        report['reference'].append(dict(intervals=rf, dt=rt, case=case, artifact=name,
                                        max_relative_residual=float(np.max(rn)), downsampled_to=L,
                                        field_sha256=h, comparator_field_sha256=w,
                                        bitwise_matches_comparator=(None if w is None else h == w),
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

    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    report['trust_radius'] = trust

    # --------------------------------------------------- the incumbent rule ----
    dcfg = {**cfg, 'strict': strict}
    Cold, Ct_old, Zstar_old, rho_old, dinfo_old = DIR.audited(params, Rb, coef_truth, Zsub, K, dcfg)
    tilde_old = jnp.asarray(rho_old) @ Rb.T
    want = cfg.get('expected_directions_sha256')
    dinfo_old['expected_sha256'] = want
    dinfo_old['bitwise_matches_comparator'] = (None if want is None
                                               else dinfo_old['directions_sha256'] == want)
    dinfo_old['residual_rows'] = int(np.asarray(rho_old).shape[0])
    print('DIRECTIONS old', round(dinfo_old['seconds'], 1), 'rank',
          dinfo_old['available_rank'], 'hash_match', dinfo_old['bitwise_matches_comparator'],
          flush=True)

    # ------------------------- the q = 0 ROM that generates the trajectories ----
    M0 = test_count('m4', K, 0, cfg['fixed_test_count'])
    data0, _ = TF.build_operators(bank, L, M0, 'dense')
    head0 = VP.head_only(params)
    C0 = Cold[:, :0]
    cold0, _ = A.build_cold(bank, TF.corrected_head(params, C0, K), Zsub, cfg['cold_axis_points'])
    qfn0 = TF.make_query(params, C0, K, 0, L, dt, trust, 'dense', 'base', Rb=Rb, linear='gj',
                         inner_damping=cfg['inner_damping'], tau_y=cfg['tau_y'],
                         ic_gtol=cfg['ic_gtol'], **strict)
    query0 = lambda u0, nu: jax.device_get(qfn0(u0, nu, data0, cold0))
    report['direction_rollout'] = dict(
        quadrature='dense', M=M0, step_budget=strict['step_budget'], gtol=strict['gtol'],
        fom_ntol=cfg['direction_fom_ntol'], fom_ltol=cfg['direction_fom_ltol'],
        note=('the q = 0 ROM used to generate the trajectory latents is the dense-quadrature '
              'M = 4K arm under the retained contract, so the directions never depend on an '
              'NNLS rule that itself depends on the directions'))
    save()

    # --------------------------------------------------- the two new rules -----
    sets = {}
    sets['old'] = dict(C=Cold, Ct=Ct_old, Zcodes=np.asarray(Zstar_old), rho=np.asarray(rho_old),
                       tilde=tilde_old, info=dinfo_old)
    for name in cfg.get('direction_sets_to_build', ['traj', 'prac']):
        t0 = time.perf_counter()
        C, Ct, Z, P, tilde, info = TD.build(
            query0, head0, Qb, Rb, train_physical[cohorts[name]], L, dt,
            cfg['direction_fom_ntol'], cfg['direction_fom_ltol'], R, qlad,
            rule=('field-metric POD of the q = 0 ROM trajectory error c*_t - h_theta(z_t) '
                  'at every internal step, nested in q'))
        info['cohort'] = dj[name]
        sets[name] = dict(C=C, Ct=Ct, Zcodes=Z, rho=P, tilde=tilde, info=info)
        print('DIRECTIONS', name, round(time.perf_counter() - t0, 1), 'rank',
              info['available_rank'], 'rows', info['residual_rows'], flush=True)
    del data0, cold0, qfn0
    jax.clear_caches()

    # ------------------------------------------------- directions: artifacts ---
    for name, s in sets.items():
        Cn = np.asarray(s['C'])
        Ctn = np.asarray(s['Ct'])
        fn = f'directions_{name}.npz'
        np.savez_compressed(out / fn, C=Cn, Ct=Ctn,
                            singular_values=np.asarray(s['info']['singular_values']))
        s['info'].update(artifact=fn,
                         prefix_sha256=TD.prefix_hashes(Cn, qlad),
                         field_orthonormal_prefix_sha256=TD.prefix_hashes(Ctn, qlad),
                         columns=int(Cn.shape[1]),
                         orthonormality_deviation=float(
                             np.max(np.abs(np.asarray(jnp.asarray(Ctn).T @ jnp.asarray(Ctn))
                                           - np.eye(Ctn.shape[1])))))
        report['direction_sets'][name] = s['info']
    report['gates']['directions_hashed_and_saved'] = dict(
        passed=all('artifact' in report['direction_sets'][n] and
                   (out / report['direction_sets'][n]['artifact']).exists() for n in sets),
        detail={n: report['direction_sets'][n]['artifact'] for n in sets})
    report['gates']['directions_rank_covers_ladder'] = dict(
        passed=all(report['direction_sets'][n]['available_rank'] >= max(qlad) for n in sets),
        detail={n: report['direction_sets'][n]['available_rank'] for n in sets})
    save()

    # --------------------------------------------- directions: comparisons ----
    cmp_q = [q for q in cfg['comparison_q'] if q > 0]
    comparison = dict(cross_capture={}, principal_angles={}, note=(
        'cross_capture[P][C][q] = || P_whitened C_{:q} ||_F^2 / || P_whitened ||_F^2 : the '
        'fraction of residual matrix P the direction set C can reach at rung q. '
        'principal_angles are field-metric, from the singular values of Ct_a[:, :q]^T Ct_b[:, :q].'))
    for pn, ps in sets.items():
        comparison['cross_capture'][pn] = {
            cn: TD.cross_capture(ps['tilde'], cs['Ct'], qlad) for cn, cs in sets.items()}
    for other in ('traj', 'prac'):
        if other in sets:
            comparison['principal_angles'][f'{other}_vs_old'] = TD.principal_angles(
                sets[other]['Ct'], sets['old']['Ct'], cmp_q)
    if 'traj' in sets and 'prac' in sets:
        comparison['principal_angles']['traj_vs_prac'] = TD.principal_angles(
            sets['traj']['Ct'], sets['prac']['Ct'], cmp_q)
    report['direction_comparison'] = comparison
    save()
    print('COMPARISON done', round(time.perf_counter() - begin, 1), flush=True)

    # ---------------------------------------------------------------- subjects -
    specs, seen, ladders = [], {}, {}

    def add(block, q):
        M = test_count(block['rule'], K, q, cfg['fixed_test_count'])
        if M <= K + q:
            return None
        ds = 'shared' if q == 0 else block['dirset']
        name = (f"q{q}_M{M}_{block['quadrature']}" if q == 0
                else f"{ds}_q{q}_M{M}_{block['quadrature']}") + block.get('tag', '')
        if name not in seen:
            seen[name] = dict(name=name, dirset=ds, q=q, M=M, rule=block['rule'],
                              quadrature=block['quadrature'],
                              fitter=block.get('fitter', 'bounded'))
            specs.append(seen[name])
        return name

    # The ladders are declared in the config and recorded here as ordered rung lists, so
    # the report never has to reconstruct which arm belongs to which ladder from a name.
    # A q = 0 rung is direction-independent and is therefore ONE arm shared by every
    # ladder that contains it.
    for lb in cfg['ladders']:
        rungs = []
        for q in lb['q']:
            nm = add(lb, q)
            if nm is not None:
                rungs.append(dict(q=q, arm=nm))
        ladders[lb['id']] = dict(id=lb['id'], dirset=lb['dirset'], rule=lb['rule'],
                                 quadrature=lb['quadrature'], label=lb['label'], rungs=rungs)
    for xb in cfg['extra_arms']:
        for q in xb['q']:
            add(xb, q)
    report['ladders'] = ladders
    save()

    built, datas, colds, recon_done = {}, {}, {}, {}
    for s in sorted(specs, key=lambda x: (x['q'], x['M'], x['name'])):
        q, M, ds = s['q'], s['M'], s['dirset']
        assert M > K + q, (s['name'], M, K + q)
        src = 'old' if ds == 'shared' else ds
        Cfull = sets[src]['C']
        C = Cfull[:, :q]
        head = TF.corrected_head(params, C, K)
        ckey = (ds, q)
        if ckey not in colds:
            Zaug = np.concatenate((Zsub, np.zeros((len(Zsub), q))), axis=1)
            colds[ckey] = A.build_cold(bank, head, Zaug, cfg['cold_axis_points'])
        cold, cinfo = colds[ckey]
        key = (('dense', M) if s['quadrature'] == 'dense'
               else ('eq', ds, q, M, s['fitter']))
        t0 = time.perf_counter()
        if key not in datas:
            if s['quadrature'] == 'eq':
                m = int(min(cfg['quadrature_multiplier'] * M, cfg['quadrature_cap']))
                if q == 0:
                    W = np.concatenate((Zold, np.zeros((len(Zold), 0))), axis=1)
                else:
                    W = TF.enriched_codes(sets[src]['Zcodes'], sets[src]['rho'], Rb,
                                          sets[src]['Ct'], q)
            else:
                m, W = None, None
            datas[key] = TF.build_operators(
                bank, L, M, s['quadrature'], head=head, Wcodes=W, m=m, fitter=s['fitter'],
                eq_seed=cfg['eq_seed'], candidate_cap=cfg['candidate_cap'],
                fit_states=cfg['fit_states'], max_fit_rows=cfg['max_fit_rows'],
                eq_seconds=cfg['eq_seconds'], blocks_wanted=cfg['eq_blocks'])
            if s['quadrature'] == 'eq':
                print('EQFIT', s['name'], 'm', datas[key][1].get('m'), 'valid',
                      datas[key][1].get('eq_rule_valid'),
                      round(datas[key][1]['eq_fit']['seconds'], 1), flush=True)
        data, info = datas[key]
        info = dict(info)
        dim = K + q
        linear = 'gj' if dim <= cfg['gauss_jordan_max'] else 'lu'
        query = TF.make_query(params, C, K, q, L, dt, trust, s['quadrature'], 'base', Rb=Rb,
                              linear=linear, inner_damping=cfg['inner_damping'],
                              tau_y=cfg['tau_y'], ic_gtol=cfg['ic_gtol'], **strict)
        info.update(arm=s['name'], family='rom', dirset=ds, q=q,
                    rule=s['rule'], fix='base', fitter=s['fitter'], solved_dimension=dim,
                    dt=dt, gtol=strict['gtol'], ic_gtol=cfg['ic_gtol'],
                    step_budget=strict['step_budget'], trust_radius=trust, linear_solve=linear,
                    cold=cinfo, correction_directions_used=q,
                    directions_prefix_sha256=sha_array(np.ascontiguousarray(np.asarray(C))),
                    total_setup_seconds=time.perf_counter() - t0)
        report['arm_setup'].append(info)
        report['declared_subjects'].append(dict(
            name=s['name'], method='rom', family='rom', dirset=ds, q=q,
            M=M, m=info.get('m'), rule=s['rule'], quadrature=s['quadrature'],
            fitter=s['fitter'], solved_dimension=dim, linear_solve=linear, dt=dt))
        built[s['name']] = dict(**s, data=data, cold=cold, query=query, head=head, C=C,
                                linear_solve=linear, m=info.get('m'))
        print('ARM', s['name'], 'M', M, 'm', info.get('m'),
              round(info['total_setup_seconds'], 1), flush=True)
        save()

    foms = {}
    for fs in cfg['fom_settings']:
        k = (fs['preconditioner'], fs['dt'])
        if k not in foms:
            foms[k] = ip.make_fom(L, fs['dt'], fs['preconditioner'])
        report['declared_subjects'].append(dict(method='fom', family='fom', **fs))
    inputs = [e.initial(L, phys) for phys in physical]
    save()

    # ------------------------------------------------- untimed diagnostics -----
    for s in specs:
        q, ds = s['q'], s['dirset']
        if (ds, q) in recon_done:
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
        entry = dict(family='rom', dirset=ds, q=q, solved_dimension=K + q, cases=rows,
                     worst_bank_projection=float(max(r['bank_projection_max'] for r in rows)),
                     worst_best_found=float(max(r['best_found_max'] for r in rows)))
        recon_done[(ds, q)] = entry
        report['reconstruction'].append(entry)
        print('RECON', ds, 'q', q, round(entry['worst_best_found'] * 100, 5), flush=True)
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
                    row.update(family='rom', dirset=b['dirset'], q=b['q'],
                               solved_dimension=K + b['q'], M=b['M'], m=b['m'], rule=b['rule'],
                               fix='base', fitter=b['fitter'], quadrature=b['quadrature'],
                               gtol=strict['gtol'], step_budget=strict['step_budget'], dt=dt,
                               linear_solve=b['linear_solve'], stop_reasons=reasons,
                               ic_iterations=int(v[5]), ic_reason=int(v[6]),
                               step_stationarity=np.asarray(v[8], dtype=float).tolist(),
                               ic_stationarity=float(v[9]),
                               step_joint_stationarity=gj.tolist(),
                               ic_joint_stationarity=float(v[13]),
                               step_inner_stationarity=np.asarray(v[14], dtype=float).tolist(),
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
                    row.update(family='fom', dt=sub['setting']['dt'],
                               nonlinear_converged=bool(
                                   np.max(v[2]) <= sub['setting']['ntol'] * (1 + 1e-9)))
                report['invocations'].append(row)
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # ------------------------------------------------------------- in-job gates
    report['gates']['nested_prefix_consistent'] = dict(
        passed=all(x['directions_prefix_sha256']
                   == report['direction_sets']['old' if x['dirset'] == 'shared' else x['dirset']
                                               ]['prefix_sha256'][str(x['q'])]
                   for x in report['arm_setup']),
        note='the C slice each arm was handed equals the first q columns of its saved matrix')
    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    assert report['checkpoint_sha256'] == report['checkpoint_sha256_after']
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('Q-TRAJDIRS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
