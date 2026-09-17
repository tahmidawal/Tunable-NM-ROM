"""p-linear: the whole Poisson 2D correction ladder on the best checkpoint, with every
comparator timed in the same job on the same GPU.

Subjects at one mesh, all on the twelve opened development sources:

* the correction ladder q in {0, 32, 64, 128, 256, 512} on the R=512 / K=32 checkpoint
  through the unchanged `correction_core` path (exact analytic elimination; the nonlinear
  iteration stays K-dimensional at every q), under two test-count rules;
* the head-only arm through the unchanged `poisson_ablation` kernel (the parent's arm);
* the free bank (identity head) at the top rung's test count;
* POD-LSPG rebuilt from the checkpoint's own 3072 training snapshots at k' up to 512;
* the incumbent R=128 / K=16 checkpoint's rungs as cross-job fidelity gates;
* the direct DST solve and unpreconditioned CG at several tolerances.

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
import correction_core as CC
import pilot as P
import arms as A
import sep_common as sc
import poisson_ablation as PA
import pbh_core as K_
import iterative_core as IC
import directions as DIR
import plin_core as L_


def resolve(here, rel):
    """A config path relative to the config directory, or its flat basename when the
    attempt is staged flat on the cluster (every file sits beside the driver there)."""
    cand = here / rel
    return cand if cand.exists() else here / Path(rel).name


def training_cohort(spec):
    draws = C.source_params(spec['seed'], spec['draw_count'])
    if spec.get('prefix') is not None:
        draws = draws[:spec['prefix']]
    if spec.get('fit_split') is not None:
        fit, _ = K_.fit_validation_split(len(draws), spec['fit_split']['split_seed'],
                                         spec['fit_split']['validation_fraction'])
        draws = draws[fit]
    return draws


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        # mechanics only: the extension is built from a 640-source prefix of the fit split so
        # the smoke stays short on the shared box; the q <= 32 gates do not depend on it
        cfg.update(intervals=64, repetitions=1, burn_seconds=0.001, cg_maxiter=4000,
                   ccrule_arms=False, attempt='smoke', extension_sources=640)
        cfg['gates'] = [g for g in cfg['gates'] if 64 in g['intervals']]
    n = int(cfg['intervals'])
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    R_ = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke), intervals=n,
              timing_contract=('host source array in to dense nodal field out, including '
                               'projection, initialization, solve, exact correction recovery '
                               'where present, and decode; the ladder path also carries its '
                               'stationarity and rank diagnostics inside the timed region, '
                               'exactly as the retained method and ccpoi01 do'),
              checkpoints=[], references=[], directions=[], arm_setup=[], reconstruction=[],
              pod=[], invocations=[], declared_subjects=[], gates=[], consistency=[],
              complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)
    save()

    # ---------------------------------------------------------------- cohorts --
    dev = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                          C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    R_['cohort'] = dict(parameters=dev.tolist(), sha256=L_.sha_array(dev),
                        groups=['existing development'] * cfg['eval_count']
                               + ['fresh development'] * cfg['fresh_count'])
    pod_draws = C.source_params(cfg['pod_cohort']['seed'], cfg['pod_cohort']['count'])
    assert not any(np.allclose(t, s) for t in pod_draws for s in dev), 'POD cohort overlaps dev'
    R_['pod_cohort'] = dict(**cfg['pod_cohort'], parameters_sha256=L_.sha_array(pod_draws))
    save()

    # ------------------------------------------------------------- checkpoints --
    loaded = {}
    for m in cfg['models']:
        ck = resolve(here, m['checkpoint'])
        bs = resolve(here, m['basis'])
        params, codes, ckcfg = sc.load_pkl(ck)
        basis = dict(np.load(bs))
        np.testing.assert_array_equal(np.asarray(codes), basis['training_latents'])
        train = training_cohort(m['training'])
        assert len(train) == len(codes), (m['id'], len(train), len(codes))
        assert not any(np.allclose(t, s) for t in train for s in dev), f"{m['id']} overlaps dev"
        loaded[m['id']] = dict(params=params, codes=np.asarray(codes), basis=basis, train=train,
                               K=int(np.asarray(codes).shape[1]),
                               R=int(np.asarray(params['h_lin']).shape[1]))
        R_['checkpoints'].append(dict(
            id=m['id'], role=m['role'], K=loaded[m['id']]['K'], R=loaded[m['id']]['R'],
            checkpoint=m['checkpoint'], basis=m['basis'],
            checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
            basis_sha256=hashlib.sha256(bs.read_bytes()).hexdigest(),
            training_codes=int(len(codes)), training_parameters_sha256=L_.sha_array(train),
            checkpoint_config={k: v for k, v in ckcfg.items() if k != 'continuation'}))
    save()

    # -------------------------------------------------------------- references --
    nhi = cfg['reference_intervals'][-1]
    nlo = cfg['reference_intervals'][0]
    chain, same = {}, {}
    for case, param in enumerate(dev):
        t0 = time.perf_counter()
        fine = P.reference(param, nhi)
        coarse = P.reference(param, nlo)
        chain[case] = np.array(fine[::nhi // n, ::nhi // n], copy=True)
        same[case] = P.reference(param, n)
        R_['references'].append(dict(case=case, fine_intervals=nhi, coarse_intervals=nlo,
                                     reference_delta=C.relative(coarse, fine[::nhi // nlo, ::nhi // nlo]),
                                     fine_sha256=L_.sha_array(fine),
                                     same_grid_sha256=L_.sha_array(same[case]),
                                     seconds=time.perf_counter() - t0))
    sources = [C.full_source(n, q) for q in dev]
    lam = jnp.asarray(C.eigenvalues(n))
    save()
    print('REFERENCES done', round(time.perf_counter() - begin, 1), flush=True)

    # ----------------------------------------------- assembly and directions ---
    fixed = int(cfg['fixed_test_count'])
    built, keep = [], []
    opcache = {}
    for m in cfg['models']:
        mid = m['id']
        L = loaded[mid]
        K, R = L['K'], L['R']
        base = C.assemble(L['params'], L['codes'], n, fixed, cfg['lm_budget'])
        assert base['info']['retained_bank_rank'] == R, (mid, base['info']['retained_bank_rank'])
        keep.append(base)
        opcache[mid] = {fixed: base}
        R_['arm_setup'].append(dict(intervals=n, arm=f'assembly@{mid}', model=mid, K=K, R=R,
                                    **{k: base['info'][k] for k in
                                       ('retained_modes', 'stored_features', 'retained_bank_rank',
                                        'trust_delta', 'operator_sha256', 'bank_sha256',
                                        'bank_build_seconds', 'weak_assembly_seconds')}))
        Cfull, dinfo = L_.extend_basis(L['params'], L['codes'], L['basis'], L['train'],
                                       cfg['training_intervals'],
                                       subset=cfg.get('extension_sources'))
        assert dinfo['retained_prefix_exact'], dinfo
        assert dinfo['extension_orthonormality_error'] < 1e-7, dinfo
        L['Cfull'] = Cfull
        R_['directions'].append(dict(model=mid, **dinfo))
        jax.clear_caches()
        save()
        print('DIRECTIONS', mid, 'orth', dinfo['extension_orthonormality_error'],
              'prefix defect', dinfo['rebuilt_prefix_subspace_defect'],
              round(dinfo['seconds'], 1), flush=True)

        def ops_for(modes, mid=mid, base=base):
            if modes not in opcache[mid]:
                opcache[mid][modes] = L_.reassemble(base, modes)
            return opcache[mid][modes]

        # the correction ladder, both test-count rules
        for rule in cfg['ladder_rules']:
            for q in cfg['ladder_q']:
                if q > R:
                    continue
                if m['role'] == 'control' and q not in cfg['control_q']:
                    continue
                modes = L_.test_count(rule, K + q, fixed)
                ops = ops_for(modes)
                M = int(ops['B'].shape[0])
                # below the top rung the K head unknowns and the q corrections must all be
                # identifiable; at q = R the head is redundant and only the R-column linear
                # part must have full rank
                need = K + q if q < R else R
                if M <= need:
                    R_['declared_subjects'].append(dict(name=f'q{q}_{rule}@{mid}', skipped=True,
                                                        reason=f'M={M} <= unknowns={need}'))
                    print('SKIP', f'q{q}_{rule}@{mid}', M, need, flush=True)
                    continue
                t0 = time.perf_counter()
                engine = CC.prepare_correction(ops, L['codes'], Cfull, q, cfg['retained'])
                assert engine['info']['linear_rank_valid'], (mid, q, rule, engine['info'])
                name = f'q{q}_{rule}@{mid}'
                built.append(dict(name=name, kind='ladder', model=mid, q=q, rule=rule, M=M,
                                  ops=ops, engine=engine, family='neural+linear',
                                  k=K + q, nonlinear_dim=K))
                R_['arm_setup'].append(dict(
                    intervals=n, arm=name, model=mid, family='neural+linear', K=K, R=R, q=q,
                    rule=rule, M=M, requested_modes=modes, nonlinear_dimension=K,
                    nominal_dimension=K + q, degenerate_full_bank=bool(q >= R),
                    directions='retained-prefix extension',
                    setup_seconds=time.perf_counter() - t0,
                    **{k: engine['info'][k] for k in
                       ('correction_count', 'linear_rank', 'linear_rank_valid',
                        'linear_condition_number', 'projected_operator_sha256',
                        'correction_matrix_sha256')}))
                print('ARM', name, 'M', M, flush=True)
                save()

        # the parent's head-only arm (poisson_ablation kernel), fixed test count
        head = lambda z, p=L['params']: sc.head(p, z)
        trust = PA.radius(L['codes'])
        linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
        kern = PA.make_query(head, base['B'], base['bank'], n, trust, cfg['lm_budget'],
                             cfg['stationarity_tolerance'], linear)
        pred = jax.jit(jax.vmap(lambda z: base['B'] @ head(z)))(jnp.asarray(L['codes']))
        built.append(dict(name=f'a_neural@{mid}', kind='pa', model=mid, k=K, kernel=kern,
                          predictions=pred, codes=jnp.asarray(L['codes']), B=base['B'],
                          bank=base['bank'], ops=base, family='neural', M=int(base['B'].shape[0]),
                          rule='m256', q=0))
        R_['arm_setup'].append(dict(intervals=n, arm=f'a_neural@{mid}', model=mid, family='neural',
                                    K=K, R=R, M=int(base['B'].shape[0]), trust_radius=trust,
                                    linear_solve=linear, operator_sha256=base['info']['operator_sha256']))

        # the free bank: identity head, R unknowns, at the top rung's test count
        fb_rule = m['freebank_rule']
        fb_modes = L_.test_count(fb_rule, K + R, fixed)
        fops = ops_for(fb_modes)
        if int(fops['B'].shape[0]) > R:
            Zfree = np.asarray(jax.jit(jax.vmap(head))(jnp.asarray(L['codes'])))
            ftrust = PA.radius(Zfree)
            fk = PA.make_query(A.identity_head(), fops['B'], fops['bank'], n, ftrust,
                               cfg['lm_budget'], cfg['stationarity_tolerance'], 'lu')
            fpred = jax.jit(jax.vmap(lambda z: fops['B'] @ z))(jnp.asarray(Zfree))
            name = f'd_freebank_{fb_rule}@{mid}'
            built.append(dict(name=name, kind='pa', model=mid, k=R, kernel=fk, predictions=fpred,
                              codes=jnp.asarray(Zfree), B=fops['B'], bank=fops['bank'], ops=fops,
                              family='free', M=int(fops['B'].shape[0]), rule=fb_rule, q=R))
            R_['arm_setup'].append(dict(intervals=n, arm=name, model=mid, family='free', K=R, R=R,
                                        M=int(fops['B'].shape[0]), requested_modes=fb_modes,
                                        trust_radius=ftrust, linear_solve='lu',
                                        operator_sha256=fops['info']['operator_sha256']))
            # the rank-R linear reduced model, solved directly (DESIGN A4): the top rung
            lk, Qt, Rr, linfo = L_.make_linear_query(fops['B'], n)
            name = f'd_linear_qr_{fb_rule}@{mid}'
            built.append(dict(name=name, kind='linear', model=mid, k=R, kernel=lk, Qt=Qt, Rr=Rr,
                              B=fops['B'], bank=fops['bank'], ops=fops, family='linear',
                              M=int(fops['B'].shape[0]), rule=fb_rule, q=R))
            R_['arm_setup'].append(dict(intervals=n, arm=name, model=mid, family='linear', K=R, R=R,
                                        requested_modes=fb_modes, degenerate_full_bank=True,
                                        operator_sha256=fops['info']['operator_sha256'], **linfo))
        save()

    # ------------------------------ cheap-corrections rule, incumbent, q=64 gate --
    if cfg.get('ccrule_arms'):
        cc = cfg['ccrule_arms']
        mid = cc['model']
        L = loaded[mid]
        K, R = L['K'], L['R']
        base = opcache[mid][fixed]
        t0 = time.perf_counter()
        train_draws = C.source_params(cc['train_seed'], cc['train_count'])
        assert not any(np.allclose(t, s) for t in train_draws for s in dev)
        bank = base['bank']
        Qb, Rb = A.whiten(bank)
        jax.block_until_ready(Rb)
        U = np.stack([np.asarray(C.dst_solve(jnp.asarray(C.full_source(n, qv)), lam))[1:-1, 1:-1].ravel()
                      for qv in train_draws])
        Ut = jnp.asarray(U.T)
        coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
        del Ut, U
        dcfg = {**cc, 'strict': dict(gtol=cfg['stationarity_tolerance']), 'q_ladder': cfg['ladder_q']}
        _, _, _, rho, dinfo = DIR.audited(L['params'], Rb, coef_truth, L['codes'], K, dcfg)
        D32 = jnp.asarray(L['basis']['coefficient_directions'])
        kept = int(D32.shape[1])
        T32, _ = jnp.linalg.qr(Rb @ D32, mode='reduced')
        tilde = rho @ Rb.T
        perp = tilde - (tilde @ T32) @ T32.T
        _, sv, Vt = jnp.linalg.svd(perp, full_matrices=False)
        Text = Vt[:R - kept].T
        Ccc = np.asarray(jnp.concatenate((D32, jnp.linalg.solve(Rb, Text)), axis=1))
        dinfo.update(model=mid, rule_family='cheap-corrections (ccpoi01) extension',
                     retained_prefix_exact=bool(np.array_equal(Ccc[:, :kept], np.asarray(D32))),
                     full_directions_sha256=L_.sha_array(Ccc), seconds=time.perf_counter() - t0)
        R_['directions'].append(dinfo)
        del Qb, Rb, rho, tilde, perp
        jax.clear_caches()
        for rule in cc['rules']:
            for q in cc['q']:
                modes = L_.test_count(rule, K + q, fixed)
                ops = opcache[mid].get(modes) or L_.reassemble(base, modes)
                opcache[mid][modes] = ops
                M = int(ops['B'].shape[0])
                engine = CC.prepare_correction(ops, L['codes'], Ccc, q, cfg['retained'])
                assert engine['info']['linear_rank_valid']
                name = f'q{q}_{rule}_ccrule@{mid}'
                built.append(dict(name=name, kind='ladder', model=mid, q=q, rule=rule, M=M, ops=ops,
                                  engine=engine, family='neural+linear', k=K + q, nonlinear_dim=K))
                R_['arm_setup'].append(dict(
                    intervals=n, arm=name, model=mid, family='neural+linear', K=K, R=R, q=q, rule=rule,
                    M=M, requested_modes=modes, nonlinear_dimension=K, nominal_dimension=K + q,
                    directions='cheap-corrections rule', degenerate_full_bank=False,
                    **{k: engine['info'][k] for k in ('correction_count', 'linear_rank',
                                                      'linear_rank_valid', 'linear_condition_number',
                                                      'projected_operator_sha256',
                                                      'correction_matrix_sha256')}))
        save()
        print('CCRULE directions', round(dinfo['seconds'], 1), flush=True)

    # ------------------------------------------------------------- POD-LSPG ----
    modes, coords_full, pinfo = K_.streaming_pod(pod_draws, n, max(cfg['pod_ranks']), cfg['pod_block'])
    keep.append(modes)
    R_['pod'].append(pinfo)
    sine_cache = {}
    for rule in cfg['pod_rules']:
        for k in cfg['pod_ranks']:
            req = L_.test_count(rule, k, fixed)
            if req not in sine_cache:
                sine_cache[req] = K_.sine_ops(n, req)
            so = sine_cache[req]
            Vk = modes[:, :k]
            Bk = PA.reduce_bank(Vk, so['S'], so['I'], so['J'], n)
            M = int(Bk.shape[0])
            name = f'e_pod{k}_{rule}@trainset'
            if M <= k:
                R_['declared_subjects'].append(dict(name=name, skipped=True, reason=f'M={M} <= k={k}'))
                print('SKIP', name, M, k, flush=True)
                continue
            Z = coords_full[:, :k]
            trust = PA.radius(Z)
            linear = 'gj' if k <= cfg['gauss_jordan_max'] else 'lu'
            kern = PA.make_query(A.identity_head(), Bk, Vk, n, trust, cfg['lm_budget'],
                                 cfg['stationarity_tolerance'], linear)
            pred = jax.jit(jax.vmap(lambda z, Bk=Bk: Bk @ z))(jnp.asarray(Z))
            built.append(dict(name=name, kind='pa', model=None, k=k, kernel=kern, predictions=pred,
                              codes=jnp.asarray(Z), B=Bk, bank=Vk, ops=so, family='pod', M=M,
                              rule=rule, q=None))
            R_['arm_setup'].append(dict(intervals=n, arm=name, family='pod', k=k, rule=rule, M=M,
                                        requested_modes=req, trust_radius=trust, linear_solve=linear,
                                        candidates=int(len(Z)), operator_sha256=C.sha(np.asarray(Bk))))
    save()
    print('POD done', round(time.perf_counter() - begin, 1), flush=True)

    # ------------------------------------------------------------------ FOM ----
    cg_kernel = IC.make_cg(n, int(cfg['cg_maxiter']))

    # ------------------------------------ untimed three-layer reconstruction ---
    for m in cfg['models']:
        mid = m['id']
        L = loaded[mid]
        K, R = L['K'], L['R']
        base = opcache[mid][fixed]
        G = base['bank']
        Rg, rank = K_.bank_r(G)
        assert rank['rank_valid'], (mid, rank)
        U = jnp.stack([jnp.asarray(same[c][1:-1, 1:-1].ravel()) for c in range(len(dev))])
        T, perp2, nu2 = K_.project_targets(G, Rg, U)
        floor = np.asarray(jnp.sqrt(perp2 / nu2))
        head = lambda z, p=L['params']: sc.head(p, z)
        linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
        rec = dict(intervals=n, model=mid, K=K, R=R, bank_rank=rank,
                   bank_projection=K_.summarise(floor), against='same-grid FD-DST truth',
                   augmented=[])
        # The parent's dense-field best-found (arms.make_reconstruction) at q = 0. It is a
        # CROSS-CHECK of the projected oracle below, not a reported quantity: its jacfwd
        # Jacobian is (starts, (n-1)^2, K) in f64 -- 2.0 GiB per array at n = 1024, with
        # 32 GiB autotuner variants, which OOMed a 40 GB A100 (job 3780691, DESIGN A7).
        # Above `dense_oracle_max_intervals` the projected oracle carries the number; the
        # two agree to ~1e-13 wherever both run, and that agreement is recorded.
        if n <= cfg['dense_oracle_max_intervals'] and (m['role'] != 'control' or not a.smoke):
            recon = A.make_reconstruction(head, K, n, cfg['recon_budget'],
                                          cfg['stationarity_tolerance'], linear)
            H = jax.jit(jax.vmap(head))(jnp.asarray(L['codes']))
            Hn = jnp.sum((H @ Rg.T) ** 2, 1)
            best, iters = [], []
            for c in range(len(dev)):
                target = U[c]
                score = Hn - 2 * (H @ (G.T @ target))
                starts = jnp.asarray(L['codes'])[jnp.argsort(score)[:cfg['recon_starts']]]
                z, rn, it, reason = jax.device_get(recon(starts, target, G))
                best.append(float(rn) / float(np.sqrt(float(nu2[c]))))
                iters.append(int(it))
            rec['best_found_dense'] = K_.summarise(best)
            rec['best_found_dense_iterations'] = iters
        for q in [qq for qq in cfg['ladder_q'] if qq <= R]:
            if m['role'] == 'control' and q not in cfg['control_q']:
                continue
            V = np.asarray(Rg) @ L['Cfull'][:, :q]
            e, it, rs = L_.oracle_projected(head, Rg, L['codes'], T, perp2, nu2, V,
                                            cfg['recon_budget'], cfg['recon_starts'],
                                            cfg['stationarity_tolerance'], linear)
            rec['augmented'].append(dict(q=q, best_found=K_.summarise(e), iterations=it.tolist(),
                                         exit_reasons=rs.tolist()))
            if q == 0 and 'best_found_dense' in rec:
                d = np.asarray(rec['best_found_dense']['per_case'])
                rec['dense_vs_projected_oracle'] = dict(
                    worst_absolute_difference=float(np.max(np.abs(d - e))),
                    worst_relative_difference=float(np.max(np.abs(d - e) / np.maximum(d, 1e-300))),
                    note='same quantity, dense field metric vs the exact QR metric')
            print('RECON', mid, 'q', q, 'floor', float(floor.max()), 'best', float(e.max()), flush=True)
        R_['reconstruction'].append(rec)
        del U, T, Rg
        jax.clear_caches()
        save()

    # ---------------------------------------------------------- timed queries --
    subjects = [dict(kind=b['kind'], name=b['name'], index=i) for i, b in enumerate(built)]
    subjects.append(dict(kind='fom', name='dst_direct'))
    for tol in cfg['cg_tolerances']:
        subjects.append(dict(kind='cg', name=f'cg_{tol:g}', tolerance=float(tol)))
    R_['declared_subjects'] += [dict(intervals=n, **{k: v for k, v in s.items() if k != 'index'})
                                for s in subjects]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, source):
        if sub['kind'] == 'ladder':
            b = built[sub['index']]
            return CC.correction_query(source, b['ops'], b['engine'], cfg['retained'])
        if sub['kind'] == 'pa':
            b = built[sub['index']]
            return PA.query_once(b['kernel'], source, b['ops'], b['predictions'], b['codes'],
                                 b['B'], b['bank'])
        if sub['kind'] == 'linear':
            b = built[sub['index']]
            return L_.linear_query_once(b['kernel'], source, b['ops'], b['Qt'], b['Rr'], b['B'], b['bank'])
        if sub['kind'] == 'cg':
            return IC.cg_query(source, cg_kernel, sub['tolerance'])
        return C.fom_query(source, lam)

    t = time.perf_counter()
    for sub in subjects:
        invoke(sub, sources[0])
    R_['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
    save()
    print('WARMUP', len(subjects), 'subjects', round(time.perf_counter() - t, 1), flush=True)

    artifacts = {}
    for rep in range(cfg['repetitions']):
        for case in range(len(dev)):
            for i in order_rng.permutation(len(subjects)):
                sub = subjects[int(i)]
                C.burn(cfg['burn_seconds'])
                field, row = invoke(sub, sources[case])
                assert np.isfinite(field).all()
                h = L_.sha_array(field)
                key = (sub['name'], case, h)
                if key not in artifacts:
                    fn = f"n{n}_{sub['name'].replace('@', '-')}_case{case}_rep{rep}.npz"
                    np.savez_compressed(out / fn, field=field)
                    artifacts[key] = fn
                b = built[sub['index']] if 'index' in sub else None
                slim = {k: v for k, v in row.items()
                        if k not in ('starts', 'projected_jacobian_singular_values')}
                R_['invocations'].append(dict(
                    intervals=n, case=case, group=R_['cohort']['groups'][case], rep=rep,
                    kind=sub['kind'], name=sub['name'], field_sha256=h, artifact=artifacts[key],
                    physical_error=C.relative(field, chain[case]),
                    same_grid_error=C.relative(field, same[case]),
                    source_sha256=C.sha(sources[case]), finite=True,
                    model=(b['model'] if b else None), family=(b['family'] if b else 'fom'),
                    k=(b['k'] if b else None), q=(b.get('q') if b else None),
                    rule=(b.get('rule') if b else None), M=(b.get('M') if b else None),
                    tolerance=sub.get('tolerance'), **slim))
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # ------------------------------------------------------------ fidelity gates -
    def per_case(name, key):
        got = {}
        for x in R_['invocations']:
            if x['name'] == name:
                got.setdefault(x['case'], x[key])
        return got

    for g in cfg['gates']:
        if n not in g['intervals']:
            continue
        ref = json.loads(resolve(here, g['file']).read_text())
        for ours, theirs in g['pairs']:
            entry = next((e for e in ref['arms'] if e['intervals'] == n and e['name'] == theirs), None)
            if entry is None:
                R_['gates'].append(dict(intervals=n, ours=ours, theirs=theirs, source=g['file'],
                                        missing_reference=True, passed=False))
                continue
            for key in ('physical_error', 'same_grid_error'):
                if key not in entry:
                    continue
                got = per_case(ours, key)
                if not got:
                    R_['gates'].append(dict(intervals=n, ours=ours, theirs=theirs, source=g['file'],
                                            metric=key, missing_subject=True, passed=False))
                    continue
                worst = max(abs(got[c] - e) / max(abs(e), 1e-300)
                            for c, e in zip(entry['cases'], entry[key]) if c in got)
                worst_abs = max(abs(got[c] - e) for c, e in zip(entry['cases'], entry[key]) if c in got)
                # an exact solver's same-grid error is round-off on both sides (~1e-16), so
                # its relative difference is 0/0; the absolute floor covers that case only
                R_['gates'].append(dict(intervals=n, ours=ours, theirs=theirs, source=g['file'],
                                        reference_job=ref.get('job_id'), metric=key,
                                        worst_relative_difference=worst,
                                        worst_absolute_difference=worst_abs,
                                        tolerance=cfg['fidelity_tolerance'],
                                        absolute_floor=cfg['fidelity_absolute_floor'],
                                        passed=bool(worst <= cfg['fidelity_tolerance']
                                                    or worst_abs <= cfg['fidelity_absolute_floor'])))
    for g in R_['gates']:
        print('GATE', g, flush=True)

    # in-job consistency checks (same job, different code paths), reported not gated
    fields = {}
    for x in R_['invocations']:
        fields.setdefault((x['name'], x['case']), x['artifact'])
    for ours, theirs, tol in cfg['consistency']:
        worst = 0.0
        found = True
        for case in range(len(dev)):
            fa, fb = fields.get((ours, case)), fields.get((theirs, case))
            if fa is None or fb is None:
                found = False
                break
            A_ = np.load(out / fa)['field']
            B_ = np.load(out / fb)['field']
            worst = max(worst, C.relative(A_, B_))
        R_['consistency'].append(dict(intervals=n, a=ours, b=theirs, found=found,
                                      worst_field_relative_difference=worst if found else None,
                                      tolerance=tol, passed=bool(found and worst <= tol)))
        print('CONSISTENCY', R_['consistency'][-1], flush=True)
    save()

    for g in R_['gates']:
        assert g['passed'], g
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('P-LINEAR SOLVE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
