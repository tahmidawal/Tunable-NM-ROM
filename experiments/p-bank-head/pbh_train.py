"""p-bank-head job 1: diagnose the incumbent, sweep the bank, sweep the head.

Layers and criteria are pre-registered in DESIGN.md. Nothing here is timed as a
query; every number is an offline representation or training diagnostic. The
frozen checkpoints this job writes are the only thing the solve job consumes.

Staged flat: every module sits beside this file on the cluster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import sep_common as sc
import arms as A
import pbh_core as K_
import pbh_fit as F_


def head_of(params):
    return lambda z: sc.head(params, z)


def weak_solve_errors(params, codes, ops, B, Fm, G, Rg, T, perp2, nu2, budget, gtol, linear):
    """The ROM's own weak solve, untimed, from the nearest cached training code.

    Identical residual, initializer policy, budget and stopping rule to arm (a);
    it is a diagnostic here because no timing contract is being measured.
    """
    head = head_of(params)
    lm = A.make_stationary_lm(lambda z, fm: jnp.asarray(B) @ head(z) - fm, budget,
                              float(np.max(np.linalg.norm(np.asarray(codes)
                                                          - np.asarray(codes).mean(0), axis=1))),
                              gtol, linear)
    pred = jax.jit(jax.vmap(lambda z: jnp.asarray(B) @ head(z)))(jnp.asarray(codes))

    @jax.jit
    def one(fm):
        idx = jnp.argmin(jnp.sum((pred - fm[None, :]) ** 2, axis=1))
        return lm(jnp.asarray(codes)[idx], (fm,), 0.)
    errs, reasons, iters, gns = [], [], [], []
    for i in range(int(Fm.shape[0])):
        z, rn, it, reason, gn = jax.device_get(one(jnp.asarray(Fm)[i]))
        res = np.asarray(jnp.asarray(Rg) @ head(jnp.asarray(z))) - np.asarray(T[i])
        errs.append(float(np.sqrt(res @ res + float(perp2[i])) / np.sqrt(float(nu2[i]))))
        reasons.append(int(reason))
        iters.append(int(it))
        gns.append(float(gn))
    return np.asarray(errs), np.asarray(reasons), np.asarray(iters), np.asarray(gns)


def correction_basis(params, Z, G, Rg, T, perp2, nu2, count):
    """`diagnose_head_corrections`: right singular vectors of the normalised
    training residuals in the exact QR physical metric, nested prefixes."""
    norm = np.sqrt(np.asarray(nu2))
    coeff = np.asarray(jax.jit(jax.vmap(head_of(params)))(jnp.asarray(Z)))
    residual = (np.asarray(T) - coeff @ np.asarray(Rg).T) / norm[:, None]
    _, s, Vt = np.linalg.svd(residual, full_matrices=False)
    count = int(min(count, Vt.shape[0]))
    physical = Vt[:count].T
    directions = np.linalg.solve(np.asarray(Rg), physical)
    W = np.asarray(G) @ directions
    orth = float(np.linalg.norm(W.T @ W - np.eye(count)))
    return dict(coefficient_directions=directions, physical_metric_directions=physical,
                R=np.asarray(Rg), singular_values=s, training_latents=np.asarray(Z),
                training_norms=norm), dict(orthogonality_error=orth, count=int(count),
                                           singular_values=s[:count].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--incumbent', required=True)
    ap.add_argument('--incumbent-basis', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bank-checkpoint', default=None,
                    help='head-only mode: skip the incumbent diagnosis and the bank sweep and '
                         'run the head sweep on this already-trained, frozen bank')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    if a.smoke:
        cfg.update(source_counts=[24], ranks=[32], latents=[8], beta_weak=[0., 1.],
                   beta_smooth=[0., 1e-2], training_intervals=31, floor_intervals=[31, 63],
                   requested_modes=16, recon_budget=60, train_oracle_subsample=8,
                   head_steps=40, knn=3, lm_budget=60)
        cfg['training_common'].update(point_batch=256, source_batch=8, timing_block_updates=5,
                                      log_every_updates=10)
        for p in cfg['bank_phases']:
            p['steps'] = 10
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    R_ = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke),
              incumbent_sha256=hashlib.sha256(Path(a.incumbent).read_bytes()).hexdigest(),
              diagnosis=[], bank_arms=[], head_arms=[], checkpoints=[],
              selection={}, complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)
    save()

    # ---------------------------------------------------------------- cohorts --
    dev = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                          C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    cohorts = {}
    for S in cfg['source_counts']:
        draws = C.source_params(cfg['train_seed'], S)
        assert not any(np.allclose(t, s) for t in draws for s in dev), \
            f'training/development overlap at S={S}'
        fit, val = K_.fit_validation_split(S, cfg['split_seed'], cfg['validation_fraction'])
        cohorts[S] = dict(draws=draws, fit=fit, val=val)
    R_['cohorts'] = {str(S): dict(sources=int(S), parameters_sha256=K_.sha_array(c['draws']),
                                  fit=c['fit'].tolist(), validation=c['val'].tolist(),
                                  fit_count=int(len(c['fit'])), validation_count=int(len(c['val'])))
                     for S, c in cohorts.items()}
    R_['development'] = dict(parameters=dev.tolist(), sha256=K_.sha_array(dev),
                             groups=['existing development'] * cfg['eval_count']
                                    + ['fresh development'] * cfg['fresh_count'],
                             disjoint_from_every_training_cohort=True)
    save()
    ntr = cfg['training_intervals']
    ops_tr = K_.sine_ops(ntr, cfg['requested_modes'])
    Fdev_tr = K_.weak_sources(dev, ops_tr, ntr)
    Udev_tr = K_.fields(dev, ntr)
    print('COHORTS done', round(time.perf_counter() - begin, 1), flush=True)

    # -------------------------------------------------- diagnosis, incumbent ---
    def diagnose(tag, params, Z, train_draws, oracle_subset, meshes):
        """D1-D8 of DESIGN.md section 6 for one checkpoint."""
        rec = dict(model=tag, K=int(np.asarray(Z).shape[1]),
                   R=int(np.asarray(params['h_lin']).shape[1]),
                   weights_sha256=K_.weights_sha(params), codes_sha256=K_.sha_array(Z),
                   training_sources=int(len(train_draws)), meshes=[])
        head = head_of(params)
        for n in meshes:
            t0 = time.perf_counter()
            G = K_.bank_of(params, n)
            Rg, rank = K_.bank_r(G)
            linear = 'gj' if rec['K'] <= cfg['gauss_jordan_max'] else 'lu'
            Utr = K_.fields(train_draws[oracle_subset], n)
            Ttr, ptr, ntr2 = K_.project_targets(G, Rg, Utr)
            Udev = K_.fields(dev, n)
            Tdev, pdev, ndev2 = K_.project_targets(G, Rg, Udev)
            d1_train = np.asarray(jnp.sqrt(ptr / ntr2))
            d1_dev = np.asarray(jnp.sqrt(pdev / ndev2))
            d2 = K_.stored_code_errors(head, Rg, np.asarray(Z)[oracle_subset], Ttr, ptr, ntr2)
            d3, it3, rs3 = K_.oracle_errors(head, Rg, Z, Ttr, ptr, ntr2, cfg['recon_budget'],
                                            cfg['recon_starts'], cfg['stationarity_tolerance'],
                                            linear)
            d5, it5, rs5 = K_.oracle_errors(head, Rg, Z, Tdev, pdev, ndev2, cfg['recon_budget'],
                                            cfg['recon_starts'], cfg['stationarity_tolerance'],
                                            linear)
            B = K_.reduce_bank(G, K_.sine_ops(n, cfg['requested_modes']), n) if n == ntr else None
            solved = None
            if n == ntr:
                se, sr, si, sg = weak_solve_errors(params, Z, ops_tr, B, Fdev_tr, G, Rg,
                                                   Tdev, pdev, ndev2, cfg['lm_budget'],
                                                   cfg['stationarity_tolerance'], linear)
                solved = dict(**K_.summarise(se), stationary=int((sr == 4).sum()),
                              exit_reasons=sr.tolist(), iterations=si.tolist(),
                              normalised_gradient=sg.tolist())
            phat_fit = K_.descriptor(train_draws)
            phat_dev = K_.descriptor(dev)
            nearest = np.sqrt(((phat_dev[:, None, :] - phat_fit[None, :, :]) ** 2).sum(-1)).min(1)
            order_d, order_n = np.argsort(np.argsort(d5)), np.argsort(np.argsort(nearest))
            spearman = float(np.corrcoef(order_d, order_n)[0, 1]) if len(d5) > 2 else None
            rec['meshes'].append(dict(
                intervals=n, bank_rank=rank, seconds=time.perf_counter() - t0,
                D1_bank_floor_training=K_.summarise(d1_train),
                D1_bank_floor_development=K_.summarise(d1_dev),
                D2_head_at_stored_codes_training=K_.summarise(d2),
                D3_head_best_found_training=K_.summarise(d3),
                D3_iterations=it3.tolist(), D3_exit_reasons=rs3.tolist(),
                D4_code_refit_gain_training=dict(worst=float(np.max(d2 - d3)),
                                                 median=float(np.median(d2 - d3)),
                                                 relative_worst=float(np.max((d2 - d3) / d3))),
                D5_head_best_found_development=K_.summarise(d5),
                D5_iterations=it5.tolist(), D5_exit_reasons=rs5.tolist(),
                D6_generalisation_gap=dict(worst_minus_worst=float(d5.max() - d3.max()),
                                           ratio_of_worst=float(d5.max() / max(d3.max(), 1e-300))),
                D7_solved_minus_best_found=(None if solved is None else
                                            float(solved['worst'] - d5.max())),
                D8_nearest_fit_parameter_distance=nearest.tolist(),
                D8_spearman_best_found_vs_distance=spearman,
                head_floor_over_bank_floor=float(d5.max() / max(d1_dev.max(), 1e-300)),
                solved_development=solved))
            print('DIAG', tag, n, 'bank', float(d1_dev.max()), 'best-found', float(d5.max()),
                  flush=True)
            del G, Rg, Utr, Udev, Ttr, Tdev
            jax.clear_caches()
        return rec

    params0, Z0, ck0 = sc.load_pkl(a.incumbent)
    basis0 = np.load(a.incumbent_basis)
    np.testing.assert_array_equal(np.asarray(Z0), basis0['training_latents'])
    inc_draws = C.source_params(cfg['train_seed'],
                                cfg['incumbent_draw_count'])[:cfg['incumbent_training_prefix']]
    assert len(inc_draws) == len(Z0), (len(inc_draws), len(Z0))
    rng = np.random.default_rng(cfg['oracle_seed'])
    sub0 = np.sort(rng.choice(len(inc_draws), min(cfg['train_oracle_subsample'], len(inc_draws)),
                              replace=False))
    if a.bank_checkpoint is None:
        R_['diagnosis'].append(diagnose('incumbent_r128_joint', params0, Z0, inc_draws, sub0,
                                        cfg['floor_intervals']))
        save()
    print('DIAGNOSIS done', round(time.perf_counter() - begin, 1), flush=True)

    # ------------------------------------------------------------ bank sweep ---
    coords_tr = K_.coords_of(ntr)
    head_only = a.bank_checkpoint is not None
    supplied_sources = None
    if head_only:
        supplied_sources = int(sc.load_pkl(a.bank_checkpoint)[2]['sources'])
        assert supplied_sources in cohorts, (supplied_sources, list(cohorts))
    Ucache = {}
    for S in cfg['source_counts']:
        if head_only and S != supplied_sources:
            continue
        Ucache[S] = K_.fields(cohorts[S]['draws'], ntr)

    def store(tag, params, Z, extra):
        dest = out / 'checkpoints' / (tag + '.pkl')
        sc.save_pkl(dest, params, Z, dict(pde='poisson2d', N=ntr + 1, k=int(np.asarray(Z).shape[1]),
                                          r=int(np.asarray(params['h_lin']).shape[1]),
                                          cell='p-bank-head', tag=tag, **extra))
        rec = dict(id=tag, path=f'checkpoints/{tag}.pkl',
                   sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                   weights_sha256=K_.weights_sha(params), codes_sha256=K_.sha_array(Z), **extra)
        R_['checkpoints'].append(rec)
        save()
        return dest

    for R in (cfg['ranks'] if not head_only else []):
        for S in cfg['source_counts']:
            tag = f'bank_R{R}_S{S}'
            t0 = time.perf_counter()
            fit = cohorts[S]['fit']
            draws = cohorts[S]['draws']
            U = Ucache[S][jnp.asarray(fit)]
            arch = dict(cfg['arch'])
            arch['g_hidden'] = R
            key = jax.random.PRNGKey(cfg['bank_phases'][0]['seed'])
            out_scale = float(jnp.sqrt(jnp.mean(U * U)))
            params = sc.init_separable(key, cfg['latents'][0], R, out_scale=out_scale, **arch)
            Z = np.asarray(0.1 * jax.random.normal(jax.random.PRNGKey(cfg['bank_phases'][0]['seed'] + 7),
                                                   (len(fit), cfg['latents'][0]), dtype=jnp.float64))
            phases = []
            for spec in cfg['bank_phases']:
                if spec['phase'] in ('joint', 'bank'):
                    latent = Z
                    if spec['phase'] == 'bank':
                        G = K_.bank_of(params, ntr)
                        Rg, _ = K_.bank_r(G)
                        T, _, _ = K_.project_targets(G, Rg, U)
                        latent = np.asarray(jnp.linalg.solve(Rg, T.T).T)
                        del G, Rg, T
                    params, latent, info = F_.train_spatial_phase(
                        params, latent, U, coords_tr, cfg['training_common'], spec['phase'],
                        spec['steps'], spec['lr'], spec['seed'], f'{tag}/{spec["tag"]}',
                        cfg['phase_wall_cap_seconds'])
                    if spec['phase'] == 'joint':
                        Z = latent
                else:
                    G = K_.bank_of(params, ntr)
                    Rg, _ = K_.bank_r(G)
                    T, perp2, nu2 = K_.project_targets(G, Rg, U)
                    Bt = K_.reduce_bank(G, ops_tr, ntr)
                    Fm = K_.weak_sources(draws[fit], ops_tr, ntr)
                    arrays = (Rg, T, perp2, nu2, Bt, Fm, jnp.sum(Fm ** 2, axis=1),
                              np.zeros((0, 2), np.int64), np.zeros((0,)))
                    params, Z, info = F_.train_head_phase(
                        params, cfg['latents'][0], R, arrays, cfg['training_common'], (0., 0.),
                        spec['steps'], spec['lr'], spec['seed'], f'{tag}/{spec["tag"]}',
                        cfg['phase_wall_cap_seconds'], init_head=False,
                        head_params=params, latent=Z)
                    del G, Rg, T, Bt, Fm
                assert info['finite'], f'{tag}/{spec["tag"]} lost finiteness'
                phases.append(info)
                jax.clear_caches()
            store(tag, params, Z, dict(layer='bank', rank=R, sources=S,
                                       fit_count=int(len(fit)), training_seconds=time.perf_counter() - t0))
            arm = dict(arm=tag, R=R, S=S, K=cfg['latents'][0], fit_count=int(len(fit)),
                       validation_count=int(len(cohorts[S]['val'])), phases=phases,
                       training_seconds=time.perf_counter() - t0, floors=[])
            head = head_of(params)
            for n in cfg['floor_intervals']:
                G = K_.bank_of(params, n)
                Rg, rank = K_.bank_r(G)
                assert rank['rank_valid'], f'{tag} rank deficient at n={n}: {rank["rank"]}'
                f_fit = K_.bank_floor(G, Rg, draws[fit], n)
                f_val = K_.bank_floor(G, Rg, draws[cohorts[S]['val']], n)
                f_dev = K_.bank_floor(G, Rg, dev, n)
                Uval = K_.fields(draws[cohorts[S]['val']], n)
                Tv, pv, nv = K_.project_targets(G, Rg, Uval)
                bf_val, _, _ = K_.oracle_errors(head, Rg, Z, Tv, pv, nv, cfg['recon_budget'],
                                                cfg['recon_starts'], cfg['stationarity_tolerance'], 'gj')
                Ud = K_.fields(dev, n)
                Td, pd, nd = K_.project_targets(G, Rg, Ud)
                bf_dev, _, _ = K_.oracle_errors(head, Rg, Z, Td, pd, nd, cfg['recon_budget'],
                                                cfg['recon_starts'], cfg['stationarity_tolerance'], 'gj')
                arm['floors'].append(dict(intervals=n, bank_rank=rank,
                                          fit=K_.summarise(f_fit), validation=K_.summarise(f_val),
                                          development=K_.summarise(f_dev),
                                          head_best_found_validation=K_.summarise(bf_val),
                                          head_best_found_development=K_.summarise(bf_dev)))
                print('BANK', tag, n, 'floor dev', float(f_dev.max()), 'val', float(f_val.max()),
                      flush=True)
                del G, Rg, Uval, Ud
                jax.clear_caches()
            R_['bank_arms'].append(arm)
            save()
            del U, params
            jax.clear_caches()

    # ------------------------------------------------------- bank selection ----
    pick_n = cfg['floor_intervals'][-1]
    if head_only:
        bp, bz, bcfg = sc.load_pkl(a.bank_checkpoint)
        best = dict(arm=bcfg['tag'], R=int(bcfg['rank']), S=int(bcfg['sources']))
        R_['selection']['bank'] = dict(
            rule=('supplied: this run is head-only, on a bank selected by DESIGN.md '
                  'amendment 4 on the common selection cohort'),
            selected=best['arm'], R=best['R'], S=best['S'], mesh=None,
            source_checkpoint=str(a.bank_checkpoint),
            source_checkpoint_sha256=hashlib.sha256(
                Path(a.bank_checkpoint).read_bytes()).hexdigest(),
            development_ranking_agrees=None, ranking=[])
        shutil.copy2(a.bank_checkpoint, out / 'checkpoints' / (best['arm'] + '.pkl'))
        R_['checkpoints'].append(dict(id=best['arm'], path=f"checkpoints/{best['arm']}.pkl",
                                      sha256=R_['selection']['bank']['source_checkpoint_sha256'],
                                      weights_sha256=K_.weights_sha(bp),
                                      codes_sha256=K_.sha_array(bz), layer='bank',
                                      rank=best['R'], sources=best['S'], supplied=True))
        save()
        print('BANK SUPPLIED', best['arm'], flush=True)

    def bank_key(arm):
        f = next(x for x in arm['floors'] if x['intervals'] == pick_n)
        return (f['validation']['worst'], f['validation']['median'], arm['R'], arm['S'])
    best = best if head_only else min(R_['bank_arms'], key=bank_key)
    if not head_only:
      R_['selection']['bank'] = dict(
        rule='lowest worst internal-validation bank projection floor at the finest floor mesh; '
             'ties by median, then smaller R, then smaller S',
        mesh=pick_n, selected=best['arm'], R=best['R'], S=best['S'],
        ranking=[dict(arm=x['arm'], validation_worst=bank_key(x)[0],
                      development_worst=next(f for f in x['floors']
                                             if f['intervals'] == pick_n)['development']['worst'])
                 for x in sorted(R_['bank_arms'], key=bank_key)])
    dev_rank = sorted(R_['bank_arms'],
                      key=lambda x: next(f for f in x['floors']
                                         if f['intervals'] == pick_n)['development']['worst'])
    R_['selection']['bank']['development_ranking_agrees'] = bool(dev_rank[0]['arm'] == best['arm'])
    save()
    print('BANK SELECTED', best['arm'], flush=True)

    # ------------------------------------------------------------ head sweep ---
    Sbest, Rbest = best['S'], best['R']
    fit = cohorts[Sbest]['fit']
    val = cohorts[Sbest]['val']
    draws = cohorts[Sbest]['draws']
    bank_params, _, _ = sc.load_pkl(out / 'checkpoints' / (best['arm'] + '.pkl'))
    assert int(np.asarray(bank_params['h_lin']).shape[1]) == Rbest, (Rbest,)
    Ufit = Ucache[Sbest][jnp.asarray(fit)]
    G = K_.bank_of(bank_params, ntr)
    Rg, rank_tr = K_.bank_r(G)
    Tfit, pfit, nfit = K_.project_targets(G, Rg, Ufit)
    Bt = K_.reduce_bank(G, ops_tr, ntr)
    Fmfit = K_.weak_sources(draws[fit], ops_tr, ntr)
    edges, ed2 = F_.knn_edges(K_.descriptor(draws[fit]), cfg['knn'])
    Uval_tr = Ucache[Sbest][jnp.asarray(val)]
    Tval, pval, nval = K_.project_targets(G, Rg, Uval_tr)
    Tdev_tr, pdev_tr, ndev_tr = K_.project_targets(G, Rg, Udev_tr)
    R_['head_layer'] = dict(bank=best['arm'], bank_rank=rank_tr, sources=Sbest, R=Rbest,
                            fit_count=int(len(fit)), validation_count=int(len(val)),
                            edges=int(len(edges)), knn=cfg['knn'],
                            descriptor_scope='offline training regulariser only; no descriptor '
                                             'reaches any online query')
    save()
    arrays = (Rg, Tfit, pfit, nfit, Bt, Fmfit, jnp.sum(Fmfit ** 2, axis=1), edges, ed2)
    arch = dict(cfg['arch'])
    arch['g_hidden'] = Rbest
    for K in cfg['latents']:
        linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
        for bw in cfg['beta_weak']:
            for bs in cfg['beta_smooth']:
                tag = f'head_K{K}_w{bw:g}_s{bs:g}'
                t0 = time.perf_counter()
                params, Z, info = F_.train_head_phase(
                    bank_params, K, Rbest, arrays, {**cfg['training_common'], 'arch': arch},
                    (bw, bs), cfg['head_steps'], cfg['head_lr'], cfg['head_seed'], tag,
                    cfg['head_wall_cap_seconds'], init_head=True)
                assert info['finite'], f'{tag} lost finiteness'
                head = head_of(params)
                bf_val, itv, rsv = K_.oracle_errors(head, Rg, Z, Tval, pval, nval,
                                                    cfg['recon_budget'], cfg['recon_starts'],
                                                    cfg['stationarity_tolerance'], linear)
                bf_dev, itd, rsd = K_.oracle_errors(head, Rg, Z, Tdev_tr, pdev_tr, ndev_tr,
                                                    cfg['recon_budget'], cfg['recon_starts'],
                                                    cfg['stationarity_tolerance'], linear)
                st = K_.stored_code_errors(head, Rg, Z, Tfit, pfit, nfit)
                se, sr, si, sg = weak_solve_errors(params, Z, ops_tr, Bt, Fdev_tr, G, Rg,
                                                   Tdev_tr, pdev_tr, ndev_tr, cfg['lm_budget'],
                                                   cfg['stationarity_tolerance'], linear)
                store(tag, params, Z, dict(layer='head', rank=Rbest, sources=Sbest, K=K,
                                           beta_weak=bw, beta_smooth=bs, bank=best['arm'],
                                           training_seconds=time.perf_counter() - t0))
                R_['head_arms'].append(dict(
                    arm=tag, K=K, R=Rbest, S=Sbest, beta_weak=bw, beta_smooth=bs, bank=best['arm'],
                    training=info, intervals=ntr,
                    head_at_stored_codes_fit=K_.summarise(st),
                    best_found_validation=K_.summarise(bf_val),
                    best_found_development=K_.summarise(bf_dev),
                    validation_exit_reasons=rsv.tolist(), development_exit_reasons=rsd.tolist(),
                    weak_solved_development=dict(**K_.summarise(se),
                                                 stationary=int((sr == 4).sum()),
                                                 exit_reasons=sr.tolist(),
                                                 iterations=si.tolist(),
                                                 normalised_gradient=sg.tolist()),
                    training_seconds=time.perf_counter() - t0))
                print('HEAD', tag, 'val', float(bf_val.max()), 'dev', float(bf_dev.max()),
                      'solved', float(se.max()), flush=True)
                save()
                jax.clear_caches()

    # ------------------------------------------------------- head selection ----
    R_['selection']['heads'] = []
    for K in cfg['latents']:
        cands = [x for x in R_['head_arms'] if x['K'] == K]
        pick = min(cands, key=lambda x: (x['best_found_validation']['worst'],
                                         x['best_found_validation']['median'],
                                         x['beta_weak'], x['beta_smooth']))
        devbest = min(cands, key=lambda x: x['best_found_development']['worst'])
        R_['selection']['heads'].append(dict(
            K=K, rule='lowest worst internal-validation best-found at the training mesh; '
                      'ties by median, then smaller beta_weak, then smaller beta_smooth',
            selected=pick['arm'], development_ranking_agrees=bool(devbest['arm'] == pick['arm']),
            ranking=[dict(arm=x['arm'], validation_worst=x['best_found_validation']['worst'],
                          development_worst=x['best_found_development']['worst'],
                          solved_development_worst=x['weak_solved_development']['worst'])
                     for x in sorted(cands, key=lambda x: x['best_found_validation']['worst'])]))
        # frozen primary: copy under a stable name and build its correction basis
        pp, pz, _ = sc.load_pkl(out / 'checkpoints' / (pick['arm'] + '.pkl'))
        Tp, pperp, pnu = K_.project_targets(G, Rg, Ufit)
        payload, binfo = correction_basis(pp, pz, G, Rg, Tp, pperp, pnu, cfg['correction_count'])
        np.savez_compressed(out / f'primary_K{K}-basis.npz', **payload)
        assert binfo['orthogonality_error'] < 1e-8, binfo
        R_['selection']['heads'][-1]['basis'] = dict(
            path=f'primary_K{K}-basis.npz', **binfo,
            sha256=hashlib.sha256((out / f'primary_K{K}-basis.npz').read_bytes()).hexdigest())
        save()
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('P-BANK-HEAD TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
