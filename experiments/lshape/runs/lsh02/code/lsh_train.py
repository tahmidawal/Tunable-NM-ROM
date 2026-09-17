"""lshape job 1: operator gates, cohorts, the bank sweep over boundary factor and rank,
the head arms, and their correction bases (DESIGN.md sections 2-5).

Nothing here is timed as a query; every number is an offline representation or training
diagnostic. The frozen checkpoints this job writes are the only thing the solve jobs
consume.

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

import sep_common as sc
import arms as A
import lsh_core as K_
import lsh_fit as F_


def weak_solve_errors(params, codes, B, Fm, Rg, T, perp2, nu2, budget, gtol, linear):
    """The ROM's own weak solve, untimed, from the nearest cached training code."""
    head = K_.head_of(params)
    trust = float(np.max(np.linalg.norm(np.asarray(codes) - np.asarray(codes).mean(0), axis=1)))
    lm = A.make_stationary_lm(lambda z, fm, B: B @ head(z) - fm, budget, trust, gtol, linear)
    pred = jax.jit(jax.vmap(lambda z, B: B @ head(z), in_axes=(0, None)))(jnp.asarray(codes), B)

    @jax.jit
    def one(fm, B, pred, codes):
        idx = jnp.argmin(jnp.sum((pred - fm[None, :]) ** 2, axis=1))
        return lm(codes[idx], (fm, B), 0.)
    errs, reasons, iters, gns = [], [], [], []
    codes_j = jnp.asarray(codes)
    for i in range(int(Fm.shape[0])):
        z, rn, it, reason, gn = jax.device_get(one(jnp.asarray(Fm)[i], B, pred, codes_j))
        res = np.asarray(jnp.asarray(Rg) @ head(jnp.asarray(z))) - np.asarray(T[i])
        errs.append(float(np.sqrt(res @ res + float(perp2[i])) / np.sqrt(float(nu2[i]))))
        reasons.append(int(reason)); iters.append(int(it)); gns.append(float(gn))
    return np.asarray(errs), np.asarray(reasons), np.asarray(iters), np.asarray(gns)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    if a.smoke:
        cfg.update(training_intervals=32, floor_intervals=[32, 64], requested_modes=16,
                   recon_budget=60, lm_budget=60, train_oracle_subsample=8, head_steps=40,
                   correction_count=8, latents=[4, 8], phase_wall_cap_seconds=60, head_wall_cap_seconds=60)
        cfg['cohorts'] = dict(training=dict(seed=0, draw=128, count=64), common=dict(seed=20260916, draw=32, count=16),
                              development=dict(seed=20260917, draw=16, count=8))
        for arm in cfg['bank_arms']:
            arm['R'] = arm['R'] // 16
            arm['arch'] = dict(arm['arch'], n_ff=arm['arch']['n_ff'] // 4)
        cfg['training_common'].update(point_batch=128, source_batch=8, timing_block_updates=5, log_every_updates=10)
        for p in cfg['bank_phases']:
            p['steps'] = 10
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    R_ = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
              backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
              smoke=bool(a.smoke), gates=[], bank_arms=[], head_arms=[], checkpoints=[], selection={},
              complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)
    save()

    # -------------------------------------------------- geometry and gates ---
    ntr = cfg['training_intervals']
    meshes = sorted(set([ntr] + cfg['floor_intervals']))
    geoms, foms = {}, {}
    for n in meshes:
        geoms[n] = K_.Geometry(n, 'lshape')
        foms[n] = K_.FOM(geoms[n], build_ic0=False)
        g = foms[n].gates()
        lam1, _, minfo = foms[n].modes(1)
        g.update(intervals=n, lambda_min=float(lam1[0]), lambda1_reference=K_.LSHAPE_LAMBDA1,
                 lambda1_relative_difference=minfo['lambda1_relative_difference'],
                 positive_definite=bool(lam1[0] > 0),
                 lambda1_within_1pct=bool(minfo['lambda1_relative_difference'] <= 1e-2) if n >= 128 else None,
                 factor_seconds=foms[n].factor_seconds, lu_nnz=foms[n].lu_nnz)
        g['passed'] = bool(g['symmetric'] and g['independent_assembly_agrees'] and g['positive_definite']
                           and (g['lambda1_within_1pct'] in (True, None)))
        R_['gates'].append(g)
        print('GATE', g, flush=True)
        assert g['passed'], g
    save()

    # ---------------------------------------------------------------- cohorts --
    cc = cfg['cohorts']
    train_draws, tinfo = K_.cohort(cc['training']['seed'], cc['training']['draw'], cc['training']['count'])
    common, cinfo = K_.cohort(cc['common']['seed'], cc['common']['draw'], cc['common']['count'])
    dev, dinfo = K_.cohort(cc['development']['seed'], cc['development']['draw'], cc['development']['count'])
    for name, (x, y) in dict(train_common=(train_draws, common), train_dev=(train_draws, dev),
                             common_dev=(common, dev)).items():
        assert not any(np.allclose(t, s) for t in x for s in y), f'cohort overlap {name}'
    S = len(train_draws)
    fit, val = K_.fit_validation_split(S, cfg['split_seed'], cfg['validation_fraction'])
    rng = np.random.default_rng(cfg['oracle_seed'])
    fit_sub = np.sort(rng.choice(fit, min(cfg['train_oracle_subsample'], len(fit)), replace=False))
    R_['cohorts'] = dict(training=dict(**tinfo, fit=fit.tolist(), validation=val.tolist(),
                                       fit_count=int(len(fit)), validation_count=int(len(val)),
                                       fit_oracle_subsample=fit_sub.tolist()),
                         common=cinfo, development=dict(**dinfo, parameters=dev.tolist()),
                         pairwise_disjoint=True)
    save()

    # ------------------------------------------------------------- fields -----
    t0 = time.perf_counter()
    U = {n: {} for n in meshes}
    U[ntr]['train'] = K_.fields(foms[ntr], train_draws)
    for n in meshes:
        U[n]['dev'] = K_.fields(foms[n], dev)
        U[n]['common'] = K_.fields(foms[n], common)
        if n != ntr:
            U[n]['val'] = K_.fields(foms[n], train_draws[val])
            U[n]['fit_sub'] = K_.fields(foms[n], train_draws[fit_sub])
        else:
            U[n]['val'] = U[ntr]['train'][val]
            U[n]['fit_sub'] = U[ntr]['train'][fit_sub]
    ops_tr = K_.weak_ops(foms[ntr], cfg['requested_modes'], cfg['mode_seed'])
    R_['weak_ops'] = dict(intervals=ntr, **ops_tr['info'])
    F_dev_int = np.stack([K_.source_interior(geoms[ntr], q) for q in dev])
    Fm_dev = K_.weak_sources(ops_tr, F_dev_int)
    R_['field_seconds'] = time.perf_counter() - t0
    R_['snapshot_sha256'] = K_.sha_array(U[ntr]['train'])
    print('FIELDS done', round(time.perf_counter() - begin, 1), flush=True)
    save()

    def store(tag, params, Z, extra):
        dest = out / 'checkpoints' / (tag + '.pkl')
        sc.save_pkl(dest, params, Z, dict(pde='poisson2d_lshape', N=ntr, k=int(np.asarray(Z).shape[1]),
                                          r=int(np.asarray(params['h_lin']).shape[1]), cell='lshape', tag=tag,
                                          **extra))
        rec = dict(id=tag, path=f'checkpoints/{tag}.pkl', sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                   weights_sha256=K_.weights_sha(params), codes_sha256=K_.sha_array(Z), **extra)
        R_['checkpoints'].append(rec)
        save()
        return dest

    def floors_and_oracles(tag, params, Z, features, K):
        head = K_.head_of(params)
        linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
        rows = []
        for n in cfg['floor_intervals']:
            t0 = time.perf_counter()
            G = K_.bank_of(features, params, geoms[n])
            Rg, rank = K_.bank_r(G)
            assert rank['rank_valid'], f'{tag} rank deficient at n={n}: {rank["rank"]}'
            row = dict(intervals=n, bank_rank=rank, seconds=None)
            for name in ('fit_sub', 'val', 'common', 'dev'):
                T, perp2, nu2 = K_.project_targets(G, Rg, U[n][name])
                row[f'floor_{name}'] = K_.summarise(np.asarray(jnp.sqrt(perp2 / nu2)))
                if name in ('val', 'dev'):
                    bf, it, rs = K_.oracle_errors(head, Rg, Z, T, perp2, nu2, cfg['recon_budget'],
                                                  cfg['recon_starts'], cfg['stationarity_tolerance'], linear)
                    row[f'head_best_found_{name}'] = K_.summarise(bf)
                    row[f'head_best_found_{name}_exit_reasons'] = rs.tolist()
            row['seconds'] = time.perf_counter() - t0
            rows.append(row)
            print('FLOOR', tag, n, 'dev', row['floor_dev']['worst'], 'common', row['floor_common']['worst'],
                  'bf dev', row['head_best_found_dev']['worst'], flush=True)
            del G, Rg
            jax.clear_caches()
        return rows

    # ------------------------------------------------------------ bank sweep ---
    coords_tr = geoms[ntr].coords
    Ufit = jnp.asarray(U[ntr]['train'][fit])
    out_scale = float(jnp.sqrt(jnp.mean(Ufit * Ufit)))
    K0 = cfg['latents'][0]
    for arm in cfg['bank_arms']:
        tag = arm['arm']
        t0 = time.perf_counter()
        R = int(arm['R'])
        E = int(arm.get('n_enrich', 0))
        features = K_.make_features(arm['factor'], E)
        key = jax.random.PRNGKey(cfg['bank_phases'][0]['seed'])
        params = K_.init_decoder(key, K0, R, arm['factor'], E, out_scale, arm['arch'], coords_tr)
        Z = np.asarray(0.1 * jax.random.normal(jax.random.PRNGKey(cfg['bank_phases'][0]['seed'] + 7),
                                               (len(fit), K0), dtype=jnp.float64))
        phases = []
        for spec in cfg['bank_phases']:
            if spec['phase'] in ('joint', 'bank'):
                latent = Z
                if spec['phase'] == 'bank':
                    G = K_.bank_of(features, params, geoms[ntr])
                    Rg, _ = K_.bank_r(G)
                    T, _, _ = K_.project_targets(G, Rg, Ufit)
                    latent = np.asarray(jnp.linalg.solve(Rg, T.T).T)
                    del G, Rg, T
                params, latent, info = F_.train_spatial_phase(
                    features, params, latent, Ufit, coords_tr, cfg['training_common'], spec['phase'],
                    spec['steps'], spec['lr'], spec['seed'], f'{tag}/{spec["tag"]}', cfg['phase_wall_cap_seconds'])
                if spec['phase'] == 'joint':
                    Z = latent
            else:
                G = K_.bank_of(features, params, geoms[ntr])
                Rg, _ = K_.bank_r(G)
                T, perp2, nu2 = K_.project_targets(G, Rg, Ufit)
                params, Z, info = F_.train_head_phase(
                    params, K0, R + E, Rg, T, perp2, nu2, {**cfg['training_common'], 'arch': arm['arch']},
                    spec['steps'], spec['lr'], spec['seed'], f'{tag}/{spec["tag"]}', cfg['phase_wall_cap_seconds'],
                    init_head=False, head_params=params, latent=Z)
                del G, Rg, T
            assert info['finite'], f'{tag}/{spec["tag"]} lost finiteness'
            phases.append(info)
            jax.clear_caches()
        extra = dict(layer='bank', factor=arm['factor'], n_enrich=E, rank=R, rank_total=R + E, sources=S,
                     arch=arm['arch'], fit_count=int(len(fit)), training_seconds=time.perf_counter() - t0)
        store(tag, params, Z, extra)
        rows = floors_and_oracles(tag, params, Z, features, K0)
        R_['bank_arms'].append(dict(arm=tag, factor=arm['factor'], n_enrich=E, R=R, R_total=R + E, K=K0, S=S,
                                    arch=arm['arch'], phases=phases, floors=rows,
                                    training_seconds=time.perf_counter() - t0))
        save()
        del params
        jax.clear_caches()

    # ------------------------------------------------------- bank selection ----
    pick_n = ntr
    order = {arm['arm']: i for i, arm in enumerate(cfg['bank_arms'])}

    def bank_key(x):
        f = next(r for r in x['floors'] if r['intervals'] == pick_n)
        return (f['floor_common']['worst'], f['floor_common']['median'], x['R'], order[x['arm']])
    ranking = sorted(R_['bank_arms'], key=bank_key)
    best = ranking[0]
    dev_rank = sorted(R_['bank_arms'], key=lambda x: next(r for r in x['floors'] if r['intervals'] == pick_n)['floor_dev']['worst'])
    R_['selection']['bank'] = dict(
        rule='lowest worst bank projection floor on the common selection cohort at the training mesh; '
             'ties by median, then smaller R, then table order (DESIGN.md section 4)',
        mesh=pick_n, selected=best['arm'], R=best['R'], factor=best['factor'],
        development_ranking_agrees=bool(dev_rank[0]['arm'] == best['arm']),
        ranking=[dict(arm=x['arm'], common_worst=bank_key(x)[0], common_median=bank_key(x)[1],
                      development_worst=next(r for r in x['floors'] if r['intervals'] == pick_n)['floor_dev']['worst'])
                 for x in ranking])
    save()
    print('BANK SELECTED', best['arm'], flush=True)

    # ------------------------------------------------------------ head arms ----
    Tdev_cache = {}
    for arm in cfg['bank_arms']:
        tag = arm['arm']
        bank_params, _, bcfg = sc.load_pkl(out / 'checkpoints' / (tag + '.pkl'))
        E = int(bcfg['n_enrich']); R = int(bcfg['rank']); Rtot = R + E
        features = K_.make_features(bcfg['factor'], E)
        G = K_.bank_of(features, bank_params, geoms[ntr])
        Rg, rank_tr = K_.bank_r(G)
        Tfit, pfit, nfit = K_.project_targets(G, Rg, Ufit)
        Tval, pval, nval = K_.project_targets(G, Rg, U[ntr]['val'])
        Tdev, pdev, ndev = K_.project_targets(G, Rg, U[ntr]['dev'])
        B = K_.reduce_bank(ops_tr, foms[ntr], G)
        Ks = cfg['latents'] if tag == best['arm'] else cfg['latents'][:1]
        for K in Ks:
            htag = f'head_{tag}_K{K}'
            t0 = time.perf_counter()
            linear = 'gj' if K <= cfg['gauss_jordan_max'] else 'lu'
            params, Z, info = F_.train_head_phase(
                bank_params, K, Rtot, Rg, Tfit, pfit, nfit, {**cfg['training_common'], 'arch': arm['arch']},
                cfg['head_steps'], cfg['head_lr'], cfg['head_seed'], htag, cfg['head_wall_cap_seconds'],
                init_head=True)
            assert info['finite'], f'{htag} lost finiteness'
            head = K_.head_of(params)
            st = K_.stored_code_errors(head, Rg, Z, Tfit, pfit, nfit)
            bf_val, _, rsv = K_.oracle_errors(head, Rg, Z, Tval, pval, nval, cfg['recon_budget'],
                                              cfg['recon_starts'], cfg['stationarity_tolerance'], linear)
            bf_dev, _, rsd = K_.oracle_errors(head, Rg, Z, Tdev, pdev, ndev, cfg['recon_budget'],
                                              cfg['recon_starts'], cfg['stationarity_tolerance'], linear)
            se, sr, si, sg = weak_solve_errors(params, Z, B, Fm_dev, Rg, Tdev, pdev, ndev, cfg['lm_budget'],
                                               cfg['stationarity_tolerance'], linear)
            payload, binfo = K_.correction_basis(head, Z, Rg, Tfit, pfit, nfit, cfg['correction_count'])
            assert binfo['orthogonality_error_scaled'] < cfg['basis_orthogonality_limit'], binfo
            np.savez_compressed(out / 'checkpoints' / f'{htag}-basis.npz', **payload)
            bsha = hashlib.sha256((out / 'checkpoints' / f'{htag}-basis.npz').read_bytes()).hexdigest()
            store(htag, params, Z, dict(layer='head', factor=bcfg['factor'], n_enrich=E, rank=R, rank_total=Rtot,
                                        sources=S, K=K, bank=tag, arch=arm['arch'], objective='reconstruction',
                                        primary=bool(tag == best['arm']), basis=f'checkpoints/{htag}-basis.npz',
                                        basis_sha256=bsha, training_seconds=time.perf_counter() - t0))
            floor_dev = float(np.max(np.asarray(jnp.sqrt(pdev / ndev))))
            R_['head_arms'].append(dict(
                arm=htag, bank=tag, factor=bcfg['factor'], n_enrich=E, K=K, R=R, R_total=Rtot, S=S,
                primary=bool(tag == best['arm']), training=info, intervals=ntr, bank_rank=rank_tr,
                bank_floor_dev=floor_dev,
                head_at_stored_codes_fit=K_.summarise(st),
                best_found_validation=K_.summarise(bf_val), best_found_development=K_.summarise(bf_dev),
                validation_exit_reasons=rsv.tolist(), development_exit_reasons=rsd.tolist(),
                weak_solved_development=dict(**K_.summarise(se), stationary=int((sr == 4).sum()),
                                             exit_reasons=sr.tolist(), iterations=si.tolist(),
                                             normalised_gradient=sg.tolist()),
                head_floor_over_bank_floor=float(bf_dev.max() / max(floor_dev, 1e-300)),
                basis=dict(path=f'checkpoints/{htag}-basis.npz', sha256=bsha, **binfo),
                training_seconds=time.perf_counter() - t0))
            print('HEAD', htag, 'floor', floor_dev, 'val', float(bf_val.max()), 'dev', float(bf_dev.max()),
                  'solved', float(se.max()), 'stationary', int((sr == 4).sum()), flush=True)
            save()
        del G, Rg, B
        jax.clear_caches()

    R_['selection']['heads'] = dict(
        rule='no selection: one objective per K (reconstruction); the two heads on the selected bank are the '
             'primaries, the K=16 heads on the other banks are the declared comparison arms',
        primaries=[x['arm'] for x in R_['head_arms'] if x['primary']],
        comparisons=[x['arm'] for x in R_['head_arms'] if not x['primary']])
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('LSHAPE TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
