"""p-linear job 3: head capacity on the frozen R=512 bank, evaluated and solved in-job.

Seven head arms (DESIGN.md section 5) are trained with `pbh_fit.train_head_phase`, the
exact `pbh02` recipe, on the selected bank frozen; only width, depth, K and (for one
control) the update count change. Each arm is measured offline at the training mesh
(stored-code fit, best-found on the validation split and the development sources, bank
floor) and then solved at the finest mesh through the unchanged `poisson_ablation`
kernel, with `pbh02`'s primary and the direct DST solve interleaved as in-job controls.

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
import pilot as P
import arms as A
import sep_common as sc
import poisson_ablation as PA
import pbh_core as K_
import pbh_fit as F_
import plin_core as L_


def resolve(here, rel):
    cand = here / rel
    return cand if cand.exists() else here / Path(rel).name


def g_track_sha(params):
    h = hashlib.sha256()
    for k in ('B', 'g', 'out_scale'):
        for x in jax.tree_util.tree_leaves(params[k]):
            h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()


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
    (out / 'checkpoints').mkdir(exist_ok=True)
    if a.smoke:
        cfg.update(solve_intervals=64, repetitions=1, burn_seconds=0.001, attempt='smoke',
                   head_wall_cap_seconds=60)
        for arm in cfg['arms']:
            arm['updates'] = 30
        cfg['arms'] = cfg['arms'][:3] + cfg['arms'][4:5]
        cfg['gates'] = [g for g in cfg['gates'] if 64 in g['intervals']]
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    R_ = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke),
              bank={}, arms=[], checkpoints=[], references=[], reconstruction=[],
              arm_setup=[], invocations=[], declared_subjects=[], gates=[], complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)
    save()

    # ------------------------------------------------------------- the bank ---
    bank_params, _, bcfg = sc.load_pkl(resolve(here, cfg['bank_checkpoint']))
    prim_params, prim_codes, pcfg = sc.load_pkl(resolve(here, cfg['primary_checkpoint']))
    assert g_track_sha(bank_params) == g_track_sha(prim_params), 'primary is not on this bank'
    Rb = int(np.asarray(bank_params['h_lin']).shape[1])
    S = int(bcfg['sources'])
    R_['bank'] = dict(checkpoint=cfg['bank_checkpoint'], tag=bcfg.get('tag'), R=Rb, sources=S,
                      g_track_sha256=g_track_sha(bank_params),
                      checkpoint_sha256=hashlib.sha256(resolve(here, cfg['bank_checkpoint']).read_bytes()).hexdigest(),
                      primary_checkpoint_sha256=hashlib.sha256(resolve(here, cfg['primary_checkpoint']).read_bytes()).hexdigest(),
                      primary_tag=pcfg.get('tag'))

    dev = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                          C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    draws = C.source_params(cfg['train_seed'], S)
    assert not any(np.allclose(t, s) for t in draws for s in dev)
    fit, val = K_.fit_validation_split(S, cfg['split_seed'], cfg['validation_fraction'])
    assert len(fit) == len(prim_codes), (len(fit), len(prim_codes))
    R_['cohorts'] = dict(sources=S, fit_count=int(len(fit)), validation_count=int(len(val)),
                         parameters_sha256=L_.sha_array(draws), development_sha256=L_.sha_array(dev))
    save()

    ntr = int(cfg['training_intervals'])
    ops_tr = K_.sine_ops(ntr, cfg['requested_modes'])
    G = K_.bank_of(bank_params, ntr)
    Rg, rank_tr = K_.bank_r(G)
    assert rank_tr['rank_valid'], rank_tr
    Ufit = K_.fields(draws[fit], ntr)
    Tfit, pfit, nfit = K_.project_targets(G, Rg, Ufit)
    Bt = K_.reduce_bank(G, ops_tr, ntr)
    Fmfit = K_.weak_sources(draws[fit], ops_tr, ntr)
    Uval = K_.fields(draws[val], ntr)
    Tval, pval, nval = K_.project_targets(G, Rg, Uval)
    Udev = K_.fields(dev, ntr)
    Tdev, pdev, ndev = K_.project_targets(G, Rg, Udev)
    floor_val = np.asarray(jnp.sqrt(pval / nval))
    floor_dev = np.asarray(jnp.sqrt(pdev / ndev))
    R_['bank'].update(bank_rank_training_mesh=rank_tr, training_intervals=ntr,
                      floor_validation=K_.summarise(floor_val), floor_development=K_.summarise(floor_dev))
    del Ufit, Uval, Udev
    arrays = (Rg, Tfit, pfit, nfit, Bt, Fmfit, jnp.sum(Fmfit ** 2, axis=1),
              np.zeros((0, 2), np.int64), np.zeros((0,)))
    save()
    print('BANK ready', round(time.perf_counter() - begin, 1), 'floor dev', float(floor_dev.max()), flush=True)

    # ---------------------------------------------------------- the pbh02 primary --
    head0 = lambda z: sc.head(prim_params, z)
    st0 = K_.stored_code_errors(head0, Rg, prim_codes, Tfit, pfit, nfit)
    bf0v, _, _ = K_.oracle_errors(head0, Rg, prim_codes, Tval, pval, nval, cfg['recon_budget'],
                                  cfg['recon_starts'], cfg['stationarity_tolerance'], 'gj')
    bf0d, _, _ = K_.oracle_errors(head0, Rg, prim_codes, Tdev, pdev, ndev, cfg['recon_budget'],
                                  cfg['recon_starts'], cfg['stationarity_tolerance'], 'gj')
    R_['arms'].append(dict(arm='pbh02_primary_K32', K=int(prim_codes.shape[1]), width=128, layers=2,
                           updates=None, role='pbh02 primary, frozen, in-job control',
                           head_at_stored_codes_fit=K_.summarise(st0),
                           best_found_validation=K_.summarise(bf0v),
                           best_found_development=K_.summarise(bf0d),
                           ratio_dev_best_found_over_floor=float(bf0d.max() / floor_dev.max())))
    save()
    print('PRIMARY dev best-found', float(bf0d.max()), flush=True)

    # -------------------------------------------------------------- the arms --
    trained = {}
    for spec in cfg['arms']:
        tag = spec['arm']
        K = int(spec['K'])
        arch = dict(cfg['arch'])
        arch.update(g_hidden=Rb, h_hidden=int(spec['width']), h_layers=int(spec['layers']))
        t0 = time.perf_counter()
        params, Z, info = F_.train_head_phase(
            bank_params, K, Rb, arrays, {**cfg['training_common'], 'arch': arch}, (0., 0.),
            int(spec['updates']), cfg['head_lr'], cfg['head_seed'], tag,
            cfg['head_wall_cap_seconds'], init_head=True)
        assert info['finite'], f'{tag} lost finiteness'
        assert g_track_sha(params) == R_['bank']['g_track_sha256'], 'bank moved'
        head = lambda z, p=params: sc.head(p, z)
        oracle_linear = 'gj' if K <= cfg['oracle_gauss_jordan_max'] else 'lu'
        st = K_.stored_code_errors(head, Rg, Z, Tfit, pfit, nfit)
        bfv, itv, rsv = K_.oracle_errors(head, Rg, Z, Tval, pval, nval, cfg['recon_budget'],
                                         cfg['recon_starts'], cfg['stationarity_tolerance'], oracle_linear)
        bfd, itd, rsd = K_.oracle_errors(head, Rg, Z, Tdev, pdev, ndev, cfg['recon_budget'],
                                         cfg['recon_starts'], cfg['stationarity_tolerance'], oracle_linear)
        dest = out / 'checkpoints' / f'{tag}.pkl'
        sc.save_pkl(dest, params, Z, dict(pde='poisson2d', N=ntr + 1, k=K, r=Rb, cell='p-linear',
                                          tag=tag, layer='head', rank=Rb, sources=S, K=K,
                                          width=int(spec['width']), layers=int(spec['layers']),
                                          updates=int(spec['updates']), bank=bcfg.get('tag'),
                                          training_seconds=time.perf_counter() - t0))
        R_['checkpoints'].append(dict(id=tag, path=f'checkpoints/{tag}.pkl',
                                      sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                                      weights_sha256=K_.weights_sha(params), codes_sha256=L_.sha_array(Z)))
        params_count = int(sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params['h'])))
        R_['arms'].append(dict(arm=tag, K=K, width=int(spec['width']), layers=int(spec['layers']),
                               updates=int(spec['updates']), role=spec['role'], training=info,
                               head_parameter_count=params_count + K * Rb,
                               oracle_linear_solve=oracle_linear,
                               head_at_stored_codes_fit=K_.summarise(st),
                               best_found_validation=K_.summarise(bfv),
                               best_found_development=K_.summarise(bfd),
                               validation_exit_reasons=rsv.tolist(), development_exit_reasons=rsd.tolist(),
                               ratio_dev_best_found_over_floor=float(bfd.max() / floor_dev.max()),
                               training_seconds=time.perf_counter() - t0))
        trained[tag] = dict(params=params, codes=np.asarray(Z), K=K)
        print('HEAD', tag, 'fit', float(st.max()), 'val', float(bfv.max()), 'dev', float(bfd.max()),
              round(time.perf_counter() - t0, 1), flush=True)
        save()
    del G, Rg, Tfit, Tval, Tdev, Bt, Fmfit
    jax.clear_caches()

    # ---------------------------------------------------- the solve at 1024 ---
    n = int(cfg['solve_intervals'])
    nhi = cfg['reference_intervals'][-1]
    chain, same = {}, {}
    for case, param in enumerate(dev):
        fine = P.reference(param, nhi)
        chain[case] = np.array(fine[::nhi // n, ::nhi // n], copy=True)
        same[case] = P.reference(param, n)
        R_['references'].append(dict(case=case, fine_intervals=nhi, fine_sha256=L_.sha_array(fine),
                                     same_grid_sha256=L_.sha_array(same[case])))
    sources = [C.full_source(n, q) for q in dev]
    lam = jnp.asarray(C.eigenvalues(n))
    base = C.assemble(prim_params, prim_codes, n, cfg['requested_modes'], cfg['lm_budget'])
    assert base['info']['retained_bank_rank'] == Rb
    R_['arm_setup'].append(dict(intervals=n, arm='shared_assembly', **{
        k: base['info'][k] for k in ('retained_modes', 'stored_features', 'retained_bank_rank',
                                     'operator_sha256', 'bank_sha256')}))
    models = {'pbh02_primary_K32': dict(params=prim_params, codes=prim_codes, K=int(prim_codes.shape[1]))}
    models.update(trained)
    built = []
    for mid, L in models.items():
        head = lambda z, p=L['params']: sc.head(p, z)
        trust = PA.radius(L['codes'])
        linear = 'gj' if L['K'] <= cfg['gauss_jordan_max'] else 'lu'
        kern = PA.make_query(head, base['B'], base['bank'], n, trust, cfg['lm_budget'],
                             cfg['stationarity_tolerance'], linear)
        pred = jax.jit(jax.vmap(lambda z: base['B'] @ head(z)))(jnp.asarray(L['codes']))
        built.append(dict(name=f'a_neural@{mid}', model=mid, k=L['K'], kernel=kern, predictions=pred,
                          codes=jnp.asarray(L['codes']), B=base['B'], bank=base['bank'], ops=base))
        R_['arm_setup'].append(dict(intervals=n, arm=f'a_neural@{mid}', model=mid, K=L['K'], R=Rb,
                                    M=int(base['B'].shape[0]), trust_radius=trust, linear_solve=linear))
    save()

    # untimed three layers at the solve mesh
    Gn = base['bank']
    Rgn, rankn = K_.bank_r(Gn)
    assert rankn['rank_valid'], rankn
    U = jnp.stack([jnp.asarray(same[c][1:-1, 1:-1].ravel()) for c in range(len(dev))])
    T, perp2, nu2 = K_.project_targets(Gn, Rgn, U)
    floor = np.asarray(jnp.sqrt(perp2 / nu2))
    for mid, L in models.items():
        head = lambda z, p=L['params']: sc.head(p, z)
        e, it, rs = L_.oracle_projected(head, Rgn, L['codes'], T, perp2, nu2, np.zeros((Rb, 0)),
                                        cfg['recon_budget'], cfg['recon_starts'],
                                        cfg['stationarity_tolerance'],
                                        'gj' if L['K'] <= cfg['oracle_gauss_jordan_max'] else 'lu')
        R_['reconstruction'].append(dict(intervals=n, model=mid, K=L['K'], R=Rb, bank_rank=rankn,
                                         bank_projection=K_.summarise(floor), best_found=K_.summarise(e),
                                         iterations=it.tolist(), exit_reasons=rs.tolist()))
        print('RECON', n, mid, 'floor', float(floor.max()), 'best', float(e.max()), flush=True)
    del U, T, Rgn
    jax.clear_caches()
    save()

    subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
    subjects.append(dict(kind='fom', name='dst_direct'))
    R_['declared_subjects'] = [dict(intervals=n, **{k: v for k, v in s.items() if k != 'index'})
                               for s in subjects]
    order_rng = np.random.default_rng(cfg['order_seed'])

    def invoke(sub, source):
        if sub['kind'] == 'rom':
            b = built[sub['index']]
            return PA.query_once(b['kernel'], source, b['ops'], b['predictions'], b['codes'], b['B'], b['bank'])
        return C.fom_query(source, lam)

    t = time.perf_counter()
    for sub in subjects:
        invoke(sub, sources[0])
    R_['compile_warmup'] = dict(seconds=time.perf_counter() - t, subjects=len(subjects))
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
                b = built[sub['index']] if sub['kind'] == 'rom' else None
                R_['invocations'].append(dict(
                    intervals=n, case=case, rep=rep, kind=sub['kind'], name=sub['name'], field_sha256=h,
                    artifact=artifacts[key], physical_error=C.relative(field, chain[case]),
                    same_grid_error=C.relative(field, same[case]), source_sha256=C.sha(sources[case]),
                    model=(b['model'] if b else None), k=(b['k'] if b else None), **row))
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    for g in cfg['gates']:
        if n not in g['intervals']:
            continue
        ref = json.loads(resolve(here, g['file']).read_text())
        for ours, theirs in g['pairs']:
            entry = next((e for e in ref['arms'] if e['intervals'] == n and e['name'] == theirs), None)
            got = {}
            for x in R_['invocations']:
                if x['name'] == ours:
                    got.setdefault(x['case'], x['physical_error'])
            if entry is None or not got:
                R_['gates'].append(dict(intervals=n, ours=ours, theirs=theirs, passed=False, missing=True))
                continue
            worst = max(abs(got[c] - e) / max(abs(e), 1e-300)
                        for c, e in zip(entry['cases'], entry['physical_error']) if c in got)
            R_['gates'].append(dict(intervals=n, ours=ours, theirs=theirs, source=g['file'],
                                    reference_job=ref.get('job_id'), worst_relative_difference=worst,
                                    tolerance=cfg['fidelity_tolerance'],
                                    passed=bool(worst <= cfg['fidelity_tolerance'])))
            print('GATE', R_['gates'][-1], flush=True)
    save()
    for g in R_['gates']:
        assert g['passed'], g
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('P-LINEAR HEAD COMPLETE', flush=True)


if __name__ == '__main__':
    main()
