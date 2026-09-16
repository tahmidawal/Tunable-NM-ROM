"""p-bank-head job 2: the frozen solve, in the unchanged head-ablation machinery.

Every reduced arm goes through `poisson_ablation.make_query` / `query_once`
unmodified, so the incumbent control is literally the `pabl01` arm (a) code path
and the new checkpoints are measured by the same kernel, initializer policy,
stopping rule, test-mode family and output contract.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    models = json.loads(Path(a.models).read_text())
    ref = json.loads(Path(a.reference).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        # The fidelity gate compares against pabl01's physical errors, which are
        # taken against the restricted 2048-interval reference, so the reference
        # chain is NOT shortened here; only the mesh list and the arm count are.
        cfg.update(intervals=[64], repetitions=1, burn_seconds=0.001,
                   pod_cohorts=[c for c in cfg['pod_cohorts'] if c['id'] == 'control'])
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    R_ = dict(config=cfg, models=models, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke),
              reference_source=ref.get('source'), gates=[], arm_setup=[], references=[],
              reconstruction=[], pod=[], invocations=[], declared_subjects=[], complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)
    save()

    dev = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                          C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    R_['cohort'] = dict(parameters=dev.tolist(), sha256=K_.sha_array(dev),
                        groups=['existing development'] * cfg['eval_count']
                               + ['fresh development'] * cfg['fresh_count'])
    pod_cohorts = {}
    for spec in cfg['pod_cohorts']:
        draws = C.source_params(spec['seed'], spec['count'])
        assert not any(np.allclose(t, s) for t in draws for s in dev), \
            f"training/development overlap in POD cohort {spec['id']}"
        pod_cohorts[spec['id']] = draws
        R_.setdefault('pod_cohorts', []).append(
            dict(id=spec['id'], seed=spec['seed'], count=spec['count'],
                 parameters_sha256=K_.sha_array(draws)))
    save()

    loaded = {}
    for m in models:
        params, codes, ckcfg = sc.load_pkl(m['checkpoint'])
        basis = np.load(m['basis'])
        np.testing.assert_array_equal(np.asarray(codes), basis['training_latents'])
        loaded[m['id']] = dict(params=params, codes=np.asarray(codes), basis=basis,
                               K=int(np.asarray(codes).shape[1]),
                               R=int(np.asarray(params['h_lin']).shape[1]))
        R_.setdefault('checkpoints', []).append(dict(
            id=m['id'], role=m['role'], K=loaded[m['id']]['K'], R=loaded[m['id']]['R'],
            checkpoint_sha256=hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest(),
            basis_sha256=hashlib.sha256(Path(m['basis']).read_bytes()).hexdigest(),
            training_codes=int(len(codes)), checkpoint_config=ckcfg))
    save()

    # ------------------------------------------------------------ references ---
    nlo, nhi = cfg['reference_intervals'][0], cfg['reference_intervals'][-1]
    fine, same = {}, {}
    for case, param in enumerate(dev):
        t0 = time.perf_counter()
        fine[case] = P.reference(param, nhi)
        coarse = P.reference(param, nlo)
        R_['references'].append(dict(case=case, fine_intervals=nhi, coarse_intervals=nlo,
                                     reference_delta=C.relative(coarse, fine[case][::nhi // nlo,
                                                                                   ::nhi // nlo]),
                                     fine_sha256=K_.sha_array(fine[case]),
                                     seconds=time.perf_counter() - t0))
    save()
    print('REFERENCES done', round(time.perf_counter() - begin, 1), flush=True)

    order_rng = np.random.default_rng(cfg['order_seed'])
    for n in cfg['intervals']:
        print('MESH', n, flush=True)
        lam = jnp.asarray(C.eigenvalues(n))
        chain = {c: np.array(fine[c][::nhi // n, ::nhi // n], copy=True) for c in fine}
        for c, param in enumerate(dev):
            same[c] = P.reference(param, n)
        sources = [C.full_source(n, q) for q in dev]

        built, engines, keep = [], {}, []
        for m in models:
            mid = m['id']
            L = loaded[mid]
            ops = C.assemble(L['params'], L['codes'], n, cfg['requested_modes'], cfg['lm_budget'])
            keep.append(ops)
            M = int(ops['B'].shape[0])
            head = lambda z, p=L['params']: sc.head(p, z)
            trust = PA.radius(L['codes'])
            linear = 'gj' if L['K'] <= cfg['gauss_jordan_max'] else 'lu'
            kern = PA.make_query(head, ops['B'], ops['bank'], n, trust, cfg['lm_budget'],
                                 cfg['stationarity_tolerance'], linear)
            pred = jax.jit(jax.vmap(lambda z: ops['B'] @ head(z)))(jnp.asarray(L['codes']))
            built.append(dict(name=f'a_neural@{mid}', model=mid, k=L['K'], kernel=kern,
                              predictions=pred, codes=jnp.asarray(L['codes']),
                              B=ops['B'], bank=ops['bank'], ops=ops, linear_solve=linear,
                              family='neural'))
            R_['arm_setup'].append(dict(intervals=n, arm=f'a_neural@{mid}', model=mid,
                                        family='neural', k=L['K'], R=L['R'], M=M,
                                        trust_radius=trust, linear_solve=linear,
                                        **{kk: ops['info'][kk] for kk in
                                           ('retained_modes', 'stored_features',
                                            'retained_bank_rank', 'operator_sha256',
                                            'bank_sha256')}))
            assert ops['info']['retained_bank_rank'] == L['R'], \
                (mid, n, ops['info']['retained_bank_rank'], L['R'])
            count = int(min(cfg['correction_count'], L['basis']['coefficient_directions'].shape[1]))
            eng = CC.prepare_correction(ops, L['codes'], L['basis']['coefficient_directions'],
                                        count, cfg['retained'])
            assert eng['info']['linear_rank_valid'], (mid, n, eng['info'])
            engines[mid] = eng
            R_['arm_setup'].append(dict(intervals=n, arm=f'a_neural_q{count}@{mid}', model=mid,
                                        family='neural+linear', k=L['K'], R=L['R'], M=M,
                                        **{kk: eng['info'][kk] for kk in
                                           ('correction_count', 'nominal_latent_dimension',
                                            'nonlinear_optimizer_dimension', 'linear_rank',
                                            'linear_rank_valid', 'projected_operator_sha256')}))
            if M > L['R']:
                Zfree = np.asarray(jax.jit(jax.vmap(head))(jnp.asarray(L['codes'])))
                ftrust = PA.radius(Zfree)
                fk = PA.make_query(A.identity_head(), ops['B'], ops['bank'], n, ftrust,
                                   cfg['lm_budget'], cfg['stationarity_tolerance'], 'lu')
                fpred = jax.jit(jax.vmap(lambda z: ops['B'] @ z))(jnp.asarray(Zfree))
                built.append(dict(name=f'd_freebank@{mid}', model=mid, k=L['R'], kernel=fk,
                                  predictions=fpred, codes=jnp.asarray(Zfree), B=ops['B'],
                                  bank=ops['bank'], ops=ops, linear_solve='lu', family='free'))
                R_['arm_setup'].append(dict(intervals=n, arm=f'd_freebank@{mid}', model=mid,
                                            family='free', k=L['R'], R=L['R'], M=M,
                                            trust_radius=ftrust, linear_solve='lu'))
            save()

        # -------------------------------------------------------- POD-LSPG ----
        ops_sine = K_.sine_ops(n, cfg['requested_modes'])
        for pid, draws in pod_cohorts.items():
            modes, coords_full, pinfo = K_.streaming_pod(draws, n, max(cfg['pod_ranks']),
                                                         cfg['pod_block'])
            keep.append(modes)
            R_['pod'].append(dict(cohort=pid, **pinfo))
            for k in cfg['pod_ranks']:
                Vk = modes[:, :k]
                Bk = PA.reduce_bank(Vk, ops_sine['S'], ops_sine['I'], ops_sine['J'], n)
                Z = coords_full[:, :k]
                trust = PA.radius(Z)
                linear = 'gj' if k <= cfg['gauss_jordan_max'] else 'lu'
                kern = PA.make_query(A.identity_head(), Bk, Vk, n, trust, cfg['lm_budget'],
                                     cfg['stationarity_tolerance'], linear)
                pred = jax.jit(jax.vmap(lambda z: Bk @ z))(jnp.asarray(Z))
                name = f'e_pod{k}' if pid == 'control' else f'e_pod{k}@{pid}'
                built.append(dict(name=name, model=None, k=k, kernel=kern, predictions=pred,
                                  codes=jnp.asarray(Z), B=Bk, bank=Vk, ops=None,
                                  linear_solve=linear, family='pod', pod_cohort=pid))
                R_['arm_setup'].append(dict(intervals=n, arm=name, family='pod', k=k,
                                            pod_cohort=pid, M=int(Bk.shape[0]),
                                            trust_radius=trust, linear_solve=linear,
                                            candidates=int(len(Z)),
                                            operator_sha256=C.sha(np.asarray(Bk))))
            save()

        # --------------------------------- untimed three-layer reconstruction --
        for m in models:
            mid = m['id']
            L = loaded[mid]
            ops = next(b['ops'] for b in built if b['name'] == f'a_neural@{mid}')
            G = ops['bank']
            Rg, rank = K_.bank_r(G)
            U = jnp.stack([jnp.asarray(chain[c][1:-1, 1:-1].ravel()) for c in range(len(dev))])
            T, perp2, nu2 = K_.project_targets(G, Rg, U)
            floor = np.asarray(jnp.sqrt(perp2 / nu2))
            head = lambda z, p=L['params']: sc.head(p, z)
            recon = A.make_reconstruction(head, L['K'], n, cfg['recon_budget'],
                                          cfg['stationarity_tolerance'],
                                          'gj' if L['K'] <= cfg['gauss_jordan_max'] else 'lu')
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
            rec = dict(intervals=n, model=mid, K=L['K'], R=L['R'], bank_rank=rank,
                       bank_projection=K_.summarise(floor), best_found=K_.summarise(best),
                       best_found_iterations=iters)
            if L['R'] <= cfg['whiten_max_rank']:
                Qb, Rb = A.whiten(G)
                alt = np.asarray(jnp.linalg.norm(Qb @ (Qb.T @ U.T) - U.T, axis=0)
                                 / jnp.linalg.norm(U.T, axis=0))
                rec['bank_projection_dense_qr_check'] = float(np.max(np.abs(alt - floor)))
                del Qb, Rb
            R_['reconstruction'].append(rec)
            print('RECON', n, mid, 'floor', float(floor.max()), 'best', float(max(best)),
                  flush=True)
            del U, T, Rg
            save()

        # ------------------------------------------------------ timed queries --
        subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
        for m in models:
            count = int(min(cfg['correction_count'],
                            loaded[m['id']]['basis']['coefficient_directions'].shape[1]))
            subjects.append(dict(kind='retained', name=f'a_neural_q{count}@{m["id"]}',
                                 model=m['id']))
        subjects.append(dict(kind='fom', name='dst_direct'))
        R_['declared_subjects'] += [dict(intervals=n, **{k: v for k, v in s.items()
                                                         if k != 'index'}) for s in subjects]

        def invoke(sub, source):
            if sub['kind'] == 'rom':
                b = built[sub['index']]
                return PA.query_once(b['kernel'], source, b['ops'] or ops_sine, b['predictions'],
                                     b['codes'], b['B'], b['bank'])
            if sub['kind'] == 'retained':
                mid = sub['model']
                ops = next(x['ops'] for x in built if x['name'] == f'a_neural@{mid}')
                return CC.correction_query(source, ops, engines[mid], cfg['retained'])
            return C.fom_query(source, lam)

        t = time.perf_counter()
        for sub in subjects:
            invoke(sub, sources[0])
        R_.setdefault('compile_warmup', []).append(dict(intervals=n,
                                                        seconds=time.perf_counter() - t))
        save()

        artifacts = {}
        for rep in range(cfg['repetitions']):
            for case in range(len(dev)):
                for i in order_rng.permutation(len(subjects)):
                    sub = subjects[int(i)]
                    C.burn(cfg['burn_seconds'])
                    field, row = invoke(sub, sources[case])
                    assert np.isfinite(field).all()
                    h = K_.sha_array(field)
                    key = (sub['name'], case, h)
                    if key not in artifacts:
                        fn = f"n{n}_{sub['name'].replace('@', '-')}_case{case}_rep{rep}.npz"
                        np.savez_compressed(out / fn, field=field)
                        artifacts[key] = fn
                    b = built[sub['index']] if sub['kind'] == 'rom' else None
                    R_['invocations'].append(dict(
                        intervals=n, case=case, group=R_['cohort']['groups'][case], rep=rep,
                        kind=sub['kind'], name=sub['name'], field_sha256=h,
                        artifact=artifacts[key],
                        physical_error=C.relative(field, chain[case]),
                        same_grid_error=C.relative(field, same[case]),
                        source_sha256=C.sha(sources[case]), finite=True,
                        model=(b['model'] if b else sub.get('model')),
                        k=(b['k'] if b else (loaded[sub['model']]['K']
                                             if sub['kind'] == 'retained' else None)),
                        family=(b['family'] if b else
                                ('neural+linear' if sub['kind'] == 'retained' else 'fom')),
                        pod_cohort=(b.get('pod_cohort') if b else None),
                        linear_solve=(b['linear_solve'] if b else None), **row))
            print('TIMED', n, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()

        # ------------------------------------------------------ fidelity gates -
        gate = []
        for entry in ref['arms']:
            if entry['intervals'] != n:
                continue
            sel = [x for x in R_['invocations'] if x['intervals'] == n
                   and x['name'] == entry['name']]
            if not sel:
                continue
            got = {}
            for x in sel:
                got.setdefault(x['case'], x['physical_error'])
            worst = max(abs(got[c] - e) / max(e, 1e-300)
                        for c, e in zip(entry['cases'], entry['physical_error'])
                        if c in got)
            gate.append(dict(intervals=n, arm=entry['name'], worst_relative_difference=worst,
                             tolerance=cfg['fidelity_tolerance'],
                             passed=bool(worst <= cfg['fidelity_tolerance'])))
        R_['gates'] += gate
        for g in gate:
            print('GATE', g, flush=True)
        save()
        del built, engines, keep
        jax.clear_caches()

    for g in R_['gates']:
        assert g['passed'], g
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('P-BANK-HEAD SOLVE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
