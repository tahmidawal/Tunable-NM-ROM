"""lshape jobs 2-4: the frozen solve on one or more meshes, every reduced model and every
full-order comparator timed in the same process on the same GPU under one contract:
host (N+1)^2 nodal source array in, host (N+1)^2 nodal solution array out
(DESIGN.md sections 6-8).

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


def make_rom_kernel(head, geom, trust, budget, gtol, linear, q, free=False):
    """One complete reduced query. Every large array is an argument.

    `free` marks the q = R rung (DESIGN.md section 6, section A6): C = I, so B_perp = (I - QQ^T)B
    is zero to round-off and there is no nonlinear unknown left -- every coefficient is
    recovered by the exact elimination below. The LM is not run on that rung: on a zero
    operator its normalised gradient is a round-off cosine, so it would spin through its
    damping ladder to exit reason 3 on every query and inflate the cost of what is, by
    construction, a linear reduced model. Its stationarity is `full_stationarity`, the
    gradient of the FULL residual, which the kernel computes for every rung regardless."""
    N, nfull = geom.N, (geom.N + 1) ** 2
    lm = A.make_stationary_lm(lambda z, fp, Bp: Bp @ head(z) - fp, budget, trust, gtol, linear)

    @jax.jit
    def kernel(source_full, idx, P, B, Bp, Q, Rq, C, G, predictions, codes):
        f = source_full.reshape(-1)[idx]
        fm = P @ f
        fp = fm - Q @ (Q.T @ fm) if q else fm
        index = jnp.argmin(jnp.sum((predictions - fp[None, :]) ** 2, axis=1))
        if free:
            z, rn, it, reason, gn = codes[index], jnp.linalg.norm(fp), jnp.int32(0), jnp.int32(4), jnp.asarray(0.)
        else:
            z, rn, it, reason, gn = lm(codes[index], (fp, Bp), 0.)
        h = head(z)
        if q:
            y = jax.scipy.linalg.solve_triangular(Rq, Q.T @ (fm - B @ h), lower=False)
            coeff = h + C @ y
        else:
            y = jnp.zeros((0,))
            coeff = h
        u = G @ coeff
        field = jnp.zeros((nfull,)).at[idx].set(u).reshape(N + 1, N + 1)
        # full augmented stationarity: the y block is solved exactly, so its gradient is
        # zero and the full gradient reduces to the z block against the FULL residual
        residual = B @ coeff - fm
        J = B @ jax.jacfwd(head)(z)
        full_gn = jnp.linalg.norm(jnp.concatenate((J.T @ residual, (B @ C).T @ residual if q else jnp.zeros((0,))))) \
            / (jnp.sqrt(jnp.sum(J * J) + (jnp.sum((B @ C) ** 2) if q else 0.)) * jnp.linalg.norm(residual) + 1e-300)
        return field, z, rn, it, reason, gn, index, jnp.linalg.norm(fm), jnp.linalg.norm(residual), full_gn, y
    return kernel


def rom_query(subject, source_full):
    start = time.perf_counter()
    src = jax.device_put(source_full)
    src.block_until_ready()
    input_end = time.perf_counter()
    s = subject
    out = s['kernel'](src, s['idx'], s['P'], s['B'], s['Bp'], s['Q'], s['Rq'], s['C'], s['G'],
                      s['predictions'], s['codes'])
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    field, z, rn, it, reason, gn, index, fmn, full_rn, full_gn, y = jax.device_get(out)
    end = time.perf_counter()
    return np.asarray(field), dict(
        total_seconds=end - start, input_seconds=input_end - start,
        device_seconds=device_end - input_end, output_seconds=end - device_end,
        residual=float(rn), full_residual=float(full_rn), relative_full_residual=float(full_rn) / max(float(fmn), 1e-300),
        iterations=int(it), reason=int(reason), stationarity=float(gn), full_stationarity=float(full_gn),
        stationary=bool(int(reason) == 4), selected_code_index=int(index), latent=np.asarray(z).tolist(),
        correction_coefficients=np.asarray(y).tolist())


def fom_splu_query(fom, source_full):
    start = time.perf_counter()
    f = fom.geom.gather(source_full)
    t1 = time.perf_counter()
    u = fom.lu.solve(f)
    t2 = time.perf_counter()
    field = fom.geom.scatter(u)
    end = time.perf_counter()
    return field, dict(total_seconds=end - start, input_seconds=t1 - start, device_seconds=None,
                       solver_seconds=t2 - t1, output_seconds=end - t2, iterations=None, reason=0,
                       stationary=None, where='cpu')


def fom_pcg_query(fom, source_full, rtol):
    start = time.perf_counter()
    f = fom.geom.gather(source_full)
    t1 = time.perf_counter()
    u, it, info = fom.pcg_ic0(f, rtol)
    t2 = time.perf_counter()
    field = fom.geom.scatter(u)
    end = time.perf_counter()
    return field, dict(total_seconds=end - start, input_seconds=t1 - start, device_seconds=None,
                       solver_seconds=t2 - t1, output_seconds=end - t2, iterations=int(it), reason=int(info),
                       converged=bool(info == 0), stationary=None, rtol=rtol, where='cpu')


def fom_cg_gpu_query(cg, source_full, rtol):
    start = time.perf_counter()
    src = jax.device_put(source_full)
    src.block_until_ready()
    t1 = time.perf_counter()
    out = cg(src, rtol)
    jax.block_until_ready(out)
    t2 = time.perf_counter()
    field, it, res = jax.device_get(out)
    end = time.perf_counter()
    return np.asarray(field), dict(total_seconds=end - start, input_seconds=t1 - start, device_seconds=t2 - t1,
                                   solver_seconds=t2 - t1, output_seconds=end - t2, iterations=int(it),
                                   final_relative_residual=float(res), converged=bool(res <= rtol), reason=0,
                                   stationary=None, rtol=rtol, where='gpu')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--intervals', type=int, nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    base = Path(a.models).resolve().parent
    models = json.loads(Path(a.models).read_text())
    for m in models:
        for key in ('checkpoint', 'basis'):
            cand = base / m[key]
            m[key] = str(cand if cand.exists() else Path(m[key]))
    if a.intervals:
        cfg['intervals'] = a.intervals
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        cfg.update(intervals=[32, 64], repetitions=2, burn_seconds=0.001, requested_modes=16,
                   pod_ranks=[4, 8], pod_count=32, correction_ladder=[0, 4, 8], fine_intervals=128,
                   recon_budget=60, lm_budget=60)
        cfg['cohorts']['development'] = dict(seed=20260917, draw=16, count=8)
        cfg['cohorts']['training'] = dict(seed=0, draw=128, count=64)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    R_ = dict(config=cfg, models=models, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
              backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
              smoke=bool(a.smoke), gates=[], fom=[], weak_ops=[], references=[], arm_setup=[], reconstruction=[],
              pod=[], invocations=[], declared_subjects=[], complete=False,
              timing_contract='host (N+1)^2 nodal source array in to host (N+1)^2 nodal solution array out, '
                              'including gather/projection, initialisation, solve and decode/scatter; '
                              'identical for every subject')
    save = lambda: K_.dump(out / 'result.json', R_)
    save()

    cc = cfg['cohorts']
    dev, dinfo = K_.cohort(cc['development']['seed'], cc['development']['draw'], cc['development']['count'])
    train_draws, tinfo = K_.cohort(cc['training']['seed'], cc['training']['draw'], cc['training']['count'])
    assert not any(np.allclose(t, s) for t in train_draws for s in dev), 'training/development overlap'
    pod_draws = train_draws[:cfg['pod_count']]
    R_['cohort'] = dict(**dinfo, parameters=dev.tolist())
    R_['pod_cohort'] = dict(**tinfo, used=int(len(pod_draws)), used_sha256=K_.sha_array(pod_draws))
    save()

    loaded = {}
    for m in models:
        params, codes, ckcfg = sc.load_pkl(m['checkpoint'])
        basis = np.load(m['basis'])
        np.testing.assert_array_equal(np.asarray(codes), basis['training_latents'])
        loaded[m['id']] = dict(params=params, codes=np.asarray(codes), basis=basis, K=int(np.asarray(codes).shape[1]),
                               Rtot=int(np.asarray(params['h_lin']).shape[1]), factor=ckcfg['factor'],
                               n_enrich=int(ckcfg['n_enrich']), cfg=ckcfg)
        R_.setdefault('checkpoints', []).append(dict(
            id=m['id'], primary=m['primary'], K=loaded[m['id']]['K'], R_total=loaded[m['id']]['Rtot'],
            factor=ckcfg['factor'], n_enrich=int(ckcfg['n_enrich']), bank=ckcfg.get('bank'),
            checkpoint_sha256=hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest(),
            basis_sha256=hashlib.sha256(Path(m['basis']).read_bytes()).hexdigest(),
            training_codes=int(len(codes)), checkpoint_config=ckcfg))
    save()

    # --------------------------------------------------- fine reference -------
    nfine = cfg['fine_intervals']
    t0 = time.perf_counter()
    gfine = K_.Geometry(nfine, 'lshape')
    ffine = K_.FOM(gfine, build_ic0=False)
    fine = {}
    for case, q in enumerate(dev):
        u, resid, floor = ffine.reference(K_.source_interior(gfine, q))
        fine[case] = gfine.scatter(u)
        R_['references'].append(dict(case=case, fine_intervals=nfine, fine_residual=resid,
                                     fine_residual_roundoff_floor=floor,
                                     fine_sha256=K_.sha_array(fine[case])))
    R_['fine_reference'] = dict(intervals=nfine, interior_unknowns=gfine.n, factor_seconds=ffine.factor_seconds,
                                lu_nnz=ffine.lu_nnz, seconds=time.perf_counter() - t0)
    del ffine
    save()
    print('FINE REFERENCE done', round(time.perf_counter() - begin, 1), flush=True)

    order_rng = np.random.default_rng(cfg['order_seed'])
    for n in cfg['intervals']:
        print('MESH', n, flush=True)
        geom = K_.Geometry(n, 'lshape')
        fom = K_.FOM(geom, build_ic0=True)
        gate = fom.gates()
        ops = K_.weak_ops(fom, cfg['requested_modes'], cfg['mode_seed'])
        gate.update(intervals=n, lambda_min=ops['info']['lambda_min'], lambda1_reference=K_.LSHAPE_LAMBDA1,
                    lambda1_relative_difference=ops['info']['lambda1_relative_difference'],
                    positive_definite=bool(ops['info']['lambda_min'] > 0),
                    lambda1_within_1pct=bool(ops['info']['lambda1_relative_difference'] <= 1e-2) if n >= 128 else None)
        gate['passed'] = bool(gate['symmetric'] and gate['independent_assembly_agrees'] and gate['positive_definite']
                              and (gate['lambda1_within_1pct'] in (True, None)))
        R_['gates'].append(gate)
        print('GATE', gate, flush=True)
        assert gate['passed'], gate
        R_['fom'].append(dict(intervals=n, interior_unknowns=geom.n, nnz=int(fom.A.nnz),
                              assembly_seconds=fom.assembly_seconds, splu_factor_seconds=fom.factor_seconds,
                              splu_lu_nnz=fom.lu_nnz, ic0_factor_seconds=fom.ic0_seconds, ic0_nnz=fom.ic0_nnz,
                              ic0_pattern_residual=fom.ic0_pattern_residual))
        R_['weak_ops'].append(dict(intervals=n, **ops['info']))
        save()

        # references and sources
        same, sources, F_int = {}, [], []
        for case, q in enumerate(dev):
            u, resid, floor = fom.reference(K_.source_interior(geom, q))
            # G-FOM-4 (DESIGN A7): the residual is round-off-limited, and its floor grows
            # like ||A||_inf ~ N^2, so the gate is the larger of the declared absolute bound
            # and that measured floor. Both numbers are recorded per case.
            limit = max(cfg['reference_residual_limit'],
                        cfg['reference_residual_roundoff_factor'] * floor)
            assert resid <= limit, (n, case, resid, floor, limit)
            same[case] = geom.scatter(u)
            sources.append(K_.source_full(geom, q))
            F_int.append(geom.gather(sources[-1]))
            R_['references'].append(dict(case=case, intervals=n, same_grid_residual=resid,
                                         same_grid_residual_roundoff_floor=floor,
                                         same_grid_residual_limit=limit,
                                         same_sha256=K_.sha_array(same[case]),
                                         discretisation_delta_vs_fine=K_.relative(same[case], geom.restrict_from(fine[case], nfine))))
        chain = {c: geom.restrict_from(fine[c], nfine) for c in fine}
        Udev = np.stack([geom.gather(same[c]) for c in range(len(dev))])
        idx = jnp.asarray(geom.idx)
        save()

        built = []
        keep = []
        empty = dict(Q=jnp.zeros((cfg['requested_modes'], 0)), Rq=jnp.zeros((0, 0)))

        # ------------------------------------------------------ neural arms ---
        for m in models:
            mid = m['id']
            L = loaded[mid]
            features = K_.make_features(L['factor'], L['n_enrich'])
            G = K_.bank_of(features, L['params'], geom)
            Rg, rank = K_.bank_r(G)
            assert rank['rank_valid'], (mid, n, rank['rank'])
            B = K_.reduce_bank(ops, fom, G)
            head = K_.head_of(L['params'])
            trust = float(np.max(np.linalg.norm(L['codes'] - L['codes'].mean(0), axis=1)))
            linear = 'gj' if L['K'] <= cfg['gauss_jordan_max'] else 'lu'
            M = int(B.shape[0])
            ladder = list(cfg['correction_ladder']) if m['primary'] else [0]
            if m['primary'] and M > L['Rtot']:
                ladder.append(L['Rtot'])
            keep.append((G, B))
            for q in ladder:
                if q > 0 and q < L['Rtot']:
                    C = jnp.asarray(L['basis']['coefficient_directions'][:, :q])
                elif q == L['Rtot']:
                    C = jnp.eye(L['Rtot'])
                else:
                    C = jnp.zeros((L['Rtot'], 0))
                if q:
                    Lq = B @ C
                    Q, Rq = jnp.linalg.qr(Lq, mode='reduced')
                    values = np.asarray(jnp.linalg.svd(Rq, compute_uv=False))
                    lrank = int(np.count_nonzero(values > values[0] * max(Lq.shape) * np.finfo(float).eps))
                    Bp = B - Q @ (Q.T @ B)
                else:
                    Q, Rq, lrank, Bp = empty['Q'], empty['Rq'], 0, B
                assert lrank == q, (mid, n, q, lrank)
                kern = make_rom_kernel(head, geom, trust, cfg['lm_budget'], cfg['stationarity_tolerance'], linear, q,
                                       free=(q == L['Rtot']))
                pred = jax.jit(jax.vmap(lambda z, Bp: Bp @ head(z), in_axes=(0, None)))(jnp.asarray(L['codes']), Bp)
                name = f'neural_q{q}@{mid}' if q < L['Rtot'] else f'freebank@{mid}'
                built.append(dict(name=name, model=mid, k=L['K'], q=q, kernel=kern, predictions=pred,
                                  codes=jnp.asarray(L['codes']), idx=idx, P=ops['P'], B=B, Bp=Bp, Q=Q, Rq=Rq, C=C,
                                  G=G, family=('neural' if q == 0 else ('free' if q == L['Rtot'] else 'neural+linear')),
                                  linear_solve=linear, primary=m['primary']))
                R_['arm_setup'].append(dict(intervals=n, arm=name, model=mid, family=built[-1]['family'], k=L['K'],
                                            q=q, R_total=L['Rtot'], M=M, trust_radius=trust, linear_solve=linear,
                                            free_rung=bool(q == L['Rtot']), nonlinear_unknowns=0 if q == L['Rtot'] else L['K'],
                                            bank_rank=rank['rank'], bank_condition=rank['condition_number'],
                                            linear_rank=lrank, operator_sha256=K_.sha_array(B),
                                            projected_operator_sha256=K_.sha_array(Bp), bank_sha256=K_.sha_array(G)))
            # untimed three-layer decomposition
            T, perp2, nu2 = K_.project_targets(G, Rg, Udev)
            floor = np.asarray(jnp.sqrt(perp2 / nu2))
            bf, it, rs = K_.oracle_errors(head, Rg, L['codes'], T, perp2, nu2, cfg['recon_budget'], cfg['recon_starts'],
                                          cfg['stationarity_tolerance'], linear)
            R_['reconstruction'].append(dict(intervals=n, model=mid, K=L['K'], R_total=L['Rtot'], bank_rank=rank,
                                             bank_projection=K_.summarise(floor), best_found=K_.summarise(bf),
                                             best_found_iterations=it.tolist(), best_found_exit_reasons=rs.tolist()))
            print('RECON', n, mid, 'floor', float(floor.max()), 'best', float(bf.max()), flush=True)
            del Rg, T
            save()

        # ---------------------------------------------------------- POD-LSPG --
        t0 = time.perf_counter()
        Upod = K_.fields(fom, pod_draws)
        pod_field_seconds = time.perf_counter() - t0
        modes, coords_full, pinfo = K_.pod_from_host(Upod, max(cfg['pod_ranks']), cfg['pod_block'])
        del Upod
        pinfo.update(intervals=n, field_seconds=pod_field_seconds)
        R_['pod'].append(pinfo)
        keep.append(modes)
        for k in cfg['pod_ranks']:
            Vk = modes[:, :k]
            Bk = K_.reduce_bank(ops, fom, Vk)
            Z = coords_full[:, :k]
            trust = float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))
            linear = 'gj' if k <= cfg['gauss_jordan_max'] else 'lu'
            kern = make_rom_kernel(A.identity_head(), geom, trust, cfg['lm_budget'], cfg['stationarity_tolerance'], linear, 0)
            pred = jax.jit(jax.vmap(lambda z, Bk: Bk @ z, in_axes=(0, None)))(jnp.asarray(Z), Bk)
            Ck = jnp.zeros((k, 0))
            built.append(dict(name=f'pod{k}', model=None, k=k, q=0, kernel=kern, predictions=pred, codes=jnp.asarray(Z),
                              idx=idx, P=ops['P'], B=Bk, Bp=Bk, Q=empty['Q'], Rq=empty['Rq'], C=Ck, G=Vk, family='pod',
                              linear_solve=linear, primary=False))
            proj = np.asarray(jnp.linalg.norm(jnp.asarray(Udev).T - Vk @ (Vk.T @ jnp.asarray(Udev).T), axis=0)
                              / jnp.linalg.norm(jnp.asarray(Udev).T, axis=0))
            R_['arm_setup'].append(dict(intervals=n, arm=f'pod{k}', family='pod', k=k, q=0, M=int(Bk.shape[0]),
                                        trust_radius=trust, linear_solve=linear, candidates=int(len(Z)),
                                        operator_sha256=K_.sha_array(Bk), projection_floor=K_.summarise(proj)))
        save()

        # --------------------------------------------------------- subjects ---
        cg = K_.make_gpu_cg(geom, cfg['cg_maxiter'])
        subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
        subjects.append(dict(kind='splu', name='fom_splu'))
        for rtol in cfg['cg_gpu_rtols']:
            subjects.append(dict(kind='cg_gpu', name=f'fom_cg_gpu_r{rtol:g}', rtol=rtol))
        for rtol in cfg['pcg_ic0_rtols']:
            subjects.append(dict(kind='pcg_ic0', name=f'fom_pcg_ic0_cpu_r{rtol:g}', rtol=rtol))
        R_['declared_subjects'] += [dict(intervals=n, **{k: v for k, v in s.items() if k != 'index'}) for s in subjects]

        def invoke(sub, source):
            if sub['kind'] == 'rom':
                return rom_query(built[sub['index']], source)
            if sub['kind'] == 'splu':
                return fom_splu_query(fom, source)
            if sub['kind'] == 'cg_gpu':
                return fom_cg_gpu_query(cg, source, sub['rtol'])
            return fom_pcg_query(fom, source, sub['rtol'])

        t = time.perf_counter()
        for sub in subjects:
            invoke(sub, sources[0])
        R_.setdefault('compile_warmup', []).append(dict(intervals=n, seconds=time.perf_counter() - t))
        save()

        artifacts = {}
        for rep in range(cfg['repetitions']):
            for case in range(len(dev)):
                for i in order_rng.permutation(len(subjects)):
                    sub = subjects[int(i)]
                    K_.burn(cfg['burn_seconds'])
                    field, row = invoke(sub, sources[case])
                    assert np.isfinite(field).all()
                    h = K_.sha_array(field)
                    key = (sub['name'], case, h)
                    if key not in artifacts:
                        fn = f"n{n}_{sub['name'].replace('@', '-')}_case{case}.npz"
                        np.savez_compressed(out / fn, interior=geom.gather(field))
                        artifacts[key] = fn
                    b = built[sub['index']] if sub['kind'] == 'rom' else None
                    R_['invocations'].append(dict(
                        intervals=n, case=case, rep=rep, kind=sub['kind'], name=sub['name'], field_sha256=h,
                        artifact=artifacts[key], same_grid_error=K_.relative(field, same[case]),
                        physical_error=K_.relative(field, chain[case]), source_sha256=K_.sha_array(sources[case]),
                        model=(b['model'] if b else None), k=(b['k'] if b else None), q=(b['q'] if b else None),
                        family=(b['family'] if b else 'fom'), primary=(b['primary'] if b else None),
                        linear_solve=(b['linear_solve'] if b else None), **row))
            print('TIMED', n, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()

        # ---------------------------------------------------- solver gate G5 --
        tight = [x for x in R_['invocations'] if x['intervals'] == n and x['kind'] in ('cg_gpu', 'pcg_ic0')
                 and x['rtol'] == min(cfg['cg_gpu_rtols'] if x['kind'] == 'cg_gpu' else cfg['pcg_ic0_rtols'])]
        worst = max(x['same_grid_error'] for x in tight)
        g5 = dict(intervals=n, gate='G-FOM-5 tight iterative solves agree with the direct reference',
                  worst_relative_difference=worst, tolerance=cfg['solver_agreement_limit'],
                  passed=bool(worst <= cfg['solver_agreement_limit']))
        R_['gates'].append(g5)
        print('GATE', g5, flush=True)
        assert g5['passed'], g5
        save()
        del built, keep, cg, fom
        jax.clear_caches()

    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('LSHAPE SOLVE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
