"""hires-poisson, L-shaped domain: the frozen lshape checkpoints (SDF boundary factor) on ONE
fine mesh per job, every comparator timed in the same allocation.

Adapted from `worktrees/2026-09-17-lshape/experiments/lshape/lsh_solve.py` @ d80fed7a. The
ROM kernel (`make_rom_kernel`), the SuperLU direct solve, the GPU CG and the IC(0)-PCG are
that file's, imported/copied unchanged; `lsh_core.py` is imported verbatim. New here:
a lean ROM variant (in-kernel full-gradient diagnostic removed, parity-gated), coarse-grid
controls, 5 repetitions on 12 development sources, physical-GPU UUID guard, no POD
(3072 sparse solves at 2048^2 are out of budget; stated omission), and the tight CG solve
of gate G-FOM-5 run once per source outside the timed loop.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import sep_common as sc
import arms as A
import lsh_core as K_


def gpu_uuid():
    cuda = ctypes.CDLL('libcuda.so.1')
    assert cuda.cuInit(0) == 0
    dev = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
    raw = (ctypes.c_ubyte * 16)()
    fn = getattr(cuda, 'cuDeviceGetUuid_v2', cuda.cuDeviceGetUuid)
    assert fn(ctypes.byref(raw), dev) == 0
    return 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))


def make_rom_kernel(head, geom, trust, budget, gtol, linear, q, free=False, lean=False):
    """`lsh_solve.make_rom_kernel` verbatim; `lean=True` drops the in-kernel full-gradient
    diagnostic (jacfwd + two dense products) from the timed kernel and nothing else."""
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
        if lean:
            return field, z, rn, it, reason, gn, index, jnp.linalg.norm(fm), jnp.asarray(0.), jnp.asarray(0.), y
        residual = B @ coeff - fm
        J = B @ jax.jacfwd(head)(z)
        full_gn = jnp.linalg.norm(jnp.concatenate((J.T @ residual, (B @ C).T @ residual if q else jnp.zeros((0,))))) \
            / (jnp.sqrt(jnp.sum(J * J) + (jnp.sum((B @ C) ** 2) if q else 0.)) * jnp.linalg.norm(residual) + 1e-300)
        return field, z, rn, it, reason, gn, index, jnp.linalg.norm(fm), jnp.linalg.norm(residual), full_gn, y
    return kernel


def rom_query(s, source_full):
    start = time.perf_counter()
    src = jax.device_put(source_full)
    src.block_until_ready()
    input_end = time.perf_counter()
    out = s['kernel'](src, s['idx'], s['P'], s['B'], s['Bp'], s['Q'], s['Rq'], s['C'], s['G'],
                      s['predictions'], s['codes'])
    jax.block_until_ready(out)
    device_end = time.perf_counter()
    field, z, rn, it, reason, gn, index, fmn, full_rn, full_gn, y = jax.device_get(out)
    end = time.perf_counter()
    return np.asarray(field), dict(
        total_seconds=end - start, input_seconds=input_end - start,
        fused_device_seconds=device_end - input_end, output_seconds=end - device_end,
        residual=float(rn), iterations=int(it), reason=int(reason), stationarity=float(gn),
        full_stationarity=float(full_gn), stationary=bool(int(reason) == 4),
        selected_code_index=int(index))


def device_query(fn, source_full):
    start = time.perf_counter()
    src = jax.device_put(source_full)
    src.block_until_ready()
    t1 = time.perf_counter()
    out = fn(src)
    jax.block_until_ready(out)
    t2 = time.perf_counter()
    field, it, res = jax.device_get(out)
    end = time.perf_counter()
    return np.asarray(field), dict(total_seconds=end - start, input_seconds=t1 - start,
                                   fused_device_seconds=t2 - t1, output_seconds=end - t2,
                                   iterations=int(it), final_relative_residual=float(res), where='gpu')


def host_up(u, n):
    """Bilinear interpolation of a full nodal (nc+1)^2 array to (n+1)^2 on the host."""
    nc = u.shape[0] - 1
    old, new = np.linspace(0., 1., nc + 1), np.linspace(0., 1., n + 1)
    a = np.stack([np.interp(new, old, u[:, j]) for j in range(nc + 1)], axis=1)
    return np.stack([np.interp(new, old, a[i]) for i in range(n + 1)], axis=0)


def device_up(u, n):
    nc = u.shape[0] - 1
    old, new = jnp.linspace(0., 1., nc + 1), jnp.linspace(0., 1., n + 1)
    a = jax.vmap(lambda col: jnp.interp(new, old, col), in_axes=1, out_axes=1)(u)
    return jax.vmap(lambda row: jnp.interp(new, old, row))(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    if a.smoke:
        cfg.update(intervals=64, repetitions=1, burn_seconds=0.001, requested_modes=200, case_count=2,
                   coarse_intervals=[16, 32], lm_budget=60)
    n = int(cfg['intervals'])
    nfine = 2 * n
    assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    uuid0 = gpu_uuid()
    inventory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'], text=True)
    assert uuid0 in inventory
    models = json.loads((here / 'models.json').read_text())
    R_ = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
              backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0,
              nvidia_smi=inventory, x64=True, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke), intervals=n, fine_intervals=nfine,
              timing_contract='host (N+1)^2 f64 nodal source in to host (N+1)^2 f64 nodal solution out for every subject',
              gates=[], arm_setup=[], invocations=[], parity=[], references=[], complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)
    cc = cfg['cohorts']
    dev, dinfo = K_.cohort(cc['development']['seed'], cc['development']['draw'], cc['development']['count'])
    dev = dev[:cfg['case_count']]
    train_draws, _ = K_.cohort(cc['training']['seed'], cc['training']['draw'], cc['training']['count'])
    assert not any(np.allclose(t, s) for t in train_draws for s in dev)
    R_['cohort'] = dict(**dinfo, used=int(len(dev)), parameters=dev.tolist(),
                        note='first case_count sources of the lshape lane development cohort')
    save()

    # ------------------------------------------------ fine reference (2n) ------
    t0 = time.perf_counter()
    gfine = K_.Geometry(nfine, 'lshape')
    mode = cfg.get('fine_reference', 'splu')
    chain = {}
    if mode == 'splu':
        ffine = K_.FOM(gfine, build_ic0=False)
        fine_info = dict(solver='SuperLU + 2 refinements', factor_seconds=ffine.factor_seconds, lu_nnz=ffine.lu_nnz)
    else:
        # DESIGN A5: SuperLU's 32-bit fill count overflows near 12.6M unknowns (2n = 4096), so the
        # 2n reference is a matrix-free GPU CG solve to a tight relative residual instead; the
        # audit verifies it by its own stencil residual like any other reference.
        ffine = None
        cgf = K_.make_gpu_cg(gfine, cfg['cg_maxiter'])
        fine_info = dict(solver=f"matrix-free GPU CG, rtol {cfg['fine_cg_rtol']}")
    for case, q in enumerate(dev):
        if ffine is not None:
            u, resid, floor = ffine.reference(K_.source_interior(gfine, q))
            assert resid <= max(cfg['reference_residual_limit'], floor), (case, resid, floor)
            full = gfine.scatter(u)
            extra = dict(fine_residual=resid, fine_roundoff_floor=floor)
        else:
            x, it, res = jax.device_get(cgf(jnp.asarray(K_.source_full(gfine, q)), cfg['fine_cg_rtol']))
            assert float(res) <= cfg['fine_cg_rtol'], (case, float(res))
            full = np.asarray(x)
            extra = dict(fine_residual=float(res), fine_cg_iterations=int(it))
        np.save(out / 'fields' / f'reference_fine_full_case{case}.npy', full)
        chain[case] = full[::2, ::2].copy()
        del full
        np.save(out / 'fields' / f'reference_fine_case{case}.npy', chain[case])
        R_['references'].append(dict(case=case, **extra))
    R_['fine_reference'] = dict(intervals=nfine, unknowns=gfine.n, seconds=time.perf_counter() - t0, **fine_info)
    del ffine, gfine
    jax.clear_caches()
    save()
    print('FINE REFERENCE', round(time.perf_counter() - begin, 1), flush=True)

    geom = K_.Geometry(n, 'lshape')
    fom = K_.FOM(geom, build_ic0=True)
    gate = fom.gates()
    ops = K_.weak_ops(fom, cfg['requested_modes'], cfg['mode_seed'])
    gate.update(gate='G-FOM-1/2/3', lambda1_relative_difference=ops['info']['lambda1_relative_difference'],
                positive_definite=bool(ops['info']['lambda_min'] > 0))
    gate['passed'] = bool(gate['symmetric'] and gate['independent_assembly_agrees'] and gate['positive_definite']
                          and (n < 128 or gate['lambda1_relative_difference'] <= 1e-2))
    R_['gates'].append(gate)
    assert gate['passed'], gate
    R_['fom'] = dict(unknowns=geom.n, splu_factor_seconds=fom.factor_seconds, splu_lu_nnz=fom.lu_nnz,
                     ic0_factor_seconds=fom.ic0_seconds, weak_modes_seconds=ops['info']['seconds'],
                     eigen_residual=ops['info']['eigen_residual'])
    save()
    print('FOM + MODES', round(time.perf_counter() - begin, 1), flush=True)

    same, sources = {}, []
    for case, q in enumerate(dev):
        u, resid, floor = fom.reference(K_.source_interior(geom, q))
        assert resid <= max(cfg['reference_residual_limit'], floor), (case, resid, floor)
        same[case] = geom.scatter(u)
        np.save(out / 'fields' / f'reference_same_case{case}.npy', same[case])
        sources.append(K_.source_full(geom, q))
        R_['references'][case].update(same_residual=resid, same_roundoff_floor=floor,
                                      discretisation_delta=K_.relative(same[case], chain[case]))
    Udev = np.stack([geom.gather(same[c]) for c in range(len(dev))])
    idx = jnp.asarray(geom.idx)

    subjects = []
    M = int(cfg['requested_modes'])
    empty = dict(Q=jnp.zeros((M, 0)), Rq=jnp.zeros((0, 0)))
    for m in models:
        params, codes, ckcfg = sc.load_pkl(here / m['checkpoint'])
        assert hashlib.sha256((here / m['checkpoint']).read_bytes()).hexdigest() == m['checkpoint_sha256']
        basis = np.load(here / m['basis'])
        codes = np.asarray(codes)
        np.testing.assert_array_equal(codes, basis['training_latents'])
        Rtot, Kc = int(np.asarray(params['h_lin']).shape[1]), int(codes.shape[1])
        features = K_.make_features(ckcfg['factor'], int(ckcfg['n_enrich']))
        G = K_.bank_of(features, params, geom)
        Rg, rank = K_.bank_r(G)
        assert rank['rank_valid'], rank
        T, perp2, nu2 = K_.project_targets(G, Rg, Udev)
        floor = np.asarray(jnp.sqrt(perp2 / nu2))
        del Rg, T
        B = K_.reduce_bank(ops, fom, G)
        head = K_.head_of(params)
        trust = float(np.max(np.linalg.norm(codes - codes.mean(0), axis=1)))
        linear = 'gj' if Kc <= cfg['gauss_jordan_max'] else 'lu'
        for q in cfg['correction_ladder']:
            assert M > Kc + q, (M, Kc, q)
            C = jnp.asarray(basis['coefficient_directions'][:, :q]) if q else jnp.zeros((Rtot, 0))
            if q:
                Q, Rq = jnp.linalg.qr(B @ C, mode='reduced')
                values = np.asarray(jnp.linalg.svd(Rq, compute_uv=False))
                assert int(np.count_nonzero(values > values[0] * M * np.finfo(float).eps)) == q
                Bp = B - Q @ (Q.T @ B)
            else:
                Q, Rq, Bp = empty['Q'], empty['Rq'], B
            pred = jax.jit(jax.vmap(lambda z, Bp: Bp @ head(z), in_axes=(0, None)))(jnp.asarray(codes), Bp)
            for variant, lean in (('retained', False), ('lean', True)):
                kern = make_rom_kernel(head, geom, trust, cfg['lm_budget'], cfg['stationarity_tolerance'],
                                       linear, q, lean=lean)
                subjects.append(dict(kind='rom', name=f"rom_q{q}_{variant}@{m['id']}", family='nm-rom', q=q,
                                     model=m['id'], variant=variant, kernel=kern, predictions=pred,
                                     codes=jnp.asarray(codes), idx=idx, P=ops['P'], B=B, Bp=Bp, Q=Q, Rq=Rq,
                                     C=C, G=G))
        R_['arm_setup'].append(dict(model=m['id'], K=Kc, R_total=Rtot, M=M, bank_rank=rank['rank'],
                                    bank_condition=rank['condition_number'],
                                    bank_floor=K_.summarise(floor), bank_sha256=K_.sha_array(G)))
        print('MODEL', m['id'], 'floor worst', float(floor.max()), round(time.perf_counter() - begin, 1), flush=True)
        save()

    cg = K_.make_gpu_cg(geom, cfg['cg_maxiter'])
    subjects.append(dict(kind='splu', name='fom_splu_cpu', family='direct-control'))
    for rtol in cfg['cg_gpu_rtols']:
        subjects.append(dict(kind='cg', name=f'cg_{rtol:g}', family='cg', tolerance=rtol,
                             fn=lambda s, rtol=rtol: cg(s, rtol)))
    for rtol in cfg['pcg_ic0_rtols']:
        subjects.append(dict(kind='pcg', name=f'pcg_ic0_cpu_{rtol:g}', family='pcg-cpu-control', tolerance=rtol))
    fmask = jnp.asarray(geom.mask)
    for nc in cfg['coarse_intervals']:
        s_ = n // nc
        assert n % nc == 0 and nc % 2 == 0
        gc = K_.Geometry(nc, 'lshape')
        fc = K_.FOM(gc, build_ic0=False)
        subjects.append(dict(kind='coarse_splu', name=f'coarse{nc}_splu_cpu', family='coarse-direct',
                             coarse_intervals=nc, geom=gc, fom=fc, stride=s_))
        ccg = K_.make_gpu_cg(gc, cfg['cg_maxiter'])
        for rtol in cfg['coarse_cg_rtols']:
            def run(src, ccg=ccg, s_=s_, rtol=rtol):
                x, it, res = ccg(src[::s_, ::s_], rtol)
                return jnp.where(fmask, device_up(x, n), 0.0), it, res
            subjects.append(dict(kind='cg', name=f'coarse{nc}_cg_{rtol:g}', family='coarse-cg',
                                 coarse_intervals=nc, tolerance=rtol, fn=jax.jit(run)))
    R_['declared_subjects'] = [s['name'] for s in subjects]

    def invoke(sub, source):
        if sub['kind'] == 'rom':
            return rom_query(sub, source)
        if sub['kind'] == 'cg':
            field, row = device_query(sub['fn'], source)
            row['converged'] = bool(row['final_relative_residual'] <= sub['tolerance'])
            return field, row
        start = time.perf_counter()
        if sub['kind'] == 'splu':
            f = geom.gather(source); t1 = time.perf_counter()
            u = fom.lu.solve(f); t2 = time.perf_counter()
            field = geom.scatter(u); extra = dict(where='cpu')
        elif sub['kind'] == 'pcg':
            f = geom.gather(source); t1 = time.perf_counter()
            u, it, info = fom.pcg_ic0(f, sub['tolerance']); t2 = time.perf_counter()
            field = geom.scatter(u); extra = dict(where='cpu', iterations=int(it), converged=bool(info == 0))
        else:
            gc, s_ = sub['geom'], sub['stride']
            f = gc.gather(source[::s_, ::s_]); t1 = time.perf_counter()
            u = sub['fom'].lu.solve(f)
            field = np.where(geom.mask, host_up(gc.scatter(u), n), 0.0); t2 = time.perf_counter()
            extra = dict(where='cpu')
        end = time.perf_counter()
        return field, dict(total_seconds=end - start, input_seconds=t1 - start, fused_device_seconds=t2 - t1,
                           output_seconds=end - t2, **extra)

    t = time.perf_counter()
    for sub in subjects:
        invoke(sub, sources[0])
    print('WARMUP', len(subjects), round(time.perf_counter() - t, 1), flush=True)
    order = np.random.default_rng(cfg['order_seed'])
    saved = {}
    for rep in range(cfg['repetitions']):
        for case in range(len(dev)):
            for i in order.permutation(len(subjects)):
                sub = subjects[int(i)]
                if case >= cfg.get('slow_subject_cases', {}).get(sub['name'], len(dev)):
                    continue
                K_.burn(cfg['burn_seconds'])
                assert gpu_uuid() == uuid0
                field, row = invoke(sub, sources[case])
                assert gpu_uuid() == uuid0
                assert np.isfinite(field).all(), sub['name']
                h = K_.sha_array(field)
                key = (sub['name'], case)
                if key not in saved:
                    fname = f"{sub['name'].replace('@', '-')}_case{case}.npy"
                    np.save(out / 'fields' / fname, field)
                    saved[key] = (fname, h)
                R_['invocations'].append(dict(
                    case=case, rep=rep, name=sub['name'], family=sub['family'], q=sub.get('q'),
                    model=sub.get('model'), variant=sub.get('variant'), tolerance=sub.get('tolerance'),
                    coarse_intervals=sub.get('coarse_intervals'), field_sha256=h, saved_field=saved[key][0],
                    matches_saved_field=bool(h == saved[key][1]),
                    same_grid_error=K_.relative(field, same[case]),
                    physical_error=K_.relative(field, chain[case]), **row))
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # G-FOM-5, once per source, untimed: a tight CG solve must reproduce the direct reference
    worst = 0.0
    for case in range(len(dev)):
        x, it, res = jax.device_get(cg(jnp.asarray(sources[case]), cfg['tight_cg_rtol']))
        worst = max(worst, K_.relative(np.asarray(x), same[case]))
    g5 = dict(gate='G-FOM-5 tight CG agrees with the sparse-direct reference', rtol=cfg['tight_cg_rtol'],
              worst_relative_difference=worst, tolerance=cfg['solver_agreement_limit'],
              passed=bool(worst <= cfg['solver_agreement_limit']))
    R_['gates'].append(g5)
    print('GATE', g5, flush=True)

    first = {}
    for x in R_['invocations']:
        first.setdefault((x['name'], x['case']), x)
    for m in models:
        for q in cfg['correction_ladder']:
            worst, ints = 0.0, True
            for case in range(len(dev)):
                xa, xb = first[(f"rom_q{q}_lean@{m['id']}", case)], first[(f"rom_q{q}_retained@{m['id']}", case)]
                worst = max(worst, K_.relative(np.load(out / 'fields' / xa['saved_field']),
                                               np.load(out / 'fields' / xb['saved_field'])))
                ints = ints and all(xa[k] == xb[k] for k in ('iterations', 'reason', 'selected_code_index'))
            R_['parity'].append(dict(candidate=f"rom_q{q}_lean@{m['id']}", baseline=f"rom_q{q}_retained@{m['id']}",
                                     worst_field_relative=worst, integers_identical=ints, limit=1e-12,
                                     passed=bool(worst <= 1e-12 and ints)))
            print('PARITY', R_['parity'][-1], flush=True)
    rom_rows = [x for x in R_['invocations'] if x['family'] == 'nm-rom']
    gates = dict(operator=all(g['passed'] for g in R_['gates']),
                 parity=bool(R_['parity']) and all(p['passed'] for p in R_['parity']),
                 rom_stationary=all(x['stationary'] for x in rom_rows),
                 retained_full_gradient=all(x['full_stationarity'] <= cfg['stationarity_tolerance']
                                            for x in rom_rows if x['variant'] == 'retained'),
                 cg_converged=all(x['converged'] for x in R_['invocations'] if 'converged' in x),
                 deterministic_gpu=all(x['matches_saved_field'] for x in R_['invocations']
                                       if x.get('where') != 'cpu'))
    R_['driver_gates'] = gates
    print('GATES', gates, flush=True)
    stats = jax.devices()[0].memory_stats() or {}
    R_['device_memory'] = {k: int(v) for k, v in stats.items() if k in ('peak_bytes_in_use', 'bytes_limit')}
    R_['device_guard_final_uuid'] = gpu_uuid()
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / ('COMPLETE' if all(gates.values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(json.dumps(gates) + '\n')
    print('HIRES LSHAPE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
