"""hires-poisson: frozen-weight mesh transfer of the R=512 / K=32 Poisson 2D checkpoint to
large square meshes, with every comparator timed in the same allocation.

Subjects at one mesh, on the twelve opened development sources:

* NM-ROM correction ladder q in cfg['ladder_q'] (m4 test rule), lean chunked kernels in
  f64 and (optionally) f32 decode; the retained single-bank `correction_core` kernel as
  the unoptimised parity/timing baseline where it fits;
* the q = R linear reduced model solved directly;
* named FOM: unpreconditioned CG (zero start, f64) at several relative-residual tolerances;
* labelled control: direct DST solve;
* coarse-grid controls: DST and CG on coarser grids + bilinear interpolation.

Errors are against (a) the same-grid FD truth and (b) a 2x-refined FD reference, both
from SciPy on the host (independent of every timed code path).
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
from scipy.fft import dstn
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import correction_core as CC
import iterative_core as IC
import pbh_core as K_
import plin_core as L_
import sep_common as sc
import hp_core as H


def reference(param, n, workers=8):
    F = C.full_source(n, param)
    c = dstn(F[1:-1, 1:-1], type=1, norm='ortho', workers=workers)
    return np.pad(dstn(c / C.eigenvalues(n), type=1, norm='ortho', workers=workers), 1)


def gpu_uuid():
    cuda = ctypes.CDLL('libcuda.so.1')
    assert cuda.cuInit(0) == 0
    dev = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
    raw = (ctypes.c_ubyte * 16)()
    fn = getattr(cuda, 'cuDeviceGetUuid_v2', cuda.cuDeviceGetUuid)
    assert fn(ctypes.byref(raw), dev) == 0
    return 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))


def training_cohort(spec):
    draws = C.source_params(spec['seed'], spec['draw_count'])
    fit, _ = K_.fit_validation_split(len(draws), spec['fit_split']['split_seed'],
                                     spec['fit_split']['validation_fraction'])
    return draws[fit]


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
        cfg.update(intervals=64, repetitions=1, burn_seconds=0.001, cg_maxiter=4000,
                   attempt='smoke', extension_sources=640, rows_per_chunk=16, eval_rows=8,
                   coarse_intervals=[16, 32], ladder_q=[0, 32, 256], eval_count=2,
                   fresh_count=1, cg_tolerances=[1e-2, 1e-6], retained_baseline=True,
                   keep_f64=True, keep_f32=True)
    n = int(cfg['intervals'])
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    assert len(jax.devices()) == 1
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    uuid0 = gpu_uuid()
    inventory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'], text=True)
    assert uuid0 in inventory, (uuid0, inventory)

    R_ = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0, nvidia_smi=inventory, x64=True,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke), intervals=n,
              timing_contract=('host f64 source array in to host f64 dense nodal field out for '
                               'EVERY subject; input copy, all device work and output copy '
                               'inside one synchronised interval; randomised subject order; '
                               'GPU burn-in before every invocation; diagnostics of the lean '
                               'ROM kernels are computed untimed after the query, the retained '
                               'baseline kernel carries them in the timed region as its '
                               'parents did'),
              arm_setup=[], invocations=[], parity=[], diagnostics=[], complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)
    save()

    # ---------------------------------------------------------------- cohort --
    dev = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                          C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    R_['cohort'] = dict(parameters=dev.tolist(), sha256=H.sha_array(dev),
                        note='the twelve OPENED development sources of p-linear / p-bank-head')
    m = cfg['model']
    params, codes, ckcfg = sc.load_pkl(here / Path(m['checkpoint']).name)
    basis = dict(np.load(here / Path(m['basis']).name))
    np.testing.assert_array_equal(np.asarray(codes), basis['training_latents'])
    train = training_cohort(m['training'])
    assert len(train) == len(codes)
    assert not any(np.allclose(t, s) for t in train for s in dev), 'training overlaps dev'
    codes = np.asarray(codes)
    K, Rw = int(codes.shape[1]), int(np.asarray(params['h_lin']).shape[1])
    R_['checkpoint'] = dict(id=m['id'], K=K, R=Rw,
                            checkpoint_sha256=hashlib.sha256(
                                (here / Path(m['checkpoint']).name).read_bytes()).hexdigest(),
                            basis_sha256=hashlib.sha256(
                                (here / Path(m['basis']).name).read_bytes()).hexdigest(),
                            training_codes=int(len(codes)))

    # ------------------------------------------------------------ references --
    same, fine_sub, refs = [], [], []
    for case, param in enumerate(dev):
        t0 = time.perf_counter()
        s_ = reference(param, n)
        f_ = reference(param, 2 * n)[::2, ::2].copy()
        same.append(s_)
        fine_sub.append(f_)
        refs.append(dict(case=case, discretisation_error_same_vs_fine=C.relative(s_, f_),
                         same_sha256=H.sha_array(s_), fine_subsampled_sha256=H.sha_array(f_),
                         seconds=time.perf_counter() - t0))
    R_['references'] = refs
    sources = [C.full_source(n, q) for q in dev]
    save()
    print('REFERENCES', round(time.perf_counter() - begin, 1), flush=True)

    # ----------------------------------------------------- directions (untimed) --
    Cfull, dinfo = L_.extend_basis(params, codes, basis, train, cfg['training_intervals'],
                                   subset=cfg.get('extension_sources'))
    assert dinfo['retained_prefix_exact'] and dinfo['extension_orthonormality_error'] < 1e-7
    R_['directions'] = {k: v for k, v in dinfo.items()
                        if k not in ('full_singular_values', 'extension_singular_values')}
    jax.clear_caches()
    print('DIRECTIONS', dinfo['directions_sha256'][:12], round(time.perf_counter() - begin, 1),
          flush=True)

    # ------------------------------------------------------------------ bank --
    ladder = [q for q in cfg['ladder_q'] if q < Rw]
    requests = {q: 4 * (K + q) for q in ladder}
    requests['linear'] = 4 * (K + Rw)
    maxmode = 0
    for req in requests.values():
        I, J, _ = H.mode_set(n, req)
        maxmode = max(maxmode, int(max(I.max(), J.max())) + 1)
    truth = np.stack([s_[1:-1, 1:-1] for s_ in same])
    bank = H.build_bank(params, n, cfg['rows_per_chunk'], cfg['eval_rows'], maxmode, truth=truth,
                        keep64=cfg['keep_f64'], keep32=cfg['keep_f32'])
    del truth
    assert bank['info']['retained_bank_rank'] == Rw, bank['info']
    R_['bank'] = dict(**bank['info'], projection_floor=K_.summarise(bank['floor']))
    save()
    print('BANK', bank['info']['build_seconds'], 'floor worst', float(bank['floor'].max()),
          flush=True)
    variants = ([('lean64', bank['chunks64'])] if cfg['keep_f64'] else []) + \
               ([('lean32', bank['chunks32'])] if cfg['keep_f32'] else [])
    single = None
    if cfg.get('retained_baseline'):
        assert cfg['keep_f64']
        single = jnp.concatenate(bank['chunks64'], axis=0)
        single.block_until_ready()

    # ------------------------------------------------------------------ arms --
    subjects = []
    for q in ladder:
        ops = H.make_ops(bank, params, codes, n, requests[q])
        M = int(ops['B'].shape[0])
        assert M > K + q, (M, K, q)
        engine = CC.prepare_correction(ops, codes, Cfull, q, cfg['retained'])
        assert engine['info']['linear_rank_valid'], (q, engine['info'])
        kern = H.make_lean(ops, engine, cfg['retained'], q)
        diag = H.make_diagnose(ops, engine, q)
        for vname, chunks in variants:
            subjects.append(dict(name=f'rom_q{q}_{vname}', kind='lean', q=q, M=M, ops=ops,
                                 engine=engine, kernel=kern, chunks=chunks, diag=diag,
                                 family='nm-rom', variant=vname))
        if single is not None:
            subjects.append(dict(name=f'rom_q{q}_retained', kind='retained', q=q, M=M,
                                 ops={**ops, 'bank': single}, engine=engine, family='nm-rom',
                                 variant='retained'))
        R_['arm_setup'].append(dict(q=q, M=M, requested_modes=requests[q],
                                    operator_sha256=ops['info']['operator_sha256'],
                                    **{k: engine['info'][k] for k in
                                       ('linear_rank', 'linear_condition_number',
                                        'projected_operator_sha256')}))
    lops = H.make_ops(bank, params, codes, n, requests['linear'])
    lk, Qt, Rr, linfo = H.make_linear(lops['B'], n)
    for vname, chunks in variants:
        subjects.append(dict(name=f'rom_q{Rw}_linear_{vname}', kind='linear', q=Rw,
                             M=linfo['M'], ops=lops, kernel=lk, Qt=Qt, Rr=Rr, chunks=chunks,
                             family='linear-rom', variant=vname))
    R_['arm_setup'].append(dict(q=Rw, linear=True, **linfo))

    lam = jnp.asarray(C.eigenvalues(n))
    cg = IC.make_cg(n, int(cfg['cg_maxiter']))
    subjects.append(dict(name='dst_direct', kind='fn', family='direct-control',
                         fn=lambda s: C.dst_solve(s, lam)))
    for tol in cfg['cg_tolerances']:
        subjects.append(dict(name=f'cg_{tol:g}', kind='fn', family='cg', tolerance=tol,
                             fn=lambda s, tol=tol: cg(s, jnp.asarray(tol))))
    for nc in cfg['coarse_intervals']:
        dk, clam = H.make_coarse(n, nc)
        subjects.append(dict(name=f'coarse{nc}_dst', kind='fn', family='coarse-dst',
                             coarse_intervals=nc, fn=lambda s, dk=dk, clam=clam: dk(s, clam)))
        ccg = IC.make_cg(nc, int(cfg['cg_maxiter']))
        ck, _ = H.make_coarse(n, nc, ccg)
        for tol in cfg['coarse_cg_tolerances']:
            subjects.append(dict(name=f'coarse{nc}_cg_{tol:g}', kind='fn', family='coarse-cg',
                                 coarse_intervals=nc, tolerance=tol,
                                 fn=lambda s, ck=ck, tol=tol: ck(s, jnp.asarray(tol))))
    R_['declared_subjects'] = [s['name'] for s in subjects]
    save()

    def invoke(sub, source):
        if sub['kind'] == 'lean':
            return H.lean_query(source, sub['ops'], sub['engine'], sub['kernel'], sub['chunks'])
        if sub['kind'] == 'retained':
            return CC.correction_query(source, sub['ops'], sub['engine'], cfg['retained'])
        if sub['kind'] == 'linear':
            return H.linear_query(source, sub['ops'], sub['kernel'], sub['Qt'], sub['Rr'],
                                  sub['chunks'])
        field, row, extra = H.generic_query(source, sub['fn'])
        if extra is not None:
            count, true, recursive, converged = extra
            row.update(iterations=int(count), true_relative_residual=float(true),
                       cg_converged=bool(converged))
        return field, row

    t = time.perf_counter()
    for sub in subjects:
        invoke(sub, sources[0])
    R_['compile_warmup_seconds'] = time.perf_counter() - t
    print('WARMUP', len(subjects), round(time.perf_counter() - t, 1), flush=True)

    order = np.random.default_rng(cfg['order_seed'])
    saved = {}
    slow_limit = cfg.get('slow_subject_cases', {})
    for rep in range(cfg['repetitions']):
        for case in range(len(dev)):
            for i in order.permutation(len(subjects)):
                sub = subjects[int(i)]
                if case >= slow_limit.get(sub['name'], len(dev)):
                    continue
                C.burn(cfg['burn_seconds'])
                assert gpu_uuid() == uuid0
                field, row = invoke(sub, sources[case])
                assert gpu_uuid() == uuid0
                assert np.isfinite(field).all(), sub['name']
                h = H.sha_array(field)
                key = (sub['name'], case)
                if key not in saved:
                    fn = f"{sub['name']}_case{case}.npy"
                    np.save(out / 'fields' / fn, field)
                    saved[key] = (fn, h)
                drop = ('starts', 'projected_jacobian_singular_values', 'augmented_latent',
                        'initial_latent', 'initial_correction_coefficients')
                if rep:            # the solved state is kept once per (subject, case)
                    drop += ('latent', 'correction_coefficients')
                slim = {k: v for k, v in row.items() if k not in drop}
                R_['invocations'].append(dict(
                    case=case, rep=rep, name=sub['name'], family=sub['family'],
                    q=sub.get('q'), M=sub.get('M'), variant=sub.get('variant'),
                    tolerance=sub.get('tolerance'), coarse_intervals=sub.get('coarse_intervals'),
                    field_sha256=h, saved_field=saved[key][0],
                    matches_saved_field=bool(h == saved[key][1]),
                    same_grid_error=C.relative(field, same[case]),
                    physical_error=C.relative(field, fine_sub[case]), **slim))
        print('TIMED', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()

    # ----------------------------------------------- untimed solver diagnostics --
    first = {}
    for x in R_['invocations']:
        first.setdefault((x['name'], x['case']), x)
    for sub in subjects:
        if sub['kind'] != 'lean' or sub['variant'] != variants[0][0]:
            continue
        for case in range(len(dev)):
            x = first[(sub['name'], case)]
            e, o = sub['engine'], sub['ops']
            full, red, rn, fn_ = jax.device_get(sub['diag'](
                jnp.asarray(sources[case]), jnp.asarray(x['latent']),
                jnp.asarray(x['correction_coefficients']), o['S'], o['I'], o['J'], o['W'],
                o['params'], e['Q'], e['C'], o['B'], e['Bp']))
            R_['diagnostics'].append(dict(
                name=sub['name'], case=case, full_stationarity=float(full),
                reduced_stationarity=float(red), weak_residual=float(rn),
                weak_source_norm=float(fn_),
                stationary=bool(full <= cfg['retained']['stationarity_tolerance']
                                and red <= cfg['retained']['stationarity_tolerance'])))

    # --------------------------------------------------------------- parity ----
    def load(name, case):
        return np.load(out / 'fields' / first[(name, case)]['saved_field'])

    base_variant = 'retained' if single is not None else variants[0][0]
    for q in ladder + [Rw]:
        tag = f'rom_q{q}_linear' if q == Rw else f'rom_q{q}'
        for vname, _ in variants:
            if q == Rw and base_variant == 'retained':
                ref_name = f'{tag}_{variants[0][0]}'
            else:
                ref_name = f'{tag}_{base_variant}'
            name = f'{tag}_{vname}'
            if name == ref_name:
                continue
            worst, ints = 0.0, True
            for case in range(len(dev)):
                worst = max(worst, C.relative(load(name, case), load(ref_name, case)))
                xa, xb = first[(name, case)], first[(ref_name, case)]
                for k in ('attempts', 'accepted', 'jacobians', 'reason',
                          'selected_training_code_index'):
                    if k in xa and k in xb:
                        ints = ints and xa[k] == xb[k]
            limit = cfg['parity']['f32_field'] if vname == 'lean32' else cfg['parity']['f64_field']
            R_['parity'].append(dict(candidate=name, baseline=ref_name,
                                     worst_field_relative=worst, integers_identical=ints,
                                     limit=limit, passed=bool(worst <= limit and ints)))
            print('PARITY', R_['parity'][-1], flush=True)
    R_['device_guard_final_uuid'] = gpu_uuid()
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('HIRES SOLVE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
