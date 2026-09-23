"""poisson-bank-knob cluster driver: one mesh, one allocation.

Phases (all on one UUID-guarded GPU, host f64 source in -> host f64 field out for every subject):
  1 MAIN     every ROM arm (R' ladder x q in {0, qmax}, q = R' linear rung, parent unrotated arms where
             the original bank fits) + CG at the loose tolerances; randomised order, GPU burn-in
             before every invocation, `repetitions` retained reps over the 12 development sources.
  2 SLOW     CG at the tight tolerances in its own phase.
  3 NEIGHBOUR  gate: each ROM arm re-timed immediately after a long CG neighbour (cases 0..2) and
             compared with its phase-1 median on the same cases.
  4 PROFILE  stage split of every ROM arm (source projection + nearest code / LM solve /
             y elimination + coefficient map / reconstruction u = G c), each stage its own jit,
             synchronised, burn-in before each chain.
GPU-query time = `fused_device_seconds` (device work between the synchronised input copy and the
output copy), the paper's scope; `total_seconds` includes the host<->device copies.
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
import sep_common as sc
import hp_core as H
import pbk_core as P


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


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t


def make_stages(ops, engine, cfg, count, Lr, decode):
    """The fused kernel split into four separately jitted stages (profile only)."""
    reduced = {**ops, 'B': engine['Bp']}
    solve = __import__('kernel_solver').make_lm_kernel(
        reduced, cfg['online_preset']['budget'], True, False, cfg['linear_backward_error_limit'],
        cfg['online_preset']['stationarity_stop'])
    tau = cfg['online_preset']['tau']

    @jax.jit
    def s_project(source, S, I, J, W, predictions, cached_codes, Q):
        f = ops['project'](source, S, I, J, W)
        fp = f - Q @ (Q.T @ f) if count else f
        _, idx = jax.lax.top_k(-jnp.sum((predictions - fp[None, :]) ** 2, axis=1), 1)
        return f, fp, cached_codes[idx[0]]

    @jax.jit
    def s_lm(z0, fp):
        return solve(z0, fp, jnp.asarray(tau))[0][0]

    @jax.jit
    def s_elim(z, f, params, Q, R, Cq, B, Lr):
        h = sc.head(params, z)
        if count:
            c = h + Cq @ jax.scipy.linalg.solve_triangular(R, Q.T @ (f - B @ h), lower=False)
        else:
            c = h
        return c if Lr is None else Lr @ c

    def run(source):
        e, o = engine, ops
        (f, fp, z0), t1 = timed(s_project, source, o['S'], o['I'], o['J'], o['W'],
                                e['cache']['predictions'], e['cache']['codes'], e['Q'])
        z, t2 = timed(s_lm, z0, fp)
        a, t3 = timed(s_elim, z, f, o['params'], e['Q'], e['R'], e['C'], o['B'], Lr)
        field, t4 = timed(decode, a)
        return field, dict(project_and_start=t1, lm_solve=t2, y_elimination_and_map=t3, reconstruction=t4)
    return run


def make_linear_stages(ops, Qt, Rr, decode):
    @jax.jit
    def s_solve(source, S, I, J, W, Qt, Rr):
        fm = (S.T @ source[1:-1, 1:-1] @ S)[I, J] * W
        return jax.scipy.linalg.solve_triangular(Rr, Qt @ fm, lower=False)

    def run(source):
        a, t1 = timed(s_solve, source, ops['S'], ops['I'], ops['J'], ops['W'], Qt, Rr)
        field, t4 = timed(decode, a)
        return field, dict(project_and_start=t1, lm_solve=0.0, y_elimination_and_map=0.0, reconstruction=t4)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    if a.smoke:
        cfg.update(intervals=128, repetitions=1, burn_seconds=0.001, rows_per_chunk=32, eval_rows=16,
                   cg_fast=[0.1, 0.01], cg_slow=[1e-4], neighbour_cases=1, profile_reps=1)
    n = int(cfg['intervals'])
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    assert len(jax.devices()) == 1
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    uuid0 = gpu_uuid()
    try:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                             '--format=csv,noheader'], text=True)
        assert uuid0 in inventory, (uuid0, inventory)
    except FileNotFoundError:
        inventory = None
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    R_ = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
              backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0,
              nvidia_smi=inventory, x64=True, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
              jax_version=jax.__version__, smoke=bool(a.smoke), intervals=n,
              timing_contract=('host f64 source in -> host f64 nodal field out for every subject; '
                               'GPU-query = fused_device_seconds; randomised order; burn-in before every '
                               'invocation; slow CG in its own phase; neighbour gate'),
              arms=[], invocations=[], slow_invocations=[], neighbour=[], profile=[], parity=[],
              complete=False)
    save = lambda: K_.dump(out / 'result.json', R_)

    m = cfg['model']
    params, codes, _ = sc.load_pkl(here / Path(m['checkpoint_path']).name)
    basis = dict(np.load(here / Path(m['basis_path']).name))
    codes = np.asarray(codes)
    np.testing.assert_array_equal(codes, basis['training_latents'])
    draws = C.source_params(m['training']['seed'], m['training']['draw_count'])
    fit, _ = K_.fit_validation_split(len(draws), m['training']['fit_split']['split_seed'],
                                     m['training']['fit_split']['validation_fraction'])
    train = draws[fit]
    dev = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                          C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    assert not any(np.allclose(t, s) for t in train for s in dev)
    R_['cohort'] = dict(parameters=dev.tolist(), sha256=H.sha_array(dev),
                        note='the twelve OPENED development sources of p-linear / hires-poisson (unchanged)')
    K, Rw = codes.shape[1], int(np.asarray(params['h_lin']).shape[1])
    prep_path = here / Path(cfg['prep_path']).name
    z = np.load(prep_path, allow_pickle=True)
    Tm, Lm, Cfull = z['T'], z['L'], z['Cfull']
    R_['rotation'] = json.loads(str(z['rotation_info']))
    R_['prep_sha256'] = hashlib.sha256(prep_path.read_bytes()).hexdigest()
    R_['checkpoint_sha256'] = hashlib.sha256((here / Path(m['checkpoint_path']).name).read_bytes()).hexdigest()

    edges = [0] + sorted(cfg['R_ladder'])
    assert edges[-1] == Rw
    same = [reference(p_, n) for p_ in dev]
    sources = [C.full_source(n, p_) for p_ in dev]
    arms = []
    for Rp in sorted(cfg['R_ladder'], reverse=True):
        for q in sorted({0, P.arm_q_max(Rp, K)}):
            arms.append(dict(Rp=Rp, q=q, kind='trunc', name=f'R{Rp}_q{q}', M=4 * (K + q)))
        arms.append(dict(Rp=Rp, q=Rp, kind='linear', name=f'R{Rp}_linear', M=4 * (K + Rp)))
    if cfg['keep_orig']:
        for q in (0, 256):
            arms.append(dict(Rp=Rw, q=q, kind='orig', name=f'orig_q{q}', M=4 * (K + q)))
    maxmode = 0
    for arm in arms:
        I, J, _ = H.mode_set(n, arm['M'])
        maxmode = max(maxmode, int(max(I.max(), J.max())) + 1)
    truth = np.stack([s_[1:-1, 1:-1] for s_ in same])
    bank = P.build_banks(params, n, cfg['rows_per_chunk'], cfg['eval_rows'], maxmode, Tm, edges, truth,
                         keep_orig=cfg['keep_orig'])
    del truth
    assert bank['info']['retained_bank_rank'] == Rw, bank['info']
    R_['bank'] = bank['info']
    R_['floors'] = {str(k): K_.summarise(v) for k, v in bank['floors'].items()}
    save()
    print('BANK', round(bank['info']['build_seconds'], 1),
          {k: round(v['worst'] * 100, 3) for k, v in R_['floors'].items()}, flush=True)

    subjects = []
    for arm in arms:
        ops = H.make_ops(bank, params, codes, n, arm['M'])
        Rp, q = arm['Rp'], arm['q']
        nb = P.nblocks(edges, Rp)
        dec_rot = jax.jit(lambda a_, rot, nb=nb: P.decode_blocks(rot, a_, n, nb))
        if arm['kind'] == 'orig':
            eng = CC.prepare_correction(ops, codes, Cfull, q, cfg['retained'])
            kern = H.make_lean(ops, eng, cfg['retained'], q)
            call = lambda s, ops=ops, eng=eng, kern=kern: H.lean_query(s, ops, eng, kern, bank['orig'])
            dec = jax.jit(lambda c_, ch: H.decode_chunks(ch, c_, n))
            prof = make_stages(ops, eng, cfg['retained'], q, None, lambda c_, dec=dec: dec(c_, bank['orig']))
        elif arm['kind'] == 'trunc':
            if Rp < Rw:            # at R' = R the truncation is the identity: P = I exactly
                ops = {**ops, 'B': ops['B'] @ jnp.asarray(Tm[:, :Rp] @ Lm[:Rp])}
            eng = CC.prepare_correction(ops, codes, Cfull, q, cfg['retained'])
            assert eng['info']['linear_rank_valid'], (arm, eng['info']['linear_rank'])
            arm['linear_condition_number'] = eng['info']['linear_condition_number']
            kern = P.make_trunc_lean(ops, eng, cfg['retained'], q, nb, n)
            Lr = jnp.asarray(Lm[:Rp])
            call = lambda s, ops=ops, eng=eng, kern=kern, Lr=Lr: P.trunc_query(s, ops, eng, kern, bank['rot'], Lr)
            prof = make_stages(ops, eng, cfg['retained'], q, Lr, lambda a_, dec=dec_rot: dec(a_, bank['rot']))
        else:
            Bp = ops['B'] @ jnp.asarray(Tm[:, :Rp])
            kern, Qt, Rr, linfo = P.make_trunc_linear(Bp, n, nb)
            arm.update(linfo)
            call = lambda s, ops=ops, kern=kern, Qt=Qt, Rr=Rr: P.trunc_linear_query(s, ops, kern, Qt, Rr, bank['rot'])
            prof = make_linear_stages(ops, Qt, Rr, lambda a_, dec=dec_rot: dec(a_, bank['rot']))
        arm['family'] = {'trunc': 'nm-rom', 'orig': 'nm-rom-parent', 'linear': 'linear-rung'}[arm['kind']]
        subjects.append(dict(name=arm['name'], family=arm['family'], arm=arm, call=call, prof=prof))
        R_['arms'].append(arm)
    cg = IC.make_cg(n, int(cfg['cg_maxiter']))

    def cg_call(tol):
        def call(s):
            field, row, extra = H.generic_query(s, lambda x: cg(x, jnp.asarray(tol)))
            count, true, recursive, converged = extra
            row.update(iterations=int(count), true_relative_residual=float(true), cg_converged=bool(converged))
            return field, row
        return call
    fast = [dict(name=f'cg_{t:g}', family='cg', tolerance=t, call=cg_call(t)) for t in cfg['cg_fast']]
    slow = [dict(name=f'cg_{t:g}', family='cg', tolerance=t, call=cg_call(t)) for t in cfg['cg_slow']]
    R_['declared_subjects'] = [s['name'] for s in subjects + fast + slow]
    save()

    saved = {}

    def record(sub, case, rep, phase, field, row, prev):
        assert np.isfinite(field).all(), sub['name']
        h = H.sha_array(field)
        key = (sub['name'], case)
        if key not in saved:
            np.save(out / 'fields' / f"{sub['name']}_case{case}.npy", field)
            saved[key] = h
        keep = {k: v for k, v in row.items() if k.endswith('seconds') or k in (
            'attempts', 'reason', 'jacobians', 'accepted', 'residual', 'selected_training_code_index',
            'iterations', 'true_relative_residual', 'cg_converged', 'max_linear_backward_error',
            'fallback_count')}
        return dict(name=sub['name'], family=sub['family'], case=case, rep=rep, phase=phase, previous=prev,
                    Rp=sub.get('arm', {}).get('Rp'), q=sub.get('arm', {}).get('q'),
                    tolerance=sub.get('tolerance'), field_sha256=h, matches_saved_field=bool(h == saved[key]),
                    same_grid_error=C.relative(field, same[case]), **keep)

    t = time.perf_counter()
    for sub in subjects + fast + slow:
        sub['call'](sources[0])
    R_['compile_warmup_seconds'] = time.perf_counter() - t
    print('WARMUP', round(R_['compile_warmup_seconds'], 1), flush=True)

    # ---- phase 1: main
    order = np.random.default_rng(cfg['order_seed'])
    main_set = subjects + fast
    prev = None
    for rep in range(cfg['repetitions']):
        for case in range(len(dev)):
            for i in order.permutation(len(main_set)):
                sub = main_set[int(i)]
                C.burn(cfg['burn_seconds'])
                assert gpu_uuid() == uuid0
                field, row = sub['call'](sources[case])
                R_['invocations'].append(record(sub, case, rep, 'main', field, row, prev))
                prev = sub['name']
        print('MAIN', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()
    # ---- phase 2: slow CG
    for rep in range(cfg['slow_repetitions']):
        for case in range(len(dev)):
            for i in order.permutation(len(slow)):
                sub = slow[int(i)]
                C.burn(cfg['burn_seconds'])
                field, row = sub['call'](sources[case])
                R_['slow_invocations'].append(record(sub, case, rep, 'slow', field, row, None))
        print('SLOW', rep, round(time.perf_counter() - begin, 1), flush=True)
        save()
    # ---- phase 3: neighbour gate
    long_sub = [s for s in slow if s['tolerance'] == cfg['neighbour_tolerance']][0]
    for case in range(cfg['neighbour_cases']):
        for i in order.permutation(len(subjects)):
            sub = subjects[int(i)]
            long_sub['call'](sources[case])
            C.burn(cfg['burn_seconds'])
            field, row = sub['call'](sources[case])
            R_['neighbour'].append(record(sub, case, 0, 'neighbour', field, row, long_sub['name']))
    gate_rows = []
    for sub in subjects:
        cases = range(cfg['neighbour_cases'])
        base = np.median([x['fused_device_seconds'] for x in R_['invocations']
                          if x['name'] == sub['name'] and x['case'] in cases])
        after = np.median([x['fused_device_seconds'] for x in R_['neighbour'] if x['name'] == sub['name']])
        gate_rows.append(dict(name=sub['name'], main_median=float(base), after_long_median=float(after),
                              ratio=float(after / base)))
    R_['neighbour_gate'] = dict(rows=gate_rows, limit=cfg['neighbour_limit'],
                                passed=bool(all(r['ratio'] <= cfg['neighbour_limit'] for r in gate_rows)))
    print('NEIGHBOUR', R_['neighbour_gate']['passed'], max(r['ratio'] for r in gate_rows), flush=True)
    save()
    # ---- phase 4: stage profile
    for rep in range(cfg['profile_reps']):
        for case in range(len(dev)):
            for i in order.permutation(len(subjects)):
                sub = subjects[int(i)]
                C.burn(cfg['burn_seconds'])
                src = jax.device_put(sources[case])
                src.block_until_ready()
                field, stages = sub['prof'](src)
                fld = np.asarray(jax.device_get(field))
                R_['profile'].append(dict(name=sub['name'], case=case, rep=rep, **stages,
                                          field_relative_to_fused=C.relative(fld, np.load(
                                              out / 'fields' / f"{sub['name']}_case{case}.npy"))))
    save()
    # ---- parity
    if cfg['keep_orig']:
        for q in (0, 256):
            worst = max(C.relative(np.load(out / 'fields' / f'R{Rw}_q{q}_case{c}.npy'),
                                   np.load(out / 'fields' / f'orig_q{q}_case{c}.npy')) for c in range(len(dev)))
            R_['parity'].append(dict(candidate=f'R{Rw}_q{q}', baseline=f'orig_q{q}', worst_field_relative=worst,
                                     limit=1e-10, passed=bool(worst <= 1e-10)))
    stats = jax.devices()[0].memory_stats() or {}
    R_['device_memory'] = {k: int(v) for k, v in stats.items() if k in ('peak_bytes_in_use', 'bytes_limit')}
    allinv = R_['invocations'] + R_['slow_invocations'] + R_['neighbour']
    R_['gates'] = dict(
        parity=all(p['passed'] for p in R_['parity']),
        deterministic=all(x['matches_saved_field'] for x in allinv),
        cg_converged=all(x.get('cg_converged', True) for x in allinv),
        neighbour=R_['neighbour_gate']['passed'],
        profile_matches_fused=all(x['field_relative_to_fused'] <= 1e-10 for x in R_['profile']),
        device_guard=gpu_uuid() == uuid0)
    print('GATES', R_['gates'], flush=True)
    R_['elapsed_seconds'] = time.perf_counter() - begin
    R_['complete'] = True
    save()
    (out / ('COMPLETE' if all(R_['gates'].values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(
        json.dumps(R_['gates']) + '\n')
    print('PBK SOLVE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
