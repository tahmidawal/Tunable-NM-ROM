"""poisson-bank-knob QUICK (local GB10, indicative timings only).

Frozen R=512/K=32 Poisson 2D checkpoint (p-linear primary_K32), the twelve opened development
sources of p-linear / hires-poisson, one mesh. Arms: every R' in the ladder x q in {0, qmax(R')},
the q = R' linear rung, and the UNROTATED parent lean arms (q = 0, 256) as parity baseline.
Writes result.json + saved fields (for the independent NumPy check).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.fft import dstn
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import correction_core as CC
import pbh_core as K_
import plin_core as L_
import sep_common as sc
import hp_core as H
import pbk_core as P


def reference(param, n, workers=8):
    F = C.full_source(n, param)
    c = dstn(F[1:-1, 1:-1], type=1, norm='ortho', workers=workers)
    return np.pad(dstn(c / C.eigenvalues(n), type=1, norm='ortho', workers=workers), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--intervals', type=int, default=None)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--prep', default=None, help='cached rotation/directions npz (built if absent)')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    n = int(a.intervals or cfg['intervals'])
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    print(f'jax_backend={jax.default_backend()} precision={os.environ.get("JAX_DEFAULT_MATMUL_PRECISION")}', flush=True)
    assert jax.default_backend() == 'gpu'
    begin = time.perf_counter()
    m = cfg['model']
    params, codes, _ = sc.load_pkl(here / m['checkpoint_path'])
    basis = dict(np.load(here / m['basis_path']))
    codes = np.asarray(codes)
    np.testing.assert_array_equal(codes, basis['training_latents'])
    draws = C.source_params(m['training']['seed'], m['training']['draw_count'])
    fit, _ = K_.fit_validation_split(len(draws), m['training']['fit_split']['split_seed'],
                                     m['training']['fit_split']['validation_fraction'])
    train = draws[fit]
    dev = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                          C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    assert not any(np.allclose(t, s) for t in train for s in dev)
    K, Rw = codes.shape[1], int(np.asarray(params['h_lin']).shape[1])
    ni = cfg['training_intervals']
    R_ = dict(config=cfg, intervals=n, gpu=jax.devices()[0].device_kind, backend=jax.default_backend(),
              precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), jax_version=jax.__version__,
              cohort=dict(parameters=dev.tolist(), sha256=H.sha_array(dev)),
              timing_label='LOCAL GB10, INDICATIVE ONLY: not a result',
              invocations=[], arms=[], parity=[])

    prep = Path(a.prep) if a.prep else out / 'prep.npz'
    if prep.exists():
        z = np.load(prep, allow_pickle=True)
        Tm, Lm, Cfull = z['T'], z['L'], z['Cfull']
        R_['rotation'] = json.loads(str(z['rotation_info']))
        R_['directions'] = json.loads(str(z['directions_info']))
    else:
        Cfull, dinfo = L_.extend_basis(params, codes, basis, train, ni)
        assert dinfo['retained_prefix_exact'] and dinfo['extension_orthonormality_error'] < 1e-7
        dinfo = {k: v for k, v in dinfo.items() if k not in ('full_singular_values', 'extension_singular_values')}
        jax.clear_caches()
        Tm, Lm, rinfo = P.make_rotation(params, codes, train, ni)
        R_['rotation'], R_['directions'] = rinfo, dinfo
        np.savez(prep, T=Tm, L=Lm, Cfull=Cfull, rotation_info=json.dumps(rinfo),
                 directions_info=json.dumps(dinfo, default=float))
        jax.clear_caches()
    print('PREP', round(time.perf_counter() - begin, 1), R_['rotation']['cumulative_energy'],
          'cond', R_['rotation']['R_G_condition_number'], 'LT-I', R_['rotation']['L_times_T_identity_deviation'], flush=True)

    edges = [0] + sorted(cfg['R_ladder'])
    assert edges[-1] == Rw
    same = [reference(p_, n) for p_ in dev]
    sources = [C.full_source(n, p_) for p_ in dev]
    arms = []
    for Rp in sorted(cfg['R_ladder'], reverse=True):
        for q in sorted({0, P.arm_q_max(Rp, K)}):
            arms.append(dict(Rp=Rp, q=q, kind='trunc', name=f'R{Rp}_q{q}', M=4 * (K + q)))
        arms.append(dict(Rp=Rp, q=Rp, kind='linear', name=f'R{Rp}_linear', M=4 * (K + Rp)))
    for q in (0, 256):
        arms.append(dict(Rp=Rw, q=q, kind='orig', name=f'orig_q{q}', M=4 * (K + q)))
    maxmode = 0
    for arm in arms:
        I, J, _ = H.mode_set(n, arm['M'])
        maxmode = max(maxmode, int(max(I.max(), J.max())) + 1)
    truth = np.stack([s_[1:-1, 1:-1] for s_ in same])
    bank = P.build_banks(params, n, cfg['rows_per_chunk'], cfg['eval_rows'], maxmode, Tm, edges, truth,
                         keep_orig=True)
    del truth
    assert bank['info']['retained_bank_rank'] == Rw
    R_['bank'] = bank['info']
    R_['floors'] = {str(k): K_.summarise(v) for k, v in bank['floors'].items()}
    print('BANK', round(bank['info']['build_seconds'], 1), {k: round(v['worst'] * 100, 3) for k, v in R_['floors'].items()}, flush=True)

    subjects = []
    for arm in arms:
        ops = H.make_ops(bank, params, codes, n, arm['M'])
        Rp, q = arm['Rp'], arm['q']
        nb = P.nblocks(edges, Rp)
        if arm['kind'] == 'orig':
            eng = CC.prepare_correction(ops, codes, Cfull, q, cfg['retained'])
            kern = H.make_lean(ops, eng, cfg['retained'], q)
            call = lambda s, ops=ops, eng=eng, kern=kern: H.lean_query(s, ops, eng, kern, bank['orig'])
        elif arm['kind'] == 'trunc':
            if Rp < Rw:            # at R' = R the truncation is the identity: P = I exactly
                ops = {**ops, 'B': ops['B'] @ jnp.asarray(Tm[:, :Rp] @ Lm[:Rp])}
            eng = CC.prepare_correction(ops, codes, Cfull, q, cfg['retained'])
            assert eng['info']['linear_rank_valid'], (arm, eng['info']['linear_rank'])
            kern = P.make_trunc_lean(ops, eng, cfg['retained'], q, nb, n)
            Lr = jnp.asarray(Lm[:Rp])
            arm['linear_condition_number'] = eng['info']['linear_condition_number']
            call = lambda s, ops=ops, eng=eng, kern=kern, Lr=Lr: P.trunc_query(s, ops, eng, kern, bank['rot'], Lr)
        else:
            Bp = ops['B'] @ jnp.asarray(Tm[:, :Rp])
            kern, Qt, Rr, linfo = P.make_trunc_linear(Bp, n, nb)
            arm.update(linfo)
            call = lambda s, ops=ops, kern=kern, Qt=Qt, Rr=Rr: P.trunc_linear_query(s, ops, kern, Qt, Rr, bank['rot'])
        subjects.append(dict(arm=arm, call=call))
        R_['arms'].append(arm)
    for sub in subjects:
        sub['call'](sources[0])
    print('WARMUP', round(time.perf_counter() - begin, 1), flush=True)
    rng = np.random.default_rng(cfg.get('order_seed', 0))
    fields = {}
    for rep in range(a.reps):
        for case in range(len(dev)):
            for i in rng.permutation(len(subjects)):
                sub = subjects[int(i)]
                field, row = sub['call'](sources[case])
                name = sub['arm']['name']
                if (name, case) not in fields:
                    fn = f'{name}_case{case}.npy'
                    np.save(out / 'fields' / fn, field)
                    fields[(name, case)] = field
                keep = {k: v for k, v in row.items() if k.endswith('seconds') or k in ('attempts', 'reason', 'jacobians', 'residual', 'selected_training_code_index')}
                R_['invocations'].append(dict(name=name, case=case, rep=rep, Rp=sub['arm']['Rp'], q=sub['arm']['q'],
                                              kind=sub['arm']['kind'], same_grid_error=C.relative(field, same[case]),
                                              field_sha256=H.sha_array(field), **keep))
    for q in (0, 256):
        worst = max(C.relative(fields[(f'R{Rw}_q{q}', c)], fields[(f'orig_q{q}', c)]) for c in range(len(dev)))
        R_['parity'].append(dict(candidate=f'R{Rw}_q{q}', baseline=f'orig_q{q}', worst_field_relative=worst,
                                 limit=1e-10, passed=bool(worst <= 1e-10)))
    print('PARITY', R_['parity'], flush=True)
    R_['elapsed_seconds'] = time.perf_counter() - begin
    K_.dump(out / 'result.json', R_)
    # compact table to stdout
    for arm in R_['arms']:
        rows = [x for x in R_['invocations'] if x['name'] == arm['name']]
        err = max(x['same_grid_error'] for x in rows)
        dt = np.median([x['fused_device_seconds'] for x in rows]) * 1e3
        print(f"{arm['name']:>14}  err {err*100:7.3f} %  floor {R_['floors'][str(arm['Rp'])]['worst']*100:6.3f} %  dev {dt:7.2f} ms")


if __name__ == '__main__':
    main()
