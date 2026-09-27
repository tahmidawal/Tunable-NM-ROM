"""spectral-fom, Poisson 2D square: DST-I fast Poisson solver vs the poisson-bank-knob Table-1 settings.

One mesh per driver run, one GPU.  ROM arms are built with the poisson-bank-knob code exactly as
`pbk_solve.py` builds them (frozen primary_K32, pinned rotation runs/prep.npz, linear rung q = R').
Spectral subjects: `dst_fft`, `dst_mm` (spec_core).  Phases:
  V  validation (untimed): spectral vs SciPy DST truth on every case; CG (the paper's FOM, iterative_core.
     make_cg) at rtol 1e-6/1e-8/1e-10 on case 0 against the spectral field (must converge to it);
     a CONTROL spectral solve with continuum eigenvalues (pi k)^2 that MUST fail the 1e-10 gate.
  T  A-B-A timing (spec_timing.ABA).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.fft import dstn
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import iterative_core as IC
import pbh_core as K_
import sep_common as sc
import hp_core as H
import pbk_core as P
import spec_core as SC
import spec_timing as ST


def reference(param, n, workers=8):
    F = C.full_source(n, param)
    c = dstn(F[1:-1, 1:-1], type=1, norm='ortho', workers=workers)
    return np.pad(dstn(c / C.eigenvalues(n), type=1, norm='ortho', workers=workers), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    if a.smoke:
        cfg.update(intervals=128, repetitions=2, burn_seconds=0.001, rows_per_chunk=32, eval_rows=16,
                   cooldown_seconds=0.1, phase_dummy_seconds=0.1, cg_validation=[1e-6, 1e-8])
    n = int(cfg['intervals'])
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    assert len(jax.devices()) == 1
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    uuid0 = ST.gpu_uuid()
    try:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                             '--format=csv,noheader'], text=True)
        assert uuid0 in inventory, (uuid0, inventory)
    except FileNotFoundError:
        inventory = None
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    R_ = dict(problem='poisson2d', config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0, nvidia_smi=inventory,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
              smoke=bool(a.smoke), intervals=n, complete=False,
              error_convention='relative L2 over all (n+1)^2 nodes vs the same-grid discrete solution (SciPy DST-I); worst over the 12 development sources')
    save = lambda: K_.dump(out / 'result.json', R_)

    # ---------------- model + arms: verbatim poisson-bank-knob construction (pbk_solve.py) ----------
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
    R_['cohort'] = dict(parameters=dev.tolist(), sha256=H.sha_array(dev))
    K, Rw = codes.shape[1], int(np.asarray(params['h_lin']).shape[1])
    prep_path = here / Path(cfg['prep_path']).name
    z = np.load(prep_path, allow_pickle=True)
    Tm = z['T']
    R_['prep_sha256'] = hashlib.sha256(prep_path.read_bytes()).hexdigest()
    R_['checkpoint_sha256'] = hashlib.sha256((here / Path(m['checkpoint_path']).name).read_bytes()).hexdigest()
    edges = [0] + sorted(cfg['R_ladder'])
    assert edges[-1] == Rw
    same = [reference(p_, n) for p_ in dev]
    sources = [C.full_source(n, p_) for p_ in dev]
    arms = [dict(Rp=int(Rp), q=int(Rp), name=f'R{Rp}_linear', M=4 * (K + int(Rp)), role=role)
            for Rp, role in ((cfg['accurate_Rp'], 'rom_accurate'), (cfg['fast_Rp'], 'rom_fast'))]
    maxmode = 0
    for arm in arms:
        I, J, _ = H.mode_set(n, arm['M'])
        maxmode = max(maxmode, int(max(I.max(), J.max())) + 1)
    truth = np.stack([s_[1:-1, 1:-1] for s_ in same])
    bank = P.build_banks(params, n, cfg['rows_per_chunk'], cfg['eval_rows'], maxmode, Tm, edges, truth,
                         keep_orig=False)
    del truth
    assert bank['info']['retained_bank_rank'] == Rw, bank['info']
    R_['bank'] = bank['info']
    R_['floors'] = {str(k): K_.summarise(v) for k, v in bank['floors'].items()}
    print('BANK', round(bank['info']['build_seconds'], 1), flush=True)

    rom = []
    for arm in arms:
        ops = H.make_ops(bank, params, codes, n, arm['M'])
        nb = P.nblocks(edges, arm['Rp'])
        Bp = ops['B'] @ jnp.asarray(Tm[:, :arm['Rp']])
        kern, Qt, Rr, linfo = P.make_trunc_linear(Bp, n, nb)
        arm.update(linfo)

        def call(case, ops=ops, kern=kern, Qt=Qt, Rr=Rr):
            field, row = P.trunc_linear_query(sources[case], ops, kern, Qt, Rr, bank['rot'])
            row.pop('linear_coefficients', None)
            return field, row, None
        rom.append(dict(name=arm['name'], role=arm['role'], call=call))
    R_['arms'] = arms

    spec = []
    for variant in cfg['spectral_variants']:
        fn = SC.make_poisson(n, 2, variant)

        def call(case, fn=fn):
            return ST.query(fn, sources[case])
        spec.append(dict(name=f'dst_{variant}', role='spectral', call=call))
    save()

    # ---------------- V: validation (untimed) ----------------
    V = dict(spectral_vs_scipy=[], cg_convergence=[], control=None)
    for s in spec:
        for case in range(len(dev)):
            f, _, _ = s['call'](case)
            V['spectral_vs_scipy'].append(dict(name=s['name'], case=case, relative=C.relative(f, same[case])))
    cg = IC.make_cg(n, int(cfg['cg_maxiter']))
    fspec = np.asarray(spec[0]['call'](0)[0])
    for tol in cfg['cg_validation']:
        t0 = time.perf_counter()
        u, info = cg(jnp.asarray(sources[0]), jnp.asarray(tol))
        u = np.asarray(u)
        count, true, recursive, conv = jax.device_get(info)
        V['cg_convergence'].append(dict(rtol=tol, case=0, iterations=int(count), true_relative_residual=float(true),
                                        converged=bool(conv), seconds=time.perf_counter() - t0,
                                        cg_vs_spectral=C.relative(u, fspec), cg_vs_scipy_truth=C.relative(u, same[0])))
        print('CGVAL', V['cg_convergence'][-1], flush=True)
    lam_c = jnp.asarray(np.add.outer((np.pi * np.arange(1, n)) ** 2, (np.pi * np.arange(1, n)) ** 2))
    fc = np.asarray(jax.jit(lambda s, l: SC.pad1(SC.dstn_fft(SC.dstn_fft(SC.interior(s)) / l)))(
        jnp.asarray(sources[0]), lam_c))
    V['control'] = dict(name='dst_continuum_eigenvalues_CONTROL', case=0, relative_vs_truth=C.relative(fc, same[0]),
                        must_fail_limit=cfg['spectral_limit'],
                        failed_as_required=bool(C.relative(fc, same[0]) > cfg['spectral_limit']))
    lim = cfg['spectral_limit']
    worst_spec = max(r['relative'] for r in V['spectral_vs_scipy'])
    cg_seq = [r['cg_vs_spectral'] for r in V['cg_convergence']]
    V['gates'] = dict(spectral_exact=bool(worst_spec <= lim),
                      cg_converges_to_spectral=bool(all(r['converged'] for r in V['cg_convergence'])
                                                   and cg_seq[-1] <= cfg['cg_agreement_limit']
                                                   and all(b < a for a, b in zip(cg_seq, cg_seq[1:]))),
                      control_failed=V['control']['failed_as_required'])
    R_['validation'] = V
    print('VALIDATION', V['gates'], worst_spec, cg_seq, V['control']['relative_vs_truth'], flush=True)
    save()

    # ---------------- T: A-B-A timing ----------------
    saved = {}

    def record(sub, case, rep, phase, field, row, raw):
        assert np.isfinite(field).all(), sub['name']
        h = H.sha_array(field)
        key = (sub['name'], case)
        if key not in saved:
            np.save(out / 'fields' / f"{sub['name']}_case{case}.npy", field)
            saved[key] = h
        return dict(field_sha256=h, matches_saved_field=bool(h == saved[key]),
                    same_grid_error=C.relative(field, same[case]))

    t = time.perf_counter()
    for s in rom + spec:
        for _ in range(2):
            s['call'](0)
    R_['compile_warmup_seconds'] = time.perf_counter() - t
    harness = ST.ABA(cfg, uuid0, record, log=lambda s: print(s, flush=True), save=save)
    R_['invocations'] = harness.invocations
    gates = harness.run(rom, spec, list(range(len(dev))), cfg['repetitions'], cfg['order_seed'])
    R_['phase_breaks'] = harness.breaks
    R_['timing_gates'] = gates
    R_['timings'] = harness.timings(rom, spec)
    errs = {}
    for s in rom + spec:
        e = [x['same_grid_error'] for x in harness.invocations if x['name'] == s['name'] and x['rep'] == 0
             and x['phase'] in ('romA1', 'spec')]
        errs[s['name']] = K_.summarise(e)
    R_['errors'] = errs
    stats = jax.devices()[0].memory_stats() or {}
    R_['device_memory'] = {k: int(v) for k, v in stats.items() if k in ('peak_bytes_in_use', 'bytes_limit')}
    R_['gates'] = dict(**V['gates'], deterministic=all(x['matches_saved_field'] for x in harness.invocations),
                       drift=gates['drift']['passed'], neighbour=gates['neighbour']['passed'],
                       device_guard=ST.gpu_uuid() == uuid0)
    print('GATES', R_['gates'], flush=True)
    print('TIMINGS', {k: round(v['median_ms'], 3) for k, v in R_['timings'].items()}, flush=True)
    R_['complete'] = True
    save()
    (out / ('COMPLETE' if all(R_['gates'].values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(
        json.dumps(R_['gates']) + '\n')


if __name__ == '__main__':
    main()
