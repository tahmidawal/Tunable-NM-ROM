"""spectral-fom, Poisson 3D cube: 3D DST-I fast Poisson solver vs the poisson-bank-knob-3d settings.

ROM arms: the linear rung q = R' built exactly as `pbk3_cube.py` (commit d2c775a9) builds it: frozen
paper-p3d final08 bank (K16, R=128), pinned rotation prep_cube.npz, frozen settings frozen-N<n>.json
(accurate R128_linear, fast R64_linear at 32^3 and 64^3), final cohort seed 920499 (64 cases).
Query scope = the cube lane's: host f64 interior forcing (n-1)^3 in -> host f64 interior field out.
Spectral subjects `dst_fft`, `dst_mm`.  Validation: spectral vs SciPy DST truth on every case; the
paper's CG (paper-p3d iterative_cg.engine) at rtol 1e-6/1e-8/1e-10 on case 0 converging to the
spectral field; CONTROL with continuum eigenvalues must fail.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import common as C
import poisson as PP
import iterative_cg as CG
import pbk3_core as P3
import pbk3_cube as PC
import spec_core as SC
import spec_timing as ST


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    if a.smoke:
        cfg.update(repetitions=2, burn_seconds=0.001, case_count=4, cooldown_seconds=0.1, phase_dummy_seconds=0.1)
    n = int(cfg['intervals'])
    assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    uuid0 = ST.gpu_uuid()
    try:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                             '--format=csv,noheader'], text=True)
        assert uuid0 in inventory
    except FileNotFoundError:
        inventory = None
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)

    bankck = pickle.loads((here / 'bank.pkl').read_bytes())
    shas = dict(bank=P3.sha_file(here / 'bank.pkl'), head=P3.sha_file(here / 'head_K16.pkl'))
    assert shas == cfg['accepted_checkpoint_sha256'], shas
    mcfg = bankck['cfg']
    Rw = int(np.asarray(bankck['rotation']).shape[1])
    prep = np.load(here / cfg['prep'], allow_pickle=False)
    Tm = prep['T']
    rinfo = json.loads(str(prep['rotation_info']))
    assert rinfo['checkpoint_sha256'] == shas
    frozen = json.loads((here / cfg['frozen_settings']).read_text())
    assert frozen['intervals'] == n and frozen['selected_on'] == 'development'
    train = C.family(mcfg['train_seed'], mcfg['train_count'])
    cases = C.family(mcfg['reserved_final_seed'], cfg['final_count'])
    assert P3.sha_array(cases) == cfg['final_parameters_sha256']
    cases = cases[:cfg['case_count']]
    assert not any(np.allclose(t, s) for t in train for s in cases)
    R_ = dict(problem='poisson3d', config=cfg, commit=os.environ.get('SOURCE_COMMIT'),
              job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
              gpu=jax.devices()[0].device_kind, gpu_uuid=uuid0, nvidia_smi=inventory,
              matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
              smoke=bool(a.smoke), intervals=n, frozen_settings=frozen, checkpoint_sha256=shas,
              cohort=dict(role='final', parameters=cases.tolist(), sha256=P3.sha_array(cases), count=int(len(cases))),
              error_convention='relative L2 over the (n-1)^3 interior vs the same-grid discrete solution (SciPy DST-I); worst over cases',
              complete=False)
    save = lambda: P3.dump(out / 'result.json', R_)

    # ---- bank + weak operator + rotated blocks: verbatim pbk3_cube.py construction
    sources = [np.asarray(PP.source(n, p)) for p in cases]
    same = [PC.reference(n, f) for f in sources]
    bank = C.bank_at(bankck['params'], n, cfg['field_chunk']) @ np.asarray(bankck['rotation'])
    bankj = jnp.asarray(bank)
    triples = C.modes(cfg['weak_tests'], n)
    lam = np.asarray(C.mode_eigenvalues(n, triples))
    ti, tj, tk = (jnp.asarray(triples[:, ax] - 1) for ax in range(3))
    col = jax.jit(lambda g: C.dst3(g.reshape((n - 1,) * 3))[ti, tj, tk] / n ** 1.5)
    operator = np.stack([np.asarray(col(bankj[:, r])) for r in range(Rw)], axis=1)
    edges = [0] + sorted(cfg['R_ladder'])
    assert edges[-1] == Rw
    rot = P3.split_blocks(bankj @ jnp.asarray(Tm), edges)
    jax.block_until_ready(rot)
    del bankj
    rom = []
    arms = []
    for name, role in ((frozen['accurate'], 'rom_accurate'), (frozen['fast'], 'rom_fast')):
        assert name.endswith('_linear')
        Rp = int(name[1:].split('_')[0])
        Bl = operator @ Tm[:, :Rp]
        Qm, Rr = np.linalg.qr(Bl, mode='reduced')
        sv = np.linalg.svd(Rr, compute_uv=False)
        kern, _ = PC.make_linear(n, triples, lam)
        blocks = rot[:P3.nblocks(edges, Rp)]
        Qt, Rrj = jnp.asarray(Qm.T), jnp.asarray(Rr)
        fn = (lambda f, kern=kern, Qt=Qt, Rrj=Rrj, blocks=blocks: kern(f, Qt, Rrj, blocks))
        arms.append(dict(name=name, Rp=Rp, role=role, qr_condition_number=float(sv[0] / sv[-1])))

        def call(case, fn=fn):
            f, row, raw = ST.query(fn, sources[case])
            return f.reshape((n - 1,) * 3), row, None
        rom.append(dict(name=name, role=role, call=call))
    R_['arms'] = arms
    spec = []
    for variant in cfg['spectral_variants']:
        fn = SC.make_poisson_interior(n, 3, variant)
        spec.append(dict(name=f'dst_{variant}', role='spectral',
                         call=(lambda case, fn=fn: ST.query(fn, sources[case]))))
    save()

    # ---- V: validation
    V = dict(spectral_vs_scipy=[], cg_convergence=[])
    for s in spec:
        for case in range(len(cases)):
            f, _, _ = s['call'](case)
            V['spectral_vs_scipy'].append(dict(name=s['name'], case=case, relative=P3.rel(f, same[case])))
    fspec = spec[0]['call'](0)[0]
    cap = int(cfg['cg_validation_cap_per_interval']) * n
    for tol in cfg['cg_validation']:
        eng = CG.engine(n, float(tol), cap, retain_history=False)
        t0 = time.perf_counter()
        u, stats = jax.device_get(eng(jnp.asarray(sources[0])))
        c = CG.counters(stats, tol, cap)
        V['cg_convergence'].append(dict(rtol=tol, case=0, iterations=c['iterations'], converged=c['cg_converged'],
                                        true_relative_residual=c['cg_true_relative_residual'],
                                        seconds=time.perf_counter() - t0, cg_vs_spectral=P3.rel(u, fspec),
                                        cg_vs_scipy_truth=P3.rel(u, same[0])))
        print('CGVAL', V['cg_convergence'][-1], flush=True)
    k = np.pi * np.arange(1, n)
    lam_c = jnp.asarray(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    fc = np.asarray(jax.jit(lambda f, l: SC.dstn_fft(SC.dstn_fft(f) / l))(jnp.asarray(sources[0]), lam_c))
    lim = cfg['spectral_limit']
    V['control'] = dict(name='dst_continuum_eigenvalues_CONTROL', case=0, relative_vs_truth=P3.rel(fc, same[0]),
                        must_fail_limit=lim, failed_as_required=bool(P3.rel(fc, same[0]) > lim))
    cg_seq = [r['cg_vs_spectral'] for r in V['cg_convergence']]
    V['gates'] = dict(spectral_exact=bool(max(r['relative'] for r in V['spectral_vs_scipy']) <= lim),
                      cg_converges_to_spectral=bool(all(r['converged'] for r in V['cg_convergence'])
                                                   and cg_seq[-1] <= cfg['cg_agreement_limit']
                                                   and all(b < a for a, b in zip(cg_seq, cg_seq[1:]))),
                      control_failed=V['control']['failed_as_required'])
    R_['validation'] = V
    print('VALIDATION', V['gates'], cg_seq, V['control']['relative_vs_truth'], flush=True)
    save()

    # ---- T: A-B-A timing
    saved = {}

    def record(sub, case, rep, phase, field, row, raw):
        assert np.isfinite(field).all(), sub['name']
        h = P3.sha_array(field)
        key = (sub['name'], case)
        if key not in saved:
            np.save(out / 'fields' / f"{sub['name']}_case{case}.npy", field)
            saved[key] = h
        return dict(field_sha256=h, matches_saved_field=bool(h == saved[key]),
                    same_grid_error=P3.rel(field, same[case]))

    for s in rom + spec:
        for _ in range(2):
            s['call'](0)
    harness = ST.ABA(cfg, uuid0, record, log=lambda s: print(s, flush=True), save=save)
    R_['invocations'] = harness.invocations
    gates = harness.run(rom, spec, list(range(len(cases))), cfg['repetitions'], cfg['order_seed'])
    R_['phase_breaks'] = harness.breaks
    R_['timing_gates'] = gates
    R_['timings'] = harness.timings(rom, spec)
    R_['errors'] = {s['name']: P3.summarise([x['same_grid_error'] for x in harness.invocations
                                             if x['name'] == s['name'] and x['rep'] == 0 and x['phase'] in ('romA1', 'spec')])
                    for s in rom + spec}
    R_['gates'] = dict(**V['gates'], deterministic=all(x['matches_saved_field'] for x in harness.invocations),
                       drift=gates['drift']['passed'], neighbour=gates['neighbour']['passed'],
                       device_guard=ST.gpu_uuid() == uuid0)
    print('GATES', R_['gates'], flush=True)
    print('TIMINGS', {k: round(v['median_ms'], 4) for k, v in R_['timings'].items()}, flush=True)
    R_['complete'] = True
    save()
    (out / ('COMPLETE' if all(R_['gates'].values()) else 'COMPLETE-WITH-FAILED-GATES')).write_text(
        json.dumps(R_['gates']) + '\n')


if __name__ == '__main__':
    main()
