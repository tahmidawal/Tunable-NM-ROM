"""Phase 2, Burgers 2D: full-bank (q = R, identity head) solved error and paired cost per bank.

Every bank is presented to the UNCHANGED head-ablation machinery (`arms.build_operators`,
`arms.build_cold`, `arms.make_query`, dense exact advection, LM with a pivoted dense solve) as
a grid bank of its orthonormalised columns with the identity head -- the construction
`ablation.py` uses for its POD-LSPG and free-bank arms. No EQ rule is fitted here: a new bank
has no certified rule, so every rung is reported DENSE (DESIGN: "an uncertified rung is
reported dense"). The project's Newton FOM is timed in the same allocation.
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

import engines as e
import arms as A
import common as C
import bf_core as K

F64 = jnp.float64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    here = Path(a.config).resolve().parent
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    smoke = bool(cfg.get('smoke', False))
    L, dt = cfg['intervals'], cfg['dt']
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    R_ = dict(lane='bank-floor', phase='solve', pde='burgers2d', config=cfg,
              source_commit=os.environ.get('SOURCE_COMMIT', 'unknown'),
              slurm_job=os.environ.get('SLURM_JOB_ID', 'local'), gpu=gpu, jax=jax.__version__,
              backend=jax.default_backend(), x64=True,
              matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'unset'),
              timing_contract='host initial field in -> six host dense fields out; burn-in before '
                              'every timed call; synchronised; medians over retained repetitions',
              quadrature='dense (no EQ rule is certified for any new bank)',
              subjects={}, complete=False)
    assert smoke or R_['backend'] == 'gpu'
    save = lambda: K.dump(out / 'result.json', R_)

    train_p = C.incumbent_draw()[:cfg['candidate_trajectories']]
    cohorts = dict(dev6=np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                                        e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases']))),
                   confirm=e.params_draw(cfg['confirm_seed'], cfg['confirm_draw'])[:cfg['confirm_count']])
    for k, v in cohorts.items():
        d = np.abs(C.incumbent_draw()[:, None, :] - v[None]).max(-1).min()
        assert d > 1e-9, f'{k} overlaps the training draw'
    R_['cohorts'] = {k: dict(count=len(v), parameters=v.tolist(), sha256=K.sha_array(v))
                     for k, v in cohorts.items()}

    # candidate states for the cold start and the trust radius: a training-prefix subsample
    qfine, _ = e.make_fom(L, dt, None, .25, dt)
    keep = jnp.arange(0, int(round(.25 / dt)) + 1, 2)
    Ucand = np.concatenate([np.asarray(qfine(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7)[0][keep]
                                       [:, 1:-1, 1:-1].reshape(len(keep), -1)) for p in train_p])
    del qfine
    fom, _ = e.make_fom(L, dt, None, .25, .05)
    inputs = {k: [e.initial(L, p) for p in v] for k, v in cohorts.items()}
    truth = {k: [np.asarray(fom(jnp.asarray(u), float(p[4]), *cfg['reference_tolerances'])[0])
                 for u, p in zip(inputs[k], v)] for k, v in cohorts.items()}

    def errs(fields, ref):
        return [float(np.linalg.norm(fields[t] - ref[t]) / np.linalg.norm(ref[t])) for t in range(len(ref))]

    def run_subject(name, invoke, meta, reps):
        rows = {}
        for coh, params in cohorts.items():
            per, times, extra = [], [], []
            for i, p in enumerate(params):
                nu = float(p[4])
                jax.block_until_ready(invoke(jnp.asarray(inputs[coh][i]), nu))      # warm, discarded
                ts = []
                for r in range(reps):
                    e.burn(cfg['burn_seconds'])
                    t0 = time.perf_counter()
                    u = jax.device_put(np.array(inputs[coh][i], copy=True))
                    value = invoke(u, nu)
                    jax.block_until_ready(value)
                    f = np.asarray(value[0])
                    ts.append(time.perf_counter() - t0)
                assert np.isfinite(f).all()
                per.append(errs(f, truth[coh][i]))
                times.append(ts)
                if len(value) > 3 and meta['kind'] == 'rom':
                    it, reason = np.asarray(value[1]), np.asarray(value[3])
                    extra.append(dict(iterations_total=int(it.sum()), iterations_max=int(it.max()),
                                      budget_exits=int((reason == 0).sum()),
                                      reasons=np.bincount(reason, minlength=5).tolist(),
                                      ic_iterations=int(value[5]), ic_reason=int(value[6])))
                if coh == 'dev6':
                    np.save(out / 'fields' / f'{name}_case{i}.npy', f)
            pe = np.asarray(per)
            rows[coh] = dict(worst_all_times=float(pe.max()), worst_evolved=float(pe[:, 1:].max()),
                             worst_t0=float(pe[:, 0].max()), median_all=float(np.median(pe)),
                             per_case_per_time=pe.tolist(), median_total_ms=float(np.median(times) * 1e3),
                             total_seconds=times, solver=extra)
        R_['subjects'][name] = dict(meta, repetitions=reps, results=rows)
        save()
        d = rows['dev6']
        print(f'SUBJECT {name}: all {d["worst_all_times"]:.4e} evolved {d["worst_evolved"]:.4e} '
              f't0 {d["worst_t0"]:.4e} total {d["median_total_ms"]:.1f} ms '
              f'budget_exits {sum(x["budget_exits"] for x in d["solver"]) if d["solver"] else "-"}', flush=True)

    for fs in cfg['fom_settings']:
        run_subject(fs['name'], lambda u, nu, fs=fs: fom(u, nu, fs['ntol'], fs['ltol']),
                    dict(kind='fom', label=f'Newton-BiCGStab(Helmholtz) ntol {fs["ntol"]:g} ltol {fs["ltol"]:g}',
                         **fs), cfg['fom_repetitions'])

    xy = K.grid(L)
    for spec in cfg['banks']:
        tag = spec['tag']
        t0 = time.perf_counter()
        path = here / spec['file']
        if path.suffix == '.npy':
            Q = jnp.asarray(np.load(path)[:, :spec['R']])
            binfo = dict(columns=int(Q.shape[1]), rank=int(Q.shape[1]), rank_valid=True)
        else:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            blocks = d['blocks'] if 'blocks' in d else [K.bank_block(d['params'])]
            Q, binfo = K.orth_basis(K.bank_of(blocks, xy))
        Rk = int(Q.shape[1])
        M = int(cfg['test_multiplier'] * Rk)
        bank = A.GridBank(Q, L)
        head = A.identity_head()
        Zc = np.asarray(jnp.asarray(Ucand) @ Q)
        trust = .01 * float(np.max(np.linalg.norm(Zc - Zc.mean(0), axis=1)))
        data, oinfo = A.build_operators(bank, L, M, 'dense')
        axis = int(max(cfg['cold_axis_points'], np.ceil(np.sqrt(cfg['cold_oversample'] * Rk))))
        cold, cinfo = A.build_cold(bank, head, Zc, axis)
        query = A.make_query(head, Rk, L, dt, trust, 'dense', linear='lu', **cfg['strict'])
        floor = {coh: float(max(K.floors(Q, np.asarray(t)[:, 1:-1, 1:-1].reshape(len(t), -1)).max()
                                for t in truth[coh])) for coh in cohorts}
        meta = dict(kind='rom', bank=tag, R=Rk, M=M, basis=dict(binfo, file=spec['file'], sha256=K.sha_file(path)),
                    trust_radius=trust, cold=cinfo, operators={k: v for k, v in oinfo.items()},
                    bank_floor_on_reference_fields=floor, setup_seconds=time.perf_counter() - t0)
        run_subject(f'{tag}_full', lambda u, nu, q=query, d=data, c=cold: q(u, nu, d, c), meta,
                    spec.get('repetitions', cfg['rom_repetitions']))
        del Q, bank, data, cold, query
        jax.clear_caches()

    R_['complete'] = True
    save()
    print('SOLVE-DONE', flush=True)


if __name__ == '__main__':
    main()
