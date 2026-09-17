"""ns2d_phase2.py -- Phase 2 driver: bank + head on the regenerated dataset (DESIGN.md).

    K=16 R=512 TRAIN_N=256 EVAL_NS=64,128,256 STEPS=30000 LR=1e-3 BATCH=0 SEED=0
    EXPECT_HASHES=configs/phase1-hashes.json   (Phase-1 cohort hashes; asserted)
    ORACLE_CASES=64 ORACLE_TIMES=0,5,10,15,20,25  SMOKE=0  OUT=output

Steps: regenerate train/dev at TRAIN_N (assert Phase-1 hashes) -> train the periodic
separable auto-decoder -> bank on each EVAL_N grid (mesh-free transfer) -> classical POD of
the training snapshots (method of snapshots) -> floors on held-out dev trajectories (bank
projection, POD-R, POD-K) -> head oracle (best-found multi-start LM) vs POD-K -> the
query-time initialiser fit (single LM from the best-scoring code) vs the oracle.  Every
gate is a NUMBER with its threshold; the checkpoint is written for Phase 3.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                              # noqa: E402

import ns2d_fom as F                                                 # noqa: E402
import ns2d_decoder as D                                             # noqa: E402

E = os.environ.get
K, R = int(E('K', '16')), int(E('R', '256'))
TRAIN_N = int(E('TRAIN_N', '256'))
EVAL_NS = [int(v) for v in E('EVAL_NS', '64,128,256').split(',')]
STEPS, LR, BATCH, SEED = int(E('STEPS', '30000')), float(E('LR', '1e-3')), int(E('BATCH', '0')), int(E('SEED', '0'))
LAM_ORTH = float(E('LAM_ORTH', '1e-4'))
ARCH = dict(n_ff=int(E('N_FF', '64')), kff=int(E('KFF', '6')), g_hidden=int(E('G_HIDDEN', '128')),
            g_layers=int(E('G_LAYERS', '2')), h_hidden=int(E('H_HIDDEN', '128')), h_layers=int(E('H_LAYERS', '2')))
DT, T, OUT_EVERY = float(E('DT', '2e-3')), float(E('T', '1.0')), int(E('OUT_EVERY', '20'))
N_TRAIN, N_DEV = int(E('TRAIN', '512')), int(E('DEV', '64'))
SEEDS = dict(train=int(E('SEED_TRAIN', '20260917')), dev=int(E('SEED_DEV', '20260918')))
NTOL, LTOL = float(E('NTOL', '1e-11')), float(E('LTOL', '1e-9'))
EXPECT = E('EXPECT_HASHES', '')
ORACLE_CASES = int(E('ORACLE_CASES', '64'))
ORACLE_TIMES = [int(v) for v in E('ORACLE_TIMES', '0,5,10,15,20,25').split(',')]
ORACLE_STARTS, ORACLE_BUDGET = int(E('ORACLE_STARTS', '8')), int(E('ORACLE_BUDGET', '300'))
PODK_LIST = [int(v) for v in E('PODK', f'{K},{R}').split(',')]
SMOKE = int(E('SMOKE', '0'))
OUT = Path(E('OUT', 'output'))
NSTEPS = int(round(T / DT))
NOUT = NSTEPS // OUT_EVERY + 1


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def log(*a):
    print(*a, flush=True)


def dump(report):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def gate(report, name, passed, **kw):
    kw['passed'] = bool(passed)
    report['gates'][name] = kw
    log(f'GATE {name}: {"PASS" if passed else "FAIL"} ' +
        ' '.join(f'{k}={v:.3e}' if isinstance(v, float) else f'{k}={v}'
                 for k, v in kw.items() if k != 'passed' and not isinstance(v, (list, dict))))


def git_commit():
    c = os.environ.get('SOURCE_COMMIT')
    if c:
        return c
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:                                                 # pragma: no cover
        return 'unknown'


def generate(N, phys_all):
    run, _ = F.make_fom(N, DT, NSTEPS, OUT_EVERY)
    U = np.empty((len(phys_all), NOUT, N * N))
    worst, mit = 0.0, 0
    t0 = time.time()
    for i, p in enumerate(phys_all):
        st, it, rn = host(run(jnp.asarray(F.initial(N, p)), float(p[12]), NTOL, LTOL))
        assert np.isfinite(st).all()
        U[i] = st.reshape(NOUT, -1)
        worst, mit = max(worst, float(rn.max())), max(mit, int(it.max()))
    del run
    jax.clear_caches()
    return U, dict(N=N, trajectories=len(phys_all), states=NOUT, worst_rel_residual=worst,
                   newton_max_it=mit, seconds=time.time() - t0, sha256=sha(U))


def pod_snapshots(U, kmax):
    """Method of snapshots on the GPU: U (S, n) -> V (n, kmax) orthonormal, singular values."""
    Ud = jnp.asarray(U)
    Gm = Ud @ Ud.T                                        # (S, S)
    w, V = jnp.linalg.eigh(Gm)
    idx = jnp.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    s = jnp.sqrt(jnp.maximum(w, 0.0))
    Vk = (Ud.T @ V[:, :kmax]) / s[None, :kmax]
    Vk, _ = jnp.linalg.qr(Vk)                              # one reorthogonalisation
    return Vk, np.asarray(s)


def floors_on_dev(Q, Udev, ks):
    """Projection floors of held-out trajectories (T, NOUT, n) on orthonormal Q (n, D):
    for each k in ks (prefix of columns), relative error per state, normalised by the
    initial state (Burgers convention) and by the current state."""
    out = {}
    Qd = jnp.asarray(Q)
    for k in ks:
        Qk = Qd[:, :k]
        fixed, cur = [], []
        for t in range(Udev.shape[0]):
            X = jnp.asarray(Udev[t])                        # (NOUT, n)
            Pe = X - (X @ Qk) @ Qk.T
            e = jnp.linalg.norm(Pe, axis=1)
            fixed.append(np.asarray(e / jnp.linalg.norm(X[0])))
            cur.append(np.asarray(e / jnp.linalg.norm(X, axis=1)))
        fixed, cur = np.stack(fixed), np.stack(cur)
        out[str(k)] = dict(k=k, worst_all_times_fixed=float(fixed.max()),
                           median_case_worst_fixed=float(np.median(fixed.max(1))),
                           worst_evolved_fixed=float(fixed[:, 1:].max()),
                           t0_fixed_worst=float(fixed[:, 0].max()),
                           t0_fixed_median=float(np.median(fixed[:, 0])),
                           worst_current=float(cur.max()),
                           median_case_worst_current=float(np.median(cur.max(1))),
                           per_case_worst_fixed=fixed.max(1).tolist())
    return out


def main():
    t_begin = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend()
    report = dict(lane='ns2d', phase=2, commit=git_commit(), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=backend, x64=bool(jax.config.jax_enable_x64),
                  matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                  jax_version=jax.__version__, host=platform.node(), device=str(jax.devices()[0]),
                  smoke=bool(SMOKE),
                  config=dict(K=K, R=R, TRAIN_N=TRAIN_N, EVAL_NS=EVAL_NS, STEPS=STEPS, LR=LR,
                              BATCH=BATCH, SEED=SEED, LAM_ORTH=LAM_ORTH, ARCH=ARCH, DT=DT, T=T,
                              OUT_EVERY=OUT_EVERY, N_TRAIN=N_TRAIN, N_DEV=N_DEV, SEEDS=SEEDS,
                              NTOL=NTOL, LTOL=LTOL, ORACLE_CASES=ORACLE_CASES,
                              ORACLE_TIMES=ORACLE_TIMES, ORACLE_STARTS=ORACLE_STARTS,
                              ORACLE_BUDGET=ORACLE_BUDGET, PODK=PODK_LIST),
                  gates={}, data={}, training={}, floors={}, oracle={}, complete=False)
    dump(report)
    gate(report, 'S0', (backend == 'gpu' and jax.config.jax_enable_x64
                        and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest') or bool(SMOKE),
         backend=backend)
    expect = json.loads(Path(EXPECT).read_text()) if EXPECT and Path(EXPECT).exists() else {}
    phys = dict(train=F.params_draw(SEEDS['train'], N_TRAIN), dev=F.params_draw(SEEDS['dev'], N_DEV))

    # ---- data at the training mesh
    Utr, itr = generate(TRAIN_N, phys['train'])
    Udev, idev = generate(TRAIN_N, phys['dev'])
    report['data'][f'train_N{TRAIN_N}'], report['data'][f'dev_N{TRAIN_N}'] = itr, idev
    for name, info in (('train', itr), ('dev', idev)):
        want = expect.get(f'{name}_N{TRAIN_N}')
        gate(report, f'B-DATA_{name}_N{TRAIN_N}', (want is None and bool(SMOKE)) or want == info['sha256'],
             expected=want, got=info['sha256'], seconds=info['seconds'])
    dump(report)
    log(f'DATA train {Utr.shape} dev {Udev.shape} in {itr["seconds"]+idev["seconds"]:.0f}s')

    # ---- train
    S = N_TRAIN * NOUT
    Uflat = Utr.reshape(S, -1)
    params, Z, tinfo = D.train_autodecoder(jax.random.PRNGKey(SEED), TRAIN_N, Uflat, K, R,
                                           steps=STEPS, lr=LR, lam_orth=LAM_ORTH, batch=BATCH,
                                           tag=f'K{K}R{R}', **ARCH)
    report['training'] = tinfo
    gate(report, 'H-TRAIN', np.isfinite(tinfo['recon_rel_l2_max']) and tinfo['recon_rel_l2_max'] < 1.0,
         recon_mean=tinfo['recon_rel_l2_mean'], recon_median=tinfo['recon_rel_l2_median'],
         recon_max=tinfo['recon_rel_l2_max'], seconds=tinfo['seconds'])
    ck = dict(params=host(params), Z_tr=np.asarray(Z), cfg=dict(K=K, R=R, TRAIN_N=TRAIN_N, ARCH=ARCH,
              STEPS=STEPS, LR=LR, LAM_ORTH=LAM_ORTH, SEED=SEED, smoke=bool(SMOKE),
              train_sha256=itr['sha256'], commit=report['commit']))
    with open(OUT / f'ckpt_K{K}_R{R}.pkl', 'wb') as f:
        pickle.dump(ck, f)
    report['checkpoint'] = dict(path=f'ckpt_K{K}_R{R}.pkl',
                                sha256=hashlib.sha256((OUT / f'ckpt_K{K}_R{R}.pkl').read_bytes()).hexdigest())
    dump(report)

    # ---- classical POD of the training snapshots at the training mesh (kept for floors)
    t0 = time.time()
    Vpod, sv = pod_snapshots(Uflat, max(PODK_LIST))
    report['pod'] = dict(N=TRAIN_N, seconds=time.time() - t0, singular_values=sv[:max(PODK_LIST)].tolist(),
                         energy_fraction={str(k): float((sv[:k] ** 2).sum() / (sv ** 2).sum()) for k in PODK_LIST})
    del Uflat
    head_fn = lambda z: D.head(params, z)                               # noqa: E731

    # ---- floors and oracle on every evaluation mesh (dev regenerated per mesh)
    for N in EVAL_NS:
        if N == TRAIN_N:
            Ud = Udev
        else:
            Ud, info = generate(N, phys['dev'])
            report['data'][f'dev_N{N}'] = info
            want = expect.get(f'dev_N{N}')
            gate(report, f'B-DATA_dev_N{N}', (want is None and bool(SMOKE)) or want == info['sha256'],
                 expected=want, got=info['sha256'])
        G = D.bank_on_grid(params, N)
        Qb, Rb = jnp.linalg.qr(G, mode='reduced')
        d = jnp.abs(jnp.diag(Rb))
        rank = int(jnp.sum(d > d.max() * 1e-12))
        cond = float(d.max() / d.min())
        gate(report, f'B-ORTH_N{N}', rank == R and np.isfinite(cond), rank=rank, cond_Rb=cond,
             mean_abs_colmean=float(jnp.mean(jnp.abs(jnp.mean(G, 0)))))
        fl = dict(bank=floors_on_dev(Qb, Ud, [R]))
        if N == TRAIN_N:
            fl['pod'] = floors_on_dev(Vpod, Ud, PODK_LIST)
        else:
            Vn, _ = pod_snapshots(generate(N, phys['train'])[0].reshape(S, -1), max(PODK_LIST))
            fl['pod'] = floors_on_dev(Vn, Ud, PODK_LIST)
        report['floors'][str(N)] = fl
        bank_w, podR_w = fl['bank'][str(R)]['worst_evolved_fixed'], fl['pod'][str(R)]['worst_evolved_fixed']
        gate(report, f'B-FLOOR_N{N}', bank_w <= 2.0 * podR_w, bank_worst_evolved=bank_w,
             podR_worst_evolved=podR_w, ratio=bank_w / podR_w,
             bank_median=fl['bank'][str(R)]['median_case_worst_fixed'],
             podR_median=fl['pod'][str(R)]['median_case_worst_fixed'],
             podK_worst_evolved=fl['pod'][str(K)]['worst_evolved_fixed'],
             podK_median=fl['pod'][str(K)]['median_case_worst_fixed'])
        dump(report)

        # oracle on held-out states: cases x times
        cases = list(range(min(ORACLE_CASES, Ud.shape[0])))
        X = np.stack([Ud[c, t] for c in cases for t in ORACLE_TIMES])           # (S_o, n)
        n0 = np.repeat(np.linalg.norm(Ud[cases][:, 0], axis=1), len(ORACLE_TIMES))
        Xd = jnp.asarray(X)
        Cc = jnp.linalg.solve(Rb, (Qb.T @ Xd.T)).T                                 # (S_o, R) bank coefficients
        perp = np.asarray(jnp.linalg.norm(Xd - (Xd @ Qb) @ Qb.T, axis=1))
        t0 = time.time()
        Zo, rn, its, reasons = D.oracle_fit(head_fn, Rb, Cc, Z, n_starts=ORACLE_STARTS,
                                            budget=ORACLE_BUDGET)
        e_or = np.sqrt(rn ** 2 + perp ** 2) / n0                                    # field error incl. floor
        # single-start initialiser (the query-time policy): best-scoring code only
        Z1, rn1, its1, r1 = D.oracle_fit(head_fn, Rb, Cc, Z, n_starts=1, budget=ORACLE_BUDGET)
        e_1 = np.sqrt(rn1 ** 2 + perp ** 2) / n0
        # POD-K linear floor on the SAME states, same normalisation
        Vk = (Vpod if N == TRAIN_N else Vn)[:, :K]
        e_podk = np.asarray(jnp.linalg.norm(Xd - (Xd @ Vk) @ Vk.T, axis=1)) / n0
        e_bank = perp / n0
        oi = dict(N=N, states=int(X.shape[0]), cases=cases, times=ORACLE_TIMES,
                  oracle_median=float(np.median(e_or)), oracle_worst=float(e_or.max()),
                  oracle_mean=float(e_or.mean()), single_start_median=float(np.median(e_1)),
                  single_start_worst=float(e_1.max()), podK_median=float(np.median(e_podk)),
                  podK_worst=float(e_podk.max()), bank_floor_median=float(np.median(e_bank)),
                  bank_floor_worst=float(e_bank.max()),
                  oracle_iters_median=float(np.median(its)), oracle_reasons={str(k): int(v) for k, v in
                                                                            zip(*np.unique(reasons, return_counts=True))},
                  seconds=time.time() - t0, per_state_oracle=e_or.tolist(), per_state_single=e_1.tolist(),
                  per_state_podK=e_podk.tolist(), per_state_bank=e_bank.tolist())
        report['oracle'][str(N)] = oi
        gate(report, f'H-ORACLE_N{N}', oi['oracle_median'] <= 0.5 * oi['podK_median']
             and oi['oracle_median'] >= oi['bank_floor_median'] * (1 - 1e-9),
             oracle_median=oi['oracle_median'], podK_median=oi['podK_median'],
             ratio_podK_over_oracle=oi['podK_median'] / oi['oracle_median'],
             bank_floor_median=oi['bank_floor_median'], oracle_worst=oi['oracle_worst'])
        gate(report, f'H-SOLVED_N{N}', oi['single_start_median'] <= 1.5 * oi['oracle_median'],
             single_start_median=oi['single_start_median'], oracle_median=oi['oracle_median'])
        np.savez_compressed(OUT / f'oracle_N{N}.npz', Z=Zo, Z1=Z1, cases=np.asarray(cases),
                            times=np.asarray(ORACLE_TIMES), e_oracle=e_or, e_single=e_1, e_podK=e_podk,
                            e_bank=e_bank, n0=n0)
        dump(report)
        del G, Qb, Rb, Xd, Cc
        jax.clear_caches()

    report['all_passed'] = all(g['passed'] for g in report['gates'].values())
    report['complete'] = not SMOKE
    report['seconds'] = time.time() - t_begin
    dump(report)
    log(f'PHASE2 done in {report["seconds"]:.0f}s; all_passed={report["all_passed"]}; '
        f'failed={[k for k, g in report["gates"].items() if not g["passed"]]}')


if __name__ == '__main__':
    main()
