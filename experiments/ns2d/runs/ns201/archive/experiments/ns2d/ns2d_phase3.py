"""ns2d_phase3.py -- Phase 3 driver: the correction ladder, POD-LSPG and the FOM tolerance
ladder on one mesh with one head (DESIGN.md "Phase 3", gates R-*).

    CKPT=ckpt_K16_R256.pkl N=256 Q_LADDER=0,16,64,256 PODK=16,64,256 MATCHED_POD=1
    FOM_NTOLS=3e-2,1e-2,3e-3,1e-3,1e-4,1e-6  REPS=3 BURN=0.25 CASES=8
    RES_SNAPSHOTS=1024 RES_STARTS=4 RES_BUDGET=200 RES_SEED=20260915
    IC_BUDGET=400 STEP_BUDGET=200 GTOL=1e-6  EXPECT_HASHES=...  SMOKE=0 OUT=output

Everything reported is written incrementally to output/result.json; the ROM fields of the
last timed repetition and the converged FOM reference of every case are saved for the
NumPy audit (audit_phase3.py).
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
import ns2d_rom as RM                                                # noqa: E402

E = os.environ.get
CKPT = E('CKPT', 'ckpt_K16_R256.pkl')
N = int(E('N', '256'))
Q_LADDER = [int(v) for v in E('Q_LADDER', '0,16,64,256').split(',')]
PODK = [int(v) for v in E('PODK', '16,64,256').split(',')]
MATCHED_POD = int(E('MATCHED_POD', '1'))
FOM_NTOLS = [float(v) for v in E('FOM_NTOLS', '3e-2,1e-2,3e-3,1e-3,1e-4,1e-6').split(',')]
REPS, BURN, CASES = int(E('REPS', '3')), float(E('BURN', '0.25')), int(E('CASES', '8'))
RES_SNAPSHOTS, RES_STARTS, RES_BUDGET, RES_SEED = (int(E('RES_SNAPSHOTS', '1024')), int(E('RES_STARTS', '4')),
                                                   int(E('RES_BUDGET', '200')), int(E('RES_SEED', '20260915')))
IC_BUDGET, STEP_BUDGET, GTOL = int(E('IC_BUDGET', '400')), int(E('STEP_BUDGET', '200')), float(E('GTOL', '1e-6'))
DT, T, OUT_EVERY = float(E('DT', '2e-3')), float(E('T', '1.0')), int(E('OUT_EVERY', '20'))
N_TRAIN, N_DEV = int(E('TRAIN', '512')), int(E('DEV', '64'))
SEEDS = dict(train=int(E('SEED_TRAIN', '20260917')), dev=int(E('SEED_DEV', '20260918')))
NTOL, LTOL = float(E('NTOL', '1e-11')), float(E('LTOL', '1e-9'))
EXPECT = E('EXPECT_HASHES', '')
TQ_STATES = int(E('TQ_STATES', '32'))
T_CHUNK = int(E('T_CHUNK', '256'))
SMOKE = int(E('SMOKE', '0'))
OUT = Path(E('OUT', 'output'))
NSTEPS = int(round(T / DT))
NOUT = NSTEPS // OUT_EVERY + 1
EVAL_IDX = list(range(0, NOUT, max(1, (NOUT - 1) // 5)))         # 6 output times


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


def generate(phys_all, ntol, ltol):
    run, _ = F.make_fom(N, DT, NSTEPS, OUT_EVERY)
    U = np.empty((len(phys_all), NOUT, N * N))
    worst, mit = 0.0, 0
    t0 = time.time()
    for i, p in enumerate(phys_all):
        st, it, rn = host(run(jnp.asarray(F.initial(N, p)), float(p[12]), ntol, ltol))
        assert np.isfinite(st).all()
        U[i] = st.reshape(NOUT, -1)
        worst, mit = max(worst, float(rn.max())), max(mit, int(it.max()))
    del run
    jax.clear_caches()
    return U, dict(N=N, trajectories=len(phys_all), worst_rel_residual=worst, newton_max_it=mit,
                   seconds=time.time() - t0, sha256=sha(U))


def residual_directions(params, Rb, coef, Zcand, K, report):
    """Nested correction directions: field-metric POD of eta - h(z*) on training snapshots."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(RES_SEED)
    idx = np.sort(rng.choice(len(coef), min(RES_SNAPSHOTS, len(coef)), replace=False))
    target = jnp.asarray(np.asarray(coef)[idx])
    head_fn = lambda z: D.head(params, z)                                # noqa: E731
    Z, rn, its, reasons = D.oracle_fit(head_fn, Rb, target, Zcand, n_starts=RES_STARTS,
                                       budget=RES_BUDGET, gtol=GTOL)
    rho = target - jax.jit(jax.vmap(head_fn))(jnp.asarray(Z))
    tilde = rho @ Rb.T
    _, sv, Vt = jnp.linalg.svd(tilde, full_matrices=False)
    rank = int(min(Vt.shape[0], coef.shape[1]))
    Ct = Vt[:rank].T
    C = jnp.linalg.solve(Rb, Ct)
    total = float(jnp.sum(sv ** 2))
    fitted = np.sqrt(np.asarray(jnp.sum(tilde * tilde, 1) / jnp.maximum(jnp.sum((target @ Rb.T) ** 2, 1), 1e-300)))
    info = dict(rule='field-metric POD of eta - h(z*), z* best-found by multi-start LM; nested in q',
                snapshots=int(len(idx)), starts=RES_STARTS, budget=RES_BUDGET, seed=RES_SEED,
                available_rank=rank, head_fit_relative_median=float(np.median(fitted)),
                head_fit_relative_worst=float(np.max(fitted)),
                reasons={str(k): int(v) for k, v in zip(*np.unique(reasons, return_counts=True))},
                singular_values=np.asarray(sv[:min(len(sv), 300)]).tolist(),
                residual_energy_captured={str(q): float(np.sum(np.asarray(sv[:q]) ** 2)) / max(total, 1e-300)
                                          for q in Q_LADDER},
                seconds=time.perf_counter() - t0, directions_sha256=sha(np.asarray(C)))
    return C, info


def pod_snapshots(U, kmax):
    Ud = jnp.asarray(U)
    w, V = jnp.linalg.eigh(Ud @ Ud.T)
    idx = jnp.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    s = jnp.sqrt(jnp.maximum(w, 0.0))
    Vk = (Ud.T @ V[:, :kmax]) / s[None, :kmax]
    Vk, _ = jnp.linalg.qr(Vk)
    return Vk, np.asarray(s)


def main():
    t_begin = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend()
    ck = pickle.load(open(CKPT, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    Ztr = np.asarray(ck['Z_tr'])
    K, R = int(ck['cfg']['K']), int(ck['cfg']['R'])
    report = dict(lane='ns2d', phase=3, commit=git_commit(), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=backend, x64=bool(jax.config.jax_enable_x64),
                  matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                  jax_version=jax.__version__, host=platform.node(), device=str(jax.devices()[0]),
                  gpu=jax.devices()[0].device_kind, smoke=bool(SMOKE),
                  checkpoint=dict(path=CKPT, sha256=hashlib.sha256(Path(CKPT).read_bytes()).hexdigest(),
                                  cfg=ck['cfg'], K=K, R=R),
                  config=dict(N=N, Q_LADDER=Q_LADDER, PODK=PODK, MATCHED_POD=MATCHED_POD, FOM_NTOLS=FOM_NTOLS,
                              REPS=REPS, BURN=BURN, CASES=CASES, RES_SNAPSHOTS=RES_SNAPSHOTS,
                              RES_STARTS=RES_STARTS, RES_BUDGET=RES_BUDGET, RES_SEED=RES_SEED,
                              IC_BUDGET=IC_BUDGET, STEP_BUDGET=STEP_BUDGET, GTOL=GTOL, DT=DT, T=T,
                              OUT_EVERY=OUT_EVERY, NOUT=NOUT, EVAL_IDX=EVAL_IDX, N_TRAIN=N_TRAIN,
                              N_DEV=N_DEV, SEEDS=SEEDS, NTOL=NTOL, LTOL=LTOL, T_CHUNK=T_CHUNK),
                  timing_contract=('dense initial vorticity on device to NOUT dense device fields, '
                                   'block_until_ready; identical for every ROM arm and the FOM ladder; '
                                   'the bank, tensor, directions and POD are built untimed'),
                  gates={}, data={}, subjects=[], invocations=[], directions={}, fom={}, complete=False)
    dump(report)
    gate(report, 'S0', (backend == 'gpu' and jax.config.jax_enable_x64
                        and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest') or bool(SMOKE),
         backend=backend)
    gate(report, 'R-CKPT', (not ck['cfg'].get('smoke', True)) or bool(SMOKE), ckpt_smoke=ck['cfg'].get('smoke'))
    expect = json.loads(Path(EXPECT).read_text()) if EXPECT and Path(EXPECT).exists() else {}
    phys_tr = F.params_draw(SEEDS['train'], N_TRAIN)
    phys_dev = F.params_draw(SEEDS['dev'], N_DEV)[:CASES]

    # ---- data: training snapshots (POD, directions) and the converged references
    Utr, itr = generate(phys_tr, NTOL, LTOL)
    report['data'][f'train_N{N}'] = itr
    want = expect.get(f'train_N{N}')
    gate(report, f'R-DATA_train_N{N}', (want is None and bool(SMOKE)) or want == itr['sha256'],
         expected=want, got=itr['sha256'])
    S = Utr.shape[0] * NOUT
    Uflat = Utr.reshape(S, -1)
    del Utr
    Uref, iref = generate(phys_dev, NTOL, LTOL)                       # converged same-grid reference
    report['data'][f'reference_N{N}'] = iref
    np.savez_compressed(OUT / f'reference_N{N}.npz', U=Uref[:, EVAL_IDX], physical=phys_dev,
                        eval_idx=np.asarray(EVAL_IDX))
    dump(report)

    # ---- the bank, its QR, the training coefficients, the directions
    t0 = time.perf_counter()
    G = D.bank_on_grid(params, N)
    Qb, Rb = jnp.linalg.qr(G, mode='reduced')
    dd = jnp.abs(jnp.diag(Rb))
    gate(report, 'R-BANK', int(jnp.sum(dd > dd.max() * 1e-12)) == R, cond_Rb=float(dd.max() / dd.min()))
    coef = np.concatenate([np.asarray(jnp.linalg.solve(Rb, Qb.T @ jnp.asarray(Uflat[s:s + 2048]).T).T)
                           for s in range(0, S, 2048)])
    report['bank'] = dict(seconds=time.perf_counter() - t0, sha256=sha(np.asarray(G)),
                          projection_rms=float(np.sqrt(np.mean([
                              float(jnp.linalg.norm(jnp.asarray(Uflat[s:s + 2048]) - (jnp.asarray(Uflat[s:s + 2048]) @ Qb) @ Qb.T) ** 2
                                    / jnp.linalg.norm(jnp.asarray(Uflat[s:s + 2048])) ** 2) for s in range(0, S, 2048)]))))
    stride = max(1, len(Ztr) // 8192)
    Zcand = Ztr[::stride]
    Cdir, dinfo = residual_directions(params, Rb, coef, Zcand, K, report)
    report['directions'] = dinfo
    gate(report, 'R-DIR', dinfo['available_rank'] >= max(Q_LADDER), available_rank=dinfo['available_rank'],
         head_fit_median=dinfo['head_fit_relative_median'])
    dump(report)

    # ---- POD of the training snapshots
    kmax = max(PODK + ([K + q for q in Q_LADDER if q > 0] if MATCHED_POD else []))
    Vpod, sv = pod_snapshots(Uflat, min(kmax, S))
    report['pod'] = dict(kmax=int(Vpod.shape[1]), singular_values=sv[:min(len(sv), 300)].tolist())
    del Uflat
    jax.clear_caches()

    # ---- test modes and tensors (largest M once; rows nested)
    M_neural = 4 * (K + max(Q_LADDER))
    pod_ks = sorted(set(PODK + ([K + q for q in Q_LADDER if q > 0] if MATCHED_POD else [])))
    M_pod = 4 * max(pod_ks)
    Phi_n, lam_n, _ = RM.fourier_modes(N, M_neural)
    Phi_p, lam_p, _ = RM.fourier_modes(N, M_pod)
    t0 = time.perf_counter()
    T_n = RM.build_T(Phi_n, G, N, chunk=T_CHUNK)
    t_tn = time.perf_counter() - t0
    T_n_rev = RM.build_T(Phi_n, G, N, chunk=T_CHUNK * 2 if not SMOKE else 64, reverse=True)
    tb = float(np.linalg.norm(T_n - T_n_rev) / np.linalg.norm(T_n))
    del T_n_rev
    gate(report, 'R-TB', tb <= 1e-12, rel=tb, M=M_neural, R=R, seconds=t_tn, bytes=int(T_n.nbytes))
    t0 = time.perf_counter()
    T_p = RM.build_T(Phi_p, Vpod, N, chunk=T_CHUNK)
    report['tensor'] = dict(neural=dict(M=M_neural, R=R, seconds=t_tn, bytes=int(T_n.nbytes)),
                            pod=dict(M=M_pod, k=int(Vpod.shape[1]), seconds=time.perf_counter() - t0,
                                     bytes=int(T_p.nbytes)))
    # R-TQ: tensor vs the dense oracle at held-out latent states (dev coefficients)
    A_n = np.asarray(jnp.asarray(Phi_n).T @ G)
    Qn = RM.symmetrize(T_n)
    worst_tq, worst_lin = 0.0, 0.0
    rng = np.random.default_rng(3)
    for _ in range(TQ_STATES):
        c_i, t_i = rng.integers(len(phys_dev)), rng.integers(NOUT)
        u = jnp.asarray(Uref[c_i, t_i])
        c = np.asarray(jnp.linalg.solve(Rb, Qb.T @ u))
        q_t = 0.5 * ((Qn @ c) @ c)
        q_d = np.asarray(RM.dense_adv_projected(Phi_n, G, N, c))
        worst_tq = max(worst_tq, float(np.linalg.norm(q_t - q_d) / np.linalg.norm(q_d)))
        lin_t = -lam_n * (A_n @ c)
        lin_d = np.asarray(jnp.asarray(Phi_n).T @ F.laplacian((G @ jnp.asarray(c)).reshape(N, N), N).ravel())
        worst_lin = max(worst_lin, float(np.linalg.norm(lin_t - lin_d) / np.linalg.norm(lin_d)))
    gate(report, 'R-TQ', worst_tq <= 1e-10, worst_rel=worst_tq, states=TQ_STATES)
    gate(report, 'R-LIN', worst_lin <= 1e-12, worst_rel=worst_lin)
    dump(report)

    # ---- subjects
    subjects = []
    head_fns = {}
    Hc = jax.jit(jax.vmap(lambda z: D.head(params, z)))(jnp.asarray(Zcand))
    Hrot_n = Hc @ Rb.T
    Hn_n = jnp.sum(Hrot_n * Hrot_n, 1)
    A_p = jnp.asarray(Phi_p).T @ Vpod
    Qp = RM.symmetrize(T_p)
    for q in Q_LADDER:
        M = 4 * (K + q)
        name = f'neural_q{q}'
        Cq = Cdir[:, :q]

        def head_q(w, Cq=Cq, K=K):
            return D.head(params, w[:K]) + Cq @ w[K:]

        ops = dict(B=G, Qb=Qb, Rb=Rb, A=jnp.asarray(A_n[:M]), Q=jnp.asarray(Qn[:M]), lam=jnp.asarray(lam_n[:M]),
                   Zcand=jnp.asarray(Zcand), Hrot=Hrot_n, Hn=Hn_n, Cq=Cq)
        subjects.append(dict(name=name, arm='neural', q=q, K=K, D=R, M=M, unknowns=K + q,
                             query=RM.make_query(head_q, K, q, None, DT, NSTEPS, OUT_EVERY, IC_BUDGET,
                                                 STEP_BUDGET, GTOL), ops=ops))
    for k in pod_ks:
        M = 4 * k
        V = Vpod[:, :k]
        ops = dict(B=V, Qb=V, Rb=jnp.eye(k, dtype=jnp.float64), A=A_p[:M, :k], Q=jnp.asarray(Qp[:M, :k, :k]),
                   lam=jnp.asarray(lam_p[:M]), Zcand=jnp.zeros((1, k)), Hrot=jnp.zeros((1, k)),
                   Hn=jnp.zeros((1,)), Cq=jnp.zeros((k, 0)))
        subjects.append(dict(name=f'pod_k{k}', arm='pod', q=0, K=k, D=k, M=M, unknowns=k,
                             query=RM.make_query(lambda w: w, k, 0, None, DT, NSTEPS, OUT_EVERY, IC_BUDGET,
                                                 STEP_BUDGET, GTOL), ops=ops))
    report['subjects'] = [{k: v for k, v in s.items() if k not in ('query', 'ops')} for s in subjects]
    dump(report)

    # ---- timed invocations: reps outermost, cases, subjects innermost in AB/BA order
    w0s = [jnp.asarray(Uref[c, 0].reshape(N, N)) for c in range(len(phys_dev))]
    nus = [float(p[12]) for p in phys_dev]
    saved = {}
    for rep in range(REPS + 1):                                    # rep 0 = warm/compile, untimed
        for c in range(len(phys_dev)):
            order = subjects if (rep + c) % 2 == 0 else list(reversed(subjects))
            for s in order:
                RM.burn(BURN)
                t0 = time.perf_counter()
                out = s['query'](w0s[c], nus[c], s['ops'])
                jax.block_until_ready(out)
                sec = time.perf_counter() - t0
                fields, it, rn, reason, W, icit, icreason, gn, icgn, icrn = host(out)
                err = RM.errors(fields[EVAL_IDX], Uref[c, EVAL_IDX].reshape(len(EVAL_IDX), N, N))
                inv = dict(subject=s['name'], case=c, rep=rep, timed=rep > 0, seconds=sec, **err,
                           iterations_total=int(it.sum()), iterations_max=int(it.max()),
                           reasons={str(k): int(v) for k, v in zip(*np.unique(reason, return_counts=True))},
                           budget_exits=int((reason == 0).sum()), worst_step_gradient=float(gn.max()),
                           ic_iterations=int(icit), ic_reason=int(icreason), ic_residual=float(icrn),
                           finite=bool(np.isfinite(fields).all()))
                report['invocations'].append(inv)
                if rep == REPS:
                    saved[(s['name'], c)] = fields[EVAL_IDX]
                log(f'INV rep{rep} case{c} {s["name"]:14s} {sec:8.3f}s worst_ev {err["worst_evolved"]:.3e} '
                    f't0 {err["t0"]:.3e} budget_exits {inv["budget_exits"]}')
            dump(report)
    np.savez_compressed(OUT / f'rom_fields_N{N}.npz',
                        **{f'{k[0]}__case{k[1]}': v for k, v in saved.items()})

    # ---- FOM ladder, timed in the same job with the same contract
    for ntol in FOM_NTOLS + [NTOL]:
        ltol = 0.1 * ntol if ntol > NTOL else LTOL
        run, _ = F.make_fom(N, DT, NSTEPS, OUT_EVERY)
        name = f'fom_ntol{ntol:g}'
        for rep in range(REPS + 1):
            for c in range(len(phys_dev)):
                RM.burn(BURN)
                t0 = time.perf_counter()
                out = run(w0s[c], nus[c], ntol, ltol)
                jax.block_until_ready(out)
                sec = time.perf_counter() - t0
                st, it, rn = host(out)
                err = RM.errors(st[EVAL_IDX], Uref[c, EVAL_IDX].reshape(len(EVAL_IDX), N, N))
                report['invocations'].append(dict(subject=name, case=c, rep=rep, timed=rep > 0, seconds=sec,
                                                  **err, iterations_total=int(it.sum()), iterations_max=int(it.max()),
                                                  worst_rel_residual=float(rn.max()), ntol=ntol, ltol=ltol,
                                                  finite=bool(np.isfinite(st).all())))
        log(f'FOM {name} done')
        del run
        jax.clear_caches()
        dump(report)

    # ---- aggregates and the pre-registered verdict
    def agg(name):
        inv = [i for i in report['invocations'] if i['subject'] == name and i['timed']]
        last = {i['case']: i for i in inv if i['rep'] == REPS}
        secs = [i['seconds'] for i in inv]
        return dict(subject=name, median_seconds=float(np.median(secs)), all_seconds=secs,
                    worst_evolved=float(max(i['worst_evolved'] for i in last.values())),
                    median_evolved=float(np.median([i['worst_evolved'] for i in last.values()])),
                    worst_all=float(max(i['worst_all'] for i in last.values())),
                    worst_t0=float(max(i['t0'] for i in last.values())),
                    median_t0=float(np.median([i['t0'] for i in last.values()])),
                    budget_exits=int(sum(i.get('budget_exits', 0) for i in last.values())),
                    finite=bool(all(i['finite'] for i in last.values())))
    names = [s['name'] for s in subjects] + [f'fom_ntol{v:g}' for v in FOM_NTOLS + [NTOL]]
    report['aggregates'] = {n: agg(n) for n in names}
    ladder = [report['aggregates'][f'neural_q{q}'] for q in Q_LADDER]
    we = [x['worst_evolved'] for x in ladder]
    mono = all(b <= a * (1 + 1e-12) for a, b in zip(we, we[1:]))
    gain = we[0] / we[-1] if we[-1] > 0 else float('inf')
    cost = ladder[-1]['median_seconds'] / ladder[0]['median_seconds']
    conv = all(x['budget_exits'] == 0 for x in ladder)
    gate(report, 'R-CONV', conv, budget_exits=[x['budget_exits'] for x in ladder])
    gate(report, 'R-LADDER', mono and gain >= 2.0, monotone=mono, gain_top_over_q0=gain,
         cost_ratio_top_over_q0=cost, worst_evolved=we, q=Q_LADDER)
    report['all_passed'] = all(g['passed'] for g in report['gates'].values())
    report['complete'] = not SMOKE
    report['seconds'] = time.time() - t_begin
    dump(report)
    log(f'PHASE3 done in {report["seconds"]:.0f}s; all_passed={report["all_passed"]}; '
        f'failed={[k for k, g in report["gates"].items() if not g["passed"]]}')


if __name__ == '__main__':
    main()
