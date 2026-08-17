"""BURGERS-2D cost-to-tolerance SURFACE: the (k, N) grid, one GPU, one process.

Every cell of the whole grid is measured SEQUENTIALLY IN ONE JOB ON ONE GPU.

WHAT ONE CELL IS
----------------
(method, N, k, M, m, tau) with method in {coord, pod}.  The ROM is the
REFERENCE implementation: `blat_common.make_weak_ops` builds the weak-form
operators (FOM-exact upwind advection inside the advection term -- operating
rule 7), and the per-step kernel is `blat_common`'s own `lm_step_jit`.  This
cell supplies only the per-step tolerance and the driver
(`ctol_tol.rollout_tau_burgers`).

The online path that is timed and graded is the WHOLE online solve:

    hyper-reduced cold start  ->  50 tau-stopped latent time steps

both stopped by the same rule, plus the 51-slice full-field decode for the
end-to-end number.  Hyper-reducing the cold start is operating rule 3; without
it the online path stays mesh-bound and the ROM's mesh independence is an
artefact of ignoring its own cold start.

TOLERANCE.  Per time step, ||r_n(z)|| <= tau * ||r_n(z_n)||: the relative
reduction of THAT step's objective from its own warm start.  The cold start
uses the same rule on the (hyper-reduced) initial-condition misfit.  Nothing in
the rule needs a held-out field, so it is deployable.  The FOM's own
backward-Euler residual at the ROM trajectory is reported per cell for
reference; it is never used to stop.

COST AND ACCURACY COME FROM THE SAME RUN.  For each held-out trajectory the
composite (cold start + rollout) is warmed TIME_WARM times and then run
TIME_REPS times under `block_until_ready`; the trajectory that is graded is the
one produced by the LAST TIMED REPETITION.

POD ARM.  Same weak objective, same test modes, same NNLS-EQ hyper-reduction
(fitted on POD-output snapshots), same LM solver, same tau-stopped cold start.
The POD basis is rebuilt at every mesh from the same training trajectories
(host f64 Gram, `blat_common.pod_basis`'s construction).

Usage:
  KS=2,4,6,8,12,16,24,32 NS=32,64,128,256 TAUS=1e-1,1e-2,1e-3 N=64 \
  PKL_DIR=../ckpt M=64 MQ=256 M_BIG=256 K_BIG=32 python ctol_burgers.py <out.json>
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(HERE, "deps", "burgers2d-rom-latent-stepping"),
           os.path.abspath(os.path.join(HERE, "..", "burgers2d-rom-latent-stepping"))):
    if os.path.isfile(os.path.join(_c, "blat_common.py")):
        sys.path.insert(0, _c)
        sys.path.insert(0, os.path.join(_c, "followup"))
        BLAT_DIR = _c
        break
else:
    raise ImportError("blat_common.py not found (deps/ or sibling experiment dir)")
sys.path.insert(0, HERE)

import blat_common as bc                                   # noqa: E402 (enables x64)
from blat_common import F64                                # noqa: E402
import fu_common as fu                                     # noqa: E402
import ctol_eq                                             # noqa: E402
import ctol_tol                                            # noqa: E402

OUT = sys.argv[1]
KS = [int(v) for v in os.environ.get("KS", "2,4,6,8,12,16,24,32").split(",") if v]
NS = [int(v) for v in os.environ.get("NS", "32,64,128,256").split(",") if v]
TAUS = [float(v) for v in os.environ.get("TAUS", "1e-1,1e-2,1e-3").split(",") if v]
M_MODES = int(os.environ.get("M", "64"))
MQ = int(os.environ.get("MQ", "256"))
M_BIG = int(os.environ.get("M_BIG", "256"))
K_BIG = int(os.environ.get("K_BIG", "32"))
MQ_SUPP = int(os.environ.get("MQ_SUPP", "1024"))
DO_SUPP = int(os.environ.get("DO_SUPP", "1"))
N_TEST = int(os.environ.get("CTOL_N_TEST", str(bc.N_TEST)))
N_POD_TRAJ = int(os.environ.get("N_POD_TRAJ", "128"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
WARM = int(os.environ.get("TIME_WARM", "2"))
PKL_DIR = os.environ.get("PKL_DIR", "../ckpt")
FOM_RES_TOL = float(os.environ.get("FOM_RES_TOL", "1e-8"))
GEN_CHUNK = int(os.environ.get("GEN_CHUNK", "16"))
POD_KMAX = max(KS)
T1 = bc.NUM_STEPS + 1
TAU_OK_STEP = ctol_tol.BURGERS_TAU_OK          # (1 tol, 4 tol_at_init)
TAU_OK_IC = ctol_tol.GENERIC_TAU_OK            # (2,) -- lm_tau_generic's code
# CONFIGS: a JSON list of {method, N, k, M, m, tau} cells to measure instead of the full
# grid.  Used by the single-GPU CONSOLIDATION job that re-times the per-(method, N)
# argmin configurations across ALL meshes in one process -- the only timing source the
# cross-N scaling figure may use (the fanned-out per-(pde, N) panel jobs are
# same-architecture but not the same physical GPU).
CONFIGS = os.environ.get("CONFIGS", "")
ARM_TAG = os.environ.get("ARM_TAG", "consolidated")
NODE = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME", "local")
log = bc.log


def build_plan():
    """plan[N][(k, M, m, arm_tag)][method] = [tau, ...]"""
    plan = {}
    if CONFIGS:
        for s_ in json.load(open(CONFIGS)):
            if s_.get("pde", "burgers2d") != "burgers2d":
                continue
            if s_["method"] not in ("coord", "pod"):
                continue
            key = (int(s_["k"]), int(s_["M"]), int(s_["m"]), s_.get("arm", ARM_TAG))
            d_ = plan.setdefault(int(s_["N"]), {}).setdefault(key, {})
            d_.setdefault(s_["method"], [])
            if float(s_["tau"]) not in d_[s_["method"]]:
                d_[s_["method"]].append(float(s_["tau"]))
        return plan
    for n_ in NS:
        arms = {}
        for k_ in KS:
            arms[(k_, M_BIG if k_ >= K_BIG else M_MODES, MQ, "primary")] = {
                "coord": list(TAUS), "pod": list(TAUS)}
        if DO_SUPP:
            for k_ in KS:
                if k_ >= K_BIG:
                    arms[(k_, M_BIG, MQ_SUPP, "supp_m4M")] = {
                        "coord": list(TAUS), "pod": list(TAUS)}
        plan[n_] = arms
    return plan


def git_commit():
    if os.environ.get("CTOL_COMMIT"):
        return os.environ["CTOL_COMMIT"]
    for d in (HERE, BLAT_DIR):
        try:
            return subprocess.check_output(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
    return "unknown"


def time_fn(fn, reps=None, warm=None):
    reps = TIME_REPS if reps is None else reps
    warm = WARM if warm is None else warm
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]


def gen_trajectories(n, cx, cy, w, a, nu, roll, chunk=None):
    """FOM trajectories at mesh n, in chunks (a 128x51x65536 f64 batch is 3.4 GB
    on device at N=256).  Aborts if the FOM's own Newton residual is too large --
    an unconverged 'truth' would silently poison every error column."""
    chunk = GEN_CHUNK if chunk is None else chunk
    outs, worst = [], 0.0
    for s in range(0, len(cx), chunk):
        e = min(s + chunk, len(cx))
        U0 = np.stack([bc.bf.blob_ic(n, cx[i], cy[i], w[i], a[i]) for i in range(s, e)])
        snaps, rr = roll(jnp.asarray(U0), jnp.asarray(nu[s:e]))
        outs.append(np.asarray(snaps).transpose(1, 0, 2))          # (b, T1, n^2)
        r = float(jnp.max(rr))
        if not np.isfinite(r) or r > worst:
            worst = r
    if not np.isfinite(worst) or worst > FOM_RES_TOL:
        raise SystemExit(f"N={n}: FOM Newton rel residual {worst:.2e} > {FOM_RES_TOL:.0e} "
                         f"-- refusing to grade or time against an unconverged baseline")
    return np.concatenate(outs, axis=0), worst


def make_fit_ic_tau(mis, K, budget):
    """Best-of-inits tau-stopped cold start.  `mis(z, u0q)` is the (weighted)
    misfit residual vector; the selection rule (smallest final ||r||) is
    `blat_common.fit_ic`'s and `fu_common.make_fit_ic_jit`'s."""
    def one(u0q, z0, tau):
        return ctol_tol.lm_tau_generic(lambda z: mis(z, u0q), K, budget)(z0, tau)

    def fit(u0q, Z0, tau):
        zs, rns, rn0s, nJs, nrs, accs, rejs, atts, reasons = jax.vmap(
            lambda z0: one(u0q, z0, tau))(Z0)
        b = jnp.argmin(rns)
        return (zs[b], rns[b], rn0s[b], jnp.sum(nJs), jnp.sum(atts), reasons[b], b)

    return jax.jit(fit)


def main():
    dev = jax.devices()[0]
    gpu_name = getattr(dev, "device_kind", str(dev))
    log(f"jax_backend={dev.platform} device={dev} gpu={gpu_name} "
        f"KS={KS} NS={NS} TAUS={TAUS} M={M_MODES} m={MQ} reps={TIME_REPS} warm={WARM} "
        f"ckpt_N={bc.N} n_test={N_TEST} n_pod_traj={N_POD_TRAJ}")
    commit = git_commit()
    plan = build_plan()
    ks_used = sorted({k_ for arms in plan.values() for (k_, _M, _m, _t) in arms})
    log(f"  plan: {len(plan)} mesh(es) {sorted(plan)}, k values {ks_used}, "
        f"{sum(len(v) for a_ in plan.values() for v in a_.values())} (arm, method) cells"
        + (f"  [CONFIGS={CONFIGS}]" if CONFIGS else ""))

    # ---------------- checkpoints (the k ladder, all trained at N=64) --------
    ck = {}
    for k in ks_used:
        p = os.path.join(PKL_DIR, f"blat_ad_N{bc.N}_K{k}.pkl")
        with open(p, "rb") as f:
            c = pickle.load(f)
        for key, val in (("bc_mode", bc.BC_MODE), ("N", bc.N), ("ad_hidden", bc.AD_HIDDEN),
                         ("ad_layers", bc.AD_LAYERS), ("n_train", bc.N_TRAIN), ("seed", bc.SEED)):
            if c["config"][key] != val:
                raise SystemExit(f"{os.path.basename(p)}: config mismatch on {key}: "
                                 f"{c['config'][key]} vs {val}")
        if c["k_lat"] != k:
            raise SystemExit(f"{p}: k_lat {c['k_lat']} != {k}")
        dec = bc.CoordDecoder(jax.tree_util.tree_map(jnp.asarray, c["params"]),
                              c["n_freq"], c["eps"], k)
        ck[k] = dict(cfg=c["config"], dec=dec, Ztr=np.asarray(c["Z_train"]),
                     fp=c["data_fingerprint"], path=os.path.basename(p))
        log(f"  ckpt k={k:2d}: {os.path.basename(p)} train_seed={c['config'].get('train_seed')}")
    fp0 = ck[ks_used[0]]["fp"]
    for k in ks_used[1:]:
        for key in ("sum", "sumsq"):
            if abs(ck[k]["fp"][key] - fp0[key]) / abs(fp0[key]) > 1e-6:
                raise SystemExit(f"k={k}: training-data fingerprint mismatch on {key} -- the "
                                 f"k ladder must be trained on one data draw")

    # family parameters (identical draw at every mesh)
    cxr, cyr, wr, ar, nur, _z = bc.bf.sample_params()
    cxt, cyt, wt, at, nut, _zt = bc.bf.sample_params(seed=bc.TEST_SEED, m=N_TEST)

    report = dict(
        config=dict(pde="burgers2d", ks=KS, ns=NS, taus=TAUS, M=M_MODES, m=MQ,
                    M_big=M_BIG, k_big=K_BIG, m_supp=MQ_SUPP, do_supp=DO_SUPP,
                    n_test=N_TEST, n_pod_traj=N_POD_TRAJ, num_steps=bc.NUM_STEPS,
                    dt=bc.DT, gn_budget=bc.GN_BUDGET, ic_budget=bc.IC_BUDGET,
                    time_reps=TIME_REPS, time_warm=WARM, seed=bc.SEED,
                    test_seed=bc.TEST_SEED, ckpt_N=bc.N,
                    cand_cap=ctol_eq.CAND_CAP, eq_snaps=ctol_eq.EQ_SNAPS,
                    eq_perturb=ctol_eq.EQ_PERTURB, eq_rows=ctol_eq.EQ_ROWS,
                    eq_seed=ctol_eq.EQ_SEED, eq_pool="grid",
                    objective="weak<M> with the FOM-exact upwind advection, "
                              "quadratured on m NNLS-EQ grid nodes",
                    stopping="per step: ||r_n(z)|| <= tau * ||r_n(z_n)||; "
                             "cold start: ||mis(z)|| <= tau * ||mis(z0)||",
                    cold_start="hyper-reduced on the cell's own EQ nodes, best of "
                               "{mean t=0 latent, nearest-IC training latent}",
                    x64=True,
                    matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
                    backend=dev.platform, device=str(dev), gpu=gpu_name, commit=commit,
                    slurm_job=os.environ.get("SLURM_JOB_ID"),
                    time_ms_definition="cold start + latent rollout + 51-slice decode",
                    train_data_fingerprint=fp0,
                    node=NODE, configs=CONFIGS or None, arm_tag=ARM_TAG,
                    ns_measured=sorted(plan), ks_measured=ks_used,
                    ckpts={k: ck[k]["path"] for k in ks_used}),
        rows=[], fom=[], supplementary=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    for n in sorted(plan):
        t_mesh = time.time()
        roll, fom_res_fn = bc.bf.make_rollout(n)
        interior = bc.interior_indices(n)
        coords = jnp.asarray(bc.grid_coords(n))
        xy_int = coords[jnp.asarray(interior)]
        n2 = n * n

        U_te, res_te = gen_trajectories(n, cxt, cyt, wt, at, nut, roll)     # (n_test, T1, n^2)
        tn = np.linalg.norm(U_te, axis=2)                                   # (n_test, T1)
        log(f"== N={n:4d}: test trajectories regenerated, FOM Newton residual {res_te:.2e}")

        # FOM baseline, per test trajectory, same protocol as the ROM
        fom_ts = []
        for i in range(N_TEST):
            U0 = jnp.asarray(U_te[i, 0])[None]
            nu1 = jnp.asarray([nut[i]])

            def fom_once(_U0=U0, _nu=nu1):
                s, r = roll(_U0, _nu); s.block_until_ready()
            med, _ = time_fn(fom_once)
            fom_ts.append(med)
        fom_med = float(np.median(fom_ts))
        report["fom"].append(dict(pde="burgers2d", method="fom", N=n, n_dof=(n - 2) ** 2,
                                  fom_rollout_s=fom_med,
                                  per_source_s=[float(v) for v in fom_ts],
                                  fom_max_rel_residual=res_te, n_sources=N_TEST,
                                  gpu=gpu_name, node=NODE,
                                  slurm_job=os.environ.get("SLURM_JOB_ID"),
                                  jax_backend=dev.platform, commit=commit))
        log(f"   FOM rollout (batch 1, median over {N_TEST} sources) {fom_med*1e3:.1f} ms")

        # POD basis at THIS mesh from the SAME training trajectories
        U_pod, res_tr = gen_trajectories(n, cxr[:N_POD_TRAJ], cyr[:N_POD_TRAJ],
                                         wr[:N_POD_TRAJ], ar[:N_POD_TRAJ],
                                         nur[:N_POD_TRAJ], roll)
        S = U_pod.reshape(-1, n2)
        Vfull, sv, orth = bc.pod_basis(S, kmax=POD_KMAX)
        Vfull = np.asarray(Vfull)
        log(f"   POD basis: {S.shape[0]} snapshots from {N_POD_TRAJ} trajectories, "
            f"orthonormality dev {orth:.2e}, sv0 {sv[0]:.3e} sv[{POD_KMAX-1}] {sv[-1]:.3e} "
            f"(train FOM residual {res_tr:.2e})")
        # nearest-IC training trajectory per test case (u0 is known online)
        U0_tr = np.stack([np.asarray(bc.bf.blob_ic(n, cxr[i], cyr[i], wr[i], ar[i])).reshape(-1)
                          for i in range(bc.N_TRAIN)])
        d_ic = np.stack([np.linalg.norm(U0_tr - U_te[i, 0], axis=1) for i in range(N_TEST)])
        nn_j = [int(np.argmin(d_ic[i])) for i in range(N_TEST)]                 # over all 512 (coord)
        nn_p = [int(np.argmin(d_ic[i, :N_POD_TRAJ])) for i in range(N_TEST)]    # over the POD set
        del U0_tr, d_ic

        cand_pos = ctol_eq.candidate_pool(interior.size)
        log(f"   EQ candidate pool: {cand_pos.size} of {interior.size} interior nodes "
            f"(cap {ctol_eq.CAND_CAP})")

        phi_cache = {}

        def phi_full(M):
            if M not in phi_cache:
                phi_cache[M] = np.asarray(bc.test_modes(n, M)[2])          # (n_i^2, M)
            return phi_cache[M]

        adv_full = jax.jit(lambda u: bc.upwind_adv_field(u, n))
        # the FOM's own backward-Euler residual along a ROM trajectory (reference only)
        fom_res_traj = jax.jit(lambda Fj, nu_: jax.vmap(
            lambda u1, u0: jnp.linalg.norm(fom_res_fn(u1, u0, nu_))
            / jnp.linalg.norm(u0))(Fj[1:], Fj[:-1]))

        for (k, M, m, arm_tag), methods in sorted(plan[n].items()):
            Phi = phi_full(M)
            for method in ("coord", "pod"):
                if method not in methods:
                    continue
                t_cell = time.time()
                if method == "coord":
                    dec = ck[k]["dec"]
                    Ztr = ck[k]["Ztr"]
                    Z_snap = Ztr.reshape(-1, k)
                    u_full_eq = jax.jit(lambda z, _d=dec: _d(z, xy_int))
                else:
                    Vk = np.ascontiguousarray(Vfull[:, :k])
                    dec = bc.PODDecoder(Vk)
                    Vint_j = jnp.asarray(Vk[interior])
                    Z_snap = S @ Vk
                    u_full_eq = jax.jit(lambda c, _V=Vint_j: _V @ c)

                keep, wq, eq_info = ctol_eq.eq_fit_burgers(
                    u_full_eq, adv_full, Phi, cand_pos, Z_snap, k, m,
                    f"burgers {method} N={n} k={k} M={M} m={m}", bc.nnls_capped)
                node_pos = cand_pos[keep]
                col = dict(kind="grid", idx=interior[node_pos], w=wq, info=eq_info)
                ops = bc.make_weak_ops(dec, n, col, kind="weak", M=M, solver="lspg")
                rollout = ctol_tol.rollout_tau_burgers(ops, bc.NUM_STEPS, bc.GN_BUDGET)

                # hyper-reduced, tau-stopped cold start on the SAME EQ nodes
                idx_q = jnp.asarray(interior[node_pos])
                xy_q = coords[idx_q]
                sw = jnp.sqrt(jnp.asarray(wq, F64))
                if method == "coord":
                    mis = lambda z, u0q, _d=dec, _p=xy_q, _w=sw: _w * (_d(z, _p) - u0q)
                else:
                    Vq_j = jnp.asarray(Vk[interior[node_pos]])
                    mis = lambda c, u0q, _V=Vq_j, _w=sw: _w * (_V @ c - u0q)
                fit_ic = make_fit_ic_tau(mis, k, bc.IC_BUDGET)

                # decode of all 51 slices (the ROM's output stage)
                if method == "coord":
                    dec_all = jax.jit(lambda ZZ, _d=dec: jax.vmap(lambda z: _d(z, coords))(ZZ))
                else:
                    Vk_j = jnp.asarray(Vk)
                    dec_all = jax.jit(lambda ZZ, _V=Vk_j: ZZ @ _V.T)

                # per-test-case cold-start inits
                if method == "coord":
                    Ztr = ck[k]["Ztr"]
                    inits = [jnp.asarray(np.stack([Ztr.mean(axis=0)[0], Ztr[nn_j[i], 0]]))
                             for i in range(N_TEST)]
                else:
                    Cm = (S.reshape(N_POD_TRAJ, T1, n2)[:, 0] @ Vk)        # t=0 coefficients
                    inits = [jnp.asarray(np.stack([Cm.mean(0), Cm[nn_p[i]]]))
                             for i in range(N_TEST)]

                u0q = [jnp.asarray(U_te[i, 0])[idx_q] for i in range(N_TEST)]
                Zt_probe = jnp.zeros((T1, k), F64)
                dec_med, _ = time_fn(lambda: dec_all(Zt_probe).block_until_ready())

                for tau in methods[method]:
                    per_t, per_err, per_jac, per_att, per_cens, per_ic = [], [], [], [], [], []
                    per_fom_res, per_red, per_red_ic = [], [], []
                    step_reasons = {}
                    blowups = 0
                    for i in range(N_TEST):
                        Z0i, u0i, nui = inits[i], u0q[i], float(nut[i])

                        def once(_Z0=Z0i, _u0=u0i, _nu=nui, _tau=tau):
                            z0, rn, rn0, nJ, att, rsn, b = fit_ic(_u0, _Z0, _tau)
                            out = rollout(z0, _nu, _tau)
                            out[0].block_until_ready()
                            return (z0, nJ, att, rsn, rn, rn0), out
                        for _ in range(WARM):
                            once()
                        ts = []
                        for _ in range(TIME_REPS):
                            t0 = time.perf_counter()
                            ic_out, out = once()
                            ts.append(time.perf_counter() - t0)
                        z0, ic_nJ, ic_att, ic_rsn, ic_rn, ic_rn0 = ic_out
                        Z, rn, rn0, nJ, att, reason = out          # LAST TIMED REPETITION
                        per_t.append(float(np.median(ts)))
                        Zall = jnp.concatenate([z0[None], Z], axis=0)
                        F = np.asarray(dec_all(Zall))                       # (T1, n^2)
                        if not np.all(np.isfinite(F)):
                            blowups += 1
                            per_err.append(float("nan"))
                            per_fom_res.append(float("nan"))
                        else:
                            per_err.append(float(np.mean(np.linalg.norm(F - U_te[i], axis=1)
                                                         / tn[i])))
                            rr = np.asarray(fom_res_traj(jnp.asarray(F), nui))
                            per_fom_res.append(float(np.mean(rr)))
                        rs = [int(r) for r in np.asarray(reason).tolist()]
                        for r in rs:
                            step_reasons[str(r)] = step_reasons.get(str(r), 0) + 1
                        miss = sum(1 for r in rs if r not in TAU_OK_STEP)
                        miss += 0 if int(ic_rsn) in TAU_OK_IC else 1
                        per_cens.append(miss / (bc.NUM_STEPS + 1))
                        per_jac.append(int(np.sum(np.asarray(nJ))) + int(ic_nJ))
                        per_att.append(int(np.sum(np.asarray(att))) + int(ic_att))
                        per_ic.append(int(ic_rsn))
                        rr0 = np.maximum(np.asarray(rn0), 1e-300)
                        per_red.append(float(np.mean(np.asarray(rn) / rr0)))
                        per_red_ic.append(float(ic_rn) / max(float(ic_rn0), 1e-300))
                    solve_ms = float(np.median(per_t)) * 1e3
                    e2e_ms = solve_ms + dec_med * 1e3
                    ok = np.isfinite(per_err)
                    row = dict(pde="burgers2d", method=method, N=n, k=k, M=M,
                               m=int(len(wq)), tau=tau, time_ms=e2e_ms,
                               err_rel_l2=float(np.mean(np.asarray(per_err)[ok]))
                               if ok.any() else float("nan"),
                               iters=float(np.mean(per_att)), jac_evals=float(np.mean(per_jac)),
                               censored=bool(np.max(per_cens) > 0), n_sources=N_TEST,
                               seed=bc.SEED, gpu=gpu_name, jax_backend=dev.platform,
                               commit=commit, node=NODE,
                               slurm_job=os.environ.get("SLURM_JOB_ID"),
                               # ---- beyond the shared schema
                               arm=arm_tag, time_ms_solve=solve_ms,
                               time_ms_decode=dec_med * 1e3,
                               time_ms_solve_per_source=[float(v) * 1e3 for v in per_t],
                               err_rel_l2_median=float(np.median(np.asarray(per_err)[ok]))
                               if ok.any() else float("nan"),
                               err_rel_l2_max=float(np.max(np.asarray(per_err)[ok]))
                               if ok.any() else float("nan"),
                               err_rel_l2_per_source=[float(v) for v in per_err],
                               n_blowup=int(blowups),
                               fom_residual_rel_mean=float(np.nanmean(per_fom_res)),
                               censored_frac=float(np.mean(per_cens)),
                               censored_frac_max=float(np.max(per_cens)),
                               rel_reduction_mean=float(np.mean(per_red)),
                               rel_reduction_max=float(np.max(per_red)),
                               rel_reduction_ic_mean=float(np.mean(per_red_ic)),
                               ic_reasons={str(r): per_ic.count(r) for r in set(per_ic)},
                               step_reasons=step_reasons,
                               eq_rel_fit=eq_info["rel_fit"], eq_info=eq_info,
                               ms_per_jac=solve_ms / max(float(np.mean(per_jac)), 1.0),
                               fom_rollout_ms=fom_med * 1e3,
                               speedup_e2e=fom_med * 1e3 / e2e_ms)
                    report["rows"].append(row)
                    log(f"   {method:5s} N={n:4d} k={k:2d} M={M:3d} m={row['m']:4d} "
                        f"tau={tau:.0e}  solve {solve_ms:8.1f} ms  e2e {e2e_ms:8.1f} ms  "
                        f"jac {row['jac_evals']:6.1f}  err {row['err_rel_l2']:.3e}  "
                        f"fomres {row['fom_residual_rel_mean']:.2e}  "
                        f"cens {row['censored_frac']*100:4.1f}%  blowups {blowups}")
                    save()
                log(f"   [cell {method} N={n} k={k} M={M} m={m}: {time.time()-t_cell:.0f}s]")
        del U_pod, S
        log(f"== N={n} done [{time.time()-t_mesh:.0f}s]")

    report["complete"] = True
    save()
    log("DONE")


if __name__ == "__main__":
    main()
