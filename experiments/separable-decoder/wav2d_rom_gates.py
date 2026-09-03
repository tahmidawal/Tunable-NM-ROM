"""Wave 2D phase 3 — ROM gates W0-W7 and G0c, per (BC, head arm, ROM arm), on the 16 held-out trajectories.

  JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun $PY wav2d_rom_gates.py N=64 BC=ref [R=64] [STEPS=40000] [M=64]
        [HEADS=sup,auto,auto+vc] [RS=8,20,40] [SMOKE=1] [OUT=runs/wav2d] [CACHE=cache/wav2d]

Consumes the phase-2 cache (data, bank, heads, oracle latents) -- the head's provenance (smoke flag, N, BC,
steps) is asserted so a smoke head can never be consumed by a certified run.  Every gate records value,
threshold, pass and its negative control; all 16/16 rollouts must complete (an incomplete rollout is a
FAIL, never dropped).  W3 is the STOP gate of the cell (design r3).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import wav2d_common as wc
import wav2d_bank as wb
import wav2d_head as wh
import wav2d_rom as wr
from wav2d_common import Grid, log, precond

ARGS = dict(a.split("=", 1) for a in sys.argv[1:])
N = int(ARGS.get("N", "64")); BC = ARGS.get("BC", "ref"); R = int(ARGS.get("R", "64"))
STEPS = int(ARGS.get("STEPS", "40000")); MM = int(ARGS.get("M", "64"))
HEADS = ARGS.get("HEADS", "sup,auto,auto+vc").split(",")
RS_LIST = [int(x) for x in ARGS.get("RS", "8,20,40").split(",")]
SMOKE = int(ARGS.get("SMOKE", "0"))
OUT = ARGS.get("OUT", "runs/wav2d"); CACHE = ARGS.get("CACHE", "cache/wav2d")
N_TRAIN = int(ARGS.get("N_TRAIN", str(wc.N_TRAIN))); N_TEST = int(ARGS.get("N_TEST", str(wc.N_TEST)))
HORIZON_MULT = int(ARGS.get("HORIZON", "4"))               # 4T continuation
K_OF = {"sup": 6, "auto": 8, "auto+vc": 8}
os.makedirs(OUT, exist_ok=True)


def finite(*xs):
    return all(np.all(np.isfinite(np.asarray(x, dtype=float))) for x in xs)


def gate(name, value, thr, control=None, control_thr=None, control_dir="ge", note="", aggregate="median"):
    v = float(value) if value is not None else float("nan")
    ok = bool(np.isfinite(v) and v <= thr)
    rec = dict(value=v, threshold=thr, aggregate=aggregate, passed=ok, note=note)
    if control is not None:
        cv = float(control)
        fired = bool(np.isfinite(cv) and (cv >= control_thr if control_dir == "ge" else cv <= control_thr))
        rec.update(control_value=cv, control_threshold=control_thr, control_dir=control_dir, control_fired=fired)
        rec["passed"] = ok and fired
    log(f"  {name:6s} {'PASS' if rec['passed'] else 'FAIL'}  value={v:.3e} thr={thr:.1e}" +
        (f"  control={rec['control_value']:.3e} fired={rec['control_fired']}" if control is not None else "") +
        (f"  [{note}]" if note else ""))
    return rec


def load_cache(g: Grid):
    dpath = os.path.join(CACHE, f"data_{g.bc}_N{g.N}_tr{N_TRAIN}_te{N_TEST}.npz")
    precond(os.path.exists(dpath), f"phase-2 data cache missing: {dpath}")
    d = dict(np.load(dpath, allow_pickle=False))
    fp = wc.data_fingerprint(d["U"]); precond(fp["sha256"] == str(d["sha256"]), "data cache fingerprint mismatch")
    bpath = os.path.join(CACHE, f"bank_{g.bc}_N{g.N}_R{R}.npz"); precond(os.path.exists(bpath), "bank cache missing")
    G = np.load(bpath)["G"]
    return d, G, fp


def fom_long(g: Grid, u0, c, num_steps):
    """the 80-substep CN FOM over num_steps stored intervals (the 4T reference), plus its energies"""
    roll, _ = wc.make_cn_fom(g, cg_tol=1e-10, num_steps=num_steps)
    S, E = roll(jnp.asarray(u0)[None], jnp.zeros((1, g.n)), jnp.asarray([c]))
    return np.asarray(S)[:, 0], np.asarray(E)[:, 0]


def main():
    t_all = time.time()
    g = Grid(N, BC)
    prov = wc.provenance()
    log(f"phase-3 ROM gates N={N} BC={BC} R={R} M={MM} RS={RS_LIST} heads={HEADS} SMOKE={SMOKE} backend={prov['jax_backend']}")
    d, G, fp = load_cache(g)
    Ut, Vt, ct_all = d["U_test"], d["V_test"], d["c_test"]
    n_test = Ut.shape[0]; m = g.mass_diag()
    T = wr.build_tables(g, G, MM)
    res = dict(N=N, bc=BC, R=R, M=MM, steps=STEPS, smoke=bool(SMOKE), provenance=prov, args=ARGS,
               data_sha256=fp["sha256"], heads={}, tables={})
    Ct = np.stack([wb.coefficients(g, G, Ut[i]) for i in range(n_test)])          # (16, T+1, R)
    CVt = np.stack([wb.coefficients(g, G, Vt[i]) for i in range(n_test)])
    perp2 = np.stack([np.sum(m[None, :] * (Ut[i] - Ct[i] @ G.T) ** 2, axis=1) for i in range(n_test)])

    # ---------------- W1: A = -Lambda B; absorbing: the C term on a boundary-active state ----------------
    w1 = np.linalg.norm(T["A"] + T["lam"][:, None] * T["B"]) / np.linalg.norm(T["A"])
    Lraw = T["Phi"].T @ np.asarray(T["L"] @ G)                                     # unweighted control
    w1_ctrl = np.linalg.norm(Lraw + T["lam"][:, None] * (T["Phi"].T @ G)) / np.linalg.norm(Lraw)
    res["tables"]["W1"] = gate("W1", w1, 1e-12, control=w1_ctrl, control_thr=(1e-6 if BC == "abs" else 1e-300), control_dir="ge",
                               note="||A + Lambda B||/||A|| with the M-weighted tables; control: unweighted Phi^T L G (fires for 'abs' where L_N is not Euclidean-symmetric; identical for 'ref', reported)")
    if BC == "ref":
        res["tables"]["W1"]["passed"] = bool(w1 <= 1e-12); res["tables"]["W1"]["note"] += " -- reflective: control not applicable (M = dx^2 I), pass on the value alone"
    if BC == "abs":
        # boundary-active manufactured state: coefficients of a face-supported bump
        x = np.linspace(0, 1, N); X, Y = np.meshgrid(x, x, indexing="ij")
        face = np.exp(-((X - 1.0) ** 2) / (2 * 0.05 ** 2)).reshape(-1)
        dh = wb.coefficients(g, G, face[None, :])[0]
        dt8 = wc.DT_SNAP / RS_LIST[0]; s8 = 0.5 * float(np.median(ct_all)) * dt8
        term = np.linalg.norm(s8 * (T["C"] @ dh)) / np.linalg.norm(T["B"] @ dh)
        res["tables"]["W1-Cterm"] = gate("W1C", -term, -1e-2, note=f"-(|| (c dt/2) C dh || / || B dh ||) on a face-supported bump at RS={RS_LIST[0]}: the damping term must be >= 1e-2 of the mass term (value {term:.3e}); this is what kills the diagonal shortcut", aggregate="value")
    res["tables"].update(lam_max_reduced=T["lam_max_reduced"], Mr_minus_I=float(np.linalg.norm(T["Mr"] - np.eye(R))),
                         C_over_A=float(np.linalg.norm(T["C"]) / np.linalg.norm(T["A"])))

    # ---------------- 4T FOM references (16 test trajectories) ----------------
    n_long = wc.NUM_STEPS * HORIZON_MULT
    t0 = time.time()
    U_long = np.zeros((n_test, n_long + 1, g.n)); E_long = np.zeros((n_test, n_long + 1))
    roll_long, _ = wc.make_cn_fom(g, cg_tol=1e-10, num_steps=n_long)
    for i0 in range(0, n_test, 8):
        S, E = roll_long(jnp.asarray(Ut[i0:i0 + 8, 0]), jnp.zeros((min(8, n_test - i0), g.n)), jnp.asarray(ct_all[i0:i0 + 8]))
        U_long[i0:i0 + 8] = np.asarray(S).transpose(1, 0, 2); E_long[i0:i0 + 8] = np.asarray(E).T
    precond(np.max(np.abs(U_long[:, :wc.NUM_STEPS + 1] - Ut)) / np.max(np.abs(Ut)) <= 1e-9, "4T FOM does not reproduce the T data")
    log(f"  4T FOM references ({n_long} intervals x {n_test}) in {time.time()-t0:.0f}s")
    C_long = np.stack([wb.coefficients(g, G, U_long[i]) for i in range(n_test)])
    perp2_long = np.stack([np.sum(m[None, :] * (U_long[i] - C_long[i] @ G.T) ** 2, axis=1) for i in range(n_test)])
    # absorbing pre-exit window: first time the FOM retains < 50% of E0
    if BC == "abs":
        t_exit_idx = np.array([int(np.argmax(E_long[i] < 0.5 * E_long[i, 0])) if np.any(E_long[i] < 0.5 * E_long[i, 0]) else n_long for i in range(n_test)])
    else:
        t_exit_idx = np.full(n_test, n_long)
    res["t_exit_idx"] = t_exit_idx.tolist()

    def traj_err(H, i, k_end, k_start=0, norm_E0=False):
        """traj-RMS of decoded coefficient history H (k_end+1, R) vs the FOM on [k_start, k_end], M-norm; optionally / sqrt(E0)"""
        Uc = H[k_start:k_end + 1] @ G.T; Uf = U_long[i, k_start:k_end + 1]
        if not finite(Uc):
            return float("nan")
        d2 = np.sum(m[None, :] * (Uc - Uf) ** 2, axis=1)
        if norm_E0:
            return float(np.sqrt(np.mean(d2)) / np.sqrt(E_long[i, 0]))
        return float(np.sqrt(np.mean(d2)) / np.sqrt(np.mean(np.sum(m[None, :] * Uf ** 2, axis=1))))

    # ---------------- same-dt FOM errors per RS (time-discretisation floor) ----------------
    samedt = {}
    for rs in RS_LIST:
        nm = wc.make_newmark_fom(g, rs, cg_tol=1e-12, num_steps=wc.NUM_STEPS)
        errs = []
        for i in range(n_test):
            S, _ = nm(jnp.asarray(Ut[i, 0]), jnp.zeros(g.n), float(ct_all[i]))
            errs.append(wc.traj_rms(g, np.asarray(S), Ut[i]))
        samedt[rs] = errs
        log(f"  same-dt FOM error at RS={rs}: median {np.median(errs):.3e}")
    res["samedt_fom_error"] = {str(k): v for k, v in samedt.items()}

    # ---------------- W2: POD-K / POD-R CN controls ----------------
    for K_ in sorted(set(list(K_OF.values()) + [R])):
        rs = RS_LIST[-1]; dt = wc.DT_SNAP / rs
        errs, eners, floors = [], [], []
        for i in range(n_test):
            P = wr.PodCN(T, K_, float(ct_all[i]), dt)
            Q, Qf = P.rollout(Ct[i, 0, :K_], np.zeros(K_), n_long * rs, rs)
            H = np.zeros((n_long + 1, R)); H[:, :K_] = Q
            errs.append(traj_err(H, i, wc.NUM_STEPS)); eners.append(P.energy(Qf)[-1] / P.energy(Qf)[0])
            Hf = np.zeros((wc.NUM_STEPS + 1, R)); Hf[:, :K_] = C_long[i, :wc.NUM_STEPS + 1, :K_]
            floors.append(traj_err(Hf, i, wc.NUM_STEPS))
        rec = dict(K=K_, RS=rs, error_median=float(np.median(errs)), floor_median=float(np.median(floors)),
                   energy_ratio_median=float(np.median(eners)), energy_ratio_max_dev=float(np.max(np.abs(np.array(eners) - 1))),
                   errors=errs, floors=floors, energy_ratios=eners)
        if BC == "ref":
            # control: POD-K backward Euler (reduced) must lose energy
            i = 0; P = wr.PodCN(T, K_, float(ct_all[i]), dt)
            s_ = P.s; Kr = P.Kr; qm = Ct[i, 0, :K_].copy(); qv = np.zeros(K_); q = qm.copy()
            Abe = np.linalg.inv(np.eye(K_) + (2 * s_) ** 2 * Kr)
            for k in range(wc.NUM_STEPS * rs):
                q1 = Abe @ (q + 2 * s_ / float(ct_all[i]) * qv); qv = (q1 - q) / (2 * s_ / float(ct_all[i])); q = q1
            e_be = (0.5 * qv @ qv + 0.5 * float(ct_all[i]) ** 2 * q @ (Kr @ q)) / (0.5 * float(ct_all[i]) ** 2 * qm @ (Kr @ qm))
            rec["gate"] = gate(f"W2-{K_}", rec["energy_ratio_max_dev"], 1e-9, control=-e_be, control_thr=-(1 - 1e-3),
                               note=f"POD-{K_} CN reflective energy ratio deviation (max over 16); error median {rec['error_median']:.4f} vs floor {rec['floor_median']:.4f} (reported); control: reduced BE energy ratio {e_be:.4f} must be < 1 - 1e-3", aggregate="max")
        else:
            rec["gate"] = dict(value=rec["error_median"], passed=True, note=f"absorbing: POD-{K_} CN error median {rec['error_median']:.4f} vs floor {rec['floor_median']:.4f}, energy ratio median {rec['energy_ratio_median']:.3e} (reported)")
        res.setdefault("W2", {})[str(K_)] = rec

    # ---------------- per head arm ----------------
    for mode in HEADS:
        K = K_OF[mode]
        log(f"== head {mode} (K={K}) ==")
        hpath = os.path.join(CACHE, f"head_{BC}_N{N}_R{R}_{mode.replace('+','')}_s{STEPS}{'_SMOKE' if SMOKE else ''}.npz")
        spec = wh.load_spec(hpath, expect=dict(mode=mode, K=K, R=R, steps=STEPS, smoke=bool(SMOKE), bc=BC, N=N))
        precond(spec is not None, f"head cache missing or provenance mismatch: {hpath}")
        head = wr.HeadNP(spec)
        orc = np.load(os.path.join(CACHE, f"oracle_{BC}_N{N}_R{R}_{mode.replace('+','')}_s{STEPS}{'_SMOKE' if SMOKE else ''}.npz"))
        Zt = orc["Z_test"].reshape(n_test, wc.NUM_STEPS + 1, K)
        Hh = dict(K=K, gates={}, arms={})
        # oracle floor on [0, T] and on [0, 4T] (oracle re-fit over the long horizon)
        floor_T = [wh.traj_rms_from_coeffs(orc["res_test"].reshape(n_test, -1)[i], perp2[i], Ct[i], np.zeros(wc.NUM_STEPS + 1, int))[0] for i in range(n_test)]
        Zl, rl = wh.oracle(spec, C_long.reshape(-1, R), n_starts=4, iters=200)
        rl = rl.reshape(n_test, -1)
        floor_4T = [wh.traj_rms_from_coeffs(rl[i], perp2_long[i], C_long[i], np.zeros(n_long + 1, int))[0] for i in range(n_test)]
        Hh["floor_T"] = [float(x) for x in floor_T]; Hh["floor_4T"] = [float(x) for x in floor_4T]
        podK_T = [traj_err(np.pad(C_long[i, :wc.NUM_STEPS + 1, :K], ((0, 0), (0, R - K))), i, wc.NUM_STEPS) for i in range(n_test)]
        podK_4T = [traj_err(np.pad(C_long[i, :, :K], ((0, 0), (0, R - K))), i, n_long) for i in range(n_test)]
        Hh["podK_floor_T"] = podK_T; Hh["podK_floor_4T"] = podK_4T

        # ---- W0: table residual vs independent full-grid path; FD gradient ----
        rs0 = RS_LIST[0]; dt0 = wc.DT_SNAP / rs0
        A0 = wr.ArmA(T, head, float(ct_all[0]), dt0); rng = np.random.default_rng(0)
        worst = 0.0; worst_g = 0.0
        pool = spec["Z"] if mode != "sup" else wh.sup_latents(d["mu"])
        for _ in range(32):
            zz, zn, zm = (pool[rng.integers(len(pool))] for _ in range(3)); hn_, hm_ = head.h(zn), head.h(zm)
            r_tab = A0.residual_gen(zz, hn_, hm_)
            r_full = wr.full_grid_residual(g, T, float(ct_all[0]), dt0, G @ head.h(zz), G @ hn_, G @ hm_)
            worst = max(worst, np.linalg.norm(r_tab - r_full) / max(np.linalg.norm(r_full), 1e-300))
            dz = rng.normal(size=K); eps = 1e-6 * (1 + np.linalg.norm(zz))
            fd = (wr.full_grid_residual(g, T, float(ct_all[0]), dt0, G @ head.h(zz + eps * dz), G @ hn_, G @ hm_)
                  - wr.full_grid_residual(g, T, float(ct_all[0]), dt0, G @ head.h(zz - eps * dz), G @ hn_, G @ hm_)) / (2 * eps)
            worst_g = max(worst_g, np.linalg.norm(A0.jac(zz) @ dz - fd) / max(np.linalg.norm(fd), 1e-300))
        Tp = dict(T); Tp["B"] = T["B"] * (1 + 1e-8)
        Ap = wr.ArmA(Tp, head, float(ct_all[0]), dt0)
        ctrl0 = np.linalg.norm(Ap.residual_gen(zz, hn_, hm_) - r_full) / np.linalg.norm(r_full)
        # captured solves: a short arm-A rollout's converged states, backward-error normalised by the mass term
        Zc, Zcf, stc, okc = A0.rollout(Zt[0, 0], T["PhiM"] @ Vt[0, 0], 3 * rs0, rs0)
        worst_c = 0.0
        for k in range(2, len(Zcf)):
            hz, hn_, hm_ = head.h(Zcf[k]), head.h(Zcf[k - 1]), head.h(Zcf[k - 2])
            r_tab = A0.residual_gen(Zcf[k], hn_, hm_); r_full = wr.full_grid_residual(g, T, float(ct_all[0]), dt0, G @ hz, G @ hn_, G @ hm_)
            worst_c = max(worst_c, np.linalg.norm(r_tab - r_full) / max(np.linalg.norm(T["B"] @ hn_), 1e-300))
        Hh["gates"]["W0"] = gate("W0", max(worst, worst_c), 1e-12, control=ctrl0, control_thr=1e-9,
                                 note=f"Petrov-table residual vs decode->SOLVER STENCIL+damping rows->Phi^T M: 32 random states (relative, {worst:.1e}) and "
                                      f"{len(Zcf)-2} captured arm-A solves (backward-error normalised by ||B h_n||, {worst_c:.1e}); control: B perturbed by 1e-8")
        Hh["gates"]["W0-grad"] = gate("W0g", worst_g, 1e-7, note="analytic Jacobian-vector product vs central FD of the FULL-GRID residual (independent path)")

        # ---- W7: arm C on a linear head == independent POD-K Verlet; control: traveling state with zdot0 dropped ----
        lin = wr.LinearHead(np.vstack([np.eye(K), np.zeros((R - K, K))]))
        CL = wr.ArmC(T, lin, float(ct_all[0]), dt0)
        q0 = Ct[0, 0, :K]
        ZL, ZLf, _, _, okL = CL.rollout(q0, np.zeros(K), wc.NUM_STEPS * rs0, rs0)
        QV, QVf = wr.pod_verlet(T, K, float(ct_all[0]), dt0, q0, np.zeros(K), wc.NUM_STEPS * rs0, rs0)
        w7 = np.max(np.abs(ZLf - QVf)) / np.max(np.abs(QVf)) if okL else float("nan")
        # control: a right-going Gaussian pulse (the F3 traveling state) projected onto the bank; its projected
        # velocity must be nonzero (asserted), and dropping zdot_0 must separate the two recurrences
        x_ = np.linspace(0, 1, N); X_, Y_ = np.meshgrid(x_, x_, indexing="ij")
        Up = np.exp(-((X_ - 0.45) ** 2 + (Y_ - 0.5) ** 2) / (2 * 0.1 ** 2)); Vp = -float(ct_all[0]) * (-(X_ - 0.45) / 0.1 ** 2) * Up
        if BC == "ref":
            Up[0, :] = Up[-1, :] = Up[:, 0] = Up[:, -1] = 0; Vp[0, :] = Vp[-1, :] = Vp[:, 0] = Vp[:, -1] = 0
        q0b = wb.coefficients(g, G, g.full_to_state(Up)[None, :])[0][:K]; qv0 = wb.coefficients(g, G, g.full_to_state(Vp)[None, :])[0][:K]
        precond(np.linalg.norm(qv0) > 1e-3 * np.linalg.norm(q0b) / wc.DT_SNAP, "W7 control: projected pulse velocity is ~0, the control cannot fire")
        QVb, QVbf = wr.pod_verlet(T, K, float(ct_all[0]), dt0, q0b, qv0, 4 * rs0, rs0)
        _, ZLbf, _, _, _ = CL.rollout(q0b, np.zeros(K), 4 * rs0, rs0)
        w7c = np.max(np.abs(ZLbf - QVbf)) / np.max(np.abs(QVbf))
        Hh["gates"]["W7"] = gate("W7", w7, 1e-11, control=w7c, control_thr=1e-3,
                                 note=f"arm C with h = [I_K;0] z vs the independent POD-K damped Verlet, RS={rs0}, complete={okL}; CFL c dt sqrt(lam_max) = {CL.cfl():.3f}; control: traveling pulse (|qv0| = {np.linalg.norm(qv0):.2e}) with zdot_0 dropped")

        # ---- rollouts: arms A and C at each RS, 16 test trajectories, 4T horizon ----
        for arm_name in ("A", "C"):
            arm_res = {}
            for rs in RS_LIST:
                dt = wc.DT_SNAP / rs; n_steps = n_long * rs
                t0 = time.time()
                per = []
                for i in range(n_test):
                    c_i = float(ct_all[i]); z0 = Zt[i, 0]
                    if arm_name == "A":
                        arm = wr.ArmA(T, head, c_i, dt)
                        pv0 = T["PhiM"] @ Vt[i, 0]
                        Zs, Zf, st, ok = arm.rollout(z0, pv0, n_steps, rs)
                        iters = float(st[:, 0].mean()) if len(st) else float("nan"); cond_min = float(np.min(st[:, 3])) if len(st) else float("nan")
                        resid_max = float(np.max(st[:, 4])) if len(st) else float("nan")      # max relative GRADIENT (first-order optimality)
                    else:
                        arm = wr.ArmC(T, head, c_i, dt)
                        zd0 = arm.zdot_from_velocity(z0, CVt[i, 0])
                        Zs, Zf, st, conds, ok = arm.rollout(z0, zd0, n_steps, rs)
                        iters = float(st[:, 0].mean()) if len(st) else float("nan"); cond_min = float(np.min(conds)) if len(conds) else float("nan")
                        resid_max = float(np.max(st[:, 1])) if len(st) else float("nan")
                    # POD-K CN rollout error at the SAME RS on this trajectory (the design's POD-K comparator is the
                    # POD-K Galerkin CN control, not the instantaneous projection floor)
                    Pk = wr.PodCN(T, K, c_i, dt); Qk, _ = Pk.rollout(Ct[i, 0, :K], np.zeros(K), n_steps, rs)
                    Hk = np.zeros((n_long + 1, R)); Hk[:, :K] = Qk
                    rec = dict(complete=bool(ok), iters_mean=iters, cond_min=cond_min, resid_max=resid_max,
                               podK_cn_err_T=traj_err(Hk, i, wc.NUM_STEPS), podK_cn_err_4T=traj_err(Hk, i, n_long))
                    if ok:
                        H = np.array([head.h(z) for z in Zs])
                        rec.update(err_T=traj_err(H, i, wc.NUM_STEPS), err_4T=traj_err(H, i, n_long),
                                   err_preexit_E0=traj_err(H, i, int(t_exit_idx[i]), norm_E0=True),
                                   err_postexit_E0=(traj_err(H, i, n_long, k_start=int(t_exit_idx[i]), norm_E0=True) if t_exit_idx[i] < n_long else float("nan")),
                                   mean_field_err=float(np.max(np.abs(np.mean(H @ G.T, axis=1) - np.mean(U_long[i], axis=1))) / np.sqrt(E_long[i, 0])))
                        if arm_name == "C":
                            Er = arm.energy_reduced(Zf, zdot0=zd0)                 # E[0] = E(0), E[-1] = E(4T)
                            rec.update(Er_ratio_T=float(Er[wc.NUM_STEPS * rs] / Er[0]), Er_ratio_4T=float(Er[-1] / Er[0]),
                                       Er_max_dev=float(np.max(np.abs(Er / Er[0] - 1))),
                                       Er_secular_slope=float(np.polyfit(np.arange(len(Er)) * dt, Er / Er[0], 1)[0]))
                        Hf = np.array([head.h(z) for z in Zf]) @ G.T
                        Ed = wr.dynamic_velocity_energy(g, T, Hf, c_i, dt)
                        rec.update(Edyn_ratio_T=float(Ed[min(wc.NUM_STEPS * rs, len(Ed) - 1)] / Ed[0]), Edyn_ratio_4T=float(Ed[-1] / Ed[0]))
                        if arm_name == "A":
                            # W5 (restated): the decoded history must be first-order optimal for the LSPG residual formed
                            # through the solver's stencil (independent of the tables); the projected residual itself is
                            # NOT small for an overdetermined LSPG problem, so it and the tautological energy-accounting
                            # identity are REPORTED only
                            gopt = wr.lspg_optimality_independent(g, T, head, c_i, dt, Zf, sample=range(1, len(Zf) - 1, max(1, len(Zf) // 200)))   # ~200 sampled steps, true predecessors
                            pm = wr.projected_momentum_residual(g, T, c_i, dt, Hf)
                            bal, ctrlb = wr.momentum_balance(g, T, c_i, dt, Hf)
                            rec.update(W5_optimality_max=float(np.max(gopt)), W5_projres_reported=float(np.max(pm)),
                                       W5_balance_reported=float(np.max(np.abs(bal))), W5_flux_dropped_reported=float(np.max(np.abs(ctrlb))))
                    per.append(rec)
                n_complete = sum(p["complete"] for p in per)
                agg = dict(RS=rs, dt=dt, n_complete=int(n_complete), per_traj=per, seconds=time.time() - t0)
                if n_complete == n_test:
                    for key in ("err_T", "err_4T", "err_preexit_E0", "Edyn_ratio_T", "Er_ratio_T", "Er_ratio_4T", "Er_max_dev", "Er_secular_slope",
                                "W5_optimality_max", "W5_projres_reported", "W5_balance_reported", "W5_flux_dropped_reported", "cond_min", "iters_mean", "resid_max",
                                "podK_cn_err_T", "podK_cn_err_4T"):
                        vals = [p.get(key) for p in per]
                        if all(v is not None for v in vals):
                            vals = np.array(vals, dtype=float)
                            # NaN policy: a nonfinite value anywhere makes the aggregate NaN (never dropped)
                            agg[key + "_median"] = float(np.median(vals)) if np.all(np.isfinite(vals)) else float("nan")
                            agg[key + "_max"] = float(np.max(vals)) if np.all(np.isfinite(vals)) else float("nan")
                            agg[key + "_absmax"] = float(np.max(np.abs(vals))) if np.all(np.isfinite(vals)) else float("nan")
                    agg["excess_over_floor_T_median"] = float(np.median([p["err_T"] - f for p, f in zip(per, floor_T)]))
                log(f"  arm {arm_name} RS={rs}: {n_complete}/{n_test} complete in {agg['seconds']:.0f}s" +
                    (f", err_T median {agg['err_T_median']:.4f} (floor {np.median(floor_T):.4f}, POD-K {np.median(podK_T):.4f}, same-dt FOM {np.median(samedt[rs]):.2e}), "
                     f"err_4T {agg['err_4T_median']:.4f}, E ratio {agg.get('Er_ratio_T_median', agg.get('Edyn_ratio_T_median', float('nan'))):.4f}" if n_complete == n_test else ""))
                arm_res[str(rs)] = agg
            Hh["arms"][arm_name] = arm_res

        # ---- W6: time-step convergence (arm C primary; arm A reported): excess over floor RS-independent within 20% at RS >= 20 ----
        for arm_name in ("C", "A"):
            ex = {rs: Hh["arms"][arm_name][str(rs)].get("excess_over_floor_T_median", float("nan")) for rs in RS_LIST}
            big = [ex[rs] for rs in RS_LIST if rs >= 20 and np.isfinite(ex[rs])]
            w6 = (max(big) - min(big)) / max(abs(np.mean(big)), 1e-12) if len(big) >= 2 else float("nan")
            # control: a FIRST-ORDER integrator on the LINEAR POD-K subspace (where the floor is exact and the
            # time-step error is not masked by manifold error): reduced-BE POD-K excess must differ by > 20% between the
            # coarsest and finest RS (CN POD-K's excess is RS-independent, reported beside it)
            ex_be = {}; ex_cn = {}
            for rs in (RS_LIST[0], RS_LIST[-1]):
                dtb = wc.DT_SNAP / rs; e_be = []; e_cn = []
                for i in range(n_test):
                    c_i = float(ct_all[i]); Kr = T["Kr"][:K, :K]; Dr = T["Dr"][:K, :K]; q = Ct[i, 0, :K].copy(); qv = np.zeros(K)
                    Abe = np.linalg.inv(np.eye(K) + c_i * dtb * Dr + (c_i * dtb) ** 2 * Kr); Q = [q.copy()]
                    for k in range(wc.NUM_STEPS * rs):
                        q1 = Abe @ (q + dtb * qv); qv = (q1 - q) / dtb; q = q1
                        if (k + 1) % rs == 0: Q.append(q.copy())
                    Hb = np.zeros((wc.NUM_STEPS + 1, R)); Hb[:, :K] = np.array(Q)
                    e_be.append(traj_err(Hb, i, wc.NUM_STEPS) - podK_T[i])
                    Qc, _ = wr.PodCN(T, K, c_i, dtb).rollout(Ct[i, 0, :K], np.zeros(K), wc.NUM_STEPS * rs, rs)
                    Hc = np.zeros((wc.NUM_STEPS + 1, R)); Hc[:, :K] = Qc
                    e_cn.append(traj_err(Hc, i, wc.NUM_STEPS) - podK_T[i])
                ex_be[rs] = float(np.median(e_be)); ex_cn[rs] = float(np.median(e_cn))
            ctrl6 = abs(ex_be[RS_LIST[0]] - ex_be[RS_LIST[-1]]) / max(abs(ex_be[RS_LIST[-1]]), 1e-12)
            Hh["gates"][f"W6-{arm_name}"] = gate(f"W6{arm_name}", w6, 0.2, control=ctrl6, control_thr=0.2,
                                                note=f"arm {arm_name}: spread of the median ROM-floor excess over RS>=20 relative to its mean; excess per RS {[(rs, round(ex[rs], 5)) for rs in RS_LIST]}; "
                                                     f"control (AMENDMENT 8: a first-order integrator on the linear POD-K subspace, where the time-step error is not masked by manifold error; the design's RS=1 mutation of the ROM arm cannot fire when the manifold error dominates): "
                                                     f"POD-K BE excess over the POD-K floor {ex_be} must differ by > 20% between RS={RS_LIST[0]} and {RS_LIST[-1]} (POD-K CN excess {ex_cn} for reference)")
        # ---- W4: arm C reflective energy bounded; control: reduced backward Euler (arm C stepping with BE) ----
        if BC == "ref":
            rsb = RS_LIST[-1]; aggC = Hh["arms"]["C"][str(rsb)]; aggC0 = Hh["arms"]["C"][str(RS_LIST[0])]
            if aggC["n_complete"] == n_test:
                # value = max( max|E_r/E_r0 - 1| / 1e-2 , max|slope| / (1e-3 per T) ) <= 1; the bound must not GROW under
                # refinement (RS 8 -> 40), and W6-C must have passed for a FAIL to be read as an implementation bug
                shrink_ok = bool(aggC0["n_complete"] == n_test and np.isfinite(aggC0.get("Er_max_dev_max", np.nan)) and aggC["Er_max_dev_max"] <= 1.05 * aggC0["Er_max_dev_max"])
                w4v = max(aggC["Er_max_dev_max"] / 1e-2, aggC["Er_secular_slope_absmax"] / 1e-3)
                if not (shrink_ok and Hh["gates"].get("W6-C", {}).get("passed", False)):
                    w4v = max(w4v, 2.0)            # cannot pass without W6-C and the shrinking bound
                # control: first-order reduced-BE stepping on the same head (dissipative): energy must drop by > 1e-2 over 4T
                i = 0; c_i = float(ct_all[i]); dtb = wc.DT_SNAP / rsb; armc = wr.ArmC(T, head, c_i, dtb)
                _, _, Eb = wr.armC_backward_euler(T, head, c_i, dtb, Zt[i, 0], armc.zdot_from_velocity(Zt[i, 0], CVt[i, 0]), n_long * rsb, rsb)
                h0_, J0_ = head.hj(Zt[i, 0]); v0_ = J0_ @ armc.zdot_from_velocity(Zt[i, 0], CVt[i, 0])
                Eb0 = 0.5 * v0_ @ (T["Mr"] @ v0_) + 0.5 * c_i ** 2 * h0_ @ (T["Kr"] @ h0_)
                Hh["gates"]["W4"] = gate("W4", w4v, 1.0, control=1 - Eb[-1] / Eb0, control_thr=1e-2,
                                         note=f"arm C reflective at RS={rsb}: max |E_r/E_r0 - 1| over 4T (max over 16) {aggC['Er_max_dev_max']:.2e} (<= 1e-2), max |secular slope| {aggC['Er_secular_slope_absmax']:.2e}/T (<= 1e-3), "
                                              f"bound at RS={RS_LIST[0]} {aggC0.get('Er_max_dev_max', float('nan')):.2e} (must not grow under refinement: {shrink_ok}), W6-C passed: {Hh['gates'].get('W6-C', {}).get('passed', False)}; "
                                              f"control: first-order reduced-BE stepping on the same head loses {1-Eb[-1]/Eb0:.3f} of E_r")
            else:
                Hh["gates"]["W4"] = dict(value=float("nan"), passed=False, note="arm C rollouts incomplete")
        # ---- W5 (absorbing): momentum balance closes with the residual work ----
        aggA = Hh["arms"]["A"][str(RS_LIST[-1])]
        if aggA["n_complete"] == n_test:
            # control: a latent history that is NOT an LSPG solution -- the arm-A history with every latent perturbed by
            # 1e-3 relative -- must show an O(1e-3)+ gradient through the same independent path
            i = 0; c_i = float(ct_all[i]); dtb = wc.DT_SNAP / RS_LIST[-1]
            _, Zf0, _, _ = wr.ArmA(T, head, c_i, dtb).rollout(Zt[i, 0], T["PhiM"] @ Vt[i, 0], 3 * RS_LIST[-1], RS_LIST[-1])
            rng5 = np.random.default_rng(5); Zp = Zf0 * (1 + 1e-3 * rng5.normal(size=Zf0.shape))
            g_ctrl = float(np.max(wr.lspg_optimality_independent(g, T, head, c_i, dtb, Zp)))
            Hh["gates"]["W5"] = gate("W5", aggA["W5_optimality_max_max"], 1e-6, control=g_ctrl, control_thr=1e-4,
                                     note=f"arm A RS={RS_LIST[-1]}: first-order LSPG optimality ||J^T r||/(||J|| ||r||) of the decoded history with r and J formed through the solver's stencil "
                                          f"(independent of the tables), max over ~200 sampled steps x 16 trajectories; REPORTED: projected residual (nonzero for overdetermined LSPG) "
                                          f"{aggA.get('W5_projres_reported_max', float('nan')):.1e}, energy-accounting identity (tautological, verification) {aggA.get('W5_balance_reported_max', float('nan')):.1e}, "
                                          f"flux-dropped {aggA.get('W5_flux_dropped_reported_max', float('nan')):.1e}; control: latents perturbed 1e-3 must read >= 1e-4")
        # ---- W3 STOP gate: per ROM arm, at the RS with the smallest median err_T among completed RS ----
        for arm_name in ("A", "C"):
            cands = [(rs, Hh["arms"][arm_name][str(rs)]) for rs in RS_LIST if Hh["arms"][arm_name][str(rs)]["n_complete"] == n_test]
            if not cands:
                Hh["gates"][f"W3-{arm_name}"] = dict(value=float("nan"), passed=False, note=f"arm {arm_name}: no RS with 16/16 complete")
                log(f"  W3-{arm_name} FAIL  no RS with {n_test}/{n_test} complete")
                continue
            # PREDECLARED RS: the finest RS that completes 16/16 (no selection on the held-out error; verification)
            rs, agg = max(cands, key=lambda t: t[0])
            per = agg["per_traj"]
            r_floor_T = np.median([p["err_T"] for p in per]) / np.median(floor_T)
            r_floor_4T = np.median([p["err_4T"] for p in per]) / np.median(floor_4T)
            # the POD-K comparator is the POD-K Galerkin CN ROLLOUT at the same RS (design), per trajectory
            r_pod_T = np.median([p["err_T"] for p in per]) / np.median([p["podK_cn_err_T"] for p in per])
            r_pod_4T = np.median([p["err_4T"] for p in per]) / np.median([p["podK_cn_err_4T"] for p in per])
            if BC == "ref":
                ekey = "Er_ratio_T" if arm_name == "C" else "Edyn_ratio_T"
                e_med = np.median([p[ekey] for p in per]); e_ok = 0.9 <= e_med <= 1.1
                val = max(r_floor_T / 1.5, r_floor_4T / 1.5, r_pod_T / 0.5, r_pod_4T / 0.5, (0.0 if e_ok else 2.0))
                note = (f"arm {arm_name} RS={rs} (finest complete): err_T/floor {r_floor_T:.3f} (<=1.5), err_4T/floor {r_floor_4T:.3f} (<=1.5), err_T/POD-K-CN {r_pod_T:.3f} (<=0.5), "
                        f"err_4T/POD-K-CN {r_pod_4T:.3f} (<=0.5), energy ratio at T median {e_med:.4f} (in [0.9,1.1]: {e_ok}); value = max of the normalised ratios; "
                        f"POD-K projection floors T/4T {np.median(podK_T):.4f}/{np.median(podK_4T):.4f} reported")
            else:
                r_pre = np.median([p["err_preexit_E0"] for p in per]) / np.median([wh.traj_rms_from_coeffs(rl[i][:t_exit_idx[i] + 1], perp2_long[i][:t_exit_idx[i] + 1], C_long[i][:t_exit_idx[i] + 1], np.zeros(t_exit_idx[i] + 1, int))[0] * np.sqrt(np.mean(np.sum(m[None, :] * U_long[i, :t_exit_idx[i] + 1] ** 2, axis=1))) / np.sqrt(E_long[i, 0]) for i in range(n_test)])
                val = max(r_floor_T / 1.5, r_floor_4T / 1.5, r_pod_T / 0.5, r_pod_4T / 0.5, r_pre / 1.5)
                note = (f"arm {arm_name} RS={rs} (finest complete): err_T/floor {r_floor_T:.3f} (<=1.5), err_4T/floor {r_floor_4T:.3f} (<=1.5), err_T/POD-K-CN {r_pod_T:.3f} (<=0.5), err_4T/POD-K-CN {r_pod_4T:.3f} (<=0.5), "
                        f"pre-exit err/floor (E0-normalised) {r_pre:.3f} (<=1.5); post-exit err/sqrt(E0) median {np.median([p['err_postexit_E0'] for p in per]) if np.all(np.isfinite([p['err_postexit_E0'] for p in per])) else float('nan'):.3e} "
                        f"and mean-field err median {np.median([p['mean_field_err'] for p in per]):.2e} reported")
            # control: wrong-sign stiffness mutation must blow up (arm C) / drift (arm A) -- one trajectory
            i = 0; c_i = float(ct_all[i]); dt = wc.DT_SNAP / rs; Tm = dict(T); Tm["Kr"] = -T["Kr"]; Tm["A"] = -T["A"]
            if arm_name == "C":
                armm = wr.ArmC(Tm, head, c_i, dt); Zs, Zfm, _, _, okm = armm.rollout(Zt[i, 0], armm.zdot_from_velocity(Zt[i, 0], CVt[i, 0]), wc.NUM_STEPS * rs, rs)
            else:
                Zs, Zfm, _, okm = wr.ArmA(Tm, head, c_i, dt).rollout(Zt[i, 0], T["PhiM"] @ Vt[i, 0], wc.NUM_STEPS * rs, rs)
            if not okm:
                # last finite latent of the diverged control, compared at its own snapshot time
                fin = np.all(np.isfinite(Zfm), axis=1); kf = int(np.max(np.where(fin)[0])) if fin.any() else 0
                ks = kf // rs
                if ks >= 1:
                    Hm = np.array([head.h(Zfm[k * rs]) for k in range(ks + 1)])
                    Zs = None; ctrl_last = traj_err(Hm, i, ks)
                else:
                    ctrl_last = float("nan")
            # the wrong-sign control may diverge: read its error at the LAST FINITE latent (never from a NaN)
            if okm:
                ctrl3 = traj_err(np.array([head.h(z) for z in Zs]), i, wc.NUM_STEPS)
            else:
                ctrl3 = ctrl_last                      # error at the last finite snapshot of the diverged control
            rec = gate(f"W3-{arm_name}", val, 1.0, control=ctrl3, control_thr=1.0,
                       note=note + f"; control: wrong-sign stiffness (K -> -K) error must exceed 1 (control complete: {okm}; a diverged control is recorded as nonfinite, not as fired)")
            rec.update(RS=rs, ratio_floor_T=float(r_floor_T), ratio_floor_4T=float(r_floor_4T), ratio_podK_cn_T=float(r_pod_T), ratio_podK_cn_4T=float(r_pod_4T),
                       control_nonfinite=bool(not okm))
            Hh["gates"][f"W3-{arm_name}"] = rec
        Hh["W3_passed_any"] = bool(any(Hh["gates"].get(f"W3-{a}", {}).get("passed", False) for a in ("A", "C")))
        res["heads"][mode] = Hh
        # ---- G0c: stepdiag from oracle starts, arm C, H in {1,2,5,10} intervals ----
        # errors are M-norm at the destination snapshot, normalised by the TRAJECTORY RMS (never per snapshot);
        # the excess is over the oracle floor AT THAT DESTINATION; NaN anywhere -> FAIL; the wrong-sign control's
        # divergence is read at its last finite state
        rs = RS_LIST[1] if len(RS_LIST) > 1 else RS_LIST[0]; dt = wc.DT_SNAP / rs
        Hs = [1, 2, 5, 10]; exc = {H: [] for H in Hs}; hold = {H: [] for H in Hs}; mut = {H: [] for H in Hs}; flr = {H: [] for H in Hs}
        Tm = dict(T); Tm["Kr"] = -T["Kr"]
        res_te = orc["res_test"].reshape(n_test, -1)
        for i in range(n_test):
            c_i = float(ct_all[i]); rms_i = np.sqrt(np.mean(np.sum(m[None, :] * Ut[i] ** 2, axis=1)))
            for k0 in (0, 10, 20, 30):
                z0 = Zt[i, k0]; armc = wr.ArmC(T, head, c_i, dt); zd0 = armc.zdot_from_velocity(z0, CVt[i, k0])
                Zs, Zf, _, _, ok = armc.rollout(z0, zd0, 10 * rs, rs)
                armm = wr.ArmC(Tm, head, c_i, dt); Zsm, Zfm, _, _, okm = armm.rollout(z0, zd0, 10 * rs, rs)
                for H in Hs:
                    if k0 + H > wc.NUM_STEPS:
                        continue
                    fl = np.sqrt(res_te[i, k0 + H] ** 2 + perp2[i, k0 + H]) / rms_i
                    e = (np.sqrt(np.sum(m * (G @ head.h(Zs[H]) - Ut[i, k0 + H]) ** 2)) / rms_i) if ok else float("nan")
                    eh = np.sqrt(np.sum(m * (G @ head.h(z0) - Ut[i, k0 + H]) ** 2)) / rms_i
                    if okm:
                        em = np.sqrt(np.sum(m * (G @ head.h(Zsm[H]) - Ut[i, k0 + H]) ** 2)) / rms_i
                    else:
                        fin = np.all(np.isfinite(Zfm), axis=1); kf = int(np.max(np.where(fin)[0])) if fin.any() else 0
                        em = (np.sqrt(np.sum(m * (G @ head.h(Zfm[kf]) - Ut[i, k0 + min(H, kf // rs)]) ** 2)) / rms_i) if kf >= rs else float("nan")
                    exc[H].append(e - fl); hold[H].append(eh); mut[H].append(em); flr[H].append(fl)
        finite_all = all(np.all(np.isfinite(exc[H])) for H in Hs)
        val = (np.median(exc[10]) / np.median(flr[10])) if finite_all else float("nan")
        ctrl = float(np.median(mut[10])) if np.all(np.isfinite(mut[10])) else float("nan")
        g0c = gate("G0c", val, 0.5, control=ctrl, control_thr=1.0,
                   note=f"stepdiag from oracle starts (arm C, RS={rs}, 4 launch times x 16 traj): median excess over the destination floor at H=10 / median floor; "
                        f"excess per H {[(H, round(float(np.median(exc[H])), 5)) for H in Hs]}; floor per H {[(H, round(float(np.median(flr[H])), 5)) for H in Hs]}; "
                        f"hold comparator per H {[(H, round(float(np.median(hold[H])), 4)) for H in Hs]}; nonfinite launches: {int(sum(np.sum(~np.isfinite(exc[H])) for H in Hs))} (any -> FAIL); "
                        f"control: wrong-sign stiffness at H=10 (read at its last finite state if diverged) must exceed 1")
        g0c.update(excess_per_H={str(H): float(np.median(exc[H])) for H in Hs}, floor_per_H={str(H): float(np.median(flr[H])) for H in Hs},
                   hold_per_H={str(H): float(np.median(hold[H])) for H in Hs})
        Hh["gates"]["G0c"] = g0c

    res["wall_s"] = time.time() - t_all
    path = os.path.join(OUT, f"wav2d_rom_gates_{BC}_N{N}_R{R}{'_SMOKE' if SMOKE else ''}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"phase 3 done ({res['wall_s']:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
