"""Numerical verification of the Wave-2D latent-stepping harness.

Checks that must hold before any cluster time is spent (all f64):

 V1  the u-only Newmark rollout at RS=80 reproduces the (u,v) CN FOM
     (`wave2d_film.make_rollout`) to CG tolerance -- i.e. the v-elimination
     v_{n+1} = 2(u_{n+1}-u_n)/dt - v_n is exact for this scheme;
 V2  the FOM's own stored trajectory drives the ROM's residual operators to
     ~0 (strong full / strong subset / weak), and a NON-solution does not;
 V3  the weak residual with ALL (n-2)^2 modes and WEAK_ALPHA=0 has exactly the
     same 2-norm as the strong full-grid residual (Phi is orthonormal) --
     "weak with all modes == full Galerkin";
 V4  the FOM energy drift, and the drift of the reconstructed-v energy of the
     FOM trajectory (the ROM's energy diagnostic applied to exact data);
 V5  the same-dt Newmark FOM error vs the 80-substep FOM as a function of RS
     -- the ROM's time-discretisation floor, which sets the usable RS;
 V6  POD floors (traj-RMS) on the val split.

Usage: N=16 [RS_LIST=1,2,4,8,20,80] python wlat_verify.py <outdir>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import wlat_common as wc
from wlat_common import wf, F64, log

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
RS_LIST = [int(s) for s in os.environ.get("RS_LIST", "1,2,4,8,20,80").split(",") if s]
N_V = int(os.environ.get("N_VERIFY", "4"))          # trajectories used for the RS sweep
N = wc.N


def main():
    log(f"jax_backend={jax.default_backend()}  N={N}  RS={wc.RS}  RS_LIST={RS_LIST}")
    rep = dict(config=dict(wc.CONFIG, rs_list=RS_LIST, n_verify=N_V),
               backend=jax.default_backend())
    interior = wc.interior_indices(N)
    ni2 = interior.size

    # --- test trajectories (fresh draw) -------------------------------------
    cxt, cyt, wt, at, ct, zt = wf.sample_params(seed=wc.TEST_SEED, m=N_V)
    U0 = np.stack([wc.blob_ic(N, cxt[i], cyt[i], wt[i], at[i]) for i in range(N_V)])
    roll, _ = wf.make_rollout(N)
    t0 = time.time()
    snaps, ens = roll(jnp.asarray(U0), jnp.asarray(ct))
    Ufom = np.asarray(snaps).transpose(1, 0, 2)                    # (N_V, T1, n^2)
    ens = np.asarray(ens)
    drift_fom = float(np.max(np.abs(ens - ens[0]) / ens[0]))
    log(f"  V4 FOM energy drift {drift_fom:.2e}  [{time.time()-t0:.0f}s]")
    rep["V4_fom_energy_drift"] = drift_fom

    # --- V1 + V5: u-only Newmark FOM at several RS --------------------------
    rs_tab = {}
    for rs in RS_LIST:
        nm = wc.make_newmark_fom(N, rs)
        t0 = time.time()
        errs, drifts = [], []
        for i in range(N_V):
            S, E = nm(jnp.asarray(U0[i]), float(ct[i]))
            S = np.asarray(S); E = np.asarray(E)
            errs.append(wc.traj_metrics(S, Ufom[i])[2])
            drifts.append(float(np.max(np.abs(E - E[0]) / E[0])))
        rs_tab[rs] = dict(traj_rel_vs_fom_mean=float(np.mean(errs)),
                          traj_rel_vs_fom_max=float(np.max(errs)),
                          energy_drift_max=float(np.max(drifts)),
                          n_latent_steps=wc.NUM_STEPS * rs, secs=time.time() - t0)
        log(f"  V5 RS={rs:3d} (dt={wc.DT_SNAP/rs:.2e}, {wc.NUM_STEPS*rs:5d} steps): "
            f"traj rel vs FOM {np.mean(errs):.3e} (max {np.max(errs):.3e}), "
            f"E-drift {np.max(drifts):.1e}  [{time.time()-t0:.0f}s]")
    rep["V5_samedt_newmark_fom"] = rs_tab
    if 80 in rs_tab:
        rep["V1_newmark_rs80_vs_cn_fom"] = rs_tab[80]["traj_rel_vs_fom_mean"]
        # threshold 1e-6, not 1e-12: the two rollouts are algebraically identical
        # but use INDEPENDENT CG solves (FOM tol 1e-10, ours 1e-12) over 4000 steps,
        # so the difference is a random walk of the CG residuals (~1e-8 observed).
        ok1 = rs_tab[80]["traj_rel_vs_fom_mean"] < 1e-6
        log(f"  V1 Newmark(RS=80) == CN FOM: {rep['V1_newmark_rs80_vs_cn_fom']:.2e} "
            f"{'OK' if ok1 else 'FAIL'}")
        rep["V1_ok"] = bool(ok1)

    # --- V2: residual operators on exact Newmark states ---------------------
    chk = wc.verify_residual_ops(N, c=float(ct[0]), M=min(32, ni2))
    log(f"  V2 residual-operator checks {chk}")
    rep["V2_residual_ops"] = chk
    rep["V2_ok"] = bool(max(chk["strong_full"], chk["strong_rand"], chk["weak_full"]) < 1e-9
                        and chk["weak_nonsolution"] > 1e-12)

    # --- V2b: the FOM's own stored trajectory through the ROM operator ------
    # (only meaningful at RS = FOM_SUBSTEPS: the stored snapshots are 80 substeps apart)
    nm_states = wc.newmark_first_states(N, wc.FOM_SUBSTEPS, U0[0], float(ct[0]), 3)
    a = (0.5 * (wc.DT_SNAP / wc.FOM_SUBSTEPS) * float(ct[0])) ** 2
    Ls = [np.asarray(wc.lap_full_field(jnp.asarray(u), N)) for u in nm_states]
    r0 = (nm_states[1] - nm_states[0]) - a * (Ls[1] + Ls[0])
    r1 = (nm_states[2] - 2 * nm_states[1] + nm_states[0]) - a * (Ls[2] + 2 * Ls[1] + Ls[0])
    rep["V2b_fom_traj_residual_rel"] = float(
        max(np.abs(r0[interior]).max(), np.abs(r1[interior]).max()) / np.linalg.norm(U0[0]))
    log(f"  V2b FOM-state residual (rel) {rep['V2b_fom_traj_residual_rel']:.2e}")

    # --- V3: weak with ALL modes, alpha=0 == strong full grid ---------------
    dec = wc.PODDecoder(np.eye(N * N))
    st = wc.newmark_first_states(N, wc.RS, U0[0], float(ct[0]), 3)
    ops_s = wc.make_strong_ops(dec, N, dict(kind="grid", idx=interior))
    ops_w = wc.make_weak_ops(dec, N, dict(kind="grid", idx=interior, w=None), M=ni2, beta=0.0)
    # perturb: a non-solution state so the residual is non-trivial
    rng = np.random.default_rng(0)
    u_bad = st[2] + 1e-2 * np.linalg.norm(st[2]) / np.sqrt(N * N) * rng.standard_normal(N * N)
    u_bad = np.asarray(jnp.asarray(u_bad) * wf.boundary_mask(N).reshape(-1))
    Ss = [ops_s["state_of"](jnp.asarray(u)) for u in st]
    Sw = [ops_w["state_of"](jnp.asarray(u)) for u in st]
    c0 = float(ct[0])
    a0 = (0.5 * wc.DT * c0) ** 2
    rs_ = ops_s["r_w"](jnp.asarray(u_bad), jnp.stack([Ss[1], Ss[0]]), c0)
    old = wc.WEAK_ALPHA
    wc.WEAK_ALPHA = 0.0
    ops_w0 = wc.make_weak_ops(dec, N, dict(kind="grid", idx=interior, w=None), M=ni2, beta=0.0)
    Sw0 = [ops_w0["state_of"](jnp.asarray(u)) for u in st]
    rw_ = ops_w0["r_w"](jnp.asarray(u_bad), jnp.stack([Sw0[1], Sw0[0]]), c0)
    wc.WEAK_ALPHA = old
    ns, nw = float(jnp.linalg.norm(rs_)), float(jnp.linalg.norm(rw_))
    rep["V3_weak_allmodes_vs_strong"] = dict(strong_norm=ns, weak_norm=nw,
                                             rel_diff=abs(ns - nw) / max(ns, 1e-300), M=int(ni2))
    log(f"  V3 ||weak_allmodes|| {nw:.6e} vs ||strong_full|| {ns:.6e}  "
        f"rel diff {abs(ns-nw)/max(ns,1e-300):.2e}")
    rep["V3_ok"] = bool(abs(ns - nw) / max(ns, 1e-300) < 1e-10)

    # --- V4b: the ROM energy diagnostic applied to the exact FOM trajectory --
    # (v reconstructed from the SNAPSHOT-level trapezoidal recursion at dt=DT_SNAP
    #  is NOT the FOM's v; done at the sub-step level with RS=80 states it is)
    nmf = wc.make_newmark_fom(N, wc.FOM_SUBSTEPS)
    S80, E80 = nmf(jnp.asarray(U0[0]), float(ct[0]))
    rep["V4b_newmark_rs80_energy_drift"] = float(np.max(np.abs(np.asarray(E80) - np.asarray(E80)[0])
                                                        / np.asarray(E80)[0]))
    log(f"  V4b Newmark(RS=80) energy drift {rep['V4b_newmark_rs80_energy_drift']:.2e}")

    # --- V6: POD floors on the training/val split (traj-RMS metric) ---------
    d = wc.build_data(N)
    U = d["U"]
    S = U[:wc.N_TRAIN].reshape(-1, N * N)
    V, sv, dev = wc.pod_basis(S, kmax=64)
    U_va3 = U[wc.N_TRAIN:]
    U_va = U_va3.reshape(-1, N * N)
    rms_va = np.repeat(np.sqrt(np.mean(np.sum(U_va3 ** 2, axis=2), axis=1)), wc.NUM_STEPS + 1)
    floors = {}
    for r in (4, 6, 8, 16, 32, 64):
        if r <= V.shape[1]:
            rec = (U_va @ V[:, :r]) @ V[:, :r].T
            floors[r] = float(np.mean(np.linalg.norm(rec - U_va, axis=1) / rms_va))
    rep["V6_pod_floors_val"] = floors
    rep["V6_pod_ortho_dev"] = dev
    rep["V6_sv"] = [float(s) for s in sv[:32]]
    log("  V6 POD val floors: " + "  ".join(f"r{r}={e:.3e}" for r, e in floors.items())
        + f"   ortho dev {dev:.1e}")

    rep["all_ok"] = bool(rep.get("V1_ok", True) and rep["V2_ok"] and rep["V3_ok"])
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"wlat_verify_N{N}.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    log(f"wrote wlat_verify_N{N}.json  all_ok={rep['all_ok']}")


if __name__ == "__main__":
    main()
