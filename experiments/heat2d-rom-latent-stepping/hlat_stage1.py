"""Stage 1 pilot: SPACE-TIME LSPG with the (z,t)-conditioned FiLM decoder from
the heat2d sweep (heat2d_film_N{N}.pkl, 5-dim true-parameter latent).

The ROM knows u0, kappa and the PDE; it never sees the held-out trajectory.  We
solve for the 5-dim z by LM on:
  (a) ic     : misfit of u(.,0;z) to u0                       [IC-fit control]
  (b) resid  : the BE residual over all 50 steps (interior + FOM boundary
               rows), the decoder supplying both u(.,t_{n+1};z) and u(.,t_n;z)
               -- a PDE-CONSISTENCY ABLATION, not a ROM (it never sees u0, so
               the low-amplitude branch of the family is an attractor)
  (c) both   : (a) and (b) stacked (IC block weighted by IC_W)
and report trajectory rel-L2 vs the FOM, next to the ORACLE (true z) error and
the latent error |z - z_true|.  Inits: family mean and the z of the training
trajectory whose IC is nearest to u0 (legit: u0 known), both with the
diffusivity coordinate set from the KNOWN kappa; best-of on the objective (never on
the held-out error).  Test trajectories come from TEST_SEED (the sweep
checkpoint was model-selected on the VAL split, so VAL is not used here).

Usage: N=64 python hlat_stage1.py <ckpt.pkl> <outdir>
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import hlat_common as bc
from hlat_common import hf, F64, log, lm_solve

CKPT = sys.argv[1]
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
N = bc.N
T1 = bc.NUM_STEPS + 1
# IC block weight: sqrt(NUM_STEPS) makes the single IC slice count like one
# residual slice per step (RMS-balanced); IC_W=1 reported as a sensitivity arm
IC_W = float(os.environ.get("IC_W", str(np.sqrt(50.0))))
BUDGET = int(os.environ.get("S1_BUDGET", "100"))


def main():
    log(f"jax_backend={jax.default_backend()}  N={N}  n_freq={hf.N_FREQ} t_freq={hf.T_FREQ}")
    with open(CKPT, "rb") as f:
        params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=F64),
                                        pickle.load(f))
    # checkpoint manifest check (the sweep's results JSON sits next to the pkl)
    rj = os.path.join(os.path.dirname(CKPT), f"heat2d_results_N{N}.json")
    if not os.path.exists(rj):
        raise SystemExit(f"missing checkpoint manifest {rj}: architecture unverifiable")
    with open(rj) as f:
        rmeta = json.load(f)
    for k_, v_ in (("N", N), ("n_freq", hf.N_FREQ), ("t_freq", hf.T_FREQ),
                   ("hidden", hf.HIDDEN), ("n_train", hf.N_TRAIN), ("seed", hf.SEED)):
        if rmeta.get(k_) != v_:
            raise SystemExit(f"checkpoint manifest mismatch {k_}: {rmeta.get(k_)} vs {v_}")
    n_feats = 2 * (2 * hf.N_FREQ + 1) + (2 * hf.T_FREQ + 1)
    got = params["trunk"][0]["W"].shape[0]
    if got != n_feats:
        raise SystemExit(f"checkpoint trunk input width {got} != coord features {n_feats}")
    d = bc.build_data(N)
    U, z_all = d["U"], d["z"].astype(np.float64)
    U_te, z_te, kap_te = d["U_test"], d["z_test"].astype(np.float64), d["kappa_test"]
    z_tr = z_all[:bc.N_TRAIN]
    U_tr0 = U[:bc.N_TRAIN, 0]
    coords = jnp.asarray(bc.grid_coords(N))
    interior = jnp.asarray(bc.interior_indices(N))
    bnd = jnp.asarray(np.setdiff1d(np.arange(N * N), np.asarray(interior)))
    st = jnp.asarray(bc.stencil_indices(np.asarray(interior), N))
    taus = jnp.asarray(np.arange(T1) / bc.NUM_STEPS)
    _, fom_residual = bc.make_fom(N)

    def field(z, tau):
        return hf.film_apply(params, z, tau, coords)

    def traj(z):
        return jax.vmap(lambda t: field(z, t))(taus)                 # (T1, n^2)

    def r_ic(z, u0):
        return field(z, 0.0) - u0

    def r_res(z, kap):
        """BE residual over all 50 steps INCLUDING the FOM's boundary rows
        (R = u_{n+1} on the walls -- the FOM's wall row is u_{n+1} - u_n with
        u_n = 0 there; the sweep decoder is not hard-BC, so its wall values must
        be penalized exactly as the FOM does)."""
        Uz = traj(z)
        def one(n):
            return jnp.concatenate([
                bc.be_residual_from_stencil(Uz[n + 1][st], Uz[n][interior], kap, N),
                Uz[n + 1][bnd] - Uz[n][bnd]])
        return jax.vmap(one)(jnp.arange(bc.NUM_STEPS)).reshape(-1)

    def r_both(z, u0, kap):
        return jnp.concatenate([IC_W * r_ic(z, u0), r_res(z, kap)])

    objs = {
        "ic": (jax.jit(lambda z, u0, kap: (r_ic(z, u0), jax.jacfwd(r_ic)(z, u0))),
               jax.jit(lambda z, u0, kap: jnp.linalg.norm(r_ic(z, u0)))),
        "resid": (jax.jit(lambda z, u0, kap: (r_res(z, kap), jax.jacfwd(r_res)(z, kap))),
                  jax.jit(lambda z, u0, kap: jnp.linalg.norm(r_res(z, kap)))),
        "both": (jax.jit(lambda z, u0, kap: (r_both(z, u0, kap),
                                             jax.jacfwd(r_both)(z, u0, kap))),
                 jax.jit(lambda z, u0, kap: jnp.linalg.norm(r_both(z, u0, kap)))),
    }
    traj_j = jax.jit(traj)

    def traj_err(z, Ut):
        P = np.asarray(traj_j(jnp.asarray(z)))
        per = np.linalg.norm(P - Ut, axis=1) / np.linalg.norm(Ut, axis=1)
        return float(per.mean()), per

    report = dict(config=dict(bc.CONFIG, ic_w=IC_W, budget=BUDGET, ckpt=os.path.basename(CKPT),
                              n_freq=hf.N_FREQ, t_freq=hf.T_FREQ),
                  backend=jax.default_backend(), data_fingerprint=bc.data_fingerprint(U))
    zmean = z_tr.mean(axis=0)
    orc = [traj_err(z_te[i], U_te[i])[0] for i in range(bc.N_TEST)]
    report["oracle_true_z"] = dict(traj_rel_mean=float(np.mean(orc)),
                                   traj_rel_max=float(np.max(orc)))
    log(f"  ORACLE (true z) traj rel mean {np.mean(orc):.3e}")
    # residual of the FOM trajectory through the FOM's OWN residual, all rows
    # (independent sanity ~1e-13), and through ours (must agree)
    Uz = jnp.asarray(U_te[0])
    rr = jax.vmap(lambda n: fom_residual(Uz[n + 1], Uz[n], float(kap_te[0])))(
        jnp.arange(bc.NUM_STEPS))
    report["fom_traj_rel_res"] = float(jnp.linalg.norm(rr) / jnp.linalg.norm(Uz))
    # scatter our interior+boundary rows back into the n^2 layout and compare ROW BY ROW
    def scatter(n):
        v = jnp.zeros(N * N, dtype=F64)
        v = v.at[interior].set(bc.be_residual_from_stencil(
            Uz[n + 1][st], Uz[n][interior], float(kap_te[0]), N))
        return v.at[bnd].set(Uz[n + 1][bnd] - Uz[n][bnd])
    r_ours = jax.vmap(scatter)(jnp.arange(bc.NUM_STEPS))
    report["ours_vs_fom_traj_res_maxabs_diff"] = float(jnp.max(jnp.abs(rr - r_ours)))
    if report["ours_vs_fom_traj_res_maxabs_diff"] > 1e-10 * float(jnp.max(jnp.abs(Uz))):
        raise SystemExit("stage-1 residual does not reproduce the FOM residual row by row")
    # the trajectory above has exactly-zero walls, so it cannot exercise the boundary
    # rows: repeat on a RANDOM pair of fields with non-zero walls (the sweep decoder is
    # not hard-BC, so its wall rows really do enter the objective)
    rg = np.random.default_rng(7)
    ua, ub = (jnp.asarray(rg.normal(size=N * N)), jnp.asarray(rg.normal(size=N * N)))
    r_fom_rand = fom_residual(ua, ub, float(kap_te[0]))
    r_our_rand = (jnp.zeros(N * N, dtype=F64)
                  .at[interior].set(bc.be_residual_from_stencil(
                      ua[st], ub[interior], float(kap_te[0]), N))
                  .at[bnd].set(ua[bnd] - ub[bnd]))
    report["ours_vs_fom_random_field_maxabs_diff"] = float(
        jnp.max(jnp.abs(r_fom_rand - r_our_rand)))
    if report["ours_vs_fom_random_field_maxabs_diff"] > 1e-9 * float(jnp.max(jnp.abs(r_fom_rand))):
        raise SystemExit("stage-1 residual differs from the FOM residual on a random field")
    log(f"  residual identity: FOM-trajectory {report['ours_vs_fom_traj_res_maxabs_diff']:.1e}, "
        f"random-field {report['ours_vs_fom_random_field_maxabs_diff']:.1e}")
    # residual value at the true z (the decoder's own PDE inconsistency)
    rt = [float(objs["resid"][1](jnp.asarray(z_te[i]), jnp.asarray(U_te[i, 0]), float(kap_te[i]))
                / np.linalg.norm(U_te[i])) for i in range(bc.N_TEST)]
    report["resid_at_true_z_rel"] = float(np.mean(rt))
    log(f"  FOM traj rel residual {report['fom_traj_rel_res']:.2e}; "
        f"decoder residual at true z (rel) {np.mean(rt):.3e}")

    for name, (rJ, rn) in objs.items():
        t0 = time.time()
        rows = []
        for i in range(bc.N_TEST):
            u0 = jnp.asarray(U_te[i, 0]); kap = float(kap_te[i])
            j = int(np.argmin(np.linalg.norm(U_tr0 - U_te[i, 0], axis=1)))
            # known kappa -> the 5th (normalized log kappa) coordinate, heat2d_film.sample_params
            z_kap = (np.log(kap) - np.log(np.sqrt(0.005))) / (0.5 * np.log(50.0))
            z0_mean = zmean.copy(); z0_mean[4] = z_kap
            z0_near = z_tr[j].copy(); z0_near[4] = z_kap
            best = None
            for iname, z0 in (("mean", z0_mean), ("nearest_ic", z0_near)):
                z, r, info = lm_solve(lambda zz: rJ(zz, u0, kap), lambda zz: rn(zz, u0, kap),
                                      jnp.asarray(z0), BUDGET)
                if best is None or r < best[1]:
                    best = (np.asarray(z), r, iname, info)
            z, r, iname, info = best
            e, per = traj_err(z, U_te[i])
            rows.append(dict(traj_rel=e, z_err=float(np.linalg.norm(z - z_te[i])),
                             init=iname, obj=float(r), accepted=info["accepted"],
                             reason=info["reason"], per_time=per.tolist()))
        tr = np.array([r["traj_rel"] for r in rows])
        report[name] = dict(traj_rel_mean=float(tr.mean()), traj_rel_median=float(np.median(tr)),
                            traj_rel_max=float(tr.max()),
                            z_err_mean=float(np.mean([r["z_err"] for r in rows])),
                            per_time_mean=np.mean([r["per_time"] for r in rows], axis=0).tolist(),
                            rows=rows, secs=time.time() - t0)
        log(f"  {name:6s}: traj rel mean {tr.mean():.3e} (med {np.median(tr):.3e}, "
            f"max {tr.max():.3e})  |z-z*| {report[name]['z_err_mean']:.3e}  "
            f"[{time.time()-t0:.0f}s]")

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"hlat_stage1_N{N}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log("wrote stage1 json")


if __name__ == "__main__":
    main()
