"""Stage 1 pilot: SPACE-TIME LSPG with the (z,t)-conditioned FiLM decoder from
the burgers2d sweep (burgers2d_film_N{N}.pkl, 5-dim true-parameter latent).

The ROM knows u0, nu and the PDE; it never sees the held-out trajectory.  We
solve for the 5-dim z by LM on:
  (a) ic     : misfit of u(.,0;z) to u0                       [IC-fit control]
  (b) resid  : the BE residual over all 50 steps, the decoder supplying both
               u(.,t_{n+1};z) and u(.,t_n;z)                  [space-time residual]
  (c) both   : (a) and (b) stacked (IC block weighted by IC_W)
and report trajectory rel-L2 vs the FOM, next to the ORACLE (true z) error and
the latent error |z - z_true|.  Inits: family mean and the z of the training
trajectory whose IC is nearest to u0 (legit: u0 known); best-of on the
objective (never on the held-out error).

Usage: N=64 python blat_stage1.py <ckpt.pkl> <outdir>
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

import blat_common as bc
from blat_common import bf, F64, log, lm_solve

CKPT = sys.argv[1]
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
N = bc.N
T1 = bc.NUM_STEPS + 1
IC_W = float(os.environ.get("IC_W", "1.0"))
BUDGET = int(os.environ.get("S1_BUDGET", "100"))


def main():
    log(f"jax_backend={jax.default_backend()}  N={N}  n_freq={bf.N_FREQ} t_freq={bf.T_FREQ}")
    with open(CKPT, "rb") as f:
        params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=F64),
                                        pickle.load(f))
    d = bc.build_data(N)
    U, z_all, nu_all = d["U"], d["z"].astype(np.float64), d["nu"]
    U_te = U[bc.N_TRAIN:bc.N_TRAIN + bc.N_TEST]
    z_te = z_all[bc.N_TRAIN:bc.N_TRAIN + bc.N_TEST]
    nu_te = nu_all[bc.N_TRAIN:bc.N_TRAIN + bc.N_TEST]
    z_tr = z_all[:bc.N_TRAIN]
    U_tr0 = U[:bc.N_TRAIN, 0]
    coords = jnp.asarray(bc.grid_coords(N))
    interior = jnp.asarray(bc.interior_indices(N))
    st = jnp.asarray(bc.stencil_indices(np.asarray(interior), N))
    taus = jnp.asarray(np.arange(T1) / bc.NUM_STEPS)

    def field(z, tau):
        return bf.film_apply(params, z, tau, coords)

    def traj(z):
        return jax.vmap(lambda t: field(z, t))(taus)                 # (T1, n^2)

    def r_ic(z, u0):
        return field(z, 0.0) - u0

    def r_res(z, nu):
        Uz = traj(z)
        def one(n):
            return bc.be_residual_from_stencil(Uz[n + 1][st], Uz[n][interior], nu, N)
        return jax.vmap(one)(jnp.arange(bc.NUM_STEPS)).reshape(-1)

    def r_both(z, u0, nu):
        return jnp.concatenate([IC_W * r_ic(z, u0), r_res(z, nu)])

    objs = {
        "ic": (jax.jit(lambda z, u0, nu: (r_ic(z, u0), jax.jacfwd(r_ic)(z, u0))),
               jax.jit(lambda z, u0, nu: jnp.linalg.norm(r_ic(z, u0)))),
        "resid": (jax.jit(lambda z, u0, nu: (r_res(z, nu), jax.jacfwd(r_res)(z, nu))),
                  jax.jit(lambda z, u0, nu: jnp.linalg.norm(r_res(z, nu)))),
        "both": (jax.jit(lambda z, u0, nu: (r_both(z, u0, nu),
                                            jax.jacfwd(r_both)(z, u0, nu))),
                 jax.jit(lambda z, u0, nu: jnp.linalg.norm(r_both(z, u0, nu)))),
    }
    traj_j = jax.jit(traj)

    def traj_err(z, Ut):
        P = np.asarray(traj_j(jnp.asarray(z)))
        per = np.linalg.norm(P - Ut, axis=1) / np.linalg.norm(Ut, axis=1)
        return float(per.mean()), per

    report = dict(config=dict(bc.CONFIG, ic_w=IC_W, budget=BUDGET, ckpt=os.path.basename(CKPT),
                              n_freq=bf.N_FREQ, t_freq=bf.T_FREQ),
                  backend=jax.default_backend(), data_fingerprint=bc.data_fingerprint(U))
    zmean = z_tr.mean(axis=0)
    orc = [traj_err(z_te[i], U_te[i])[0] for i in range(bc.N_TEST)]
    report["oracle_true_z"] = dict(traj_rel_mean=float(np.mean(orc)),
                                   traj_rel_max=float(np.max(orc)))
    log(f"  ORACLE (true z) traj rel mean {np.mean(orc):.3e}")
    # residual of the FOM trajectory through the same residual (sanity ~1e-13)
    Uz = jnp.asarray(U_te[0])
    rr = jax.vmap(lambda n: bc.be_residual_from_stencil(Uz[n + 1][st], Uz[n][interior],
                                                        float(nu_te[0]), N))(
        jnp.arange(bc.NUM_STEPS))
    report["fom_traj_rel_res"] = float(jnp.linalg.norm(rr) / jnp.linalg.norm(Uz))
    # residual value at the true z (the decoder's own PDE inconsistency)
    rt = [float(objs["resid"][1](jnp.asarray(z_te[i]), jnp.asarray(U_te[i, 0]), float(nu_te[i]))
                / np.linalg.norm(U_te[i])) for i in range(bc.N_TEST)]
    report["resid_at_true_z_rel"] = float(np.mean(rt))
    log(f"  FOM traj rel residual {report['fom_traj_rel_res']:.2e}; "
        f"decoder residual at true z (rel) {np.mean(rt):.3e}")

    for name, (rJ, rn) in objs.items():
        t0 = time.time()
        rows = []
        for i in range(bc.N_TEST):
            u0 = jnp.asarray(U_te[i, 0]); nu = float(nu_te[i])
            j = int(np.argmin(np.linalg.norm(U_tr0 - U_te[i, 0], axis=1)))
            best = None
            for iname, z0 in (("mean", zmean), ("nearest_ic", z_tr[j])):
                z, r, info = lm_solve(lambda zz: rJ(zz, u0, nu), lambda zz: rn(zz, u0, nu),
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
    with open(os.path.join(OUTDIR, f"blat_stage1_N{N}.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log("wrote stage1 json")


if __name__ == "__main__":
    main()
