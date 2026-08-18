"""EXPERIMENT 1 (hypotheses C + D): where does the weak-form minimum actually sit?

For every k and every one of the 16 held-out sources, run the SAME LM on the SAME
weak objective from THREE initial guesses:
    mean      = mean training latent      (what the ROM actually does)
    nearest   = latent of the nearest training sample in PARAMETER space
    ceiling   = the ORACLE latent (LM fit of z to the true field)
and report, for each, the final field error AND the final objective value.

The comparison obj(z_from_ceiling) vs obj(z_from_mean) splits the diagnosis:
  obj(ceil-start) >  obj(mean-start)  -> the mean-start run found a LOWER objective
                                          than the neighbourhood of the ceiling, i.e.
                                          the OBJECTIVE's minimum is not at the ceiling.
  obj(ceil-start) <  obj(mean-start)  -> the mean-start run STALLED above a better
                                          objective value it failed to reach.
"""
import json, sys, time
import numpy as np, jax
import jax.numpy as jnp
import kcommon as kc, klm

M = int(sys.argv[1]) if len(sys.argv) > 1 else 64
KS = [4, 6, 8, 12, 16, 24, 32]
TAU = 1e-3
BUDGET = 60
NN = 64

grid, Fs, U, tn, nn_idx, par = kc.testbed(NN)
spec, pts, wq, PhiT, Wl = kc.weak_full(grid, M)
f_ms = [kc.fm(grid, spec, Fs[i]) for i in range(kc.N_TEST)]
n_i = grid.n_i
print(f"jax_backend={jax.default_backend()} N={NN} M={M} n_modes={PhiT.shape[0]} "
      f"pts={pts.shape[0]} tau={TAU} budget={BUDGET}", flush=True)

out = []
for k in KS:
    dec, Z = kc.load(k)
    r_of, rJ, rn = klm.make(dec, k, pts, wq, PhiT, Wl)
    u_full = jax.jit(lambda z, _d=dec: _d(z, grid.coords_int))
    # ---- oracle ceiling latent: LM on the data misfit, best of the two inits
    mis = lambda z, u: u_full(z) - u
    mrJ = jax.jit(lambda z, u: (mis(z, u), jax.jacfwd(mis)(z, u)))
    mrn = jax.jit(lambda z, u: jnp.linalg.norm(mis(z, u)))
    z_mean = jnp.asarray(Z.mean(0))
    rows = []
    for i in range(kc.N_TEST):
        ui = jnp.asarray(U[i].reshape(-1))
        z_near = jnp.asarray(Z[nn_idx[i]])
        best = None
        for nm, z0 in (("mean", z_mean), ("nearest", z_near)):
            zc, v, _, _, _, _, _ = klm.lm_trace(mrJ, mrn, k, z0, ui, 0.0, 300)
            if best is None or v < best[1]:
                best = (zc, v, nm)
        z_ceil, ceil_err, ceil_init = best[0], best[1]/tn[i], best[2]
        rec = dict(k=k, src=i, ceil_err=ceil_err, ceil_init=ceil_init,
                   obj_ceil=float(rn(jnp.asarray(z_ceil), f_ms[i])))
        for nm, z0 in (("mean", z_mean), ("nearest", z_near),
                       ("ceil", jnp.asarray(z_ceil))):
            z, val, v0, att, acc, rsn, _ = klm.lm_trace(rJ, rn, k, z0, f_ms[i],
                                                        TAU, BUDGET)
            e = float(np.linalg.norm(np.asarray(u_full(jnp.asarray(z))).reshape(n_i, n_i)
                                     - U[i]) / tn[i])
            rec[nm] = dict(err=e, obj=val, obj0=v0, red=val/v0, iters=att,
                           acc=acc, reason=rsn,
                           dz_from_ceil=float(np.linalg.norm(np.asarray(z) - z_ceil)))
        rows.append(rec); out.append(rec)
    em = np.array([r["mean"]["err"] for r in rows])
    ec = np.array([r["ceil"]["err"] for r in rows])
    en = np.array([r["nearest"]["err"] for r in rows])
    cl = np.array([r["ceil_err"] for r in rows])
    print(f"k={k:2d} ceiling mean {cl.mean():.3e} | ROM<-mean {em.mean():.3e} "
          f"(med {np.median(em):.3e}) | ROM<-nearest {en.mean():.3e} "
          f"(med {np.median(en):.3e}) | ROM<-ceil {ec.mean():.3e} (med {np.median(ec):.3e})",
          flush=True)
    for r in rows:
        if r["mean"]["err"] > 5*np.median(em) or r["ceil"]["err"] > 5*np.median(ec):
            print(f"    src {r['src']:2d} ceil {r['ceil_err']:.3e} | "
                  f"mean: e {r['mean']['err']:.3e} obj {r['mean']['obj']:.3e} "
                  f"red {r['mean']['red']:.2e} it {r['mean']['iters']} rsn {r['mean']['reason']} "
                  f"|dz-ceil| {r['mean']['dz_from_ceil']:.2f} | "
                  f"ceil-start: e {r['ceil']['err']:.3e} obj {r['ceil']['obj']:.3e} "
                  f"red {r['ceil']['red']:.2e} it {r['ceil']['iters']} rsn {r['ceil']['reason']} "
                  f"|dz-ceil| {r['ceil']['dz_from_ceil']:.2f} | obj@ceil {r['obj_ceil']:.3e}",
                  flush=True)
json.dump(out, open(f"exp1_M{M}.json", "w"), indent=1, default=float)
