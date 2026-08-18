"""EXPERIMENT 2 (hypotheses B, E, F): what the stall looks like.

(1) Per-iteration LM trace (lam, ||dz||, ||r||, accept, cond(J) ALONG THE PATH) for
    a failing (k, source) and a working one.
(2) 1-D slice of the objective and of the field error along the segment
    z0 -> z_ceiling, to see whether there is a barrier / a spurious minimum.
"""
import json, sys
import numpy as np, jax
import jax.numpy as jnp
import kcommon as kc, klm

M = 64; TAU = 1e-3; BUDGET = 60; NN = 64
CASES = [(12, 9), (12, 14), (12, 0), (8, 14), (8, 9), (6, 14), (6, 0), (32, 14), (32, 0)]

grid, Fs, U, tn, nn_idx, par = kc.testbed(NN)
spec, pts, wq, PhiT, Wl = kc.weak_full(grid, M)
print(f"jax_backend={jax.default_backend()}", flush=True)
cache = {}
out = {}
for k, i in CASES:
    if k not in cache:
        dec, Z = kc.load(k)
        r_of, rJ, rn = klm.make(dec, k, pts, wq, PhiT, Wl)
        u_full = jax.jit(lambda z, _d=dec: _d(z, grid.coords_int))
        mis = lambda z, u: u_full(z) - u
        mrJ = jax.jit(lambda z, u: (mis(z, u), jax.jacfwd(mis)(z, u)))
        mrn = jax.jit(lambda z, u: jnp.linalg.norm(mis(z, u)))
        cache[k] = (dec, Z, rJ, rn, u_full, mrJ, mrn)
    dec, Z, rJ, rn, u_full, mrJ, mrn = cache[k]
    f_m = kc.fm(grid, spec, Fs[i]); ui = jnp.asarray(U[i].reshape(-1))
    z_mean = jnp.asarray(Z.mean(0)); z_near = jnp.asarray(Z[nn_idx[i]])
    best = None
    for nm, z0 in (("mean", z_mean), ("nearest", z_near)):
        zc, v, _, _, _, _, _ = klm.lm_trace(mrJ, mrn, k, z0, ui, 0.0, 300)
        if best is None or v < best[1]: best = (zc, v, nm)
    z_ceil = jnp.asarray(best[0]); ceil_err = best[1]/tn[i]
    z, val, v0, att, acc, rsn, tr = klm.lm_trace(rJ, rn, k, z_mean, f_m, TAU, BUDGET, trace=True)
    err = float(np.linalg.norm(np.asarray(u_full(jnp.asarray(z))).reshape(grid.n_i,grid.n_i)-U[i])/tn[i])
    print(f"\n### k={k} src={i}  ceiling {ceil_err:.3e}  ROM {err:.3e}  "
          f"obj0 {v0:.4e} -> {val:.4e} (red {val/v0:.2e})  obj@ceil {float(rn(z_ceil,f_m)):.4e}  "
          f"iters {att} acc {acc} reason {rsn}  |z0-zceil| {float(np.linalg.norm(np.asarray(z_mean)-np.asarray(z_ceil))):.3f} "
          f"|zfin-zceil| {float(np.linalg.norm(z-np.asarray(z_ceil))):.3f}", flush=True)
    print("   it   lam       ||r||       ||r_new||   acc  ||dz||    ||z||   cond(J)   ||g||")
    for t in tr:
        print(f"   {t['it']:3d} {t['lam']:.2e} {t['val']:.4e} {t['val_new']:.4e} "
              f"{'A' if t['accept'] else '.'}  {t['dznorm']:.3e} {t['znorm']:.3f} "
              f"{t['condJ']:.3e} {t['gnorm']:.3e}", flush=True)
    # ---- slice along z0 -> z_ceil, extended past both ends
    ts = np.linspace(-0.3, 1.6, 39)
    sl = []
    for t in ts:
        zz = jnp.asarray((1-t)*np.asarray(z_mean) + t*np.asarray(z_ceil))
        e = float(np.linalg.norm(np.asarray(u_full(zz)).reshape(grid.n_i,grid.n_i)-U[i])/tn[i])
        sl.append((float(t), float(rn(zz, f_m)), e))
    print("   slice t: obj / err")
    print("   " + "  ".join(f"{t:+.2f}:{o:.3e}/{e:.2e}" for t,o,e in sl[::3]), flush=True)
    out[f"k{k}_s{i}"] = dict(k=k, src=i, ceil=ceil_err, rom=err, obj0=v0, obj=val,
                             obj_ceil=float(rn(z_ceil,f_m)), trace=tr, slice=sl,
                             reason=rsn, iters=att, acc=acc)
json.dump(out, open("exp2.json","w"), indent=1, default=float)
