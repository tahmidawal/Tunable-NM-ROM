"""EXPERIMENT 5 (hypothesis G): is the k pattern structural, or a 16-sample lottery?

Re-runs the SAME base solver on ALL 64 held-out sources (the study uses only the
first 16) at N=32, and counts blow-ups per k.  If the good/bad k set is a property
of k (a Fourier-feature / FiLM resonance), it must survive a 4x larger, disjoint
sample.  If it is a per-source lottery it will move.
Also records, per (k, source), the norm of the iterate after the first accepted
step relative to the radius of the training latent cloud -- the quantity that
separated the working from the failing solves in experiment 2.
"""
import json, sys
import numpy as np, jax
import jax.numpy as jnp
import kcommon as kc, klm, klm2
import ms_parametric as mp

NN = int(sys.argv[1]) if len(sys.argv) > 1 else 32
M = 64; TAU = 1e-3; BUDGET = 60
KS = [4, 6, 8, 12, 16, 24, 32]
NSRC = 64

cx, cy, w, a, _z = mp.sample_params()
grid = kc.pc.Grid(NN)
Fs = np.stack([mp.source_interior(NN, cx[kc.N_TRAIN+i], cy[kc.N_TRAIN+i],
                                  w[kc.N_TRAIN+i], a[kc.N_TRAIN+i]) for i in range(NSRC)])
op = lambda v: mp.neg_lap_interior(v, NN)
solve = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(op, F, tol=mp.CG_TOL,
                                                     maxiter=mp.CG_MAXITER)[0])
U = np.asarray(jax.lax.map(solve, jnp.asarray(Fs)))
tn = np.array([np.linalg.norm(U[i]) for i in range(NSRC)])
spec, pts, wq, PhiT, Wl = kc.weak_full(grid, M)
f_ms = [kc.fm(grid, spec, Fs[i]) for i in range(NSRC)]
print(f"jax_backend={jax.default_backend()} N={NN} sources={NSRC}", flush=True)

out = []
for k in KS:
    dec, Z = kc.load(k)
    _, rJ, rn = klm.make(dec, k, pts, wq, PhiT, Wl)
    u_full = jax.jit(lambda z, _d=dec: _d(z, grid.coords_int))
    z_mean = jnp.asarray(Z.mean(0))
    R = float(np.max(np.linalg.norm(Z - Z.mean(0), axis=1)))
    for arm, kw in (("base", dict(variant="base", lam0=1e-6)),
                    ("tr",   dict(variant="tr", lam0=1e-6, delta=R))):
        errs, esc = [], []
        for i in range(NSRC):
            z, val, v0, att, acc, rsn = klm2.lm(rJ, rn, k, z_mean, f_ms[i], TAU, BUDGET, **kw)
            e = float(np.linalg.norm(np.asarray(u_full(jnp.asarray(z))).reshape(
                grid.n_i, grid.n_i) - U[i]) / tn[i])
            errs.append(e); esc.append(float(np.linalg.norm(z - np.asarray(z_mean)) / R))
        errs = np.array(errs); esc = np.array(esc)
        nb = int((errs > 5*np.median(errs)).sum())
        print(f"k={k:2d} {arm:4s} R_train={R:.3f} mean {errs.mean():.3e} med {np.median(errs):.3e} "
              f"blown {nb:2d}/{NSRC} idx {list(np.where(errs>5*np.median(errs))[0])[:12]} "
              f"| escape ratio med {np.median(esc):.2f} max {esc.max():.2f}", flush=True)
        out.append(dict(N=NN, k=k, arm=arm, R=R, err=[float(v) for v in errs],
                        esc=[float(v) for v in esc], n_blown=nb,
                        mean=float(errs.mean()), med=float(np.median(errs))))
        json.dump(out, open(f"exp5_N{NN}.json","w"), indent=1, default=float)
