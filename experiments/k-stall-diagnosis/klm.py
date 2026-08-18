"""Instrumented copy of ctol_tol.lm_tau_poisson: SAME damping schedule, SAME
acceptance test, SAME stopping tests, SAME lam0/clamps -- but a host loop so the
per-iteration trace (lam, |dz|, ||r||, accept, cond(J)) is observable."""
import numpy as np, jax
import jax.numpy as jnp
F64 = jnp.float64

def make(dec, K, pts, wq, PhiT, Wl):
    pts = jnp.asarray(pts); wq = jnp.asarray(wq)
    PhiT = jnp.asarray(PhiT); Wl = jnp.asarray(Wl)
    def r_of(z, f_m):
        return Wl * (PhiT @ (wq * dec(z, pts))) - f_m
    rJ = jax.jit(lambda z, f_m: (r_of(z, f_m), jax.jacfwd(r_of)(z, f_m)))
    rn = jax.jit(lambda z, f_m: jnp.linalg.norm(r_of(z, f_m)))
    return r_of, rJ, rn

def lm_trace(rJ, rn, K, z0, f_m, tau, budget=60, trace=False):
    z = jnp.asarray(z0)
    r, J = rJ(z, f_m); val = float(jnp.linalg.norm(r)); v0 = val
    tol = tau * v0
    lam = 1e-6; acc = 0; att = 0; reason = 2 if (tau > 0 and v0 <= tol) else 0
    tr = []
    eye = jnp.eye(K, dtype=F64)
    while reason == 0 and att < budget:
        H = J.T @ J; g = J.T @ r
        D = jnp.diag(jnp.diag(H)) + 1e-30 * eye
        dz = jnp.linalg.solve(H + lam * D, -g)
        finite = bool(jnp.all(jnp.isfinite(dz)))
        z_new = z + (dz if finite else 0.0)
        v_new = float(rn(z_new, f_m)) if finite else np.inf
        accept = finite and np.isfinite(v_new) and v_new < val
        rel_dec = (val - v_new) / (abs(val) + 1e-300) if accept else 1.0
        step = float(jnp.linalg.norm(dz)) / (1.0 + float(jnp.linalg.norm(z)))
        if trace:
            sv = np.linalg.svd(np.asarray(J), compute_uv=False)
            tr.append(dict(it=att, lam=lam, val=val, val_new=v_new, accept=bool(accept),
                           step=step, dznorm=float(jnp.linalg.norm(dz)),
                           znorm=float(jnp.linalg.norm(z)),
                           condJ=float(sv[0]/max(sv[-1],1e-300)),
                           gnorm=float(jnp.linalg.norm(g))))
        if accept:
            z = z_new; val = v_new
            r, J = rJ(z, f_m)
            lam = max(lam/3.0, 1e-12); acc += 1
        else:
            lam = min(lam*10.0, 1e12)
        att += 1
        if accept and tau > 0 and val <= tol:      reason = 2
        elif accept and (rel_dec < 1e-12 or step < 1e-13): reason = 1
        elif (not accept) and lam >= 1e12:         reason = 3
    return np.asarray(z), val, v0, att, acc, reason, tr
