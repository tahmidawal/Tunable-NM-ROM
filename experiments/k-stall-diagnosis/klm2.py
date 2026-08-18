"""LM variants for the fix test.  `base` is character-for-character the current
solver (ctol_tol.lm_tau_poisson); the others change ONE thing each."""
import numpy as np, jax
import jax.numpy as jnp
F64 = jnp.float64

def lm(rJ, rn, K, z0, f_m, tau, budget=60, variant="base", lam0=1e-6, delta=np.inf,
       rho_min=0.0):
    """variant:
      base    : lam0=1e-6, accept iff ||r_new|| < ||r||          (CURRENT SOLVER)
      lam0    : identical but a larger initial damping
      tr      : base + a TRUST REGION -- any step with ||dz|| > delta is refused
                and treated as a failed step (lam *= 10), so the iterate can never
                leave the neighbourhood of the training latent cloud in one jump
      nielsen : proper gain-ratio LM (accept iff rho > rho_min; lam scaled by
                max(1/3, 1-(2rho-1)^3) on accept, doubled nu on reject)
    """
    z = jnp.asarray(z0)
    r, J = rJ(z, f_m); val = float(jnp.linalg.norm(r)); v0 = val
    tol = tau * v0
    lam = lam0; nu = 2.0; acc = 0; att = 0
    reason = 2 if (tau > 0 and v0 <= tol) else 0
    eye = jnp.eye(K, dtype=F64)
    while reason == 0 and att < budget:
        H = J.T @ J; g = J.T @ r
        D = jnp.diag(jnp.diag(H)) + 1e-30 * eye
        dz = jnp.linalg.solve(H + lam * D, -g)
        finite = bool(jnp.all(jnp.isfinite(dz)))
        dzn = float(jnp.linalg.norm(dz)) if finite else np.inf
        too_long = dzn > delta
        z_new = z + (dz if finite and not too_long else 0.0)
        v_new = float(rn(z_new, f_m)) if (finite and not too_long) else np.inf
        if variant == "nielsen":
            pred = float(val**2 - jnp.sum((r + J @ dz)**2)) if finite else -1.0
            rho = ((val**2 - v_new**2) / pred) if (pred > 0 and np.isfinite(v_new)) else -1.0
            accept = finite and (not too_long) and np.isfinite(v_new) and rho > rho_min
        else:
            accept = finite and (not too_long) and np.isfinite(v_new) and v_new < val
        rel_dec = (val - v_new) / (abs(val) + 1e-300) if accept else 1.0
        step = dzn / (1.0 + float(jnp.linalg.norm(z))) if finite else 0.0
        if accept:
            z = z_new; val = v_new; r, J = rJ(z, f_m); acc += 1
            if variant == "nielsen":
                lam = max(lam * max(1.0/3.0, 1.0 - (2.0*rho - 1.0)**3), 1e-12); nu = 2.0
            else:
                lam = max(lam / 3.0, 1e-12)
        else:
            if variant == "nielsen":
                lam = min(lam * nu, 1e12); nu = min(nu * 2.0, 1e8)
            else:
                lam = min(lam * 10.0, 1e12)
        att += 1
        if accept and tau > 0 and val <= tol:               reason = 2
        elif accept and (rel_dec < 1e-12 or step < 1e-13):  reason = 1
        elif (not accept) and lam >= 1e12:                  reason = 3
    return np.asarray(z), val, v0, att, acc, reason
