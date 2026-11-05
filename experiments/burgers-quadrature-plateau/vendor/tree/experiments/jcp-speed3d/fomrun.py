"""jcp-speed3d full-order models (DESIGN H4, A1.9, A2.5): F-UB upwind + BE, F-UC upwind + CN-R, F-CC central + CN-R
(lane copy `fomcore.py` of E0 refcore, import-only change), F-PS (psjax). Every FOM query maps (prepared input, nu) to
the six fields on the 63^3 scoring lattice on the device (lattice slicing inside the timed call).

Specs: ('fd', fam, L, dt, tol_level) with fam in UB | UC | CC and tol_level 0 (1e-8, 1e-9), 1 (1e-6, 1e-7),
2 (1e-4, 1e-5); ('ps', N, dt).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import fomcore as FC
import psjax

MAX_NEWTON = 20          # fomcore.make_solver default (E0 refcore)
TOLS = ((1e-8, 1e-9), (1e-6, 1e-7), (1e-4, 1e-5))
FAM = dict(UB=('upwind', 'be'), UC=('upwind', 'cnr'), CC=('central', 'cnr'))


def key(sp):
    if sp[0] == 'fd':
        return f'fd|{sp[1]}|L{sp[2]}|{sp[3]:.8g}|t{sp[4]}'
    return f'ps|N{sp[1]}|{sp[2]:.8g}'


def parse(k):
    p = k.split('|')
    if p[0] == 'fd':
        return ('fd', p[1], int(p[2][1:]), S_canon(p[3]), int(p[4][1:]))
    return ('ps', int(p[1][1:]), S_canon(p[2]))


def S_canon(x):
    k = round(0.05 / float(x))
    assert k >= 1 and abs(0.05 / k - float(x)) <= 1e-7 * float(x), x
    return 0.05 / k


def lattice_index(L):
    s = L // 64
    assert s * 64 == L, L
    idx = np.arange(1, 64) * s - 1
    return jnp.asarray(idx)


class FOMs:
    """Solvers cached per (L, adv) (one AOT compile each, E0 refcore) and PS queries per (N, dt)."""

    def __init__(self):
        self.fd, self.ps, self.lat = {}, {}, {}

    def solver(self, L, adv):
        k = (L, adv)
        if k not in self.fd:
            self.fd[k] = FC.make_solver(3, L, adv=adv)
        return self.fd[k]

    def run(self, sp, u_in, nu):
        """-> (lattice fields (6, 63^3) device array, info dict). u_in: interior field (FD, flat) or sine coefficients (PS)."""
        if sp[0] == 'ps':
            k = (sp[1], sp[2])
            if k not in self.ps:
                self.ps[k] = psjax.make_ps_query(sp[1], sp[2])
            f = self.ps[k](u_in, nu)
            return f, dict(converged=True)
        _, fam, L, dt, lev = sp
        adv, scheme = FAM[fam]
        ntol, ltol = TOLS[lev]
        o = self.solver(L, adv)(u_in, nu, dt, scheme, ntol, ltol)
        if L not in self.lat:
            idx = lattice_index(L)
            n1 = L - 1
            self.lat[L] = jax.jit(lambda out: out.reshape(6, n1, n1, n1)[:, idx][:, :, idx][:, :, :, idx].reshape(6, -1))
        f = self.lat[L](o['out'])
        return f, o

    @staticmethod
    def converged(sp, info, F=None):
        """A2.5 (audit code-1 #8): finite lattice output; FD: complete per-stage evidence (stages <= buffer, every recorded
        relative residual finite and <= ntol, Newton counts recorded); a stage that stopped at the Newton limit counts as
        converged only if its residual met ntol (the refcore loop exits on either), and is flagged."""
        fin = F is None or bool(np.isfinite(np.asarray(F)).all())
        if sp[0] == 'ps':
            return fin, dict(finite=fin)
        ntol = TOLS[sp[4]][0]
        _, fam, L, dt, _ = sp
        scheme = FAM[fam][1]
        k = int(np.asarray(info['stages']))
        want = FC.n_stages(dt, scheme)
        units_ok = int(np.asarray(info['units'])) == 2 * int(round(0.25 / dt))
        resall, nwtall = np.asarray(info['res']), np.asarray(info['nwt'])
        complete = bool(k == want and units_ok and k <= len(resall) and k <= len(nwtall))
        res, nwt = resall[:k], nwtall[:k]
        fin = fin and bool(np.isfinite(np.asarray(info['out'])).all())
        # A4.2: a stage is converged iff its recorded relative residual <= ntol (the refcore loop exits on that test or at the
        # Newton limit; a limit exit with the residual met counts as converged and is flagged); every count in [0, limit]
        ok = bool(fin and complete and np.all(np.isfinite(res)) and np.all(res <= ntol)
                  and np.all((nwt >= 0) & (nwt <= MAX_NEWTON)))
        return ok, dict(stages=k, complete=complete, worst_res=float(np.max(res)) if len(res) else None,
                        max_newton=int(nwt.max()) if len(nwt) else None,
                        at_newton_limit=int(np.sum(nwt >= MAX_NEWTON)) if len(nwt) else None)
