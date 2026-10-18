"""jcp-fom2d full-order models (DESIGN §2–§3): F-UB upwind + BE and F-CC central + CN-R (E0 `refcore.make_solver`, d = 2,
byte-identical copy), F-PS (psjax2d). Every query maps (prepared input, nu) to the six output fields. FD: E0's compiled solve
followed by the compiled output map, dispatched back-to-back in one timed interval (A2.6); PS: one jitted program.

Spec keys: 'fd|<UB|CC>|L<L>|<dt>|t<level>' with level in 0..3 or X (control), 'ps|N<N>|<dt>'.
Output contracts: 'lattice' (257^2 with zero walls) or ('mesh', M) ((M+1)^2 with zero walls). FD at L >= target: nodes
by stride; L < target: sine (DST-I) interpolation, `sine_interp`.
"""
from __future__ import annotations

import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import refcore as FC          # E0 refcore, byte-identical copy (G-FD1 = sha256 identity, INHERITED.json)
import psjax2d
import cohorts as CH

E2 = CH.E2                    # vendor engines (2D): dst2, initial, make_fom
MAX_NEWTON = 20
TOLS = {'0': (1e-10, 1e-11), '1': (1e-8, 1e-9), '2': (1e-6, 1e-7), '3': (1e-4, 1e-5), 'X': (1e-2, 1e-3)}
NEWTON_LIMIT = {'X': 1}
FAM = dict(UB=('upwind', 'be'), CC=('central', 'cnr'))
LAT = 256


def canon_dt(x):
    k = round(0.05 / float(x))
    assert k >= 1 and abs(0.05 / k - float(x)) <= 1e-9 * float(x), x
    return 0.05 / k


def key(sp):
    if sp[0] == 'fd':
        return f'fd|{sp[1]}|L{sp[2]}|{sp[3]:.8g}|t{sp[4]}'
    return f'ps|N{sp[1]}|{sp[2]:.8g}'


def parse(k):
    p = k.split('|')
    if p[0] == 'fd':
        assert p[1] in FAM and p[4][0] == 't' and p[4][1:] in TOLS, k
        return ('fd', p[1], int(p[2][1:]), canon_dt(p[3]), p[4][1:])
    assert p[0] == 'ps' and p[1][0] == 'N', k
    return ('ps', int(p[1][1:]), canon_dt(p[2]))


def family(k):
    p = k.split('|')
    return p[1] if p[0] == 'fd' else 'PS'


# --------------------------------------------------------------------------------------------- output maps ----

def sine_interp(U, L, M):
    """Interior values U (..., L-1, L-1) on mesh L -> interior values (..., M-1, M-1) on mesh M (L | M) of the sine
    interpolant: orthonormal DST-I coefficients b, c = b * (M / L) (sqrt(M/L) per axis), zero-padded, inverse DST-I.
    Exact at the coincident nodes; exact for any field in span{sin(k pi x) sin(l pi y), k, l < L}."""
    assert M % L == 0 and M >= L, (L, M)
    if M == L:
        return U
    b = E2.dst2(U) * (M / L)
    pad = [(0, 0)] * (U.ndim - 2) + [(0, M - L), (0, M - L)]
    return E2.dst2(jnp.pad(b, pad))


def fd_map(L, target):
    """jit: flat interior outputs (6, (L-1)^2) on mesh L -> (6, T+1, T+1) on the target (T = 256 for 'lattice')."""
    T = LAT if target == 'lattice' else int(target[1])

    def f(out):
        U = out.reshape(6, L - 1, L - 1)
        if L >= T:
            assert L % T == 0, (L, T)
            s = L // T
            return jnp.pad(U, ((0, 0), (1, 1), (1, 1)))[:, ::s, ::s]
        return jnp.pad(sine_interp(U, L, T), ((0, 0), (1, 1), (1, 1)))
    return jax.jit(f)


# ------------------------------------------------------------------------------------------------- inputs -----

def fd_input(desc, L):
    """Nodal IC (engines.initial: Gaussian at every node, wall nodes zeroed) -> interior, flat, x slowest."""
    return jnp.asarray(np.asarray(E2.initial(L, desc), float)[1:-1, 1:-1].reshape(-1))


def prepare(sp, desc):
    """Prepared input of a spec (outside every timer); returns (input, seconds)."""
    t = time.perf_counter()
    x = fd_input(desc, sp[2]) if sp[0] == 'fd' else psjax2d.ic(desc, sp[1])
    x = jax.block_until_ready(x)
    return x, time.perf_counter() - t


# -------------------------------------------------------------------------------------------------- runner -----

class FOMs:
    """FD solvers cached per (L, adv, newton limit) (one AOT compile each; dt, scheme, tolerances traced), output maps per
    (L, target), PS queries per (N, dt, target)."""

    def __init__(self):
        self.fd, self.ps, self.maps = {}, {}, {}

    def solver(self, L, adv, lim):
        k = (L, adv, lim)
        if k not in self.fd:
            self.fd[k] = FC.make_solver(2, L, adv=adv, max_newton=lim)
        return self.fd[k]

    def fmap(self, L, target):
        k = (L, target)
        if k not in self.maps:
            self.maps[k] = fd_map(L, target)
        return self.maps[k]

    def run(self, sp, x, nu, target='lattice'):
        """-> (fields (6, T+1, T+1) device array, info). Untimed callers block on the result themselves."""
        if sp[0] == 'ps':
            k = (sp[1], sp[2], target)
            if k not in self.ps:
                self.ps[k] = psjax2d.make_ps_query(sp[1], sp[2], target)
            F, um = self.ps[k](x, nu)
            return F, dict(kind='ps', umax=um)
        _, fam, L, dt, lev = sp
        adv, scheme = FAM[fam]
        ntol, ltol = TOLS[lev]
        o = self.solver(L, adv, NEWTON_LIMIT.get(lev, MAX_NEWTON))(x, nu, dt, scheme, ntol, ltol)
        return self.fmap(L, target)(o['out']), o

    def callable(self, sp, x, nu, target):
        """A zero-argument function doing exactly the timed work of one query (solve + output map)."""
        if sp[0] == 'ps':
            k = (sp[1], sp[2], target)
            if k not in self.ps:
                self.ps[k] = psjax2d.make_ps_query(sp[1], sp[2], target)
            q = self.ps[k]
            return lambda: q(x, nu)[0]        # the running maximum is computed in the same program (validity evidence)
        _, fam, L, dt, lev = sp
        adv, scheme = FAM[fam]
        ntol, ltol = TOLS[lev]
        s = self.solver(L, adv, NEWTON_LIMIT.get(lev, MAX_NEWTON))
        m = self.fmap(L, target)
        return lambda: m(s(x, nu, dt, scheme, ntol, ltol)['out'])

    @staticmethod
    def converged(sp, info, F):
        """FD (speed3d A2.5/A4.2): finite output; complete per-stage evidence (stage count = schedule, time units reach
        0.25); every recorded relative residual finite and <= ntol; Newton counts in [0, limit]; a stage stopped at the
        Newton limit with its residual met counts as converged and is flagged. PS: finite output and finite running
        maximum of |u| (A2.6). The evidence dict persists the per-stage residuals and Newton counts and the stage/unit
        counts, so `match2.evidence_ok` recomputes the predicate offline."""
        fin = bool(np.isfinite(np.asarray(F)).all())
        if sp[0] == 'ps':
            um = float(np.asarray(info['umax']))
            return bool(fin and np.isfinite(um)), dict(finite=fin, umax=um)
        _, fam, L, dt, lev = sp
        ntol = TOLS[lev][0]
        lim = NEWTON_LIMIT.get(lev, MAX_NEWTON)
        scheme = FAM[fam][1]
        k = int(np.asarray(info['stages']))
        want = FC.n_stages(dt, scheme)
        units_ok = int(np.asarray(info['units'])) == 2 * int(round(0.25 / dt))
        resall, nwtall = np.asarray(info['res']), np.asarray(info['nwt'])
        complete = bool(k == want and units_ok and k <= len(resall))
        res, nwt = resall[:k], nwtall[:k]
        fin = fin and bool(np.isfinite(np.asarray(info['out'])).all())
        ok = bool(fin and complete and np.all(np.isfinite(res)) and np.all(res <= ntol) and np.all((nwt >= 0) & (nwt <= lim)))
        return ok, dict(stages=k, want_stages=want, units=int(np.asarray(info['units'])), want_units=2 * int(round(0.25 / dt)),
                        ntol=ntol, newton_limit=lim, res=[float(x) for x in res], nwt=[int(x) for x in nwt],
                        complete=complete, finite=fin, worst_res=float(np.max(res)) if len(res) else None,
                        max_newton=int(nwt.max()) if len(nwt) else None, newton_total=int(nwt.sum()) if len(nwt) else None,
                        at_newton_limit=int(np.sum(nwt >= lim)) if len(nwt) else None, umax=float(np.asarray(info['umax'])))
