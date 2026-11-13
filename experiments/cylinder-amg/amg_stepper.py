"""AMG variants of the frozen-Oseen linear solve (PROTOCOL.md of this lane).

AmgStepper subclasses the vendored transient.Stepper (unchanged) and overrides only the frozen-scheme refresh
(prepare_frozen) and solve.  Solver spec (dict):
  kind   : 'lu' (original) | 'amg1' (block preconditioner, one V-cycle per block) | 'amg2' (blocks solved by AMG-Krylov)
  rtol   : outer relative residual tolerance ||r - K d|| <= rtol ||r||, r = rhs - K x_bc (free unknowns)
  amg    : 'sa' (smoothed aggregation) | 'rs' (classical Ruge-Stueben)
  schur  : 'cc' (Cahouet-Chabard: coef*Lp^-1 + nu*Mp^-1) | 'mass' ((1/nu) Mp^-1)
  inner_rtol : inner tolerance for amg2 (default 1e-8)
  restart, maxiter : outer FGMRES restart (60) and iteration cap (300)
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import pyamg
from pyamg import krylov

import transient as tr


def to_scipy(m):
    r, c, v = m.COO()
    return sp.csr_matrix((np.asarray(v), (np.asarray(r), np.asarray(c))), shape=(m.height, m.width))


def build_amg(A, kind, spd):
    if kind == 'sa':
        if spd:
            return pyamg.smoothed_aggregation_solver(A, max_coarse=50)
        return pyamg.smoothed_aggregation_solver(A, symmetry='nonsymmetric', max_coarse=50)
    if kind == 'rs':
        return pyamg.ruge_stuben_solver(A, max_coarse=50)
    raise ValueError(kind)


class AmgStepper(tr.Stepper):
    def __init__(self, cfg, out, ng, solver):
        super().__init__(cfg, out, ng)
        self.sv = dict(solver)
        self.st = {'setups': 0, 'extract_seconds': 0.0, 'amg_setup_seconds': 0.0, 'krylov_seconds': 0.0,
                   'outer_iters': [], 'inner_iters': 0, 'final_rel': [], 'levels': None, 'blocks_identical': None,
                   'pressure_setup_seconds': 0.0}
        self._hist = []
        if self.sv['kind'] == 'lu':
            return
        t0 = time.perf_counter()
        nv = self.V.ndof
        n = nv // 2
        fr = self.free
        self.fidx = np.flatnonzero(fr)
        self.ivx = np.flatnonzero(fr[:n])
        self.ivy = n + np.flatnonzero(fr[n:nv])
        self.ip = nv + np.flatnonzero(fr[nv:])
        if len(self.ip) != self.Q.ndof:
            raise AssertionError('pressure has constrained dofs')
        # local positions inside the free vector
        pos = -np.ones(self.X.ndof, dtype=np.int64); pos[self.fidx] = np.arange(len(self.fidx))
        self.lx, self.ly, self.lp = pos[self.ivx], pos[self.ivy], pos[self.ip]
        if not np.array_equal(np.r_[self.lx, self.ly, self.lp], np.arange(len(self.fidx))):
            raise AssertionError('free unknowns are not ordered [ux, uy, p]')
        ng = self.ng
        p, q = self.Q.TnT()
        mp = ng.BilinearForm(self.Q); mp += p * q * ng.dx; mp.Assemble()
        lp = ng.BilinearForm(self.Q); lp += ng.grad(p) * ng.grad(q) * ng.dx; lp.Assemble()
        Mp = to_scipy(mp.mat); Lp = to_scipy(lp.mat).tolil()
        out_free = np.array([bool(b) for b in ng.H1(self.mesh, order=self.cfg['pressure_order'], dirichlet='outlet').FreeDofs()])
        od = np.flatnonzero(~out_free)
        if len(od) == 0:
            raise AssertionError('no outlet pressure dofs')
        Lp[od, :] = 0; Lp[:, od] = 0; Lp[od, od] = 1.0
        self.Lp = Lp.tocsr()
        self.Mp_dinv = 1.0 / Mp.diagonal()
        self.ml_p = build_amg(self.Lp, self.sv.get('amg', 'sa'), spd=True)
        self.Pp = self.ml_p.aspreconditioner(cycle='V')
        self.st['pressure_setup_seconds'] = time.perf_counter() - t0
        self.st['pressure_levels'] = len(self.ml_p.levels)

    # ---- refresh: same assembly as the original, AMG setup instead of LU
    def prepare_frozen(self, coef, w, force=False):
        if self.sv['kind'] == 'lu':
            return super().prepare_frozen(coef, w, force)
        c = self.cfg
        if not force and self.fo_key == coef and self.fo_age < c['refactor_every']:
            self.fo_age += 1
            return False
        t0 = time.perf_counter()
        ng = self.ng
        self.fo_form = self._factor(coef, w)
        self.fo_C = ng.BilinearForm(self.X)
        self.fo_C += (ng.Grad(self.u) * self.wgf) * self.v * self.dx
        self.fo_C.Assemble()
        t1 = time.perf_counter()
        K = to_scipy(self.fo_form.mat)
        self.Kfr = K[self.fidx, :].tocsr()          # free rows, all columns (for r = rhs - K x_bc)
        self.Kff = self.Kfr[:, self.fidx].tocsr()
        Fx = self.Kff[self.lx][:, self.lx].tocsr()
        Fy = self.Kff[self.ly][:, self.ly].tocsr()
        Bt = self.Kff[np.r_[self.lx, self.ly]][:, self.lp].tocsr()
        Fxy = self.Kff[self.lx][:, self.ly]
        if Fxy.nnz and abs(Fxy).max() > 0:
            raise AssertionError('velocity components are coupled in F')
        same = Fx.shape == Fy.shape and abs(Fx - Fy).max() <= 1e-14 * abs(Fx).max()
        t2 = time.perf_counter()
        kind = self.sv.get('amg', 'sa')
        self.ml_x = build_amg(Fx, kind, spd=False)
        self.ml_y = self.ml_x if same else build_amg(Fy, kind, spd=False)
        self.Px, self.Py = self.ml_x.aspreconditioner(cycle='V'), self.ml_y.aspreconditioner(cycle='V')
        self.Fx, self.Fy, self.Bt = Fx, Fy, Bt
        self.coef = coef
        t3 = time.perf_counter()
        self.st['setups'] += 1
        self.st['extract_seconds'] += t2 - t1
        self.st['amg_setup_seconds'] += t3 - t2
        self.st['blocks_identical'] = bool(same)
        self.st['levels'] = len(self.ml_x.levels)
        self.fo_key, self.fo_age = coef, 1
        self.factorizations += 1
        self.check_next = True
        self.timers['factor'] += time.perf_counter() - t0
        return True

    # ---- block preconditioner
    def _fsolve(self, P, A, b):
        if self.sv['kind'] == 'amg1':
            return P @ b
        res = []
        x, info = krylov.gmres(A, b, x0=np.zeros_like(b), tol=self.sv.get('inner_rtol', 1e-8), restart=40, maxiter=10,
                               M=P, residuals=res)
        self.st['inner_iters'] += len(res) - 1
        if info != 0:
            raise ArithmeticError(f'inner velocity solve did not converge ({info})')
        return x

    def _lpsolve(self, b):
        if self.sv['kind'] == 'amg1':
            return self.Pp @ b
        res = []
        x, info = krylov.cg(self.Lp, b, x0=np.zeros_like(b), tol=self.sv.get('inner_rtol', 1e-8), maxiter=400,
                            M=self.Pp, residuals=res)
        self.st['inner_iters'] += len(res) - 1
        if info != 0:
            raise ArithmeticError(f'inner pressure solve did not converge ({info})')
        return x

    def _prec(self, r):
        nx, ny = len(self.lx), len(self.ly)
        ru, rp = r[:nx + ny], r[nx + ny:]
        # pressure: zp = S^-1 rp,  S^-1 ~ -(coef Lp^-1 + nu Mp^-1)  (cc)  or  -(1/nu) Mp^-1  (mass)
        if self.sv.get('schur', 'cc') == 'cc':
            zp = -(self.coef * self._lpsolve(rp) + self.nu * self.Mp_dinv * rp)
        else:
            zp = -(self.Mp_dinv * rp) / self.nu
        gu = ru - self.Bt @ zp
        zx = self._fsolve(self.Px, self.Fx, gu[:nx])
        zy = self._fsolve(self.Py, self.Fy, gu[nx:])
        z = np.empty_like(r)
        z[:nx], z[nx:nx + ny], z[nx + ny:] = zx, zy, zp
        return z

    def solve(self, coef, rhs, t_new, w=None, frozen=False):
        if self.sv['kind'] == 'lu' or not frozen:
            return super().solve(coef, rhs, t_new, w=w, frozen=frozen)
        t0 = time.perf_counter()
        xb = self.bc(t_new)
        r = rhs[self.fidx] - self.Kfr @ xb
        rnorm = float(np.linalg.norm(r))
        # initial guess: previous solution (linear extrapolation when the two previous solves share coef)
        h = [z for z in self._hist if z[0] == coef]
        if len(h) >= 2:
            g = 2 * h[-1][1] - h[-2][1]
        elif len(h) == 1:
            g = h[-1][1]
        else:
            g = xb
        d0 = (g - xb)[self.fidx]
        M = spl.LinearOperator(self.Kff.shape, matvec=self._prec, dtype=float)
        res = []
        tk = time.perf_counter()
        if rnorm == 0:
            d, info = np.zeros(len(r)), 0
        else:
            d, info = krylov.fgmres(self.Kff, r, x0=d0, tol=self.sv['rtol'], restart=self.sv.get('restart', 60),
                                    maxiter=self.sv.get('maxiter', 300) // self.sv.get('restart', 60), M=M, residuals=res)
        self.st['krylov_seconds'] += time.perf_counter() - tk
        rel = float(np.linalg.norm(r - self.Kff @ d)) / max(rnorm, 1e-300)
        self.st['outer_iters'].append(max(len(res) - 1, 0))
        self.st['final_rel'].append(rel)
        if rel > self.sv["rtol"] * 1.5:
            raise ArithmeticError(f'AMG Krylov relative residual {rel:.3e} > {self.sv["rtol"]} at t={t_new} (info {info})')
        x = xb.copy()
        x[self.fidx] += d
        self._hist = (self._hist + [(coef, x.copy())])[-2:]
        if self.check_next:
            self.lin_checks.append({'t': t_new, 'relative_linear_residual': rel})
            self.check_next = False
        self.timers['solve'] += time.perf_counter() - t0
        return x

    def solver_stats(self):
        s = dict(self.st)
        it = np.asarray(s.pop('outer_iters'), dtype=float)
        fr = np.asarray(s.pop('final_rel'), dtype=float)
        s['solver'] = self.sv
        if len(it):
            s.update(outer_iters_mean=float(it.mean()), outer_iters_max=int(it.max()), outer_iters_total=int(it.sum()),
                     solves=int(len(it)), final_rel_max=float(fr.max()))
        return s
