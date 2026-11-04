"""Transient curved Taylor-Hood reference for DFG 2D-2 (G5, DESIGN-G5.md).

Geometry, spaces, gradient-form weak form, outlet and force/normal conventions are
those of exp/2026-10-27-ns-obstacle-2d fom.py; the mesh is the admitted curved
binary mesh loaded by bytes (never regenerated or re-curved).

Schemes (both second order, one linear solve per step):
  sbdf2      BDF2 + implicit Stokes/viscous + extrapolated explicit convection
             2N(u^n)-N(u^{n-1}); constant matrix, factored once per (dt, nu).
  oseen_bdf2 BDF2 with linearly implicit convection (grad u) w, w=2u^n-u^{n-1};
             matrix reassembled every step, LU refreshed every `refactor_every`
             steps and used as GMRES preconditioner in between.
  frozen_oseen BDF2 with implicit convection (grad u) w_f, w_f = u^n frozen and the LU
             refreshed every `refactor_every` steps, plus the explicit second-order
             extrapolated correction 2R(u^n)-R(u^{n-1}), R(u)=(grad u)u-(grad u)w_f
             (amendment G5-A1, after the smoke showed sbdf2 unstable at U=1).
Startup from a single state: `startup_substeps` first-order IMEX/Oseen Euler
substeps spanning one dt, then BDF2 (variable first interval is avoided).
All arrays float64. NGSolve kernels run on CPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from scalars import scalar_value  # noqa: E402
import cycles as cyc  # noqa: E402

EXIT_WALL = 75


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Stepper:
    def __init__(self, cfg, out, ng):
        self.cfg, self.out, self.ng = cfg, Path(out), ng
        c = cfg
        self.nu, self.H = float(c['nu']), float(c['H'])
        self.D = 2 * float(c['radius'])
        mesh_path = Path(c['mesh_path'])
        if sha(mesh_path) != c['mesh_sha256']:
            raise ValueError('mesh bytes do not match the admitted mesh hash')
        self.mesh = mesh = ng.Mesh(str(mesh_path))
        if int(mesh.GetCurveOrder()) != c['geometry_order']:
            raise ValueError('loaded curve order mismatch')
        from oriented_normals import fluid_outward_normal
        self.normal, cert = fluid_outward_normal(mesh, c['center'], order=c['diagnostic_order'],
                                                 evidence_path=self.out / 'normal_orientation.json')
        if not cert['passed']:
            raise ArithmeticError('normal orientation certificate failed')
        k = c['velocity_order']
        if c['pressure_order'] != k - 1 or k < 2:
            raise ValueError('Taylor-Hood k/(k-1) required')
        self.V = ng.VectorH1(mesh, order=k, dirichlet='inlet|wall.*|obstacle')
        self.Q = ng.H1(mesh, order=k - 1)
        self.X = X = self.V * self.Q
        if X.ndof > c['max_dofs']:
            raise MemoryError(f'DOF guard {X.ndof}')
        self.free = np.array([bool(b) for b in X.FreeDofs()], dtype=bool)
        lay = hashlib.sha256()
        lay.update(np.array([self.V.ndof, self.Q.ndof, X.ndof], dtype=np.int64).tobytes())
        lay.update(np.packbits(self.free).tobytes())
        self.layout_sha256 = lay.hexdigest()
        self.row_scale = np.r_[np.full(self.V.ndof, 1.0 ** 2 * self.D), np.full(self.Q.ndof, 1.0 * self.D)]
        (u, p), (v, q) = X.TnT()
        self.u, self.p, self.v, self.q = u, p, v, q
        self.dx = ng.dx(bonus_intorder=c['assembly_bonus_order'])
        grad = ng.Grad
        self.visc = self.nu * ng.InnerProduct(grad(u), grad(v)) - p * ng.div(v) - q * ng.div(u)
        m = ng.BilinearForm(X); m += u * v * self.dx; m.Assemble(); self.M = m.mat
        a = ng.BilinearForm(X); a += self.visc * self.dx; a.Assemble(); self.A = a.mat
        self.conv = ng.BilinearForm(X, nonassemble=True)
        self.conv += (grad(u) * u) * v * self.dx
        # Unit-mean inlet profile; the time factor multiplies this Dirichlet vector.
        gb = ng.GridFunction(X)
        y = ng.y
        gb.components[0].Set(ng.CF((6 * y * (self.H - y) / self.H ** 2, 0)), definedon=mesh.Boundaries('inlet'))
        self.bc_unit = gb.vec.FV().NumPy().copy()
        if np.any(self.bc_unit[self.free] != 0):
            raise AssertionError('inlet trace touched free DOFs')
        kick = c.get('kick')
        self.kick_vec = np.zeros(X.ndof)
        if kick:
            xc, yc = kick['center']
            shape = ng.exp(-((ng.x - xc) ** 2 + (ng.y - yc) ** 2) / (2 * kick['sigma'] ** 2))
            f = ng.LinearForm(X); f += ng.CF((0, kick['amplitude'] * shape)) * v * self.dx; f.Assemble()
            self.kick_vec = f.vec.FV().NumPy().copy()
        # Reaction test vectors: unit velocity component on obstacle trace DOFs.
        self.chi = []
        for i in range(2):
            t = ng.GridFunction(X)
            t.components[0].Set(ng.CF(tuple(1.0 if j == i else 0.0 for j in range(2))),
                                definedon=mesh.Boundaries('obstacle'))
            self.chi.append(t.vec.FV().NumPy().copy())
        self.g = ng.GridFunction(X)
        self.uh, self.ph = self.g.components
        gub = ng.BoundaryFromVolumeCF(grad(self.uh))
        self.tau_n = (self.nu * gub - self.ph * ng.Id(2)) * self.normal
        self.probe_pts = [mesh(c['center'][0] - c['radius'], c['center'][1]),
                          mesh(c['center'][0] + c['radius'], c['center'][1])]
        if any(mp.nr < 0 for mp in self.probe_pts):
            raise ValueError('exact pressure probe outside the FE mesh')
        self.tmp = [self.g.vec.CreateVector() for _ in range(3)]
        self.solver_key = None
        self.wgf = ng.GridFunction(self.V)
        self.lin_checks = []
        self.timers = {k: 0.0 for k in ('conv_apply', 'factor', 'solve', 'observe', 'diagnostics')}
        self.fo_key, self.fo_age = None, 0

    # ---- vector helpers -------------------------------------------------
    def apply(self, op, x):
        a, b = self.tmp[0], self.tmp[1]
        a.FV().NumPy()[:] = x
        if op == 'conv':
            t0 = time.perf_counter()
            self.conv.Apply(a, b)
            self.timers['conv_apply'] += time.perf_counter() - t0
        elif op == 'Cf':
            b.data = self.fo_C.mat * a
        else:
            b.data = {'M': self.M, 'A': self.A}[op] * a
        return b.FV().NumPy().copy()

    def prepare_frozen(self, coef, w, force=False):
        """(Re)assemble/factor coef*M + A + C(w_f) when due; returns True on refresh."""
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
        self.fo_inv = self.fo_form.mat.Inverse(self.X.FreeDofs(), inverse=c['inverse_backend'])
        self.fo_key, self.fo_age = coef, 1
        self.factorizations += 1
        self.check_next = True
        self.timers['factor'] += time.perf_counter() - t0
        return True

    def U_mean(self, t):
        c = self.cfg
        tr, U0, U1 = c['ramp_time'], c['U_mean0'], c['U_mean']
        return U1 if t >= tr else U0 + (U1 - U0) * math.sin(math.pi * t / (2 * tr)) ** 2

    def forcing(self, t):
        k = self.cfg.get('kick')
        if not k or not (k['t_start'] <= t <= k['t_start'] + k['duration']):
            return 0.0 * self.kick_vec
        return math.sin(math.pi * (t - k['t_start']) / k['duration']) ** 2 * self.kick_vec

    def bc(self, t):
        return self.U_mean(t) * self.bc_unit

    # ---- linear solves --------------------------------------------------
    def _factor(self, coef, w=None):
        ng = self.ng
        form = ng.BilinearForm(self.X)
        integrand = coef * self.u * self.v + self.visc
        if w is not None:
            self.wgf.vec.FV().NumPy()[:] = w[:self.V.ndof]
            integrand = integrand + (ng.Grad(self.u) * self.wgf) * self.v
        form += integrand * self.dx
        form.Assemble()
        return form

    def solve(self, coef, rhs, t_new, w=None, frozen=False):
        """Solve K x = rhs with x = bc(t_new) on Dirichlet DOFs."""
        c = self.cfg
        ng = self.ng
        oseen = w is not None
        t_solve0 = time.perf_counter()
        if frozen:
            self.form, self.inv = self.fo_form, self.fo_inv
        elif not oseen:
            key = ('sbdf', coef)
            if self.solver_key != key:
                self.form = self._factor(coef)
                self.inv = self.form.mat.Inverse(self.X.FreeDofs(), inverse=c['inverse_backend'])
                self.solver_key = key
                self.factorizations += 1
        else:
            self.form = self._factor(coef, w)
            refresh = self.solver_key is None or self.solver_key[0] != 'oseen' or self.solver_key[1] != coef \
                or self.oseen_age >= c['refactor_every']
            if refresh:
                self.inv = self.form.mat.Inverse(self.X.FreeDofs(), inverse=c['inverse_backend'])
                self.solver_key = ('oseen', coef)
                self.oseen_age = 0
                self.factorizations += 1
            self.oseen_age += 1
        xb = self.bc(t_new)
        r, d, kx = self.tmp
        r.FV().NumPy()[:] = xb
        kx.data = self.form.mat * r
        r.FV().NumPy()[:] = rhs - kx.FV().NumPy()
        rnorm = float(np.linalg.norm((r.FV().NumPy() / self.row_scale)[self.free]))
        if oseen and self.oseen_age > 1:
            from ngsolve.solvers import GMRes
            d.FV().NumPy()[:] = 0
            GMRes(A=self.form.mat, b=r, pre=self.inv, x=d,
                  tol=c['linear_relative_tolerance'] * 1e-2, maxsteps=c['gmres_maxsteps'], printrates=False)
        else:
            d.data = self.inv * r
        x = xb + d.FV().NumPy()
        x[~self.free] = xb[~self.free]
        if self.check_next or oseen:
            kx.FV().NumPy()[:] = x
            r.data = self.form.mat * kx
            res = (r.FV().NumPy() - rhs) / self.row_scale
            rel = float(np.linalg.norm(res[self.free])) / max(rnorm, 1e-300)
            self.lin_checks.append({'t': t_new, 'relative_linear_residual': rel})
            if rel > c['linear_relative_tolerance']:
                raise ArithmeticError(f'linear residual {rel:.3e} at t={t_new}')
            self.check_next = False
        self.timers['solve'] += time.perf_counter() - t_solve0
        return x

    # ---- outputs --------------------------------------------------------
    def observe(self, x, dudt, Nx, t):
        t0 = time.perf_counter()
        try:
            return self._observe(x, dudt, Nx, t)
        finally:
            self.timers['observe'] += time.perf_counter() - t0

    def _observe(self, x, dudt, Nx, t):
        ng = self.ng
        self.g.vec.FV().NumPy()[:] = x
        res = self.apply('M', 0 * x if dudt is None else dudt) + self.apply('A', x) + Nx - self.forcing(t)
        react = [-float(ch @ res) for ch in self.chi]
        qo = self.cfg['diagnostic_order']
        surf = [-scalar_value(ng.Integrate(self.tau_n[i], self.mesh, definedon=self.mesh.Boundaries('obstacle'),
                                           order=qo), 'surface_force') for i in range(2)]
        norm = 0.5 * self.cfg['U_mean'] ** 2 * self.D
        probes = [scalar_value(self.ph(mp), 'probe') for mp in self.probe_pts]
        E = 0.5 * float(x @ self.apply('M', x))
        return [t, react[0] / norm, react[1] / norm, surf[0] / norm, surf[1] / norm,
                probes[0], probes[1], E]

    def steady_forces(self, x, U):
        """R1a: forces of the Newton state (exact zero time derivative), Re-20 scaling."""
        row = self.observe(x, 0 * x, self.apply('conv', x), 0.0)
        scale = 1.0 / U ** 2  # observe() normalizes by U_mean=1; rescale to U_mean0
        rec = {'U_mean': U, 'nu': self.nu, 'CD_reaction': row[1] * scale, 'CL_reaction': row[2] * scale,
               'CD_surface': row[3] * scale, 'CL_surface': row[4] * scale, 'dp': row[5] - row[6],
               'admitted_P4P3_reference': self.cfg.get('steady_reference')}
        ref = rec['admitted_P4P3_reference']
        if ref:
            rec['CD_relative_error'] = abs(rec['CD_reaction'] - ref['CD']) / abs(ref['CD'])
            rec['R1a_drag_pass'] = rec['CD_relative_error'] <= 2e-3
        dump(self.out / 'steady_forces.json', rec)
        return rec

    def diagnostics(self, x, dudt, t):
        t0 = time.perf_counter()
        try:
            return self._diagnostics(x, dudt, t)
        finally:
            self.timers['diagnostics'] += time.perf_counter() - t0

    def _diagnostics(self, x, dudt, t):
        ng = self.ng
        self.g.vec.FV().NumPy()[:] = x
        uh, ph, n = self.uh, self.ph, self.normal
        qo = self.cfg['diagnostic_order']
        mesh = self.mesh
        def integ(cf, b=None):
            if b:
                return scalar_value(ng.Integrate(cf, mesh, definedon=mesh.Boundaries(b), order=qo), 'bint')
            return scalar_value(ng.Integrate(cf, mesh, order=qo), 'vint')
        div = ng.div(uh)
        vol = integ(1)
        bnames = sorted(set(mesh.GetBoundaries()))
        gub = ng.BoundaryFromVolumeCF(ng.Grad(uh))
        tau = self.nu * gub - ph * ng.Id(2)
        qin, qout = -integ(uh * n, 'inlet'), integ(uh * n, 'outlet')
        terms = {'dissipation': self.nu * integ(ng.InnerProduct(ng.Grad(uh), ng.Grad(uh))),
                 'convection_boundary': sum(0.5 * integ((uh * uh) * (uh * n), b) for b in bnames),
                 'gradient_traction_work': -sum(integ(uh * (tau * n), b) for b in bnames),
                 'finite_divergence_convection': -0.5 * integ((uh * uh) * div),
                 'finite_divergence_pressure': -integ(ph * div)}
        body = -float(self.forcing(t) @ x)
        terms['body_force_work'] = body
        if dudt is not None:
            dt_gf = ng.GridFunction(self.X); dt_gf.vec.FV().NumPy()[:] = dudt
            terms['kinetic_rate'] = integ(uh * dt_gf.components[0])
        return {'t': t, 'divergence': self.D * math.sqrt(max(integ(div * div), 0)) / (self.U_mean(t) * math.sqrt(vol)),
                'Qin': qin, 'Qout': qout, 'mass': abs(qout - qin) / abs(qin),
                'energy_terms': terms,
                'energy_balance': (abs(sum(terms.values())) / max(sum(abs(z) for z in terms.values()), 1e-12)
                                   if dudt is not None else None)}

    # ---- initial states -------------------------------------------------
    def steady(self, U):
        """Damped Newton steady state at inlet mean U (Stokes start)."""
        c = self.cfg
        ng = self.ng
        g = self.g
        res_form = ng.BilinearForm(self.X)
        res_form += (self.visc + (ng.Grad(self.u) * self.u) * self.v) * self.dx
        x = U * self.bc_unit
        # Stokes start
        r, d, kx = self.tmp
        r.FV().NumPy()[:] = x
        kx.data = self.A * r
        r.FV().NumPy()[:] = -kx.FV().NumPy()
        d.data = self.A.Inverse(self.X.FreeDofs(), inverse=c['inverse_backend']) * r
        x = x + d.FV().NumPy()
        def resid(xx):
            g.vec.FV().NumPy()[:] = xx
            out = g.vec.CreateVector()
            res_form.Apply(g.vec, out)
            return out.FV().NumPy().copy()
        def fn(rr):
            return float(np.linalg.norm((rr / self.row_scale)[self.free]))
        r0 = fn(resid(x)); den = max(r0, 1.0)
        hist = []
        for it in range(c['newton_max'] + 1):
            rr = resid(x); val = fn(rr)
            hist.append({'iteration': it, 'relative_residual': val / den})
            if val / den <= c['nonlinear_relative_tolerance']:
                break
            if it == c['newton_max']:
                raise ArithmeticError('steady Newton iteration limit')
            g.vec.FV().NumPy()[:] = x
            res_form.AssembleLinearization(g.vec)
            r.FV().NumPy()[:] = rr
            d.data = res_form.mat.Inverse(self.X.FreeDofs(), inverse=c['inverse_backend']) * r
            for damp in c['damping_sequence']:
                cand = x - damp * d.FV().NumPy()
                if fn(resid(cand)) <= (1 - 1e-4 * damp) * val:
                    x = cand; hist[-1]['damping'] = damp; break
            else:
                raise ArithmeticError('no Newton damping accepted')
        dump(self.out / 'steady_newton_history.json', hist)
        return x

    def load_state(self, spec):
        ng = self.ng
        path = Path(spec['path'])
        if sha(path) != spec['sha256']:
            raise ValueError('state hash mismatch')
        z = np.load(path)
        if str(z['mesh_sha256']) != self.cfg['mesh_sha256']:
            raise ValueError('state on another mesh')
        k = int(z['velocity_order'])
        if k == self.cfg['velocity_order']:
            if str(z['layout_sha256']) != self.layout_sha256:
                raise ValueError('state layout mismatch')
            return np.array(z['x'], dtype=np.float64), float(z['t'])
        # Exact polynomial embedding P_k -> P_k' (k' > k) on the same curved mesh.
        if k > self.cfg['velocity_order']:
            raise ValueError('only order-raising interpolation is allowed')
        Vs = ng.VectorH1(self.mesh, order=k, dirichlet='inlet|wall.*|obstacle')
        Xs = Vs * ng.H1(self.mesh, order=k - 1)
        gs = ng.GridFunction(Xs); gs.vec.FV().NumPy()[:] = z['x']
        bo = self.cfg['assembly_bonus_order']
        self.g.components[0].Set(gs.components[0], bonus_intorder=bo); self.g.components[1].Set(gs.components[1], bonus_intorder=bo)
        x = self.g.vec.FV().NumPy().copy()
        # G5-A5 witness: embedded fields equal the source fields to roundoff (L2, high order).
        qo = 2 * self.cfg['velocity_order'] + 6
        du = self.g.components[0] - gs.components[0]; dp = self.g.components[1] - gs.components[1]
        ev = math.sqrt(max(scalar_value(ng.Integrate(du * du, self.mesh, order=qo), 'emb'), 0)) / \
            math.sqrt(scalar_value(ng.Integrate(gs.components[0] * gs.components[0], self.mesh, order=qo), 'emb'))
        ep = math.sqrt(max(scalar_value(ng.Integrate(dp * dp, self.mesh, order=qo), 'emb'), 0)) / \
            math.sqrt(scalar_value(ng.Integrate(gs.components[1] * gs.components[1], self.mesh, order=qo), 'emb'))
        self.interpolation = {'from_order': k, 'to_order': self.cfg['velocity_order'],
                              'velocity_relative_L2_change': ev, 'pressure_relative_L2_change': ep, 'bound': 1e-10}
        if not (ev <= 1e-10 and ep <= 1e-10):
            raise ArithmeticError(f'embedding is not exact: velocity {ev:.3e}, pressure {ep:.3e}')
        return x, float(z['t'])

    def save_state(self, name, x, t):
        p = self.out / f'{name}.npz'
        np.savez(p, x=x, t=t, nu=self.nu, velocity_order=self.cfg['velocity_order'],
                 mesh_sha256=self.cfg['mesh_sha256'], layout_sha256=self.layout_sha256)
        return {'file': p.name, 'sha256': sha(p), 't': t}

    # ---- time loop ------------------------------------------------------
    def run(self):
        c = self.cfg
        self.factorizations = 0
        self.check_next = True
        t_wall0 = time.perf_counter()
        def on_term(signum, frame):
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, on_term)  # main() writes FAILED.json for pre-loop kills
        init = c['initial']
        steady_start = init['kind'] == 'steady_newton'
        if steady_start:
            x, t = self.steady(c['U_mean0']), 0.0
            self.steady_forces(x, c['U_mean0'])
        else:
            x, t = self.load_state(init)
        rows, walls, diag_rows, stage_records, states, snap_files = [], [], [], [], [], []
        buf = {'x': [], 't': [], 'chunk': 0}
        def flush():
            if not buf['x']:
                return
            p = self.out / f"snapshots_{buf['chunk']:04d}.npz"
            np.savez(p, t=np.asarray(buf['t']), x=np.asarray(buf['x']))
            snap_files.append({'file': p.name, 'sha256': sha(p), 'count': len(buf['x'])})
            buf.update(x=[], t=[], chunk=buf['chunk'] + 1)
            meta('running')
        def persist(status):
            flush()
            meta(status)
        def meta(status):
            np.savez(self.out / 'linear_checks.npz', t=np.array([z['t'] for z in self.lin_checks]),
                     rel=np.array([z['relative_linear_residual'] for z in self.lin_checks]))
            np.savez(self.out / 'history.npz', rows=np.asarray(rows), wall=np.asarray(walls),
                     columns=np.array(['t', 'CD_reaction', 'CL_reaction', 'CD_surface', 'CL_surface',
                                       'p_front', 'p_back', 'kinetic_energy', 'stage']))
            dump(self.out / 'diagnostics.json', diag_rows)
            dump(self.out / 'stages.json', {'status': status, 'stages': stage_records, 'states': states,
                                            'snapshots': snap_files, 'linear_checks_count': len(self.lin_checks),
                                            'linear_checks_max': max([z['relative_linear_residual'] for z in self.lin_checks], default=None),
                                            'factorizations': self.factorizations, 'timers_seconds': self.timers,
                                            'interpolation': getattr(self, 'interpolation', None),
                                            'layout_sha256': self.layout_sha256, 'ndof': self.X.ndof,
                                            'elapsed_seconds': time.perf_counter() - t_wall0})
        if c['scheme'] not in ('sbdf2', 'oseen_bdf2', 'frozen_oseen'):
            raise ValueError('unknown scheme')
        sbdf = c['scheme']
        try:
            self._stages(c, x, t, sbdf, steady_start, rows, walls, diag_rows, stage_records, states, buf, flush, persist, t_wall0)
        except BaseException as exc:
            if not (isinstance(exc, SystemExit) and exc.code == EXIT_WALL):
                persist(f'aborted: {exc!r}')
            raise
        persist('completed')

    def _stages(self, c, x, t, sbdf, steady_start, rows, walls, diag_rows, stage_records, states, buf, flush, persist, t_wall0):
        for si, st in enumerate(c['stages']):
            dt = float(st['dt'])
            n_steps = int(round(st['duration'] / dt))
            if abs(n_steps * dt - st['duration']) > 1e-9 * max(1.0, st['duration']):
                raise ValueError('stage duration must be a multiple of dt')
            save_every = st.get('save_every_steps')
            diag_every = int(st.get('diagnostics_every_saves', 0))
            rec = {'name': st['name'], 'dt': dt, 't_start': t, 'planned_steps': n_steps,
                   'until_periodic': bool(st.get('until_periodic', False))}
            w0 = time.perf_counter()
            stage_rows = []
            n_saves = 0
            def emit(k, x, dudt, Nx, t):
                nonlocal n_saves
                row = self.observe(x, dudt, Nx, t) + [si]
                if dudt is None:  # stage start: no consistent time derivative for the reaction force
                    row[1] = row[2] = float('nan')
                rows.append(row); stage_rows.append(row); walls.append(time.perf_counter() - w0)
                if save_every and k % save_every == 0:
                    buf['x'].append(x.copy()); buf['t'].append(t)
                    if diag_every and n_saves % diag_every == 0:
                        diag_rows.append(dict(self.diagnostics(x, dudt, t), stage=si))
                    n_saves += 1
                    if len(buf['x']) >= c['snapshot_chunk']:
                        flush()
            # k=0: the stage's initial state (derivative unknown; reaction uses zero).
            x0, N0 = x, self.apply('conv', x)
            # A steady initial state has an exactly zero time derivative.
            emit(0, x0, 0 * x0 if (si == 0 and steady_start) else None, N0, t)
            # Startup: first-order substeps spanning exactly one dt.
            hs = dt / int(c['startup_substeps'])
            xs, Ns, ts = x0, N0, t
            for j in range(int(c['startup_substeps'])):
                tn = ts + hs
                rhs = self.apply('M', xs) / hs + self.forcing(tn)
                if sbdf == 'sbdf2':
                    xn = self.solve(1.0 / hs, rhs - Ns, tn)
                elif sbdf == 'oseen_bdf2':
                    xn = self.solve(1.0 / hs, rhs, tn, w=xs)
                else:
                    self.prepare_frozen(1.0 / hs, x0, force=(j == 0))
                    xn = self.solve(1.0 / hs, rhs - (Ns - self.apply('Cf', xs)), tn, frozen=True)
                dudt = (xn - xs) / hs
                xs, ts, Ns = xn, tn, self.apply('conv', xn)
            x_prev, N_prev, x, Nx, t = x0, N0, xs, Ns, t + dt
            emit(1, x, None, Nx, t)  # startup derivative is first order: no reaction force here
            self.check_next = True
            coef = 1.5 / dt
            k = 1
            periodic = False
            while k < n_steps:
                if (k + 1) % 200 == 0 or k + 1 == n_steps:
                    self.check_next = True
                tn = t + dt
                rhs = self.apply('M', 4 * x - x_prev) / (2 * dt) + self.forcing(tn)
                if sbdf == 'sbdf2':
                    xn = self.solve(coef, rhs - 2 * Nx + N_prev, tn)
                elif sbdf == 'oseen_bdf2':
                    xn = self.solve(coef, rhs, tn, w=2 * x - x_prev)
                else:
                    self.prepare_frozen(coef, x)
                    Rn = Nx - self.apply('Cf', x)
                    Rp = N_prev - self.apply('Cf', x_prev)
                    xn = self.solve(coef, rhs - 2 * Rn + Rp, tn, frozen=True)
                dudt = (3 * xn - 4 * x + x_prev) / (2 * dt)
                x_prev, N_prev = x, Nx
                x, t, Nx = xn, tn, self.apply('conv', xn)
                k += 1
                if not np.isfinite(x).all() or abs(rows[-1][7]) > c['energy_blowup']:
                    rec.update(status='nonfinite_or_energy_blowup', steps=k)
                    stage_records.append(rec); persist('failed'); raise ArithmeticError('instability')
                emit(k, x, dudt, Nx, t)
                if time.perf_counter() - t_wall0 > c['wall_limit_seconds']:
                    rec.update(status='wall_limit', steps=k, t_end=t)
                    stage_records.append(rec)
                    states.append(self.save_state('interrupted', x, t))
                    persist('wall_limit'); sys.exit(EXIT_WALL)
                if k % 200 == 0:
                    if rec['until_periodic']:
                        sr = np.asarray(stage_rows)[1:]
                        recs = cyc.cycles(sr[:, 0], sr[:, 1], sr[:, 2], sr[:, 5] - sr[:, 6])
                        if cyc.periodic(recs, **c['periodicity']):
                            periodic = True
                            break
            sr = np.asarray(stage_rows)[1:]
            recs = cyc.cycles(sr[:, 0], sr[:, 1], sr[:, 2], sr[:, 5] - sr[:, 6])
            rec.update(status='completed', steps=k, t_end=t, periodic_reached=periodic or cyc.periodic(recs, **c['periodicity']),
                       cycles=recs, wall_seconds=time.perf_counter() - w0,
                       seconds_per_step=(time.perf_counter() - w0) / max(k, 1))
            stage_records.append(rec)
            if st.get('save_state'):
                states.append(self.save_state(st['name'], x, t))
            persist('running')
            if rec['until_periodic'] and st.get('require_periodic') and not rec['periodic_reached']:
                raise ArithmeticError(f"stage {st['name']} not periodic by cap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest' or os.environ.get('JAX_ENABLE_X64') != 'true':
        raise RuntimeError('required precision environment missing')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'config.json', cfg)
    try:
        import ngsolve as ng
        ng.SetNumThreads(int(cfg['threads']))
        with ng.TaskManager():
            Stepper(cfg, out, ng).run()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == EXIT_WALL:
            raise
        dump(out / 'FAILED.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                                   'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                   'job_id': os.environ.get('SLURM_JOB_ID')})
        raise


if __name__ == '__main__':
    main()
