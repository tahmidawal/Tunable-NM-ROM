"""G5 reduced models on the common quadrature Q (DESIGN-G5 §5.2–5.5). NumPy only.

Gradient layout everywhere: G[q, component, derivative, function] (physical coordinates).
Field errors use the norm expansion |u-v|^2 = |u|^2 - 2<u,v> + |v|^2 from streamed inner
products, so an error e has absolute precision ~ eps*|u|^2/(2e): about 1e-10 relative at
e = 1e-3, negligible for every floor/error of interest (>= 1e-4).
An arm = lift + velocity basis (POD prefix or bank functions) + p_r supremizers, all
L2_Q-orthonormalized together, and the first p_r pressure functions.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

RANK_CUTOFF = 1e-12   # A1 relative eigenvalue cutoff for the orthonormalization
CHUNK = 8192


class Export:
    def __init__(self, d):
        d = Path(d)
        q = np.load(d / 'quadrature.npz'); b = np.load(d / 'basis.npz')
        self.w = q['weights']; self.vol = float(self.w.sum())
        self.names = [str(n) for n in b['names']]
        self.V, self.G, self.P = b['V'], b['G'], b['P']
        self.probes = b['probes']
        self.bank = {'frozen': (b['frozen_bank_V'], b['frozen_bank_G'])}
        self.out_w = q['outlet_weights']; self.out_V = b['outlet_V']; self.out_bank = {'frozen': b['frozen_bank_outlet_V']}
        # inlet/outlet traces for the reduced energy balance (normals (-1,0) and (1,0))
        self.bnd = {lab: {'w': q[f'{lab}_weights'], 'V': b[f'{lab}_V'], 'G': b[f'{lab}_G'], 'P': b[f'{lab}_P'],
                          'n': np.array([-1.0, 0.0]) if lab == 'inlet' else np.array([1.0, 0.0])}
                    for lab in ('inlet', 'outlet')}
        self.bnd_bank = {'frozen': {lab: (b[f'frozen_bank_{lab}_V'], b[f'frozen_bank_{lab}_G']) for lab in ('inlet', 'outlet')}}
        self.dir = d

    def col(self, name):
        return self.names.index(name)

    def add_bank(self, key, V, G, outlet_V, bnd=None):
        self.bank[key] = (V, G); self.out_bank[key] = outlet_V
        if bnd is not None:
            self.bnd_bank[key] = bnd


def orthonormal_transform(gram):
    """Eigen-truncated transform, then one Cholesky re-orthogonalization pass, verified."""
    lam, U = np.linalg.eigh(gram)
    keep = lam > RANK_CUTOFF * lam.max()
    T = U[:, keep] / np.sqrt(lam[keep])
    G1 = T.T @ gram @ T
    T = T @ np.linalg.inv(np.linalg.cholesky(G1)).T
    defect = float(np.max(np.abs(T.T @ gram @ T - np.eye(T.shape[1]))))
    if defect > 1e-12:
        raise ArithmeticError(f'basis not orthonormal after re-orthogonalization: {defect:.2e}')
    return T, int(keep.sum()), len(lam), defect


def wdot(w, A, B):
    """sum_q w_q A[q,...,i] B[q,...,j] over all non-function axes -> (i, j)."""
    n = A.shape[0]
    return (A.reshape(n, -1, A.shape[-1]) * w[:, None, None]).reshape(-1, A.shape[-1]).T @ B.reshape(-1, B.shape[-1])


class Arm:
    """Reduced spaces and exact operators for one method/rank."""

    def __init__(self, ex: Export, kind, r, bank_key=None, orthonormalize=True):
        self.kind, self.r, self.ex, self.bank_key = kind, r, ex, bank_key
        w = ex.w
        L = ex.col('lift')
        sup = [ex.col(f'sup{i}') for i in range(r)]
        if kind == 'pod':
            raw_V = np.concatenate([ex.V[..., [ex.col(f'pod{i}') for i in range(r)]], ex.V[..., sup]], axis=-1)
            raw_G = np.concatenate([ex.G[..., [ex.col(f'pod{i}') for i in range(r)]], ex.G[..., sup]], axis=-1)
            raw_out = np.concatenate([ex.out_V[..., [ex.col(f'pod{i}') for i in range(r)]], ex.out_V[..., sup]], axis=-1)
            self.raw_cols = [ex.col(f'pod{i}') for i in range(r)] + sup
        else:
            bv, bg = ex.bank[bank_key]
            if bv.shape[-1] != r:
                raise ValueError('bank rank mismatch')
            raw_V = np.concatenate([bv, ex.V[..., sup]], axis=-1)
            raw_G = np.concatenate([bg, ex.G[..., sup]], axis=-1)
            raw_out = np.concatenate([ex.out_bank[bank_key], ex.out_V[..., sup]], axis=-1)
            self.raw_cols = [('bank', i) for i in range(r)] + sup
        gram = wdot(w, raw_V, raw_V)
        self.T, self.retained, self.raw_count, self.orth_defect = orthonormal_transform(gram)
        if not orthonormalize:   # quadrature check (G5-A5 rev.2): operators in the raw basis
            self.T = np.eye(len(gram))
        # Basis-only (no supremizer) transform for the gated projection floor.
        self.T0, self.retained0, _, self.orth_defect0 = orthonormal_transform(gram[:r, :r])
        self.raw_V, self.raw_G = raw_V, raw_G
        if orthonormalize:
            # Verify the EVALUATED basis (Codex ROM review 02): re-orthogonalize against wdot(Vb, Vb).
            self.T, self.orth_defect = self._evaluated_orth(raw_V, self.T)
            self.T0, self.orth_defect0 = self._evaluated_orth(raw_V[..., :r], self.T0)
        self.Vb = raw_V @ self.T; self.Gb = raw_G @ self.T
        self.lV, self.lG = ex.V[..., L], ex.G[..., L]
        self.Pb = ex.P[:, :r]
        self.out_b = raw_out @ self.T; self.out_l = ex.out_V[..., L]
        self.bnd_b = {}
        for lab, bd in ex.bnd.items():
            if kind == 'pod':
                cols = [ex.col(f'pod{i}') for i in range(r)] + sup
                rv, rg = bd['V'][..., cols], bd['G'][..., cols]
            else:
                bbv, bbg = ex.bnd_bank[bank_key][lab]
                rv = np.concatenate([bbv, bd['V'][..., sup]], -1); rg = np.concatenate([bbg, bd['G'][..., sup]], -1)
            self.bnd_b[lab] = {'w': bd['w'], 'n': bd['n'], 'V': rv @ self.T, 'G': rg @ self.T, 'P': bd['P'][:, :r],
                               'lV': bd['V'][..., L], 'lG': bd['G'][..., L]}
        self.chiV = ex.V[..., [ex.col('chi_x'), ex.col('chi_y')]]
        self.chiG = ex.G[..., [ex.col('chi_x'), ex.col('chi_y')]]
        self.build()

    def _evaluated_orth(self, raw_V, T, passes=3):
        w = self.ex.w
        for _ in range(passes):
            Mv = wdot(w, raw_V @ T, raw_V @ T)
            defect = float(np.max(np.abs(Mv - np.eye(Mv.shape[0]))))
            if defect <= 1e-13:
                return T, defect
            T = T @ np.linalg.inv(np.linalg.cholesky(Mv)).T
        Mv = wdot(w, raw_V @ T, raw_V @ T)
        defect = float(np.max(np.abs(Mv - np.eye(Mv.shape[0]))))
        if defect > 1e-12:
            raise ArithmeticError(f'evaluated basis not orthonormal: {defect:.2e}')
        return T, defect

    def build(self):
        w, V, G, lV, lG, P = self.ex.w, self.Vb, self.Gb, self.lV, self.lG, self.Pb
        d = V.shape[-1]
        self.d, self.dp = d, P.shape[-1]          # velocity and pressure dimensions; total d + dp
        div = G[:, 0, 0, :] + G[:, 1, 1, :]; ldiv = lG[:, 0, 0] + lG[:, 1, 1]
        self.M = wdot(w, V, V)
        self.K = wdot(w, G, G)
        self.KL = wdot(w, G, lG[..., None])[:, 0]
        self.B = wdot(w, P, div)                          # B[k, j] = int psi_k div phi_j
        self.BL = wdot(w, P, ldiv[:, None])[:, 0]
        self.Gp = wdot(w, P, P)
        conv_l = np.einsum('qcd,qd->qc', lG, lV)          # (grad lift) lift
        self.c0 = wdot(w, V, conv_l[..., None])[:, 0]
        self.C1 = np.zeros((d, d)); self.C2 = np.zeros((d, d, d))
        # chi (force) functionals use the same structures with chi as the test function.
        cV, cG = self.chiV, self.chiG
        self.f_m = wdot(w, cV, V); self.f_k = wdot(w, cG, G); self.f_kl = wdot(w, cG, lG[..., None])[:, 0]
        self.f_c0 = wdot(w, cV, conv_l[..., None])[:, 0]
        self.f_b = wdot(w, P, cG[:, 0, 0, :] + cG[:, 1, 1, :])   # (k, 2)
        self.f_C1 = np.zeros((2, d)); self.f_C2 = np.zeros((2, d, d))
        for s in range(0, len(w), CHUNK):
            e = slice(s, s + CHUNK); ww = w[e]
            Ve, Ge, lVe, lGe, cVe = V[e], G[e], lV[e], lG[e], cV[e]
            # (grad phi_j) lift + (grad lift) phi_j  -> (q, c, j)
            A1 = np.einsum('qcdj,qd->qcj', Ge, lVe) + np.einsum('qcd,qdj->qcj', lGe, Ve)
            self.C1 += wdot(ww, Ve, A1); self.f_C1 += wdot(ww, cVe, A1)
            A2 = np.einsum('qcdj,qdk->qcjk', Ge, Ve).reshape(len(ww), 2, d * d)
            self.C2 += wdot(ww, Ve, A2).reshape(d, d, d); self.f_C2 += wdot(ww, cVe, A2).reshape(2, d, d)
        self.D2 = (div * w[:, None]).T @ div
        self.D1 = (div * w[:, None]).T @ ldiv; self.D0 = float(w @ (ldiv * ldiv))
        self.L2l = float(np.einsum('q,qc,qc->', w, lV, lV)); self.lb = wdot(w, lV[..., None], V)[0]
        ow = self.ex.out_w
        self.qout_l = float(ow @ self.out_l[:, 0]); self.qout_b = ow @ self.out_b[:, 0, :]

    # ---- residual and Jacobian ------------------------------------------
    def conv(self, a):
        return self.c0 + self.C1 @ a + np.einsum('ijk,j,k->i', self.C2, a, a)

    def dconv(self, a):
        return self.C1 + np.einsum('ijk,k->ij', self.C2, a) + np.einsum('ijk,j->ik', self.C2, a)

    def step(self, nu, coef, rhs_m, a_guess, c_guess, tol=1e-12, maxit=20):
        """Solve coef*M a + nu(K a + KL) + N(a) - B^T c = rhs_m ; B a + BL = 0."""
        d, dp = self.d, self.dp
        a, c = a_guess.copy(), c_guess.copy()
        scale = max(np.linalg.norm(rhs_m), np.linalg.norm(self.c0), 1e-30)
        hist = []
        for it in range(maxit + 1):
            rm = coef * self.M @ a + nu * (self.K @ a + self.KL) + self.conv(a) - self.B.T @ c - rhs_m
            rc = self.B @ a + self.BL
            res = math.sqrt(float(rm @ rm + rc @ rc)) / scale
            hist.append(res)
            if res <= tol:
                return a, c, it, hist
            J = np.zeros((d + dp, d + dp))
            J[:d, :d] = coef * self.M + nu * self.K + self.dconv(a)
            J[:d, d:] = -self.B.T; J[d:, :d] = self.B
            delta = np.linalg.lstsq(J, -np.r_[rm, rc], rcond=None)[0] if not np.isfinite(np.linalg.cond(J)) \
                else np.linalg.solve(J, -np.r_[rm, rc])
            a, c = a + delta[:d], c + delta[d:]
        raise ArithmeticError(f'reduced Newton did not converge: {hist[-3:]}')

    def force(self, nu, a, adot, c):
        """Variational reaction force coefficients (C_D, C_L), U_mean=1, D=0.1."""
        conv = self.f_c0 + self.f_C1 @ a + np.einsum('ijk,j,k->i', self.f_C2, a, a)
        F = -(self.f_m @ adot + nu * (self.f_k @ a + self.f_kl) + conv - self.f_b.T @ c)
        return F / (0.5 * 1.0 ** 2 * 0.1)

    def rollout(self, nu, dt, a0, n_steps, substeps=4):
        """BDF2 from a single projected state; returns a(t), c(t), forces, probes, Newton counts."""
        d, dp = self.d, self.dp
        A = np.zeros((n_steps + 1, d)); C = np.zeros((n_steps + 1, dp)); F = np.full((n_steps + 1, 2), np.nan)
        its = np.zeros(n_steps + 1, int)
        A[0] = a0
        # Consistent initial pressure (G5-A5): [M -B^T; B 0][adot; c] = [-F(a0); 0].
        F0 = nu * (self.K @ a0 + self.KL) + self.conv(a0)
        S = np.zeros((d + dp, d + dp)); S[:d, :d] = self.M; S[:d, d:] = -self.B.T; S[d:, :d] = self.B
        sol = np.linalg.solve(S, np.r_[-F0, np.zeros(dp)])
        c = sol[d:]; C[0] = c
        defect = float(np.linalg.norm(self.B @ a0 + self.BL))
        self.initial = {'continuity_defect': defect,
                        'continuity_scale': float(max(np.linalg.norm(self.BL), 1e-300)),
                        # characteristic scaling: tested divergence vs (U/D) sqrt|Omega|, U = 1, D = 0.1
                        'characteristic_scaled_defect': defect * 0.1 / math.sqrt(self.ex.vol),
                        'initial_acceleration_norm': float(np.linalg.norm(sol[:d]))}
        hs = dt / substeps; a = a0.copy(); it_start = 0
        for _ in range(substeps):
            a_new, c, it, _ = self.step(nu, 1 / hs, self.M @ a / hs, a, c)
            it_start += it; a = a_new
        A[1], C[1], its[1] = a, c, it_start
        for n in range(1, n_steps):
            rhs = self.M @ (4 * A[n] - A[n - 1]) / (2 * dt)
            guess = 2 * A[n] - A[n - 1]
            a, c, it, _ = self.step(nu, 1.5 / dt, rhs, guess, C[n])
            A[n + 1], C[n + 1], its[n + 1] = a, c, it
            adot = (3 * A[n + 1] - 4 * A[n] + A[n - 1]) / (2 * dt)
            F[n + 1] = self.force(nu, A[n + 1], adot, C[n + 1])
            if not np.isfinite(a).all() or np.linalg.norm(a) > 1e6:
                raise ArithmeticError(f'rollout blow-up at step {n + 1}')
        probes = C @ self.ex.probes[:dp]
        return A, C, F, probes, its

    # ---- projections and errors from streamed inner products ------------
    def ref_coeffs(self, proj):
        """<u_ref - lift, phi_j> for the orthonormal basis, per saved time."""
        raw = np.stack([self._column(proj, c) for c in self.raw_cols], axis=1)
        lift_raw = wdot(self.ex.w, self.lV[..., None], self.raw_V)[0]
        return (raw - lift_raw[None, :]) @ self.T, (raw[:, :self.r] - lift_raw[None, :self.r]) @ self.T0

    def _column(self, proj, c):
        """FE columns from 'uv'; bank columns from the identity-tagged 'uv_<bank key>' only."""
        if isinstance(c, tuple):
            key = f'uv_{self.bank_key}'
            if key not in proj:
                raise KeyError(f'no projections for bank {self.bank_key!r}')
            return proj[key][:, c[1]]
        return proj['uv'][:, c]

    def _nonneg(self, e2, scale):
        """Squared errors from norm expansions: roundoff negatives (|.| <= 1e-12 scale) -> 0, else refuse."""
        bad = e2 < -1e-12 * np.abs(scale)
        if np.any(bad):
            raise ArithmeticError(f'negative squared error beyond roundoff: {e2[bad].min():.3e}')
        self.clipped = getattr(self, 'clipped', 0) + int(np.sum(e2 < 0))
        return np.maximum(e2, 0)

    def energy_balance(self, nu, a, adot, c):
        """Reduced gradient-form energy balance (walls/cylinder carry u = 0), normalized residual."""
        w = self.ex.w
        u = self.lV + self.Vb @ a; g = self.lG + self.Gb @ a; ut = self.Vb @ adot; p = self.Pb @ c
        div = g[:, 0, 0] + g[:, 1, 1]
        t = {'kinetic_rate': float(w @ np.einsum('qc,qc->q', u, ut)),
             'dissipation': nu * float(w @ np.einsum('qcd,qcd->q', g, g)),
             'finite_divergence_convection': -0.5 * float(w @ (np.einsum('qc,qc->q', u, u) * div)),
             'finite_divergence_pressure': -float(w @ (p * div))}
        for lab, bd in self.bnd_b.items():
            ub = bd['lV'] + bd['V'] @ a; gb = bd['lG'] + bd['G'] @ a; pb = bd['P'] @ c; n = bd['n']
            un = ub @ n
            t[f'convection_{lab}'] = 0.5 * float(bd['w'] @ (np.einsum('qc,qc->q', ub, ub) * un))
            tn = nu * np.einsum('qcd,d->qc', gb, n) - pb[:, None] * n[None, :]
            t[f'traction_work_{lab}'] = -float(bd['w'] @ np.einsum('qc,qc->q', ub, tn))
        return {'terms': t, 'residual': abs(sum(t.values())) / max(sum(abs(z) for z in t.values()), 1e-12)}

    def floors(self, proj):
        uu = proj['uu']; ul = proj['uv'][:, self.ex.col('lift')]
        fluct2 = uu - 2 * ul + self.L2l
        full, base = self.ref_coeffs(proj)
        # Best-approximation norms with the actual evaluated Gram matrices (not assumed identity).
        M0 = wdot(self.ex.w, self.raw_V[..., :self.r] @ self.T0, self.raw_V[..., :self.r] @ self.T0)
        pf = np.einsum('tj,tj->t', full, np.linalg.solve(self.M, full.T).T)
        pb = np.einsum('tj,tj->t', base, np.linalg.solve(M0, base.T).T)
        e_full = np.sqrt(self._nonneg(fluct2 - pf, uu)) / np.sqrt(uu)
        e_base = np.sqrt(self._nonneg(fluct2 - pb, uu)) / np.sqrt(uu)
        pv = proj['pv'][:, :self.dp]
        cp = np.linalg.solve(self.Gp, pv.T).T
        e_p = np.sqrt(self._nonneg(proj['pp'] - np.einsum('tk,tk->t', cp, pv), proj['pp'])) / math.sqrt(self.ex.vol)
        return {'velocity_basis_only': e_base, 'velocity_with_supremizers': e_full, 'pressure': e_p}

    def errors(self, proj, A, C, idx):
        """Phase-sensitive field errors at saved times; idx maps saved times to rollout steps."""
        uu = proj['uu']; ul = proj['uv'][:, self.ex.col('lift')]
        full, _ = self.ref_coeffs(proj)            # <u_ref - lift, phi>
        a = A[idx]
        # ||lift + Phi a - u_ref||^2 = ||u_ref - lift||^2 - 2 a.<u_ref - lift, phi> + |a|^2 (orthonormal)
        fluct2 = uu - 2 * ul + self.L2l
        e_u = np.sqrt(self._nonneg(fluct2 - 2 * np.einsum('tj,tj->t', a, full) + np.einsum('tj,jk,tk->t', a, self.M, a), uu)) / np.sqrt(uu)
        c = C[idx]; pv = proj['pv'][:, :self.dp]
        e_p = np.sqrt(self._nonneg(proj['pp'] - 2 * np.einsum('tk,tk->t', c, pv) + np.einsum('tk,kl,tl->t', c, self.Gp, c), proj['pp'])) / math.sqrt(self.ex.vol)
        div = np.sqrt(np.maximum(self.D0 + 2 * a @ self.D1 + np.einsum('tj,jk,tk->t', a, self.D2, a), 0)) * 0.1 / math.sqrt(self.ex.vol)
        qin = 1.0 * 0.41  # U_mean*H: the lift carries the inlet trace; fluctuation traces vanish there
        qout = self.qout_l + a @ self.qout_b
        return {'e_u': e_u, 'e_p': e_p, 'divergence': div, 'qout': qout, 'mass': np.abs(qout - qin) / qin}
