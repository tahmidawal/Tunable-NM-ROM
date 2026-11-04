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
        self.dir = d

    def col(self, name):
        return self.names.index(name)

    def add_bank(self, key, V, G, outlet_V):
        self.bank[key] = (V, G); self.out_bank[key] = outlet_V


def wdot(w, A, B):
    """sum_q w_q A[q,...,i] B[q,...,j] over all non-function axes -> (i, j)."""
    n = A.shape[0]
    return (A.reshape(n, -1, A.shape[-1]) * w[:, None, None]).reshape(-1, A.shape[-1]).T @ B.reshape(-1, B.shape[-1])


class Arm:
    """Reduced spaces and exact operators for one method/rank."""

    def __init__(self, ex: Export, kind, r, bank_key=None):
        self.kind, self.r, self.ex = kind, r, ex
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
        lam, U = np.linalg.eigh(gram)
        keep = lam > RANK_CUTOFF * lam.max()
        self.raw_count, self.retained = len(lam), int(keep.sum())
        self.T = U[:, keep] / np.sqrt(lam[keep])          # raw -> orthonormal
        # Basis-only (no supremizer) transform for the gated projection floor.
        g0 = gram[:r, :r]; l0, U0 = np.linalg.eigh(g0); k0 = l0 > RANK_CUTOFF * l0.max()
        self.T0 = U0[:, k0] / np.sqrt(l0[k0])
        self.raw_V, self.raw_G = raw_V, raw_G
        self.Vb = raw_V @ self.T; self.Gb = raw_G @ self.T
        self.lV, self.lG = ex.V[..., L], ex.G[..., L]
        self.Pb = ex.P[:, :r]
        self.out_b = raw_out @ self.T; self.out_l = ex.out_V[..., L]
        self.chiV = ex.V[..., [ex.col('chi_x'), ex.col('chi_y')]]
        self.chiG = ex.G[..., [ex.col('chi_x'), ex.col('chi_y')]]
        self.build()

    def build(self):
        w, V, G, lV, lG, P = self.ex.w, self.Vb, self.Gb, self.lV, self.lG, self.Pb
        d = V.shape[-1]
        self.d, self.dp = d, P.shape[-1]
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
        c = np.linalg.lstsq(self.B.T, nu * (self.K @ a0 + self.KL) + self.conv(a0), rcond=None)[0]  # diagnostic only
        C[0] = c
        hs = dt / substeps; a = a0.copy()
        for _ in range(substeps):
            a_new, c, it, _ = self.step(nu, 1 / hs, self.M @ a / hs, a, c)
            adot = (a_new - a) / hs; a = a_new
        A[1], C[1], its[1] = a, c, it
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
        uv = proj['uv']
        raw = np.stack([uv[:, self._colindex(c)] for c in self.raw_cols], axis=1)
        lift_raw = wdot(self.ex.w, self.lV[..., None], self.raw_V)[0]
        return (raw - lift_raw[None, :]) @ self.T, (raw[:, :self.r] - lift_raw[None, :self.r]) @ self.T0

    def _colindex(self, c):
        if isinstance(c, tuple):
            return len(self.ex.names) + c[1]
        return c

    def floors(self, proj):
        uu = proj['uu']; ul = proj['uv'][:, self.ex.col('lift')]
        fluct2 = uu - 2 * ul + self.L2l
        full, base = self.ref_coeffs(proj)
        e_full = np.sqrt(np.maximum(fluct2 - np.sum(full ** 2, 1), 0)) / np.sqrt(uu)
        e_base = np.sqrt(np.maximum(fluct2 - np.sum(base ** 2, 1), 0)) / np.sqrt(uu)
        pv = proj['pv'][:, :self.dp]
        cp = np.linalg.solve(self.Gp, pv.T).T
        e_p = np.sqrt(np.maximum(proj['pp'] - np.einsum('tk,tk->t', cp, pv), 0)) / math.sqrt(self.ex.vol)
        return {'velocity_basis_only': e_base, 'velocity_with_supremizers': e_full, 'pressure': e_p}

    def errors(self, proj, A, C, idx):
        """Phase-sensitive field errors at saved times; idx maps saved times to rollout steps."""
        uu = proj['uu']; ul = proj['uv'][:, self.ex.col('lift')]
        full, _ = self.ref_coeffs(proj)            # <u_ref - lift, phi>
        a = A[idx]
        # ||lift + Phi a - u_ref||^2 = ||u_ref - lift||^2 - 2 a.<u_ref - lift, phi> + |a|^2 (orthonormal)
        fluct2 = uu - 2 * ul + self.L2l
        e_u = np.sqrt(np.maximum(fluct2 - 2 * np.einsum('tj,tj->t', a, full) + np.einsum('tj,tj->t', a, a), 0)) / np.sqrt(uu)
        c = C[idx]; pv = proj['pv'][:, :self.dp]
        e_p = np.sqrt(np.maximum(proj['pp'] - 2 * np.einsum('tk,tk->t', c, pv) + np.einsum('tk,kl,tl->t', c, self.Gp, c), 0)) / math.sqrt(self.ex.vol)
        div = np.sqrt(np.maximum(self.D0 + 2 * a @ self.D1 + np.einsum('tj,jk,tk->t', a, self.D2, a), 0)) * 0.1 / math.sqrt(self.ex.vol)
        qin = 1.0 * 0.41  # U_mean*H: the lift carries the inlet trace; fluctuation traces vanish there
        qout = self.qout_l + a @ self.qout_b
        return {'e_u': e_u, 'e_p': e_p, 'divergence': div, 'qout': qout, 'mass': np.abs(qout - qin) / qin}
