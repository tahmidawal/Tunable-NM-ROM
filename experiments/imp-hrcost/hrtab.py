"""imp-hrcost tables (DESIGN 1, 2): every arm is the precontracted Galerkin ODE

    dc/dt = f(c) = -P^T [(Gq c) * (Gs c)] + nu L0 c,

with an arm-specific point set and tables (Gq, Gs, P (m, r); L0 (r, r)), built offline on the device.

  sine-tested (arms R, F3):   A = Qm Rm (thin QR of the mesh-projected A, M = 4r frozen modes),
                              P = Psi Qm Rm^{-T}, Psi = L w psi (qcore.offmesh_data 'point'),
                              L0 = -Rm^{-1} Qm^T Lambda A
  bank-tested (arms F2, F2+F3): Mass = int G^T G, Stiff = int grad G^T grad G (Gauss 384^2, checked against 320^2),
                              P = diag(w) Gq Mass^{-1},  L0 = -Mass^{-1} Stiff

Fitted rules (F3, F2+F3) are row subsets of the Gauss 128^2 parent tables with new weights:
P_fit = P_parent[S] * (w_hat_S / w_parent_S). The duplicated-rule control (C5) repeats every row with half weight.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

import p1core as T2

T2.CKPT = T2.HERE / 'inputs' / 'sep_hfit_dense_mid_N256_dense.pkl'    # lane copy (PROVENANCE.json); read before Bank()
import p1common as P  # noqa: E402
from p1common import Q  # noqa: E402

MASS_RULE, MASS_CHECK, MASS_FALLBACK, MASS_BAR = 'gauss384', 'gauss320', 'gauss448', 1e-10


def sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def rule(name):
    """(X, w) of a classical off-mesh rule (qcore.offmesh_rule; sum w = 1)."""
    return Q.offmesh_rule(name)


def _vg(bank, X, Rp, chunk=16384):
    """Rotated bank values and x+y gradient sum, and separately gx, gy, at X (device arrays (m, Rp))."""
    return bank.mdl.values_grads(np.asarray(X), Rp, chunk=chunk)


def mass_stiff(bank, Rp, rule_name, chunk=16384):
    """int G^T G and int (Gx^T Gx + Gy^T Gy) by an off-mesh rule, accumulated in point chunks."""
    X, w = rule(rule_name)
    Mm = jnp.zeros((Rp, Rp))
    Km = jnp.zeros((Rp, Rp))
    for s in range(0, len(X), chunk):
        g, gx, gy = _vg(bank, X[s:s + chunk], Rp)
        ws = jnp.asarray(w[s:s + chunk])[:, None]
        Mm = Mm + g.T @ (ws * g)
        Km = Km + gx.T @ (ws * gx) + gy.T @ (ws * gy)
    return jax.block_until_ready(.5 * (Mm + Mm.T)), jax.block_until_ready(.5 * (Km + Km.T))


def bank_operators(bank, Rp):
    """(Mass, Stiff, info) with the P3 rule-convergence check (DESIGN 1)."""
    t0 = time.perf_counter()
    Mm, Km = mass_stiff(bank, Rp, MASS_RULE)
    Mc, Kc = mass_stiff(bank, Rp, MASS_CHECK)
    rel = lambda a, b: float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    dM, dK = rel(Mc, Mm), rel(Kc, Km)
    used = MASS_RULE
    if max(dM, dK) > MASS_BAR:
        Mf, Kf = mass_stiff(bank, Rp, MASS_FALLBACK)
        dM, dK = rel(Mm, Mf), rel(Km, Kf)
        Mm, Km, used = Mf, Kf, MASS_FALLBACK
    ev = np.linalg.eigvalsh(np.asarray(Mm))
    info = dict(rule=used, dM=dM, dK=dK, p3_pass=bool(max(dM, dK) <= MASS_BAR), cond_mass=float(ev[-1] / ev[0]),
                mass_min_eig=float(ev[0]), seconds=time.perf_counter() - t0, sha_mass=sha(Mm), sha_stiff=sha(Km))
    return Mm, Km, info


class Base:
    """Bank on the run mesh plus the R-arm (sine-tested) linear operators for one r; shared by every arm."""

    def __init__(self, L, Rp, policy='frozen', edges=(0, 128, 256, 384, 512)):
        self.L, self.Rp, self.policy = int(L), int(Rp), policy
        self.bank = P.Bank(L, list(edges))
        ops = self.bank.ops(policy)
        self.M = 4 * self.Rp
        self.kx, self.ky = ops['kx'][:self.M], ops['ky'][:self.M]
        self.mode_hash = P.list_hash(self.kx, self.ky)
        A = ops['Arot'][:self.M, :self.Rp]
        lam = ops['lam'][:self.M]
        Qm, Rm = jnp.linalg.qr(A, mode='reduced')
        self.A, self.lam, self.Qm, self.Rm = A, lam, Qm, Rm
        self.L0_sine = -solve_triangular(Rm, Qm.T @ (lam[:, None] * A), lower=False)
        self.G = Q.BK.prefix(self.bank.Grot, self.bank.edges, self.Rp)
        self.cold, self.trust = self.bank.cold_trust(self.Rp)
        self._bank_ops = None

    def bank_ops(self):
        if self._bank_ops is None:
            Mm, Km, info = bank_operators(self.bank, self.Rp)
            self._bank_ops = (Mm, Km, -jnp.linalg.solve(Mm, Km), info)
        return self._bank_ops

    # -------------------------------------------------------------- tables ----

    def tables(self, proj, rule_name):
        """Classical-rule tables of an arm. proj: 'sine' (R, F3) or 'bank' (F2, F2+F3).
        Returns (d, aux, meta, (X, w)): d = stepper tables (Gq, Gs, P, L0); aux['T'] = weighted tested rows (m, k) with
        tested vector T^T[(Gq c)(Gs c)] (sine: Psi, k = M, the inherited N(c); bank: diag(w) Gq, k = r, DESIGN A2.1);
        aux['fit'] = unweighted per-point fitting rows for the NNLS families (A2.2)."""
        X, w = rule(rule_name)
        wj = jnp.asarray(w)[:, None]
        t0 = time.perf_counter()
        if proj == 'sine':
            blk = Q.offmesh_data(self.bank.mdl, self.Rp, X, w, self.L, self.kx, self.ky, 'point')
            Pt = solve_triangular(self.Rm, self.Qm.T @ blk['Psi'].T, lower=False)          # (r, m) = Rm^-1 Q^T Psi^T
            d = dict(Gq=blk['Gq'], Gs=blk['Gs'], P=Pt.T, L0=self.L0_sine)
            aux = dict(T=blk['Psi'])
        elif proj == 'bank':
            Mm, _, L0, _ = self.bank_ops()
            g, gx, gy = _vg(self.bank, X, self.Rp)
            Tw = wj * g
            Pm = jnp.linalg.solve(Mm, Tw.T).T                                              # diag(w) G M^-1
            d = dict(Gq=g, Gs=gx + gy, P=Pm, L0=L0)
            aux = dict(T=Tw)
            del gx, gy
        else:
            raise ValueError(proj)
        d, aux = jax.block_until_ready((d, aux))
        meta = dict(proj=proj, rule=rule_name, m=int(len(X)), rule_sha256=sha(np.concatenate([X.ravel(), w])),
                    L=self.L, Rp=self.Rp, M=self.M, mode_hash=self.mode_hash, build_seconds=time.perf_counter() - t0)
        return d, aux, meta, (X, w)

    def fit_rows(self, family, aux, w):
        """Unweighted per-point rows of the NNLS system (A2.2): F3a Q^T(L psi) (m, r); F3b L psi (m, M); F2F3 G (m, r)."""
        wj = jnp.asarray(w)[:, None]
        if family == 'F3a':
            return (aux['T'] @ self.Qm) / wj
        if family == 'F3b':
            return aux['T'] / wj
        if family == 'F2F3':
            return aux['T'] / wj
        raise ValueError(family)


def fitted(parent, parent_w, idx, w_hat, name):
    """Row subset of parent tables with new non-negative weights (F3, F2+F3). parent = (d, aux, meta)."""
    d, aux, meta = parent
    idx = np.asarray(idx, int)
    sc = jnp.asarray(np.asarray(w_hat, float) / np.asarray(parent_w, float)[idx])[:, None]
    out = dict(Gq=d['Gq'][idx], Gs=d['Gs'][idx], P=d['P'][idx] * sc, L0=d['L0'])
    ax = dict(T=aux['T'][idx] * sc)
    m2 = dict(meta, rule=name, m=int(len(idx)), parent_rule=meta['rule'],
              rule_sha256=sha(np.concatenate([idx.astype(float), np.asarray(w_hat, float)])))
    return jax.block_until_ready((out, ax)) + (m2,)


def duplicated(parent):
    """C5: every row twice with half weight (same mathematics, 2x point work)."""
    d, aux, meta = parent
    cat = lambda a, f=1.: jnp.concatenate([f * a, f * a])
    out = dict(Gq=cat(d['Gq']), Gs=cat(d['Gs']), P=cat(d['P'], .5), L0=d['L0'])
    return out, dict(T=cat(aux['T'], .5)), dict(meta, rule=meta['rule'] + '-dup', m=2 * meta['m'])


def nan_injected(parent):
    """C4 / P6: one NaN in P (and in the tested rows)."""
    d, aux, meta = parent
    return (dict(d, P=d['P'].at[0, 0].set(jnp.nan)), dict(T=aux['T'].at[0, 0].set(jnp.nan)),
            dict(meta, rule=meta['rule'] + '-nan'))


def table_fingerprint(d):
    return P.sha(np.concatenate([np.asarray(P._reduce(d[k])) for k in ('Gq', 'Gs', 'P', 'L0')]
                                + [np.asarray(d[k].shape, float) for k in ('Gq', 'Gs', 'P', 'L0')]))


def advection(d, c):
    """v(c) = P^T [(Gq c) * (Gs c)], the advection contribution to dc/dt (Q2-projected indicator, A1.4)."""
    return d['P'].T @ ((d['Gq'] @ c) * (d['Gs'] @ c))


def tested(d, aux, c):
    """The binding tested vector (A2.1): T^T [(Gq c) * (Gs c)] (sine: N(c), M entries; bank: T(c), r entries)."""
    return aux['T'].T @ ((d['Gq'] @ c) * (d['Gs'] @ c))
