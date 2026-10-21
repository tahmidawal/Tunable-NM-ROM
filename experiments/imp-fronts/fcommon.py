"""Shared loaders and numerics for the imp-fronts Phase A diagnostics (CPU, NumPy/SciPy only; read-only inputs).

Inputs (never written):
  * incumbent bank: the vendored NumPy evaluator of jcp-regimes (consolidated worktree) -> dn256b ckpt + rotation_R512;
  * E0 ST references dev6/val32 via jcp-references/refstore.py on the local store copy (manifest-verified);
  * jcp-regimes regime references store/2d/<cohort>/ST/cNNN.npz (finest final level meta['final'][2], as rscore.py) + eligibility from checks/analysis.json;
  * jcp-regimes saved ROM coefficient outputs runs/rom1024/output/qacc_rom1024/R_*.npz.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
CONS = ROOT / 'worktrees/2026-10-20-jcp-consolidated/experiments'
REG = ROOT / 'worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes'      # data (gitignored npz) live here
E0STORE = ROOT / 'worktrees/2026-10-06-jcp-references/experiments/jcp-references/store'
HERE = Path(__file__).resolve().parent
CHECKS = HERE / 'checks'
ROMDIR = REG / 'runs/rom1024/output/qacc_rom1024'

os.environ.setdefault('JCPREF_STORE', str(E0STORE))
os.environ.setdefault('JCPREF_META', str(E0STORE))
sys.path.insert(0, str(CONS / 'jcp-regimes'))
sys.path.insert(0, str(CONS / 'jcp-references'))
import npcheck as NC  # noqa: E402
import refstore as RS  # noqa: E402

IN_RANGE = ('dev6', 'val32')
REGIMES = ('N0', 'N0s', 'Ap', 'Am', 'C1', 'C2', 'B1', 'B2')
NI = 255                                     # interior nodes per axis of the 257^2 lattice
XL = np.arange(257) / 256.


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance():
    return dict(bank_ckpt=str(NC.CKPT), bank_ckpt_sha256=sha(NC.CKPT), rotation=str(NC.ROT), rotation_sha256=sha(NC.ROT),
                e0_store_version=RS.store_version(), regimes_analysis_sha256=sha(REG / 'checks/analysis.json'))


# ---------------------------------------------------------------- bank on the lattice
def G(r):
    """(66049, r) incumbent bank functions on the 257^2 lattice (zero walls)."""
    if r not in NC._G257:
        NC.decode257(np.zeros((1, r)), r)
    return NC._G257[r]


def orth(B, rtol=1e-13):
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    return U[:, s > s[0] * rtol]


# ---------------------------------------------------------------- Laplacian-ordered sines (DST-I, orthonormal)
_k = np.arange(1, NI + 1)
_KX, _KY = np.meshgrid(_k, _k, indexing='ij')
LAP_ORDER = np.argsort((_KX ** 2 + _KY ** 2).ravel(), kind='stable')


def dst_coef(F):
    """F (..., 257, 257) with zero walls -> orthonormal DST-I coefficients (..., 65025) in Laplacian order."""
    F = np.asarray(F)
    C = dstn(F[..., 1:-1, 1:-1], type=1, norm='ortho', axes=(-2, -1))
    return C.reshape(*C.shape[:-2], -1)[..., LAP_ORDER]


def sine_truncate(F, K):
    """Orthogonal lattice projection of zero-wall fields F (n, 257, 257) onto the first K Laplacian sines."""
    C = dstn(np.asarray(F)[:, 1:-1, 1:-1], type=1, norm='ortho', axes=(1, 2)).reshape(len(F), -1)
    keep = np.zeros(NI * NI, bool)
    keep[LAP_ORDER[:K]] = True
    C[:, ~keep] = 0.
    out = np.zeros_like(np.asarray(F, float))
    out[:, 1:-1, 1:-1] = idstn(C.reshape(len(F), NI, NI), type=1, norm='ortho', axes=(1, 2))
    return out


def sine_cols(idx):
    """Lattice columns (66049, len(idx)) of the sines with Laplacian ranks idx (orthonormal on the interior)."""
    idx = np.asarray(idx)
    kx, ky = _KX.ravel()[LAP_ORDER[idx]], _KY.ravel()[LAP_ORDER[idx]]
    c = np.sqrt(2. / 256.)
    S = (c * np.sin(np.pi * XL[:, None] * kx[None]))[:, None, :] * (c * np.sin(np.pi * XL[:, None] * ky[None]))[None]
    return S.reshape(257 * 257, len(idx))


# ---------------------------------------------------------------- references
def e0_cases(cohort):
    return [c for c in range(len(list((E0STORE / f'2d/{cohort}/ST').glob('c*.npz'))))]


def load_ref(cohort, case):
    """-> (fields (6,257,257), meta dict with n0, eps_ref, eligible, desc)."""
    if cohort in IN_RANGE:
        f, meta = RS.load(2, cohort, case, 'ST')
        st = RS.reference_status(2, cohort, case, 'ST')
        return np.asarray(f), dict(n0=float(np.linalg.norm(f[0])), n0_judge=meta['judge']['n0'], eps_ref=st['eps_ref'],
                                   eligible=st['eligible'], desc=meta['desc'])
    z = np.load(REG / f'store/2d/{cohort}/ST/c{case:03d}.npz')
    meta = json.loads((REG / f'store/2d/{cohort}/ST/c{case:03d}.json').read_text())
    pc = _regime_elig().get(f'{cohort}|{case}', {})
    f = np.asarray(z[meta['final'][2]], float)
    return f, dict(n0=float(np.linalg.norm(f[0])), n0_judge=meta['judge']['n0'], eps_ref=meta['judge']['eps_ref'],
                   eligible=bool(pc.get('eligible', False)), desc=meta['desc'])


_EL = None


def _regime_elig():
    global _EL
    if _EL is None:
        _EL = json.loads((REG / 'checks/analysis.json').read_text())['configs']['R384|LSPG_CNR_1']['per_case']
    return _EL


def regime_cases(cohort):
    return sorted(int(p.stem[1:]) for p in (REG / f'store/2d/{cohort}/ST').glob('c*.npz'))


def all_cases(cohorts):
    out = []
    for coh in cohorts:
        cs = e0_cases(coh) if coh in IN_RANGE else regime_cases(coh)
        out += [(coh, c) for c in cs]
    return out


# ---------------------------------------------------------------- metrics
def err(v, u, n0):
    """Project metric: max over t1..t5 of ||v - u|| / n0 (inf if non-finite)."""
    d = np.linalg.norm((np.asarray(v) - np.asarray(u)).reshape(6, -1)[1:], axis=1) / n0
    return float(d.max()) if np.isfinite(d).all() else float('inf')


def floor_Q(u, Q, n0):
    """max over t1..t5 of ||(I - Q Q^T) u(t)|| / n0, Q orthonormal lattice columns."""
    R = np.asarray(u).reshape(6, -1)[1:].T
    res = R - Q @ (Q.T @ R)
    return float(np.max(np.linalg.norm(res, axis=0)) / n0)


def write_json(name, obj):
    CHECKS.mkdir(exist_ok=True)
    p = CHECKS / f'{name}.json'
    p.write_text(json.dumps(obj, indent=1, default=float))
    return p
