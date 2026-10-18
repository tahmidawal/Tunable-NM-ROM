"""jcp-hr-baselines E0 scoring (DESIGN.md 2; A1.7; A2.11 N-swap) through the byte-identical E0 loader `e0/refstore.py`
(store version and STATUS pinned, as speed3d score3.py). CPU / numpy. JCPREF_STORE must point at the E0 store
(cluster: /cluster/tufts/paralab/tawal01/jcp-ref/store; local: the references worktree's store).

Lattice conventions (refstore): 2D (6, 257, 257) with zero walls; 3D (6, 63^3) interior, flat, x slowest.
Cohort names: 2D 'dev6', 'val32'; 3D 'val64' (= seed 923801). Only these are readable here.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'e0'))
import refstore as RS  # noqa: E402

import hrcore as H  # noqa: E402

STORE_VERSION = '3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0'
STATUS_SHA = 'af7d824e94b42afb83753d6b11a0c4d71407cff0923ebee9c245eb8d51f81005'
ALLOWED = {2: ('dev6', 'val32'), 3: ('val64',)}


def check_store():
    """Pins the exact metadata the loader reads (code audit 1, item 17): JCPREF_META must be unset or point at this
    lane's e0/store; the manifest version and the STATUS.json the loader uses (RS._meta_dir()) are hashed."""
    meta = os.environ.get('JCPREF_META')
    assert meta is None or Path(meta).resolve() == (HERE / 'e0' / 'store').resolve(), f'JCPREF_META={meta} not pinned'
    assert RS._meta_dir().resolve() == (HERE / 'e0' / 'store').resolve()
    assert RS.store_version() == STORE_VERSION, 'E0 store version differs from the pin'
    st = hashlib.sha256((RS._meta_dir() / 'STATUS.json').read_bytes()).hexdigest()
    assert st == STATUS_SHA, 'E0 STATUS.json differs from the pin'
    return STORE_VERSION


def ic_lattice(dim, fam, j):
    """Our canonical IC on the scoring lattice (the t0 field a reference must equal)."""
    m = H.LATTICE[dim]
    if dim == 2:
        return np.asarray(H.E2.initial(m, fam.D[j]), float)
    return np.asarray(H.C3.initial_interior(m + 1, fam.T, j), float)


def kind_allowed(dim, kind, option=None):
    """S is used only for 2D BE at dt = 0.005 (time-matched); anything else raises (A2.11 kind control)."""
    if kind == 'ST':
        return True
    if kind == 'S' and dim == 2 and option is not None and option[0] == 'BE' and abs(option[1] - 0.005) < 1e-15:
        return True
    raise PermissionError(f'S reference not allowed for dim={dim}, option={option}')


class Refs:
    """Cached E0 fields + metadata of one cohort's cases (only the allowed cohorts)."""

    def __init__(self, dim, cohort, fam, cases):
        check_store()
        assert cohort in ALLOWED[dim], cohort
        self.dim, self.cohort = dim, cohort
        self.f, self.meta, self.st = {}, {}, {}
        for j in cases:
            f, meta = RS.load(dim, cohort, j, 'ST')
            ic = ic_lattice(dim, fam, j)
            t0 = np.asarray(f[0])
            assert t0.shape == ic.shape and np.max(np.abs(t0 - ic)) <= 1e-12 * max(1., np.max(np.abs(ic))), \
                (dim, cohort, j, 't0 differs from our IC')
            n0 = float(np.linalg.norm(ic))
            assert abs(n0 - meta['judge']['n0']) <= 1e-12 * n0, (dim, cohort, j, 'n0 mismatch')
            self.f[j], self.meta[j] = np.asarray(f), meta
            self.st[j] = RS.reference_status(dim, cohort, j, 'ST')

    def n0(self, j):
        return float(self.meta[j]['judge']['n0'])

    def score(self, fields, j, ref_case=None, shift=0):
        """e = max_{k>=1} ||F_k - R_k|| / n0 (refstore metric). ref_case / shift implement the N-swap controls:
        shift = 1 compares F_k with R_{k+1} for k = 1..4 (four comparisons, A4.4)."""
        rc = j if ref_case is None else ref_case
        R = self.f[rc]
        F = np.asarray(fields, float)
        n0 = self.n0(rc)
        if F.shape != R.shape or not np.all(np.isfinite(F)) or not (np.isfinite(n0) and n0 > 0):
            raise ValueError(f'invalid field for scoring: shape {F.shape} vs {R.shape}, finite={np.all(np.isfinite(F))}')
        ks = range(1, 6 - shift)
        e_t = [float(np.linalg.norm((F[k] - R[k + shift]).ravel())) / self.n0(rc) for k in ks]
        return dict(e=max(e_t), e_t=e_t, eps_ref=float(self.st[rc]['eps_ref']), eligible=bool(self.st[rc]['eligible']))

    def score_safe(self, fields, j, **kw):
        """score() that records an invalid (non-finite / wrong-shape) field as e = nan instead of raising."""
        try:
            return self.score(fields, j, **kw)
        except ValueError as ex:
            return dict(e=float('nan'), error=str(ex), eps_ref=float(self.st[j]['eps_ref']), eligible=False)

    def score_refstore(self, fields, j):
        """The loader's own scorer (cross-check of self.score)."""
        return RS.score(fields, self.dim, self.cohort, j, 'ST')


def lattice_distance(Fa, Fb, n0):
    """Q1 distance (A1.7): max_{k>=1} ||Fa_k - Fb_k|| / n0 on the scoring lattice (non-finite inputs -> inf)."""
    Fa, Fb = np.asarray(Fa, float), np.asarray(Fb, float)
    if Fa.shape != Fb.shape or not (np.all(np.isfinite(Fa)) and np.all(np.isfinite(Fb))) or not (np.isfinite(n0) and n0 > 0):
        return float('inf')
    return max(float(np.linalg.norm((np.asarray(Fa[k]) - np.asarray(Fb[k])).ravel())) for k in range(1, 6)) / n0
