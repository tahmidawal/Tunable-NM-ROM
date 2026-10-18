"""jcp-dims certificate helpers (DESIGN A2.1, A3.3) shared by the 2D and 3D drivers.

  * square-system LSPG (kappa = 1, M = r): tolerance levels with a residual alternative (A2.1) and the weighted
    residual certificate eta_w = ||D R_n(c_{n+1})|| / scale <= 1e-7 (scale = 0: certified iff the raw weighted residual
    is exactly 0);
  * G-cert by exact inversion of the solver's record tolratio = r_n / (tol + 1e-300): r_rec = tolratio * (tol + 1e-300),
    finite required, |r_cert - r_rec| <= 1e-6 r_rec + 1e-15 (A3.3), used for square LSPG and for GAL.
"""
from __future__ import annotations

import numpy as np

ETA_W_BAR = 1e-7          # 100 x the production tolf 1e-9 (A2.1)
TOL_SQUARE_LSPG = {'prod': (1e-3, 1e-9), 'tight': (1e-7, 1e-13), 'tighter': (1e-8, 1e-15)}


def tol_of(form, level, square=False):
    if form == 'LSPG' and square:
        return TOL_SQUARE_LSPG[level]
    if form == 'LSPG':
        return {'prod': (1e-3, 1e-9), 'tight': (1e-7, 0.), 'tighter': (1e-8, 0.)}[level]
    return {'prod': (0., 1e-10), 'tight': (0., 1e-13), 'tighter': (0., 1e-15)}[level]


def eta_w(raw, scale):
    raw, sc = np.asarray(raw, float), float(scale)
    if sc > 0:
        return raw / sc
    return np.where(raw == 0, 0., np.inf)


def certified_square(raw, scale):
    e = eta_w(raw, scale)
    return np.isfinite(e) & (e <= ETA_W_BAR)


def recovered(tolratio, tol):
    return np.asarray(tolratio, float) * (np.asarray(tol, float) + 1e-300)


def gcert(raw_cert, tolratio, tol):
    """Per-step G-cert (A3.3); tol may be a scalar (square LSPG: tolf * scale) or per-step (GAL: tolf * ||A c_n||)."""
    rc = np.asarray(raw_cert, float)
    rr = recovered(tolratio, tol)
    if not rc.size:
        return False
    return bool(np.all(np.isfinite(rc) & np.isfinite(rr) & (np.abs(rc - rr) <= 1e-6 * np.abs(rr) + 1e-15)))
