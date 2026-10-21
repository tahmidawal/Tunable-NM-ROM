"""Shared paths and read-only loaders for the why-spectral analyses (CPU only).

References:
  * E0 second-order references (central + Crank-Nicolson, Richardson-checked), read through the lane's own loader
    `refstore.py` (sha256-verified against the committed manifest) from the local store of
    worktrees/2026-10-06-jcp-references. 2D cohorts dev6 + val32 (38 cases, the campaign's validation set),
    3D cohorts dev4 + val64 (68 cases). Sealed cohorts are never touched.
  * jcp-regimes second-order references (same E0 procedure, copied byte-identically into that lane) for the steepness
    and viscosity regimes N0, Am (A-), C1, C2 (16 cases each), read from that lane's local store.
Nothing here writes outside this folder.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
W = ROOT / 'worktrees'
HERE = Path(__file__).resolve().parent
OUT = HERE / 'analysis'
FIGS = HERE / 'figs'
REF = W / '2026-10-06-jcp-references/experiments/jcp-references'
REG = W / '2026-10-19-jcp-regimes/experiments/jcp-regimes'
FOM = W / '2026-10-18-jcp-fom2d/experiments/jcp-fom2d'
S3D = W / '2026-10-14-jcp-speed3d/experiments/jcp-speed3d'
P1 = W / '2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling'
IDEA = ROOT / 'reports/2026-10-20-improvement-ideation'
FINAL = ROOT / 'reports/2026-10-17-full-experiment-plan/final'

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))
import refstore as RS  # noqa: E402  (read-only; verifies sha256 of every file it reads)

TIMES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25)


def J(p):
    return json.loads(Path(p).read_text())


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def e0_cases(dim):
    """[(cohort, case)] of the validation cohorts (2D dev6 + val32 = 38, 3D dev4 + val64 = 68)."""
    if dim == 2:
        return [('dev6', c) for c in range(6)] + [('val32', c) for c in range(32)]
    return [('dev4', c) for c in range(4)] + [('val64', c) for c in range(64)]


def load_e0(dim, cohort, case):
    """-> fields, meta. 2D fields (6, 257, 257) incl. zero walls; 3D (6, 63**3) interior nodes, x slowest."""
    f, meta = RS.load(dim, cohort, case, 'ST')
    return np.asarray(f, np.float64), meta


def steepness(meta, dim):
    d = meta['desc']
    if dim == 2:
        return float(d['a']) / float(d['nu'])
    return float(d['A']) / float(d['nu'])


REGIMES = ('N0', 'Am', 'C1', 'C2')


def load_regime(regime, case):
    """jcp-regimes reference (finest level) with identity check; fields (6, 257, 257)."""
    stem = REG / 'store' / '2d' / regime / 'ST' / f'c{case:03d}'
    meta = json.loads(Path(str(stem) + '.json').read_text())
    if (meta['cohort'], meta['case'], meta['kind']) != (regime, case, 'ST'):
        raise RuntimeError('regime reference identity')
    z = np.load(str(stem) + '.npz')
    return np.asarray(z[meta['final'][2]], np.float64), meta


def regime_cases(regime):
    return sorted(int(p.stem[1:]) for p in (REG / 'store' / '2d' / regime / 'ST').glob('c*.json'))
