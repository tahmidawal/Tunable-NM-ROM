"""imp-fronts reference adapter: the four fresh cohorts (FI, FN0s, FC1, FC2) as refjob cohorts (NumPy only).

Descriptors are read from ../fresh_cohorts.json (written and hashed by phaseb/fcohorts.py, DESIGN B3); its canonical
sha256 is re-verified on load. No other cohort is admissible (DESIGN Amendment 1 B3: the adapter allow-lists exactly the
four fresh cohort names). Interface as jcp-regimes rcohorts.Cohort (.dim, .name, .n, .nu(j), .ic(L, j) interior,
.full_ic(L, j), .desc(j), .sha256). The nodal IC is the single-bump training convention (rcohorts.initial_2d, wall nodes
zero), copied here so the adapter has no dependency on another lane's cohort code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLE = HERE.parent / 'fresh_cohorts.json'
NAMES = ('FI', 'FN0s', 'FC1', 'FC2')
KEYS = ('cx', 'cy', 'w', 'a', 'nu')
TIGHT_ALLOWANCE = {}                 # DESIGN B3: tight-path allowance 0 for every fresh cohort
DEV_SMOKE = {'FC1': [0], 'FC2': [0]}  # Amendment 2 B3: development references (smoke only, never scored with test)


class FreshError(ValueError):
    pass


def canonical_sha(tab):
    root = {k: v for k, v in tab['seed_root'].items() if k != 'numpy'}
    blob = json.dumps(dict(cohorts=tab['cohorts'], seed_root=root), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()


_TAB = None


def table():
    global _TAB
    if _TAB is None:
        tab = json.loads(TABLE.read_text())
        if canonical_sha(tab) != tab.get('sha256'):
            raise FreshError('fresh_cohorts.json content differs from its recorded sha256')
        if set(tab['cohorts']) != set(NAMES):
            raise FreshError(f'unexpected cohorts in fresh_cohorts.json: {sorted(tab["cohorts"])}')
        _TAB = tab
    return _TAB


def admissible_cases(name, split):
    """'test' -> every j >= 4; 'dev-smoke' -> the registered development reference cases only."""
    rows = table()['cohorts'][name]
    if split == 'test':
        return [r['j'] for r in rows if r['split'] == 'test']
    if split == 'dev-smoke':
        return list(DEV_SMOKE.get(name, []))
    raise FreshError(f'split {split} not admissible (references exist only for test and the dev smoke cases)')


def initial_2d(L, row):
    cx, cy, w, a, _ = row
    x = np.arange(L + 1) / L
    X, Y = np.meshgrid(x, x, indexing='ij')
    u = np.array(a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w * w)), float)
    u[[0, -1], :] = 0.
    u[:, [0, -1]] = 0.
    return u


class Cohort:
    def __init__(self, dim, name):
        if dim != 2 or name not in NAMES:
            raise FreshError(f'{dim}D cohort {name} not admissible in imp-fronts (fresh cohorts only)')
        tab = table()
        rows = tab['cohorts'][name]
        assert [r['j'] for r in rows] == list(range(len(rows)))
        self.dim, self.name = dim, name
        self.rows = rows
        self.D = np.array([[r[k] for k in KEYS] for r in rows], float)
        self.n = len(rows)
        self.sha256 = hashlib.sha256(np.ascontiguousarray(self.D).tobytes()).hexdigest()
        self.table_sha256 = tab['sha256']

    def nu(self, j):
        return float(self.D[j][4])

    def full_ic(self, L, j):
        return initial_2d(L, self.D[j])

    def ic(self, L, j):
        return self.full_ic(L, j)[1:-1, 1:-1]

    def desc(self, j):
        d = dict(zip(KEYS, map(float, self.D[j])))
        d['s'] = d['a'] / d['nu']
        d['split'] = self.rows[j]['split']
        d['j'] = int(j)
        return d
