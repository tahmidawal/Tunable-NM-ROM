"""imp-fronts fresh-cohort cases for the ROM drivers (DESIGN B3, Amendments 1-3). Descriptor handling is pure Python /
NumPy; the IC builder imports the vendored engines lazily.

Table: experiments/imp-fronts/phaseb/fresh_cohorts.json (written by fcohorts.py, not here), schema
    {"cohorts": {"FI": [{"j": 0, "split": "dev", "cx", "cy", "w", "a", "nu"}, ...], "FN0s": [...], "FC1": [...],
                 "FC2": [...]},
     "sha256": <hex>, ...}
sha256 definition assumed here (TABLE_SHA): sha256 of json.dumps(table["cohorts"], sort_keys=True, separators=(',', ':'))
(UTF-8). A table whose sha256 does not verify is refused. Splits (Amendment 1 B3): j = 0, 1 dev; j = 2, 3 val; j >= 4 test;
every row's split must equal the split its j implies.

Case ids are '<cohort>|<j>' (r2acc's 'coh|c' form). The IC is built exactly as r2acc / p1common build dev6 / val32 ICs:
engines.initial(L, (cx, cy, w, a, nu)) (nodal (L+1)^2 Gaussian, wall nodes zero), nu = descriptor[4], and the scoring
normaliser n0 = ||u0[::L/256, ::L/256]||. Development cohorts dev6 / val32 are also admitted (descriptors from the
vendored qcore.cohort, as r2acc). Test-split cases are admitted only with test_once=True (the final scoring jobs).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
TABLE = LANE / 'phaseb' / 'fresh_cohorts.json'
FRESH = ('FI', 'FN0s', 'FC1', 'FC2')
DEVCOH = ('dev6', 'val32')
KEYS = ('cx', 'cy', 'w', 'a', 'nu')
# Amendment 2 B2: tight/tighter anchors for the fresh populations (frozen)
ANCHORS = ('FI|4', 'FI|5', 'FC1|4', 'FC1|5')


class CaseError(ValueError):
    pass


def split_of(j):
    return 'dev' if j < 2 else 'val' if j < 4 else 'test'


def table_sha(cohorts):
    return hashlib.sha256(json.dumps(cohorts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_table(path=TABLE):
    path = Path(path)
    if not path.exists():
        raise CaseError(f'fresh-cohort table {path} does not exist (fcohorts.py writes it)')
    t = json.loads(path.read_text())
    coh = t.get('cohorts')
    if not isinstance(coh, dict) or set(coh) - set(FRESH):
        raise CaseError(f'cohorts must be a subset of {FRESH}: {sorted(coh or {})}')
    if t.get('sha256') != table_sha(coh):
        raise CaseError('fresh-cohort table sha256 does not verify (definition: fcases.table_sha)')
    for name, rows in coh.items():
        js = [r['j'] for r in rows]
        if js != list(range(len(rows))):
            raise CaseError(f'{name}: rows must be j = 0..n-1 in order')
        for r in rows:
            if r.get('split') != split_of(r['j']):
                raise CaseError(f'{name}|{r["j"]}: split {r.get("split")} != {split_of(r["j"])}')
            d = [r[k] for k in KEYS]
            if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in d) or r['nu'] <= 0 or r['w'] <= 0:
                raise CaseError(f'{name}|{r["j"]}: bad descriptor {d}')
    return t, path


def descriptor(table, cid):
    coh, j = cid.split('|')
    rows = table['cohorts'].get(coh)
    if rows is None or not (0 <= int(j) < len(rows)):
        raise CaseError(f'{cid}: not in the fresh-cohort table')
    r = rows[int(j)]
    return np.array([float(r[k]) for k in KEYS]), r['split']


def case_list(ids, table_path=TABLE, test_once=False):
    """[(cid, cohort, j, descriptor, split)] for explicit case ids ('FC1|0', 'dev6|3', ...), in the given order."""
    out, table = [], None
    if len(set(ids)) != len(ids):
        raise CaseError('duplicate case ids')
    for cid in ids:
        coh, j = cid.split('|')
        j = int(j)
        if coh in DEVCOH:
            import p1core as T2                      # vendored descriptors, as r2acc (lazy: needs the JAX closure)
            ph = np.asarray(T2.load_qcore().cohort(coh), float)
            out.append((cid, coh, j, ph[j], 'dev'))
            continue
        if coh not in FRESH:
            raise CaseError(f'{cid}: cohort {coh} not admissible (allow-list {FRESH + DEVCOH})')
        if table is None:
            table, _ = load_table(table_path)
        d, sp = descriptor(table, cid)
        if sp == 'test' and not test_once:
            raise CaseError(f'{cid} is test-once: admitted only by a final scoring config (test_once=true)')
        out.append((cid, coh, j, d, sp))
    return out


def initial(L, desc):
    """Nodal IC exactly as r2acc: Q.e.initial(L, ph) with ph = (cx, cy, w, a, nu); returns (u0 numpy, nu, n0)."""
    import engines as e
    u0n = np.asarray(e.initial(int(L), desc), float)
    s = int(L) // 256
    return u0n, float(desc[4]), float(np.linalg.norm(u0n[::s, ::s]))


def anchors_in(ids, explicit=None):
    """Tight / tighter anchor cases inside this job: the frozen fresh anchors, or an explicit (smoke) list."""
    want = ANCHORS if explicit is None else tuple(explicit)
    return [c for c in ids if c in want]


def write_fixture(path, seed_root=0x46495854):
    """Small fixture table for tests (seed root 'FIXT', never the lane's 'IMPF' root): 2 rows per cohort, with the B3
    families' ranges (rcohorts formulas)."""
    coh = {}
    for g, name in enumerate(FRESH):
        rows = []
        for j in range(6 if name in ('FI', 'FC1') else 2):
            r = np.random.default_rng([seed_root, g, j])
            cx, cy, w = r.uniform(.15, .85), r.uniform(.15, .85), r.uniform(.05, .20)
            if name == 'FI':
                nu = float(np.exp(r.uniform(np.log(.01), np.log(.1))))
                a = r.uniform(.5, 2.)
            elif name == 'FN0s':
                nu = float(np.exp(r.uniform(np.log(.01), np.log(.0126))))
                a = r.uniform(1.5, 2.)
            elif name == 'FC1':
                nu = 10. ** r.uniform(-2.5, -2.15)
                a = r.uniform(250. * nu, min(2., 500. * nu))
            else:
                nu = 10. ** r.uniform(-2.8, -2.45)
                a = r.uniform(500. * nu, min(2., 1000. * nu))
            rows.append(dict(j=j, split=split_of(j), cx=float(cx), cy=float(cy), w=float(w), a=float(a), nu=float(nu)))
        coh[name] = rows
    t = dict(cohorts=coh, sha256=table_sha(coh), fixture=True, seed_root=seed_root,
             note='TEST FIXTURE ONLY (seed root FIXT); never scored; the real table is phaseb/fresh_cohorts.json')
    Path(path).write_text(json.dumps(t, indent=1) + '\n')
    return t
