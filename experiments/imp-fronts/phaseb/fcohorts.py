"""imp-fronts Phase B fresh cohorts FI, FN0s, FC1, FC2 (DESIGN B3 + Amendment 1 B3 split + Amendment 2 B3 counts) and the
C-B5 freshness check (Amendment 1 B2). CPU only (NumPy; the training draws are taken from the vendored
burgers2d_film.sample_params, which imports JAX: run under JAX_PLATFORMS=cpu through the campaign jaxslot).

    python phaseb/fcohorts.py            -> writes phaseb/fresh_cohorts.json and phaseb/freshness.json

Rows: row j of cohort g is drawn from numpy.random.default_rng([0x494D5046, g, j]) (seed root "IMPF"), g = 0 FI,
1 FN0s, 2 FC1, 3 FC2; draw order within a row c_x, c_y, w, nu-variable, a-variable (as jcp-regimes rcohorts).
Families (B3 table): FI = the in-range box (rcohorts N0), FN0s = the stiff corner (rcohorts N0s), FC1 = s in [250, 500]
drawn independently (not paired with an A- row), FC2 = s in [500, 1000] (rcohorts C2).
Split (Amendment 1 B3): j = 0, 1 dev; j = 2, 3 val (both reference-free; FC1/FC2 dev j = 0 get smoke references,
Amendment 2 B3); j >= 4 test-once. Draws FI 12, FN0s 12, FC1 12, FC2 8 -> test 8, 8, 8, 4 (28 references).

Freshness (C-B5, Amendment 1): canonical descriptor (cx, cy, w, a, nu) match at relative tolerance 1e-9 against the
explicit allow-list only:
  * E0 dev6 / val32 descriptors (fcommon.load_ref meta 'desc'; no other E0 store directory is traversed),
  * jcp-regimes cohort descriptors (rcohorts.descriptors_2d; single-bump regimes; the two-bump B1/B2 rows have a
    different layout and are recorded as not comparable),
  * the incumbent training draws (burgers2d_film.sample_params seed 0 x 576 and seed 1000 x 4032, the jcp-robust
    Stage A / Stage B EXTRA_SEED=1000 EXTRA_TRAJ=4032 streams), identical for arm (a),
  * the arm-(b) coverage draws (same streams; log nu ~ U(log max(0.0016, a/1000), log 0.1), DESIGN B1(b)),
plus within-fresh duplicates across all rows and splits; the injected val32|0 descriptor must be flagged (must-fail).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
WT = LANE.parents[1]
REG = WT / 'experiments' / 'jcp-regimes'          # the lane worktree's own (consolidated) copy, read-only import
OUT = HERE / 'fresh_cohorts.json'
FRESH_REPORT = HERE / 'freshness.json'

TAG = 0x494D5046                                  # "IMPF"
COHORTS = ('FI', 'FN0s', 'FC1', 'FC2')
G_INDEX = {'FI': 0, 'FN0s': 1, 'FC1': 2, 'FC2': 3}
DRAWS = {'FI': 12, 'FN0s': 12, 'FC1': 12, 'FC2': 8}
RTOL = 1e-9
KEYS = ('cx', 'cy', 'w', 'a', 'nu')
# B3 table boxes (inclusive; tiny tolerance only on derived s bounds and the log10 nu edges, as rcohorts)
BOX = {'FI': dict(nu=(.01, .1), a=(.5, 2.), s=(5., 200.)),
       'FN0s': dict(nu=(.01, .0126), a=(1.5, 2.), s=(1.5 / .0126, 200.)),
       'FC1': dict(lg=(-2.5, -2.15), a=(.5, 2.), s=(250., 500.)),
       'FC2': dict(lg=(-2.8, -2.45), a=(.5, 2.), s=(500., 1000.))}
REGIME_OF = {'FI': 'N0', 'FN0s': 'N0s', 'FC1': 'C1', 'FC2': 'C2'}   # rcohorts box used as the second bound check


class FreshError(ValueError):
    pass


def split_of(j):
    return 'dev' if j < 2 else ('val' if j < 4 else 'test')


def draw_row(name, j):
    r = np.random.default_rng([TAG, G_INDEX[name], int(j)])
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
    elif name == 'FC2':
        nu = 10. ** r.uniform(-2.8, -2.45)
        a = r.uniform(500. * nu, min(2., 1000. * nu))
    else:
        raise FreshError(name)
    return np.array([cx, cy, w, a, nu], float)


def check_bounds(name, row):
    """B3 table box (and the matching rcohorts regime box when rcohorts is importable). Raises FreshError."""
    row = np.asarray(row, float)
    if row.shape != (5,) or not np.isfinite(row).all():
        raise FreshError(f'{name}: malformed row {row}')
    cx, cy, w, a, nu = row
    b = BOX[name]
    s = a / nu
    tol = 1e-9
    ok = .15 <= cx <= .85 and .15 <= cy <= .85 and .05 <= w <= .20 and b['a'][0] <= a <= b['a'][1] and nu > 0
    if 'nu' in b:
        ok = ok and b['nu'][0] <= nu <= b['nu'][1]
    if 'lg' in b:
        lg = np.log10(nu)
        ok = ok and b['lg'][0] - 1e-12 <= lg <= b['lg'][1] + 1e-12
    ok = ok and b['s'][0] * (1 - tol) <= s <= b['s'][1] * (1 + tol)
    if not ok:
        raise FreshError(f'{name} row out of bounds: {row.tolist()} (s = {s})')
    RG = _rg()
    try:
        RG.check_bounds_2d(REGIME_OF[name], row)
    except RG.CohortError as e:
        raise FreshError(f'{name}: rcohorts {REGIME_OF[name]} box: {e}')


def rows_of(name, n=None):
    n = DRAWS[name] if n is None else n
    out = np.array([draw_row(name, j) for j in range(n)])
    for r_ in out:
        check_bounds(name, r_)
    return out


def table():
    """{'cohorts': {name: [ {j, split, cx, cy, w, a, nu, s}, ... ]}, 'seed_root': ...} (no hash)."""
    co = {}
    for name in COHORTS:
        co[name] = [dict(j=j, split=split_of(j), **{k: float(v) for k, v in zip(KEYS, r_)}, s=float(r_[3] / r_[4]))
                    for j, r_ in enumerate(rows_of(name))]
    root = dict(generator='numpy.random.default_rng([tag, g, j])', tag=TAG, tag_hex=hex(TAG), tag_ascii='IMPF',
                g=G_INDEX, draws=DRAWS, numpy=np.__version__)
    return dict(cohorts=co, seed_root=root)


def canonical_sha(tab):
    """sha256 of the canonical cohort content: cohorts + seed_root (minus the numpy version), sorted keys, compact,
    floats as Python repr (round-trip exact)."""
    root = {k: v for k, v in tab['seed_root'].items() if k != 'numpy'}
    blob = json.dumps(dict(cohorts=tab['cohorts'], seed_root=root), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()


def load(path=OUT, verify=True):
    tab = json.loads(Path(path).read_text())
    if verify:
        if canonical_sha(tab) != tab['sha256']:
            raise FreshError('fresh_cohorts.json content differs from its sha256')
        regen = table()
        if regen['cohorts'] != tab['cohorts']:
            raise FreshError('fresh_cohorts.json differs from the generator')
    return tab


def rows_array(tab, name):
    return np.array([[r_[k] for k in KEYS] for r_ in tab['cohorts'][name]], float)


# ------------------------------------------------------------------------------------------------- allow-list ---

def _rg():
    if str(REG) not in sys.path:
        sys.path.insert(0, str(REG))
    import rcohorts as RG
    return RG


def e0_rows():
    """E0 dev6 / val32 canonical descriptors via fcommon.load_ref meta 'desc' (only those two store directories)."""
    if str(LANE) not in sys.path:
        sys.path.insert(0, str(LANE))
    import fcommon as F
    out = {}
    for coh in F.IN_RANGE:
        for c in F.e0_cases(coh):
            out[f'{coh}|{c}'] = np.array([F.load_ref(coh, c)[1]['desc'][k] for k in KEYS], float)
    return out


def regime_rows():
    RG = _rg()
    out, skipped = {}, {}
    for g in RG.REGIMES_2D:
        D = RG.descriptors_2d(g)
        if g in RG.TWO_BUMP:
            skipped[g] = len(D)
            continue
        for j, r_ in enumerate(D):
            out[f'{g}|{j}'] = np.asarray(r_, float)
    return out, skipped


def _bf():
    p = LANE / 'deps'
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import burgers2d_film as bf
    return bf


TRAIN_STREAMS = ((0, 576), (1000, 4032))


def training_rows():
    """Incumbent (and arm (a)) training draws: burgers2d_film.sample_params(seed, m), (cx, cy, w, a, nu) columns."""
    bf = _bf()
    out = []
    for seed, m in TRAIN_STREAMS:
        cx, cy, w, a, nu, _z = bf.sample_params(seed=seed, m=m)
        out.append(np.stack([cx, cy, w, a, nu], 1))
    T = np.concatenate(out)
    ref = _rg().training_rows_2d()                    # independent re-implementation (rcohorts) must agree exactly
    if not np.array_equal(T, ref):
        raise FreshError('burgers2d_film.sample_params differs from rcohorts.training_rows_2d')
    return T


def coverage_rows():
    """Arm (b) draws (DESIGN B1(b)): same streams and order; log nu ~ U(log max(0.0016, a/1000), log 0.1)."""
    out = []
    for seed, m in TRAIN_STREAMS:
        r = np.random.default_rng(seed)
        cx, cy, w, a = r.uniform(.15, .85, m), r.uniform(.15, .85, m), r.uniform(.05, .20, m), r.uniform(.5, 2., m)
        lo = np.log(np.maximum(.0016, a / 1000.))
        nu = np.exp(r.uniform(lo, np.log(.1), m))
        # identity: the same uniform variates as the incumbent's log nu draw, affinely re-mapped
        r2 = np.random.default_rng(seed)
        for _ in range(4):
            r2.uniform(size=m)
        u = r2.uniform(size=m)
        nu2 = np.exp(lo + (np.log(.1) - lo) * u)
        if np.max(np.abs(nu2 / nu - 1)) > 1e-13:
            raise FreshError('coverage stream is not the incumbent stream re-mapped')
        out.append(np.stack([cx, cy, w, a, nu], 1))
    return np.concatenate(out)


def matches(row, others, rtol=RTOL):
    """Indices of rows of `others` equal to `row` componentwise within relative tolerance rtol."""
    o = np.asarray(others, float)
    if o.size == 0:
        return np.zeros(0, int)
    scale = np.maximum(np.abs(o), np.abs(np.asarray(row))[None, :])
    hit = np.all(np.abs(o - np.asarray(row)[None, :]) <= rtol * scale, axis=1)
    return np.flatnonzero(hit)


def freshness(tab, allow, inject=None, rtol=RTOL):
    """allow: {table_name: (labels list, rows array)}. inject: optional (label, row) added to the fresh rows.
    Returns dict(pass, hits, within, counts)."""
    fresh = [(f'{n}|{r_["j"]}|{r_["split"]}', np.array([r_[k] for k in KEYS], float))
             for n in COHORTS for r_ in tab['cohorts'][n]]
    if inject is not None:
        fresh.append(inject)
    hits = []
    for lab, row in fresh:
        for tname, (labels, rows) in allow.items():
            for i in matches(row, rows, rtol):
                hits.append(dict(fresh=lab, table=tname, other=labels[i]))
    within = []
    F_ = np.array([r_ for _, r_ in fresh])
    for i, (lab, row) in enumerate(fresh):
        for k in matches(row, F_, rtol):
            if k > i:
                within.append(dict(a=lab, b=fresh[k][0]))
    return dict(passed=not hits and not within, hits=hits, within_fresh_duplicates=within,
                n_fresh=len(fresh), counts={t: len(v[1]) for t, v in allow.items()})


def allow_list():
    e0 = e0_rows()
    reg, skipped = regime_rows()
    T = training_rows()
    C = coverage_rows()
    lab_t = [f'train_s{s}|{i}' for s, m in TRAIN_STREAMS for i in range(m)]
    lab_c = [f'cover_s{s}|{i}' for s, m in TRAIN_STREAMS for i in range(m)]
    allow = {'E0_dev6_val32': (list(e0), np.array(list(e0.values()))),
             'jcp_regimes': (list(reg), np.array(list(reg.values()))),
             'train_incumbent_and_arm_a': (lab_t, T),
             'train_arm_b_coverage': (lab_c, C)}
    return allow, skipped, e0


def seed_root_check():
    """The fresh streams' seed root differs from every earlier public cohort root. NumPy's SeedSequence ignores trailing
    zero words ([t, g, 0] == [t, g]; [t, 0, 0] == t), so the effective forms are also compared."""
    RG = _rg()
    sys.path.insert(0, str(REG))
    import cohorts as CH
    public_one_word = set(RG.E0_HISTORICAL) | set(RG.E0_PUBLIC) | set(RG.TRAIN_SEEDS) | set(CH.HISTORICAL_SEEDS) \
        | {s for d in RG.SEEDS.values() for s in d.values()}
    effective = {'FI j=0': [TAG], 'FN0s/FC1/FC2 j=0': [TAG, 'g'], 'general': [TAG, 'g', 'j']}
    rgen = np.random.default_rng([TAG, 1, 0]).random() == np.random.default_rng([TAG, 1]).random()
    return dict(tag=TAG, differs_from_regimes_tag=TAG != RG.TAG, tag_not_a_public_seed=TAG not in public_one_word,
                regimes_tag=RG.TAG, public_one_word_seeds=sorted(public_one_word), effective_forms=effective,
                numpy_trailing_zero_equivalence=bool(rgen),
                caveat=('E0 sealed / OOD 2D rows use default_rng([secret_seed, j]); with trailing-zero equivalence '
                        'the fresh j=0 rows of g=1,2,3 coincide with such a stream only if a secret seed equals the tag '
                        'and j equals g. The secret seeds are never opened by this lane; this residual possibility is '
                        'recorded, not checked.'),
                passed=bool(TAG != RG.TAG and TAG not in public_one_word))


def build():
    tab = table()
    tab['sha256'] = canonical_sha(tab)
    allow, skipped, e0 = allow_list()
    rep = dict(fresh_sha256=tab['sha256'], rtol=RTOL, seed_root=seed_root_check(),
               main=freshness(tab, allow), two_bump_not_comparable=skipped)
    inj = freshness(tab, allow, inject=('INJECT|val32|0', e0['val32|0']))
    rep['must_fail_injection'] = dict(label='val32|0', flagged=not inj['passed'], hits=inj['hits'],
                                      within_fresh_duplicates=inj['within_fresh_duplicates'])
    rep['C_B5_pass'] = bool(rep['main']['passed'] and rep['seed_root']['passed'] and rep['must_fail_injection']['flagged'])
    s = {n: [r_['s'] for r_ in tab['cohorts'][n]] for n in COHORTS}
    nu = {n: [r_['nu'] for r_ in tab['cohorts'][n]] for n in COHORTS}
    rep['ranges'] = {n: dict(s=[min(s[n]), max(s[n])], nu=[min(nu[n]), max(nu[n])],
                             test=sum(r_['split'] == 'test' for r_ in tab['cohorts'][n])) for n in COHORTS}
    return tab, rep


def main():
    tab, rep = build()
    if not rep['C_B5_pass']:
        print(json.dumps(rep, indent=1, default=str))
        raise SystemExit('C-B5 freshness FAILED: fresh_cohorts.json not written')
    if OUT.exists():
        old = json.loads(OUT.read_text())
        if old.get('sha256') != tab['sha256']:
            raise SystemExit('fresh_cohorts.json exists with different content: refusing to overwrite')
    OUT.write_text(json.dumps(tab, indent=1) + '\n')
    FRESH_REPORT.write_text(json.dumps(rep, indent=1, default=str) + '\n')
    print('fresh sha256', tab['sha256'])
    for n, v in rep['ranges'].items():
        print(f"{n:5s} test={v['test']} s=[{v['s'][0]:.1f}, {v['s'][1]:.1f}] nu=[{v['nu'][0]:.4g}, {v['nu'][1]:.4g}]")
    print('freshness main pass', rep['main']['passed'], 'counts', rep['main']['counts'])
    print('must-fail injection flagged', rep['must_fail_injection']['flagged'], rep['must_fail_injection']['hits'])
    print('seed root pass', rep['seed_root']['passed'])


if __name__ == '__main__':
    main()
