"""jcp-references LOADER API — score any lane's fields against the second-order references (DESIGN §11, A1.12, A2.5,
A2.10, A3.3, A3.7, A3.8).

    import sys; sys.path.insert(0, '<worktree>/experiments/jcp-references')
    import refstore as RS
    fields, meta = RS.load(2, 'val32', 7)                  # R_ST finest level, (6, 257, 257)
    s = RS.score(my_fields, 2, 'val32', 7)                 # {'e', 'e_t', 'eps_ref', 'eligible', ...}
    RS.resolved(s_arm_a, s_arm_b)                          # True iff |e_a - e_b| > 2 eps_ref (both eligible)

Conventions (identical to the old first-order references, so a lane switches by changing the loader only):
  2D  fields (6, 257, 257) on x = i/256 incl. the zero walls (old `f257`), t = 0, .05, ..., .25
  3D  fields (6, 63**3) on the interior nodes x = k/64, flat, x slowest (old `ref_<seed>.npz['c<j>']`)
  metric  e(t_k) = ||f(t_k) - R(t_k)|| / ||R(t_0)||, k >= 1; e = max_k e(t_k) (fixed-t0 normalisation)
Cohorts: 2D dev6, val32, test64; 3D dev4, val64, test32 (= seeds 923651, 923801, 923901). Kinds: 'ST' (central +
Crank-Nicolson, the primary reference) and 'S' (central + backward Euler at the ROM's dt = 0.005).
Sealed / OOD cohorts (2D sealed128, ood32; 3D sealed64, ood16) are NOT readable through this loader (A3.8).

Every load verifies the sha256 of both files against the committed store/MANIFEST.sha256. The store root is
`<lane>/store` locally, or $JCPREF_STORE (e.g. the cluster copy /cluster/tufts/paralab/tawal01/jcp-ref/store).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import verdict as V  # noqa: E402

SEALED = {(2, 'sealed128'), (2, 'ood32'), (3, 'sealed64'), (3, 'ood16')}
COHORTS = {2: ('dev6', 'val32', 'test64'), 3: ('dev4', 'val64', 'test32')}
LATTICE = {2: 256, 3: 64}
STRIDES = {2: (2, 4, 8, 16, 32, 64, 128), 3: (2, 4, 8, 16, 32)}
# pre-registered cross-check cases (cohorts.crosscheck_cases; pinned here so membership never comes from STATUS alone)
CROSSCHECK = {2: ('val32', (0, 1, 2, 3, 4, 5, 19, 26)), 3: ('val64', (0, 1, 2, 3, 5, 33, 43, 47))}


class SealedCohortError(PermissionError):
    pass


class ReferenceIntegrityError(RuntimeError):
    pass


def _root():
    return Path(os.environ.get('JCPREF_STORE', HERE / 'store'))


def _manifest():
    m = {}
    for line in (HERE / 'store' / 'MANIFEST.sha256').read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            m[p.strip()] = h
    return m


def _status():
    st = json.loads((HERE / 'store' / 'STATUS.json').read_text())
    if st.get('store_version') != store_version():
        raise ReferenceIntegrityError('STATUS.json was written for another store version')
    for d, (coh, cases) in CROSSCHECK.items():
        if sorted(st['crosscheck_cases'][str(d)]) != [f'{coh}/{c}' for c in sorted(cases)]:
            raise ReferenceIntegrityError('STATUS.json cross-check case list differs from the pre-registered one')
    return st


def store_version():
    return hashlib.sha256((HERE / 'store' / 'MANIFEST.sha256').read_bytes()).hexdigest()


def _check(dim, cohort, kind):
    if (dim, cohort) in SEALED:
        raise SealedCohortError(f'{dim}D {cohort} is sealed; it cannot be read through refstore (DESIGN A3.8)')
    if cohort not in COHORTS[dim]:
        raise KeyError(f'unknown cohort {cohort} for {dim}D (known: {COHORTS[dim]})')
    if kind not in ('ST', 'S'):
        raise KeyError(kind)


def _read(rel):
    man = _manifest()
    if rel not in man:
        raise ReferenceIntegrityError(f'{rel} not in the committed manifest')
    b = (_root() / rel).read_bytes()
    if hashlib.sha256(b).hexdigest() != man[rel]:
        raise ReferenceIntegrityError(f'sha256 mismatch for {rel}')
    return b


def _load_raw(dim, cohort, case, kind):
    _check(dim, cohort, kind)
    rel = f'{dim}d/{cohort}/{kind}/c{case:03d}'
    meta = json.loads(_read(rel + '.json'))
    import io
    arr = np.load(io.BytesIO(_read(rel + '.npz')))
    if (meta['dim'], meta['cohort'], meta['case'], meta['kind']) != (dim, cohort, case, kind):
        raise ReferenceIntegrityError(f'{rel}: metadata identity mismatch')
    return arr, meta


def load(dim, cohort, case, kind='ST', field='finest'):
    """-> (fields, meta). field 'finest' (the primary reference, uncertainty eps_ref) or 'richardson' (diagnostic
    extrapolation, NO uncertainty attached) or 'levels' (the three final level fields, coarse -> fine)."""
    arr, meta = _load_raw(dim, cohort, case, kind)
    ids = meta['final']
    if field == 'finest':
        return np.asarray(arr[ids[2]]), meta
    if field == 'levels':
        return [np.asarray(arr[i]) for i in ids], meta
    if field == 'richardson':
        jd = meta['judge']
        m2 = dict(meta, field='richardson', uncertainty=None,
                  note='diagnostic extrapolation; eps_ref does NOT apply to this field')
        m2['judge'] = dict(jd, eps_ref=None, eps_t=None)
        return V.richardson_field(arr[ids[1]], arr[ids[2]], jd['p'], jd['h'][1], jd['h'][2]), m2
    raise KeyError(field)


def reference_status(dim, cohort, case, kind='ST'):
    _check(dim, cohort, kind)
    meta = json.loads(_read(f'{dim}d/{cohort}/{kind}/c{case:03d}.json'))
    st = _status()
    dstat = st['dimension_status'][str(dim)]
    is_xc = cohort == CROSSCHECK[dim][0] and case in CROSSCHECK[dim][1]
    xc = st['crosscheck'].get(str(dim), {}).get(f'{cohort}/{case}') if is_xc else None
    eligible = bool(meta['outcome'] == 'accepted' and dstat == 'validated' and (not is_xc or xc == 'PASS'))
    return dict(accepted=meta['outcome'] == 'accepted', outcome=meta['outcome'], eps_ref=meta['judge']['eps_ref'],
                dimension_status=dstat, crosscheck=xc, eligible=eligible)


def score(field, dim, cohort, case, kind='ST'):
    """One arm's error against the reference (no ranking claim). `field` in the conventions above."""
    ref, meta = load(dim, cohort, case, kind)
    f = np.asarray(field, np.float64)
    if f.shape != ref.shape:
        raise ValueError(f'field shape {f.shape} != reference shape {ref.shape}')
    n0 = meta['judge']['n0']
    e, e_t = V.rel_err(f, ref, n0)
    st = reference_status(dim, cohort, case, kind)
    return dict(dim=dim, cohort=cohort, case=case, kind=kind, metric_id=V.METRIC_ID[dim],
                store_version=store_version(), e=e, e_t=e_t, eps_ref=st['eps_ref'], eligible=st['eligible'],
                outcome=st['outcome'])


def resolved(sa, sb):
    """Pairwise ranking rule (A2.10/A3.7): True iff |e_a - e_b| > 2 eps_ref. Raises unless both scores are eligible and
    refer to the same (dim, cohort, case, kind, metric_id, store_version); eps_ref is re-read from the store."""
    keys = ('dim', 'cohort', 'case', 'kind', 'metric_id', 'store_version')
    if any(sa[k] != sb[k] for k in keys):
        raise ValueError('scores refer to different cases, kinds, metrics or store versions')
    if sa['store_version'] != store_version():
        raise ValueError('scores were computed against another store version')
    st = reference_status(sa['dim'], sa['cohort'], sa['case'], sa['kind'])
    if not (st['eligible'] and sa['eligible'] and sb['eligible']):
        raise ValueError(f"case not eligible for ranking (outcome {st['outcome']}, dimension {st['dimension_status']}, "
                         f"cross-check {st['crosscheck']})")
    eps = st['eps_ref']
    if eps is None or not np.isfinite(eps) or eps < 0:
        raise ValueError('invalid eps_ref')
    return bool(abs(sa['e'] - sb['e']) > 2 * eps)


def restrict_field(fields, dim, factor):
    """Restrict lattice fields to the sub-lattice x = k*factor/lattice (coordinate-wise)."""
    if factor not in STRIDES[dim]:
        raise ValueError(f'factor must be one of {STRIDES[dim]}')
    f = np.asarray(fields)
    if dim == 2:
        return f[:, ::factor, ::factor]
    m = LATTICE[3]
    g = f.reshape(6, m - 1, m - 1, m - 1)
    idx = np.arange(factor, m, factor) - 1
    return g[:, idx][:, :, idx][:, :, :, idx].reshape(6, -1)


def restrict(dim, cohort, case, kind, factor):
    """Reference levels restricted to a coarser sub-lattice, with n0, p, eps_ref, c AND the outcome recomputed there.
    DIAGNOSTIC ONLY: eligible is always False (A3.3)."""
    levels, meta = load(dim, cohort, case, kind, field='levels')
    arr, _ = _load_raw(dim, cohort, case, kind)
    t0 = restrict_field(np.asarray(arr['t0'])[None], dim, factor)[0]
    att = {a['id']: a for a in meta['attempts']}
    lv = [dict(L=att[i]['L'], lat=restrict_field(u, dim, factor), tau=att[i]['tau'], steps_ok=att[i]['steps_ok'],
               umax=att[i]['umax_running']) for i, u in zip(meta['final'], levels)]
    jd = V.judge(lv, t0, meta['desc']['nu'])
    out = dict(judge=jd, factor=factor, metric_id=V.METRIC_ID[dim].replace(f'lattice={LATTICE[dim]}',
                                                                            f'lattice={LATTICE[dim] // factor}'),
               eligible=False, fields=lv[2]['lat'], crosscheck=None)
    if kind == 'ST' and cohort == CROSSCHECK[dim][0] and case in CROSSCHECK[dim][1]:
        out['crosscheck'] = _restricted_crosscheck(dim, cohort, case, factor, jd, lv[2]['lat'])
    return out


def _restricted_crosscheck(dim, cohort, case, factor, jd, u3):
    """The ordered cross-check recomputed on the sub-lattice from the stored PS runs (diagnostic, A2.10)."""
    import io
    man = _manifest()
    cands = sorted(k for k in man if k.startswith(f'{dim}d/{cohort}/PS/c{case:03d}_a') and k.endswith('.json'))
    if not cands:
        return None
    rel = cands[-1][:-5]
    ps = json.loads(_read(rel + '.json'))
    a = np.load(io.BytesIO(_read(rel + '.npz')))
    runs = {r['id']: r for r in ps['runs']}
    R = lambda i: restrict_field(np.asarray(a[i]), dim, factor)
    full = next(i for i in ps['selected_spatial'] if runs[i]['N'] == ps['Ns'][-1])
    return V.crosscheck(jd, u3, dict(spatial=[(runs[i]['N'], R(i)) for i in ps['selected_spatial']],
                                     half=R(ps['selected_half']), full=R(full), finite=ps['finite'],
                                     gates_ok=ps['gates_ok'], stability_ok=ps['stability_ok']), jd['n0'])
