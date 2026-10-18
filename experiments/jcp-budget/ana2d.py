"""Offline 2D B2 analysis (DESIGN 4, A1.1-A1.6, A2.1-A2.5, A3.1-A3.2; CPU, NumPy only, independent decode npcheck.decode257).

From a pulled b2run output directory (B2_*.npz, B2CHK_*.npz, STAR_*.npz, STARCHK_*.npz, CTRL_b2.npz, result.json) and the
E0 store (refstore; every score through the identity-checked `score_e0`), per member and case, per output time
k = 1..5, relative to n0 = ||R(t0)||:

  pass 1  G-B4: check-rule replays of every Q-containing corner type and of the u_star family (check cases); a corner
          type that fails is treated as INVALID on every case (A3.2), a failed u_star family makes the ideal end
          unavailable (e_dyn*, chi*, U_star, D_a closure);
  pass 2  corners F_S (decoded), Moebius effects I_S, main effects, pairs, closure (G-B2), floor e_P, e_dyn*,
          chi_x = 2<I_x, F_0 - Pi R> and chi*_x = 2<I_x, F_star - Pi R> with the numerical background chi - chi*,
          U_star, D_a and its sum / RSS comparison, delta_form_tight, IC errors and IC-effect persistence, the B0 identity
          for every corner with its count (G-B0) and the real-data mismatched-normaliser fixture, G-B3 coverage, C-F;
  decide  A1.4 / A2.4 / A3.2 rules: largest (ties, incomplete), dominance (population after exclusions, 75 % complete),
          pair labels (eligible-case denominators), controls C-Q / C-S / C-I / C-T with comparator validity, G-B1,
          C-score on the actual scoring path.
Mesh comparison (A1.4): `python ana2d.py --compare a.json b.json c.json` -> same / differs / inconclusive on sub16 and
component ratios with case-bootstrap intervals.

    python ana2d.py --run runs/<attempt>/archive/output/b2run_<cfg> [--minpop 8] [--out ...]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import npcheck as NC  # noqa: E402

E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
sys.path.insert(0, str(E0))
import refstore as RS  # noqa: E402
sys.path.pop(0)
WT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees')
ING = ('Q', 'T', 'F', 'S', 'I', 'M')
TAU, CONV_BAR = 2.5e-4, 2.5e-5
METRIC_ID = RS.V.METRIC_ID[2]
TIME_TAG = 't0,t1,t2,t3,t4,t5'
SUB16 = {f'dev6|{c}' for c in range(6)} | {f'val32|{c}' for c in range(10)}


def case_set(which):
    """The configured case list (p1common.select names) as ordered case ids, independent of any produced record."""
    if which == 'all38':
        return [f'dev6|{c}' for c in range(6)] + [f'val32|{c}' for c in range(32)]
    if which == 'sub16':
        return [f'dev6|{c}' for c in range(6)] + [f'val32|{c}' for c in range(10)]
    if which == 'dev6':
        return [f'dev6|{c}' for c in range(6)]
    if which == 'dev2':
        return ['dev6|0', 'dev6|1']
    return [f'{a}|{b}' for a, b in which]


def subsets_of(design):
    S1 = [frozenset()] + [frozenset([x]) for x in ING]
    S2 = [frozenset(c) for c in itertools.combinations(ING, 2)]
    full = [frozenset(c) for n in (3, 4, 5) for c in itertools.combinations(('Q', 'T', 'F', 'S', 'I'), n)]
    return S1 + S2 + (full if design == 'full' else [])
STARS = ('star', 'star16', 'starx', 'star16x')
RNG = np.random.default_rng(0)


class IdentityError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ scoring path --

def load_e0(cid, kind='ST', expect_kind='ST'):
    """The E0 reference of a case with identity checks (C-score): returns (fields, n0, eps_ref, eligible, sha)."""
    coh, c = cid.split('|')
    R, meta = RS.load(2, coh, int(c), kind)
    if (meta['cohort'], meta['case'], meta['kind'], meta['dim']) != (coh, int(c), expect_kind, 2):
        raise IdentityError(f"reference identity {meta['cohort']}|{meta['case']}|{meta['kind']} != {cid}|{expect_kind}")
    if meta.get('metric_id', METRIC_ID) != METRIC_ID or len(meta['times']) != 6:
        raise IdentityError('metric / time list')
    st = RS.reference_status(2, coh, int(c), kind)
    sha = RS._manifest()[f'2d/{coh}/{kind}/c{int(c):03d}.npz']
    return np.asarray(R), float(meta['judge']['n0']), st['eps_ref'], st['eligible'], sha


def score_e0(F, rec):
    """Identity-checked E0 score of fields F (6, 257, 257). rec: case_id (the field's own case), requested (cohort, case),
    time_tag of the field, kind (ST; S only for LSPG-BE at dt0), expect_kind (what the caller believes it scores)."""
    coh, c = rec['requested']
    if rec['case_id'] != f'{coh}|{c}':
        raise IdentityError('case swap')
    if rec['time_tag'] != TIME_TAG:
        raise IdentityError('time shift')
    kind = rec.get('kind', 'ST')
    if kind == 'S' and not (rec.get('scheme') == 'BE' and abs(rec.get('dt', 0) - .005) < 1e-15):
        raise IdentityError('S is matched only to LSPG-BE at dt0')
    R, n0, eps, el, sha = load_e0(rec['case_id'], kind, rec.get('expect_kind', kind))
    e_t = [float(np.linalg.norm(F[k] - R[k])) / n0 for k in range(6)]
    return dict(e=max(e_t[1:]), e_t=e_t, eps_ref=eps, eligible=el, reference_sha256=sha, kind=kind, time_tag=TIME_TAG)


def b0_fixtures():
    """combined A3.5 analytic negative fixtures (must break the identity)."""
    defect = lambda v, y, P: float(np.linalg.norm(v - y) ** 2 - np.linalg.norm(y - P @ y) ** 2 - np.linalg.norm(v - P @ y) ** 2)
    e1, e2 = np.array([1., 0.]), np.array([0., 1.])
    f1 = defect(e1 + 1e-3 * e2, e2, np.diag([1., 0.]))
    f2 = defect(e1, e2, np.array([[1., .5], [0., 0.]]))
    return dict(f1=f1, f2=f2, passed=bool(abs(f1 + 2e-3) < 1e-12 and abs(f2 - .5) < 1e-12))


class Span:
    """Orthonormal basis of the bank prefix at the 255^2 interior lattice nodes (Pi of DESIGN 4.2)."""

    def __init__(self, r):
        NC.decode257(np.zeros((1, r)), r)
        G = NC._G257[r].reshape(257, 257, r)[1:-1, 1:-1].reshape(-1, r)
        self.Q, _ = np.linalg.qr(G)

    def proj(self, X):
        """X interior (n, 255^2) or fields (n, 257, 257) -> Pi X on the interior."""
        X = inner(X) if np.asarray(X).ndim == 3 else np.asarray(X)
        return (X @ self.Q) @ self.Q.T


def inner(F):
    return np.asarray(F)[:, 1:-1, 1:-1].reshape(len(F), -1)


def nt(X, n0):
    return np.linalg.norm(np.asarray(X).reshape(len(X), -1), axis=1) / n0


def mobius(corners, avail):
    """{subset-key: vector effect} for every subset whose sub-corners are all available (key '0' = the deployed)."""
    sets = {k: frozenset('' if k == '0' else k) for k in corners}
    inv = {v: k for k, v in sets.items()}
    out = {}
    for k, S in sets.items():
        if not S:
            continue
        subs = [frozenset(c) for n in range(len(S) + 1) for c in itertools.combinations(sorted(S), n)]
        if all(T in inv and avail[inv[T]] for T in subs):
            out[k] = sum(((-1) ** (len(S) - len(T))) * corners[inv[T]] for T in subs)
    return out


def skey(S):
    return ''.join(x for x in ING if x in S) or '0'


# ------------------------------------------------------------------------------------------------- archives --

def archive_f257(mesh, member):
    """G-B3: jcp-qualify / P1 archived fields of the SAME configuration, verified by the archive's own metadata."""
    r, form, scheme, fac = int(member['Rp']), member['form'], member['scheme'], float(member['factor'])
    pats = [f'{WT}/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/*{mesh}/archive/output/*/R_R{r}_{form}_{scheme}_{fac:g}.npz']
    if form == 'LSPG' and fac == 1.:
        pats.append(f'{WT}/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/acc{mesh}/archive/output/*/R_R{r}_{scheme}.npz')
    out, srcs = {}, []
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            z = np.load(f)
            meta = dict(rule=str(z['rule']), Rp=int(z['Rp']), mesh=int(z['mesh']), scheme=str(z['scheme']),
                        form=str(z['form']) if 'form' in z.files else 'LSPG', policy=str(z['policy']),
                        dt=float(z['dt']) if 'dt' in z.files else None)
            want = dict(rule=member['rule'], Rp=r, mesh=int(mesh), scheme=scheme, form=form, policy='frozen')
            if any(meta[k] != v for k, v in want.items()) or (meta['dt'] is not None and abs(meta['dt'] - .005 * fac) > 1e-12):
                continue                              # another configuration (fallback rule, other step): not this member
            srcs.append(f)
            for i, c in enumerate(z['case_ids']):
                if np.isfinite(z['f257'][i]).all():
                    out.setdefault(str(c), np.asarray(z['f257'][i]))
    return out, srcs


# ----------------------------------------------------------------------------------------------------- pass 1 --

def gb4_member(run, mk, r, refs, exp_corners, exp_cases):
    """Per Q-containing corner type: max over check cases of the check-rule distance; None entries = invalid run. The
    expected inventory (configured design's Q corners x configured check cases) is checked independently: a missing
    corner type or case is a FAILED check."""
    z = np.load(run / f"B2_{mk.replace('|', '_')}.npz")
    zc = np.load(run / f"B2CHK_{mk.replace('|', '_')}.npz")
    got_cases, got_corners = [str(x) for x in zc['case_ids']], [str(x) for x in zc['corners']]
    if sorted(got_cases) != sorted(exp_cases) or sorted(got_corners) != sorted(exp_corners):
        return {c: dict(worst=None, n=0, passed=False, note='check inventory differs from the configuration')
                for c in exp_corners}
    ck = [str(x) for x in z['case_ids']]
    corners = [str(x) for x in z['corners']]
    res = {}
    for cj, cid in enumerate([str(x) for x in zc['case_ids']]):
        ci = ck.index(cid)
        n0 = refs(cid)[1]
        for kk, c in enumerate([str(x) for x in zc['corners']]):
            s_ = corners.index(c)
            ok = bool(zc['valid'][cj, kk]) and bool(z['valid'][ci, s_])
            d = float(max(np.linalg.norm((NC.decode257(zc['W'][cj, kk], r) - NC.decode257(z['W'][ci, s_], r))[k])
                          for k in range(1, 6)) / n0) if ok else None
            res.setdefault(c, []).append(d)
    return {c: dict(worst=max([v for v in vs if v is not None], default=None), n=len(vs),
                    passed=bool(vs and all(v is not None and v <= CONV_BAR for v in vs))) for c, vs in res.items()}


def gb4_star(run, r, refs):
    zs, zsc = np.load(run / f'STAR_R{r}.npz'), np.load(run / f'STARCHK_R{r}.npz')
    ids = {str(c): i for i, c in enumerate(zs['case_ids'])}
    vals = []
    for i, c in enumerate(zsc['case_ids']):
        j = ids[str(c)]
        for k in STARS:
            if zsc[f'valid_{k}'][i] and zs[f'valid_{k}'][j]:
                vals.append(float(max(np.linalg.norm((NC.decode257(zsc[f'W_{k}'][i], r) - NC.decode257(zs[f'W_{k}'][j], r))[t])
                                      for t in range(1, 6)) / refs(str(c))[1]))
            else:
                vals.append(None)
    return dict(worst=max([v for v in vals if v is not None], default=None), n=len(vals),
                passed=bool(vals and all(v is not None and v <= CONV_BAR for v in vals)))


# ----------------------------------------------------------------------------------------------------- pass 2 --

def analyse_member(run, mk, member, mesh, span, star, refs, failed, star_ok, chk_ids):
    z = np.load(run / f"B2_{mk.replace('|', '_')}.npz")
    ck = [str(x) for x in z['case_ids']]
    corners = [str(x) for x in z['corners']]
    r = int(member['Rp'])
    arch, srcs = archive_f257(mesh, member)
    per = []
    st = dict(gb0=0., gb0_n=0, gb2=0., gb2_n=0, mism=[], gb3=[], gb3_missing=0)
    far = 'QTFSI' if member['form'] == 'LSPG' else 'QTSI'          # the corner that equals u_star's specification
    for ci, cid in enumerate(ck):
        R, n0, eps, elig, rsha = refs(cid)
        raw = {c: bool(z['valid'][ci, j]) for j, c in enumerate(corners)}                # run validity (identities)
        valid = {c: raw[c] and c not in failed for c in corners}                         # + quadrature gate (effects)
        F = {c: NC.decode257(z['W'][ci, j], r) for j, c in enumerate(corners)}
        Fi = {c: inner(F[c]) for c in corners}
        RI = inner(R)
        PR = span.proj(RI)
        eP = nt(RI - PR, n0)
        d = dict(case=cid, eligible=elig, eps_ref=eps, reference_sha256=rsha, valid=valid, e_P=float(eP[1:].max()),
                 e_P_t0=float(eP[0]))
        for c in corners:                                                            # G-B0, every valid corner
            if raw[c]:
                ev, ed = nt(Fi[c] - RI, n0), nt(Fi[c] - PR, n0)
                st['gb0'] = max(st['gb0'], float((np.abs(ev ** 2 - eP ** 2 - ed ** 2) / np.maximum(ev ** 2, 1e-300))[1:].max()))
                st['gb0_n'] += 5
        if valid['0']:          # real-data mismatched-normaliser fixture (combined A12.1): must break the identity
            ev, ed = nt(Fi['0'] - RI, n0), nt(Fi['0'] - PR, n0)
            st['mism'].append(float(abs(ev[5] ** 2 - (eP[5] / .1) ** 2 - ed[5] ** 2) / max(ev[5] ** 2, 1e-300)))
            sc = score_e0(F['0'], dict(case_id=cid, requested=tuple(cid.split('|')), time_tag=TIME_TAG))
            d['e_tot'] = sc['e']
            d['e_t0'] = float(nt(Fi['0'] - RI, n0)[0])
        else:
            d['e_tot'] = d['e_t0'] = None
        if valid.get('I'):
            d['e_t0_ls'] = float(nt(Fi['I'] - RI, n0)[0])
        eff = mobius(Fi, valid)
        mx = lambda X: float(nt(X, n0)[1:].max())
        d['main'] = {x: (mx(eff[x]) if x in eff else None) for x in ING}
        d['pairs'] = {''.join(p): (mx(eff[''.join(p)]) if ''.join(p) in eff else None) for p in itertools.combinations(ING, 2)}
        d['higher'] = {k: mx(v) for k, v in eff.items() if len(k) >= 3}
        if 'I' in eff:                                     # IC-effect persistence (6.4): t5 vs t1
            e_ = nt(eff['I'], n0)
            d['I_t1'], d['I_t5'] = float(e_[1]), float(e_[5])
        if 'QTFSI' in corners and all(raw[c] for c in corners if set(c) <= set('QTFSI') or c == '0'):
            effr = mobius({c: Fi[c] for c in corners if set(c) <= set('QTFSI') or c == '0'}, raw)   # bookkeeping check
            tot = sum(v for k, v in effr.items() if set(k) <= set('QTFSI'))
            # A6: the bar is relative to the rounding scale of the 32-term sum (the largest corner norm), not to the
            # (much smaller) difference u_all - u_0
            scale = max(float(np.linalg.norm(Fi[c])) for c in corners if set(c) <= set('QTFSI') or c == '0')
            rel = float(np.linalg.norm(Fi['QTFSI'] - Fi['0'] - tot) / max(scale, 1e-300))
            st['gb2'] = max(st['gb2'], rel)
            st['gb2_n'] += 1
            d['closure_rel'] = rel
        si = star['ids'].get(cid)
        if star_ok and si is not None and all(star['valid'][k][si] for k in STARS):
            Fs = {k: inner(NC.decode257(star['W'][k][si], r)) for k in STARS}
            Us = nt(Fs['star'] - Fs['star16'], n0) + nt(Fs['star'] - Fs['starx'], n0) + nt(Fs['star16'] - Fs['star16x'], n0)
            d['U_star'] = float(Us[1:].max())
            d['e_dyn_star'] = mx(Fs['star'] - PR)
            es, ed = nt(Fs['star'] - RI, n0), nt(Fs['star'] - PR, n0)
            st['gb0'] = max(st['gb0'], float((np.abs(es ** 2 - eP ** 2 - ed ** 2) / np.maximum(es ** 2, 1e-300))[1:].max()))
            st['gb0_n'] += 5
            if valid['0']:
                d['D_a'] = mx(Fi['0'] - Fs['star'])
                ms = [d['main'][x] for x in ING if d['main'].get(x) is not None and x != 'M']
                d['D_a_over_sum'] = d['D_a'] / sum(ms) if ms and sum(ms) > 0 else None
                d['D_a_over_rss'] = d['D_a'] / float(np.sqrt(sum(m * m for m in ms))) if ms and sum(ms) > 0 else None
            if valid.get(far):                       # the far corner has u_star's specification: determinism / closure end
                d['far_vs_star'] = mx(Fi[far] - Fs['star'])
            d['chi_star'], d['chi'], d['chi_background'] = {}, {}, {}
            for x in ING:
                if x in eff and valid['0']:
                    cs = 2 * np.sum(eff[x] * (Fs['star'] - PR), axis=1) / n0 ** 2
                    ca = 2 * np.sum(eff[x] * (Fi['0'] - PR), axis=1) / n0 ** 2
                    k = 1 + int(np.argmax(np.abs(ca[1:])))
                    d['chi'][x] = float(ca[k])                             # chi = chi* + background at the SAME time k
                    d['chi_star'][x] = float(cs[k])
                    d['chi_background'][x] = float((ca - cs)[k])
                    d.setdefault('chi_star_peak', {})[x] = float(cs[1 + int(np.argmax(np.abs(cs[1:])))])
            if cid in chk_ids and valid['0'] and all(x in eff for x in ING):
                d['C_F'] = control_floor(RI, PR, eP, n0, eps or 0., d['U_star'], span, Fs['star'], eff, d)
        else:
            d['U_star'] = d['e_dyn_star'] = None
        if valid.get('S') and valid.get('FS'):
            d['delta_form_tight'] = mx(Fi['FS'] - Fi['S'])
        if valid['0']:
            if cid in arch:
                d['G_B3'] = float(max(np.linalg.norm(F['0'][k] - arch[cid][k]) for k in range(1, 6)) / n0)
                st['gb3'].append(d['G_B3'])
            else:
                st['gb3_missing'] += 1
        per.append(d)
    return per, st, srcs


def control_floor(RI, PR, eP, n0, eps, U, span, Fstar, eff, d):
    """C-F: inject eta_k v_k, v_k orthogonal to the span AND to the floor residual r_k (unit lattice norm),
    eta_k = n0 (10 max_c + 2 eps + 3 U); e_P must become sqrt(e_P^2 + (eta/n0)^2) per time, e_dyn* unchanged, and the
    decision code must label the floor largest."""
    x = np.arange(257) / 256.
    edyn = nt(Fstar - PR, n0)
    mains = {k: nt(v, n0) for k, v in eff.items() if len(k) == 1}
    ident, seeds = [], []
    RI2 = RI.copy()
    for k in range(1, 6):
        maxc = max([eP[k], edyn[k]] + [m[k] for m in mains.values()])
        eta = n0 * (10 * maxc + 2 * eps + 3 * U)
        rk = RI[k] - PR[k]
        for q in (255, 254, 253, 252):
            h = np.outer(np.sin(q * np.pi * x), np.sin(q * np.pi * x))[1:-1, 1:-1].reshape(-1)
            hp = h - span.Q @ (span.Q.T @ h)
            if np.linalg.norm(hp) < 1e-8 * np.linalg.norm(h):
                continue
            nr = np.linalg.norm(rk)
            v = hp - (hp @ rk) / nr ** 2 * rk if nr > 0 else hp
            if np.linalg.norm(v) < 1e-8 * np.linalg.norm(hp):
                continue
            v /= np.linalg.norm(v)
            seeds.append(q)
            break
        else:
            return dict(available=False)
        RI2[k] = RI[k] + eta * v
        eP2 = np.linalg.norm(RI2[k] - span.Q @ (span.Q.T @ RI2[k])) / n0
        want = np.sqrt(eP[k] ** 2 + (eta / n0) ** 2)
        ident.append(abs(eP2 - want) / want)
    PR2 = span.proj(RI2)
    d2 = dict(d, e_P=float(nt(RI2 - PR2, n0)[1:].max()), e_dyn_star=float(nt(Fstar - PR2, n0)[1:].max()))
    lab = largest(d2)['label']
    dyn_same = bool(abs(d2['e_dyn_star'] - float(edyn[1:].max())) <= 1e-12 * max(float(edyn[1:].max()), 1e-300))
    return dict(available=True, identity_worst=float(max(ident)), seeds=seeds, dyn_unchanged=dyn_same, label=lab,
                fired=bool(max(ident) <= 1e-10 and dyn_same and lab == 'e_P'))


# ---------------------------------------------------------------------------------------------------- decide --

def largest(d):
    """A1.4 / A2.4 / A3.2: the largest candidate with tie thresholds; 'incomplete' if any candidate is unavailable."""
    U, eps = d.get('U_star'), d['eps_ref'] or 0.
    cand = {'e_P': ('ref', d['e_P']), 'e_dyn_star': ('ref', d['e_dyn_star'])}
    cand.update({x: ('traj', d['main'].get(x)) for x in ING})
    if U is None or any(v is None for _, v in cand.values()):
        return dict(label='incomplete', available={k: v for k, (_, v) in cand.items() if v is not None})
    res = {}
    for k, (kind, v) in cand.items():
        res[k] = (v >= 3 * U) if kind == 'traj' else ((v > 2 * eps) if k == 'e_P' else (v > 2 * eps and v >= 3 * U))
    order = sorted(((v, k) for k, (_, v) in cand.items() if res[k]), reverse=True)
    if not order:
        return dict(label='none resolved')
    if len(order) > 1:
        (va, ka), (vb, kb) = order[0], order[1]
        ta, tb = cand[ka][0], cand[kb][0]
        tau = 3 * U if ta == tb == 'traj' else (2 * eps if ta == tb == 'ref' else 2 * eps + 3 * U)
        if va - vb <= tau:
            return dict(label='tied', top=[ka, kb])
    return dict(label=order[0][1])


def pair_label(d, x, y):
    U = d.get('U_star')
    a, b, p = d['main'].get(x), d['main'].get(y), d['pairs'].get(''.join(sorted((x, y), key=ING.index)))
    if U is None or a is None or b is None or p is None:
        return 'unavailable'
    big = max(a, b)
    if p > .25 * big and p >= 3 * U:
        return 'interacting'
    if p <= .25 * big and big >= 3 * U:
        return 'additive'
    return 'unresolved'


def summarise(per, minpop):
    el = [d for d in per if d['eligible']]
    lab = [largest(d)['label'] for d in el]
    comp = [l for l in lab if l != 'incomplete']
    out = dict(n_eligible=len(el), n_complete=len(comp), counts={l: lab.count(l) for l in set(lab)})
    if len(comp) < minpop:
        out['dominant'] = 'inconclusive (population)'
    elif len(comp) < .75 * len(el):
        out['dominant'] = 'inconclusive (incomplete)'
    else:
        top = max(set(comp), key=comp.count)
        out['dominant'] = top if (top not in ('tied', 'none resolved') and comp.count(top) > .5 * len(comp)) else 'no single dominant'
    pl = {}
    for x, y in itertools.combinations(ING, 2):
        labs = [pair_label(d, x, y) for d in el]
        av = [l for l in labs if l != 'unavailable']
        if len(av) < minpop:
            pl[x + y] = 'inconclusive (population)'
        elif sum(l == 'interacting' for l in labs) > .25 * len(el):          # eligible-case denominators (A1.4)
            pl[x + y] = 'interacting'
        elif sum(l == 'additive' for l in labs) >= .75 * len(el):
            pl[x + y] = 'additive'
        else:
            pl[x + y] = 'mixed'
    out['pairs'] = pl
    med = lambda vals: float(np.median(vals)) if vals else None
    out['median_main'] = {x: med([d['main'][x] for d in el if d['main'].get(x) is not None]) for x in ING}
    mains = {x: v for x, v in out['median_main'].items() if v is not None}
    out['two_largest'] = sorted(mains, key=mains.get, reverse=True)[:2]
    for k in ('e_P', 'e_dyn_star', 'U_star', 'D_a', 'e_tot', 'delta_form_tight', 'D_a_over_sum', 'D_a_over_rss', 'e_t0',
              'e_t0_ls', 'e_P_t0', 'far_vs_star'):
        v = [d[k] for d in el if d.get(k) is not None]
        out[f'median_{k}'] = med(v)
        out[f'worst_{k}'] = float(np.max(v)) if v else None
    for nm in ('chi', 'chi_star', 'chi_background'):
        out[f'median_{nm}'] = {x: med([d[nm][x] for d in el if d.get(nm, {}).get(x) is not None]) for x in ING}
    pers = [(d['I_t5'], d['I_t1']) for d in el if 'I_t1' in d]
    out['IC_persists'] = dict(cases=len(pers), t5_gt_t1=sum(a > b for a, b in pers))
    return out


# ------------------------------------------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run')
    ap.add_argument('--minpop', type=int, default=None)
    ap.add_argument('--compare', nargs=3, help='three budget2d json files (256, 1024, 4096) -> mesh comparison')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.compare:
        return mesh_compare(a.compare, a.out)
    run = Path(a.run)
    res = json.loads((run / 'result.json').read_text())
    cfg = res['config']
    mesh = int(cfg['mesh'])
    minpop = a.minpop if a.minpop is not None else (6 if mesh == 4096 else 8)
    out = dict(run=str(run), mesh=mesh, store_version=RS.store_version(), label='E0 R_ST, conditional Richardson eps_ref',
               B0_fixtures=b0_fixtures(), members={}, gates={}, controls={})
    cache = {}

    def refs(cid):
        if cid not in cache:
            cache[cid] = load_e0(cid)
        return cache[cid]

    tot = dict(gb0=0., gb0_n=0, gb2=0., gb2_n=0, mism=[], gb3=[], gb3_missing=0)
    gb4 = {}
    case_cov_bad = []
    for rk, rr in res['b2'].items():
        r = int(rk[1:])
        span = Span(r)
        zs = np.load(run / f'STAR_R{r}.npz')
        star = dict(ids={str(c): i for i, c in enumerate(zs['case_ids'])}, W={k: zs[f'W_{k}'] for k in STARS},
                    valid={k: zs[f'valid_{k}'] for k in STARS})
        sg = gb4_star(run, r, refs)
        zsc_ = np.load(run / f'STARCHK_R{r}.npz')
        exp_chk_ = [c for c in case_set(cfg['b2'].get('check_cases', 'dev6')) if c in case_set(cfg['cases'])]
        if sorted(str(x) for x in zsc_['case_ids']) != sorted(exp_chk_) or \
                [str(x) for x in zs['case_ids']] != case_set(cfg['cases']):
            sg = dict(sg, passed=False, note='star inventory differs from the configuration')
        chk_ids = {str(c) for c in np.load(run / f'STARCHK_R{r}.npz')['case_ids']}
        for mk, mm in rr['members'].items():
            exp_q = [skey(S) for S in subsets_of(cfg['b2'].get('design', 'pairwise')) if 'Q' in S]
            exp_chk = [c for c in case_set(cfg['b2'].get('check_cases', 'dev6')) if c in case_set(cfg['cases'])]
            g4 = gb4_member(run, mk, r, refs, exp_q, exp_chk)
            zb = np.load(run / f"B2_{mk.replace('|', '_')}.npz")
            cov_bad = ([str(x) for x in zb['case_ids']] != case_set(cfg['cases']) or
                       [str(x) for x in zb['corners']] != [skey(S) for S in subsets_of(cfg['b2'].get('design', 'pairwise'))])
            if cov_bad:
                case_cov_bad.append(mk)
            failed = {c for c, v in g4.items() if not v['passed']}
            per, st, srcs = analyse_member(run, mk, mm['member'], mesh, span, star, refs, failed, sg['passed'], chk_ids)
            for k in ('gb0', 'gb2'):
                tot[k] = max(tot[k], st[k])
                tot[k + '_n'] += st[k + '_n']
            tot['mism'] += st['mism']
            tot['gb3'] += st['gb3']
            tot['gb3_missing'] += st['gb3_missing']
            gb4[mk] = dict(per_corner=g4, failed=sorted(failed), star_family=sg)
            out['members'][mk] = dict(member=mm['member'], diagnostic=mm['member']['form'] == 'GAL', archive_sources=srcs,
                                      summary=summarise(per, minpop), per_case=per,
                                      quadrature=('every Q corner type verified on the check cases; extrapolated to the rest'
                                                  if not failed else f'withheld for corner types {sorted(failed)}'),
                                      ideal_end='verified' if sg['passed'] else 'quadrature-unverified')
    conf = [f"R{m['Rp']}|{m['form']}|{m['scheme']}|{float(m['factor']):g}" for m in cfg['members']]
    prod = [mk for rr in res['b2'].values() for mk in rr['members']]
    out['coverage'] = dict(configured=conf, missing=[k for k in conf if k not in prod], extra=[k for k in prod if k not in conf],
                           star_missing=[r_ for r_ in sorted({int(m['Rp']) for m in cfg['members']}) if f'R{r_}' not in res['b2']])
    out['coverage']['case_or_corner_inventory_differs'] = case_cov_bad
    # G-B6 offline: the saved logical tuple of every corner of every member against the configuration
    g6 = []
    for m in cfg['members']:
        mk_ = f"R{m['Rp']}|{m['form']}|{m['scheme']}|{float(m['factor']):g}"
        saved = next((rr['members'][mk_]['tuples'] for rr in res['b2'].values() if mk_ in rr['members']), None)
        exp = []
        for S in subsets_of(cfg['b2'].get('design', 'pairwise')):
            form = {'LSPG': 'GAL', 'GAL': 'LSPG'}[m['form']] if 'F' in S else m['form']
            sc, dt = ('BDF2', .005 / 8) if 'T' in S else (m['scheme'], .005 * float(m['factor']))
            exp.append(dict(S=skey(S), role='qinf' if 'Q' in S else 'main', form=form, scheme=sc, dt=dt,
                            level='tight' if 'S' in S else 'prod', init='ls' if 'I' in S else 'deployed',
                            M=2048 if 'M' in S else 4 * int(m['Rp']), rule=cfg['b2']['qinf'] if 'Q' in S else m['rule']))
        ok = saved is not None and len(saved) == len(exp) and all(
            all((abs(a_[k] - e_[k]) <= 1e-15) if k == 'dt' else a_[k] == e_[k] for k in e_) for a_, e_ in zip(saved, exp))
        g6.append(dict(member=mk_, n=len(exp), passed=bool(ok)))
    G_B6 = dict(members=g6, passed=bool(g6 and all(x['passed'] for x in g6)))
    out['coverage']['passed'] = not (out['coverage']['missing'] or out['coverage']['extra'] or out['coverage']['star_missing']
                                     or case_cov_bad)
    out['gates'] = dict(
        G_B0=dict(worst=tot['gb0'], n=tot['gb0_n'], passed=bool(tot['gb0_n'] > 0 and tot['gb0'] <= 1e-10)),
        B0_real_mismatch=dict(min_defect=min(tot['mism']) if tot['mism'] else None,
                              passed=bool(tot['mism'] and min(tot['mism']) > 1e-6)),
        G_B2=dict(worst=tot['gb2'], n=tot['gb2_n'], passed=bool(tot['gb2_n'] > 0 and tot['gb2'] <= 1e-12) if
                  cfg['b2'].get('design') == 'full' else 'not applicable (pairwise design)'),
        G_B3=dict(worst=max(tot['gb3']) if tot['gb3'] else None, n=len(tot['gb3']), missing=tot['gb3_missing'],
                  passed=bool(tot['gb3'] and tot['gb3_missing'] == 0 and max(tot['gb3']) <= 1e-8)),
        G_B4=gb4, G_B5={k: v.get('G_B5') for k, v in res['setup'].items() if k.startswith('LS')}, G_B6=G_B6,
        B0_fixtures=out['B0_fixtures']['passed'])
    out['controls'] = controls(run, res, refs)
    ct = out['members'].get('R128|LSPG|BE|1')
    if ct is not None:
        chk = [d for d in ct['per_case'] if d['case'] in chk_ids]
        v = [d['main']['T'] for d in chk if d['main'].get('T') is not None]
        out['controls']['C_T'] = dict(values=v, n_check=len(chk), fired=bool(v and len(v) == len(chk) and
                                                                          sum(x > 1e-3 for x in v) > len(v) / 2))
    cf = [d['C_F'] for m in out['members'].values() for d in m['per_case'] if d.get('C_F', {}).get('available')]
    out['controls']['C_F'] = dict(n=len(cf), worst_identity=max([x['identity_worst'] for x in cf], default=None),
                                  labels=sorted({x['label'] for x in cf}), fired=bool(cf and all(x['fired'] for x in cf)))
    dest = Path(a.out or HERE / 'checks' / f"budget2d-{run.parents[1].name}-{mesh}.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(dest, json.dumps({k: (v.get('passed') if isinstance(v, dict) and 'passed' in v else '...') for k, v in out['gates'].items()}),
          json.dumps({k: v.get('fired') for k, v in out['controls'].items() if isinstance(v, dict) and 'fired' in v}))


def controls(run, res, refs):
    out = {}
    p = run / 'CTRL_b2.npz'
    if not p.exists():
        return out
    z = np.load(p)
    ids = [str(x) for x in z['case_ids']]
    r = int(res['config']['controls'].get('Rp', 384))
    n0 = {c: refs(c)[1] for c in ids}
    V = lambda k: [bool(x) for x in z[f'valid_{k}']] if f'valid_{k}' in z.files else [False] * len(ids)   # unknown = not valid
    dmax = lambda A, B, c: float(max(np.linalg.norm((NC.decode257(A, r) - NC.decode257(B, r))[k]) for k in range(1, 6)) / n0[c])
    q = [dmax(z['Q_ctrl'][i], z['Q_qinf'][i], c) if V('Q_qinf')[i] else None for i, c in enumerate(ids)]
    out['C_Q'] = dict(dist=q, comparator_valid=V('Q_qinf'), fired=bool(sum(v is not None and v > TAU for v in q) >= 1))
    s = [dmax(z['S_b0'][i], z['S_tight'][i], c) if V('S_tight')[i] else None for i, c in enumerate(ids)]
    out['C_S'] = dict(dist=s, comparator_valid=V('S_tight'), fired_majority=bool(sum(v is not None and v > TAU for v in s) > len(s) / 2),
                      fired_any=bool(sum(v is not None and v > TAU for v in s) >= 1))
    out['C_S']['fired'] = out['C_S']['fired_majority']
    ci = []
    for i, c in enumerate(ids):
        Fp, F0 = NC.decode257(z['I_pert'][i][:1], r), NC.decode257(z['S_prod'][i][:1], r)
        got = float(np.linalg.norm(Fp[0] - F0[0])) / n0[c]
        want = .01 * float(np.linalg.norm(F0[0])) / n0[c]
        ci.append(dict(got=got, want=want, rel=abs(got - want) / want if want > 0 else None))
    out['C_I'] = dict(rows=ci, fired=bool(all(x['rel'] is not None and x['rel'] <= 1e-12 and x['got'] > 0 for x in ci)))
    ctl = res.get('controls', {})
    out['G_B1'] = dict(sham_rule_bitwise=ctl.get('b2_controls', {}).get('sham_rule_bitwise'),
                       prod_rerun_bitwise=ctl.get('b2_controls', {}).get('prod_rerun_bitwise'),
                       offline_bitwise=bool(np.array_equal(z['sham'], z['S_prod']) and np.array_equal(z['S_prod2'], z['S_prod'])))
    out['G_B1']['passed'] = bool(out['G_B1']['offline_bitwise'] and out['G_B1']['sham_rule_bitwise'] and out['G_B1']['prod_rerun_bitwise'])
    gq = run / 'CTRL_gq.npz'                     # recomputed from the certificate evidence; missing evidence fails
    if gq.exists():
        zg = np.load(gq)
        cl, cg = np.asarray(zg['cert_LSPG'], float), np.asarray(zg['cert_GAL'], float)
        S_ = int(round(.25 / .005))                       # the schedule of the fixture: dt0 to T = 0.25 (50 steps)
        conf = case_set(res['config']['controls'].get('cases', 'dev6'))     # configured control cases, not produced ids
        n_ = min(2, len(conf))
        gids = [str(x) for x in zg['case_ids']] if 'case_ids' in zg.files else None     # the certificates' own case ids
        evaluable = bool(ids == conf and gids == conf[:n_] and len(set(gids)) == n_ and cl.shape == (n_, S_) and
                         cg.shape == (n_, S_) and np.isfinite(cl).all() and np.isfinite(cg).all())
        fl = float(np.mean(cl > 1e-3)) if evaluable else None             # registered bars (gamma 1e-3, GAL eta 1e-8)
        fg = float(np.mean(cg > 1e-8)) if evaluable else None
        out['G_Q'] = dict(evaluable=evaluable, fraction_LSPG=fl, fraction_GAL=fg, wrong_table_raised=bool(zg['wrong_table_raised']),
                          fired=bool(evaluable and fl > .01 and fg > .01 and bool(zg['wrong_table_raised'])))
    else:
        out['G_Q'] = dict(fired=False, note='no certificate evidence')
    # C-score on the actual scoring path (score_e0 / load_e0): each mismatch must raise
    c0 = ids[0]
    coh, cc = c0.split('|')
    F = NC.decode257(z['S_prod'][0], r)
    raised = {}
    other = next((c for c in ids if c != c0), 'dev6|5' if c0 != 'dev6|5' else 'dev6|4')
    tests = (('case_swap', dict(case_id=other, requested=(coh, cc), time_tag=TIME_TAG)),
             ('time_shift', dict(case_id=c0, requested=(coh, cc), time_tag='t1,t2,t3,t4,t5,t6')),
             ('ST_S_swap', dict(case_id=c0, requested=(coh, cc), time_tag=TIME_TAG, kind='S', expect_kind='ST', scheme='BE',
                                dt=.005)))
    for nm, rec in tests:
        try:
            score_e0(F, rec)
            raised[nm] = False
        except IdentityError:
            raised[nm] = True
    good = score_e0(F, dict(case_id=c0, requested=(coh, cc), time_tag=TIME_TAG))
    Fs = np.roll(F, 1, axis=0)
    shifted = float(max(np.linalg.norm(Fs[k] - refs(c0)[0][k]) for k in range(1, 6)) / refs(c0)[1])
    out['C_score'] = dict(raised=raised, aligned_e=good['e'], shifted_e=shifted, fired=bool(all(raised.values())))
    return out


def mesh_compare(files, outp):
    """A1.4: dominant ingredient on sub16 at the three meshes (same / differs / inconclusive) and per-component median
    ratios 4096/1024 and 256/1024 with case-bootstrap 95 % intervals."""
    D = {}
    for f in files:
        j = json.loads(Path(f).read_text())
        D[j['mesh']] = j
    out = dict(members={})
    common = set.intersection(*[set(D[m]['members']) for m in D])
    for mk in sorted(common):
        res = {}
        for mesh in sorted(D):
            per = [d for d in D[mesh]['members'][mk]['per_case'] if d['case'] in SUB16]
            res[mesh] = summarise(per, 6)['dominant']
        doms = list(res.values())
        if any(x.startswith('inconclusive') or x == 'no single dominant' for x in doms):
            lab = 'inconclusive'
        else:
            lab = 'same' if len(set(doms)) == 1 else 'differs'
        ratios = {}
        for x in ING + ('e_P', 'e_dyn_star'):
            for num in (4096, 256):
                if num not in D or 1024 not in D:
                    continue
                a = {d['case']: (d['main'].get(x) if x in ING else d.get(x)) for d in D[num]['members'][mk]['per_case']}
                b = {d['case']: (d['main'].get(x) if x in ING else d.get(x)) for d in D[1024]['members'][mk]['per_case']}
                cs = [c for c in a if c in b and a[c] is not None and b[c] not in (None, 0.)]
                v = np.array([a[c] / b[c] for c in cs])
                if len(v):
                    bt = np.median(v[RNG.integers(0, len(v), (10000, len(v)))], axis=1)
                    ratios[f'{x}|{num}/1024'] = dict(median=float(np.median(v)), ci=[float(np.percentile(bt, 2.5)),
                                                                                     float(np.percentile(bt, 97.5))], n=len(v),
                                                     excluded=len(a) - len(v))
        out['members'][mk] = dict(dominant=res, label=lab, ratios=ratios)
    Path(outp).write_text(json.dumps(out, indent=1) + '\n')
    print(outp)


if __name__ == '__main__':
    main()
