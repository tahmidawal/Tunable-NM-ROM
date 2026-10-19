"""jcp-regimes offline scoring, qualification and labels (DESIGN §6, §7, A1.5–A1.8). CPU only, NumPy, no JAX.

Inputs (pulled locally; nothing large):
  runs/<attempt>/output/qacc_<cfg>/R_<setting>_<arm>.npz   ROM coefficients, in-job certificates, f257 probes (no f257)
  runs/<attempt>/output/qacc_<cfg>/result.json             in-job rows
  store/2d/<regime>/ST/c<j>.npz|json                       this lane's references (finest-level lattice fields + meta)
  E0 store (read-only, refstore) for dev6 / val32
Q1/Q2: jcp-qualify `qscore.qualify` semantics (copied here, adapted to the f257 probe), offline NumPy decode.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import npcheck as NC  # noqa: E402
import rcohorts as RG  # noqa: E402

E0 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
STORE = HERE / 'store'
BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, GAMMA, ETA = 2.5e-4, 2.5e-5, 2.5e-4, .116, 1e-3, 1e-8
FACTOR = 2.0                    # DESIGN §7 rule 3
NBOOT, NPERM = 10000, 10000
MIN_ELIG = 8
UNUSABLE = .10
BASE = ('N0', 'dev6', 'val32')
E0_COHORTS = ('dev6', 'val32')


class IdentityError(RuntimeError):
    pass


def _rs():
    if str(E0) not in sys.path:
        sys.path.append(str(E0))           # appended last: never shadows this lane's modules
    import refstore as RS
    return RS


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def finite(*xs):
    return all(np.isfinite(np.asarray(x, float)).all() for x in xs)


def dmax(xs):
    xs = np.asarray(list(xs), float)
    return float('inf') if xs.size == 0 or not np.isfinite(xs).all() else float(xs.max())


def agree(off, job, rel=1e-9, ab=1e-13):
    return bool(abs(off - job) <= rel * abs(job) + ab)


# ------------------------------------------------------------------------------------------------ references ---

def ref_lane(regime, case):
    """(finest lattice field (6,257,257), meta) of a regime case from this lane's store; asserts identity."""
    stem = STORE / '2d' / regime / 'ST' / f'c{case:03d}'
    meta = json.loads(Path(str(stem) + '.json').read_text())
    z = np.load(str(stem) + '.npz')
    if (meta['cohort'], meta['case'], meta['kind']) != (regime, case, 'ST'):
        raise IdentityError(f'reference identity {meta["cohort"]}|{meta["case"]} != {regime}|{case}')
    D = RG.descriptors_2d(regime)
    if meta['cohort_sha256'] != RG.sha_rows(D):
        raise IdentityError(f'{regime}: cohort hash differs from the generator')
    return np.asarray(z[meta['final'][2]], float), np.asarray(z['t0'], float), meta


def reference(cohort, case):
    """dict(field, t0, n0, eligible, eps_ref, outcome, sigma, meta) for any allowed cohort."""
    if cohort in E0_COHORTS:
        RS = _rs()
        f, meta = RS.load(2, cohort, case, 'ST')
        st = RS.reference_status(2, cohort, case, 'ST')
        jd = meta['judge']
        return dict(field=np.asarray(f, float), n0=float(jd['n0']), eligible=bool(st.get('eligible', meta['outcome'] == 'accepted')),
                    eps_ref=jd.get('eps_ref'), outcome=meta['outcome'], sigma=None, source='E0')
    if cohort not in RG.REGIMES_2D:
        raise IdentityError(f'cohort {cohort} not admissible')
    f, t0, meta = ref_lane(cohort, case)
    jd = meta['judge']
    return dict(field=f, n0=float(np.linalg.norm(t0.ravel())), eligible=meta['outcome'] == 'accepted',
                eps_ref=jd.get('eps_ref'), outcome=meta['outcome'], sigma=sigma_of(meta), source='lane', meta=meta,
                t0=t0)


def sigma_of(meta):
    """DESIGN A1.5: max_{k=1..5} g_k / max|u(t0)| at the finest level of the final triple; plus the middle-level ratio."""
    att = {a['id']: a for a in meta['attempts']}
    fin, mid = att[meta['final'][2]], att[meta['final'][1]]
    if fin.get('grad_max') is None:
        return None
    u0max = meta['desc'].get('a')               # max|u0| over nodes: a for every family (two-bump peak-normalised)
    s3 = max(fin['grad_max'][1:]) / u0max
    s2 = max(mid['grad_max'][1:]) / u0max if mid.get('grad_max') else None
    return dict(sigma=s3, sigma_mid=s2, ratio=(s3 / s2 if s2 else None), level=fin['L'],
                underresolved_mid=bool(s2 and abs(s3 / s2 - 1) > .1))


def error(field, ref):
    """E0 metric: max_{k>=1} ||f - R|| / ||R(t0)|| on the lattice."""
    f = np.asarray(field, float)
    R = ref['field']
    if f.shape != R.shape:
        raise IdentityError(f'shape {f.shape} vs {R.shape}')
    e = [float(np.linalg.norm((f[k] - R[k]).ravel())) / ref['n0'] for k in range(1, 6)]
    return float('inf') if not np.isfinite(e).all() else max(e)


# ------------------------------------------------------------------------------------------- qualification ---

def certified(cert, form):
    c = np.asarray(cert, float)
    return np.isfinite(c) & (c <= (GAMMA if form == 'LSPG' else ETA))


def gcert_ok(R, form, tolf):
    c = np.asarray(R['cert'], float)
    if not c.size:
        return False
    if form == 'LSPG':
        r = np.asarray(R['rec'], float)
        return bool(np.all(np.isfinite(c) & np.isfinite(r) & (np.abs(c - r) <= 1e-7 * np.abs(r) + 1e-12)))
    tr, sc, raw = (np.asarray(R[k], float) for k in ('tolratio', 'scale', 'raw'))
    pos = sc > 0
    ok_pos = np.isfinite(c) & np.isfinite(tr) & (np.abs(c - tr * tolf) <= 1e-6 * np.abs(tr * tolf) + 1e-13)
    ok_zero = np.isfinite(tr) & np.isfinite(raw) & (np.abs(raw - tr * 1e-300) <= 1e-6 * np.abs(tr * 1e-300) + 1e-15)
    return bool(np.all(np.where(pos, ok_pos, ok_zero)))


def q2_label(q2, disagreement=False):
    if disagreement:
        return 'fail (offline disagreement)'
    if not (q2['successor_valid'] and q2['check_valid']):
        return 'unverified (successor invalid)'
    if q2['rho_max'] is None:
        return 'unverified (no rho indicators)'
    if not (q2['rollout'] and q2['rho']):
        return 'fail'
    if not q2['convergence']:
        return 'unverified (finer rule not converged)'
    return 'pass'


def rule_tables(rule, Rp, kx, ky, L):
    if rule.startswith('gauss'):
        X, w = NC.gauss_rule(int(rule[5:]))
    else:
        sys.path.insert(0, str(HERE / 'vendor' / 'quad2d' / 'vendor'))
        import hari_quadrature as HQ
        a, b, k = 1, 1, 2
        n = int(rule[3:])
        while b < n:
            a, b, k = b, a + b, k + 1
        X, w = HQ.fibonacci_lattice(k, seed=0)
        X, w = np.asarray(X, float), np.asarray(w, float)
    return NC.tables(X, w, Rp, kx, ky, L)


def qualify(R, rows, expect, rho_tables=None, subset=None):
    """jcp-qualify qscore.qualify (copied; A1.7), with two adaptations: the decode check uses the saved f257 probe and
    per-time norms (the f257 array stays on the cluster), and `subset` restricts the evaluation to one regime's cases
    (Q1/Q2 per regime, every sampled case). Returns per_case, Q1, Q2, numerically_qualified, disagreements."""
    form, S = expect['form'], int(expect['steps'])
    Rp = int(R['Rp'])
    ids_all = [str(x) for x in R['case_ids']]
    keep = [j for j, c in enumerate(ids_all) if subset is None or c.split('|')[0] in subset]
    out = dict(per_case={}, disagreements=[])
    dec = lambda W: NC.decode257(np.asarray(W)[None], Rp)[0] if np.ndim(W) == 1 else NC.decode257(W, Rp)
    rowby = {f"{r_['cohort']}|{r_['case']}": r_ for r_ in rows}

    def cmp(cid, key, off):
        r_ = rowby.get(cid)
        if r_ is None or r_.get(key) is None:
            out['disagreements'].append((cid, key, 'missing in-job value'))
        elif not agree(off, r_[key]):
            out['disagreements'].append((cid, key, off, r_[key]))

    cert = np.asarray(R['cert'], float)[keep]
    reason = np.asarray(R['reason'])[keep]
    nonc = int(np.sum(~certified(cert, form)))
    Rk = {k: (np.asarray(R[k])[keep] if k in ('cert', 'rec', 'tolratio', 'scale', 'raw') else R[k])
          for k in ('cert', 'rec', 'tolratio', 'scale', 'raw') if k in R}
    out.update(pooled_noncertified=nonc, pooled_steps=int(cert.size), gcert_all=gcert_ok(Rk, form, float(np.asarray(R['tol'])[1])),
               budget_exit_steps=int(np.sum(reason == 0)), certificate=('gamma' if form == 'LSPG' else 'eta'),
               n_cases=len(keep))
    for j in keep:
        cid = ids_all[j]
        n0 = float(R['n0'][j])
        if not (np.isfinite(n0) and n0 > 0):
            raise ValueError('non-positive normalisation')
        fp = dec(R['W'][j])
        probe = fp.reshape(6, -1)[:, R['f257_probe_idx']]
        pj = np.asarray(R['f257_probe'][j])
        rec_err = float(np.max(np.abs(probe - pj)) / max(np.max(np.abs(pj)), 1e-300)) if finite(pj) else float('inf')
        tn_err = float(np.max(np.abs(np.sqrt(np.sum(fp.reshape(6, -1) ** 2, 1)) - R['f257_tnorm'][j]) /
                              np.maximum(R['f257_tnorm'][j], 1e-300))) if finite(R['f257_tnorm'][j]) else float('inf')
        d_t = NC.fdist(fp, dec(R['W_tight'][j]), n0)
        valid = bool(finite(R['W'][j], R['w0'][j], R['states'][j], pj)
                     and not np.any(np.asarray(R['reason'][j]) == 3) and bool(R['valid'][j]))
        out['per_case'][cid] = dict(valid=valid, tight_valid=bool(R['valid_tight'][j] and finite(R['W_tight'][j])),
                                    dist_tight=d_t, decode_rel=rec_err, tnorm_rel=tn_err)
        if valid and not (rec_err <= 1e-12 and tn_err <= 1e-12):
            out['disagreements'].append((cid, 'decode', rec_err, tn_err))
        cmp(cid, 'dist_tight', d_t)
    ix = {cid: j for j, cid in enumerate(ids_all)}
    devc = [str(x) for x in R['dev_case_ids']]
    tt, dev_keep = [], []
    for k, cid in enumerate(devc):
        if cid not in out['per_case']:
            continue
        dev_keep.append(cid)
        j = ix[cid]
        d_tt = NC.fdist(dec(R['W_tight'][j]), dec(R['W_tighter'][k]), float(R['n0'][j]))
        out['per_case'][cid].update(dist_tight_tighter=d_tt, tighter_valid=bool(R['valid_tighter'][k] and finite(R['W_tighter'][k])))
        cmp(cid, 'dist_tight_tighter', d_tt)
        tt.append(d_tt)
    q2case = [str(x) for x in R['q2_case_ids']]
    q2 = None
    q2_here = [c for c in q2case if c in out['per_case']]
    if q2_here:
        if rho_tables is None:
            raise ValueError('Q2 needs the independent rho recomputation')
        dq, valid_qp, conv, valid_qpp = [], [], [], []
        rho_job = np.asarray(R['rho'], float)
        rho_rows = []
        for k, cid in enumerate(q2case):
            if cid not in out['per_case']:
                continue
            j = ix[cid]
            d = NC.fdist(dec(R['W'][j]), dec(R['W_qp'][k]), float(R['n0'][j]))
            dq.append(d)
            rho_rows.append(rho_job[k])
            out['per_case'][cid]['dist_qp'] = d
            cmp(cid, 'dist_qp', d)
            valid_qp.append(bool(R['valid_qp'][k] and finite(R['W_qp'][k])))
            if cid in dev_keep:
                for kk in range(6):
                    off = NC.rho(np.asarray(R['W'][j, kk]), *rho_tables)
                    if not (np.isnan(off) and np.isnan(rho_job[k, kk])) and not agree(off, rho_job[k, kk]):
                        out['disagreements'].append((cid, f'rho[{kk}]', off, float(rho_job[k, kk])))
        if expect.get('qpp'):
            for k, cid in enumerate(devc):
                if cid not in dev_keep or cid not in q2case:
                    continue
                kq, j = q2case.index(cid), ix[cid]
                c_ = NC.fdist(dec(R['W_qp'][kq]), dec(R['W_qpp'][k]), float(R['n0'][j]))
                conv.append(c_)
                out['per_case'][cid]['dist_qp_qpp'] = c_
                cmp(cid, 'dist_qp_qpp', c_)
                valid_qpp.append(bool(R['valid_qpp'][k] and finite(R['W_qpp'][k])))
        rho = np.asarray(rho_rows, float).ravel()
        rho_ok = rho[~np.isnan(rho)]
        q2 = dict(population=len(dq), rollout=bool(dq) and dmax(dq) <= BAR_Q2, worst_dist_qp=dmax(dq),
                  rho_max=dmax(rho_ok) if rho_ok.size else None, rho_trivial=int(np.sum(np.isnan(rho))),
                  rho_infinite=int(np.sum(np.isinf(rho))), rho=bool(rho_ok.size) and dmax(rho_ok) <= BAR_RHO,
                  successor_valid=bool(valid_qp) and all(valid_qp),
                  check_valid=(not expect.get('qpp')) or (bool(valid_qpp) and all(valid_qpp)),
                  convergence=(not expect.get('qpp')) or (bool(conv) and dmax(conv) <= BAR_Q2),
                  worst_dist_qp_qpp=dmax(conv) if conv else None)
    pcs = out['per_case'].values()
    q1 = dict(all_valid=bool(out['per_case']) and all(v['valid'] and v['tight_valid'] for v in pcs),
              certificate=bool(cert.size == len(keep) * S) and nonc <= .01 * cert.size, gcert=out['gcert_all'],
              distance=bool(out['per_case']) and dmax(v['dist_tight'] for v in pcs) <= BAR_Q1,
              tight_tighter=(not dev_keep) or (bool(tt) and dmax(tt) <= BAR_TT and
                                               all(out['per_case'][c]['tighter_valid'] for c in dev_keep)))
    q1['offline_agrees'] = not out['disagreements']
    q1['pass'] = all(q1.values())
    out['Q1'] = q1
    if q2 is None:
        out['Q2'] = dict(label='not run here')
    else:
        q2['label'] = q2_label(q2, bool(out['disagreements']))
        out['Q2'] = q2
    out['numerically_qualified'] = bool(out['Q1']['pass'] and out['Q2'].get('label') == 'pass')
    return out


# --------------------------------------------------------------------------------------------------- labels ---

def boot_ratio(num, den, seed=0, n=NBOOT, paired=False):
    """95 % percentile interval of median(num)/median(den) (independent resampling within each set) or, paired,
    of median(num/den) over pairs. Returns (point, lo, hi, dropped)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    if paired:
        q = num / den
        pt = float(np.median(q))
        reps = np.median(q[rng.integers(0, len(q), (n, len(q)))], axis=1)
        dropped = 0
    else:
        pt = float(np.median(num) / np.median(den)) if np.median(den) > 0 else float('nan')
        a = np.median(num[rng.integers(0, len(num), (n, len(num)))], axis=1)
        b = np.median(den[rng.integers(0, len(den), (n, len(den)))], axis=1)
        ok = np.isfinite(b) & (b > 0)
        dropped = int(np.sum(~ok))
        reps = a[ok] / b[ok]
    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    return pt, lo, hi, dropped


def acc_label(pt, lo, hi, dropped, n_num, n_den, n=NBOOT):
    if n_num < MIN_ELIG or n_den < MIN_ELIG:
        return 'inconclusive (coverage)'
    if dropped > .01 * n or not np.isfinite(pt):
        return 'INCONCLUSIVE'
    if hi < 1.:
        return 'IMPROVES'
    if hi <= FACTOR:
        return 'HOLDS'
    if lo > FACTOR:
        return 'DEGRADES'
    return 'INCONCLUSIVE'


def spearman_perm(x, y, seed=0, n=NPERM):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return dict(rho=None, p=None, n=int(len(x)), excluded=int(np.sum(~ok)))
    rx, ry = rankdata(x), rankdata(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, ry[rng.permutation(len(ry))])[0, 1] for _ in range(n)])
    p = float((np.sum(np.abs(perm) >= abs(r) - 1e-15) + 1) / (n + 1))
    slope = float(np.polyfit(x, np.log10(np.exp(y)) if False else y / np.log(10), 1)[0])
    return dict(rho=r, p=p, n=int(len(x)), excluded=int(np.sum(~ok)), ols_slope_log10e_per_unit=slope,
                grows=bool(r > 0 and p < .05))


def separation(sig, regimes):
    """DESIGN 4.4 / A1.5 separation checks on eligible-case sigma values. sig: {regime: [sigma of eligible cases]}."""
    inn = list(sig.get('N0', [])) + list(sig.get('N0s', []))
    res = {}
    if len(inn) < MIN_ELIG:
        return {g: 'undetermined (in-range coverage)' for g in regimes}
    lo, hi, q90 = min(inn), max(inn), float(np.percentile(inn, 90))
    res['in_range'] = dict(min=lo, max=hi, q90=q90, n=len(inn))
    frac_in = lambda v: float(np.mean([(lo <= x <= hi) for x in v]))
    for g in regimes:
        v = list(sig.get(g, []))
        if len(v) < MIN_ELIG:
            res[g] = dict(verdict='undetermined (coverage)', n=len(v))
            continue
        if g in ('Ap', 'Am', 'B1', 'B2'):
            f = frac_in(v)
            res[g] = dict(n=len(v), frac_in_range=f, passed=bool(f >= .75))
        elif g == 'C1':
            f = float(np.mean([x > q90 for x in v]))
            am = sig.get('Am', [])
            res[g] = dict(n=len(v), frac_above_q90=f, median=float(np.median(v)),
                          median_Am=float(np.median(am)) if am else None,
                          passed=bool(f >= .75 and am and np.median(v) > np.median(am)))
        elif g == 'C2':
            c1 = sig.get('C1', [])
            res[g] = dict(n=len(v), median=float(np.median(v)), median_C1=float(np.median(c1)) if c1 else None,
                          passed=bool(c1 and np.median(v) > np.median(c1)))
    return res


def projection_floor(ref_field, Rp):
    """DESIGN §6 diagnostic: max_k ||R - Pi R|| / ||R(t0)|| with Pi the lattice least-squares projection on the first Rp
    bank functions (npcheck's lattice basis, zero walls)."""
    NC.decode257(np.zeros((1, Rp)), Rp)
    G = NC._G257[Rp]
    R = np.asarray(ref_field, float).reshape(6, -1)
    n0 = float(np.linalg.norm(R[0]))
    C, *_ = np.linalg.lstsq(G, R[1:].T, rcond=None)
    res = R[1:].T - G @ C
    return float(np.max(np.linalg.norm(res, axis=0)) / n0)
