"""jcp-robust E1 decision rules (DESIGN 2.3 as amended by A1.5, A2.2, A2.3, A3.2-A3.4, A4.2). Pure functions, no I/O.

Every predicate is evaluated on explicit per-case states; nothing passes through an empty reduction.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta

BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, BAR_TC, GAMMA, ETA = 2.5e-4, 2.5e-5, 2.5e-4, .116, 2.5e-5, 1e-3, 1e-8
NDRAW = 16


def cp_upper_one_sided(k, n, conf=.95):
    """One-sided Clopper-Pearson upper bound on a failure probability after k failures in n trials (A1.5)."""
    if n <= 0:
        return float('nan')
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def h_e1a(qualified, expected=NDRAW):
    """qualified: per draw True / False / None (None = unavailable draw). A2.3: recorded failures are False. The
    registered population (16 draws) is required; any other count is 'unavailable (population)'."""
    if qualified is None or len(qualified) != expected:
        return dict(label='unavailable (population)', failures=None, n=len(qualified or []))
    if any(q is None for q in qualified):
        return dict(label='unavailable', failures=None, n=len(qualified))
    k = sum(1 for q in qualified if not q)
    n = len(qualified)
    lab = 'robust' if k == 0 else ('inconclusive' if k == 1 else 'fragile')
    fresh = qualified[1:]
    kf = sum(1 for q in fresh if not q)
    return dict(label=lab, failures=k, n=n, cp_upper_all=cp_upper_one_sided(k, n), cp_upper_fresh=cp_upper_one_sided(kf, len(fresh)),
                failed_draws=[i for i, q in enumerate(qualified) if not q])


def num(x):
    """JSON evidence -> float: None (serialised non-finite) -> +inf; never arithmetic on a missing record."""
    return math.inf if x is None else float(x)


def rank_interval(v, delta):
    """A2.2: v = values per draw (index 0 = deployed; +inf = failed), lower is better. Returns (k_lo, k_hi, label)."""
    v = [float(x) for x in v]
    n = len(v)
    if n < 2 or any(math.isnan(x) for x in v):
        return None, None, 'unavailable'
    v0 = v[0]
    lo = 1 + sum(1 for s in range(1, n) if v[s] < v0 - delta)      # IEEE: inf compares correctly on both sides
    hi = n - sum(1 for s in range(1, n) if v[s] > v0 + delta)
    if lo == 1 and hi == n:
        lab = 'insensitive'
    elif hi <= 2:
        lab = 'favourable'
    elif lo >= n - 1:
        lab = 'unfavourable'
    elif lo >= 3 and hi <= n - 2:
        lab = 'typical'
    else:
        lab = 'indeterminate'
    return lo, hi, lab


def h_e1b(stats):
    """stats: dict name -> label (from rank_interval). Configuration label of A2.2."""
    labs = list(stats.values())
    if not labs or any(l == 'unavailable' for l in labs):
        return 'unavailable'
    rest = [l for l in labs if l != 'insensitive']
    if not rest:
        return 'insensitive; typicality not assessed'
    fav, unf = 'favourable' in rest, 'unfavourable' in rest
    if fav and not unf:
        return 'atypically favourable'
    if unf and not fav:
        return 'atypically unfavourable'
    if fav and unf:
        return 'mixed'
    if all(l == 'typical' for l in rest):
        return 'typical'
    return 'indeterminate'


def h_e1c(e, e0, failed, eps, econv, eligible):
    """A3.2/A3.3/A4.2. Per eligible case i: within margin iff no draw failed on i and max_s |e_si - e_0i| <= margin_i,
    margin_i = max(2 eps_i, 0.1 e_conv_i) (e_conv unavailable on i -> 2 eps_i). e: (draws, cases); failed: (draws, cases)."""
    e, failed = np.asarray(e, float), np.asarray(failed, bool)
    idx = [i for i in range(e.shape[1]) if eligible[i]]
    if not idx:
        return dict(label='unavailable', n=0)
    within = []
    for i in idx:
        if failed[:, i].any():
            within.append(False)
            continue
        m = 2 * eps[i]
        if econv[i] is not None and np.isfinite(econv[i]):
            m = max(m, .1 * econv[i])
        col = e[:, i]
        within.append(bool(np.isfinite(col).all() and np.max(np.abs(col - col[0])) <= m))
    frac = sum(within) / len(within)
    lab = 'insensitive' if frac >= .9 else ('sensitive' if frac < .75 else 'inconclusive')
    return dict(label=lab, fraction_within=frac, n=len(within), outside=[idx[k] for k, w in enumerate(within) if not w])


def pooled_nonstationary(cert_rows, form):
    """Pooled non-stationary fraction over every step of every case (V4-2): LSPG gamma > 1e-3 or non-finite; GAL eta > 1e-8."""
    bar = GAMMA if form == 'LSPG' else ETA
    tot = bad = 0
    for c in cert_rows:
        c = np.asarray(c, float)
        tot += c.size
        bad += int(np.sum(~(np.isfinite(c) & (c <= bar))))
    return (bad / tot) if tot else float('nan')


def draw_qualified(flags, d, target_ok, n_cases, n_tight):
    """A1.2 + A3.2 + A3.3 + A4.2 for one draw. flags: per-case validity lists (valid, valid_tight, valid_succ: n_cases;
    valid_tighter: n_tight); d: per-case distances / indicator maxima (tight, succ, rho_succ, target, rho_tgt: n_cases;
    tt: n_tight) and 'nonst_frac'. Values may be None (serialised non-finite) -> treated as +inf (a failure of the bar).
    Wrong lengths or a missing list -> 'missing' (unavailable). target_ok: True / False (a target run failed on some
    case) / None (target not certified). Returns (qualified_full, qualified_own_family, status)."""
    need = dict(valid=n_cases, valid_tight=n_cases, valid_succ=n_cases, valid_tighter=n_tight)
    needd = dict(tight=n_cases, succ=n_cases, rho_succ=n_cases, target=n_cases, rho_tgt=n_cases, tt=n_tight)
    if n_cases <= 0 or n_tight <= 0:
        return None, None, 'missing'
    for k, n in need.items():
        if flags.get(k) is None or len(flags[k]) != n:
            return None, None, 'missing'
    for k, n in needd.items():
        if d.get(k) is None or len(d[k]) != n:
            return None, None, 'missing'
    if d.get('nonst_frac') is None or not np.isfinite(d['nonst_frac']):
        return None, None, 'missing'
    for k, st in (('valid', 'failed-prod'), ('valid_tight', 'failed-tight'), ('valid_tighter', 'failed-tighter'),
                  ('valid_succ', 'failed-succ')):
        if not all(bool(x) for x in flags[k]):
            return False, False, st
    mx = lambda xs: max(num(x) for x in xs)
    own = bool(d['nonst_frac'] <= .01 and mx(d['tight']) <= BAR_Q1 and mx(d['tt']) <= BAR_TT and mx(d['succ']) <= BAR_Q2
               and mx(d['rho_succ']) <= BAR_RHO)
    if target_ok is None:
        return None, own, 'target not certified'
    if target_ok is False:
        return None, own, 'failed-target'
    cross = bool(mx(d['target']) <= BAR_Q2 and mx(d['rho_tgt']) <= BAR_RHO)
    return bool(own and cross), own, 'ok'


def fixture_fired(cpu_states, job_rho, n_states=6):
    """A1.3 + A4.2: admitted (CPU: finite, target norm > 0, rho >= 1 at every state) and in-job rho >= 0.116 at >= 1 state
    and agreement within 1 % relative at every state."""
    if not cpu_states or not job_rho or len(cpu_states) != n_states or len(job_rho) != n_states:
        return dict(admitted=False, fired=False, reason='missing')
    admitted = all(s['ok'] for s in cpu_states)
    if not admitted:
        return dict(admitted=False, fired=False, reason='fixture not admitted')
    job_rho = [num(j) for j in job_rho]
    agree = all(np.isfinite(j) and abs(j - s['rho_tgt']) <= .01 * abs(s['rho_tgt']) for s, j in zip(cpu_states, job_rho))
    flagged = any((not np.isfinite(j)) or j >= BAR_RHO for j in job_rho)
    return dict(admitted=True, fired=bool(agree and flagged), agree=agree, flagged=flagged)
