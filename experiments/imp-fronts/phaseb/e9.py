"""imp-fronts E9: out-of-regime indicator s = a / nu (DESIGN B7 as restated by Amendments 1-2). CPU, NumPy / SciPy.

    python phaseb/e9.py freeze                       -> phaseb/e9_frozen.json (dev data only; before any fresh case is scored)
    python phaseb/e9.py score <K1-policy.json> [<timing.json>] [<out.json>]

Freeze (dev data only):
  tau_s    = numpy.percentile(s, 95) (linear) over the in-range cases dev6 + val32 + N0 (all 54 descriptors; dev6 / val32
             from the jcp-regimes copy of the E0 cohort generator (cohorts.descriptors_2d), N0 from rcohorts), cross-checked
             against the E0 store descriptors (fcommon.load_ref meta 'desc').
  target A = e > 2 x K0's dev in-range median of R384|LSPG_CNR_1 (jcp-regimes checks/analysis.json; eligible, valid,
             finite cases of dev6 + val32 + N0), one frozen absolute threshold for every bank (Amendment 1 B7).
  target B = e > 1 % (0.01; the error metric is a fraction).
Score (once, at the end; K1's terminal policy errors on the fresh TEST cases, Amendment 2 B7; arms descriptive):
  alarm = s > tau_s or invalid run; invalid runs count as bad and alarmed. Ineligible references are excluded and counted.
  AUC (score s; invalid -> +inf), detection = P(alarm | bad) with its one-sided 95 % Clopper-Pearson LOWER bound,
  false alarm = P(alarm | good) with its one-sided 95 % Clopper-Pearson UPPER bound. Fewer than 30 good or 20 bad cases
  under a target -> "inconclusive (F12 minimum 30 good / 20 bad not met)", never a pass (necessarily so with 28 cases).
  Routing reported both ways: ROM-then-fallback (every query runs the ROM; flagged ones also the FOM) and direct-to-FOM
  (flagged queries skip the ROM; s is known before solving). Cost only if a paired same-GPU timing file is given,
  else "unmeasured" (Amendment 2 B7). Flagged fractions overall and per cohort.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, rankdata

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
WT = LANE.parents[1]
REG_WT = WT / 'experiments' / 'jcp-regimes'                       # code (generators), read-only
REG_DATA = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes')
ANALYSIS = REG_DATA / 'checks' / 'analysis.json'
FROZEN = HERE / 'e9_frozen.json'
CFG = 'R384|LSPG_CNR_1'
IN_RANGE = ('dev6', 'val32', 'N0')
Q = 95
TARGET_B = 0.01
MIN_GOOD, MIN_BAD = 30, 20
ALPHA = 0.05


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _content_sha(d):
    return hashlib.sha256(json.dumps({k: v for k, v in d.items() if k != 'content_sha256'}, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def in_range_s():
    for p in (REG_WT, REG_WT / '..' / 'mr-burgers2d'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import cohorts as CH
    import rcohorts as RG
    sys.path.insert(0, str(LANE))
    import fcommon as F
    s = {}
    for coh in ('dev6', 'val32'):
        D = CH.descriptors_2d(coh)
        for j, r_ in enumerate(D):
            e0 = F.load_ref(coh, j)[1]['desc']                      # cross-check against the E0 store
            assert all(abs(e0[k] - float(v)) <= 1e-12 * abs(float(v)) for k, v in zip(('cx', 'cy', 'w', 'a', 'nu'), r_)), \
                (coh, j)
            s[f'{coh}|{j}'] = float(r_[3] / r_[4])
    for j, r_ in enumerate(RG.descriptors_2d('N0')):
        s[f'N0|{j}'] = float(RG.desc_2d('N0', r_)['s'])
    return s


def freeze():
    s = in_range_s()
    A = json.loads(ANALYSIS.read_text())
    pc = A['configs'][CFG]['per_case']
    vals = np.array(list(s.values()))
    tau_s = float(np.percentile(vals, Q))                            # numpy default method = 'linear'
    ok = {k: v for k, v in pc.items() if k.split('|')[0] in IN_RANGE and v.get('eligible') and v.get('valid')
          and v.get('e') is not None and np.isfinite(v['e'])}
    med = float(np.median([v['e'] for v in ok.values()]))
    elig_valid_s = np.array([s[k] for k in ok])
    out = dict(
        what='E9 frozen threshold and targets (DESIGN B7, Amendments 1-2), from dev data only; no fresh case scored',
        frozen_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        indicator='s = a / nu', percentile=Q, percentile_method='numpy.percentile linear (default)',
        population=list(IN_RANGE), n_population=len(vals), tau_s=tau_s,
        s_population=s,
        target_A=dict(definition=f'e > 2 x K0 dev in-range median of {CFG}', median_in_range=med, threshold=2 * med,
                      n_cases=len(ok), excluded=sorted(set(k for k in pc if k.split('|')[0] in IN_RANGE) - set(ok))),
        target_B=dict(definition='e > 1 %', threshold=TARGET_B),
        descriptive_alternative=dict(note='ideation ood_indicator convention: s over eligible valid in-range cases only '
                                          '(not the frozen threshold)',
                                     tau_s=float(np.percentile(elig_valid_s, Q)), n=len(elig_valid_s)),
        sources=dict(analysis=str(ANALYSIS), analysis_sha256=sha(ANALYSIS),
                     cohorts_py_sha256=sha(REG_WT / 'cohorts.py'), rcohorts_py_sha256=sha(REG_WT / 'rcohorts.py')),
        scoring=dict(bank='K1 terminal policy errors (fresh test cases)', min_good=MIN_GOOD, min_bad=MIN_BAD,
                     alarm='s > tau_s or invalid', alpha_one_sided=ALPHA))
    out['content_sha256'] = _content_sha(out)
    if FROZEN.exists():
        old = json.loads(FROZEN.read_text())
        keep = lambda d: {k: v for k, v in d.items() if k not in ('frozen_utc', 'content_sha256')}
        if keep(old) != keep(out):
            raise SystemExit('e9_frozen.json exists with different content: refusing to overwrite')
        print('e9_frozen.json already frozen (identical content)')
        return old
    FROZEN.write_text(json.dumps(out, indent=1) + '\n')
    return out


def load_frozen():
    f = json.loads(FROZEN.read_text())
    if _content_sha(f) != f['content_sha256']:
        raise SystemExit('e9_frozen.json was modified after freezing')
    return f


def cp_lower(k, n, a=ALPHA):
    return 0. if k == 0 else float(beta.ppf(a, k, n - k + 1))


def cp_upper(k, n, a=ALPHA):
    return 1. if k == n else float(beta.ppf(1 - a, k + 1, n - k))


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    rk = rankdata(score)
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def score(policy, timing=None, frozen=None, fresh=None):
    """policy: {'<cohort>|<j>': {e, valid, eligible}} for the fresh TEST cases (K1 terminal policy runs)."""
    frozen = frozen or load_frozen()
    if fresh is None:
        sys.path.insert(0, str(HERE))
        import fcohorts as FC
        fresh = FC.load()
    srow = {f"{c}|{r['j']}": r['s'] for c, rows in fresh['cohorts'].items() for r in rows if r['split'] == 'test'}
    unknown = set(policy) - set(srow)
    if unknown:
        raise ValueError(f'not fresh test cases: {sorted(unknown)}')
    keys = [k for k in srow if k in policy and policy[k].get('eligible')]
    out = dict(frozen_sha256=frozen['content_sha256'], tau_s=frozen['tau_s'], n_test=len(srow),
               n_missing=len([k for k in srow if k not in policy]),
               n_ineligible=len([k for k in srow if k in policy and not policy[k].get('eligible')]), n=len(keys),
               targets={})
    if not keys:
        out['status'] = 'no eligible scored cases'
        return out
    s = np.array([srow[k] for k in keys])
    inv = np.array([not (policy[k].get('valid') and policy[k].get('e') is not None and np.isfinite(policy[k]['e']))
                    for k in keys])
    e = np.array([np.inf if b else float(policy[k]['e']) for k, b in zip(keys, inv)])
    alarm = (s > frozen['tau_s']) | inv
    sc = np.where(inv, np.inf, s)
    sc = np.where(np.isfinite(sc), sc, np.max(s) * 10 + 1)
    for name, thr in (('A', frozen['target_A']['threshold']), ('B', frozen['target_B']['threshold'])):
        y = e > thr
        nb, ng = int(y.sum()), int((~y).sum())
        kd, kf = int(alarm[y].sum()), int(alarm[~y].sum())
        out['targets'][name] = dict(
            threshold=thr, n_bad=nb, n_good=ng, auc=auc(sc, y),
            detection=kd / nb if nb else None, detection_cp95_lower=cp_lower(kd, nb) if nb else None,
            false_alarm=kf / ng if ng else None, false_alarm_cp95_upper=cp_upper(kf, ng) if ng else None,
            label=('inconclusive (F12 minimum 30 good / 20 bad not met)' if (ng < MIN_GOOD or nb < MIN_BAD)
                   else 'evaluable (descriptive)'))
    coh = np.array([k.split('|')[0] for k in keys])
    frac = float(alarm.mean())
    out['flagged_fraction'] = frac
    out['flagged_by_cohort'] = {c: float(alarm[coh == c].mean()) for c in sorted(set(coh))}
    out['n_invalid'] = int(inv.sum())
    route = dict(rom_then_fallback='every query runs the ROM; flagged queries also run the accuracy-matched FOM',
                 direct_to_fom='flagged queries go straight to the FOM (s is known before solving)')
    if timing:
        tr, tf, gpu = float(timing['rom_seconds']), float(timing['fom_seconds']), timing['gpu']
        route['rom_then_fallback_seconds_per_query'] = frac * tf + tr
        route['direct_to_fom_seconds_per_query'] = frac * tf + (1 - frac) * tr
        route['gpu'] = gpu
    else:
        route['cost'] = 'unmeasured (no paired same-GPU timing job)'
    out['routing'] = route
    out['note'] = 's is a known input on this family: the indicator is a safety feature, not a speed one'
    return out


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('freeze', 'score'):
        raise SystemExit(__doc__)
    if sys.argv[1] == 'freeze':
        f = freeze()
        print(f"E9 FROZEN tau_s = {f['tau_s']!r} (95th pct linear of s over {f['n_population']} in-range cases)")
        print(f"target A threshold = {f['target_A']['threshold']!r} (2 x median {f['target_A']['median_in_range']!r},"
              f" n = {f['target_A']['n_cases']}); target B = {f['target_B']['threshold']}")
        print(f"descriptive alternative (eligible valid only): {f['descriptive_alternative']}")
        return
    pol = json.loads(Path(sys.argv[2]).read_text())
    tim = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 and sys.argv[3] != '-' else None
    out = score(pol, tim)
    s = json.dumps(out, indent=1)
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(s + '\n')
    print(s)


if __name__ == '__main__':
    main()
