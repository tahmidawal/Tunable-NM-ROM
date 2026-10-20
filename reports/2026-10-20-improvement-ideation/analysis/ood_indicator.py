"""Out-of-regime indicator proxy on saved jcp-regimes data (CPU, JSON + the lane's deterministic cohort generator).

Per frozen ROM configuration (9: 8 LSPG + the GAL-CN diagnostic) and per case (dev6, val32 and the fresh regimes N0,
N0s, A+, A-, C1, C2, B1, B2; 158 rows per configuration), joins the per-case E0-style error e
(checks/analysis.json -> configs.<cfg>.per_case) with indicators:
  online (from runs/rom1024/output/qacc_rom1024/result.json -> rows):
    dist_qp     rollout distance to the successor quadrature rule (rule doubling; costs a second rollout),
    rho_max     largest relative tested-advection change under the successor rule on the run's own states,
    dist_tight  distance to a tighter solve (costs a second rollout),  it_sum  LM iterations (free);
  parameter-only controls, known before the query (rcohorts.desc_2d; dev6/val32 descriptors from jcp-combined):
    s           nominal steepness a / nu,
    d_box       log-distance of (nu, s) outside the training box (d_nu + d_s; 0 inside).
Cases with an ineligible reference (eligible: false; unresolved truth) are excluded. Invalid ROM runs are kept: they count as "bad" and as alarmed (a deployed system must fall back on them).
Targets: (A) e > 2 x the configuration's in-range median (dev6 + val32 + N0); (B) e > 1 %.
Threshold for detection / false alarm = the in-range 95th percentile of the indicator (in-sample: chosen and scored
on the same in-range cases; the four online indicators were also compared after the fact). Writes ood_indicator.json.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
LANE = ROOT / 'worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes'
sys.path.insert(0, str(LANE))
import rcohorts as RG  # noqa: E402

A = json.loads((LANE / 'checks/analysis.json').read_text())
R = json.loads((LANE / 'runs/rom1024/output/qacc_rom1024/result.json').read_text())['rows']
coh = json.loads((ROOT / 'worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/runs/local2dfb/result.json').read_text())['cohorts']
desc = {}
for c in ('dev6', 'val32'):
    for i, p in enumerate(coh[c]['physical']):
        desc[f'{c}|{i}'] = RG.desc_2d('N0', np.array(p))      # single-bump layout (cx, cy, w, a, nu)
for c in ('N0', 'N0s', 'Ap', 'Am', 'C1', 'C2', 'B1', 'B2'):
    for i, row in enumerate(RG.descriptors_2d(c)):
        desc[f'{c}|{i}'] = RG.desc_2d(c, row)
IN = ('dev6', 'val32', 'N0')
ONLINE = ('dist_qp', 'rho_max', 'dist_tight', 'it_sum')
PARAM = ('s', 'd_box')


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    rk = rankdata(score)
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


out = dict(sources=[str((LANE / 'checks/analysis.json').relative_to(ROOT)),
                    str((LANE / 'runs/rom1024/output/qacc_rom1024/result.json').relative_to(ROOT)),
                    str((LANE / 'rcohorts.py').relative_to(ROOT))], configs={})
for cfg, v in A['configs'].items():
    st, arm = cfg.split('|')
    pc = v['per_case']
    rows = [x for x in R if x['setting'] == st and x['arm'] == arm]
    # cases whose reference is ineligible (unresolved truth label) are excluded from scoring and calibration
    excluded = sorted({f"{x['cohort']}|{x['case']}" for x in rows if pc.get(f"{x['cohort']}|{x['case']}", {}).get('eligible') is False})
    rows = [x for x in rows if f"{x['cohort']}|{x['case']}" not in excluded]
    keys = [f"{x['cohort']}|{x['case']}" for x in rows]
    e = np.array([pc.get(k, {}).get('e') if (x['valid'] and pc.get(k, {}).get('e') is not None) else np.inf
                  for k, x in zip(keys, rows)], float)
    invalid = ~np.isfinite(e)
    cohort = np.array([x['cohort'] for x in rows])
    inr = np.isin(cohort, IN) & ~invalid
    med_in = float(np.median(e[inr]))
    ind = {k: np.array([float(x[k]) for x in rows]) for k in ONLINE}
    ind['s'] = np.array([desc[k]['s'] for k in keys])
    ind['d_box'] = np.array([desc[k]['d_nu'] + desc[k]['d_s'] for k in keys])
    res = dict(n=len(rows), excluded_ineligible_reference=excluded, n_invalid=int(invalid.sum()), invalid_cases=[k for k, b in zip(keys, invalid) if b],
               n_in_range=int(inr.sum()), median_in_range=med_in, targets={}, indicators={})
    res['per_regime_dist_qp_alarm_rate'] = {}
    for target, y in (('A_e_gt_2x_inrange_median', e > 2 * med_in), ('B_e_gt_1pct', e > 0.01)):
        res['targets'][target] = dict(n_bad=int(y.sum()), n_good=int((~y).sum()))
        for name, s in ind.items():
            s_eff = np.where(invalid, np.inf, s)
            thr = float(np.quantile(s[inr], 0.95))
            alarm = (s_eff > thr) | invalid
            d = res['indicators'].setdefault(name, dict(kind='online' if name in ONLINE else 'parameter-only',
                                                        spearman_with_e_valid=float(spearmanr(s[~invalid], e[~invalid])[0])))
            d[target] = dict(auc=auc(np.where(np.isfinite(s_eff), s_eff, np.nanmax(s) * 10 + 1), y),
                             threshold_inrange_q95=thr,
                             detection=float(alarm[y].mean()) if y.any() else None,
                             false_alarm=float(alarm[~y].mean()) if (~y).any() else None)
    thr_q = res['indicators']['dist_qp']['A_e_gt_2x_inrange_median']['threshold_inrange_q95']
    for c in sorted(set(cohort)):
        sel = cohort == c
        res['per_regime_dist_qp_alarm_rate'][c] = float((((ind['dist_qp'] > thr_q) | invalid)[sel]).mean())
    out['configs'][cfg] = res
Path(__file__).with_name('ood_indicator.json').write_text(json.dumps(out, indent=1))
for t in ('A_e_gt_2x_inrange_median', 'B_e_gt_1pct'):
    print(t)
    for name in ONLINE + PARAM:
        a_ = [c['indicators'][name][t]['auc'] for c in out['configs'].values()]
        d_ = [c['indicators'][name][t]['detection'] for c in out['configs'].values()]
        f_ = [c['indicators'][name][t]['false_alarm'] for c in out['configs'].values()]
        print(f'  {name:10s} AUC {min(a_):.2f}-{max(a_):.2f}  det {min(d_):.2f}-{max(d_):.2f}  fa {min(f_):.2f}-{max(f_):.2f}')
