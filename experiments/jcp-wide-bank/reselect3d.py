"""jcp-wide-bank: offline re-selection of a w3d.py job under DESIGN A7 (pooled 3D eligibility) and A0-2 (metric-only
control discrimination), from the job's persisted per-case records. Uses the SAME selection functions as w3d.py
(arm_eligible, select_mstar). Writes <out>: a copy of result.json whose per-setting 'gates', 'controls', 'selection' and
'deployed' are replaced by the amended values, with the as-run values kept under 'as_run'.

    python reselect3d.py runs/j2/archive/output/result.json runs/j2/archive/output/result_A7.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w3d import arm_eligible, select_mstar  # noqa: E402


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    rep = json.loads(src.read_text())
    cfg = rep['config']
    nsf, rho_bar = float(cfg['nonstat_fraction']), float(cfg['rho_bar'])
    taus = dict(primary=float(cfg['tau']), secondary=float(cfg['tau_secondary']))
    out = copy.deepcopy(rep)
    out['amendment'] = 'A7 (pooled 3D eligibility) + A0-2 (metric-only controls), re-selected offline by reselect3d.py'
    for n, m_ in out['meshes'].items():
        for bn, b in m_['banks'].items():
            for key, S in b['settings'].items():
                S['as_run'] = dict(gates=S['gates'], controls=S.get('controls'), selection=S['selection'],
                                   deployed=S.get('deployed'))
                recs = {nm: a['cases'] for nm, a in S['arms'].items() if 'cases' in a}
                for nm, a in S['arms'].items():
                    if 'cases' in a:
                        a['arm_eligible_pooled'] = arm_eligible(a['cases'], nsf)
                crecs = S['certification']
                conv_ok = arm_eligible(recs['conv'], nsf) and arm_eligible(crecs['conv'], nsf)
                chk_ok = arm_eligible(recs['check'], nsf) and arm_eligible(crecs['check'], nsf)
                chk_d = S['gates']['converged']['check_worst_distance']
                gates = dict(converged=dict(conv_all_eligible=conv_ok, check_all_eligible=chk_ok, check_worst_distance=chk_d,
                                            bar=cfg['conv_bar'], passed=bool(conv_ok and chk_ok and chk_d <= cfg['conv_bar'])),
                             target=S['gates']['target'])
                rho_max = {nm: v['worst'] for nm, v in S['rho']['rules'].items()}
                arm_list = [dict(name=nm, family=a['family'], m=a['m'], control=a['control']) for nm, a in S['arms'].items()
                            if a['family'] not in ('ref', 'tensor') and a.get('m') is not None]
                ctrl = {}
                for nm, a in S['arms'].items():
                    if a['control']:
                        dm = max(r['dist_conv'] for r in recs[nm])
                        fd = {t: bool(not (np.isfinite(dm) and dm <= tv)) for t, tv in taus.items()}
                        fr = bool(not (np.isfinite(rho_max[nm]) and rho_max[nm] <= rho_bar))
                        ctrl[nm] = dict(worst_distance=dm, rho_max=rho_max[nm], all_eligible=arm_eligible(recs[nm], nsf),
                                        fails_distance=fd, fails_rho=fr,
                                        would_be_selected={t: bool(not fd[t] and not fr) for t in taus})
                disc = {t: bool(ctrl and not any(v['would_be_selected'][t] for v in ctrl.values())) for t in taus}
                valid = bool(gates['converged']['passed'] and gates['target']['passed'])
                sel = {}
                for t, tv in taus.items():
                    for fam in cfg['families']:
                        nm_, mm = select_mstar(recs, rho_max, arm_list, fam, tv, rho_bar, nonstat_frac=nsf)
                        nd, md = select_mstar(recs, rho_max, arm_list, fam, tv, rho_bar, use_rho=False, nonstat_frac=nsf)
                        ok = valid and disc[t]
                        sel[f'{t}|{fam}'] = dict(tau=tv, family=fam, gates_passed=ok, available=bool(ok and nm_ is not None),
                                                 reason=(None if ok and nm_ is not None else
                                                         ('gates' if not valid else 'controls' if not disc[t]
                                                          else 'no ladder member qualifies')),
                                                 arm=nm_ if ok else None, m=mm if ok else None, arm_raw=nm_, m_raw=mm,
                                                 arm_d=nd, m_d=md)
                for fam in cfg['families']:
                    nr, mr = select_mstar(recs, rho_max, arm_list, fam, 0., rho_bar, use_d=False, nonstat_frac=nsf)
                    sel[f'rho_only|{fam}'] = dict(arm_rho=nr, m_rho=mr)
                tim = S['timing']['subjects']
                tvalid = bool(S['timing']['gates']['drift_pass'] and S['timing']['gates']['deterministic'])
                dep = [((tim[e['arm']]['median_ms'] if tvalid else e['m']), e['family'], e['arm']) for k, e in sel.items()
                       if k.startswith('primary|') and e.get('arm') is not None]
                if dep:
                    _, fam, arm = min(dep)
                    S['deployed'] = dict(family=fam, arm=arm, m=S['arms'][arm]['m'], median_ms=tim[arm]['median_ms'],
                                         timing_valid=tvalid, chosen_by=('timed_median (setting panel)' if tvalid else
                                                                         'smaller_m (K-time failed; diagnostic only)'),
                                         final_panel=False)
                else:
                    S['deployed'] = None
                S['gates'], S['controls'], S['selection'] = gates, ctrl, dict(valid=valid, discriminating=disc, entries=sel)
    for n, m_ in out['meshes'].items():
        ft = m_.get('final_timing')
        timed = set(ft['subjects']) if ft else set()
        new = [f"{bn}|{k}|{S['deployed']['arm']}" for bn, b in m_['banks'].items() for k, S in b['settings'].items()
               if S.get('deployed')]
        m_['final_panel_coverage'] = dict(job_final_panel_ran=bool(ft), amended_deployed=new,
                                          covered=[k for k in new if k in timed], missing=[k for k in new if k not in timed])
    dst.write_text(json.dumps(out, indent=1))
    print('written', dst)


if __name__ == '__main__':
    main()
