"""jcp-speed3d replication analysis (DESIGN A1.11): the frozen configurations on the historical test cohort 923901
(E0 test32, 32 cases). Replication only: nothing is selected here. CPU only; reads runs/r1/archive/output, the E0 scores
checks/scores-r1.json (score3.py) and checks/FROZEN-SELECTION.json; writes checks/analysis-r1.json including a generated
markdown block that make_report.py embeds verbatim.

Per frozen ROM configuration and path: E-a/E-b (certified, recomputed from rows), E-c (distance to its ADT tight on this
cohort), E0 median / worst. Per comparator: does the frozen winner still tie-or-beat the comparator on median and worst
here, and the replication timing panel's S = 1 / ratio. Per matched FOM: does it still tie-or-beat its ROM here.
Missing evidence gives null / "unavailable", never a pass or a fail.

    python analyze_r1.py
"""
import offline  # noqa: F401
import json
from pathlib import Path

import numpy as np

import analyze as AN
import analyze_pm as PM

HERE = Path(__file__).resolve().parent


def main():
    outp = HERE / 'checks' / 'analysis-r1.json'
    outp.unlink(missing_ok=True)                       # a failed run must not leave an earlier analysis behind
    rp = (HERE / 'runs' / 'r1' / 'archive' / 'output').resolve()
    res = PM.jl(rp / 'result.json')
    sc = PM.jl(HERE / 'checks' / 'scores-r1.json')
    fz = PM.jl(HERE / 'checks' / 'FROZEN-SELECTION.json')
    assert Path(sc['run']).resolve() == rp and sc['job_id'] == res['job_id'], 'scores-r1.json is of another run'
    cfg, R, T = res['config'], res['replication'], res['timing']
    blk = cfg['rep']
    cases = res['cohort']['cases']
    integ = bool(res.get('complete') and res['cohort']['seed'] == 923901 and sorted(cases) == list(range(32))
                 and sc['controls'].get('K4_fires'))
    out = dict(job=res['job_id'], gpu=res['gpu'], commit=res['commit'], integrity_ok=integ, roms={}, comparators={}, foms={})
    S = {}
    for r in blk['roms']:
        for path in r['paths']:
            tag = f"{r['setting']}|{r['rule']}" + ('|mf' if path == 'mf' else '')
            ab = PM.rows_elig(res, tag, r['spec'], cases)
            sp = r['spec'].split('|')
            ak = f"ad|LSPG|{sp[1]}|{float(sp[2]):.8g}|tight"
            ad = PM.rows_elig(res, tag, ak, cases)
            dist = (R.get('dist_to_adt') or {}).get(f"{r['label']}|{path}")
            e_c = PM.ec(dist, [True] * len(cases), len(cases)) if dist is not None else dict(Ec=None, dist_worst=None)
            okev = bool(ab['rows_complete'] and ad['rows_complete'] and e_c.get('dist_worst') is not None)
            el = None if not okev else bool(ab['Ea'] and ab['Eb'] and ad['Ea'] and e_c['Ec'])
            s = PM.scores_of(sc, f"{tag}|{r['spec']}", cases)
            S[(r['label'], path)] = s
            out['roms'][f"{r['label']}|{path}"] = dict(role=r.get('role'), key=f"{r['setting']}|{r['rule']}|{r['spec']}", path=path,
                                                      eligible=el, Eb_frac=ab['Eb_frac'], dist_worst=e_c.get('dist_worst'),
                                                      scores_ok=s['ok'], median=s['median'], worst=s['worst'])
    for c, v in fz['comparators'].items():
        w = v['winner_label']
        sc_, sw = S.get((c, 'mesh')), S.get((w, 'mesh')) if w else None
        tob = None
        if sc_ and sw and sc_['ok'] and sw['ok']:
            ea, ec_, eta, ids = AN.common(sw, sc_)
            tob = bool(all(AN.ties_or_beats(ea, ec_, eta)))
        p = PM.panel(T.get(f'rep-confirm|{c}|{w}'), w, A=c, cases=cases[:blk['confirm_cases']], reps=blk['confirm_reps']) if w and w != c else None
        out['comparators'][c] = dict(winner=w, winner_ties_or_beats=tob,
                                     S_replication=None if not p or not p['valid'] else 1. / p['ratio'],
                                     S_validation=v['H2'].get('S_confirmed'), panel_reason=None if not p else p.get('reason'))
    for l_, fam in fz['fom_matches'].items():
        s = S.get((l_, 'mesh'))
        for f_, m in fam.items():
            fk = m.get('matched')
            if not fk:
                continue
            cs = (R.get('foms') or {}).get(fk)
            st = None
            if cs and s and s['ok'] and all(str(j) in cs or j in cs for j in cases):
                cc = {int(k): x for k, x in cs.items()}
                if all(cc[j].get('converged') and np.isfinite(cc[j].get('e', np.nan)) for j in cases):
                    rk = [j for j in cases if cc[j].get('eligible')]
                    ef, er, eta, ids = AN.common(dict(e=[cc[j]['e'] for j in rk], eta=[cc[j]['eps_ref'] for j in rk], ranked_ids=rk), s)
                    st = bool(all(AN.ties_or_beats(ef, er, eta)))
            out['foms'][f'{l_}|{f_}'] = dict(fom=fk, still_ties_or_beats=st)
    if not integ:
        PM.suppress(out, 'r1 integrity failed')
    f = lambda x: '–' if x is None else (('yes' if x else 'no') if isinstance(x, bool) else f'{x:#.3g}'.rstrip('.'))
    md = [f"Job {out['job']} ({out['gpu']}), integrity {f(integ)}; replication only (nothing selected on this cohort).", '',
          '| configuration | path | role | eligible | distance to ADT | median | worst |', '|---|---|---|---|---|---|---|']
    for k, v in out['roms'].items():
        md.append(f"| {k.split('|')[0]} | {v['path']} | {v['role']} | {f(v['eligible'])} | {f(v['dist_worst'])} | {f(v['median'])} | {f(v['worst'])} |")
    md += ['', '| comparator | winner | winner ties-or-beats here | S replication | S validation |', '|---|---|---|---|---|']
    for c, v in out['comparators'].items():
        md.append(f"| {c} | {v['winner']} | {f(v['winner_ties_or_beats'])} | {f(v['S_replication'])} | {f(v['S_validation'])} |")
    md += ['', '| ROM, family | matched FOM | still ties-or-beats the ROM here |', '|---|---|---|']
    for k, v in out['foms'].items():
        md.append(f"| {k.replace('|', ', ')} | {v['fom'].replace('|', ' ')} | {f(v['still_ties_or_beats'])} |")
    out['markdown'] = '\n'.join(md)
    tmp = outp.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, indent=1, default=float) + '\n')
    tmp.replace(outp)
    print(out['markdown'])


if __name__ == '__main__':
    main()
