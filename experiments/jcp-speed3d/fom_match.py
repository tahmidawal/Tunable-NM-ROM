"""jcp-speed3d FOM matcher (DESIGN H4.3, A1.9, A1.10 K10, A4.5): for each ROM configuration and FOM family, the cheapest
calibrated FOM configuration (screening cost = median warm time of the f1 job, same job, same GPU) whose E0 median AND
worst tie-or-beat the ROM's (Step 2 rule, analyze.ties_or_beats). A candidate is admissible only if it was calibrated
(tolerance qualified), converged with complete evidence on every one of the 64 validation cases, has finite scores on the
same ranked cases as the ROM, and complete hash-checked warm timings. Every match is an UPPER BOUND relative to an
unrestricted FOM family (finite ladder); a match at the family's cheapest admissible member is flagged 'boundary'.

K10: two synthetic entries — zero cost with an unconverged case; zero cost with errors 10x the ROM's — are injected into
every family and must be rejected (if either is selected the matcher fails and nothing is matched).

    python fom_match.py --f1 runs/f1/archive/output --rom-scores <json with label -> e, eta per case> --out checks/fom-match.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze as AN

FAMILIES = ('UB', 'UC', 'CC', 'PS')


def candidates(f1, cases):
    """{family: [dict(key, e, eta, cost, admissible, why)]} from an f1 result.json."""
    out = {f: [] for f in FAMILIES}

    def add(fam, key, ent, cal_ok):
        cs = ent.get('cases', {})
        why = []
        if not cal_ok:
            why.append('not calibrated')
        if sorted(int(j) for j in cs) != sorted(cases):
            why.append('case population')
        if not all(c.get('converged') is True and np.isfinite(c.get('e', np.nan)) for c in cs.values()):
            why.append('not converged / non-finite')
        w = ent.get('warm') or {}
        times = [x.get('seconds') for x in w.get('times', [])]
        if not (times and w.get('hashes_ok') is True and np.isfinite(times).all() and min(times) > 0):
            why.append('warm timing')
        e = np.array([cs[str(j)]['e'] if str(j) in cs else np.nan for j in cases], float)
        eta = np.array([cs[str(j)].get('eps_ref', np.nan) if str(j) in cs else np.nan for j in cases], float)
        out[fam].append(dict(key=key, e=e, eta=eta, cost=float(np.median(times)) if times else np.inf,
                             admissible=not why, why=why))
    for k, v in f1['fom']['fd'].items():
        fam = k.split('|')[0]
        add(fam, v.get('key', k), v, v.get('calibration', {}).get('chosen') is not None)
    for k, v in f1['fom']['ps'].items():
        add('PS', k, v, True)
    return out


def match_one(rom, cands, k10=True):
    """rom: dict(e, eta, label) on the cases; cands: one family's list. Returns the match record."""
    pool = list(cands)
    if k10:
        n = len(rom['e'])
        bad1 = dict(key='K10|unconverged', e=np.zeros(n), eta=rom['eta'], cost=0., admissible=False, why=['not converged / non-finite'])
        bad2 = dict(key='K10|inaccurate', e=10 * np.asarray(rom['e']), eta=rom['eta'], cost=0., admissible=True, why=[])
        pool += [bad1, bad2]
    adm = [c for c in pool if c['admissible']]
    ok = []
    for c in adm:
        med, wst = AN.ties_or_beats(c['e'], rom['e'], rom['eta'])
        if med and wst:
            ok.append(c)
    k10_fail = any(c['key'].startswith('K10') for c in ok)
    if k10_fail:
        return dict(status='K10 FAILED: a synthetic entry was matched', matched=None)
    if not ok:
        return dict(status='no matched FOM in the ladder (ROM more accurate than every admissible configuration)',
                    matched=None, admissible=[c['key'] for c in adm if not c['key'].startswith('K10')])
    best = min(ok, key=lambda c: c['cost'])
    real = [c for c in adm if not c['key'].startswith('K10')]
    cheapest = min(real, key=lambda c: c['cost'])
    return dict(status='matched', matched=best['key'], cost_screen=best['cost'], median=float(np.median(best['e'])),
                worst=float(np.max(best['e'])), boundary=best['key'] == cheapest['key'], upper_bound=True,
                n_admissible=len(real), n_matching=len([c for c in ok if not c['key'].startswith('K10')]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f1', required=True)
    ap.add_argument('--roms', required=True, help='json: {label: {e: [...64], eta: [...64]}} on cases 0..63')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    f1 = json.loads((Path(a.f1) / 'result.json').read_text())
    cases = f1['cohort']['cases']
    assert f1['cohort']['seed'] == 923801 and len(cases) == 64 and f1.get('complete') is True
    roms = json.loads(Path(a.roms).read_text())
    C = candidates(f1, cases)
    out = dict(f1_job=f1.get('job_id'), gpu=f1.get('gpu'), label='every speed-up against a matched FOM is an upper bound '
               '(finite FOM ladder); boundary = matched at the family\'s cheapest admissible configuration', matches={},
               admissibility={f: {c['key']: c['why'] for c in v} for f, v in C.items()})
    for lab, r in roms.items():
        r = dict(e=np.asarray(r['e'], float), eta=np.asarray(r['eta'], float))
        out['matches'][lab] = {f: match_one(r, C[f]) for f in FAMILIES}
    out['K10_all_rejected'] = all('K10' not in str(m.get('matched')) and 'K10 FAILED' not in m['status']
                                  for v in out['matches'].values() for m in v.values())
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps({k: {f: (m['status'][:20], m.get('matched')) for f, m in v.items()} for k, v in out['matches'].items()}, indent=1))


if __name__ == '__main__':
    main()
