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


def validate(fam, key, ent, cal_ok, cases):
    """One admissibility validator for real and synthetic (K10) entries: calibrated; converged with complete evidence and
    finite scores on every one of the cases; hash-checked finite positive warm timings. Returns the candidate record with
    per-case errors, eta and reference eligibility keyed by case id."""
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
    e = {int(j): c.get('e', np.nan) for j, c in cs.items()}
    eta = {int(j): c.get('eps_ref', np.nan) for j, c in cs.items()}
    el = {int(j): c.get('eligible', False) for j, c in cs.items()}
    return dict(family=fam, key=key, e=e, eta=eta, eligible=el, cost=float(np.median(times)) if times else np.inf,
                admissible=not why, why=why)


def candidates(f1, cases):
    out = {f: [] for f in FAMILIES}
    for k, v in f1['fom']['fd'].items():
        fam = k.split('|')[0]
        out[fam].append(validate(fam, v.get('key', k), v, v.get('calibration', {}).get('chosen') is not None, cases))
    for k, v in f1['fom']['ps'].items():
        out['PS'].append(validate('PS', k, v, True, cases))
    return out


def aligned(rom, cand):
    """ROM and FOM errors on the common reference-eligible cases (ROM ranked ids that the FOM record also marks eligible);
    eta must agree (same reference, same store)."""
    ids = [j for j in rom['ranked_ids'] if cand['eligible'].get(j)]
    er = np.array([rom['e'][rom['ranked_ids'].index(j)] for j in ids])
    eta = np.array([rom['eta'][rom['ranked_ids'].index(j)] for j in ids])
    ef = np.array([cand['e'][j] for j in ids])
    assert np.allclose(eta, [cand['eta'][j] for j in ids], rtol=0, atol=1e-15), 'eta differs between ROM and FOM scores'
    return ef, er, eta, ids


def k10_entries(rom, cases, template):
    """Two synthetic raw entries through the SAME validator (audit select-1): (a) one unconverged case, otherwise perfect
    and zero-cost; (b) converged and zero-cost with errors beyond the ROM's upper endpoint on every case by a margin
    (e = 10 e_ROM + 4 eta + 1e-3), so the tie rule must reject it at any accuracy level."""
    er = {j: rom['e'][rom['ranked_ids'].index(j)] if j in rom['ranked_ids'] else 1. for j in cases}
    eta = {j: rom['eta'][rom['ranked_ids'].index(j)] if j in rom['ranked_ids'] else 0. for j in cases}
    warm = dict(times=[dict(case=cases[0], seconds=1e-9, hash_ok=True)], hashes_ok=True)
    a = dict(cases={str(j): dict(converged=(i != 0), e=0., eps_ref=eta[j], eligible=True) for i, j in enumerate(cases)}, warm=warm)
    b = dict(cases={str(j): dict(converged=True, e=10 * er[j] + 4 * eta[j] + 1e-3, eps_ref=eta[j], eligible=True) for j in cases},
             warm=warm)
    return [validate('K10', 'K10|unconverged', a, True, cases), validate('K10', 'K10|inaccurate', b, True, cases)]


def match_one(rom, cands, cases):
    pool = list(cands) + k10_entries(rom, cases, None)
    adm = [c for c in pool if c['admissible']]
    real_adm = [c for c in adm if not c['key'].startswith('K10')]
    ok = []
    for c in adm:
        ef, er, eta, ids = aligned(rom, c)
        if not ids:
            continue
        med, wst = AN.ties_or_beats(ef, er, eta)
        if med and wst:
            ok.append(c)
    if any(c['key'].startswith('K10') for c in ok):
        return dict(status='K10 FAILED: a synthetic entry was matched', matched=None)
    if not real_adm:
        return dict(status='unavailable (no admissible FOM configuration in the family)', matched=None,
                    inadmissible={c['key']: c['why'] for c in cands})
    if not ok:
        return dict(status='no joint median-and-worst match in the ladder', matched=None, admissible=[c['key'] for c in real_adm])
    best = min(ok, key=lambda c: c['cost'])
    cheapest = min(real_adm, key=lambda c: c['cost'])
    ef, er, eta, ids = aligned(rom, best)
    return dict(status='matched', matched=best['key'], cost_screen=best['cost'], median=float(np.median(ef)),
                worst=float(np.max(ef)), ranked_cases=len(ids), boundary=best['key'] == cheapest['key'], upper_bound=True,
                n_admissible=len(real_adm), n_matching=len(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f1', required=True)
    ap.add_argument('--roms', required=True, help='checks/rom-scores.json: {label: {key, e, eta, ranked_ids, eligible}}')
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
        if r.get('eligibility') != 'eligible':          # only eligible configurations receive H4.3 matches
            out['matches'][lab] = {f: dict(status=f"ROM not eligible ({r.get('eligibility')})", matched=None) for f in FAMILIES}
            continue
        out['matches'][lab] = {f: match_one(r, C[f], cases) for f in FAMILIES}
    out['K10_all_rejected'] = all('K10 FAILED' not in m['status'] for v in out['matches'].values() for m in v.values())
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps({k: {f: (m['status'][:20], m.get('matched')) for f, m in v.items()} for k, v in out['matches'].items()}, indent=1))


if __name__ == '__main__':
    main()
