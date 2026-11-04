"""R1b time / R1c space gate decisions from compare.py outputs (DESIGN-G5 §4.4, G5-A5).

Fail-closed: any missing, invalid or nonfinite evidence fails the gate.
Usage: r1_gates.py --dir <dir with compare_*.json, startup_*.json, space_*.json> --out R1BC-GATES.json

Candidate dt0 = 1/c, c in (800, 1600, 3200), in that order; c qualifies iff
  (i)   compare_c_2c passes all four tolerances,
  (ii)  its frozen three-level stencil gives empirical orders in [1.5, 2.5] for every metric
        whose two differences both exceed that metric's noise floor; stencil(800) =
        (800,1600,3200), stencil(1600) = stencil(3200) = (1600,3200,6400); only the candidate's
        own stencil counts (a non-asymptotic coarse stencil does not disqualify finer c),
  (iii) startup_c (4 vs 16 substeps at c) passes 0.2 x every tolerance.
R1b passes with the first qualifying c. R1c passes iff space_c (P3/P2 vs P4/P3 at the selected
c) passes all four tolerances. Noise floors (G5-A5 rev.2) are demonstrated, not assumed: the
roundoff control `roundoff_1600.json` compares the 1/1600 P4 run with an otherwise identical
8-thread run (only floating-point reduction order differs); floor_y = 10 x that gap. A missing
control fails. Every record must share one initial-state hash and the fixed window W.
"""
import argparse
import json
import math
from pathlib import Path

TOL = {'e_u_max': 2e-3, 'e_p_max': 4e-3, 'e_D': 3e-3, 'e_L': 2e-4}
STENCIL = {800: (800, 1600, 3200), 1600: (1600, 3200, 6400), 3200: (1600, 3200, 6400)}


def load(p):
    if not p.exists():
        return None, f'missing {p.name}'
    rec = json.loads(p.read_text())
    if rec.get('valid') is not True:
        return None, f'{p.name} not validated: {rec.get("invalid_reason")}'
    if any(not (isinstance(rec.get(k), (int, float)) and math.isfinite(rec[k])) for k in TOL):
        return None, f'{p.name} nonfinite metric'
    return rec, None


def passes(rec, scale=1.0):
    return {k: rec[k] <= scale * v for k, v in TOL.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); d = Path(a.dir)
    res = {'tolerances': TOL, 'stencils': {str(k): v for k, v in STENCIL.items()},
           'candidates': {}, 'evidence_errors': []}
    rc, err = load(d / 'roundoff_1600.json')
    if rc is None:
        res['evidence_errors'].append(err); res['R1b_time_gate'] = 'fail_missing_roundoff_control'
        Path(a.out).write_text(json.dumps(res, indent=2) + '\n'); print(res['R1b_time_gate']); return
    FLOOR = {k: 10 * rc[k] for k in TOL}
    res['noise_floors'] = FLOOR
    ident = {(rc['initial_state_sha256'], rc['W'], round(rc['window'][0], 9))}
    pairs = {}
    for c in (800, 1600, 3200):
        rec, err = load(d / f'compare_{c}_{2 * c}.json')
        pairs[c] = rec
        if rec is not None:
            ident.add((rec['initial_state_sha256'], rec['W'], round(rec['window'][0], 9)))
        if err:
            res['evidence_errors'].append(err)
    dt0 = None
    for c in (800, 1600, 3200):
        cand = {'pair': None, 'orders': None, 'startup': None, 'qualifies': False, 'reasons': []}
        if pairs[c] is None:
            cand['reasons'].append('pair evidence missing/invalid')
            res['candidates'][str(c)] = cand
            continue
        cand['pair'] = {'metrics': {k: pairs[c][k] for k in TOL}, 'passes': passes(pairs[c])}
        ok = all(cand['pair']['passes'].values())
        if not ok:
            cand['reasons'].append('pair tolerance failed')
        s0, s1, s2 = STENCIL[c]
        lo, hi = pairs.get(s0), pairs.get(s1)
        if lo is None or hi is None:
            cand['reasons'].append('stencil evidence missing'); ok = False
        else:
            orders = {}
            for k in TOL:
                if lo[k] > FLOOR[k] and hi[k] > FLOOR[k]:
                    orders[k] = math.log2(lo[k] / hi[k])
                else:
                    orders[k] = None
            cand['orders'] = orders
            bad = [k for k, v in orders.items() if v is not None and not 1.5 <= v <= 2.5]
            if bad:
                cand['reasons'].append(f'empirical order outside [1.5,2.5]: {bad}'); ok = False
        st, err = load(d / f'startup_{c}.json')
        if st is not None:
            ident.add((st['initial_state_sha256'], st['W'], round(st['window'][0], 9)))
        if st is None:
            cand['reasons'].append(err); ok = False
        else:
            cand['startup'] = {'metrics': {k: st[k] for k in TOL}, 'passes': passes(st, 0.2)}
            if not all(cand['startup']['passes'].values()):
                cand['reasons'].append('startup check failed'); ok = False
        cand['qualifies'] = ok
        res['candidates'][str(c)] = cand
        if ok and dt0 is None:
            dt0 = c
    if len(ident) != 1:
        res['evidence_errors'].append(f'records disagree on initial state/window: {sorted(ident)}')
        dt0 = None
    res['dt0_denominator'] = dt0
    res['R1b_time_gate'] = 'pass' if dt0 else 'fail'
    if dt0:
        sp, err = load(d / f'space_{dt0}.json')
        if sp is not None and (sp['initial_state_sha256'], sp['W'], round(sp['window'][0], 9)) not in ident:
            sp, err = None, 'space record differs in initial state/window'
        if sp is None:
            res['R1c_space_gate'] = 'pending_or_missing'; res['R1c_reason'] = err
        else:
            res['space'] = {'metrics': {k: sp[k] for k in TOL}, 'passes': passes(sp)}
            res['R1c_space_gate'] = 'pass' if all(res['space']['passes'].values()) else 'fail'
    else:
        res['R1c_space_gate'] = 'not_evaluated_R1b_failed'
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: res[k] for k in ('dt0_denominator', 'R1b_time_gate', 'R1c_space_gate')}))


if __name__ == '__main__':
    main()
