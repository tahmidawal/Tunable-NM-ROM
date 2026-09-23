"""Independent re-computation (separate code from spec_timing.gates) of every timing median, the drift gate and the
within-phase neighbour gate from the raw invocation records of each accepted run, plus the retained-sample counts
(>= 5 reps per case).  Also recomputes each subject's worst error from the invocation records.  Exits non-zero on
any disagreement with the recorded values.   python check_timing.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from make_report import ACCEPTED, HERE

bad = 0
for prob, runs in ACCEPTED.items():
    for att, od in runs:
        p = HERE / 'runs' / att / 'archive' / od / 'result.json'
        if not p.exists():
            continue
        R = json.loads(p.read_text())
        inv = R['invocations']
        lim = R['config']['neighbour_limit']
        by = defaultdict(list)
        for x in inv:
            by[(x['name'], x['phase'])].append(x)
        names = sorted({x['name'] for x in inv})
        # medians
        for nm in names:
            role = R['timings'][nm]['role']
            ph = ('romA1', 'romA2') if role.startswith('rom') else ('spec',)
            v = [x['fused_device_seconds'] for p_ in ph for x in by[(nm, p_)]]
            m = 1e3 * float(np.median(v))
            if abs(m - R['timings'][nm]['median_ms']) > 1e-9:
                bad += 1; print('MEDIAN MISMATCH', att, od, nm)
            reps = defaultdict(int)
            for p_ in ph:
                for x in by[(nm, p_)]:
                    reps[(p_, x['case'])] += 1
            if min(reps.values()) < 5:
                bad += 1; print('FEWER THAN 5 REPS', att, od, nm)
        # drift
        dr_ok = True
        for nm in names:
            if R['timings'][nm]['role'].startswith('rom'):
                a1 = np.median([x['fused_device_seconds'] for x in by[(nm, 'romA1')]])
                a2 = np.median([x['fused_device_seconds'] for x in by[(nm, 'romA2')]])
                dr_ok &= bool(1 / lim <= a2 / a1 <= lim)
        # neighbour: rebuild predecessor sequences from phase order
        nb_ok, worst = True, 0.0
        for ph in ('romA1', 'romA2', 'spec'):
            seq = [x for x in inv if x['phase'] == ph]
            subs = sorted({x['name'] for x in seq})
            if len(subs) < 2:
                continue
            med = {s: np.median([x['fused_device_seconds'] for x in seq if x['name'] == s]) for s in subs}
            prev = [None] + [x['name'] for x in seq[:-1]]
            assert all(x['previous'] == p_ for x, p_ in zip(seq, prev)), 'predecessor record mismatch'
            for s in subs:
                oth = sorted((o for o in subs if o != s), key=lambda o: med[o])
                if len(oth) == 1:
                    o = oth[0]
                    slow, fast = ([o], [s]) if med[o] >= med[s] else ([s], [o])
                else:
                    k = max(1, len(oth) // 3)
                    fast, slow = oth[:k], oth[-k:]
                hi = [x['fused_device_seconds'] for x, p_ in zip(seq, prev) if x['name'] == s and p_ in slow]
                lo = [x['fused_device_seconds'] for x, p_ in zip(seq, prev) if x['name'] == s and p_ in fast]
                if len(hi) < 3 or len(lo) < 3:
                    nb_ok = False
                    continue
                r = np.median(hi) / np.median(lo)
                worst = max(worst, r)
                nb_ok &= bool(r <= lim)
        rec = R['timing_gates']
        agree = (dr_ok == rec['drift']['passed']) and (nb_ok == rec['neighbour']['passed'])
        if not agree:
            bad += 1
        # errors
        key = 'same_grid_error' if prob.startswith('poisson') else ('same_grid_evolved' if prob.startswith('burgers') else 'same_grid_worst')
        for nm in names:
            e = max(x[key] for x in inv if x['name'] == nm)
            e0 = R['errors'][nm]['worst']
            if abs(e - e0) > 1e-12 * max(e0, 1e-300) + 1e-16:
                bad += 1; print('ERROR MISMATCH', att, od, nm, e, e0)
        # controls on the real records (must be detected): a scaled sample moves the median; a swapped predecessor label
        nm0 = names[0]
        ph0 = 'romA1' if R['timings'][nm0]['role'].startswith('rom') else 'spec'
        v = sorted(x['fused_device_seconds'] for p_ in (('romA1', 'romA2') if ph0 == 'romA1' else ('spec',)) for x in by[(nm0, p_)])
        v2 = [t * 1.5 if i >= len(v) // 2 - 1 else t for i, t in enumerate(v)]
        ctrl_median = abs(1e3 * float(np.median(v2)) - R['timings'][nm0]['median_ms']) > 1e-9
        seq = [x for x in inv if x['phase'] == ph0]
        ctrl_prev = any(x['previous'] != p_ for x, p_ in list(zip(seq, [None, None] + [x['name'] for x in seq[:-2]]))[2:])   # off-by-one alignment
        if not (ctrl_median and ctrl_prev):
            bad += 1; print('CONTROL NOT DETECTED', att, od)
        print(f'{prob:10s} {att}/{od} n={R["intervals"]}: drift {dr_ok} neighbour {nb_ok} (worst {worst:.3f}) agrees_with_recorded={agree} controls_detected={ctrl_median and ctrl_prev}')
print('CHECK', 'PASS' if bad == 0 else f'FAIL ({bad})')
sys.exit(1 if bad else 0)
