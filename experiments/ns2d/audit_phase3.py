"""audit_phase3.py -- INDEPENDENT NumPy audit of a Phase-3 result directory.

    python experiments/ns2d/audit_phase3.py <output-dir> [audit.json]

Imports neither the driver nor JAX.  From rom_fields_N*.npz (the last timed repetition of
every subject x case) and reference_N*.npz (the converged same-grid FOM at the evaluation
times) it recomputes, per subject and case, the three error metrics, then the per-subject
aggregates (worst/median over cases), and compares them with result.json to 1e-12 relative.
It also re-derives the pre-registered verdict (monotone ladder and the 2x gain) from the
recomputed numbers alone, and checks the timing medians against the retained repetitions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def close(a, b, rtol=1e-12, atol=1e-15):
    return abs(a - b) <= atol + rtol * abs(b)


def main():
    out = Path(sys.argv[1])
    res = json.loads((out / 'result.json').read_text())
    N = res['config']['N']
    ref = np.load(out / f'reference_N{N}.npz')
    U = ref['U']                                                       # (cases, n_eval, n)
    rom = np.load(out / f'rom_fields_N{N}.npz')
    audit = dict(source=str(out / 'result.json'), checks=[], all_match=True)

    def check(name, rec, rep):
        ok = close(rec, rep)
        audit['checks'].append(dict(name=name, recomputed=rec, reported=rep, match=ok))
        audit['all_match'] &= ok
        if not ok:
            print(f'MISMATCH {name}: recomputed {rec:.6e} reported {rep:.6e}')

    reps = res['config']['REPS']
    last = {(i['subject'], i['case']): i for i in res['invocations'] if i['rep'] == reps}
    per_subject = {}
    for key in rom.files:
        subject, case = key.split('__case')
        case = int(case)
        f = rom[key].reshape(U.shape[1], -1)
        r = U[case]
        n0 = np.linalg.norm(r[0])
        e = np.linalg.norm(f - r, axis=1) / n0
        inv = last[(subject, case)]
        check(f'{subject}.case{case}.worst_evolved', float(e[1:].max()), inv['worst_evolved'])
        check(f'{subject}.case{case}.worst_all', float(e.max()), inv['worst_all'])
        check(f'{subject}.case{case}.t0', float(e[0]), inv['t0'])
        per_subject.setdefault(subject, {})[case] = e
    agg = res.get('aggregates', {})
    rec_worst = {}
    for subject, cases in per_subject.items():
        we = [e[1:].max() for e in cases.values()]
        rec_worst[subject] = float(max(we))
        if subject in agg:
            check(f'{subject}.aggregate.worst_evolved', float(max(we)), agg[subject]['worst_evolved'])
            check(f'{subject}.aggregate.median_evolved', float(np.median(we)), agg[subject]['median_evolved'])
            secs = [i['seconds'] for i in res['invocations'] if i['subject'] == subject and i['timed']]
            check(f'{subject}.aggregate.median_seconds', float(np.median(secs)), agg[subject]['median_seconds'])
    q = res['config']['Q_LADDER']
    we = [rec_worst.get(f'neural_q{v}') for v in q]
    if all(v is not None for v in we):
        mono = all(b <= a * (1 + 1e-12) for a, b in zip(we, we[1:]))
        gain = we[0] / we[-1]
        g = res['gates'].get('R-LADDER', {})
        audit['verdict'] = dict(monotone=mono, gain=gain, passed=bool(mono and gain >= 2.0),
                                reported_passed=g.get('passed'))
        ok = audit['verdict']['passed'] == g.get('passed')
        audit['all_match'] &= ok
        print('VERDICT recomputed', audit['verdict'])
    audit['n_checks'] = len(audit['checks'])
    print('ALL_MATCH', audit['all_match'], 'checks', audit['n_checks'])
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
