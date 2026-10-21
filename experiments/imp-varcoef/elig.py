"""Freeze the eligibility mask and timing slots (DESIGN A1.6, A2.5, A2.25, A3.25, A5.5).

    python elig.py --results runs/<ref job>/result.json [...] --out checks/eligibility.json

Later result files override earlier ones per case (fallback reruns). Eligible = E0 judge 'accepted', eps_ref <= 1e-3, and
the contrast's cross-check not failed. Coverage rule: the gate is decidable at a contrast only with >= 24 of the 32 pooled
cases eligible. Slots: the first six eligible dev cases (index order), completed with eligible val cases; anchor = dev6 s.
Writes the mask and a sha256 line to checks/eligibility.FROZEN.
"""
import argparse
import hashlib
import json
from pathlib import Path

import vcoef as VC

ELIG_EPS = 1e-3
COVERAGE = 24
NSLOT = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cases, xc = {}, {}
    srcs = []
    for r in a.results:
        raw = Path(r).read_bytes()
        d = json.loads(raw)
        assert d.get('complete') is True, ('incomplete reference job', r)
        assert d.get('backend') == 'gpu', ('reference job not on gpu', r)
        srcs.append(dict(path=r, sha256=hashlib.sha256(raw).hexdigest()))
        cases.update(d['cases'])
        xc.update(d['crosscheck'])
    out = dict(sources=srcs, rule=dict(eps_max=ELIG_EPS, coverage_min=COVERAGE, nslot=NSLOT), contrasts={})
    for C in sorted({k.split('|')[0][1:] for k in cases}, key=float):
        rows = {k.split('|', 1)[1]: v for k, v in cases.items() if k.split('|')[0] == f'C{C}'}
        xcs = {k.split('|', 1)[1]: v for k, v in xc.items() if k.split('|')[0] == f'C{C}'}
        want_xc = {f'dev|{j}' for j in range(4)}
        xc_missing = sorted(want_xc - set(xcs))
        xc_failed = bool(xc_missing) or any(not xcs[c]['passed'] for c in want_xc if c in xcs)
        per = {}
        for cid, v in rows.items():
            ok = bool(v['judge']['outcome'] == 'accepted' and v['eps_ref'] is not None and v['eps_ref'] <= ELIG_EPS
                      and not xc_failed)
            per[cid] = dict(outcome=v['judge']['outcome'], eps_ref=v['eps_ref'], p=v['judge']['p'], eligible=ok,
                            desc=v['desc'], ref_sha256=v['ref_sha256'], realised=v['realised'])
        el = [c for c in per if per[c]['eligible']]
        dev = sorted([c for c in el if c.startswith('dev|')], key=lambda c: int(c.split('|')[1]))
        val = sorted([c for c in el if c.startswith('val|')], key=lambda c: int(c.split('|')[1]))
        fom = (dev + val)[:NSLOT]
        slots = [dict(slot=s, anchor=['dev6', s], fom=f) for s, f in enumerate(fom)]
        want_ids = {f'dev|{j}' for j in range(len(VC.cohort('dev')))} | {f'val|{j}' for j in range(len(VC.cohort('val')))}
        pooled = len(want_ids)
        exact = set(per) == want_ids
        out['contrasts'][C] = dict(cases=per, eligible=el, n_cases=len(per), n_pooled_expected=pooled,
                                   exact_case_set=exact, crosscheck_missing=xc_missing,
                                   coverage=len(el), decidable=bool(len(el) >= COVERAGE and exact
                                                                     and not xc_failed and len(slots) == NSLOT),
                                   crosscheck={k: dict(passed=v['passed'], converged=v['converged'],
                                                       distance=v['distance'], bar=v['bar'], U_xc=v['U_xc'])
                                               for k, v in xcs.items()},
                                   crosscheck_failed=xc_failed, slots=slots,
                                   excluded={c: per[c]['outcome'] for c in per if not per[c]['eligible']})
    raw = json.dumps(out, indent=1, default=float).encode()
    Path(a.out).write_bytes(raw)
    h = hashlib.sha256(raw).hexdigest()
    Path(a.out).with_suffix('.FROZEN').write_text(f'{h}  {Path(a.out).name}\n')
    for C, v in out['contrasts'].items():
        print(C, 'eligible', v['coverage'], '/', v['n_cases'], 'decidable', v['decidable'], 'xc_failed', v['crosscheck_failed'],
              'excluded', v['excluded'])


if __name__ == '__main__':
    main()
