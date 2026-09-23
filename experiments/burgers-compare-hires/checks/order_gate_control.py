"""Control for the `no_order_effect_between_arms` gate (DESIGN section 6), on REAL timing data: repanel's br1024
(job 4187625) invocations, the panel whose gate failed with a 106 % gap.

Three measurements:
  1. repanel's own definition (per arm, medians POOLED over the six cases, predecessor >= 1 s): must reproduce
     repanel's failure -- it does, so the data are read correctly;
  2. this lane's definition (every sample normalised by its arm's PER-CASE median): on the same data;
  3. the control proper: a synthetic slowdown of +6 % / +10 % injected into the samples that follow a slow
     predecessor, in the same real data. This lane's gate MUST fail on both, or it cannot see the effect it exists
     to catch.

    python checks/order_gate_control.py <repanel br1024 result.json> checks/order-gate-control.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_cmp import order_effect

rep = json.loads(Path(sys.argv[1]).read_text())
rows = [dict(name=r['name'], case=r['case'], gpu_seconds=r['gpu_seconds']) for r in rep['invocations']]


def pooled(rows, slow=1.):
    """repanel audit_repanel's definition: per arm, median after a >= 1 s predecessor vs after a short one,
    pooled over cases."""
    g = {}
    for prev, r in zip(rows[:-1], rows[1:]):
        g.setdefault(r['name'], ([], []))[0 if prev['gpu_seconds'] >= slow else 1].append(r['gpu_seconds'])
    gaps = {n: float(np.median(s) / np.median(f) - 1) for n, (s, f) in g.items() if s and f}
    return max(abs(v) for v in gaps.values()), gaps


def inject(rows, factor, split):
    med = {}
    for r in rows:
        med.setdefault((r['name'], r['case']), []).append(r['gpu_seconds'])
    med = {k: float(np.median(v)) for k, v in med.items()}
    out = [dict(rows[0])]
    for prev, r in zip(rows[:-1], rows[1:]):
        x = dict(r)
        slow = prev['gpu_seconds'] >= (1. if split == 'a' else 3. * med[(r['name'], r['case'])])
        if slow:
            x['gpu_seconds'] = r['gpu_seconds'] * factor
        out.append(x)
    return out


worst_pooled, gaps_pooled = pooled(rows)
real = order_effect(rows)
controls = {}
for split in ('a', 'b'):
    for factor in (1.06, 1.10):
        res = order_effect(inject(rows, factor, split))
        controls[f'split_{split}_x{factor}'] = dict(gate_passed=res['passed'],
                                                   worst_gap_split=res['splits'][split]['worst_gap'],
                                                   judged=res['splits'][split]['judged_arms'])
out = dict(source=dict(file=sys.argv[1], job_id=rep.get('job_id'), invocations=len(rows)),
           repanel_pooled_definition=dict(worst_gap=worst_pooled, gaps=gaps_pooled,
                                          reproduces_repanel_failure=worst_pooled > .05),
           this_lane_definition_on_real_data=real,
           injected_controls=controls,
           control_fails_as_required=all(v['gate_passed'] is False for v in controls.values()),
           reading=('repanel\'s pooled gap is a case-mix artefact if this lane\'s per-case-normalised gap on the same '
                    'data is within the bar while the injected controls fail'))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')
print('repanel pooled definition on br1024: worst gap %.3f (reproduces the failure: %s)' % (worst_pooled, worst_pooled > .05))
print('this lane per-case definition on br1024: passed =', real['passed'], '| worst a', real['splits']['a']['worst_gap'],
      '| worst b', real['splits']['b']['worst_gap'])
for k, v in controls.items():
    print('injected', k, '-> gate passed =', v['gate_passed'], 'worst', v['worst_gap_split'], 'judged', v['judged'])
assert out['control_fails_as_required'], 'the gate does not catch an injected slowdown'
