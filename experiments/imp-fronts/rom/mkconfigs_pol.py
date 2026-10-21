"""imp-fronts §B5 production policy configs (one per bank; DESIGN B5, Amendments 2-5, 10-12). The case population,
ceiling and schedule membership come from schedule.py (shared with resources.py; fpolicy.py re-validates them).

    python mkconfigs_pol.py <label> --schedule S1|S3      -> configs/pol<label>.json
<label> = 1 (K1; selected rule gauss160 from jcp-robust E2) or an arm label in the schedule. For an arm the selection
outcome checks/select2d-<label>.json is REQUIRED (written by select2d_r384.py, which persists an explicit
'incomplete selection' outcome when the job did not complete); outcome 'selected' -> its rule, anything else ->
selected = null (uncertified start at fib4181, Amendment 4). Cases: the 40 of schedule.POLICY_IDS.
"""
import json
import sys
from pathlib import Path

import bankguard as BG
import schedule as SCH

HERE = Path(__file__).resolve().parent
K1_SELECTED = 'gauss160'          # jcp-robust E2 procedure-selected R384 rule of seed 1 (DESIGN B1)


def selected_of(label):
    if label == '1':
        return K1_SELECTED, 'jcp-robust E2 (K1)'
    p = HERE / 'checks' / f'select2d-{label}.json'
    if not p.exists():
        raise FileNotFoundError(f'{p}: no selection outcome; run select2d_r384.py (it persists incomplete outcomes too)')
    s = json.loads(p.read_text())['selected']['384']
    return (s['rule'] if s['outcome'] == 'selected' and s.get('rule') else None), s['outcome']


def make(label, sch):
    assert sch in ('S1', 'S3'), 'S2 is withdrawn and S4 has no policy jobs'
    assert label in SCH.SCHEDULES[sch]['banks'], f'bank {label} not in {sch}'
    BG.verify({'dir': f'inputs/banks/s{label}'}, HERE)
    sel, outcome = selected_of(label)
    c = dict(attempt=f'pol{label}', driver='fpolicy', mesh=1024, edges=[0, 128, 256, 384, 512], Rp=384, scheme='CNR',
             bank={'dir': f'inputs/banks/s{label}'}, bank_label=label, selected=sel, selection_outcome=outcome,
             cases=list(SCH.POLICY_IDS), table='phaseb/fresh_cohorts.json', test_once=True, anchors=None,
             ceiling_hours=SCH.POLICY_CEILING_H, stop_fraction=SCH.CUTOFF, max_rounds=10, controls={}, schedule=sch)
    SCH.check_policy_config(c)
    return c


def main():
    label = sys.argv[1]
    sch = sys.argv[sys.argv.index('--schedule') + 1]
    c = make(label, sch)
    p = HERE / 'configs' / f'pol{label}.json'
    p.write_text(json.dumps(c, indent=1) + '\n')
    print(p, len(c['cases']), 'cases, start', c['selected'] or 'fib4181 (uncertified start)')


if __name__ == '__main__':
    main()
