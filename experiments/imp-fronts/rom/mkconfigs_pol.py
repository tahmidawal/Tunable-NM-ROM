"""imp-fronts §B5 production policy configs (one per bank; DESIGN B5, Amendments 2-5, 10-12). The case population,
ceiling and schedule membership come from schedule.py (shared with resources.py; fpolicy.py re-validates them).

    python mkconfigs_pol.py <label> --schedule S1|S3|S3C  -> configs/pol<label>.json
    python mkconfigs_pol.py e10s                          -> configs/pole10s.json (E10-s, re-used test cases)
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
    assert sch in ('S1', 'S3', 'S3C'), 'S2 is withdrawn and S4 has no policy jobs'
    assert label in SCH.SCHEDULES[sch]['banks'], f'bank {label} not in {sch}'
    BG.verify({'dir': f'inputs/banks/s{label}'}, HERE)
    sel, outcome = selected_of(label)
    c = dict(attempt=f'pol{label}', driver='fpolicy', mesh=1024, edges=[0, 128, 256, 384, 512], Rp=384, scheme='CNR',
             bank={'dir': f'inputs/banks/s{label}'}, bank_label=label, selected=sel, selection_outcome=outcome,
             cases=list(SCH.POLICY_IDS), table='phaseb/fresh_cohorts.json', test_once=True, anchors=None,
             ceiling_hours=SCH.POLICY_CEILING_H, stop_fraction=SCH.CUTOFF, max_rounds=10, controls={}, schedule=sch)
    SCH.check_policy_config(c)
    return c


def make_e10s():
    """POL-e10s (E10-DESIGN.md sections 8-9): K1 from gauss160, the 40 Phase B cases RE-USED (marked), factor set
    {1, .5, .25, .125}, round cap 11, ceiling 0.8 h, schedule E10."""
    BG.verify({'dir': 'inputs/banks/s1'}, HERE)
    c = dict(attempt='pole10s', driver='fpolicy', mesh=1024, edges=[0, 128, 256, 384, 512], Rp=384, scheme='CNR',
             bank={'dir': 'inputs/banks/s1'}, bank_label='1', selected=K1_SELECTED, selection_outcome='jcp-robust E2 (K1)',
             factors=list(SCH.E10_FACTORS), max_rounds=SCH.E10_ROUNDS, cases=list(SCH.POLICY_IDS),
             table='phaseb/fresh_cohorts.json', test_once=True, reused_test_cases=True,
             reuse_note='E10 re-uses the Phase B fresh test cases already scored for K1 and arm (b) (E10-DESIGN.md section 8): '
                        'a separate, re-use evaluation, not test-once',
             anchors=None, ceiling_hours=SCH.E10_CEIL['policy_e10s'], stop_fraction=SCH.CUTOFF, controls={}, schedule='E10')
    SCH.check_e10s_config(c)
    return c


def main():
    if sys.argv[1] == 'e10s':
        c = make_e10s()
        p = HERE / 'configs' / 'pole10s.json'
        p.write_text(json.dumps(c, indent=1) + '\n')
        print(p, len(c['cases']), 'cases (re-used), factors', c['factors'])
        return
    label = sys.argv[1]
    sch = sys.argv[sys.argv.index('--schedule') + 1]
    c = make(label, sch)
    p = HERE / 'configs' / f'pol{label}.json'
    p.write_text(json.dumps(c, indent=1) + '\n')
    print(p, len(c['cases']), 'cases, start', c['selected'] or 'fib4181 (uncertified start)')


if __name__ == '__main__':
    main()
