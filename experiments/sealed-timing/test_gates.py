"""Must-fail controls for gates.py (PROTOCOL.md §3). Run: python test_gates.py"""
import copy
import random

import gates

SUBJ = ['rom_lat', 'fom', 'rom_prep', 'fom_prep', 'rom_e2e', 'fom_e2e']
BASE = dict(rom_lat=42.0, fom=6.5, rom_prep=3.0, fom_prep=10.0)
BASE.update(rom_e2e=BASE['rom_prep'] + BASE['rom_lat'], fom_e2e=BASE['fom_prep'] + BASE['fom'])


def synthetic(seed=0):
    r = random.Random(seed)
    inv = []
    for blk, n in (('discard_main', 1), ('main', 3), ('discard_confirm', 1), ('confirm', 5)):
        for q in range(n):
            for s in SUBJ:
                for c in range(6):
                    inv.append(dict(block=blk, subject=s, case=c, rep=q, seconds=BASE[s] * (1 + 0.01 * r.random()) / 1e3,
                                    diff=0.0, **({'bitwise': True} if s.endswith('prep') else {})))
    return dict(cases=list(range(6)), subjects=SUBJ, invocations=inv,
                schedule=dict(main_reps=3, confirm_reps=5))


def test_clean_passes():
    e = gates.evaluate(synthetic())
    assert e['prepared_qualified'] and e['e2e_qualified'] and e['panel_wide_pass'], e['reasons']


def test_injected_fault_fails():
    m = synthetic()
    for row in m['invocations']:
        if row['block'] == 'confirm' and row['subject'] == 'fom' and row['rep'] == 2:
            row['seconds'] *= 1.5
    e = gates.evaluate(m)
    assert not e['prepared_qualified'] and not e['e2e_qualified'] and 'fom' in e['reasons'], e['reasons']


def test_discarded_block_ignored():
    m = synthetic()
    for row in m['invocations']:
        if row['block'] == 'discard_confirm':
            row['seconds'] *= 3
    assert gates.evaluate(m)['panel_wide_pass']


def test_doubled_work_fused_fails():
    m = synthetic()
    for row in m['invocations']:
        if row['subject'] == 'rom_e2e':
            row['seconds'] *= 2
    e = gates.evaluate(m)
    assert e['prepared_qualified'] and not e['e2e_qualified'] and 'component_sum_vs_fused' in e['reasons'], e['reasons']


def test_output_mismatch_fails():
    m = synthetic()
    m['invocations'][-1]['diff'] = 1e-6
    assert not gates.evaluate(m)['panel_wide_pass']


def test_prep_not_bitwise_fails():
    m = copy.deepcopy(synthetic())
    for row in m['invocations']:
        if row['subject'] == 'rom_prep' and row['block'] == 'main':
            row['bitwise'] = False
            break
    e = gates.evaluate(m)
    assert e['prepared_qualified'] and not e['e2e_qualified']


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f()
            print('pass', k)
