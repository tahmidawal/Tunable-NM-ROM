"""imp-fronts unit tests (no JAX): the §B5 policy state machine (rom/policy_sm.py) driven through every transition and
terminal state with synthetic run records; the 10-round bound; the successor maps; the C-B1 bank hash guard; the fresh-
cohort table guard; the selection-config equivalence; the cost certificate self-test.

    /home/tahmid/Dev/.venv/bin/python -m pytest -q experiments/imp-fronts/tests/test_policy.py
"""
from __future__ import annotations

import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import pytest

T = Path(__file__).resolve().parent
LANE = T.parent
ROM = LANE / 'rom'
sys.path.insert(0, str(ROM))
sys.path.insert(0, str(LANE))

import policy_sm as SM  # noqa: E402
import bankguard as BG  # noqa: E402
import fcases as FC  # noqa: E402

FIXTURE = T / 'fixtures' / 'fresh_cohorts_fixture.json'
INF, NAN = float('inf'), float('nan')


# ----------------------------------------------------------------------------------------------- maps (Amendment 3)
def test_maps_are_amendment3_verbatim():
    assert SM.CHAIN == ('fib4181', 'fib6765', 'gauss96', 'fib10946', 'gauss128', 'gauss160', 'gauss192', 'gauss256')
    assert SM.NEXT_CANDIDATE == {a: b for a, b in zip(SM.CHAIN, SM.CHAIN[1:])}
    assert 'gauss256' not in SM.NEXT_CANDIDATE
    assert SM.Q2_SUCCESSOR == {'fib4181': 'fib6765', 'fib6765': 'fib10946', 'gauss96': 'gauss128', 'fib10946': 'fib17711',
                               'gauss128': 'gauss160', 'gauss160': 'gauss192', 'gauss192': 'gauss256', 'gauss256': 'gauss384'}
    assert set(SM.Q2_SUCCESSOR) == set(SM.CHAIN)
    assert len(set(SM.Q2_SUCCESSOR.values())) == len(SM.Q2_SUCCESSOR)        # injective: a succ key fixes its prod


def test_maps_agree_with_registered_select2d_ladder():
    from make_configs_e2 import CANDIDATES
    assert [c for c, _ in CANDIDATES] == list(SM.CHAIN[:-1])                 # select2d ladder order, + gauss256
    assert {c: s for c, s in CANDIDATES} == {k: v for k, v in SM.Q2_SUCCESSOR.items() if k != 'gauss256'}


def test_needed_and_factors():
    assert SM.needed('gauss160', 1.) == [('prod', 'gauss160', 1.), ('tight', 'gauss160', 1.), ('succ', 'gauss192', 1.),
                                         ('half', 'gauss160', .5)]
    assert SM.needed('gauss160', .5)[-1] == ('half', 'gauss160', .25)
    assert SM.needed('gauss256', .25) == [('prod', 'gauss256', .25), ('tight', 'gauss256', .25), ('succ', 'gauss384', .25)]
    assert SM.FACTORS == (1., .5, .25)


def test_start_states():
    assert SM.start_state('gauss160') == ('gauss160', 1., False)
    assert SM.start_state(None) == ('fib4181', 1., True)                     # uncertified start
    with pytest.raises(ValueError):
        SM.start_state('gauss320')


# ----------------------------------------------------------------------------------------------- decide(): every row
P_OK = dict(valid=True)
S_OK = dict(valid=True, dist_succ=1e-5, rho_succ_max=.01)
H_OK = dict(valid=True, dist_half=1e-5)
H_BAD = dict(valid=True, dist_half=1e-3)


@pytest.mark.parametrize('prod', [None, {}, dict(valid=False), dict(valid=None), dict(valid='yes')])
def test_1_invalid_prod_beats_everything(prod):
    assert SM.decide('gauss160', 1., prod, S_OK, H_OK) == ('terminal', 'INVALID')
    assert SM.decide('gauss160', .25, prod, None, None) == ('terminal', 'INVALID')


@pytest.mark.parametrize('succ', [None, {}, dict(valid=False, dist_succ=0., rho_succ_max=0.),
                                  dict(valid=True, dist_succ=None, rho_succ_max=.01),
                                  dict(valid=True, dist_succ=NAN, rho_succ_max=.01),
                                  dict(valid=True, dist_succ=INF, rho_succ_max=.01),
                                  dict(valid=True, dist_succ=1e-5, rho_succ_max=None),
                                  dict(valid=True, dist_succ=1e-5, rho_succ_max=NAN),
                                  dict(valid=True, dist_succ=1e-5, rho_succ_max=INF),
                                  dict(valid=True, dist_succ=2.5000001e-4, rho_succ_max=.01),
                                  dict(valid=True, dist_succ=1e-5, rho_succ_max=.1160001),
                                  dict(valid=True, dist_succ=1e-5)])
def test_2_q2_missing_nonfinite_or_failing_escalates(succ):
    assert SM.decide('gauss160', 1., P_OK, succ, H_OK) == ('next', ('gauss192', 1.))
    assert SM.decide('gauss160', .25, P_OK, succ, None) == ('next', ('gauss192', .25))     # beats (3)
    assert SM.decide('gauss128', .5, P_OK, succ, None) == ('next', ('gauss160', .5))       # beats (4)
    assert SM.decide('gauss256', 1., P_OK, succ, H_OK) == ('terminal', 'Q2-FAIL')          # chain exhausted
    assert SM.decide('fib6765', 1., P_OK, succ, H_OK) == ('next', ('gauss96', 1.))         # next_candidate, not q2_successor


def test_2_q2_bars_inclusive():
    s = dict(valid=True, dist_succ=2.5e-4, rho_succ_max=.116)
    assert SM.decide('gauss160', 1., P_OK, s, H_OK) == ('terminal', 'CONVERGED')


def test_3_f_quarter_step_unconverged():
    assert SM.decide('gauss160', .25, P_OK, S_OK, None) == ('terminal', 'STEP-UNCONVERGED')
    assert SM.decide('gauss160', .25, P_OK, S_OK, H_OK) == ('terminal', 'STEP-UNCONVERGED')  # no finer check exists


@pytest.mark.parametrize('half', [None, {}, dict(valid=False, dist_half=0.), dict(valid=True, dist_half=None),
                                  dict(valid=True, dist_half=NAN), dict(valid=True, dist_half=INF), dict(valid=True)])
def test_4_half_unavailable(half):
    assert SM.decide('gauss160', 1., P_OK, S_OK, half) == ('terminal', 'STEP-UNAVAILABLE')
    assert SM.decide('gauss160', .5, P_OK, S_OK, half) == ('terminal', 'STEP-UNAVAILABLE')


def test_5_converged_inclusive_and_6_halving():
    assert SM.decide('gauss160', 1., P_OK, S_OK, dict(valid=True, dist_half=2.5e-4)) == ('terminal', 'CONVERGED')
    assert SM.decide('gauss160', 1., P_OK, S_OK, H_BAD) == ('next', ('gauss160', .5))
    assert SM.decide('gauss160', .5, P_OK, S_OK, H_BAD) == ('next', ('gauss160', .25))


# ----------------------------------------------------------------------------------------------- synthetic driver
class Synth:
    """Synthetic executor. outcome(cid, rule, f) -> one of
         'ok'      prod valid, Q2 pass, half within bar
         'stepbad' prod valid, Q2 pass, half beyond bar
         'q2bad'   prod valid, Q2 fails (distance)
         'rhobad'  prod valid, Q2 fails (rho non-finite)
         'succinv' prod valid, succ invalid
         'prodinv' prod invalid
         'halfinv' prod valid, Q2 pass, half invalid
       Counts every request (every run charged) and refuses duplicates."""

    def __init__(self, outcome, budget_after=None):
        self.outcome, self.calls, self.seen, self.budget_after, self.flag = outcome, [], set(), budget_after, False

    def __call__(self, reqs):
        out = {}
        for q in reqs:
            if self.budget_after is not None and len(self.calls) >= self.budget_after:
                self.flag = True
                break
            assert q['key'] not in self.seen, f"run requested twice: {q['key']}"
            self.seen.add(q['key'])
            self.calls.append(q)
            o = self.outcome(q['cid'], q['state_rule'], q['state_f'])
            if q['kind'] == 'prod':
                out[q['key']] = dict(valid=o != 'prodinv', q1_case=True)
            elif q['kind'] == 'tight':
                out[q['key']] = dict(valid=True, dist_tight=1e-6)
            elif q['kind'] == 'succ':
                out[q['key']] = dict(valid=o != 'succinv', dist_succ=(1e-3 if o == 'q2bad' else 1e-6),
                                     rho_succ_max=(INF if o == 'rhobad' else .01))
            else:
                # a half run is also the next state's prod: its own validity follows the NEXT state's outcome
                nxt = self.outcome(q['cid'], q['rule'], q['f'])
                out[q['key']] = dict(valid=(o != 'halfinv') and nxt != 'prodinv', q1_case=True,
                                     dist_half=(1e-3 if o == 'stepbad' else 1e-6))
        return out

    def over(self):
        return self.flag


def run_cases(outcome, selected='gauss160', n=1, **kw):
    ex = Synth(outcome, kw.pop('budget_after', None))
    cs = [SM.Case(f'FC1|{i}', selected) for i in range(n)]
    store = SM.drive(cs, ex, over_budget=ex.over, **kw)
    return cs, store, ex


def test_drive_converged_first_round():
    (c,), store, ex = run_cases(lambda *a: 'ok')
    assert (c.terminal, c.rounds, c.scored) == ('CONVERGED', 1, 'FC1|0|cert|gauss160|1')
    assert [q['kind'] for q in ex.calls] == ['prod', 'tight', 'succ', 'half']
    assert SM.qualified(c, lambda k: store[k]['q1_case'])
    assert not SM.qualified(c, lambda k: False)


def test_drive_halving_reuses_half_as_prod_and_rechecks_q2_at_new_factor():
    (c,), store, ex = run_cases(lambda cid, r, f: 'stepbad' if f == 1. else 'ok')
    assert (c.terminal, c.rounds, c.scored) == ('CONVERGED', 2, 'FC1|0|cert|gauss160|0.5')
    keys = [q['key'] for q in ex.calls]
    assert keys.count('FC1|0|cert|gauss160|0.5') == 1                       # requested once (as half), reused as prod
    assert 'FC1|0|tight|gauss160|0.5' in keys                                 # the promoted half gets its tight solve then
    assert 'FC1|0|succ|gauss192|0.5' in keys                                  # Q2 re-checked on its own reached states
    assert c.history[0]['value'] == ('gauss160', .5)


def test_drive_step_unconverged_at_quarter():
    (c,), _, ex = run_cases(lambda *a: 'stepbad')
    assert (c.terminal, c.rounds, c.scored) == ('STEP-UNCONVERGED', 3, 'FC1|0|cert|gauss160|0.25')
    assert min(q['f'] for q in ex.calls) == .25                              # never below the finest factor


def test_drive_escalation_keeps_factor_then_q2_fail():
    (c,), _, ex = run_cases(lambda cid, r, f: 'stepbad' if f == 1. else 'q2bad')
    assert [h['value'] for h in c.history] == [('gauss160', .5), ('gauss192', .5), ('gauss256', .5), 'Q2-FAIL']
    assert (c.terminal, c.scored) == ('Q2-FAIL', 'FC1|0|cert|gauss256|0.5')
    assert 'FC1|0|succ|gauss384|0.5' in [q['key'] for q in ex.calls]


@pytest.mark.parametrize('o,label', [('prodinv', 'INVALID'), ('halfinv', 'STEP-UNAVAILABLE')])
def test_drive_invalid_and_unavailable(o, label):
    (c,), _, _ = run_cases(lambda *a: o)
    assert c.terminal == label
    assert c.scored == (None if label == 'INVALID' else 'FC1|0|cert|gauss160|1')


@pytest.mark.parametrize('o', ['q2bad', 'rhobad', 'succinv'])
def test_drive_q2_variants_walk_the_chain(o):
    (c,), _, _ = run_cases(lambda *a: o, selected=None)
    assert c.uncertified_start and c.terminal == 'Q2-FAIL' and c.rounds == 8
    assert [h['rule'] for h in c.history] == list(SM.CHAIN)


def test_drive_invalid_after_halving_and_escalation_scored_none():
    out = {('gauss160', 1.): 'stepbad', ('gauss160', .5): 'q2bad', ('gauss192', .5): 'prodinv'}
    (c,), _, ex = run_cases(lambda cid, r, f: out.get((r, f), 'ok'))
    assert c.terminal == 'INVALID' and c.scored is None and c.latest_valid_prod == 'FC1|0|cert|gauss160|0.5'
    assert c.rounds == 3


def test_promoted_half_invalid_is_step_unavailable_not_invalid():
    # the half run IS the next prod: if it is invalid the coarser state ends STEP-UNAVAILABLE (precedence (4))
    (c,), _, _ = run_cases(lambda cid, r, f: 'stepbad' if f == 1. else 'prodinv')
    assert c.terminal == 'STEP-UNAVAILABLE' and c.scored == 'FC1|0|cert|gauss160|1'


def test_round_cap_labels_open_state_after_full_round():
    (c,), _, _ = run_cases(lambda *a: 'q2bad', selected=None, max_rounds=2)
    assert (c.terminal, c.rounds, c.scored) == ('ROUND-CAP', 2, 'FC1|0|cert|fib6765|1')


def test_round_cap_does_not_override_terminal_in_last_round():
    (c,), _, _ = run_cases(lambda *a: 'ok', max_rounds=1)
    assert c.terminal == 'CONVERGED'


def test_budget_cap_before_any_run():
    ex = Synth(lambda *a: 'ok')
    cs = [SM.Case('FC1|0', 'gauss160')]
    SM.drive(cs, ex, over_budget=lambda: True)
    assert (cs[0].terminal, cs[0].scored, ex.calls) == ('BUDGET-CAP', None, [])


def test_budget_cap_mid_round_scores_latest_valid_prod():
    # case 0 completes round 1 (stepbad), case 1 too; budget dies after 2 runs of round 2
    cs, _, ex = run_cases(lambda *a: 'stepbad', n=2, budget_after=10)
    assert [c.terminal for c in cs] == ['BUDGET-CAP', 'BUDGET-CAP']
    assert cs[0].scored == 'FC1|0|cert|gauss160|0.5' and cs[1].scored == 'FC1|1|cert|gauss160|0.5'
    assert len(ex.calls) == 10


def test_budget_cap_prod_of_current_round_counts():
    cs, _, ex = run_cases(lambda *a: 'ok', n=1, budget_after=1)              # prod done, succ not run
    assert cs[0].terminal == 'BUDGET-CAP' and cs[0].scored == 'FC1|0|cert|gauss160|1'


def test_budget_cap_does_not_hit_finished_cases():
    out = lambda cid, r, f: 'ok' if cid == 'FC1|0' else 'stepbad'
    cs, _, _ = run_cases(out, n=2, budget_after=7)
    assert cs[0].terminal == 'CONVERGED' and cs[1].terminal == 'BUDGET-CAP'


# ----------------------------------------------------------------------------------------------- the 10-round bound
OUTCOMES = ('ok', 'stepbad', 'q2bad', 'prodinv', 'halfinv')


def all_paths(selected):
    """Every outcome sequence (adversarial, per round) from a start: returns the set of (terminal, rounds)."""
    res = []

    def rec(prefix):
        script = list(prefix)

        def outcome(cid, r, f, _s=script, _seen={}):
            k = (r, f)
            if k not in _seen:
                _seen[k] = _s[len(_seen)] if len(_seen) < len(_s) else 'ok'
            return _seen[k]
        (c,), _, _ = run_cases(outcome, selected=selected)
        res.append((c.terminal, c.rounds))
        if c.rounds == len(prefix) and c.terminal is not None and len(prefix) < 12:
            # the last scripted outcome decided the terminal: try every outcome at that depth instead (DFS over paths)
            pass
        return c
    # breadth: every script of length <= 10 over transition outcomes (terminals end early)
    for n in range(1, 11):
        for s in itertools.product(('stepbad', 'q2bad'), repeat=n - 1):
            for last in OUTCOMES:
                rec(s + (last,))
    return res


@pytest.mark.parametrize('selected', [None] + list(SM.CHAIN))
def test_ten_round_bound_never_binding(selected):
    res = all_paths(selected)
    assert all(t != 'ROUND-CAP' and t in SM.TERMINALS for t, _ in res)
    assert max(r for _, r in res) <= 10
    if selected is None:
        assert max(r for _, r in res) == 10                                  # 1 + 7 escalations + 2 halvings


def test_every_terminal_reachable():
    seen = set()
    for sel in [None] + list(SM.CHAIN):
        seen |= {t for t, _ in all_paths(sel)}
    seen |= {'ROUND-CAP', 'BUDGET-CAP'}                                       # covered by the cap tests above
    assert seen == set(SM.TERMINALS)


def test_path_enumeration_and_solve_inventory():
    ps = SM.paths(None)
    assert max(map(len, ps)) == 10
    longest = [p for p in ps if len(p) == 10]
    assert all(p[-1] == ('gauss256', .25) for p in longest)
    p = [('gauss160', 1.), ('gauss160', .5), ('gauss192', .5)]
    inv = SM.path_solves(p)
    assert inv.count(('prod', 'gauss160', .5)) == 0 and ('half', 'gauss160', .5) in inv     # promoted half not re-solved
    assert ('tight', 'gauss160', .5) in inv and len(inv) == 4 + 3 + 4
    assert len(SM.paths('gauss256')) == 3


# ----------------------------------------------------------------------------------------------- C-B1 hash guard
def _fake_bank(d, label='a'):
    d.mkdir(parents=True)
    (d / 'ckpt.pkl').write_bytes(b'ckpt-bytes')
    (d / 'rotation_R512.npz').write_bytes(b'rot-bytes')
    m = dict(label=label, seed=1, ckpt_sha256=BG.sha_file(d / 'ckpt.pkl'), rotation_sha256=BG.sha_file(d / 'rotation_R512.npz'))
    (d / 'BANK.json').write_text(json.dumps(m))
    return m


def test_cb1_guard_accepts_matched_and_rejects_every_mismatch(tmp_path):
    m = _fake_bank(tmp_path / 'sa')
    assert BG.verify({'dir': 'sa'}, tmp_path, expect_label='a')['ckpt_sha256'] == m['ckpt_sha256']
    assert BG.must_reject({'dir': 'sa'}, tmp_path)['fired'] is False
    # swapped hashes in BANK.json (the deliberate mismatch)
    (tmp_path / 'sa' / 'BANK.json').write_text(json.dumps(dict(m, ckpt_sha256=m['rotation_sha256'], rotation_sha256=m['ckpt_sha256'])))
    assert BG.must_reject({'dir': 'sa'}, tmp_path)['fired'] is True
    (tmp_path / 'sa' / 'BANK.json').write_text(json.dumps(m))
    # config hash differs from BANK.json
    assert BG.must_reject({'dir': 'sa', 'ckpt_sha256': '0' * 64}, tmp_path)['fired'] is True
    # wrong expected label (bank mix-up)
    assert BG.must_reject({'dir': 'sa'}, tmp_path, expect_label='b')['fired'] is True
    # file changed after registration
    (tmp_path / 'sa' / 'rotation_R512.npz').write_bytes(b'other')
    assert BG.must_reject({'dir': 'sa'}, tmp_path)['fired'] is True
    # no BANK.json; malformed hash
    shutil.rmtree(tmp_path / 'sa')
    _fake_bank(tmp_path / 'sb')
    (tmp_path / 'sb' / 'BANK.json').unlink()
    assert BG.must_reject({'dir': 'sb'}, tmp_path)['fired'] is True
    (tmp_path / 'sb' / 'BANK.json').write_text(json.dumps(dict(ckpt_sha256='x', rotation_sha256='y')))
    assert BG.must_reject({'dir': 'sb'}, tmp_path)['fired'] is True


def test_cb1_real_k1_bank_and_mismatched_bank_json(tmp_path):
    v = BG.verify({'dir': 'inputs/banks/s1'}, ROM, expect_label='1')
    assert v['ckpt_sha256'].startswith('244e1107') and v['rotation_sha256'].startswith('73bb603b')
    d = tmp_path / 'k1bad'
    d.mkdir()
    for nm in ('ckpt.pkl', 'rotation_R512.npz'):
        (d / nm).symlink_to(ROM / 'inputs' / 'banks' / 's1' / nm)
    meta = json.loads((ROM / 'inputs' / 'banks' / 's1' / 'BANK.json').read_text())
    (d / 'BANK.json').write_text(json.dumps(dict(meta, rotation_sha256=meta['ckpt_sha256'])))
    assert BG.must_reject({'dir': str(d)}, '/')['fired'] is True


# ----------------------------------------------------------------------------------------------- fresh-cohort table guard
def fixture_table():
    FIXTURE.parent.mkdir(exist_ok=True)
    if not FIXTURE.exists():
        FC.write_fixture(FIXTURE)
    return json.loads(FIXTURE.read_text())


def test_fixture_table_loads_and_splits():
    t = fixture_table()
    assert t['fixture'] is True and t['seed_root'] != 0x494D5046
    cs = FC.case_list(['FC1|0', 'FC1|1', 'FI|2', 'FN0s|1', 'FC2|0'], FIXTURE)
    assert [c[4] for c in cs] == ['dev', 'dev', 'val', 'dev', 'dev']
    with pytest.raises(FC.CaseError):
        FC.case_list(['FC1|4'], FIXTURE)                                     # test-once without the flag
    assert FC.case_list(['FC1|4'], FIXTURE, test_once=True)[0][4] == 'test'
    assert FC.anchors_in(['FI|4', 'FC1|5', 'FC1|0']) == ['FI|4', 'FC1|5']
    for bad in (['C1|0'], ['FC1|0', 'FC1|0'], ['FC1|99']):
        with pytest.raises(FC.CaseError):
            FC.case_list(bad, FIXTURE)


def test_table_guard_rejects_tampering(tmp_path):
    t = fixture_table()
    p = tmp_path / 't.json'
    t2 = json.loads(json.dumps(t))
    t2['cohorts']['FC1'][0]['a'] += 1e-9
    p.write_text(json.dumps(t2))
    with pytest.raises(FC.CaseError):
        FC.load_table(p)
    t3 = json.loads(json.dumps(t))
    t3['cohorts']['FC1'][0]['split'] = 'test'
    t3['sha256'] = FC.table_sha(t3['cohorts'])
    p.write_text(json.dumps(t3))
    with pytest.raises(FC.CaseError):
        FC.load_table(p)
    with pytest.raises(FC.CaseError):
        FC.load_table(tmp_path / 'missing.json')


# ----------------------------------------------------------------------------------------------- selection configs, cost
def test_selection_config_generator_reproduces_e2s1():
    import mkconfigs_sel as MS
    assert MS.text(MS.sel_config('1')) == (ROM / 'configs' / 'e2s1.json').read_text()


def test_cost_certificate_selftest():
    import resources as RES
    RES.selftest()


def test_collector_writes_exhaustion_records_for_killed_job(tmp_path, monkeypatch):
    import fpolicy as FP
    import fcollect as FCL
    ex = Synth(lambda *a: 'stepbad')
    cs = [SM.Case(f'FC1|{i}', 'gauss160') for i in range(2)]
    seen = {}

    def killed_after_round_1(cases, store):          # the job dies after persisting round 1
        seen.update(store)
        FP.dump(dict(cases=[c.as_dict() for c in cases], records=store, anchors=[], elapsed=1.,
                     rows=[dict(seconds=1.)]), tmp_path / 'progress.json')
        raise SystemExit
    with pytest.raises(SystemExit):
        SM.drive(cs, ex, on_round=killed_after_round_1)
    for k in seen:                                   # make the persisted prods look like real cert records
        if '|cert|' in k:
            seen[k].update(gcert_ok=True, steps=100, noncertified=0)
    pg = json.loads((tmp_path / 'progress.json').read_text())
    pg['records'] = seen
    (tmp_path / 'progress.json').write_text(json.dumps(pg))
    monkeypatch.setattr(sys, 'argv', ['fcollect.py', str(tmp_path)])
    FCL.main()
    r = json.loads((tmp_path / 'result.json').read_text())
    assert r['killed'] and r['counts']['BUDGET-CAP'] == 2 and r['qualified'] == 0
    assert [p['scored'] for p in r['policy']] == ['FC1|0|cert|gauss160|0.5', 'FC1|1|cert|gauss160|0.5']


def test_selection_config_r384_group_identical_to_e2s1():
    import mkconfigs_sel as MS
    full = json.loads((ROM / 'configs' / 'e2s1.json').read_text())
    c = MS.sel_config('1', r384=True)
    assert [g['Rp'] for g in c['groups']] == [384]
    assert c['groups'] == [g for g in full['groups'] if g['Rp'] == 384]
    assert {k: v for k, v in c.items() if k != 'groups'} == {k: v for k, v in full.items() if k != 'groups'}


# ----------------------------------------------------------------------------------------------- audit B-1 fixes
def test_schedule_definition_and_policy_config_validation():
    import schedule as SCH
    import mkconfigs_pol as MP
    assert abs(SCH.total('S1') - 15.55) < 1e-9 and abs(SCH.total('S3') - 9.85) < 1e-9 and SCH.total('S1') > SCH.CAP
    assert len(SCH.POLICY_IDS) == 40 and SCH.POLICY_IDS[:16] == SCH.DEVVAL_IDS and SCH.CEIL['policy_K1'] == .5
    c = MP.make('1', 'S3')
    assert c['cases'] == list(SCH.POLICY_IDS) and c['ceiling_hours'] == .5 and c['selected'] == 'gauss160'
    for bad in (dict(cases=c['cases'][:39]), dict(ceiling_hours=.4), dict(max_rounds=1), dict(schedule='S2'),
                dict(bank_label='a', schedule='S3')):
        with pytest.raises(ValueError):
            SCH.check_policy_config(dict(c, **bad))
    with pytest.raises(AssertionError):
        MP.make('1', 'S2')


def test_incomplete_selection_persists_explicit_null(tmp_path, monkeypatch):
    import mkconfigs_pol as MP
    import select2d_r384 as S384
    out = tmp_path / 'r2acc_e2sb'
    out.mkdir()
    assert S384.incomplete(out, 'b') == 'result.json missing'
    (out / 'result.json').write_text(json.dumps(dict(complete=False)))
    assert 'not complete' in S384.incomplete(out, 'b')
    monkeypatch.setattr(MP, 'HERE', tmp_path)
    with pytest.raises(FileNotFoundError):
        MP.selected_of('b')                                   # no persisted outcome -> refused, never silently null
    (tmp_path / 'checks').mkdir()
    (tmp_path / 'checks' / 'select2d-b.json').write_text(json.dumps(dict(selected={'384': dict(rule=None, outcome=S384.INCOMPLETE)})))
    assert MP.selected_of('b') == (None, S384.INCOMPLETE)
    (tmp_path / 'checks' / 'select2d-b.json').write_text(json.dumps(dict(selected={'384': dict(rule='gauss128', outcome='selected')})))
    assert MP.selected_of('b')[0] == 'gauss128'
    assert SM.start_state(None) == ('fib4181', 1., True)


def test_collector_without_progress_json(tmp_path, monkeypatch):
    import fcollect as FCL
    cfg = dict(cases=['FC1|0', 'FC1|1'], selected=None)
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'RUN.json').write_text(json.dumps(dict(config=cfg)))
    monkeypatch.setattr(sys, 'argv', ['fcollect.py', str(tmp_path / 'a')])
    FCL.main()
    r = json.loads((tmp_path / 'a' / 'result.json').read_text())
    assert r['counts']['BUDGET-CAP'] == 2 and r['collected_from'] == 'config' and all(p['scored'] is None for p in r['policy'])
    cp = tmp_path / 'cfg.json'
    cp.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, 'argv', ['fcollect.py', str(tmp_path / 'b'), str(cp)])   # died before writing anything
    FCL.main()
    assert json.loads((tmp_path / 'b' / 'result.json').read_text())['counts']['BUDGET-CAP'] == 2


def test_test_once_registration(tmp_path):
    import testonce as TO
    bank = dict(ckpt_sha256='a' * 64, rotation_sha256='b' * 64)
    cfg = dict(attempt='pol1', test_once=True, cases=['FC1|4'])
    reg = tmp_path / 'reg'
    with pytest.raises(TO.TestOnceError):
        TO.require(bank, cfg, 'c' * 64, tmp_path / 'out', reg=reg)          # unregistered
    k, _ = TO.register(bank, cfg, 'c' * 64, reg=reg)
    with pytest.raises(TO.TestOnceError):
        TO.register(bank, cfg, 'c' * 64, reg=reg)                          # second evaluation refused
    assert TO.require(bank, cfg, 'c' * 64, tmp_path / 'out', reg=reg) == k
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out' / 'RUN.json').write_text('{}')
    with pytest.raises(TO.TestOnceError):
        TO.require(bank, cfg, 'c' * 64, tmp_path / 'out', reg=reg)          # existing output refused
    k2, _ = TO.key_of(bank, cfg, 'd' * 64)
    assert k2 != k                                                           # another cohort table -> another key
    assert TO.key_of(bank, dict(cfg, note='x'), 'c' * 64)[0] == k            # notes do not change the key


def test_register_arm_writes_canonical_names(tmp_path):
    import register_arm as RA
    src = tmp_path / 'train2d_s1'
    (src / 'stageB').mkdir(parents=True)
    (src / 'rotation').mkdir()
    (src / RA.CK).write_bytes(b'c')
    (src / RA.ROT).write_bytes(b'r')
    m = RA.write_bank(src, tmp_path / 'sb', dict(label='b', seed=1))
    assert m['ckpt'] == 'ckpt.pkl' and m['rotation'] == 'rotation_R512.npz'
    assert BG.verify({'dir': 'sb'}, tmp_path, expect_label='b')['ckpt_sha256'] == m['ckpt_sha256']
    meta = json.loads((tmp_path / 'sb' / 'BANK.json').read_text())
    (tmp_path / 'sb' / 'BANK.json').write_text(json.dumps(dict(meta, ckpt='../x.pkl')))
    assert BG.must_reject({'dir': 'sb'}, tmp_path)['fired']


def test_render_templates_are_valid_bash(tmp_path):
    import subprocess
    for args in (['sel', 'b'], ['pol', 'K1', 'pol1'], ['smr']):
        t = subprocess.run([sys.executable, str(LANE / 'cluster' / 'rom' / 'render.py'), *args], capture_output=True,
                           text=True, check=True).stdout
        assert '{{' not in t and '}}' not in t and '--partition=gpu' in t and 'preempt' not in t
        assert 'jax_backend' in t and 'exit(0 if b == ' in t and 'JAX_DEFAULT_MATMUL_PRECISION=highest' in t
        f = tmp_path / 'x.sh'
        f.write_text(t)
        subprocess.run(['bash', '-n', str(f)], check=True)
    smr = subprocess.run([sys.executable, str(LANE / 'cluster' / 'rom' / 'render.py'), 'smr'], capture_output=True,
                         text=True, check=True).stdout
    assert 'smkpolK1' in smr and 'smkdryK1' in smr and 'refjob.py' in smr and 'gpu:h200' in smr and '00:24:00' in smr
