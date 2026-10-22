"""imp-fronts Phase B unit tests (C-B2 stop-rule evaluator with synthetic inputs; fcohorts bounds and C-B5 freshness incl.
the must-fail injection; E9 scoring helpers). CPU only.

    JAX_PLATFORMS=cpu bash .../jaxslot.sh 3600 /home/tahmid/Dev/.venv/bin/python -m pytest tests/test_phaseb.py -q
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

LANE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LANE / 'phaseb'))
sys.path.insert(0, str(LANE))
import verdict as V  # noqa: E402
import fcohorts as FC  # noqa: E402
import e9 as E9  # noqa: E402

INF = float('inf')
FI = [str(j) for j in range(4, 12)]
FN = [str(j) for j in range(4, 12)]
C1 = [str(j) for j in range(4, 12)]


def bank(fi=1.0, fn=1.0, e=1.0, tau=1.0, qual=True, nodes=25600, cert=True, rom=True):
    """Synthetic bank: constant floors / errors / tails on every case (overridable per case afterwards)."""
    return dict(trained=True, rom_available=rom,
                floors=dict(FI={c: dict(f384=fi) for c in FI}, FN0s={c: dict(f384=fn) for c in FN}),
                policy=dict(FC1={c: dict(e=e, valid=math.isfinite(e), qualified=qual) for c in C1},
                            FI={c: dict(tau=tau, valid=math.isfinite(tau), qualified=qual) for c in FI}),
                selection=dict(certified=cert, nodes=nodes, rule='gauss160'))


ELIG = dict(FI=[int(c) for c in FI], FN0s=[int(c) for c in FN], FC1=[int(c) for c in C1])
K1 = bank()
ELIG_V = dict(ELIG, FI=ELIG['FI'][:6])          # verdict(): scheduled references only (Amendment 10)


def ev(name, arm, ctrl=None, elig=None):
    return V.evaluate(name, arm, ctrl or K1, elig or ELIG)


# ------------------------------------------------------------------------------------------- C-B2: arm (a) ---
@pytest.mark.parametrize('fn,want', [(1 / 3 - 1e-12, V.CONT), (0.25, V.CONT), (0.5, V.STOP_E), (1.0, V.STOP_E)])
def test_floor_route_both_sides(fn, want):
    assert ev('a', bank(fn=fn))['outcome'] == want


def test_floor_route_exact_tie_at_factor():
    ctrl = bank(fn=3.0)
    assert ev('a', bank(fn=1.0), ctrl)['routes']['floor']['status'] == V.PASS        # 1.0 <= 3.0 / 3 exactly
    assert ev('a', bank(fn=np.nextafter(1.0, 2.)), ctrl)['routes']['floor']['status'] == V.FAIL


def test_floor_route_min_population():
    el = dict(ELIG, FN0s=ELIG['FN0s'][:4])
    r = ev('a', bank(fn=0.1), elig=el)
    assert r['routes']['floor']['status'] == V.UNAV and r['outcome'] == V.INCONC
    el = dict(ELIG, FN0s=ELIG['FN0s'][:5])
    assert ev('a', bank(fn=0.1), elig=el)['outcome'] == V.CONT


@pytest.mark.parametrize('bad', [INF, float('nan'), None])
def test_floor_route_missing_or_nonfinite(bad):
    arm = bank(fn=0.1)
    arm['floors']['FN0s']['5'] = dict(f384=bad)
    assert ev('a', arm)['routes']['floor']['status'] == V.UNAV
    ctrl = bank()
    ctrl['floors']['FN0s']['5'] = dict(f384=bad)
    assert ev('a', bank(fn=0.1), ctrl)['routes']['floor']['status'] == V.UNAV
    arm = bank(fn=0.1)
    del arm['floors']['FN0s']['6']
    assert ev('a', arm)['routes']['floor']['status'] == V.UNAV


# ------------------------------------------------------------------------------------------------- guard ---
def test_guard_both_sides_and_tie():
    assert ev('a', bank(fn=0.1, fi=2.0))['guard']['status'] == V.PASS                 # exactly 2x: not "more than"
    assert ev('a', bank(fn=0.1, fi=2.0))['outcome'] == V.CONT
    r = ev('a', bank(fn=0.1, fi=np.nextafter(2.0, 3.)))
    assert r['guard']['status'] == V.FAIL and r['outcome'] == V.STOP_G


def test_guard_min_population():
    el = dict(ELIG, FI=ELIG['FI'][:5])
    r = ev('a', bank(fn=0.1, fi=10.), elig=el)
    assert r['guard']['status'] == V.UNAV and r['outcome'] == V.INCONC                # not evaluable: no STOP


def test_guard_missing_floor_unavailable_but_violation_still_stops():
    arm = bank(fn=0.1)
    arm['floors']['FI']['4'] = dict(f384=INF)
    r = ev('a', arm)
    assert r['guard']['status'] == V.UNAV and r['outcome'] == V.INCONC                # route PASS, guard UNAV
    arm = bank(fn=0.1, fi=5.0)
    arm['floors']['FI']['4'] = dict(f384=None)
    r = ev('a', arm)
    assert r['guard']['status'] == V.FAIL and r['outcome'] == V.STOP_G               # Amendments 5, 8
    arm = bank(fn=0.1, fi=2.0)
    arm['floors']['FI']['4'] = dict(f384=None)                                       # 2.0 = 2 x complete K1 max: no stop
    assert ev('a', arm)['guard']['status'] == V.UNAV
    ctrl = bank()
    ctrl['floors']['FI']['5'] = dict(f384=INF)                                       # K1 incomplete: no early stop
    arm = bank(fn=0.1, fi=5.0)
    arm['floors']['FI']['4'] = dict(f384=None)
    assert ev('a', arm, ctrl)['guard']['status'] == V.UNAV
    ctrl = bank()
    ctrl['floors']['FI']['5'] = dict(f384=0.1)                                       # complete K1 max = 1.0
    arm = bank(fn=0.1, fi=2.5)
    arm['floors']['FI']['4'] = dict(f384=float('nan'))
    assert ev('a', arm, ctrl)['guard']['status'] == V.FAIL
    ctrl = bank()
    del ctrl['floors']['FI']['7']
    assert ev('a', bank(fn=0.1), ctrl)['guard']['status'] == V.UNAV


def test_guard_precedes_endpoint():
    r = ev('a', bank(fn=0.1, fi=3.0))
    assert r['routes']['floor']['status'] == V.PASS and r['outcome'] == V.STOP_G


# ------------------------------------------------------------------------------------------- C-B2: arm (b) ---
@pytest.mark.parametrize('e,want', [(0.5, V.CONT), (0.4, V.CONT), (0.6, V.STOP_E)])
def test_rom_route_both_sides_and_tie(e, want):
    assert ev('b', bank(e=e))['outcome'] == want


def test_rom_route_unqualified_case_fails_even_if_better():
    arm = bank(e=0.01)
    arm['policy']['FC1']['9']['qualified'] = False
    r = ev('b', arm)
    assert r['routes']['rom']['status'] == V.FAIL and r['outcome'] == V.STOP_E
    arm = bank(e=0.01)
    del arm['policy']['FC1']['9']                                                     # missing record = not qualified
    assert ev('b', arm)['routes']['rom']['status'] == V.FAIL


def test_rom_route_arm_invalid_infinite():
    arm = bank(e=0.01)
    arm['policy']['FC1']['4'] = dict(e=INF, valid=False, qualified=False)
    assert ev('b', arm)['outcome'] == V.STOP_E


def test_rom_route_control_infinite_median_unavailable():
    ctrl = bank()
    for c in C1[:4]:
        ctrl['policy']['FC1'][c] = dict(e=None, valid=False, qualified=False)        # 4 of 8 invalid -> median inf
    r = ev('b', bank(e=0.01), ctrl)
    assert r['routes']['rom']['status'] == V.UNAV and r['outcome'] == V.INCONC
    ctrl = bank()
    for c in C1[:3]:
        ctrl['policy']['FC1'][c] = dict(e=INF, valid=False, qualified=False)        # 3 of 8 -> median finite
    assert ev('b', bank(e=0.01), ctrl)['routes']['rom']['status'] == V.PASS


def test_rom_route_control_unqualified_still_counts():
    ctrl = bank(qual=False)                                                           # diagnostic median regardless
    assert ev('b', bank(e=0.5), ctrl)['outcome'] == V.CONT


def test_rom_route_min_population_and_missing_control():
    assert ev('b', bank(e=0.1), elig=dict(ELIG, FC1=ELIG['FC1'][:5]))['routes']['rom']['status'] == V.UNAV
    ctrl = bank()
    del ctrl['policy']['FC1']['11']
    assert ev('b', bank(e=0.1), ctrl)['routes']['rom']['status'] == V.UNAV
    assert ev('b', bank(e=0.1, rom=False))['routes']['rom']['status'] == V.UNAV


def test_rom_route_uncertified_start_evaluated_normally():
    r = ev('b', bank(e=0.1, cert=False))                                              # Amendment 4
    assert r['routes']['rom']['status'] == V.PASS and r['outcome'] == V.CONT


# ------------------------------------------------------------------------------------------- C-B2: arm (c) ---
def test_mstar_route_bar_and_tie():
    assert ev('c', bank(nodes=12800, tau=1.))['routes']['mstar']['status'] == V.PASS  # exactly 25600 / 2
    assert ev('c', bank(nodes=12801, tau=1.))['routes']['mstar']['status'] == V.FAIL
    assert ev('c', bank(nodes=96 * 96, tau=1.))['outcome'] == V.CONT
    r = ev('c', bank(nodes=96 * 96, cert=False, tau=1.))
    assert r['routes']['mstar']['status'] == V.FAIL and r['outcome'] == V.STOP_E      # tail FAIL too


@pytest.mark.parametrize('tau,want', [(1 / 3, V.PASS), (0.3, V.PASS), (0.34, V.FAIL)])
def test_tail_route_both_sides(tau, want):
    ctrl = bank(tau=1.0)
    r = ev('c', bank(tau=tau, nodes=25600), ctrl)
    assert r['routes']['tail']['status'] == want


def test_tail_route_exact_tie():
    assert ev('c', bank(tau=1.0), bank(tau=3.0))['routes']['tail']['status'] == V.PASS


def test_either_c_route_suffices():
    assert ev('c', bank(nodes=12800, tau=1.0))['outcome'] == V.CONT                   # m* PASS, tail FAIL
    assert ev('c', bank(nodes=25600, tau=0.1))['outcome'] == V.CONT                   # m* FAIL, tail PASS
    assert ev('c', bank(nodes=25600, tau=1.0))['outcome'] == V.STOP_E                 # both FAIL


def test_mstar_pass_tail_unavailable_continue_if_guard_pass():
    ctrl = bank()
    for c in FI[:3]:
        ctrl['policy']['FI'][c] = dict(tau=None, valid=False, qualified=False)        # 5 common < 6 -> UNAV
    r = ev('c', bank(nodes=9216), ctrl)
    assert r['routes']['tail']['status'] == V.UNAV and r['routes']['mstar']['status'] == V.PASS
    assert r['outcome'] == V.CONT
    arm = bank(nodes=9216)
    arm['floors']['FI']['4'] = dict(f384=None)                                       # guard UNAV -> INCONCLUSIVE
    assert ev('c', arm, ctrl)['outcome'] == V.INCONC


def test_guard_violated_and_endpoint_unavailable_stops():
    ctrl = bank()
    for c in FI[:3]:
        ctrl['policy']['FI'][c] = dict(tau=None, valid=False, qualified=False)
    r = ev('c', bank(fi=5.0, nodes=25600, cert=False), ctrl)
    assert r['routes']['tail']['status'] == V.UNAV and r['outcome'] == V.STOP_G
    r = ev('a', bank(fi=5.0), elig=dict(ELIG, FN0s=[4]))
    assert r['routes']['floor']['status'] == V.UNAV and r['outcome'] == V.STOP_G


def test_tail_common_set_requires_both_valid():
    arm = bank(tau=0.1)
    arm['policy']['FI']['4'] = dict(tau=INF, valid=False, qualified=False)
    r = ev('c', arm)
    assert r['routes']['tail']['status'] == V.FAIL                                   # qualification over every case
    assert r['routes']['tail']['n'] == 7
    ctrl = bank(tau=1.0)
    ctrl['policy']['FI']['5']['tau'] = 1e9                                            # in the common set
    ctrl['policy']['FI']['6'] = dict(tau=None, valid=False)
    r = ev('c', bank(tau=0.3), ctrl)
    assert r['routes']['tail']['n'] == 7 and r['routes']['tail']['ctrl_median'] == 1.0


def test_tail_route_unqualified_fails():
    arm = bank(tau=0.01)
    arm['policy']['FI']['11']['qualified'] = False
    assert ev('c', arm)['routes']['tail']['status'] == V.FAIL


def test_not_run():
    assert ev('a', None)['outcome'] == V.NOT_RUN
    assert ev('b', dict(trained=False))['outcome'] == V.NOT_RUN


def test_decide_table():
    P, F_, U = V.PASS, V.FAIL, V.UNAV
    assert V.decide(False, P, {'x': P}) == (V.NOT_RUN, 1)
    assert V.decide(True, F_, {'x': P}) == (V.STOP_G, 2)
    assert V.decide(True, U, {'x': F_, 'y': F_}) == (V.STOP_E, 3)
    assert V.decide(True, P, {'x': F_, 'y': P}) == (V.CONT, 4)
    assert V.decide(True, U, {'x': P}) == (V.INCONC, 5)
    assert V.decide(True, P, {'x': U, 'y': F_}) == (V.INCONC, 5)


def test_resolved_wins_ties_losses():
    r = V.resolved({'1': 1.0, '2': 1.0, '3': 2.0, '4': INF, '5': 1.0, '6': INF},
                   {'1': 2.0, '2': 1.1, '3': 1.0, '4': 1.0, '5': 1.0, '6': INF},
                   {'1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1, '6': 0.1})
    assert r == dict(wins=1, ties=1, losses=2, unresolved=1, both_invalid=1)


def test_passes_against_all_K1_K3():
    inp = dict(arms=dict(a=bank(fn=0.1)), controls=dict(K1=bank(), K2=bank(), K3=bank(fn=0.6)), eligible=ELIG_V)
    out = V.verdict(inp)
    assert out['arms']['a']['outcome'] == V.CONT and out['arms']['a']['passes_against_all_K1_K3'] is True
    inp['controls']['K3'] = bank(fn=0.25)
    out = V.verdict(inp)
    assert out['arms']['a']['outcome'] == V.CONT and out['arms']['a']['passes_against_all_K1_K3'] is False
    assert out['arms']['b']['outcome'] == V.NOT_RUN and out['arms']['b']['passes_against_all_K1_K3'] is False


# --------------------------------------------------------------------------------------------- fcohorts ---
def test_fresh_rows_in_bounds_and_split():
    tab = FC.table()
    for n in FC.COHORTS:
        rows = tab['cohorts'][n]
        assert len(rows) == FC.DRAWS[n]
        assert [r['split'] for r in rows[:4]] == ['dev', 'dev', 'val', 'val']
        assert all(r['split'] == 'test' for r in rows[4:])
        for r in rows:
            FC.check_bounds(n, [r[k] for k in FC.KEYS])
    assert sum(r['split'] == 'test' for n in FC.COHORTS for r in tab['cohorts'][n]) == 28
    s = {n: [r['s'] for r in tab['cohorts'][n]] for n in FC.COHORTS}
    assert max(s['FI']) <= 200 + 1e-9 and min(s['FC1']) >= 250 - 1e-9 and max(s['FC2']) <= 1000 + 1e-9


@pytest.mark.parametrize('name,row', [('FI', [.5, .5, .1, 1., .2]), ('FN0s', [.5, .5, .1, 1.4, .011]),
                                      ('FC1', [.5, .5, .1, .5, .005]), ('FC2', [.5, .5, .1, 2.5, .003]),
                                      ('FI', [.9, .5, .1, 1., .05]), ('FC1', [.5, .5, .1, 1., float('nan')])])
def test_bounds_reject(name, row):
    with pytest.raises(FC.FreshError):
        FC.check_bounds(name, row)


def test_draw_order_and_seed_root():
    r = np.random.default_rng([FC.TAG, 2, 7])
    cx, cy, w = r.uniform(.15, .85), r.uniform(.15, .85), r.uniform(.05, .2)
    nu = 10. ** r.uniform(-2.5, -2.15)
    a = r.uniform(250 * nu, min(2., 500 * nu))
    assert np.array_equal(FC.draw_row('FC1', 7), np.array([cx, cy, w, a, nu]))
    assert FC.canonical_sha(FC.table()) == FC.canonical_sha(FC.table())


def test_freshness_synthetic_and_within_duplicates():
    tab = FC.table()
    row = FC.rows_array(tab, 'FC2')[5]
    allow = {'x': (['x0', 'x1'], np.array([[.3, .3, .1, 1., .05], row * (1 + 5e-10)]))}
    rep = FC.freshness(tab, allow)
    assert not rep['passed'] and rep['hits'][0]['fresh'] == 'FC2|5|test' and rep['hits'][0]['other'] == 'x1'
    allow = {'x': (['x1'], np.array([row * (1 + 5e-9)]))}
    assert FC.freshness(tab, allow)['passed']
    t2 = copy.deepcopy(tab)
    t2['cohorts']['FI'][9] = dict(t2['cohorts']['FI'][1], j=9, split='test')       # dev row duplicated into test
    rep = FC.freshness(t2, {})
    assert not rep['passed'] and rep['within_fresh_duplicates']


def test_freshness_e0_injection_must_fail():
    tab = FC.table()
    e0 = FC.e0_rows()
    allow = {'E0_dev6_val32': (list(e0), np.array(list(e0.values())))}
    assert FC.freshness(tab, allow)['passed']
    inj = FC.freshness(tab, allow, inject=('INJECT|val32|0', e0['val32|0']))
    assert not inj['passed'] and any(h['other'] == 'val32|0' for h in inj['hits'])


def test_coverage_rows_bounds():
    C = FC.coverage_rows()
    s = C[:, 3] / C[:, 4]
    assert len(C) == 4608 and s.max() <= 1000 + 1e-9 and C[:, 4].min() >= .0016 - 1e-15 and s.max() >= 900


def test_written_table_matches_generator():
    if not FC.OUT.exists():
        pytest.skip('fresh_cohorts.json not yet written')
    tab = FC.load()
    assert tab['sha256'] == FC.canonical_sha(tab)


# ------------------------------------------------------------------------------------------------- E9 ---
def test_clopper_pearson():
    assert E9.cp_lower(0, 10) == 0. and E9.cp_upper(10, 10) == 1.
    assert abs(E9.cp_upper(0, 10) - (1 - 0.05 ** 0.1)) < 1e-12
    assert abs(E9.cp_lower(10, 10) - 0.05 ** 0.1) < 1e-12


def test_e9_score_synthetic():
    fresh = dict(cohorts=dict(FI=[dict(j=j, split='test' if j >= 4 else 'dev', s=10. * (j + 1)) for j in range(12)],
                              FC1=[dict(j=j, split='test' if j >= 4 else 'dev', s=300. + j) for j in range(12)]))
    frozen = dict(content_sha256='x', tau_s=150., target_A=dict(threshold=0.002), target_B=dict(threshold=0.01))
    pol = {f'FI|{j}': dict(e=0.001, valid=True, eligible=True) for j in range(4, 10)}
    pol.update({f'FC1|{j}': dict(e=0.05, valid=True, eligible=True) for j in range(4, 11)})
    pol['FC1|11'] = dict(e=None, valid=False, eligible=True)
    pol['FI|9']['eligible'] = False
    out = E9.score(pol, frozen=frozen, fresh=fresh)
    assert out['n'] == 13 and out['n_ineligible'] == 1 and out['n_invalid'] == 1 and out['n_test'] == 14
    tA = out['targets']['A']
    assert tA['n_bad'] == 8 and tA['detection'] == 1.0 and tA['false_alarm'] == 0.0 and tA['auc'] == 1.0
    assert tA['label'].startswith('inconclusive')
    assert 'unmeasured' in out['routing']['cost']
    with pytest.raises(ValueError):
        E9.score({'FI|0': dict(e=0.1, valid=True, eligible=True)}, frozen=frozen, fresh=fresh)


# ------------------------------------------------------------------------------------- reference adapter ---
def test_reference_adapter_allow_list():
    if not FC.OUT.exists():
        pytest.skip('fresh_cohorts.json not yet written')
    sys.path.insert(0, str(LANE / 'phaseb' / 'frefs'))
    import fresh as FR
    assert FR.admissible_cases('FI', 'test') == list(range(4, 10))                  # Amendment 10
    assert FR.admissible_cases('FN0s', 'test') == list(range(4, 12))
    assert FR.admissible_cases('FC2', 'test') == [4, 5]
    assert FR.admissible_cases('FC1', 'dev-smoke') == [0] and FR.admissible_cases('FI', 'dev-smoke') == []
    for bad in ('val', 'dev', None):
        with pytest.raises(FR.FreshError):
            FR.admissible_cases('FI', bad)
    for name in ('N0', 'val32', 'dev6', 'sealed128', 'ood32'):
        with pytest.raises(FR.FreshError):
            FR.Cohort(2, name)
    c = FR.Cohort(2, 'FC1')
    tab = FC.load()
    assert c.nu(5) == tab['cohorts']['FC1'][5]['nu'] and c.full_ic(256, 5).shape == (257, 257)
    assert c.full_ic(256, 5)[0].max() == 0. and c.ic(256, 5).shape == (255, 255)


def test_rom_driver_loads_fresh_table():
    """Interface contract: rom/fcases.load_table accepts phaseb/fresh_cohorts.json (sha256 definition and splits)."""
    if not FC.OUT.exists():
        pytest.skip('fresh_cohorts.json not yet written')
    sys.path.insert(0, str(LANE / 'rom'))
    import fcases
    t, _ = fcases.load_table(FC.OUT)
    assert fcases.table_sha(t['cohorts']) == FC.load()['sha256']


# ------------------------------------------------------------------------- audit codex-code-B-1 fixes ---
def test_route_substitution_probe_refused():
    """Unchanged arm-(a) evidence that STOPs must not become CONTINUE by supplying routes=['mstar'] (Amendment 12)."""
    arm = bank(fn=1.0, nodes=9216)
    arm['selection']['rule'] = 'gauss96'
    inp = dict(arms=dict(a=arm), controls=dict(K1=bank()), eligible=ELIG_V)
    assert V.verdict(inp)['arms']['a']['outcome'] == V.STOP_E
    arm2 = dict(arm, routes=['mstar'])
    with pytest.raises(V.InputError):
        V.verdict(dict(inp, arms=dict(a=arm2)))
    with pytest.raises(V.InputError):
        V.evaluate('a', arm2, bank(), ELIG)
    assert V.verdict(dict(inp, arms=dict(a=dict(arm, routes=['floor']))))['arms']['a']['outcome'] == V.STOP_E


@pytest.mark.parametrize('elig', [dict(ELIG_V, FI=[4, 5, 5, 6, 7, 8]), dict(ELIG_V, FN0s=[2, 4, 5, 6, 7]),
                                  dict(ELIG_V, FC1=['4', 'x']), dict(ELIG_V, BAD=[4]), dict(ELIG_V, FI=[4, 5, 6, 7, 8, 999]),
                                  dict(ELIG_V, FI=['FI|999']), dict(ELIG_V, FI=[4, 5, 6, 7, 8, 10]),
                                  dict(ELIG_V, FC2=[4, 6]), dict(ELIG_V, FC1=['FN0s|4']), dict(ELIG_V, FC1=[True])])
def test_eligible_ids_validated(elig):
    with pytest.raises((V.InputError, ValueError)):
        V.verdict(dict(arms=dict(a=bank(fn=0.1)), controls=dict(K1=bank()), eligible=elig))


def test_selection_rule_node_consistency():
    assert V.rule_nodes('gauss160') == 25600 and V.rule_nodes('fib4181') == 4181
    arm = bank(nodes=9216)
    arm['selection']['rule'] = 'gauss128'                                             # 16384 != 9216
    with pytest.raises(V.InputError):
        V.verdict(dict(arms=dict(c=arm), controls=dict(K1=bank()), eligible=ELIG_V))
    ctrl = bank(nodes=100)
    with pytest.raises(V.InputError):
        V.verdict(dict(arms=dict(a=bank(fn=0.1)), controls=dict(K1=ctrl), eligible=ELIG_V))
    arm = bank(nodes=None, cert=False)                                                # uncertified: no check
    V.verdict(dict(arms=dict(c=arm), controls=dict(K1=bank()), eligible=ELIG_V))


def test_floors_bank_json_formats(tmp_path):
    import json
    import floors_b as B
    b = B.resolve_bank('K1')                                                          # jcp-robust format
    assert b['ckpt'].endswith('s1/ckpt.pkl') and b['rotation'].endswith('s1/rotation_R512.npz')
    d = tmp_path / 'arm'
    d.mkdir()
    (d / 'BANK.json').write_text(json.dumps(dict(ckpt='bank.pkl', rotation='rot.npz', ckpt_sha256='a' * 64,
                                                 rotation_sha256='b' * 64)))
    b = B.resolve_bank(str(d / 'BANK.json'))
    assert b['ckpt'] == str(d / 'bank.pkl') and b['rotation'] == str(d / 'rot.npz')
    for bad in (dict(ckpt='/abs/x.pkl'), dict(ckpt='../x.pkl'), dict(ckpt_sha256='short')):
        (d / 'BANK.json').write_text(json.dumps(dict(dict(ckpt='bank.pkl', rotation='rot.npz', ckpt_sha256='a' * 64,
                                                          rotation_sha256='b' * 64), **bad)))
        with pytest.raises(B.BankError):
            B.resolve_bank(str(d / 'BANK.json'))


def test_e9_routing_separates_input_and_failure():
    fresh = dict(cohorts=dict(FI=[dict(j=j, split='test' if j >= 4 else 'dev', s=10.) for j in range(10)],
                              FC1=[dict(j=j, split='test' if j >= 4 else 'dev', s=300.) for j in range(12)]))
    frozen = dict(content_sha256='x', tau_s=150., target_A=dict(threshold=0.002), target_B=dict(threshold=0.01))
    pol = {f'FI|{j}': dict(e=0.001, valid=True, eligible=True) for j in range(4, 10)}
    pol['FI|4'] = dict(e=None, valid=False, eligible=True)                            # failure found after solving
    pol.update({f'FC1|{j}': dict(e=0.05, valid=True, eligible=True) for j in range(4, 10)})
    out = E9.score(pol, frozen=frozen, fresh=fresh)
    assert out['input_flagged_fraction'] == 0.5 and abs(out['failure_fallback_fraction'] - 1 / 12) < 1e-15
    assert abs(out['flagged_fraction'] - 7 / 12) < 1e-15 and 'unmeasured' in out['routing']['cost']
    t = dict(rom_seconds=1., fom_seconds=10., gpu='H200', job_id='123', paired_same_job=True)
    r = E9.score(pol, timing=t, frozen=frozen, fresh=fresh)['routing']
    assert abs(r['input_routing_seconds_per_query'] - (0.5 * 10 + 0.5 * 1 + 10 / 12)) < 1e-12
    assert abs(r['rom_then_fallback_seconds_per_query'] - (1 + (0.5 + 1 / 12) * 10)) < 1e-12
    r = E9.score(pol, timing=dict(rom_seconds=1., fom_seconds=10., gpu='H200'), frozen=frozen, fresh=fresh)['routing']
    assert 'unmeasured' in r['cost'] and 'timing_refused' in r


def test_validation_accepts_scheduled_and_registered():
    arm = bank(nodes=9216)
    arm['selection']['rule'] = 'gauss96'
    assert V.validate(dict(eligible=dict(FI=['FI|4', 5, '6', 7, 8, 9], FN0s=list(range(4, 12)), FC1=list(range(4, 12)),
                                         FC2=[4, 5]), arms=dict(c=arm), controls=dict(K1=bank())))


@pytest.mark.parametrize('rule,nodes', [('fib1', 1), ('gauss256', 65536), ('gauss384', 147456), ('gauss160', 25600.0),
                                        ('gauss160', '25600'), ('gauss160', True), ('fib4181', 4180), ('gauss64', 4096)])
def test_selection_rule_registered_and_exact(rule, nodes):
    arm = bank()
    arm['selection'] = dict(certified=True, rule=rule, nodes=nodes)
    with pytest.raises(V.InputError):
        V.verdict(dict(arms=dict(c=arm), controls=dict(K1=bank()), eligible=ELIG_V))


def test_prefixed_and_numeric_ids_give_identical_verdicts():
    def pref(b):
        b = copy.deepcopy(b)
        for key in ('floors', 'policy'):
            b[key] = {coh: {f'{coh}|{c}': v for c, v in m.items()} for coh, m in b[key].items()}
        return b
    arms = dict(a=bank(fn=0.1), b=bank(e=0.4), c=bank(tau=0.2))
    num = dict(arms=arms, controls=dict(K1=bank(), K2=bank(), K3=bank()), eligible=ELIG_V,
               eps_ref={'FI': {c: 1e-6 for c in FI}})
    pre = dict(arms={k: pref(v) for k, v in arms.items()},
               controls={k: pref(bank()) for k in ('K1', 'K2', 'K3')},
               eligible={coh: [f'{coh}|{j}' for j in js] for coh, js in ELIG_V.items()},
               eps_ref={'FI': {f'FI|{c}': 1e-6 for c in FI}})
    vn, vp = V.verdict(num), V.verdict(pre)
    assert vn == vp
    assert [vn['arms'][k]['outcome'] for k in 'abc'] == [V.CONT, V.CONT, V.CONT]
    mixed = copy.deepcopy(num)
    mixed['arms']['a']['floors']['FI']['FI|4'] = dict(f384=1.0)                      # same case given twice
    with pytest.raises(V.InputError):
        V.verdict(mixed)


def test_reference_ledger_outcomes(tmp_path):
    """Job audit refT-1: explicit per-case outcomes; deadline outcomes are not failures; empty refslim finalises."""
    import subprocess
    py, refs = sys.executable, LANE / 'cluster' / 'refs'
    cfgs = sorted(str(p) for p in (refs / 'configs').glob('t_*.json'))
    out, logs = tmp_path / 'output', tmp_path / 'logs'
    (out / 'refs_t_FI_4_5/2d/FI/ST').mkdir(parents=True)
    logs.mkdir()
    (out / 'refs_t_FI_4_5/2d/FI/ST/c004.json').write_text('{"outcome": "accepted", "judge": {"eps_ref": 1e-6}}')
    (out / 'refs_t_FI_4_5/2d/FI/ST/c005.json').write_text('{"outcome": "order-fail", "judge": {}}')
    lines = [f'{Path(c).name}\t124\t1' for c in cfgs]
    (logs / 'steps_gpu1.tsv').write_text('\n'.join(lines) + '\n')
    run = lambda: subprocess.run([py, str(refs / 'ref_ledger.py'), str(out), str(logs / 'steps_gpu1.tsv'), '--configs',
                                  *cfgs, '--out', str(tmp_path / 'L.json')], capture_output=True, text=True)
    r = run()
    import json
    L = json.loads((tmp_path / 'L.json').read_text())
    assert r.returncode == 0 and L['ok'] and L['counts']['scheduled'] == 24
    assert L['counts']['accepted'] == 1 and L['counts']['ineligible'] == 1 and L['counts']['unfinished-at-deadline'] == 22
    (logs / 'steps_gpu1.tsv').write_text('\n'.join(lines[1:]) + '\n')                # one step without a record
    r = run()
    L = json.loads((tmp_path / 'L.json').read_text())
    assert r.returncode == 1 and L['counts']['failed'] >= 1 and not L['ok']


def test_score_nonfinite_forces_invalid():
    """Audit codex-code-B-5 row 7: non-finite coefficients / fields / errors -> invalid, e = inf, not qualified."""
    import score_b as SB
    u = np.zeros((6, 257, 257))
    u[:, 100:110, 100:110] = 1.
    W = np.zeros((6, 384))
    ok = SB.decode_and_measure(W, u, 10., True, decode=lambda w: np.zeros((6, 257, 257)))
    assert ok['valid'] and ok['qualified'] and math.isfinite(ok['e'])
    Wn = W.copy()
    Wn[3, 7] = np.nan
    for args in ((Wn, None), (W, lambda w: np.full((6, 257, 257), np.inf)), (np.zeros((6, 10)), None)):
        r = SB.decode_and_measure(args[0], u, 10., True, decode=args[1] or (lambda w: np.zeros((6, 257, 257))))
        assert r['valid'] is False and r['qualified'] is False and r['e'] == math.inf


def test_rom_route_invalid_case_cannot_pass():
    arm = bank(e=0.01)
    arm['policy']['FC1']['5'] = dict(e=INF, valid=False, qualified=True)             # stale qualified flag
    assert ev('b', arm)['routes']['rom']['status'] == V.FAIL
    arm = bank(e=0.01)
    arm['policy']['FC1']['5'] = dict(e=float('nan'), valid=True, qualified=True)
    assert ev('b', arm)['routes']['rom']['status'] == V.FAIL
    arm = bank(tau=0.01)
    arm['policy']['FI']['6'] = dict(tau=INF, valid=False, qualified=True)
    assert ev('c', arm)['routes']['tail']['status'] == V.FAIL


# ------------------------------------------------------------------------------- arm (c) architecture ---
def _synthetic_sine_even(tmp_path, R=512, H=24, seed=3):
    import pickle
    r = np.random.default_rng(seed)
    B0 = r.normal(0., 4., (2, 128))
    g = [(r.normal(0, 1 / 16, (256, H)), r.normal(0, .1, H)), (r.normal(0, 1 / 5, (H, H)), r.normal(0, .1, H)),
         (r.normal(0, 1 / 5, (H, R)), r.normal(0, .1, R))]
    params = dict(B=B0, Kc=np.abs(np.round(2 * B0)), g=g, out_scale=np.float64(0.7))
    f = tmp_path / 'ckpt.pkl'
    pickle.dump(dict(params=params, Z_tr=np.zeros((4, 16)), cfg=dict(arch='sine_even', seed=1)), open(f, 'wb'))
    return f, params


def test_arm_c_bank_decodes_as_sine_even_ref(tmp_path):
    """floors_b / score_b load an arm-(c) checkpoint through npbank_arch: decode == sine_even_ref exactly."""
    import json
    import hashlib
    import floors_b as B
    sys.path.insert(0, str(LANE / 'rom' / 'deps'))
    import sine_even_ref as SE
    f, params = _synthetic_sine_even(tmp_path)
    rot = B.ROBUST / 's1' / 'rotation_R512.npz'
    sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (tmp_path / 'rotation_R512.npz').write_bytes(rot.read_bytes())
    (tmp_path / 'BANK.json').write_text(json.dumps(dict(ckpt='ckpt.pkl', rotation='rotation_R512.npz',
                                                        ckpt_sha256=sh(f), rotation_sha256=sh(rot))))
    try:
        b = B.use(str(tmp_path / 'BANK.json'))
        assert b['arch'] == 'sine_even'
        T = np.load(rot)['T']
        X = np.random.default_rng(5).uniform(0, 1, (300, 2))
        got = B.F.NC.NB.features_grad(B.F.NC.bank()[0], X)
        ref = SE.features_grad(params, X)
        assert all(np.array_equal(a, c) for a, c in zip(got, ref))
        W = np.random.default_rng(6).normal(size=(2, 384))
        V = B.F.NC.decode257(W, 384)
        x = np.arange(257) / 256.
        L = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
        G = SE.features(params, L)
        G[(L[:, 0] == 0) | (L[:, 0] == 1) | (L[:, 1] == 0) | (L[:, 1] == 1)] = 0.
        Vr = (W @ (G @ T[:, :384]).T).reshape(2, 257, 257)
        assert np.max(np.abs(V - Vr)) <= 1e-12 * np.max(np.abs(Vr))
        # K1 (incumbent arch) unchanged by the dispatch: identical to the original npbank formula
        b1 = B.use('K1')
        assert b1['arch'] == 'bc16poly'
        P = B.F.NC.bank()[0]
        assert all(np.array_equal(a, c) for a, c in zip(B.F.NC.NB.features_grad(P, X), B.ORIG_FEATURES_GRAD(P, X)))
    finally:
        B.use('K0')


# ------------------------------------------------------------------- audit codex-code-C-2 rows 12-13 ---
def test_e10q_requires_validated_complete_evidence():
    import json
    import assemble as A
    q = json.loads((A.CHK / 'e10q_every_step.json').read_text())
    assert A.e10q_check(q) == (True, 'ok')
    for mut in (lambda d: d['validation'].__setitem__('passed', False),
                lambda d: d['validation'].__setitem__('max_rel_err', 1.0),
                lambda d: d['validation'].__setitem__('tolerance', 1e-3),
                lambda d: d.__setitem__('runs', d['runs'][1:]),
                lambda d: d.__setitem__('result_sha256', '0' * 64),
                lambda d: d['validation'].__setitem__('per_run', d['validation']['per_run'][1:]),
                lambda d: d['runs'][0].__setitem__('finite_states', 0),
                lambda d: d['summary'].__setitem__('states_over_bar', 5)):
        d = copy.deepcopy(q)
        mut(d)
        assert A.e10q_check(d)[0] is False
    assert A.e10q_check(None)[0] is False


def test_e10_outcomes_need_verified_evidence(tmp_path, monkeypatch):
    import json
    import assemble as A
    k1s = A.load('score_K1.json')
    elig = {'FC1': list(range(4, 12))}
    q = json.loads((A.CHK / 'e10q_every_step.json').read_text())
    fake = copy.deepcopy(k1s)                                                         # E10-s: qualified, 10x better
    for r in fake['cases']:
        r.update(qualified=True, valid=True, terminal='CONVERGED', e=(A._f(r['e']) / 10.))
    bad_q = copy.deepcopy(q)
    bad_q['validation']['passed'] = False
    store = {'score_e10s.json': fake, 'e10q_every_step.json': q}
    monkeypatch.setattr(A, 'load', lambda n: store.get(n))
    ev = tmp_path / 'E10-EVIDENCE.json'
    ev.write_text(json.dumps({'passed': True}))                                      # the audit's unbound probe
    out = A.e10(k1s, elig, str(ev))
    assert out['E10-s']['outcome'] == 'INCONCLUSIVE' and out['E10-q']['outcome'] == 'STOP'
    assert A.e10(k1s, elig, str(tmp_path / 'missing.json'))['E10-s']['outcome'] == 'INCONCLUSIVE'
    monkeypatch.setattr(A, 'verify_e10s_evidence', lambda p=None: (True, 'ok'))
    assert A.e10(k1s, elig, None)['E10-s']['outcome'] == 'CONTINUE'
    store['e10q_every_step.json'] = bad_q
    assert A.e10(k1s, elig, None)['E10-q']['outcome'] == 'INCONCLUSIVE'
    store.pop('score_e10s.json')
    assert A.e10(k1s, elig, None)['E10-s']['outcome'] == 'NOT RUN'
