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
    assert r['guard']['status'] == V.FAIL and r['outcome'] == V.STOP_G               # Amendment 5
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
    inp = dict(arms=dict(a=bank(fn=0.1)), controls=dict(K1=bank(), K2=bank(), K3=bank(fn=0.6)), eligible=ELIG)
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
    pol = {f'FI|{j}': dict(e=0.001, valid=True, eligible=True) for j in range(4, 12)}
    pol.update({f'FC1|{j}': dict(e=0.05, valid=True, eligible=True) for j in range(4, 11)})
    pol['FC1|11'] = dict(e=None, valid=False, eligible=True)
    pol['FI|11']['eligible'] = False
    out = E9.score(pol, frozen=frozen, fresh=fresh)
    assert out['n'] == 15 and out['n_ineligible'] == 1 and out['n_invalid'] == 1
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
    assert FR.admissible_cases('FI', 'test') == list(range(4, 12))
    assert FR.admissible_cases('FC2', 'test') == [4, 5, 6, 7]
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
