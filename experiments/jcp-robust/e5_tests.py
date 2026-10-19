"""jcp-robust E5 CPU unit tests (no JAX). C-E5.3 drift band, A3.7 fixtures, A4.1 missing/blocked jobs, empty reductions.

    JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python e5_tests.py        (or: python -m pytest e5_tests.py -q)
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import e5common as EC
import e5analyze as E5A

HERE = Path(__file__).resolve().parent
R128, R384 = E5A.FAMS
ROMS = ('ckpt:x|R128|fib4181|frozen|BE|0.005|LSPG|prod', 'ckpt:x|R384|gauss128|frozen|CNR|0.005|LSPG|prod')
PS, CC1, CC2 = 'ps|N96|0.0125', 'fd|CC|L1024|0.0125|t2', 'fd|CC|L1024|0.005|t1'
PINNED = {ROMS[0]: {'|H|union|joint': PS, '|H|CC|same_mesh': CC1}, ROMS[1]: {'|H|union|joint': PS, '|H|CC|same_mesh': CC2}}
COST_LEVEL = {R128: .25, R384: 1.0}
CROSS_LEVEL = {ROMS[0]: .24, ROMS[1]: 1.0, PS: .07, CC1: 1.05, CC2: 2.4}


def write_job(root, name, gtype, seed, ncase=6, reps=3, cost_scale=None, cross_scale=None, doubled=2.0, meshes=(256, 1024),
              sigma=.02):
    """A synthetic E5 job directory in the formats e5time writes (P1 p1time / fom2d c2time per-triplet records)."""
    rng = np.random.default_rng(seed)
    d = Path(root) / name
    (d / 'pcost').mkdir(parents=True)
    cases = [f'dev6|{i}' for i in range(ncase)]
    cost_scale = cost_scale or {}
    cross_scale = cross_scale or {}
    inv, dbl = [], []
    subs = {}
    for f in E5A.FAMS:
        for L in meshes:
            lb = f'{L}|{f}'
            subs[lb] = dict(case_ids=cases)
            for c in cases:
                for r in range(reps):
                    ratio = COST_LEVEL[f] * cost_scale.get((f, L), 1.) * float(np.exp(sigma * rng.standard_normal()))
                    inv.append(dict(B=lb, case=c, rep=r, ratio=ratio, valid=True, t_evolve=.02 * ratio, hashes_ok=True,
                                    all_hashes_ok=True, attempts=[dict(hashes_ok=True)]))
    for lb in ('1024|' + R128, '1024|' + R384):
        for c in cases:
            for r in range(reps):
                dbl.append(dict(subject=lb, case=c, rep=r, ratio=doubled * float(np.exp(sigma * rng.standard_normal())),
                                drift_ok=True, identity_ok=True, same_iterations=True))
    pc = dict(config=dict(reps=reps), subjects=subs, invocations=inv, fused=[], doubled=dbl,
              equivalence=[dict(passed=True)], complete=True)
    st2 = []
    for s in list(ROMS) + [PS, CC1, CC2]:
        for c in cases:
            for r in range(reps):
                ratio = CROSS_LEVEL[s] * cross_scale.get(s, 1.) * float(np.exp(sigma * rng.standard_normal()))
                st2.append(dict(B=s, case=c, rep=r, ratio=ratio, valid=True, tB=.085 * ratio))
    xc = dict(config=dict(cases=[c.split('|')[0:1] + [int(c.split('|')[1])] for c in cases], roms=[dict(label=r) for r in ROMS]),
              stage2=st2, pinned=PINNED, veto=[], complete=True, gates={})
    top = dict(attempt=name, complete=True, gpu_type=gtype, gpu_type_ok=True, backend='gpu', source_check=[dict(ok=True)],
               records=[], pcost=pc, pcross=xc)
    (d / 'result.json').write_text(json.dumps(top))
    (d / 'COMPLETE').write_text('complete\n')
    return d


def four_jobs(root, h200_cost=None, h200_cross=None, **kw):
    js = [('A100-80G', write_job(root, 'a1', 'A100-80G', 1, **kw)), ('A100-80G', write_job(root, 'a2', 'A100-80G', 2, **kw)),
          ('H200', write_job(root, 'h1', 'H200', 3, cost_scale=h200_cost, cross_scale=h200_cross, **kw)),
          ('H200', write_job(root, 'h2', 'H200', 4, cost_scale=h200_cost, cross_scale=h200_cross, **kw))]
    return [E5A.load_job(t, d) for t, d in js]


EXPECT = {'A100-80G': 2, 'H200': 2}


class Tmp:
    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix='e5test-', dir=_scratch())
        return self.d

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


def _scratch():
    p = Path('/tmp/claude-1002/jcp-robust-e5-scratch')
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# ------------------------------------------------------------------ C-E5.3 ----

def test_c_e5_3_drift_band():
    assert EC.drift_ok(1.0) and EC.drift_ok(1.1) and EC.drift_ok(1 / 1.1)
    assert not EC.drift_ok(1.5) and not EC.drift_ok(1 / 1.5)

    def stepped(steps):
        def attempt(a):
            return dict(tA1=.1, tB=.05, tA2=.1 * steps[a], hashes_ok=True)
        return attempt
    x = EC.aba_triplet(stepped([1.5, 1.5, 1.5]))            # a 1.5x step inside every attempt: rejected after 2 retries
    assert x['valid'] is False and x['retries'] == 2 and len(x['attempts']) == 3
    x = EC.aba_triplet(stepped([1.5, 1.0, 1.0]))            # a transient step: one retry, then valid
    assert x['valid'] is True and x['retries'] == 1 and abs(x['ratio'] - .5) < 1e-12
    x = EC.aba_triplet(lambda a: dict(tA1=.1, tB=.05, tA2=.1, hashes_ok=False))
    assert x['valid'] is False                                 # identity failure invalidates regardless of drift
    # the driver uses this function for P-cross, and P1's p1time (P-cost) keeps the same inline band and retry count
    assert 'EC.aba_triplet(' in (HERE / 'e5time.py').read_text()
    assert EC.p1time_band_is_ours((HERE / 'e5' / 'p1time.py').read_text())
    assert 'BAND = (1 / 1.1, 1.1)' in (HERE / 'e5' / 'c2time.py').read_text()


# ------------------------------------------------------------------ fixtures (A3.7) ----

def test_fixtures_fire_and_null_consistent():
    with Tmp() as t:
        j = E5A.load_job('A100-80G', write_job(t, 'a1', 'A100-80G', 11))
        assert j['status'] == 'ok', j['status']
        fx = E5A.fixtures(j)
        for k in ('q1', 'q2', 'q3'):
            assert fx[k]['ok'], (k, fx[k])
            assert fx[k]['injected']['label'] == 'hardware-dependent'
            assert abs(fx[k]['injected']['phi'] - 1.5) < 1e-12
            assert fx[k]['null']['label'] == 'hardware-consistent'
            assert fx[k]['null']['lo'] == 1. and fx[k]['null']['hi'] == 1.


def test_fixture_too_few_cases_never_passes():
    with Tmp() as t:
        j = E5A.load_job('A100-80G', write_job(t, 'a1', 'A100-80G', 12, ncase=4))
        fx = E5A.fixtures(j)
        assert all(not v['ok'] for v in fx.values())


# ------------------------------------------------------------------ end to end ----

def test_consistent_and_dependent():
    with Tmp() as t:
        res = E5A.analyze(four_jobs(t), EXPECT)
        assert res['types_complete'] == {'A100-80G': True, 'H200': True}
        assert all(v['ok'] for v in res['fixtures'].values())
        assert len(res['case_sets']['C_cost']) == 6 and len(res['case_sets']['C_cross']) == 6
        for k in ('phi1', 'phi2', 'phi3'):
            assert all(v['label'] == 'hardware-consistent' for v in res['phi'][k].values()), (k, res['phi'][k])
        assert res['agreement'][f'H-cost-N(2)|{R128}'] == 'verdict-robust'
        assert res['per_type']['H200'][f'H-cost-N(2)|{R128}']['label'] == 'pass'
        assert res['agreement'][f'S2|{ROMS[1]}|{CC2}'] == 'verdict-robust'          # S = 2.4 / 1.0 >= 2
        assert res['agreement'][f'S2|{ROMS[0]}|{PS}'] == 'verdict-robust'           # S << 2: fail on both
        assert res['per_type']['A100-80G'][f'S2|{ROMS[0]}|{PS}']['label'] == 'fail'
        assert set(res['unit_to_unit']) == {'A100-80G', 'H200'}
    with Tmp() as t:
        res = E5A.analyze(four_jobs(t, h200_cost={(R128, 1024): 1.5}, h200_cross={CC2: .5}), EXPECT)
        assert res['phi']['phi1'][f'{R128}|1024']['label'] == 'hardware-dependent'
        assert res['phi']['phi1'][f'{R128}|256']['label'] == 'hardware-consistent'
        assert res['phi']['phi2'][R128]['label'] == 'hardware-dependent'
        assert res['phi']['phi3'][f'{ROMS[1]}|{CC2}']['label'] == 'hardware-dependent'
        assert res['agreement'][f'H-cost-N(2)|{R128}'] == 'hardware-dependent verdict'   # pass on A100, fail on H200
        assert res['agreement'][f'S2|{ROMS[1]}|{CC2}'] == 'hardware-dependent verdict'


# ------------------------------------------------------------------ A4.1 ----

def _assert_unavailable(res):
    assert res['phi'] == {k: 'unavailable (job missing)' for k in ('phi1', 'phi2', 'phi3')}
    assert all(v == 'unavailable (job missing)' for v in res['agreement'].values()) and res['agreement']


def test_missing_job_gives_unavailable():
    with Tmp() as t:
        js = four_jobs(t)[:3]
        res = E5A.analyze(js, EXPECT)
        assert res['types_complete'] == {'A100-80G': True, 'H200': False}
        _assert_unavailable(res)
        assert res['per_type']['H200'] == 'unavailable (job missing)'
        assert all('per-type only' in v['scope'] for v in res['per_type']['A100-80G'].values())
        assert res['per_job']['H200#2']['status'] == 'missing'
        js = four_jobs(t + '/b')
        (Path(js[2]['dir']) / 'COMPLETE').unlink()                       # an incomplete job is "failed", same path
        res = E5A.analyze([E5A.load_job(j['type'], j['dir']) for j in js], EXPECT)
        _assert_unavailable(res)


def test_blocked_job_gives_unavailable():
    with Tmp() as t:
        js = four_jobs(t)
        d = write_job(t, 'h2bad', 'H200', 9, doubled=1.0)              # doubled-work control does not fire
        js[3] = E5A.load_job('H200', d)
        assert js[3]['status'] == 'blocked (control C-E5.1)'
        res = E5A.analyze(js, EXPECT)
        _assert_unavailable(res)
        assert res['per_job']['H200#2']['scope'] == 'per-job only'


# ------------------------------------------------------------------ empty / degenerate reductions ----

def test_empty_reductions_never_pass():
    assert E5A.c_e5_1([])['fires'] is False
    assert E5A.c_e5_1([dict(subject='s', case='c', ratio=2., drift_ok=False, identity_ok=True, same_iterations=True)])['fires'] is False
    for f in (E5A.phi_label, E5A.cost_label, E5A.s2_label):
        assert f(None, None) == 'unavailable' and f(float('nan'), 1.) == 'unavailable'
    assert E5A.interval([]) == (None, None)
    assert E5A.agreement({}) == 'unavailable'
    assert E5A.agreement({'a': 'pass', 'b': 'inconclusive'}) == 'inconclusive'
    assert E5A.agreement({'a': 'pass', 'b': 'unavailable (x)'}) == 'unavailable'
    assert E5A.type_label('pass', ['pass', 'fail']) == 'inconclusive (jobs disagree)'
    assert E5A.type_label('pass', ['pass', 'inconclusive']) == 'pass'
    assert E5A.case_set({'g': [dict(cost={})]}, [('x', 1)], ['c0', 'c1'], 'cost') == []
    sp = E5A.C2A.speedup({'c0': 1.}, {'c0': 2.})
    assert sp['status'] == 'invalid timing'
    with Tmp() as t:                                                    # 4 cases everywhere: |C| < 5 -> unavailable
        res = E5A.analyze(four_jobs(t, ncase=4), EXPECT)
        assert str(res['phi']['phi1']).startswith('unavailable') and str(res['phi']['phi3']).startswith('unavailable')
        assert all(v != 'verdict-robust' for v in res['agreement'].values())
        assert all(not v['ok'] for v in res['fixtures'].values())


def test_failed_fixture_blocks():
    with Tmp() as t:
        js = four_jobs(t)
        real = E5A.fixtures
        try:
            E5A.fixtures = lambda j: dict(real(j), q2=dict(ok=False, reason='forced'))
            res = E5A.analyze(js, EXPECT)
        finally:
            E5A.fixtures = real
        assert all(v['label'].startswith('blocked') for v in res['phi']['phi2'].values())
        assert all(not v['label'].startswith('blocked') for v in res['phi']['phi1'].values())
        assert all(res['agreement'][k].startswith('blocked') for k in res['agreement'] if k.startswith('H-cost-N'))
        assert all(not res['agreement'][k].startswith('blocked') for k in res['agreement'] if k.startswith('S2'))


if __name__ == '__main__':
    import time
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith('test_') and callable(v)):
        t0 = time.time()
        try:
            fn()
            print(f'PASS {name} ({time.time() - t0:.1f}s)', flush=True)
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {name}: {e!r}', flush=True)
    print('ALL PASS' if not fails else f'{fails} FAILED')
    raise SystemExit(1 if fails else 0)
