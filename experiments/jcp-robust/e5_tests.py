"""jcp-robust E5 CPU unit tests (no JAX). C-E5.3 drift band, A3.7 fixtures, A4.1 missing/blocked jobs, empty reductions.

    JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python e5_tests.py        (or: python -m pytest e5_tests.py -q)
"""
from __future__ import annotations

import json
import zlib
import shutil
import tempfile
from pathlib import Path

import numpy as np

import e5common as EC
import e5analyze as E5A

HERE = Path(__file__).resolve().parent
R128, R384 = E5A.FAMS
ROMS = EC.E5_ROMS
PS, CC1, CC2 = 'ps|N96|0.0125', 'fd|CC|L1024|0.0125|t2', 'fd|CC|L1024|0.005|t1'
PINNED = EC.E5_PINNED
assert PINNED == {ROMS[0]: {'|H|union|joint': PS, '|H|CC|same_mesh': CC1}, ROMS[1]: {'|H|union|joint': PS, '|H|CC|same_mesh': CC2}}
COST_LEVEL = {R128: .25, R384: 1.0}
SPEC = {R128: dict(Rp=128, rule='fib4181', policy='frozen', scheme='BE'), R384: dict(Rp=384, rule='gauss128', policy='frozen', scheme='CNR')}


def w_of(label, cases):
    """Deterministic scoring-run coefficients per (subject, case): identical in a synthetic job and its P1 reference."""
    rp = 128 if '|R128|' in label else 384
    return np.stack([np.full((6, rp), float(len(label))) + int(c.split('|')[1]) for c in cases])


def smi_row(uuid, name='NVIDIA A100-SXM4-80GB', idx='0', bus='00000000:07:00.0', app='1410 MHz'):
    return {'index': idx, 'pci.bus_id': bus, 'uuid': uuid, 'name': name, 'memory.total': '81920 MiB',
            'driver_version': '550.54.15', 'clocks.sm': '1410 MHz', 'clocks.mem': '1593 MHz',
            'clocks.applications.graphics': app, 'clocks.applications.memory': app, 'power.limit': '400.00 W',
            'power.draw': '90.1 W', 'temperature.gpu': '34'}


def telem(tag, uuid, row=None):
    r = smi_row(uuid)
    r.update(row or {})
    return dict(tag=tag, device_uuid=uuid, env={'JAX_ENABLE_X64': 'true', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest'},
                versions={'jax': '0.10.2', 'jaxlib': '0.10.2', 'jax_cuda_plugin': 'jax-cuda12-plugin==0.10.2',
                          'jax_cuda_pjrt': 'jax-cuda12-pjrt==0.10.2'},
                nvidia_smi=dict(row=r, error=None, n_rows=1, cuda_driver_version='12.4'))


def write_s(d, label, cases, W=None):
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"S_{label.replace('|', '_')}.npz", case_ids=np.array(cases), W=w_of(label, cases) if W is None else W)


def make_ref(root, meshes=(256, 1024)):
    """A synthetic P1c archive (output/p1time_time/S_*.npz over all of dev6 + the archive listing OUTPUTS.sha256)."""
    arch = Path(root) / 'p1arch'
    d = arch / 'output' / 'p1time_time'
    for f in E5A.FAMS:
        for L in meshes:
            write_s(d, f'{L}|{f}', [f'dev6|{i}' for i in range(6)])
    (arch / 'OUTPUTS.sha256').write_text(''.join(f'{EC.sha_file(q)}  output/p1time_time/{q.name}\n' for q in sorted(d.glob('S_*.npz'))))
    return str(d)
CROSS_LEVEL = {ROMS[0]: .24, ROMS[1]: 1.0, PS: .07, CC1: 1.05, CC2: 2.4}


def write_job(root, name, gtype, seed, ncase=6, reps=3, cost_scale=None, cross_scale=None, doubled=2.0, meshes=(256, 1024),
              sigma=.02, gates=None):
    """A synthetic E5 job directory in the formats e5time writes (P1 p1time / fom2d c2time per-triplet records)."""
    rng = np.random.default_rng(seed)
    att = f"test-{gtype}-{ncase}x{reps}-{'-'.join(map(str, meshes))}"
    cl = [f'dev6|{i}' for i in range(ncase)]
    EC.E5_CONTRACT[att] = EC.make_contract(gtype, meshes, cl, reps, cl, reps)
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
    for lb in subs:
        write_s(d / 'pcost', lb, cases)
    specs = [dict(SPEC[f], mesh=L) for f in E5A.FAMS for L in meshes]
    scored = sorted(set(subs) | {EC.E5_A})
    pc = dict(config=dict(reps=reps, cases=[['dev6', int(c.split('|')[1])] for c in cases], A=dict(SPEC[R384], mesh=1024),
                          subjects=specs), subjects=subs, invocations=inv, fused=[], doubled=dbl,
              equivalence=[dict(subject=lb, fields_rel=0., same_iterations=True, passed=True) for lb in scored], complete=True)
    st2 = []
    for s in list(ROMS) + [PS, CC1, CC2]:
        for c in cases:
            for r in range(reps):
                ratio = CROSS_LEVEL[s] * cross_scale.get(s, 1.) * float(np.exp(sigma * rng.standard_normal()))
                st2.append(dict(B=s, case=c, rep=r, ratio=ratio, valid=True, tB=.085 * ratio))
    xc = dict(config=dict(cases=[c.split('|')[0:1] + [int(c.split('|')[1])] for c in cases], roms=[dict(label=r) for r in ROMS]),
              stage2=st2, pinned=PINNED, veto=[], complete=True,
              gates=gates if gates is not None else {'G-RR': True, 'G-FR': True, 'A-valid': True})
    uuid = f'GPU-{name}-{seed}'
    top = dict(attempt=att, complete=True, gpu_type=gtype, gpu_type_ok=True, backend='gpu', source_check=[dict(ok=True)],
               device=dict(resolved=True, uuid=uuid, method='test'), device_uuid=uuid, job_id=f'{seed}{name}-{zlib.crc32(str(d).encode())}',
               records=[telem('start', uuid), telem('between-panels', uuid), telem('end', uuid)], pcost=pc, pcross=xc)
    (d / 'result.json').write_text(json.dumps(top))
    (d / 'COMPLETE').write_text('complete\n')
    return d


def four_jobs(root, h200_cost=None, h200_cross=None, **kw):
    js = [('A100-80G', write_job(root, 'a1', 'A100-80G', 1, **kw)), ('A100-80G', write_job(root, 'a2', 'A100-80G', 2, **kw)),
          ('H200', write_job(root, 'h1', 'H200', 3, cost_scale=h200_cost, cross_scale=h200_cross, **kw)),
          ('H200', write_job(root, 'h2', 'H200', 4, cost_scale=h200_cost, cross_scale=h200_cross, **kw))]
    ref = make_ref(root)
    return [E5A.load_job(t, d, p1ref=ref) for t, d in js]


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
        j = E5A.load_job('A100-80G', write_job(t, 'a1', 'A100-80G', 11), p1ref=make_ref(t))
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
        j = E5A.load_job('A100-80G', write_job(t, 'a1', 'A100-80G', 12, ncase=4), p1ref=make_ref(t))
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
        res = E5A.analyze([E5A.load_job(j['type'], j['dir'], p1ref=make_ref(t + '/b')) for j in js], EXPECT)
        _assert_unavailable(res)


def test_blocked_job_gives_unavailable():
    with Tmp() as t:
        js = four_jobs(t)
        d = write_job(t, 'h2bad', 'H200', 9, doubled=1.0)              # doubled-work control does not fire
        js[3] = E5A.load_job('H200', d, p1ref=make_ref(t))
        assert js[3]['status'] == 'blocked (control C-E5.1)'
        res = E5A.analyze(js, EXPECT)
        _assert_unavailable(res)
        assert res['per_job']['H200#2']['scope'] == 'per-job only'


# ------------------------------------------------------------------ empty / degenerate reductions ----

def test_empty_reductions_never_pass():
    ct = EC.make_contract('A100-80G', (256, 1024), ['dev6|0'], 1, ['dev6|0'], 1)
    assert E5A.c_e5_1([], ct)['fires'] is False and E5A.c_e5_1([], ct)['complete'] is False
    rows = [dict(subject=s_, case='dev6|0', rep=0, ratio=2., drift_ok=True, identity_ok=True, same_iterations=True) for s_ in ct['doubled']]
    assert E5A.c_e5_1(rows, ct)['fires'] is True
    rows[0]['drift_ok'] = False
    assert E5A.c_e5_1(rows, ct)['fires'] is False
    assert EC.population_check([], lambda x: x, [])[0] is False
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


# ------------------------------------------------------------------ code audit 3: gates, smoke mode, inventory ----

def test_pcross_gates_required():
    with Tmp() as t:
        for g in ({'G-RR': False, 'G-FR': True, 'A-valid': True}, {'G-RR': True, 'G-FR': True}, {}):
            j = E5A.load_job('A100-80G', write_job(t, f'g{len(g)}{sum(g.values())}', 'A100-80G', 21, gates=g), p1ref=make_ref(t))
            assert j['status'].startswith('blocked (gate P-cross'), (g, j['status'])


def test_g_e5_1_required_complete_and_finite():
    with Tmp() as t:
        ref = make_ref(t)
        d = write_job(t, 'a1', 'A100-80G', 22)
        assert E5A.load_job('A100-80G', d, p1ref=ref)['status'] == 'ok'
        j = E5A.load_job('A100-80G', d)                                    # no --p1ref
        assert j['status'] == 'blocked (gate G-E5.1)' and not j['g_e5_1']['passed']
        lb = f'1024|{R128}'
        cases = [f'dev6|{i}' for i in range(6)]
        W = w_of(lb, cases)
        W[2, 0, 0] *= 1 + 1e-6                                              # numerics differ
        write_s(d / 'pcost', lb, cases, W)
        assert E5A.load_job('A100-80G', d, p1ref=ref)['status'] == 'blocked (gate G-E5.1)'
        W[2, 0, 0] = np.nan                                                 # non-finite
        write_s(d / 'pcost', lb, cases, W)
        assert E5A.load_job('A100-80G', d, p1ref=ref)['status'] == 'blocked (gate G-E5.1)'
        write_s(d / 'pcost', lb, cases[:5])                                 # a registered case missing
        assert E5A.load_job('A100-80G', d, p1ref=ref)['status'] == 'blocked (gate G-E5.1)'
        write_s(d / 'pcost', lb, cases)
        (d / 'pcost' / f"S_{('256|' + R384).replace('|', '_')}.npz").unlink()   # a registered subject missing
        j = E5A.load_job('A100-80G', d, p1ref=ref)
        assert j['status'] == 'blocked (gate G-E5.1)' and 'missing job file' in str(j['g_e5_1']['rows'])


def test_smoke_mode_single_job():
    with Tmp() as t:
        ref = make_ref(t)
        good = E5A.load_job('A100-80G', write_job(t, 'a1', 'A100-80G', 31, ncase=5, reps=2), p1ref=ref)
        r = E5A.smoke(good)
        assert r['ALL'] is True and all(v['ok'] for v in r['checks'].values()), {k: v['ok'] for k, v in r['checks'].items()}
        assert sum(k.startswith('fixture') for k in r['checks']) == 3
        bad = E5A.load_job('A100-80G', write_job(t, 'a2', 'A100-80G', 32, ncase=5, reps=2, doubled=1.0), p1ref=ref)
        r = E5A.smoke(bad)
        assert r['ALL'] is False and not r['checks'][next(k for k in r['checks'] if k.startswith('C-E5.1'))]['ok']
        r = E5A.smoke(E5A.load_job('A100-80G', write_job(t, 'a3', 'A100-80G', 33, ncase=5, reps=2)))   # no p1ref
        assert r['ALL'] is False
        r = E5A.smoke(E5A.load_job('A100-80G', None))
        assert r['ALL'] is False
        res = E5A.analyze([good], {'A100-80G': 1})                          # --expect A100-80G=1 must not crash
        assert res['phi']['phi1'].startswith('unavailable') and all(v.startswith('unavailable') for v in res['agreement'].values())
        import subprocess
        import sys
        out = Path(t) / 'smk.json'
        rc = subprocess.run([sys.executable, str(HERE / 'e5analyze.py'), '--smoke', '--job', f"A100-80G={bad['dir']}",
                             '--p1ref', ref, '--out', str(out)], capture_output=True).returncode
        assert rc == 1 and json.loads(out.read_text())['ALL'] is False


def test_inventory_e5time():
    for f in ('e5smk4', 'e5prod_a100', 'e5prod_h200'):
        c = json.loads((HERE / 'configs' / f'{f}.json').read_text())
        k = EC.inv_e5time(c)
        assert k and len(k) == len(set(k)), f
        assert all(isinstance(EC.klass(x), str) and EC.klass(x).startswith('e5time|') for x in k)
        n = lambda kind, panel: sum(1 for x in k if x[3] == kind and x[1] == panel)
        pc, xc = c['pcost'], c['pcross']
        npc = len(EC._cases(pc['cases']))
        assert n('triplet', 'pcost') == len(pc['subjects']) * npc * pc['reps']
        nf = len({v for d in xc['expected_comparators'].values() for v in d.values()})
        assert n('triplet', 'pcross') == (len(xc['roms']) + nf) * len(xc['cases']) * xc['reps']
        assert n('doubled', 'pcost') == len(pc['doubled']) * npc * pc['reps']
    assert any('4096|' in x[2] for x in EC.inv_e5time(json.loads((HERE / 'configs' / 'e5prod_h200.json').read_text())))


# ------------------------------------------------------------------ code audit 4: contract, populations, smoke type ----

def _rewrite(d, fn):
    r = json.loads((Path(d) / 'result.json').read_text())
    fn(r)
    (Path(d) / 'result.json').write_text(json.dumps(r))


def test_contract_matches_configs_and_unknown_attempt_fails():
    for att, ct in EC.E5_CONTRACT.items():
        if att.startswith('test-'):
            continue
        c = json.loads((HERE / 'configs' / f'{att}.json').read_text())
        lab = lambda sp: f"{sp['mesh']}|R{sp['Rp']}|{sp['rule']}|{sp.get('policy', 'frozen')}|{sp['scheme']}"
        assert c['gpu_type'] == ct['gpu_type'] and {lab(x) for x in c['pcost']['subjects']} == set(ct['pcost_subjects'])
        assert EC._cases(c['pcost']['cases']) == list(ct['pcost_cases']) and c['pcost']['reps'] == ct['pcost_reps']
        assert EC._cases(c['pcross']['cases']) == list(ct['pcross_cases']) and c['pcross']['expected_comparators'] == ct['pinned']
    with Tmp() as t:
        d = write_job(t, 'a1', 'A100-80G', 40)
        _rewrite(d, lambda r: r.update(attempt='not-registered'))
        assert 'no registered E5 contract' in E5A.load_job('A100-80G', d, p1ref=make_ref(t))['status']
        d = write_job(t, 'h1', 'H200', 41)
        assert 'registered for H200' in E5A.load_job('A100-80G', d, p1ref=make_ref(t))['status']


def test_g_e5_1_reference_hash_duplicates_shapes():
    with Tmp() as t:
        ref = make_ref(t)
        d = write_job(t, 'a1', 'A100-80G', 42)
        assert E5A.load_job('A100-80G', d, p1ref=ref)['status'] == 'ok'
        lb = f'256|{R128}'
        cases = [f'dev6|{i}' for i in range(6)]
        write_s(d / 'pcost', lb, cases + ['dev6|5'], np.concatenate([w_of(lb, cases), w_of(lb, ['dev6|5'])]))   # duplicate id
        j = E5A.load_job('A100-80G', d, p1ref=ref)
        assert j['status'] == 'blocked (gate G-E5.1)' and j['g_e5_1']['rows'][lb]['status'] == 'duplicate case ids'
        write_s(d / 'pcost', lb, cases, w_of(lb, cases)[:, :, :1])                                 # broadcastable shape
        j = E5A.load_job('A100-80G', d, p1ref=ref)
        assert j['status'] == 'blocked (gate G-E5.1)' and 'shape' in j['g_e5_1']['rows'][lb]['status']
        write_s(d / 'pcost', lb, cases)
        assert E5A.load_job('A100-80G', d, p1ref=ref)['status'] == 'ok'
        write_s(Path(ref), lb, cases, w_of(lb, cases) * (1 + 1e-14))                               # reference not as archived
        j = E5A.load_job('A100-80G', d, p1ref=ref)
        assert j['status'] == 'blocked (gate G-E5.1)' and 'listing' in j['g_e5_1']['rows'][lb]['status']


def test_c_e5_1_and_triplet_populations_complete():
    with Tmp() as t:
        ref = make_ref(t)
        for name, fn, want in (
                ('d1', lambda r: r['pcost'].update(doubled=r['pcost']['doubled'][:1]), 'blocked (control C-E5.1: incomplete)'),
                ('d2', lambda r: r['pcost']['doubled'].append(dict(r['pcost']['doubled'][0])), 'blocked (control C-E5.1: incomplete)'),
                ('d3', lambda r: r['pcost']['doubled'][3].update(drift_ok=False), 'blocked (control C-E5.1: incomplete)'),
                ('x1', lambda r: r['pcross'].update(stage2=r['pcross']['stage2'][1:]), 'blocked (P-cross triplets: incomplete)'),
                ('x2', lambda r: r['pcross']['stage2'].append(dict(r['pcross']['stage2'][0])), 'blocked (P-cross triplets: incomplete)'),
                ('c1', lambda r: r['pcost'].update(invocations=r['pcost']['invocations'][:-1]), 'blocked (P-cost triplets: incomplete)')):
            d = write_job(t, name, 'A100-80G', 43, ncase=5, reps=2)
            _rewrite(d, fn)
            j = E5A.load_job('A100-80G', d, p1ref=ref)
            assert want in j['reasons'], (name, j['reasons'])
            r = E5A.smoke(j)
            assert r['ALL'] is False, name
            if name.startswith('x'):
                assert all(v['label'].startswith('unavailable') for v in r['per_job']['S'].values())


def test_smoke_expected_gpu_type():
    with Tmp() as t:
        ref = make_ref(t)
        h = E5A.load_job('H200', write_job(t, 'h1', 'H200', 44, ncase=5, reps=2), p1ref=ref)
        assert E5A.smoke(h, 'H200')['ALL'] is True
        assert E5A.smoke(h, 'A100-80G')['ALL'] is False
        import subprocess
        import sys
        out = Path(t) / 'smk.json'
        rc = subprocess.run([sys.executable, '-c', 'import sys, e5_tests, e5analyze; '
                             f"e5_tests.write_job({t!r}, 'h2', 'H200', 45, ncase=5, reps=2); "
                             "sys.argv = ['e5analyze.py', '--smoke', '--gpu-type', 'H200', '--job', "
                             f"{str(Path(t) / 'h2')!r}, '--p1ref', {ref!r}, '--out', {str(out)!r}]; e5analyze.main()"],
                            cwd=str(HERE), capture_output=True, text=True)
        assert rc.returncode == 0 and json.loads(out.read_text())['ALL'] is True, rc.stderr[-800:]


# ------------------------------------------------------------------ code audit 5: equivalence population, G-E5.2 ----

def test_equivalence_population_required():
    with Tmp() as t:
        ref = make_ref(t)
        d0 = write_job(t, 'e0', 'A100-80G', 50, ncase=5, reps=2)
        scored = EC.E5_CONTRACT[json.loads((d0 / 'result.json').read_text())['attempt']]['scored']
        assert E5A.load_job('A100-80G', d0, p1ref=ref)['status'] == 'ok'
        cases = [(f'del{i}', (lambda i: lambda r: r['pcost'].update(equivalence=[x for x in r['pcost']['equivalence']
                                                                                 if x['subject'] != scored[i]]))(i))
                 for i in range(len(scored))]                              # delete each required record in turn
        cases += [('dup', lambda r: r['pcost']['equivalence'].append(dict(r['pcost']['equivalence'][0]))),
                  ('fail', lambda r: r['pcost']['equivalence'][1].update(passed=False)),
                  ('anon', lambda r: r['pcost'].update(equivalence=[dict(passed=True)])),
                  ('none', lambda r: r['pcost'].update(equivalence=[]))]
        for name, fn in cases:
            d = write_job(t, name, 'A100-80G', 50, ncase=5, reps=2)
            _rewrite(d, fn)
            j = E5A.load_job('A100-80G', d, p1ref=ref)
            assert 'blocked (control: split/fused equivalence)' in j['reasons'], (name, j['reasons'])
            assert E5A.smoke(j, 'A100-80G', ref)['ALL'] is False, name


def test_smoke_reports_g_e5_2_descriptive():
    with Tmp() as t:
        ref = make_ref(t)
        d = write_job(t, 'a1', 'A100-80G', 51, ncase=5, reps=2)
        (Path(ref) / 'result.json').write_text(json.dumps(json.loads((d / 'result.json').read_text())['pcost']))
        r = E5A.smoke(E5A.load_job('A100-80G', d, p1ref=ref), 'A100-80G', ref)
        g = r['g_e5_2_descriptive']
        assert r['ALL'] is True and g['role'].startswith('descriptive') and g['result']['cells']
        assert all(v['within'] for v in g['result']['cells'].values())
        d2 = write_job(t, 'a2', 'A100-80G', 52, ncase=5, reps=2, cost_scale={(R128, 1024): 3.})
        r = E5A.smoke(E5A.load_job('A100-80G', d2, p1ref=ref), 'A100-80G', ref)
        assert r['ALL'] is True and not r['g_e5_2_descriptive']['result']['cells'][f'{R128}|1024']['within']   # no veto
        r = E5A.smoke(E5A.load_job('A100-80G', d2, p1ref=make_ref(t + '/x')), 'A100-80G', make_ref(t + '/x'))
        assert 'error' in r['g_e5_2_descriptive']['result'] and r['ALL'] is True          # no P1 result.json: still no veto


# ------------------------------------------------------------------ job audit E5: device, duplicates, coverage, telemetry ----

def test_resolve_executing_device():
    rows = [smi_row('GPU-a', idx='0', bus='00000000:07:00.0'), smi_row('GPU-b', idx='1', bus='00000000:0F:00.0')]
    r = EC.resolve_device(rows, 0, cvd='1', cu_uuid='GPU-b')
    assert r['resolved'] and r['uuid'] == 'GPU-b' and r['row']['index'] == '1'          # driver uuid wins over position
    assert EC.resolve_device(rows, 0, pci_bus_id='0000:0f:00.0')['uuid'] == 'GPU-b'
    assert EC.resolve_device(rows, 0, cvd='GPU-a,GPU-b')['uuid'] == 'GPU-a'
    assert EC.resolve_device(rows, 0, cvd='1', device_order='PCI_BUS_ID')['uuid'] == 'GPU-b'
    assert not EC.resolve_device(rows, 0, cvd='1')['resolved']                           # index without PCI order: ambiguous
    assert not EC.resolve_device(rows, 0)['resolved']
    assert not EC.resolve_device(rows, 0, cvd='GPU-a', cu_uuid='GPU-b')['resolved']      # conflicting evidence
    assert not EC.resolve_device(rows, 0, cu_uuid='GPU-zzz')['resolved']                 # uuid with no smi row
    assert EC.resolve_device(rows[:1], 0)['method'] == 'single visible GPU'
    assert EC.gpu_type_ok('A100-80G', rows[0])[0] and not EC.gpu_type_ok('H200', rows[0])[0]
    assert not EC.gpu_type_ok('A100-80G', None)[0]
    txt = ', '.join(rows[0][k] for k in EC.SMI_KEYS)
    assert EC.parse_smi(txt) == [rows[0]]


def test_telemetry_fails_closed():
    u = 'GPU-x'
    assert EC.telemetry_problems(telem('start', u), u) == []
    assert EC.telemetry_problems(telem('start', u, {'clocks.applications.graphics': 'unsupported',
                                                    'clocks.applications.memory': 'unsupported'}), u) == []
    for bad in ({'clocks.sm': '[N/A]'}, {'power.limit': ''}, {'temperature.gpu': 'ERROR'}, {'uuid': 'GPU-other'},
                {'clocks.applications.graphics': '[N/A]'}, {'driver_version': '[Not Supported]'}):
        assert EC.telemetry_problems(telem('start', u, bad), u), bad
    r = telem('start', u)
    r['env']['JAX_DEFAULT_MATMUL_PRECISION'] = None
    assert EC.telemetry_problems(r, u)
    r = telem('start', u)
    r['versions'].pop('jaxlib')
    assert EC.telemetry_problems(r, u)
    assert EC.telemetry_problems(None, u) == ['record missing']
    with Tmp() as t:
        ref = make_ref(t)
        for name, fn in (('t1', lambda r: r.update(records=r['records'][:2])),                        # no end record
                         ('t2', lambda r: r['records'][2]['nvidia_smi'].update(error='ERROR: nvidia-smi failed')),
                         ('t3', lambda r: r['records'][0]['nvidia_smi']['row'].update({'clocks.mem': '[N/A]'})),
                         ('t4', lambda r: r.update(device=dict(resolved=False, reason='x'))),
                         ('t5', lambda r: r.update(job_id=None))):
            d = write_job(t, name, 'A100-80G', 60, ncase=5, reps=2)
            _rewrite(d, fn)
            j = E5A.load_job('A100-80G', d, p1ref=ref)
            assert j['status'].startswith('failed'), (name, j['status'])
            assert E5A.smoke(j, 'A100-80G')['ALL'] is False


def test_duplicate_jobs_rejected_and_same_gpu_labelled():
    with Tmp() as t:
        ref = make_ref(t)
        js = four_jobs(t)
        res = E5A.analyze(js[:3] + [E5A.load_job('H200', js[2]['dir'], p1ref=ref)], EXPECT)        # same dir twice
        assert res['duplicates'] and res['types_complete']['H200'] is False
        assert res['phi']['phi1'] == 'unavailable (job missing)'
        d = write_job(t, 'h9', 'H200', 77)
        _rewrite(d, lambda r: r.update(job_id=js[2]['job_id']))                                     # same SLURM job id
        res = E5A.analyze(four_jobs(t + '/b')[:3] + [E5A.load_job('H200', d, p1ref=ref)], EXPECT)
        assert res['types_complete']['H200'] is True                                                 # different id there
        res = E5A.analyze(js[:3] + [E5A.load_job('H200', d, p1ref=ref)], EXPECT)
        assert res['duplicates'] and res['types_complete']['H200'] is False
        d = write_job(t, 'a9', 'A100-80G', 78)                                                       # same physical GPU
        _rewrite(d, lambda r: [r.update(device=dict(r['device'], uuid=js[0]['device_uuid']), device_uuid=js[0]['device_uuid']),
                               [x['nvidia_smi']['row'].update(uuid=js[0]['device_uuid']) for x in r['records']]])
        j = E5A.load_job('A100-80G', d, p1ref=ref)
        assert j['status'] == 'ok'
        res = E5A.analyze([js[0], j, js[2], js[3]], EXPECT)
        assert res['same_device']['A100-80G']['label'] == 'same physical GPU'
        assert res['same_device']['H200']['label'] == 'distinct GPUs'
        assert res['unit_to_unit']['A100-80G']['device'] == 'same physical GPU'
        assert res['phi']['phi1'] != 'unavailable (job missing)'


def test_smoke_requires_full_panel_coverage():
    with Tmp() as t:
        ref = make_ref(t)
        for name, sub in (('cv1', CC1), ('cv2', f'256|{R384}')):
            d = write_job(t, name, 'A100-80G', 70, ncase=5, reps=2)

            def inval(r, sub=sub):
                rows = r['pcross']['stage2'] if not sub[0].isdigit() else r['pcost']['invocations']
                x = next(x for x in rows if x['B'] == sub and x['case'] == 'dev6|3')
                x['valid'] = False
            _rewrite(d, inval)
            j = E5A.load_job('A100-80G', d, p1ref=ref)
            assert j['status'] == 'ok', j['status']                                  # rows complete, one invalid timing
            r = E5A.smoke(j, 'A100-80G')
            assert r['ALL'] is False, name
            bad = [k for k, v in r['checks'].items() if not v['ok']]
            assert len(bad) == 1 and ('P-cross' in bad[0] if name == 'cv1' else 'P-cost' in bad[0]), bad


# ------------------------------------------------------------------ job audit round 2, item 8 ----

class _Dist:
    def __init__(self, name, version):
        self.metadata, self.version = {'Name': name}, version


def test_telemetry_round2_versions_unknown_error_three_records():
    v = EC.versions([_Dist('jax', '0.10.2'), _Dist('jaxlib', '0.10.2'), _Dist('jax_cuda12_plugin', '0.10.2'),
                     _Dist('jax-cuda12-pjrt', '0.10.2')])
    assert v['jax_cuda_plugin'] == 'jax-cuda12-plugin==0.10.2' and v['jax_cuda_pjrt'] == 'jax-cuda12-pjrt==0.10.2'
    v = EC.versions([_Dist('jax', '0.10.2'), _Dist('jaxlib', '0.10.2')])
    assert v['jax_cuda_plugin'] == 'not installed' and v['jax_cuda_pjrt'] == 'not installed'
    u = 'GPU-x'
    for mut in (lambda r: r['versions'].update(jax_cuda_plugin='not installed'), lambda r: r['versions'].pop('jax_cuda_pjrt'),
                lambda r: r['nvidia_smi'].update(cuda_driver_version=None),
                lambda r: r['nvidia_smi'].update(cuda_driver_version='ERROR: x'),
                lambda r: r['nvidia_smi']['row'].update({'power.draw': '[Unknown Error]'}),
                lambda r: r['nvidia_smi']['row'].update({'clocks.applications.graphics': '[Unknown Error]'})):
        r = telem('start', u)
        mut(r)
        assert EC.telemetry_problems(r, u), r
    assert EC._unsupported('[N/A]') and EC._unsupported('[Not Supported]') and not EC._unsupported('[Unknown Error]')
    with Tmp() as t:
        ref = make_ref(t)
        for name, fn in (('b1', lambda r: r.update(records=[x for x in r['records'] if x['tag'] != 'between-panels'])),
                         ('b2', lambda r: r['records'][1]['nvidia_smi']['row'].update({'temperature.gpu': '[Unknown Error]'})),
                         ('b3', lambda r: r['records'].append(dict(telem('pcross-after-warm', r['device_uuid']),
                                                                   env={'JAX_DEFAULT_MATMUL_PRECISION': 'default'})))):
            d = write_job(t, name, 'A100-80G', 80, ncase=5, reps=2)
            _rewrite(d, fn)
            j = E5A.load_job('A100-80G', d, p1ref=ref)
            assert j['status'].startswith('failed') and 'telemetry' in j['status'], (name, j['status'])
            assert E5A.smoke(j, 'A100-80G')['ALL'] is False


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
