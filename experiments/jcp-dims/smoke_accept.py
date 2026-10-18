"""jcp-dims smoke acceptance (DESIGN 7, A1.1-A1.4, A2.1-A2.4, A3, A4): every required control recomputed from the smoke
archives (smk2: dacc smk2 + dtime smk2t; smk3: d3run smk3) and the CPU checks. Writes checks/smoke-accept-<part>.json;
exit 1 if any required check fails. CPU only.

    JAX_PLATFORMS=cpu python smoke_accept.py smk2|smk3
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import dscore as DSC
import qscore as SC
import npcheck as NC

HERE = Path(__file__).resolve().parent
QARCH = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/smkA/archive/output/qacc_smkA')
OUT = {}


def req(name, ok, **kw):
    OUT[name] = dict(required=True, passed=bool(ok), **kw)
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)


def rep(name, **kw):
    OUT[name] = dict(required=False, **kw)


def rel_w(Wa, Wb):
    return float(np.max([np.max(np.linalg.norm(Wa[j] - Wb[j], axis=1)) / np.linalg.norm(Wb[j][0]) for j in range(len(Wa))]))


def smk2_acc():
    d = DSC.out_dir('smk2', 'dacc', 'smk2')
    req('smk2 dacc archive present', d is not None)
    if d is None:
        return
    res = json.loads((d / 'result.json').read_text())
    req('smk2 dacc complete', res['complete'])
    req('smk2 dacc memory: no rejection and job peak <= 120 GB', not res.get('memory_rejected')
        and res['memory']['end']['peak'] <= 120e9, peak=res['memory']['end']['peak'])
    rep('smk2 dacc peaks per setting', peaks={k: v.get('peak') for k, v in res['memory'].items() if isinstance(v, dict)})
    # G-modes
    z = DSC.load(d, 'R512M4096gauss256', 'LSPG', 'CNR')
    F = json.loads((HERE / 'inputs' / 'modes_F2048.json').read_text())
    req('G-modes: first 2048 entries of F_4096 equal P1 F_2048', np.array_equal(z['kx'][:2048], F['kx']) and np.array_equal(z['ky'][:2048], F['ky']))
    # reproduction gates (dev6, H200 lane vs A100 qualify smkA): <= 1e-10
    for setting, rp in (('R384M1536gauss128', 'R384'), ('R128M512fib4181', 'R128')):
        for sch in ('CNR', 'BE'):
            R = DSC.load(d, setting, 'LSPG', sch)
            A = dict(np.load(QARCH / f'R_{rp}_LSPG_{sch}_1.npz'))
            ids = [str(x) for x in R['case_ids']]
            ai = [str(x) for x in A['case_ids']]
            same = ids == ai and int(R['M']) == int(A['A'].shape[0]) and str(R['rule']) == str(A['rule'])
            w = rel_w(R['W'], A['W']) if same else float('inf')
            req(f'reproduction {setting} LSPG-{sch} vs qualify smkA <= 1e-10', same and w <= 1e-10, worst=w)
    # offline Q1 recomputation and labels for every smoke configuration
    rows = res['rows']
    for s in res['config']['settings']:
        for a in s['arms']:
            R = DSC.load(d, s['name'], a['form'], a['scheme'], a['factor'])
            rr = [r_ for r_ in rows if r_['setting'] == s['name'] and r_['arm'] == f"{a['form']}_{a['scheme']}_{float(a['factor']):g}"]
            q = DSC.q1(R, rr)
            k = f"{s['name']}|{a['form']}|{a['scheme']}"
            req(f'{k}: offline recomputation agrees with the job', not q['disagreements'],
                disagreements=[str(x) for x in q['disagreements'][:4]])
            info = dict(Q1=q['Q1'], pooled=q['pooled_noncertified'], steps=q['pooled_steps'], square=bool(R['square']))
            if bool(R['square']) and a['form'] == 'LSPG':
                info.update(jcond_max=float(np.nanmax(R['jcond'])), unweighted_max=float(np.nanmax(R['unweighted'])),
                            gamma_lit_median=float(np.nanmedian(R['gamma_lit'])))
                req(f'{k}: square diagnostics recorded (cond(J), unweighted residual)', np.isfinite(R['jcond']).any()
                    and np.isfinite(R['unweighted']).any())
            rep(f'{k}: Q1', **info)
            if s['name'] == 'R384M1536gauss128' and a['form'] == 'GAL':
                req('positive GAL fixture (R384 gauss128 GAL-CN dt0 dev6): certificate, G-cert, validity',
                    q['Q1']['certificate'] and q['Q1']['gcert'] and q['Q1']['all_valid'], Q1=q['Q1'])
            if s['name'] == 'R384M1536gauss128':
                zu = SC.zero_update_distance(R, 0)
                req(f'{k}: zero-update motion > 100 x 2.5e-4 (establishment)', zu['motion'] > 100 * 2.5e-4, motion=zu['motion'])
                req(f'{k}: zero-update control fires', zu['dist'] > 2.5e-4, dist=zu['dist'])
    # Q2 ladder path on the heavy R512 M4096 cell (gauss256 -> 320 -> 384)
    for form, sch in (('LSPG', 'CNR'), ('GAL', 'CN')):
        Rs = [DSC.load(d, f'R512M4096{g}', form, sch) for g in ('gauss256', 'gauss320', 'gauss384')]
        q2 = DSC.ladder_q2(Rs[0], Rs[1], Rs[2])
        rep(f'ladder Q2 path R512 M4096 gauss256 ({form}-{sch})', **{k: (str(v) if k == 'check' else v) for k, v in q2.items()})
        req(f'ladder Q2 path R512 M4096 ({form}) produced a label', isinstance(q2['label'], str))
    # controls block (qualify's run_controls, unchanged)
    c = res.get('controls', {})
    z = np.load(d / 'CTRL_under_resolved.npz')
    q2 = SC.q2_from_runs(z['W'], z['W_succ'], z['rho'], z['n0'], int(z['Rp']), bool(np.all(z['succ_valid'])))
    req('under-resolved rule (gauss32 -> gauss64): Q2 fail', q2['label'] == 'fail', worst=q2['worst_dist_qp'], rho=q2['rho_max'])
    for form in ('LSPG', 'GAL'):
        b = c.get(f'budget0_{form}', {})
        req(f'Q1 negative ({form}, budget 0): pooled non-certified > 1 %', (b.get('pooled_fraction') or 0) > .01,
            pooled=b.get('pooled_fraction'))
    req('wrong-mesh table refused', c.get('wrong_mesh', {}).get('passed'))
    # fixgate component (b)
    fg = res.get('fixgate', [])
    req('fixgate blocks ran at both family cells', len(fg) == 2)
    for f in fg:
        bad = f['rules'].get('badlat121393', {})
        req(f"fixgate R{f['r']} M{f['M']}: bad-vector rule flagged by component (b)", bad.get('flags') is True,
            rho_max=bad.get('rho_max'))
        if 'gauss384' in f['rules']:
            req(f"fixgate R{f['r']}: target against itself gives rho 0", max(f['rules']['gauss384']['rho_tgt']) == 0.)
    # draws: tables differ by draw, same family size
    meta = {s['name']: s for s in res['settings'].values()} if isinstance(res['settings'], dict) else {}
    shas = {k: v['meta']['main']['rule_sha256'] for k, v in res['settings'].items()}
    for fam in ('fib2584', 'sobol2048', 'halton2048', 'mc2048'):
        k1 = [k for k in shas if k.startswith('R128M512f' + fam)]
        req(f'draws of {fam} give different tables', len(set(shas[k] for k in k1)) == len(k1) >= 2, settings=k1)
    # LMF stability inputs
    lm, df = res['settings'].get('R384M1536lmfgauss256'), res['settings'].get('R384M1536gauss256')
    req('LMF list: rank(A) = r', lm is not None and lm['A_rank'] == 384, rank=lm and lm['A_rank'])
    rep('LMF stability: cond(A) ratio LMF / default (bar 10)', ratio=lm['A_cond'] / df['A_cond'] if lm and df else None,
        lmf=lm and lm['A_cond'], default=df and df['A_cond'])
    del meta


def smk2_time():
    d = DSC.out_dir('smk2', 'dtime', 'smk2t')
    req('smk2t dtime archive present', d is not None)
    if d is None:
        return
    res = json.loads((d / 'result.json').read_text())
    req('smk2t complete', res['complete'])
    req('smk2t memory: no rejection, every block released (<= 1 GB over base)', not res.get('memory_rejected')
        and all(b['in_use_over_base'] <= 1e9 for b in res['blocks']))
    req('split = fused (fields 1e-12, iterations equal)', all(x['passed'] for x in res['equivalence']))
    bad = res['config']['inject_identity_fault']
    inj = [x for x in res['invocations'] if x['B'] in bad]
    req('injected identity fault invalidates every triplet of its subject', inj and all(not x['valid'] for x in inj))
    ok = [x for x in res['invocations'] if x['B'] not in bad]
    req('identity holds for every other subject', ok and all(x['all_hashes_ok'] for x in ok))
    dbl = res['doubled']
    med = float(np.median([x['ratio'] for x in dbl])) if dbl else None
    req('doubled work B/A in [1.8, 2.2] with identical iterations and identity', dbl and 1.8 <= med <= 2.2 and
        all(x['same_iterations'] and x['identity_ok'] and x['drift_ok'] for x in dbl), median=med)
    req('invocation counts complete', res.get('counts_complete') is True)
    for lb, s in res['subjects'].items():
        rows = s['certification']
        req(f'{lb}: G-cert holds in the timing job', all(r_['gcert_ok'] for r_ in rows))
        if s['spec'].get('qpp'):
            p = d / f"S_{lb.replace('|', '_')}_qpp.npz"
            req(f'{lb}: second certification phase (q++) present', p.exists() and all('dist_qp_qpp' in r_ for r_ in rows))
        rep(f'timing-job certification {lb}', rows=[{k: r_.get(k) for k in ('case', 'valid', 'noncertified', 'dist_tight',
                                                                              'dist_qp', 'dist_qp_qpp')} for r_ in rows])


def cpu_checks():
    t = json.loads((HERE / 'checks' / 'test_dims.json').read_text())
    req('CPU decision-rule / generator tests all pass', t['all_pass'])
    f = json.loads((HERE / 'checks' / 'fixture-badlat.json').read_text())
    req('bad-vector fixtures admitted (independent CPU establishment)', all(p['admitted'] for p in f['panels']))


def smk3():
    d = DSC.out_dir('smk3', 'd3run', 'smk3')
    req('smk3 archive present', d is not None)
    if d is None:
        return
    res = json.loads((d / 'result.json').read_text())
    req('smk3 complete', res['complete'])
    req('smk3 memory: no rejection and job peak <= 120 GB', not res.get('memory_rejected') and res['peak_bytes_job'] <= 120e9,
        peak=res['peak_bytes_job'])
    B = {(b['kind'], b.get('label')): b for b in res['blocks']}
    ov = next(b['result'] for b in res['blocks'] if b['kind'] == 'override')
    req('3D M override: slices, N(c), shapes and independent mode hash', ov['passed'], **{k: ov[k] for k in ('A', 'lam', 'P', 'N')})
    fx = next(b['result'] for b in res['blocks'] if b['kind'] == 'fixgate')
    req('3D fixgate: bad-vector rule flagged by component (b)', fx['rules']['badlat262144']['flags'] is True,
        rho=max(fx['rules']['badlat262144']['rho_tgt']))
    req('3D fixgate: target against itself gives rho 0', max(fx['rules']['gl64']['rho_tgt']) == 0.)
    rd = next(b['result'] for b in res['blocks'] if b['kind'] == 'rounds')
    ctl = rd.get('control|W512|lat8192|fs|BE|0.01|K1')
    req('3D under-resolved rule (W512 lat8192 -> lat16384): rollout bar fails', ctl is not None and max(ctl['dist_qp']) > 2.5e-4,
        worst=ctl and max(ctl['dist_qp']))
    hv = rd.get('W1024|lat65536|fs|CN|0.01|K1')
    req('heavy round member (W1024 lat65536 + q+ + q++) completed on 64 cases', hv is not None and len(hv['dist_qp']) == 64)
    pan = res['timing']
    req('smoke panel accepted (certified links, hashes, inventory, drift)', pan['smoke']['valid'] is True, acc=pan['smoke']['acceptance'])
    req('uncertified panel rejected (control)', pan['smoke-uncertified']['valid'] is False,
        acc=pan['smoke-uncertified']['acceptance'])
    t1 = [x['ratio'] for x in pan['smoke']['invocations'] if x['B'].startswith('T1|')]
    req('T1 (C_old twice) ratio in [1.8, 2.2]', bool(t1) and 1.8 <= float(np.median(t1)) <= 2.2, median=float(np.median(t1)) if t1 else None)
    want = [(st, rl, f) for st, rls in (('W512k1', ('lat65536', 'lat131072')), ('W512', ('lat65536',)),
                                        ('W512k8', ('lat131072', 'lat262144'))) for rl in rls for f in ('LSPG', 'GAL')]
    for st, rl, f in want:                         # exact adaptive inventory; G-cert is an implementation check
        p = d / f'AD_{st}_{rl}_ad_{f}_CN_0.01.npz'
        req(f'adaptive {st} {rl} {f}-CN: evidence present', p.exists())
        if not p.exists():
            continue
        z = np.load(p)
        req(f'adaptive {st} {rl} {f}-CN: G-cert holds on every case', bool(z['gcert'].all()))
        req(f'adaptive {st} {rl} {f}-CN: square flag matches M = r', bool(z['square']) == (st == 'W512k1'))
        if st == 'W512k1' and f == 'LSPG':
            req(f'adaptive {st} {rl}: square diagnostics recorded', np.isfinite(z['jcond']).any() and np.isfinite(z['unweighted']).any())
        rep(f'adaptive {st} {rl} {f}-CN (scientific outcome, no gate)', valid=bool(z['valid'].all()), noncert=int(z['noncert'].sum()),
            worst_tight=float(np.max(z['dist_tight'])), worst_tight_tighter=float(np.max(z['dist_tight_tighter'])))
    # reproduction (default M) vs speed3d archives, offline
    W = np.load(d / 'W_smk3.npz')
    S3 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/runs')
    for key in ('W512|lat16384|fs|CN|0.01|K1', 'M2_512|gl24|fs|BDF2|0.0125|K1'):
        mine = W[key]
        found = None
        for g in ('g1', 'g2', 'g3'):
            for f in (S3 / g / 'archive').rglob('W_*.npz'):
                z = np.load(f)
                if key in z.files:
                    found = (z[key], list(z[f'cases|{key}']) if f'cases|{key}' in z.files else list(range(len(z[key]))))
                    break
            if found:
                break
        if found is None:
            req(f'3D reproduction {key}: speed3d archive found', False)
            continue
        Wa, ca = found
        w = max(float(np.max(np.linalg.norm(mine[i] - Wa[ca.index(i)], axis=1)) / np.linalg.norm(Wa[ca.index(i)][0])) for i in range(4))
        req(f'3D reproduction {key} (default M) vs speed3d <= 1e-10', w <= 1e-10, worst=w)
    del B


def main():
    part = sys.argv[1]
    cpu_checks()
    if part == 'smk2':
        smk2_acc()
        smk2_time()
    else:
        smk3()
    fails = [k for k, v in OUT.items() if v['required'] and not v['passed']]
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'smoke-accept-{part}.json').write_text(json.dumps(dict(results=OUT, failures=fails,
                                                                                accepted=not fails), indent=1, default=str) + '\n')
    print('ACCEPTED' if not fails else f'FAILURES ({len(fails)}): ' + '; '.join(fails))
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
