"""Smoke acceptance (DESIGN 7, A1.6, A2.2, A3.6): every gate must pass and every control must fire on at least one smoke
case, each predicate recomputed from the raw outputs by the offline analyses (anaref, ana2d, ana7 / ana3d), never read
from a job's own verdict field alone. Missing coverage fails (fail-closed). Exit status 1 if anything fails.
Writes checks/smoke-<attempt>.json with one row per predicate.

    python smoke_accept.py --attempt smk1          (2D smoke)
    python smoke_accept.py --attempt smk2 --dim 3  (3D smoke)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = '/home/tahmid/Dev/.venv/bin/python'


def run(args):
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_DEFAULT_MATMUL_PRECISION='highest')
    r = subprocess.run([PY] + args, cwd=HERE, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f'{args[0]} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}')
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--dim', type=int, default=2)
    a = ap.parse_args()
    arc = HERE / 'runs' / a.attempt / 'archive'
    out = arc / 'output'
    man = arc / 'OUTPUTS.sha256'
    logs = '\n'.join(p.read_text() for p in (arc / 'logs').glob('*.out'))
    errs = '\n'.join(p.read_text() for p in (arc / 'logs').glob('*.err'))
    both = logs + '\n' + errs
    rows = []
    add = lambda name, kind, ok, detail=None: rows.append(dict(name=name, kind=kind, passed=bool(ok), detail=detail))
    add('jax_backend=gpu', 'gate', 'jax_backend=gpu' in logs and 'jax_backend=cpu' not in logs)
    add('ALL-DONE', 'gate', 'ALL-DONE' in logs)
    add('no captured-constants warning (stdout and stderr)', 'gate', 'constants were captured' not in both)
    add('no Traceback / OOM / Killed in either stream', 'gate', not any(t in both for t in ('Traceback', 'RESOURCE_EXHAUSTED',
                                                                                         'Out of memory', 'Killed')))
    add('every step COMPLETE', 'gate', all((d / 'COMPLETE').exists() for d in out.iterdir() if d.is_dir()))
    ra = HERE / 'checks' / f'refs-{a.attempt}.json'
    run(['anaref.py', '--run', str(out), '--out', str(ra)])
    R = json.loads(ra.read_text())
    s = R['summary']
    if a.dim == 2:
        add('G-R1 (E0 reproduction, T = 0.25)', 'gate', s['G_R1'], R['G_R1'])
    add('G-R2 (final triple, first six outputs vs E0)', 'gate', s['G_R2'], [c['G_R2'] for c in R['cases']])
    add('G-R3 (fine vs lattice shared nodes; flag and LOCAL recomputation for every case)', 'gate',
        s['G_R3'] and s['G_R3_local_all'])
    add('judge recomputed offline agrees (whole run and windows)', 'gate', s['judge_agree'])
    add('C-R1 (first-order ladder fails order, prerequisites true)', 'control', s['C_R1'], R['x1'])
    refdir = next(out.glob('ref_refrun_*_ref'))
    if a.dim == 2:
        tb = json.loads((HERE / 'checks' / 'test_budcore.json').read_text())
        add('G-C1 (budcore nout = 6 bitwise p1core; prefix)', 'gate', tb['passed'])
        b2 = next(out.glob('b2run_*b1024*'))
        bj = HERE / 'checks' / f'budget2d-{a.attempt}.json'
        run(['ana2d.py', '--run', str(b2), '--minpop', '1', '--out', str(bj)])
        B = json.loads(bj.read_text())
        g = B['gates']
        add('B2 produced for every CONFIGURED member', 'gate', B['coverage']['passed'], B['coverage'])
        add('G-B0 (floor identity, every corner; n > 0)', 'gate', g['G_B0']['passed'], g['G_B0'])
        add('G-B0 analytic fixtures break', 'gate', g['B0_fixtures'])
        add('G-B0 real-data mismatched normaliser breaks', 'gate', g['B0_real_mismatch']['passed'], g['B0_real_mismatch'])
        add('G-B2 (Moebius closure; n > 0)', 'gate', g['G_B2']['passed'], g['G_B2'])
        add('G-B3 (every deployed corner reproduces the qualify / P1 archive)', 'gate', g['G_B3']['passed'], g['G_B3'])
        add('G-B4 (converged rule, every Q corner type and the u_star family)', 'gate',
            all(not v['failed'] and v['star_family']['passed'] for v in g['G_B4'].values()),
            {k: dict(failed=v['failed'], star=v['star_family']) for k, v in g['G_B4'].items()})
        add('G-B5 (LS init accepted)', 'gate', bool(g['G_B5']) and all(g['G_B5'].values()), g['G_B5'])
        add('G-B6 (saved logical tuples of every corner match the configuration)', 'gate', g['G_B6']['passed'], g['G_B6'])
        c = B['controls']
        add('G-B1 (sham switches bitwise)', 'gate', c.get('G_B1', {}).get('passed'), c.get('G_B1'))
        add('C-Q (>= 1 smoke case, valid comparator)', 'control', c['C_Q']['fired'], c['C_Q'])
        add('C-S (>= 1 smoke case, converged comparator)', 'control', c['C_S']['fired_any'], c['C_S'])
        add('C-I', 'control', c['C_I']['fired'], c['C_I'])
        add('C-F', 'control', c['C_F']['fired'], c['C_F'])
        add('C-score (identity errors on the scoring path)', 'control', c['C_score']['fired'], c['C_score'])
        ct = c.get('C_T', {})
        add('C-T (>= 1 smoke case)', 'control', bool(ct.get('values')) and any(x > 1e-3 for x in ct['values']), ct)
        add('G-Q (budget-0 certifier fires; wrong table raises)', 'control', c.get('G_Q', {}).get('fired'), c.get('G_Q'))
        b7j = HERE / 'checks' / f'b7b8-{a.attempt}.json'
        run(['ana7.py', '--run', str(b2), '--refs', str(refdir), '--manifest', str(man), '--anaref', str(ra), '--fine',
             str(refdir), '--minpop', '1', '--out', str(b7j)])
        S7 = json.loads(b7j.read_text())
        add('G-L1 (every B7 run reproduces the archive on its first six outputs)', 'gate',
            bool(S7['b7']) and all(v['G_L1']['passed'] for v in S7['b7'].values()), {k: v['G_L1'] for k, v in S7['b7'].items()})
        b8 = S7['b8']
        add('B7 and B8 produced for every CONFIGURED member', 'gate', S7['coverage']['passed'], S7['coverage'])
        add('C-A (alias: raw detection everywhere; production classifier says inconsistent on >= 1 resolved case)', 'control',
            bool(b8) and any(v['C_A']['resolved'] for v in b8.values()) and
            all(all(v['C_A']['raw']) and (v['C_A']['fired'] or not v['C_A']['resolved']) for v in b8.values() if v['C_A']['raw']),
            {k: v['C_A'] for k, v in b8.items()})
        add('C-E (balance closed form)', 'control', bool(b8) and all(v['C_E']['fired'] for v in b8.values()),
            {k: v['C_E'] for k, v in b8.items()})
        add('G-D1 (zero walls, ROM and reference)', 'gate', bool(b8) and all(v['G_D1'] for v in b8.values()) and
            all(r_.get('ref_walls_zero', True) for v in S7['b7'].values() for r_ in v['rows']))
        add('fine-route recomputation offline (1e-9)', 'gate', S7.get('fine_recompute', {}).get('passed'), S7.get('fine_recompute'))
        rows.append(dict(name='B7 Q1/Q2 at T (reported, no smoke verdict)', kind='report', passed=True,
                         detail={k: dict(Q1=v['Q1'], Q2=v['Q2']) for k, v in S7['b7'].items()}))
        pil = next(out.glob('b2run_*pilot4096*'), None)
        peak = None
        if pil:
            pr = json.loads((pil / 'result.json').read_text())
            peak = max((v.get('peak', -1) for x in pr.get('pilot', {}).values() for v in x.values() if isinstance(v, dict)), default=-1)
        add('4096^2 combined-path memory pilot ran and peak <= 120 GiB', 'gate', peak is not None and 0 < peak <= 120 * 2 ** 30,
            dict(peak_GiB=(peak or 0) / 2 ** 30))
    else:
        b3 = next(out.glob('b3run_*'))
        bj = HERE / 'checks' / f'b3-{a.attempt}.json'
        run(['ana3d.py', '--run', str(b3), '--refs', str(refdir), '--manifest', str(man), '--anaref', str(ra), '--minpop', '1',
             '--out', str(bj)])
        B = json.loads(bj.read_text())
        add('3D B2 and B7 produced for every CONFIGURED member', 'gate', B['coverage']['passed'], B['coverage'])
        for k in ('G_B0', 'G_B2', 'G_B3'):
            add(f'{k.replace("_", "-")} (3D)', 'gate', B['gates'][k]['passed'], B['gates'][k])
        add('G-B4 (3D, every Q corner type and the u_star family)', 'gate',
            bool(B['members']) and all(not v['failed'] and v['star_family']['passed'] for v in B['members'].values()),
            {k: dict(failed=v['failed'], star=v['star_family']) for k, v in B['members'].items()})
        add('G-L1 (3D)', 'gate', bool(B['b7']) and all(v['G_L1']['passed'] for v in B['b7'].values()),
            {k: v['G_L1'] for k, v in B['b7'].items()})
        add('C-Q (3D, >= 1)', 'control', B['controls']['C_Q']['fired'], B['controls']['C_Q'])
        add('C-T (3D, >= 1)', 'control', B['controls']['C_T']['fired_any'], B['controls']['C_T'])
        add('C-S (3D, >= 1)', 'control', B['controls']['C_S']['fired_any'], B['controls']['C_S'])
        add('C-F (3D)', 'control', B['controls']['C_F']['fired'], B['controls']['C_F'])
        add('C-score (3D)', 'control', B['controls']['C_score']['fired'], B['controls']['C_score'])
        add('G-Q (3D q0 certifier)', 'control', (B['controls'].get('G_Q') or {}).get('fired'))
        rows.append(dict(name='3D B7 Q1/Q2 at T (reported)', kind='report', passed=True,
                         detail={k: dict(Q1=v['Q1'], Q2=v['Q2']) for k, v in B['b7'].items()}))
    ok = all(r['passed'] for r in rows if r['kind'] in ('gate', 'control'))
    dest = HERE / 'checks' / f'smoke-{a.attempt}.json'
    dest.write_text(json.dumps(dict(attempt=a.attempt, all_passed=ok, rows=rows), indent=1, default=float) + '\n')
    for r in rows:
        print(('PASS ' if r['passed'] else 'FAIL ') + r['kind'][:7].ljust(8) + r['name'])
    print('ALL', ok)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
