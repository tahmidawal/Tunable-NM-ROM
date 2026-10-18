"""Smoke acceptance (DESIGN 7, A1.6, A2.2, A3.6): every gate must pass and every control must fire on at least one smoke
case, each predicate recomputed from the raw outputs by the offline analyses (anaref, ana2d, ana7 / ana3d), never
read from a job's own verdict field alone. Writes checks/smoke-<attempt>.json with one row per predicate.

    python smoke_accept.py --attempt smk1          (2D smoke)
    python smoke_accept.py --attempt smk2 --dim 3  (3D smoke)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = '/home/tahmid/Dev/.venv/bin/python'


def run(args):
    env = dict(__import__('os').environ, JAX_PLATFORMS='cpu', JAX_DEFAULT_MATMUL_PRECISION='highest')
    r = subprocess.run([PY] + args, cwd=HERE, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f'{args[0]} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}')
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--dim', type=int, default=2)
    a = ap.parse_args()
    out = HERE / 'runs' / a.attempt / 'archive' / 'output'
    logs = '\n'.join(p.read_text() for p in (out.parent / 'logs').glob('*.out'))
    rows = []
    add = lambda name, kind, ok, detail=None: rows.append(dict(name=name, kind=kind, passed=bool(ok), detail=detail))
    add('jax_backend=gpu', 'gate', 'jax_backend=gpu' in logs and 'jax_backend=cpu' not in logs)
    add('ALL-DONE', 'gate', 'ALL-DONE' in logs)
    add('no captured-constants warning', 'gate', 'constants were captured' not in logs)
    # references
    ra = HERE / 'checks' / f'refs-{a.attempt}.json'
    run(['anaref.py', '--run', str(out), '--out', str(ra)])
    R = json.loads(ra.read_text())
    s = R['summary']
    if a.dim == 2:
        add('G-R1 (E0 reproduction, T = 0.25)', 'gate', s['G_R1'], R['G_R1'])
    add('G-R2 (T = 0.5 first six outputs vs E0)', 'gate', s['G_R2'], [c.get('G_R2') for c in R['cases']])
    add('G-R3 (fine vs lattice shared nodes)', 'gate', s['G_R3'])
    add('judge recomputed offline agrees', 'gate', s['judge_agree'])
    add('C-R1 (first-order ladder fails order)', 'control', s['C_R1'], R['x1'])
    if a.dim == 2:
        tb = json.loads((HERE / 'checks' / 'test_budcore.json').read_text())
        add('G-C1 (budcore nout = 6 bitwise p1core; prefix)', 'gate', tb['passed'])
        b2 = next(out.glob('b2run_*b1024*'))
        refdir = next(out.glob('ref_refrun_*_ref'))
        bj = HERE / 'checks' / f'budget2d-{a.attempt}.json'
        run(['ana2d.py', '--run', str(b2), '--minpop', '1', '--out', str(bj)])
        B = json.loads(bj.read_text())
        g = B['gates']
        add('G-B0 (floor identity, every corner)', 'gate', g['G_B0']['passed'], g['G_B0'])
        add('G-B0 analytic fixtures break', 'gate', g['B0_fixtures'])
        add('G-B2 (Moebius closure)', 'gate', g['G_B2']['passed'], g['G_B2'])
        add('G-B3 (reproduces qualify/P1 archive)', 'gate', g['G_B3']['passed'], g['G_B3'])
        add('G-B4 (converged rule, every Q corner type)', 'gate',
            all(all(v['per_corner'].values()) and v['star_family'] for v in g['G_B4'].values()),
            {k: v['worst'] for k, v in g['G_B4'].items()})
        add('G-B5 (LS init accepted)', 'gate', all(v.get('G_B5', True) for k, v in g['G_B5'].items() if k.startswith('LS')))
        c = B['controls']
        add('G-B1 (sham switches bitwise)', 'gate', c['G_B1']['offline_bitwise'] and c['G_B1']['sham_rule_bitwise']
            and c['G_B1']['prod_rerun_bitwise'])
        for k in ('C_Q', 'C_S', 'C_I', 'C_F', 'C_score'):
            add(k.replace('_', '-'), 'control', c.get(k, {}).get('fired'), {kk: vv for kk, vv in c.get(k, {}).items() if kk != 'rows'})
        ct = c.get('C_T', {})
        add('C-T (smoke: >= 1 case)', 'control', bool(ct.get('values')) and any(x > 1e-3 for x in ct['values']), ct)
        gq = c.get('G_Q', {})
        add('G-Q (budget-0 certifier fires; wrong table raises)', 'control',
            all(v.get('fired') for k, v in gq.items() if k.startswith('GQ_budget0')) and gq.get('GQ_wrong_table', {}).get('raised'))
        # B7 / B8
        b7j = HERE / 'checks' / f'b7b8-{a.attempt}.json'
        run(['ana7.py', '--run', str(b2), '--refs', str(refdir), '--anaref', str(ra), '--out', str(b7j)])
        S7 = json.loads(b7j.read_text())
        add('G-L1 (B7 first six outputs reproduce the archive)', 'gate',
            all(v['G_L1']['passed'] for v in S7['b7'].values() if v['G_L1']['n']), {k: v['G_L1'] for k, v in S7['b7'].items()})
        b8 = S7['b8']
        add('C-A (alias: raw detection and classifier)', 'control', bool(b8) and any(v['C_A']['raw'] for v in b8.values())
            and all(all(v['C_A']['raw']) and all(v['C_A']['classifier']) for v in b8.values() if v['C_A']['raw']),
            {k: v['C_A'] for k, v in b8.items()})
        add('C-E (balance closed form)', 'control', bool(b8) and all(v['C_E']['fired'] for v in b8.values()),
            {k: v['C_E'] for k, v in b8.items()})
        add('G-D1 (zero walls)', 'gate', bool(b8) and all(v['G_D1'] for v in b8.values()))
        rows.append(dict(name='B7 Q1/Q2 at T (reported, no smoke verdict)', kind='report', passed=True,
                         detail={k: dict(Q1=v['Q1'], Q2=v['Q2']) for k, v in S7['b7'].items()}))
        pil = next(out.glob('b2run_*pilot4096*'), None)
        if pil:
            pr = json.loads((pil / 'result.json').read_text())
            peak = max(v.get('peak', -1) for x in pr.get('pilot', {}).values() for v in x.values() if isinstance(v, dict))
            add('4096^2 combined-path memory pilot peak <= 120 GiB', 'gate', 0 < peak <= 120 * 2 ** 30,
                dict(peak_GiB=peak / 2 ** 30, record=pr.get('pilot')))
    else:
        b3 = next(out.glob('b3run_*'))
        bj = HERE / 'checks' / f'b3-{a.attempt}.json'
        refdir = next(out.glob('ref_refrun_*_ref'))
        run(['ana3d.py', '--run', str(b3), '--refs', str(refdir), '--anaref', str(ra), '--minpop', '1', '--out', str(bj)])
        B = json.loads(bj.read_text())
        add('G-B0 (3D)', 'gate', B['gates']['G_B0']['passed'], B['gates']['G_B0'])
        add('G-B3 (3D, reproduces qualify q3d)', 'gate', B['gates']['G_B3']['passed'], B['gates']['G_B3'])
        add('G-B4 (3D)', 'gate', all(all(v['G_B4'].values()) for v in B['members'].values()),
            {k: v['G_B4'] for k, v in B['members'].items()})
        for k in ('C_Q', 'C_T', 'C_S', 'C_F'):
            add(k.replace('_', '-') + ' (3D)', 'control', B['controls'].get(k, {}).get('fired'), B['controls'].get(k))
        add('G-Q (3D q0 certifier)', 'control', (B['controls'].get('G_Q') or {}).get('fired'))
    ok = all(r['passed'] for r in rows if r['kind'] in ('gate', 'control'))
    dest = HERE / 'checks' / f'smoke-{a.attempt}.json'
    dest.write_text(json.dumps(dict(attempt=a.attempt, all_passed=ok, rows=rows), indent=1, default=float) + '\n')
    for r in rows:
        print(('PASS ' if r['passed'] else 'FAIL ') + r['kind'][:7].ljust(8) + r['name'])
    print('ALL', ok)


if __name__ == '__main__':
    main()
