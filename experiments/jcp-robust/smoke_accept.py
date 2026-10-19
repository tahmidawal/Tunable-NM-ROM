"""jcp-robust smoke acceptance (DESIGN 6; A1.12, A2.9, A3.6, A4.2; code audit 2 #27). CPU only. Read-only on the smoke
outputs; every predicate is recomputed from the raw outputs through the production analysis code with the smoke's own
explicit population (its config), and the result is labelled smoke (never a verdict).

    python smoke_accept.py smk1 <archive output dir> [--fetched <dir>]
    python smoke_accept.py smk2 <archive output dir>
    python smoke_accept.py smk3 <archive output dir>
    python smoke_accept.py smk4 <archive output dir>
Writes checks/smoke-accept-<smk>.json with every check {name: {ok, evidence}} and ALL = every check ok.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = '/home/tahmid/Dev/.venv/bin/python'
ENV = dict(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', JAX_DEFAULT_MATMUL_PRECISION='highest')


def run(args):
    import os
    r = subprocess.run([PY] + args, cwd=HERE, env=dict(os.environ, **ENV), capture_output=True, text=True)
    return r.returncode, r.stdout[-4000:] + r.stderr[-4000:]


def chk(C, name, ok, ev=None):
    C[name] = dict(ok=bool(ok), evidence=ev)


def smk1(out, fetched):
    C = {}
    d2 = out / 'r2acc_smk1m1024'
    rc, log = run(['e1analyze.py', str(d2), '--tag', 'smoke-smk1-2d'] + (['--fetched', fetched] if fetched else []))
    chk(C, '2d analysis runs', rc == 0, log[-800:])
    if rc == 0:
        a = json.loads((HERE / 'checks' / 'smoke-smk1-2d.json').read_text())
        g = a['groups']['R128']
        chk(C, '2d job complete on gpu', a['complete'] and a['backend'] == 'gpu')
        for k in ('G-E1.1', 'G-E1.3'):
            chk(C, f'2d {k} passes', g['controls'].get(k, {}).get('passed'), g['controls'].get(k))
        chk(C, '2d C-E1.2 fires', g['controls'].get('C-E1.2', {}).get('fired'), g['controls'].get('C-E1.2'))
        chk(C, '2d C-E1.1 fires (admitted CPU fixture, in-job agreement 1 %, flagged)', g['controls'].get('C-E1.1', {}).get('fired'),
            g['controls'].get('C-E1.1'))
        for ak, av in g['arms'].items():
            chk(C, f'2d {ak} integrity (distances recomputed = in-job, G-cert, population)', av['integrity']['ok'], av['integrity'])
            chk(C, f'2d {ak} G-E1.2 reproduction on dev6', (av.get('G-E1.2') or {}).get('passed'), av.get('G-E1.2'))
            chk(C, f'2d {ak} target certified on dev6', av['target']['certified'], av['target']['check_rows'])
            chk(C, f'2d {ak} own-family labels computed for draws 0-2',
                all(av['rules'][f'd{s}']['own_family_qualified'] is not None for s in range(3)),
                {l: (r['status'], r['qualified'], r['own_family_qualified']) for l, r in av['rules'].items()})
            b = av['rules'].get('bad', {})
            chk(C, f'2d {ak} bad vector far from the target (evidence; D or invalid)',
                b and (b.get('prod_failed') or (b.get('D') is not None and b['D'] > 2.5e-4)), dict(D=b.get('D'), failed=b.get('prod_failed')))
    d3 = out / 'e1run3d_smk13d'
    sc = HERE / 'checks' / 'smoke-smk1-3d-scores.json'
    rc, log = run(['score3.py', '--run', str(d3), '--out', str(sc)])
    chk(C, '3d E0 scoring runs', rc == 0, log[-800:])
    if rc == 0:
        rc, log = run(['e1analyze3d.py', str(d3), str(sc), '--tag', 'smoke-smk1-3d'])
        chk(C, '3d analysis runs', rc == 0, log[-800:])
    if rc == 0:
        a = json.loads((HERE / 'checks' / 'smoke-smk1-3d.json').read_text())
        cv = a['configs']['E1-3D-a']
        ct = cv['controls']
        chk(C, '3d integrity (recomputed target / successor / ADT / tighter distances = in-job)', cv['integrity']['ok'], cv['integrity'])
        chk(C, '3d C-E1.1 fires (and job M = CPU M)', ct.get('C-E1.1', {}).get('fired'), ct.get('C-E1.1'))
        chk(C, '3d C-E1.2 fires', ct.get('C-E1.2', {}).get('fired'), ct.get('C-E1.2'))
        chk(C, '3d G-E1.1 passes', ct.get('G-E1.1', {}).get('passed'), ct.get('G-E1.1'))
        chk(C, '3d G-E1.3 passes', ct.get('G-E1.3', {}).get('passed'), ct.get('G-E1.3'))
        chk(C, '3d G-E1.2 reproduction on cases 0-7', ct.get('G-E1.2', {}).get('passed'), ct.get('G-E1.2'))
        chk(C, '3d target certified on cases 0-7', cv['target_cert']['certified'], cv['target_cert'])
        chk(C, '3d own-family labels computed for draws 0-2',
            all(cv['draws'][str(s)]['own_family_qualified'] is not None for s in range(3)),
            {s: (x['status'], x['own_family_qualified']) for s, x in cv['draws'].items()})
        bs = ct.get('bad_rollout_summary', {})
        chk(C, '3d bad vector far from the target (evidence)', bs.get('any_invalid') or E_num(bs.get('D')) > 2.5e-4, bs)
    return C


def E_num(x):
    return math.inf if x is None else float(x)


def smk2(out):
    C = {}
    import e1analyze as EA
    want = EA.expected_ids('dev6')
    d = out / 'r2acc_smk2r'
    res = json.loads((d / 'result.json').read_text())
    chk(C, 'incumbent-through-swap job complete on gpu', res.get('complete') and res['backend'] == 'gpu')
    chk(C, 'swap path loaded the incumbent files', res['checkpoint_sha256'] == '18f0266ae6f0' + res['checkpoint_sha256'][12:]
        and res['bank'] != 'incumbent', dict(ck=res['checkpoint_sha256'], bank=res['bank']))
    for r, lab, arm, nm in ((128, 'fib4181', 'LSPG_BE_1', 'R_R128_BE.npz'), (128, 'fib4181', 'LSPG_CNR_1', 'R_R128_CNR.npz'),
                            (384, 'gauss128', 'LSPG_BE_1', 'R_R384_BE.npz'), (384, 'gauss128', 'LSPG_CNR_1', 'R_R384_CNR.npz')):
        z = np.load(d / f'R_R{r}_{lab}_{arm}.npz')
        rp = EA.reproduce(z['W'], [str(x) for x in z['case_ids']], [('p1', 'acc1024', nm)], None, want)
        chk(C, f'G-E2.3 smoke replay R{r} {arm} (swap-neutral, dev6)', rp['passed'], rp)
    t = out / 'train2d_s1'
    chk(C, 'train2d smoke SEED-COMPLETE', (t / 'SEED-COMPLETE').exists())
    rr = t / 'run_record.json'
    if rr.exists():
        rec = json.loads(rr.read_text())
        chk(C, 'train2d smoke record (steps, rotation inverse check, device)', True, rec)
    for tag in ('smk2t', 'smk2u'):
        dd = out / f'r2acc_{tag}'
        r_ = json.loads((dd / 'result.json').read_text())
        chk(C, f'{tag} complete on gpu, bank runtime-hashed', r_.get('complete') and r_['backend'] == 'gpu'
            and (r_['bank'] or {}).get('runtime_hashed') is not False, dict(bank=r_.get('bank'), ck=r_['checkpoint_sha256']))
        chk(C, f'{tag} ran on the trained (not the incumbent) bank', not r_['checkpoint_sha256'].startswith('18f0266a'))
        g = r_['config']['groups'][0]
        for arm in g['arms']:
            ak = f"{arm['form']}_{arm['scheme']}_{float(arm['factor']):g}"
            import e2bank
            e2bank.use(dict(r_['config']['bank'], ckpt_sha256=r_['checkpoint_sha256'], rotation_sha256=r_['rotation_sha256']))
            av = EA.analyse_arm(dd, r_, g, ak, int(g['Rp']))
            chk(C, f'{tag} {ak} analysis integrity', av['integrity']['ok'], av['integrity'])
            chk(C, f'{tag} {ak} own-family labels computed', all(x['own_family_qualified'] is not None for x in av['rules'].values()),
                {l: (x['status'], x['own_family_qualified']) for l, x in av['rules'].items()})
    return C


def smk3(out):
    C = {}
    p = out / 'train3d_pilot'
    chk(C, 'train3d pilot SEED-COMPLETE', (p / 'SEED-COMPLETE').exists())
    if (p / 'run_record.json').exists():
        rec = json.loads((p / 'run_record.json').read_text())
        chk(C, 'train3d pilot device peak <= 120 GB (memory gate)', E_num(rec.get('device_peak_GiB')) * 2 ** 30 <= 120e9, rec)
    d = out / 'e2run3d_smk3q'
    res = json.loads((d / 'result.json').read_text())
    chk(C, 'e2run3d smoke complete on gpu', res.get('complete') and res['backend'] == 'gpu')
    chk(C, 'heaviest cell (W1024 gl64 + gl80) under the 120 GB device bar', not res.get('memory_rejected') and
        res.get('peak_bytes_job', math.inf) <= 120e9, dict(peak=res.get('peak_bytes_job'), rejected=res.get('memory_rejected')))
    lad = {}
    for blk in res['blocks']:
        if blk['kind'] == 'ladder':
            lad.update(blk['result'])
    for k in ('W1024|gl64|fs|CN|0.01|K1', 'W256|lat8192|qn|BDF2|0.025|k4', 'W256p|gl24|fs|CN|0.01|K1'):
        x = lad.get(k)
        chk(C, f'ladder rung {k} ran valid with ADT anchors', x is not None and all(x['valid']) and all(x['adt_valid']),
            None if x is None else dict(valid=x['valid'], dist_adt=x['dist_adt']))
    z = np.load(d / f"W_{res['mode']}.npz")
    QA = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/q3d3/archive/output/q3run_q3d3/W_q2.npz')
    a = np.load(QA)
    ak = 'W256|lat8192|qn|BDF2|0.025|k4#q|qn|BDF2|0.025|k4'
    Wa, ida = a[ak], [int(x) for x in a['cases|' + ak]]
    nk = 'W256|lat8192|qn|BDF2|0.025|k4'
    Wn, idn = z[nk], [int(x) for x in z['cases|' + nk]]
    rel = [float(np.max(np.abs(Wn[i] - Wa[ida.index(j)])) / float(np.max(np.abs(Wa[ida.index(j)])))) for i, j in enumerate(idn)]
    chk(C, 'G-E2.3 smoke replay W256 lat8192 qn BDF2 (cases 0-7) <= 1e-10', idn == list(range(8)) and max(rel) <= 1e-10, max(rel))
    return C


def smk4(out):
    C = {}
    rc, log = run(['e5_tests.py'])
    chk(C, 'E5 CPU tests (C-E5.3 drift, fixtures, missing-job paths)', rc == 0, log[-600:])
    P1T = '/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/time/archive/output/p1time_time'
    rc, log = run(['e5analyze.py', '--job', f"A100-80G={out / 'e5time_e5smk4'}", '--expect', 'A100-80G=1', '--p1ref', P1T,
                   '--out', str(HERE / 'checks' / 'smoke-smk4-e5.json')])
    chk(C, 'e5analyze runs on the A100 smoke', rc == 0, log[-800:])
    if rc == 0:
        a = json.loads((HERE / 'checks' / 'smoke-smk4-e5.json').read_text())
        chk(C, 'smoke analysis summary', True, {k: a[k] for k in list(a)[:12]})
    return C


def main():
    which, out = sys.argv[1], Path(sys.argv[2])
    fetched = sys.argv[sys.argv.index('--fetched') + 1] if '--fetched' in sys.argv else None
    C = {'smk1': lambda: smk1(out, fetched), 'smk2': lambda: smk2(out), 'smk3': lambda: smk3(out), 'smk4': lambda: smk4(out)}[which]()
    rep = dict(smoke=which, source=str(out), checks=C, ALL=all(v['ok'] for v in C.values()),
               label='smoke acceptance (plumbing and controls at real size); never a verdict')
    (HERE / 'checks' / f'smoke-accept-{which}.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    for k, v in C.items():
        print('OK ' if v['ok'] else 'FAIL', k)
    print('ALL', rep['ALL'])


if __name__ == '__main__':
    main()
