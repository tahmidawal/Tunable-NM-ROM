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
PILOT_BANK = None          # smk3: the pilot bank.pkl fetched from the cluster archive into scratch (--pilot-bank)
ENV = dict(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true', JAX_DEFAULT_MATMUL_PRECISION='highest')


def run(args):
    import os
    r = subprocess.run([PY] + args, cwd=HERE, env=dict(os.environ, **ENV), capture_output=True, text=True)
    return r.returncode, r.stdout[-4000:] + r.stderr[-4000:]


def chk(C, name, ok, ev=None):
    C[name] = dict(ok=bool(ok), evidence=ev)


def resource_contract(C, out, att):
    """A6.1 (code audit 5 #3): the collected JOBSPEC equals the registered jobs/<att>.json at the job's commit; every driver
    step's inventory equals certify.INV regenerated from the committed config at that commit (keys compared exactly);
    every shell step lists its registered args / env / projection; the reservation equals the spec's rounded time limit
    x GPUs and is <= 2 GPU-h."""
    import certify as CF
    root = out.parent
    ok, ev = True, {}
    try:
        commit = (root / 'COMMIT.txt').read_text().strip()
        git = lambda path: subprocess.check_output(['git', 'show', f'{commit}:experiments/jcp-robust/{path}'], cwd=HERE)
        spec = json.loads((root / 'JOBSPEC.json').read_text())
        reg = json.loads(git(f'jobs/{att}.json'))
        ok &= spec == reg
        I = json.loads((root / f'inventory-{att}.json').read_text())
        R = json.loads((root / 'RESERVATION.json').read_text())
        lim = math.ceil(float(reg['hours']) * 4) / 4 * int(reg.get('gpus', 1))
        ok &= bool(I.get('smoke')) and abs(R['time_limit_hours'] - lim) < 1e-9 and lim <= 2.0 + 1e-9
        ident = dict(attempt=R.get('attempt') == att, commit=R.get('commit') == commit, gpu=R.get('gpu') == reg['gpu'],
                     gpus=R.get('gpus') == int(reg.get('gpus', 1)), groups=R.get('groups') == reg['groups'],
                     time_limit=abs(float(R.get('time_limit', -1)) - math.ceil(float(reg['hours']) * 4) / 4) < 1e-9)
        ev['reservation_identity'] = ident                       # code audit 6 #1
        ok &= all(ident.values())
        for g in reg['groups']:
            for st in g:
                if 'driver' in st:
                    keys = CF.INV[st['driver']](json.loads(git(st['config'])))
                    want = [list(map(str, k)) for k in sorted(set(map(tuple, keys)), key=str)]
                    got = I['inventory'].get(st['config'], {})
                    same = got.get('keys') == want and got.get('n') == len(keys)
                    ev[st['config']] = dict(n=len(keys), same=same)
                    ok &= same
                else:
                    got = I['inventory'].get(st['sh'], {})
                    same = (got.get('args') == st.get('args') and got.get('env') == st.get('env')
                            and got.get('projection_hours') == st.get('projection_hours'))
                    ev[st['sh']] = dict(same=same)
                    ok &= same
    except Exception as e_:                                  # missing artefacts -> the contract fails (never passes)
        ok, ev = False, dict(error=repr(e_))
    chk(C, 'smoke resource contract (registered spec, regenerated inventories, reservation <= 2 GPU-h)', ok, ev)


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
        chk(C, '2d registered arm inventory (LSPG_BE_1, LSPG_CNR_1)', sorted(g['arms']) == ['LSPG_BE_1', 'LSPG_CNR_1'], sorted(g['arms']))
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
        chk(C, '3d job complete on gpu', a.get('backend') == 'gpu' and (d3 / 'COMPLETE').exists(), a.get('backend'))
        chk(C, '3d registered configuration inventory (E1-3D-a)', sorted(a['configs']) == ['E1-3D-a'], sorted(a['configs']))
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
    resource_contract(C, out, 'smk1b')
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
    import e2bank
    e2bank.OUTPUT_ROOT = str(out)                      # code audit 3 #11: trained-bank files from the collected output
    for tag in ('smk2t', 'smk2u'):
        dd = out / f'r2acc_{tag}'
        r_ = json.loads((dd / 'result.json').read_text())
        chk(C, f'{tag} complete on gpu, bank runtime-hashed', r_.get('complete') and r_['backend'] == 'gpu'
            and (r_.get('bank') or {}).get('runtime_hashed') is True, dict(bank=r_.get('bank'), ck=r_['checkpoint_sha256']))
        chk(C, f'{tag} ran on the trained (not the incumbent) bank', not r_['checkpoint_sha256'].startswith('18f0266a'))
        g = r_['config']['groups'][0]
        e2bank.use(r_['config']['bank'], (r_['checkpoint_sha256'], r_['rotation_sha256']))
        for arm in g['arms']:
            ak = f"{arm['form']}_{arm['scheme']}_{float(arm['factor']):g}"
            av = EA.analyse_arm(dd, r_, g, ak, int(g['Rp']))
            chk(C, f'{tag} {ak} analysis integrity', av['integrity']['ok'], av['integrity'])
            chk(C, f'{tag} {ak} own-family labels computed', bool(av['rules']) and
                all(x['own_family_qualified'] is not None for x in av['rules'].values()),
                {l: (x['status'], x['own_family_qualified']) for l, x in av['rules'].items()})
    rc, log = run(['select2d.py', str(out / 'r2acc_smk2t'), 'smk', '--output-root', str(out)])
    sp_ = HERE / 'checks' / 'select2d-smk-smoke.json'
    sel = json.loads(sp_.read_text())['selected']['128'] if (rc == 0 and sp_.exists()) else None
    oc = None if sel is None else sel['outcome']
    chk(C, 'select2d (smoke contract) runs on the trained bank and returns a defined outcome',
        oc is not None and (oc in ('selected', 'none within ladder', 'diagnostic pick (target unverified)') or oc.startswith('unavailable')),
        dict(outcome=oc, rule=None if sel is None else sel['rule'], log=log[-400:]))
    # selection -> evaluation chain (code audit 4 #8): the 1024^2 'sel' row is the candidate the selection returned;
    # the 256^2 row exists only if smk2u evaluated that same rule; otherwise H-acc-N must report 'verdict not reached'
    import e2analyze as E2A
    pick = None if sel is None else sel['rule']
    r2u = json.loads((out / 'r2acc_smk2u' / 'result.json').read_text())
    u_rule = r2u['config']['groups'][0]['rules'][0]['rule']
    C2 = {1024: E2A.records(out / 'r2acc_smk2t', lambda r, labels: [(pick, 'sel')] if pick in labels else [])}
    C2[256] = E2A.records(out / 'r2acc_smk2u', lambda r, labels: [('sel', 'sel')] if u_rule == pick else [])
    h = E2A.h_acc_n(C2, 128, 'sel', 'BE')
    want = ('two-mesh evidence, registered three-mesh verdict not reached' if (pick and u_rule == pick)
            else 'verdict not reached')
    chk(C, 'selection result drives the H-acc-N rows (selected rule evaluated -> two-mesh outcome; else verdict not reached)',
        h.get('procedure_outcome', h.get('outcome')) == want, dict(pick=pick, evaluated=u_rule, outcome=h.get('procedure_outcome', h.get('outcome'))))
    resource_contract(C, out, 'smk2')
    return C


def smk3(out):
    C = {}
    p = out / 'train3d_pilot'
    chk(C, 'train3d pilot SEED-COMPLETE', (p / 'SEED-COMPLETE').exists())
    rec = json.loads((p / 'run_record.json').read_text()) if (p / 'run_record.json').exists() else None
    chk(C, 'train3d pilot run record present', rec is not None)
    jid = sorted(x.stem for x in (out.parent / 'logs').glob('*.out'))
    rss = None
    if jid:
        r = subprocess.run(['timeout', '60', 'ssh', '-o', 'ConnectTimeout=20', 'tufts-login',
                            f'sacct -j {jid[-1]} -n -P -o MaxRSS'], capture_output=True, text=True)
        vals = [x for x in r.stdout.split() if x.strip()]
        conv = lambda v: float(v[:-1]) * {'K': 2 ** 10, 'M': 2 ** 20, 'G': 2 ** 30}[v[-1]] if v[-1] in 'KMG' else float(v)
        rss = max((conv(v) for v in vals), default=None)
    req = 420 * 2 ** 30
    chk(C, 'memory gate: device peak <= 120 GB (pilot)', rec is not None and math.isfinite(E_num(rec.get('device_peak_GiB')))
        and E_num(rec.get('device_peak_GiB')) * 2 ** 30 <= 120e9, None if rec is None else rec.get('device_peak_GiB'))
    chk(C, 'memory gate: host MaxRSS x 1.25 <= requested 420G (sacct)', rss is not None and rss * 1.25 <= req,
        dict(max_rss_bytes=rss, requested=req))
    d = out / 'e2run3d_smk3q'
    res = json.loads((d / 'result.json').read_text())
    chk(C, 'e2run3d smoke complete on gpu', res.get('complete') and res['backend'] == 'gpu')
    chk(C, 'heaviest cell (W1024 gl64 + gl80) under the 120 GB device bar', not res.get('memory_rejected') and
        E_num(res.get('peak_bytes_job')) <= 120e9, dict(peak=res.get('peak_bytes_job'), rejected=res.get('memory_rejected')))
    lad = {}
    for blk in res['blocks']:
        if blk['kind'] == 'ladder':
            lad.update(blk['result'])
    for k, n in (('W1024|gl64|fs|CN|0.01|K1', 4), ('W1024|gl80|fs|CN|0.01|K1', 4), ('W256|lat8192|qn|BDF2|0.025|k4', 8),
                 ('W256|lat16384|qn|BDF2|0.025|k4', 8), ('W256p|gl24|fs|CN|0.01|K1', 8), ('W256p|gl32|fs|CN|0.01|K1', 8)):
        x = lad.get(k)
        chk(C, f'ladder rung {k} ran valid with ADT anchors on its {n} cases', x is not None and list(x['cases']) == list(range(n))
            and list(x['tighter_cases']) == list(range(n)) and len(x['adt_tighter_valid']) == n and all(x['adt_tighter_valid'])
            and len(x['valid']) == n and
            all(x['valid']) and len(x['adt_valid']) == n and all(x['adt_valid']),
            None if x is None else dict(valid=x['valid'], dist_adt=x['dist_adt']))
    x = lad.get('W256p|gl24|fs|CN|0.01|K1')                  # the rung-selection predicate on the pilot bank (own-family)
    xn = lad.get('W256p|gl32|fs|CN|0.01|K1')
    pop_ok = bool(x is not None and xn is not None and list(x['cases']) == list(range(8)) and list(x['tighter_cases']) == list(range(8))
                  and len(x['adt_tighter_valid']) == 8 and len(x['adt_tight_tighter']) == 8 and len(x['dist_next']) == 8
                  and list(xn['cases']) == list(range(8)))
    chk(C, 'pilot rung pair carries the registered populations (cases 0-7, tighter 0-7)', pop_ok)
    z = np.load(d / f"W_{res['mode']}.npz")
    pb = PILOT_BANK                                          # code audit 6 #6: pilot arrays decoded on the pilot bank
    rec_ok, d_rec = False, None
    if x is not None and pop_ok and pb:
        import hashlib
        import offline  # noqa: F401
        import c3common as C3
        import score3 as SC
        large = {l.split()[1]: l.split()[0] for l in (out.parent / 'LARGE.sha256').read_text().splitlines()} \
            if (out.parent / 'LARGE.sha256').exists() else {}
        want = large.get('output/train3d_pilot/bank.pkl')
        if want and hashlib.sha256(Path(pb).read_bytes()).hexdigest() == want:
            C3.BANKS['W1024pilot'] = (Path(pb), want)
            SC.BANKS = {'W256p': ('W1024pilot', 256)}
            SC._G.clear()
            G = np.asarray(SC.glat('W256p'))
            tab = C3.C.table(923801, 64)
            n0 = {j: float(np.linalg.norm(np.asarray(C3.C.initial_interior(65, tab, j)))) for j in range(8)}
            sp = 'fs|CN|0.01|K1'
            keys = dict(p=f'W256p|gl24|{sp}', s=f'W256p|gl32|{sp}', t=f'W256p|gl24|ad|LSPG|CN|0.01|tight',
                        tt=f'W256p|gl24|ad|LSPG|CN|0.01|tighter')
            arr = {k: z[v] for k, v in keys.items() if v in z.files}
            maps_ok = len(arr) == 4 and all([int(c) for c in z['cases|' + keys[k]]] == list(range(8)) for k in keys) and \
                all(arr[k].shape == (8, 6, 256) for k in arr) and all(np.isfinite(arr[k]).all() for k in arr)   # audit 7 #5
            if maps_ok:
                def dd(A_, B_):
                    v = [[float(np.linalg.norm((A_[i][k] - B_[i][k]) @ G)) / n0[i] for k in range(1, 6)] for i in range(8)]
                    return [max(r_) if all(math.isfinite(x_) for x_ in r_) else math.inf for r_ in v]
                d_rec = dict(tight=dd(arr['p'], arr['t']), succ=dd(arr['p'], arr['s']), tt=dd(arr['t'], arr['tt']))
                cr = [r_ for r_ in res['rows'] if r_.get('tag') == 'W256p|gl24' and r_.get('spec') == sp and 'gam_cert' in r_]
                rec_ok = sorted(int(r_['case']) for r_ in cr) == list(range(8)) and all(len(r_['gam_cert']) == 25 for r_ in cr)
                gam = [E_num(v) for r_ in cr for v in r_['gam_cert']]
                d_rec['nonst'] = (sum(1 for v in gam if not (math.isfinite(v) and v <= 1e-3)) / len(gam)) if (rec_ok and gam) else None
    chk(C, 'pilot rung evidence recomputed from its saved arrays on the pilot bank (mappings, shapes, certificate rows)',
        bool(rec_ok and d_rec), None if d_rec is None else {k: (max(v) if isinstance(v, list) else v) for k, v in d_rec.items()})
    if x is not None and pop_ok and rec_ok and d_rec:
        import e1stats as E
        qual, own, status = E.draw_qualified(
            dict(valid=x['valid'], valid_tight=x['adt_valid'], valid_tighter=x['adt_tighter_valid'],
                 valid_succ=lad.get('W256p|gl32|fs|CN|0.01|K1', {}).get('valid')),
            dict(tight=d_rec['tight'], tt=d_rec['tt'], succ=d_rec['succ'],
                 rho_succ=[max(E.num(v) for v in rr) for rr in x['rho']], target=None, rho_tgt=None,
                 nonst_frac=d_rec['nonst']), None, 8, 8)
        chk(C, 'pilot-bank rung-selection predicate evaluates (own-family label defined)', own is not None, dict(own=own, status=status))
    resource_contract(C, out, 'smk3')
    z = np.load(d / f"W_{res['mode']}.npz")
    QD = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/q3d3/archive')
    QA = QD / 'output' / 'q3run_q3d3' / 'W_q2.npz'
    import hashlib
    pin = next((l.split()[0] for l in (QD / 'OUTPUTS.sha256').read_text().splitlines() if l.split()[1] == 'output/q3run_q3d3/W_q2.npz'), None)
    ok_pin = hashlib.sha256(QA.read_bytes()).hexdigest() == pin
    a = np.load(QA)
    ak = 'W256|lat8192|qn|BDF2|0.025|k4#q|qn|BDF2|0.025|k4'
    Wa, ida = a[ak], [int(x) for x in a['cases|' + ak]]
    nk = 'W256|lat8192|qn|BDF2|0.025|k4'
    Wn, idn = z[nk], [int(x) for x in z['cases|' + nk]]
    fin = bool(np.isfinite(Wn).all() and np.isfinite(Wa).all())
    rel = [float(np.max(np.abs(Wn[i] - Wa[ida.index(j)])) / float(np.max(np.abs(Wa[ida.index(j)])))) for i, j in enumerate(idn)] if fin else [math.inf]
    chk(C, 'G-E2.3 smoke replay W256 lat8192 qn BDF2 (cases 0-7) <= 1e-10, archive hash pinned, finite',
        ok_pin and fin and idn == list(range(8)) and max(rel) <= 1e-10, dict(worst=max(rel), pin_ok=ok_pin))
    return C


def smk5(out):
    """H200 E5 panel (DESIGN 6 / A5.2), run as its own smoke after smk1 moved to an A100 (queue; A7.1)."""
    C = {}
    P1T = '/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/time/archive/output/p1time_time'
    rc, log = run(['e5analyze.py', '--smoke', '--gpu-type', 'H200', '--job', f"H200={out / 'e5time_e5smk1'}", '--p1ref', P1T,
                   '--out', str(HERE / 'checks' / 'smoke-smk5-e5.json')])
    chk(C, 'H200 E5 single-job smoke: ALL (job ok, C-E5.1, fixtures, G-E5.1, P-cross gates)', rc == 0, log[-1200:])
    resource_contract(C, out, 'smk5')
    return C


def smk4(out):
    C = {}
    rc, log = run(['e5_tests.py'])
    chk(C, 'E5 CPU tests (C-E5.3 drift, fixtures, missing-job paths)', rc == 0, log[-600:])
    P1T = '/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/time/archive/output/p1time_time'
    rc, log = run(['e5analyze.py', '--smoke', '--gpu-type', 'A100-80G', '--job', f"A100-80G={out / 'e5time_e5smk4'}", '--p1ref', P1T,
                   '--out', str(HERE / 'checks' / 'smoke-smk4-e5.json')])
    chk(C, 'E5 single-job smoke analysis: ALL (job ok, C-E5.1, q1/q2/q3/null fixtures, G-E5.1, P-cross gates)', rc == 0, log[-1200:])
    resource_contract(C, out, 'smk4')
    return C


def main():
    which, out = sys.argv[1], Path(sys.argv[2])
    fetched = sys.argv[sys.argv.index('--fetched') + 1] if '--fetched' in sys.argv else None
    global PILOT_BANK
    PILOT_BANK = sys.argv[sys.argv.index('--pilot-bank') + 1] if '--pilot-bank' in sys.argv else None
    C = {'smk1b': lambda: smk1(out, fetched), 'smk1': lambda: smk1(out, fetched), 'smk2': lambda: smk2(out), 'smk3': lambda: smk3(out), 'smk4': lambda: smk4(out), 'smk5': lambda: smk5(out)}[which]()
    rep = dict(smoke=which, source=str(out), checks=C, ALL=all(v['ok'] for v in C.values()),
               label='smoke acceptance (plumbing and controls at real size); never a verdict')
    (HERE / 'checks' / f'smoke-accept-{which}.json').write_text(json.dumps(rep, indent=1, default=str) + '\n')
    for k, v in C.items():
        print('OK ' if v['ok'] else 'FAIL', k)
    print('ALL', rep['ALL'])
    sys.exit(0 if rep['ALL'] else 1)


if __name__ == '__main__':
    main()
