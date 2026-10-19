"""jcp-regimes smoke acceptance (DESIGN §8, A1.9, A2.2, A2.5, A3.1). CPU only; recomputes every predicate from the pulled
raw outputs (never from in-job flags alone).

    JAX_PLATFORMS=cpu python smoke_accept.py runs/smk1      -> checks/smoke-accept.json, exit 1 if a blocking check fails
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import npcheck as NC
import rscore as RS

HERE = Path(__file__).resolve().parent
FRONT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/FRONT-2D.json')
CHECKS = []


def check(name, ok, blocking=True, **detail):
    CHECKS.append(dict(check=name, passed=bool(ok), blocking=blocking, **detail))
    print(f"{'PASS' if ok else ('FAIL' if blocking else 'NOTE')} {name} {json.dumps(detail, default=str)[:300]}", flush=True)


def main(run):
    run = Path(run)
    out = run / 'output'
    logs = ''.join(p.read_text() for p in sorted((run / 'logs').glob('*.out')))
    check('jax_backend=gpu in log', 'jax_backend=gpu' in logs)
    check('job finished', 'ALL-DONE' in logs)
    check('no captured-constants warning', 'large amount of constants' not in logs)
    ug = json.loads((out / 'unit_gates.json').read_text())
    check('unit gates (G-*, R-order, R-peclet)', ug['all_pass'], failed=[g['gate'] for g in ug['gates'] if not g['pass_']])
    # ---------------------------------------------------------------------------------------------- references
    RS.STORE = out / 'refslim_smk1-slim'
    res = {g: json.loads((out / f'refjob_smk1-ref-{g}' / 'result.json').read_text()) for g in ('dev6', 'C2', 'B2', 'Ap')}
    for g, r in res.items():
        check(f'refjob {g} complete on gpu', r.get('complete') and r['job']['backend'] == 'gpu' and r['gates_all_pass'],
              gpu=r['job']['gpu'])
    # R-repro: regenerated dev6 c0 vs the E0 store
    rs = RS._rs()
    f0, m0 = rs.load(2, 'dev6', 0, 'ST')
    js = json.loads((out / 'refjob_smk1-ref-dev6' / '2d' / 'dev6' / 'ST' / 'c000.json').read_text())
    z = np.load(out / 'refjob_smk1-ref-dev6' / '2d' / 'dev6' / 'ST' / 'c000.npz')
    f1 = z[js['final'][2]]
    rel = float(np.max(np.abs(f1 - f0)) / np.max(np.abs(f0)))
    dp = max(abs(a - b) for a, b in zip(js['judge']['p'], m0['judge']['p']))
    check('R-repro dev6 c0 == E0 store', rel <= 1e-12 and js['outcome'] == m0['outcome'] and dp <= 1e-9, rel=rel, dp=dp,
          outcome=js['outcome'])
    # R-upwind (A2.5): exactly order-fail with solver, t0 and floor passing
    pr = {p['case']: p for p in res['C2'].get('probes', [])}
    up = pr.get(9, {}).get('upwind')
    ok_up = bool(up and up['outcome'] == 'order-fail' and up['checks']['solver'] and up['checks']['t0_shared']
                 and up['checks']['floor'])
    check('R-upwind C2 c9 rejected on order', ok_up, outcome=up and up['outcome'], p=up and up['p'])
    # R-calib (A2.2, A3.1): all three probes must 'agree' for the pass; otherwise a label (not blocking)
    verdicts = {}
    for g, c in (('C2', 9), ('B2', 15), ('Ap', 13)):
        pp = {p['case']: p for p in res[g].get('probes', [])}.get(c, {}).get('tight_coarse')
        verdicts[f'{g}|{c}'] = pp and dict(verdict=pp['verdict'], dev=pp['dev'], tau=pp['tau_prod'])
    check('R-calib probes agree (A2.2)', all(v and v['verdict'] == 'agree' for v in verdicts.values()), blocking=False,
          probes=verdicts)
    # pilots: outcome, sigma available, cost (certificate input)
    pil = {}
    for g, c in (('dev6', 0), ('C2', 9), ('B2', 15), ('Ap', 13)):
        m = json.loads((out / f'refjob_smk1-ref-{g}' / '2d' / g / 'ST' / f'c{c:03d}.json').read_text())
        sig = RS.sigma_of(m)
        pil[f'{g}|{c}'] = dict(outcome=m['outcome'], refined=m['refined'], tight_used=m['tight_used'],
                               tight_skipped=m.get('tight_skipped'), refine_skipped=m.get('refine_skipped'),
                               seconds=m['seconds'], eps_ref=m['judge']['eps_ref'], p=m['judge']['p'],
                               sigma=sig, peclet=m['judge']['peclet'])
    check('pilot references ran (outcomes recorded)', all(v['seconds'] for v in pil.values()), blocking=True, pilots=pil)
    check('sigma available for regime pilots', all(pil[k]['sigma'] and pil[k]['sigma']['sigma'] for k in pil if not k.startswith('dev6')))
    # ------------------------------------------------------------------------------------------------------ ROM
    qd = out / 'qacc_smk1-rom'
    rep = json.loads((qd / 'result.json').read_text())
    check('qacc complete on gpu', rep.get('complete') and rep['backend'] == 'gpu', gpu=rep['gpu'])
    check('IC n0 == lattice IC norm, every cohort (A1.6 iv)',
          all(v['equal'] and abs(v['n0'] - v['n0_lattice']) <= 1e-12 * v['n0_lattice'] for v in rep['ic_check'].values()),
          ic=rep['ic_check'])
    # Q-repro against FRONT-2D per_case (dev6)
    front = json.loads(FRONT.read_text())['meshes']['1024']
    entries = {x['config']: x for grp in ('members', 'other_qualified') for x in front[grp]}
    worst_rep = {}
    for setting, arm, key in (('R384', 'LSPG_CNR_1', '1024|R384|LSPG|CNR|1'), ('R128', 'LSPG_BE_1', '1024|R128|LSPG|BE|1')):
        R = np.load(qd / f'R_{setting}_{arm}.npz')
        ids = [str(x) for x in R['case_ids']]
        w = 0.
        for j, cid in enumerate(ids):
            if not cid.startswith('dev6|'):
                continue
            ref = RS.reference('dev6', int(cid.split('|')[1]))
            e = RS.error(NC.decode257(R['W'][j], int(R['Rp'])), ref)
            e_f = entries[key]['errors']['per_case'][cid]['e']
            w = max(w, abs(e - e_f) / e_f)
        worst_rep[key] = w
    check('Q-repro dev6 errors == FRONT-2D per_case (1e-9)', all(v <= 1e-9 for v in worst_rep.values()), worst=worst_rep)
    # controls, recomputed offline
    cu = np.load(qd / 'CTRL_under_resolved.npz')
    d_ur = [NC.fdist(NC.decode257(cu['W'][i], int(cu['Rp'])), NC.decode257(cu['W_succ'][i], int(cu['Rp'])), float(cu['n0'][i]))
            for i in range(len(cu['W']))]
    ids_ur = [str(x) for x in cu['case_ids']]
    check('Q-under-resolved exceeds the Q2 bar on >= 1 case', max(d_ur) > RS.BAR_Q2,
          dist=dict(zip(ids_ur, d_ur)), regime_fired=any(d > RS.BAR_Q2 for d, c in zip(d_ur, ids_ur) if not c.startswith('dev6')))
    for form in ('LSPG', 'GAL'):
        cb = np.load(qd / f'CTRL_budget0_{form}.npz')
        cert = np.asarray(cb['cert'], float)
        frac = float(np.mean(~RS.certified(cert, form)))
        detail = dict(in_job_fraction=frac)
        if form == 'LSPG':          # independent NumPy gamma for the first case
            t = RS.rule_tables(str(cb['rule']), cb['A'].shape[1], cb['kx'], cb['ky'], int(cb['L']))
            g_np = NC.gammas(cb['states'][0], cb['A'], cb['lam'], t, str(cb['scheme']), .005, float(cb['nu'][0]))
            detail['numpy_gamma_case0_max_rel_diff'] = float(np.max(np.abs(g_np - cert[0]) / np.maximum(np.abs(cert[0]), 1e-300)))
            detail['numpy_fraction_case0'] = float(np.mean(~(g_np <= RS.GAMMA)))
        check(f'Q-budget0 {form} pooled non-certified > 1 %', frac > .01, **detail)
    check('Q-wrong-mesh refused', rep['controls']['wrong_mesh']['passed'])
    # offline Q1/Q2 machinery per regime (smoke populations), no in-job/offline disagreement
    rows = rep['rows']
    qsum = {}
    for p in sorted(qd.glob('R_*.npz')):
        R = np.load(p)
        name, arm = p.stem[2:].split('_', 1)
        form = str(R['form'])
        rw = [r_ for r_ in rows if r_['setting'] == name and r_['arm'] == arm]
        setting = next(s for s in rep['config']['settings'] if s['name'] == name)
        rt = (RS.rule_tables(setting['rule'], int(R['Rp']), R['kx'], R['ky'], 1024),
              RS.rule_tables(setting['qp'], int(R['Rp']), R['kx'], R['ky'], 1024))
        for coh in dict.fromkeys(str(c).split('|')[0] for c in R['case_ids']):
            q = RS.qualify(R, rw, dict(form=form, steps=int(R['steps']), qpp=bool(setting.get('qpp'))), rt, subset=[coh])
            qsum[f'{name}|{arm}|{coh}'] = dict(Q1=q['Q1']['pass'], Q2=q['Q2'].get('label'), dis=q['disagreements'][:3])
    check('offline Q1/Q2 agree with in-job values (every arm x cohort)', all(not v['dis'] for v in qsum.values()),
          n=len(qsum), disagreeing=[k for k, v in qsum.items() if v['dis']])
    secs = {}
    for r_ in rows:
        secs.setdefault(r_['cohort'], []).append(r_['seconds'] + (r_.get('seconds_tight') or 0.))
    acc = dict(run=str(run), all_blocking_pass=all(c['passed'] for c in CHECKS if c['blocking']), checks=CHECKS,
               qualify_smoke=qsum, rom_seconds_per_case_arm={k: float(np.median(v)) for k, v in secs.items()},
               qacc_elapsed=rep.get('elapsed_seconds'), pilots=pil)
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / 'smoke-accept.json').write_text(json.dumps(RS_clean(acc), indent=1) + '\n')
    print('ALL-BLOCKING-PASS' if acc['all_blocking_pass'] else 'BLOCKING-FAIL')
    sys.exit(0 if acc['all_blocking_pass'] else 1)


def RS_clean(x):
    if isinstance(x, dict):
        return {str(k): RS_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [RS_clean(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


if __name__ == '__main__':
    main(sys.argv[1])
