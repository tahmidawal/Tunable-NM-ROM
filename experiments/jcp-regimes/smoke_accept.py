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
import rcohorts as RG
import rscore as RS
import verdict as V

HERE = Path(__file__).resolve().parent
BASE_LEVELS = [(2048, 1.25e-3), (4096, 6.25e-4), (8192, 3.125e-4)]      # E0 LEVELS['ST'][2]['base'] (refjob)
CTRL = ['dev6|0', 'dev6|1', 'C2|9', 'C2|11', 'B2|15', 'B2|0']          # A2.5 registered control cases


def n_stages_cnr(dt):
    # refcore.n_stages(dt, 'cnr') re-implemented: 4 BE half steps replace 2 CN steps
    steps = int(round(.25 / dt))
    return 4 + steps - 2


def canon_t0(g, c):
    return RG.initial_2d(256, g, RG.descriptors_2d(g)[c])


def record_ok(cm, g, c):
    return bool(cm['cohort'] == g and cm['case'] == c and cm['kind'] == 'ST'
                and cm['desc'] == RG.desc_2d(g, RG.descriptors_2d(g)[c]))


def lat_ok(a):
    a = np.asarray(a)
    return a.shape == (6, 257, 257) and bool(np.isfinite(a).all())
FRONT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs/FRONT-2D.json')
CHECKS = []


def check(name, ok, blocking=True, **detail):
    CHECKS.append(dict(check=name, passed=bool(ok), blocking=blocking, **detail))
    print(f"{'PASS' if ok else ('FAIL' if blocking else 'NOTE')} {name} {json.dumps(detail, default=str)[:300]}", flush=True)


def main(run):
    run = Path(run)
    out = run / 'output'
    logs = ''.join(p.read_text() for p in sorted((run / 'logs').glob('*.out')))
    errs = ''.join(p.read_text() for p in sorted((run / 'logs').glob('*.err')))
    check('jax_backend=gpu in log', 'jax_backend=gpu' in logs)
    check('job finished', 'ALL-DONE' in logs)
    bad = [w for w in ('large amount of constants', 'Traceback', 'Segmentation', 'Out of memory', 'RESOURCE_EXHAUSTED',
                       'No space left') if w in logs + errs]
    check('no captured-constants / traceback / OOM / disk-full in .out or .err', not bad, found=bad)
    ug = json.loads((out / 'unit_gates.json').read_text())
    check('unit gates in job (file)', ug['all_pass'], failed=[g['gate'] for g in ug['gates'] if not g['pass_']])
    import subprocess
    rr = subprocess.run([sys.executable, str(HERE / 'test_regimes.py')], capture_output=True, text=True, cwd=HERE,
                        env=dict(__import__('os').environ, JAX_PLATFORMS='cpu'))
    check('unit gates re-run locally', rr.returncode == 0 and 'ALL-PASS' in rr.stdout, tail=rr.stdout[-300:])
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
    # R-upwind recomputed from the saved evidence (verdict.judge on the three upwind lattices)
    pr = {p['case']: p for p in res['C2'].get('probes', [])}
    ev = np.load(out / 'refjob_smk1-ref-C2' / '2d' / 'C2' / 'ST' / 'probe_c009.npz')
    ulv = {m_['L']: m_ for m_ in pr[9]['upwind_levels']}
    lv = []
    for L_, dt_ in BASE_LEVELS:                      # independently expected levels and stage counts
        rs_ = ev[f'upwind_res_L{L_}']
        lat_ = ev[f'upwind_lat_L{L_}']
        nexp = n_stages_cnr(dt_)
        lv.append(dict(L=L_, lat=lat_, tau=V.floor_proxy(rs_, nexp),
                       steps_ok=bool(len(rs_) == nexp and V.steps_ok(rs_, nexp) and lat_ok(lat_)), umax=ulv[L_]['umax_running']))
    nu9 = RG.nu_2d('C2', RG.descriptors_2d('C2')[9])
    t0c = canon_t0('C2', 9)
    cm9 = json.loads((out / 'refjob_smk1-ref-C2' / '2d' / 'C2' / 'ST' / 'c009.json').read_text())
    ident_up = bool(record_ok(cm9, 'C2', 9) and np.array_equal(ev['t0'], t0c)
                    and all(np.array_equal(x['lat'][0], t0c) for x in lv))
    jd = V.judge(lv, t0c, nu9)
    ok_up = bool(ident_up and jd['outcome'] == 'order-fail' and jd['checks']['solver'] and jd['checks']['t0_shared'] and jd['checks']['floor'])
    check('R-upwind C2 c9 rejected on order (recomputed)', ok_up, outcome=jd['outcome'], p=jd['p'],
          in_job=pr[9]['upwind']['outcome'])
    # R-calib (A2.2, A3.1): all three probes must 'agree' for the pass; otherwise a label (not blocking)
    verdicts = {}
    for g, c in (('C2', 9), ('B2', 15), ('Ap', 13)):
        base = out / f'refjob_smk1-ref-{g}' / '2d' / g / 'ST'
        ev = np.load(base / f'probe_c{c:03d}.npz')
        cm = json.loads((base / f'c{c:03d}.json').read_text())
        cz = np.load(base / f'c{c:03d}.npz')
        # the production member: the last prod / retry attempt at L = 2048 of the case record (A3.1), identical arrays
        a0 = [x for x in cm['attempts'] if x['L'] == 2048 and x['purpose'] in ('prod', 'retry')][-1]
        t0c = canon_t0(g, c)
        ident = bool(record_ok(cm, g, c) and np.array_equal(ev['prod_lat'], cz[a0['id']])
                     and np.array_equal(ev['prod_res'], cz['res_' + a0['id']]) and np.array_equal(ev['t0'], t0c)
                     and np.array_equal(cz['t0'], t0c) and np.array_equal(ev['prod_lat'][0], t0c)
                     and np.array_equal(ev['tight_lat'][0], t0c))
        nexp = n_stages_cnr(1.25e-3)
        n0 = float(np.linalg.norm(ev['t0'].ravel()))
        dev = max(float(np.linalg.norm((ev['tight_lat'][k] - ev['prod_lat'][k]).ravel())) / n0 for k in range(1, 6))
        tau = V.floor_proxy(ev['prod_res'], nexp)
        valid = bool(ident and len(ev['prod_res']) == nexp and len(ev['tight_res']) == nexp
                     and V.steps_ok(ev['prod_res'], nexp) and V.steps_ok(ev['tight_res'], nexp)
                     and lat_ok(ev['prod_lat']) and lat_ok(ev['tight_lat']) and np.isfinite(dev))
        vd = 'probe invalid' if not valid else ('agree' if dev <= tau else 'disagree')
        pp = {p['case']: p for p in res[g].get('probes', [])}.get(c, {}).get('tight_coarse')
        verdicts[f'{g}|{c}'] = dict(verdict=vd, dev=dev, tau=tau, in_job=pp and pp['verdict'])
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
    # IC identity recomputed: every R record's n0 equals the norm of the generator's lattice IC (A1.6 iv)
    worst_n0 = 0.
    for p in sorted(qd.glob('R_*.npz')):
        Rz = np.load(p)
        for j, cid in enumerate(str(x) for x in Rz['case_ids']):
            coh, c = cid.split('|')
            lat = RS._engines().initial(256, RS.e0_desc(coh)[int(c)]) if coh in RS.E0_COHORTS else \
                RG.initial_2d(256, coh, RG.descriptors_2d(coh)[int(c)])
            worst_n0 = max(worst_n0, abs(float(Rz['n0'][j]) - float(np.linalg.norm(lat))) / float(np.linalg.norm(lat)))
    check('IC n0 == generator lattice IC norm, every record (recomputed)', worst_n0 <= 1e-12, worst=worst_n0,
          in_job=rep['ic_check'])
    # Q-repro against FRONT-2D per_case (dev6)
    front = json.loads(FRONT.read_text())['meshes']['1024']
    entries = {x['config']: x for grp in ('members', 'other_qualified') for x in front[grp]}
    worst_rep = {}
    for setting, arm, key in (('R384', 'LSPG_CNR_1', '1024|R384|LSPG|CNR|1'), ('R128', 'LSPG_BE_1', '1024|R128|LSPG|BE|1')):
        R = np.load(qd / f'R_{setting}_{arm}.npz')
        ids = [str(x) for x in R['case_ids']]
        w = 0.
        assert sum(c.startswith('dev6|') for c in ids) == 6, 'Q-repro needs all six dev6 cases'
        for j, cid in enumerate(ids):
            if not cid.startswith('dev6|'):
                continue
            ref = RS.reference('dev6', int(cid.split('|')[1]))
            e = RS.error(NC.decode257(R['W'][j], int(R['Rp'])), ref, R['n0'][j])
            e_f = entries[key]['errors']['per_case'][cid]['e']
            w = max(w, abs(e - e_f) / e_f)
        worst_rep[key] = w
    check('Q-repro dev6 errors == FRONT-2D per_case (1e-9)', all(v <= 1e-9 for v in worst_rep.values()), worst=worst_rep)
    # controls, recomputed offline
    cu = np.load(qd / 'CTRL_under_resolved.npz')
    check('control populations are the registered six cases', [str(x) for x in cu['case_ids']] == CTRL and
          all([str(x) for x in np.load(qd / f'CTRL_budget0_{f_}.npz')['case_ids']] == CTRL for f_ in ('LSPG', 'GAL'))
          and np.shape(cu['W']) == (6, 6, int(cu['Rp'])) and np.shape(cu['W_succ']) == (6, 6, int(cu['Rp'])))
    d_ur = [NC.fdist(NC.decode257(cu['W'][i], int(cu['Rp'])), NC.decode257(cu['W_succ'][i], int(cu['Rp'])), float(cu['n0'][i]))
            for i in range(len(cu['W']))]
    ids_ur = [str(x) for x in cu['case_ids']]
    check('Q-under-resolved exceeds the Q2 bar on >= 1 case', max(d_ur) > RS.BAR_Q2,
          dist=dict(zip(ids_ur, d_ur)), regime_fired=any(d > RS.BAR_Q2 for d, c in zip(d_ur, ids_ur) if not c.startswith('dev6')))
    for form in ('LSPG', 'GAL'):
        cb = np.load(qd / f'CTRL_budget0_{form}.npz')
        cert = np.asarray(cb['cert'], float)
        # identity of the control (code audit 4): registered cases, their generator viscosities, form / scheme, and the
        # operator / mode arrays equal those of the same setting's production arm (R384 LSPG-CNR)
        Rref = np.load(qd / 'R_R384_LSPG_CNR_1.npz')
        nus = [RS.e0_desc(cid.split('|')[0])[int(cid.split('|')[1])][4] if cid.startswith('dev6') else
               RG.nu_2d(cid.split('|')[0], RG.descriptors_2d(cid.split('|')[0])[int(cid.split('|')[1])]) for cid in CTRL]
        dims_ok = bool(cert.shape == (6, 50) and np.shape(cb['states']) == (6, 51, 384) and len(cb['nu']) == 6
                       and str(cb['rule']) == 'gauss128' and int(cb['L']) == 1024 and np.shape(cb['A']) == (1536, 384)
                       and [str(x) for x in cb['case_ids']] == CTRL and np.array_equal(np.asarray(cb['nu'], float), np.asarray(nus))
                       and str(cb['form']) == form and str(cb['scheme']) == {'LSPG': 'CNR', 'GAL': 'CN'}[form]
                       and all(np.array_equal(cb[k], Rref[k]) for k in ('A', 'lam', 'kx', 'ky')))
        check(f'Q-budget0 {form} evidence complete (6 cases x 50 steps, R384, gauss128, 1024^2)', dims_ok,
              cert=cert.shape, states=np.shape(cb['states']))
        frac = float(np.mean(~RS.certified(cert, form)))
        detail = dict(in_job_fraction=frac)
        if form == 'LSPG':          # independent NumPy gamma for the first case
            t = RS.rule_tables(str(cb['rule']), cb['A'].shape[1], cb['kx'], cb['ky'], int(cb['L']))
            g_all = np.array([NC.gammas(cb['states'][i], cb['A'], cb['lam'], t, str(cb['scheme']), .005, float(cb['nu'][i]))
                              for i in range(len(cb['nu']))])
            detail['numpy_gamma_case0_max_rel_diff'] = float(np.max(np.abs(g_all - cert) / np.maximum(np.abs(cert), 1e-300)))
            detail['numpy_fraction_case0'] = float(np.mean(~(g_all <= RS.GAMMA)))     # pooled over all six cases
        ok_b = frac > .01 and dims_ok
        if form == 'LSPG':          # the independent NumPy gamma must agree with the in-job certificate (gated)
            ok_b = ok_b and detail['numpy_gamma_case0_max_rel_diff'] <= 1e-6 and detail['numpy_fraction_case0'] > .01
        if form == 'LSPG':
            check('Q-budget0 LSPG pooled non-certified > 1 % (independent NumPy gamma)', ok_b, **detail)
        else:                       # GAL is the registered diagnostic row; no independent NumPy eta exists (in-job record)
            check('Q-budget0 GAL pooled non-certified > 1 % (in-job eta; diagnostic row)', ok_b, blocking=False, **detail)
    check('Q-wrong-mesh refused in job', rep['controls']['wrong_mesh']['passed'], raised=rep['controls']['wrong_mesh']['raised'])
    import os
    import subprocess
    import guard as G
    wm = G.check_local(HERE / 'checks' / 'wrongmesh-local.json', HERE)
    if wm.exists():
        wm.unlink()                                 # never accept a stale result
    rr = subprocess.run(['bash', '/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/jaxslot.sh',
                         '1800', 'jaxrun', sys.executable, str(HERE / 'ctrl_wrongmesh.py'), str(wm)], cwd=HERE,
                        capture_output=True, text=True, env=dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
                                                                 JAX_DEFAULT_MATMUL_PRECISION='highest'))
    wmr = json.loads(wm.read_text()) if wm.exists() else {}
    check('Q-wrong-mesh refused when re-executed locally (CPU, same code path)', rr.returncode == 0 and wmr.get('passed'),
          result=wmr, rc=rr.returncode)
    fv = json.loads((out / 'fverify_smk1-fverify' / 'fverify.json').read_text())
    import hashlib
    want_fv = json.loads((HERE / 'configs' / 'smk1-fverify.json').read_text())['arms']
    bound = sorted(fv['arms']) == sorted(want_fv) and all(
        fv['arms'][k]['R_file_sha256'] == hashlib.sha256((qd / f'R_{k}.npz').read_bytes()).hexdigest()
        and fv['arms'][k]['f257_sha256'] == str(np.load(qd / f'R_{k}.npz')['f257_sha256']) for k in want_fv)
    check('full-field independent decode (on-cluster fverify), bound to the pulled R files and config',
          fv['all_pass'] and bound and fv['config_sha256'] == hashlib.sha256((HERE / 'configs' / 'smk1-fverify.json').read_bytes()).hexdigest(),
          arms={k: dict(passed=v['passed'], max_rel=v['max_rel']) for k, v in fv['arms'].items()})
    # offline Q1/Q2 machinery per regime (smoke populations), no in-job/offline disagreement
    rows = rep['rows']
    qsum = {}
    want_arms = sorted(f"R_{s_['name']}_{a_['form']}_{a_['scheme']}_{a_['factor']:g}.npz" for s_ in rep['config']['settings']
                       for a_ in s_['arms'])
    have_arms = sorted(p.name for p in qd.glob('R_*.npz'))
    check('every configured arm produced its R file', have_arms == want_arms, missing=sorted(set(want_arms) - set(have_arms)))
    for p in sorted(qd.glob('R_*.npz')):
        R = np.load(p)
        name, arm = p.stem[2:].split('_', 1)
        form = str(R['form'])
        rw = [r_ for r_ in rows if r_['setting'] == name and r_['arm'] == arm]
        setting = next(s for s in rep['config']['settings'] if s['name'] == name)
        rt = (RS.rule_tables(setting['rule'], int(R['Rp']), R['kx'], R['ky'], 1024),
              RS.rule_tables(setting['qp'], int(R['Rp']), R['kx'], R['ky'], 1024))
        arm_cfg = next(a_ for a_ in setting['arms'] if f"{a_['form']}_{a_['scheme']}_{a_['factor']:g}" == arm)
        ex = RS.arm_expect(rep['config'], arm_cfg, setting)
        for coh in dict.fromkeys(c.split('|')[0] for c in ex['cases']):
            q = RS.qualify(R, rw, ex, rt, subset=[coh])
            qsum[f'{name}|{arm}|{coh}'] = dict(Q1=q['Q1']['pass'], Q2=q['Q2'].get('label'), dis=q['disagreements'][:3])
    check('offline Q1/Q2 agree with in-job values (every arm x cohort)', all(not v['dis'] for v in qsum.values()),
          n=len(qsum), disagreeing=[k for k, v in qsum.items() if v['dis']])
    secs = {}
    for r_ in rows:
        secs.setdefault(r_['cohort'], []).append(r_['seconds'] + (r_.get('seconds_tight') or 0.))
    # resource inputs (resources.py): per-step seconds from the STEP timestamps, probe seconds, calibration verdict
    import datetime as _dt
    stamps = [(ln.split()[1], ln.split()[2], _dt.datetime.fromisoformat(ln.split()[3].replace('Z', '+00:00')))
              for ln in logs.splitlines() if ln.startswith('STEP ')]
    step_s = {}
    for i, (drv, cf, t) in enumerate(stamps):
        if i + 1 < len(stamps):
            step_s[f'{drv}:{cf}'] = (stamps[i + 1][2] - t).total_seconds()
    probe_s = {f'{g}|{p_["case"]}': p_['tight_coarse']['seconds'] for g in ('C2', 'B2', 'Ap')
               for p_ in res[g].get('probes', []) if p_.get('tight_coarse')}
    acc = dict(run=str(run), step_seconds=step_s, probe_seconds=probe_s,
               calib_agree=all(v['verdict'] == 'agree' for v in verdicts.values()), all_blocking_pass=all(c['passed'] for c in CHECKS if c['blocking']), checks=CHECKS,
               qualify_smoke=qsum, rom_seconds_per_case_arm={k: float(np.median(v)) for k, v in secs.items()},
               qacc_elapsed=rep.get('elapsed_seconds'), pilots=pil)
    (HERE / 'checks').mkdir(exist_ok=True)
    import guard
    guard.check_local(HERE / 'checks' / 'smoke-accept.json', HERE)
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
