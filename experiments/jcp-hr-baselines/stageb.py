"""jcp-hr-baselines phase-II Stage-B + timing job for one cell (DESIGN A1.6, A2.6, A5.3, A6.6, A24.4, A26.5).

Inputs: the frozen Stage-B membership (checks/FROZEN-STAGEB-<cell>.json, sha256-checked) and the tuning job's
artifacts (basis, HR data; sha256-checked against the frozen file). Steps:
  1. G-ours again (A24.5): every admitted X replayed on the 6 timing cases (exported errors to 1e-10 relative);
  2. every Stage-B member on the full verdict cohort (2D dev6 + val32, 3D 923801 0-63): prod runs with its solver,
     tight native anchors on every case (shared per discrete problem), tighter on the cohort's first two cases, Q1,
     E0 scores; on the screening cases the outputs are compared with the tuning job's (replay of the construction);
  3. A-B-A timing (A6.6): every admitted X against every member Q1-qualified here, on the 6 timing cases, 3 triples per
     case with drift retries, each timed output checked against its scored output (<= 1e-12).
Picks and labels are computed offline from this job's data (A26.5). Writes <out>/stageb.json and <out>/arrays/.

Usage: python stageb.py <config.json> <out_dir>
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'e0'))
import hrcore as H  # noqa: E402
import rom as RM  # noqa: E402
import build as BD  # noqa: E402
import pipeline as PL  # noqa: E402
import score as SC  # noqa: E402
import stats as ST  # noqa: E402

T0 = time.perf_counter()
TIMED_TOL = 1e-12
REPLAY_TOL = 1e-8          # A32.7: max |W_stageB - W_tune| / max(1, max |W_tune|) on the screening cases (same GPU type)
DEFER_FRAC = 0.9


def log(*a):
    print(f'[{time.perf_counter() - T0:8.1f}s]', *a, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def disc_finite(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float('nan')
    return float(np.max(np.abs(a - b)))


def timed_triples(fA, fB, argsA, argsB, checkA, checkB, reps=3, burn_s=0.1, max_retry=3, rec=None, save=None):
    """A-B-A triples (A3.9, A6.6), as smoke.timed_triples: every attempt kept (appended to `rec` and checkpointed
    through `save` as it completes, A33.3); drift |tA1/tA2 - 1| > 0.25 retried."""
    rec = [] if rec is None else rec
    for rep in range(reps):
        for attempt in range(max_retry + 1):
            ts, ds = [], []
            for f, a, chk in ((fA, argsA, checkA), (fB, argsB, checkB), (fA, argsA, checkA)):
                PL.burn(burn_s)
                t0 = time.perf_counter()
                o = jax.block_until_ready(f(*a))
                ts.append(time.perf_counter() - t0)
                ds.append(float(chk(o)))
            ok = abs(ts[0] / ts[2] - 1.) <= 0.25
            rec.append(dict(rep=rep, attempt=attempt, tA1=ts[0], tB=ts[1], tA2=ts[2], valid=bool(ok),
                            dA1=ds[0], dB=ds[1], dA2=ds[2]))
            if save is not None:
                save()
            if ok:
                break
    return rec


def parse(cid):
    kind, r, hp, scheme, dt, solver, gen = cid.split('|')
    return kind, int(r), hp, scheme, float(dt), solver


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    (out / 'arrays').mkdir(parents=True, exist_ok=True)
    if not cfg.get('allow_cpu'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    log(f'jax_backend={jax.default_backend()} devices={jax.devices()}')
    res = dict(config=cfg, env=dict(backend=jax.default_backend(), devices=[str(x) for x in jax.devices()],
                                    job=os.environ.get('SLURM_JOB_ID'), commit=cfg.get('commit')),
               phases={}, ours={}, members={}, timing={}, inventory={})

    last_save = [0.]
    jl = open(out / 'evidence.jsonl', 'a', buffering=1)          # A42: append-only per-unit evidence, flushed per line

    def save(force=False):
        # A42: the full JSON (tens of MB) is rewritten at most every 60 s and at every unit boundary (force=True);
        # every completed case, anchor and timing attempt is appended to evidence.jsonl at once, so a kill loses none
        now = time.perf_counter()
        if not force and now - last_save[0] < 60.:
            return
        PL.dump(out / 'stageb.json.tmp', res)
        os.replace(out / 'stageb.json.tmp', out / 'stageb.json')
        last_save[0] = now

    def ev(kind, **kw):
        jl.write(json.dumps(dict(kind=kind, t=time.time(), **kw), default=float) + '\n')

    def phase(name, fn):
        t0 = time.perf_counter()
        res['active_phase'] = dict(name=name, started_unix=time.time())        # A35.4: bounds unfinished work
        save(force=True)
        try:
            v = fn()
            res['phases'][name] = dict(status='ok', seconds=time.perf_counter() - t0, device_peak=PL.device_peak())
        except Exception as e:  # noqa: BLE001
            res['phases'][name] = dict(status='error', error=f'{type(e).__name__}: {e}',
                                       trace=traceback.format_exc()[-3000:], seconds=time.perf_counter() - t0)
            log(f'ERR  {name}: {type(e).__name__}: {e}')
            v = None
        res['active_phase'] = None
        save(force=True)
        return v

    # ---- inputs, integrity ------------------------------------------------------------------------------------------
    fz_path = HERE / cfg['frozen']
    assert sha(fz_path) == cfg['frozen_sha256'], 'frozen Stage-B file differs from the audited one'
    FZ = json.loads(fz_path.read_text())
    tdir = Path(cfg['tune_dir'])
    for a, h in FZ['artifacts'].items():
        assert sha(tdir / a) == h, f'artifact {a} differs from the frozen hash'
    tune = json.loads((tdir / 'tune.json').read_text())
    assert sha(tdir / 'tune.json') == FZ['tune_json_sha256']
    d, L = tune['config']['d'], tune['config']['L']
    assert (d, L) == (cfg['d'], cfg['L'])
    budget_s = 3600. * cfg.get('budget_hours', 1e9)
    zf = np.load(tdir / 'artifacts' / 'Phi.npz')
    allr = [int(x) for x in zf['ranks']]
    cell = PL.Cell(d, L, log=log)
    cell.set_basis(jnp.asarray(zf['Phi']), {r: float(t) for r, t in zip(allr, zf['trust'])})
    cell.constructible_r = {int(x) for x in zf['constructible']}
    del zf
    # cohorts: case identities '<cohort>|<j>'; the timing cases are the first six of the selection cohort
    if d == 2:
        cohorts = [('dev6', H.Family(2, 'dev6')), ('val32', H.Family(2, 'val32'))]
    else:
        cohorts = [('val64', H.Family(3, 'val64', count=64))]
    if cfg.get('dry'):
        cohorts = [(c, f) for c, f in cohorts[:1]]
    fam = {c: f for c, f in cohorts}
    cases = [f'{c}|{j}' for c, f in cohorts for j in (range(f.n) if not cfg.get('dry') else range(3))]
    timing = [f'{cohorts[0][0]}|{j}' for j in (ST.TIMING_CASES if not cfg.get('dry') else (0, 1, 2))]
    refs = {c: SC.Refs(d, c, f, list(range(f.n)) if not cfg.get('dry') else [0, 1, 2]) for c, f in cohorts}
    n0 = {k: refs[k.split('|')[0]].n0(int(k.split('|')[1])) for k in cases}

    def ic(k):
        c, j = k.split('|')
        return fam[c].ic(L, int(j)), fam[c].nu(int(j))

    def score(k, F):
        c, j = k.split('|')
        return refs[c].score_safe(F, int(j))

    # ---- 1. G-ours again ---------------------------------------------------------------------------------------------
    admitted = [k for k, x in FZ['X'].items() if 'front' in x]
    OX, XS = {}, {}
    res['eligible_cases'] = [f'{c}|{j}' for c, f in cohorts for j in (range(f.n) if not cfg.get('dry') else range(3))
                             if bool(refs[c].st[j]['eligible'])]          # A32.1: fixed mask from the store
    res['eps_ref'] = {k: float(refs[k.split('|')[0]].st[int(k.split('|')[1])]['eps_ref']) for k in cases}   # A35.1

    def make_x(key, ox_p1=None):
        import ours as OU
        if d == 2:
            ox = ox_p1 if ox_p1 is not None else OU.OursP1(L, [key], log=log)
            return ox, ox.fn(key)
        o_ = OU.OursS3(*key.split('|')[:2], log=log)
        return o_, o_.fn(key)

    def ours_pass():
        import ours as OU
        if d == 2:
            exp = json.loads((HERE / 'inputs' / 'p1_analysis.json').read_text())['configurations']
            ref_e = {k: {t: exp[k]['e0_per_case'][t] for t in timing} for k in admitted}
            ox = None                                  # A36.3: one comparator object live at a time
        else:
            exp = json.loads((HERE / 'inputs' / 'speed3d_batch1_e0.json').read_text())['configs']
            ref_e = {k: {t: exp[k]['e0'][t.split('|')[1]] for t in timing} for k in admitted}
        for key in admitted:
            rec = dict(status='construction_failed', cases={}, batch=FZ['X'][key].get('batch'),
                       qualification=FZ['X'][key].get('qualification'))
            res['ours'][key] = rec
            def replay(key=key, rec=rec):              # A37.1: the comparator lives only inside this call
                o_, fx = make_x(key)
                try:
                    for t in timing:
                        c, j = t.split('|')
                        u, v = o_.u0(fam[c], int(j)), jnp.float64(fam[c].nu(int(j)))
                        Fw = o_.lattice(jax.block_until_ready(fx(u, v)))
                        e = score(t, Fw)['e']
                        rel = abs(e - ref_e[key][t]) / abs(ref_e[key][t]) if np.isfinite(e) else float('inf')
                        np.savez(out / 'arrays' / f'X~{key.replace("|", "~")}~{t.replace("|", "~")}.npz', F=Fw)
                        XS.setdefault(key, {})[t] = Fw                             # scored output (A32.7 binding)
                        rec['cases'][t] = dict(e0=e, e0_exported=ref_e[key][t], replay_rel=rel)   # after its array
                        ev('ours_case', key=key, case=t, **rec['cases'][t])
                        save(force=True)
                finally:
                    del o_, fx
                    gc.collect()
            try:
                replay()
                worst = max(c_['replay_rel'] for c_ in rec['cases'].values())
                rec.update(status='complete', G_ours=bool(worst <= 1e-10), replay_worst_rel=worst)
                if rec['G_ours']:
                    OX[key] = True                     # A35.3: objects released; rebuilt per X for timing
            except Exception as e:  # noqa: BLE001
                rec.update(status='run_failed', error=f'{type(e).__name__}: {e}')
                gc.collect()
            save(force=True)
            log(f'   ours {key}: {rec["status"]} G-ours={rec.get("G_ours")}')
        return len(OX)
    phase('ours_pass', ours_pass)

    # ---- 2. Stage-B members on the full verdict cohort ---------------------------------------------------------------
    def phi(r):                                        # A35.3: no cached prefixes; one live via the ARM cache
        return cell.Phi[:, :r]

    ARM = {}

    def arm(kind, r, hp):                              # online data per (kind, r, hp), one live at a time
        k = (kind, r, hp)
        if k not in ARM:
            ARM.clear()
            Phi_r = phi(r)
            if kind == 'none':
                ARM[k] = cell.arm_data('none', r, '-')[0]
            else:
                z = np.load(tdir / 'artifacts' / f'HR~{kind}~{r}~{hp}.npz')
                if kind == 'ecsw':
                    dat, _ = BD.ecsw_data(Phi_r, z['nodes'], z['weights'], d, L)
                    ARM[k] = dict(dat, Phi=Phi_r)
                else:
                    idx = H.stencil_index(z['P'], d, L)
                    ARM[k] = jax.block_until_ready(dict(Pst=H.stencil_rows(Phi_r, idx), T1=jnp.asarray(z['T1']),
                                                        T2=jnp.asarray(z['T2']), T3E=jnp.asarray(z['T3E']), Phi=Phi_r))
        return ARM[k]

    groups = {}
    for m in FZ['members']:
        kind, r, hp, scheme, dt, solver = parse(m)
        groups.setdefault((kind, r, hp, scheme, dt), []).append(m)
        res['inventory'][m] = dict(status='not_run')
    SCORED = {}                                        # member -> timing case -> certified lattice fields
    QUAL = []
    for (kind, r, hp, scheme, dt), mems in sorted(groups.items(), key=lambda kv: str(kv[0])):
        def grp(kind=kind, r=r, hp=hp, scheme=scheme, dt=dt, mems=mems):
            dat = arm(kind, r, hp)
            opt = (scheme, dt)
            tight, tighter = {}, {}
            ap = res.setdefault('anchors_partial', {}).setdefault(f'{kind}|{r}|{hp}|{scheme}|{dt:.8g}', {})
            for k in cases:
                tight[k] = cell.run(kind, r, dat, 'native', opt, 'tight', *ic(k))
                ap[f'tight|{k}'] = dict(cert=tight[k]['cert'], W=tight[k]['W'])
                ev('anchor', group=f'{kind}|{r}|{hp}|{scheme}|{dt:.8g}', level='tight', case=k, cert=tight[k]['cert'],
                   W=np.asarray(tight[k]['W']).tolist())
                save()                                 # A35.4: each anchor checkpointed at once
                if k in cases[:2]:
                    tighter[k] = cell.run(kind, r, dat, 'native', opt, 'tighter', *ic(k))
                    ap[f'tighter|{k}'] = dict(cert=tighter[k]['cert'], W=tighter[k]['W'])
                    ev('anchor', group=f'{kind}|{r}|{hp}|{scheme}|{dt:.8g}', level='tighter', case=k,
                       cert=tighter[k]['cert'], W=np.asarray(tighter[k]['W']).tolist())
                    save()
            for m in mems:
                solver = parse(m)[5]
                part = res.setdefault('partial', {}).setdefault(m, {})
                try:
                    recs = {}
                    for k in cases:                    # A32.7: per-case checkpoint
                        recs[k] = cell.run(kind, r, dat, solver, opt, 'prod', *ic(k))
                        part[k] = dict(cert=recs[k]['cert'], e0=score(k, recs[k]['fields_lat']), W=recs[k]['W'])
                        ev('member_case', member=m, case=k, cert=part[k]['cert'], e0=part[k]['e0'],
                           W=np.asarray(part[k]['W']).tolist())
                        save()
                    q1 = PL.q1_config(recs, tight, tighter, n0, cases[:2], cases)
                    e0 = {k: part[k]['e0'] for k in cases}
                    tz = np.load(tdir / 'arrays' / f"{tune['stageA'][m]['stage']}~{m.replace('|', '~')}.npz")
                    scale = max(1., float(np.max(np.abs(tz['W']))))
                    rep = max(float(np.max(np.abs(recs[t]['W'] - tz['W'][i]))) for i, t in enumerate(timing)) / scale
                    replay_ok = bool(np.isfinite(rep) and rep <= REPLAY_TOL)
                    np.savez(out / 'arrays' / f'B~{m.replace("|", "~")}.npz',
                             W=np.stack([recs[k]['W'] for k in cases]), Wtight=np.stack([tight[k]['W'] for k in cases]),
                             Wtighter=np.stack([tighter[k]['W'] for k in cases[:2]]),
                             gamma=np.stack([recs[k]['gamma'] for k in cases]))
                    es = [e0[k]['e'] for k in res['eligible_cases']]
                    res['members'][m] = dict(q1=q1, e0=e0, replay_vs_tune_W=rep, replay_ok=replay_ok, cases=cases,
                                             certs={k: recs[k]['cert'] for k in cases},
                                             anchors=dict(tight={k: tight[k]['cert'] for k in cases},
                                                          tighter={k: tighter[k]['cert'] for k in cases[:2]}),
                                             W=float(np.max(es)) if es and all(np.isfinite(es)) else float('nan'),
                                             med=float(np.median(es)) if es and all(np.isfinite(es)) else float('nan'))
                    res['inventory'][m] = dict(status='complete' if replay_ok else 'replay_failed')
                    res['partial'].pop(m, None)
                    log(f'   {m}: q1={q1["q1"]} W={res["members"][m]["W"]:.4g} replay={rep:.1e}')
                    if q1['q1'] and replay_ok:
                        QUAL.append(m)
                        SCORED[m] = {t: recs[t]['fields_lat'] for t in timing}
                except Exception as e:  # noqa: BLE001
                    res['inventory'][m] = dict(status='run_failed', reason=f'{type(e).__name__}: {e}')
                save(force=True)
            return True
        if phase(f'B|{kind}|{r}|{hp}|{scheme}|{dt:.8g}', grp) is None:
            for m in mems:
                if res['inventory'][m]['status'] == 'not_run':
                    res['inventory'][m] = dict(status='run_failed', reason='anchors / construction failed')
            save(force=True)

    # ---- 3. A-B-A timing: every admitted X against every member Q1-qualified (and replayed) here --------------------
    def aba():
        mi = FZ['member_info']                         # A33.3: the registered A1.12 total order, actual statistics
        mem = sorted(QUAL, key=lambda m: ST.total_order_key(dict(
            W=res['members'][m]['W'], med=res['members'][m]['med'], t=mi[m]['t'], r=parse(m)[1], kind=parse(m)[0],
            hr_points=mi[m]['hr_points'], scheme=parse(m)[3], dt=parse(m)[4], solver=parse(m)[5], id=m)))
        for key in OX:
            for m in mem:
                res['timing'][f'{key}||{m}'] = dict(status='not_timed')
        save(force=True)
        for key in OX:                                 # A35.3: one comparator object live at a time
            if time.perf_counter() - T0 > DEFER_FRAC * budget_s:
                for m in mem:
                    res['timing'][f'{key}||{m}'] = dict(status='not_timed', reason='budget')
                save(force=True)
                continue
            def time_x(key=key):                       # A37.1: comparator and every closure scoped to this call
                o_, fx = make_x(key)
                try:
                    for m in mem:                              # registered A1.12 total order
                        kind, r, hp, scheme, dt, solver = parse(m)
                        pk = f'{key}||{m}'
                        if time.perf_counter() - T0 > DEFER_FRAC * budget_s:
                            res['timing'][pk] = dict(status='not_timed', reason='budget')
                            save(force=True)
                            continue
                        q = cell.query(kind, r, solver)
                        sch = RM.schedule(d, scheme, dt, 'prod_native' if solver == 'native' else 'prod')
                        dat = arm(kind, r, hp)
                        fB = lambda u, v, q=q, dat=dat, sch=sch: q(u, v, dat, sch)['fields']
                        fX = lambda u, v, fx=fx: fx(u, v)['fields']
                        rec = {}
                        res['timing'][pk] = dict(status='running', cases=rec)
                        try:
                            for t in timing:
                                c, j = t.split('|')
                                ux, v = o_.u0(fam[c], int(j)), jnp.float64(fam[c].nu(int(j)))
                                ub = jnp.asarray(fam[c].ic(L, int(j)))
                                jax.block_until_ready(fx(ux, v))
                                jax.block_until_ready(fB(ub, v))                 # warm both
                                sX, sB = XS[key][t], SCORED[m][t]                # bound to the scored / certified outputs
                                chkX = lambda o, s=sX: disc_finite(o_.lattice(dict(fields=o)), s)
                                chkB = lambda o, s=sB: disc_finite(cell.lattice(o), s)
                                rec[t] = dict(triples=[])
                                def cb(pk=pk, t=t, rec=rec):          # A42: each attempt durable at once
                                    ev('triple', pair=pk, case=t, **rec[t]['triples'][-1])
                                    save()
                                timed_triples(fX, fB, (ux, v), (ub, v), chkX, chkB, rec=rec[t]['triples'], save=cb)
                            ok = all(all(np.isfinite(x) and x <= TIMED_TOL for tr in rc['triples']
                                         for x in (tr['dA1'], tr['dB'], tr['dA2'])) for rc in rec.values())
                            res['timing'][pk] = dict(status='complete', cases=rec, outputs_ok=bool(ok))
                        except Exception as e:  # noqa: BLE001
                            res['timing'][pk] = dict(status='run_failed', reason=f'{type(e).__name__}: {e}', cases=rec)
                        save(force=True)
                finally:
                    del o_, fx
                    gc.collect()
            try:
                time_x()
                save(force=True)                       # comparator end (A42)
            except Exception as e:  # noqa: BLE001
                for m in mem:
                    if res['timing'][f'{key}||{m}'].get('status') in ('not_timed', 'running'):
                        res['timing'][f'{key}||{m}'] = dict(status='run_failed', reason=f'comparator: {e}')
                save(force=True)
                gc.collect()
        return len(res['timing'])
    try:
        phase('aba', aba)
        res['total_seconds'] = time.perf_counter() - T0
        save(force=True)
    finally:
        jl.close()                                     # A42: closed on every exit
    log('done')


if __name__ == '__main__':
    main()
