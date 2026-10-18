"""jcp-fom2d C2 paired-timing job for ONE ROM mesh (DESIGN §5 as amended by A1.3–A1.5). One job, one GPU.

  0  bank on mesh N (P1 `p1common.Bank`, explicit edges), ROM settings (P1 `Setting`, non-recording fused query =
     the deployed full-field query), dev6 inputs: ROM full field on mesh N (sampled at quadrature points inside the
     timed query), FOM prepared inputs (fomrun2.prepare, outside every timer, seconds recorded).
  1  scoring runs (untimed) of every subject on every case: identity reference, E0 R_ST score of the lattice restriction;
     G-RR (ROM vs its target errors) and G-FR (FOM vs its ladder errors), |de| <= 1e-9.
  2  stage 0: every FOM candidate, full-field contract, warm, unpaired: dev6 cases 0, 1 (median) -> screening cost.
  3  stage 1 (selection block): per group (target, family, criterion) the three stage-0-cheapest candidates; A-B-A
     triplets on every case x 1 repetition; selected = smallest median-over-cases ratio. FROZEN-SELECTION.json written
     (sha256 recorded) before stage 2.
  4  stage 2 (inference block): every ROM subject and every selected FOM, FRESH triplets, every case x reps, randomised.
  5  controls: T-dbl (A twice, two separately dispatched synchronised executions), T-fault (A2 interval holds two A
     executions), T-id (a wrong identity reference for one subject -> subject-level veto).
Every timed execution's identity is compared with its scoring run (ROM: W, w0, per-step exit reasons and iteration
counts, plus the output fingerprint; FOM: sha256 of the lattice restriction plus the full-output fingerprint).
Drift t_A2/t_A1 outside [1/1.1, 1.1] -> the triplet is repeated (<= 2 retries) else marked invalid.

    python c2time.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import p1common as P
from p1common import Q, sha
import jax
import jax.numpy as jnp

import fomrun2 as FR
import refstore as RS
import cohorts as CH

BAND = (1 / 1.1, 1.1)
TOL_REPRO = 1e-9


def sync_time(fn):
    t = time.perf_counter()
    o = fn()
    jax.block_until_ready(o)
    return time.perf_counter() - t, o


_BURN = jax.jit(lambda a: jnp.tanh(a @ a) * .5)


def burn(seconds):
    a = jnp.ones((512, 512), jnp.float64) * .01
    jax.block_until_ready(_BURN(a))
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        a = jax.block_until_ready(_BURN(a))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    N = int(cfg['mesh'])
    bsec = float(cfg.get('burn', 0.05))
    reps = int(cfg.get('reps', 3))
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, store_version=RS.store_version(), checkpoint_sha256=P.QS.sha_file(P.T2.CKPT),
               mesh=N, subjects={}, prep={}, stage0={}, stage1=[], selection={}, stage2=[], controls={}, memory={},
               gates={}, complete=False)
    save = lambda: (out / 'result.json').write_text(json.dumps(clean(rep), indent=1) + '\n')
    co = {c: CH.Cohort(2, c) for c in ('dev6', 'val32')}
    cases = [(c, int(j), co[c].D[int(j)]) for c, j in cfg['cases']]
    cid = lambda c: f'{c[0]}|{c[1]}'

    # ---- 0. bank, ROM settings, inputs -------------------------------------------------------------------------------
    bank = P.Bank(N, cfg['edges'], nrb=cfg.get('bank_blocks'))
    rep['memory']['bank'] = dict(P.mem(), bank_bytes=bank.bytes, build_seconds=bank.build_seconds)
    print('BANK', N, el(), P.mem(), flush=True)
    settings = {}

    forms_needed = {}
    for r_ in [cfg['A']] + cfg['roms']:
        forms_needed.setdefault((int(r_['Rp']), r_['rule'], r_.get('policy', 'frozen')), set()).add(r_.get('form', 'LSPG'))

    def setting(sp):
        k = (int(sp['Rp']), sp['rule'], sp.get('policy', 'frozen'))
        if k not in settings:
            fs = tuple(['LSPG'] + sorted(f for f in forms_needed[k] if f != 'LSPG'))
            settings[k] = P.Setting(bank, f'{k}', k[0], k[1], k[2], forms=fs, nrec=0)
        return settings[k]

    roms = {cfg['A']['label']: cfg['A']}
    for r in cfg['roms']:
        assert r['label'] not in roms, r['label']
        roms[r['label']] = r
    U0 = {}
    for c in cases:
        u0n = Q.e.initial(N, c[2])
        U0[cid(c)] = (jnp.asarray(u0n), float(c[2][4]))
    foms = FR.FOMs()
    fkeys = list(cfg['foms'])
    X = {}
    for k in fkeys:
        sp = FR.parse(k)
        tp = []
        for c in cases:
            x, t = FR.prepare(sp, c[2])
            X[(k, cid(c))] = x
            tp.append(t)
        rep['prep'][k] = dict(median=float(np.median(tp)), max=float(np.max(tp)))
    rep['memory']['inputs'] = P.mem()

    def rom_call(lb, c):
        sp = roms[lb]
        st = setting(sp)
        st.check('main', bank)
        u0, nu = U0[cid(c)]
        form = sp.get('form', 'LSPG')
        assert int(round(.25 / float(sp['dt']))) <= st.nstep, ('per-step evidence buffer too short', sp['dt'])
        q = st.q0[form]
        d = st.data[('main', form)]
        sch = st.sched(form, sp['scheme'], float(sp['dt']), sp.get('level', 'prod'))
        cold = st.cold
        return lambda: q['fused'](u0, nu, d, cold, sch)

    def rom_ident(o):
        s_ = o['stats']
        return sha(np.concatenate([np.asarray(o['W']).ravel(), np.asarray(o['w0']).ravel(),
                                   np.asarray(s_['step_reason'], float), np.asarray(s_['step_it'], float),
                                   np.asarray(P._reduce(o['fields']))]))

    def fom_call(k, c):
        kk = k[len('CTRL-ID|'):] if k.startswith('CTRL-ID|') else k
        return foms.callable(FR.parse(kk), X[(k, cid(c))], float(c[2][4]), ('mesh', N))

    def fom_ident(F):
        return sha(np.concatenate([np.asarray(F[:, ::N // 256, ::N // 256]).ravel(), np.asarray(P._reduce(F))]))

    def call(lb, c):
        return rom_call(lb, c) if lb in roms else fom_call(lb, c)

    def ident(lb, o):
        return rom_ident(o) if lb in roms else fom_ident(o)

    def fields(lb, o):
        return o['fields'] if lb in roms else o

    # ---- 1. scoring runs, identities, G-RR / G-FR --------------------------------------------------------------------
    REF = {}
    for lb in list(roms) + fkeys:
        rows = []
        for c in cases:
            fn = call(lb, c)
            t, o = sync_time(fn)
            REF[(lb, cid(c))] = ident(lb, o)
            F = np.asarray(fields(lb, o)[:, ::N // 256, ::N // 256])
            s = RS.score(F, 2, c[0], c[1], 'ST')
            want = (roms[lb].get('target_e') or {}).get(cid(c)) if lb in roms else cfg['foms'][lb]['ladder_e'].get(cid(c))
            row = dict(case=cid(c), e=s['e'], eligible=s['eligible'], first_seconds=t, want=want,
                       repro=None if want is None else bool(abs(s['e'] - want) <= TOL_REPRO),
                       finite=bool(np.isfinite(F).all()))
            if lb in roms:
                row['valid'] = P.run_valid(o)
            rows.append(row)
        rep['subjects'][lb] = dict(kind='rom' if lb in roms else 'fom', spec=roms.get(lb) or dict(key=lb), scoring=rows,
                                   repro_ok=all(r['repro'] is not False for r in rows),
                                   repro_checked=sum(r['repro'] is not None for r in rows))
        print('SCORED', lb, rep['subjects'][lb]['repro_ok'], el(), flush=True)
    rep['gates']['G-RR'] = all(rep['subjects'][lb]['repro_ok'] and rep['subjects'][lb]['repro_checked'] == len(cases)
                               and all(r_.get('valid') is True for r_ in rep['subjects'][lb]['scoring'])
                               for lb in roms if lb != cfg['A']['label'])           # every target ROM: target_e and Q1-valid runs
    rep['gates']['A-valid'] = all(r_.get('valid') is True for r_ in rep['subjects'][cfg['A']['label']]['scoring'])
    rep['gates']['G-FR'] = all(rep['subjects'][k]['repro_ok'] and rep['subjects'][k]['repro_checked'] == len(cases)
                               for k in fkeys)
    rep['memory']['scoring'] = P.mem()
    save()

    VETO = set()

    def check(lb, c, o):
        ok = ident(lb, o) == REF[(lb, cid(c))]
        if not ok:
            VETO.add(lb)
        return ok

    # ---- 2. stage 0: full-field screening (unpaired) -----------------------------------------------------------------
    for k in fkeys:
        ts, ok = [], True
        for c in cases[:2]:
            fn = fom_call(k, c)
            jax.block_until_ready(fn())
            burn(bsec)
            t, o = sync_time(fn)
            ok = check(k, c, o) and ok
            ts.append(t)
        rep['stage0'][k] = dict(times=ts, cost=float(np.median(ts)), identity_ok=ok)
    save()
    print('STAGE0', len(fkeys), el(), flush=True)

    A = cfg['A']['label']

    def timed_A(c):
        fn = rom_call(A, c)
        burn(bsec)
        t, o = sync_time(fn)
        return t, check(A, c, o)

    def triplet(lb, c, r_, stage, fault=None):
        tries = []
        fnB = call(lb, c)
        for attempt in range(3):
            ta1, ha1 = timed_A(c)
            burn(bsec)
            tb, ob = sync_time(fnB)
            hb = check(lb, c, ob) if fault != 'identity' else False
            if fault == 'identity':
                VETO.add(lb)
            if fault == 'drift':        # A2 interval holds two separately dispatched, synchronised A executions
                fnA = rom_call(A, c)
                burn(bsec)
                t0 = time.perf_counter()
                o1 = jax.block_until_ready(fnA())
                o2 = jax.block_until_ready(fnA())
                ta2 = time.perf_counter() - t0
                ha2 = check(A, c, o1) and check(A, c, o2)
            else:
                ta2, ha2 = timed_A(c)
            drift = ta2 / ta1
            tries.append(dict(B=lb, case=cid(c), rep=r_, stage=stage, attempt=attempt, tA1=ta1, tA2=ta2, tB=tb, drift=drift,
                              ratio=tb / (.5 * (ta1 + ta2)), hashes_ok=bool(ha1 and ha2 and hb)))
            if BAND[0] <= drift <= BAND[1]:
                break
        x = dict(tries[-1], valid=bool(BAND[0] <= tries[-1]['drift'] <= BAND[1] and all(t_['hashes_ok'] for t_ in tries)),
                 retries=len(tries) - 1, drifts=[t_['drift'] for t_ in tries], attempts=tries)
        return x

    # warm every subject on every case once before any timing (first-call effects)
    for lb in list(roms) + fkeys:
        for c in cases:
            o = jax.block_until_ready(call(lb, c)())
            check(lb, c, o)

    # ---- 3. stage 1: selection block ---------------------------------------------------------------------------------
    rng = np.random.default_rng(int(cfg.get('seed', 1)))
    top = {}
    for g in cfg['groups']:
        key0 = lambda k: (rep['stage0'][k]['cost'], k)
        g = dict(g, candidates=[k for k in g['candidates'] if k not in VETO])     # vetoed candidates never shortlisted
        if g.get('pool_by_family'):          # A2.1 union group: the three stage-0-cheapest of every family, pooled
            fams = sorted({FR.family(k) for k in g['candidates']})
            top[g['id']] = sorted({k for f in fams for k in sorted([c for c in g['candidates'] if FR.family(c) == f], key=key0)[:3]},
                                  key=key0)
        else:
            top[g['id']] = sorted(g['candidates'], key=key0)[:3]
    s1keys = sorted({k for v in top.values() for k in v})
    order = [(k, ci) for k in s1keys for ci in range(len(cases))]
    rng.shuffle(order)
    for k, ci in order:
        rep['stage1'].append(triplet(k, cases[ci], 0, 1))
    s1 = {}
    for x in rep['stage1']:
        s1.setdefault(x['B'], {})[x['case']] = x
    for g in cfg['groups']:
        sl = [k for k in top[g['id']] if k not in VETO]          # a candidate vetoed during stage 0/1 is never selected
        # A2.1: selection population = cases valid for EVERY shortlisted candidate; >= 5 of the cases or no selection
        pop = [cid(c) for c in cases if all(s1.get(k, {}).get(cid(c), {}).get('valid') for k in sl)]
        cost = {k: float(np.median([s1[k][c]['ratio'] for c in pop])) for k in sl} if pop else {}
        ok = len(sl) > 0 and len(pop) >= len(cases) - 1
        best = min(sl, key=lambda k: (cost[k], rep['stage0'][k]['cost'], k)) if ok else None
        rep['selection'][g['id']] = dict(target=g['target'], family=g['family'], criterion=g['criterion'], shortlist=sl,
                                         stage0={k: rep['stage0'][k]['cost'] for k in sl}, stage1=cost, population=pop,
                                         selected=best, status='selected' if ok else
                                         ('no candidate' if not sl else 'invalid timing (selection population < 5)'))
    frozen = dict(mesh=N, cases=[cid(c) for c in cases], store_version=RS.store_version(), selection=rep['selection'],
                  roms=sorted(roms), veto_after_stage1=sorted(VETO))
    fb = json.dumps(clean(frozen), indent=1, sort_keys=True).encode()
    (out / f'FROZEN-SELECTION-{N}.json').write_bytes(fb)
    rep['frozen_selection_sha256'] = hashlib.sha256(fb).hexdigest()
    save()
    print('STAGE1', len(s1keys), el(), flush=True)

    # ---- 4. stage 2: inference block (fresh triplets) ----------------------------------------------------------------
    sel = sorted({v['selected'] for v in rep['selection'].values() if v['selected']})
    s2 = [lb for lb in roms if lb != A] + sel
    order = [(lb, ci, r_) for lb in s2 for ci in range(len(cases)) for r_ in range(reps)]
    rng.shuffle(order)
    for i, (lb, ci, r_) in enumerate(order):
        rep['stage2'].append(triplet(lb, cases[ci], r_, 2))
        if (i + 1) % 24 == 0:
            save()
            print('STAGE2', i + 1, len(order), el(), flush=True)
    rep['memory']['stage2'] = P.mem()
    save()

    # ---- 5. controls --------------------------------------------------------------------------------------------------
    ctl = cfg.get('controls', {})
    dbl = []
    for c in [x for x in cases if [x[0], x[1]] in [list(y) for y in ctl.get('doubled_cases', [])]]:
        fnA = rom_call(A, c)
        for r_ in range(reps):
            burn(bsec)
            t1, o1 = sync_time(fnA)
            burn(bsec)
            t0 = time.perf_counter()
            oa = jax.block_until_ready(fnA())
            ob = jax.block_until_ready(fnA())
            tb = time.perf_counter() - t0
            burn(bsec)
            t2, o2 = sync_time(fnA)
            dbl.append(dict(case=cid(c), rep=r_, t1=t1, t_double=tb, t2=t2, ratio=tb / (.5 * (t1 + t2)), drift=t2 / t1,
                            identity_ok=all(check(A, c, o_) for o_ in (o1, oa, ob, o2))))
    med = float(np.median([d['ratio'] for d in dbl])) if dbl else float('nan')
    rep['controls']['T-dbl'] = dict(rows=dbl, median=med, fires=bool(1.8 <= med <= 2.2))
    if ctl.get('fault_subject'):
        lb = ctl['fault_subject']
        x = triplet(lb, cases[0], 0, 'control', fault='drift')
        rep['controls']['T-fault'] = dict(row=x, separation_ok=bool(x['drift'] > BAND[1]), fires=bool(not x['valid']))
    if ctl.get('identity_fault_subject'):
        src_lb = ctl['identity_fault_subject']               # control-only alias: same program, its own label and veto
        lb = 'CTRL-ID|' + src_lb
        if src_lb in roms:
            roms[lb] = dict(roms[src_lb], label=lb)
        else:
            cfg['foms'][lb] = cfg['foms'][src_lb]
            for c in cases:
                X[(lb, cid(c))] = X[(src_lb, cid(c))]
        for c in cases:
            REF[(lb, cid(c))] = REF[(src_lb, cid(c))]
        x = triplet(lb, cases[0], 0, 'control', fault='identity')
        rep['controls']['T-id'] = dict(row=x, alias=lb, source=src_lb, source_vetoed=src_lb in VETO,
                                       fires=bool(lb in VETO and not x['valid']),
                                       note='control-only alias; the production subject is not vetoed by the control')
    for k in cfg.get('pilot_foms', []):                  # A1.6 combined-workload memory pilot (untimed, bank resident)
        sp = FR.parse(k)
        x, tp = FR.prepare(sp, cases[0][2])
        t, F = sync_time(foms.callable(sp, x, float(cases[0][2][4]), ('mesh', N)))
        rep.setdefault('pilot_foms', {})[k] = dict(seconds_first=t, prep_seconds=tp, finite=bool(np.isfinite(np.asarray(F[:, ::64, ::64])).all()),
                                                   mem=P.mem())
        del F, x
        print('PILOT', k, round(t, 2), P.mem(), flush=True)
    rep['compile_log'] = foms.compile_log
    rep['veto'] = sorted(VETO)
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('C2TIME COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
