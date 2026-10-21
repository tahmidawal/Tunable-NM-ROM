"""imp-varcoef paired-timing job (DESIGN §8, A1.13-A1.16, A2.14, A2.16, A2.25, A2.26, A3.14, A3.25, A5.12).

A modified copy of jcp-fom2d `c2time.py` (stages, triplets, controls and identity logic unchanged). Changes:
  * ROM subjects (A = the C2 baseline; the frozen per-target ROM denominators of checks/rom-box.json) run on the dev6 anchor
    cases exactly as C2 (bank on the 4096^2 mesh, fused full-field query), scored with the E0 store (G-RR).
  * FOM subjects are imp-varcoef competitors 'key@C<contrast>' (vkeys.py), run on the variable-coefficient FOM case of each
    timing SLOT (A3.25): slot s = (anchor dev6 case s, FOM case cfg['slots'][C][s]); every FOM triplet uses A on the slot's
    anchor case. Full-field contract: six fields on the (N+1)^2 nodes of mesh N, one program per call; IC prepared outside.
  * 'KERNEL|key@C' and 'SETUP|key@C' subjects (A3.14): prepared-setup kernel and setup-only programs.
  * FOM scoring: lattice restriction F[:, ::N//256, ::N//256] against this lane's references; G-FR = reproduction of the
    ladder error per slot within 1e-9 and run validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'jcp-fom2d'))

import numpy as np
import p1common as P
from p1common import Q, sha
import jax
import jax.numpy as jnp

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vcoef as VC
import vkeys as VK
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
    ctl0 = cfg.get('controls', {})
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
    if cfg.get('bindings'):                                 # production: authenticate the hash chain (code audit 1 item 5)
        b_ = cfg['bindings']
        for name_, h_ in b_['files'].items():
            assert hashlib.sha256((HERE / name_).read_bytes()).hexdigest() == h_, ('binding mismatch', name_)
        el_ = json.loads((HERE / b_['eligibility']).read_text())
        ca_ = json.loads((HERE / b_['candidates']).read_text())
        for C_, v_ in cfg['slots'].items():
            assert v_ == [x['fom'] for x in el_['contrasts'][C_]['slots']], ('slots differ from the frozen mask', C_)
        qual_ = set(ca_['candidates'])
        for g_ in cfg['groups']:
            fam_, tau_, C_ = g_['id'].split('|')
            if fam_ == 'union':
                want_ = sorted({f'{k}@{C_}' for f2 in ('PSI', 'ETD', 'FDMG', 'CCV')
                                for k in ca_['groups'][C_[1:]].get(f'{f2}|{tau_}|{C_}', {}).get('candidates', [])})
            else:
                want_ = [f'{k}@{C_}' for k in ca_['groups'][C_[1:]][g_['id']]['candidates']]
            assert sorted(g_['candidates']) == sorted(want_), ('group differs from the qualified ladder group', g_['id'])
        for k_, v_ in cfg['foms'].items():
            base_ = VK.split_key(k_)
            if base_[2] != 1.0 and base_[0] != 'SETUP':
                bk_ = f'{base_[1]}@C{base_[2]:g}'
                assert bk_ in qual_ and v_['slot_e'] == ca_['candidates'][bk_]['slot_e'], ('slot_e not from the ladder', k_)
    if cfg.get('slot_e_from'):                              # smoke only: ladder errors from the in-job smoke ladder
        lr = json.loads(Path(cfg['slot_e_from']).read_text())
        for k_, v_ in cfg['foms'].items():
            pre_, kk_, C_ = VK.split_key(k_)
            v_['slot_e'] = {r['case']: r['e'] for r in lr['runs'] if r['key'] == kk_ and float(r['C']) == C_}
    fkeys = list(cfg['foms'])
    vcases = {sp_: VC.cohort(sp_) for sp_ in ('dev', 'val')}
    slots = {str(k): v for k, v in cfg['slots'].items()}          # contrast -> [fom case id per anchor index]
    assert all(len(v) == len(cases) for v in slots.values()), 'one FOM case per anchor slot (A3.25)'
    refdirs = cfg['refdir'] if isinstance(cfg['refdir'], list) else [cfg['refdir']]
    PROGS, XS = {}, {}

    def base(k):
        k = k[len('CTRL-ID|'):] if k.startswith('CTRL-ID|') else k
        return k

    def fcase(k, c):
        pre, kk, C = VK.split_key(base(k))
        Cs = 100.0 if C == 1.0 else C                     # C = 1 cost reading uses the C = 100 physical cases (paired)
        cidf = slots[f'{Cs:g}'][[cid(x) for x in cases].index(cid(c))]
        s_, j_ = cidf.split('|')
        return cidf, vcases[s_][int(j_)], C

    def prog(k):
        pre, kk, C = VK.split_key(base(k))
        if kk not in PROGS:
            PROGS[kk] = VK.program(VK.parse(kk), ('mesh', N))
        return PROGS[kk]

    def xget(k, c):
        pre, kk, C = VK.split_key(base(k))
        cidf, case, C = fcase(k, c)
        xk = (kk, cidf, C, pre)
        if xk not in XS:
            sp = VK.parse(kk)
            x0, tp = VK.prepare(sp, case)
            kp = jnp.asarray(VC.kappa_params(case, C))
            S_ = None
            if pre == 'KERNEL':
                S_ = jax.block_until_ready(prog(k).setup_only(kp))
            XS[xk] = ((x0, kp, S_), tp)
        return XS[xk][0]
    for k in fkeys:
        for c in cases:
            xget(k, c)
        tp = [XS[(VK.split_key(base(k))[1], fcase(k, c)[0], fcase(k, c)[2], VK.split_key(base(k))[0])][1] for c in cases]
        rep['prep'][k] = dict(median=float(np.median(tp)), max=float(np.max(tp)))
    rep['memory']['inputs'] = dict(P.mem(), n_inputs=len(XS))
    REFS = {}

    def fref(cidf, C):
        if (cidf, C) not in REFS:
            s_, j_ = cidf.split('|')
            fs = [Path(d_) / f'ref_C{C:g}_{s_}_{j_}.npz' for d_ in refdirs]
            fs = [f_ for f_ in fs if f_.exists()]
            assert fs, ('no reference', cidf, C)
            z = np.load(fs[-1])
            if cfg.get('bindings'):
                want_ = json.loads((HERE / cfg['bindings']['eligibility']).read_text())['contrasts'][f'{C:g}']['cases'][cidf]['ref_sha256']
                assert sha(np.ascontiguousarray(np.asarray(z['lat'], np.float64))) == want_, ('reference hash', cidf, C)
            REFS[(cidf, C)] = (z['lat'], float(np.linalg.norm(z['t0'].ravel())))
        return REFS[(cidf, C)]

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
        pre, kk, C = VK.split_key(base(k))
        x0, kp, S_ = xget(k, c)
        p_ = prog(k)
        if pre == 'SETUP':
            return lambda: p_.setup_only(kp)
        if pre == 'KERNEL':
            return lambda: p_.kernel(x0, S_)
        return lambda: p_.query(x0, kp)

    def fom_ident(k, o):
        pre = VK.split_key(base(k))[0]
        if pre == 'SETUP':
            return sha(np.concatenate([np.asarray(x, np.float64).ravel() for x in jax.tree_util.tree_leaves(o)]))
        F = o[0]
        return sha(np.concatenate([np.asarray(F[:, ::N // 256, ::N // 256]).ravel(), np.asarray(P._reduce(F))]))

    def call(lb, c):
        return rom_call(lb, c) if lb in roms else fom_call(lb, c)

    def ident(lb, o):
        return rom_ident(o) if lb in roms else fom_ident(lb, o)

    def fields(lb, o):
        return o['fields'] if lb in roms else o[0]

    # ---- 1. scoring runs, identities, G-RR / G-FR --------------------------------------------------------------------
    REF = {}
    for lb in list(roms) + fkeys:
        rows = []
        for c in cases:
            fn = call(lb, c)
            t, o = sync_time(fn)
            REF[(lb, cid(c))] = ident(lb, o)
            if lb not in roms and (VK.split_key(base(lb))[0] == 'SETUP' or VK.split_key(base(lb))[2] == 1.0):
                if VK.split_key(base(lb))[0] != 'SETUP':           # C = 1 cost-reading subject: validity only, no reference
                    okv, ev = VK.validity(VK.parse(VK.split_key(base(lb))[1]), o[0], o[1])
                    rows.append(dict(case=cid(c), e=None, eligible=None, first_seconds=t, want=None, repro=None,
                                     finite=bool(np.isfinite(np.asarray(o[0])).all()), valid=okv, kind='C1 diagnostic'))
                    continue
                rows.append(dict(case=cid(c), e=None, eligible=None, first_seconds=t, want=None, repro=None, finite=True,
                                 valid=True, kind='setup-only'))
                continue
            F = np.asarray(fields(lb, o)[:, ::N // 256, ::N // 256])
            if lb in roms:
                s = RS.score(F, 2, c[0], c[1], 'ST')
                want = (roms[lb].get('target_e') or {}).get(cid(c))
            else:
                cidf, case_, C_ = fcase(lb, c)
                R_, n0_ = fref(cidf, C_)
                okv, ev = VK.validity(VK.parse(VK.split_key(base(lb))[1]), F, o[1])
                s = dict(e=VK.lat_err(F, R_, n0_) if okv else None, eligible=True)
                want = cfg['foms'][lb]['slot_e'].get(cidf)
            row = dict(case=cid(c), e=s['e'], eligible=s['eligible'], first_seconds=t, want=want,
                       repro=None if (want is None or s['e'] is None) else bool(abs(s['e'] - want) <= TOL_REPRO),
                       finite=bool(np.isfinite(F).all()))
            if lb in roms:
                row['valid'] = P.run_valid(o)
            else:
                row.update(valid=okv, fom_case=cidf, evidence=ev)
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
                               and all(r_.get('valid') is True for r_ in rep['subjects'][k]['scoring'])
                               for k in fkeys if VK.split_key(k)[0] != 'SETUP' and VK.split_key(k)[2] != 1.0)  # query + KERNEL
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
        del fn, o                                           # no stage-0 closure/output survives (A1.6 release)
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
            hb = check(lb, c, ob)
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
            del o

    # ---- 3. stage 1: selection block ---------------------------------------------------------------------------------
    rng = np.random.default_rng(int(cfg.get('seed', 1)))
    top = {}
    pre_veto = {}
    for g in cfg['groups']:
        key0 = lambda k: (rep['stage0'][k]['cost'], k)
        pre_veto[g['id']] = len(g['candidates'])
        g = dict(g, candidates=[k for k in g['candidates'] if k not in VETO])     # vetoed candidates never shortlisted
        if g.get('pool_by_family'):          # A2.1 union group: the three stage-0-cheapest of every family, pooled
            fams = sorted({VK.family(k) for k in g['candidates']})
            top[g['id']] = sorted({k for f in fams for k in sorted([c for c in g['candidates'] if VK.family(c) == f], key=key0)[:3]},
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
                                         (('all candidates vetoed (identity)' if pre_veto[g['id']] else 'no candidate') if not sl
                                          else 'invalid timing (selection population < 5)'))
    keep = {v['selected'] for v in rep['selection'].values() if v['selected']}
    keep |= {k for k in (ctl0.get('fault_subject'), ctl0.get('identity_fault_subject')) if k and k not in roms}
    keep |= {k for k in fkeys if VK.split_key(k)[0] is not None or VK.split_key(k)[2] == 1.0}   # diagnostics stay
    keepk = {VK.split_key(base(x))[1] for x in keep if x not in roms}
    for k_ in list(PROGS):                                  # release programs and inputs no longer needed
        if k_ not in keepk:
            del PROGS[k_]
    for k_ in list(XS):
        if k_[0] not in keepk:
            del XS[k_]
    import gc
    gc.collect()
    rep['memory']['after_stage1_release'] = P.mem()
    frozen = dict(mesh=N, cases=[cid(c) for c in cases], store_version=RS.store_version(), selection=rep['selection'],
                  roms=sorted(roms), veto_after_stage1=sorted(VETO))
    fb = json.dumps(clean(frozen), indent=1, sort_keys=True).encode()
    (out / f'FROZEN-SELECTION-{N}.json').write_bytes(fb)
    rep['frozen_selection_sha256'] = hashlib.sha256(fb).hexdigest()
    save()
    print('STAGE1', len(s1keys), el(), flush=True)

    # ---- 4. stage 2: inference block (fresh triplets) ----------------------------------------------------------------
    sel = sorted({v['selected'] for v in rep['selection'].values() if v['selected']})
    selb = {VK.split_key(x)[1] for x in sel}
    diag = [k for k in fkeys if (VK.split_key(k)[0] is not None and VK.split_key(k)[1] + '@C' + f"{VK.split_key(k)[2]:g}" in
            {VK.split_key(x)[1] + '@C' + f"{VK.split_key(x)[2]:g}" for x in sel})            # A3.14 for selected winners
            or (VK.split_key(k)[0] is None and VK.split_key(k)[2] == 1.0 and VK.split_key(k)[1] in selb)]   # C=1 reading
    s2 = [lb for lb in roms if lb != A] + sel + diag
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
        for c in cases:                                      # a deliberately WRONG reference: the ordinary check must catch it
            REF[(lb, cid(c))] = 'corrupted-' + REF[(src_lb, cid(c))]
        pre = src_lb in VETO
        x = triplet(lb, cases[0], 0, 'control')
        rep['controls']['T-id'] = dict(row=x, alias=lb, source=src_lb, source_vetoed_before=pre, source_vetoed_after=src_lb in VETO,
                                       fires=bool(lb in VETO and not x['valid'] and (src_lb in VETO) == pre),
                                       note='control-only alias with a corrupted identity reference, through the ordinary '
                                            'check(); the production subject is unaffected')
    for k in cfg.get('pilot_foms', []):                  # combined-workload memory pilot (untimed, bank resident)
        c = cases[0]
        F, st = fom_call(k, c)()
        F = jax.block_until_ready(F)
        okv, ev = VK.validity(VK.parse(VK.split_key(base(k))[1]), F, st)
        rep.setdefault('pilot_foms', {})[k] = dict(valid=bool(okv), evidence=ev, shape=list(F.shape), mem=P.mem())
        del F, st
        print('PILOT', k, okv, P.mem(), flush=True)
    rep['pilots_ok'] = all(rep.get('pilot_foms', {}).get(k, {}).get('valid') is True for k in cfg.get('pilot_foms', []))
    rep['veto'] = sorted(VETO)
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('C2TIME COMPLETE', el(), flush=True)


if __name__ == '__main__':
    main()
