"""imp-faststep timing block (DESIGN 8, A1.11, A2.9 as replaced by A3.6). One GPU, one job, warm JIT, burn-in, f64.

Subjects expose prepared (function, arguments) pairs for init / evolve / decode / fused, built untimed, so only the
dispatch is inside a timed interval. Triplet types (each repetition: <= 2 drift retries, band [1/1.1, 1.1]):
  E  evolve triplets   A = arm 1 (L-CNR, same r) evolve | B init | B evolve | B decode | A evolve   (bank arms)
  F  fused triplets    A = the job anchor (L-CNR r = 384) fused | B fused | B init | B evolve | B decode | A fused
  D  direct triplets   X fused | B fused | X fused, for (X, B*(X)) and (G-CN, G-R1), after the in-job selection
Identity (A3.6): the untimed warm-up of every stage must reproduce the scored production coefficients to 1e-12
relative (fused and evolve); every timed evolve / fused output must then hash-equal its warm-up output (bitwise: same
compiled program, same inputs). Raw records are kept.
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

import fscore as F
import fsselect as FS
import p1common as P
from p1common import QS

BAND = FS.BAND


def sync_time(fn, args):
    t = time.perf_counter()
    o = fn(*args)
    jax.block_until_ready(o)
    return time.perf_counter() - t, o


def _close(Wa, Wb):
    Wa, Wb = np.asarray(Wa), np.asarray(Wb)
    return bool(np.isfinite(Wa).all() and np.max(np.abs(Wa - Wb)) <= 1e-12 * max(np.max(np.abs(Wb)), 1e-300))


# ------------------------------------------------------------------ subjects ----

def bank_subjects(st, settings, tabs, arms, evolver, U0, refs, Rp):
    """Bank subjects of one rank. refs[(arm, ci)] = W (6, r) of the scored production run."""
    subj = {}
    qg, ql = st.q['GAL'], st.q['LSPG']
    dg = st.data[('main', 'GAL')]
    for arm in arms:
        sid = f'{arm}|{Rp}'
        if arm.startswith('L-CNR'):
            stx = settings['M1152'] if arm == 'L-CNR-M1152' else st
            q = stx.q['LSPG']
            d = stx.data[('main', 'LSPG')]
            sch = stx.sched('LSPG', 'CNR', P.DT0, 'prod')

            def stages(ci, q=q, d=d, sch=sch, stx=stx):
                u0, nu, _ = U0[ci]
                w0, scale = q['init'](u0, stx.cold)
                W, s_ = q['evolve'](w0, nu, scale, d, sch)
                return dict(init=(q['init'], (u0, stx.cold)), evolve=(q['evolve'], (w0, nu, scale, d, sch)),
                            decode=(q['decode'], (W, d)), fused=(q['fused'], (u0, nu, d, stx.cold, sch)))
            h_ev = lambda o: F.sha(np.concatenate([np.asarray(o[0]).ravel(), np.asarray(o[1]['states']).ravel(),
                                                    np.asarray(o[1]['step_reason'], float), np.asarray(o[1]['step_it'], float)]))
            W_fu = lambda o: o['W']
            W_ev = lambda o: o[0]
        else:
            from fsrun import ARMS
            kind, dt = ARMS[arm]['kind'], ARMS[arm]['dt']
            ev = evolver(kind)
            tab = tabs['main']
            sch = F.sched(dt)

            def fused_fn(u0, nu, cold, tab, d, sch, ev=ev):
                w0, _ = qg['init'](u0, cold)
                W, s_ = ev(w0, nu, tab, sch)
                return qg['decode'](W, d), W, s_['Z']
            fz = jax.jit(fused_fn)

            def stages(ci, ev=ev, tab=tab, sch=sch, fz=fz):
                u0, nu, _ = U0[ci]
                w0, _ = qg['init'](u0, st.cold)
                W, s_ = ev(w0, jnp.float64(nu), tab, sch)
                return dict(init=(qg['init'], (u0, st.cold)), evolve=(ev, (w0, jnp.float64(nu), tab, sch)),
                            decode=(qg['decode'], (W, dg)), fused=(fz, (u0, jnp.float64(nu), st.cold, tab, dg, sch)))
            h_ev = lambda o: F.sha(np.concatenate([np.asarray(o[0]).ravel(), np.asarray(o[1]['Z'])[:int(o[1]['steps']) + 1].ravel()]))
            W_fu = lambda o: o[1]
            W_ev = lambda o: o[0]
        subj[sid] = dict(id=sid, family='bank', arm=arm, r=Rp, stages=stages, h_ev=h_ev, W_fu=W_fu, W_ev=W_ev,
                         ref={ci: refs[(arm, ci)] for ci in U0 if (arm, ci) in refs})
    return subj


def pod_subjects(T):
    subj = {}
    for mid, s in T.items():
        if s['kind'] == 'native':
            def stages(ci, s=s):
                return dict(fused=(s['q'], (s['U0i'][ci], jnp.float64(s['U0'][ci][1]), s['dat'], s['sch'])))
            subj[mid] = dict(id=mid, family='native', r=s['r'], stages=stages, h_ev=None, W_fu=lambda o: o['W'], W_ev=None,
                             ref=dict(s['ref']))
            continue
        ev, tb, Phi, init, decode = s['ev'], s['tb'], s['Phi'], s['init'], s['decode']
        sch = F.sched(s['dt'])

        def fused_fn(u, nu, Phi, tb, sch, ev=ev, init=init, decode=decode):
            W, s_ = ev(init(u, Phi), nu, tb, sch)
            return decode(W, Phi), W, s_['Z']
        fz = jax.jit(fused_fn)

        def stages(ci, s=s, ev=ev, tb=tb, Phi=Phi, init=init, decode=decode, sch=sch, fz=fz):
            u, nu = s['U0i'][ci], jnp.float64(s['U0'][ci][1])
            c0 = init(u, Phi)
            W, s_ = ev(c0, nu, tb, sch)
            return dict(init=(init, (u, Phi)), evolve=(ev, (c0, nu, tb, sch)), decode=(decode, (W, Phi)),
                        fused=(fz, (u, nu, Phi, tb, sch)))
        h_ev = lambda o: F.sha(np.concatenate([np.asarray(o[0]).ravel(), np.asarray(o[1]['Z'])[:int(o[1]['steps']) + 1].ravel()]))
        subj[mid] = dict(id=mid, family='pod', r=s['r'], stages=stages, h_ev=h_ev, W_fu=lambda o: o[1], W_ev=lambda o: o[0],
                         ref=dict(s['ref']))
    return subj


# ------------------------------------------------------------------ triplets ----

def _rep(timed, A, B, ci, kind, burn, cache):
    """One repetition with <= 2 drift retries. timed: names of B's stages to time between the two A brackets."""
    sa = cache(A, ci)
    sb = cache(B, ci)
    a_stage = 'evolve' if kind == 'E' else 'fused'
    tries = []
    for attempt in range(3):
        QS.burn(burn)
        ta1, oa1 = sync_time(*sa[a_stage])
        rec = dict(attempt=attempt)
        ok = True
        for nm in timed:
            if nm not in sb:
                continue
            QS.burn(burn)
            t, o = sync_time(*sb[nm])
            rec[f't_{nm}'] = t
            if nm == 'evolve' and B['h_ev'] is not None:
                ok = ok and (B['h_ev'](o) == B['warm'][ci]['h_ev'])
            if nm == 'fused':
                ok = ok and (F.sha(np.asarray(B['W_fu'](o))) == B['warm'][ci]['h_fu'])
        ok = ok and B['warm'][ci]['scored_ok']
        QS.burn(burn)
        ta2, oa2 = sync_time(*sa[a_stage])
        if kind == 'E':
            ok = ok and A['h_ev'](oa1) == A['warm'][ci]['h_ev'] and A['h_ev'](oa2) == A['warm'][ci]['h_ev']
        else:
            ok = ok and all(F.sha(np.asarray(A['W_fu'](o_))) == A['warm'][ci]['h_fu'] for o_ in (oa1, oa2))
        ok = ok and A['warm'][ci]['scored_ok']
        main = rec.get('t_evolve' if kind == 'E' else 't_fused')
        rec.update(tA1=ta1, tA2=ta2, drift=ta2 / ta1, identity_ok=bool(ok),
                   ratio=(.5 * (ta1 + ta2)) / main if (kind == 'E' and main) else (main / (.5 * (ta1 + ta2)) if main else None))
        tries.append(rec)
        if BAND[0] <= rec['drift'] <= BAND[1]:
            break
    last = dict(tries[-1])
    last.update(valid=bool(BAND[0] <= last['drift'] <= BAND[1] and last['identity_ok']), retries=len(tries) - 1,
                attempts=tries, A=A['id'], B=B['id'], case=ci, type=kind)
    return last


def run(cfg, subjects, U0, dev, cases, selector):
    """subjects: {id: subject}; selector(screen) -> list of (X, B) direct pairs (in-job frozen selection, A3.6)."""
    tc = cfg['timing']
    reps, burn = int(tc.get('reps', 3)), float(tc.get('burn', .05))
    rng = np.random.default_rng(int(tc.get('seed', 20261021)))
    STG = {}

    def cache(S, ci):
        k = (S['id'], ci)
        if k not in STG:
            STG[k] = S['stages'](ci)
            outs = {}
            for nm, (fn, args) in STG[k].items():        # warm every compiled stage once (JIT, not timed)
                outs[nm] = fn(*args)
                jax.block_until_ready(outs[nm])
            Wsc = S['ref'].get(ci)
            w = dict(h_fu=F.sha(np.asarray(S['W_fu'](outs['fused']))),
                     scored_ok=bool(Wsc is not None and _close(S['W_fu'](outs['fused']), Wsc)))
            if 'evolve' in outs and S['h_ev'] is not None:
                w['h_ev'] = S['h_ev'](outs['evolve'])
                w['scored_ok'] = w['scored_ok'] and _close(S['W_ev'](outs['evolve']), Wsc)
            S.setdefault('warm', {})[ci] = w
        return STG[k]
    cid = lambda i: f'{cases[i][0]}|{cases[i][1]}'
    anchor = subjects[tc['anchor']]
    recs = dict(E=[], F=[], D=[])
    order = []
    for sid, S in subjects.items():
        if S['family'] == 'bank':
            order += [('E', sid, ci, r_) for ci in dev for r_ in range(reps)]
        order += [('F', sid, ci, r_) for ci in dev for r_ in range(reps)]
    rng.shuffle(order)
    for kind, sid, ci, r_ in order:
        S = subjects[sid]
        if kind == 'E':
            A = subjects[f"L-CNR|{S['r']}"]
            x = _rep(('init', 'evolve', 'decode'), A, S, ci, 'E', burn, cache)
        else:
            x = _rep(('fused', 'init', 'evolve', 'decode'), anchor, S, ci, 'F', burn, cache)
        x.update(rep=r_, case=cid(ci))
        recs[kind].append(x)
    devc = [cid(i) for i in dev]
    screen = {sid: FS.case_ratio_stats([t for t in recs['F'] if t['B'] == sid], devc, reps) for sid in subjects}
    pairs = selector({k: (v['value'] if v['status'] == 'ok' else None) for k, v in screen.items()})
    dorder = [(X, B, ci, r_) for X, B in pairs for ci in dev for r_ in range(reps)]
    rng.shuffle(dorder)
    for X, B, ci, r_ in dorder:
        x = _rep(('fused',), subjects[X], subjects[B], ci, 'D', burn, cache)
        x.update(rep=r_, case=cid(ci), pair=f'{X}~{B}')
        recs['D'].append(x)
    warm = {sid: {cid(ci): dict(v) for ci, v in S.get('warm', {}).items()} for sid, S in subjects.items()}
    return dict(records=recs, pairs=pairs, screen=screen, anchor=tc['anchor'], reps=reps, burn=burn, warm=warm)
