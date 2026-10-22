"""imp-faststep timing block (DESIGN 8, A1.11, A3.6, A5.7-A5.9). One GPU, one job, warm JIT, burn-in, f64.

Subjects expose prepared (function, arguments) pairs for init / evolve / decode / fused, built untimed, so only the
dispatch is inside a timed interval. Triplet types (each repetition: <= 2 drift retries, band [1/1.1, 1.1]):
  E  evolve triplets   A = arm 1 (L-CNR, same r) evolve | B init | B evolve | B decode | A evolve   (bank arms)
                        ratio = mean(t_A) / t_B,evolve  (speed-up of B's evolution over the incumbent)
  F  fused triplets    A = the job anchor (L-CNR r = 384) fused | B fused | B init | B evolve | B decode | A fused
                        ratio = t_B,fused / mean(t_A)   (relative full-query time; screening only)
  D  direct triplets   X fused | B fused | X fused, for (X, B*(X)) and (G-CN, G-R1) after the in-job selection
                        ratio = t_B,fused / mean(t_X)
Identity (A5.7): a timed evolve must be bitwise equal to the scored production run (same compiled program, same
inputs; hash of W and states); a timed fused query must be within 1e-12 relative of the scored coefficients (different
compiled program; for the native replays, whose scored run IS the fused program, bitwise) and bitwise equal to its own
warm-up (W, states and fields hashed). A failure in ANY attempt marks the record and the whole block.
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
    return bool(Wa.shape == Wb.shape and np.isfinite(Wa).all()
                and np.max(np.abs(Wa - Wb)) <= 1e-12 * max(np.max(np.abs(Wb)), 1e-300))


def _hash_tree(o):
    return F.sha(np.concatenate([np.asarray(x, np.float64).ravel() for x in jax.tree_util.tree_leaves(o)
                                 if hasattr(x, 'shape') and np.asarray(x).dtype.kind in 'fiub']))


# ------------------------------------------------------------------ subjects ----

def bank_subjects(st, settings, tabs, arms, evolver, U0, refs, Rp, check_tab, lcheck):
    """Bank subjects of one rank. refs[(arm, ci)] = dict(W, h_ev) of the scored production run."""
    from fsrun import ARMS
    subj = {}
    qg = st.q['GAL']
    dg = st.data[('main', 'GAL')]
    for arm in arms:
        sid = f'{arm}|{Rp}'
        if arm.startswith('L-CNR'):
            stx = settings['M1152'] if arm == 'L-CNR-M1152' else st
            q = stx.q['LSPG']
            d = stx.data[('main', 'LSPG')]
            sch = stx.sched('LSPG', 'CNR', P.DT0, 'prod')

            def stages(ci, q=q, d=d, sch=sch, stx=stx):
                lcheck(stx)
                u0, nu, _ = U0[ci]
                w0, scale = q['init'](u0, stx.cold)
                W, s_ = q['evolve'](w0, nu, scale, d, sch)
                return dict(init=(q['init'], (u0, stx.cold)), evolve=(q['evolve'], (w0, nu, scale, d, sch)),
                            decode=(q['decode'], (W, d)), fused=(q['fused'], (u0, nu, d, stx.cold, sch)))
            h_ev = lambda o: F.sha(np.concatenate([np.asarray(o[0]).ravel(), np.asarray(o[1]['states']).ravel(),
                                                    np.asarray(o[1]['step_reason'], float), np.asarray(o[1]['step_it'], float)]))
            subj[sid] = dict(id=sid, family='bank', arm=arm, r=Rp, stages=stages, h_ev=h_ev, W_fu=lambda o: o['W'],
                             W_ev=lambda o: o[0], fused_is_scored=False,
                             ref={ci: refs[(arm, ci)] for ci in U0 if (arm, ci) in refs})
            continue
        kind, dt = ARMS[arm]['kind'], ARMS[arm]['dt']
        ev = evolver(kind)
        tab = tabs['main']
        sch = F.sched(dt)

        def fused_fn(u0, nu, cold, tab, d, sch, ev=ev):
            w0, _ = qg['init'](u0, cold)
            W, s_ = ev(w0, nu, tab, sch)
            return dict(fields=qg['decode'](W, d), W=W, Z=s_['Z'])
        fz = jax.jit(fused_fn)

        def stages(ci, ev=ev, tab=tab, sch=sch, fz=fz):
            check_tab(tab, 'main')
            u0, nu, _ = U0[ci]
            w0, _ = qg['init'](u0, st.cold)
            W, s_ = ev(w0, jnp.float64(nu), tab, sch)
            return dict(init=(qg['init'], (u0, st.cold)), evolve=(ev, (w0, jnp.float64(nu), tab, sch)),
                        decode=(qg['decode'], (W, dg)), fused=(fz, (u0, jnp.float64(nu), st.cold, tab, dg, sch)))
        h_ev = lambda o: F.sha(np.concatenate([np.asarray(o[0]).ravel(), np.asarray(o[1]['Z'])[:int(o[1]['steps']) + 1].ravel()]))
        subj[sid] = dict(id=sid, family='bank', arm=arm, r=Rp, stages=stages, h_ev=h_ev, W_fu=lambda o: o['W'],
                         W_ev=lambda o: o[0], fused_is_scored=False, ref={ci: refs[(arm, ci)] for ci in U0 if (arm, ci) in refs})
    return subj


def pod_subjects(T):
    subj = {}
    for mid, s in T.items():
        if s['kind'] == 'native':
            def stages(ci, s=s):
                return dict(fused=(s['q'], (s['U0i'][ci], jnp.float64(s['U0'][ci][1]), s['dat'], s['sch'])))
            subj[mid] = dict(id=mid, family='native', r=s['r'], stages=stages, h_ev=None, W_fu=lambda o: o['W'], W_ev=None,
                             fused_is_scored=True, ref=dict(s['ref']))
            continue
        ev, tb, Phi, init, decode = s['ev'], s['tb'], s['Phi'], s['init'], s['decode']
        sch = F.sched(s['dt'])

        def fused_fn(u, nu, Phi, tb, sch, ev=ev, init=init, decode=decode):
            W, s_ = ev(init(u, Phi), nu, tb, sch)
            return dict(fields=decode(W, Phi), W=W, Z=s_['Z'])
        fz = jax.jit(fused_fn)

        def stages(ci, s=s, ev=ev, tb=tb, Phi=Phi, init=init, decode=decode, sch=sch, fz=fz):
            s['guard'](tb)
            u, nu = s['U0i'][ci], jnp.float64(s['U0'][ci][1])
            c0 = init(u, Phi)
            W, s_ = ev(c0, nu, tb, sch)
            return dict(init=(init, (u, Phi)), evolve=(ev, (c0, nu, tb, sch)), decode=(decode, (W, Phi)),
                        fused=(fz, (u, nu, Phi, tb, sch)))
        h_ev = lambda o: F.sha(np.concatenate([np.asarray(o[0]).ravel(), np.asarray(o[1]['Z'])[:int(o[1]['steps']) + 1].ravel()]))
        subj[mid] = dict(id=mid, family='pod', r=s['r'], stages=stages, h_ev=h_ev, W_fu=lambda o: o['W'], W_ev=lambda o: o[0],
                         fused_is_scored=False, ref=dict(s['ref']))
    return subj


# ------------------------------------------------------------------ triplets ----

def _rep(timed, A, B, ci, kind, burn, cache):
    """One repetition with <= 2 drift retries. timed: names of B's stages timed between the two A brackets."""
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
                ok = ok and B['h_ev'](o) == B['ref'][ci]['h_ev']
            if nm == 'fused':
                ok = ok and _hash_tree(o) == B['warm'][ci]['h_fu']
        ok = ok and B['warm'][ci]['scored_ok']
        QS.burn(burn)
        ta2, oa2 = sync_time(*sa[a_stage])
        if kind == 'E':
            ok = ok and A['h_ev'](oa1) == A['ref'][ci]['h_ev'] and A['h_ev'](oa2) == A['ref'][ci]['h_ev']
        else:
            ok = ok and all(_hash_tree(o_) == A['warm'][ci]['h_fu'] for o_ in (oa1, oa2))
        ok = ok and A['warm'][ci]['scored_ok']
        main = rec.get('t_evolve' if kind == 'E' else 't_fused')
        rec.update(tA1=ta1, tA2=ta2, drift=ta2 / ta1, identity_ok=bool(ok),
                   ratio=(.5 * (ta1 + ta2)) / main if (kind == 'E' and main) else (main / (.5 * (ta1 + ta2)) if main else None))
        tries.append(rec)
        if BAND[0] <= rec['drift'] <= BAND[1]:
            break
    last = dict(tries[-1])
    all_id = all(t_['identity_ok'] for t_ in tries)
    last.update(identity_ok=bool(all_id), valid=bool(BAND[0] <= last['drift'] <= BAND[1] and all_id),
                retries=len(tries) - 1, attempts=tries, A=A['id'], B=B['id'], type=kind)
    return last


def run(cfg, subjects, U0, dev, cases, selector):
    """selector(screen, records) -> list of (X, B) direct pairs (in-job frozen selection, A3.6)."""
    tc = cfg['timing']
    reps, burn = int(tc.get('reps', 3)), float(tc.get('burn', .05))
    if cfg.get('headline'):
        assert reps == 3 and len(dev) == 6, 'A5.8: headline timing needs the six dev6 cases x 3 repetitions'
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
            R = S['ref'].get(ci)
            Wf = S['W_fu'](outs['fused'])
            if R is None:
                sok = False
            elif S['fused_is_scored']:
                sok = bool(F.sha(np.asarray(Wf)) == F.sha(np.asarray(R['W'])))
            else:
                sok = _close(Wf, R['W'])
            w = dict(h_fu=_hash_tree(outs['fused']), scored_fused_ok=sok)
            if 'evolve' in outs and S['h_ev'] is not None:
                w['h_ev_warm'] = S['h_ev'](outs['evolve'])
                w['scored_evolve_ok'] = bool(R is not None and w['h_ev_warm'] == R['h_ev'])
                sok = sok and w['scored_evolve_ok']
            w['scored_ok'] = bool(sok)
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
            x = _rep(('init', 'evolve', 'decode'), subjects[f"L-CNR|{S['r']}"], S, ci, 'E', burn, cache)
        else:
            x = _rep(('fused', 'init', 'evolve', 'decode'), anchor, S, ci, 'F', burn, cache)
        x.update(rep=r_, case=cid(ci))
        recs[kind].append(x)
    devc = [cid(i) for i in dev]
    screen = {sid: FS.case_ratio_stats([t for t in recs['F'] if t['B'] == sid], devc, reps) for sid in subjects}
    pairs = selector({k: (v['value'] if v['status'] == 'ok' else None) for k, v in screen.items()}, recs)
    dorder = [(X, B, ci, r_) for X, B in pairs for ci in dev for r_ in range(reps)]
    rng.shuffle(dorder)
    for X, B, ci, r_ in dorder:
        x = _rep(('fused',), subjects[X], subjects[B], ci, 'D', burn, cache)
        x.update(rep=r_, case=cid(ci), pair=f'{X}~{B}')
        recs['D'].append(x)
    ident = all(t['identity_ok'] for k in recs for t in recs[k])
    warm = {sid: {cid(ci): dict(v) for ci, v in S.get('warm', {}).items()} for sid, S in subjects.items()}
    return dict(records=recs, pairs=pairs, screen=screen, anchor=tc['anchor'], reps=reps, burn=burn, warm=warm,
                status='ok' if ident else 'failed (identity)', headline=bool(cfg.get('headline')),
                dev_cases=devc)
