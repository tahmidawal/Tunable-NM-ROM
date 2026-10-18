"""jcp-dims 3D job driver (DESIGN 2.3, 3, 4, 6.5, A1.2-A1.7, A2.1-A2.4, A3). One job, one GPU, cohort 923801, n = 65.

Uses speed3d's context (s3ctx.Ctx, copied unchanged) after d3common.install(cfg) (explicit test count M per setting,
lane rule names incl. lat262144 and generated draws). Blocks (config 'blocks', run in order):

  ladder    {setting, specs: [spec keys], rungs: [rule names], cases?, tighter_cases?}
            per rung: production rollouts of every spec on every case (speed3d `rollouts`: validity + in-job stationarity
            certificate for fs / qn); ADT anchors (adaptive LSPG, Step 2 `ad`) at level tight on every case and tighter on
            the tighter cases, one per (scheme, dt) of the specs; rho of the rung's own six output states against the next
            rung's table. The Q2 rollout bar is computed offline from consecutive rungs (DESIGN 3). Rule data live: the
            current and the next rung only.
  adaptive  {setting, forms: ['LSPG|CN', 'GAL|CN'], dt, rungs, ...} recording adaptive solves through p1core.make_evolve
            (nl3 / nlJ3, t2core form data), certified in-job (LSPG gamma; square LSPG eta_w, A2.1; GAL eta, qualify A1.3),
            G-cert (A3.3), tight on every case, tighter on the tighter cases (square tolerances for M = r), rho vs the next rung.
  rounds    {members: [{key, qp, qpp}], control?} qualify q3run.member (Q2 of speed3d configurations), unchanged.
  override  {setting_default, setting_override, rule}: the A1.2 independent M-slicing check.
  fixgate   {setting, target, rules, fixture}: component (b) of the cross-family gate (A2.2, A3.2).
  pilot     {items: [{setting, rules}]}: build tables, record peak device bytes.
  panel     {label, A: subject, subjects: [...], cases, reps, seed}: A-B-A timing in blocks (A2.4) with full per-step
            payload identity (A3.1). A subject is {setting, rule, spec} (spec key or 'ad|<form>|<scheme>|<dt>|prod').

    python d3run.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import d3common as D3
import c3common as C3
import s3common as S
import p1core as P1C
import qcert as QC
import dcert as DC
from s3ctx import Ctx
from offmesh import TB

GAMMA = 1e-3
BAND = (1 / 1.1, 1.1)


# ----------------------------------------------------------------------------------------------- identity payload
def payload_hash(o):
    """A3.1: sha256 of a deterministic serialisation of every array of the output pytree (sorted flattened paths; name,
    dtype, shape, bytes)."""
    h = hashlib.sha256()
    leaves = jax.tree_util.tree_leaves_with_path(o)
    items = sorted((jax.tree_util.keystr(p), np.asarray(v)) for p, v in leaves)
    for name, a in items:
        a = np.ascontiguousarray(a)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest(), [n for n, _ in items]


# ----------------------------------------------------------------------------------------------- adaptive forms
class Adaptive:
    """Recording adaptive LSPG / GAL solves on a speed3d setting (A1.6)."""

    def __init__(self, cx):
        self.cx, self.q, self.cert = cx, {}, {}

    def query(self, sname, form, scheme, dt, level):
        st = self.cx.ST[sname]
        square = st.M == st.Rp
        k = (sname, form, scheme, round(dt, 12), level)
        if k not in self.q:
            S_, _ = S.nsteps(dt)
            ev = P1C.make_evolve(C3.nl3, C3.nlJ3, form, st.Rp, st.trust, nrec=S_ + 1, nstep=max(P1C.NSTEP, S_ + 1))
            g, t_ = DC.tol_of(form, level, square=square)
            sch = P1C.sched(scheme, dt, g, t_)

            def run(u0, nu, d):
                w0, s0 = S.project_mesh(u0, d)
                W, stt = ev(w0[0], nu[0], s0[0], d, sch)
                return dict(W=W, w0=w0[0], scale=s0[0], stats=stt)
            self.q[k] = (jax.jit(run), sch, t_, square)
        return self.q[k]

    def certifier(self, form, Rp):
        k = (form, Rp)
        if k not in self.cert:
            self.cert[k] = P1C.make_certifier(C3.nl3, Rp) if form == 'LSPG' else QC.make_gal_certifier(C3.nl3, Rp)
        return self.cert[k]


def form_data(st, rule, form):
    return P1C.form_data(st.data(rule), {}, form)


def ad_valid(o):
    """Validity of a recording adaptive run (p1common.run_valid semantics): finite outputs and states, no reason-3 step."""
    st = o['stats']
    k = int(st['steps'])
    ex = np.asarray(st['exits'])
    ok = bool(np.isfinite(np.asarray(o['W'])).all() and np.isfinite(np.asarray(st['states'])).all() and ex[3] == 0)
    return ok and bool(np.isfinite(np.asarray(st['step_gn'])[:k]).all())


def run_adaptive(cx, ad, blk, out):
    """Adaptive-form block: rungs ascending, rho vs the next rung, Q1 evidence in-job."""
    sname = blk['setting']
    st = cx.ST[sname]
    cases = blk.get('cases') or cx.cases
    tcs = [j for j in blk.get('tighter_cases', list(range(8))) if j in cases]
    dt = float(blk['dt'])
    S_, _ = S.nsteps(dt)
    res = {}
    rungs = blk['rungs']
    for i, rule in enumerate(rungs):
        nxt = rungs[i + 1] if i + 1 < len(rungs) else None
        for fs_ in blk['forms']:
            form, scheme = fs_.split('|')
            d = form_data(st, rule, form)
            fq, sch, tolf, square = ad.query(sname, form, scheme, dt, 'prod')
            cert = ad.certifier(form, st.Rp)
            n = len(cases)
            R = dict(W=np.full((n, 6, st.Rp), np.nan), states=np.full((n, S_ + 1, st.Rp), np.nan),
                     cert=np.full((n, S_), np.nan), gamma_lit=np.full((n, S_), np.nan), raw=np.full((n, S_), np.nan),
                     scale=np.full((n, S_), np.nan), tolratio=np.full((n, S_), np.nan), it=np.full((n, S_), -1),
                     reason=np.full((n, S_), -1), valid=np.zeros(n, bool), gcert=np.zeros(n, bool),
                     noncert=np.zeros(n, int), W_tight=np.full((n, 6, st.Rp), np.nan), valid_tight=np.zeros(n, bool),
                     W_tighter=np.full((len(tcs), 6, st.Rp), np.nan), valid_tighter=np.zeros(len(tcs), bool),
                     rho=np.full((n, 6), np.nan), secs=np.full(n, np.nan), payload=np.array([''] * n, dtype=object))
            for ci, j in enumerate(cases):
                t1 = time.perf_counter()
                o = jax.block_until_ready(fq(cx.U0[j], cx.NU[j], d))
                R['secs'][ci] = time.perf_counter() - t1
                stt = o['stats']
                nu = cx.NU[j][0]
                R['W'][ci], R['states'][ci] = np.asarray(o['W']), np.asarray(stt['states'])
                R['it'][ci], R['reason'][ci] = (np.asarray(stt[k])[:S_] for k in ('step_it', 'step_reason'))
                tr = np.asarray(stt['step_tolratio'])[:S_]
                R['tolratio'][ci] = tr
                c = cert(stt['states'], nu, d, sch)
                scl = float(o['scale'])
                if form == 'LSPG' and square:
                    g, rn = np.asarray(c[0]), np.asarray(c[1])
                    R['gamma_lit'][ci], R['raw'][ci], R['scale'][ci] = g, rn, scl
                    R['cert'][ci] = DC.eta_w(rn, scl)
                    R['gcert'][ci] = DC.gcert(rn, tr, tolf * scl)
                    R['noncert'][ci] = int(np.sum(~DC.certified_square(rn, scl)))
                elif form == 'LSPG':
                    g = np.asarray(c[0])
                    rec = np.asarray(stt['step_gn'])[:S_]
                    R['cert'][ci] = R['gamma_lit'][ci] = g
                    R['gcert'][ci] = bool(np.all(np.abs(g - rec) <= 1e-7 * np.abs(rec) + 1e-12) and np.isfinite(g).all())
                    R['noncert'][ci] = int(np.sum(~(g <= GAMMA)))
                else:
                    eta, gam, _, rn, sc_ = (np.asarray(x) for x in c)
                    R['cert'][ci], R['gamma_lit'][ci], R['raw'][ci], R['scale'][ci] = eta, gam, rn, sc_
                    R['gcert'][ci] = DC.gcert(rn, tr, tolf * sc_)
                    R['noncert'][ci] = int(np.sum(~QC.certified_gal(eta)))
                R['valid'][ci] = ad_valid(o)
                R['payload'][ci] = payload_hash(o)[0]
                ft, _, _, _ = ad.query(sname, form, scheme, dt, 'tight')
                ot = jax.block_until_ready(ft(cx.U0[j], cx.NU[j], d))
                R['W_tight'][ci], R['valid_tight'][ci] = np.asarray(ot['W']), ad_valid(ot)
                if j in tcs:
                    ftt, _, _, _ = ad.query(sname, form, scheme, dt, 'tighter')
                    ott = jax.block_until_ready(ftt(cx.U0[j], cx.NU[j], d))
                    R['W_tighter'][tcs.index(j)], R['valid_tighter'][tcs.index(j)] = np.asarray(ott['W']), ad_valid(ott)
                cx.acc_sha[(f'{sname}|{rule}', f'ad|{form}|{scheme}|{dt:.8g}|prod', j)] = R['payload'][ci]
            if nxt is not None:
                dn = st.data(nxt)
                for ci in range(len(cases)):
                    for kk in range(6):
                        c_ = jnp.asarray(R['W'][ci, kk])
                        R['rho'][ci, kk] = P1rho(C3.nl3(c_, st.data(rule)), C3.nl3(c_, dn))
            R['dist_tight'] = cx.dist_cases(sname, R['W'], R['W_tight'], cases)[:, 1:].max(1)
            idx = [cases.index(j) for j in tcs]
            R['dist_tight_tighter'] = cx.dist_cases(sname, R['W_tight'][idx], R['W_tighter'], tcs)[:, 1:].max(1)
            key = f'{sname}|{rule}|ad|{form}|{scheme}|{dt:.8g}'
            np.savez_compressed(out / f"AD_{key.replace('|', '_')}.npz", case_ids=np.array(cases), tighter_ids=np.array(tcs),
                                rule=rule, next_rule=str(nxt), M=st.M, Rp=st.Rp, square=square, form=form, scheme=scheme,
                                dt=dt, mode_hash=st.mode_hash, tol_prod=np.array(DC.tol_of(form, 'prod', square)),
                                **{k: v for k, v in R.items() if k != 'payload'}, payload=R['payload'].astype(str))
            res[key] = dict(valid=bool(R['valid'].all()), noncert=int(R['noncert'].sum()), steps=int(len(cases) * S_),
                            gcert=bool(R['gcert'].all()), worst_dist_tight=float(np.max(R['dist_tight'])),
                            worst_tight_tighter=float(np.max(R['dist_tight_tighter'])) if len(tcs) else None,
                            rho_max=float(np.nanmax(R['rho'])) if nxt else None, M=st.M, square=square,
                            rule_sha256=st._d[rule][1]['sha256'])
            print('AD', key, res[key], cx.el(), flush=True)
            del d
        if i >= 1:
            st.drop(rungs[i - 1])
        gc.collect()
    st.drop(rungs[-1])
    gc.collect()
    return res


def P1rho(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float('inf')
    nb = float(np.linalg.norm(b))
    if nb == 0.:
        return float('nan') if float(np.linalg.norm(a)) == 0. else float('inf')
    return float(np.linalg.norm(a - b) / nb)


# ----------------------------------------------------------------------------------------------- ladder (fs / qn)
def run_ladder(cx, blk):
    sname = blk['setting']
    st = cx.ST[sname]
    cases = blk.get('cases') or cx.cases
    tcs = [j for j in blk.get('tighter_cases', list(range(8))) if j in cases]
    specs = [S.parse_key(k) for k in blk['specs']]
    rungs = blk['rungs']
    res = {}
    for i, rule in enumerate(rungs):
        nxt = rungs[i + 1] if i + 1 < len(rungs) else None
        tag = f'{sname}|{rule}'
        r = cx.rollouts(sname, rule, specs, cases, tag=tag)
        adt = {}
        for sp in specs:
            k_ad = (sp[1], round(sp[2], 12))
            if k_ad not in adt:
                at = ('ad', sp[1], sp[2], 'tight')
                att = ('ad', sp[1], sp[2], 'tighter')
                ra = cx.rollouts(sname, rule, [at], cases, tag=tag)[S.spec_key(at)]
                rb = cx.rollouts(sname, rule, [att], tcs, tag=tag)[S.spec_key(att)] if tcs else None
                idx = [cases.index(j) for j in tcs]
                adt[k_ad] = dict(W=ra['W'], valid=ra['valid'],
                                 tt=(cx.dist_cases(sname, ra['W'][idx], rb['W'], tcs)[:, 1:].max(1).tolist() if rb else []),
                                 valid_tighter=(rb['valid'] if rb else []))
        dn = st.data(nxt) if nxt else None
        for sp in specs:
            key = S.spec_key(sp)
            a = adt[(sp[1], round(sp[2], 12))]
            W = r[key]['W']
            rho = np.full((len(cases), 6), np.nan)
            if dn is not None:
                for ci in range(len(cases)):
                    for kk in range(6):
                        c_ = jnp.asarray(W[ci, kk])
                        rho[ci, kk] = P1rho(C3.nl3(c_, st.data(rule)), C3.nl3(c_, dn))
            res[f'{tag}|{key}'] = dict(setting=sname, rule=rule, next_rule=nxt, spec=key, M=st.M, Rp=st.Rp,
                                       valid=[bool(x) for x in r[key]['valid']], nonstationary=int(sum(r[key]['nonst'])),
                                       steps=int(sum(r[key]['steps'])),
                                       dist_adt=cx.dist_cases(sname, W, a['W'], cases)[:, 1:].max(1).tolist(),
                                       adt_valid=[bool(x) for x in a['valid']], adt_tight_tighter=a['tt'],
                                       adt_tighter_valid=[bool(x) for x in a['valid_tighter']], rho=rho.tolist(),
                                       cases=list(cases), tighter_cases=tcs, rule_sha256=st._d[rule][1]['sha256'],
                                       mode_hash=st.mode_hash)
            print('LADDER', f'{tag}|{key}', 'valid', all(res[f'{tag}|{key}']['valid']),
                  'worst dist_adt', max(res[f'{tag}|{key}']['dist_adt']), 'rho', np.nanmax(rho) if dn is not None else None,
                  cx.el(), flush=True)
        if i >= 1:
            st.drop(rungs[i - 1])
        gc.collect()
        cx.save()
        cx.dump()
    st.drop(rungs[-1])
    gc.collect()
    return res


# ----------------------------------------------------------------------------------------------- override check
def run_override(cx, blk):
    """A1.2: the override's A, lam, P and N(c) equal the slices of the default setting's (same rule), and its mode hash
    equals an independently computed hash of C.modes(65, M)."""
    sd, so, rule = cx.ST[blk['setting_default']], cx.ST[blk['setting_override']], blk['rule']
    dd, do = sd.data(rule), so.data(rule)
    M = so.M
    z = np.load(Path(__file__).resolve().parent / blk['fixture'])
    C = np.asarray(z['W'])[:, :so.Rp]
    rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / (np.max(np.abs(np.asarray(b))) + 1e-300))
    out = dict(M_default=sd.M, M_override=M, shapes=dict(A=list(np.asarray(do['A']).shape), P=list(np.asarray(do['P']).shape)),
               A=rel(do['A'], np.asarray(dd['A'])[:M]), lam=rel(do['lam'], np.asarray(dd['lam'])[:M]),
               P=rel(do['P'], np.asarray(dd['P'])[:, :M]),
               N=max(rel(C3.nl3(jnp.asarray(c), do), np.asarray(C3.nl3(jnp.asarray(c), dd))[:M]) for c in C))
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent / 'vendor' / 'quad3d'))
    import npcore as N3                     # independent NumPy mode order (quadrature-burgers3d npcore.modes)
    k_ind, _ = N3.modes(65, M)
    out['mode_hash_ok'] = bool(so.mode_hash == C3.sha(np.asarray(k_ind, np.int64)[:M]))
    out['shapes_ok'] = bool(out['shapes']['A'] == [M, so.Rp] and out['shapes']['P'][1] == M)
    out['passed'] = bool(out['shapes_ok'] and out['mode_hash_ok'] and out['A'] <= 1e-13 and out['lam'] <= 1e-13
                         and out['P'] <= 1e-13 and out['N'] <= 1e-12)
    print('OVERRIDE', out, flush=True)
    sd.drop(rule)
    so.drop(rule)
    return out


# ----------------------------------------------------------------------------------------------- fixgate (b)
def run_fixgate(cx, blk):
    st = cx.ST[blk['setting']]
    z = np.load(Path(__file__).resolve().parent / blk['fixture'])
    C = np.asarray(z['W'])[:, :st.Rp]
    Nt = [np.asarray(C3.nl3(jnp.asarray(c), st.data(blk['target']))) for c in C]
    out = dict(setting=blk['setting'], M=st.M, target=blk['target'], fixture=blk['fixture'], fixture_source=str(z['source']),
               rules={})
    for rl in blk['rules']:
        Nr = [np.asarray(C3.nl3(jnp.asarray(c), st.data(rl))) for c in C]
        rh = [P1rho(a, b) for a, b in zip(Nr, Nt)]
        out['rules'][rl] = dict(rho_tgt=rh, flags=bool(any((not np.isfinite(x)) or x > 0.116 for x in rh)),
                                sha256=st._d[rl][1]['sha256'])
        print('FIXGATE3', rl, max(rh), flush=True)
        if rl != blk['target']:
            st.drop(rl)
        gc.collect()
    st.drop(blk['target'])
    return out


# ----------------------------------------------------------------------------------------------- timing panels
def subject(cx, ad, sp_):
    """{name, run(j) -> output pytree, data builder}; rule data built at block start (A2.4)."""
    sname, rule, key = sp_['setting'], sp_['rule'], sp_['spec']
    st = cx.ST[sname]
    if key.startswith('ad|'):
        _, form, scheme, dt, lev = key.split('|')
        fq, _, _, _ = ad.query(sname, form, scheme, float(dt), lev)
        d = form_data(st, rule, form)
        run = lambda j: fq(cx.U0[j], cx.NU[j], d)
    else:
        f = cx.q_of(sname, S.parse_key(key), 'mesh', 65)
        d = st.data(rule)
        run = lambda j: f(cx.U0[j], cx.NU[j], d)
    name = f'{sname}|{rule}|{key}'
    if sp_.get('double'):                 # speed3d T1 control: the same query twice, each synchronised (ratio in [1.8, 2.2])
        one_ = run
        run = lambda j: (jax.block_until_ready(one_(j)), jax.block_until_ready(one_(j)))
        name = 'T1|' + name
    return dict(name=name, run=run, setting=sname, rule=rule)


def run_panel(cx, ad, blk):
    """A-B-A in blocks (A2.4): A persistent; per (setting, rule) block the subjects are built, scored untimed (payload
    hash = expected), warmed, timed in randomised order, then released. Invocation counts validated."""
    burn = cx.cfg['timing']['burn']
    cases = list(range(int(blk['cases'])))
    reps = int(blk['reps'])
    rng = np.random.default_rng(int(blk['seed']))
    A = subject(cx, ad, blk['A'])
    expA = {ci: payload_hash(jax.block_until_ready(A['run'](cx.cases[ci])))[0] for ci in cases}

    def one(x, ci, expect):
        C3.burn(burn)
        t1 = time.perf_counter()
        o = jax.block_until_ready(x['run'](cx.cases[ci]))
        secs = time.perf_counter() - t1
        h = payload_hash(o)[0]
        del o
        return secs, bool(h == expect)
    for ci in cases:
        one(A, ci, expA[ci])
    blocks = {}
    for sp_ in blk['subjects']:
        blocks.setdefault((sp_['setting'], sp_['rule']), []).append(sp_)
    border = list(blocks)
    rng.shuffle(border)
    inv, base = [], C3.peak_bytes()
    mem0 = jax.devices()[0].memory_stats().get('bytes_in_use', -1)
    blog = []
    for bk in border:
        Bs = [subject(cx, ad, sp_) for sp_ in blocks[bk]]
        exp, scored = {}, {}
        for x in Bs:
            for ci in cases:
                o = jax.block_until_ready(x['run'](cx.cases[ci]))
                exp[(x['name'], ci)], names = payload_hash(o)
                scored[(x['name'], ci)] = scored_equals_certified(cx, x, ci, o, exp[(x['name'], ci)])
                del o
        cx.rep.setdefault('scored_equals_certified', {}).update({f'{k[0]}#{cx.cases[k[1]]}': v for k, v in scored.items()})
        for x in Bs:                                  # warm
            for ci in cases:
                one(x, ci, exp[(x['name'], ci)])
        order_ = [(b, ci, rr) for b in range(len(Bs)) for ci in cases for rr in range(reps)]
        rng.shuffle(order_)
        for b, ci, rr in order_:
            ta1, h1 = one(A, ci, expA[ci])
            tb, hb = one(Bs[b], ci, exp[(Bs[b]['name'], ci)])
            ta2, h2 = one(A, ci, expA[ci])
            inv.append(dict(B=Bs[b]['name'], case=cx.cases[ci], rep=rr, tA1=ta1, tB=tb, tA2=ta2,
                            ratio=tb / (.5 * (ta1 + ta2)), drift=ta2 / ta1, hashes_ok=bool(h1 and hb and h2)))
        cx.rep.setdefault('payload_keys', {})[bk[0] + '|' + bk[1]] = names
        for x in Bs:
            cx.ST[x['setting']].drop(x['rule'])
        del Bs
        for k_ in [k_ for k_ in list(ad.q) if k_[0] == bk[0]]:
            pass                                       # compiled adaptive queries hold no data (data are arguments)
        gc.collect()
        m_ = jax.devices()[0].memory_stats().get('bytes_in_use', -1)
        blog.append(dict(block=list(bk), in_use=m_, over_start=m_ - mem0))
        if m_ - mem0 > 1e9:
            cx.rep['timing'][blk['label']] = dict(invocations=inv, blocks=blog, valid=False, error='memory not released')
            cx.save()
            raise SystemExit('memory not released after block %s' % (bk,))
    import timing as TM
    cnt = {}
    for x in inv:
        cnt[x['B']] = cnt.get(x['B'], 0) + 1
    want = len(cases) * reps
    complete = all(cnt.get(('T1|' if s.get('double') else '') + f"{s['setting']}|{s['rule']}|{s['spec']}", 0) == want
                   for s in blk['subjects'])
    v = TM.panel_validity(inv)
    cx.rep['timing'][blk['label']] = dict(A=A['name'], subjects=sorted(cnt), invocations=inv, cases=[cx.cases[c] for c in cases],
                                          reps=reps, burn=burn, seed=blk['seed'], blocks=blog, counts=cnt,
                                          counts_complete=complete, peak_before=base, **v,
                                          valid_all=bool(v['valid'] and complete))
    cx.save()
    print('PANEL', blk['label'], len(inv), 'valid', v['valid'], 'complete', complete, cx.el(), flush=True)


def scored_equals_certified(cx, x, ci, o, payload):
    """A1.1 (3D): the panel's untimed scoring run must equal the run certified earlier in the same job: fs / qn -
    bitwise equal six output vectors with the `rollouts` archive (Wstore '<setting>|<rule>|<spec>'); adaptive - equal
    payload hash with the adaptive block's record. None if the subject was not certified in this job."""
    if x['name'].startswith('T1|'):
        return None
    j = cx.cases[ci]
    tag, key = f"{x['setting']}|{x['rule']}", x['name'].replace('T1|', '', 1).split('|', 2)[2]
    if key.startswith('ad|'):
        p = cx.acc_sha.get((tag, key, j))
        return None if p is None else bool(p == payload)
    W = cx.Wstore.get(f'{tag}|{key}')
    if W is None:
        return None
    ids = list(cx.Cstore[f'{tag}|{key}'])
    if j not in ids:
        return None
    return bool(np.array_equal(np.asarray(o['W'])[:, 0], W[ids.index(j)]))


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    assert int(cfg['cohort_seed']) == 923801, 'jcp-dims 3D runs on the validation cohort only'
    D3.install(cfg)
    cx = Ctx(cfg, a.out)
    cx.rep['peak_after_setup'] = C3.peak_bytes()
    cx.rep['settings_M'] = {k: dict(M=v.M, Rp=v.Rp, mode_hash=v.mode_hash) for k, v in cx.ST.items()}
    cx.rep['blocks'] = []
    ad = Adaptive(cx)
    import q3run
    for blk in cfg['blocks']:
        t0 = time.perf_counter()
        kind = blk['kind']
        if kind == 'ladder':
            r = run_ladder(cx, blk)
        elif kind == 'adaptive':
            r = run_adaptive(cx, ad, blk, cx.out)
        elif kind == 'rounds':
            r = {}
            qpp_cases = blk.get('qpp_cases', list(range(8)))
            for m in blk['members']:
                r[m['key']] = q3run.member(cx, m, list(cx.cases), qpp_cases, 'member')
                for rl in (m['key'].split('|')[1], m['qp'], m.get('qpp')):
                    if rl:
                        cx.ST[m['key'].split('|')[0]].drop(rl)
                gc.collect()
                cx.save()
                cx.dump()
            if blk.get('control'):
                r['control|' + blk['control']['key']] = q3run.member(cx, blk['control'], list(cx.cases), [], 'control')
        elif kind == 'override':
            r = run_override(cx, blk)
        elif kind == 'fixgate':
            r = run_fixgate(cx, blk)
        elif kind == 'pilot':
            r = []
            for it in blk['items']:
                st = cx.ST[it['setting']]
                for rl in it['rules']:
                    st.data(rl)
                    for f_ in it.get('forms', []):
                        form_data(st, rl, f_)
                r.append(dict(it, peak_bytes=C3.peak_bytes()))
                for rl in it['rules']:
                    st.drop(rl)
                gc.collect()
        elif kind == 'panel':
            run_panel(cx, ad, blk)
            r = dict(label=blk['label'])
        else:
            raise KeyError(kind)
        cx.rep['blocks'].append(dict(kind=kind, label=blk.get('label', blk.get('setting')), seconds=time.perf_counter() - t0,
                                     peak_bytes=C3.peak_bytes(), result=r))
        cx.save()
        cx.dump()
    cx.rep['peak_bytes_job'] = C3.peak_bytes()
    cx.rep['elapsed_seconds'] = cx.el()
    cx.rep['complete'] = True
    cx.save()
    cx.dump()
    (cx.out / 'COMPLETE').write_text('complete\n')
    print('D3RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
