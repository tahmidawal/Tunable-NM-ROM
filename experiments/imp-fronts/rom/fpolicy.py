"""imp-fronts §B5 policy driver: every round of the adaptive rule / step policy inside ONE job per bank (DESIGN B5 as
replaced by Amendments 2-3, B5/B8/B9 of Amendments 4-5; B2 C-B1, C-B6 as amended). Uses the byte-identical r2acc
closure in this directory for every solve (r2acc.run / steps_of / rule_identity, p1common.Setting / Bank / fdist /
run_valid / gcert_ok / rho, p1core.make_certifier); the state machine is policy_sm.py (pure, tests/test_policy.py).

Solves per visited state (rule, f) [r = Rp, LSPG, scheme CNR, mesh `mesh`, dt = DT0 * f] (Amendment 5 Q1 scope):
  prod   (rule, f)               recorded; in-job per-step certificate (fresh-Jacobian gamma, G-cert vs the solver's
                                 record, non-stationary count)
  tight  (rule, f) tight tol.    Q1 tight-solve distance of prod (a promoted half gets its tight solve when visited)
  succ   (q2_successor[rule], f) recorded + per-step certificate, replayed from prod's own initial state (the same IC
                                 fit); dist_succ = rollout distance, rho_succ_max = max over the six outputs of
                                 rho(N_rule(c), N_succ(c)) at prod's own reached coefficient states (as r2acc)
  half   (rule, f / 2), f > .25  recorded + per-step certificate, no tight; dist_half = rollout distance to prod
After the loop: one tighter solve per anchor case at its terminal (scored) state -> dist_tight_tighter.
Q1 of a scored prod: valid, G-cert, tight valid, dist_tight <= 2.5e-4, (anchor) tighter valid and tight/tighter <= 2.5e-5;
pooled non-stationary fraction over the scored prods <= 1 % (configuration-level). qualified = CONVERGED and Q1.
Self-timer (Amendment 4 B8): `ceiling_hours` is the job's wall limit (= its ceiling GPU-h on one GPU); no solve is
started after `stop_fraction` (0.9) of it, nor one whose estimated end passes that point; open cases -> BUDGET-CAP
(scored at their latest valid prod). progress.json is rewritten after every round and every solve record, so
fcollect.py can write the exhaustion records if the job is killed. Every solve is charged (rows).
Outputs (out dir): result.json, progress.json, runs/<run>.npz (W [6, Rp], w0, states, cert, ... of every solve),
CB6.json (if configured), COMPLETE.

Config:
  {"attempt", "driver": "fpolicy", "mesh", "edges", "Rp": 384, "scheme": "CNR",
   "bank": {"dir": "inputs/banks/s1"} | {"ckpt", "ckpt_sha256", "rotation", "rotation_sha256"}, "bank_label": "1",
   "selected": "gauss160" | null,                  (null -> uncertified start at fib4181)
   "cases": ["FC1|0", ...], "table": "phaseb/fresh_cohorts.json" (lane-relative), "test_once": false,
   "anchors": null | ["FC1|0", ...]               (null -> the frozen fresh anchors FI|4, FI|5, FC1|4, FC1|5),
   "ceiling_hours": 0.4, "stop_fraction": 0.9, "max_rounds": 10,   (max_rounds < 10 only in smoke attempts 'smk*')
   "controls": {"cb1": true, "cb6": {"case": "FC1|0", "step": 10}},
   "allow_cpu_smoke": false}

    python fpolicy.py --config <cfg.json> --out <dir> [--t0-epoch <job start, s since epoch>] [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
ROOT = HERE.parents[2]
for _p in (ROOT / 'experiments' / 'separable-decoder', ROOT / 'experiments' / 'mr-burgers2d'):   # see PROVENANCE.json note
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import policy_sm as SM  # noqa: E402
import bankguard as BG  # noqa: E402
import fcases as FC  # noqa: E402
import schedule as SCH  # noqa: E402
import testonce as TO  # noqa: E402

GAMMA, BAR_Q1, BAR_TT, NONST_POOLED = 1e-3, 2.5e-4, 2.5e-5, .01


def safe(k):
    return k.replace('|', '__').replace('.', 'p')


def m_of(rule):
    import rules2d
    return rules2d.size_of(rule)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def dump(obj, p):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=1))
    os.replace(tmp, p)


def q1_of(store, key, anchor):
    """Q1 of a scored prod (cert record) from its own record, its tight record and (anchors) the tighter record."""
    c = store.get(key) or {}
    t = store.get(key.replace('|cert|', '|tight|')) or {}
    ok = bool(c.get('valid') is True and c.get('gcert_ok') and t.get('valid') is True and SM.finite(t.get('dist_tight'))
              and t['dist_tight'] <= BAR_Q1)
    if anchor:
        tt = store.get(key.replace('|cert|', '|tighter|')) or {}
        ok = ok and bool(tt.get('valid') is True and SM.finite(tt.get('dist_tight_tighter')) and tt['dist_tight_tighter'] <= BAR_TT)
    return ok


class Executor:
    """Runs policy requests with the r2acc closure; keeps per-run records, fields and coefficients in host memory."""

    def __init__(self, cfg, out, cases, begin):
        import jax
        import jax.numpy as jnp
        import p1common as P
        import p1core as T2
        import r2acc as RA
        self.jax, self.jnp, self.P, self.T2, self.RA = jax, jnp, P, T2, RA
        self.cfg, self.out, self.begin = cfg, out, begin
        self.L, self.Rp, self.scheme = int(cfg['mesh']), int(cfg.get('Rp', 384)), cfg.get('scheme', 'CNR')
        self.stop_at = float(cfg.get('ceiling_hours', .4)) * 3600. * float(cfg.get('stop_fraction', .9))
        self.stopped, self.rows, self.fields, self.W, self.prev = False, [], {}, {}, {}
        self.rate, self.shapes, self.progress = {}, set(), None
        self.cases = {c[0]: c for c in cases}
        self.anchors = set(FC.anchors_in(list(self.cases), cfg.get('anchors')))
        self.U0 = {}
        s256 = self.L // 256
        for cid, coh, j, desc, sp in cases:
            u0n, nu, n0 = FC.initial(self.L, desc)
            assert np.array_equal(u0n[::s256, ::s256], FC.initial(256, desc)[0]), 'lattice IC differs from the 256 IC'
            self.U0[cid] = (jnp.asarray(u0n), nu, n0)

    def setup(self, bank_cfg):
        P, T2 = self.P, self.T2
        t0 = time.perf_counter()
        self.bank = P.Bank(self.L, self.cfg['edges'], nrb=self.cfg.get('bank_blocks'), ckpt=HERE / bank_cfg['ckpt'],
                           rotation=HERE / bank_cfg['rotation'])
        self.st = P.Setting(self.bank, 'POL', self.Rp, 'fib4181', 'frozen', roles={}, forms=('LSPG',), nrec=0)
        self.nl = self.st.q['LSPG']['nl']
        self.cert = T2.make_certifier(self.nl, self.Rp)
        return dict(seconds=time.perf_counter() - t0, bank_bytes=self.bank.bytes, mode_hash=self.st.mode_hash,
                    M=self.st.M, mem=P.mem())

    def elapsed(self):
        return time.perf_counter() - self.begin

    def over_budget(self):
        return self.stopped or self.elapsed() >= self.stop_at

    def load(self, rule, with_succ):
        self.st.add_role(self.bank, 'd', rule, forms=('LSPG',))
        self.RA.rule_identity(self.st, 'd', rule)
        if with_succ:
            self.st.add_role(self.bank, 'ds', SM.Q2_SUCCESSOR[rule], forms=('LSPG',))
            self.RA.rule_identity(self.st, 'ds', SM.Q2_SUCCESSOR[rule])

    def unload(self):
        for k in ('d', 'ds'):
            self.st.drop_role(k)
            self.st.meta.pop(k, None)
            self.st.rules.pop(k, None)
        self.RA.QQ.c.clear()
        gc.collect()

    def solve(self, role, rule, f, level, cid, record, kind):
        """One charged solve. Returns (v, fields, seconds) or None when the self-timer forbids starting it."""
        S = self.RA.steps_of(self.P.DT0 * f)
        m = m_of(rule)
        r_ = self.rate.get((level, bool(record)))
        if self.elapsed() + (0. if r_ is None else r_ * m * S) >= self.stop_at:
            self.stopped = True
            return None
        u0, nu, n0 = self.U0[cid]
        shape = (rule, f, bool(record))
        first = shape not in self.shapes
        t1 = time.perf_counter()
        v = self.RA.run(self.bank, self.st, role, 'LSPG', self.scheme, self.P.DT0 * f, level, u0, nu, record=record)
        self.jax.block_until_ready(v)
        secs = time.perf_counter() - t1
        self.shapes.add(shape)
        k = (level, bool(record))
        if not first or k not in self.rate:
            self.rate[k] = max(self.rate.get(k, 0.), secs / (m * S))
        it = np.asarray(v['stats']['step_it'])[:S]
        self.rows.append(dict(case=cid, kind=kind, level=level, rule=rule, m=m, factor=f, steps=S, recorded=bool(record),
                              seconds=secs, first_of_shape=first, elapsed=self.elapsed(), valid=self.P.run_valid(v),
                              it_mean=float(np.mean(it)), it_max=int(np.max(it))))
        return v, self.P.restrict(v['fields'], self.L), secs

    def certified(self, q, role):
        """Recorded solve + the in-job per-step certificate (r2acc's LSPG branch). Returns (record, R arrays, fields)."""
        P = self.P
        cid, rule, f = q['cid'], q['rule'], q['f']
        u0, nu, n0 = self.U0[cid]
        r = self.solve(role, rule, f, 'prod', cid, True, q['kind'])
        if r is None:
            return None
        v, fl, secs = r
        S = self.RA.steps_of(P.DT0 * f)
        sch = self.T2.sched(self.scheme, P.DT0 * f, *P.tol_of('LSPG', 'prod'))
        stt = v['stats']
        gm, _ = self.cert(stt['states'], nu, self.st.data[(role, 'LSPG')], sch)
        gm, rec_g = np.asarray(gm), np.asarray(stt['step_gn'])[:S]
        R = dict(W=np.asarray(v['W']), w0=np.asarray(v['w0']), n0=n0, states=np.asarray(stt['states']), cert=gm, rec=rec_g,
                 it=np.asarray(stt['step_it'])[:S], reason=np.asarray(stt['step_reason'])[:S])
        rec = dict(kind=q['kind'], rule=rule, factor=f, dt=P.DT0 * f, steps=S, valid=bool(P.run_valid(v)),
                   gcert_ok=bool(P.gcert_ok(gm, rec_g) and np.isfinite(gm).all()), noncertified=int(np.sum(~(gm <= GAMMA))),
                   budget_exits=P.budget_exits(v), it_sum=int(stt['it_sum']), seconds=secs,
                   rule_sha256=self.st.meta[role]['rule_sha256'], W_sha=P.sha(v['W']))
        self.last_states = (q['key'], R['states'], sch, nu)
        del v
        return rec, R, fl

    def save_npz(self, key, **arrs):
        np.savez_compressed(self.out / 'runs' / f'{safe(key)}.npz', key=key, mesh=self.L, Rp=self.Rp, scheme=self.scheme,
                            form='LSPG', time_tag='t0,t1,t2,t3,t4,t5', **arrs)

    def cert_run(self, q):                     # prod or half
        r = self.certified(q, 'd')
        if r is None:
            return None
        rec, R, fl = r
        if q['kind'] == 'half':
            pf = self.fields.get(q['prod_key'])
            rec.update(dist_half=self.P.fdist(pf, fl, R['n0']) if pf is not None else None, half_of=q['prod_key'])
        self.fields[q['key']], self.W[q['key']] = fl, R['W']
        self.save_npz(q['key'], level='prod', **R)
        return rec

    def tight_run(self, q, level='tight', ref_key=None):
        P = self.P
        cid, rule, f = q['cid'], q['rule'], q['f']
        ref_key = ref_key or q['prod_key']
        rt = self.solve('d', rule, f, level, cid, False, level)
        if rt is None:
            return None
        vt, ft, secs = rt
        n0 = self.U0[cid][2]
        d = P.fdist(self.fields[ref_key], ft, n0)
        rec = dict(kind=level, rule=rule, factor=f, valid=bool(P.run_valid(vt)), seconds=secs, of=ref_key,
                   **({'dist_tight': d} if level == 'tight' else {'dist_tight_tighter': d}))
        self.fields[q['key']] = ft
        self.save_npz(q['key'], level=level, W=np.asarray(vt['W']))
        del vt
        return rec

    def succ_run(self, q):
        P, jnp = self.P, self.jnp
        r = self.certified(q, 'ds')
        if r is None:
            return None
        rec, R, fq = r
        Wp = self.W[q['prod_key']]
        rho = np.full(6, np.nan)
        for kk in range(6):                    # rho at prod's own reached states (r2acc)
            c = jnp.asarray(Wp[kk])
            rho[kk] = P.rho(np.asarray(self.nl(c, self.st.data[('d', 'LSPG')])), np.asarray(self.nl(c, self.st.data[('ds', 'LSPG')])))
        rec.update(dist_succ=P.fdist(self.fields[q['prod_key']], fq, R['n0']), rho_succ=rho.tolist(),
                   rho_succ_max=(float(np.max(rho)) if np.isfinite(rho).all() else float('inf')), succ_of=q['prod_key'])
        self.save_npz(q['key'], level='prod', rho_succ=rho, **R)
        return rec

    def __call__(self, reqs, controls=None):
        recs = {}
        for rule in dict.fromkeys(q['state_rule'] for q in reqs):
            if self.over_budget():
                break
            grp = [q for q in reqs if q['state_rule'] == rule]
            self.load(rule, any(q['kind'] == 'succ' for q in grp))
            for q in grp:                      # state order: prod, tight, succ, half (policy_sm.needed)
                if self.over_budget():
                    break
                if q['kind'] != 'prod' and not (recs.get(q['prod_key']) or self.prev.get(q['prod_key']) or {}).get('valid'):
                    # precedence (1): the state is INVALID whatever the other runs say; nothing else is run for it
                    recs[q['key']] = dict(kind=q['kind'], valid=None, skipped='prod missing or invalid')
                    continue
                fn = dict(prod=self.cert_run, half=self.cert_run, succ=self.succ_run, tight=self.tight_run)[q['kind']]
                rec = fn(q)
                if rec is None:
                    break
                recs[q['key']] = rec
                self.prev[q['key']] = rec
                if controls is not None:
                    controls(q, rec, self)
                if self.progress:
                    self.progress()
            self.unload()
        self.prev.update(recs)
        return recs

    def tighter(self, cases, store):
        """One tighter solve per anchor case at its terminal (scored) state (Amendment 5)."""
        for c in cases:
            if c.cid not in self.anchors or not c.scored or self.over_budget():
                continue
            _, cid, _, rule, f = c.scored.split('|')
            f = float(f)
            tk = c.scored.replace('|cert|', '|tight|')
            if tk not in self.fields:
                continue
            self.load(rule, False)
            q = dict(key=c.scored.replace('|cert|', '|tighter|'), cid=c.cid, kind='tighter', rule=rule, f=f)
            rec = self.tight_run(q, 'tighter', ref_key=tk)
            self.unload()
            if rec is not None:
                store[q['key']] = rec


# ------------------------------------------------------------------------------------- C-B6 fixture (Amendment 3)
def cb6_search(ex, key, step):
    """Perturb the accepted state of `step` (1-based) of run `key` along the top right singular vector of the fresh Jacobian;
    eps in {1e-3, 1e-2, 1e-1} * ||c||, signs + then -; at each candidate the ACTUAL residual and a FRESH Jacobian give gamma;
    the first candidate with gamma > 1e-3 is the fixture, and the in-job certificate (p1core.make_certifier, the code used on
    every run) must flag it. None found -> Q1 certification blocked."""
    jax, jnp, T2 = ex.jax, ex.jnp, ex.T2
    k_, states, sch, nu = ex.last_states
    assert k_ == key, (k_, key)
    k = int(step) - 1                               # certifier index: states[k + 1] is the accepted state of step k + 1
    data = ex.st.data[('d', 'LSPG')]
    co = jnp.where(k < sch['nstart'], jnp.asarray(T2.BE_CO, jnp.float64), sch['co'])
    S = jnp.asarray(states)
    ck, cm, cn = S[k], S[max(k - 1, 0)], S[k + 1]
    A, lam = data['A'], data['lam']
    Fn = jnp.where(co[4] != 0., ex.nl(ck, data) + nu * lam * (A @ ck), 0.)
    r = T2.lspg_residual(ex.nl, co, sch['dt'], nu, data, (A @ ck, A @ cm, Fn))

    def gamma_at(c):
        rv, J = r(c), jax.jacfwd(r)(c)
        return float(jnp.linalg.norm(J.T @ rv) / (jnp.linalg.norm(J) * jnp.linalg.norm(rv) + 1e-300)), float(jnp.linalg.norm(rv))

    J0 = np.asarray(jax.jacfwd(r)(cn))
    _, sv, Vt = np.linalg.svd(J0, full_matrices=False)
    g0, r0 = gamma_at(cn)
    cnorm = float(np.linalg.norm(np.asarray(cn)))
    rep = dict(run=key, step=int(step), gamma_accepted=g0, residual_accepted=r0, c_norm=cnorm, sigma1=float(sv[0]),
               candidates=[], fixture=None)
    for eps in (1e-3, 1e-2, 1e-1):
        for sg in (1., -1.):
            c = cn + jnp.asarray(sg * eps * cnorm * Vt[0])
            g, rn = gamma_at(c)
            rep['candidates'].append(dict(eps_rel=eps, sign=sg, gamma=g, residual=rn))
            if g > GAMMA:
                Sp = np.array(states, copy=True)
                Sp[k + 1] = np.asarray(c)
                gm, _ = ex.cert(jnp.asarray(Sp), nu, data, sch)
                gk = float(np.asarray(gm)[k])
                rep['fixture'] = dict(eps_rel=eps, sign=sg, gamma_measured=g, gamma_certificate=gk,
                                      flagged=bool(not (gk <= GAMMA)), agree=bool(abs(gk - g) <= 1e-6 * abs(g) + 1e-12))
                break
        if rep['fixture'] is not None:
            break
    rep['fired'] = bool(rep['fixture'] and rep['fixture']['flagged'])
    rep['blocked'] = not rep['fired']
    rep['verdict'] = ('FIRED (certificate flags the perturbed state)' if rep['fired'] else
                      'BLOCKED: no candidate with gamma > 1e-3 -> Q1 certification blocked, no production ROM job'
                      if rep['fixture'] is None else 'NOT FIRED: certificate did not flag the fixture -> Q1 certification blocked')
    return rep


def cb1_controls(out, bank):
    """C-B1 must-fail controls on the job's bank: (i) a BANK.json with ckpt / rotation hashes swapped, (ii) a config hash
    differing from BANK.json; both must be rejected, and the matched bank must pass (positive control)."""
    d = out / 'cb1_fixture'
    d.mkdir(exist_ok=True)
    src = BG.resolve(bank, HERE)
    for nm, p in (('ckpt.pkl', src['ckpt']), ('rotation_R512.npz', src['rotation'])):
        if not (d / nm).exists():
            os.symlink(p, d / nm)
    meta = json.loads(src['meta_path'].read_text())
    (d / 'BANK.json').write_text(json.dumps(dict(meta, ckpt_sha256=meta['rotation_sha256'], rotation_sha256=meta['ckpt_sha256'])))
    i = BG.must_reject({'dir': str(d)}, '/')
    ii = BG.must_reject(dict(bank, ckpt_sha256='0' * 64), HERE)
    pos = BG.must_reject(bank, HERE)
    return dict(swapped_bank_json=i, config_hash_mismatch=ii, matched_positive=dict(pos, passed=not pos['fired']),
                fired=bool(i['fired'] and ii['fired'] and not pos['fired']))


def summarise(rep, cases, store, anchors, rows, exhausted):
    """Per-case outcome, counts and the pooled Q1 criterion. Shared with fcollect.py (killed-job exhaustion records)."""
    term = [store.get(c.scored) for c in cases if c.scored]
    tot = sum(int(r_['steps']) for r_ in term if r_ and 'steps' in r_)
    bad = sum(int(r_['noncertified']) for r_ in term if r_ and 'noncertified' in r_)
    pooled = (bad / tot) if tot else None
    pooled_ok = pooled is not None and pooled <= NONST_POOLED
    outc = []
    for c in cases:
        d = c.as_dict()
        q1 = q1_of(store, c.scored, c.cid in anchors) if c.scored else None
        d.update(q1_terminal=q1, qualified=bool(c.terminal == 'CONVERGED' and q1 and pooled_ok),
                 e_infinite=c.terminal == 'INVALID' or c.scored is None)
        outc.append(d)
    rep.update(records=store, rows=rows, policy=outc,
               counts={t: sum(1 for c in cases if c.terminal == t) for t in SM.TERMINALS},
               uncertified_start=sum(c.uncertified_start for c in cases),
               pooled_nonstationary=dict(fraction=pooled, steps=tot, bad=bad, ok=pooled_ok),
               q1_failing_terminal=sum(1 for d in outc if d['q1_terminal'] is not True),
               qualified=sum(d['qualified'] for d in outc), budget_exhausted=bool(exhausted),
               solve_seconds=sum(r_['seconds'] for r_ in rows))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--t0-epoch', type=float, default=None,
                    help='job start (date +%%s): the self-timer counts the wall limit from here, not from driver start')
    a = ap.parse_args()
    begin = time.perf_counter() - (max(0., time.time() - a.t0_epoch) if a.t0_epoch else 0.)
    cfg = json.loads(Path(a.config).read_text())
    smoke = str(cfg.get('attempt', '')).startswith('smk')
    max_rounds = int(cfg.get('max_rounds', SM.MAX_ROUNDS))
    assert max_rounds == SM.MAX_ROUNDS or smoke, 'the round cap is 10 except in smokes'
    assert (cfg.get('scheme', 'CNR') == 'CNR' and int(cfg.get('Rp', 384)) == 384) or smoke, 'production policy: r = 384, LSPG-CN-R'
    table = LANE / cfg.get('table', 'phaseb/fresh_cohorts.json')
    cases = FC.case_list(cfg['cases'], table, test_once=bool(cfg.get('test_once')))
    if a.dry_run:
        ps = SM.paths(cfg.get('selected'))
        print(json.dumps(dict(cases=len(cases), start=SM.start_state(cfg.get('selected')), n_paths=len(ps),
                              longest=max(map(len, ps)), max_solves=max(len(SM.path_solves(p)) for p in ps))))
        return
    out = Path(a.out)
    bcfg = BG.verify(cfg['bank'], HERE, expect_label=cfg.get('bank_label'))      # C-B1 hash guard (raises BankMismatch)
    tbl_sha = FC.load_table(table)[0]['sha256'] if any(c[1] in FC.FRESH for c in cases) else None
    tkey = None
    if cfg.get('test_once'):                          # production: registered schedule + immutable test-once registration
        SCH.check_policy_config(cfg)
        tkey = TO.require(bcfg, cfg, tbl_sha, out)
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    pcases = [SM.Case(c[0], cfg.get('selected')) for c in cases]
    dump(dict(config=cfg, testonce_key=tkey, bank=bcfg, table_sha256=tbl_sha, started=time.time(),
              policy_sha256=TO.policy_sha(cfg)), out / 'RUN.json')
    dump(dict(cases=[c.as_dict() for c in pcases], records={}, rows=[], anchors=[], elapsed=0.,
              stage='before setup'), out / 'progress.json')          # fcollect works from the first second on
    import jax
    import p1common as P
    P.check_env(cfg)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'), bank=bcfg,
               table=str(table) if tbl_sha else None, table_sha256=tbl_sha,
               cases={c[0]: dict(cohort=c[1], j=c[2], split=c[4], desc=dict(zip(FC.KEYS, map(float, c[3])))) for c in cases},
               ceiling_hours=float(cfg.get('ceiling_hours', .4)), stop_fraction=float(cfg.get('stop_fraction', .9)),
               max_rounds=max_rounds, testonce_key=tkey, policy_sha256=TO.policy_sha(cfg), setup={}, controls={},
               complete=False)
    if (cfg.get('controls') or {}).get('cb1'):
        rep['controls']['cb1'] = cb1_controls(out, cfg['bank'])
    dump(rep, out / 'result.json')
    ex = Executor(cfg, out, cases, begin)
    rep['setup'] = ex.setup(bcfg)
    rep['anchors'] = sorted(ex.anchors)
    dump(rep, out / 'result.json')
    ex.progress = lambda: dump(dict(cases=[c.as_dict() for c in pcases], records=ex.prev, rows=ex.rows,
                                    anchors=sorted(ex.anchors), elapsed=ex.elapsed()), out / 'progress.json')
    cb6 = (cfg.get('controls') or {}).get('cb6')

    def controls(q, rec, ex_):
        if cb6 and q['cid'] == cb6['case'] and q['kind'] == 'prod' and 'CB6' not in rep['controls']:
            rep['controls']['CB6'] = cb6_search(ex_, q['key'], cb6.get('step', 10))
            dump(rep['controls']['CB6'], out / 'CB6.json')

    store = SM.drive(pcases, lambda reqs: ex(reqs, controls), max_rounds=max_rounds, over_budget=ex.over_budget,
                     on_round=lambda cs, st_: ex.progress())
    ex.tighter(pcases, store)
    summarise(rep, pcases, store, ex.anchors, ex.rows, ex.stopped)
    rep.update(elapsed_seconds=ex.elapsed(), shapes=sorted(map(list, ex.shapes)), memory=P.mem(), complete=True)
    dump(rep, out / 'result.json')
    ex.progress()
    (out / 'COMPLETE').write_text('complete\n')
    print('FPOLICY COMPLETE', round(ex.elapsed(), 1), rep['counts'], flush=True)


if __name__ == '__main__':
    main()
