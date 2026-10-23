"""imp-box stage `time` (DESIGN §5, A1.4, A2.5, A3.3, A3.7): finalists per (arm, tau, stat) on val32 with Q1 (+Q2),
paired A-B-A timing against every FOM finalist, S = max_b min_a S(a, b), T-dbl, then diag.py (components, r_x, floors,
finer lattice) and the freeze (frozen.json).

Statuses per (arm, tau, stat): finalist | arm-censored (no valid visited row meets tau) | arm-unqualified (rows meet tau,
all attempted, all fail Q1/Q2) | search-incomplete (18 attempts reached; counts as missing).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import finalize as FZ
import ladder as LD
import ladders as LS
import podsine as PS
import qual as QL
import rom as RM
import score as SC

STATS = ('median', 'worst')
BATCH, MAX_ATTEMPTS, N_FINAL = 6, 18, 2
N_PAIR = 8
FOM_ARMS = ('spec', 'specad')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


# ------------------------------------------------------------------------------------------------ arms ---------

class Arm:
    """One selectable arm: its rows, builder, Q1 and Q2."""

    def __init__(self, name, ctx):
        self.name, self.ctx = name, ctx
        self.sps = self.Phi = self.subs = None
        out = ctx.out
        if name in FOM_ARMS:
            self.build = LS.build_spec
        elif name in ('podg', 'ecsw'):
            self.operator_ok = json.loads((out / 'pod.json').read_text())['pod'].get('operator_ok', False)
            self.sps, self.Phi, _ = LS.pod_spaces(out, 'pod')
            self.build = LS.build_reduced(self.sps)
            if name == 'ecsw':
                self.subs = {}
                for f in sorted(out.glob('ecsw_r*_k*.npz')):
                    r, k = f.stem.split('_')[1:]
                    z = np.load(f)
                    self.subs[(int(r[1:]), int(k[1:]))] = self.sps[12].with_weights(z['nodes'], z['weights'])
                self.build = LS.build_ecsw(self.subs)
        elif name == 'trans':
            self.operator_ok = json.loads((out / 'pod.json').read_text())['podc'].get('operator_ok', False)
            self.sps, self.Phi, _ = LS.pod_spaces(out, 'podc')
            self.build = LS.build_reduced(self.sps, self.Phi)
        else:
            self.sps, _ = LS.bank_spaces(out)
            self.build = LS.build_reduced(self.sps)

    def construction_failures(self):
        """Expected-but-not-completed construction (code audit 2): build errors in this arm's ladder rows, ECSW fits
        that returned no support."""
        fam = {'bank_lspg': 'bank', 'bank_e4': 'bank'}.get(self.name, self.name)
        st = {'bank_lspg': 'lspg_cnr', 'bank_e4': 'gal_etdrk4'}.get(self.name)
        n = sum(1 for r in self.ctx.L.done.values() if r['cfg'].get('fam') == fam and r.get('error')
                and (st is None or r['cfg'].get('st') == st))
        if self.name == 'ecsw' and (self.ctx.out / 'ecsw_fits.json').exists():
            for f in json.loads((self.ctx.out / 'ecsw_fits.json').read_text()):
                n += sum(1 for v in f['sizes'].values() if v is None)
        return n

    def rows(self):
        fam = {'bank_lspg': 'bank', 'bank_e4': 'bank'}.get(self.name, self.name)
        st = {'bank_lspg': 'lspg_cnr', 'bank_e4': 'gal_etdrk4'}.get(self.name)
        return [r for r in self.ctx.L.done.values() if r['cfg'].get('fam') == fam and r.get('valid')
                and (st is None or r['cfg'].get('st') == st)]

    def build_on(self, cfg, sp, z_out):
        q = cfg['q']
        form = 'tensor' if str(q).startswith('T') else 'point'
        if self.name == 'trans':
            if z_out:
                inner = RM.make_query(sp, cfg['r'], 'gal_etdrk4', cfg['nsteps'], form=form, out='z')
                import hoist as HZ                    # A16: the outermost callable is hoisted (code audit 12)
                return HZ.hoisted(lambda d: inner(d.at[0].set(0.5).at[1].set(0.5)))
            return PS.make_trans_query(sp, self.Phi, cfg['r'], cfg['nsteps'], form=form)
        return RM.make_query(sp, cfg['r'], cfg['st'], cfg['nsteps'], form=form, tol=cfg.get('tol', 'prod'),
                             out='z' if z_out else 'lattice')

    def q2_spaces(self, cfg):
        if not str(cfg['q']).startswith('T'):
            return self.sps
        return {12: self.sps[12], 14: self.sps[14]}

    def qualify(self, cfg, O, st, w_scale=1.0):
        c = self.ctx
        if self.name in FOM_ARMS:
            return dict(Q1=QL.q1_fom(cfg, self.build, c.Dv, O), Q2=None)
        fam = 'trans' if self.name == 'trans' else cfg['fam']
        q1 = FZ.q1(fam, cfg, self.build, c.Dv, c.Dd, c.Uv, c.Ud, O, st)
        if self.name == 'ecsw':
            pc = dict(fam='podg', r=cfg['r'], q=12, st='gal_etdrk4', nsteps=cfg['nsteps'])
            key = f"podg-parent|r{cfg['r']}|nt{cfg['nsteps']}"
            if key not in c.cache_q2:
                c.cache_q2[key] = QL.q2(pc, self.sps, self.build_on, c.Dv, c.Uv)
            sub = self.subs[(cfg['r'], cfg['kappa'])]
            bs = lambda cf, sp, zo: RM.make_query(sp, cf['r'], 'gal_etdrk4', cf['nsteps'], form='point',
                                                  out='z' if zo else 'lattice')
            q2 = QL.q2_ecsw(cfg, sub, self.sps[12], bs, bs, c.Dv, c.Uv, c.cache_q2[key], w_scale=w_scale)
        else:
            q2 = QL.q2(cfg, self.q2_spaces(cfg), self.build_on, c.Dv, c.Uv, w_scale=w_scale)
        return dict(Q1=q1, Q2=q2)


class Ctx:
    def __init__(self, a, out, D):
        self.a, self.out = a, out
        rj = json.loads((out / 'ref.json').read_text())
        self.U = rj['allowance']['U']
        import cell as CL
        self.Dv, self.Uv = CL.eligible(D['val'], np.load(out / 'refs_val.npz'))
        self.Dd, self.Ud = CL.eligible(D['dev'], np.load(out / 'refs_dev.npz'))
        self.L = LD.Ladder(out / 'ladder.jsonl', self.Uv, self.Dv, self.U, tag='val')
        self.cache_off, self.cache_qual, self.cache_q2, self.cache_scr = {}, {}, {}, {}


# ------------------------------------------------------------------------------------------------ selection ----

def rescreen(q, D, reps=3):
    for d in D[:1]:
        jax.block_until_ready(q(jnp.asarray(d)))
    return float(np.mean([FZ.screen_time(q, d, reps) for d in D]))


def select(arm, tau, stat):
    """Statuses (A2.5 + code audit 1): finalist | finalist-partial (>= 1 finalist, attempt cap reached before
    exhaustion) | arm-censored (valid rows exist, none meets tau) | arm-unqualified | search-incomplete |
    missing-construction (no valid row at all: build failures / no ECSW fit) | operator-check-failed."""
    c = arm.ctx
    allrows = arm.rows()
    rows = [r for r in allrows if SC.three_way(r[stat], c.U, tau) == 'meets' and r.get('screen') is not None]
    rows.sort(key=lambda r: (r['screen'], r['key']))
    rec = dict(arm=arm.name, tau=tau, stat=stat, n_meeting=len(rows), n_valid_rows=len(allrows), attempts=[],
               finalists=[])
    if getattr(arm, 'operator_ok', True) is False:
        rec['status'] = 'operator-check-failed'
        return rec
    nfail = arm.construction_failures() if hasattr(arm, 'construction_failures') else 0
    nscr = sum(1 for r in allrows if SC.three_way(r[stat], c.U, tau) == 'meets' and r.get('screen') is None)
    nfail += nscr                              # a failed screen is incomplete evidence, not censoring (code audit 3)
    rec['construction_failures'] = nfail
    rec['screen_failures'] = nscr
    if not allrows or (not rows and nfail):
        rec['status'] = 'missing-construction'
        return rec
    if not rows:
        rec['status'] = 'arm-censored'
        return rec
    att = 0
    for b0 in range(0, len(rows), BATCH):
        batch = rows[b0:b0 + BATCH]
        for r in batch:
            if r['key'] not in c.cache_scr:
                c.cache_scr[r['key']] = rescreen(arm.build(r['cfg']), c.Dv[:4])
        batch.sort(key=lambda r: (c.cache_scr[r['key']], r['key']))
        for r in batch:
            if att >= MAX_ATTEMPTS:
                rec['status'] = 'search-incomplete' if not rec['finalists'] else 'finalist-partial'
                return rec
            att += 1
            k = r['key']
            if k not in c.cache_off or c.cache_off[k][0] is None:
                O, st = FZ.run_unbatched(arm.build(r['cfg']), c.Dv)
                e = SC.err(O, c.Uv)
                e = np.where(LD.stats_valid(r['cfg'], st), e, np.inf)
                c.cache_off[k] = (O, st, dict(worst=float(np.max(e)), median=float(np.median(e)), e=e.tolist()))
            O, st, sc = c.cache_off[k]
            meets = SC.three_way(sc[stat], c.U, tau) == 'meets'
            if meets and k not in c.cache_qual:
                c.cache_qual[k] = jsonable(arm.qualify(r['cfg'], O, st))
            ql = c.cache_qual.get(k)
            ok = bool(meets and ql and ql['Q1']['passed'] and (ql['Q2'] is None or ql['Q2']['passed']))
            rec['attempts'].append(dict(key=k, rescreen=c.cache_scr[k], official=sc, meets=meets, qualified=ok))
            if not ok and k in c.cache_off:                  # bound the output cache (code audit 1): keep stats only
                c.cache_off[k] = (None, c.cache_off[k][1], c.cache_off[k][2])
            if ok:
                rec['finalists'].append(dict(key=k, cfg=r['cfg'], rescreen=c.cache_scr[k], official=sc))
                if len(rec['finalists']) >= N_FINAL:
                    rec['status'] = 'finalist-partial' if nfail else 'finalist'
                    return rec
    if nfail:                                  # incomplete construction: a finalist set can only be partial
        rec['status'] = 'finalist-partial' if rec['finalists'] else 'missing-construction'
    else:
        rec['status'] = 'finalist' if rec['finalists'] else 'arm-unqualified'
    return rec


# ------------------------------------------------------------------------------------------------ timing -------

def warm(q, d, seconds=2.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        jax.block_until_ready(q(d))


def timed_pair(qA, qB, D):
    dj = jnp.asarray(D[0])
    warm(qA, dj)
    warm(qB, dj)
    n0 = (qA._cache_size(), qB._cache_size())
    res, OA, OB = FZ.paired(qA, qB, D)
    res['compiles_in_timed_region'] = int(qA._cache_size() + qB._cache_size() - sum(n0))
    ts = np.array([r['b'] for r in res['raw']])
    res['outliers_B'] = int(np.sum(ts > 1.2 * np.median(ts)))
    return res, OA, OB


def fom_extend(c, sel, arm_by):
    """A2.4/A3.6 on the QUALIFIED finalists (code audit 1): N at the ladder maximum (or no finalist) -> add N = 384, then
    512; AD tol at the minimum -> add 1e-8; re-select after every extension. Returns the extension log."""
    log_ = []
    for fam in FOM_ARMS:
        for ext in LS.SPEC_EXT:
            rows = arm_by[fam].rows()
            nmax = max([r['cfg']['N'] for r in rows] or [0])
            need = any((not sel[(fam, t, s_)]['finalists']) or
                       any(f['cfg']['N'] == nmax for f in sel[(fam, t, s_)]['finalists'])
                       for t in LS.TAUS for s_ in STATS)
            if nmax >= ext:
                continue
            if not need or c.a.smoke:
                break
            if fam == 'spec':
                c.L.step_ladder(f'spec|N{ext}', dict(fam='spec', N=ext), LS.build_spec, 5)
            else:
                for tol in LS.AD_TOLS:
                    c.L.run(f'specad|N{ext}|tol{tol:.0e}', dict(fam='specad', N=ext, tol=tol), LS.build_spec)
            log_.append(dict(fam=fam, N=ext))
            for t in LS.TAUS:
                for s_ in STATS:
                    sel[(fam, t, s_)] = select(arm_by[fam], t, s_)
    done_ext = set()
    while not c.a.smoke:                       # every distinct binding tol finalist, until closure (code audit 2)
        todo = sorted({(f['cfg']['N'], f['cfg'].get('xform', 'fft')) for t in LS.TAUS for s_ in STATS
                       for f in sel[('specad', t, s_)]['finalists'] if f['cfg']['tol'] == min(LS.AD_TOLS)} - done_ext)
        if not todo:
            break
        for N_, xf in todo:                        # the implementation (xform) is part of the configuration (audit 10)
            cf = dict(fam='specad', N=N_, tol=LS.AD_EXT, xform=xf)
            c.L.run(f"specad|N{N_}|tol{LS.AD_EXT:.0e}" + ('|dense' if xf == 'dense' else ''), cf, LS.build_spec)
            log_.append(dict(fam='specad', N=N_, tol=LS.AD_EXT, xform=xf))
            done_ext.add((N_, xf))
        for t in LS.TAUS:
            for s_ in STATS:
                sel[('specad', t, s_)] = select(arm_by['specad'], t, s_)
    # boundary labels after the extensions
    for fam in FOM_ARMS:
        nmax = max([r['cfg']['N'] for r in arm_by[fam].rows()] or [0])
        for t in LS.TAUS:
            for s_ in STATS:
                rec = sel[(fam, t, s_)]
                lab = []
                if any(f['cfg']['N'] == nmax for f in rec['finalists']):
                    lab.append('search truncated at N')
                if fam == 'specad' and any(f['cfg']['tol'] == LS.AD_EXT for f in rec['finalists']):
                    lab.append('search truncated at tol')
                rec['boundary_labels'] = lab
    return log_


def t_dbl_raw(q, D):
    """finalize.t_dbl with the repetition arrays kept (code audit 2)."""
    d1, d2 = jnp.asarray(D[0]), jnp.asarray(D[1])
    for _ in range(2):
        jax.block_until_ready(q(d1)); jax.block_until_ready(q(d2))
    t1, t2, tb = [], [], []
    for _ in range(5):
        t = time.perf_counter(); jax.block_until_ready(q(d1)); jax.block_until_ready(q(d2)); tb.append(time.perf_counter() - t)
        t = time.perf_counter(); jax.block_until_ready(q(d1)); t1.append(time.perf_counter() - t)
        t = time.perf_counter(); jax.block_until_ready(q(d2)); t2.append(time.perf_counter() - t)
    r = float(np.median(tb) / (np.median(t1) + np.median(t2)))
    return dict(ratio=r, passed=bool(0.9 <= r <= 1.1), both=tb, one=t1, two=t2)


def term_labels(out, sel):
    """Attach the step-ladder termination reason of every finalist's (size) ladder to its selection entry."""
    f = out / 'ladder_terms.jsonl'
    terms = {}
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            terms[r['base']] = r['reason']
    for rec in sel.values():
        for fz in rec['finalists']:
            base = fz['key'].rsplit('|nt', 1)[0]
            fz['ladder_termination'] = terms.get(base, 'not a step ladder')


def stage_time(a, out, D):
    c = Ctx(a, out, D)
    arms = list(FOM_ARMS) + ['podg', 'ecsw'] + (['trans'] if a.trans else []) + \
        (['bank_lspg', 'bank_e4'] if a.bank and (out / 'bank.pkl').exists() else [])
    sel, Darm = {}, {}
    for name in FOM_ARMS:
        arm = Arm(name, c)
        for tau in LS.TAUS:
            for stat in STATS:
                sel[(name, tau, stat)] = select(arm, tau, stat)
        Darm[name] = arm
    ext_log = fom_extend(c, sel, Darm)
    tdbl = {}

    def tdbl_of(arm, f):
        if f['key'] not in tdbl:
            tdbl[f['key']] = t_dbl_raw(arm.build(f['cfg']), c.Dv)
        return tdbl[f['key']]['passed']
    pairs_all, T = {}, {}
    D8 = c.Dv[:N_PAIR]
    for name in arms:
        if name in FOM_ARMS:
            continue
        arm = Arm(name, c)
        for tau in LS.TAUS:
            for stat in STATS:
                rec = select(arm, tau, stat)
                sel[(name, tau, stat)] = rec
                fins_a = [(fn, f) for fn in FOM_ARMS for f in sel[(fn, tau, stat)]['finalists']]
                pr = {}
                for fn, fa in fins_a:
                    for fb in rec['finalists']:
                        pk = (fa['key'], fb['key'])
                        if pk not in pairs_all:
                            qA, qB = Darm[fn].build(fa['cfg']), arm.build(fb['cfg'])
                            res, OA, OB = timed_pair(qA, qB, D8)
                            res['T_id_A'] = FZ.t_id(OA, c.cache_off[fa['key']][0][:len(D8)])
                            res['T_id_B'] = FZ.t_id(OB, c.cache_off[fb['key']][0][:len(D8)])
                            res['T_dbl_A'], res['T_dbl_B'] = tdbl_of(Darm[fn], fa), tdbl_of(arm, fb)
                            res['valid'] = bool(res['T_id_A'] <= 1e-12 and res['T_id_B'] <= 1e-12
                                                and res['compiles_in_timed_region'] == 0
                                                and res['T_dbl_A'] and res['T_dbl_B'])
                            pairs_all[pk] = res
                            log(f'pair {fa["key"]} vs {fb["key"]}: S {res["S"]:.3g} valid {res["valid"]}')
                        r_ = pairs_all[pk]
                        pr[(fn, fa['key'], fb['key'])] = r_['S'] if r_['valid'] else None
                T[(name, tau, stat)] = FZ.maxmin(pr, [f['key'] for f in rec['finalists']],
                                                 [(fn, f['key']) for fn, f in fins_a])
        import diag
        diag.arm_diag(c, arm, sel, T, out)
        del arm
        RM.clear_cache()
    for name in FOM_ARMS:
        for rec in [v for (nm, _, _), v in sel.items() if nm == name]:
            for f in rec['finalists']:
                tdbl_of(Darm[name], f)
    import diag
    diag.fom_diag(c, Darm, sel, out)
    term_labels(out, sel)
    import diag as DG
    final = dict(scorable=True, U=c.U, claim_context=DG.population(c), selection={f'{k[0]}|{k[1]}|{k[2]}': v for k, v in sel.items()},
                 S={f'{k[0]}|{k[1]}|{k[2]}': v for k, v in T.items()},
                 pairs={f'{k[0]} || {k[1]}': v for k, v in pairs_all.items()},
                 qual=c.cache_qual, q2_parent=c.cache_q2, tdbl=tdbl, fom_extensions=ext_log,
                 n_val=len(c.Dv), n_dev=len(c.Dd))
    (out / 'final.json').write_text(json.dumps(jsonable(final), indent=1))
    freeze(a, out, final)


QUERY_SOURCES = ('geom.py', 'spec.py', 'rom.py', 'podsine.py', 'bank.py', 'builders.py', 'hoist.py', 'vendor/pscore.py',
                 'vendor/sep_common.py')


POLICY_SOURCES = ('ladders.py', 'ladder.py', 'timing.py', 'qual.py', 'finalize.py', 'score.py', 'diag.py', 'cell.py')


def policy_identity():
    """sha256 of the ladder / selection policy sources (bound to the cost certificate, not to the test freeze)."""
    here = Path(__file__).resolve().parent
    return {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in POLICY_SOURCES}


def source_identity():
    """sha256 of every query-defining source file (verified again in the test stage)."""
    here = Path(__file__).resolve().parent
    return {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in QUERY_SOURCES}


def freeze(a, out, final):
    """A3.7: frozen finalists and the val pair attaining S per (arm, tau, stat), with provenance (code audit 1)."""
    import os
    arts = {}
    for p in sorted(list(out.glob('*_Phi.npy')) + list(out.glob('ecsw_r*_k*.npz')) + list(out.glob('bank.pkl'))):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 24), b''):
                h.update(blk)
        arts[p.name] = h.hexdigest()
    for nm in ('pod.json', 'bank.json', 'data.json', 'ref.json', 'cohorts.json'):     # query-defining metadata too
        if (out / nm).exists():
            arts[nm] = hashlib.sha256((out / nm).read_bytes()).hexdigest()
    rd = json.loads((out / 'ref.json').read_text())
    dd = json.loads((out / 'data.json').read_text())
    fr = dict(W=a.W, lev=a.lev, commit=os.environ.get('LANE_COMMIT'), artifacts=arts, source=source_identity(),
              policy=policy_identity(),
              cohorts=json.loads((out / 'cohorts.json').read_text()), U_val=final['U'],
              ref_scorable=rd['scorable'], training_gates={k: v['gate'] for k, v in dd.items()
                                                           if isinstance(v, dict) and 'gate' in v},
              claim_context=final.get('claim_context'), selection={}, pairs={})
    for k, v in final['selection'].items():            # qualifications retained with every frozen entry (audit 4)
        fr['selection'][k] = dict(status=v['status'], boundary_labels=v.get('boundary_labels', []),
                                  construction_failures=v.get('construction_failures', 0),
                                  screen_failures=v.get('screen_failures', 0),
                                  finalists=[dict(key=f['key'], cfg=f['cfg'],
                                                  ladder_termination=f.get('ladder_termination'))
                                             for f in v['finalists']])
    for k, v in final['S'].items():
        if v and v.get('S') is not None:
            fr['pairs'][k] = dict(competitor_key=v['competitor_key'], arm_key=v['bank_key'], S_val=v['S'],
                                  excluded=v.get('excluded', []), per_family=v.get('per_family'),
                                  arm_status=final['selection'][k]['status'])
    (out / 'frozen.json').write_text(json.dumps(fr, indent=1))
