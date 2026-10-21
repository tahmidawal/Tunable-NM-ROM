"""imp-hrcost job driver (DESIGN.md with amendments A1-A3; E5R when adopted): one mesh, one GPU, all arms in one job.

    python hjob.py --config configs/<attempt>.json --out <dir>

Phases (each timed, peak device bytes recorded):
  setup     Base (bank on the run mesh, R operators), bank Mass / Stiff (P3), identities, initial conditions
  ladder    every registered rule of each projection on the Q2 population (sub16): own runs (prod tolerance); Q2 rows
            (successor rollout distance, tested-vector rho = binding, projected rho = indicator, check distances)
  q1        for R and the F2 candidates (increasing m, until the first Q1+Q2 pass), F2 at Gauss 128^2 and every fitted
            rule: production runs on the case set, in-job certificate (fresh AD Jacobians), tight / tighter anchors
  fit       parent rollouts on training draws, ECSW systems, support-targeted greedy NNLS (F3a, F3b, F2F3)
  timing    paired A-B-A (A = R) end-to-end and evolution-only, burn-in before each block; Jacobian microbenchmark (C5)
  controls  C1-C5, C7, P1, P3-P6 (smoke)
Outputs: result.json, arrays/<arm>.npz (W, W_tight, W_tighter, states, eta, gamma, cond, it, reason, eta_rec), COMPLETE.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import hrtab as HT
import gstep as GS
import fit as FT
import hlabel as HL
from hrtab import P, Q

TOL = dict(prod=1e-10, tight=1e-13, tighter=1e-15)
GAUSS = [64, 80, 96, 112, 128, 160, 192]
FIB = [4181, 6765, 10946, 17711, 28657]
MMAX_CAND = 16384


def ladders(cfg):
    out = {'gauss': [f'gauss{p}' for p in cfg.get('gauss', GAUSS)]}
    if cfg.get('fib', True):
        out['fib'] = [f'fib{n}' for n in cfg.get('fib_list', FIB)]
    return out


class Job:
    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, Path(out)
        (self.out / 'arrays').mkdir(parents=True, exist_ok=True)
        self.rep = dict(config=cfg, phases={}, mem={}, setup={}, q2={}, q1={}, fits={}, timing={}, controls={},
                        errors=[], gpu=P.gpu_info(), jax=jax.__version__)
        self.t0 = time.perf_counter()
        self.L, self.Rp, self.dt = int(cfg['mesh']), int(cfg['Rp']), float(cfg.get('dt', .005))
        self.S = int(round(.25 / self.dt))
        self.sch = {k: GS.sched(self.dt, v, budget=cfg.get('budget', 50)) for k, v in TOL.items()}
        self.tabs = {}
        self.keep = set()

    # ------------------------------------------------------------------------------------------- utils ----
    def save(self):
        (self.out / 'result.json').write_text(json.dumps(self.rep, indent=1, default=_js))

    def phase(self, name, t):
        self.rep['phases'][name] = time.perf_counter() - t
        self.rep['mem'][name] = dict(P.mem(), host_maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(f'PHASE {name} {self.rep["phases"][name]:.1f}s mem={self.rep["mem"][name]}', flush=True)
        self.save()

    # ------------------------------------------------------------------------------------------- setup ----
    def setup(self):
        t = time.perf_counter()
        cfg = self.cfg
        self.base = B = HT.Base(self.L, self.Rp, cfg.get('policy', 'frozen'))
        self.Qy = GS.make_query(self.L, nrec=self.S + 1)
        self.cert = GS.make_certifier()
        self.U = {'sine': GS.gram_factor('sine', B)}
        cases = P.case_list(cfg)
        self.cases = cases
        self.ids = [f'{c[0]}|{c[1]}' for c in cases]
        s256 = self.L // 256
        self.U0 = {}
        for i, (coh, c, ph) in enumerate(cases):
            u0n = Q.e.initial(self.L, ph)
            self.U0[self.ids[i]] = (jnp.asarray(u0n), float(ph[4]), float(np.linalg.norm(u0n[::s256, ::s256])))
        self.sets = {k: [self.ids[i] for i in P.select(cases, k)] for k in ('all38', 'sub16', 'dev6', 'dev2')}
        self.case_set = self.sets[cfg.get('cases', 'all38')]
        self.q2_set = self.sets[cfg.get('q2_cases', 'sub16')]
        self.dev = [c for c in self.sets['dev6'] if c in self.case_set]
        ckpt = HT.T2.CKPT
        self.rep['setup'] = dict(L=self.L, Rp=self.Rp, M=B.M, dt=self.dt, steps=self.S, mode_hash=B.mode_hash,
                                 policy=B.policy, ckpt_sha256=_fsha(ckpt), rotation_sha256=_fsha(Q.ROTATION),
                                 cold_sha256=HT.sha(np.concatenate([np.asarray(B.cold[0]).ravel(), np.asarray(B.cold[1])])),
                                 mesh_hash=B.bank.mesh_hash, cohort_hashes=P.cohort_hashes(),
                                 stepper_sha256=_fsha(Path(GS.__file__)), tables_sha256=_fsha(Path(HT.__file__)),
                                 A_sha=HT.sha(B.A), cond_Rm=float(np.linalg.cond(np.asarray(B.Rm))),
                                 n0={k: v[2] for k, v in self.U0.items()})
        if cfg.get('bank_arms', True):
            Mm, Km, L0, info = B.bank_ops()
            self.U['bank'] = GS.gram_factor('bank', B)
            self.rep['setup']['bank_ops'] = info
            np.savez(self.out / 'arrays' / 'bank_ops.npz', Mass=np.asarray(Mm), Stiff=np.asarray(Km), A=np.asarray(B.A),
                     lam=np.asarray(B.lam), Rm=np.asarray(B.Rm), kx=np.asarray(B.kx), ky=np.asarray(B.ky))
        else:
            np.savez(self.out / 'arrays' / 'bank_ops.npz', A=np.asarray(B.A), lam=np.asarray(B.lam), Rm=np.asarray(B.Rm),
                     kx=np.asarray(B.kx), ky=np.asarray(B.ky))
        self.phase('setup', t)

    # -------------------------------------------------------------------------------------------- runs ----
    def table(self, proj, rule):
        k = (proj, rule)
        if k not in self.tabs:
            d, aux, meta, (X, w) = self.base.tables(proj, rule)
            meta['fingerprint'] = HT.table_fingerprint(d)
            self.tabs[k] = (d, aux, meta, w)
        return self.tabs[k]

    def drop(self, keep=()):
        for k in [k for k in self.tabs if k not in keep]:
            del self.tabs[k]

    def run(self, d, proj, cid, level='prod', sch=None):
        u0, nu, n0 = self.U0[cid]
        o = self.Qy['fused'](u0, nu, d, self.U[proj], self.base.G, self.base.cold, sch or self.sch[level])
        o = jax.device_get(o)
        f257 = P.restrict(o['fields'], self.L)
        st = o['stats']
        return dict(W=np.asarray(o['W']), f257=f257, states=np.asarray(st['states'])[:self.S + 1],
                    it=np.asarray(st['it'])[:self.S], reason=np.asarray(st['reason'])[:self.S],
                    eta_rec=np.asarray(st['eta_rec'])[:self.S], valid=GS.run_valid(o, self.S), n0=n0, nu=nu)

    def certify(self, d, proj, r):
        eta, gam, cond = self.cert(jnp.asarray(r['states']), r['nu'], d, self.U[proj], self.dt, self.S)
        return np.asarray(eta), np.asarray(gam), np.asarray(cond)

    # -------------------------------------------------------------------------------------- Q2 ladder ----
    def own_runs(self, proj, rule, cases, d=None):
        d = d if d is not None else self.table(proj, rule)[0]
        return {cid: self.run(d, proj, cid) for cid in cases}

    def q2(self, name, proj, own_tab, own_runs, succ_tab, succ_runs, chk_runs, inherited_proj=None):
        """Q2 rows of one arm (A2.1): successor rollout distances and the tested-vector rho at own output states."""
        d, aux = own_tab[0], own_tab[1]
        ds, auxs = succ_tab[0], succ_tab[1]
        Ug = self.U[proj]
        rows = {}
        rho_b, rho_p = [], []
        for cid in self.q2_set:
            o, s_ = own_runs[cid], succ_runs[cid]
            rb, rp = [], []
            for k in range(6):
                c = jnp.asarray(o['W'][k])
                rb.append(HL.rho(np.asarray(HT.tested(d, aux, c)), np.asarray(HT.tested(ds, auxs, c))))
                va, vb = HT.advection(d, c), HT.advection(ds, c)
                rp.append(HL.rho(np.asarray(Ug @ va), np.asarray(Ug @ vb)))
            rho_b.append(rb)
            rho_p.append(rp)
            rows[cid] = dict(own_valid=o['valid'], succ_valid=s_['valid'],
                             dist_succ=HL.fdist(o['f257'], s_['f257'], o['n0']), rho=rb, rho_proj=rp)
        dev = [c for c in self.q2_set if c.startswith('dev6')]
        chk = {cid: HL.fdist(succ_runs[cid]['f257'], chk_runs[cid]['f257'], succ_runs[cid]['n0']) for cid in dev}
        args = ([rows[c]['own_valid'] for c in self.q2_set], [rows[c]['succ_valid'] for c in self.q2_set],
                [rows[c]['dist_succ'] for c in self.q2_set])
        chk_args = ([succ_runs[c]['valid'] for c in dev], [chk_runs[c]['valid'] for c in dev], [chk[c] for c in dev])
        lab = HL.q2_label(*args, np.asarray(rho_b), *chk_args)
        labp = HL.q2_label(*args, np.asarray(rho_p), *chk_args)
        self.rep['q2'][name] = dict(proj=proj, own=own_tab[2]['rule'], succ=succ_tab[2]['rule'], m=own_tab[2]['m'],
                                    label=lab['label'], detail=lab, label_projected=labp['label'], detail_projected=labp,
                                    rows=rows, chk=chk)
        np.savez(self.out / 'arrays' / f'q2_{name}.npz', W=np.stack([own_runs[c]['W'] for c in self.q2_set]),
                 Ws=np.stack([succ_runs[c]['W'] for c in self.q2_set]), case_ids=np.array(self.q2_set))
        return lab['label']

    def ladder(self, proj, rules, name_of):
        """Own runs of every rule of a ladder on the Q2 population; Q2 of every rule that has a successor and check."""
        runs = {}
        for i, rl in enumerate(rules):
            runs[rl] = self.own_runs(proj, rl, self.q2_set)
            if i >= 2:
                q, qp, qpp = rules[i - 2], rules[i - 1], rl
                self.q2(name_of(q), proj, self.table(proj, q), runs[q], self.table(proj, qp), runs[qp], runs[qpp])
                self.drop(keep=[(proj, rules[i - 1]), (proj, rl)] + list(self.keep))
        return runs

    # -------------------------------------------------------------------------------------------- Q1 ----
    def q1(self, name, proj, d, meta, cases=None):
        cases = cases or self.case_set
        rows, arr = {}, {k: [] for k in ('W', 'W_tight', 'states', 'eta', 'gamma', 'cond', 'it', 'reason', 'eta_rec')}
        arr_tt = []
        for cid in cases:
            o = self.run(d, proj, cid)
            eta, gam, cond = self.certify(d, proj, o)
            tt = self.run(d, proj, cid, 'tight')
            row = dict(valid=o['valid'], eta=eta, eta_rec=o['eta_rec'], tight_valid=tt['valid'] and bool(np.all(tt['reason'] == 1)),
                       dist_tight=HL.fdist(o['f257'], tt['f257'], o['n0']), it_mean=float(o['it'].mean()),
                       it_max=int(o['it'].max()), budget_exits=int((o['reason'] == 0).sum()))
            if cid in self.dev:
                t2 = self.run(d, proj, cid, 'tighter')
                row.update(tighter_valid=t2['valid'] and bool(np.all(t2['reason'] == 1)),
                           dist_tt=HL.fdist(tt['f257'], t2['f257'], o['n0']))
                arr_tt.append(t2['W'])
            rows[cid] = row
            for k, v in (('W', o['W']), ('W_tight', tt['W']), ('states', o['states']), ('eta', eta), ('gamma', gam),
                         ('cond', cond), ('it', o['it']), ('reason', o['reason']), ('eta_rec', o['eta_rec'])):
                arr[k].append(v)
        lab = HL.q1_label(rows, self.dev)
        np.savez(self.out / 'arrays' / f'q1_{name}.npz', case_ids=np.array(cases), W_tighter=np.asarray(arr_tt),
                 dev_ids=np.array(self.dev), **{k: np.asarray(v) for k, v in arr.items()})
        self.rep['q1'][name] = dict(proj=proj, rule=meta['rule'], m=meta['m'], meta=meta, label=lab['label'],
                                    reasons=lab['reasons'][:20], nonconv_frac=lab['nonconv_frac'],
                                    rows={c: {k: v for k, v in r.items() if k not in ('eta', 'eta_rec')} for c, r in rows.items()},
                                    gamma_max=float(np.nanmax(np.asarray(arr['gamma']))),
                                    cond_max=float(np.nanmax(np.asarray(arr['cond']))))
        self.save()
        return lab['label']

    # ------------------------------------------------------------------------------------------- fits ----
    def fit_family(self, fam, parent_key):
        """A1.5 / A2.2 / A2.5: training rollouts of the parent arm, ECSW system, greedy NNLS; returns {kappa: tables}."""
        t = time.perf_counter()
        proj, rule = parent_key
        d, aux, meta, w = self.table(proj, rule)
        D = FT.train_draw_2d(seed=0, count=576)
        FT.assert_not_dev_val(D[:FT.NDRAW + FT.MAX_REPL], np.stack([c[2] for c in self.cases]))
        steps = FT.SNAP_STEPS[fam]
        rows_used, snaps, repl = [], [], 0
        j = 0
        while len(rows_used) < FT.NDRAW and j < FT.NDRAW + FT.MAX_REPL:
            ph = D[j]
            u0 = jnp.asarray(Q.e.initial(self.L, ph))
            o = self.Qy['fused'](u0, float(ph[4]), d, self.U[proj], self.base.G, self.base.cold, self.sch['prod'])
            o = jax.device_get(o)
            if GS.run_valid(o, self.S) and np.all(np.asarray(o['stats']['reason'])[:self.S] == 1):
                rows_used.append(j)
                snaps.append(np.asarray(o['stats']['states'])[list(steps)])
            else:
                repl += 1
            j += 1
        info = dict(family=fam, parent=rule, parent_fingerprint=meta['fingerprint'], draws_sha=HT.sha(D),
                    rows_used=rows_used, replacements=repl, steps=list(steps))
        if len(rows_used) < FT.NDRAW:
            info['status'] = 'not built (too many invalid parent rollouts)'
            self.rep['fits'][fam] = info
            return {}
        snaps = np.concatenate(snaps)
        info['snap_sha'] = HT.sha(snaps)
        info['rollout_seconds'] = time.perf_counter() - t
        rows_unw = self.base.fit_rows(fam, aux, w)
        t1 = time.perf_counter()
        try:
            C, b, sinfo = FT.build_system(rows_unw, d['Gq'], d['Gs'], w, snaps)
        except FloatingPointError as e:
            info['status'] = f'not built ({e})'
            self.rep['fits'][fam] = info
            return {}
        info.update(system=sinfo, system_seconds=time.perf_counter() - t1, mem_system=P.mem())
        if sinfo['snapshots'] == 0:
            info['status'] = 'not built (all blocks dropped)'
            self.rep['fits'][fam] = info
            return {}
        res = FT.fit(C, b, self.Rp, log=lambda s: print(fam, s, flush=True))
        info.update({k: v for k, v in res.items() if k != 'rules'}, mem_fit=P.mem())
        del C
        drift_bad = not res.get('drift_ok', True)
        out = {}
        info['rules'] = {}
        for k, rr in res['rules'].items():
            if rr is None:
                info['rules'][k] = dict(status='not reached')
                continue
            if drift_bad:
                info['rules'][k] = dict(status='not built (drift)')
                continue
            ok = bool(np.all(np.isfinite(rr['w']) & (rr['w'] > 0)) and rr['size'] >= k * self.Rp)
            chk = FT.check_residual(rows_unw, d['Gq'], d['Gs'], w, snaps, rr['idx'], rr['w'])
            agree = abs(chk - rr['relres']) <= 1e-8 * max(abs(rr['relres']), 1e-300) + 1e-14
            st = 'built' if (ok and agree) else 'not built (support check)'
            info['rules'][k] = dict(status=st, size=rr['size'], relres=rr['relres'], relres_check=chk,
                                    w_sum=float(rr['w'].sum()))
            np.savez(self.out / 'arrays' / f'fit_{fam}_k{k}.npz', idx=rr['idx'], w=rr['w'])
            if st == 'built':
                out[k] = HT.fitted((d, aux, meta), w, rr['idx'], rr['w'], f'{fam}k{k}')
        info['status'] = 'done'
        info['seconds'] = time.perf_counter() - t
        self.rep['fits'][fam] = info
        self.save()
        return out

    # ------------------------------------------------------------------------------------------ timing ----
    def timing(self, pairs, cases, reps, burn):
        """Paired A-B-A per case (A = R): end-to-end fused query and evolution only (A1.7)."""
        dR, projR = pairs[0][1], pairs[0][2]
        res = {}
        ev = self.Qy['evolve']
        for name, dB, projB in pairs:
            per = {}
            for cid in cases:
                u0, nu, n0 = self.U0[cid]
                fA = lambda: self.Qy['fused'](u0, nu, dR, self.U[projR], self.base.G, self.base.cold, self.sch['prod'])
                fB = lambda: self.Qy['fused'](u0, nu, dB, self.U[projB], self.base.G, self.base.cold, self.sch['prod'])
                w0 = self.Qy['init'](u0, self.base.cold)
                eA = lambda: ev(w0, nu, dR, self.U[projR], self.sch['prod'])
                eB = lambda: ev(w0, nu, dB, self.U[projB], self.sch['prod'])
                rec = {}
                for tag, a_, b_ in (('e2e', fA, fB), ('evo', eA, eB)):
                    for _ in range(burn):
                        jax.block_until_ready(a_())
                        jax.block_until_ready(b_())
                    tA1, tB, tA2 = [], [], []
                    for _ in range(reps):
                        for lst, f in ((tA1, a_), (tB, b_), (tA2, a_)):
                            s = time.perf_counter()
                            jax.block_until_ready(f())
                            lst.append(time.perf_counter() - s)
                    a1, bb, a2 = np.median(tA1), np.median(tB), np.median(tA2)
                    rec[tag] = dict(A1=tA1, B=tB, A2=tA2, ratio=float(.5 * (a1 + a2) / bb),
                                    outliers=int(sum(x > 3 * np.median(l_) for l_ in (tA1, tB, tA2) for x in l_)))
                ob = jax.device_get(fB())
                rec['timed_W_sha'] = HT.sha(np.asarray(ob['W']))
                rec['timed_valid'] = GS.run_valid(ob, self.S)
                per[cid] = rec
            r_e = np.array([per[c]['e2e']['ratio'] for c in cases])
            r_v = np.array([per[c]['evo']['ratio'] for c in cases])
            res[name] = dict(per_case=per, e2e_median=float(np.median(r_e)), evo_median=float(np.median(r_v)),
                             e2e_ci=_boot(r_e), evo_ci=_boot(r_v))
            print('TIMING', name, res[name]['e2e_median'], res[name]['evo_median'], flush=True)
            self.rep['timing'][name] = res[name]
            self.save()
        return res

    def jac_bench(self, d, n=200, warm=20):
        rng = np.random.default_rng(0)
        cs = jnp.asarray(rng.standard_normal((n + warm, self.Rp)) * 1e-2)
        f = jax.jit(lambda c, acc: acc + jnp.sum(GS.rhs_jac(c, .05, d)))
        acc = jnp.float64(0.)
        for i in range(warm):
            acc = f(cs[i], acc)
        jax.block_until_ready(acc)
        ts = []
        for i in range(warm, warm + n):
            s = time.perf_counter()
            acc = jax.block_until_ready(f(cs[i], acc))
            ts.append(time.perf_counter() - s)
        return float(np.median(ts)), float(acc)


def _boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return [float(np.quantile(meds, .025)), float(np.quantile(meds, .975))]


def _fsha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _js(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    P.check_env(cfg)
    import hjob_main
    J = Job(cfg, a.out)
    try:
        hjob_main.run(J)
    except Exception:  # noqa: BLE001
        J.rep['errors'].append(traceback.format_exc())
        J.save()
        raise
    J.rep['total_seconds'] = time.perf_counter() - J.t0
    J.save()
    (Path(a.out) / 'COMPLETE').write_text('ok\n')


if __name__ == '__main__':
    main()
