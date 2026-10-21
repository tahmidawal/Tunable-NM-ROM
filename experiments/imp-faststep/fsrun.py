"""imp-faststep job driver (DESIGN.md + A1/A2): one mesh per job, one GPU, every arm in the same job.

Per rank spec (r, rule, successor rule) on the frozen-policy bank of the mesh:
  arm 1  L-CNR    p1-scaling LSPG CN-R at dt0 (p1common.Setting / p1core split query), certified (autodiff gamma)
  arm 2  G-CN     precontracted Galerkin CN, Newton (fscore 'cn'), fresh-residual certificate
  arm 3a G-E4a    ETDRK4 at 0.0125, per-step reconstruction certificate
  arm 3b G-E4b    ETDRK4 at 0.025
  arm 4  G-R1     one-pass linearised CN at dt0, Jacobian + increment certificate
  arm 6  L-CNR-M1152 (r = 384 only; M override)
on every case of the configured set: production run, certificate, stability monitor, E0 ST score (in-job, refstore),
successor-rule rollout + own-state rho (Q2, every case), anchors (arms 1, 6, 2: tight every case, tighter dev6),
dev6 twins (arms 3, 4), dev6 step halving, the F5 order block, the stability classification of tripped runs; then the
POD block (podarm.py, 1024^2 only), the controls block (smoke), and the paired timing block. Outputs: result.json,
A_<arm>_<r>.npz (pulled: coefficients, states, certificates), F_<arm>_<r>.npz (lattice fields, archive), COMPLETE.

    python fsrun.py --config configs/<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import fscore as F  # noqa: E402  (puts jcp-p1-scaling on sys.path)
import fsselect as FS  # noqa: E402
import p1common as P  # noqa: E402
from p1common import Q, QS, T2, DT0, sha  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(HERE.parent / 'jcp-references'))
import refstore as RS  # noqa: E402

NREC = int(round(.25 / DT0)) + 1
ARMS = {'L-CNR': dict(kind='lspg', dt=DT0), 'G-CN': dict(kind='cn', dt=DT0), 'G-E4a': dict(kind='etd', dt=.0125),
        'G-E4b': dict(kind='etd', dt=.025), 'G-R1': dict(kind='ros', dt=DT0), 'L-CNR-M1152': dict(kind='lspg', dt=DT0)}
GROWTH = 1.05
BAR_Q1, BAR_TT, BAR_Q2, BAR_RHO, GAMMA = 2.5e-4, 2.5e-5, 2.5e-4, .116, 1e-3
TIME_TAG = 't0,t1,t2,t3,t4,t5'


# ------------------------------------------------------------------ settings ----

class SettingM(P.Setting):
    """p1common.Setting with an explicit test count M (A1.11: arm 6); M = 4 r reproduces Setting exactly."""

    def __init__(self, bank, name, Rp, rule, M=None, roles=None, forms=('LSPG', 'GAL'), nrec=NREC):
        self.name, self.Rp, self.policy = name, int(Rp), 'frozen'
        self.M = int(M) if M else 4 * self.Rp
        self.L = bank.L
        assert self.Rp in bank.edges and self.M <= P.MMAX
        ops = bank.ops('frozen')
        self.kx, self.ky = ops['kx'][:self.M], ops['ky'][:self.M]
        self.mode_hash = P.list_hash(self.kx, self.ky)
        self.base = dict(A=ops['Arot'][:self.M, :self.Rp], lam=ops['lam'][:self.M], G=Q.BK.prefix(bank.Grot, bank.edges, self.Rp))
        self.cold, self.trust = bank.cold_trust(self.Rp)
        self.rules = {'main': rule}
        self.rules.update(roles or {})
        self.m, self.data, self.meta, self.table_bytes = {}, {}, {}, {}
        for role, rl in list(self.rules.items()):
            self.add_role(bank, role, rl, forms)
        self.forms, self.nstep, self.nrec = forms, T2.NSTEP, nrec
        self.q = {f: T2.make_split_query(Q, 'point', f, self.Rp, self.L, self.trust, nrec=nrec) for f in ('LSPG', 'GAL')}
        self.q0 = {f: T2.make_split_query(Q, 'point', f, self.Rp, self.L, self.trust, nrec=0) for f in ('LSPG',)}


def anchor_ok(v, level):
    """LSPG anchor validity (A1.3): run valid and every step gamma <= gtol(level), or residual <= tol, or a round-off
    stall (budget / tiny-step exit) with gamma <= 100 gtol (hr-baselines A9.1)."""
    g_, _ = P.tol_of('LSPG', level)
    st = v['stats']
    k = int(st['steps'])
    gn = np.asarray(st['step_gn'])[:k]
    tr = np.asarray(st['step_tolratio'])[:k]
    rs = np.asarray(st['step_reason'])[:k]
    ok = (gn <= g_) | (tr <= 1.) | (np.isin(rs, (0, 2)) & (gn <= 100 * g_))
    return bool(P.run_valid(v) and k > 0 and np.all(ok)), int(np.sum(~ok))


def e0(field, coh, c):
    s = RS.score(np.asarray(field), 2, coh, int(c), 'ST')
    assert (s['cohort'], s['case'], s['kind']) == (coh, int(c), 'ST')
    return dict(e=float(s['e']), e_t=[float(x) for x in s['e_t']], eps_ref=s['eps_ref'], eligible=bool(s['eligible']),
                outcome=s['outcome'], store_version=s['store_version'], metric_id=s['metric_id'])


# ------------------------------------------------------------------ main ----

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
    L = int(cfg['mesh'])
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT), e0_store=RS.store_version(),
               cohorts=P.cohort_hashes(), setup={}, ranks={}, runs=[], blocks={}, memory={}, elapsed={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    bank = P.Bank(L, cfg['edges'])
    ops = bank.ops('frozen')
    rep['setup']['bank'] = dict(edges=bank.edges, seconds=bank.build_seconds, bytes=bank.bytes, mem=P.mem(),
                                list_hash_2048=P.list_hash(ops['kx'], ops['ky']))
    cases = P.case_list(cfg)
    ids = P.select(cases, cfg['cases'])
    dev = [i for i in P.select(cases, 'dev6') if i in ids]
    s256 = L // 256
    U0 = {}
    for i in ids:
        u0n = Q.e.initial(L, cases[i][2])
        U0[i] = (jnp.asarray(u0n), float(cases[i][2][4]), float(np.linalg.norm(u0n[::s256, ::s256])))
    cid = lambda i: f'{cases[i][0]}|{cases[i][1]}'
    rep['setup']['cases'] = [cid(i) for i in ids]
    rep['elapsed']['setup'] = el()
    save()
    print('SETUP', el(), P.mem(), flush=True)

    EV = {}

    def evolver(kind, adv='pre', mutation=None):
        key = (kind, adv, mutation)
        if key not in EV:
            if adv == 'pre':
                EV[key] = jax.jit(F.make_evolve(kind, F.adv_pre, F.jac_pre, mutation=mutation))
            else:
                EV[key] = jax.jit(F.make_evolve(kind, F.make_adv_orig(adv), None, mutation=mutation))
        return EV[key]

    timing_subjects = {}
    REFS = {}
    for spec in cfg['ranks']:
        Rp = int(spec['Rp'])
        t_r = time.perf_counter()
        st = SettingM(bank, f'R{Rp}', Rp, spec['rule'], roles={'qp': spec['qp']})
        nl = st.q['GAL']['nl']
        dg = {role: st.data[(role, 'GAL')] for role in ('main', 'qp')}
        tab, chk = F.precontract(dg['main'])
        tab_qp, chk_qp = F.precontract(dg['qp'], basis=(tab['D'], tab['V']))
        tabs = {'main': tab, 'qp': tab_qp}
        twin_tab = dict(tab, orig=dg['main'])
        manifest = dict(L=L, Rp=Rp, M=st.M, rule=spec['rule'], qp=spec['qp'], mode_hash=st.mode_hash,
                        rule_sha256=st.meta['main']['rule_sha256'], qp_sha256=st.meta['qp']['rule_sha256'],
                        tab_fp=F.table_fingerprint(tab), tab_qp_fp=F.table_fingerprint(tab_qp))
        rr = rep['ranks'][str(Rp)] = dict(m=st.m, M=st.M, trust=st.trust, checks=chk, checks_qp=chk_qp, manifest=manifest,
                                          arms={})
        cert_cn, cert_etd, cert_ros = F.make_cert_cn(nl), F.make_cert_etd(nl), F.make_cert_ros(nl)
        lcert = T2.make_certifier(nl, Rp)

        def check_tab(t, role):                                  # C12 guard (control roles such as 'c7' are unchecked)
            if role not in ('main', 'qp'):
                return
            if F.table_fingerprint(t) != manifest['tab_fp' if role == 'main' else 'tab_qp_fp']:
                raise ValueError('precontracted table differs from the manifest (substituted or wrong-mesh data)')
            st.check(role, bank)

        def g_run(kind, dt, ci, role='main', level='prod', mutation=None, adv='pre'):
            u0, nu, n0 = U0[ci]
            t = tabs[role] if adv == 'pre' else dict(tabs[role], orig=dg[role])
            if adv == 'pre':
                check_tab(t, role)
            qg = st.q['GAL']
            w0, scale = qg['init'](u0, st.cold)
            ev = evolver(kind, 'pre' if adv == 'pre' else nl, mutation)
            W, s = ev(w0, jnp.float64(nu), t, F.sched(dt, level=level))
            fields = qg['decode'](W, st.data[(role, 'GAL')])
            jax.block_until_ready(fields)
            return dict(W=np.asarray(W), w0=np.asarray(w0), Z=np.asarray(s['Z']), steps=int(s['steps']),
                        it=np.asarray(s['it']), tolratio=np.asarray(s['tolratio']), reason=np.asarray(s['reason']),
                        stage_max=float(s['stage_max']), growth=float(s['growth']), z_out=np.asarray(s['z_out']),
                        f=P.restrict(fields, L), nu=nu, n0=n0)

        def g_valid(v, kind):
            k = v['steps']
            ok = bool(np.isfinite(v['W']).all() and np.isfinite(v['Z'][:k + 1]).all() and np.isfinite(v['f']).all()
                      and np.isfinite(v['stage_max']))
            if kind == 'cn':
                ok = ok and not np.any(v['reason'][:k] == 3)
            return ok

        def g_cert(v, kind, dt, role='main', level='prod'):
            k, nu = v['steps'], v['nu']
            Z = jnp.asarray(v['Z'][:k + 1])
            d_, t_ = dg[role], tabs[role]
            if kind == 'cn':
                rn, nA, gam = (np.asarray(x) for x in cert_cn(Z, jnp.float64(nu), d_, t_['V'], jnp.float64(dt), k))
                bad = ~(rn <= (1 + 1e-6) * F.TOL_CN[level] * nA)
                bud = (v['reason'][:k] == 0)
                return dict(unconverged=int(np.sum(bad)), steps=k, worst_ratio=float(np.max(rn / (F.TOL_CN[level] * nA))),
                            gamma_max=float(np.nanmax(gam)), budget_steps=int(np.sum(bud)),
                            budget_unconverged=int(np.sum(bud & bad)), iters=int(np.sum(v['it'][:k])),
                            it_max=int(np.max(v['it'][:k])), ok=bool(np.sum(bad) <= .01 * k))
            if kind == 'etd':
                co = F.etd_coefs_indep(-nu * np.asarray(t_['D']), dt)
                mis = np.asarray(cert_etd(Z, d_, t_['V'], {kk: jnp.asarray(vv) for kk, vv in co.items()}, k))
                return dict(worst_mismatch=float(np.max(mis)), steps=k, failures=int(np.sum(~(mis <= 1e-10))),
                            ok=bool(np.all(mis <= 1e-10)))
            jr, lr = (np.asarray(x) for x in cert_ros(Z, jnp.float64(nu), d_, t_['V'], t_, jnp.float64(dt), k))
            return dict(worst_jac=float(np.max(jr)), worst_lin=float(np.max(lr)), steps=k,
                        failures=int(np.sum(~((jr <= 1e-9) & (lr <= 1e-9)))), ok=bool(np.all((jr <= 1e-9) & (lr <= 1e-9))))

        def own_rho(Wm):
            """A1.4: both forces at the same main-rule output states (p1common.rho semantics)."""
            return [P.rho(np.asarray(nl(jnp.asarray(Wm[kk]), dg['main'])), np.asarray(nl(jnp.asarray(Wm[kk]), dg['qp'])))
                    for kk in range(6)]

        def lspg_run(stx, ci, role='main', level='prod'):
            u0, nu, n0 = U0[ci]
            v = P.run_query(bank, stx, role, 'LSPG', 'CNR', DT0, level, u0, nu, record=True)
            jax.block_until_ready(v)
            return v, P.restrict(v['fields'], L)

        arm_list = [a_ for a_ in cfg['arms'] if a_ != 'L-CNR-M1152' or Rp == 384]
        settings = {'main': st}
        if 'L-CNR-M1152' in arm_list:
            settings['M1152'] = SettingM(bank, 'R384M1152', Rp, spec['rule'], M=1152, roles={'qp': spec['qp']},
                                         forms=('LSPG',))
        for arm in arm_list:
            t_arm = time.perf_counter()
            kind, dt = ARMS[arm]['kind'], ARMS[arm]['dt']
            n = len(ids)
            A_ = dict(case_ids=np.array([cid(i) for i in ids]), W=np.full((n, 6, Rp), np.nan), w0=np.full((n, Rp), np.nan),
                      W_qp=np.full((n, 6, Rp), np.nan), rho=np.full((n, 6), np.nan), n0=np.full(n, np.nan),
                      nu=np.full(n, np.nan), W_tight=np.full((n, 6, Rp), np.nan))
            Fl = dict(case_ids=A_['case_ids'], f257=np.full((n, 6, 257, 257), np.nan))
            rows = []
            stx = settings['M1152'] if arm == 'L-CNR-M1152' else st
            for j, ci in enumerate(ids):
                coh, c, _ = cases[ci]
                u0, nu, n0 = U0[ci]
                row = dict(arm=arm, Rp=Rp, case=cid(ci), dt=dt)
                if kind == 'lspg':
                    v, f = lspg_run(stx, ci)
                    stt = v['stats']
                    states = np.asarray(stt['states'])
                    g, _ = lcert(stt['states'], nu, stx.data[('main', 'LSPG')], stx.sched('LSPG', 'CNR', DT0, 'prod'))
                    g = np.asarray(g)
                    Ast = np.asarray(stx.base['A'])
                    nA = np.linalg.norm(states @ Ast.T, axis=1)
                    growth = float(np.max(nA / nA[0])) if np.isfinite(nA).all() else float('inf')
                    W = np.asarray(v['W'])
                    row.update(valid=P.run_valid(v), nonstationary=int(np.sum(~(g <= GAMMA))), steps=len(g),
                               gcert_ok=P.gcert_ok(g, np.asarray(stt['step_gn'])[:len(g)]), it_sum=int(stt['it_sum']),
                               budget_exits=P.budget_exits(v), growth=growth, stage_finite=bool(np.isfinite(states).all()))
                    A_.setdefault('states', np.full((n, NREC, Rp), np.nan))[j] = states
                    A_.setdefault('gamma', np.full((n, NREC - 1), np.nan))[j] = g
                    vt, ft = lspg_run(stx, ci, level='tight')
                    aok, abad = anchor_ok(vt, 'tight')
                    A_['W_tight'][j] = np.asarray(vt['W'])
                    row.update(dist_tight=P.fdist(f, ft, n0), tight_anchor_ok=aok, tight_anchor_bad_steps=abad)
                    if ci in dev:
                        vtt, ftt = lspg_run(stx, ci, level='tighter')
                        aok2, abad2 = anchor_ok(vtt, 'tighter')
                        row.update(dist_tight_tighter=P.fdist(ft, ftt, n0), tighter_anchor_ok=aok2, tighter_anchor_bad_steps=abad2)
                        del vtt
                    vq, fq = lspg_run(stx, ci, role='qp')
                    A_['W_qp'][j] = np.asarray(vq['W'])
                    row.update(qp_valid=P.run_valid(vq), dist_qp=P.fdist(f, fq, n0))
                    if arm == 'L-CNR':
                        A_['rho'][j] = own_rho(W)
                    else:
                        A_['rho'][j] = [P.rho(np.asarray(nl(jnp.asarray(W[kk]), stx.data[('main', 'LSPG')])),
                                              np.asarray(nl(jnp.asarray(W[kk]), stx.data[('qp', 'LSPG')]))) for kk in range(6)]
                    del v, vt, vq
                else:
                    v = g_run(kind, dt, ci)
                    W, f = v['W'], v['f']
                    cert = g_cert(v, kind, dt)
                    row.update(valid=g_valid(v, kind), growth=v['growth'], stage_max=v['stage_max'], steps=v['steps'],
                               stage_finite=bool(np.isfinite(v['stage_max'])), cert=cert,
                               out_vs_states=float(np.max(np.abs(v['z_out'][1:] - v['Z'][[v['steps'] // 5 * k for k in range(1, 6)]]))))
                    S_ = v['steps']
                    A_.setdefault('Z', np.full((n, S_ + 1, Rp), np.nan))[j] = v['Z'][:S_ + 1]
                    if kind == 'cn':
                        A_.setdefault('it', np.full((n, S_), -1))[j] = v['it'][:S_]
                        A_.setdefault('tolratio', np.full((n, S_), np.nan))[j] = v['tolratio'][:S_]
                        vt = g_run(kind, dt, ci, level='tight')
                        ct = g_cert(vt, kind, dt, level='tight')
                        A_['W_tight'][j] = vt['W']
                        row.update(dist_tight=P.fdist(f, vt['f'], n0), tight_anchor_ok=bool(g_valid(vt, kind) and ct['unconverged'] == 0),
                                   tight_anchor_bad_steps=ct['unconverged'])
                        if ci in dev:
                            vtt = g_run(kind, dt, ci, level='tighter')
                            ctt = g_cert(vtt, kind, dt, level='tighter')
                            row.update(dist_tight_tighter=P.fdist(vt['f'], vtt['f'], n0),
                                       tighter_anchor_ok=bool(g_valid(vtt, kind) and ctt['unconverged'] == 0),
                                       tighter_anchor_bad_steps=ctt['unconverged'])
                    elif ci in dev:                                     # dev6 twin (un-precontracted RHS), A1.3
                        vw = g_run(kind, dt, ci, adv='orig')
                        row.update(twin_dist=P.fdist(f, vw['f'], n0), twin_valid=g_valid(vw, kind))
                    vq = g_run(kind, dt, ci, role='qp')
                    cq = g_cert(vq, kind, dt, role='qp')
                    A_['W_qp'][j] = vq['W']
                    row.update(qp_valid=bool(g_valid(vq, kind) and cq['ok']), dist_qp=P.fdist(f, vq['f'], n0))
                    A_['rho'][j] = own_rho(W)
                    if ci in dev:                                       # dev6 step halving (time-error estimate)
                        vh = g_run(kind, dt / 2, ci)
                        row.update(halving_dist=P.fdist(f, vh['f'], n0), halving_valid=g_valid(vh, kind))
                row['rho_max'] = float(np.nanmax(A_['rho'][j])) if np.isfinite(np.nanmax(A_['rho'][j])) else float('inf')
                row['e0'] = e0(f, coh, c)
                row['stable'] = bool(row['growth'] <= GROWTH and row['stage_finite'])
                A_['W'][j], A_['n0'][j], A_['nu'][j] = W, n0, nu
                if ci in dev:
                    REFS[(arm, Rp, ci)] = np.asarray(W)
                Fl['f257'][j] = f
                rows.append(row)
                rep['runs'].append(row)
                print('RUN', arm, Rp, cid(ci), round(row['e0']['e'], 6), el(), flush=True)
            A_.update(A=np.asarray(stx.base['A']), lam=np.asarray(stx.base['lam']), kx=np.asarray(stx.kx), ky=np.asarray(stx.ky),
                      D=np.asarray(tab['D']), V=np.asarray(tab['V']), dt=dt, Rp=Rp, mesh=L, rule=spec['rule'], qp=spec['qp'],
                      M=stx.M, time_tag=TIME_TAG)
            np.savez_compressed(out / f'A_{arm}_{Rp}.npz', **A_)
            np.savez_compressed(out / f'F_{arm}_{Rp}.npz', **Fl)
            rr['arms'][arm] = dict(n=len(rows), seconds=time.perf_counter() - t_arm)
            rep['elapsed'][f'{arm}|{Rp}'] = time.perf_counter() - t_arm
            save()
            del A_, Fl
            gc.collect()

        # ---------------------------------------------------- dev6 blocks: F5 order, stability classification ----
        blk = rep['blocks'].setdefault(str(Rp), {})
        if cfg.get('order', True) and 'G-R1' in arm_list:
            t_b = time.perf_counter()
            od = []
            for ci in dev:
                ref8 = g_run('etd', DT0 / 8, ci)
                ref16 = g_run('etd', DT0 / 16, ci)
                d_ref = P.fdist(ref8['f'], ref16['f'], ref8['n0'])
                lev = {}
                for fct in (1, 2, 4):
                    vr = g_run('ros', DT0 / fct, ci)
                    lev[fct] = P.fdist(vr['f'], ref8['f'], ref8['n0'])
                x = np.log(np.array([DT0, DT0 / 2, DT0 / 4]))
                y = np.log(np.array([lev[1], lev[2], lev[4]]))
                slope = float(np.polyfit(x, y, 1)[0]) if np.isfinite(y).all() else float('nan')
                od.append(dict(case=cid(ci), ref_dist=d_ref, levels=lev, slope=slope,
                               ref_converged=bool(d_ref <= 1e-3 * min(lev.values()))))
            blk['order_F5'] = od
            rep['elapsed'][f'order|{Rp}'] = time.perf_counter() - t_b
        tripped = [r_ for r_ in rows_all(rep, Rp) if not r_['stable']]
        cls = []
        for r_ in tripped[:cfg.get('max_classify', 8)]:
            ci = ids[[cid(i) for i in ids].index(r_['case'])]
            kind, dt = ARMS[r_['arm']]['kind'], ARMS[r_['arm']]['dt']
            if kind == 'lspg':
                cls.append(dict(arm=r_['arm'], case=r_['case'], note='LSPG refined classification not run (descriptive)'))
                continue
            v8, v16 = (g_run(kind, dt / f_, ci) for f_ in (8, 16))
            ok_ = all(g_valid(v_, kind) and g_cert(v_, kind, dt / f_)['ok'] for v_, f_ in ((v8, 8), (v16, 16)))
            cls.append(dict(arm=r_['arm'], case=r_['case'], g8=v8['growth'], g16=v16['growth'], refined_ok=ok_,
                            label=FS.classify_growth(v8['growth'], v16['growth'], ok_)))
        blk['stability_classification'] = cls
        save()

        # ---------------------------------------------------- controls (smoke) ----
        if cfg.get('controls'):
            import fscontrols as FC
            t_b = time.perf_counter()
            blk['controls'] = FC.run(cfg, st, dg, tabs, nl, g_run, g_valid, g_cert, lspg_run, U0, dev, ids, cases, bank, L,
                                     evolver, settings)
            rep['elapsed'][f'controls|{Rp}'] = time.perf_counter() - t_b
            save()

        # ---------------------------------------------------- timing subjects of this rank ----
        if cfg.get('timing'):
            import fstiming as FT
            U0d = {ci: U0[ci] for ci in dev}
            refs_r = {(a_, ci): W_ for (a_, r_, ci), W_ in REFS.items() if r_ == Rp}
            timing_subjects.update(FT.bank_subjects(st, settings, tabs, arm_list, evolver, U0d, refs_r, Rp))
        rep['memory'][f'R{Rp}'] = P.mem()
        rep['elapsed'][f'rank|{Rp}'] = time.perf_counter() - t_r
        if not cfg.get('timing'):
            del st, settings, tabs, dg
            gc.collect()
        save()

    # ---------------------------------------------------- POD block ----
    if cfg.get('pod'):
        import podarm as PA
        t_b = time.perf_counter()
        rep['pod'] = PA.run(cfg, L, cases, ids, dev, U0, out, e0, el)
        rep['elapsed']['pod'] = time.perf_counter() - t_b
        save()

    # ---------------------------------------------------- timing ----
    if cfg.get('timing'):
        import fstiming as FT
        t_b = time.perf_counter()
        subj = dict(timing_subjects)
        if cfg.get('pod'):
            subj.update(FT.pod_subjects(PA.timing_subjects()))
        rep['timing'] = FT.run(cfg, subj, U0, dev, cases, lambda screen: select_pairs(rep, screen, dev, cases, cfg))
        rep['elapsed']['timing'] = time.perf_counter() - t_b
        save()
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('FSRUN COMPLETE', el(), flush=True)


def subject_rows(rep):
    """{subject id: [{case, valid, stable, e}]} for the bank arms and the POD pool (in-job selection inputs)."""
    out = {}
    for r_ in rep['runs']:
        out.setdefault(f"{r_['arm']}|{r_['Rp']}", []).append(dict(case=r_['case'], valid=r_['valid'], stable=r_['stable'],
                                                                  e=r_['e0']['e']))
    pool = {}
    for grp in ('members', 'native'):
        for mid, m in (rep.get('pod') or {}).get(grp, {}).items():
            pool[mid] = [dict(case=x['case'], valid=x['valid'], stable=x['stable'], e=x['e0']['e']) for x in m['rows']]
    return out, pool


def select_pairs(rep, screen, dev, cases, cfg):
    """A3.6 in-job frozen selection: B*(X) for every bank subject X (dev6 only), plus the F5 pairs (G-CN, G-R1)."""
    devc = [f'{cases[i][0]}|{cases[i][1]}' for i in dev]
    bank_rows, pool = subject_rows(rep)
    pairs, sel = [], {}
    for X, rows in bank_rows.items():
        B = FS.matched_competitor(rows, pool, screen, devc) if pool else None
        sel[X] = B
        if B is not None:
            pairs.append((X, B))
    for r_ in [s_['Rp'] for s_ in cfg['ranks']]:
        if f'G-CN|{r_}' in bank_rows and f'G-R1|{r_}' in bank_rows:
            pairs.append((f'G-CN|{r_}', f'G-R1|{r_}'))
    rep['selection'] = dict(matched=sel, pairs=pairs, screen={k: v for k, v in screen.items()})
    return pairs


def rows_all(rep, Rp):
    return [r_ for r_ in rep['runs'] if r_['Rp'] == Rp]


if __name__ == '__main__':
    main()
