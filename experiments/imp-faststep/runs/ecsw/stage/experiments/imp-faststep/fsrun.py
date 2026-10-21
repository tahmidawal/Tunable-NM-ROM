"""imp-faststep job driver (DESIGN.md + A1-A5): one mesh per job, one GPU, every arm in the same job.

Per rank spec (r, rule, successor rule) on the frozen-policy bank of the mesh, on every case of the configured set:
  arm 1  L-CNR    p1-scaling LSPG CN-R at dt0, split recording program (init | evolve | decode), fresh autodiff gamma
  arm 2  G-CN     precontracted Galerkin CN (Newton), fresh-residual certificate
  arm 3a G-E4a    ETDRK4 at 0.0125, per-step reconstruction certificate        arm 3b G-E4b  the same at 0.025
  arm 4  G-R1     one-pass linearised CN at dt0, fresh Jacobian + saved-increment certificate
  arm 6  L-CNR-M1152 (r = 384 only)
Per run: map/identity gates (A5.1), stability monitor, E0 ST score with provenance (A5.6), successor-rule rollout and
own-state rho (every case), anchors (LSPG and CN: tight every case, tighter dev6, fresh certificates, A5.2), dev6 twins
(arms 3, 4), dev6 step halving (every arm). Blocks: F5 order (dev6), stability classification of every tripped run
(A5.5), controls (fscontrols), POD (podarm, if configured), in-job best stepper and timing (fstiming, A3.6/A5.7-A5.9).
COMPLETE marks execution only; acceptance is checks.py (A5.10).

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
from jax.scipy.linalg import solve_triangular  # noqa: E402

sys.path.insert(0, str(HERE.parent / 'jcp-references'))
import refstore as RS  # noqa: E402

NREC = int(round(.25 / DT0)) + 1
ARMS = {'L-CNR': dict(kind='lspg', dt=DT0), 'G-CN': dict(kind='cn', dt=DT0), 'G-E4a': dict(kind='etd', dt=.0125),
        'G-E4b': dict(kind='etd', dt=.025), 'G-R1': dict(kind='ros', dt=DT0), 'L-CNR-M1152': dict(kind='lspg', dt=DT0)}
GROWTH = 1.05
GAMMA = 1e-3
TIME_TAG = 't0,t1,t2,t3,t4,t5'
GATE = 1e-10


class IdentityError(ValueError):
    pass


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
        self.q0 = {'LSPG': T2.make_split_query(Q, 'point', 'LSPG', self.Rp, self.L, self.trust, nrec=0)}


def make_e0(cfg):
    pin_store, pin_metric = cfg['e0_store_version'], cfg['e0_metric_id']
    man = RS._manifest()

    def e0(field, rec):
        """A5.6: the field carries its case id and time tag; both, the store version, the metric and the reference kind
        are checked; the reference sha256 is recorded."""
        coh, c = rec['requested']
        if rec['case_id'] != f'{coh}|{c}':
            raise IdentityError(f"case swap: field {rec['case_id']} scored as {coh}|{c}")
        if rec['time_tag'] != TIME_TAG:
            raise IdentityError(f"time swap: {rec['time_tag']}")
        s = RS.score(np.asarray(field), 2, coh, int(c), 'ST')
        if (s['cohort'], s['case'], s['kind']) != (coh, int(c), 'ST') or s['store_version'] != pin_store or \
                s['metric_id'] != pin_metric:
            raise IdentityError('score identity differs from the request or the pinned store / metric')
        return dict(e=float(s['e']), e_t=[float(x) for x in s['e_t']], eps_ref=s['eps_ref'], eligible=bool(s['eligible']),
                    outcome=s['outcome'], reference_sha256=man[f'2d/{coh}/ST/c{int(c):03d}.npz'], case_id=rec['case_id'])
    return e0


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
    assert RS.store_version() == cfg['e0_store_version'], 'E0 store differs from the pinned version'
    e0 = make_e0(cfg)
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT), e0_store=RS.store_version(),
               cohorts=P.cohort_hashes(), setup={}, ranks={}, runs=[], blocks={}, memory={}, elapsed={}, executed=False)
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
    rec_of = lambda i: dict(case_id=cid(i), time_tag=TIME_TAG, requested=(cases[i][0], cases[i][1]))
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
    W0REF = {}
    for spec in cfg['ranks']:
        Rp = int(spec['Rp'])
        t_r = time.perf_counter()
        st = SettingM(bank, f'R{Rp}', Rp, spec['rule'], roles={'qp': spec['qp']})
        nl = st.q['GAL']['nl']
        dg = {role: st.data[(role, 'GAL')] for role in ('main', 'qp')}
        tab, chk = F.precontract(dg['main'])
        tab_qp, chk_qp = F.precontract(dg['qp'], basis=(tab['D'], tab['V']))
        for c_ in (chk, chk_qp):                                           # A5.1 table gates
            assert c_['eig_resid'] <= GATE and c_['v_orth'] <= GATE and c_['qr_orth'] <= GATE and c_['D_min'] > 0, c_
        tabs = {'main': tab, 'qp': tab_qp}
        manifest = dict(L=L, Rp=Rp, M=st.M, rule=spec['rule'], qp=spec['qp'], mode_hash=st.mode_hash,
                        rule_sha256=st.meta['main']['rule_sha256'], qp_sha256=st.meta['qp']['rule_sha256'],
                        fp={'main': F.table_fingerprint(tab), 'qp': F.table_fingerprint(tab_qp)})
        rr = rep['ranks'][str(Rp)] = dict(m=st.m, M=st.M, trust=st.trust, checks=chk, checks_qp=chk_qp, manifest=manifest,
                                          arms={})
        cert_cn, cert_etd, cert_ros = F.make_cert_cn(nl), F.make_cert_etd(nl), F.make_cert_ros(nl)
        lcert = T2.make_certifier(nl, Rp)
        lrec801 = {}

        def check_tab(t, role):                                  # C12 guard (control roles such as 'c7' are unchecked)
            if role not in manifest['fp']:
                return
            if F.table_fingerprint(t) != manifest['fp'][role]:
                raise ValueError('precontracted table differs from the manifest (substituted or wrong-mesh data)')
            st.check(role, bank)

        def g_run(kind, dt, ci, role='main', level='prod', mutation=None, adv='pre', tab_override=None):
            u0, nu, n0 = U0[ci]
            t = tabs[role] if tab_override is None else tab_override
            if adv != 'pre':
                t = dict(t, orig=dg[role])
            elif tab_override is None:
                check_tab(t, role)
            qg = st.q['GAL']
            w0, scale = qg['init'](u0, st.cold)
            ev = evolver(kind, 'pre' if adv == 'pre' else nl, mutation)
            W, s = ev(w0, jnp.float64(nu), t, F.sched(dt, level=level))
            fields = qg['decode'](W, st.data[(role, 'GAL')])
            jax.block_until_ready(fields)
            k = int(s['steps'])
            Z = np.asarray(s['Z'])[:k + 1]
            Wn, w0n = np.asarray(W), np.asarray(w0)
            Rm, Vt = dg[role]['Rm'], t['V']
            z0r = np.asarray(Vt.T @ (Rm @ w0))
            keep = k // 5
            cr = np.asarray(jax.vmap(lambda z: solve_triangular(Rm, Vt @ z, lower=False))(jnp.asarray(Z[[keep * j for j in range(6)]])))
            return dict(W=Wn, w0=w0n, Z=Z, steps=k, it=np.asarray(s['it'])[:k], tolratio=np.asarray(s['tolratio'])[:k],
                        reason=np.asarray(s['reason'])[:k], stage_max=float(s['stage_max']), growth=float(s['growth']),
                        z_out=np.asarray(s['z_out']), f=P.restrict(fields, L), nu=nu, n0=n0,
                        z0_check=float(np.linalg.norm(Z[0] - z0r) / (np.linalg.norm(Z[0]) + 1e-300)),
                        W_check=float(np.max(np.linalg.norm(Wn - cr, axis=1) / (np.linalg.norm(Wn, axis=1) + 1e-300))),
                        out_vs_states=float(np.max(np.abs(np.asarray(s['z_out']) - Z[[keep * j for j in range(6)]]))),
                        h_ev=F.sha(np.concatenate([Wn.ravel(), Z.ravel()])))

        def g_cert(v, kind, dt, role='main', level='prod'):
            k, nu = v['steps'], v['nu']
            Z = jnp.asarray(v['Z'])
            d_, t_ = dg[role], tabs.get(role, tabs['main'])
            if kind == 'cn':
                rn, nA, gam = (np.asarray(x) for x in cert_cn(Z, jnp.float64(nu), d_, t_['V'], jnp.float64(dt), k))
                bad = ~(rn <= (1 + 1e-6) * F.TOL_CN[level] * nA)
                bud = v['reason'] == 0
                return dict(unconverged=int(np.sum(bad)), steps=k, worst_ratio=float(np.max(rn / (F.TOL_CN[level] * nA))),
                            gamma_max=float(np.nanmax(gam)), budget_steps=int(np.sum(bud)),
                            budget_unconverged=int(np.sum(bud & bad)), iters=int(np.sum(v['it'])),
                            it_max=int(np.max(v['it'])), ok=bool(np.sum(bad) <= .01 * k), arrays=dict(rn=rn, nA=nA, gamma=gam))
            if kind == 'etd':
                co = F.etd_coefs_indep(-nu * np.asarray(t_['D']), dt)
                mis = np.asarray(cert_etd(Z, d_, t_['V'], {kk: jnp.asarray(vv) for kk, vv in co.items()}, k))
                return dict(worst_mismatch=float(np.max(mis)), steps=k, failures=int(np.sum(~(mis <= 1e-10))),
                            ok=bool(np.all(mis <= 1e-10)), arrays=dict(mismatch=mis))
            jr, lr = (np.asarray(x) for x in cert_ros(Z, jnp.float64(nu), d_, t_['V'], t_, jnp.float64(dt), k))
            return dict(worst_jac=float(np.max(jr)), worst_lin=float(np.max(lr)), steps=k,
                        failures=int(np.sum(~((jr <= 1e-9) & (lr <= 1e-9)))), ok=bool(np.all((jr <= 1e-9) & (lr <= 1e-9))),
                        arrays=dict(jac=jr, lin=lr))

        def g_valid(v, kind, cert=None, ci=None):
            ok = bool(np.isfinite(v['W']).all() and np.isfinite(v['Z']).all() and np.isfinite(v['f']).all()
                      and np.isfinite(v['stage_max']) and v['z0_check'] <= 1e-12 and v['W_check'] <= 1e-12
                      and v['out_vs_states'] == 0.)
            if kind == 'cn':
                ok = ok and not np.any(v['reason'] == 3)
                if cert is not None:
                    ok = ok and cert['budget_unconverged'] == 0                       # A5.3
            if ci is not None and ci in W0REF.get(Rp, {}):
                ok = ok and bool(np.array_equal(v['w0'], W0REF[Rp][ci]))             # A5.1: same c0 as arm 1
            return ok

        def own_rho(Wm, dm, dq):
            return [P.rho(np.asarray(nl(jnp.asarray(Wm[kk]), dm)), np.asarray(nl(jnp.asarray(Wm[kk]), dq))) for kk in range(6)]

        def lspg_run(stx, ci, role='main', level='prod', dt=DT0, record=True):
            """Split recording program (init | evolve | decode) of p1core; the timed evolve is this same program."""
            u0, nu, n0 = U0[ci]
            stx.check(role, bank)
            q = (stx.q if record else stx.q0)['LSPG']
            d = stx.data[(role, 'LSPG')]
            sch = stx.sched('LSPG', 'CNR', dt, level)
            if record:
                assert int(round(.25 / dt)) + 1 == stx.nrec
            w0, scale = q['init'](u0, stx.cold)
            W, s = q['evolve'](w0, nu, scale, d, sch)
            fields = q['decode'](W, d)
            jax.block_until_ready(fields)
            v = dict(W=W, w0=w0, stats=s, fields=fields, scale=float(scale))
            return v, P.restrict(fields, L), sch

        def lspg_fresh(stx, v, ci, sch, level, role='main'):
            """Fresh autodiff gamma of every step (p1core certifier) and the anchor rule (A5.2)."""
            u0, nu, _ = U0[ci]
            g, rn = lcert(v['stats']['states'], nu, stx.data[(role, 'LSPG')], sch)
            g, rn = np.asarray(g), np.asarray(rn)
            gt, tf = P.tol_of('LSPG', level)
            ok_steps = (g <= gt) | (rn <= tf * v['scale'])
            return g, rn, bool(P.run_valid(v) and np.all(ok_steps)), int(np.sum(~ok_steps))

        def lspg_growth(stx, states):
            Ast = np.asarray(stx.base['A'])
            nA = np.linalg.norm(np.asarray(states) @ Ast.T, axis=1)
            return float(np.max(nA / nA[0])) if np.isfinite(nA).all() else float('inf')

        arm_list = [a_ for a_ in cfg['arms'] if a_ != 'L-CNR-M1152' or Rp == 384]
        assert arm_list[0] == 'L-CNR'
        settings = {'main': st}
        if 'L-CNR-M1152' in arm_list:
            settings['M1152'] = SettingM(bank, 'R384M1152', Rp, spec['rule'], M=1152, roles={'qp': spec['qp']}, forms=('LSPG',))
        rows_rank = []
        for arm in arm_list:
            t_arm = time.perf_counter()
            kind, dt = ARMS[arm]['kind'], ARMS[arm]['dt']
            n = len(ids)
            A_ = dict(case_ids=np.array([cid(i) for i in ids]), W=np.full((n, 6, Rp), np.nan), w0=np.full((n, Rp), np.nan),
                      W_qp=np.full((n, 6, Rp), np.nan), rho=np.full((n, 6), np.nan), n0=np.full(n, np.nan),
                      nu=np.full(n, np.nan), W_tight=np.full((n, 6, Rp), np.nan),
                      dev_case_ids=np.array([cid(i) for i in dev]), W_tighter=np.full((len(dev), 6, Rp), np.nan),
                      W_half=np.full((len(dev), 6, Rp), np.nan))
            Fl = dict(case_ids=A_['case_ids'], f257=np.full((n, 6, 257, 257), np.nan))
            stx = settings['M1152'] if arm == 'L-CNR-M1152' else st
            for j, ci in enumerate(ids):
                coh, c, _ = cases[ci]
                u0, nu, n0 = U0[ci]
                row = dict(arm=arm, Rp=Rp, case=cid(ci), dt=dt)
                if kind == 'lspg':
                    v, f, sch = lspg_run(stx, ci)
                    stt = v['stats']
                    g, rn, _, _ = lspg_fresh(stx, v, ci, sch, 'prod')
                    W = np.asarray(v['W'])
                    w0n = np.asarray(v['w0'])
                    if arm == 'L-CNR':
                        W0REF.setdefault(Rp, {})[ci] = w0n
                    row.update(valid=P.run_valid(v), nonstationary=int(np.sum(~(g <= GAMMA))), steps=len(g),
                               gcert_ok=P.gcert_ok(g, np.asarray(stt['step_gn'])[:len(g)]), it_sum=int(stt['it_sum']),
                               budget_exits=P.budget_exits(v), growth=lspg_growth(stx, stt['states']),
                               stage_finite=bool(np.isfinite(np.asarray(stt['states'])).all()))
                    A_.setdefault('states', np.full((n, NREC, Rp), np.nan))[j] = np.asarray(stt['states'])
                    A_.setdefault('gamma', np.full((n, NREC - 1), np.nan))[j] = g
                    A_.setdefault('gn_rec', np.full((n, NREC - 1), np.nan))[j] = np.asarray(stt['step_gn'])[:NREC - 1]
                    A_.setdefault('reason', np.full((n, NREC - 1), -1))[j] = np.asarray(stt['step_reason'])[:NREC - 1]
                    h_ev = F.sha(np.concatenate([W.ravel(), np.asarray(stt['states']).ravel(),
                                                 np.asarray(stt['step_reason'], float), np.asarray(stt['step_it'], float)]))
                    vt, ft, scht = lspg_run(stx, ci, level='tight')
                    gt_, _, aok, abad = lspg_fresh(stx, vt, ci, scht, 'tight')
                    A_['W_tight'][j] = np.asarray(vt['W'])
                    A_.setdefault('gamma_tight', np.full((n, NREC - 1), np.nan))[j] = gt_
                    row.update(dist_tight=P.fdist(f, ft, n0), tight_anchor_ok=aok, tight_anchor_bad_steps=abad)
                    if ci in dev:
                        k_ = dev.index(ci)
                        vtt, ftt, schtt = lspg_run(stx, ci, level='tighter')
                        gtt_, _, aok2, abad2 = lspg_fresh(stx, vtt, ci, schtt, 'tighter')
                        A_['W_tighter'][k_] = np.asarray(vtt['W'])
                        A_.setdefault('gamma_tighter', np.full((len(dev), NREC - 1), np.nan))[k_] = gtt_
                        row.update(dist_tight_tighter=P.fdist(ft, ftt, n0), tighter_anchor_ok=aok2, tighter_anchor_bad_steps=abad2)
                        vh, fh, _ = lspg_run(stx, ci, dt=DT0 / 2, record=False)
                        A_['W_half'][k_] = np.asarray(vh['W'])
                        row.update(halving_dist=P.fdist(f, fh, n0), halving_valid=P.run_valid(vh))
                        del vtt, vh
                    vq, fq, schq = lspg_run(stx, ci, role='qp')
                    gq_, _, _, _ = lspg_fresh(stx, vq, ci, schq, 'prod', role='qp')
                    A_['W_qp'][j] = np.asarray(vq['W'])
                    row.update(qp_valid=bool(P.run_valid(vq)), qp_nonstationary=int(np.sum(~(gq_ <= GAMMA))),
                               dist_qp=P.fdist(f, fq, n0))
                    A_['rho'][j] = own_rho(W, stx.data[('main', 'LSPG')], stx.data[('qp', 'LSPG')])
                    del v, vt, vq
                else:
                    v = g_run(kind, dt, ci)
                    W, f, w0n = v['W'], v['f'], v['w0']
                    cert = g_cert(v, kind, dt)
                    arrs = cert.pop('arrays')
                    h_ev = v['h_ev']
                    row.update(valid=g_valid(v, kind, cert, ci), growth=v['growth'], stage_max=v['stage_max'], steps=v['steps'],
                               stage_finite=bool(np.isfinite(v['stage_max'])), cert=cert, z0_check=v['z0_check'],
                               W_check=v['W_check'], out_vs_states=v['out_vs_states'],
                               c0_identical=bool(np.array_equal(w0n, W0REF[Rp][ci])))
                    S_ = v['steps']
                    A_.setdefault('Z', np.full((n, S_ + 1, Rp), np.nan))[j] = v['Z']
                    for nm_, ar_ in arrs.items():
                        A_.setdefault(f'cert_{nm_}', np.full((n, S_), np.nan))[j] = ar_
                    if kind == 'cn':
                        A_.setdefault('it', np.full((n, S_), -1))[j] = v['it']
                        A_.setdefault('reason', np.full((n, S_), -1))[j] = v['reason']
                        vt = g_run(kind, dt, ci, level='tight')
                        ct = g_cert(vt, kind, dt, level='tight')
                        ct.pop('arrays')
                        A_['W_tight'][j] = vt['W']
                        A_.setdefault('Z_tight', np.full((n, S_ + 1, Rp), np.nan))[j] = vt['Z']
                        row.update(dist_tight=P.fdist(f, vt['f'], n0),
                                   tight_anchor_ok=bool(g_valid(vt, kind, ct) and ct['unconverged'] == 0),
                                   tight_anchor_bad_steps=ct['unconverged'])
                        if ci in dev:
                            k_ = dev.index(ci)
                            vtt = g_run(kind, dt, ci, level='tighter')
                            ctt = g_cert(vtt, kind, dt, level='tighter')
                            ctt.pop('arrays')
                            A_['W_tighter'][k_] = vtt['W']
                            row.update(dist_tight_tighter=P.fdist(vt['f'], vtt['f'], n0),
                                       tighter_anchor_ok=bool(g_valid(vtt, kind, ctt) and ctt['unconverged'] == 0),
                                       tighter_anchor_bad_steps=ctt['unconverged'])
                    elif ci in dev:                                     # dev6 twin (un-precontracted RHS), A1.3
                        vw = g_run(kind, dt, ci, adv='orig')
                        row.update(twin_dist=P.fdist(f, vw['f'], n0), twin_valid=bool(np.isfinite(vw['W']).all()))
                    vq = g_run(kind, dt, ci, role='qp')
                    cq = g_cert(vq, kind, dt, role='qp')
                    cq.pop('arrays')
                    A_['W_qp'][j] = vq['W']
                    row.update(qp_valid=bool(g_valid(vq, kind, cq) and cq['ok']), qp_cert=cq, dist_qp=P.fdist(f, vq['f'], n0))
                    A_['rho'][j] = own_rho(W, dg['main'], dg['qp'])
                    if ci in dev:                                       # dev6 step halving (time-error estimate)
                        k_ = dev.index(ci)
                        vh = g_run(kind, dt / 2, ci)
                        A_['W_half'][k_] = vh['W']
                        row.update(halving_dist=P.fdist(f, vh['f'], n0), halving_valid=g_valid(vh, kind))
                rh = A_['rho'][j]
                fin = rh[~np.isnan(rh)]
                row['rho_max'] = None if fin.size == 0 else (float(fin.max()) if np.isfinite(fin).all() else float('inf'))
                row['e0'] = e0(f, rec_of(ci))
                row['stable'] = bool(row['growth'] <= GROWTH and row['stage_finite'])
                row['h_ev'] = h_ev
                A_['W'][j], A_['w0'][j], A_['n0'][j], A_['nu'][j] = W, w0n, n0, nu
                Fl['f257'][j] = f
                if ci in dev:
                    REFS[(arm, Rp, ci)] = dict(W=np.asarray(W), h_ev=h_ev)
                rows_rank.append(row)
                rep['runs'].append(row)
                print('RUN', arm, Rp, cid(ci), round(row['e0']['e'], 6), el(), flush=True)
            A_.update(A=np.asarray(stx.base['A']), lam=np.asarray(stx.base['lam']), kx=np.asarray(stx.kx), ky=np.asarray(stx.ky),
                      D=np.asarray(tab['D']), V=np.asarray(tab['V']), Rm=np.asarray(dg['main']['Rm']), dt=dt, Rp=Rp, mesh=L,
                      rule=spec['rule'], qp=spec['qp'], M=stx.M, time_tag=TIME_TAG)
            np.savez_compressed(out / f'A_{arm}_{Rp}.npz', **A_)
            np.savez_compressed(out / f'F_{arm}_{Rp}.npz', **Fl)
            rr['arms'][arm] = dict(n=n, seconds=time.perf_counter() - t_arm)
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
                ref8, ref16 = g_run('etd', DT0 / 8, ci), g_run('etd', DT0 / 16, ci)
                d_ref = P.fdist(ref8['f'], ref16['f'], ref8['n0'])
                lev = {fct: P.fdist(g_run('ros', DT0 / fct, ci)['f'], ref8['f'], ref8['n0']) for fct in (1, 2, 4)}
                x = np.log(np.array([DT0, DT0 / 2, DT0 / 4]))
                y = np.log(np.array([lev[1], lev[2], lev[4]]))
                slope = float(np.polyfit(x, y, 1)[0]) if np.isfinite(y).all() else float('nan')
                od.append(dict(case=cid(ci), ref_dist=d_ref, levels=lev, slope=slope,
                               ref_converged=bool(d_ref <= 1e-3 * min(lev.values()))))
            blk['order_F5'] = od
            rep['elapsed'][f'order|{Rp}'] = time.perf_counter() - t_b
        cls = []
        for r_ in [x for x in rows_rank if not x['stable']]:                 # A5.5: every tripped run
            ci = ids[[cid(i) for i in ids].index(r_['case'])]
            kind, dt = ARMS[r_['arm']]['kind'], ARMS[r_['arm']]['dt']
            if kind == 'lspg':
                stx = settings['M1152'] if r_['arm'] == 'L-CNR-M1152' else st
                key = stx.name
                if key not in lrec801:
                    lrec801[key] = T2.make_split_query(Q, 'point', 'LSPG', Rp, L, stx.trust, nrec=801, nstep=801)
                gs, oks = [], []
                for f_ in (8, 16):
                    u0, nu, _ = U0[ci]
                    sch = stx.sched('LSPG', 'CNR', DT0 / f_, 'prod')
                    o = lrec801[key]['fused'](u0, nu, stx.data[('main', 'LSPG')], stx.cold, sch)
                    stt_ = o['stats']
                    k_ = int(stt_['steps'])
                    gs.append(lspg_growth(stx, np.asarray(stt_['states'])[:k_ + 1]))
                    oks.append(bool(np.isfinite(np.asarray(o['W'])).all() and np.asarray(stt_['exits'])[3] == 0))
                g8, g16, ok_ = gs[0], gs[1], all(oks)
            else:
                v8, v16 = (g_run(kind, dt / f_, ci) for f_ in (8, 16))
                ok_ = all(g_valid(v_, kind) and g_cert(v_, kind, dt / f_)['ok'] for v_, f_ in ((v8, 8), (v16, 16)))
                g8, g16 = v8['growth'], v16['growth']
            cls.append(dict(arm=r_['arm'], case=r_['case'], g8=g8, g16=g16, refined_ok=ok_,
                            label=FS.classify_growth(g8, g16, ok_)))
        blk['stability_classification'] = cls
        save()

        # ---------------------------------------------------- controls ----
        if cfg.get('controls'):
            import fscontrols as FC
            t_b = time.perf_counter()
            blk['controls'] = FC.run(cfg, st, dg, tabs, nl, g_run, g_valid, g_cert, lspg_run, U0, dev, ids, cases, bank, L,
                                     evolver, settings, check_tab, e0, rec_of)
            rep['elapsed'][f'controls|{Rp}'] = time.perf_counter() - t_b
            save()

        if cfg.get('timing'):
            import fstiming as FT
            U0d = {ci: U0[ci] for ci in dev}
            refs_r = {(a_, ci): v_ for (a_, r_, ci), v_ in REFS.items() if r_ == Rp}
            timing_subjects.update(FT.bank_subjects(st, settings, tabs, arm_list, evolver, U0d, refs_r, Rp, check_tab,
                                                    lambda stx: stx.check('main', bank)))
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
        rep['pod'] = PA.run(cfg, L, cases, ids, dev, U0, out, e0, rec_of, el)
        rep['elapsed']['pod'] = time.perf_counter() - t_b
        save()

    # ---------------------------------------------------- timing, in-job selection ----
    if cfg.get('timing'):
        import fstiming as FT
        t_b = time.perf_counter()
        subj = dict(timing_subjects)
        if cfg.get('pod'):
            subj.update(FT.pod_subjects(PA.timing_subjects()))
        rep['timing'] = FT.run(cfg, subj, U0, dev, cases, lambda screen, recs: select_pairs(rep, screen, recs, dev, cases, cfg))
        rep['elapsed']['timing'] = time.perf_counter() - t_b
        save()
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['executed'] = True
    save()
    (out / 'COMPLETE').write_text('executed (acceptance is checks.py)\n')
    print('FSRUN COMPLETE', el(), flush=True)


# ------------------------------------------------------------------ in-job selection (A3.4, A3.6, A5.9) ----

def subject_rows(rep):
    out = {}
    for r_ in rep['runs']:
        out.setdefault(f"{r_['arm']}|{r_['Rp']}", []).append(r_)
    pool = {}
    for grp in ('members', 'native'):
        for mid, m in (rep.get('pod') or {}).get(grp, {}).items():
            if m.get('duplicate_of'):
                continue
            pool[mid] = [dict(case=x['case'], valid=x['valid'], stable=x['stable'], e=x['e0']['e']) for x in m['rows']]
    return out, pool


def dev_qualified(rows, devc):
    """dev6-only numerical evidence of one bank subject (A5.9): validity, stability, certificates, Q2 bars, anchors."""
    by = {r_['case']: r_ for r_ in rows}
    for c in devc:
        r_ = by.get(c)
        if r_ is None or not (r_['valid'] and r_['stable'] and r_.get('qp_valid')):
            return False
        if not (r_.get('dist_qp', 1.) <= 2.5e-4 and r_.get('rho_max') is not None and r_['rho_max'] <= .116):
            return False
        cert = r_.get('cert')
        if cert is not None and not cert['ok']:
            return False
        if 'nonstationary' in r_ and r_['nonstationary'] > .01 * r_['steps']:
            return False
        if 'tight_anchor_ok' in r_ and not (r_['tight_anchor_ok'] and r_.get('tighter_anchor_ok', False)
                                            and r_['dist_tight'] <= 2.5e-4 and r_.get('dist_tight_tighter', 1.) <= 2.5e-5):
            return False
        if 'twin_dist' in r_ and not r_['twin_dist'] <= 2.5e-5:
            return False
    return True


def select_pairs(rep, screen, recs, dev, cases, cfg):
    devc = [f'{cases[i][0]}|{cases[i][1]}' for i in dev]
    bank_rows, pool = subject_rows(rep)
    simple = {k: [dict(case=r_['case'], valid=r_['valid'], stable=r_['stable'], e=r_['e0']['e']) for r_ in v]
              for k, v in bank_rows.items()}
    pairs, sel = [], {}
    for X, rows in simple.items():
        B = FS.matched_competitor(rows, pool, screen, devc) if pool else None
        sel[X] = B
        if B is not None:
            pairs.append((X, B))
    ranks = [s_['Rp'] for s_ in cfg['ranks']]
    for r_ in ranks:
        if f'G-CN|{r_}' in simple and f'G-R1|{r_}' in simple:
            pairs.append((f'G-CN|{r_}', f'G-R1|{r_}'))
    espeed = {}
    for sid in simple:
        st_ = FS.case_ratio_stats([t for t in recs['E'] if t['B'] == sid], devc, cfg['timing'].get('reps', 3))
        espeed[sid] = st_['value'] if st_['status'] == 'ok' else None
    cands = []
    for arm in ('G-CN', 'G-E4a', 'G-E4b', 'G-R1'):
        ok = True
        for r_ in ranks:
            X, A1 = f'{arm}|{r_}', f'L-CNR|{r_}'
            if X not in bank_rows or not dev_qualified(bank_rows[X], devc):
                ok = False
                break
            if not (FS.ok_on(simple[X], devc) and FS.worst(simple[X], devc) <= FS.worst(simple[A1], devc)):
                ok = False
                break
        X = f'{arm}|384'
        if ok and 384 in ranks and espeed.get(X) is not None and espeed[X] >= 3 and screen.get(X) is not None:
            cands.append((arm, screen[X], FS.worst(simple[X], devc)))
    order = {'G-E4b': 0, 'G-E4a': 1, 'G-R1': 2, 'G-CN': 3}
    best = None
    if cands:
        ts = min(c[1] for c in cands)
        tie = [c for c in cands if c[1] <= FS.TIE * ts]
        best = sorted(tie, key=lambda c: (c[2], order[c[0]]))[0][0]
    rep['selection'] = dict(matched=sel, pairs=pairs, best_stepper=best, best_candidates=cands, evolve_speed_dev6=espeed,
                            screen=dict(screen), smoke_subset=not cfg.get('headline', False))
    return pairs


if __name__ == '__main__':
    main()
