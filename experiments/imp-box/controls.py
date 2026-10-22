"""imp-box stage `controls`: must-fail controls on real data (DESIGN §7, A1.6, A2.9, A3.5, A4) with their positive
controls. Calibrated numerical faults must move the checker statistic past its threshold by >= 2x; Boolean controls
fire iff the decision flips. Writes controls.json. (MF-REF-real, MF-PAD-chk, MF-IC, G7 are in ref.json; G-OPS in
pod.json; MF-FROZEN / MF-LEAK in the test stage; T-id and T-dbl in final.json.)
"""
from __future__ import annotations

import json
import time

import numpy as np
import jax
import jax.numpy as jnp

import finalize as FZ
import geom as G
import ladder as LD
import ladders as LS
import podsine as PS
import qual as QL
import rom as RM
import score as SC
import spec as SP
import stoprule as STP


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))


def stage_controls(a, out, D):
    import timing as TM
    c = TM.Ctx(a, out, D)
    d0 = jnp.asarray(c.Dv[0])
    D2 = c.Dv[:2]
    res = {}
    sps, Phi, kq = LS.pod_spaces(out, 'pod', rules=(10, 12, 14))
    sp = sps[12]
    r = min(32, sp.R)
    n = 20
    # G4 + MF-TENS
    qp = RM.make_query(sp, r, 'gal_etdrk4', n, form='point')
    qt = RM.make_query(sp, r, 'gal_etdrk4', n, form='tensor')
    qs = RM.make_query(sp, r, 'gal_etdrk4', n, form='tensor', slice_scale=(0, 1.01))
    Op, _ = FZ.run_unbatched(qp, D2)
    Ot, _ = FZ.run_unbatched(qt, D2)
    Os, _ = FZ.run_unbatched(qs, D2)
    g4, mt = rel(Ot, Op), rel(Os, Op)
    res['G4'] = dict(diff=g4, passed=bool(g4 <= 1e-10))
    res['MF_TENS'] = dict(diff=mt, fires=bool(mt > 2e-10))
    # MF-Q1-E4 (coefficient fault in the update path) and its positive control
    _, st0 = RM.make_query(sp, r, 'gal_etdrk4', n, cert=True)(d0)
    _, st1 = RM.make_query(sp, r, 'gal_etdrk4', n, cert=True, perturb_coef=1e-6)(d0)
    res['MF_Q1_E4'] = dict(defect_clean=float(st0[4]), coef_cert_clean=float(st0[1]), defect_fault=float(st1[4]),
                           positive=bool(float(st0[4]) <= 1e-9 and float(st0[1]) <= 1e-10),
                           fires=bool(float(st1[4]) > 2e-9))
    # MF-Q1-LSPG (perturbed iterate before the certificate) on the POD span (Mt = min(4r, R))
    _, sl0 = RM.make_query(sp, r, 'lspg_cnr', n, cert=True)(d0)
    _, sl1 = RM.make_query(sp, r, 'lspg_cnr', n, cert=True, perturb_cert=1e-3)(d0)
    f0, f1 = float(sl0[0] / max(sl0[2], 1)), float(sl1[0] / max(sl1[2], 1))
    res['MF_Q1_LSPG'] = dict(nonstat_clean=f0, nonstat_fault=f1, positive=bool(f0 <= 0.01), fires=bool(f1 > 0.02))
    # MF-NAN-stage (reduced, FOM fixed, FOM adaptive)
    _, sn = RM.make_query(sp, r, 'gal_etdrk4', n, nan_stage=True)(d0)
    _, sf = SP.make_fom(16, 20, nan_stage=True)(d0)
    _, sa = SP.make_fom_ad(16, 1e-4, nan_stage=True)(d0)
    res['MF_NAN_stage'] = dict(reduced_invalid=not LD.stats_valid(dict(fam='podg', st='gal_etdrk4'), np.asarray(sn)[None]),
                               fom_invalid=not SP.fom_valid('spec', np.asarray(sf)[None]),
                               ad_invalid=not SP.fom_valid('specad', np.asarray(sa)[None]))
    res['MF_NAN_stage']['fires'] = bool(all(res['MF_NAN_stage'].values()))
    # MF-AD-rej / MF-AD-cap
    tol_rej = None
    for tl in (1e-3, 1e-2, 1e-1, 1.0):                  # a tolerance whose clean first trial is accepted (0 rejects)
        _, s_ok = SP.make_fom_ad(32, tl)(d0)
        if float(s_ok[5]) == 0:
            tol_rej = tl
            break
    _, s_rej = SP.make_fom_ad(32, tol_rej or 1.0, force_reject_first=True)(d0)
    _, s_cap = SP.make_fom_ad(32, 1e-4, force_inf_all=True)(d0)
    s_ok, s_rej, s_cap = (np.asarray(x) for x in (s_ok, s_rej, s_cap))
    res['MF_AD_rej'] = dict(tol=tol_rej, rejects_clean=float(s_ok[5]), rejects_fault=float(s_rej[5]),
                            accepted_clean=float(s_ok[2]), accepted_fault=float(s_rej[2]),
                            fires=bool(tol_rej is not None and s_rej[5] >= 1 and SP.fom_valid('specad', s_rej[None])))
    res['MF_AD_cap'] = dict(cap_hit=float(s_cap[3]), rejects=float(s_cap[5]), expected_rejects=SP.AD_J - SP.AD_J0 + 1,
                            fires=bool(s_cap[3] == 1 and s_cap[5] == SP.AD_J - SP.AD_J0 + 1
                                       and not SP.fom_valid('specad', s_cap[None])))
    # MF-Q2 (configuration's own rule weights x 1.5) and positive control
    arm = TM.Arm('podg', c)
    cfg = dict(fam='podg', r=r, q=12, st='gal_etdrk4', nsteps=n)
    p_ = QL.q2(cfg, arm.sps, arm.build_on, D2, c.Uv[:2])
    f_ = QL.q2(cfg, arm.sps, arm.build_on, D2, c.Uv[:2], w_scale=1.5)
    res['MF_Q2'] = dict(rho_clean=p_['rho_max'], passed_clean=p_['passed'], rho_fault=f_['rho_max'],
                        fires=bool(f_['rho_max'] >= 2 * QL.RHO_BAR and not f_['passed']))
    # MF-ECSW (weights x 1.5 vs the parent rule)
    fits = sorted(out.glob('ecsw_r*_k*.npz'), key=lambda f: (-int(f.stem.split('_')[2][1:]), int(f.stem.split('_')[1][1:])))
    if fits:
        bs = lambda cf, s_, zo: RM.make_query(s_, cf['r'], 'gal_etdrk4', cf['nsteps'], form='point',
                                              out='z' if zo else 'lattice')
        chosen = None
        for f in fits[:4]:                              # largest kappa first; prefer a fit whose clean Q2 passes
            z = np.load(f)
            rr, kk = (int(t[1:]) for t in f.stem.split('_')[1:])
            sub = arm.sps[12].with_weights(z['nodes'], z['weights'])
            pc = dict(fam='podg', r=rr, q=12, st='gal_etdrk4', nsteps=n)
            parent_q2 = QL.q2(pc, arm.sps, arm.build_on, D2, c.Uv[:2])
            ce = dict(fam='ecsw', r=rr, kappa=kk, q=12, st='gal_etdrk4', nsteps=n)
            pe = QL.q2_ecsw(ce, sub, arm.sps[12], bs, bs, D2, c.Uv[:2], parent_q2)
            chosen = (rr, kk, sub, ce, parent_q2, pe)
            if pe['passed']:
                break
        rr, kk, sub, ce, parent_q2, pe = chosen
        fe = QL.q2_ecsw(ce, sub, arm.sps[12], bs, bs, D2, c.Uv[:2], parent_q2, w_scale=1.5)
        res['MF_ECSW'] = dict(r=rr, kappa=kk, rho_clean=pe['rho_max'], passed_clean=pe['passed'],
                              rho_fault=fe['rho_max'], positive=bool(pe['passed']),
                              fires=bool(fe['rho_max'] >= 2 * QL.RHO_BAR and not fe['passed']))
    # MF-TRANS-id / MF-TRANS-sign
    N = Phi.shape[1] + 1
    dec0 = PS.make_shift_decoder(Phi[:r])
    zr = np.linalg.solve(sp.M[:r, :r], sp.Pi[:, :r].T @ (sp.wi * G.ic('B', np.asarray(d0), sp.meta['ic_points'][:, 0],
                                                                              sp.meta['ic_points'][:, 1], np)))
    idv = rel(dec0(jnp.asarray(zr), jnp.zeros(2)), sp.Pl[:, :r] @ zr)
    b1 = np.zeros((N - 1, N - 1))
    b1[2, 1] = 1.0                                       # u = sin(3 pi x) sin(2 pi y): exactly representable
    delta = np.array([0.13, -0.07])
    pts = G.lattice('B')[0]
    direct = np.sin(3 * np.pi * (pts[:, 0] - delta[0])) * np.sin(2 * np.pi * (pts[:, 1] - delta[1]))
    good = PS.make_shift_decoder(b1[None], sign=-1.0)(jnp.ones(1), jnp.asarray(delta))
    bad = PS.make_shift_decoder(b1[None], sign=+1.0)(jnp.ones(1), jnp.asarray(delta))
    eg, eb = rel(good, direct), rel(bad, direct)
    res['MF_TRANS'] = dict(zero_shift_identity=idv, id_passed=bool(idv <= 1e-12), sign_good=eg, sign_bad=eb,
                           sign_passed=bool(eg <= 1e-12), sign_fires=bool(eb > 1e-2))
    # G-XFORM (A11): the dense-transform nonlinear term equals pscore.nonlinear (FFT) on real states, N in {10, 24, 64}
    def xdiff(a1, a2):                                  # non-finite inputs or differences fail (audit 10)
        if not (np.all(np.isfinite(a1)) and np.all(np.isfinite(a2))):
            return float('inf')
        return float(np.max(np.abs(a1 - a2)) / (np.max(np.abs(a1)) + 1e-300))
    per = {}
    for Nx in [n_ for n_ in LS.SPEC_N if n_ <= LS.DENSE_MAX_N]:
        bx = SP.ICTable(Nx).coefs(d0)
        per[str(Nx)] = xdiff(np.asarray(SP.nl_fun(Nx)(bx)), np.asarray(SP.nl_fun(Nx, xform='dense')(bx)))
    gx = max(per.values())
    bx = np.asarray(SP.nl_fun(24)(SP.ICTable(24).coefs(d0)))
    nan_mut = xdiff(bx, np.full_like(bx, np.nan))       # mutation: an all-NaN dense output must fail
    res['G_XFORM'] = dict(per_N=per, max_rel=gx, passed=bool(gx <= 1e-12), nan_mutation=nan_mut,
                          nan_fires=bool(not np.isfinite(nan_mut)))
    # MF-FLOOR
    zf = np.random.default_rng(3).standard_normal((6, r))
    Uin = zf @ sp.Pl[:, :r].T
    Urand = np.random.default_rng(4).standard_normal((6, sp.Pl.shape[0]))
    fi, fr = RM.floor_err(sp, r, Uin), RM.floor_err(sp, r, Urand)
    res['MF_FLOOR'] = dict(in_span=fi, random=fr, fires=bool(fi <= 1e-10 and fr >= 0.5))
    # MF-SETUP: a leaky FOM query with coefficients frozen at nu0 must differ from the official one at nu != nu0
    N0, n0 = 24, 40
    lam = SP.lam_grid(N0)
    co0 = SP._coefs(lam, float(d0[4]), G.T_END / n0)
    tab = SP.ICTable(N0)
    NL = SP.nl_fun(N0)
    syn = SP.synth_fn(N0)

    def leaky(d):
        v = tab.coefs(d)
        outs = [syn(v)]
        for k in range(5):
            v = jax.lax.fori_loop(0, n0 // 5, lambda i, x: SP._etd(x, co0, NL)[0], v)
            outs.append(syn(v))
        return jnp.stack(outs)
    d1 = d0.at[4].multiply(1.5)
    off = SP.make_fom(N0, n0)(d1)[0]
    lk = jax.jit(leaky)(d1)
    same = rel(jax.jit(leaky)(d0), SP.make_fom(N0, n0)(d0)[0])
    res['MF_SETUP'] = dict(leaky_vs_official_at_nu1=rel(lk, off), leaky_vs_official_at_nu0=same,
                           fires=bool(rel(lk, off) > 1e-6 and same <= 1e-12),
                           structural='every timed callable is jax.jit of a function of d alone (builders receive no '
                                      'descriptor); nu enters only through d')
    # MF-ENV (bank cells)
    if (out / 'bank.pkl').exists():
        import bank as BK
        import sep_common as sc
        b = BK.load(out / 'bank.pkl')
        pj = jax.tree_util.tree_map(jnp.asarray, b['params'])
        bs_ = jnp.asarray(G.boundary_samples('B'))
        withe = float(jnp.max(jnp.abs(BK.features('B', pj, bs_))))
        X = bs_ / G.SIDE
        ang = 2.0 * jnp.pi * (X @ pj['B'])
        raw = pj['out_scale'] * sc.apply_mlp(pj['g'], jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], -1))
        res['MF_ENV'] = dict(with_envelope=withe, without=float(jnp.max(jnp.abs(raw))),
                             positive=bool(withe <= 1e-12), fires=bool(float(jnp.max(jnp.abs(raw))) > 2e-8))
    # selector fixtures, MF-NOMATCH, MF-STOP, score fixtures
    res['selector'] = selector_fixture(c)
    res['MF_NOMATCH'] = dict(fires=bool(FZ.maxmin({}, ['b'], []) is None))
    res['MF_STOP'] = STP.fixtures()
    res['score_selftest'] = SC.selftest()
    import teststage as TS
    res.update(TS.leak_controls(out, a.W, a.lev))
    res['test_fixtures'] = TS.test_fixtures()
    import cell as CL                                   # injected exclusion (code audit 2): row 0 marked ineligible
    zv = np.load(out / 'refs_val.npz')
    zf = dict(U=zv['U'], eligible=np.asarray(zv['eligible'], bool).copy())
    i_ex = int(np.argmax(zf['eligible']))               # a currently eligible row (code audit 3)
    zf['eligible'][i_ex] = False
    Df, Uf = CL.eligible(D['val'], zf)
    res['eligibility_propagates'] = dict(
        row=i_ex, fires=bool(len(Df) == int(np.sum(zv['eligible'])) - 1
                             and not np.any(np.all(np.isclose(Df, D['val'][i_ex]), axis=1))
                             and len(c.Dv) == int(np.sum(zv['eligible'])) and len(Uf) == len(Df)))
    import timing as TMd                                # representative (not microsecond) queries for the harness check
    r64 = min(64, sp.R)
    res['T_dbl_rom'] = TMd.t_dbl_raw(RM.make_query(sp, r64, 'gal_etdrk4', 40, form='tensor'), c.Dv)
    res['T_dbl_fom'] = TMd.t_dbl_raw(SP.make_fom(48, 80), c.Dv)
    rj = json.loads((out / 'ref.json').read_text())
    pj = json.loads((out / 'pod.json').read_text())
    reg = {                                             # mandatory registry (code audit 1): name -> required outcome
        'G7': rj['G7']['passed'], 'MF_IC': rj['MF_IC']['fires'], 'MF_REF_real': rj['MF_REF_real']['fires'],
        'MF_PAD_chk': rj['MF_PAD_chk']['fires'], 'refgate_selftest': rj['refgate_selftest']['passed'],
        'G_OPS': pj['G_OPS']['passed'], 'G4': res['G4']['passed'], 'MF_TENS': res['MF_TENS']['fires'],
        'G_XFORM': res['G_XFORM']['passed'] and res['G_XFORM']['nan_fires'],
        'MF_Q1_E4': res['MF_Q1_E4']['positive'] and res['MF_Q1_E4']['fires'],
        'MF_Q1_LSPG': res['MF_Q1_LSPG']['positive'] and res['MF_Q1_LSPG']['fires'],
        'MF_NAN_stage': res['MF_NAN_stage']['fires'], 'MF_AD_rej': res['MF_AD_rej']['fires'],
        'MF_AD_cap': res['MF_AD_cap']['fires'], 'MF_Q2': res['MF_Q2']['passed_clean'] and res['MF_Q2']['fires'],
        'MF_TRANS': res['MF_TRANS']['id_passed'] and res['MF_TRANS']['sign_passed'] and res['MF_TRANS']['sign_fires'],
        'MF_FLOOR': res['MF_FLOOR']['fires'], 'MF_SETUP': res['MF_SETUP']['fires'], 'selector': res['selector']['fires'],
        'MF_NOMATCH': res['MF_NOMATCH']['fires'], 'MF_STOP': res['MF_STOP']['passed'],
        'score_selftest': all(res['score_selftest'].values()), 'MF_LEAK': res['MF_LEAK']['fires'],
        'MF_FROZEN': res['MF_FROZEN']['fires'] and res['MF_FROZEN'].get('positive', True),
        'test_fixtures': res['test_fixtures']['fires'], 'eligibility_propagates': res['eligibility_propagates']['fires'],
        'T_dbl': res['T_dbl_rom']['passed'] and res['T_dbl_fom']['passed']}
    # required controls are declared from the cell's arms before reading results (code audit 2): missing = failure
    reg['MF_ECSW'] = bool(res.get('MF_ECSW', {}).get('fires', False))   # positive control reported (A6.6)
    if a.bank:
        reg['MF_ENV'] = bool(res.get('MF_ENV', {}).get('positive', False) and res.get('MF_ENV', {}).get('fires', False))
    res['registry'] = {k: bool(v) for k, v in reg.items()}
    res['all_fire'] = bool(all(reg.values()))
    res['not_firing'] = [k for k, v in reg.items() if not v]
    (out / 'controls.json').write_text(json.dumps(res, indent=1, default=float))
    print(time.strftime('%H:%M:%S'), 'controls: all fire', res['all_fire'], 'not firing', res['not_firing'], flush=True)
    if not res['all_fire']:
        raise SystemExit(f"mandatory controls failed: {res['not_firing']} (results of this cell are not accepted)")


def selector_fixture(c):
    """A2.9: six failing candidates then a qualifying seventh -> selected; a forced failure is never a finalist."""
    import timing as TM

    class FakeArm:
        name = 'fake'

        def __init__(self, ctx):
            self.ctx = ctx

        def rows(self):
            return [dict(key=f'fake|{i}', cfg=dict(fam='fake', i=i), valid=True, worst=0.001, median=0.001,
                         screen=0.001 * (i + 1)) for i in range(9)]

        def build(self, cfg):
            U0 = jnp.asarray(self.ctx.Uv[0])
            return jax.jit(lambda d: (U0, jnp.zeros(7)))

        def qualify(self, cfg, O, st):
            return dict(Q1=dict(passed=cfg['i'] >= 6), Q2=None)

    class FC:
        pass
    fc = FC()
    fc.Dv, fc.Uv, fc.U = c.Dv[:1], c.Uv[:1], 0.0
    fc.cache_off, fc.cache_qual, fc.cache_q2, fc.cache_scr = {}, {}, {}, {}
    rec = TM.select(FakeArm(fc), 0.03, 'worst')
    keys = [f['key'] for f in rec['finalists']]
    att = rec['attempts']
    ok = (len(att) >= 7 and not any(x['qualified'] for x in att[:6]) and att[6]['qualified']
          and all(int(k.split('|')[1]) >= 6 for k in keys) and len(keys) >= 1)
    return dict(finalists=keys, attempts=[(x['key'], x['qualified']) for x in att], fires=bool(ok))
