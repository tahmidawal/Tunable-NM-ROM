"""imp-faststep in-job controls (DESIGN 7, A1.9, A2.7, A3.7, A4.1). Each entry records the measured quantity, the
threshold and whether the detector fired (must-fire controls) or passed (positive controls). Escalations follow A2.7.
Offline-only controls (C10, C17, C19-C24) live in checks.py; C9 is read from the timing block; C11 from the ecsw job.
"""
from __future__ import annotations

import types

import numpy as np
import jax
import jax.numpy as jnp

import fscore as F
import fsselect as FS
import p1common as P
from p1common import Q, T2, DT0

NREC = int(round(.25 / DT0)) + 1


def _slope(dts, errs):
    x, y = np.log(np.asarray(dts)), np.log(np.asarray(errs, float))
    return float(np.polyfit(x, y, 1)[0]) if np.isfinite(y).all() else float('nan')


def run(cfg, st, dg, tabs, nl, g_run, g_valid, g_cert, lspg_run, U0, dev, ids, cases, bank, L, evolver, settings,
        check_tab, e0, rec_of):
    C = {}
    Rp = st.Rp
    tab = tabs['main']
    d0, d1 = dev[0], dev[min(1, len(dev) - 1)]
    val = [i for i in ids if cases[i][0] == 'val32']
    vcase = val[0] if val else d1
    u0, nu, n0 = U0[d0]
    w0, scale = st.q['GAL']['init'](u0, st.cold)
    z0 = np.asarray(tab['VtRm'] @ w0)
    # C1 / C1m -------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(20261021)
    Zs = [z0] + [rng.standard_normal(Rp) * np.linalg.norm(z0) / np.sqrt(Rp) for _ in range(5)]
    v = g_run('etd', .0125, d0)
    Zs += [v['z_out'][k] for k in range(1, 6)]
    a1 = F.rhs_agreement(Zs, nu, dg['main'], tab, nl)
    C['C1'] = dict(value=a1, bar=1e-11, passed=bool(a1 <= 1e-11))
    bad, _ = F.precontract(dg['main'], mutation='wrong_congruence')
    ce = F.make_cert_etd(nl)
    evb = jax.jit(F.make_evolve('etd', F.adv_pre, F.jac_pre))
    Wb, sb = evb(w0, jnp.float64(nu), bad, F.sched(.0125))
    k = int(sb['steps'])
    co = {a: jnp.asarray(b) for a, b in F.etd_coefs_indep(-nu * np.asarray(tab['D']), .0125).items()}
    mis = float(np.nanmax(np.asarray(ce(jnp.asarray(np.asarray(sb['Z'])[:k + 1]), dg['main'], tab['V'], co, k))))
    C['C1m'] = dict(rhs_agreement=F.rhs_agreement(Zs, nu, dg['main'], bad, nl), reconstruction=mis, bar=1e-10,
                    fired=bool(not mis <= 1e-10))
    # C2 / C2m / C16 -------------------------------------------------------------------------------------------
    evg = jax.jit(T2.make_evolve(nl, Q.tested_jac('point', L), 'GAL', Rp, st.trust))
    c2, c2a, c2g = [], [], []
    for ci in (d0, d1):
        uu, nn, n00 = U0[ci]
        ww, sc_ = st.q['GAL']['init'](uu, st.cold)
        Wg, sg = evg(ww, nn, sc_, dg['main'], T2.sched('CN', DT0, 0., 1e-13))
        fg = P.restrict(st.q['GAL']['decode'](Wg, dg['main']), L)
        vt = g_run('cn', DT0, ci, level='tight')
        ct = g_cert(vt, 'cn', DT0, level='tight')
        c2.append(P.fdist(vt['f'], fg, n00))
        c2a.append(bool(g_valid(vt, 'cn', ct) and ct['unconverged'] == 0))
        c2g.append(bool(np.isfinite(np.asarray(Wg)).all() and np.asarray(sg['exits'])[3] == 0))
    C['C2'] = dict(values=c2, bar=1e-8, anchors_valid=c2a, t2core_valid=c2g,
                   passed=bool(max(c2) <= 1e-8 and all(c2a) and all(c2g)))
    vm = g_run('cn', DT0, d0, mutation='newton0')
    cm = g_cert(vm, 'cn', DT0)
    C['C2m'] = dict(unconverged=cm['unconverged'], steps=cm['steps'], fired=bool(cm['unconverged'] > .01 * cm['steps']))
    c16 = []
    for b_ in (1, 0):
        ev1 = jax.jit(F.make_evolve('cn', F.adv_pre, F.jac_pre, budget_override=b_))
        W1, s1 = ev1(w0, jnp.float64(nu), tab, F.sched(DT0, level='tight'))
        v1 = dict(W=np.asarray(W1), Z=np.asarray(s1['Z']), steps=int(s1['steps']), reason=np.asarray(s1['reason']),
                  it=np.asarray(s1['it']), nu=nu, f=np.zeros(1), stage_max=float(s1['stage_max']))
        c1 = g_cert(v1, 'cn', DT0, level='tight')
        c16.append(dict(budget=b_, unconverged=c1['unconverged'], anchor_invalid=bool(c1['unconverged'] > 0)))
        if c16[-1]['anchor_invalid']:
            break
    C['C16'] = dict(attempts=c16, fired=bool(c16[-1]['anchor_invalid']))
    # C3 linear exactness ------------------------------------------------------------------------------------
    ev0 = jax.jit(F.make_evolve('etd', lambda z, t: 0. * z))
    W0_, s0_ = ev0(w0, jnp.float64(nu), tab, F.sched(.0125))
    Dn = np.asarray(tab['D'])
    exact = np.stack([np.exp(-nu * Dn * .05 * kk) * z0 for kk in range(6)])
    c3 = float(np.max(np.abs(np.asarray(s0_['z_out']) - exact)) / np.max(np.abs(exact)))
    C['C3'] = dict(value=c3, bar=1e-12, passed=bool(c3 <= 1e-12))
    # C3m ETDRK4 order and the f2_as_f1 mutant (A2.7: fixture dt 0.0125, escalate to 0.025) ----------------------
    r32, r64 = g_run('etd', .025 / 32, d0), g_run('etd', .025 / 64, d0)
    dref = P.fdist(r32['f'], r64['f'], n0)
    errs = [P.fdist(g_run('etd', dt, d0)['f'], r32['f'], n0) for dt in (.025, .0125, .00625)]
    conv = bool(dref <= 1e-3 * min(errs))
    mm = {}
    for dt in (.0125, .025):
        vmut = g_run('etd', dt, d0, mutation='f2_as_f1')
        mm[dt] = g_cert(vmut, 'etd', dt)['worst_mismatch']
        if mm[dt] > 1e-8:
            break
    em = [P.fdist(g_run('etd', dt, d0, mutation='f2_as_f1')['f'], r32['f'], n0) for dt in (.025, .0125, .00625)]
    C['C3m'] = dict(ref_dist=dref, ref_converged=conv, errors=errs, slope=_slope((.025, .0125, .00625), errs),
                    slope_ok=bool(_slope((.025, .0125, .00625), errs) >= 3.5), mutant_mismatch=mm,
                    mutant_errors=em, mutant_slope=_slope((.025, .0125, .00625), em),
                    fired=bool(max(mm.values()) > 1e-8), demonstrated=bool(conv))
    # C4m stability monitor (A2.7 escalation -nu, -10 nu) -----------------------------------------------------
    c4 = []
    for fac in (1., 10.):
        evn = jax.jit(F.make_evolve('etd', F.adv_pre, mutation='neg_nu'))
        Wn, sn = evn(w0, jnp.float64(nu * fac), tab, F.sched(.0125))
        g_, sm = float(sn['growth']), float(sn['stage_max'])
        c4.append(dict(factor=fac, growth=g_, stage_max=sm, fired=bool(not (g_ <= 1.05 and np.isfinite(sm)))))
        if c4[-1]['fired']:
            break
    C['C4m'] = dict(attempts=c4, fired=bool(c4[-1]['fired']))
    # C5 one-pass order, mutant J = 0 ----------------------------------------------------------------------------
    e8 = g_run('etd', DT0 / 8, d0)
    e16 = g_run('etd', DT0 / 16, d0)
    dr = P.fdist(e8['f'], e16['f'], n0)
    er = [P.fdist(g_run('ros', DT0 / f_, d0)['f'], e8['f'], n0) for f_ in (1, 2, 4)]
    ez = []
    for f_ in (1, 2, 4):
        vz = g_run('ros', DT0 / f_, d0, mutation='jzero')
        ez.append(P.fdist(vz['f'], e8['f'], n0) if g_valid(vz, 'ros') and vz['growth'] <= 1.05 else float('inf'))
    sz = _slope((DT0, DT0 / 2, DT0 / 4), ez)
    C['C5'] = dict(ref_dist=dr, ref_converged=bool(dr <= 1e-3 * min(er)), errors=er, slope=_slope((DT0, DT0 / 2, DT0 / 4), er),
                   slope_ok=bool(_slope((DT0, DT0 / 2, DT0 / 4), er) >= 1.8),
                   mutant_errors=ez, mutant_slope=sz, fired=bool(not np.isfinite(sz) or sz < 1.8))
    # C6m LSPG Jacobian (analytic vs autodiff) and the jflip run -----------------------------------------------
    jac = T2.make_jac(nl)
    nlJ = Q.tested_jac('point', L)
    dl = st.data[('main', 'LSPG')]
    co_ = jnp.asarray(T2.SCHEMES['CNR'][:5])
    cv = jnp.asarray(w0)
    Ja = np.asarray(jac(cv, co_, DT0, nu, dl))
    S_ = 1. / (1. + DT0 * .5 * nu * np.asarray(dl['lam']))
    dN = np.asarray(nlJ(cv, dl, None)[1])
    Jan = np.asarray(dl['A']) + (DT0 * .5 * S_)[:, None] * dN
    Jfl = np.asarray(dl['A']) - (DT0 * .5 * S_)[:, None] * dN
    rel = lambda X_: float(np.linalg.norm(X_ - Ja) / np.linalg.norm(Ja))
    stj = P.Setting(bank, 'jflip', Rp, st.rules['main'], 'frozen', forms=('LSPG',), nrec=NREC, mutation='jflip')
    vj = P.run_query(bank, stj, 'main', 'LSPG', 'CNR', DT0, 'prod', u0, nu, record=True)
    gj, _ = T2.make_certifier(nl, Rp)(vj['stats']['states'], nu, stj.data[('main', 'LSPG')], stj.sched('LSPG', 'CNR', DT0, 'prod'))
    gj = np.asarray(gj)
    C['C6m'] = dict(analytic_vs_autodiff=rel(Jan), jflip_vs_autodiff=rel(Jfl), nonstationary_jflip=int(np.sum(~(gj <= 1e-3))),
                    steps=len(gj), fired=bool(rel(Jfl) > 1e-6), positive_ok=bool(rel(Jan) <= 1e-10))
    del stj, vj
    # C7m coarse-rule Q2 witness (gauss32 -> gauss16 -> gauss8) ------------------------------------------------
    c7 = []
    for rl in ('gauss32', 'gauss16', 'gauss8'):
        st.add_role(bank, 'c7', rl, forms=('GAL',))
        dg['c7'] = st.data[('c7', 'GAL')]
        tabs['c7'], _ = F.precontract(dg['c7'], basis=(tab['D'], tab['V']))
        worst_rho, worst_d = 0., 0.
        for ci in dev:
            vm_ = g_run('etd', .0125, ci)
            vc_ = g_run('etd', .0125, ci, role='c7')
            rho = max(P.rho(np.asarray(nl(jnp.asarray(vm_['W'][kk]), dg['main'])), np.asarray(nl(jnp.asarray(vm_['W'][kk]), dg['c7'])))
                      for kk in range(6))
            worst_rho, worst_d = max(worst_rho, rho), max(worst_d, P.fdist(vm_['f'], vc_['f'], vm_['n0']))
        c7.append(dict(rule=rl, rho=worst_rho, dist=worst_d, fired=bool(worst_rho > .116 or worst_d > 2.5e-4)))
        st.drop_role('c7')
        del tabs['c7'], dg['c7']
        if c7[-1]['fired']:
            break
    C['C7m'] = dict(attempts=c7, fired=bool(c7[-1]['fired']))
    # C8 timing identity: an evolve at the other dt must not reproduce the warm-up hash -----------------------
    ev = evolver('etd')
    hs = lambda W, s: F.sha(np.concatenate([np.asarray(W).ravel(), np.asarray(s['Z'])[:int(s['steps']) + 1].ravel()]))
    Wa, sa = ev(w0, jnp.float64(nu), tab, F.sched(.0125))
    Wa2, sa2 = ev(w0, jnp.float64(nu), tab, F.sched(.0125))
    Wb2, sb2 = ev(w0, jnp.float64(nu), tab, F.sched(.025))
    C['C8'] = dict(repeat_identical=bool(hs(Wa, sa) == hs(Wa2, sa2)), other_dt_differs=bool(hs(Wa, sa) != hs(Wb2, sb2)),
                   fired=bool(hs(Wa, sa) != hs(Wb2, sb2)), positive_ok=bool(hs(Wa, sa) == hs(Wa2, sa2)))
    # C12 guards -------------------------------------------------------------------------------------------------
    c12 = {}
    try:
        check_tab(tabs['qp'], 'main')                    # the production guard, fed the successor table as 'main'
        c12['table_swap'] = False
    except ValueError:
        c12['table_swap'] = True
    try:
        rr_ = dict(rec_of(d0))
        rr_['case_id'] = rec_of(d1)['case_id']           # a field of case d1 requested as d0 (production scorer)
        e0(v['f'], rr_)
        c12['e0_case_swap'] = d0 == d1
    except ValueError:
        c12['e0_case_swap'] = True
    try:
        st.check('main', types.SimpleNamespace(L=2 * L, mesh_hash='x', ops=bank.ops))
        c12['wrong_mesh'] = False
    except ValueError:
        c12['wrong_mesh'] = True
    C['C12'] = dict(c12, fired=bool(all(c12.values())))
    # C13 / C14 / C15 / C18 ------------------------------------------------------------------------------------
    uv, nv, _ = U0[vcase]
    v13 = g_run('etd', .0125, vcase, mutation='f3_scale')
    C['C13'] = dict(case=f'{cases[vcase][0]}|{cases[vcase][1]}', mismatch=g_cert(v13, 'etd', .0125)['worst_mismatch'],
                    fired=bool(g_cert(v13, 'etd', .0125)['worst_mismatch'] > 1e-10))
    v14 = g_run('etd', .0125, d0, mutation='wrong_nu')
    C['C14'] = dict(mismatch=g_cert(v14, 'etd', .0125)['worst_mismatch'], fired=bool(g_cert(v14, 'etd', .0125)['worst_mismatch'] > 1e-10))
    v15 = g_run('ros', DT0, d0, mutation='half_update')
    c15 = g_cert(v15, 'ros', DT0)
    C['C15'] = dict(worst_lin=c15['worst_lin'], fired=bool(c15['worst_lin'] > 1e-9))
    v18 = g_run('etd', .0125, d0, mutation='late_out')
    k18 = v18['steps']
    o18 = float(np.max(np.abs(v18['z_out'][1:] - v18['Z'][[k18 // 5 * j for j in range(1, 6)]])))
    C['C18'] = dict(out_vs_states=o18, fired=bool(o18 > 0.))
    # C22 classifier fixtures (A4.1) ------------------------------------------------------------------------------
    fx = [((1.057, 1.054), 'unresolved'), ((1.054, 1.057), 'unresolved'), ((1.049, 1.053), 'unresolved'),
          ((1.045, 1.044), 'unresolved'), ((1.055, 1.056), 'unresolved'), ((1.06, 1.07), 'unresolved'),
          ((1.06, 1.058), 'growth present in the tested refinements'), ((1.01, 1.012), 'growth absent in the tested refinements')]
    C['C22'] = dict(cases=[dict(pair=p_, expected=e_, got=FS.classify_growth(*p_)) for p_, e_ in fx])
    C['C22']['passed'] = all(c_['got'] == c_['expected'] for c_ in C['C22']['cases'])
    must = ('C1m', 'C2m', 'C16', 'C3m', 'C4m', 'C5', 'C6m', 'C7m', 'C8', 'C12', 'C13', 'C14', 'C15', 'C18')
    pos = ('C1', 'C2', 'C3', 'C22')
    C['summary'] = dict(must_fire={m: bool(C[m]['fired']) for m in must}, positive={p_: bool(C[p_]['passed']) for p_ in pos},
                        demonstrated=dict(C3m=C['C3m']['demonstrated'], C5=C['C5']['ref_converged']),
                        positive_slopes=dict(C3m=C['C3m']['slope_ok'], C5=C['C5']['slope_ok']),
                        all_ok=bool(all(C[m]['fired'] for m in must) and all(C[p_]['passed'] for p_ in pos)
                                    and C['C3m']['demonstrated'] and C['C3m']['slope_ok'] and C['C5']['ref_converged']
                                    and C['C5']['slope_ok'] and C['C6m']['positive_ok'] and C['C8']['positive_ok']))
    return C
