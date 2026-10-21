"""imp-hrcost job orchestration (DESIGN 2, 3, 5, 6; A1-A4): called by hjob.main with a Job."""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

import fit as FT
import gstep as GS
import hlabel as HL
import hrtab as HT
import npcheck as NC
from hrtab import Q

R_LADDER = ['gauss128', 'gauss160', 'gauss192']


def m_of(rule):
    return int(rule[5:]) ** 2 if rule.startswith('gauss') else int(rule[3:])


def run(J):
    cfg = J.cfg
    J.setup()
    if cfg.get('controls_only_setup_check'):
        return
    # ------------------------------------------------------------------------ R (sine, Gauss 128^2) ----
    t = time.perf_counter()
    J.keep.add(('sine', 'gauss128'))
    runs_sine = J.ladder('sine', R_LADDER, lambda q: 'R' if q == 'gauss128' else f'sine_{q}')
    dR, auxR, metaR, wR, XR = J.table('sine', 'gauss128')
    J.q1('R', 'sine', dR, metaR)
    J.phase('R', t)
    # ------------------------------------------------------------------------ F2 ladder (bank) ----
    lad = {}
    if not J.bank_ok:
        J.rep['F2'] = dict(status='not built (P3 failed or bank arms disabled)')
    if J.bank_ok:
        t = time.perf_counter()
        J.keep.add(('bank', 'gauss128'))
        from hjob import ladders
        runs_bank = {}
        for fam, rules in ladders(cfg).items():
            runs_bank.update(J.ladder('bank', rules, lambda q: f'F2_{q}'))
            lad[fam] = rules
        if any(q not in runs_bank for q in R_LADDER):           # fit parent (Gauss 128^2) and its successor / check
            runs_bank.update(J.ladder('bank', R_LADDER, lambda q: f'F2_{q}'))
        J.phase('F2_q2', t)
        t = time.perf_counter()
        cands = sorted([q for rules in lad.values() for q in rules[:-2] if m_of(q) <= cfg.get('mmax', 16384)], key=m_of)
        status, first = {}, {'label': None, 'label_projected': None}
        unres = {'label': False, 'label_projected': False}
        for q in cands:
            labs = {k: J.rep['q2'][f'F2_{q}'][k] for k in first}
            need = any(labs[k] == 'pass' and first[k] is None for k in first) or q == 'gauss128'
            if need:
                d, aux, meta, _, _ = J.table('bank', q)
                l1 = J.q1(f'F2_{q}', 'bank', d, meta)
                for k in first:
                    if labs[k] == 'pass' and l1 == 'pass' and first[k] is None:
                        first[k] = q
                status[q] = dict(q2=labs['label'], q2_projected=labs['label_projected'], q1=l1)
                if q != 'gauss128':
                    J.drop(keep=list(J.keep))
            else:
                status[q] = dict(q2=labs['label'], q2_projected=labs['label_projected'], q1='not evaluated')
            for k in first:
                if first[k] is None and labs[k].startswith('unverified'):
                    unres[k] = True
        J.rep['F2'] = dict(candidates=cands, status=status, first_qualified=first['label'],
                           first_qualified_projected=first['label_projected'],
                           mstar_identified=bool(first['label'] is not None and not unres['label']),
                           unresolved_below=unres['label'], unresolved_below_projected=unres['label_projected'],
                           ladders=lad)
        J.phase('F2_q1', t)
    # ------------------------------------------------------------------------ fits ----
    fitted = {}
    fams = [('F3a', ('sine', 'gauss128')), ('F3b', ('sine', 'gauss128'))]
    if J.bank_ok:
        fams.append(('F2F3', ('bank', 'gauss128')))
    for fam, pk in fams:
        if fam not in cfg.get('fits', ['F3a', 'F3b', 'F2F3']):
            continue
        t = time.perf_counter()
        rules = J.fit_family(fam, pk)
        proj = pk[0]
        par_runs = runs_sine if proj == 'sine' else runs_bank
        par_name = 'R' if proj == 'sine' else 'F2_gauss128'
        for k, (d, aux, meta) in rules.items():
            name = f'{fam}_k{k}'
            own = J.own_runs(proj, None, J.q2_set, d=d)
            lab = J.q2(name, proj, (d, aux, meta), own, J.table(*pk)[:3], par_runs['gauss128'], par_runs['gauss160'])
            for key in ('label', 'label_projected'):          # the parent's label of the same reading (code audit 1)
                if J.rep['q2'][name][key] == 'pass' and J.rep['q2'][par_name][key] != 'pass':
                    J.rep['q2'][name][key] = 'unverified (parent not converged)'
            J.q1(name, proj, d, meta)
            fitted[name] = (proj, d, aux, meta)
        J.phase(f'fit_{fam}', t)
    J.fitted = fitted
    # ------------------------------------------------------------------------ controls (smoke) ----
    if cfg.get('controls'):
        t = time.perf_counter()
        controls(J, runs_sine, runs_bank if J.bank_ok else None)
        J.phase('controls', t)
    # ------------------------------------------------------------------------ timing ----
    t = time.perf_counter()
    tc = cfg.get('timing', {})
    pairs = [('AA', dR, 'sine')]
    if J.bank_ok:
        for q in dict.fromkeys([J.rep['F2']['first_qualified'], J.rep['F2']['first_qualified_projected'], 'gauss128']):
            if q is not None:
                pairs.append((f'F2_{q}', J.table('bank', q)[0], 'bank'))
    for name, (proj, d, aux, meta) in fitted.items():
        pairs.append((name, d, proj))
    dup = HT.duplicated((dR, auxR, metaR))
    pairs.append(('dup', dup[0], 'sine'))
    scored = {}
    for name in [p_[0] for p_ in pairs] + ['R']:
        f = J.out / 'arrays' / f'q1_{"R" if name == "AA" else name}.npz'
        if f.exists():
            z = np.load(f)
            scored[name] = {str(c): z['W'][i] for i, c in enumerate(z['case_ids'])}
    J.timing((dR, 'sine'), pairs, J.sets[tc.get('cases', 'all38')], tc.get('reps', 5), tc.get('burn', 3), scored=scored)
    J.rep['jac_bench'] = dict(R=J.jac_bench(dR), dup=J.jac_bench(dup[0]))
    J.rep['jac_bench']['ratio'] = J.rep['jac_bench']['dup'][0] / J.rep['jac_bench']['R'][0]
    J.phase('timing', t)


# ============================================================================== controls ====

def _np_tables(d):
    return {k: np.asarray(v) for k, v in d.items()}


def _np_rhs(c, nu, t):
    return -t['P'].T @ ((t['Gq'] @ c) * (t['Gs'] @ c)) + nu * (t['L0'] @ c)


def _np_eta(states, k, nu, t, U, dt):
    cn, c1 = states[k], states[k + 1]
    rv = c1 - cn - .5 * dt * (_np_rhs(c1, nu, t) + _np_rhs(cn, nu, t))
    return float(np.linalg.norm(U @ rv) / np.linalg.norm(U @ cn))


def controls(J, runs_sine, runs_bank):
    C = J.rep['controls']
    B = J.base
    dR, auxR, metaR, wR, XR = J.table('sine', 'gauss128')
    dev2 = J.sets['dev2']
    # ---- C1 coarse rules
    for proj, runs0 in (('sine', runs_sine), ('bank', runs_bank)):
        if runs0 is None:
            continue
        lad = ['gauss16', 'gauss24', 'gauss32', 'gauss48']
        own = {q: J.own_runs(proj, q, J.q2_set) for q in lad}
        pick = None
        for i in range(3):
            q, qp, qpp_ = lad[i], lad[i + 1], lad[min(i + 2, 3)]
            if all(own[x][c]['valid'] for x in (q, qp, qpp_) for c in J.q2_set):
                pick = i
                break
        if pick is None:
            C[f'C1_{proj}'] = dict(fired=False, fixture='none valid')
            continue
        q, qp = lad[pick], lad[pick + 1]
        qpp = lad[min(pick + 2, 3)]
        J.q2(f'C1_{proj}', proj, J.table(proj, q)[:3], own[q], J.table(proj, qp)[:3], own[qp], own[qpp])
        det = J.rep['q2'][f'C1_{proj}']['detail']
        C[f'C1_{proj}'] = dict(fixture=q, successor=qp, fired=bool(det.get('fail_rollout') and det.get('fail_rho')),
                               rollout_max=det['rollout_max'], rho_max=det['rho_max'])
        J.drop(keep=list(J.keep))
    # ---- C2 permuted weights (F3b k8, else F3a k8)
    fx = next((n for n in ('F3b_k8', 'F3a_k8') if n in J.fitted), None)
    if fx:
        fam = fx.split('_')[0]
        z = np.load(J.out / 'arrays' / f'fit_{fam}_k8.npz')
        perm = np.random.default_rng(0).permutation(len(z['w']))
        wperm = z['w'][perm]
        dist = float(np.linalg.norm(wperm - z['w']) / np.linalg.norm(z['w']))
        d2, a2, m2 = HT.fitted((dR, auxR, metaR), XR, wR, z['idx'], wperm, f'{fx}-perm')
        own = J.own_runs('sine', None, J.q2_set, d=d2)
        J.q2('C2', 'sine', (d2, a2, m2), own, (dR, auxR, metaR), runs_sine['gauss128'], runs_sine['gauss160'])
        C['C2'] = dict(fixture=fx, weight_change=dist, fixture_valid=dist >= .5,
                       label=J.rep['q2']['C2']['label'], fired=bool(J.rep['q2']['C2']['label'] == 'fail' and dist >= .5))
    else:
        C['C2'] = dict(fired=False, fixture='no k8 fit built')
    # ---- C3 corrupted states (R, dev cases)
    zq = np.load(J.out / 'arrays' / 'q1_R.npz')
    ids = list(zq['case_ids'])
    tR = _np_tables(dR)
    U = np.asarray(J.U['sine'])
    S = J.S
    res = {}
    rng = np.random.default_rng(0)
    for cid in J.dev:
        j = ids.index(cid)
        st0 = zq['states'][j].copy()
        nu = J.U0[cid][1]
        base_ok = bool(np.all(HL.certified(zq['eta'][j])))
        rr = {}
        for tag, ks in (('a', [S]), ('b', [25]), ('c', [10, 20, 30, 40])):
            st = st0.copy()
            for k in ks:
                v = rng.standard_normal(J.Rp)
                st[k] = st[k] + 1e-6 * np.linalg.norm(st[k]) * v / np.linalg.norm(v)
            eta, _, _ = J.cert(jnp.asarray(st), nu, dR, J.U['sine'], J.dt, S)
            flagged = sorted(np.nonzero(~HL.certified(np.asarray(eta)))[0].tolist())
            want = sorted({kk - 1 for kk in ks} | {kk for kk in ks if kk < S})
            np_cross = all(_np_eta(st, kk, nu, tR, U, J.dt) > HL.ETA_BAR for kk in want)
            rr[tag] = dict(flagged=flagged, expected=want, ok=bool(flagged == want and np_cross),
                           eta_perturbed=np.asarray(eta).tolist())
            np.savez(J.out / 'arrays' / f'C3_{cid.replace("|", "_")}_{tag}.npz', states=st, eta=np.asarray(eta), ks=np.asarray(ks))
        res[cid] = dict(base_certified=base_ok, **rr)
    rows = {cid: dict(valid=True, eta=np.asarray(res[cid]['c']['eta_perturbed']), eta_rec=np.asarray(res[cid]['c']['eta_perturbed']),
                      tight_valid=True, dist_tight=0., tighter_valid=True, dist_tt=0.) for cid in J.dev}
    labc = HL.q1_label(rows, J.dev)
    C['C3'] = dict(per_case={c: {k: (v if k == 'base_certified' else dict(flagged=v['flagged'], expected=v['expected'], ok=v['ok']))
                                 for k, v in r_.items()} for c, r_ in res.items()},
                   pooled_c=labc['nonconv_frac'], label_c=labc['label'],
                   fired=bool(all(r_['base_certified'] and r_['a']['ok'] and r_['b']['ok'] and r_['c']['ok'] for r_ in res.values())
                              and labc['label'] == 'fail' and labc['nonconv_frac'] > .01))
    # ---- C4 NaN table
    dn, an, mn = HT.nan_injected((dR, auxR, metaR))
    o = J.run(dn, 'sine', dev2[0])
    lab4 = J.q1('C4', 'sine', dn, mn, cases=dev2)
    C['C4'] = dict(valid=o['valid'], q1_label=lab4, fired=(not o['valid']) and lab4 == 'fail')
    # ---- C5 duplicated rule
    dd = HT.duplicated((dR, auxR, metaR))
    eq = []
    for cid in dev2:
        a_, b_ = J.run(dR, 'sine', cid), J.run(dd[0], 'sine', cid)
        eq.append(float(np.linalg.norm(a_['W'] - b_['W']) / np.linalg.norm(a_['W'])))
    bR, bD = J.jac_bench(dR), J.jac_bench(dd[0])
    tb = bR[0], bD[0]
    C['C5'] = dict(max_rel_diff=max(eq), jac_R=tb[0], jac_dup=tb[1], ratio=tb[1] / tb[0], reps_R=bR[2], reps_dup=bD[2],
                   fired=bool(max(eq) <= 1e-12 and tb[1] / tb[0] >= 1.5))
    # ---- C7 leakage
    try:
        D = FT.train_draw_2d(0, 576)[:4]
        FT.assert_not_dev_val(np.concatenate([D, J.cases[0][2][None]]), np.stack([c[2] for c in J.cases]))
        C['C7'] = dict(fired=False)
    except AssertionError as e:
        C['C7'] = dict(fired=True, msg=str(e))
    # ---- P1 fitted machinery at parent weights
    dp, ap, mp = HT.fitted((dR, auxR, metaR), XR, wR, np.arange(len(wR)), wR, 'parent-as-fit')
    eq = []
    for cid in dev2:
        a_, b_ = J.run(dR, 'sine', cid), J.run(dp, 'sine', cid)
        eq.append(float(np.linalg.norm(a_['W'] - b_['W']) / np.linalg.norm(a_['W'])))
    C['P1'] = dict(max_rel_diff=max(eq), passed=max(eq) <= 1e-12)
    # ---- P3
    if 'bank_ops' in J.rep['setup']:
        C['P3'] = dict(J.rep['setup']['bank_ops'], stiff_min_eig=float(np.linalg.eigvalsh(np.asarray(B.bank_ops()[1]))[0]))
        C['P3']['passed'] = bool(C['P3']['p3_pass'] and C['P3']['mass_min_eig'] > 0 and C['P3']['stiff_min_eig'] > 0)
    # ---- P4 uncontracted RHS and Jacobians
    states = [zq['W'][ids.index(c)] for c in J.dev]
    nl = Q.tested_value('point', J.L)
    p4 = dict(rhs=[], jac=[])
    for c6 in states:
        for c in c6:
            cj = jnp.asarray(c)
            nu = .05
            ref = -jax.scipy.linalg.solve_triangular(B.Rm, B.Qm.T @ (nl(cj, dict(Gq=dR['Gq'], Gs=dR['Gs'], Psi=auxR['T']))
                                                                     + nu * B.lam * (B.A @ cj)), lower=False)
            got = GS.rhs(cj, nu, dR)
            p4['rhs'].append(float(jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref)))
    fixtures = [('R', dR)]
    if runs_bank is not None:
        fixtures.append(('F2_gauss96', J.table('bank', 'gauss96')[0]))
    fx4 = next((n for n in ('F3a_k8', 'F3b_k8') if n in J.fitted), None)
    if fx4:
        fixtures.append((fx4, J.fitted[fx4][1]))
    for nm, d in fixtures:
        for c in np.concatenate(states):
            cj = jnp.asarray(c)
            Ja = GS.rhs_jac(cj, .05, d)
            Jd = jax.jacfwd(lambda x: GS.rhs(x, .05, d))(cj)
            p4['jac'].append((nm, float(jnp.linalg.norm(Ja - Jd) / jnp.linalg.norm(Jd))))
    need = 3 if runs_bank is not None else 2
    C['P4'] = dict(rhs_max=max(p4['rhs']), jac_max={nm: max(v for n_, v in p4['jac'] if n_ == nm) for nm, _ in fixtures},
                   fixtures=[nm for nm, _ in fixtures],
                   passed=bool(len(fixtures) == need and max(p4['rhs']) <= 1e-12 and max(v for _, v in p4['jac']) <= 1e-10))
    # ---- P5 independent bank operators
    if runs_bank is not None:
        C['P5'] = p5(J, states)
    # ---- P6 invalid anchors, successor, check
    cid = dev2[0]
    o = J.run(dR, 'sine', cid)
    tt = J.run(dR, 'sine', cid, sch=GS.sched(J.dt, 1e-12, budget=1))
    ks = np.nonzero(tt['reason'] == 0)[0]
    tR = _np_tables(dR)
    missed = bool(len(ks)) and any(_np_eta(tt['states'], int(k), J.U0[cid][1], tR, U, J.dt) > 1e-12 for k in ks[:5])
    tight_valid = tt['valid'] and bool(np.all(tt['reason'] == 1))
    lab6 = HL.q1_label({cid: dict(valid=o['valid'], eta=np.zeros(S), eta_rec=np.zeros(S), tight_valid=tight_valid,
                                  dist_tight=HL.fdist(o['f257'], tt['f257'], o['n0']))}, [])
    succ_nan = {c: J.run(dn, 'sine', c) for c in J.q2_set}
    J.q2('P6b', 'sine', (dR, auxR, metaR), runs_sine['gauss128'], (dn, an, mn), succ_nan, runs_sine['gauss160'])
    J.q2('P6c', 'sine', (dR, auxR, metaR), runs_sine['gauss128'], J.table('sine', 'gauss160')[:3],
         runs_sine['gauss160'], succ_nan)
    C['P6'] = dict(a_missed_tol=missed, a_label=lab6['label'], b_label=J.rep['q2']['P6b']['label'],
                   c_label=J.rep['q2']['P6c']['label'],
                   passed=bool(missed and lab6['label'] == 'fail' and J.rep['q2']['P6b']['label'].startswith('unverified')
                               and J.rep['q2']['P6c']['label'].startswith('unverified')))
    J.save()


def p5(J, states):
    """P5 (A2.6, A6): (i) bank RHS vs host NumPy M^-1[-T(c) - nu K c] with npcheck tables; (ii') AD-Laplacian diffusion
    identity; (iii') advection energy identity; (iv') host NumPy assembly of Mass / Stiff at Gauss 768^2; mesh sums
    reported descriptively."""
    import hops as H
    import sep_common as sc
    B = J.base
    Mm, Km, L0, _ = B.bank_ops()
    Mm_, Km_ = np.asarray(Mm), np.asarray(Km)
    out = dict(rhs_np=[], lap_identity=[], energy=[])
    # (i)
    X, w = HT.rule('gauss96')
    d = J.table('bank', 'gauss96')[0]
    Gq, Gx, Gy = NC.values_grads(X, J.Rp)
    nu = .05
    allst = np.concatenate(states)                                        # every dev6 output state (A6, code audit 1)
    for c in allst:
        u, s_ = Gq @ c, (Gx + Gy) @ c
        ref = np.linalg.solve(Mm_, -(w[:, None] * Gq).T @ (u * s_) - nu * Km_ @ c)
        got = np.asarray(GS.rhs(jnp.asarray(c), nu, d))
        out['rhs_np'].append(float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
    # (ii') and (iii') at Gauss 1024^2 (device, chunked)
    X, w = HT.rule('gauss1024')
    mdl = B.bank.mdl
    Tp = jnp.asarray(mdl.T[:, :J.Rp])

    def lap1(x):
        f = lambda y: sc.features(mdl.params, y[None])[0]
        tot = 0.
        for e in (jnp.array([1., 0.]), jnp.array([0., 1.])):
            g = lambda y: jax.jvp(f, (y,), (e,))[1]
            tot = tot + jax.jvp(g, (x,), (e,))[1]
        return tot
    lapv = jax.jit(jax.vmap(lap1))
    C6 = jnp.asarray(allst)                                                 # (n, r)
    GtLap = jnp.zeros((len(allst), J.Rp))
    Tc = jnp.zeros((len(allst), J.Rp))
    for s0 in range(0, len(X), 16384):
        Xs = X[s0:s0 + 16384]
        ws = jnp.asarray(w[s0:s0 + 16384])
        g, gx, gy = mdl.values_grads(Xs, J.Rp)
        lg = lapv(jnp.asarray(Xs)) @ Tp
        U, Lu, S_ = C6 @ g.T, C6 @ lg.T, C6 @ (gx + gy).T              # (n, chunk)
        GtLap = GtLap + (ws[None] * Lu) @ g
        Tc = Tc + (ws[None] * U * S_) @ g
    for j in range(len(allst)):
        c = C6[j]
        out['lap_identity'].append(float(jnp.linalg.norm(GtLap[j] + Km @ c) / jnp.linalg.norm(Km @ c)))
        out['energy'].append(float(jnp.abs(c @ Tc[j]) / (jnp.linalg.norm(c) * jnp.linalg.norm(Tc[j]))))
    # (iv') host assembly at Gauss 768^2
    X, w = NC.gauss_rule(768)
    Mh, Kh = np.zeros((J.Rp, J.Rp)), np.zeros((J.Rp, J.Rp))
    for s0 in range(0, len(X), 32768):
        g, gx, gy = NC.values_grads(X[s0:s0 + 32768], J.Rp)
        ws = w[s0:s0 + 32768][:, None]
        Mh += g.T @ (ws * g)
        Kh += gx.T @ (ws * gx) + gy.T @ (ws * gy)
    Md, Kd = (np.asarray(a_) for a_ in HT.mass_stiff(B.bank, J.Rp, 'gauss768'))
    out['host_mass'] = float(np.linalg.norm(Mh - Md) / np.linalg.norm(Md))
    out['host_stiff'] = float(np.linalg.norm(Kh - Kd) / np.linalg.norm(Kd))
    # descriptive mesh sums
    L, h = J.L, 1. / J.L
    Gm = np.asarray(H.bank_apply(B.G, jnp.eye(J.Rp)))
    out['mesh_mass_rel'] = float(np.linalg.norm(h * h * Gm.T @ Gm - Mm_) / np.linalg.norm(Mm_))
    # descriptive only (A6): five-point stiffness and the FD right-hand side projected on the mesh (dev case 0 states)
    G3 = Gm.reshape(L - 1, L - 1, J.Rp)
    Gp = np.pad(G3, ((1, 1), (1, 1), (0, 0)))
    lapG = ((Gp[2:, 1:-1] + Gp[:-2, 1:-1] + Gp[1:-1, 2:] + Gp[1:-1, :-2] - 4 * G3) / h ** 2).reshape(-1, J.Rp)
    Kh = -h * h * Gm.T @ lapG
    out['mesh_stiff_rel'] = float(np.linalg.norm(.5 * (Kh + Kh.T) - Km_) / np.linalg.norm(Km_))
    del Gp, lapG, Kh
    Mh = h * h * Gm.T @ Gm
    dbank = J.table('bank', 'gauss128')[0]
    out['mesh_fd_rhs_rel'] = []
    for c in states[0]:
        U_ = (Gm @ c).reshape(L - 1, L - 1)
        Up = np.pad(U_, 1)
        ds = (Up[2:, 1:-1] - Up[:-2, 1:-1] + Up[1:-1, 2:] - Up[1:-1, :-2]) / (2 * h)
        lu = (Up[2:, 1:-1] + Up[:-2, 1:-1] + Up[1:-1, 2:] + Up[1:-1, :-2] - 4 * U_) / h ** 2
        cm = np.linalg.solve(Mh, h * h * Gm.T @ (-U_ * ds + nu * lu).reshape(-1))
        got = np.asarray(GS.rhs(jnp.asarray(c), nu, dbank))
        out['mesh_fd_rhs_rel'].append(float(np.linalg.norm(got - cm) / np.linalg.norm(cm)))
    del Gm, G3
    out['passed'] = bool(max(out['rhs_np']) <= 1e-10 and max(out['lap_identity']) <= 1e-5 and max(out['energy']) <= 1e-6
                         and out['host_mass'] <= 1e-11 and out['host_stiff'] <= 1e-11)
    return out
