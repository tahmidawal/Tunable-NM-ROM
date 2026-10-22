"""imp-faststep POD block (DESIGN 3.5, A1.8, A2.2, A2.6, A3.1, A3.4, A5.1, A5.5, A5.11, A5.12): POD-Galerkin +
Galerkin-ECSW with the same ETDRK4 step, the hyper-reduction-free POD-Galerkin ETDRK4, and the hr-baselines registered
picks replayed natively (5L).

POD-Galerkin ODE of the central FOM (hrcore operator): c' = -Phi^T N_h(Phi c) - nu K c, K = Phi^T(-Lap_h)Phi = W D_P W^T,
y = W^T c. Advection providers (precontracted, used by the stepper) and their un-precontracted twins (certificates):

  ecsw   N_y(y) = Wt^T [N_E(PstW y)],  PstW = Pst W, Wt = diag(xi) Phi_E W      | twin: W^T Pst0^T(xi N_E(Pst W y))
  none   N_y(y) = PhiW^T N_h(PhiW y),   PhiW = Phi W                            | twin: W^T Phi^T N_h(Phi W y)

N_E / N_h: the central advection u_i (L/2) sum_a (u_{+a} - u_{-a}) (hrcore.stencil_F / adv_central). The stepper is
fscore.make_evolve('etd', ...) with tab D = D_P, VtRm = W^T, RinvV = W. Every large array is a jit argument.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import fscore as F  # noqa: E402
import fsselect as FS  # noqa: E402
import p1common as P  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

HRB = HERE.parent / 'jcp-hr-baselines'
if str(HRB) not in sys.path:
    sys.path.append(str(HRB))          # appended: never shadows the p1-scaling modules of the same name
import hrcore as H  # noqa: E402
import rom as RM  # noqa: E402

GATE = 1e-10
_TIMING = {}


def sha_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rule_id(nodes, weights):
    o = np.argsort(nodes)
    return hashlib.sha256(np.ascontiguousarray(np.asarray(nodes)[o]).tobytes() +
                          np.ascontiguousarray(np.asarray(weights, np.float64)[o]).tobytes()).hexdigest()


def make_adv(kind, L):
    sh = (L - 1, L - 1)
    if kind == 'ecsw':
        def adv(y, tb):
            us = jnp.einsum('esq,q->es', tb['PstW'], y)
            return tb['Wt'].T @ H.stencil_F(us, 0., L, 2)[1]

        def twin(y, tb):
            c = tb['W'] @ y
            us = jnp.einsum('esr,r->es', tb['Pst'], c)
            return tb['W'].T @ (tb['Pst'][:, 0, :].T @ (tb['xi'] * H.stencil_F(us, 0., L, 2)[1]))
    else:
        def adv(y, tb):
            u = tb['PhiW'] @ y
            return tb['PhiW'].T @ H.adv_central(u.reshape(sh), L).reshape(-1)

        def twin(y, tb):
            u = tb['Phi'] @ (tb['W'] @ y)
            return tb['W'].T @ (tb['Phi'].T @ H.adv_central(u.reshape(sh), L).reshape(-1))
    return adv, twin


def force_c(kind, L):
    """Reduced advection force in c coordinates (A2.2 force ratio): ECSW sum_e xi_e Phi_e^T N_e, none Phi^T N_h."""
    sh = (L - 1, L - 1)
    if kind == 'ecsw':
        return jax.jit(lambda c, tb: tb['Pst'][:, 0, :].T @ (tb['xi'] * H.stencil_F(jnp.einsum('esr,r->es', tb['Pst'], c), 0., L, 2)[1]))
    return jax.jit(lambda c, tb: tb['Phi'].T @ H.adv_central((tb['Phi'] @ c).reshape(sh), L).reshape(-1))


def successor_plan(rules):
    """Production successor construction (A2.2/A3.1/A5.11), shared with the offline controls C19/C21: rules = available
    [(kappa, nodes, weights)] in kappa order. Members = one per distinct rule (smallest kappa name, all names kept) plus
    'none'; successor of a member (by rule id) = fsselect.pod_successors over the full ordered list."""
    members, seen, order = [], {}, []
    for k, nd, wt in rules:
        hid = rule_id(nd, wt)
        order.append((hid, hid))
        if hid in seen:
            seen[hid]['names'].append(k)
            continue
        seen[hid] = dict(kind='ecsw', k=k, nd=nd, wt=wt, hid=hid, names=[k])
        members.append(seen[hid])
    members.append(dict(kind='none', k=None, hid='none', names=[]))
    succ = FS.pod_successors(order + [('none', 'none')])
    return members, succ


def run(cfg, L, cases, ids, dev, U0, out, e0, rec_of, el):
    pc = cfg['pod']
    tdir = Path(pc['tune_dir'])
    rec = dict(tune_dir=str(tdir), members={}, native={}, ranks={}, gecsw={}, classification=[],
               labels=['snapshot-limited basis (hr-baselines A1.3)', 'tuning-budget band unresolved (hr-baselines 2D)',
                       'developmental, in-sample, unadjusted', 'no H-base conclusion at 4096^2'])
    t0 = time.perf_counter()
    rec['phi_sha256'] = sha_file(tdir / 'artifacts' / 'Phi.npz')
    assert rec['phi_sha256'] == pc['phi_sha256'], 'POD basis differs from the pinned digest'
    for nm, h in pc['native_sha256'].items():
        p_ = tdir / ('artifacts/' + nm if nm.startswith('HR~') else nm)
        assert sha_file(p_) == h, f'native artifact {nm} differs from its pinned digest'
    zf = np.load(tdir / 'artifacts' / 'Phi.npz')
    rmax = max(pc['ranks'] + [int(m.split('|')[1]) for m in pc.get('native', [])])
    Phi_all = jnp.asarray(np.asarray(zf['Phi'])[:, :rmax])
    trust = {int(r): float(t) for r, t in zip(zf['ranks'], zf['trust'])}
    del zf
    rec['load_seconds'] = time.perf_counter() - t0
    lat = jnp.asarray(H.lattice_nodes(2, L))
    cid = lambda i: f'{cases[i][0]}|{cases[i][1]}'
    U0i = {i: jnp.asarray(np.asarray(U0[i][0])[1:-1, 1:-1].reshape(-1)) for i in ids}
    init = jax.jit(lambda u, Phi: Phi.T @ u)
    decode = jax.jit(lambda Wc, Phi: Wc @ Phi.T)
    to_lat = lambda fl: H.to_lattice(np.asarray(fl[:, lat]), 2)
    EVC = {}
    for r in pc['ranks']:
        Phi = Phi_all[:, :r]
        LPhi = H.lap_cols(Phi, 2, L)
        K = -(Phi.T @ LPhi)
        del LPhi
        K = .5 * (K + K.T)
        D, Wm = jnp.linalg.eigh(K)
        g_ = dict(D_min=float(D[0]), D_max=float(D[-1]), phi_orth=float(jnp.max(jnp.abs(Phi.T @ Phi - jnp.eye(r)))),
                  eig_resid=float(jnp.linalg.norm(K @ Wm - Wm * D[None]) / jnp.linalg.norm(D)),
                  w_orth=float(jnp.max(jnp.abs(Wm.T @ Wm - jnp.eye(r)))), trust=trust.get(r))
        assert g_['D_min'] > 0 and g_['eig_resid'] <= GATE and g_['w_orth'] <= GATE and g_['phi_orth'] <= 1e-8, g_   # A5.1
        rec['ranks'][str(r)] = g_
        gpath = HERE / pc['gecsw'].format(r=r)
        z = np.load(gpath)
        rules = [(k, np.asarray(z[f'k{k}_nodes']), np.asarray(z[f'k{k}_weights'])) for k in pc['kappas'] if f'k{k}_nodes' in z.files]
        rec['gecsw'][str(r)] = dict(file=str(pc['gecsw'].format(r=r)), sha256=sha_file(gpath), available=[k for k, _, _ in rules])
        members, succ = successor_plan(rules)
        TB = {}
        for m in members:
            if m['kind'] == 'ecsw':
                idx = H.stencil_index(m['nd'], 2, L)
                Pst = H.stencil_rows(Phi, idx)
                xi = jnp.asarray(m['wt'])
                tb = dict(PstW=jnp.einsum('esr,rq->esq', Pst, Wm), Wt=(xi[:, None] * Pst[:, 0, :]) @ Wm, Pst=Pst, xi=xi, W=Wm,
                          D=D, VtRm=Wm.T, RinvV=Wm)
            else:
                tb = dict(PhiW=Phi @ Wm, Phi=Phi, W=Wm, D=D, VtRm=Wm.T, RinvV=Wm)
            TB[m['hid']] = (jax.block_until_ready(tb), F.table_fingerprint(tb))

        def guard(tb, fps=frozenset(v[1] for v in TB.values())):
            if F.table_fingerprint(tb) not in fps:
                raise ValueError('POD table differs from its manifest')
        fc = {kd: force_c(kd, L) for kd in ('ecsw', 'none')}
        mname = lambda m, dt: f'pod-{m["kind"]}|{r}|{"k%d" % m["k"] if m["k"] else "-"}|ETDRK4|{dt}'
        for dt in pc['dts']:
            for m in members:
                tb, fp = TB[m['hid']]
                mid = mname(m, dt)
                adv, twin = make_adv(m['kind'], L)
                key = (m['kind'], r)
                if key not in EVC:
                    EVC[key] = (jax.jit(F.make_evolve('etd', adv)), jax.jit(F.make_evolve('etd', twin)), F.make_cert_etd_adv(twin))
                ev, evt, cert = EVC[key]
                rows, W_all, Z_all, C0, MIS = [], [], [], [], []
                for ci in ids:
                    _, nu, n0 = U0[ci]
                    guard(tb)
                    c0 = init(U0i[ci], Phi)
                    W, s = ev(c0, jnp.float64(nu), tb, F.sched(dt))
                    f = to_lat(decode(W, Phi))
                    k_ = int(s['steps'])
                    Z = np.asarray(s['Z'])[:k_ + 1]
                    keep = k_ // 5
                    Wn, c0n = np.asarray(W), np.asarray(c0)
                    co = {a: jnp.asarray(b) for a, b in F.etd_coefs_indep(-nu * np.asarray(D), dt).items()}
                    mis = np.asarray(cert(jnp.asarray(Z), tb, co, k_))
                    y0_check = float(np.linalg.norm(Z[0] - np.asarray(Wm).T @ c0n) / (np.linalg.norm(Z[0]) + 1e-300))
                    cW = Z[[keep * j for j in range(6)]] @ np.asarray(Wm).T
                    W_check = float(np.max(np.linalg.norm(Wn - cW, axis=1) / (np.linalg.norm(Wn, axis=1) + 1e-300)))
                    valid = bool(np.isfinite(Wn).all() and np.isfinite(Z).all() and np.isfinite(float(s['stage_max']))
                                 and y0_check <= 1e-12 and W_check <= 1e-12)
                    row = dict(member=mid, case=cid(ci), valid=valid, growth=float(s['growth']), stage_max=float(s['stage_max']),
                               cert_worst=float(np.max(mis)), cert_ok=bool(np.all(mis <= 1e-10)), y0_check=y0_check,
                               W_check=W_check, e0=e0(f, rec_of(ci)), rule=m['hid'])
                    row['stable'] = bool(row['growth'] <= 1.05 and np.isfinite(row['stage_max']))
                    if ci in dev:
                        Wt_, _ = evt(c0, jnp.float64(nu), tb, F.sched(dt))
                        row['twin_dist'] = P.fdist(f, to_lat(decode(Wt_, Phi)), n0)
                        h_ev = F.sha(np.concatenate([Wn.ravel(), Z.ravel()]))
                        _TIMING.setdefault(mid, dict(r=r, dt=dt, kind=m['kind'], ev=ev, tb=tb, Phi=Phi, init=init, decode=decode,
                                                     U0i=U0i, U0=U0, ref={}, guard=guard))['ref'][ci] = dict(W=Wn, h_ev=h_ev)
                    rows.append(row)
                    W_all.append(Wn)
                    Z_all.append(Z)
                    C0.append(c0n)
                    MIS.append(mis)
                rec['members'][mid] = dict(r=r, kappa=m['k'], kappas=m['names'], dt=dt, kind=m['kind'], rule=m['hid'],
                                           rows=rows, successor=None if m['kind'] == 'none' else
                                           mname(next(x for x in members if x['hid'] == succ[m['hid']]), dt),
                                           size=None if m['kind'] == 'none' else int(len(m['nd'])))
                np.savez_compressed(out / f'P_{mid.replace("|", "_")}.npz', W=np.stack(W_all), Z=np.stack(Z_all), c0=np.stack(C0),
                                    cert_mismatch=np.stack(MIS), case_ids=np.array([cid(i) for i in ids]))
                print('POD', mid, round(max(rw['e0']['e'] for rw in rows), 5), el(), flush=True)
                # A5.5: classify every tripped run with ETDRK4 refinements of the same POD ODE
                for j, ci in enumerate(ids):
                    if rows[j]['stable']:
                        continue
                    _, nu, _ = U0[ci]
                    gs, oks, ev_ = [], [], {}
                    for f_ in (8, 16):
                        W2, s2 = ev(init(U0i[ci], Phi), jnp.float64(nu), tb, F.sched(dt / f_))
                        k2 = int(s2['steps'])
                        Z2 = np.asarray(s2['Z'])[:k2 + 1]
                        co2 = {a: jnp.asarray(b) for a, b in F.etd_coefs_indep(-nu * np.asarray(D), dt / f_).items()}
                        m2 = np.asarray(cert(jnp.asarray(Z2), tb, co2, k2))
                        gs.append(float(s2['growth']))
                        oks.append(bool(np.isfinite(Z2).all() and np.isfinite(float(s2['stage_max'])) and np.isfinite(np.asarray(W2)).all()
                                        and np.all(m2 <= 1e-10)))
                        ev_.update({f'Z_{f_}': Z2, f'W_{f_}': np.asarray(W2), f'mismatch_{f_}': m2, f'stage_max_{f_}': float(s2['stage_max'])})
                    np.savez_compressed(out / f"CLS_{mid.replace('|', '_')}_{cid(ci).replace('|', '_')}.npz", **ev_)
                    rec['classification'].append(dict(member=mid, case=cid(ci), g8=gs[0], g16=gs[1], refined_ok=all(oks),
                                                      label=FS.classify_growth(gs[0], gs[1], all(oks))))
            # Q2-POD on own states (A2.2/A3.1): rollout distance to the successor and the force ratio at own states
            for m in members[:-1]:
                mid = mname(m, dt)
                sid = rec['members'][mid]['successor']
                Wm_ = np.load(out / f'P_{mid.replace("|", "_")}.npz')['W']
                Ws_ = np.load(out / f'P_{sid.replace("|", "_")}.npz')['W']
                tbm = TB[m['hid']][0]
                sm = next(x for x in members if x['hid'] == succ[m['hid']])
                tbs = TB[sm['hid']][0]
                for j, ci in enumerate(ids):
                    n0 = U0[ci][2]
                    fm = to_lat(decode(jnp.asarray(Wm_[j]), Phi))
                    fs = to_lat(decode(jnp.asarray(Ws_[j]), Phi))
                    rho = [P.rho(np.asarray(fc[m['kind']](jnp.asarray(Wm_[j, q]), tbm)),
                                 np.asarray(fc[sm['kind']](jnp.asarray(Wm_[j, q]), tbs))) for q in range(6)]
                    srow = rec['members'][sid]['rows'][j]
                    rec['members'][mid]['rows'][j].update(succ_dist=P.fdist(fm, fs, n0), succ_rho=rho,
                                                          succ_certified=bool(srow['valid'] and srow['stable'] and srow['cert_ok']))
        del TB, Phi
    # ---------------------------------------------------- 5L: hr-baselines registered picks, native code ----
    for mid in pc.get('native', []):
        kind, r, hp, scheme, dt, solver, _ = mid.split('|')
        r, dt = int(r), float(dt)
        Phi = Phi_all[:, :r]
        zz = np.load(tdir / 'artifacts' / f'HR~{kind}~{r}~{hp}.npz')
        if kind == 'ecsw':
            idx = H.stencil_index(zz['nodes'], 2, L)
            dat = dict(Pst=H.stencil_rows(Phi, idx), sw=jnp.sqrt(jnp.asarray(zz['weights'])), Phi=Phi)
        else:
            idx = H.stencil_index(zz['P'], 2, L)
            dat = dict(Pst=H.stencil_rows(Phi, idx), T1=jnp.asarray(zz['T1']), T2=jnp.asarray(zz['T2']),
                       T3E=jnp.asarray(zz['T3E']), Phi=Phi)
        q = RM.make_query(kind, 2, L, solver, trust=trust[r])
        certf = RM.make_certifier(kind, 2, L, H.NMAX[2])
        rows, Cs, Gs, Ws, AC, AG, RS_, LM_ = [], [], [], [], {}, {}, [], []
        for ci in ids:
            _, nu, n0 = U0[ci]
            sch = RM.schedule(2, scheme, dt, 'prod')
            o = q(U0i[ci], jnp.float64(nu), dat, sch)
            f = to_lat(o['fields'])
            gam = certf(o['C'], jnp.float64(nu), dat, sch)
            cr = RM.certify(o, gam, sch, 'prod')
            Cn = np.asarray(o['C'])[:int(sch['steps']) + 1]
            nC = np.linalg.norm(Cn, axis=1)
            kp = (len(Cn) - 1) // 5
            sched_ok = bool(np.array_equal(np.asarray(o['W']), Cn[[kp * j for j in range(6)]]))
            RS_.append(np.asarray(o['stats']['reason'])[:int(sch['steps'])])
            LM_.append(np.asarray(o['stats']['lam'])[:int(sch['steps'])])
            row = dict(member=mid, case=cid(ci), valid=bool(cr['valid'] and sched_ok), sched_ok=sched_ok, nonstat=cr['nonstat'],
                       steps=cr['steps'],
                       growth=float(np.max(nC / nC[0])) if np.isfinite(nC).all() else float('inf'), e0=e0(f, rec_of(ci)))
            row['stable'] = bool(row['growth'] <= 1.05)
            Cs.append(Cn)
            Gs.append(np.asarray(gam)[:int(sch['steps'])])
            Ws.append(np.asarray(o['W']))
            if ci in dev:
                tt = {}
                for lev in ('tight', 'tighter'):
                    sl = RM.schedule(2, scheme, dt, lev)
                    ol = q(U0i[ci], jnp.float64(nu), dat, sl)
                    gl = certf(ol['C'], jnp.float64(nu), dat, sl)
                    cl = RM.certify(ol, gl, sl, lev)
                    Cl = np.asarray(ol['C'])[:int(sl['steps']) + 1]
                    okl = bool(np.array_equal(np.asarray(ol['W']), Cl[[(len(Cl) - 1) // 5 * j for j in range(6)]]))
                    tt[lev] = (to_lat(ol['fields']), bool(cl['anchor_ok'] and okl))
                    AG[f'{lev}_reason_{cid(ci)}'] = np.asarray(ol['stats']['reason'])[:int(sl['steps'])]
                    AG[f'{lev}_lam_{cid(ci)}'] = np.asarray(ol['stats']['lam'])[:int(sl['steps'])]
                    AC[f'{lev}_C_{cid(ci)}'] = np.asarray(ol['C'])[:int(sl['steps']) + 1]
                    AG[f'{lev}_gamma_{cid(ci)}'] = np.asarray(gl)[:int(sl['steps'])]
                    AC[f'{lev}_W_{cid(ci)}'] = np.asarray(ol['W'])
                row.update(dist_tight=P.fdist(f, tt['tight'][0], n0), tight_anchor_ok=tt['tight'][1],
                           dist_tight_tighter=P.fdist(tt['tight'][0], tt['tighter'][0], n0), tighter_anchor_ok=tt['tighter'][1])
                _TIMING.setdefault(mid, dict(r=r, dt=dt, kind='native', q=q, dat=dat, sch=sch, U0i=U0i, U0=U0, ref={}))['ref'][ci] = \
                    dict(W=np.asarray(o['W']))
            rows.append(row)
        np.savez_compressed(out / f'N_{mid.replace("|", "_")}.npz', C=np.stack(Cs), gamma=np.stack(Gs), W=np.stack(Ws),
                            reason=np.stack(RS_), lam=np.stack(LM_),
                            case_ids=np.array([cid(i) for i in ids]), **{k.replace('|', '_'): v for k, v in {**AC, **AG}.items()})
        rec['native'][mid] = dict(r=r, rows=rows, labels=['anchors on dev6 only', 'hr-baselines registered pick, replayed'])
        print('NATIVE', mid, round(max(rw['e0']['e'] for rw in rows), 5), el(), flush=True)
    del Phi_all
    return rec


def timing_subjects():
    return _TIMING
