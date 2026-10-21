"""imp-faststep POD block (DESIGN 3.5, A1.8, A2.2, A2.6, A3.1, A3.4): POD-Galerkin + Galerkin-ECSW with the same ETDRK4
step, the hyper-reduction-free POD-Galerkin ETDRK4, and the hr-baselines registered picks replayed natively (5L).

POD-Galerkin ODE of the central FOM (hrcore operator): c' = -Phi^T N_h(Phi c) - nu K c, K = Phi^T(-Lap_h)Phi = W D_P W^T,
y = W^T c. Advection providers (precontracted, used by the stepper) and their un-precontracted twins (certificates):

  ecsw   N_y(y) = Wt^T [N_E(PstW y)],  PstW = Pst W, Wt = diag(xi) Phi_E W      | twin: W^T Pst0^T(xi N_E(Pst W y))
  none   N_y(y) = PhiW^T N_h(PhiW y),   PhiW = Phi W                            | twin: W^T Phi^T N_h(Phi W y)

N_E / N_h: the central advection u_i (L/2) sum_a (u_{+a} - u_{-a}) (hrcore.stencil_F / adv_central). The stepper is
fscore.make_evolve('etd', ...) with tab D = D_P, VtRm = W^T, RinvV = W. Every large array is a jit argument.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import fscore as F  # noqa: E402
import p1common as P  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

HRB = HERE.parent / 'jcp-hr-baselines'
if str(HRB) not in sys.path:
    sys.path.append(str(HRB))          # appended: never shadows the p1-scaling modules of the same name
import hrcore as H  # noqa: E402
import rom as RM  # noqa: E402

KAPPAS = (2, 4, 8, 16)
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


# ------------------------------------------------------------------ advection providers ----

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


# ------------------------------------------------------------------ the block ----

def run(cfg, L, cases, ids, dev, U0, out, e0, el):
    pc = cfg['pod']
    tdir = Path(pc['tune_dir'])
    rec = dict(tune_dir=str(tdir), members={}, native={}, ranks={}, gecsw={}, labels=['snapshot-limited basis (hr-baselines A1.3)',
               'tuning-budget band unresolved (hr-baselines 2D)', 'developmental, in-sample, unadjusted', 'no H-base conclusion at 4096^2'])
    t0 = time.perf_counter()
    rec['phi_sha256'] = sha_file(tdir / 'artifacts' / 'Phi.npz')
    zf = np.load(tdir / 'artifacts' / 'Phi.npz')
    rmax = max(pc['ranks'])
    Phi_all = jnp.asarray(np.asarray(zf['Phi'])[:, :rmax])
    trust = {int(r): float(t) for r, t in zip(zf['ranks'], zf['trust'])}
    del zf
    rec['load_seconds'] = time.perf_counter() - t0
    lat = jnp.asarray(H.lattice_nodes(2, L))
    sh = (L - 1, L - 1)
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
        rec['ranks'][str(r)] = dict(D_min=float(D[0]), D_max=float(D[-1]), orth=float(jnp.max(jnp.abs(Phi.T @ Phi - jnp.eye(r)))),
                                    eig_resid=float(jnp.linalg.norm(K @ Wm - Wm * D[None]) / jnp.linalg.norm(D)), trust=trust.get(r))
        assert float(D[0]) > 0
        z = np.load(HERE / pc['gecsw'].format(r=r))
        rules = []                                      # (kappa, nodes, weights) available, in kappa order
        for k in pc['kappas']:
            if f'k{k}_nodes' in z.files:
                rules.append((k, np.asarray(z[f'k{k}_nodes']), np.asarray(z[f'k{k}_weights'])))
        rec['gecsw'][str(r)] = dict(file=pc['gecsw'].format(r=r), sha256=sha_file(HERE / pc['gecsw'].format(r=r)),
                                    available=[k for k, _, _ in rules])
        members = [('ecsw', k, nd, wt) for k, nd, wt in rules] + [('none', None, None, None)]
        TB = {}
        for kind, k, nd, wt in members:
            if kind == 'ecsw':
                idx = H.stencil_index(nd, 2, L)
                Pst = H.stencil_rows(Phi, idx)
                xi = jnp.asarray(wt)
                tb = dict(PstW=jnp.einsum('esr,rq->esq', Pst, Wm), Wt=(xi[:, None] * Pst[:, 0, :]) @ Wm, Pst=Pst, xi=xi, W=Wm,
                          D=D, VtRm=Wm.T, RinvV=Wm)
                hid = rule_id(nd, wt)
            else:
                tb = dict(PhiW=Phi @ Wm, Phi=Phi, W=Wm, D=D, VtRm=Wm.T, RinvV=Wm)
                hid = 'none'
            TB[(kind, k)] = (tb, hid)
        # successor of each member: the next available DISTINCT rule, else none (A3.1)
        succ = {}
        for a_, (kind, k, _, _) in enumerate(members):
            if kind == 'none':
                continue
            h0 = TB[(kind, k)][1]
            nxt = ('none', None)
            for kind2, k2, _, _ in members[a_ + 1:]:
                if kind2 == 'none' or TB[(kind2, k2)][1] != h0:
                    nxt = (kind2, k2)
                    break
            succ[(kind, k)] = nxt
        fc = {kd: force_c(kd, L) for kd in ('ecsw', 'none')}
        for dt in pc['dts']:
            for kind, k, _, _ in members:
                tb, hid = TB[(kind, k)]
                mid = f'pod-{kind}|{r}|{"k%d" % k if k else "-"}|ETDRK4|{dt}'
                adv, twin = make_adv(kind, L)
                key = (kind, r)
                if key not in EVC:
                    EVC[key] = (jax.jit(F.make_evolve('etd', adv)), jax.jit(F.make_evolve('etd', twin)),
                                F.make_cert_etd_adv(twin))
                ev, evt, cert = EVC[key]
                rows, W_all, Z_all = [], [], []
                for ci in ids:
                    coh, c, _ = cases[ci]
                    _, nu, n0 = U0[ci]
                    c0 = init(U0i[ci], Phi)
                    W, s = ev(c0, jnp.float64(nu), tb, F.sched(dt))
                    f = to_lat(decode(W, Phi))
                    k_ = int(s['steps'])
                    Z = np.asarray(s['Z'])[:k_ + 1]
                    co = {a: jnp.asarray(b) for a, b in F.etd_coefs_indep(-nu * np.asarray(D), dt).items()}
                    mis = np.asarray(cert(jnp.asarray(Z), tb, co, k_))
                    row = dict(member=mid, case=cid(ci), valid=bool(np.isfinite(np.asarray(W)).all() and np.isfinite(Z).all()
                                                                   and np.isfinite(float(s['stage_max']))),
                               growth=float(s['growth']), stage_max=float(s['stage_max']), cert_worst=float(np.max(mis)),
                               cert_ok=bool(np.all(mis <= 1e-10)), e0=e0(f, coh, c), rule=hid)
                    row['stable'] = bool(row['growth'] <= 1.05 and np.isfinite(row['stage_max']))
                    if ci in dev:
                        Wt_, st_ = evt(c0, jnp.float64(nu), tb, F.sched(dt))
                        row['twin_dist'] = P.fdist(f, to_lat(decode(Wt_, Phi)), n0)
                    rows.append(row)
                    W_all.append(np.asarray(W))
                    Z_all.append(Z)
                    if ci in dev:
                        _TIMING.setdefault(mid, dict(r=r, dt=dt, kind=kind, ev=ev, tb=tb, Phi=Phi, init=init, decode=decode,
                                                     U0i=U0i, U0=U0, ref={}))['ref'][ci] = np.asarray(W)
                rec['members'][mid] = dict(r=r, kappa=k, dt=dt, kind=kind, rule=hid, rows=rows,
                                           successor=None if kind == 'none' else
                                           f'pod-{succ[(kind, k)][0]}|{r}|{"k%d" % succ[(kind, k)][1] if succ[(kind, k)][1] else "-"}|ETDRK4|{dt}',
                                           size=None if kind == 'none' else int(len(TB[(kind, k)][0]['xi'])))
                np.savez_compressed(out / f'P_{mid.replace("|", "_")}.npz', W=np.stack(W_all), Z=np.stack(Z_all),
                                    case_ids=np.array([cid(i) for i in ids]))
                print('POD', mid, round(max(rw['e0']['e'] for rw in rows), 5), el(), flush=True)
            # Q2-POD on own states (A2.2/A3.1): rollout distance to the successor and the force ratio
            for kind, k, _, _ in members:
                if kind == 'none':
                    continue
                mid = f'pod-{kind}|{r}|k{k}|ETDRK4|{dt}'
                sid = rec['members'][mid]['successor']
                Wm_ = np.load(out / f'P_{mid.replace("|", "_")}.npz')['W']
                Ws_ = np.load(out / f'P_{sid.replace("|", "_")}.npz')['W']
                tbm = TB[(kind, k)][0]
                skd, sk = succ[(kind, k)]
                tbs = TB[(skd, sk)][0]
                for j, ci in enumerate(ids):
                    n0 = U0[ci][2]
                    fm = to_lat(decode(jnp.asarray(Wm_[j]), Phi))
                    fs = to_lat(decode(jnp.asarray(Ws_[j]), Phi))
                    rho = [P.rho(np.asarray(fc[kind](jnp.asarray(Wm_[j, q]), tbm)), np.asarray(fc[skd](jnp.asarray(Wm_[j, q]), tbs)))
                           for q in range(6)]
                    rec['members'][mid]['rows'][j].update(succ_dist=P.fdist(fm, fs, n0), succ_rho=rho)
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
        rows = []
        for ci in ids:
            coh, c, _ = cases[ci]
            _, nu, n0 = U0[ci]
            sch = RM.schedule(2, scheme, dt, 'prod')
            o = q(U0i[ci], jnp.float64(nu), dat, sch)
            f = to_lat(o['fields'])
            gam = certf(o['C'], jnp.float64(nu), dat, sch)
            cr = RM.certify(o, gam, sch, 'prod')
            Cn = np.asarray(o['C'])[:int(sch['steps']) + 1]
            nC = np.linalg.norm(Cn, axis=1)
            row = dict(member=mid, case=cid(ci), valid=cr['valid'], nonstat=cr['nonstat'], steps=cr['steps'],
                       growth=float(np.max(nC / nC[0])) if np.isfinite(nC).all() else float('inf'), e0=e0(f, coh, c),
                       artifact_sha256=sha_file(tdir / 'artifacts' / f'HR~{kind}~{r}~{hp}.npz'))
            row['stable'] = bool(row['growth'] <= 1.05)
            if ci in dev:
                ot = q(U0i[ci], jnp.float64(nu), dat, RM.schedule(2, scheme, dt, 'tight'))
                row['dist_tight'] = P.fdist(f, to_lat(ot['fields']), n0)
                _TIMING.setdefault(mid, dict(r=r, dt=dt, kind='native', q=q, dat=dat, sch=sch, U0i=U0i, U0=U0, ref={}))['ref'][ci] = \
                    np.asarray(o['W'])
            rows.append(row)
        rec['native'][mid] = dict(r=r, rows=rows, labels=['anchors on dev6 only', 'hr-baselines registered pick, replayed'])
        print('NATIVE', mid, round(max(rw['e0']['e'] for rw in rows), 5), el(), flush=True)
    del Phi_all
    return rec


def timing_subjects():
    return _TIMING
