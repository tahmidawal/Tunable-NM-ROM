"""Manufactured tests of the fixed-sweep LMM solver fs3 (DESIGN A1.3, A2.2; gate G-FS2). CPU only:

    JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python experiments/jcp-combined/test_fs.py [--out checks/test_fs.json]

Problem, oracle and the adaptive LSPG comparator are C2's (test_lmm.py, inherited byte-identical): M=24, R'=6, quadratic
N(c) = sum T c c, so Ju(c) = (T + T^T) c is linear in c and N = 0.5 Ju c (the off-mesh structure fs3 relies on).
  (a) reported: fixed-sweep (K = 1, 2) and adaptive LSPG errors against the Radau oracle at h in {0.05/16, /32, /64}, slopes;
  (b) GATE: K = 2 fixed-sweep vs adaptive LSPG distance (same scheme, step, startup) <= 0.1 x the adaptive run's oracle
      error, at every tested step, for CN, CN-R, BDF2 (K = 1 reported). The adaptive comparator is t2core LSPG at the
      job's tight level gtol 1e-7 (C2 A5.1: 1e-12 is unattainable; on this system it ends steps on tiny-step exits and is
      unverified), required verified; an independent SciPy rollout of the exact intended LSPG steps is reported beside it
      and must agree with the adaptive comparator within 1e-3 x the adaptive run's oracle error, i.e. 100x tighter than the
      gate itself (code-time amendment A4.1);
  (c) GATE: mutations, each on its designated scheme (A2.2), must be detected: the mutated trajectory differs from the
      unmutated one by > 10 x the gate-(b) distance at the finest step, AND an independent SciPy least-squares solve of the
      INTENDED step equation (from the run's own history) differs from the run's step by > 10 x the unmutated run's
      discrepancy on the designated step;
  (d) reported: the previous-state predictor ('prev_pred') vs the quadratic predictor;
  (e) GATE (code audit 1 item 3): an independent NumPy re-implementation of the vendor fixed-sweep BE algorithm
      (offmesh.make_fsc_rule: quadratic predictor with extrapolated Ju, fused LM on the first a steps, then one damped,
      clipped LM sweep per step with damping carried) reproduces the fs3 BE K=1 trajectory (states <= 1e-9 relative,
      LM iterations and exit reasons equal); CONTROL: the same NumPy solver with the previous-state predictor must
      differ by > 1e-9 (the comparison can fail).
The script exits non-zero unless every gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.optimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_lmm as TL  # noqa: E402  (C2, inherited: problem, oracle, adaptive LSPG runner)
import fs3  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

A, LAM, TT, C0, NU, RP = TL.A, TL.LAM, TL.TT, TL.C0, TL.NU, TL.RP
DTS = TL.ORDER_DTS
DATA = dict(A=jnp.asarray(A), lam=jnp.asarray(LAM), T=jnp.asarray(TT))
GTOL = 1e-3          # deployed fixed-sweep gtol (only decides the reported reason in the fixed phase)
AFIRST = 3


def contract(d, w):
    T = d['T']
    return jnp.einsum('mij,j->mi', T + T.transpose(0, 2, 1), w)


_FS = {}
_AD = {}
_EX = {}


def ad_run(scheme, dt):
    """Adaptive LSPG (t2core.make_evolve) at gtol 1e-7, must be verified."""
    if (scheme, dt) not in _AD:
        import t2core as T2
        if 'ev' not in _AD:
            _AD['ev'] = jax.jit(T2.make_evolve(TL.nl, TL.nlJ, 'LSPG', RP, np.inf, budget=600))
        W, st = _AD['ev'](jnp.asarray(C0), NU, 1., TL.data('LSPG'), T2.sched(scheme, dt, 1e-7, 0.))
        st = {k: np.asarray(v) for k, v in st.items()}
        _AD[(scheme, dt)] = (np.asarray(W), int(st['nfail']))
    return _AD[(scheme, dt)]


def exact_run(scheme, dt):
    """Independent SciPy rollout: every step the exact least-squares solution of the intended LSPG step."""
    if (scheme, dt) not in _EX:
        Wall = [C0.copy()]
        for k in range(int(round(.25 / dt))):
            Wall.append(intended_step(scheme, k, np.array(Wall + [Wall[-1]]), dt))
        _EX[(scheme, dt)] = at_outputs(np.array(Wall), dt)
    return _EX[(scheme, dt)]


def fs_run(scheme, dt, K, mutation=None):
    key = (scheme, dt, K, mutation)
    if key not in _FS:
        steps = int(round(.25 / dt))
        a = fs3.a_effective(scheme, dt, dt, AFIRST, steps, mutation)
        ev = jax.jit(fs3.make_fs_evolve(RP, contract, scheme, dt, K, GTOL, np.inf, a, steps, mutation=mutation,
                                        unroll=False))
        o = ev(jnp.asarray(C0), NU, 1e-13, DATA)
        _FS[key] = {k: np.asarray(v) for k, v in o.items()}
    return _FS[key]


def at_outputs(W_all, dt):
    keep = int(round(.05 / dt))
    return W_all[::keep]


def err(W6, ref):
    return TL.err(W6, ref)


def dist(Wa, Wb):
    return max(np.linalg.norm(Wa[k] - Wb[k]) for k in range(1, 6)) / np.linalg.norm(C0)


def intended_step(scheme, k, Wall, dt):
    """SciPy least-squares solution of the INTENDED LSPG step k (BE start per the scheme table), history from Wall."""
    a0, a1, a2, b0, b1, ns = fs3.SCHEMES[scheme]
    if k < ns:
        a0, a1, a2, b0, b1 = 1., -1., 0., 1., 0.
    cn, cm = Wall[k], (Wall[k - 1] if k >= 1 else Wall[0])
    S = 1. / (a0 + dt * b0 * NU * LAM)

    def res(c):
        return S * (A @ (a0 * c + a1 * cn + a2 * cm) + dt * (b0 * TL.F_np(c) + b1 * TL.F_np(cn)))

    def jac(c):
        return S[:, None] * (a0 * A + dt * b0 * TL.dF_np(c))
    sol = scipy.optimize.least_squares(res, Wall[k + 1], jac=jac, xtol=1e-15, ftol=1e-15, gtol=1e-15, method='lm')
    assert sol.success, ('SciPy least squares failed', scheme, k, sol.message)
    return sol.x


def intended_gn(scheme, k, Wall, dt):
    """Independent stationarity ratio ||J^T r|| / (||J|| ||r||) of the INTENDED LSPG step k at the run's own state."""
    a0, a1, a2, b0, b1, ns = fs3.SCHEMES[scheme]
    if k < ns:
        a0, a1, a2, b0, b1 = 1., -1., 0., 1., 0.
    cn, cm, c = Wall[k], (Wall[k - 1] if k >= 1 else Wall[0]), Wall[k + 1]
    S = 1. / (a0 + dt * b0 * NU * LAM)
    r = S * (A @ (a0 * c + a1 * cn + a2 * cm) + dt * (b0 * TL.F_np(c) + b1 * TL.F_np(cn)))
    J = S[:, None] * (a0 * A + dt * b0 * TL.dF_np(c))
    return float(np.linalg.norm(J.T @ r) / (np.linalg.norm(J) * np.linalg.norm(r) + 1e-300))


def step_discrepancy(scheme, k, Wall, dt):
    return float(np.linalg.norm(Wall[k + 1] - intended_step(scheme, k, Wall, dt)) / np.linalg.norm(C0))


def chol_solve(H, g):
    import scipy.linalg
    c = scipy.linalg.cho_factor(H, lower=True)
    return scipy.linalg.cho_solve(c, g)


def np_fused_lm(evalJ, w0, tol, gtol, lam0, budget=50, trust=np.inf):
    """NumPy transcription of common.make_fused_lm (clip=True, Cholesky), including its exit semantics: an initial
    non-finite residual exits with reason 3 before any iteration; the tiny-step test uses the PRE-update state."""
    ratio = lambda g, J, rn: np.linalg.norm(g) / (np.linalg.norm(J) * rn + 1e-300)
    w = w0.copy()
    r, J = evalJ(w)
    rn = np.linalg.norm(r)
    g = J.T @ r
    reason = (4 if ratio(g, J, rn) <= gtol else (1 if rn <= tol else 0)) if np.isfinite(rn) else 3
    lam, it, rej = lam0, 0, 0
    while it < budget and reason == 0:
        Hm = J.T @ J
        d = np.diag(Hm) + 1e-30
        try:
            step = chol_solve(Hm + np.diag(lam * d), -g)
        except (np.linalg.LinAlgError, ValueError):       # non-PD or non-finite matrix: a non-finite step (rejected)
            step = np.full_like(g, np.nan)
        nz = np.linalg.norm(step)
        step = step * (trust / (nz + 1e-300) if nz > trust else 1.)
        ok = bool(np.all(np.isfinite(step)))
        wn = w + (step if ok else 0.)
        r2, J2 = evalJ(wn)
        rn2 = np.linalg.norm(r2)
        acc = ok and np.isfinite(rn2) and rn2 < rn
        tiny = ok and np.linalg.norm(step) <= 1e-14 * (1 + np.linalg.norm(w))       # pre-update w, as the vendor
        if acc:
            w, r, J, rn = wn, r2, J2, rn2
        g = J.T @ r
        gn = ratio(g, J, rn)
        reason = 4 if gn <= gtol else (1 if rn <= tol else (2 if tiny else (3 if (not acc) and lam >= 1e14 else 0)))
        lam = max(lam / 3, 1e-12) if acc else min(lam * 10, 1e14)
        it += 1
        rej += int(not acc)
    return w, rn, it, reason, lam


def np_vendor_be(dt, a=AFIRST, prev_pred=False, trust=np.inf):
    """Independent NumPy re-implementation of offmesh.make_fsc_rule (BE) on the manufactured system."""
    S = 1. / (1. + dt * NU * LAM)
    Juf = lambda c: np.einsum('mij,j->mi', TT + TT.transpose(0, 2, 1), c)
    rj = lambda Ju, c, p: ((A @ c - p + dt * (0.5 * Ju @ c + NU * LAM * (A @ c))) * S, A + (dt * S)[:, None] * Ju)
    steps = int(round(.25 / dt))
    wv = wp = wp2 = C0.copy()
    Jv = Jp = Jp2 = Juf(C0)
    lam = 1e-6
    Wall, its, reasons = [C0.copy()], [], []
    for k in range(steps):
        p = A @ wv
        if prev_pred:
            cand, Ju3 = [wv], [Jv]
        else:
            we, Je = 2 * wv - wp, 2 * Jv - Jp
            wq, Jq = (3 * wv - 3 * wp + wp2, 3 * Jv - 3 * Jp + Jp2) if k >= 2 else (we, Je)
            cand, Ju3 = [wv, we, wq], [Jv, Je, Jq]
        rs = [np.linalg.norm(rj(J_, c_, p)[0]) for J_, c_ in zip(Ju3, cand)]
        i = int(np.argmin([x if np.isfinite(x) else np.inf for x in rs]))
        w, Ju = cand[i], Ju3[i]
        if k < a:
            w2, rn, it, reason, lam = np_fused_lm(lambda c: rj(Juf(c), c, p), w, 1e-13, GTOL, lam, trust=trust)
            Ju2 = Juf(w2)
        else:
            r, J = rj(Ju, w, p)
            rn = np.linalg.norm(r)
            Hm = J.T @ J
            g = J.T @ r
            d = np.diag(Hm) + 1e-30
            try:
                dw = chol_solve(Hm + np.diag(lam * d), -g)
            except (np.linalg.LinAlgError, ValueError):
                dw = np.full_like(g, np.nan)
            nz = np.linalg.norm(dw)
            dw = dw * (trust / (nz + 1e-300) if nz > trust else 1.)
            okf = bool(np.all(np.isfinite(dw)))
            wn = w + (dw if okf else 0.)
            Jun = Juf(wn)
            r2, J2 = rj(Jun, wn, p)
            rn2 = np.linalg.norm(r2)
            acc = okf and np.isfinite(rn2) and rn2 < rn
            lam = max(lam / 3, 1e-12) if acc else min(lam * 10, 1e14)
            w2, Ju2, r, J, rn = (wn, Jun, r2, J2, rn2) if acc else (w, Ju, r, J, rn)
            gn = np.linalg.norm(J.T @ r) / (np.linalg.norm(J) * rn + 1e-300)
            it, reason = 1, ((4 if gn <= GTOL else 0) if np.isfinite(rn) else 3)
        wv, wp, wp2 = w2, wv, wp
        Jv, Jp, Jp2 = Ju2, Jv, Jp
        Wall.append(w2)
        its.append(int(it))
        reasons.append(int(reason))
    return np.array(Wall), its, reasons


TRUST_FIX = 0.02 * float(np.linalg.norm(C0))      # finite-trust fixture: the step clip is active in early steps


def fs_run_trust(dt):
    key = ('BE-trust', dt)
    if key not in _FS:
        steps = int(round(.25 / dt))
        ev = jax.jit(fs3.make_fs_evolve(RP, contract, 'BE', dt, 1, GTOL, TRUST_FIX, AFIRST, steps, unroll=False))
        _FS[key] = {k: np.asarray(v) for k, v in ev(jnp.asarray(C0), NU, 1e-13, DATA).items()}
    return _FS[key]


def vendor_be_trajectory_check(dt, trust=False):
    o = fs_run_trust(dt) if trust else fs_run('BE', dt, 1)
    tr = TRUST_FIX if trust else np.inf
    Wn, its, reasons = np_vendor_be(dt, trust=tr)
    Wc, _, _ = np_vendor_be(dt, prev_pred=True, trust=tr)
    nrm = np.linalg.norm(C0)
    d = float(np.max(np.linalg.norm(o['W_all'] - Wn, axis=1)) / nrm)
    dc = float(np.max(np.linalg.norm(o['W_all'] - Wc, axis=1)) / nrm)
    return dict(state_distance=d, iterations_equal=its == o['it'].tolist(), reasons_equal=reasons == o['reason'].tolist(),
                control_distance=dc, passed=bool(d <= 1e-9 and its == o['it'].tolist() and reasons == o['reason'].tolist()
                                                and dc > 1e-9))


def nonfinite_jacobian_fixture():
    """Finite residual, NaN Jacobian: the vendor LM (common.make_fused_lm) and the NumPy transcription must agree on
    iterations and exit reason (code audit 3 item 3)."""
    from offmesh import C as Cv
    r0 = np.array([1., 2., 3.])
    evj = lambda w, _a=None: (jnp.asarray(r0) + 0. * jnp.sum(w), jnp.full((3, 2), jnp.nan))
    lm = Cv.make_fused_lm(evj, 2, 50, jnp.inf, GTOL, clip=True)
    w, rn, it, reason, gn, rej, lam = lm(jnp.zeros(2), None, 0., 1e-6)
    _, _, it_n, reason_n, _ = np_fused_lm(lambda w_: (r0.copy(), np.full((3, 2), np.nan)), np.zeros(2), 0., GTOL, 1e-6)
    return dict(vendor=(int(it), int(reason)), numpy=(int(it_n), int(reason_n)), passed=(int(it), int(reason)) == (int(it_n), int(reason_n)))


def vendor_be_check(dt):
    """(e) recompute the BE fixed-sweep trajectory with the vendor residual expression written out here independently."""
    o = fs_run('BE', dt, 1)
    Wall = o['W_all']
    S = 1. / (1. + dt * NU * LAM)
    worst = 0.
    for k in range(AFIRST, len(Wall) - 1):
        p = A @ Wall[k]
        # one vendor sweep from the run's own chosen predictor is not reconstructible independently of the predictor;
        # instead check the vendor residual identity on the returned state: r = (Ac - p + dt(0.5 Ju c + nu lam Ac)) S
        c = Wall[k + 1]
        Ju = np.einsum('mij,j->mi', TT + TT.transpose(0, 2, 1), c)
        r_v = (A @ c - p + dt * (0.5 * Ju @ c + NU * LAM * (A @ c))) * S
        r_g = S * (A @ (c - Wall[k]) + dt * TL.F_np(c))
        worst = max(worst, float(np.linalg.norm(r_v - r_g) / max(np.linalg.norm(r_g), 1e-300)))
    return worst, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'checks' / 'test_fs.json'))
    a = ap.parse_args()
    ref = TL.oracle()
    rep = dict(problem=dict(seed=TL.SEED, M=TL.M, Rp=RP, nu=NU), dts=DTS, a_first=AFIRST, schemes={}, mutations={},
               gates={})
    ok = True
    gate_b = {}
    for sc in ('CN', 'CNR', 'BDF2'):
        ent = dict(adaptive_err=[], fs1_err=[], fs2_err=[], fs1_vs_ad=[], fs2_vs_ad=[], prev_pred_err=[],
                   adaptive_nfail=[], adaptive_vs_exact=[], fs2_vs_exact=[])
        for dt in DTS:
            Wad, nf = ad_run(sc, dt)
            We = exact_run(sc, dt)
            ent['adaptive_nfail'].append(nf)
            ent['adaptive_vs_exact'].append(dist(Wad, We))
            e_ad = err(Wad, ref)
            W1 = at_outputs(fs_run(sc, dt, 1)['W_all'], dt)
            W2 = at_outputs(fs_run(sc, dt, 2)['W_all'], dt)
            Wpp = at_outputs(fs_run(sc, dt, 1, 'prev_pred')['W_all'], dt)
            ent['adaptive_err'].append(e_ad)
            ent['fs1_err'].append(err(W1, ref))
            ent['fs2_err'].append(err(W2, ref))
            ent['fs1_vs_ad'].append(dist(W1, Wad))
            ent['fs2_vs_ad'].append(dist(W2, Wad))
            ent['fs2_vs_exact'].append(dist(W2, We))
            ent['prev_pred_err'].append(err(Wpp, ref))
        for k_ in ('adaptive_err', 'fs1_err', 'fs2_err'):
            e = ent[k_]
            ent[k_ + '_slopes'] = [float(np.log2(e[i] / e[i + 1])) for i in range(len(e) - 1)]
        ent['gate_b_pass'] = bool(all(d <= .1 * e for d, e in zip(ent['fs2_vs_ad'], ent['adaptive_err'])) and
                                  all(n == 0 for n in ent['adaptive_nfail']) and
                                  all(x <= 1e-3 * e for x, e in zip(ent['adaptive_vs_exact'], ent['adaptive_err'])))
        ok &= ent['gate_b_pass']
        gate_b[sc] = ent['fs2_vs_ad'][-1]
        rep['schemes'][sc] = ent
        print(sc, 'adaptive', ent['adaptive_err'], 'fs2-vs-ad', ent['fs2_vs_ad'], 'gate', ent['gate_b_pass'], flush=True)
    dtf = DTS[-1]
    designated = dict(stale_fn=('CN', 2), stale_hist=('BDF2', 4), no_startup=('BDF2', 0), wrong_weight=('BDF2', 4))
    for mut, (sc, kstep) in designated.items():
        Wm_all = fs_run(sc, dtf, 2, mut)['W_all']
        Wu_all = fs_run(sc, dtf, 2)['W_all']
        dtraj = dist(at_outputs(Wm_all, dtf), at_outputs(Wu_all, dtf))
        ks = max(kstep, AFIRST)            # first fixed-sweep step at or after the designated one
        sd_m = step_discrepancy(sc, ks, Wm_all, dtf)
        sd_u = step_discrepancy(sc, ks, Wu_all, dtf)
        if mut == 'no_startup':            # the startup step is adaptive: check the intended equation at step 0
            sd_m, sd_u = step_discrepancy(sc, 0, Wm_all, dtf), step_discrepancy(sc, 0, Wu_all, dtf)
        kk = 0 if mut == 'no_startup' else ks
        gm, gu = intended_gn(sc, kk, Wm_all, dtf), intended_gn(sc, kk, Wu_all, dtf)
        det = bool(dtraj > 10 * gate_b[sc] and sd_m > 10 * max(sd_u, 1e-16) and gm > 10 * max(gu, 1e-16))
        rep['mutations'][mut] = dict(scheme=sc, step=ks if mut != 'no_startup' else 0, trajectory_distance=dtraj,
                                     bar=10 * gate_b[sc], step_discrepancy_mutated=sd_m, step_discrepancy_unmutated=sd_u,
                                     intended_gn_mutated=gm, intended_gn_unmutated=gu, detected=det)
        ok &= det
        print('MUT', mut, rep['mutations'][mut], flush=True)
    vb, _ = vendor_be_check(dtf)
    rep['gates']['vendor_be_residual_identity'] = vb          # reported only (algebraic identity on one trajectory)
    vt = {f'{dt:g}': vendor_be_trajectory_check(dt) for dt in DTS}
    vt.update({f'{dt:g}|trust': vendor_be_trajectory_check(dt, trust=True) for dt in DTS})
    rep['gates']['trust_fixture_clip_active'] = bool(any(np.linalg.norm(np.diff(fs_run_trust(dt)['W_all'], axis=0), axis=1).max()
                                                         >= .999 * TRUST_FIX for dt in DTS))
    ok &= rep['gates']['trust_fixture_clip_active']
    rep['gates']['nonfinite_jacobian'] = nonfinite_jacobian_fixture()
    ok &= rep['gates']['nonfinite_jacobian']['passed']
    print('NONFINITE', rep['gates']['nonfinite_jacobian'], flush=True)
    rep['gates']['vendor_be_trajectory'] = vt
    ok &= all(v['passed'] for v in vt.values())
    print('VENDOR-BE', vt, flush=True)
    rep['gates']['all_pass'] = bool(ok)
    rep['all_pass'] = bool(ok)
    rep['source_sha256'] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                            for f in ('fs3.py', 'test_fs.py', 't2core.py', 'test_lmm.py', 'vendor/quad3d/offmesh.py',
                                      'vendor/quad3d/vendor/burgers3d-span/common.py',
                                      'vendor/quad3d/vendor/burgers3d-retry/tables.py')}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print('ALL_PASS', ok)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
