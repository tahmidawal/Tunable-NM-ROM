"""jcp-hr-baselines offline construction (DESIGN.md 3.3-3.5; A1.2, A1.3, A1.8, A1.13, A1.15):
snapshots of the central FOM on the training draws, POD by the method of snapshots, per-rank trust radii, advection
POD and Q-DEIM points (pivoted QR on the device), the Householder R factor of B = [Phi, Lap Phi, U_p], ECSW training
states, contribution matrices and NNLS weights, and the per-configuration online data.

Large arrays: snapshots live on the host (f64) and are streamed to the device in column blocks; every phase frees its
device arrays. Timers record GPU-seconds per phase (offline cost, DESIGN 11).
"""
from __future__ import annotations

import math
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import hrcore as H
import nnls as NN

CAND = 32768
ECSW_CAP, ECSW_CAP_RETRY = 8192, 16384
ECSW_TAUS = (1e-2, 1e-3, 1e-4)                 # recorded along the path (reported), not the knob (A15.1)
ECSW_KAPPA = (2, 4, 8)                          # the ECSW knob: support targets s = kappa * r (A15.1)
ECSW_KAPPA_PATH = (2, 4, 8, 16)                 # every path is built through the extension target 16r (A16.2)
ECSW_SEED_TRAJ, ECSW_SEED_CAND = 20261017, 20261018
N_ECSW_TRAJ = 24


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        tm = self

        class _C:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                tm.t[name] = tm.t.get(name, 0.) + time.perf_counter() - self.t0
        return _C()


# ---------------------------------------------------------------------------------------------- snapshots ---

def snapshots(d, L, fam, rows, save_steps=H.SNAP_STEPS, scheme=H.SNAP_SCHEME, dt=H.SNAP_DT, ntol=1e-10, ltol=1e-12,
              log=print, adv='central'):
    """Host array (len(rows) * len(save_steps), n) of FOM states (central operator; 'upwind' = the banks' generator,
    used with scheme 'BE' by the generator probe, A2.3); asserts every step converged."""
    n = (L - 1) ** d
    fom = H.make_fom(d, L, adv=adv)
    sch = H.sched(scheme, dt, save_steps=save_steps)
    S = np.empty((len(rows) * len(save_steps), n))
    worst, t0 = 0.0, time.perf_counter()
    for i, j in enumerate(rows):
        o = fom(fam.ic(L, j), fam.nu(j), sch, ntol, ltol)
        res = np.asarray(o['res'])
        assert np.all(np.isfinite(res)) and float(np.max(res)) <= ntol * (1 + 1e-9), (j, float(np.nanmax(res)))
        worst = max(worst, float(np.max(res)))
        S[i * len(save_steps):(i + 1) * len(save_steps)] = np.asarray(o['out'])
        if log and (i % 64 == 0 or i == len(rows) - 1):
            log(f'   snapshots {i + 1}/{len(rows)} worst rel residual {worst:.2e} [{time.perf_counter() - t0:.0f}s]')
    return S, dict(worst_rel_residual=worst, rows=len(rows), steps=list(save_steps), scheme=scheme, dt=dt, adv=adv)


def _blocks(n, nb):
    return [(i, min(i + nb, n)) for i in range(0, n, nb)]


def gram(S, nb=1 << 16):
    """G = S S^T in f64, S on the host, column blocks streamed to the device."""
    m = S.shape[0]
    G = jnp.zeros((m, m))
    f = jax.jit(lambda G, X: G + X @ X.T)
    for a, b in _blocks(S.shape[1], nb):
        G = f(G, jnp.asarray(S[:, a:b]))
    return G


def pod(S, rmax, nb=1 << 16):
    """Method of snapshots: Phi (n, rmax) on the device, singular values, orthonormality deviation."""
    G = gram(S, nb)
    lam, V = jnp.linalg.eigh(G)
    lam, V = lam[::-1], V[:, ::-1]
    sig = jnp.sqrt(jnp.maximum(lam, 0.))
    # A14.2: beyond the numerical rank sigma ~ 0; a floored divisor keeps W finite (those columns become completion
    # directions re-orthonormalised by the QR below; numerical rank is diagnostic only, and a prefix is admitted only
    # through the A6.1 certificate -- which may pass a saturated prefix)
    W = V[:, :rmax] / jnp.maximum(sig[:rmax], 1e-14 * sig[0])
    del G, V
    blocks = [jnp.asarray(S[:, a:b]).T @ W for a, b in _blocks(S.shape[1], nb)]
    Phi = jnp.concatenate(blocks, 0)
    del blocks
    for _ in range(2):
        Phi, _r = jnp.linalg.qr(Phi, mode='reduced')
    # fix signs so that the basis is deterministic (largest-|.| entry of each column positive)
    am = jnp.concatenate([jnp.argmax(jnp.abs(Phi[:, a:b]), axis=0) for a, b in _blocks(rmax, 128)])   # column chunks
    sgn = jnp.sign(Phi[am, jnp.arange(rmax)])
    Phi = Phi * jnp.where(sgn == 0, 1., sgn)[None, :]
    dev = float(jnp.max(jnp.abs(Phi.T @ Phi - jnp.eye(rmax))))
    return jax.block_until_ready(Phi), np.asarray(sig), dev


def pod_rsvd(S, rmax, oversample=64, power=2, nb=1 << 16, seed=0):
    """A4.7 fallback: blocked randomised SVD of S^T (n, n_s) with power iterations and re-orthonormalisation.
    Returns (Phi (n, rmax), singular values (rmax + oversample,), orthonormality deviation)."""
    m, n = S.shape
    k = min(rmax + oversample, m, n)                  # A6.2: l = min(q_max + 64, n_snapshots, n_spatial)
    assert k >= rmax, f'requested rank {rmax} exceeds the snapshot span ({m} snapshots): not constructible'
    Om = jnp.asarray(np.random.default_rng(seed).standard_normal((m, k)))

    def StX(X):                         # S^T X  -> (n, k), assembled in row blocks of S^T
        return jnp.concatenate([jnp.asarray(S[:, a:b]).T @ X for a, b in _blocks(n, nb)], 0)

    def SY(Y):                          # S Y -> (m, k)
        out = jnp.zeros((m, Y.shape[1]))
        for a, b in _blocks(n, nb):
            out = out + jnp.asarray(S[:, a:b]) @ Y[a:b]
        return out
    Y = StX(Om)
    Qy, _ = jnp.linalg.qr(Y, mode='reduced')
    del Y, Om                                       # code audit 9: release the sketch before the power iterations
    for _ in range(power):
        Z, _ = jnp.linalg.qr(SY(Qy), mode='reduced')
        Qy, _ = jnp.linalg.qr(StX(Z), mode='reduced')
    Bm = SY(Qy).T                       # (k, m) = Q^T S^T
    Ub, sb, _ = jnp.linalg.svd(Bm, full_matrices=False)
    Phi = Qy @ Ub[:, :rmax]
    del Qy, Bm
    Phi, _ = jnp.linalg.qr(Phi, mode='reduced')
    dev = float(jnp.max(jnp.abs(Phi.T @ Phi - jnp.eye(rmax))))
    return jax.block_until_ready(Phi), np.asarray(sb), dev


def project_codes(S, Phi, nb=1 << 16):
    """C = S Phi (n_s, r) with S on the host."""
    C = jnp.zeros((S.shape[0], Phi.shape[1]))
    f = jax.jit(lambda C, X, P: C + X @ P)
    for a, b in _blocks(S.shape[1], nb):
        C = f(C, jnp.asarray(S[:, a:b]), Phi[a:b])
    return C


def projection_errors(S, Phi, ranks, sig, nb=1 << 16):
    """Per rank: RMS relative projection error of the snapshot rows, the relative Frobenius error, and the optimal
    (POD-tail) value sqrt((||S||_F^2 - sum_{i<=r} sig_i^2) / ||S||_F^2) for the G-pod optimality gate (A3.3)."""
    C = np.asarray(project_codes(S, Phi, nb))
    n2 = np.zeros(S.shape[0])
    for a, b in _blocks(S.shape[1], nb):
        n2 += np.sum(S[:, a:b] ** 2, 1)
    tot = float(np.sum(n2))                       # ||S||_F^2 directly (exact; independent of how many sigma are known)
    out = {}
    for r in ranks:
        e2 = np.maximum(n2 - np.sum(C[:, :r] ** 2, 1), 0.)
        out[r] = dict(rms_rel=float(np.sqrt(np.mean(e2 / np.maximum(n2, 1e-300)))),
                      frob_rel=float(np.sqrt(np.sum(e2) / np.sum(n2))),
                      pod_tail=float(np.sqrt(max(tot - float(np.sum(sig[:r] ** 2)), 0.) / tot)),
                      pod_tail_sq=float((tot - float(np.sum(sig[:r] ** 2))) / tot))      # unclipped radicand (A6.1)
    return out, C


def trust_radii(C, ranks, d):
    kind, frac = H.TRUST[d]
    out = {}
    for r in ranks:
        X = C[:, :r] - C[:, :r].mean(0)
        nrm = np.linalg.norm(X, axis=1)
        out[r] = float(frac * (np.max(nrm) if kind == 'max' else np.sqrt(np.mean(nrm ** 2))))
    return out


def advection_in_place(S, d, L, nb_rows=64):
    """Overwrite each snapshot row u by N_h(u) (central advection), on the device in row batches."""
    sh = H.shape_of(d, L)
    f = jax.jit(jax.vmap(lambda u: H.adv_central(u.reshape(sh), L).reshape(-1)))
    for a in range(0, S.shape[0], nb_rows):
        S[a:a + nb_rows] = np.asarray(f(jnp.asarray(S[a:a + nb_rows])))
    return S


# ------------------------------------------------------------------------------------------------ Q-DEIM ----

def _pivots(p):
    def run(R):
        n = R.shape[1]

        def body(j, c):
            R, piv, taken = c
            nrm = jnp.where(taken, -1., jnp.sum(R * R, 0))
            i = jnp.argmax(nrm)
            q = R[:, i] / jnp.sqrt(nrm[i])
            R = R - q[:, None] * (q @ R)[None, :]
            return R, piv.at[j].set(i), taken.at[i].set(True)
        _, piv, _ = jax.lax.fori_loop(0, p, body, (R, jnp.zeros(p, jnp.int32), jnp.zeros(n, bool)))
        return piv
    return jax.jit(run)


_PIV = {}


def qdeim_points(U):
    """Q-DEIM point selection (Drmac-Gugercin): greedy column pivoting of U^T (U (n, p) on the device) by the largest
    remaining column norm after orthogonal projection against the already selected columns (modified Gram-Schmidt with
    Businger-Golub pivoting). This is the pivot rule of LAPACK geqp3 but not its Householder implementation: equality of
    the pivot sequence with scipy/LAPACK is checked on the registered small fixtures (G-qdeim) only, not claimed in
    general (A12.3)."""
    p = U.shape[1]
    if p not in _PIV:
        _PIV[p] = _pivots(p)
    return np.asarray(_PIV[p](U.T))


def qdeim_data(Phi_r, LPhi_r, U_p, P, d, L):
    """Online Q-DEIM data (A1.2): Pst (p, 2d+1, r), T1, T2, T3E; plus diagnostics."""
    r, p = Phi_r.shape[1], U_p.shape[1]
    B = jnp.concatenate([Phi_r, LPhi_r, U_p], 1)
    Rb = jnp.linalg.qr(B, mode='r')
    del B
    M = U_p[jnp.asarray(P)]                                   # P^T U_p (p, p)
    sv = jnp.linalg.svd(M, compute_uv=False)
    T3E = jnp.linalg.solve(M.T, Rb[:, 2 * r:].T).T            # R3 M^{-1}
    bwd = float(jnp.linalg.norm(T3E @ M - Rb[:, 2 * r:]) / (jnp.linalg.norm(T3E) * jnp.linalg.norm(M) + 1e-300))
    idx = H.stencil_index(P, d, L)
    data = dict(Pst=H.stencil_rows(Phi_r, idx), T1=Rb[:, :r], T2=Rb[:, r:2 * r], T3E=T3E)
    info = dict(p=int(p), interp_const=float(1. / sv[-1]), cond_PtU=float(sv[0] / sv[-1]), solve_backward_error=bwd,
                solve_ok=bool(bwd <= 1e-12),
                rank_B_proxy=int(jnp.sum(jnp.abs(jnp.diag(Rb)) > 1e-13 * jnp.max(jnp.abs(jnp.diag(Rb))))),
                stencil_nodes=int(len(np.unique(idx[idx < (L - 1) ** d]))))
    return jax.block_until_ready(data), info


# -------------------------------------------------------------------------------------------------- ECSW ----

def ecsw_states(d, L, fam_train, Phi_max, log=print, ntol=1e-10, ltol=1e-12):
    """Training states for ECSW (A1.8): list of dict(option, k, u_k, u_{k-1}, u_{k-2}) on the host."""
    rng = np.random.default_rng(ECSW_SEED_TRAJ)
    traj = rng.permutation(fam_train.n)[:N_ECSW_TRAJ]
    opts = H.OPTIONS[d]
    fom = H.make_fom(d, L)
    states = []
    for i, j in enumerate(traj):
        sc, dt = opts[i % len(opts)]
        N = int(round(H.HORIZON / dt))
        start = {'BE': [1], 'CN': [1], 'CNR': [1, 2, 3], 'BDF2': [1, 2]}[sc]     # A2.8: start-up states included
        ks = sorted(set(start) | {math.ceil(0.15 * N), math.ceil(0.4 * N), N})
        save = sorted({max(k - q, 0) for k in ks for q in (0, 1, 2)})
        sch = H.sched(sc, dt, save_steps=tuple(save))
        o = fom(fam_train.ic(L, j), fam_train.nu(j), sch, ntol, ltol)
        res = np.asarray(o['res'])
        assert np.all(np.isfinite(res[:N])) and float(np.max(res[:N])) <= ntol * (1 + 1e-9)
        out = np.asarray(o['out'])
        pos = {k: i_ for i_, k in enumerate(save)}
        for k in ks:
            states.append(dict(option=(sc, dt), k=k, traj=int(j), nu=fam_train.nu(j), fom_res=float(res[k - 1]),
                               u=out[pos[k]], un=out[pos[k - 1]], um=out[pos[max(k - 2, 0)]]))
    per = {}
    for st in states:
        per.setdefault(f'{st["option"][0]}|{st["option"][1]:.8g}', []).append(st['k'])
    return states, dict(trajectories=traj.tolist(), n_states=len(states), per_option=per)


def ecsw_contrib(states, Phi_r, d, L, cand_nodes=None, seed=ECSW_SEED_CAND, n_cand=CAND):
    """Contribution matrix G (rows = states x r, columns = candidates) and target b, block-scaled (A1.8).
    Candidates: all nodes if n <= n_cand, else half uniform and half drawn proportional to sum_s ||g_i(s)||."""
    import rom as RM
    n = (L - 1) ** d
    sc_c, rJ = RM.make_arm('none', d, L)
    LPhi, DPhi = H.lap_cols(Phi_r, d, L), H.dsum_cols(Phi_r, d, L)
    data = dict(Phi=Phi_r, LPhi=LPhi, DPhi=DPhi)
    be = jnp.asarray(H.BE_CO)

    @jax.jit
    def one_(u, un, um, nu, co, dt, data):           # large arrays are arguments, never closed over (captured constants)
        P = data['Phi']
        c, cn, cm = P.T @ u, P.T @ un, P.T @ um
        const = sc_c(cn, cm, nu, co, dt, data)
        r, J = rJ(c, const, nu, co, dt, data)
        return r, J, J.T @ r
    one = lambda u, un, um, nu, co, dt: one_(u, un, um, jnp.float64(nu), co, jnp.float64(dt), data)
    rows = []
    for s in states:
        sc, dt = s['option']
        a0, a1, a2, b0, b1, ns = H.SCHEMES[sc]
        co = be if s['k'] <= ns else jnp.asarray([a0, a1, a2, b0, b1])
        rows.append((s, co, dt))
    imp = np.zeros(n)
    if cand_nodes is None:
        if n <= n_cand:
            cand_nodes = np.arange(n)
        else:
            for s, co, dt in rows:
                r, J, _ = one(jnp.asarray(s['u']), jnp.asarray(s['un']), jnp.asarray(s['um']), s['nu'], co, dt)
                imp += np.asarray(jnp.abs(r) * jnp.linalg.norm(J, axis=1))
            rng = np.random.default_rng(seed)
            half = n_cand // 2
            uni = rng.choice(n, half, replace=False)
            rest = np.setdiff1d(np.arange(n), uni)
            pr = imp[rest] / imp[rest].sum()
            k_imp = min(n_cand - half, int(np.sum(pr > 0)))
            impn = rng.choice(rest, k_imp, replace=False, p=pr)
            cand_nodes = np.sort(np.concatenate([uni, impn]))
    cj = jnp.asarray(cand_nodes)
    r_ = Phi_r.shape[1]
    nc = len(cand_nodes)
    Gh = np.empty((len(rows) * r_, nc))             # host assembly, one device transfer (A15.4)
    bh = np.empty(len(rows) * r_)
    nb = np.empty(len(rows))
    for i, (s, co, dt) in enumerate(rows):
        r, J, b = one(jnp.asarray(s['u']), jnp.asarray(s['un']), jnp.asarray(s['um']), s['nu'], co, dt)
        Gh[i * r_:(i + 1) * r_] = np.asarray((J[cj] * r[cj][:, None]).T)
        bh[i * r_:(i + 1) * r_] = np.asarray(b)
        nb[i] = float(jnp.linalg.norm(b))
        del r, J, b
    floor = max(1e-8 * float(np.median(nb)), 1e-14 * float(np.max(nb)), 1e-300)     # A2.8 absolute positive floor
    keep = nb >= floor
    if not np.any(keep):
        raise RuntimeError('ECSW construction failure: every training target is below the floor')
    sc_ = 1. / np.maximum(nb, floor)
    rowsel = np.repeat(keep, r_)
    Gh *= np.repeat(sc_, r_)[:, None]
    bh *= np.repeat(sc_, r_)
    if not keep.all():
        Gh, bh = Gh[rowsel], bh[rowsel]
    G = jax.block_until_ready(jnp.asarray(Gh))
    b = jnp.asarray(bh)
    del Gh, bh
    return G, b, np.asarray(cand_nodes), dict(dropped_blocks=int(np.sum(~keep)), n_cand=int(nc),
                                              block_norms=nb.tolist())


NNLS_MAX_ITER = 40000                            # single-index LH needs about 1.15 x support iterations (A17)
BAN_RETRY_FRAC = 0.01


def ecsw_weights(G, b, cand_nodes, r, kappas=ECSW_KAPPA_PATH, taus=ECSW_TAUS, log=None):
    """A15.1/A16: "ECSW with support-targeted greedy NNLS". One greedy path per (mesh, rank) is built through the largest
    target (16 r), recorded at first crossings of s = kappa r (actual sizes and training residuals reported) and at the
    tau crossings (reported). A16.3/A16.5: whenever more than 1 % of the candidates were banned, the path is rebuilt once
    with banning disabled (same NNLS_MAX_ITER); one whole path is selected by the A16.5 order; both attempts are returned.
    A target unreached on the selected path is a construction failure."""
    targets = [int(k * r) for k in kappas]
    cap = min(max(targets), len(cand_nodes))
    sup = [t for t in targets if t <= cap]

    def attempt(max_drops):
        o = NN.greedy_nnls(G, b, taus=taus, supports=sup, cap=cap, log=log, max_drops=max_drops, max_iter=NNLS_MAX_ITER)
        cps = {}
        opt = o['reason'] == 'kkt' and o['kkt']['ok']          # A23: path ended at a KKT-certified NNLS optimum
        for k, t in zip(kappas, targets):
            x = o['records'].get(('s', t))
            if x is None and opt and len(o['support']) > 0:
                pos = o['weights'] > 0
                x = dict(support=o['support'][pos], weights=o['weights'][pos], relres=o['relres'], size=int(pos.sum()),
                         resvec=None, optimum=True)
            cps[f'k{k}'] = None if x is None else dict(nodes=cand_nodes[x['support']], weights=x['weights'],
                                                       relres=x['relres'], size=x['size'], optimum=x.get('optimum', False))
        cps_raw = {f'k{k}': o['records'].get(('s', t)) for k, t in zip(kappas, targets)}
        for k_, v in cps_raw.items():
            if v is not None and cps[k_] is not None:
                cps[k_]['resvec'] = v['resvec']
        for k_, v in cps.items():
            if v is not None and v.get('optimum'):
                Sv = jnp.asarray(np.searchsorted(cand_nodes, v['nodes']))
                v['resvec'] = np.asarray(jnp.asarray(b) - G[:, Sv] @ jnp.asarray(v['weights']))
        cps['_drops'] = o['drops']
        cps['_bnorm'] = o['bnorm']
        summ = dict(max_drops=max_drops, reason=o['reason'], final_relres=o['relres'], final_support=int(len(o['support'])),
                    method=o.get('method'), qr_drift=o.get('qr_drift'), drift_ok=o.get('drift_ok'),
                    drift_events=o.get('drift_events'), dependent_skips=o.get('dependent_skips'),
                    refactor_every=o.get('refactor_every'), capacity=o.get('capacity'),
                    banned=o['banned'], kkt=o['kkt'], iterations=len(o['path']),
                    taus={str(t): (None if o['records'][t] is None else dict(size=o['records'][t]['size'],
                                                                             relres=o['records'][t]['relres'])) for t in taus},
                    reached=sum(v is not None for k, v in cps.items() if not k.startswith('_')),
                    checkpoints={k: (None if v is None else dict(size=v['size'], relres=v['relres'],
                                                                     optimum=v.get('optimum', False)))
                                 for k, v in cps.items() if not k.startswith('_')})
        return cps, summ
    cps0, s0 = attempt(3)
    attempts = {'initial': (cps0, s0)}
    if s0['banned'] > BAN_RETRY_FRAC * len(cand_nodes):
        attempts['retry'] = attempt(10 ** 9)
    sel = 'initial'
    if 'retry' in attempts:
        s1 = attempts['retry'][1]
        if s1['reached'] > s0['reached']:
            sel = 'retry'
        elif s1['reached'] == s0['reached']:
            common = [k for k in s0['checkpoints'] if s0['checkpoints'][k] and s1['checkpoints'][k]]
            l0 = sum(np.log10(max(s0['checkpoints'][k]['relres'], 1e-300)) for k in common)
            l1 = sum(np.log10(max(s1['checkpoints'][k]['relres'], 1e-300)) for k in common)
            sel = 'retry' if l1 < l0 else 'initial'
    rec = {int(k[1:]): v for k, v in attempts[sel][0].items() if not k.startswith('_')}
    summary = dict(selected=sel, attempts={a: v[1] for a, v in attempts.items()},
                   checkpoints_all={a: v[0] for a, v in attempts.items()},
                   targets={f'k{k}': t for k, t in zip(kappas, targets)}, ban_retry_frac=BAN_RETRY_FRAC,
                   max_iter=NNLS_MAX_ITER, n_cand=int(len(cand_nodes)))
    return rec, summary


def ecsw_data(Phi_r, nodes, weights, d, L):
    idx = H.stencil_index(nodes, d, L)
    return jax.block_until_ready(dict(Pst=H.stencil_rows(Phi_r, idx), sw=jnp.sqrt(jnp.asarray(weights)))), \
        dict(sample=int(len(nodes)), stencil_nodes=int(len(np.unique(idx[idx < (L - 1) ** d]))))


def none_data(Phi_r, d, L):
    return jax.block_until_ready(dict(Phi=Phi_r, LPhi=H.lap_cols(Phi_r, d, L), DPhi=H.dsum_cols(Phi_r, d, L)))
