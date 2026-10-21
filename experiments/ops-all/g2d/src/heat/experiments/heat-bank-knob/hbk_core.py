"""heat-bank-knob: deployment-time nested truncation R' of the frozen heat banks (2D wide R=128, 3D R=320). No training.

Offline, TRAINING data only, at the training mesh: G = Q_G R_G. Samples
    a_i = [Q_G^T u_i ; R_G h(z_i)] / ||u_i||   (projection coefficients of training field u_i, and the head's
                                                stored-code coefficients, both in the orthonormal bank metric)
    A = U S V_s^T,  T = R_G^{-1} V_s,  L = V_s^T R_G = T^{-1}.
This is the SVD of G Sigma^{1/2} (Sigma = training second moment of coefficient vectors): POD of the training
decoded fields inside span(G). The rotated bank G' = G T has its columns ordered by training energy. A truncation
R' keeps G'_{R'} = G' [:, :R'] and the model

    u = G'_{R'} L_{R'} (h(z) + D_q y)       (L_{R'} = first R' rows of L)

i.e. the parent model with head h -> L_{R'} h and corrections D -> L_{R'} D, written in R'-dimensional coefficient
coordinates. Every other piece (weak matrix a' = Phi^T G'_{R'}, triangular factor of G'_{R'}, elimination of y,
nearest-code starts, LM, CN / batched-fit stepping) is the parent code (heat3d-bank core.make_stages), applied to
the truncated quantities; only the bank reads (encode G'^T u, decode G' c) touch exactly R' columns.
The linear rung (q = R', head dropped) is the free-coefficient model in G'_{R'} with the same stepping families.

The rotated bank is stored as row blocks x nested COLUMN blocks (edges = the R' ladder) so a truncated query
reads only its first R' columns, with no slicing copy (slicing the 42 GB 256^3 bank duplicated it: job 4175066).
"""
from __future__ import annotations
import hashlib, time
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import core as C


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


# ------------------------------------------------------------------ rotation (offline, training only)
def training_fields(tcfg):
    d, n = tcfg['d'], tcfg['train_intervals']
    draws = C.family(tcfg['family'], tcfg['train_seed'], tcfg['train_count'])
    prop = C.make_propagate(d); lam = C.eig_grid(n, d); t = jnp.asarray(tcfg['times'])
    u = np.concatenate([np.asarray(prop(C.initial_grid(n, d, p), lam, t, tcfg['diffusivity'])).reshape(len(t), -1) for p in draws])
    return draws, u


def make_rotation(model, tcfg):
    """Returns T, L (R x R) and diagnostics. Training draws and stored training codes only."""
    t0 = time.perf_counter()
    draws, u = training_fields(tcfg)
    assert len(u) == len(model['codes']), (len(u), len(model['codes']))
    G = np.concatenate([np.asarray(b) for b in C.bank_at(model, tcfg['train_intervals'])])
    Q, Rg = np.linalg.qr(G, mode='reduced'); sgn = np.where(np.diag(Rg) < 0, -1., 1.); Q = Q * sgn; Rg = Rg * sgn[:, None]
    nu = np.sqrt(np.sum(u * u, axis=1))[:, None]
    A_proj = (u @ Q) / nu
    heads = np.asarray(jax.jit(model['head_fn'])(model['head_params'], model['codes']))
    A_head = (heads @ Rg.T) / nu
    A = np.concatenate((A_proj, A_head))
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    Vs = Vt.T
    Vs = Vs * np.where(Vs[np.argmax(np.abs(Vs), axis=0), np.arange(Vs.shape[1])] < 0, -1., 1.)   # sign convention
    T = np.linalg.solve(Rg, Vs); L = Vs.T @ Rg
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    ep = np.cumsum(np.sum((A_proj @ Vs) ** 2, axis=0)) / np.sum(A_proj ** 2)
    eh = np.cumsum(np.sum((A_head @ Vs) ** 2, axis=0)) / np.sum(A_head ** 2)
    sv = np.linalg.svd(Rg, compute_uv=False)
    ks = [k for k in (8, 16, 32, 48, 64, 96, 128, 192, 256, 320) if k <= len(s)]
    # training projection floor in the truncated rotated bank (field-relative, worst over training fields)
    perp_full = np.maximum(1. - np.sum(A_proj ** 2, axis=1), 0.)
    P = A_proj @ Vs
    floors = {str(k): float(np.sqrt(np.max(perp_full + np.sum(P[:, k:] ** 2, axis=1)))) for k in ks}
    info = dict(training_intervals=int(tcfg['train_intervals']), training_draws=int(len(draws)), training_fields=int(len(u)),
                training_draws_sha=C.sha(draws),
                sample_rule='rows = [Q_G^T u_i ; R_G h(z_i)] / ||u_i||, training draws and stored training codes only',
                R_G_condition_number=float(sv[0] / sv[-1]), L_times_T_identity_deviation=float(np.linalg.norm(L @ T - np.eye(len(s)))),
                singular_values=s.tolist(), cumulative_energy={str(k): float(energy[k - 1]) for k in ks},
                cumulative_energy_projection_samples={str(k): float(ep[k - 1]) for k in ks},
                cumulative_energy_head_samples={str(k): float(eh[k - 1]) for k in ks},
                training_projection_floor_worst=floors,
                T_sha256=sha_array(T), L_sha256=sha_array(L), seconds=time.perf_counter() - t0)
    return T, L, info


# ------------------------------------------------------------------ nested rotated bank at a mesh
def build_bank(model, n, T, edges, keep_orig, chunk=1 << 18):
    """Evaluates the frozen feature network once per row block. Returns the rotated nested bank
    [[B_{r,j}]_j]_r (B_{r,j} = (G T)[rows r, edges j:j+1]) and optionally the ORIGINAL row blocks (parity)."""
    d = model['d']; a = axis_nodes = C.axis_nodes(n); N = (n - 1) ** d
    rot = jnp.asarray(model['rotation']); Tj = jnp.asarray(T)

    @jax.jit
    def piece(p, idx):
        sub = jnp.stack(jnp.unravel_index(idx, (n - 1,) * d), -1)
        return model['feature_fn'](p, jnp.asarray(axis_nodes)[sub]) @ rot

    @jax.jit
    def rotate(g, Tj):
        gr = g @ Tj
        return tuple(gr[:, lo:hi] for lo, hi in zip(edges[:-1], edges[1:]))

    nested, orig = [], []
    for lo, hi in C.row_blocks(N):
        parts = [piece(model['bank_params'], jnp.arange(s, min(s + chunk, hi))) for s in range(lo, hi, chunk)]
        g = C.block(jnp.concatenate(parts) if len(parts) > 1 else parts[0]); del parts
        nested.append(list(C.block(rotate(g, Tj))))
        if keep_orig: orig.append(g)
        del g
    return nested, (orig if keep_orig else None)


def nproject(nested, vec, nb):
    """(G'_{R'})^T vec, R' = edges[nb]; reads column blocks 0..nb-1 only."""
    acc = [0.] * nb; lo = 0
    for row in nested:
        v = vec[lo:lo + row[0].shape[0]]; lo += row[0].shape[0]
        for j in range(nb): acc[j] = acc[j] + row[j].T @ v
    return jnp.concatenate(acc) if nb > 1 else acc[0]


def nexpand(coefs, nested, nb):
    """coefs @ (G'_{R'})^T, coefs [T, R']; reads column blocks 0..nb-1 only."""
    out = []; col = 0
    for row in nested:
        acc = None; col = 0
        for j in range(nb):
            w = row[j].shape[1]; part = coefs[:, col:col + w] @ row[j].T; col += w
            acc = part if acc is None else acc + part
        out.append(acc)
    return out[0] if len(out) == 1 else jnp.concatenate(out, axis=1)


def nested_tsqr_r(nested, chunk=1 << 20):
    """Triangular factor of the full rotated bank; its leading R' x R' block is the factor of G'_{R'} (nested QR)."""
    rs = []
    for row in nested:
        g = np.concatenate([np.asarray(b) for b in row], axis=1)
        rs += [np.linalg.qr(g[s:s + chunk], mode='r') for s in range(0, g.shape[0], chunk)]
        del g
    r = np.linalg.qr(np.concatenate(rs), mode='r') if len(rs) > 1 else rs[0]
    return r * np.where(np.diag(r) < 0, -1., 1.)[:, None]


def nested_weak_matrix(nested, modes, n, d, cols=16):
    """a' = tests^T G' (M x R) via DST of rotated bank columns."""
    f = jax.jit(lambda b: C.moments(jnp.moveaxis(b.reshape((n - 1,) * d + (b.shape[1],)), -1, 0), modes, n, d))
    out = []
    for j in range(len(nested[0])):
        w = nested[0][j].shape[1]
        for s in range(0, w, cols):
            out.append(np.asarray(f(jnp.concatenate([row[j][:, s:s + cols] for row in nested]))))
    return np.concatenate(out).T


# ------------------------------------------------------------------ truncated NM-ROM (parent make_stages, bank ops injected)
def truncated_model(model, L, Rp):
    """Head h -> L_{R'} h (L passed as a jit argument inside head_params), directions D -> L_{R'} D."""
    Lr = jnp.asarray(L[:Rp])
    base = model['head_fn']
    return dict(model, head_fn=lambda p, z: base(p[0], z) @ p[1].T, head_params=(model['head_params'], Lr),
                directions=np.asarray(L[:Rp]) @ np.asarray(model['directions']))


def make_stages(model, setup, q, opt, project, expand):
    """Verbatim port of heat3d-bank core.make_stages @5f1b048d with bank_project/bank_expand replaced by
    project(bank, vec) / expand(coefs, bank). With project/expand = the parent's functions it IS the parent."""
    d, n, modes = model['d'], setup['n'], setup['modes']; head_fn = model['head_fn']; hp = model['head_params']
    codes = model['codes']; times = np.asarray(setup['times']); nu = setup['nu']; lam = jnp.asarray(setup['mode_lam'])
    assert 0 <= q <= np.asarray(setup['directions']).shape[1] and q + codes.shape[1] <= setup['a'].shape[0], ('invalid q', q)
    D = np.asarray(setup['directions'])[:, :q]; Dj = jnp.asarray(D)
    E0 = C.eliminate(setup['rtri'] if opt['init'] == 'field' else setup['a'], D, opt.get('compress', False))
    E1 = C.eliminate(setup['a'], D, opt.get('compress', False))
    rtri_t = jnp.asarray(setup['rtri'].T)
    fit0 = C.lm(head_fn, opt['fit_budget'], opt['tolerance'], opt.get('cholesky', False)); fit1 = C.lm(head_fn, opt['step_budget'], opt['tolerance'], opt.get('cholesky', False))
    def proj(E, t):
        tp = t - E['qq'] @ (E['qq'].T @ t); scale = jnp.maximum(jnp.linalg.norm(tp), 1e-14)
        return (tp if E['rows'] is None else E['rows'].T @ tp), scale
    lib0 = head_fn(hp, codes) @ E0['ap'].T; lib1 = head_fn(hp, codes) @ E1['ap'].T
    def recover(E, z, t):
        h = head_fn(hp, z)
        return h + Dj @ jsl.solve_triangular(E['rr'], E['qq'].T @ (t - E['a'] @ h), lower=False) if q else h
    def starts(lib, tp, k, mean):
        z = codes[jnp.argsort(jnp.sum((lib - tp) ** 2, axis=1))[:k]]
        return jnp.concatenate((z, jnp.mean(codes, axis=0)[None])) if mean else z
    def best_fit(fit, E, lib, t, k, mean):
        tp, scale = proj(E, t); zs, stats = jax.vmap(lambda z: fit(hp, E['ap'], tp, z, scale))(starts(lib, tp, k, mean))
        best = jnp.argmin(stats[:, 3]); z = zs[best]; return z, recover(E, z, t), stats, stats[best]

    def encode(u0, bank):
        m0 = C.moments(u0, modes, n, d)
        t0 = jsl.solve_triangular(rtri_t, project(bank, u0.reshape(-1)), lower=True) if opt['init'] == 'field' else m0
        return t0, m0
    def init(t0):
        z, c0, stats, chosen = best_fit(fit0, E0, lib0, t0, opt['starts'], opt.get('mean_start', False))
        return z, c0, jnp.concatenate((stats, chosen[None]))
    nout = len(times) - 1; dtout = float(times[1] - times[0])
    if opt['stepping'] == 'cn':
        dt = opt['dt']; stride = int(round(dtout / dt)); nsteps = stride * nout
        assert np.allclose(np.arange(len(times)) * stride * dt, times)
        factor = (1 - dt * nu * lam / 2) / (1 + dt * nu * lam / 2)
    else:
        stride, nsteps = 1, nout; factor = jnp.exp(-nu * lam * dtout)
    def evolve(z, coef, m0):
        if opt['stepping'] == 'exact_direct':
            targets = jnp.exp(-nu * lam[None] * jnp.asarray(times[1:])[:, None]) * m0[None]
            _, coefs, _, chosen = jax.vmap(lambda t: best_fit(fit1, E1, lib1, t, opt.get('direct_starts', 1), False))(targets)
            return coefs, chosen
        assert opt['stepping'] == 'cn', opt['stepping']
        def step(carry, _):
            zp, cp = carry; t = factor * (E1['a'] @ cp)
            zn, info = fit1(hp, E1['ap'], *proj(E1, t)[:1], zp, proj(E1, t)[1]); cn = recover(E1, zn, t)
            return (zn, cn), (cn, info)
        _, (coefs, infos) = jax.lax.scan(step, (z, coef), None, length=nsteps)
        return coefs[stride - 1::stride], infos
    def decode(coefs, bank):
        return expand(coefs, bank).reshape((len(times),) + (n - 1,) * d)
    def query(u0, bank):
        t0, m0 = encode(u0, bank); z, c0, s0 = init(t0); coefs, s1 = evolve(z, c0, m0)
        return decode(jnp.concatenate((c0[None], coefs)), bank), s0, s1
    return dict(encode=jax.jit(encode), init=jax.jit(init), evolve=jax.jit(evolve), decode=jax.jit(decode), query=jax.jit(query))


def make_linear(setup, init, stepping, dt, project, expand):
    """LINEAR RUNG (q = R', head dropped): free coefficients c in the truncated bank.
    init 'field': c0 = argmin ||G' c - u0|| (exact projection); 'moments': c0 = pinv(a') Phi^T u0.
    stepping 'cn': the reduced weak Crank-Nicolson step c <- pinv(a') (factor * a' c) (the NM-ROM's CN step with
    free coefficients), composed offline into one map per output time; 'exact_direct': each output fitted to the
    exactly propagated moments of the supplied field (the NM-ROM batched fit with free coefficients).
    Returns query(u0, bank) -> (fields,) and query stages for the profile."""
    a = np.asarray(setup['a']); pinv = np.linalg.pinv(a); n, d, modes = setup['n'], setup['d'], setup['modes']
    times = np.asarray(setup['times']); nu = setup['nu']; lam = np.asarray(setup['mode_lam'])
    rt = jnp.asarray(setup['rtri']); pj = jnp.asarray(pinv)
    if stepping == 'cn':
        stride = int(round((times[1] - times[0]) / dt)); assert np.allclose(np.arange(len(times)) * stride * dt, times)
        step = pinv @ (((1 - dt * nu * lam / 2) / (1 + dt * nu * lam / 2))[:, None] * a)
        per = np.linalg.matrix_power(step, stride)
        maps = [np.eye(a.shape[1])]
        for _ in range(len(times) - 1): maps.append(per @ maps[-1])
        mj = jnp.asarray(np.stack(maps))
    def encode(u0, bank):
        m0 = C.moments(u0, modes, n, d)
        if init == 'field':
            c0 = jsl.solve_triangular(rt, jsl.solve_triangular(rt.T, project(bank, u0.reshape(-1)), lower=True), lower=False)
        else:
            c0 = pj @ m0
        return c0, m0
    def evolve(c0, m0):
        if stepping == 'cn':
            return mj @ c0
        later = (jnp.exp(-nu * jnp.asarray(lam)[None] * jnp.asarray(times[1:])[:, None]) * m0[None]) @ pj.T
        return jnp.concatenate((c0[None], later))
    def decode(coefs, bank):
        return expand(coefs, bank).reshape((len(times),) + (n - 1,) * d)
    def query(u0, bank):
        c0, m0 = encode(u0, bank); return (decode(evolve(c0, m0), bank),)
    return dict(encode=jax.jit(encode), evolve=jax.jit(evolve), decode=jax.jit(decode), query=jax.jit(query))


def lm_failures(s0, s1):
    """Non-stationary LM exits (reason != 1) of the selected init start and every step/fit."""
    i = np.asarray(s0).reshape(-1, 5)[-1]; st = np.asarray(s1).reshape(-1, 5)
    return int((i[2] != 1) + (st[:, 2] != 1).sum()), float(i[0]), float(st[:, 0].mean()), float(st[:, 0].max())
