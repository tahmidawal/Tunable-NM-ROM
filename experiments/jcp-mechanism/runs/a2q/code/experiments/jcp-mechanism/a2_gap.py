"""jcp-mechanism A2 (DESIGN.md section 4, amendments A1-3, A1-4, A2-2, A2-3): the consistency gap of mesh-tested
advection sums against the continuum tested advection, as a function of the mesh spacing h, on FIXED reached states.

For a state c (coefficients of the frozen ordered bank, u = G_hat c) and a mesh of spacing h = 1/L, with the
normalised tested sums (the common L^{d/2} removed, so every quantity approximates int psi_a f):

    upwind   Nh_a = h^d sum_i psi_a(x_i) u_i (A_h u)_i          the FOM sign-upwind stencil (vendor code)
    central  Nh_a = h^d sum_i psi_a(x_i) u_i (sum_j d0_j u)_i     second-order central differences, ghost zeros (vendor)
    nodes    Nh_a = h^d sum_i psi_a(x_i) u_i (1 . grad u)(x_i)   the A1 arm: analytic gradient at the nodes
    target   N_a  = sum_q w_q psi_a(x_q) u (1 . grad u)(x_q)      Gauss 80^3 (3D) / 640^2 (2D); checks: Gauss 64^3 / 768^2
                                                                   and an independent family (lat32768 / fib121393)
    gap g(h; c) = ||Nh - N|| / ||N||  over the FROZEN tests (3D: first M modes of the n = 65 ordering; 2D: of L = 1024)

The tested sums use the vendor DST (3D, common.phiT) and the vendor separable sine tables (2D, hops.sep_project); no
test matrix is formed. Fields on the mesh come from the vendor off-mesh point code (3D offmesh.point_blocks; 2D
qcore.Model.values_grads) evaluated at the interior nodes. Controls (amendment A1-3 / A2-2), on the manufactured
state u* = mu(x) (1 + 0.3 sin(2 pi x1 + 0.4) + 0.2 cos(3 pi x2 - 0.1) [+ 0.25 sin(pi x3 + 1.1)]):
    C-pos   upwind and central gaps, and their predicted leading terms
            upwind  -h/2 int psi u Lap u,   central  h^2/6 int psi u sum_j u_jjj   (Gauss 80^3 / 640^2)
    C-neg-a a constant injected vector error 1e-2 ||N|| e/||e|| (gap identically 1e-2: tests normalisation and fit)
    C-neg-b central x 1.01 (descriptive)
Slopes are fitted by make_report.py, not here (the window and the screen are fixed in DESIGN.md).

Output: result.json (gates, sizes, timings) and gaps_{3d,2d}.npz (per state, per mesh, per stencil gaps, target checks).
Usage: python a2_gap.py --config configs/a2q.json --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
block = jax.block_until_ready


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def rel_max(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def rho(v, t):
    return np.linalg.norm(v - t, axis=-1) / np.maximum(np.linalg.norm(t, axis=-1), 1e-300)


# ------------------------------------------------------------------ manufactured state (C-pos), d = 2 or 3
def u_star(x):
    d = x.shape[-1]
    mu = 4.0 ** d * jnp.prod(x * (1 - x), axis=-1)
    br = 1 + 0.3 * jnp.sin(2 * jnp.pi * x[..., 0] + 0.4) + 0.2 * jnp.cos(3 * jnp.pi * x[..., 1] - 0.1)
    if d == 3:
        br = br + 0.25 * jnp.sin(jnp.pi * x[..., 2] + 1.1)
    return mu * br


def u_star_derivs(X):
    """u, 1.grad u, Lap u, sum_j u_jjj at points X (P, d), by nested forward-mode derivatives of the scalar u*."""
    d = X.shape[1]
    f = lambda x: u_star(x)
    e = jnp.eye(d)

    def one(x):
        g1 = [jax.jvp(f, (x,), (e[j],))[1] for j in range(d)]
        d1 = lambda x_, j: jax.jvp(f, (x_,), (e[j],))[1]
        d2 = lambda x_, j: jax.jvp(lambda y: d1(y, j), (x_,), (e[j],))[1]
        d3 = lambda x_, j: jax.jvp(lambda y: d2(y, j), (x_,), (e[j],))[1]
        return (f(x), sum(g1), sum(d2(x, j) for j in range(d)), sum(d3(x, j) for j in range(d)))
    f1 = jax.jit(jax.vmap(one))
    parts = [f1(jnp.asarray(X[s:s + (1 << 20)])) for s in range(0, len(X), 1 << 20)]
    return [np.concatenate([np.asarray(p_[i]) for p_ in parts]) for i in range(4)]


def chunks(N, size):
    return [(s, min(N, s + size)) for s in range(0, N, size)]


# ====================================================================================================== 3D
def run3d(cfg, rep, out, log):
    sys.path.insert(0, str(HERE / 'vendor' / 'quad3d'))
    import offmesh as OM
    from offmesh import C
    b3 = C.b3
    mp = ROOT / cfg['model3d']
    assert sha_file(mp) == cfg['expected_model3d_sha256']
    b = pickle.loads(mp.read_bytes())
    bank, T = jax.tree_util.tree_map(jnp.asarray, b['params']), np.asarray(b['rotation'])
    rz = np.load(ROOT / cfg['rules3d'])
    assert sha_file(ROOT / cfg['rules3d']) == cfg['expected_rules3d_sha256']
    R3 = {nm: (np.asarray(rz[f'{nm}_X']), np.asarray(rz[f'{nm}_w'])) for nm in ('gl80', 'gl64', 'lat32768')}
    S = np.load(HERE / 'inputs' / 'a2_states.npz')
    meshes = cfg['meshes3d']
    res = {}
    arrays = {}

    def tested(fields, n, idx):                  # normalised: (n-1)^{-3/2} Phi^T f  ~  int psi f
        return C.phiT(fields, n, idx) / (n - 1) ** 1.5

    tested_j = jax.jit(tested, static_argnums=(1,))
    upw = jax.jit(jax.vmap(lambda u, n: C.upwind(u, n), in_axes=(0, None)), static_argnums=(1,))
    cen = jax.jit(jax.vmap(lambda u, n: b3.central_adv_field_3d(u, n), in_axes=(0, None)), static_argnums=(1,))

    def target(Cs, X, w, kx):                     # normalised continuum value: continuum_adv with n = 2
        return np.asarray(OM.continuum_adv(2, bank, T, X, w, kx, Cs))

    for Rp in cfg['Rps3d']:
        order, lam = C.mode_order(65)
        M = C.complete_M(65, 4 * Rp, order, lam)
        kx = C.modes(65, M)[0]
        arrays[f'kx_R{Rp}'] = kx
        idx = tuple(jnp.asarray(kx[:, a]) for a in range(3))
        sets = {'fixed': S[f'd3_fixed_n65_R{Rp}']}
        arrays[f'R{Rp}_fixed_labels'] = S[f'd3_fixed_n65_R{Rp}_labels']
        for n in meshes:
            if f'd3_own_n{n}_R{Rp}' in S.files:
                sets[f'own{n}'] = S[f'd3_own_n{n}_R{Rp}']
        tg = split_targets(sets, {nm: target(np.concatenate(list(sets.values()), 0), *R3[r_], kx)
                                  for nm, r_ in (('t', 'gl80'), ('chk', 'gl64'), ('ind', 'lat32768'))})
        for k_ in sets:
            arrays[f'R{Rp}_{k_}_check_rho'] = rho(tg[k_]['chk'], tg[k_]['t'])
            arrays[f'R{Rp}_{k_}_lat32768_rho'] = rho(tg[k_]['ind'], tg[k_]['t'])
        log(f'3D R{Rp} M{M}: targets done; check worst ' +
            ', '.join(f"{k_} {arrays[f'R{Rp}_{k_}_check_rho'].max():.1e}" for k_ in sets))
        for n in meshes:
            assert kx.max() <= n - 2, (n, kx.max())
            use = ['fixed'] + ([f'own{n}'] if f'own{n}' in sets else [])
            Call = np.concatenate([sets[k_] for k_ in use], 0)
            X = C.interior_coords(n)
            N = len(X)
            U = np.empty((N, len(Call)))
            DU = np.empty((N, len(Call)))
            t0 = time.perf_counter()
            Cj = jnp.asarray(Call)
            for s, e in chunks(N, 1 << 18):
                bb, dd = OM.point_blocks(bank, T[:, :Rp], X[s:e], chunk=1 << 15)
                U[s:e] = np.asarray(bb @ Cj.T)
                DU[s:e] = np.asarray(dd @ Cj.T)
                del bb, dd
            if n == meshes[1]:                    # parity: point_blocks rows = the vendor mesh bank rows (decode)
                import tables as TB
                G = TB.feature_rows(bank, T[:, :Rp], X[:4096])
                rep['gates'][f'3d_point_vs_feature_rows_R{Rp}'] = rel_max(U[:4096, :4], np.asarray((Cj[:4] @ G).T))
                assert rep['gates'][f'3d_point_vs_feature_rows_R{Rp}'] < 1e-12, rep['gates']
            tb = time.perf_counter() - t0
            sb = max(1, int(2 ** 31 // (N * 8 * 40)))
            gaps = {k: [] for k in ('upwind', 'central', 'nodes', 'central101')}
            for s, e in chunks(len(Call), sb):
                u = jnp.asarray(U[:, s:e].T)
                du = jnp.asarray(DU[:, s:e].T)
                t_up = np.asarray(tested_j(upw(u, n), n, idx))
                t_ce = np.asarray(tested_j(cen(u, n), n, idx))
                t_nd = np.asarray(tested_j(u * du, n, idx))
                for k, v in (('upwind', t_up), ('central', t_ce), ('nodes', t_nd), ('central101', 1.01 * t_ce)):
                    gaps[k].append(v)
                del u, du
            off = 0
            for k_ in use:
                m_ = len(sets[k_])
                for st, vals in gaps.items():
                    V = np.concatenate(vals, 0)[off:off + m_]
                    arrays[f'R{Rp}_{k_}_n{n}_{st}'] = rho(V, tg[k_]['t'])
                off += m_
            rep['sizes'][f'3d_n{n}_R{Rp}'] = dict(N=N, states=len(Call), bank_seconds=tb)
            log(f'3D n{n} R{Rp}: fixed median gaps ' + ', '.join(
                f"{st} {np.median(arrays[f'R{Rp}_fixed_n{n}_{st}']):.2e}" for st in ('upwind', 'central', 'nodes')))
            del U, DU
            save_npz(out / 'gaps_3d.npz', arrays)
    # ---------------------------------------------------------------- manufactured state (C-pos / C-neg)
    kx = arrays[f'kx_R{cfg["Rps3d"][0]}']
    idx = tuple(jnp.asarray(kx[:, a]) for a in range(3))
    Xc, wc = R3['gl80']
    tgt = np.zeros(len(kx))
    lead_up = np.zeros(len(kx))
    lead_ce = np.zeros(len(kx))
    for s, e in chunks(len(wc), 1 << 15):
        u, g, lap, d3 = u_star_derivs(Xc[s:e])
        P = np.asarray(OM.psi(Xc[s:e], kx)) * wc[s:e, None]
        tgt += (u * g) @ P
        lead_up += (-0.5 * u * lap) @ P
        lead_ce += (u * d3 / 6.0) @ P
    mres = dict(target_norm=float(np.linalg.norm(tgt)), lead_upwind_rel=float(np.linalg.norm(lead_up) / np.linalg.norm(tgt)),
                lead_central_rel=float(np.linalg.norm(lead_ce) / np.linalg.norm(tgt)), meshes={})
    rng = np.random.default_rng(7)
    evec = rng.normal(size=len(kx))
    for n in meshes:
        X = C.interior_coords(n)
        u, g, _, _ = u_star_derivs(X)
        assert u.min() > 0, u.min()
        uj = jnp.asarray(u)[None]
        t_up = np.asarray(tested_j(upw(uj, n), n, idx))[0]
        t_ce = np.asarray(tested_j(cen(uj, n), n, idx))[0]
        t_nd = np.asarray(tested_j(uj * jnp.asarray(g)[None], n, idx))[0]
        h = 1.0 / (n - 1)
        neg_a = tgt + 1e-2 * np.linalg.norm(tgt) * evec / np.linalg.norm(evec)
        mres['meshes'][str(n)] = dict(
            h=h, upwind=float(rho(t_up, tgt)), central=float(rho(t_ce, tgt)), nodes=float(rho(t_nd, tgt)),
            central101=float(rho(1.01 * t_ce, tgt)), neg_a=float(rho(neg_a, tgt)),
            upwind_vs_lead=float(np.linalg.norm((t_up - tgt) - h * lead_up) / np.linalg.norm(h * lead_up)),
            central_vs_lead=float(np.linalg.norm((t_ce - tgt) - h * h * lead_ce) / np.linalg.norm(h * h * lead_ce)))
        log(f"3D manufactured n{n}: {mres['meshes'][str(n)]}")
    rep['manufactured_3d'] = mres


def split_targets(sets, vals):
    """Split target arrays computed on the concatenation of the state sets back into the sets."""
    out, off = {}, 0
    for k_, Cs in sets.items():
        out[k_] = {nm: v[off:off + len(Cs)] for nm, v in vals.items()}
        off += len(Cs)
    return out


def save_npz(path, arrays):
    np.savez(path, **arrays)


# ====================================================================================================== 2D
def run2d(cfg, rep, out, log):
    sys.path.insert(0, str(HERE / 'q2d'))
    import qcore as Q
    H, e_ = Q.H, Q.e
    assert sha_file(Q.CKPT) == cfg['expected_ckpt2d_sha256']
    mdl = Q.Model()
    S = np.load(HERE / 'inputs' / 'a2_states.npz')
    meshes = cfg['meshes2d']
    arrays = {}
    upw = jax.jit(jax.vmap(lambda u, L: e_.spatial(u, L)[0], in_axes=(0, None)), static_argnums=(1,))

    def central(u, L):
        p = jnp.pad(u.reshape(L - 1, L - 1), 1)
        c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
        return (c * L * ((xp - xm) + (yp - ym)) / 2.0).reshape(-1)
    cen = jax.jit(jax.vmap(central, in_axes=(0, None)), static_argnums=(1,))

    def tested(F, sx, sy, L):                     # normalised: Phi^T f / L  ~  int psi f
        return H.sep_project(F.reshape(-1, L - 1, L - 1), sx, sy, L) / L

    tested_j = jax.jit(tested, static_argnums=(3,))

    def target(Cs, rule, Rp, kx, ky):             # normalised continuum value: the off-mesh point form with L = 1
        X, w = Q.offmesh_rule(rule)
        f = jax.jit(jax.vmap(Q.tested_value('point', 1), in_axes=(0, None)))
        acc = np.zeros((len(Cs), len(kx)))
        for s, e in chunks(len(X), 32768):
            d = Q.offmesh_data(mdl, Rp, X[s:e], w[s:e], 1, kx, ky, 'point')
            acc += np.asarray(f(jnp.asarray(Cs), d))
            del d
        return acc

    for s_name in cfg['settings2d']:
        Rp, M = Q.SETTINGS[s_name]['Rp'], Q.SETTINGS[s_name]['M']
        kx, ky, _ = H.modes_lean(1024, M)
        arrays[f'{s_name}_kx'], arrays[f'{s_name}_ky'] = kx, ky
        sets = {'fixed': S[f'd2_fixed_L1024_{s_name}']}
        arrays[f'{s_name}_fixed_labels'] = S[f'd2_fixed_L1024_{s_name}_labels']
        for L in meshes:
            if f'd2_own_L{L}_{s_name}' in S.files:
                sets[f'own{L}'] = S[f'd2_own_L{L}_{s_name}']
        tg = split_targets(sets, {nm: target(np.concatenate(list(sets.values()), 0), r_, Rp, kx, ky)
                                  for nm, r_ in (('t', 'gauss640'), ('chk', 'gauss768'), ('ind', 'fib121393'))})
        for k_ in sets:
            arrays[f'{s_name}_{k_}_check_rho'] = rho(tg[k_]['chk'], tg[k_]['t'])
            arrays[f'{s_name}_{k_}_fib121393_rho'] = rho(tg[k_]['ind'], tg[k_]['t'])
        log(f'2D {s_name}: targets done; check worst ' +
            ', '.join(f"{k_} {arrays[f'{s_name}_{k_}_check_rho'].max():.1e}" for k_ in sets))
        for L in meshes:
            assert max(kx.max(), ky.max()) <= L - 1, L
            use = ['fixed'] + ([f'own{L}'] if f'own{L}' in sets else [])
            Call = np.concatenate([sets[k_] for k_ in use], 0)
            x = np.arange(1, L) / L
            N = (L - 1) ** 2
            U = np.empty((N, len(Call)))
            GS = np.empty((N, len(Call)))
            Cj = jnp.asarray(Call)
            t0 = time.perf_counter()
            rows = max(1, (1 << 18) // (L - 1))
            for i0 in range(0, L - 1, rows):
                i1 = min(L - 1, i0 + rows)
                X = np.stack(np.meshgrid(x[i0:i1], x, indexing='ij'), -1).reshape(-1, 2)
                v, gx, gy = mdl.values_grads(X, Rp, chunk=8192)
                U[i0 * (L - 1):i1 * (L - 1)] = np.asarray(v @ Cj.T)
                GS[i0 * (L - 1):i1 * (L - 1)] = np.asarray((gx + gy) @ Cj.T)
                del v, gx, gy
            if L == 256:                          # parity: the mesh values are the vendor rotated mesh bank (decode)
                Gm = Q.build_rotated_bank(mdl, L, [0, 128, 384][:2 + (Rp > 128)])
                Gp = Q.BK.prefix(Gm, [0, 128, 384][:2 + (Rp > 128)], Rp)
                rep['gates'][f'2d_values_vs_mesh_bank_{s_name}'] = rel_max(
                    U[:, :4], np.asarray(jnp.stack([Q.H.bank_apply(Gp, c_) for c_ in Cj[:4]], 1)))
                del Gm, Gp
                assert rep['gates'][f'2d_values_vs_mesh_bank_{s_name}'] < 1e-12, rep['gates']
            tb = time.perf_counter() - t0
            sx, sy = (jnp.asarray(t_) for t_ in H.sine_tables(L, kx, ky))
            sb = max(1, int(2 ** 31 // (N * 8 * 24)))
            gaps = {k: [] for k in ('upwind', 'central', 'nodes', 'central101')}
            for s, e in chunks(len(Call), sb):
                u = jnp.asarray(U[:, s:e].T)
                gs = jnp.asarray(GS[:, s:e].T)
                t_up = np.asarray(tested_j(upw(u, L), sx, sy, L))
                t_ce = np.asarray(tested_j(cen(u, L), sx, sy, L))
                t_nd = np.asarray(tested_j(u * gs, sx, sy, L))
                for k, v in (('upwind', t_up), ('central', t_ce), ('nodes', t_nd), ('central101', 1.01 * t_ce)):
                    gaps[k].append(v)
                del u, gs
            off = 0
            for k_ in use:
                m_ = len(sets[k_])
                for st, vals in gaps.items():
                    V = np.concatenate(vals, 0)[off:off + m_]
                    arrays[f'{s_name}_{k_}_L{L}_{st}'] = rho(V, tg[k_]['t'])
                off += m_
            rep['sizes'][f'2d_L{L}_{s_name}'] = dict(N=N, states=len(Call), bank_seconds=tb)
            log(f'2D L{L} {s_name}: fixed median gaps ' + ', '.join(
                f"{st} {np.median(arrays[f'{s_name}_fixed_L{L}_{st}']):.2e}" for st in ('upwind', 'central', 'nodes')))
            del U, GS
            save_npz(out / 'gaps_2d.npz', arrays)
    # ---------------------------------------------------------------- manufactured state (C-pos / C-neg)
    s0 = cfg['settings2d'][0]
    kx, ky = arrays[f'{s0}_kx'], arrays[f'{s0}_ky']
    Xc, wc = Q.offmesh_rule('gauss640')
    tgt, lead_up, lead_ce = (np.zeros(len(kx)) for _ in range(3))
    for s, e in chunks(len(wc), 1 << 15):
        u, g, lap, d3 = u_star_derivs(Xc[s:e])
        psi_, _ = Q.continuum_tests(Xc[s:e], kx, ky)
        P = psi_ * wc[s:e, None]
        tgt += (u * g) @ P
        lead_up += (-0.5 * u * lap) @ P
        lead_ce += (u * d3 / 6.0) @ P
    mres = dict(target_norm=float(np.linalg.norm(tgt)), lead_upwind_rel=float(np.linalg.norm(lead_up) / np.linalg.norm(tgt)),
                lead_central_rel=float(np.linalg.norm(lead_ce) / np.linalg.norm(tgt)), meshes={})
    evec = np.random.default_rng(7).normal(size=len(kx))
    for L in meshes:
        x = np.arange(1, L) / L
        X = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
        u, g, _, _ = u_star_derivs(X)
        assert u.min() > 0, u.min()
        sx, sy = (jnp.asarray(t_) for t_ in H.sine_tables(L, kx, ky))
        uj = jnp.asarray(u)[None]
        t_up = np.asarray(tested_j(upw(uj, L), sx, sy, L))[0]
        t_ce = np.asarray(tested_j(cen(uj, L), sx, sy, L))[0]
        t_nd = np.asarray(tested_j(uj * jnp.asarray(g)[None], sx, sy, L))[0]
        h = 1.0 / L
        neg_a = tgt + 1e-2 * np.linalg.norm(tgt) * evec / np.linalg.norm(evec)
        mres['meshes'][str(L)] = dict(
            h=h, upwind=float(rho(t_up, tgt)), central=float(rho(t_ce, tgt)), nodes=float(rho(t_nd, tgt)),
            central101=float(rho(1.01 * t_ce, tgt)), neg_a=float(rho(neg_a, tgt)),
            upwind_vs_lead=float(np.linalg.norm((t_up - tgt) - h * lead_up) / np.linalg.norm(h * lead_up)),
            central_vs_lead=float(np.linalg.norm((t_ce - tgt) - h * h * lead_ce) / np.linalg.norm(h * h * lead_ce)))
        log(f"2D manufactured L{L}: {mres['meshes'][str(L)]}")
    rep['manufactured_2d'] = mres


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.get('local_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    log = lambda s: print(f'[{round(time.perf_counter() - begin, 1)}s] {s}', flush=True)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    st = HERE / 'inputs' / 'a2_states.npz'
    assert sha_file(st) == cfg['expected_states_sha256']
    rep = dict(config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               host=os.uname().nodename, backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
               nvidia_smi=smi, jax_version=jax.__version__, precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
               states_sha256=cfg['expected_states_sha256'], gates={}, sizes={}, complete=False)
    save = lambda: (out / 'result.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    save()
    if cfg.get('run3d', True):
        run3d(cfg, rep, out, log)
        save()
    if cfg.get('run2d', True):
        run2d(cfg, rep, out, log)
        save()
    rep['complete'] = True
    rep['seconds'] = time.perf_counter() - begin
    save()
    print('A2 COMPLETE', flush=True)


if __name__ == '__main__':
    main()
