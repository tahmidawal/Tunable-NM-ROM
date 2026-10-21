"""CPU unit tests of mmcore on tiny meshes (run: JAX_PLATFORMS=cpu python test_mmcore.py)."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import mmcore as M

H, BD = M.H, M.BD


def rand_basis(L, r, seed=0):
    rng = np.random.default_rng(seed)
    n = (L - 1) ** 2
    # smooth random modes: low sine coefficients only, so restriction is nearly exact
    m = L - 1
    C = np.zeros((r, m, m))
    C[:, :6, :6] = rng.standard_normal((r, 6, 6))
    V = np.asarray(M._dst2(jnp.asarray(C))).reshape(r, n).T
    Q, _ = np.linalg.qr(V)
    return Q


def test_resample():
    Ls, Lt, r = 16, 64, 10
    P = rand_basis(Ls, r)
    Pt, R, info = M.resample_basis(P, Ls, Lt)
    assert R is None and info['orth'] <= 1e-12, info
    e = M.node_consistency(P, Ls, Pt, Lt)
    assert e <= 1e-12, e
    for mut in ('shift', 'transpose'):
        Pm, _, im = M.resample_basis(P, Ls, Lt, mutate=mut)
        em = M.node_consistency(P, Ls, Pm, Lt)
        print('mutation', mut, 'node consistency', em, 'orth', im['orth'])
        assert em > 1e-6, (mut, em)
    # round trip
    rt = M.roundtrip_error(P, Ls, Lt, cols=4)
    assert rt <= 1e-12, rt
    # restriction with and without QR
    Pf = rand_basis(64, r, 1)
    Pc, Rr, ic = M.resample_basis(Pf, 64, 32)
    assert ic['qr'] and ic['orth'] <= 1e-12 and ic['rank_ratio'] > 1e-10, ic
    Pc0, _, ic0 = M.resample_basis(Pf, 64, 32, qr=False)
    print('restriction without QR orth', ic0['orth'], 'with', ic['orth'])
    # codes: physical field u_f = Pf c on 64; restricted field (low-pass) in the coarse basis
    c = np.random.default_rng(3).standard_normal(r)
    uf = Pf @ c
    uc = np.asarray(M.resample(uf[:, None], 64, 32))[:, 0] * (32 / 64)      # physical (low-pass) nodal values on 32
    cc = np.asarray(Pc).T @ uc
    ct = M.transform_codes(c[None, :], 64, 32, r, Rr)[0]
    assert np.max(np.abs(cc - ct)) <= 1e-10 * np.max(np.abs(cc)), (cc[:3], ct[:3])
    # codes under prolongation
    up = np.asarray(M.resample((P @ c)[:, None], Ls, Lt))[:, 0] * (Lt / Ls)
    cp = np.asarray(Pt).T @ up
    assert np.max(np.abs(cp - M.transform_codes(c[None], Ls, Lt, r)[0])) <= 1e-10 * np.max(np.abs(cp))
    print('resample ok')


def test_map():
    Ls, Lt = 16, 64
    p = np.arange((Ls - 1) ** 2)
    q = M.map_nodes(p, Ls, Lt)
    i, j = np.unravel_index(p, (Ls - 1, Ls - 1))
    k, l = np.unravel_index(q, (Lt - 1, Lt - 1))
    assert np.all((i + 1) / Ls == (k + 1) / Lt) and np.all((j + 1) / Ls == (l + 1) / Lt)
    assert np.any(M.map_nodes(p, Ls, Lt, mutate=True) != q)
    print('map ok')


def test_blocked():
    L, r = 32, 12
    fam = H.Family(2, 'train')
    states, _ = BD.ecsw_states(2, L, fam, None)
    states = states[:20]
    P = rand_basis(L, r, 5)
    G0, b0, c0, i0 = BD.ecsw_contrib(states, jnp.asarray(P), 2, L, n_cand=256)
    G1, b1, c1, i1 = M.ecsw_contrib_blocked(states, jnp.asarray(P), r, L, block=100, n_cand=256, log=None)
    assert np.array_equal(c0, c1), 'candidates differ'
    eg = float(jnp.linalg.norm(G0 - G1) / jnp.linalg.norm(G0))
    eb = float(jnp.linalg.norm(b0 - b1) / jnp.linalg.norm(b0))
    print('blocked vs dense', eg, eb)
    assert eg <= 1e-10 and eb <= 1e-10
    G2, b2, c2, _ = M.ecsw_contrib_blocked(states, jnp.asarray(P), r, L, block=100, n_cand=256, log=None,
                                           mutation='drop_adv_jac', cand_nodes=c0)
    em = float(jnp.linalg.norm(G0 - G2) / jnp.linalg.norm(G0))
    print('mutation drop_adv_jac', em)
    assert em > 1e-6
    print('blocked ok')


if __name__ == '__main__':
    test_resample()
    test_map()
    test_blocked()
    print('ALL OK')
