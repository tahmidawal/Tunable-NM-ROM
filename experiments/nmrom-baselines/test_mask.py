"""CPU-only structural test: the index tables reproduce the paper's dense mask exactly."""
import numpy as np, kimae
for nx, ny, b, db in ((5, 7, 6, 2), (4, 4, 10, 1), (6, 5, 3, 3), (58, 58, 100, 10)):
    idx, valid, M2 = kimae.mask_tables(nx, ny, b, db)
    if nx * ny > 2000:
        assert M2 == 33730 and idx.shape[1] == 320; print('kim 58x58: M2', M2, 'nnz', valid.sum()); continue
    S = kimae.dense_mask(nx, ny, b, db)
    T = np.zeros_like(S)
    for i in range(nx * ny):
        cols = idx[i][valid[i]]
        assert len(set(cols)) == len(cols), 'duplicate column'
        T[i, cols] = True
    assert (S == T).all(), (nx, ny, b, db)
    print('ok', nx, ny, b, db, 'nnz', S.sum())
Phi = np.linalg.qr(np.random.default_rng(0).normal(size=(200, 7)))[0]
z = kimae.greedy_samples(Phi, 11); assert len(z) == 11
print('cond ZtPhi', np.linalg.cond(Phi[z]))

# slice form == gather form (values and gradients), asymmetric random weights
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
for nx, ny, b, db in ((5, 7, 6, 2), (6, 5, 9, 3), (7, 4, 10, 10)):
    idx, valid, M2 = kimae.mask_tables(nx, ny, b, db)
    r = np.random.default_rng(1)
    W2 = jnp.asarray(r.normal(size=idx.shape) * valid); h = jnp.asarray(r.normal(size=M2))
    f1 = lambda W, h: jnp.einsum('np,np->n', W, h[jnp.asarray(idx)])
    f2 = lambda W, h: kimae.masked_out(W, h, ny, b, db)
    assert np.allclose(f1(W2, h), f2(W2, h), atol=1e-12)
    c = jnp.asarray(r.normal(size=nx * ny))
    g1 = jax.grad(lambda W, h: c @ f1(W, h), (0, 1))(W2, h); g2 = jax.grad(lambda W, h: c @ f2(W, h), (0, 1))(W2, h)
    assert np.allclose(g1[0] * valid, g2[0] * valid, atol=1e-12) and np.allclose(g1[1], g2[1], atol=1e-12)
    print('slice==gather', nx, ny, b, db)
