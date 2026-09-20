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
