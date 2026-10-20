import numpy as np, time
z = np.load('G.npz'); G, b = z['G'], z['b']
m, N = G.shape
Gb = G.T @ b
S = []; x = np.zeros(0); drops_new = 0; drops_old = 0; t = time.time()
for it in range(400):
    r = b - (G[:, S] @ x if S else 0)
    w = G.T @ r
    if S: w[S] = -np.inf
    j = int(np.argmax(w))
    if w[j] <= 0: print('kkt'); break
    S.append(j); x = np.append(x, 0.)
    while True:
        zz = np.linalg.lstsq(G[:, S], b, rcond=None)[0]
        if np.all(zz > 0): x = zz; break
        neg = zz <= 0
        al = np.where(neg, x / (x - zz), np.inf); a = min(1., al.min())
        x = x + a * (zz - x)
        keep = x > 1e-14 * np.max(np.abs(x))
        dropped = [S[i] for i in range(len(S)) if not keep[i]]
        drops_new += int(j in dropped); drops_old += len(dropped) - int(j in dropped)
        S = [S[i] for i in range(len(S)) if keep[i]]; x = x[keep]
    if it % 50 == 0: print(it, len(S), np.linalg.norm(G[:, S] @ x - b) / np.linalg.norm(b), drops_new, drops_old, '%.1fs' % (time.time() - t), flush=True)
