import numpy as np, scipy.sparse.linalg as spla, time, aniso3d as a3
# manufactured solution check: u = sin(pi x) sin(2 pi y) sin(pi z), constant rotated K
mu = np.array([30., 40., -2., .5, .5, .5]); K = a3.ktensor(mu)
ev = a3.Evaluator3(64); X, Y, Z = np.meshgrid(ev.x, ev.x, ev.x, indexing="ij")
kv = np.array([np.pi, 2*np.pi, np.pi])
uex = np.sin(kv[0]*X)*np.sin(kv[1]*Y)*np.sin(kv[2]*Z)
def fex(x, y, z):
    s = [np.sin(kv[0]*x), np.sin(kv[1]*y), np.sin(kv[2]*z)]; c = [np.cos(kv[0]*x), np.cos(kv[1]*y), np.cos(kv[2]*z)]
    out = 0
    for i in range(3):
        for j in range(3):
            if i == j: out = out + K[i, i]*kv[i]**2*s[0]*s[1]*s[2]
            else:
                t = [s[0], s[1], s[2]]; t[i] = c[i]; t[j] = c[j]
                out = out - K[i, j]*kv[i]*kv[j]*t[0]*t[1]*t[2]
    return out
for p, N in ((1, 8), (1, 16), (2, 4), (2, 8), (4, 2), (4, 4)):
    sp_ = a3.Space(N, p); A = sp_.matrix(mu)
    old = a3.source; a3.source = lambda mu, x, y, z: fex(x[:, None, None], y[None, :, None], z[None, None, :])
    b = sp_.load(mu); a3.source = old
    u = spla.spsolve(A.tocsc(), b)
    print(p, N, sp_.n**3, "nnz", A.nnz, "asym %.1e" % abs(A - A.T).max(), "err", ev.errors(ev.field(sp_, u), uex))
