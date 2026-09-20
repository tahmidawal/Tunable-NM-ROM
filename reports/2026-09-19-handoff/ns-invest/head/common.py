"""common.py -- numpy re-implementation of the ns2d periodic separable decoder (bank + head),
read from the lane's pickled checkpoints. CPU only; no JAX."""
import pickle, numpy as np
LANE = '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-ns2d/experiments/ns2d'
RUNS = {'ns203': f'{LANE}/runs/ns203/archive/experiments/ns2d/output',
        'ns204': f'{LANE}/runs/ns204/archive/experiments/ns2d/output',
        'ns302': f'{LANE}/runs/ns302/archive/experiments/ns2d/output',
        'ns303': f'{LANE}/runs/ns303/archive/experiments/ns2d/output',
        'ns301': f'{LANE}/runs/ns301/archive/experiments/ns2d/output'}
CKPT = {'ns203': 'ckpt_K16_R256.pkl', 'ns204': 'ckpt_K32_R512.pkl', 'ns302': 'ckpt_K16_R256.pkl', 'ns303': 'ckpt_K16_R256.pkl'}
NOUT = 26
TIMES = np.array([0, 5, 10, 15, 20, 25])

def load_ckpt(run):
    ck = pickle.load(open(f'{RUNS[run]}/{CKPT[run]}', 'rb'))
    p = {k: (np.asarray(v, dtype=np.float64) if not isinstance(v, list) else [(np.asarray(w), np.asarray(b)) for w, b in v]) for k, v in ck['params'].items()}
    return p, np.asarray(ck['Z_tr']), ck['cfg']

def load_oracle(run, N=256):
    return dict(np.load(f'{RUNS[run]}/oracle_N{N}.npz'))

def silu(x):
    return x / (1.0 + np.exp(-x))

def dsilu(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))

def mlp(layers, x):
    for w, b in layers[:-1]:
        x = silu(x @ w + b)
    w, b = layers[-1]
    return x @ w + b

def bank_on_grid(p, N):
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing='ij')
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    ang = 2.0 * np.pi * (xy @ p['B'])
    ff = np.concatenate([np.sin(ang), np.cos(ang)], -1)
    G = float(p['out_scale']) * mlp(p['g'], ff)
    return G - G.mean(0, keepdims=True)

def head(p, z):
    return mlp(p['h'], z) + z @ p['h_lin']

def head_jac(p, z):
    """Jacobian dh/dz (R, K) at one code z (K,)."""
    x = z[None, :]
    acts = []
    for w, b in p['h'][:-1]:
        a = x @ w + b
        acts.append(a)
        x = silu(a)
    J = p['h'][-1][0].T                                # (R, hidden)
    for (w, b), a in zip(reversed(p['h'][:-1]), reversed(acts)):
        J = (J * dsilu(a)) @ w.T                       # (R, prev)
    return J + p['h_lin'].T

def whiten(G):
    Q, R = np.linalg.qr(G)
    return Q, R

def coefs(Q, R, U):
    """U (S, n) -> whitened coefficients a = Q^T u (S, R) and perp norms (S,)."""
    A = U @ Q
    perp = np.linalg.norm(U - A @ Q.T, axis=1)
    return A, perp

def poly_features(P, deg):
    """polynomial features up to degree deg of P (S, d) (with bias)."""
    S, d = P.shape
    cols = [np.ones(S)]
    cols += [P[:, i] for i in range(d)]
    if deg >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(P[:, i] * P[:, j])
    if deg >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(P[:, i] * P[:, j] * P[:, k])
    return np.stack(cols, 1)

def ridge_fit(X, Y, lam):
    XtX = X.T @ X
    return np.linalg.solve(XtX + lam * np.trace(XtX) / X.shape[1] * np.eye(X.shape[1]), X.T @ Y)

def state_params(phys, cases, times, nout=NOUT):
    """(params, t) rows for cases x times: [a(6), b(6), log10 nu, t]."""
    rows = []
    for c in cases:
        for t in times:
            rows.append(np.concatenate([phys[c, :-1], [np.log10(phys[c, -1]), t / (nout - 1)]]))
    return np.asarray(rows)
