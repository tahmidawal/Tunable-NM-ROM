"""analyze_b.py -- diagnostics that need fields (64^2 regenerated cohorts, data64.npz):
sanity of the numpy decoder vs archived N64 oracle numbers; data coverage (NN distances);
residual smoothness in the parameters; parameter-conditioned regression baselines (the proxy for a
parameter-conditioned head); quadratic/cubic manifold on POD coordinates (proxy for the recommended
design) with linear-encoder and best-fit (LM) errors. CPU numpy."""
import json, sys, time, numpy as np
from common import *

def params_draw(seed, count, nmodes=6):
    r = np.random.default_rng(seed)
    cols = [r.standard_normal(count) for _ in range(2 * nmodes)]
    cols.append(np.exp(r.uniform(np.log(1e-3), np.log(1e-2), count)))
    return np.stack(cols, axis=1)

MODES = np.array([(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)], float)
def feats_phys(phys, cases, times, nout=NOUT):
    """[a/urms (6), b/urms (6), log10 nu, t] : the IC is exactly linear in the first 12."""
    k2 = (MODES ** 2).sum(1)
    rows = []
    for c in cases:
        a, b = phys[c, :6], phys[c, 6:12]
        urms = np.sqrt(np.sum((a * a + b * b) / (8 * np.pi ** 2 * k2)))
        for t in times:
            rows.append(np.concatenate([a / urms, b / urms, [np.log10(phys[c, -1]), t / (nout - 1)]]))
    return np.asarray(rows)

d = np.load('data64.npz')
Udev, Utr, phys_dev, phys_tr = d['U_dev'], d['U_tr'], d['phys_dev'], d['phys_train']
ntr = Utr.shape[0]
N = 64
print('data', Udev.shape, Utr.shape, flush=True)
cases = np.arange(64)
Xdev = np.stack([Udev[c, t] for c in cases for t in TIMES])       # (384, n) same order as the archived oracle
n0 = np.repeat(np.linalg.norm(Udev[:, 0], axis=1), 6)
Utr_flat = Utr.reshape(-1, N * N)
tidx_tr = np.tile(np.arange(NOUT), ntr)
out = {'ntraj_train_regenerated': int(ntr)}

# ---------------------------------------------------------------- data coverage (field metric)
t0 = time.time()
nn_all, nn_same_t = [], []
for i, x in enumerate(Xdev):
    dd = np.linalg.norm(Utr_flat - x, axis=1)
    nn_all.append(dd.min() / n0[i])
    m = tidx_tr == TIMES[i % 6]
    nn_same_t.append(dd[m].min() / n0[i])
nn_all, nn_same_t = np.array(nn_all), np.array(nn_same_t)
# NN in parameter space (same time) -> field distance
Pd, Pt = feats_phys(phys_dev, cases, [0])[:, :-1], feats_phys(phys_tr, range(ntr), [0])[:, :-1]
Pd_n, Pt_n = (Pd - Pt.mean(0)) / Pt.std(0), (Pt - Pt.mean(0)) / Pt.std(0)
nnp = np.argmin(np.linalg.norm(Pd_n[:, None] - Pt_n[None], axis=2), 1)
nn_param = np.array([np.linalg.norm(Utr[nnp[i // 6], TIMES[i % 6]] - Xdev[i]) / n0[i] for i in range(384)])
out['coverage'] = dict(nn_field_any_time_median=float(np.median(nn_all)),
                       nn_field_same_time_median=float(np.median(nn_same_t)),
                       nn_field_same_time_per_time_median=np.median(nn_same_t.reshape(-1, 6), 0).round(4).tolist(),
                       nn_param_neighbour_same_time_median=float(np.median(nn_param)),
                       nn_param_neighbour_per_time_median=np.median(nn_param.reshape(-1, 6), 0).round(4).tolist(),
                       seconds=time.time() - t0)
print('coverage', out['coverage'], flush=True)

# ---------------------------------------------------------------- POD of the regenerated training snapshots
t0 = time.time()
Um = Utr_flat
U_, s_, Vt = np.linalg.svd(Um, full_matrices=False)
V = Vt.T                                                             # (n, min)
out['pod_energy_16_32_256'] = [float((s_[:k] ** 2).sum() / (s_ ** 2).sum()) for k in (16, 32, 256)]
def pod_err(K):
    Vk = V[:, :K]
    return np.linalg.norm(Xdev - (Xdev @ Vk) @ Vk.T, axis=1) / n0
e_pod = {K: pod_err(K) for K in (16, 32, 64, 256)}
out['pod_dev_median'] = {str(K): float(np.median(v)) for K, v in e_pod.items()}
print('pod', out['pod_dev_median'], time.time() - t0, flush=True)

# ---------------------------------------------------------------- per-run: sanity, residual smoothness
for run in ('ns203', 'ns204'):
    p, Ztr, cfg = load_ckpt(run)
    K, R = cfg['K'], cfg['R']
    G = bank_on_grid(p, N)
    Q, Rb = whiten(G)
    orc = load_oracle(run, 64)
    Zo = orc['Z']
    Adev, perp = coefs(Q, Rb, Xdev)
    e_bank = perp / n0
    Ho = head(p, Zo)
    e_or = np.linalg.norm(Ho @ G.T - Xdev, axis=1) / n0
    res = dict(K=K, R=R,
               sanity_e_bank_vs_archived_max_rel=float(np.max(np.abs(e_bank - orc['e_bank']) / orc['e_bank'])),
               sanity_e_oracle_vs_archived_max_rel=float(np.max(np.abs(e_or - orc['e_oracle']) / orc['e_oracle'])),
               oracle_median=float(np.median(e_or)), podK_median_archived=float(np.median(orc['e_podK'])))
    # whitened residual of the best fit
    Rw = Adev - Ho @ Rb.T                                            # (384, R)
    res['residual_over_oracle_err_check'] = float(np.median(np.sqrt((Rw ** 2).sum(1) + perp ** 2) / n0 / e_or))
    # (i) how much of the residual lies in the span of the training-image directions vs. orthogonal
    Ht = head(p, Ztr[::2]) @ Rb.T
    Uh, sh, Vh = np.linalg.svd(Ht - Ht.mean(0), full_matrices=False)
    for kk in (16, 32, 64, 128):
        proj = Rw @ Vh[:kk].T
        res[f'residual_energy_frac_in_top{kk}_train_image_dirs'] = float((proj ** 2).sum() / (Rw ** 2).sum())
    # (ii) is the residual a smooth function of (params, t)?  CV ridge over cases
    F = feats_phys(phys_dev, cases, TIMES)
    Fn = (F - F.mean(0)) / F.std(0)
    folds = np.repeat(np.arange(64) % 8, 6)
    for deg in (1, 2):
        X = poly_features(Fn, deg)
        sse = 0.0
        for f in range(8):
            tr, te = folds != f, folds == f
            W = ridge_fit(X[tr], Rw[tr], 1e-3)
            sse += ((X[te] @ W - Rw[te]) ** 2).sum()
        res[f'R2_residual_vs_params_deg{deg}_cv8'] = float(1 - sse / ((Rw - Rw.mean(0)) ** 2).sum())
    # residual vs POD-K per state: does the head fail where POD fails?
    res['corr_residual_norm_vs_podK'] = float(np.corrcoef(np.linalg.norm(Rw, axis=1) / n0, orc['e_podK'])[0, 1])
    res['corr_residual_norm_vs_nn_same_t'] = float(np.corrcoef(np.linalg.norm(Rw, axis=1) / n0, nn_same_t)[0, 1])
    # (iii) linear-in-z best fit: least squares over the head's LINEAR skip only (a 16-dim linear subspace)
    Wl = p['h_lin'] @ Rb.T                                           # (K, R) whitened
    Qs, _ = np.linalg.qr(Wl.T)
    e_lin = np.sqrt(np.linalg.norm(Adev - (Adev @ Qs) @ Qs.T, axis=1) ** 2 + perp ** 2) / n0
    res['linear_skip_subspace_only_median'] = float(np.median(e_lin))
    # (iv) optimal K-dim linear subspace of the bank coefficients (POD inside the bank span) for reference
    Atr, _ = coefs(Q, Rb, Utr_flat[::3])
    Ua, sa, Va = np.linalg.svd(Atr - Atr.mean(0), full_matrices=False)
    m = Atr.mean(0)
    for kk in (K,):
        Pk = Va[:kk].T
        e = np.sqrt(np.linalg.norm((Adev - m) - ((Adev - m) @ Pk) @ Pk.T, axis=1) ** 2 + perp ** 2) / n0
        res[f'pod{kk}_inside_bank_span_median'] = float(np.median(e))
    out[run] = res
    print(run, json.dumps(res, indent=1), flush=True)

# ---------------------------------------------------------------- parameter-conditioned regression baselines
# target: whitened coefficients in the ns203 bank (R=256), which carries the bank floor (1.2 %).
p, Ztr, cfg = load_ckpt('ns203')
G = bank_on_grid(p, N); Q, Rb = whiten(G)
Adev, perp = coefs(Q, Rb, Xdev)
Atr, perp_tr = coefs(Q, Rb, Utr_flat)
Ftr = feats_phys(phys_tr, range(ntr), range(NOUT))
Fdev = feats_phys(phys_dev, cases, TIMES)
mu_f, sd_f = Ftr.mean(0), Ftr.std(0)
Ftr_n, Fdev_n = (Ftr - mu_f) / sd_f, (Fdev - mu_f) / sd_f
sel = np.repeat(cases >= 32, 6); rep = ~sel
def field_err(Apred):
    return np.sqrt(np.linalg.norm(Apred - Adev, axis=1) ** 2 + perp ** 2) / n0
def report(e, tag):
    return dict(tag=tag, median_all=float(np.median(e)), median_report=float(np.median(e[rep])),
                median_select=float(np.median(e[sel])), per_time_median=np.median(e.reshape(-1, 6), 0).round(4).tolist())
preg = {}
t0 = time.time()
for deg in (1, 2, 3):
    X, Xd = poly_features(Ftr_n, deg), poly_features(Fdev_n, deg)
    best = None
    for lam in (1e-8, 1e-6, 1e-4, 1e-2):
        W = ridge_fit(X, Atr, lam)
        e = field_err(Xd @ W)
        r = report(e, f'poly{deg}_lam{lam:g}')
        if best is None or r['median_select'] < best['median_select']:
            best = r
    preg[f'poly_deg{deg}'] = best
    print('poly', deg, best, time.time() - t0, flush=True)
# RBF kernel ridge from (params, t)
def krr(Xtr, Ytr, Xte, gamma, lam):
    K_ = np.exp(-gamma * ((Xtr[:, None, :] - Xtr[None, :, :]) ** 2).sum(2))
    Kte = np.exp(-gamma * ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(2))
    alpha = np.linalg.solve(K_ + lam * np.eye(len(Xtr)), Ytr)
    return Kte @ alpha
sub = np.arange(0, ntr * NOUT, 2)          # every other snapshot keeps the kernel matrix at ~6k
best = None
for gamma in (0.02, 0.05, 0.1, 0.2):
    for lam in (1e-6, 1e-4, 1e-2):
        e = field_err(krr(Ftr_n[sub], Atr[sub], Fdev_n, gamma, lam))
        r = report(e, f'krr_g{gamma}_lam{lam:g}')
        if best is None or r['median_select'] < best['median_select']:
            best = r
preg['krr_rbf'] = best
print('krr', best, time.time() - t0, flush=True)
# learning curve of the best polynomial degree and of krr vs number of training trajectories
lc = {}
for n_use in (64, 128, 256, ntr):
    m = np.arange(ntr * NOUT) < n_use * NOUT
    X, Xd = poly_features(Ftr_n, 2), poly_features(Fdev_n, 2)
    e2 = field_err(Xd @ ridge_fit(X[m], Atr[m], 1e-6))
    X, Xd = poly_features(Ftr_n, 3), poly_features(Fdev_n, 3)
    e3 = field_err(Xd @ ridge_fit(X[m], Atr[m], 1e-4))
    idx = np.nonzero(m)[0][::2]
    ek = field_err(krr(Ftr_n[idx], Atr[idx], Fdev_n, 0.1, 1e-4))
    lc[str(n_use)] = dict(poly2=float(np.median(e2[rep])), poly3=float(np.median(e3[rep])), krr=float(np.median(ek[rep])))
preg['learning_curve_report_median'] = lc
out['param_regression'] = preg
print('learning curve', lc, flush=True)

# ---------------------------------------------------------------- quadratic / cubic manifold on POD coordinates
qm = {}
for K in (16, 32):
    Vk = V[:, :K]
    Ztr_pod = Utr_flat @ Vk                                           # linear encoder on training
    Rtr = Utr_flat - Ztr_pod @ Vk.T                                   # residual to be modelled
    zs = Ztr_pod.std(0)
    Zdev_pod = Xdev @ Vk
    for deg in (2, 3):
        if deg == 3 and K == 32:
            continue                                                  # 6545 features x 12k rows: skip
        Xf = poly_features(Ztr_pod / zs, deg)
        W = ridge_fit(Xf, Rtr, 1e-6)
        Xd = poly_features(Zdev_pod / zs, deg)
        e_enc = np.linalg.norm(Xdev - Zdev_pod @ Vk.T - Xd @ W, axis=1) / n0
        e_tr = np.linalg.norm(Rtr - Xf @ W, axis=1) / np.linalg.norm(Utr_flat, axis=1)
        r = dict(K=K, deg=deg, n_features=int(Xf.shape[1]), train_recon_median=float(np.median(e_tr)),
                 dev_linear_encoder_median=float(np.median(e_enc)),
                 dev_linear_encoder_per_time=np.median(e_enc.reshape(-1, 6), 0).round(4).tolist(),
                 podK_median=float(np.median(e_pod[K])), ratio_podK_over_encoder=float(np.median(e_pod[K]) / np.median(e_enc)))
        # best-fit (oracle) z by Gauss-Newton/LM from the linear-encoder start, on every 4th state to bound cost
        if deg == 2:
            def decode(z):
                return z @ Vk.T + poly_features((z / zs)[None], 2)[0] @ W
            def jac(z):
                # d/dz of quadratic features: numerical is cheap enough (K columns)
                eps = 1e-6
                base = decode(z)
                J = np.empty((N * N, K))
                for j in range(K):
                    dz = np.zeros(K); dz[j] = eps
                    J[:, j] = (decode(z + dz) - base) / eps
                return J
            e_fit = []
            for i in range(0, 384, 4):
                z = Zdev_pod[i].copy(); u = Xdev[i]
                r_ = decode(z) - u; val = np.linalg.norm(r_); lam = 1e-6
                for it in range(40):
                    J = jac(z); H = J.T @ J; g = J.T @ r_
                    dz = np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-30 * np.eye(K), -g)
                    r2 = decode(z + dz) - u; v2 = np.linalg.norm(r2)
                    if v2 < val:
                        z, r_, val, lam = z + dz, r2, v2, max(lam / 3, 1e-12)
                        if np.linalg.norm(dz) < 1e-10 * (1 + np.linalg.norm(z)):
                            break
                    else:
                        lam = min(lam * 10, 1e12)
                        if lam >= 1e12:
                            break
                e_fit.append(val / n0[i])
            e_fit = np.array(e_fit)
            r['dev_bestfit_median_every4th'] = float(np.median(e_fit))
            r['podK_median_every4th'] = float(np.median(e_pod[K][::4]))
            r['ratio_podK_over_bestfit'] = float(np.median(e_pod[K][::4]) / np.median(e_fit))
        qm[f'K{K}_deg{deg}'] = r
        print('quadratic manifold', r, flush=True)
out['quadratic_manifold'] = qm
json.dump(out, open('analyze_b.json', 'w'), indent=1)
print('DONE')
