"""analyze_burgers.py -- the SAME diagnostics on the working Burgers case (b-seeds seed1 head, stage-C
codes and the extraction npz): chart smoothness R^2(params,t -> codes), code geometry, data coverage in
the whitened bank metric, parameter-conditioned regression baseline, best-K linear subspace inside the
bank, quadratic manifold on the linear-K coordinates, and the head oracle on the 408 test states."""
import json, time, numpy as np, pickle
from common import silu, mlp, head_jac, poly_features, ridge_fit
B = '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-seeds/experiments/b-seeds'
d = np.load(f'{B}/runs/s1/archive/output/train/sep_coeff_N256_K16_R512.npz')
ck = pickle.load(open(f'{B}/checkpoints/sep_hfit_seed1.pkl', 'rb'))
p = {k: (np.asarray(v, dtype=np.float64) if not isinstance(v, list) else [(np.asarray(w), np.asarray(b)) for w, b in v]) for k, v in ck['params'].items()}
Ztr = np.asarray(ck['Z_tr']); K = Ztr.shape[1]
Gram = d['Gram']; R = Gram.shape[0]
L = np.linalg.cholesky(Gram + 1e-12 * np.trace(Gram) / R * np.eye(R))
A_tr = d['C_tr'] @ L; A_te = d['C_te'] @ L
un_tr, fl_tr = np.sqrt(d['un2_tr']), np.sqrt(d['fl2_tr'])
un_te, fl_te = np.sqrt(d['un2_te']), np.sqrt(d['fl2_te'])
mu_tr, t_tr, traj_tr = d['mu_tr'], d['t_tr'], d['traj_tr']
mu_te, t_te, traj_te = d['mu_te'], d['t_te'], d['traj_te']
T = 51
out = dict(K=K, R=R, n_train_states=int(A_tr.shape[0]), n_traj=int(len(np.unique(traj_tr))), n_test_states=int(A_te.shape[0]))
# n0 = ||u(0)|| of the test trajectory (Burgers convention normalises by the initial state)
n0_of = {int(tr): float(un_te[(traj_te == tr) & (t_te == 0)][0]) for tr in np.unique(traj_te)}
n0 = np.array([n0_of[int(tr)] for tr in traj_te])
def head(z):
    return mlp(p['h'], z) + z @ p['h_lin']
def field_err(Apred, A, fl, nn):
    return np.sqrt(np.linalg.norm(Apred - A, axis=1) ** 2 + fl ** 2) / nn
out['bank_floor_test_median_n0'] = float(np.median(fl_te / n0))
out['bank_floor_test_median_own'] = float(np.median(fl_te / un_te))
# ---- training recon of the head with its own codes
rec = field_err(head(Ztr[::8]) @ L, A_tr[::8], fl_tr[::8], un_tr[::8])
out['train_recon_median_own_norm'] = float(np.median(rec))
# ---- chart smoothness: (mu, t) -> codes, CV by trajectory
P = np.column_stack([mu_tr[:, :4], np.log(mu_tr[:, 4]), t_tr / (T - 1)])
Pn = (P - P.mean(0)) / P.std(0)
rng = np.random.default_rng(0)
sub = rng.choice(A_tr.shape[0], 30000, replace=False)
folds = traj_tr[sub] % 5
for deg in (1, 2, 3):
    X = poly_features(Pn[sub], deg); Y = Ztr[sub]
    sse = 0.0
    for f in range(5):
        tr, te = folds != f, folds == f
        W = ridge_fit(X[tr], Y[tr], 1e-6)
        sse += ((X[te] @ W - Y[te]) ** 2).sum()
    out[f'R2_params_to_codes_deg{deg}_cv'] = float(1 - sse / ((Y - Y.mean(0)) ** 2).sum())
for deg in (1, 2):
    X = poly_features((Ztr[sub] - Ztr.mean(0)) / Ztr.std(0), deg); Y = Pn[sub]
    sse = 0.0
    for f in range(5):
        tr, te = folds != f, folds == f
        W = ridge_fit(X[tr], Y[tr], 1e-6)
        sse += ((X[te] @ W - Y[te]) ** 2).sum()
    out[f'R2_codes_to_params_deg{deg}_cv'] = float(1 - sse / ((Y - Y.mean(0)) ** 2).sum())
sv = np.linalg.svd(Ztr[sub] - Ztr.mean(0), compute_uv=False)
out['Ztr_sv_norm'] = (sv / sv[0]).round(4).tolist()
out['Ztr_participation_ratio'] = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
# code geometry: consecutive-time step within trajectory vs NN distance to other trajectories
order = np.lexsort((t_tr, traj_tr))
Zs, ts, trs = Ztr[order], t_tr[order], traj_tr[order]
same = (trs[1:] == trs[:-1]) & (ts[1:] == ts[:-1] + 1)
step = np.linalg.norm(Zs[1:] - Zs[:-1], axis=1)[same]
pick = rng.choice(A_tr.shape[0], 1500, replace=False)
dz = np.linalg.norm(Ztr[pick][:, None, :] - Ztr[None, ::4, :], axis=2)
dz[traj_tr[pick][:, None] == traj_tr[None, ::4]] = np.inf
out['code_step_median'] = float(np.median(step)); out['code_nn_other_traj_median'] = float(np.median(dz.min(1)))
out['code_norm_median'] = float(np.median(np.linalg.norm(Ztr, axis=1)))
# ---- coverage: NN distance in the whitened metric from each test state to training states (same time / any time)
t0 = time.time()
nn_any, nn_same = [], []
for i in range(A_te.shape[0]):
    dd = np.linalg.norm(A_tr - A_te[i], axis=1)
    nn_any.append(dd.min() / n0[i]); nn_same.append(dd[t_tr == t_te[i]].min() / n0[i])
nn_any, nn_same = np.array(nn_any), np.array(nn_same)
out['coverage'] = dict(nn_any_time_median=float(np.median(nn_any)), nn_same_time_median=float(np.median(nn_same)),
                       nn_same_time_t0_median=float(np.median(nn_same[t_te == 0])), nn_same_time_tlate_median=float(np.median(nn_same[t_te >= 25])),
                       seconds=time.time() - t0)
print('coverage', out['coverage'], flush=True)
# ---- best K-dim linear subspace inside the bank (POD-K analogue) on test states
m = A_tr[::4].mean(0)
Ua, sa, Va = np.linalg.svd(A_tr[::4] - m, full_matrices=False)
lin = {}
for kk in (16, 32, 64):
    Pk = Va[:kk].T
    e = np.sqrt(np.linalg.norm((A_te - m) - ((A_te - m) @ Pk) @ Pk.T, axis=1) ** 2 + fl_te ** 2) / n0
    lin[str(kk)] = float(np.median(e))
out['podK_inside_bank_test_median'] = lin
# ---- head oracle on the test states (LM in whitened space, 8 nearest training codes as starts)
def lm(z0, a, iters=60):
    z = z0.copy(); r = head(z[None])[0] @ L - a; val = np.linalg.norm(r); lam = 1e-6
    for _ in range(iters):
        J = L.T @ head_jac(p, z)
        H = J.T @ J; g = J.T @ r
        dz = np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-30 * np.eye(K), -g)
        r2 = head(z[None] + dz[None])[0] @ L - a; v2 = np.linalg.norm(r2)
        if np.isfinite(v2) and v2 < val:
            z, r, val, lam = z + dz, r2, v2, max(lam / 3, 1e-12)
            if np.linalg.norm(dz) < 1e-12 * (1 + np.linalg.norm(z)):
                break
        else:
            lam = min(lam * 10, 1e12)
            if lam >= 1e12:
                break
    return z, val, r
Zc = Ztr[::16]; Hc = head(Zc) @ L
e_or, Zo, Rw = [], [], []
t0 = time.time()
TE = np.arange(0, A_te.shape[0], 4)
for i in TE:
    sc_ = ((Hc - A_te[i]) ** 2).sum(1)
    best = None
    for z0 in Zc[np.argsort(sc_)[:1]]:
        z, v, r = lm(z0, A_te[i])
        if best is None or v < best[1]:
            best = (z, v, r)
    Zo.append(best[0]); Rw.append(best[2]); e_or.append(np.sqrt(best[1] ** 2 + fl_te[i] ** 2) / n0[i])
e_or, Zo, Rw = np.array(e_or), np.array(Zo), np.array(Rw)
t_te_full, traj_te_full = t_te, traj_te
t_te, traj_te = t_te[TE], traj_te[TE]
out['head_oracle_test_median_n0'] = float(np.median(e_or)); out['head_oracle_t0_median'] = float(np.median(e_or[t_te == 0])); out['oracle_states'] = int(len(TE)); out['oracle_protocol'] = '1 nearest-code start, 60 LM iterations, every 4th test state (light)'
out['head_oracle_tlate_median'] = float(np.median(e_or[t_te >= 25]))
out['ratio_lin16_over_oracle'] = lin['16'] / float(np.median(e_or))
out['oracle_seconds'] = time.time() - t0
print('oracle', out['head_oracle_test_median_n0'], out['ratio_lin16_over_oracle'], flush=True)
# oracle codes vs training cloud
mu_, C = Ztr.mean(0), np.cov(Ztr[sub].T); Ci = np.linalg.inv(C)
mah_tr = np.sqrt(np.einsum('ij,jk,ik->i', Ztr[sub] - mu_, Ci, Ztr[sub] - mu_)); mah_o = np.sqrt(np.einsum('ij,jk,ik->i', Zo - mu_, Ci, Zo - mu_))
out['mahalanobis_train_q50_q95_q99'] = np.percentile(mah_tr, [50, 95, 99]).round(2).tolist()
out['mahalanobis_oracle_q50_q95_max'] = np.percentile(mah_o, [50, 95, 100]).round(2).tolist()
dn = np.linalg.norm(Zo[:, None, :] - Ztr[None, ::4, :], axis=2).min(1)
out['oracle_nn_train_code_median'] = float(np.median(dn)); out['oracle_nn_over_train_nn_other'] = float(np.median(dn) / out['code_nn_other_traj_median'])
# residual smoothness in params (CV over test trajectories, 8 folds)
Pte = np.column_stack([mu_te[:, :4], np.log(mu_te[:, 4]), t_te_full / (T - 1)]); Pte_n = (Pte - P.mean(0)) / P.std(0)
Pte_o = Pte_n[TE]
folds = traj_te % 8
for deg in (1, 2):
    X = poly_features(Pte_o, deg); sse = 0.0
    for f in range(8):
        tr, te = folds != f, folds == f
        W = ridge_fit(X[tr], Rw[tr], 1e-3); sse += ((X[te] @ W - Rw[te]) ** 2).sum()
    out[f'R2_residual_vs_params_deg{deg}_cv8'] = float(1 - sse / ((Rw - Rw.mean(0)) ** 2).sum())
# jacobian local dimension at oracle codes
dims = {th: [] for th in (0.1, 0.01)}
for z in Zo[::4]:
    s = np.linalg.svd(L.T @ head_jac(p, z), compute_uv=False)
    for th in dims: dims[th].append(int(np.sum(s > th * s[0])))
out['jac_local_dim_median'] = {str(th): float(np.median(v)) for th, v in dims.items()}
t_te, traj_te = t_te_full, traj_te_full
# ---- parameter-conditioned regression baseline: (mu, t) -> whitened coefficients
preg = {}
for deg in (1, 2, 3):
    X = poly_features(Pn, deg); Xd = poly_features(Pte_n, deg)
    best = None
    for lam in (1e-8, 1e-6, 1e-4):
        W = ridge_fit(X, A_tr, lam); e = field_err(Xd @ W, A_te, fl_te, n0)
        if best is None or np.median(e) < best[0]:
            best = (float(np.median(e)), lam, float(np.median(e[t_te == 0])), float(np.median(e[t_te >= 25])))
    preg[f'poly_deg{deg}'] = dict(median=best[0], lam=best[1], t0=best[2], tlate=best[3])
def krr(Xtr, Ytr, Xte, gamma, lam):
    K_ = np.exp(-gamma * ((Xtr[:, None, :] - Xtr[None, :, :]) ** 2).sum(2)); Kte = np.exp(-gamma * ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(2))
    return Kte @ np.linalg.solve(K_ + lam * np.eye(len(Xtr)), Ytr)
sub2 = rng.choice(A_tr.shape[0], 3000, replace=False)
best = None
for gamma in (0.05, 0.2, 0.5, 1.0):
    for lam in (1e-6, 1e-4, 1e-2):
        e = field_err(krr(Pn[sub2], A_tr[sub2], Pte_n, gamma, lam), A_te, fl_te, n0)
        if best is None or np.median(e) < best[0]:
            best = (float(np.median(e)), gamma, lam)
preg['krr_rbf_6000_states'] = dict(median=best[0], gamma=best[1], lam=best[2])
# learning curve of the regression by number of trajectories (first n of the canonical 576, all times)
lc = {}
for n_use in (72, 144, 288, 576):
    m_ = traj_tr < n_use
    X = poly_features(Pn[m_], 3); Xd = poly_features(Pte_n, 3)
    e3 = field_err(Xd @ ridge_fit(X, A_tr[m_], 1e-6), A_te, fl_te, n0)
    idx = np.nonzero(m_)[0]; idx = idx[rng.choice(len(idx), min(3000, len(idx)), replace=False)]
    ek = field_err(krr(Pn[idx], A_tr[idx], Pte_n, best[1], best[2]), A_te, fl_te, n0)
    lc[str(n_use)] = dict(poly3=float(np.median(e3)), krr=float(np.median(ek)), n_states=int(m_.sum()))
preg['learning_curve'] = lc
out['param_regression'] = preg
print('preg', preg, flush=True)
# ---- quadratic manifold on the best-16 linear coordinates inside the bank (linear encoder)
Pk = Va[:16].T
Ztr_l = (A_tr - m) @ Pk; zs = Ztr_l.std(0)
Rtr = (A_tr - m) - Ztr_l @ Pk.T
Xf = poly_features(Ztr_l[::2] / zs, 2); W = ridge_fit(Xf, Rtr[::2], 1e-6)
Zte_l = (A_te - m) @ Pk
pred = m + Zte_l @ Pk.T + poly_features(Zte_l / zs, 2) @ W
e_q = field_err(pred, A_te, fl_te, n0)
out['quadratic_manifold_K16_linear_encoder_median'] = float(np.median(e_q))
Xf3 = poly_features(Ztr_l[::2] / zs, 3); W3 = ridge_fit(Xf3, Rtr[::2], 1e-4)
e_c = field_err(m + Zte_l @ Pk.T + poly_features(Zte_l / zs, 3) @ W3, A_te, fl_te, n0)
out['cubic_manifold_K16_linear_encoder_median'] = float(np.median(e_c))
json.dump(out, open('analyze_burgers2.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
