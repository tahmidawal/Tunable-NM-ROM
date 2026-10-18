"""analyze_a.py -- artifact-only diagnostics (no fields): latent-code structure, oracle codes vs
training codes, head Jacobian rank, coefficient-vector spectra. CPU numpy."""
import json, sys, numpy as np
from common import *
sys.path.insert(0, LANE)
import importlib.util
spec = importlib.util.spec_from_file_location('fomdraw', f'{LANE}/ns2d_fom.py')
# params_draw is pure numpy; import only what we need without triggering jax
def params_draw(seed, count, nmodes=6):
    r = np.random.default_rng(seed)
    cols = [r.standard_normal(count) for _ in range(2 * nmodes)]
    cols.append(np.exp(r.uniform(np.log(1e-3), np.log(1e-2), count)))
    return np.stack(cols, axis=1)

out = {}
for run, nm in [('ns203', 6), ('ns204', 6), ('ns303', 3), ('ns302', 6)]:
    p, Ztr, cfg = load_ckpt(run)
    K, R = cfg['K'], cfg['R']
    ntraj = Ztr.shape[0] // NOUT
    N = 64
    G = bank_on_grid(p, N)
    Q, Rb = whiten(G)
    orc = load_oracle(run, 256 if run in ('ns302', 'ns303') else 64)
    Zo, Z1 = orc['Z'], orc['Z1']
    res = dict(K=K, R=R, ntraj=ntraj)
    # ---- training-code geometry
    Zt = Ztr.reshape(ntraj, NOUT, K)
    sv = np.linalg.svd(Ztr - Ztr.mean(0), compute_uv=False)
    pr = (sv ** 2).sum() ** 2 / (sv ** 4).sum()
    res['Ztr_sv_norm'] = (sv / sv[0]).round(4).tolist()
    res['Ztr_participation_ratio'] = float(pr)
    res['Ztr_std_per_dim'] = Ztr.std(0).round(3).tolist()
    step = np.linalg.norm(np.diff(Zt, axis=1), axis=2)            # (ntraj, 25)
    # NN distance of each snapshot to codes of OTHER trajectories (subsample for cost)
    rng = np.random.default_rng(0)
    pick = rng.choice(Ztr.shape[0], 2000, replace=False)
    traj_of = np.arange(Ztr.shape[0]) // NOUT
    d = np.linalg.norm(Ztr[pick][:, None, :] - Ztr[None, :, :], axis=2)
    d[traj_of[pick][:, None] == traj_of[None, :]] = np.inf
    nn_other = d.min(1)
    res['code_step_median'] = float(np.median(step))
    res['code_step_t0_median'] = float(np.median(step[:, 0]))
    res['code_nn_other_traj_median'] = float(np.median(nn_other))
    res['code_traj_span_median'] = float(np.median(np.linalg.norm(Zt[:, -1] - Zt[:, 0], axis=1)))
    res['code_norm_median'] = float(np.median(np.linalg.norm(Ztr, axis=1)))
    # ---- are the training codes a smooth function of (params, t)?  ridge regression, CV by trajectory
    phys = params_draw(20260917, 512, nm)
    if run == 'ns302':
        phys = np.concatenate([phys, params_draw(20260920, 1536, nm)])
    P = state_params(phys, range(ntraj), range(NOUT))
    Pn = (P - P.mean(0)) / P.std(0)
    folds = np.arange(ntraj) % 5
    for deg in (1, 2, 3):
        X = poly_features(Pn, deg)
        sse = 0.0
        for f in range(5):
            tr = np.repeat(folds != f, NOUT); te = ~tr
            W = ridge_fit(X[tr], Ztr[tr], 1e-6)
            sse += ((X[te] @ W - Ztr[te]) ** 2).sum()
        res[f'R2_params_to_codes_deg{deg}_cv'] = float(1 - sse / ((Ztr - Ztr.mean(0)) ** 2).sum())
    # inverse: are the params a smooth function of the codes? (chart consistency) -- deg 2 in z
    for deg in (1, 2):
        X = poly_features((Ztr - Ztr.mean(0)) / Ztr.std(0), deg)
        sse = 0.0; tot = ((Pn - Pn.mean(0)) ** 2).sum()
        for f in range(5):
            tr = np.repeat(folds != f, NOUT); te = ~tr
            W = ridge_fit(X[tr], Pn[tr], 1e-6)
            sse += ((X[te] @ W - Pn[te]) ** 2).sum()
        res[f'R2_codes_to_params_deg{deg}_cv'] = float(1 - sse / tot)
    # ---- oracle codes vs the training cloud
    mu, C = Ztr.mean(0), np.cov(Ztr.T)
    Ci = np.linalg.inv(C)
    mah_tr = np.sqrt(np.einsum('ij,jk,ik->i', Ztr - mu, Ci, Ztr - mu))
    mah_o = np.sqrt(np.einsum('ij,jk,ik->i', Zo - mu, Ci, Zo - mu))
    res['mahalanobis_train_q50_q95_q99'] = np.percentile(mah_tr, [50, 95, 99]).round(2).tolist()
    res['mahalanobis_oracle_q50_q95_max'] = np.percentile(mah_o, [50, 95, 100]).round(2).tolist()
    res['oracle_frac_outside_train_q99'] = float(np.mean(mah_o > np.percentile(mah_tr, 99)))
    # per-PC range check
    U_, s_, Vt = np.linalg.svd(Ztr - mu, full_matrices=False)
    ptr, po = (Ztr - mu) @ Vt.T, (Zo - mu) @ Vt.T
    lo, hi = ptr.min(0), ptr.max(0)
    res['oracle_frac_outside_train_box'] = float(np.mean(((po < lo) | (po > hi)).any(1)))
    dn = np.linalg.norm(Zo[:, None, :] - Ztr[None, :, :], axis=2).min(1)
    res['oracle_nn_train_code_median'] = float(np.median(dn))
    res['oracle_nn_over_train_nn_other'] = float(np.median(dn) / np.median(nn_other))
    # multi-start vs single-start
    dz = np.linalg.norm(Zo - Z1, axis=1)
    res['frac_8start_differs_from_1start(>0.1 code norm)'] = float(np.mean(dz > 0.1 * np.median(np.linalg.norm(Ztr, axis=1))))
    res['median_e_single_over_e_oracle'] = float(np.median(orc['e_single'] / orc['e_oracle']))
    # ---- head Jacobian rank at oracle codes (whitened metric): local manifold dimension
    dims = {1e-1: [], 1e-2: [], 1e-3: []}
    svs = []
    for z in Zo[::4]:
        J = Rb @ head_jac(p, z)
        s = np.linalg.svd(J, compute_uv=False)
        svs.append(s / s[0])
        for th in dims:
            dims[th].append(int(np.sum(s > th * s[0])))
    svs = np.array(svs)
    res['jac_sv_median_normalised'] = np.median(svs, 0).round(4).tolist()
    res['jac_local_dim_median'] = {str(th): float(np.median(v)) for th, v in dims.items()}
    # same at training codes
    dims_tr = {1e-1: [], 1e-2: [], 1e-3: []}
    for z in Ztr[rng.choice(Ztr.shape[0], 96, replace=False)]:
        s = np.linalg.svd(Rb @ head_jac(p, z), compute_uv=False)
        for th in dims_tr:
            dims_tr[th].append(int(np.sum(s > th * s[0])))
    res['jac_local_dim_median_train'] = {str(th): float(np.median(v)) for th, v in dims_tr.items()}
    # ---- spectra of coefficient vectors (whitened): head image at oracle codes, and at training codes
    Ho = head(p, Zo) @ Rb.T
    Ht = head(p, Ztr[::4]) @ Rb.T
    for name, H in [('oracle_img', Ho), ('train_img', Ht)]:
        s = np.linalg.svd(H - H.mean(0), compute_uv=False)
        e = np.cumsum(s ** 2) / (s ** 2).sum()
        res[f'{name}_dims_for_90_99_999_energy'] = [int(np.searchsorted(e, q) + 1) for q in (0.9, 0.99, 0.999)]
    # ---- per-time medians (archived errors)
    eo, ep, eb = orc['e_oracle'].reshape(-1, 6), orc['e_podK'].reshape(-1, 6), orc['e_bank'].reshape(-1, 6)
    res['per_time_oracle_median'] = np.median(eo, 0).round(4).tolist()
    res['per_time_podK_median'] = np.median(ep, 0).round(4).tolist()
    res['per_time_bank_median'] = np.median(eb, 0).round(4).tolist()
    res['per_time_ratio'] = (np.median(ep, 0) / np.median(eo, 0)).round(3).tolist()
    # correlation of the oracle error with nu and with the POD error across cases
    cases = orc['cases']
    physd = params_draw(20260918, 64, nm)
    lognu = np.log10(physd[cases, -1])
    ecase = eo[:, 1:].max(1)
    res['corr_log_nu_vs_case_worst_oracle'] = float(np.corrcoef(lognu, ecase)[0, 1])
    res['corr_podK_vs_oracle_per_state'] = float(np.corrcoef(orc['e_podK'], orc['e_oracle'])[0, 1])
    out[run] = res
    print(run, json.dumps(res, indent=1))
json.dump(out, open('analyze_a.json', 'w'), indent=1)
