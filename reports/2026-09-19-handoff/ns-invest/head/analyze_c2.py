"""analyze_c.py -- two structure-exploiting candidates tested on the 64^2 cohorts (numpy):
(1) quadratic manifold on POD-K coordinates whose polynomial ALSO sees (log nu, t);
(2) translation alignment: shift every field so the (1,0) and (0,1) Fourier phases vanish, then POD-K
    and the quadratic manifold on aligned fields (a 'shift + manifold' decoder, 2 extra online unknowns)."""
import json, numpy as np
from common import poly_features, ridge_fit, NOUT, TIMES
d = np.load('data64.npz')
Udev, Utr, phys_dev, phys_tr = d['U_dev'], d['U_tr'], d['phys_dev'], d['phys_train']
N = 64; ntr = Utr.shape[0]
cases = np.arange(64)
Xdev = np.stack([Udev[c, t] for c in cases for t in TIMES]); n0 = np.repeat(np.linalg.norm(Udev[:, 0], axis=1), 6)
Utr_flat = Utr.reshape(-1, N * N)
out = {}
def pod(Utrain):
    _, s, Vt = np.linalg.svd(Utrain, full_matrices=False)
    return Vt.T, s
def nu_t_feats(phys, cases_, times_):
    return np.array([[np.log10(phys[c, -1]), t / (NOUT - 1)] for c in cases_ for t in times_])
F_tr = nu_t_feats(phys_tr, range(ntr), range(NOUT)); F_dev = nu_t_feats(phys_dev, cases, TIMES)
Fm, Fs = F_tr.mean(0), F_tr.std(0)
def quad_with_nut(V, K, Utrain, Xd, tag):
    Vk = V[:, :K]; Z = Utrain @ Vk; zs = Z.std(0); Rr = Utrain - Z @ Vk.T
    Zd = Xd @ Vk
    res = {}
    for name, deg in (('quad_z_nu_t', 2),):
        Xf = poly_features(np.column_stack([Z / zs, (F_tr - Fm) / Fs]), deg)
        Xf_d = poly_features(np.column_stack([Zd / zs, (F_dev - Fm) / Fs]), deg)
        for lam in (1e-6, 1e-3):
            W = ridge_fit(Xf, Rr, lam)
            e = np.linalg.norm(Xd - Zd @ Vk.T - Xf_d @ W, axis=1) / n0
            e_tr = np.linalg.norm(Rr - Xf @ W, axis=1) / np.linalg.norm(Utrain, axis=1)
            res[f'{name}_lam{lam:g}'] = dict(n_features=int(Xf.shape[1]), train_recon_median=float(np.median(e_tr)),
                                              dev_median=float(np.median(e)), per_time=np.median(e.reshape(-1, 6), 0).round(4).tolist())
    e_pod = np.linalg.norm(Xd - Zd @ Vk.T, axis=1) / n0
    res['podK_median'] = float(np.median(e_pod)); res['podK_per_time'] = np.median(e_pod.reshape(-1, 6), 0).round(4).tolist()
    out[tag] = res; print(tag, json.dumps(res), flush=True)
V, s = pod(Utr_flat)
#quad_with_nut(V, 16, Utr_flat, Xdev, 'raw_K16')
#quad_with_nut(V, 32, Utr_flat, Xdev, 'raw_K32')
# ---------------------------------------------------------------- translation alignment
kx = np.fft.fftfreq(N, 1.0 / N)
def align(U):
    """shift each (n,) field so that the phases of modes (1,0) and (0,1) are zero (spectral shift)."""
    W = np.fft.fft2(U.reshape(-1, N, N))
    p10, p01 = np.angle(W[:, 1, 0]), np.angle(W[:, 0, 1])
    amp = np.stack([np.abs(W[:, 1, 0]), np.abs(W[:, 0, 1])], 1) / np.sqrt((np.abs(W) ** 2).sum((1, 2)))[:, None]
    # shift by s: W_k -> W_k exp(-2 pi i k.s); we want angle(W_10) + (-2 pi s_x) = 0 -> s_x = p10 / (2 pi)
    sx, sy = p10 / (2 * np.pi), p01 / (2 * np.pi)
    ph = np.exp(-1j * 2 * np.pi * (kx[None, :, None] * sx[:, None, None] + kx[None, None, :] * sy[:, None, None]))
    Ua = np.real(np.fft.ifft2(W * ph)).reshape(U.shape[0], -1)
    return Ua, np.stack([sx, sy], 1), amp
Utr_a, s_tr, amp_tr = align(Utr_flat)
Xdev_a, s_dev, amp_dev = align(Xdev)
chk = np.fft.fft2(Utr_a[:5].reshape(-1, N, N))
out['align_check_phase_10_01_after'] = [float(np.abs(np.angle(chk[:, 1, 0])).max()), float(np.abs(np.angle(chk[:, 0, 1])).max())]
out['align_relative_amplitude_of_modes_10_01_q05_median'] = [np.percentile(amp_tr, 5, axis=0).round(4).tolist(), np.median(amp_tr, 0).round(4).tolist()]
Va, sa = pod(Utr_a)
out['energy_16_raw_vs_aligned'] = [float((s[:16] ** 2).sum() / (s ** 2).sum()), float((sa[:16] ** 2).sum() / (sa ** 2).sum())]
for K in (14, 16, 18, 30, 32):
    Vk = Va[:, :K]
    e = np.linalg.norm(Xdev_a - (Xdev_a @ Vk) @ Vk.T, axis=1) / n0
    out[f'aligned_pod{K}_dev_median'] = float(np.median(e)); out[f'aligned_pod{K}_per_time'] = np.median(e.reshape(-1, 6), 0).round(4).tolist()
    print('aligned pod', K, out[f'aligned_pod{K}_dev_median'], flush=True)
#quad_with_nut(Va, 14, Utr_a, Xdev_a, 'aligned_K14')
#quad_with_nut(Va, 16, Utr_a, Xdev_a, 'aligned_K16')
# nearest-neighbour coverage after alignment (same time)
tidx = np.tile(np.arange(NOUT), ntr)
nn = []
for i, x in enumerate(Xdev_a):
    m = tidx == TIMES[i % 6]
    nn.append(np.linalg.norm(Utr_a[m] - x, axis=1).min() / n0[i])
out['aligned_nn_field_same_time_median'] = float(np.median(nn))
json.dump(out, open('analyze_c2.json', 'w'), indent=1)
print('DONE', json.dumps(out, indent=1))
