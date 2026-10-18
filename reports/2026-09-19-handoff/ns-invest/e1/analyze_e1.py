"""analyze_e1.py -- the two pre-registered E1 statistics (and the side columns) on the 64^2 cohorts.
CPU numpy only (no JAX).  usage: python analyze_e1.py [data_*.npz ...]  (default: all data_*.npz here)
Statistics, exactly as in the head investigation (analyze_b.py, report Sec 2.3):
  * POD-K floor: basis = SVD of ALL training states (ntr x 26 snapshots); held-out error
    ||x - V_K V_K^T x|| / ||omega_dev(case, t=0)|| on the 6 output times TIMES; reported as
      floor_all   = median over the ndev x 6 held-out states          (the head report's 0.239 for A)
      floor_worst = per case max over the 5 evolved times, median over cases   (the task's wording)
      floor_t0 / floor_evolved = medians of the t=0 states / of the t>0 states
  * coverage: for each held-out state, min over training states AT THE SAME OUTPUT INDEX of
    ||u_tr - x|| / ||omega_dev(case, 0)||; median over the ndev x 6 states (the head's 0.644 for A).
Side columns: POD energy fractions, energy/enstrophy decay, Newton it/step, CFL, any-time coverage,
own-norm floors, and (NOT pre-registered) the same two statistics after the head's spectral alignment
of the (1,0)/(0,1) Fourier phases (analyze_c2.py) -- the frame a shifted-manifold decoder (E2) sees."""
import sys, os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NOUT = 26
TIMES = np.array([0, 5, 10, 15, 20, 25])
TH_FLOOR, TH_COV = 0.35, 0.25


def energy_enstrophy(U, N):
    k = np.arange(N)
    l1 = (2.0 - 2.0 * np.cos(2.0 * np.pi * k / N)) * (N * N)
    lam = l1[:, None] + l1[None, :]; lam[0, 0] = 1.0
    W = np.fft.fft2(U.reshape(-1, N, N)) / lam; W[:, 0, 0] = 0.0
    psi = np.real(np.fft.ifft2(W)).reshape(U.shape[0], -1)
    return 0.5 * (psi * U).sum(1) / (N * N), 0.5 * (U * U).sum(1) / (N * N)


def align(U, N):
    """shift each field so the phases of Fourier modes (1,0) and (0,1) vanish (head analyze_c2.py)."""
    kx = np.fft.fftfreq(N, 1.0 / N)
    W = np.fft.fft2(U.reshape(-1, N, N))
    sx, sy = np.angle(W[:, 1, 0]) / (2 * np.pi), np.angle(W[:, 0, 1]) / (2 * np.pi)
    ph = np.exp(-1j * 2 * np.pi * (kx[None, :, None] * sx[:, None, None] + kx[None, None, :] * sy[:, None, None]))
    return np.real(np.fft.ifft2(W * ph)).reshape(U.shape[0], -1)


def stats(Utr, Udev, N, tag, it=None, cfl=None, ntr_used=None):
    ndev = Udev.shape[0]
    Xdev = np.stack([Udev[c, t] for c in range(ndev) for t in TIMES])
    n0 = np.repeat(np.linalg.norm(Udev[:, 0], axis=1), len(TIMES))
    nt = np.linalg.norm(Xdev, axis=1)
    Utr_flat = Utr.reshape(-1, N * N)
    res = dict(tag=tag, ntr=int(Utr.shape[0]), ndev=int(ndev), n_train_states=int(Utr_flat.shape[0]))
    # ---- POD
    _, s, Vt = np.linalg.svd(Utr_flat, full_matrices=False)
    res['pod_energy_16_32_64'] = [float((s[:k] ** 2).sum() / (s ** 2).sum()) for k in (16, 32, 64)]
    for K in (16, 32, 64):
        Vk = Vt[:K].T
        e = np.linalg.norm(Xdev - (Xdev @ Vk) @ Vk.T, axis=1) / n0
        E = e.reshape(ndev, len(TIMES))
        res[f'pod{K}'] = dict(floor_all=float(np.median(e)), floor_worst_evolved=float(np.median(E[:, 1:].max(1))),
                              floor_t0=float(np.median(E[:, 0])), floor_evolved=float(np.median(E[:, 1:])),
                              per_time=np.median(E, 0).round(4).tolist(),
                              own_norm_all=float(np.median(e * n0 / nt)))
    # ---- coverage
    tidx = np.tile(np.arange(NOUT), Utr.shape[0])
    nn_same, nn_any = np.empty(len(Xdev)), np.empty(len(Xdev))
    for i, x in enumerate(Xdev):
        dd = np.linalg.norm(Utr_flat - x, axis=1)
        nn_any[i] = dd.min() / n0[i]
        nn_same[i] = dd[tidx == TIMES[i % len(TIMES)]].min() / n0[i]
    C = nn_same.reshape(ndev, len(TIMES))
    res['coverage'] = dict(same_time_all=float(np.median(nn_same)), same_time_t0=float(np.median(C[:, 0])),
                           same_time_evolved=float(np.median(C[:, 1:])), per_time=np.median(C, 0).round(4).tolist(),
                           any_time_all=float(np.median(nn_any)),
                           same_time_own_norm=float(np.median(nn_same * n0 / nt)))
    # ---- invariants over the horizon (held-out trajectories)
    E_, Z_ = energy_enstrophy(Udev.reshape(-1, N * N), N)
    E_, Z_ = E_.reshape(ndev, NOUT), Z_.reshape(ndev, NOUT)
    res['energy_ratio_per_time_median'] = np.median(E_[:, TIMES] / E_[:, :1], 0).round(4).tolist()
    res['enstrophy_ratio_per_time_median'] = np.median(Z_[:, TIMES] / Z_[:, :1], 0).round(4).tolist()
    res['norm_ratio_T_median'] = float(np.median(np.linalg.norm(Udev[:, -1], axis=1) / np.linalg.norm(Udev[:, 0], axis=1)))
    if it is not None:
        res['newton_per_step'] = dict(mean=float(it.mean()), max=int(it.max()), frac_gt2=float((it > 2).mean()))
    if cfl is not None:
        res['cfl'] = float(cfl)
    # ---- NOT pre-registered: after spectral (1,0)/(0,1) phase alignment
    Utr_a, Xdev_a = align(Utr_flat, N), align(Xdev, N)
    _, sa, Vta = np.linalg.svd(Utr_a, full_matrices=False)
    ea = np.linalg.norm(Xdev_a - (Xdev_a @ Vta[:16].T) @ Vta[:16], axis=1) / n0
    nn_a = np.array([np.linalg.norm(Utr_a[tidx == TIMES[i % 6]] - x, axis=1).min() / n0[i] for i, x in enumerate(Xdev_a)])
    res['aligned_not_preregistered'] = dict(pod16_floor_all=float(np.median(ea)), coverage_same_time_all=float(np.median(nn_a)))
    # ---- verdict on the pre-registered pair
    f, c = res['pod16']['floor_all'], res['coverage']['same_time_all']
    res['verdict'] = dict(floor_pass=bool(f >= TH_FLOOR), coverage_pass=bool(c <= TH_COV),
                          both=bool(f >= TH_FLOOR and c <= TH_COV),
                          floor_worst_pass=bool(res['pod16']['floor_worst_evolved'] >= TH_FLOOR))
    return res


def main(files):
    out = {}
    # the head investigation's cohort (282 train / 64 dev) as the reproduction anchor
    head = os.path.join(HERE, '..', 'head', 'data64.npz')
    if os.path.exists(head):
        d = np.load(head)
        out['A_head282x64'] = stats(d['U_tr'], d['U_dev'], int(d['N']), 'A (head cohort 282/64)')
        print('A_head282x64', json.dumps(out['A_head282x64']['pod16']), json.dumps(out['A_head282x64']['coverage']), flush=True)
    for f in files:
        d = np.load(f)
        fam = os.path.basename(f)[5:-4]
        it = np.concatenate([d['it_tr'].ravel(), d['it_dev'].ravel()])
        r = stats(d['U_tr'], d['U_dev'], int(d['N']), fam, it=it, cfl=float(d['cfl']))
        r['meta'] = json.loads(str(d['meta'])); r['worst_newton_residual'] = float(max(d['rn_tr'].max(), d['rn_dev'].max()))
        r['gen_seconds'] = float(d['seconds']); r['seeds'] = d['seeds'].tolist()
        out[fam] = r
        print(fam, json.dumps({k: r[k] for k in ('pod16', 'coverage', 'newton_per_step', 'verdict')}), flush=True)
    json.dump(out, open(os.path.join(HERE, 'analyze_e1.json'), 'w'), indent=1)
    # ---- markdown table (generated; the report includes this file verbatim)
    rows = ['| family | params (P) | T | POD-16 floor (all / worst-evolved) | POD-32 | POD-64 | coverage (same time) | t=0 split: POD-16 / cov | E(T)/E(0) · Z(T)/Z(0) | Newton it/step (mean/max) | verdict |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for fam, r in out.items():
        m = r.get('meta', {}); T = m.get('T', 1.0); P = m.get('P', '')
        nw = r.get('newton_per_step', {}); nws = f"{nw['mean']:.2f} / {nw['max']}" if nw else 'n/a'
        v = r['verdict']
        verdict = 'PASS both' if v['both'] else ('floor only' if v['floor_pass'] else ('coverage only' if v['coverage_pass'] else 'FAIL both'))
        rows.append(f"| {r['tag'] if fam.startswith('A_') else fam} | {P} | {T} | **{r['pod16']['floor_all']:.3f}** / {r['pod16']['floor_worst_evolved']:.3f} | "
                    f"{r['pod32']['floor_all']:.3f} | {r['pod64']['floor_all']:.3f} | **{r['coverage']['same_time_all']:.3f}** | "
                    f"{r['pod16']['floor_t0']:.3f} / {r['coverage']['same_time_t0']:.3f} | "
                    f"{r['energy_ratio_per_time_median'][-1]:.3f} · {r['enstrophy_ratio_per_time_median'][-1]:.3f} | {nws} | {verdict} |")
    rows2 = ['| family | POD-16 per output time (t = 0 … T) | coverage per output time | POD energy 16/32/64 | aligned (not pre-reg.) POD-16 / coverage | own-norm POD-16 / coverage | CFL | worst Newton res |',
             '|---|---|---|---|---|---|---|---|']
    for fam, r in out.items():
        a = r['aligned_not_preregistered']
        rows2.append(f"| {r['tag'] if fam.startswith('A_') else fam} | {' '.join(f'{x:.3f}' for x in r['pod16']['per_time'])} | "
                     f"{' '.join(f'{x:.3f}' for x in r['coverage']['per_time'])} | "
                     f"{' / '.join(f'{x:.3f}' for x in r['pod_energy_16_32_64'])} | {a['pod16_floor_all']:.3f} / {a['coverage_same_time_all']:.3f} | "
                     f"{r['pod16']['own_norm_all']:.3f} / {r['coverage']['same_time_own_norm']:.3f} | {r.get('cfl', float('nan')):.2f} | {r.get('worst_newton_residual', float('nan')):.1e} |")
    open(os.path.join(HERE, 'table.md'), 'w').write('\n'.join(rows) + '\n\n' + '\n'.join(rows2) + '\n')
    print('\n'.join(rows)); print(); print('\n'.join(rows2))


if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, 'data_?.npz')))
    main(files)
