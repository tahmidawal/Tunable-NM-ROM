"""extras_e1.py -- side checks for the report (numpy): (1) how the two statistics of B2 / B scale with the
number of training trajectories (32/64/128 -> extrapolation to 512); (2) how far the dipoles actually travel
(peak-tracked displacement of the positive vortex over the horizon, torus-unwrapped), per family B/B2/B3."""
import json, numpy as np
from analyze_e1 import TIMES, NOUT
out = {}
for fam in ('B', 'B2', 'B3'):
    d = np.load(f'data_{fam}.npz'); Utr, Udev, P = d['U_tr'], d['U_dev'], d['P_dev']; N = int(d['N'])
    ndev = Udev.shape[0]
    Xdev = np.stack([Udev[c, t] for c in range(ndev) for t in TIMES]); n0 = np.repeat(np.linalg.norm(Udev[:, 0], axis=1), 6)
    r = {}
    for ntr in (32, 64, 128):
        U = Utr[:ntr].reshape(-1, N * N)
        _, s, Vt = np.linalg.svd(U, full_matrices=False); Vk = Vt[:16].T
        e = np.linalg.norm(Xdev - (Xdev @ Vk) @ Vk.T, axis=1) / n0
        tidx = np.tile(np.arange(NOUT), ntr)
        nn = np.array([np.linalg.norm(U[tidx == TIMES[i % 6]] - x, axis=1).min() / n0[i] for i, x in enumerate(Xdev)])
        r[f'ntr{ntr}'] = dict(pod16_floor_all=float(np.median(e)), pod16_floor_worst_evolved=float(np.median(e.reshape(ndev, 6)[:, 1:].max(1))),
                             coverage=float(np.median(nn)))
    c = np.array([r[f'ntr{n}']['coverage'] for n in (32, 64, 128)])
    slope = np.polyfit(np.log([32, 64, 128]), np.log(c), 1)[0]
    r['coverage_log_slope_vs_ntr'] = float(slope)
    r['coverage_extrapolated_512'] = float(c[-1] * (512 / 128) ** slope)
    r['coverage_extrapolated_2048'] = float(c[-1] * (2048 / 128) ** slope)
    # dipole travel: track the positive-vortex peak over the 26 outputs, unwrap on the torus
    disp = []
    for c_ in range(ndev):
        pos = np.array([np.unravel_index(np.argmax(Udev[c_, t].reshape(N, N)), (N, N)) for t in range(NOUT)]) / N
        step = (np.diff(pos, axis=0) + 0.5) % 1.0 - 0.5
        disp.append(np.linalg.norm(step.sum(0)))
    disp = np.array(disp)
    G, dd = P[:, 0], P[:, 1]
    r['travel_domain_lengths_median_q10_q90'] = np.percentile(disp, [50, 10, 90]).round(3).tolist()
    r['point_vortex_speed_Gamma_over_2pi_d_median'] = float(np.median(G / (2 * np.pi * dd)))
    r['T'] = float(json.loads(str(d['meta']))['T'])
    out[fam] = r
    print(fam, json.dumps(r, indent=1), flush=True)
json.dump(out, open('extras_e1.json', 'w'), indent=1)
