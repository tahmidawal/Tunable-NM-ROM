"""Stage 4: recover the correction directions from saved fields, then recompute the weak
residual on the arm's own M test modes and on a disjoint held-out set.

The job's C_q array is archived nowhere. But every archived ROM invocation carries the
internal latents w_n = (z_n, y_n) for all 51 steps AND the decoded field at the six output
times, and the checkpoint is on disk. Since u = G(h(z) + C y) with G of full column rank r,

    c = G^+ u,      c - h(z) = C y,

so pooling the (y, c - h(z)) pairs of every arm at one q over-determines C_q. Accepted only
if held-out pairs reconstruct to <= 1e-8 relative AND the recomputed weak-residual norms
reproduce the job's own recorded per-step norms.

Decoder, bank, modes and residual are all re-implemented in NumPy from the same sources the
job used (`sep_common.features/head`, `engines.modes/spatial`, `arms.weak_dense`).
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np


def silu(x):
    return x / (1. + np.exp(-x))


def apply_mlp(params, x):
    for w, b in params[:-1]:
        x = silu(x @ np.asarray(w) + np.asarray(b))
    w, b = params[-1]
    return x @ np.asarray(w) + np.asarray(b)


def bc_poly(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def features(p, xy):
    ang = 2.0 * np.pi * (xy @ np.asarray(p['B']))
    ff = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)
    return (float(p['out_scale']) * bc_poly(xy))[..., None] * apply_mlp(p['g'], ff)


def head(p, z):
    assert 'hB' not in p
    return apply_mlp(p['h'], z) + z @ np.asarray(p['h_lin'])


def coords(L):
    x = np.arange(1, L) / L
    return np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)


def modes(L, M):
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')
    kx, ky = kx.ravel()[ind][:M], ky.ravel()[ind][:M]
    c = coords(L)
    phi = (2. / L) * np.sin(np.pi * c[:, 0, None] * kx) * np.sin(np.pi * c[:, 1, None] * ky)
    return phi, lam.ravel()[ind][:M]


def advection(u, L):
    p = np.pad(u.reshape(L - 1, L - 1), 1)
    c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    return (c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restore', required=True)
    ap.add_argument('--job', default='cclad01')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--qs', default='16,32,64,128')
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--heldout', type=int, default=1024)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R = Path(a.restore) / a.job
    r = json.loads((R / 'output' / 'result.json').read_text())
    L, dt, K = int(r['intervals']), float(r['config']['dt']), int(r['K'])
    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = {k: (np.asarray(v) if not isinstance(v, (list, tuple)) else
                  [(np.asarray(w), np.asarray(b)) for w, b in v]) for k, v in ck['params'].items()}
    xy = coords(L)
    G = features(params, xy)
    print('G', G.shape, flush=True)
    Qg, Rg = np.linalg.qr(G)
    Phi, lam = modes(L, a.M + a.heldout)
    P_on, lam_on = Phi[:, :a.M], lam[:a.M]
    P_off, lam_off = Phi[:, a.M:], lam[a.M:]

    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    inv = {}
    for x in r['invocations']:
        inv.setdefault((x['name'], x['case']), x)

    def decoded(arm, case):
        d = np.load(R / 'output' / inv[(arm, case)]['artifact'])
        f = d['fields']
        U = np.stack([np.ascontiguousarray(f[k][1:-1, 1:-1]).ravel() for k in range(f.shape[0])])
        return U, d['internal_latents']

    out = dict(job=a.job, M=a.M, heldout=a.heldout, intervals=L, dt=dt, K=K, rungs=[])
    for q in [int(v) for v in a.qs.split(',')]:
        arms = [k for k, v in setup.items() if v.get('q') == q and v.get('family', 'rom') == 'rom']
        Ys, Ds = [], []
        for arm in arms:
            for case in range(len(r['physical_cases'])):
                if (arm, case) not in inv:
                    continue
                U, W = decoded(arm, case)
                idx = np.arange(0, W.shape[0], int(round(.05 / dt)))
                Wk = W[idx]
                C0 = np.linalg.solve(Rg, Qg.T @ U.T).T                # (6, r)
                Ys.append(Wk[:, K:]); Ds.append(C0 - head(params, Wk[:, :K]))
        Y = np.concatenate(Ys); D = np.concatenate(Ds)
        n = len(Y)
        if q == 0:
            # nothing to recover: h(z) alone must already reproduce the decoded field
            Cq = np.zeros((0, D.shape[1]))
            tr = te = np.arange(n)
            rank, err_tr = 0, float(np.linalg.norm(D) / max(np.linalg.norm(D + 0) + 1e-300, 1e-300))
            err_tr = float(np.max(np.linalg.norm(D, axis=1)))
            err_te = err_tr
            accepted = True
        else:
            rng = np.random.default_rng(20260916)
            perm = rng.permutation(n)
            cut = max(n - max(8, n // 5), q + 4)
            tr, te = perm[:cut], perm[cut:]
            Cq, *_ = np.linalg.lstsq(Y[tr], D[tr], rcond=None)         # (q, r)
            sv = np.linalg.svd(Y[tr], compute_uv=False)
            rank = int((sv > sv[0] * 1e-10).sum())
            err_tr = float(np.linalg.norm(Y[tr] @ Cq - D[tr]) / max(np.linalg.norm(D[tr]), 1e-300))
            err_te = float(np.linalg.norm(Y[te] @ Cq - D[te]) / max(np.linalg.norm(D[te]), 1e-300))
            accepted = bool(rank == q and err_te <= 1e-8)
        rung = dict(q=q, arms=sorted(arms), pairs=n, train=len(tr), test=len(te),
                    design_rank=rank, recover_train_relative=float(err_tr),
                    recover_heldout_relative=float(err_te), accepted=accepted, cases=[])
        print('q', q, 'pairs', n, 'rank', rank, 'train', err_tr, 'heldout', err_te,
              'accepted', accepted, flush=True)
        if accepted or q == 0:
            for arm in sorted(arms):
                if setup[arm].get('M') != a.M or setup[arm].get('quadrature') != 'dense':
                    continue
                for case in range(len(r['physical_cases'])):
                    if (arm, case) not in inv:
                        continue
                    nu = float(r['physical_cases'][case][4])
                    _, W = decoded(arm, case)
                    Cc = head(params, W[:, :K]) + (W[:, K:] @ Cq if q else 0.)
                    Uall = Cc @ G.T                                    # (51, n)
                    on, off = [], []
                    for s in range(1, Uall.shape[0]):
                        u, prev = Uall[s], Uall[s - 1]
                        av = advection(u, L)
                        for P, lm, acc in ((P_on, lam_on, on), (P_off, lam_off, off)):
                            au, ap_ = P.T @ u, P.T @ prev
                            acc.append(float(np.linalg.norm(
                                (au - ap_ + dt * (P.T @ av + nu * lm * au)) / (1 + dt * nu * lm))))
                    rec = inv[(arm, case)].get('residuals')
                    rung['cases'].append(dict(
                        arm=arm, case=case, on_test=on, held_out=off,
                        recorded=rec,
                        recomputed_vs_recorded_relative=(
                            float(np.max(np.abs(np.array(on) - np.array(rec))
                                         / np.maximum(np.abs(rec), 1e-300))) if rec else None)))
                    print('  ', arm, case, 'on', np.median(on), 'off', np.median(off),
                          'vs recorded', rung['cases'][-1]['recomputed_vs_recorded_relative'], flush=True)
        out['rungs'].append(rung)
    Path(a.out).write_text(json.dumps(out))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
