"""audit_phase1.py -- INDEPENDENT NumPy audit of a Phase-1 result directory.

    python experiments/ns2d/audit_phase1.py <output-dir> [audit.json]

Imports neither the driver nor JAX.  Recomputes, from the saved fields alone:
  * F-TG:      relative error of tg_N*.npz `final` against the closed-form fully-discrete
               TG solution and against the semi-discrete / continuum forms, and the
               backward-Euler control -- all from the stored (w0, lam_h, nu, dt, nsteps);
  * F-MMS:     spatial errors from mms_space_N*.npz (final vs exact) and their orders;
  * F-BUDGET:  the per-step enstrophy/energy identities from budget_N*.npz using its own
               5-point Laplacian and its own FFT Poisson solve; nu=0 conservation; the BE control;
  * F-MESH:    errors against the finest mesh from mesh_*.npz and their orders;
  * F-INDEP:   the per-step relative difference between the two saved trajectories;
and compares every recomputed number with result.json to 1e-12 relative (or 1e-14 absolute).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PI = np.pi


def rel(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def lap(w, N):
    return (N * N) * (np.roll(w, 1, 0) + np.roll(w, -1, 0) + np.roll(w, 1, 1) + np.roll(w, -1, 1) - 4 * w)


def poisson(w, N):
    k = np.arange(N)
    l1 = (2 - 2 * np.cos(2 * PI * k / N)) * N * N
    lam = l1[:, None] + l1[None, :]
    lam[0, 0] = 1.0
    W = np.fft.fft2(w) / lam
    W[0, 0] = 0.0
    return np.real(np.fft.ifft2(W))


def orders(e):
    e = np.asarray(e, float)
    return (np.log(e[:-1] / e[1:]) / np.log(2.0)).tolist()


def close(a, b, rtol=1e-12, atol=1e-14):
    return abs(a - b) <= atol + rtol * abs(b)


def main():
    out = Path(sys.argv[1])
    res = json.loads((out / 'result.json').read_text())
    G = res['gates']
    audit = dict(source=str(out / 'result.json'), checks=[], all_match=True)

    def check(name, recomputed, reported):
        ok = close(recomputed, reported)
        audit['checks'].append(dict(name=name, recomputed=recomputed, reported=reported, match=ok))
        audit['all_match'] &= ok
        print(f'{"ok " if ok else "MISMATCH"} {name}: recomputed {recomputed:.6e} reported {reported:.6e}')

    for f in sorted(out.glob('tg_N*.npz')):
        d = np.load(f)
        N = int(f.stem[4:])
        w0, lam_h, nu, dt, n = d['w0'], float(d['lam_h']), float(d['nu']), float(d['dt']), int(d['nsteps'])
        a = dt * nu * lam_h
        ex = w0 * ((1 - a / 2) / (1 + a / 2)) ** n
        g = G[f'F-TG_N{N}']
        check(f'F-TG_N{N}.exact_rel', rel(d['final'], ex), g['exact_rel'])
        check(f'F-TG_N{N}.semi_rel', rel(d['final'], w0 * np.exp(-nu * lam_h * dt * n)), g['semi_rel'])
        check(f'F-TG_N{N}.cont_rel', rel(d['final'], w0 * np.exp(-nu * 8 * PI * PI * dt * n)), g['cont_rel'])
        check(f'F-TG_N{N}.be_rel', rel(d['be_final'], ex), g['be_rel'])
        lam_c = 8 * PI * PI
        pred = abs(np.exp(-nu * dt * n * lam_h) - np.exp(-nu * dt * n * lam_c)) / np.exp(-nu * dt * n * lam_c)
        check(f'F-TG_N{N}.cont_pred', pred, g['cont_pred'])

    sp = []
    for f in sorted(out.glob('mms_space_N*.npz'), key=lambda p: int(p.stem.split('N')[-1])):
        d = np.load(f)
        sp.append(rel(d['final'], d['exact']))
    if sp and 'F-MMS' in G:
        for i, e in enumerate(sp):
            check(f'F-MMS.spatial_errors[{i}]', e, G['F-MMS']['spatial_errors'][i])
        for i, o in enumerate(orders(sp)):
            check(f'F-MMS.spatial_orders[{i}]', o, G['F-MMS']['spatial_orders'][i])

    for f in sorted(out.glob('budget_N*.npz')):
        d = np.load(f)
        N = int(f.stem[8:])
        st, nu, dt = d['states'], float(d['nu']), float(d['dt'])
        Z = 0.5 * np.sum(st * st, axis=(1, 2)) / (N * N)
        Eg = np.array([0.5 * np.sum(poisson(s, N) * s) / (N * N) for s in st])
        wz = we = 0.0
        for n in range(len(st) - 1):
            wm = 0.5 * (st[n] + st[n + 1])
            L = lap(wm, N)
            dZ = dt * nu * np.sum(wm * L) / (N * N)
            dE = dt * nu * np.sum(poisson(wm, N) * L) / (N * N)
            wz = max(wz, abs((Z[n + 1] - Z[n]) - dZ) / abs(dZ))
            we = max(we, abs((Eg[n + 1] - Eg[n]) - dE) / abs(dE))
        g = G[f'F-BUDGET_N{N}']
        check(f'F-BUDGET_N{N}.identity_enstrophy', wz, g['identity_enstrophy'])
        check(f'F-BUDGET_N{N}.identity_energy', we, g['identity_energy'])
        s0 = d['states_nu0']
        Z0 = 0.5 * np.sum(s0 * s0, axis=(1, 2)) / (N * N)
        E0 = np.array([0.5 * np.sum(poisson(s, N) * s) / (N * N) for s in s0])
        check(f'F-BUDGET_N{N}.conservation_enstrophy', float(np.max(np.abs(Z0 - Z0[0])) / Z0[0]), g['conservation_enstrophy'])
        check(f'F-BUDGET_N{N}.conservation_energy', float(np.max(np.abs(E0 - E0[0])) / E0[0]), g['conservation_energy'])
        sb = d['states_be_nu0']
        Zb = 0.5 * np.sum(sb * sb, axis=(1, 2)) / (N * N)
        check(f'F-BUDGET_N{N}.be_control_enstrophy', float(np.max(np.abs(Zb - Zb[0])) / Zb[0]), g['be_control_enstrophy'])

    for f in sorted(out.glob('mesh_*.npz')):
        d = np.load(f)
        tag = f.stem[5:]
        Ns = sorted(int(k[1:]) for k in d.files if k.startswith('N'))
        fine = d[f'N{Ns[-1]}']
        errs = [rel(d[f'N{N}'], fine[::Ns[-1] // N, ::Ns[-1] // N]) for N in Ns[:-1]]
        g = G[f'F-MESH_{tag}']
        for i, e in enumerate(errs):
            check(f'F-MESH_{tag}.errors[{i}]', e, g['errors_vs_finest'][i])
        for i, o in enumerate(orders(errs)):
            check(f'F-MESH_{tag}.orders[{i}]', o, g['orders'][i])

    if (out / 'indep.npz').exists():
        d = np.load(out / 'indep.npz')
        per = [rel(d['jax'][k], d['numpy'][k]) for k in range(d['jax'].shape[0])]
        check('F-INDEP.worst_rel', float(max(per)), G['F-INDEP']['worst_rel'])

    # dev8 cohort hashes: recompute physical parameters' hash from the stored array
    import hashlib
    for f in sorted(out.glob('dev*_N*.npz')):
        d = np.load(f)
        N = int(f.stem.split('N')[-1])
        h = hashlib.sha256(np.ascontiguousarray(d['physical']).tobytes()).hexdigest()
        audit['checks'].append(dict(name=f'dev8_N{N}.physical_prefix_sha256', value=h))

    audit['n_checks'] = len(audit['checks'])
    print('ALL_MATCH', audit['all_match'], 'checks', audit['n_checks'])
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
