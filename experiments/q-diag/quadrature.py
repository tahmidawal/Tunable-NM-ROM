"""Stage 2: the empirical-quadrature rule's own error, measured on saved fields.

`arms.weak_eq` approximates ONLY the advection term of the weak residual; the mass and
Laplacian terms go through A = Phi^T G exactly. So the rule's entire contribution to the
residual error is

    rho(u) = || sum_j w_j Phi(x_j) a(u)(x_j) - Phi^T a(u) || / || Phi^T a(u) ||,

with a(u) the upwind advection field of `engines.spatial`. That is a functional of ONE
field, so it can be evaluated on every archived output field with no internal state, no
decoder and no correction directions. Nodes and weights come from the archived
`arm_setup.eq_indices` / `eq_weights` of the job that ran the arm.

Also reported: how much of the advection mass the m sampled cells actually see, and how
close they are to the advection front (the top 1 % of |a|).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def coords(L):
    x = np.arange(1, L) / L
    return np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)


def modes(L, M):
    """engines.modes, verbatim in NumPy."""
    k = np.arange(1, L)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    lam = 4 * L ** 2 * (np.sin(np.pi * kx / (2 * L)) ** 2 + np.sin(np.pi * ky / (2 * L)) ** 2)
    ind = np.argsort(lam.ravel(), kind='stable')[:M]
    kx, ky = kx.ravel()[ind], ky.ravel()[ind]
    c = coords(L)
    phi = (2. / L) * np.sin(np.pi * c[:, 0, None] * kx) * np.sin(np.pi * c[:, 1, None] * ky)
    return phi, lam.ravel()[ind], np.stack((kx, ky), axis=1)


def advection(u, L):
    """engines.spatial(...)[0], verbatim in NumPy."""
    p = np.pad(u.reshape(L - 1, L - 1), 1)
    c, xm, xp, ym, yp = p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    adv = c * L * (np.where(c > 0, c - xm, xp - c) + np.where(c > 0, c - ym, yp - c))
    return adv.reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--restore', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    R = Path(a.restore)
    L = 256
    n = (L - 1) ** 2

    # every EQ rule in the five archives, keyed by (job, arm)
    rules, results = {}, []
    jobs = {}
    for job in ('btq101', 'btq102', 'btq201', 'cclad01', 'qlad01'):
        r = json.loads((R / job / 'output' / 'result.json').read_text())
        jobs[job] = r
        for s in r.get('arm_setup', []):
            if s.get('quadrature') != 'eq' or 'eq_indices' not in s:
                continue
            rules[(job, s['arm'])] = dict(idx=np.asarray(s['eq_indices'], dtype=np.int64),
                                          w=np.asarray(s['eq_weights'], dtype=float),
                                          M=int(s['M']), q=s.get('q'), m=int(s['m']),
                                          fit=s.get('eq_relative_fit'))
    print('eq rules:', len(rules), flush=True)

    phi_cache = {}

    def PHI(M):
        if M not in phi_cache:
            phi_cache[M] = modes(L, M)[0]
        return phi_cache[M]

    # the fields rho is evaluated on: every arm's own trajectory, plus the converged FOM
    for (job, arm), ru in sorted(rules.items()):
        r = jobs[job]
        Phi = PHI(ru['M'])
        sub = Phi[ru['idx']] * ru['w'][:, None]                       # (m, M)
        inv = {(x['name'], x['case']): x['artifact'] for x in r['invocations']}
        for src in (arm, 'fft_tight'):
            for case in range(len(r['physical_cases'])):
                key = (src, case)
                if key not in inv:
                    continue
                f = np.load(R / job / 'output' / inv[key])['fields']
                for k in range(f.shape[0]):
                    u = np.ascontiguousarray(f[k][1:-1, 1:-1]).ravel()
                    av = advection(u, L)
                    exact = Phi.T @ av
                    approx = sub.T @ av[ru['idx']]
                    ne = np.linalg.norm(exact)
                    aa = np.abs(av)
                    tot = aa.sum()
                    thr = np.quantile(aa, 0.99)
                    front = aa >= thr
                    near = np.zeros(n, dtype=bool)
                    ij = np.stack(np.unravel_index(ru['idx'], (L - 1, L - 1)), 1)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            q2 = np.clip(ij + np.array([di, dj]), 0, L - 2)
                            near[np.ravel_multi_index((q2[:, 0], q2[:, 1]), (L - 1, L - 1))] = True
                    results.append(dict(
                        job=job, arm=arm, q=ru['q'], M=ru['M'], m=ru['m'],
                        eq_relative_fit=ru['fit'], evaluated_on=src, case=case, time_index=k,
                        rho=float(np.linalg.norm(approx - exact) / max(ne, 1e-300)),
                        exact_norm=float(ne),
                        mass_fraction_sampled=float(aa[ru['idx']].sum() / max(tot, 1e-300)),
                        front_covered=float(near[front].mean()),
                        front_cells=int(front.sum())))
        print('rho done', job, arm, flush=True)
    Path(a.out).write_text(json.dumps(results))
    print('wrote', a.out, len(results))


if __name__ == '__main__':
    main()
