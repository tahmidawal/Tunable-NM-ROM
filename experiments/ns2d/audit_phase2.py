"""audit_phase2.py -- INDEPENDENT NumPy audit of a Phase-2 result directory.

    python experiments/ns2d/audit_phase2.py <phase2-output-dir> <phase1-output-dir> [audit.json]

Imports neither the driver nor JAX.  From the pickled checkpoint (plain NumPy arrays) it
re-implements the periodic bank (integer Fourier features -> SiLU MLP -> mean-zero columns)
and the head (MLP + linear skip) and, on the first 8 development trajectories archived by
Phase 1 (dev8_N*.npz, all 26 output states), recomputes:

  * the bank's numerical rank on every evaluation grid (gate B-ORTH);
  * the per-state ORACLE and SINGLE-START errors  ||G h(z) - u|| / ||u_case(0)||  at the
    saved codes (oracle_N*.npz) -- an exact recomputation, no projection involved;
  * the per-state bank floor and the per-case bank floor (worst over the 26 states) with an
    SVD projection onto the NUMERICAL range of G (rank-deficient banks make the QR-based
    projection of the driver depend on roundoff directions; the difference is reported and
    bounded, not hidden);
  * every gate's arithmetic from the stored per-state arrays and floors;
  * the physical parameters of dev8 against a column-by-column re-draw of the dev seed.
Exact quantities are compared at 1e-9 relative; projection-based ones at a value tolerance
that is recorded with the check.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

PI = np.pi
MODES = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)]


def silu(x):
    return x / (1.0 + np.exp(-x))


def mlp(params, x):
    for w, b in params[:-1]:
        x = silu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def bank(p, N):
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing='ij')
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    ang = 2.0 * PI * (xy @ np.asarray(p['B']))
    ff = np.concatenate([np.sin(ang), np.cos(ang)], -1)
    G = float(p['out_scale']) * mlp(p['g'], ff)
    return G - G.mean(0, keepdims=True)


def head(p, z):
    return mlp(p['h'], z) + z @ np.asarray(p['h_lin'])


def params_draw(seed, count, nu_lo=1e-3, nu_hi=1e-2):
    r = np.random.default_rng(seed)
    cols = [r.standard_normal(count) for _ in range(12)]
    cols.append(np.exp(r.uniform(np.log(nu_lo), np.log(nu_hi), count)))
    return np.stack(cols, 1)


def main():
    out, p1 = Path(sys.argv[1]), Path(sys.argv[2])
    res = json.loads((out / 'result.json').read_text())
    cfg = res['config']
    K, R = cfg['K'], cfg['R']
    ck = pickle.load(open(out / res['checkpoint']['path'], 'rb'))
    p = {k: (np.asarray(v) if not isinstance(v, list) else [(np.asarray(w), np.asarray(b)) for w, b in v])
         for k, v in ck['params'].items()}
    G_ = res['gates']
    audit = dict(source=str(out / 'result.json'), checks=[], all_match=True, notes=[])

    def check(name, rec, rep, rtol=1e-9, atol=1e-14, kind='exact'):
        ok = bool(abs(rec - rep) <= atol + rtol * abs(rep))
        audit['checks'].append(dict(name=name, recomputed=float(rec), reported=float(rep), match=ok, rtol=rtol, kind=kind))
        audit['all_match'] = bool(audit['all_match'] and ok)
        print(f'{"ok " if ok else "MISMATCH"} {name}: recomputed {rec:.6e} reported {rep:.6e} ({kind}, rtol {rtol:g})')

    # -- the physical parameters of the archived dev8 are the first 8 column-wise draws of the dev seed
    ref_any = np.load(p1 / f'dev8_N{cfg["EVAL_NS"][0]}.npz')
    phys = params_draw(cfg['SEEDS']['dev'], cfg['N_DEV'])[:8]
    check('dev8.physical_max_abs_diff', float(np.max(np.abs(ref_any['physical'] - phys))), 0.0, atol=1e-15)

    # -- the rank cap is structural: the g-track's last layer is linear over g_hidden units
    gh = cfg['ARCH']['g_hidden']
    audit['notes'].append(f'structural rank bound of the bank: min(R, g_hidden) = {min(R, gh)} (R={R}, g_hidden={gh})')

    for N in cfg['EVAL_NS']:
        Ns = str(N)
        d8 = np.load(p1 / f'dev8_N{N}.npz')
        U8 = d8['U']                                                   # (8, NOUT, n)
        G = bank(p, N)                                                 # (n, R)
        s = np.linalg.svd(G, compute_uv=False)
        rank = int(np.sum(s > s.max() * 1e-12 * np.sqrt(G.shape[0])))  # numerical rank (SVD)
        g = G_[f'B-ORTH_N{N}']
        check(f'B-ORTH_N{N}.rank', rank, g['rank'], atol=0.5, kind='exact-integer')
        check(f'B-ORTH_N{N}.rank_le_structural_bound', float(rank <= min(R, gh)), 1.0, atol=0.0)
        Uq, sq, _ = np.linalg.svd(G, full_matrices=False)
        Q = Uq[:, :rank]                                               # numerical range only
        o = res['oracle'][Ns]
        onp = np.load(out / f'oracle_N{N}.npz')
        cases, times = onp['cases'], onp['times']
        nt = len(times)
        # per-state arrays in result.json equal the npz arrays
        for key in ('per_state_oracle', 'per_state_single', 'per_state_podK', 'per_state_bank'):
            arr = {'per_state_oracle': 'e_oracle', 'per_state_single': 'e_single',
                   'per_state_podK': 'e_podK', 'per_state_bank': 'e_bank'}[key]
            check(f'oracle_N{N}.{key}.max_abs_diff_json_vs_npz', float(np.max(np.abs(np.asarray(o[key]) - onp[arr]))), 0.0, atol=1e-15)
        # exact recomputation of the oracle / single-start errors on the archived 8 cases
        worst_or, worst_sg, worst_n0 = 0.0, 0.0, 0.0
        n_states = 0
        for ci, c in enumerate(cases):
            if c >= 8:
                continue
            for ti, t in enumerate(times):
                i = ci * nt + ti
                u = U8[c, t]
                n0 = np.linalg.norm(U8[c, 0])
                worst_n0 = max(worst_n0, abs(n0 - onp['n0'][i]) / n0)
                e_or = np.linalg.norm(G @ head(p, onp['Z'][i]) - u) / n0
                e_sg = np.linalg.norm(G @ head(p, onp['Z1'][i]) - u) / n0
                worst_or = max(worst_or, abs(e_or - onp['e_oracle'][i]) / onp['e_oracle'][i])
                worst_sg = max(worst_sg, abs(e_sg - onp['e_single'][i]) / onp['e_single'][i])
                n_states += 1
        check(f'oracle_N{N}.n0.worst_rel_diff', worst_n0, 0.0, atol=1e-12)
        check(f'oracle_N{N}.e_oracle.worst_rel_diff_over_{n_states}_states', worst_or, 0.0, atol=1e-9)
        check(f'oracle_N{N}.e_single.worst_rel_diff_over_{n_states}_states', worst_sg, 0.0, atol=1e-9)
        # bank floor on the same states, projection onto the NUMERICAL range (value-level check)
        worst_bk = 0.0
        for ci, c in enumerate(cases):
            if c >= 8:
                continue
            for ti, t in enumerate(times):
                i = ci * nt + ti
                u = U8[c, t]
                e_bk = np.linalg.norm(u - Q @ (Q.T @ u)) / np.linalg.norm(U8[c, 0])
                worst_bk = max(worst_bk, abs(e_bk - onp['e_bank'][i]) / onp['e_bank'][i])
        tol_bk = 1e-9 if rank == R else 5e-2
        check(f'oracle_N{N}.e_bank.worst_rel_diff_numerical_range_vs_driver_QR', worst_bk, 0.0, atol=tol_bk,
              kind='value (driver QR spans R columns; rank-deficient banks add roundoff directions)')
        # per-case bank floor (worst over all 26 states) for the 8 archived cases
        fl = res['floors'][Ns]['bank'][str(R)]
        per_case = np.asarray(fl['per_case_worst_fixed'])[:8]
        rec = []
        for c in range(8):
            X = U8[c]
            e = np.linalg.norm(X - (X @ Q) @ Q.T, axis=1) / np.linalg.norm(X[0])
            rec.append(e.max())
        check(f'B-FLOOR_N{N}.bank.per_case_worst_fixed[:8].worst_rel_diff', float(np.max(np.abs(np.asarray(rec) - per_case) / per_case)), 0.0,
              atol=tol_bk, kind='value (same caveat)')
        # gate arithmetic from the stored arrays
        e_or, e_sg, e_pk, e_bk = onp['e_oracle'], onp['e_single'], onp['e_podK'], onp['e_bank']
        check(f'H-ORACLE_N{N}.oracle_median', float(np.median(e_or)), o['oracle_median'])
        check(f'H-ORACLE_N{N}.podK_median', float(np.median(e_pk)), o['podK_median'])
        check(f'H-ORACLE_N{N}.bank_floor_median', float(np.median(e_bk)), o['bank_floor_median'])
        check(f'H-ORACLE_N{N}.oracle_worst', float(e_or.max()), o['oracle_worst'])
        passed = bool(np.median(e_or) <= 0.5 * np.median(e_pk) and np.median(e_or) >= np.median(e_bk) * (1 - 1e-9))
        check(f'H-ORACLE_N{N}.passed', float(passed), float(G_[f'H-ORACLE_N{N}']['passed']), atol=0.0)
        check(f'H-SOLVED_N{N}.single_start_median', float(np.median(e_sg)), o['single_start_median'])
        check(f'H-SOLVED_N{N}.passed', float(np.median(e_sg) <= 1.5 * np.median(e_or)), float(G_[f'H-SOLVED_N{N}']['passed']), atol=0.0)
        bf = G_[f'B-FLOOR_N{N}']
        podR = res['floors'][Ns]['pod'][str(R)]
        check(f'B-FLOOR_N{N}.bank_worst_evolved', float(np.max(fl['per_case_worst_fixed'])), bf['bank_worst_evolved'])
        check(f'B-FLOOR_N{N}.ratio', fl['worst_evolved_fixed'] / podR['worst_evolved_fixed'], bf['ratio'])
        check(f'B-FLOOR_N{N}.passed', float(fl['worst_evolved_fixed'] <= 2.0 * podR['worst_evolved_fixed']), float(bf['passed']), atol=0.0)
        check(f'B-FLOOR_N{N}.bank_median', float(np.median(fl['per_case_worst_fixed'])), bf['bank_median'])
    audit['all_passed_reported'] = res.get('all_passed')
    audit['failed_gates_reported'] = [k for k, g in G_.items() if not g['passed']]
    audit['n_checks'] = len(audit['checks'])
    print('ALL_MATCH', audit['all_match'], 'checks', audit['n_checks'], 'failed gates (reported):', audit['failed_gates_reported'])
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
