"""audit_headfit.py -- INDEPENDENT NumPy/SciPy audit of a head-fit (DESIGN §A9, ns301) result.

    python experiments/ns2d/audit_headfit.py <output-dir> <phase1-output-dir> [audit.json]

Imports neither the driver nor JAX (bank/head/params_draw are the NumPy re-implementations of
audit_phase2.py).  From the frozen checkpoint and the per-arm head pickles it recomputes:
  * the frozen bank's numerical rank on the 256^2 grid;
  * the bank floor on the archived Phase-1 dev8 states (dev cases 0..7 x the six times), which
    are the first 48 of the driver's dev-report states, EXACTLY;
  * for every arm, a SciPy Levenberg-Marquardt fit from the SAME single start the driver uses
    (the training code whose head output is nearest in the whitened metric) on those 48 states,
    compared with the driver's single-start error at a VALUE tolerance (two LM implementations
    need not stop at the same point) and the ordering oracle <= single-start for every state;
  * every reported median, ratio, held-out/training gap, log-log slope and §A9 verdict from the
    stored per-state arrays.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_phase2 import bank, head, params_draw          # noqa: E402  (NumPy re-implementations)


def main():
    out, p1 = Path(sys.argv[1]), Path(sys.argv[2])
    res = json.loads((out / 'result.json').read_text())
    cfg = res['config']
    K, R, N = cfg['K'], cfg['R'], cfg['TRAIN_N']
    ck = pickle.load(open(out.parent / 'checkpoints' / Path(res['checkpoint']['path']).name, 'rb'))
    pb = {k: (np.asarray(v) if not isinstance(v, list) else [(np.asarray(w), np.asarray(b)) for w, b in v])
          for k, v in ck['params'].items()}
    audit = dict(source=str(out / 'result.json'), checks=[], all_match=True, notes=[])

    def check(name, rec, rep, rtol=1e-9, atol=1e-14, kind='exact'):
        ok = bool(abs(rec - rep) <= atol + rtol * abs(rep))
        audit['checks'].append(dict(name=name, recomputed=float(rec), reported=float(rep), match=ok, rtol=rtol, kind=kind))
        audit['all_match'] = bool(audit['all_match'] and ok)
        print(f'{"ok " if ok else "MISMATCH"} {name}: recomputed {rec:.6e} reported {rep:.6e} ({kind}, rtol {rtol:g})')

    d8 = np.load(p1 / f'dev8_N{N}.npz')
    phys = params_draw(cfg['SEEDS']['dev'], cfg['N_DEV'])[:8]
    check('dev8.physical_max_abs_diff', float(np.max(np.abs(d8['physical'] - phys))), 0.0, atol=1e-15)
    times = cfg['ORACLE_TIMES']
    X = np.stack([d8['U'][c, t] for c in range(8) for t in times])            # (48, n)
    n0 = np.repeat(np.linalg.norm(d8['U'][:8, 0], axis=1), len(times))
    G = bank(pb, N)
    s = np.linalg.svd(G, compute_uv=False)
    rank = int(np.sum(s > s.max() * 1e-12 * np.sqrt(G.shape[0])))
    check('B-ORTH_N256.rank', rank, res['gates'][f'B-ORTH_N{N}']['rank'], kind='exact-integer')
    Q, Rb = np.linalg.qr(G)
    C48 = np.linalg.solve(Rb, Q.T @ X.T).T                                     # (48, R)
    perp = np.linalg.norm(X - (X @ Q) @ Q.T, axis=1)
    e_bank = perp / n0

    for tag, arm in res['arms'].items():
        ev = arm['eval']['dev_report']
        eo, e1, ep, eb = (np.asarray(ev[k]) for k in ('per_state_oracle', 'per_state_single', 'per_state_podK', 'per_state_bank'))
        check(f'{tag}.bank_floor.first48_worst_rel_diff', float(np.max(np.abs(e_bank - eb[:48]) / eb[:48])), 0.0, atol=1e-9)
        check(f'{tag}.oracle_le_single_all_states', float(np.all(eo <= e1 * (1 + 1e-12))), 1.0, atol=0.0)
        check(f'{tag}.oracle_ge_bank_all_states', float(np.all(eo >= eb * (1 - 1e-9))), 1.0, atol=0.0)
        check(f'{tag}.oracle_median', float(np.median(eo)), ev['oracle_median'])
        check(f'{tag}.podK_median', float(np.median(ep)), ev['podK_median'])
        check(f'{tag}.ratio_podK_over_oracle', float(np.median(ep) / np.median(eo)), ev['ratio_podK_over_oracle'])
        check(f'{tag}.budget_exits', float(ev['oracle_reasons'].get('0', 0)), ev['budget_exits'], kind='exact-integer')
        tr = arm['eval']['train']
        check(f'{tag}.heldout_over_train_oracle', float(np.median(eo) / np.median(np.asarray(tr['per_state_oracle']))),
              arm['heldout_over_train_oracle_median'])
        check(f'{tag}.heldout_over_training_recon', float(np.median(eo) / arm['training']['recon_rel_l2_median']),
              arm['heldout_over_training_median'])
        # single-start fit from the driver's start rule, SciPy LM, whitened metric
        hp = pickle.load(open(out / f'head_{tag}.pkl', 'rb'))
        ph = {'h': [(np.asarray(w), np.asarray(b)) for w, b in hp['params']['h']], 'h_lin': np.asarray(hp['params']['h_lin'])}
        Z = np.asarray(hp['Z'])
        Zc = Z[::max(1, len(Z) // 8192)]
        Hc = head(ph, Zc) @ Rb.T
        Hn = np.sum(Hc * Hc, 1)
        e_np = np.empty(48)
        for i in range(48):
            c = C48[i]
            z0 = Zc[np.argmin(Hn - 2 * (Hc @ (Rb @ c)))]
            sol = least_squares(lambda z: Rb @ (head(ph, z) - c), z0, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-12, max_nfev=20000)
            e_np[i] = np.linalg.norm(G @ head(ph, sol.x) - X[i]) / n0[i]
        rel = np.abs(e_np - e1[:48]) / e1[:48]
        audit['notes'].append(f'{tag}: SciPy-LM single-start vs driver single-start on 48 states: median rel diff {np.median(rel):.2e}, worst {rel.max():.2e}; '
                              f'SciPy better on {int(np.sum(e_np < e1[:48] * (1 - 1e-9)))}, driver better on {int(np.sum(e_np > e1[:48] * (1 + 1e-9)))}')
        check(f'{tag}.single_start.first48_median_rel_diff', float(np.median(rel)), 0.0, atol=1e-3,
              kind='value (independent LM from the same start; local minima may differ per state)')
        check(f'{tag}.scipy_single_ge_oracle_all_states', float(np.all(e_np >= eo[:48] * (1 - 1e-6))), 1.0, atol=0.0,
              kind='value (the driver oracle must be at least as good as an independent single-start fit)')
    for name, sm in res['summary'].items():
        ns = np.asarray(sm['n_traj'], float)
        ys = np.asarray([res['arms'][f'n{int(n)}_{name}']['eval']['dev_report']['oracle_median'] for n in ns])
        slope = float(np.polyfit(np.log(ns), np.log(ys), 1)[0])
        check(f'summary.{name}.loglog_slope', slope, sm['loglog_slope'])
        verdict = 'needs-data' if slope <= -0.25 else 'head-limited' if slope >= -0.10 else 'ambiguous'
        check(f'summary.{name}.verdict_matches_A9_rule', float(verdict == sm['verdict']), 1.0, atol=0.0)
    audit['n_checks'] = len(audit['checks'])
    print('ALL_MATCH', audit['all_match'], 'checks', audit['n_checks'])
    for n in audit['notes']:
        print('NOTE', n)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
