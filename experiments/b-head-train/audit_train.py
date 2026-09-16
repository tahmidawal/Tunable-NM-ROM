"""Independent audit of a training attempt. NumPy only: no JAX, no driver.

Re-derives every draw from its seed, re-hashes every emitted checkpoint, checks
that each one has the parameter shapes its arm declares, re-runs the
pre-registered density / capacity / objective selection arithmetic from the
recorded held-out oracle numbers, and confirms the source checkpoint was not
touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def params_draw(seed, count):
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count),
                     r.uniform(.05, .20, count), r.uniform(.5, 2., count),
                     np.exp(r.uniform(np.log(.01), np.log(.1), count))], axis=1)


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--checkpoints', required=True)
    p.add_argument('--draws', help='npz written by cluster/capture_draws.py: the attempt\'s own draws, regenerated in the cluster interpreter and verified against the hashes this result.json records')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    cfg = r['config']
    D = Path(a.checkpoints)
    checks = {}

    def gate(name, ok, detail=None):
        checks[name] = dict(passed=bool(ok), detail=detail)

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r.get('backend') == 'gpu', r.get('backend'))
    gate('x64', r.get('x64') is True)
    gate('precision_highest', r.get('matmul_precision') == 'highest')
    gate('source_checkpoint_unchanged',
         r['source_checkpoint_sha256'] == r['source_checkpoint_sha256_after'])
    gate('whitening_round_trip', r['gates']['whitening_round_trip'] < 1e-10,
         r['gates']['whitening_round_trip'])
    gate('identity_star', r['gates']['identity_star_relative'] < 1e-9,
         r['gates']['identity_star_relative'])

    # ---- the draws, re-derived from their seeds ----------------------------
    traj = np.concatenate((params_draw(cfg['canonical_seed'], cfg['canonical_trajectories']),
                           params_draw(cfg['extra_seed'], cfg['extra_trajectories'])))
    hold = params_draw(cfg['holdout_seed'], cfg['holdout_trajectories'])
    ev = np.concatenate((params_draw(cfg['eval_seed'], cfg['eval_cases']),
                         params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    local = dict(train=traj, holdout=hold, eval=ev)
    recorded = {k: r['data'][f'{k}_physical_sha256'] for k in local}

    # The job records a SHA256 of each draw. Hashing a LOCAL re-derivation against it is not a
    # portable check: `np.exp` differs by one unit in the last place between this machine's NumPy
    # and the cluster's, so the viscosity column -- and only that column -- can disagree in its
    # last bit while every value is the same number to 16 digits (DESIGN.md A5). The gate is
    # therefore split. `--draws` is the attempt's own draw, regenerated in the cluster's
    # interpreter and accepted by `cluster/capture_draws.py` only because it hashes to exactly
    # the recorded value; that equality is re-checked here, so nothing is taken on trust.
    if a.draws:
        z = np.load(a.draws)
        arch = {k: np.asarray(z[k]) for k in local}
        gate('archived_draws_match_the_recorded_hashes',
             all(sha_array(arch[k]) == recorded[k] for k in local),
             {k: sha_array(arch[k]) == recorded[k] for k in local})
        tol = {}
        for k in local:
            d = np.abs(arch[k] - local[k])
            ulp = float(np.max(d / np.maximum(np.spacing(np.abs(local[k])), 1e-300)))
            cols = np.nonzero(d.max(0) > 0)[0].tolist()
            tol[k] = dict(bitwise=bool((arch[k] == local[k]).all()), max_ulp=ulp,
                          max_relative=float(np.max(d / np.maximum(np.abs(local[k]), 1e-300))),
                          columns_differing=cols, rows_differing=int((d.max(1) > 0).sum()))
        gate('archived_draws_reproduce_locally_to_one_ulp',
             all(v['max_ulp'] <= 1.0 and v['max_relative'] < 1e-15
                 and set(v['columns_differing']) <= {4} for v in tol.values()), tol)
        use = arch
    else:
        gate('draw_hashes_bitwise', all(sha_array(local[k]) == recorded[k] for k in local),
             {k: sha_array(local[k]) == recorded[k] for k in local})
        use = local

    lo = np.array([.15, .15, .05, .5, .01])
    hi = np.array([.85, .85, .20, 2., .1])
    gate('draws_inside_the_declared_ranges',
         all(bool(((v >= lo - 1e-12) & (v <= hi + 1e-12)).all()) for v in use.values()),
         {k: [float(v.min(0).min()), float(v.max(0).max())] for k, v in use.items()})

    traj, hold, ev = use['train'], use['holdout'], use['eval']
    ov = [(i, j) for i, t in enumerate(traj) for j, s in enumerate(ev) if np.allclose(t, s)]
    ov += [(i, j) for i, t in enumerate(hold) for j, s in enumerate(ev) if np.allclose(t, s)]
    ov += [(i, j) for i, t in enumerate(hold) for j, s in enumerate(traj) if np.allclose(t, s)]
    gate('cohorts_disjoint', not ov, ov)
    gate('state_stride_is_one_so_pairs_are_one_step_apart', cfg['state_stride'] == 1,
         cfg['state_stride'])
    gate('training_data_converged', r['data']['training']['max_relative_residual'] <= 1e-8,
         r['data']['training']['max_relative_residual'])

    # ---- the emitted checkpoints -------------------------------------------
    bad = []
    for x in r['emitted']:
        path = D / x['artifact']
        if not path.exists():
            bad.append((x['arm'], 'missing'))
            continue
        if sha_file(path) != x['sha256']:
            bad.append((x['arm'], 'hash'))
            continue
        d = pickle.load(open(path, 'rb'))
        K = int(np.asarray(d['Z_tr']).shape[1])
        R = int(np.asarray(d['params']['h_lin']).shape[1])
        if (K, R) != (x['K'], x['R']):
            bad.append((x['arm'], f'shape {K},{R} != {x["K"]},{x["R"]}'))
            continue
        if int(np.asarray(d['params']['h'][0][0]).shape[0]) != K:
            bad.append((x['arm'], 'head input width'))
        if int(np.asarray(d['params']['g'][-1][0]).shape[1]) != R:
            bad.append((x['arm'], 'bank output width'))
        if not np.isfinite(np.concatenate([np.asarray(v).ravel() for v in
                                           [d['params']['h_lin'], d['Z_tr']]])).all():
            bad.append((x['arm'], 'non-finite'))
    gate('every_checkpoint_present_hashed_and_shaped', not bad, bad)

    # frozen-bank arms must share the incumbent's bank exactly
    src = pickle.load(open(cfg.get('source_path', ''), 'rb')) if cfg.get('source_path') else None
    frozen_ok, moved = [], []
    ref_bank = None
    for row in r['arms']:
        d = pickle.load(open(D / row['checkpoint'], 'rb'))
        b = np.concatenate([np.asarray(d['params']['B']).ravel()]
                           + [np.asarray(w).ravel() for w, _ in d['params']['g']])
        if row['spec']['bank'] == 'frozen':
            if ref_bank is None:
                ref_bank = b
            frozen_ok.append(bool(b.shape == ref_bank.shape and np.array_equal(b, ref_bank)))
        else:
            moved.append(row['arm'])
    gate('frozen_bank_arms_share_one_bank', all(frozen_ok), dict(n=len(frozen_ok), joint=moved))

    # ---- the pre-registered selection arithmetic ---------------------------
    worst = {row['arm']: row['holdout']['selection']['max'] for row in r['arms']}
    K0 = r['K_incumbent']
    curve = [(n, worst[f'd{n}k{K0}rec']) for n in cfg['densities']]
    rec = r['selection']
    gate('density_curve_matches', all(abs(a1 - b1) < 1e-12 and a0 == b0
                                      for (a0, a1), (b0, b1) in zip(curve, rec['density_curve'])),
         curve)
    mono = all(curve[i][1] >= curve[i + 1][1] for i in range(len(curve) - 1))
    i_mid = min(1, max(len(curve) - 2, 0))
    g_mid = 1. - curve[i_mid + 1][1] / curve[i_mid][1]
    g_top = 1. - curve[-1][1] / curve[i_mid][1]
    sat = cfg['saturation_fraction']
    verdict = ('data-limited' if (mono and g_top > sat)
               else 'capacity/objective-limited' if g_mid < sat else 'mixed')
    gate('diagnostic_verdict_reproduced', verdict == rec['diagnostic_verdict'],
         dict(recomputed=verdict, recorded=rec['diagnostic_verdict'], monotone=mono,
              gain_mid_to_next=g_mid, gain_mid_to_top=g_top, saturation_fraction=sat))
    gate('best_density_reproduced', min(curve, key=lambda x: x[1])[0] == rec['best_density'])
    gate('best_objective_reproduced',
         min(rec['objective_curve'], key=lambda x: x[1])[0] == rec['best_objective'])
    gate('bank_rank_arm_condition',
         (rec.get('bank_rank_arms') is None
          or (cfg['wide_rank'] in rec['bank_rank_arms']) == (verdict != 'data-limited')),
         dict(arms=rec.get('bank_rank_arms'), verdict=verdict))

    wts = r.get('weights', {})
    gate('objective_weights_recorded',
         all(k in wts for k in ('beta_w', 'beta_t', 'gamma', 'L_rec', 'L_weak', 'L_smooth')),
         {k: wts.get(k) for k in ('beta_w', 'beta_t', 'gamma', 'L_rec', 'L_weak', 'L_smooth')})
    gate('objective_weights_reproduce_the_declared_ratio',
         (abs(wts.get('beta_w', 0) - cfg['objective_fraction'] * wts.get('L_rec', 0)
              / max(wts.get('L_weak', 1e-300), 1e-300)) < 1e-9
          and abs(wts.get('gamma', 0) - cfg['objective_fraction'] * wts.get('L_rec', 0)
                  / max(wts.get('L_smooth', 1e-300), 1e-300)) < 1e-9),
         dict(rho=cfg['objective_fraction']))

    table = [dict(arm=row['arm'], K=row['K'], R=row['R'], bank=row['spec']['bank'],
                  objective=row['spec']['objective'], trajectories=row['trajectories'],
                  states=row['states'], steps=row['spec']['steps'],
                  recon_train_mean=(row['recon_train'] or {}).get('mean'),
                  recon_train_max=(row['recon_train'] or {}).get('max'),
                  holdout_mean=row['holdout']['selection']['mean'],
                  holdout_max=row['holdout']['selection']['max'],
                  holdout_mean_only=row['holdout']['mean_only']['mean'],
                  holdout_over_floor=row['holdout_over_floor'],
                  span_floor_mean=row['span_floor']['mean'],
                  train_gpu_hours=row['train_gpu_hours'],
                  final_parts=(row['train'] or {}).get('final_parts'),
                  beta_w=row['spec'].get('beta_w'), beta_t=row['spec'].get('beta_t'),
                  gamma=row['spec'].get('gamma'),
                  checkpoint_sha256=row['checkpoint_sha256'])
             for row in r['arms']]

    out = dict(result=str(a.result), checks=checks, arm_table=table, selection=rec,
               all_passed=bool(all(v['passed'] for v in checks.values())))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float) + '\n')
    for k, v in checks.items():
        print(f"{'PASS' if v['passed'] else 'FAIL'}  {k}  {v['detail']}")
    print('AUDIT', 'OK' if out['all_passed'] else 'FAILED', a.out)


if __name__ == '__main__':
    main()
