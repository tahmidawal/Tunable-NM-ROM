"""burgers-heldout bh1, last step: correction directions C for the new (bank, head), by the incumbent's rule.

Mirrors q-trajdirs `qtd_run.py` (commit f064b882, the job that produced `directions_qtd02.npz`) up to
and including the `old` direction set, arithmetic unchanged: snapshots of `params_draw(0, 128)` at 256
intervals, stride 2, FOM tolerances 1e-9 / 1e-7; bank coefficients through the thin QR of the bank on
that grid; `directions.audited` with qtd02's residual configuration. Writes `directions_bh.npz`
(keys C, Ct, singular_values) and `directions.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import arms as A
import ladder as LD
import directions as DIR

QTD02 = dict(intervals=256, dt=.005, train_seed=0, train_trajectories=128, train_state_stride=2,
             snapshot_ntol=1e-9, snapshot_ltol=1e-7, residual_snapshots=1024, residual_starts=4,
             residual_budget=200, residual_seed=20260915, decoder_code_subsample=8192,
             gauss_jordan_max=64, cold_axis_points=48,
             strict=dict(ic_budget=400, step_budget=600, gtol=1e-6), ic_gtol=1e-6, inner_damping=1e-10,
             tau_y=.1, q_ladder=[0, 16, 32, 64, 128, 256])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--smoke', type=int, default=0, help='local smoke: trajectories and snapshots scaled down')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(QTD02)
    if a.smoke:
        cfg.update(intervals=a.smoke, train_trajectories=4, residual_snapshots=16, decoder_code_subsample=256)
    print(f'jax_backend={jax.default_backend()}', flush=True)
    t0 = time.perf_counter()
    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, LD.host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    L = cfg['intervals']
    train_physical = e.params_draw(cfg['train_seed'], cfg['train_trajectories'])
    U, sinfo = LD.generate_snapshots(L, cfg['dt'], train_physical, cfg['train_state_stride'],
                                     cfg['snapshot_ntol'], cfg['snapshot_ltol'])
    bank = A.CoordBank(params, K, R)
    G = bank.on_grid(L)
    Qb, Rb = A.whiten(G)
    Ut = jnp.asarray(U.T)
    coef_truth = jnp.linalg.solve(Rb, Qb.T @ Ut).T
    sinfo['bank_projection_relative_rms'] = float(jnp.linalg.norm(Ut - Qb @ (Qb.T @ Ut)) / jnp.linalg.norm(Ut))
    del Ut, U
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    Zsub = np.asarray(Zold[::stride])
    C, Ct, Zstar, rho, info = DIR.audited(params, Rb, coef_truth, Zsub, K, dict(cfg))
    C, Ct = np.asarray(C), np.asarray(Ct)
    orth = float(np.max(np.abs((np.asarray(Rb) @ C).T @ (np.asarray(Rb) @ C) - np.eye(C.shape[1]))))
    np.savez_compressed(out / 'directions_bh.npz', C=C, Ct=Ct,
                        singular_values=np.asarray(info.get('singular_values', [])))
    rep = dict(config=cfg, job=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
               gpu=jax.devices()[0].device_kind, matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
               checkpoint_sha256=LD.sha_file(a.checkpoint), K=K, R=R, snapshots=sinfo,
               directions={k: v for k, v in info.items() if k != 'singular_values'},
               singular_values_head=list(np.asarray(info.get('singular_values', []))[:64]),
               field_orthonormality_max_abs_dev=orth, columns=int(C.shape[1]),
               directions_file_sha256=LD.sha_file(out / 'directions_bh.npz'),
               seconds=time.perf_counter() - t0, complete=True)
    LD.dump(out / 'directions.json', rep)
    print('DIRECTIONS', C.shape, 'orth', f'{orth:.2e}', 'head-fit median/worst',
          info.get('head_fit_relative_median'), info.get('head_fit_relative_worst'),
          round(time.perf_counter() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
