"""Corrected incumbent diagnosis D1(train)/D2/D3/D4/D6/D8 on the RIGHT cohort.

`pbh01` ran the incumbent's training-side diagnostics against
`core.source_params(0, 512)`, which is NOT the incumbent's training set: that
checkpoint was trained on `core.source_params(0, 576)[:512]`, and because every
call draws each parameter array at its own length the two cohorts share nothing.
The `pbh01` values of D2, D3, D4, D6 and D8 are therefore retracted and replaced
by this check; D1 on the development cohort, D5 and D7 are unaffected, since they
never touch the training parameters.

Local GB10 diagnostic, `jaxrun`, f64, highest matmul precision. Run from a flat
staging directory that holds the cell's modules and the incumbent checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import core as C
import sep_common as sc
import pbh_core as K_

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--parameters', default=str(HERE / 'incumbent-training-parameters.npy'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--intervals', type=int, default=255)
    ap.add_argument('--subsample', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--budget', type=int, default=400)
    ap.add_argument('--starts', type=int, default=8)
    a = ap.parse_args()
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    t0 = time.perf_counter()

    params, Z, ckcfg = sc.load_pkl(a.checkpoint)
    train = np.load(a.parameters)
    assert len(train) == len(Z), (len(train), len(Z))
    dev = np.concatenate((C.source_params(7090703, 6), C.source_params(7090732, 6)))
    rng = np.random.default_rng(a.seed)
    sub = np.sort(rng.choice(len(train), min(a.subsample, len(train)), replace=False))

    n = a.intervals
    G = K_.bank_of(params, n)
    Rg, rank = K_.bank_r(G)
    head = lambda z: sc.head(params, z)

    Utr = K_.fields(train[sub], n)
    Ttr, ptr, ntr = K_.project_targets(G, Rg, Utr)
    Udev = K_.fields(dev, n)
    Tdev, pdev, ndev = K_.project_targets(G, Rg, Udev)

    d1_train = np.asarray(jnp.sqrt(ptr / ntr))
    d1_dev = np.asarray(jnp.sqrt(pdev / ndev))
    d2 = K_.stored_code_errors(head, Rg, np.asarray(Z)[sub], Ttr, ptr, ntr)
    d3, it3, rs3 = K_.oracle_errors(head, Rg, Z, Ttr, ptr, ntr, a.budget, a.starts, 1e-6, 'gj')
    d5, it5, rs5 = K_.oracle_errors(head, Rg, Z, Tdev, pdev, ndev, a.budget, a.starts, 1e-6, 'gj')
    phat_tr, phat_dev = K_.descriptor(train), K_.descriptor(dev)
    nearest = np.sqrt(((phat_dev[:, None, :] - phat_tr[None, :, :]) ** 2).sum(-1)).min(1)
    order = lambda x: np.argsort(np.argsort(x))
    spearman = float(np.corrcoef(order(d5), order(nearest))[0, 1])

    out = dict(
        scope=__doc__.strip().splitlines()[0],
        retraction=('pbh01 D2, D3, D4, D6 and D8 for the incumbent are retracted: they were '
                    'computed against core.source_params(0, 512), which is not the incumbent '
                    'training cohort. D1 on the development cohort, D5 and D7 stand.'),
        machine='local GB10', backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
        x64=True, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
        jax_version=jax.__version__, intervals=n, bank_rank=rank,
        checkpoint=str(a.checkpoint),
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        K=int(np.asarray(Z).shape[1]), R=int(np.asarray(params['h_lin']).shape[1]),
        training_sources=int(len(train)), subsample=sub.tolist(),
        training_parameters_sha256=hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),
        D1_bank_floor_training=K_.summarise(d1_train),
        D1_bank_floor_development=K_.summarise(d1_dev),
        D2_head_at_stored_codes_training=K_.summarise(d2),
        D3_head_best_found_training=K_.summarise(d3),
        D3_exit_reasons=rs3.tolist(), D3_iterations=it3.tolist(),
        D4_code_refit_gain_training=dict(worst=float(np.max(d2 - d3)),
                                         median=float(np.median(d2 - d3)),
                                         relative_worst=float(np.max((d2 - d3) / d3)),
                                         relative_median=float(np.median((d2 - d3) / d3))),
        D5_head_best_found_development=K_.summarise(d5),
        D5_exit_reasons=rs5.tolist(),
        D6_generalisation_gap=dict(worst_minus_worst=float(d5.max() - d3.max()),
                                   ratio_of_worst=float(d5.max() / d3.max()),
                                   ratio_of_median=float(np.median(d5) / np.median(d3))),
        D8_nearest_training_parameter_distance=nearest.tolist(),
        D8_spearman_best_found_vs_distance=spearman,
        head_floor_over_bank_floor=float(d5.max() / d1_dev.max()),
        seconds=time.perf_counter() - t0)
    K_.dump(a.out, out)
    for k in ('D1_bank_floor_training', 'D1_bank_floor_development',
              'D2_head_at_stored_codes_training', 'D3_head_best_found_training',
              'D5_head_best_found_development'):
        print(f"{k:38s} worst {out[k]['worst']*100:9.4f}%  median {out[k]['median']*100:9.4f}%")
    print('D4 relative worst', out['D4_code_refit_gain_training']['relative_worst'])
    print('D6 ratio of worst', out['D6_generalisation_gap']['ratio_of_worst'])
    print('D8 spearman', spearman)


if __name__ == '__main__':
    main()
