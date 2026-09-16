"""Bank selection on the common cohort of DESIGN.md amendment 4.

Every bank arm is scored on ONE cohort — `core.source_params(20260916, 256)`, a fresh
seed asserted disjoint from every training cohort and from the development cohort —
so the maximum over it is comparable across arms, which the per-arm validation splits
of the original rule were not. Mesh 255; the floor is measured to be mesh-independent
to three significant figures.

Local GB10 diagnostic, `jaxrun`, f64, highest matmul precision. No cluster job.
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

import core as C
import sep_common as sc
import pbh_core as K_


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path, help='collected pbh01 output directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--count', type=int, default=256)
    ap.add_argument('--intervals', type=int, default=255)
    a = ap.parse_args()
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    t0 = time.perf_counter()
    d = json.loads((a.run / 'result.json').read_text())
    assert d['complete']
    cfg = d['config']

    common = C.source_params(a.seed, a.count)
    dev = np.concatenate((C.source_params(cfg['eval_seed'], cfg['eval_count']),
                          C.source_params(cfg['fresh_seed'], cfg['fresh_count'])))
    disjoint = {}
    for S in cfg['source_counts']:
        tr = C.source_params(cfg['train_seed'], S)
        n = int(sum(1 for t in tr for s in common if np.allclose(t, s)))
        assert n == 0, (S, n)
        disjoint[str(S)] = n
    ndev = int(sum(1 for t in dev for s in common if np.allclose(t, s)))
    assert ndev == 0, ndev

    arms = []
    for ck in d['checkpoints']:
        if ck.get('layer') != 'bank':
            continue
        src = a.run / ck['path']
        assert hashlib.sha256(src.read_bytes()).hexdigest() == ck['sha256']
        params, Z, ckcfg = sc.load_pkl(src)
        G = K_.bank_of(params, a.intervals)
        Rg, rank = K_.bank_r(G)
        assert rank['rank_valid'], (ck['id'], rank['rank'])
        err = K_.bank_floor(G, Rg, common, a.intervals)
        arm = next(x for x in d['bank_arms'] if x['arm'] == ck['id'])
        rep = next(f for f in arm['floors'] if f['intervals'] == a.intervals)
        arms.append(dict(arm=ck['id'], R=int(ckcfg['rank']), S=int(ckcfg['sources']),
                         checkpoint_sha256=ck['sha256'], bank_rank=rank['rank'],
                         common=K_.summarise(err),
                         reported_validation_worst=rep['validation']['worst'],
                         reported_development_worst=rep['development']['worst'],
                         reported_fit_worst=rep['fit']['worst']))
        print('COMMON', ck['id'], 'worst', float(err.max()), flush=True)
        del G, Rg
        jax.clear_caches()

    key = lambda x: (x['common']['worst'], x['common']['median'], x['R'], x['S'])
    order = sorted(arms, key=key)
    old = sorted(arms, key=lambda x: x['reported_validation_worst'])
    devo = sorted(arms, key=lambda x: x['reported_development_worst'])
    out = dict(
        scope=__doc__.strip().splitlines()[0],
        rule=('lowest worst bank projection floor on the common selection cohort at '
              f'{a.intervals} intervals; ties by median, then smaller R, then smaller S'),
        common_cohort=dict(seed=a.seed, count=a.count,
                           sha256=K_.sha_array(common),
                           disjoint_from_training=disjoint,
                           disjoint_from_development=ndev),
        machine='local GB10', backend=jax.default_backend(), gpu=jax.devices()[0].device_kind,
        x64=True, matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
        jax_version=jax.__version__, intervals=a.intervals,
        source_result_sha256=hashlib.sha256((a.run / 'result.json').read_bytes()).hexdigest(),
        arms=arms, selected=order[0]['arm'],
        ranking_common=[x['arm'] for x in order],
        ranking_original_rule=[x['arm'] for x in old],
        ranking_development=[x['arm'] for x in devo],
        original_rule_selected=d['selection']['bank']['selected'],
        rules_agree=bool(order[0]['arm'] == d['selection']['bank']['selected']),
        development_ranking_agrees=bool(devo[0]['arm'] == order[0]['arm']),
        seconds=time.perf_counter() - t0)
    K_.dump(a.out, out)
    for x in order:
        print(f"{x['arm']:20s} R{x['R']:4d} S{x['S']:5d} common worst "
              f"{x['common']['worst']*100:8.4f}%  median {x['common']['median']*100:8.4f}%  "
              f"(val {x['reported_validation_worst']*100:7.4f}%  dev "
              f"{x['reported_development_worst']*100:7.4f}%)")
    print('SELECTED', out['selected'], '| original rule', out['original_rule_selected'],
          '| agree', out['rules_agree'])


if __name__ == '__main__':
    main()
