"""Write configs/panel{n}.json from the pulled training records (validation data only).

Usage: make_panel_config.py <mesh> [--smoke-checkpoint-root DIR]
Reads runs/tr{n}_<arm>/output/result.json for every arm in configs/ops/, applies the
pre-registered size rule (DESIGN.md section 4: per family, lower validation mean-case-max
of the selected checkpoint; within 1 % relative the smaller arm), and records the cluster
checkpoint path + the sha256 the training job itself wrote. Arms without a complete result
are recorded as not trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NS = '/cluster/tufts/paralab/tawal01/nsops_20260923'
SMALLER = {'fno': 'fno-s', 'unet': 'unet-s', 'transolver': 'tsol-s', 'deeponet': 'don-s'}
REF = {32: 'experiments/ns3d-shift-head/runs/a2_h32/output/summary.json',
       64: 'experiments/ns3d-shift-head/runs/a2_h64/output/summary.json',
       96: 'experiments/ns3d-shift-head/runs/b2_heldout96/output/summary.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh', type=int)
    ap.add_argument('--smoke-checkpoint-root', type=Path, default=None)
    args = ap.parse_args()
    n = args.mesh
    root = HERE.parents[1]
    ref_path = root / REF[n]
    s = json.loads(ref_path.read_text())
    ref_values = {'head_k8': s['frontier']['k8_q0_dt0.02_it3']['errors'],
                  'span16': s['span']['span16_dt0.02_it3']['errors']}
    ref_map = {'nmrom_accurate_head_k8': 'head_k8', 'nmrom_fast_span16': 'span16'}
    steps = [200, 100, 80, 70, 60, 50, 40, 20, 10]
    for st in steps:
        e = s['cnab2'][str(st)]
        if e['errors'] is not None:
            ref_values[f'cnab2_s{st}'] = e['errors']
            ref_map[f'cnab2_s{st}'] = f'cnab2_s{st}'
    frozen_dir = f'experiments/ns3d-operators/frozen/h{n}' if n in (32, 64) else 'experiments/ns3d-shift-head/frozen'
    fsha = {name: sha(root / frozen_dir / name) for name in ('head_k8.npz', 'rotation.npz', 'bank_probe.npz')}
    ops, by_family = [], {}
    for cfgfile in sorted((HERE / 'configs' / 'ops').glob('*.json')):
        arm = cfgfile.stem
        family = json.loads(cfgfile.read_text())['family']
        job = f'tr{n}_{arm}'
        res_path = HERE / 'runs' / job / 'output' / 'result.json'
        if args.smoke_checkpoint_root is not None:
            res_path = args.smoke_checkpoint_root / f'smoke_{arm}' / 'result.json'
        if not res_path.exists():
            ops.append(dict(arm=arm, family=family, trained=False, reason='no result.json'))
            continue
        res = json.loads(res_path.read_text())
        if not res.get('complete') or int(res['mesh']) != n:
            ops.append(dict(arm=arm, family=family, trained=False, reason='incomplete or wrong mesh'))
            continue
        ckpt = (str(args.smoke_checkpoint_root / f'smoke_{arm}' / 'best.pt') if args.smoke_checkpoint_root
                else f'{NS}/{job}/output/best.pt')
        entry = dict(arm=arm, family=family, trained=True, checkpoint=ckpt, sha256=res['best_checkpoint_sha256'],
                     validation_mean_case_max=res['validation_best_checkpoint']['mean_case_max'],
                     training={k: res[k] for k in ('best_epoch', 'best_step', 'epochs_completed', 'optimisation_steps',
                                                   'stop_reason', 'training_seconds', 'micro_batch',
                                                   'real_parameter_count', 'parameter_dtype')}
                     | dict(validation_mean_case_max=res['validation_best_checkpoint']['mean_case_max'],
                            validation_worst_case_max=res['validation_best_checkpoint']['worst_case_max'],
                            result_sha256=sha(res_path)))
        ops.append(entry)
        by_family.setdefault(family, []).append(entry)
    for family, entries in by_family.items():
        best = min(e['validation_mean_case_max'] for e in entries)
        within = [e for e in entries if e['validation_mean_case_max'] <= best * 1.01]
        pick = next((e for e in within if e['arm'] == SMALLER[family]), None) or \
            min(within, key=lambda e: e['validation_mean_case_max'])
        for e in entries:
            e['selected'] = e is pick
    trained = [e for e in ops if e['trained']]
    heldout = n == 96
    cfg = dict(
        n=n, horizon=0.2, dt_truth=0.001, rank=64, modes=292, check_modes=8, gram_block=64,
        train_seed=202609201, train_cases=128, operator_train_cases=512,
        eval_seed=202609221 if heldout else 202609202, eval_cases=32 if heldout else 16,
        span_rank=16, rom_dt=0.02, rom_iters=3, damping=1e-6, ic_iters=12, cnab2_steps=steps,
        frozen_dir=frozen_dir, frozen_sha256=fsha,
        reference_summary=REF[n], reference_summary_sha256=sha(ref_path),
        reference_errors=ref_map, reference_values=ref_values,
        operators=trained, operators_not_trained=[e for e in ops if not e['trained']],
        full_cases=[] if heldout else [0, 1],
        sample_seed=20260923 + n, sample_points=8192, min_free_gb=40, timing_rounds=3, burn_calls=2,
        timing_seed=923 + n,
        smoke=dict(eval_cases=2, cnab2_steps=[200, 50, 40], timing_rounds=1, burn_calls=1))
    if args.smoke_checkpoint_root is not None:
        cfg['reference_values'] = {k: v[:2] for k, v in ref_values.items()}
    name = f'panel{n}_smoke.json' if args.smoke_checkpoint_root else f'panel{n}.json'
    out = HERE / 'configs' / name
    out.write_text(json.dumps(cfg, indent=1) + '\n')
    print(out, json.dumps({e['arm']: [round(e['validation_mean_case_max'], 5), e.get('selected')] for e in trained}))


if __name__ == '__main__':
    main()
