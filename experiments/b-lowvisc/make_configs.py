"""Generate every b-lowvisc configuration from the audited b-panel parent.

Nothing in the emitted configurations is hand-typed:

  * the full-order Newton grid, the meshes, the time step, the cohort seeds, the reference
    mesh and the snapshot settings are read verbatim out of `comparators/bpn301-config-256.json`
    (b-panel's 256^2 configuration, job 3780638);
  * the incumbent-family comparator numbers the gate job checks itself against -- the POD
    `best-found` floor of every rank and the median GPU cost of every full-order setting --
    are read out of `comparators/bpn301-summary.json`, the b-panel report's own machine
    readable table;
  * the incumbent development cases are recomputed here with NumPy and written into the
    configuration as VALUES, because `np.exp` differs by one ulp between the GB10 and the
    cluster and a hash gate would fail spuriously (CLAUDE.md landmine; b-seeds A2).

Run:  python experiments/b-lowvisc/make_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lv_common as LV

HERE = Path(__file__).resolve().parent
PARENT = json.loads((HERE / 'comparators/bpn301-config-256.json').read_text())
SUMMARY = json.loads((HERE / 'comparators/bpn301-summary.json').read_text())['rows']
JOB = '3780638'


def comparator(family, metric):
    return {r['subject']: r['value'] for r in SUMMARY
            if r['family'] == family and r['metric'] == metric and r['job_id'] == JOB}


def main():
    best = comparator('pod', 'best_found_percent')
    ranks = PARENT['pod_ranks']
    pod_floor = {}
    for k in ranks:
        key = next(s for s in best if s.startswith(f'pod{k}_'))
        pod_floor[str(k)] = best[key]
    fom_ms = comparator('fom', 'median_gpu_ms')
    fom_err = comparator('fom', 'worst_evolved_percent')

    cfg = dict(
        attempt='lvg01',
        lane='b-lowvisc',
        purpose=('DESIGN.md sections 2 and 6: both viscosity families, one allocation, one GPU, '
                 'no trained model. Leg (a) is the POD projection floor of every panel rank on '
                 'the six development cases; leg (b) is the tuned full-order Newton grid of job '
                 f'{JOB} timed for both families in one randomised order.'),
        intervals=PARENT['intervals'], dt=PARENT['dt'],
        families=[dict(name='incumbent', nu_lo=LV.INCUMBENT_NU[0], nu_hi=LV.INCUMBENT_NU[1]),
                  dict(name='lowvisc', nu_lo=LV.LOWVISC_NU[0], nu_hi=LV.LOWVISC_NU[1])],
        eval_seed=PARENT['eval_seed'], eval_cases=PARENT['eval_cases'],
        eval_fresh_seed=PARENT['eval_fresh_seed'], eval_fresh_cases=PARENT['eval_fresh_cases'],
        expected_incumbent_cases=LV.cohort(PARENT, *LV.INCUMBENT_NU).tolist(),
        expected_incumbent_cases_note=(
            'values, not a hash: np.exp differs by one ulp between the GB10 that wrote this file '
            'and the cluster NumPy that will recompute it, so the gate compares to 4e-16 relative '
            'and reports bitwise equality beside it as a probe'),
        reference_mesh=PARENT['reference_mesh'], reference_dt=PARENT['reference_dt'],
        train_seed=PARENT['train_seed'], train_trajectories=PARENT['train_trajectories'],
        train_state_stride=PARENT['train_state_stride'],
        snapshot_ntol=PARENT['snapshot_ntol'], snapshot_ltol=PARENT['snapshot_ltol'],
        snapshot_residual_bar=1e-8,
        pod_ranks=ranks,
        fom_settings=PARENT['fom_settings'],
        same_grid_reference=PARENT['same_grid_reference'],
        reps=PARENT['reps'], order_seed=20260917, burn_seconds=PARENT['burn_seconds'],
        mesh_probe=dict(meshes=[512, 1024], settings=['fft_tight', 'nt1e-3_dt005'],
                        families=['incumbent', 'lowvisc']),
        pod_probe_meshes=[512],
        pod_probe_note=('the whole leg-(a) measurement repeated at 512 intervals for both '
                        'families, so a degradation measured at 256 can be separated from the '
                        'first-order-upwind numerical diffusion of that grid; it runs last and '
                        'is guarded, so a failure there voids nothing above it. 1024 is left '
                        'out: the snapshot matrix would be 27.9 GB on device.'),
        comparator_source=f'b-panel job {JOB}, reports/summary.json, family=pod metric=best_found_percent',
        comparator_pod_best_found_percent=pod_floor,
        comparator_tolerance=1e-3,
        comparator_fom_median_gpu_ms=fom_ms,
        comparator_fom_worst_evolved_percent=fom_err,
        comparator_fom_note=('recorded for the report only: b-panel ran on its own allocation, so '
                             'no cost of this job may be divided by one of those. The incumbent '
                             'family is re-timed HERE and every cost ratio this lane states is '
                             'between two subjects of this job.'),
        leg_a_rank=512, leg_a_bar=2.0, leg_b_bar=1.5)
    (HERE / 'config-gate.json').write_text(json.dumps(cfg, indent=2) + '\n')

    smoke = dict(cfg)
    smoke.update(attempt='smoke', intervals=64, reference_mesh=256, reference_dt=0.00125,
                 train_trajectories=8, pod_ranks=[4, 8, 16], reps=1, burn_seconds=0.0,
                 expected_incumbent_cases=None, comparator_pod_best_found_percent=None,
                 leg_a_rank=16,
                 mesh_probe=dict(meshes=[128], settings=['fft_tight'],
                                 families=['incumbent', 'lowvisc']),
                 pod_probe_meshes=[128],
                 purpose='local 64-interval smoke of the same driver; validates no number')
    (HERE / 'config-smoke64.json').write_text(json.dumps(smoke, indent=2) + '\n')
    print('pod comparator floors:', json.dumps(pod_floor, indent=1))
    print('wrote config-gate.json, config-smoke64.json')


if __name__ == '__main__':
    main()
