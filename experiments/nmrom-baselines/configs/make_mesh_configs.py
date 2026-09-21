"""Write configs/fam256.json and configs/fam512.json from the 128^2 selection (runs/fam128a/output/summary.json).

Rules fixed before the 128^2 result was read (HANDOFF / DESIGN s.7 item 4):
- every Kim hyper-parameter = the variant selected on the 128^2 tuning subset; only K changes (8/16/32);
- encoder width capped at M1 <= 4096 above 128^2 (the published M1 = 2n does not fit; that arm is kept in the
  config and is recorded by family.py's device-memory precheck, never attempted);
- wall budgets are scaled so every arm gets at least the epoch count the selected 128^2 arm got
  (epoch cost measured at 128^2 for M1 = 4096 and the selected width, scaled by n and data size);
- 256^2 only: one data-matched arm, K = 16, 576 fit trajectories (= the frozen bank's 576 training trajectories),
  from the protocol's future train prefixes (train indices 128..591);
- HR (exploratory; HR gate failed) grid = the 128^2 finals' grid, selected per mesh on the tuning subset;
- no finals block: each variant is registered for timing directly.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
S = json.loads((LANE / 'runs/fam128a/output/summary.json').read_text())
sel = S['selection']['selected']
bv = dict(S['selection']['selected_variant'])
hr = S['config']['finals']['overrides']['hr']
tr = S['training']
n128 = S['n']
ep_sel = tr[sel]['epochs']


def sec_per_epoch_at(n, M1, fit_traj=112):
    """Scale the measured 128^2 epoch cost of the M1=4096 sweep arm (encoder-dominated) by n and data size."""
    ref = tr.get('kim_K16_M1_4096') or tr[sel]
    return ref['seconds_per_epoch'] * (n * M1) / (n128 * ref['M1']) * fit_traj / 112


def mesh(L, gpu_note, data_matched):
    n = (L - 1) ** 2
    M1 = 4096 if bv['M1'] == '2n' else min(int(bv['M1']), 4096)
    variants = []
    for K in (8, 16, 32):
        wall = max(1200, int(1.2 * ep_sel * sec_per_epoch_at(n, M1)))
        variants.append(dict(bv, K=K, M1=M1, wall=wall, hr=hr, timed=True, name=f'kim_K{K}_sel'))
    if data_matched:
        wall = max(1200, int(1.2 * ep_sel * sec_per_epoch_at(n, M1, 576)))
        variants.append(dict(bv, K=16, M1=M1, wall=wall, hr=[], timed=True, fit_traj=576, name='kim_K16_sel_fit576'))
    variants.append(dict(bv, K=16, M1='2n', wall=1200, hr=[], timed=False, name='kim_K16_published_M1'))
    for v in variants:
        v['max_epochs'] = max(int(v['max_epochs']), 3000)
    cfg = dict(intervals=L, pod_ks=[8, 16, 32, 64, 128], rom_qs=[0, 256], gn_cap=20, hr_residual_cases=16, variants=variants,
               timing=dict(reps=5, cases=6, burn=.25),
               provenance=dict(selected_at_128=sel, source=str((LANE / 'runs/fam128a/output/summary.json').relative_to(LANE)),
                               source_job=S['job_id'], selected_epochs_128=ep_sel, gpu=gpu_note))
    (HERE / f'fam{L}.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print(L, [(v['name'], v['M1'], v['wall']) for v in variants])


mesh(256, 'a100-80G', True)
mesh(512, 'h200 --mem 240G', False)
