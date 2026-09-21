"""Write the 256^2 and 512^2 family configs (DESIGN s.8). Rules fixed before any 256^2 number exists.

fam256 (`python make_mesh_configs.py 256`):
- the passed gate is attempt 3 (sigmoid), but the 128^2 sweep (fam128a) ran swish, so the 256^2 job runs its own
  sigmoid mini-sweep at K = 16 on the tuning subset. Candidates = the gate recipe (ic / per-feature) and the two
  swish-sweep variants with the best 128^2 TUNING score (zero reference with per-feature or global scaling);
- encoder width capped at M1 = 4096 (published M1 = 2n needs 135 GB of weights+grad+Adam at 256^2: kept as an
  arm so the precheck records it, never attempted, excluded from selection);
- wall 2400 s per arm (~1000 epochs at the measured 128^2 M1=4096 epoch cost x4), max_epochs 10000 as in the gate;
- finals K = 8/16/32 of the selected variant with the 128^2 HR grid (exploratory; HR gate failed), and one
  data-matched arm: K = 16, 576 fit trajectories (= the frozen bank's training count), wall 7200 s.
fam512 (`python make_mesh_configs.py 512`, after fam256's SELECTED line exists): the fam256-selected variant at
  K = 8/16/32, no sweep, same HR grid, published-M1 precheck arm; wall 4800 s per arm.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
HR = [[32, 48, 'gappy'], [64, 96, 'gappy'], [128, 192, 'gappy'], [0, 64, 'colloc'], [0, 256, 'colloc'], [0, 1024, 'colloc']]
BASE = dict(K=16, ref='ic', scale='feature', b=100, db=10, M1=4096, lr=0.001, patience=10, seed=0, max_epochs=10000,
            wall=2400, dtype='float32', act='sigmoid', hr=[])
COMMON = dict(pod_ks=[8, 16, 32, 64, 128], rom_qs=[0, 256], gn_cap=20, hr_residual_cases=16, timing=dict(reps=5, cases=6, burn=.25))


def fam256():
    sweep = [dict(BASE, name='sig_K16_ic_feature'),
             dict(BASE, ref='zero', name='sig_K16_zero_feature'),
             dict(BASE, ref='zero', scale='global', name='sig_K16_zero_global'),
             dict(BASE, M1='2n', select=False, name='sig_K16_published_M1')]
    cfg = dict(intervals=256, variants=sweep, finals=dict(Ks=[8, 16, 32], overrides=dict(hr=HR),
                                                          data_matched=dict(K=16, fit_traj=576, wall=7200)), **COMMON)
    return cfg


def fam512():
    s = json.loads((LANE / 'runs/fam256/output/summary.json').read_text()) if (LANE / 'runs/fam256/output/summary.json').exists() else None
    if s is None:   # job still running: read the SELECTED variant from the live summary copied by the caller
        s = json.loads(Path(sys.argv[2]).read_text())
    bv = dict(s['selection']['selected_variant'])
    vs = [dict(bv, K=K, wall=4800, hr=HR, timed=True, name=f'sig_K{K}_sel') for K in (8, 16, 32)]
    vs.append(dict(bv, K=16, M1='2n', hr=[], select=False, name='sig_K16_published_M1'))
    return dict(intervals=512, variants=vs, provenance=dict(selected_at_256=s['selection']['selected'], source_job=s['job_id']), **COMMON)


L = int(sys.argv[1])
cfg = fam256() if L == 256 else fam512()
(HERE / f'fam{L}.json').write_text(json.dumps(cfg, indent=1) + '\n')
print(L, [(v['name'], v['M1'], v['wall']) for v in cfg['variants']])
