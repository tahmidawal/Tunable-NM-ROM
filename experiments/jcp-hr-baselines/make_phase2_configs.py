"""Writes the phase-II tuning-job configurations (DESIGN 3.3, 6, A1.3, A1.4, A21 batch 1, A24.4):
configs/tune_2d256.json, configs/tune_2d1024.json, configs/tune_3d64.json, and configs/tune_cpu_{2d,3d}.json (local CPU
dry run at tiny size; never submitted)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT2 = [['BE', 0.005], ['CNR', 0.005]]
OPT3 = [[s, 0.05 / k] for s in ('BE', 'CN', 'BDF2') for k in (5, 4, 3, 2)]     # dt0 x {1, 1.25, 5/3, 2.5}, dt0 = 0.01
P1_256 = [f'256|{r}|{s}' for r in ('R128', 'R256', 'R384', 'R384nom') for s in ('BE', 'CNR')]
P1_1024 = [f'1024|{r}|{s}' for r in ('R128', 'R256', 'R384', 'R512') for s in ('BE', 'CNR')]
S3 = [f'{c}|fs|BE|0.01|K1' for c in ('M2_512|gl32', 'W256|gl24', 'W512|lat32768', 'W768|lat32768', 'W1024|lat65536')] + \
    ['W256|gl24|fs|CN|0.01|K1', 'W256|gl24|fs|CN|0.0125|K1', 'W256|lat8192|qn|BDF2|0.025|k4',     # A29: batch 2
     'W512|lat16384|qn|BDF2|0.025|k4']
R2, R3 = [32, 64, 128, 256, 384], [64, 128, 256, 512, 768, 1024]
# A25.2: ours' C-search GPU-hours (sacct ElapsedRaw x 1 GPU): 2D combined c2grid 5209526 (13,596 s) + c2fbgrid 5228777
# (658 s) = 3.959444 h, split evenly over the two 2D cells; 3D combined c3grid 5219449 (15,949 s) + speed3d g1-g3
# (1.7472 + 2.4092 + 3.6294 h, whole jobs: an upper bound on their grid phase) = 12.216 h.
A2, A3 = (13596 + 658) / 3600 / 2, 15949 / 3600 + 1.7472 + 2.4092 + 3.6294
WALL = dict(tune_2d256=2, tune_2d1024=5, tune_3d64=12)        # A26.4: Slurm wall limits (h), the hard GPU-h bound
cfgs = dict(
    tune_2d256=dict(d=2, L=256, ranks=R2, options=OPT2, saturation=True, ours=P1_256, n_traj=None, ext_rank=512,
                    search_allocation_hours=A2),
    tune_2d1024=dict(d=2, L=1024, ranks=R2, options=OPT2, saturation=False, ours=P1_1024, n_traj=None, ext_rank=512,
                     search_allocation_hours=A2),
    tune_3d64=dict(d=3, L=64, ranks=R3, options=OPT3, saturation=False, ours=S3, n_traj=None,
                   search_allocation_hours=A3),
    tune_cpu_2d=dict(d=2, L=256, ranks=[8, 16], options=OPT2, saturation=True, ours=['256|R128|BE'], n_traj=12, ext_rank=24,
                     allow_cpu=True, dry=True),
    tune_cpu_3d=dict(d=3, L=64, ranks=[8, 16], options=OPT3[:1] + OPT3[8:9], saturation=False, ours=S3[1:2] + S3[7:8], n_traj=12,
                     allow_cpu=True, dry=True))
for k, v in cfgs.items():
    if k in WALL:
        v.update(wall_hours=WALL[k], budget_hours=WALL[k] - 0.25,
                 ecsw_path_seconds=(2400 if v['d'] == 3 else 2700))      # A38.2: per-attempt NNLS wall budget
    (HERE / 'configs' / f'{k}.json').write_text(json.dumps(v, indent=1) + '\n')
    print(k, len(v['ranks']), len(v['options']), len(v['ours']))
