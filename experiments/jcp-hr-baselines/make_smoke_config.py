"""Writes configs/smoke.json (cluster, real size) and configs/smoke_cpu.json (local CPU dry run, tiny meshes).
N-under fixtures (A1.9): the two selection-subset cases with the smallest nu (ties -> larger amplitude), from the
cohort descriptors only (no outcome)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hrcore as H  # noqa: E402


def fixtures(d):
    if d == 2:
        D = H._raw_rows(2, 'dev6')
        nu, amp = D[:6, 4], D[:6, 3]
    else:
        T = H.C3.b3.draw_param_table(923801, 6)
        nu, amp = T['nu'], T['A']
    order = sorted(range(6), key=lambda j: (nu[j], -amp[j], j))
    return order[:2]                     # smallest nu first (N-grad uses the first, A1.9)


def cell(d, L, n_traj, rmax, ranks, pmax, rs, ps, n_adeq, n_probe, opts, pilot):
    return dict(d=d, L=L, n_traj=n_traj, rmax=rmax, ranks=ranks, pmax=pmax, rs=rs, ps=ps, kappas=[2, 4], kappa_run=4,
                cases=[0, 1], under_fixtures=fixtures(d), n_adequacy=n_adeq, n_probe=n_probe, options=opts,
                pilot_support=pilot)


real = dict(wall_seconds=3300, cells=[
    cell(2, 1024, 192, 512, [128, 512], 2048, 128, 256, 8, 4, [['BE', 0.005], ['CNR', 0.005]], 1024),
    cell(3, 64, 342, 1024, [256, 1024], 4096, 256, 512, 8, 4, [['BE', 0.01], ['CN', 0.01], ['BDF2', 0.025]], 1024)])
cpu = dict(wall_seconds=5000, allow_cpu=True, dry_skip_ours=True, cells=[
    cell(2, 256, 24, 48, [16, 48], 96, 16, 32, 4, 2, [["BE", 0.005], ["CNR", 0.005]], 64),
    cell(3, 64, 24, 48, [16, 48], 96, 16, 32, 4, 2, [["BE", 0.01], ["CN", 0.01], ["BDF2", 0.025]], 64)])
(HERE / 'configs').mkdir(exist_ok=True)
for nm, c in (('smoke', real), ('smoke_cpu', cpu)):
    (HERE / 'configs' / f'{nm}.json').write_text(json.dumps(c, indent=1))
print(json.dumps(real['cells'][0]['under_fixtures']), json.dumps(real['cells'][1]['under_fixtures']))
