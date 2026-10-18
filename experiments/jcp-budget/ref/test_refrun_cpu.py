"""Local CPU functional test of refrun.py (reduced levels; never a reference): fd mode at T = 0.5 on dev6 case 0 with
levels L = 256/512/1024, fine lattice = 256 (patched), plus x1 at reduced levels. Checks the files, the per-window
judges and G-R3 run end to end.  JAX_PLATFORMS=cpu python ref/test_refrun_cpu.py <outdir>"""
import json
import sys
from pathlib import Path

import refrun as RR
import refcore as RC

out = Path(sys.argv[1])
RR.LEVELS[2] = dict(base=[(256, 0.01), (512, 0.005), (1024, 0.0025)], refine=(1536, 0.05 / 30))
RR.X1_LEVELS[2] = [(128, 0.01), (256, 0.005), (512, 0.0025)]
RC.FINE[2] = 256
for mode in ('fd', 'x1'):
    cfg = dict(mode=mode, dim=2, cohort='dev6', cases=[0], horizon=0.5, fine=True, allow_refine=False, allow_cpu=True,
               skip_g1=True)
    p = out / f'{mode}.json'
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg))
    sys.argv = ['refrun', '--config', str(p), '--out', str(out / mode)]
    RR.main()
r = json.loads((out / 'fd' / 'result.json').read_text())
print(json.dumps(r['cases'], indent=1))
m = json.loads((out / 'fd' / '2d' / 'dev6' / 'T0.5' / 'c000.json').read_text())
print('windows', {k: (v['outcome'], v['eps_ref']) for k, v in m['judge_windows'].items()})
print('fine windows', {k: (v['outcome'], v['eps_ref']) for k, v in (m['judge_fine_windows'] or {}).items()})
print('x1', json.loads((out / 'x1' / 'result.json').read_text())['x1'])
