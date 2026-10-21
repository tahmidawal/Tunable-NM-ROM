"""Write configs/panel_n{65,129,257}.json (+ smoke) from the Table 1 held-out configs (copied unchanged in
lanes/burgers3d-retry/configs) and the Table 1 held-out results (numbers pinned below, from
burgers3d-retry runs/ho{65,129,257}/archive/code/output/result.json, jobs 4253861/4253865/4253867)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = {65: ['fno-l', 'unet-l', 'tsol-l', 'don-s'], 129: ['fno-l', 'unet-l', 'tsol-l', 'don-s'],
       257: ['unet-s', 'tsol-s', 'don-s', 'fno-s']}
EXPECTED = json.loads((HERE / 'configs' / 'table1_heldout_expected.json').read_text())
for n in (65, 129, 257):
    c = json.loads((HERE / 'lanes' / 'burgers3d-retry' / 'configs' / f'heldout_n{n}.json').read_text())
    c.update(model='lanes/burgers3d-retry/inputs/model_M2', audit_arms=[], operators=OPS[n],
             expected_cohort_sha256=EXPECTED[str(n)]['cohort_sha256'],
             expected_nmrom_worst=EXPECTED[str(n)]['nmrom_worst'], table1_job=EXPECTED[str(n)]['job'],
             table1_reference_restricted=f'configs/table1_refs/ref_n{n}.npz')
    (HERE / 'configs' / f'panel_n{n}.json').write_text(json.dumps(c, indent=1) + '\n')
c = json.loads((HERE / 'lanes' / 'burgers3d-retry' / 'configs' / 'heldout_n65.json').read_text())
c.update(model='lanes/burgers3d-retry/inputs/model_M2', audit_arms=[], operators=['fno-s', 'unet-s', 'tsol-s', 'don-s'],
         mesh=33, cohort_count=2, reps=1, local_smoke=True, fom_grid=[[0.01, 0.01, 0.1], [0.025, 0.01, 0.5]],
         arms=[{'kind': 'span', 'Rp': 64, 'solver': 'fsc', 'dt': 0.01}])
(HERE / 'configs' / 'panel_smoke.json').write_text(json.dumps(c, indent=1) + '\n')
print('ok')
