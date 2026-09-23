"""Held-out configs from the frozen validation selection (DESIGN.md section 7): the selected accurate and fast arms of
each mesh, the whole FOM grid, the sealed cohort 923401 (32 cases). Refuses unless selection.json exists and is
committed."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(R):
    sel_path = HERE / 'selection.json'
    sel = json.loads(sel_path.read_text())
    rel = 'experiments/burgers3d-span/selection.json'
    root = HERE.parents[1]
    committed = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{rel}'])
    assert committed == sel_path.read_bytes(), 'selection.json must be committed before the held-out configs'
    for n, s in sel['meshes'].items():
        val = json.loads((HERE / 'configs' / f'val_R{R}A3_n{n}.json').read_text())
        arms = [s['accurate_spec']] + ([s['fast_spec']] if s['fast_spec'] and s['fast'] != s['accurate'] else [])
        c = {k: v for k, v in val.items() if k not in ('arms', 'parity', 'audit_arms', 'cohort_seed', 'cohort_count')}
        c.update(arms=[{k: v for k, v in a.items() if k in ('kind', 'rule', 'Rp', 'K', 'gtol', 'solver', 'dt')} for a in arms],
                 cohort_seed=923401, cohort_count=32, parity=dict(arm=None),
                 audit_arms=[s['accurate'], s['fast'], 'fom_dt0.005_nt0.001_lt0.5'], sealed=True,
                 selection_sha256=sel['sha256_note'] if 'sha256_note' in sel else None)
        (HERE / 'configs' / f'heldout_R{R}_n{n}.json').write_text(json.dumps(c, indent=1) + '\n')
        print(n, [a for a in c['arms']])


if __name__ == '__main__':
    main(int(sys.argv[1]))
