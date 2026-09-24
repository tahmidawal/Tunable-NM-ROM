"""DESIGN §9: held-out configs from the frozen selection.json (accurate + fast arm per mesh, whole FOM grid, sealed
cohort 923901 x 32). Writes configs/heldout_n{n}.json."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    sel_p = HERE / 'selection.json'
    sel = json.loads(sel_p.read_text())
    sha = hashlib.sha256(sel_p.read_bytes()).hexdigest()
    for n, s in sel['meshes'].items():
        val = json.loads((HERE / 'configs' / f'val_n{n}.json').read_text())
        byname = {}
        for a in val['arms']:
            nm = f"span_R{a['Rp']}_{a['solver']}_dt{a['dt']:g}"
            byname[nm] = a
        arms = [byname[s['accurate']]] + ([byname[s['fast']]] if s['fast'] != s['accurate'] else [])
        cfg = dict(val, mode='panel', cohort_seed=923901, cohort_count=32, arms=arms, sealed=True, selection_sha256=sha,
                   audit_arms=[s['accurate'], s['fast'], s['fom']] if int(n) < 257 else [s['fast']])
        (HERE / 'configs' / f'heldout_n{n}.json').write_text(json.dumps(cfg, indent=1) + '\n')
        print(n, [x for x in arms], sha)


if __name__ == '__main__':
    main()
