"""Write configs/c3grid.json from the committed checks/FROZEN-SELECTION.json '3d_rules' (DESIGN 3.3, A10.3): every 3D
setting with status 'selected' gets its deployed rule; a setting without one is left out and listed. The config records
the frozen file's sha256 and the rule-selection label (pre-registered or POST HOC, A10) so every grid result carries it.

    python make_c3grid_config.py
"""
import hashlib
import json
from pathlib import Path

import make_configs as MC

HERE = Path(__file__).resolve().parent


def main():
    fp = HERE / 'checks' / 'FROZEN-SELECTION.json'
    fz = json.loads(fp.read_text())
    r3 = fz['3d_rules']
    sets, missing = [], []
    for s in MC.SETTINGS3:
        e = r3['settings'].get(s['name'])
        if not e or e['status'] != 'selected' or not e.get('deployed'):
            missing.append(s['name'])
            continue
        sets.append(dict(s, rule=e['deployed'], converged=e['converged']))
    names = [s['name'] for s in sets]
    assert sets, 'no setting has a deployed rule'
    c = MC.c3('c3grid', 'grid', sets)
    if 'W512' not in names:
        assert 'M2_512' in names, 'neither common baseline has a deployed rule'
    c.update(frozen_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(), rule_selection_label=r3.get('label', 'pre-registered'),
             rule_selection_posthoc=bool(r3.get('posthoc')), settings_without_rule=missing)
    (HERE / 'configs' / 'c3grid.json').write_text(json.dumps(c, indent=1) + '\n')
    print('c3grid', names, 'missing', missing, c['rule_selection_label'])


if __name__ == '__main__':
    main()
