"""jcp-robust E1 final verdicts with the registered precedence (DESIGN A1.4 as enumerated by A2.4). CPU only, no JAX.

Inputs: checks/e1-2d-{256,1024,4096}.json (e1analyze.py) and checks/e1-3d.json (e1analyze3d.py). Precedence, first
applicable wins, per (dimension, configuration, mesh):
  1 job integrity (complete, backend gpu, registered population, offline/in-job distance agreement, G-cert)  -> invalid job
  2 must-fail controls of the dimension: C-E1.1 (2D: established and fired at 1024^2, applies to every 2D mesh; 3D: on
    E1-3D-a, applies to both 3D configurations) and C-E1.2                                                -> blocked (control)
  3 positive gates: G-E1.1 (identity), G-E1.2 (reproduction of the configuration), G-E1.3 (permuted batching; dimension)
                                                                                                          -> blocked (gate)
  4 missing evidence                                                                                      -> unavailable
  5 the registered rule (H-E1a, H-E1b, H-E1c; indicator table descriptive)
GAL-CN rows are diagnostic (no verdict). Writes checks/e1-verdicts.json.

    python e1verdicts.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    p = HERE / 'checks' / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = dict(dimension={'2D': {}, '3D': {}}, rows=[])
    meshes = {m: load(f'e1-2d-{m}.json') for m in (256, 1024, 4096)}
    m1024 = meshes[1024]
    ctl2 = (m1024 or {}).get('groups', {}).get('R128', {}).get('controls', {}) if m1024 else {}
    c11 = ctl2.get('C-E1.1', {})
    c12 = ctl2.get('C-E1.2', {})
    perm2 = [((meshes[m] or {}).get('groups', {}).get('R128', {}).get('controls', {}).get('G-E1.3') or {}).get('passed')
             for m in meshes]
    g11 = [((meshes[m] or {}).get('groups', {}).get('R128', {}).get('controls', {}).get('G-E1.1') or {}).get('passed')
           for m in meshes]
    out['dimension']['2D'] = dict(C_E1_1=c11, C_E1_2=c12, G_E1_3=perm2, G_E1_1=g11)
    dim_block2 = None
    if not c11.get('fired'):
        dim_block2 = 'blocked (control C-E1.1: ' + ('unavailable' if not c11.get('admitted') else 'did not fire') + ')'
    elif not c12.get('fired'):
        dim_block2 = 'blocked (control C-E1.2 did not fire)'
    elif not all(x is True for x in g11):
        dim_block2 = 'blocked (gate G-E1.1)'
    elif not all(x is True for x in perm2):
        dim_block2 = 'blocked (gate G-E1.3)'
    for m, rep in meshes.items():
        if rep is None:
            for arm in ('LSPG_BE_1', 'LSPG_BE_2', 'LSPG_BE_2.5', 'LSPG_CNR_1'):
                out['rows'].append(dict(dim='2D', mesh=m, config=arm, label='unavailable (no analysis)'))
            continue
        for ak, av in rep['groups'].get('R128', {}).get('arms', {}).items():
            row = dict(dim='2D', mesh=m, config=f'R128|fib4181|{ak}')
            if av.get('diagnostic'):
                row.update(label='diagnostic (GAL-CN): no verdict', H_E1=None, descriptive=av.get('verdicts'))
            elif not (rep['complete'] and rep['backend'] == 'gpu' and av['integrity']['ok']):
                row.update(label='invalid job', integrity=av['integrity'])
            elif dim_block2:
                row.update(label=dim_block2)
            elif not (av.get('G-E1.2') or {}).get('passed'):
                row.update(label='blocked (gate G-E1.2)', reason=(av.get('G-E1.2') or {}).get('reason'))
            elif not isinstance(av.get('verdicts'), dict) or 'H_E1a' not in av['verdicts']:
                row.update(label='unavailable')
            else:
                v = av['verdicts']
                row.update(label='verdict', H_E1a=v['H_E1a'], H_E1b=v['H_E1b'], H_E1c=v['H_E1c'], indicator=v['indicator'],
                           eligible_cases=v['eligible_cases'])
            out['rows'].append(row)
    r3 = load('e1-3d.json')
    if r3 is None:
        out['rows'] += [dict(dim='3D', config=c, label='unavailable (no analysis)') for c in ('E1-3D-a', 'E1-3D-b')]
    else:
        a = r3['configs'].get('E1-3D-a', {}).get('controls', {})
        c11, c12 = a.get('C-E1.1', {}), a.get('C-E1.2', {})
        out['dimension']['3D'] = dict(C_E1_1=c11, C_E1_2=c12, G_E1_3=a.get('G-E1.3'))
        blk = None
        if not c11.get('fired'):
            blk = 'blocked (control C-E1.1: ' + ('unavailable' if not c11.get('admitted') else 'did not fire') + ')'
        elif not c12.get('fired'):
            blk = 'blocked (control C-E1.2 did not fire)'
        elif not (a.get('G-E1.3') or {}).get('passed'):
            blk = 'blocked (gate G-E1.3)'
        supp_names = {c['name'] for c in r3.get('config_supplementary', [])}
        for nm, cv in r3['configs'].items():
            row = dict(dim='3D', config=f"{nm}|{cv['setting']}|{cv['spec']}", supplementary=nm in supp_names)
            ctl = cv.get('controls', {})
            if not (r3['backend'] == 'gpu' and cv['integrity']['ok']):
                row.update(label='invalid job', integrity=cv['integrity'])
            elif blk:
                row.update(label=blk)
            elif not (ctl.get('G-E1.1') or {}).get('passed'):
                row.update(label='blocked (gate G-E1.1)')
            elif not (ctl.get('G-E1.2') or {}).get('passed') and nm not in supp_names:
                row.update(label='blocked (gate G-E1.2)', reason=(ctl.get('G-E1.2') or {}).get('reason'))
            else:
                v = cv['verdicts']
                row.update(label='verdict', H_E1a=v['H_E1a'], H_E1b=v['H_E1b'], H_E1c=v['H_E1c'], indicator=v['indicator'],
                           eligible_cases=v['eligible_cases'])
            out['rows'].append(row)
    (HERE / 'checks' / 'e1-verdicts.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    for r in out['rows']:
        print(r['dim'], r.get('mesh', ''), r['config'], r['label'],
              {k: r[k]['label'] for k in ('H_E1a', 'H_E1b', 'H_E1c') if isinstance(r.get(k), dict)})


if __name__ == '__main__':
    main()
