"""fp64-query: job configs = the b2d-query-speed final configs (src/configs/ftest64.json, ffresh64.json, byte-identical
copies) with only these changes (PROTOCOL.md section 4):
  - widths restricted to 96, 128, 176, 192, 256, 384 (width 512 dropped; not in the rebuttal claim);
  - every frozen "qs" arm (accurate, fast) kept unchanged as the mixed-precision control, plus a copy with
    qs.stencil32 = False (the solver's own float64 path), tag '_f32' -> '_f64', role '<role>_fp64', require_fp64;
  - precision_audit on; the batched-throughput phase removed (latency only);
  - attempt name; for the rehearsal, the six dev cases and two widths.
    python make_fp64_config.py   ->  configs/p64test.json, configs/p64fresh.json, configs/p64dev.json
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTHS = [96, 128, 176, 192, 256, 384]


def convert(src, attempt, widths=WIDTHS):
    cfg = json.loads((HERE / 'src/configs' / src).read_text())
    fz = json.loads((HERE / 'src/frozen.json').read_text())
    assert cfg['frozen'] == fz
    arms = []
    for a in cfg['arms']:
        if int(a['Rp']) not in widths:
            continue
        arms.append(a)
        if a.get('impl') == 'qs':
            assert a['qs'].get('stencil32') is True and '_f32' in a['tag'], a      # every frozen setting is mixed
            b = copy.deepcopy(a)
            b['qs']['stencil32'] = False
            b['tag'] = a['tag'].replace('_f32', '_f64')
            b['role'] = a['role'] + '_fp64'
            b['require_fp64'] = True
            b['control_of'] = a['tag']
            arms.append(b)
    roles = sorted({(a['Rp'], a['role']) for a in arms})
    cfg.update(attempt=attempt, arms=arms, precision_audit=True,
               purpose='fp64-query: frozen b2d-query-speed settings in float64 next to the mixed-precision originals, '
                       'baseline and operators in one allocation (fp64-query PROTOCOL.md)')
    cfg.pop('batch', None)
    return cfg, roles


def main():
    out = HERE / 'configs'
    out.mkdir(exist_ok=True)
    for src, att in (('ftest64.json', 'p64test'), ('ffresh64.json', 'p64fresh')):
        cfg, roles = convert(src, att)
        (out / f'{att}.json').write_text(json.dumps(cfg, indent=1) + '\n')
        print(att, len(cfg['arms']), 'arms', roles)
    cfg, roles = convert('ftest64.json', 'p64dev', widths=[176, 384])
    cfg.update(eval_draws=[[7090702, 4], [911702, 2]], expected_physical_sha256=None,
               disjoint_draws=[[20260916, 64], [2026111402, 64]],
               cohort_name='dev6 = params_draw(7090702,4) ++ params_draw(911702,2) (first six dev38 cases)',
               purpose='fp64-query rehearsal on six dev cases (pipeline check only; not a reported result)')
    (out / 'p64dev.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print('p64dev', len(cfg['arms']), 'arms', roles)


if __name__ == '__main__':
    main()
