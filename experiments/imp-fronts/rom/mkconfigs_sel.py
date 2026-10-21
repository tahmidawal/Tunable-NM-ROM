"""imp-fronts §B4 selection configs for new banks (DESIGN B4, Amendment 1 B4: jcp-robust E2 procedure and select2d.py
unchanged). Separate-file twin of make_configs_e2.sel, which casts the bank label to int and so cannot name an arm bank:
the groups, candidates, successors, target, check, arms and case set are imported from the unchanged make_configs_e2,
the bank block is read from inputs/banks/s<label>/BANK.json through the C-B1 guard (bankguard.verify).

    python mkconfigs_sel.py <label> [--check]        -> configs/e2s<label>.json   (select2d.py <out> <label> reads it back)
--check: for label 1 the output must equal the copied configs/e2s1.json byte for byte (the equivalence test).
"""
import json
import sys
from pathlib import Path

import bankguard as BG
from make_configs_e2 import CANDIDATES, EDGES, RANKS, SEL_ARMS

HERE = Path(__file__).resolve().parent


def sel_config(label):
    groups = [dict(name=f'R{r}', Rp=r, target='gauss384', check='gauss320', check_cases='dev6', tighter_cases='dev6',
                   arms=SEL_ARMS, rules=[dict(label=c, rule=c, succ=s, level='full') for c, s in CANDIDATES]) for r in RANKS]
    c = dict(attempt=f'e2s{label}', driver='r2acc', mesh=1024, edges=EDGES, cases='all38', groups=groups)
    v = BG.verify({'dir': f'inputs/banks/s{label}'}, HERE)
    meta = json.loads((HERE / v['bank_json']).read_text())
    c['bank'] = dict(ckpt=v['ckpt'], ckpt_sha256=v['ckpt_sha256'], rotation=v['rotation'], rotation_sha256=v['rotation_sha256'],
                     seed=int(meta['seed']))
    if 'label' in meta:
        c['bank']['label'] = meta['label']
    return c


def text(c):
    return json.dumps(c, indent=1) + '\n'


def main():
    label = sys.argv[1]
    t = text(sel_config(label))
    p = HERE / 'configs' / f'e2s{label}.json'
    if '--check' in sys.argv:
        assert p.read_text() == t, f'{p.name} differs from the regenerated config'
        print('identical', p.name)
        return
    assert label != '1', 'configs/e2s1.json is a byte-identical copy (K1 selection is taken from jcp-robust, not rerun)'
    p.write_text(t)
    print(p)


if __name__ == '__main__':
    main()
