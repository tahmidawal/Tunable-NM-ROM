"""jcp-robust E2 2D rule selection (DESIGN 3.2 as amended by A1.8, A2.5). CPU only.

For every r: the selected rule is the first candidate (ordered by m) that is qualified under BOTH LSPG-BE dt0 and
LSPG-CN-R dt0 by the full contract (Q1 + Q2 against its registered successor on every case + the cross-family check
against gauss384 on every case), all recomputed offline by e1analyze.analyse_arm. If the target is not certified on either
arm, the pick is a DIAGNOSTIC pick on Q1 + successor Q2 only (A2.5). "none within ladder" if no candidate qualifies.

    JAX_PLATFORMS=cpu python select2d.py <e2s output dir> <bank>
"""
import json
import sys
from pathlib import Path

import numpy as np

import e1analyze as EA
import e2bank
from make_configs_e2 import CANDIDATES, DEPLOYED, RANKS

HERE = Path(__file__).resolve().parent


def main():
    out, b = Path(sys.argv[1]), sys.argv[2]
    res = json.loads((out / 'result.json').read_text())
    assert res.get('complete'), 'selection job incomplete'
    e2bank.use(res['config'].get('bank'))
    rep = dict(source=str(out), bank=b, job_id=res['job_id'], commit=res['commit'], selected={}, cells={})
    for r in RANKS:
        g = [x for x in res['config']['groups'] if x['Rp'] == r][0]
        labels = [x['label'] for x in g['rules']]
        cells, tcert = {}, True
        for arm in ('LSPG_BE_1', 'LSPG_CNR_1'):
            av, _ = EA.analyse_arm(out, res, g['name'], arm, labels, r)
            tcert = tcert and av['target_cert']['certified']
            cells[arm] = {l: dict(qualified=av['draws'][l]['qualified'], own=av['draws'][l]['own_family_qualified'],
                                  status=av['draws'][l]['status'], D=av['draws'][l]['D'],
                                  median_e=float(np.median(av['draws'][l]['e'])), worst_e=float(np.max(av['draws'][l]['e'])),
                                  mismatches=len(av['draws'][l]['mismatches']))
                          for l in labels}
        key = 'qualified' if tcert else 'own'
        pick = next((c for c, _ in CANDIDATES if all(cells[a][c][key] for a in cells)), None)
        rep['selected'][str(r)] = dict(rule=pick, label=('selected' if tcert else 'diagnostic pick (target unverified)')
                                       if pick else 'none within ladder', target_certified=tcert,
                                       deployed=DEPLOYED[r], equals_deployed=pick == DEPLOYED[r])
        rep['cells'][str(r)] = cells
        print(r, rep['selected'][str(r)], flush=True)
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f'select2d-{b}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
