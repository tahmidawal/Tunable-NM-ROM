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
    e2bank.use(res['config'].get('bank'), (res.get('checkpoint_sha256'), res.get('rotation_sha256')))
    smoke = str(res['config'].get('attempt', '')).startswith('smk')
    rep = dict(source=str(out), bank=b, job_id=res['job_id'], commit=res['commit'], selected={}, cells={})
    ranks = RANKS if not smoke else tuple(int(x['Rp']) for x in res['config']['groups'])
    cand_all = [c for c, _ in CANDIDATES]
    for r in ranks:
        g = [x for x in res['config']['groups'] if x['Rp'] == r][0]
        labels = [x['label'] for x in g['rules']]
        if smoke:      # smoke contract: an ordered subset of the registered candidates (labelled smoke, never a selection)
            assert labels == [c for c in cand_all if c in labels], 'smoke candidates out of the registered order'
        else:
            assert labels == cand_all, 'candidate list differs from the registered one'
        cells, tcert, integ = {}, True, True
        for arm in ('LSPG_BE_1', 'LSPG_CNR_1'):
            av = EA.analyse_arm(out, res, g, arm, r)
            tcert = tcert and av['target']['certified'] and bool(av['target']['all_valid'])
            integ = integ and av['integrity']['ok']
            cells[arm] = {l: dict(qualified=av['rules'][l]['qualified'], own=av['rules'][l]['own_family_qualified'],
                                  status=av['rules'][l]['status'], D=av['rules'][l]['D'],
                                  median_e=float(np.median(av['rules'][l]['e'])), worst_e=float(np.max(av['rules'][l]['e'])))
                          for l in labels}
            cells[arm]['_integrity'] = av['integrity']
        key = 'qualified' if tcert else 'own'
        pick, outcome = None, 'none within ladder'
        if not integ:
            outcome = 'unavailable (integrity: distance / G-cert / population check failed)'
        else:
            for c in labels:                           # three-state (code audit 1 #10): undetermined evidence stops the search
                v = [cells[a][c][key] for a in ('LSPG_BE_1', 'LSPG_CNR_1')]
                if all(x is True for x in v):
                    pick = c
                    outcome = 'selected' if tcert else 'diagnostic pick (target unverified)'
                    break
                if any(x is None for x in v):
                    outcome = f'unavailable (evidence for {c} undetermined)'
                    break
        rep['selected'][str(r)] = dict(rule=pick, outcome=outcome, smoke=smoke, target_certified=tcert, deployed=DEPLOYED[r],
                                       equals_deployed=pick == DEPLOYED[r])
        rep['cells'][str(r)] = cells
        print(r, rep['selected'][str(r)], flush=True)
    (HERE / 'checks').mkdir(exist_ok=True)
    (HERE / 'checks' / f"select2d-{b}{'-smoke' if smoke else ''}.json").write_text(json.dumps(rep, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
