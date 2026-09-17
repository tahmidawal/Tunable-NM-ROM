"""Extract the per-case cross-job fidelity references from the parent archives.

Reads only the two collected `result.json` files (pbh02 from p-bank-head, ccpoi01 from
cheap-corrections) and writes compact per-case error tables for the arms this cell
reproduces. Nothing is typed by hand.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WT = ROOT.parent
PBH02 = ROOT / 'experiments/p-bank-head/artifacts/pbh02/result.json'
CCPOI01 = WT / '2026-09-15-cheap-corrections/experiments/cheap-corrections/artifacts/ccpoi01/result.json'
OUT = ROOT / 'experiments/p-linear/references'

PBH02_ARMS = ['a_neural@incumbent', 'a_neural_q32@incumbent', 'd_freebank@incumbent',
              'a_neural@new_K32', 'a_neural_q32@new_K32',
              'e_pod32@trainset', 'e_pod64@trainset', 'e_pod128@trainset', 'dst_direct']
CCPOI01_ARMS = ['q0_m256', 'q32_m256', 'q64_m256', 'q64_m4', 'dst_direct']


def table(d, arms, keys):
    rows = []
    spread = 0.0
    for n in sorted({x['intervals'] for x in d['invocations']}):
        for name in arms:
            sel = [x for x in d['invocations'] if x['intervals'] == n and x['name'] == name]
            if not sel:
                continue
            cases = sorted({x['case'] for x in sel})
            entry = dict(intervals=n, name=name, cases=cases)
            for key in keys:
                vals = []
                for c in cases:
                    v = [x[key] for x in sel if x['case'] == c]
                    vals.append(v[0])
                    spread = max(spread, (max(v) - min(v)) / max(abs(v[0]), 1e-300))
                entry[key] = vals
            rows.append(entry)
    return rows, spread


def main():
    OUT.mkdir(exist_ok=True)
    for src, arms, keys, name in ((PBH02, PBH02_ARMS, ('physical_error', 'same_grid_error'), 'pbh02'),
                                  (CCPOI01, CCPOI01_ARMS, ('physical_error',), 'ccpoi01')):
        d = json.loads(src.read_text())
        assert d['complete']
        rows, spread = table(d, arms, keys)
        payload = dict(source=str(src.relative_to(WT)), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                       job_id=d['job_id'], gpu=d['gpu'], commit=d['commit'],
                       repetition_spread=spread, arms=rows,
                       note=(f'per-case errors of {name} at every mesh it ran, first repetition; '
                             'the largest relative spread across its repetitions is recorded'))
        (OUT / f'{name}-reference.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(name, 'arms', len(rows), 'spread', spread)


if __name__ == '__main__':
    main()
