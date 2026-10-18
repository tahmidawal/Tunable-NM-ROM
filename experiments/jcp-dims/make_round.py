"""jcp-dims 3D rounds 5-8 (DESIGN 2.3 (3R), A1.7): the next round is qualify's front3d.round_next over speed3d's 436
candidates with every Q2-tested configuration removed unless it passed (qualify's 55 labels + this lane's round
labels from checks/rounds3d.json, written by the analysis after each round), ladders extended by lat262144. If the next
round is empty, the remaining rounds are waived in the ledger with that reason (never silently).

    python make_round.py <k>        -> configs/q3r<k>.json, or prints EMPTY
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import front3d as F3
import make_configs3d as MC

HERE = Path(__file__).resolve().parent
QUALIFY = MC.QUALIFY


def tested_labels():
    t = dict(json.loads((QUALIFY / 'runs' / 'EXPORT.json').read_text())['threeD']['tested'])
    p = HERE / 'checks' / 'rounds3d.json'
    assert p.exists(), 'predecessor round labels missing (run the analysis of the previous round first)'
    if p.exists():
        for k, v in json.loads(p.read_text())['labels'].items():
            t[k] = 'pass' if v == 'pass' else 'fail' if v == 'fail' else v      # unverified stays non-passing
    return t


def main():
    k = int(sys.argv[1])
    assert 5 <= k <= 8
    prev = json.loads((HERE / 'configs' / f'q3r{k - 1}.json').read_text())
    lab = json.loads((HERE / 'checks' / 'rounds3d.json').read_text())['labels']
    missing = [m['key'] for b in prev['blocks'] if b['kind'] == 'rounds' for m in b['members'] if m['key'] not in lab]
    assert not missing, ('previous round not analysed', missing)
    F3.LATTICE[:] = ['lat2048', 'lat4096'] + MC.LAT                 # successors include lat262144
    mem = F3.round_next(tested_labels(), limit=20)
    if not mem:
        print('EMPTY')
        return
    members = []
    for m in mem:
        qp, qpp = MC.successors(m['key'].split('|')[1])
        members.append(dict(key=m['key'], qp=qp, qpp=qpp))
    MC.job(f'q3r{k}', [MC.setting(n, n) for n in MC.BASE], [dict(kind='rounds', members=members, control=MC.CONTROL)],
           MC.GL + ['gl80'] + MC.LAT)
    print(f'q3r{k}', len(members))


if __name__ == '__main__':
    main()
