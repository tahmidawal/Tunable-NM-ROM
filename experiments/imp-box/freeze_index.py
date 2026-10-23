"""Build the campaign freeze (A3.7, A6.7, A16): copy every retained cell's frozen.json to runs/frozen/<W>_<lev>.json, write
runs/frozen/INDEX.json (cells -> sha256, retained list) and runs/frozen/SOURCE-EQUIVALENCE.json (A16: for every query
source whose hash differs between a cell's freeze and the current tree, the frozen and current hashes). Commit the
result before any test job is staged."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import timing as TM   # noqa: E402

CELLS = {'W40_A1': 'W40_A1', 'W40_A2': 'W40_A2', 'W40_A3': 'W40_A3t', 'W28_A1': 'W28_A1', 'W28_A2': 'W28_A2',
         'W28_A3': 'W28_A3t', 'W08_A1': 'W08_A1', 'W08_A2': 'W08_A2', 'W16_A1': 'W16_A1'}


def main():
    fd = HERE / 'runs' / 'frozen'
    fd.mkdir(parents=True, exist_ok=True)
    cur = TM.source_identity()
    idx, eq = dict(cells={}, retained=[], archive={}), {}
    for cell, src in CELLS.items():
        f = HERE / 'runs' / 'cells' / src / 'out' / 'cell' / 'frozen.json'
        if not f.exists():
            print('missing freeze', cell)
            continue
        shutil.copy(f, fd / f'{cell}.json')
        idx['cells'][cell] = hashlib.sha256((fd / f'{cell}.json').read_bytes()).hexdigest()
        idx['retained'].append(cell)
        idx['archive'][cell] = src
        frz = json.loads(f.read_text())['source']
        for k in sorted(set(cur) | set(frz)):
            if cur.get(k) != frz.get(k):
                e = eq.setdefault(k, {'from': [], 'to': cur.get(k), 'amendment': 'A16 (constants as executable arguments)'})
                if frz.get(k) not in e['from']:
                    e['from'].append(frz.get(k))
    (fd / 'INDEX.json').write_text(json.dumps(idx, indent=1))
    (fd / 'SOURCE-EQUIVALENCE.json').write_text(json.dumps(eq, indent=1))
    print(json.dumps(dict(retained=idx['retained'], equivalence_files=sorted(eq))))


if __name__ == '__main__':
    main()
