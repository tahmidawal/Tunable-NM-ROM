"""Prove which files this lane inherits byte-for-byte from `no-second` at its fork point.

    python check_inherited.py            # writes checks/inherited-sources.json

DESIGN 2 claims a file list is identical to `experiments/no-second/` at commit ea812685 and
that four named files differ in stated ways. This regenerates that claim from the Git blobs
instead of trusting the prose, and fails if an unlisted file differs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

FORK = 'ea812685'
LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[1]
SOURCE = 'experiments/no-second'
EXPECTED_CHANGED = {'DESIGN.md', 'families.py', 'model.py', 'smoke_second.py', 'training_smoke_second.py',
                    'worker_second.py', 'cluster/stage.py'}
EXPECTED_NEW = {'DESIGN.md', 'check_inherited.py', 'specs/don01.json',
                'configs/deeponet/small.json', 'configs/deeponet/medium.json', 'configs/deeponet/large.json'}


def blob(path):
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'show', f'{FORK}:{SOURCE}/{path}'],
                              check=True, capture_output=True).stdout
    except subprocess.CalledProcessError:
        return None


def main():
    identical, changed, new = {}, {}, {}
    for path in sorted(LANE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        name = str(path.relative_to(LANE))
        if name.split('/')[0] in ('runs', 'reports', 'checks'):
            continue
        mine = path.read_bytes()
        theirs = blob(name)
        digest = hashlib.sha256(mine).hexdigest()
        if theirs is None:
            new[name] = digest
        elif theirs == mine:
            identical[name] = digest
        else:
            changed[name] = dict(sha256=digest, fork_sha256=hashlib.sha256(theirs).hexdigest())
    problems = [f'undeclared change: {n}' for n in changed if n not in EXPECTED_CHANGED]
    problems += [f'undeclared new file: {n}' for n in new if n not in EXPECTED_NEW]
    problems += [f'declared changed but identical: {n}' for n in EXPECTED_CHANGED if n in identical]
    result = dict(fork_commit=FORK, source=SOURCE, identical=identical, changed=changed,
                  new=new, problems=problems, passed=not problems)
    (LANE / 'checks').mkdir(exist_ok=True)
    (LANE / 'checks/inherited-sources.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(identical=len(identical), changed=sorted(changed), new=sorted(new),
                          problems=problems, passed=result['passed']), indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
