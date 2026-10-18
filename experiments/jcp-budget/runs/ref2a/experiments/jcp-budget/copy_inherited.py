"""One-off: copy inherited code byte-identically into this lane and record provenance (PROVENANCE.json)."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees')
SRC = {
    'qualify': (WT / '2026-10-18-jcp-qualify', 'experiments/jcp-qualify', 'exp/2026-10-18-jcp-qualify'),
    'references': (WT / '2026-10-06-jcp-references', 'experiments/jcp-references', 'exp/2026-10-06-jcp-references'),
}
FILES = [  # (source key, source relative path, destination relative path)
    *[('qualify', f, f) for f in ('p1core.py', 'p1common.py', 'qcert.py', 'npcheck.py', 'ledger.py', 'resources.py',
                                   'timing.py', 'c3common.py', 's3common.py', 's3ctx.py', 'fs3.py', 'fs3p.py', 'qn3.py',
                                   't2core.py', 'cluster/stage.py', 'cluster/submit.sh', 'cluster/collect.py',
                                   'inputs/modes_F2048.json', 'inputs/model_W1024/bank.pkl',
                                   'inputs/model_W1024/training.json', 'rules3d/rules_extra.npz')],
    *[('references', f, 'ref/' + f) for f in ('refcore.py', 'verdict.py', 'cohorts.py', 'pscore.py', 'gates.py', 'sealio.py')],
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


import os
OLD = {e['file']: e for e in json.loads((HERE / 'PROVENANCE.json').read_text())} if (HERE / 'PROVENANCE.json').exists() else {}
prov = []
for key, rel, dst in FILES:
    root, lane, branch = SRC[key]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    s = root / lane / rel
    blob = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{lane}/{rel}'])
    assert blob == s.read_bytes(), f'{s} differs from its committed blob'
    d = HERE / dst
    if dst in OLD:
        prov.append(OLD[dst])
        continue
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(s, d)
    prov.append(dict(file=dst, source=str(s), source_branch=branch, source_commit=commit, sha256=sha(d),
                     status='copied unchanged (edits, if any, are recorded below with the new sha256)'))
(HERE / 'PROVENANCE.json').write_text(json.dumps(prov, indent=1) + '\n')
print(len(prov), 'files')
