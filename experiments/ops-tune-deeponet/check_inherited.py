"""Prove which files this lane inherited unchanged, and that only the declared ones differ.

DESIGN.md section 3 makes two byte-identity claims: against the forked lane
`experiments/ops-deeponet-b2d` at `306c939d`, and against the six generator files the pinned
128-case cache's own index records as its provenance. Both are checked here against Git blobs,
not against whatever happens to be in a sibling worktree.

    python check_inherited.py --output checks/inherited-sources.json

Exit code is nonzero if any claim fails. Run it after every edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

LANE = Path(__file__).resolve().parent
REPO = LANE.parents[1]
FORK = '306c939d'
FORK_LANE = 'experiments/ops-deeponet-b2d'
GENERATOR = '5169c095'

# Claimed byte-identical to the forked lane at FORK.
INHERITED = ['dataset.py', 'model.py', 'evaluate_cohort.py', 'prepare_diagnosis_cohort.py',
             'spectral_conv_f64.py', 'NEURALOPERATOR-LICENSE', 'check_fno_parity.py',
             'check_burgers_error_definition.py', 'smoke_second.py', 'training_smoke_second.py']
# Claimed changed, and only in the ways DESIGN section 3 tabulates.
CHANGED = ['families.py', 'train.py', 'audit.py', 'check_inherited.py',
           'cluster/stage.py', 'cluster/collect.py']
# New in this lane; no counterpart in the forked lane.
NEW = ['gen_more.py', 'build_pool.py', 'worker_gen.py', 'worker_tune.py', 'compose.py',
       'make_fin_spec.py']
# Claimed byte-identical to the pinned generator at GENERATOR.
VENDORED = {'data.py': 'experiments/neural-operator-burgers/data.py',
            'refine.py': 'experiments/neural-operator-burgers/refine.py',
            'protocol.json': 'experiments/neural-operator-burgers/protocol.json',
            'protocol-refined.json': 'experiments/neural-operator-burgers/protocol-refined.json',
            'engines.py': 'experiments/mr-burgers2d/engines.py',
            'sep_common.py': 'experiments/separable-decoder/sep_common.py'}
# The forked lane's files this lane does NOT carry, and why.
DROPPED = {'timing.py': 'no speed number is admissible from this lane (DESIGN section 7)',
           'worker_second.py': 'replaced by worker_tune.py (DESIGN section 3)'}


def blob(commit, path):
    result = subprocess.run(['git', '-C', str(REPO), 'show', f'{commit}:{path}'],
                            capture_output=True)
    return result.stdout if result.returncode == 0 else None


def sha(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def main(args):
    report = dict(lane=str(LANE.relative_to(REPO)), fork=FORK, fork_lane=FORK_LANE,
                  generator_commit=GENERATOR, inherited={}, changed={}, new={}, vendored={},
                  dropped=DROPPED, failures=[])
    for name in INHERITED:
        mine = (LANE / name).read_bytes()
        theirs = blob(FORK, f'{FORK_LANE}/{name}')
        same = theirs is not None and mine == theirs
        report['inherited'][name] = dict(sha256=sha(mine), fork_sha256=sha(theirs), identical=same)
        if not same:
            report['failures'].append(f'{name} is claimed byte-identical to the fork and is not')
    for name in CHANGED:
        mine = (LANE / name).read_bytes()
        theirs = blob(FORK, f'{FORK_LANE}/{name}')
        report['changed'][name] = dict(sha256=sha(mine), fork_sha256=sha(theirs),
                                       identical=theirs is not None and mine == theirs)
        if theirs is not None and mine == theirs:
            report['failures'].append(f'{name} is declared changed but is byte-identical to the fork')
    for name in NEW:
        mine = (LANE / name).read_bytes()
        report['new'][name] = dict(sha256=sha(mine), exists_in_fork=blob(FORK, f'{FORK_LANE}/{name}') is not None)
        if report['new'][name]['exists_in_fork']:
            report['failures'].append(f'{name} is declared new but exists in the fork')
    for name, path in VENDORED.items():
        mine = (LANE / name).read_bytes()
        theirs = blob(GENERATOR, path)
        same = theirs is not None and mine == theirs
        report['vendored'][name] = dict(sha256=sha(mine), source=path, generator_sha256=sha(theirs),
                                        identical=same)
        if not same:
            report['failures'].append(f'{name} is claimed byte-identical to {GENERATOR}:{path} and is not')
    for name in DROPPED:
        if (LANE / name).exists():
            report['failures'].append(f'{name} is declared dropped but is present')
    declared = set(INHERITED) | set(CHANGED) | set(NEW) | set(VENDORED)
    present = {str(p.relative_to(LANE)) for p in LANE.rglob('*.py')
               if 'runs' not in p.parts and 'reports' not in p.parts and 'checks' not in p.parts}
    report['undeclared_python_files'] = sorted(present - declared)
    if report['undeclared_python_files']:
        report['failures'].append('python files present but not declared in any list: '
                                  + ', '.join(report['undeclared_python_files']))
    report['passed'] = not report['failures']
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], failures=report['failures'],
                          inherited=len(INHERITED), changed=len(CHANGED), new=len(NEW),
                          vendored=len(VENDORED)), indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='checks/inherited-sources.json')
    sys.exit(main(parser.parse_args()))
