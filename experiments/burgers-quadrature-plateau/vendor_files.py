"""Vendor the frozen code and inputs byte-identically into vendor/tree/<original repo path> and write vendor/PROVENANCE.json.

Every file is read with `git show <commit>:<path>` (no checkout of the source lanes), written unchanged, and its sha256 is
recorded with the source branch and commit. Run once from the lane directory:  python vendor_files.py [--check]
--check re-verifies every vendored file against PROVENANCE.json and against git (exit 1 on any mismatch).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE / 'vendor' / 'tree'
ENV = dict(os.environ, GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='gc.auto', GIT_CONFIG_VALUE_0='0')

QS = ('exp/2026-10-01-quadrature-study', '84d28e3039ae77b7b3e351fb9c4e2842df0797cc')
Q3 = ('exp/2026-10-01-quadrature-burgers3d', 'aab6b6927')
E0 = ('exp/2026-10-06-jcp-references', 'cc04f4b54a1b717d00eb803a276d7efb86994773')
F2 = ('exp/2026-10-18-jcp-fom2d', '12103d331')
S3 = ('exp/2026-10-14-jcp-speed3d', '739154488')

q = 'experiments/quadrature-study/'
b = 'experiments/quadrature-burgers3d/'
r = 'experiments/jcp-references/'
SOURCES = [
    # 2D off-mesh code and frozen model (quadrature-study tip)
    (QS, [q + p for p in ('qcore.py', 'qstudy.py', 'DESIGN.md', 'vendor/arms.py', 'vendor/hops.py', 'vendor/hfast.py',
                          'vendor/bkfast.py', 'vendor/hari_quadrature.py', 'vendor/PROVENANCE.json',
                          'inputs/rotation_R512.npz', 'inputs/rule_q0_m1024_qrg304_reachable.npz',
                          'checks/t256-summary.json', 'checks/t1024-summary.json', 'checks/t4096-summary.json')] +
     ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
      'experiments/separable-decoder/sep_common.py',
      'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']),
    # 3D off-mesh code and frozen model (quadrature-burgers3d tip)
    (Q3, [b + p for p in ('offmesh.py', 'qpanel.py', 'rules.py', 'npcore.py', 'refine_q.py', 'DESIGN.md',
                          'rules/rules.json', 'results/refined-ho.json', 'inputs/model_M2/bank.pkl', 'vendor/burgers3d-span/common.py',
                          'vendor/burgers3d-retry/tables.py', 'vendor/paper-b3d/vendor/b3d_common.py')]),
    # E0 references loader + FD/PS cores (jcp-references tip)
    (E0, [r + p for p in ('refstore.py', 'verdict.py', 'refcore.py', 'pscore.py', 'cohorts.py', 'store/MANIFEST.sha256',
                          'store/STATUS.json', 'vendor/quad3d/vendor/burgers3d-span/common.py',
                          'vendor/quad3d/vendor/paper-b3d/vendor/b3d_common.py')]),
    # 2D FOM runner (jcp-fom2d tip); its refcore/pscore/cohorts are E0 byte-identical copies (checked below)
    (F2, ['experiments/jcp-fom2d/fomrun2.py', 'experiments/jcp-fom2d/psjax2d.py', 'experiments/jcp-fom2d/refcore.py',
          'experiments/jcp-fom2d/pscore.py', 'experiments/jcp-fom2d/cohorts.py']),
    # 3D FOM runner (jcp-speed3d tip)
    (S3, ['experiments/jcp-speed3d/fomrun.py', 'experiments/jcp-speed3d/fomcore.py', 'experiments/jcp-speed3d/psjax.py',
          'experiments/jcp-speed3d/pscore.py', 'experiments/jcp-speed3d/vendor/quad3d/vendor/burgers3d-span/common.py',
          'experiments/jcp-speed3d/vendor/quad3d/vendor/paper-b3d/vendor/b3d_common.py']),
]
# copies that must be byte-identical to each other (same module imported under one name)
SAME = [('experiments/jcp-fom2d/refcore.py', r + 'refcore.py'), ('experiments/jcp-fom2d/pscore.py', r + 'pscore.py'),
        ('experiments/jcp-fom2d/cohorts.py', r + 'cohorts.py'), ('experiments/jcp-speed3d/pscore.py', r + 'pscore.py'),
        (b + 'vendor/burgers3d-span/common.py', r + 'vendor/quad3d/vendor/burgers3d-span/common.py'),
        (b + 'vendor/burgers3d-span/common.py', 'experiments/jcp-speed3d/vendor/quad3d/vendor/burgers3d-span/common.py'),
        (b + 'vendor/paper-b3d/vendor/b3d_common.py', r + 'vendor/quad3d/vendor/paper-b3d/vendor/b3d_common.py'),
        (b + 'vendor/paper-b3d/vendor/b3d_common.py', 'experiments/jcp-speed3d/vendor/quad3d/vendor/paper-b3d/vendor/b3d_common.py')]


def git_blob(commit, path):
    return subprocess.run(['git', '-c', 'gc.auto=0', 'show', f'{commit}:{path}'], cwd=HERE, env=ENV, check=True,
                          capture_output=True).stdout


def full(commit):
    return subprocess.run(['git', '-c', 'gc.auto=0', 'rev-parse', commit], cwd=HERE, env=ENV, check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    check = '--check' in sys.argv
    files, bad = [], []
    for (branch, commit), paths in SOURCES:
        cfull = full(commit)
        for p in paths:
            data = git_blob(cfull, p)
            h = hashlib.sha256(data).hexdigest()
            dst = TREE / p
            if check:
                if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest() != h:
                    bad.append(p)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
            files.append(dict(path=f'vendor/tree/{p}', source_branch=branch, source_commit=cfull, source_path=p,
                              sha256=h, bytes=len(data), modified=False))
    by = {f['source_path']: f['sha256'] for f in files}
    same = [dict(a=a, b=b_, identical=by[a] == by[b_]) for a, b_ in SAME]
    assert all(s['identical'] for s in same), same
    prov = dict(note='byte-identical copies (git show <commit>:<path>); never edit — copy into the lane dir to change',
                files=files, identical_copies=same)
    if check:
        old = json.loads((HERE / 'vendor' / 'PROVENANCE.json').read_text())
        if old['files'] != files:
            bad.append('PROVENANCE.json')
        print('vendor check:', 'OK' if not bad else f'MISMATCH {bad}')
        sys.exit(1 if bad else 0)
    (HERE / 'vendor' / 'PROVENANCE.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(f'vendored {len(files)} files, {sum(f["bytes"] for f in files) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
