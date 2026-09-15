"""Build a verified staging tree for one mesh-ladder cluster attempt.

Every staged byte is checked against the committed blob for its path, so a dirty
working-tree edit cannot reach the cluster unnoticed. The tree preserves the
worktree's ``experiments/`` layout because the drivers resolve their imports and
their frozen model files relative to the repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Python sources the two drivers import, plus the frozen model files they load.
PACKAGES = [
    'experiments/mesh-ladder',
    'experiments/mr-burgers2d',
    'experiments/separable-decoder',
    'experiments/multiresolution-poisson',
    'experiments/cost-to-tolerance',
    'experiments/wave2d-rom-latent-stepping/deps/multistage-precision',
]
MODELS = [
    'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
    'experiments/multiresolution-poisson/runs/correction_accuracy10/checkpoints/r128_joint.pkl',
    'experiments/multiresolution-poisson/runs/correction_accuracy10/basis.npz',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*arguments):
    return subprocess.run(('git', '-C', str(ROOT)) + arguments,
                          check=True, capture_output=True, text=True).stdout


def committed_blob(relative):
    """SHA-256 of the committed content at HEAD, or None when untracked."""
    try:
        blob = git('rev-parse', f'HEAD:{relative}').strip()
    except subprocess.CalledProcessError:
        return None
    raw = subprocess.run(('git', '-C', str(ROOT), 'cat-file', 'blob', blob),
                         check=True, capture_output=True).stdout
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='staging directory; must not exist')
    parser.add_argument('--allow-untracked', action='store_true',
                        help='permit staging files that are not committed at HEAD')
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists(), out

    commit = git('rev-parse', 'HEAD').strip()
    dirty = git('status', '--porcelain').strip()
    files = []
    for package in PACKAGES:
        for path in sorted((ROOT / package).glob('*.py')):
            files.append(path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT / package).glob('*.json')):
            files.append(path.relative_to(ROOT).as_posix())
    files += MODELS
    files = sorted(set(files))

    records = []
    mismatched = []
    for relative in files:
        source = ROOT / relative
        assert source.is_file(), relative
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        staged, blob = digest(target), committed_blob(relative)
        if blob is None:
            state = 'untracked'
        elif blob == staged:
            state = 'matches HEAD'
        else:
            state = 'differs from HEAD'
            mismatched.append(relative)
        records.append(dict(path=relative, bytes=source.stat().st_size,
                            sha256=staged, committed_sha256=blob, state=state))

    if mismatched and not args.allow_untracked:
        print('staged content differs from HEAD:', file=sys.stderr)
        for relative in mismatched:
            print('  ' + relative, file=sys.stderr)
        raise SystemExit('commit the changes, or pass --allow-untracked deliberately')

    (out / 'MANIFEST.sha256').write_text(
        ''.join(f"{r['sha256']}  {r['path']}\n" for r in records))
    (out / 'PROVENANCE.json').write_text(json.dumps(dict(
        commit=commit, working_tree_clean=not dirty,
        untracked_staged=[r['path'] for r in records if r['state'] == 'untracked'],
        differing_from_head=mismatched, file_count=len(records),
        total_bytes=sum(r['bytes'] for r in records), files=records), indent=2) + '\n')
    print(f'staged {len(records)} files, {sum(r["bytes"] for r in records) / 1e6:.1f} MB, commit {commit}')
    print(f'untracked: {sum(1 for r in records if r["state"] == "untracked")}, '
          f'differing: {len(mismatched)}')


if __name__ == '__main__':
    main()
