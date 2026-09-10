"""Describe a scoped integration for review; never apply files or merge branches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'experiments/modcp-eq/'
BRANCHES = {
    'burgers': 'exp/2026-09-10-modcp-burgers2d',
    'wave': 'exp/2026-09-10-modcp-wave2d',
}
DEPENDENCIES = ('experiments/fresh-wave-head/fresh_fom.py',
                'experiments/fresh-wave-head/test_fresh_fom.py')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def files(commit, *paths):
    records = {}
    for item in git('ls-tree', '-rz', commit, '--', *paths).split(b'\0'):
        if not item:
            continue
        metadata, path = item.split(b'\t', 1)
        mode, kind, blob = metadata.decode().split()
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError(f'Unexpected integration object: {path!r}')
        records[path.decode()] = dict(mode=mode, blob=blob)
    return records


def describe_file(commit, path, record):
    content = git('cat-file', 'blob', record['blob'])
    return dict(**record, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                source_commit=commit, source_path=path)


def generated_file(content):
    payload = content.encode()
    return dict(mode='100644', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                generated_content=content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    heads = {name: git('rev-parse', branch).decode().strip() for name, branch in BRANCHES.items()}
    catalogs = {name: files(head, PREFIX) for name, head in heads.items()}
    proposed = {}
    collisions = []
    for name, catalog in catalogs.items():
        for path, record in catalog.items():
            if path in (PREFIX+'.gitignore', PREFIX+'DESIGN.md'):
                continue
            if path in proposed and proposed[path]['blob'] != record['blob']:
                collisions.append(path)
                continue
            if path not in proposed:
                proposed[path] = describe_file(heads[name], path, record)
    if collisions:
        raise ValueError(f'Unresolved shared source collisions: {collisions}')
    for name, catalog in catalogs.items():
        path = PREFIX+'DESIGN.md'
        target = PREFIX+f'DESIGN-{name.upper()}.md'
        if target in proposed:
            raise ValueError(f'Design preservation would overwrite {target}')
        proposed[target] = describe_file(heads[name], path, catalog[path])
    ignore_lines = []
    for name in BRANCHES:
        content = git('cat-file', 'blob', catalogs[name][PREFIX+'.gitignore']['blob']).decode()
        for line in content.splitlines():
            if line not in ignore_lines:
                ignore_lines.append(line)
    proposed[PREFIX+'.gitignore'] = generated_file('\n'.join(ignore_lines)+'\n')
    proposed[PREFIX+'DESIGN.md'] = generated_file(
        '# Modified CP with empirical quadrature\n\n'
        'This directory contains the Burgers and fresh-wave architecture pilots. '
        'Their approved protocols are preserved separately.\n\n'
        '- [Burgers protocol](DESIGN-BURGERS.md)\n'
        '- [Reflective and absorbing wave protocol](DESIGN-WAVE.md)\n\n'
        'The shared decoder, training, quadrature, and latent-solver modules live in `common/`. '
        'Burgers uses Newton–BiCGStab as its full-order comparison. Waves use the verified '
        'fresh reference implementation in `../fresh-wave-head/`, with CG and direct/explicit controls.\n\n'
        'The canonical project history is `../../LAB-LOG.md`. Generated comparison results '
        'and their audit/retention manifests live in `../../reports/`. Large field archives '
        'are retained separately; integrating source files does not authorize deleting '
        'the original worktrees or rewriting immutable run provenance.\n')
    dependencies = files(heads['wave'], *DEPENDENCIES)
    if set(dependencies) != set(DEPENDENCIES):
        raise ValueError('Required fresh-wave dependency missing from the committed wave tree')
    for path, record in dependencies.items():
        proposed[path] = describe_file(heads['wave'], path, record)
    main_head = git('rev-parse', 'HEAD').decode().strip()
    existing = files(main_head, PREFIX, *DEPENDENCIES)
    conflicts = []
    for path, record in existing.items():
        if path in proposed:
            content = git('cat-file', 'blob', record['blob'])
            if hashlib.sha256(content).hexdigest() != proposed[path]['sha256']:
                conflicts.append(path)
    result = dict(
        status='Review proposal only: no merge, branch creation, file application, or worktree cleanup performed.',
        strategy='Integrate only the listed experiment files and required fresh-wave dependencies; preserve both designs.',
        main_head=main_head, sources={name: dict(branch=BRANCHES[name], commit=head) for name, head in heads.items()},
        file_count=len(proposed), total_file_bytes=sum(r['bytes'] for r in proposed.values()),
        existing_main_conflicts=conflicts,
        retention='Keep the source worktrees and root raw archives until report paths and immutable provenance are deliberately migrated.',
        review='Re-run this manifest after final evaluation commits, inspect all listed paths and generated text, and obtain the required user merge decision.',
        files=dict(sorted(proposed.items())))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('file_count', 'total_file_bytes', 'existing_main_conflicts')}))


if __name__ == '__main__':
    main()
