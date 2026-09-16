"""Measure what a code-only GitHub mirror would drop and keep, and generate the
tables of `2026-09-16-code-only-github-mirror-plan.md`.

Read-only against the source repository: it lists objects and their sizes and
never writes a ref, an object or a note. Nothing here pushes anything.

    /home/tahmid/Dev/.venv/bin/python reports/generate_mirror_plan.py \
        --emit-json reports/2026-09-16-code-only-github-mirror-plan.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
from pathlib import Path

REPO = '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude'

# The same rule the mirror script applies, kept in one place in each file and
# asserted equal by `--check-script`.
DROP_RE = (r'(\.part-?[0-9]+$)|(/archive-parts/part-?[0-9]+$)'
           r'|(\.(npz|npy|pkl|pt|pth|h5|hdf5)$)|(\.(tar|tgz|zip)$)'
           r'|(\.tar\.gz$)|(\.gz$)')
DROP = re.compile(DROP_RE)

GIB = 1 << 30
MIB = 1 << 20


def git(*args: str) -> str:
    return subprocess.run(('git', '-C', REPO) + args,
                          capture_output=True, text=True, check=True).stdout


def inventory(refspec: list[str]):
    names = git('rev-list', '--objects', *refspec)
    chk = subprocess.run(
        ('git', '-C', REPO, 'cat-file',
         '--batch-check=%(objecttype) %(objectname) %(objectsize) '
         '%(objectsize:disk) %(rest)'),
        input=names, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in chk.splitlines():
        p = line.split(' ', 4)
        rows.append({'type': p[0], 'sha': p[1], 'size': int(p[2]),
                     'disk': int(p[3]), 'path': p[4] if len(p) > 4 else ''})
    return rows


def classify(path: str) -> str:
    base = os.path.basename(path)
    if re.search(r'\.part-?\d+$', base) or re.search(r'(^|/)archive-parts/part-?\d+$', path):
        return 'chunked run archive'
    if base.endswith(('.tar', '.tar.gz', '.tgz', '.zip', '.gz')):
        return 'whole archive'
    if base.endswith(('.npz', '.npy')):
        return 'saved field array'
    if base.endswith(('.pkl', '.pt', '.pth', '.h5', '.hdf5')):
        return 'checkpoint'
    if base.endswith(('.json', '.jsonl')):
        return 'result / audit JSON'
    if base.endswith(('.md', '.txt', '.sha256', '.log', '.out', '.err', '.csv',
                      '.yaml', '.yml', '.tex', '.sty', '.bib', '.mmd', '.sbatch')):
        return 'text / manifest'
    if base.endswith(('.py', '.sh', '.c', '.cu', '.h', '.ipynb')):
        return 'source'
    if base.endswith(('.png', '.pdf', '.jpg', '.gif', '.mp4', '.html', '.svg')):
        return 'figure / document'
    return 'other'


def table(head, body, align=None):
    align = align or ['---' if i == 0 else '---:' for i in range(len(head))]
    return '\n'.join(['| ' + ' | '.join(head) + ' |',
                      '|' + '|'.join(align) + '|']
                     + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in body])


def commit_ladder(ref: str, sizes: dict[str, int], top: int):
    commits = git('rev-list', '--first-parent', ref).split()
    seen: set[str] = set()
    rows = []
    for c in reversed(commits):
        added = 0
        dirs: collections.Counter = collections.Counter()
        for line in git('ls-tree', '-r', c).splitlines():
            meta, path = line.split('\t', 1)
            sha = meta.split()[2]
            if sha in seen:
                continue
            seen.add(sha)
            nbytes = sizes.get(sha, 0)
            added += nbytes
            dirs[os.path.dirname(path)] += nbytes
        subject = git('log', '-1', '--format=%h|%ad|%s', '--date=short', c).strip()
        h, date, subj = subject.split('|', 2)
        dom, dom_bytes = dirs.most_common(1)[0] if dirs else ('', 0)
        rows.append({'commit': h, 'date': date, 'subject': subj,
                     'added_bytes': added, 'dominant_dir': dom,
                     'dominant_bytes': dom_bytes})
    rows.sort(key=lambda r: -r['added_bytes'])
    return rows[:top], sum(1 for r in rows if r['added_bytes'] > 2 * GIB)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--emit-json', default=None)
    ap.add_argument('--mirror', default=None,
                    help='a built throwaway mirror, to report its measured pack size')
    ap.add_argument('--lineage', default='exp/2026-09-13-nmrom-consolidated')
    a = ap.parse_args()

    rows = inventory(['--all'])
    blobs = [r for r in rows if r['type'] == 'blob']
    sizes = {r['sha']: r['size'] for r in blobs}
    keep = [r for r in blobs if not DROP.search(r['path'])]
    drop = [r for r in blobs if DROP.search(r['path'])]
    meta = [r for r in rows if r['type'] in ('tree', 'commit', 'tag')]

    by_class: collections.Counter = collections.Counter()
    n_class: collections.Counter = collections.Counter()
    for r in blobs:
        c = classify(r['path'])
        by_class[c] += r['size']
        n_class[c] += 1

    chunk_dirs: collections.Counter = collections.Counter()
    for r in drop:
        chunk_dirs[os.path.dirname(r['path'])] += r['size']

    ladder, over2gib = commit_ladder(a.lineage, sizes, top=10)

    mirror_pack = None
    if a.mirror:
        for line in subprocess.run(('git', '-C', a.mirror, 'count-objects', '-vH'),
                                   capture_output=True, text=True).stdout.splitlines():
            if line.startswith('size-pack:'):
                mirror_pack = line.split(':', 1)[1].strip()

    payload = {
        'repository': REPO,
        'refspec': '--all',
        'objects_total': len(rows),
        'blobs_total': len(blobs),
        'blob_bytes_total': sum(r['size'] for r in blobs),
        'blob_bytes_on_disk': sum(r['disk'] for r in blobs),
        'metadata_objects': len(meta),
        'metadata_bytes_on_disk': sum(r['disk'] for r in meta),
        'drop_rule': DROP_RE,
        'dropped_blobs': len(drop),
        'dropped_bytes': sum(r['size'] for r in drop),
        'kept_blobs': len(keep),
        'kept_bytes': sum(r['size'] for r in keep),
        'kept_bytes_on_disk': sum(r['disk'] for r in keep),
        'kept_blobs_over_50MB': [r['path'] for r in keep if r['size'] > 50 * 10 ** 6],
        'kept_blobs_over_100MB': [r['path'] for r in keep if r['size'] > 100 * 10 ** 6],
        'largest_kept': [{'path': r['path'], 'bytes': r['size']}
                         for r in sorted(keep, key=lambda r: -r['size'])[:10]],
        'by_class': {c: {'count': n_class[c], 'bytes': by_class[c]}
                     for c in sorted(by_class, key=lambda c: -by_class[c])},
        'dropped_directories_count': len(chunk_dirs),
        'dropped_directories': [{'dir': d, 'bytes': b}
                                for d, b in chunk_dirs.most_common()],
        'lineage': a.lineage,
        'commits_over_2GiB_on_lineage': over2gib,
        'largest_commits': ladder,
        'measured_mirror_pack': mirror_pack,
    }
    if a.emit_json:
        Path(a.emit_json).write_text(json.dumps(payload, indent=1) + '\n')

    print('## Measured object inventory (all refs)\n')
    print(table(['quantity', 'value'], [
        ['objects reachable from every ref', f"{payload['objects_total']:,}"],
        ['blobs', f"{payload['blobs_total']:,}"],
        ['blob bytes, uncompressed', f"{payload['blob_bytes_total']/GIB:.2f} GiB"],
        ['blob bytes, as packed on disk today', f"{payload['blob_bytes_on_disk']/GIB:.2f} GiB"],
        ['trees + commits + tags', f"{payload['metadata_objects']:,}"],
        ['trees + commits + tags, as packed', f"{payload['metadata_bytes_on_disk']/MIB:.1f} MiB"],
    ]))
    print('\n## What the rule drops and keeps\n')
    print(table(['', 'blobs', 'uncompressed', 'share'], [
        ['dropped (raw archives and arrays)', f"{payload['dropped_blobs']:,}",
         f"{payload['dropped_bytes']/GIB:.2f} GiB",
         f"{100*payload['dropped_bytes']/payload['blob_bytes_total']:.2f} %"],
        ['kept (code, reports, result JSONs, manifests)', f"{payload['kept_blobs']:,}",
         f"{payload['kept_bytes']/GIB:.3f} GiB",
         f"{100*payload['kept_bytes']/payload['blob_bytes_total']:.2f} %"],
    ]))
    print('\n## Blob bytes by kind\n')
    print(table(['kind', 'blobs', 'uncompressed', 'mirrored?'],
                [[c, f"{v['count']:,}", f"{v['bytes']/GIB:.3f} GiB",
                  'dropped' if c in ('chunked run archive', 'whole archive',
                                     'saved field array', 'checkpoint') else 'kept']
                 for c, v in payload['by_class'].items()]))
    top_dirs = payload['dropped_directories'][:25]
    rest = payload['dropped_directories'][25:]
    print('\n## Where the dropped bytes live — 25 largest directories\n')
    print(table(['directory', 'uncompressed'],
                [[f"`{d['dir']}`", f"{d['bytes']/GIB:.3f} GiB"] for d in top_dirs]))
    print(f"\n…and **{len(rest)}** further directories holding "
          f"{sum(d['bytes'] for d in rest)/GIB:.2f} GiB, for "
          f"{payload['dropped_directories_count']} directories in total. Every one of "
          f"them keeps its manifests (`SHA256SUMS`, `MANIFEST.sha256`, `archive.json`, "
          f"`ARCHIVE.json`), its `result.json` and its `audit.json`.")
    print('\n## Commits that alone exceed GitHub\'s per-push pack limit\n')
    print(f"On `{a.lineage}` (first-parent), **{over2gib} commits** each introduce "
          f"more than 2 GiB of new blob bytes:\n")
    print(table(['commit', 'date', 'added', 'dominant directory', 'subject'],
                [[f"`{r['commit']}`", r['date'], f"{r['added_bytes']/GIB:.2f} GiB",
                  f"`{r['dominant_dir']}`", r['subject'][:64]]
                 for r in ladder if r['added_bytes'] > 2 * GIB],
                align=['---', '---', '---:', '---', '---']))
    print('\n## Largest files surviving in the mirror\n')
    print(table(['file', 'size'],
                [[f"`{r['path']}`", f"{r['bytes']/MIB:.2f} MiB"]
                 for r in payload['largest_kept']]))
    print(f"\nKept blobs above GitHub's 50 MB warning: "
          f"**{len(payload['kept_blobs_over_50MB'])}**; above its 100 MB hard "
          f"rejection: **{len(payload['kept_blobs_over_100MB'])}**.")
    if mirror_pack:
        print(f"\nMeasured pack size of the built throwaway mirror: **{mirror_pack}**.")


if __name__ == '__main__':
    main()
