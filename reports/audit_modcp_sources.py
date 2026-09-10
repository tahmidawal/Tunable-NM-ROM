"""Check collected pilot source bytes against recorded Git objects locally."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from modcp_audit import digest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'reports'/'2026-09-10-modified-cp-source-audit.json'


def check(run):
    run = Path(run).resolve()
    if (run/'PROVENANCE.json').exists():
        declared = json.loads((run/'PROVENANCE.json').read_text())
        records = [dict(source=r['source'], staged=run/r['staged'], sha256=r['sha256'], commit=r['commit'])
                   for r in declared]
    else:
        declared = json.loads((run/'submission.json').read_text())
        records = [dict(source='experiments/'+name.removeprefix('code/'), staged=run/'cluster'/name,
                        sha256=sha, commit=declared['source_commit']) for name, sha in declared['source_sha256'].items()]
    if not records:
        raise ValueError('No declared source files')
    for row in records:
        committed = subprocess.check_output(['git', '-C', str(ROOT), 'show', row['commit']+':'+row['source']])
        if hashlib.sha256(committed).hexdigest() != row['sha256'] or digest(row['staged']) != row['sha256']:
            raise ValueError(f'Collected source differs from recorded Git object: {row["source"]}')
        row['staged'] = str(row['staged'].relative_to(ROOT))
    return dict(passed=True, files_checked=len(records), source_commits=sorted({r['commit'] for r in records}), files=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {
        'scope': 'Collected source content against immutable Git objects; source provenance only, not numerical accuracy.', 'runs': {}}
    for path in args.run:
        key = str(path.resolve().relative_to(ROOT))
        result['runs'][key] = check(path)
        print(json.dumps({'run': key, 'files_checked': result['runs'][key]['files_checked'], 'passed': True}))
    OUTPUT.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
