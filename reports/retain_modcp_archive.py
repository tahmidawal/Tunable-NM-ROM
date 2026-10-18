"""Anchor immutable pilot field archives outside worktrees and Git history."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tarfile

from modcp_audit import digest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'reports'/'2026-09-10-modified-cp-raw-artifacts.json'
RAW = ROOT/'artifacts'/'modcp-eq'/'2026-09-10'/'raw'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError('Archive name must be a basename')
    source = args.input.resolve()
    if digest(source) != args.sha256:
        raise ValueError('Source archive does not match its declared checksum')
    RAW.mkdir(parents=True, exist_ok=True)
    destination = RAW/args.name
    if not destination.exists():
        # Both names refer to one immutable archive. The root name survives
        # unlinking the worktree name, without duplicating large compressed data.
        os.link(source, destination)
    elif digest(destination) != args.sha256:
        raise ValueError('Refusing to replace a different retained archive')
    with tarfile.open(destination, 'r:*') as archive:
        names = archive.getnames()
    document = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        'policy': 'Full raw fields retained locally outside Git; this tracked manifest identifies immutable root archive paths. '
                  'Raw timing/error JSON, source, proofs, audits, and checkpoints are tracked separately.',
        'extraction': 'Verify SHA256, create an empty destination, then tar -xf ARCHIVE -C DESTINATION. '
                      'Consult member_prefixes before choosing a destination; never overwrite an existing run.',
        'archives': {}}
    record = dict(path=str(destination.relative_to(ROOT)), sha256=args.sha256,
                  bytes=destination.stat().st_size, source_path=str(source.relative_to(ROOT)),
                  role=args.role, job_id=args.job_id, member_count=len(names),
                  member_prefixes=sorted({name.split('/')[0] for name in names}),
                  retention='hard link under main; worktree cleanup may unlink only the worktree name')
    if args.name in document['archives'] and document['archives'][args.name] != record:
        raise ValueError('Refusing to rewrite a different archive manifest entry')
    document['archives'][args.name] = record
    document['updated_utc'] = datetime.now(timezone.utc).isoformat()
    MANIFEST.write_text(json.dumps(document, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
