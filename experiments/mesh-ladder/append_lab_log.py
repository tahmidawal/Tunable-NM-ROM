"""Append this session's entry to the one canonical LAB-LOG.md under an exclusive lock.

Several agents share that file, so the append is a locked read-modify-write with an
fsync, and it only ever adds to the chronology -- it never rewrites the
"Where things stand" block at the top.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path

LAB = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md')


def append(text):
    with LAB.open('r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        existing = stream.read()
        if not existing.endswith('\n'):
            existing += '\n'
        stream.seek(0)
        stream.write(existing + text)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    return len(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True, help='markdown file holding the entry to append')
    args = parser.parse_args()
    body = Path(args.entry).read_text()
    assert body.lstrip().startswith('## '), 'the entry must open with its dated ## header'
    if not body.startswith('\n'):
        body = '\n' + body
    written = append(body)
    print(f'appended {written} bytes to {LAB}')


if __name__ == '__main__':
    main()
