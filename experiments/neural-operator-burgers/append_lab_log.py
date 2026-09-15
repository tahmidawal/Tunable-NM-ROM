"""Append one dated session entry to the canonical lab log under an exclusive lock.

The canonical log lives on main at the repository root and is shared by every
worktree, so the append must hold ``fcntl.flock`` for the whole read/write.
"""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, required=True)
    args = parser.parse_args()
    body = args.entry.read_text()
    assert body.lstrip('\n').startswith('## '), 'entry must begin with a dated header'
    body = '\n\n' + body.strip('\n') + '\n'
    append(body)
    print('appended', len(body), 'characters to', LAB)


if __name__ == '__main__':
    main()
