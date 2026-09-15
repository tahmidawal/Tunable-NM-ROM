"""Append one dated session entry to the canonical lab log under an exclusive lock.

Never rewrites the "Where things stand" block; appends only.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('entry', help='file holding the exact markdown entry to append')
    a = p.parse_args()
    text = Path(a.entry).read_text()
    assert text.startswith('\n\n## '), 'entry must begin with a blank line and a dated ## header'
    assert '\n### ' in text, 'entry must contain a session ### header'
    before = LAB.stat().st_size
    append(text)
    print(f'appended {len(text)} bytes; {before} -> {LAB.stat().st_size}')


if __name__ == '__main__':
    main()
