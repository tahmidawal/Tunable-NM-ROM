"""Assemble the local reference tree runs/refs-local/<d>d/<cohort>/T<T>/... from collected reference job archives (hard
links, no copies) and verify every file against the cluster refs/REFS-MANIFEST.sha256 (pulled to runs/). Files absent
locally (fine fields left on the cluster) are reported, never invented.

    python assemble_refs.py ref2a ref2b ref2c ps2a ps2b ref3 ps3
"""
import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / 'runs' / 'refs-local'


def main():
    man = {}
    for line in (HERE / 'runs' / 'REFS-MANIFEST.sha256').read_text().splitlines():
        if line.strip():
            h, f = line.split(None, 1)
            man[f.strip().lstrip('./')] = h
    linked, bad = 0, []
    for att in sys.argv[1:]:
        out = HERE / 'runs' / att / 'archive' / 'output'
        for step in out.iterdir():
            for d in ('2d', '3d'):
                root = step / d
                if not root.is_dir():
                    continue
                for f in root.rglob('c*'):
                    rel = str(f.relative_to(step))
                    if rel not in man:
                        bad.append(f'{rel}: not in the cluster manifest')
                        continue
                    if hashlib.sha256(f.read_bytes()).hexdigest() != man[rel]:
                        bad.append(f'{rel}: sha256 differs from the cluster manifest')
                        continue
                    t = DEST / rel
                    t.parent.mkdir(parents=True, exist_ok=True)
                    if not t.exists():
                        os.link(f, t)
                    linked += 1
    missing = sorted(set(man) - {str(p.relative_to(DEST)) for p in DEST.rglob('c*')})
    print('linked', linked, 'bad', len(bad), 'missing locally', len(missing))
    for b in bad:
        print('BAD', b)
    (HERE / 'runs' / 'refs-local.missing.txt').write_text('\n'.join(missing) + '\n')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
