"""Build the current report/PDF in dependency order and reject stale direct inputs."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from inventory import read, sha, write

HERE = Path(__file__).resolve().parent
PY = '/home/tahmid/Dev/.venv/bin/python'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true', help='Require closed queue and reconciled final accounting')
    args = ap.parse_args()
    checkpoint_args = ['--final'] if args.final else []
    for script, extra in [('checkpoint.py', checkpoint_args), ('make_report.py', []), ('update_research.py', [])]:
        subprocess.run([PY, str(HERE / script), *extra], check=True)
    pins = read(HERE / 'PROVENANCE.json')
    stale = [path for path, digest in pins.items() if not Path(path).is_file() or sha(Path(path)) != digest]
    assert not stale, ('report input changed during build', stale)
    presentation = HERE.parent / '2026-10-23-research-update'
    preservation = read(HERE / 'RESEARCH-UPDATE-PRESERVATION.json')
    pdf = presentation / 'Research-Update.pdf'
    assert sha(pdf) == preservation['generated_pdf_sha256']
    info = subprocess.check_output(['/usr/bin/pdfinfo', str(pdf)], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
    deck = read(presentation / 'project/deck.json')
    assert pages == len(deck['order'])
    for relative, digest in preservation['original'].items():
        if relative.startswith('project/slides/'):
            assert sha(presentation / relative) == digest
    for record in preservation.get('original_artifact_backups', {}).values():
        assert sha(Path(record['backup'])) == record['sha256']
    write(HERE / 'BUILD-CHECK.json', dict(direct_inputs_current=True, original_slides_preserved=True,
        report_filename=read(HERE / 'VALUES.json')['report_filename'],
        report_sha256=sha(HERE / read(HERE / 'VALUES.json')['report_filename']),
        pdf_pages=pages, pdf_sha256=sha(pdf), preserved_slides=preservation['preserved_slide_count'],
        added_slides=preservation['added_slide_count'],
        note='Build/integrity checks only; numerical and publication audit gates remain separate.'))
    print('Report/PDF built; direct inputs current, original slides preserved, page count verified.')


if __name__ == '__main__':
    main()
