"""Verify the UI-exported Overleaf source and PDF against the local revision."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper_revision'
DEST = ROOT / 'manuscript_tools' / 'overleaf'
sha = lambda data: hashlib.sha256(data).hexdigest()
required = ['main.tex', 'preamble.tex', 'natbib.sty', 'iclr2027_conference.bst',
            'iclr2027_conference.sty', 'fancyhdr.sty', 'fig_cost_scaling.pdf']
required += ['figs/' + p.name for p in sorted((PAPER / 'figs').iterdir())
             if p.suffix in {'.pdf', '.png'} and p.name != 'F8_ns_head_span.png']
with ZipFile(DEST / 'after-update.zip') as archive:
    for name in required:
        assert archive.read(name) == (PAPER / name).read_bytes(), name
    hashes = {name: sha(archive.read(name)) for name in required}
info = subprocess.check_output(['pdfinfo', str(DEST / 'compiled.pdf')], text=True)
page_count = int(re.search(r'Pages:\s+(\d+)', info)[1])
subprocess.run(['pdftotext', '-layout', str(DEST / 'compiled.pdf'),
                str(DEST / 'compiled.txt')], check=True)
text = (DEST / 'compiled.txt').read_text()
pages = text.split('\f')
checks = {
    'main_text_conclusion_page_10': 'CONCLUSIONANDFUTUREWORK' in re.sub(r'\s+', '', pages[9]),
    'figure_2_page_8': 'Figure 2:' in pages[7],
    'table_4_page_9': 'Table 4:' in pages[8],
    'operator_accuracy_only_text': 'prediction errors' in text,
}
assert all(checks.values()), checks
manifest_path = DEST / 'upload-manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest.update({
    'overleaf_pdf_pages': page_count,
    'required_file_sha256': hashes,
    'required_files_byte_match': True,
    'after_update_source_sha256': sha((DEST / 'after-update.zip').read_bytes()),
    'compiled_pdf_sha256': sha((DEST / 'compiled.pdf').read_bytes()),
    'pdf_checks': checks,
    'proof_screenshots': ['compiled-project.png', 'zero-errors.png'],
})
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'matching_files': len(hashes), 'pdf_pages': page_count,
                  'pdf_checks': checks}, indent=2))
