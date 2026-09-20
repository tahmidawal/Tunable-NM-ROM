"""Check the bounded September manuscript repair against its exact prior commit.

CPU only. No solver, training, or field revalidation is implied.
"""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = '10498e2e0e264661d702b7606f49133ef7d5be17'


def run(args, cwd=ROOT):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True).stdout


def original(path):
    return run(['git', 'show', f'{BASE}:{path}'])


def boxes(log):
    return Counter(re.findall(r'Overfull \\[hv]box \(([^)]+)\)', log))


before = json.loads(original('paper/tables-md/numbers.json'))
after = json.loads((HERE / 'tables-md/numbers.json').read_text())
changed = {k: [v, after.get(k)] for k, v in before.items() if after.get(k) != v}
scientific_changes = {k: v for k, v in changed.items()
                      if k.startswith('n') and k != 'nNsRom'}
assert not scientific_changes, scientific_changes

tracked = run(['git', 'ls-tree', '-r', '--name-only', BASE, 'paper/tables',
               'paper/tables-md']).decode().splitlines()
existing_tables = [p for p in tracked if Path(p).name.startswith('T')
                   and not Path(p).name.startswith('T02_provenance')]
table_changes = [p for p in existing_tables if (ROOT / p).read_bytes() != original(p)]
format_only = {'paper/tables/T00_glance.tex', 'paper/tables/T01b_spec.tex'}
assert not set(table_changes) - format_only, table_changes
for p in table_changes:
    # Exact equality of the generated Markdown twin verifies unchanged cells,
    # independent of LaTeX column widths, font commands and path wrapping.
    md = 'paper/tables-md/' + Path(p).stem + '.md'
    assert (ROOT / md).read_bytes() == original(md), md

prov = json.loads((HERE / 'tables/provenance.json').read_text())['sources']
assert all(v.get('reachable', True) for v in prov.values())
assert prov['lowvisc_summary']['pinned_commit'] == '5760a7e256ef3c003956bef7a185a86c88b823b6'

# Build the original sources with the same local figure assets and TeX toolchain.
# The temporary checkout is inside this worker's existing worktree and is removed.
with tempfile.TemporaryDirectory(prefix='.integrity-baseline-', dir=ROOT) as td:
    td = Path(td)
    shutil.copytree(HERE, td / 'paper', ignore=shutil.ignore_patterns(
        '*.aux', '*.log', '*.out', '*.fls', '*.fdb_latexmk', '*.toc',
        'integrity-repair-2026-09-20.json', '__pycache__'))
    archive = run(['git', 'archive', BASE, 'paper'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(td, filter='data')
    subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                    'main.tex'], cwd=td / 'paper', check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=60)
    base_log = (td / 'paper/main.log').read_text()

new_log = (HERE / 'main.log').read_text()
new_boxes = boxes(new_log) - boxes(base_log)
assert not new_boxes, dict(new_boxes)
assert not re.search(r'(?:Reference|Citation).*undefined|Undefined control sequence', new_log)
page9 = run(['pdftotext', '-f', '9', '-l', '9', str(HERE / 'main.pdf'), '-']).decode()
assert 'C ONCLUSION' in page9 and 'another Navier' in page9
assert 'AI USE STATEMENT' not in page9
summary = {
    'baseline_commit': BASE,
    'scope': 'manuscript numeric preservation, provenance pin, and PDF build/layout; no new scientific run',
    'passed': True,
    'scientific_macros_compared': sum(k.startswith('n') and k != 'nNsRom' for k in before),
    'scientific_macro_changes': scientific_changes,
    'existing_table_files_compared': len(existing_tables),
    'existing_scientific_table_changes': [],
    'format_only_table_changes_with_identical_markdown_cells': table_changes,
    'changed_existing_metadata_or_status_macros': changed,
    'added_macros': {k: v for k, v in after.items() if k not in before},
    'baseline_overfull_boxes': dict(boxes(base_log)),
    'current_overfull_boxes': dict(boxes(new_log)),
    'new_overfull_boxes': dict(new_boxes),
    'undefined_references_or_commands': False,
    'main_text_ends_on_page': 9,
    'pdf_pages': int(re.search(r'^Pages:\s+(\d+)', run(
        ['pdfinfo', str(HERE / 'main.pdf')]).decode(), re.M).group(1)),
    'pdf_sha256': hashlib.sha256((HERE / 'main.pdf').read_bytes()).hexdigest(),
    'source_sha256': {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()
                      for p in ['main.tex', 'gen_tables.py', 'sections/appendix.tex',
                                'sections/extended-results.tex', 'sections/method-details.tex',
                                'check_integrity_repair.py']},
}
(HERE / 'integrity-repair-2026-09-20.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in
                  ['changed_existing_metadata_or_status_macros', 'added_macros', 'source_sha256']}, indent=2))
