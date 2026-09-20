"""Check the bounded supplement against the exact preceding manuscript."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
BASE = '187e34b6bdac6946647db019b54b34f3feb4d5b4'


def run(*args):
    return subprocess.check_output(args, cwd=HERE)


def original(path):
    return run('git', 'show', f'{BASE}:paper/{path}')


before = json.loads(original('tables-md/numbers.json'))
after = json.loads((HERE / 'tables-md/numbers.json').read_text())
assert before == after, 'Existing macro values changed'
files = run('git', 'ls-tree', '-r', '--name-only', BASE, 'tables', 'tables-md').decode().splitlines()
# git ls-tree paths above are relative to the paper working directory.
assert files
existing = []
for path in files:
    if Path(path).name.startswith('T'):
        assert (HERE / path).read_bytes() == original(path), path
        existing.append(path)
abstract = lambda s: s.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert abstract(original('main.tex').decode()) == abstract((HERE / 'main.tex').read_text())
log = (HERE / 'main.log').read_text()
assert not re.search(r'Overfull \\[hv]box|(?:Reference|Citation).*undefined|Undefined control sequence|Float too large', log)
pages = run('pdftotext', '-layout', 'main.pdf', '-').decode().split('\f')
assert 'C ONCLUSION' in pages[8] and 'another Navier' in pages[8]
assert 'AI USE STATEMENT' not in pages[8]
counts = json.loads((HERE / 'tables/campaign-provenance.json').read_text())['audit']
result = dict(baseline_commit=BASE, passed=True,
    preserved_existing_macros=len(before),
    preserved_scientific_macros=sum(k.startswith('n') and k != 'nNsRom' for k in before),
    preserved_existing_table_files=len(existing), abstract_unchanged=True,
    main_text_ends_on_page=9, pdf_pages=len([p for p in pages if p.strip()]),
    overfull_boxes=0, undefined_references=0,
    new_evidence=counts,
    pdf_sha256=hashlib.sha256((HERE / 'main.pdf').read_bytes()).hexdigest())
(HERE / 'campaign-integration-2026-09-20.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
