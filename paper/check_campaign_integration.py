"""Check claim-scope repairs and the provisional supplement against exact baselines."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
BASE = 'ac4d010b2d53f2d14e0f4a5ef987911794b56e72'
LEGACY_TABLE_BASE = '187e34b6bdac6946647db019b54b34f3feb4d5b4'


def run(*args):
    return subprocess.check_output(args, cwd=HERE)


def original(path, base=BASE):
    return run('git', 'show', f'{base}:paper/{path}')


def sha(data):
    return hashlib.sha256(data).hexdigest()


before = json.loads(original('tables-md/numbers.json'))
after = json.loads((HERE / 'tables-md/numbers.json').read_text())
assert before == after, 'Existing macro values changed'
assert (HERE / 'tables/provenance.json').read_bytes() == original('tables/provenance.json')
files = run('git', 'ls-tree', '-r', '--name-only', LEGACY_TABLE_BASE,
            'tables', 'tables-md').decode().splitlines()
assert files
existing = []
for path in files:
    if Path(path).name.startswith('T'):
        assert (HERE / path).read_bytes() == original(path), path
        existing.append(path)
for path in ('tables/TC_oracle_confirmation.tex', 'tables-md/TC_oracle_confirmation.md',
             'tables/campaign-numbers.tex', 'tables-md/campaign-numbers.json'):
    assert (HERE / path).read_bytes() == original(path), path
old_source = original('main.tex').decode()
new_source = (HERE / 'main.tex').read_text()
ledger = json.loads((HERE / 'claim-scope-disposition-2026-09-20.json').read_text())
assert sha(old_source.encode()) == ledger['source_sha256']
assert sha(new_source.encode()) == ledger['repaired_source_sha256']
replayed_source = old_source
for edit in ledger['all_wording_edits']:
    assert replayed_source.count(edit['before']) == 1
    replayed_source = replayed_source.replace(edit['before'], edit['after'])
assert replayed_source == new_source, 'Unrecorded main-source edit'
assert len(ledger['findings']) == 6
assert all(f['status'] == 'repaired; measured values unchanged' and f['edits']
           for f in ledger['findings'])
abstract = lambda s: s.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert sha(abstract(old_source).encode()) == ledger['abstract_before_sha256']
assert sha(abstract(new_source).encode()) == ledger['abstract_after_sha256']
assert abstract(old_source) != abstract(new_source)
assert ledger['abstract_unchanged'] is False
assert len(ledger['abstract_changes']) == 2
assert re.search(r'\\title\{[^\n]+', old_source).group() == re.search(
    r'\\title\{[^\n]+', new_source).group()
scientific_uses = lambda s: Counter(re.findall(r'\\(?:n|prov)[A-Z][A-Za-z]+', s))
assert scientific_uses(old_source) == scientific_uses(new_source)
# Literal mesh/rank/tolerance numbers in main.tex must also survive the prose repair.
assert Counter(re.findall(r'\d+(?:\.\d+)?', old_source)) == Counter(
    re.findall(r'\d+(?:\.\d+)?', new_source))
log = (HERE / 'main.log').read_text()
assert not re.search(r'Overfull \\[hv]box|(?:Reference|Citation).*undefined|Undefined control sequence|Float too large', log)
pages = run('pdftotext', '-layout', 'main.pdf', '-').decode().split('\f')
assert 'C ONCLUSION' in pages[8] and 'another Navier' in pages[8]
assert 'AI USE STATEMENT' not in pages[8]
counts = json.loads((HERE / 'tables/campaign-provenance.json').read_text())['audit']
assert counts['status'] == 'development only; final confirmation pending'
assert counts['nmrom_training_records'] == 11
assert counts['operator_training_records'] == 16
result = dict(baseline_commit=BASE, legacy_table_baseline_commit=LEGACY_TABLE_BASE,
    passed=True, preserved_existing_macros=len(before),
    preserved_scientific_macros=sum(k.startswith('n') and k != 'nNsRom' for k in before),
    preserved_existing_table_files=len(existing), title_unchanged=True,
    existing_source_pins_unchanged=True, historical_oracle_tables_unchanged=True,
    main_scientific_macro_uses_and_literal_numbers_unchanged=True,
    abstract_unchanged=False, abstract_wording_corrections=len(ledger['abstract_changes']),
    disposed_claim_findings=len(ledger['findings']),
    claim_disposition_sha256=sha((HERE / 'claim-scope-disposition-2026-09-20.json').read_bytes()),
    main_text_ends_on_page=9, pdf_pages=len([p for p in pages if p.strip()]),
    overfull_boxes=0, undefined_references=0, new_evidence=counts,
    pdf_sha256=sha((HERE / 'main.pdf').read_bytes()))
(HERE / 'campaign-integration-2026-09-20.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
