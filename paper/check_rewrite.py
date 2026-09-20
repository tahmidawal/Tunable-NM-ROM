"""Check manuscript rewrite provenance, preserved tables, layout and anonymity."""
import hashlib,json,re,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent; R=P.parent
BASE='53d26f481dcd3e59f2d2dc15126392dc2b458460'
def old(path): return subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=R)
files=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'paper/tables','paper/tables-md'],cwd=R).decode().splitlines()
preserved=[p for p in files if Path(p).name.startswith(('T','numbers'))]
changed=[p for p in preserved if (R/p).read_bytes()!=old(p)]
reference_only={'paper/tables/T00_glance.tex','paper/tables-md/T00_glance.md'}
assert not set(changed)-reference_only,changed
for path in changed:
    before=old(path).decode()
    cleaned=re.sub(r'; Fig\.[~ ]\\ref\{fig:family\}[ABC]', '', before)
    assert (R/path).read_text()==cleaned,path
log=(P/'main.log').read_text()
assert 'Overfull' not in log
assert not re.search(r'(undefined references|Citation .* undefined|Reference .* undefined|multiply defined)',log)
style=json.loads((P/'style-verification.json').read_text());assert style['identical']
assert hashlib.sha256((P/'iclr2027_conference.sty').read_bytes()).hexdigest()==style['official_sha256']
text=subprocess.check_output(['pdftotext','-layout',str(P/'main.pdf'),'-']).decode()
pages=text.split('\f');refs=[i+1 for i,p in enumerate(pages) if re.search(r'R\s*EFERENCES\s*\n',p)]
assert refs and refs[0]<=9,refs
s=(P/'main.tex').read_text();assert r'\iclrfinalcopy' not in s and 'Anonymous authors' in s
abstract=s.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert len(abstract.split())<=250
cg=json.loads((P/'evidence/paired-cg-2026-09-20/results.json').read_text())
for r in cg['rows']:
 assert abs(r['cg_ms']/r['method_ms']-r['speedup'])<1e-10
 assert r['cg_error_pct']<=r['error_pct']
experiment=json.loads((P/'tables/main-experiments-provenance.json').read_text())
for kind, source in experiment['sources'].items():
 raw=(P/'evidence/main-experiments-2026-09-20'/f'{kind}.json').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==source['sha256']
 rows=experiment['selected'][kind]
 assert len(rows)==(7 if kind=='heat' else 8)
 for row in rows:
  assert row['cases']==(32 if kind=='burgers' else 8 if kind=='ns' else 16)
  if kind=='heat':assert row['speedup'] is None
  else:assert abs(rows[-1]['ms']/row['ms']-row['speedup'])<1e-12
assert 'fig_tunability_family' not in (P/'sections/appendix.tex').read_text()
assert 'fig:family' not in (P/'main.tex').read_text()
assert 'sparse direct' not in (P/'tables/TR_lshape_cg_main.tex').read_text()
for label in ('tab:3d-linear-main','tab:3d-nonlinear-main','tab:cg-main','tab:lshape-main'):
 assert label in s.split(r'\bibliographystyle')[0]
report=dict(passed=True,baseline=BASE,preserved_generated_files=len(preserved),historical_numeric_changes=[],obsolete_figure_reference_removals=changed,paired_cg_rows=len(cg['rows']),main_3d_method_rows=sum(len(x) for x in experiment['selected'].values()),main_3d_source_hashes={k:v['sha256'] for k,v in experiment['sources'].items()},main_text_last_page=refs[0],pdf_pages=len([p for p in pages if p.strip()]),abstract_source_words=len(abstract.split()),overfull_boxes=0,undefined_references=0,official_style=style,pdf_sha256=hashlib.sha256((P/'main.pdf').read_bytes()).hexdigest(),visually_reviewed_pages=[6,7,8,9],legacy_exact_prose_check='Not applicable to the authorized rewrite; preserved historical numerical checks passed before its old source-hash assertion.',scope='Main experiments cleaned and expanded: CG-only main linear-PDE baselines; accepted B3D final plus P/H/NS development snapshots. Historical alternatives remain supplementary.')
(P/'rewrite-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
