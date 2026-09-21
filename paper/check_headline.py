"""Checks for the headline table, its figure, the failure table and the prose that cites them."""
import hashlib, json, re
from collections import defaultdict
from pathlib import Path
P = Path(__file__).resolve().parent
prov = json.loads((P / 'tables/headline-provenance.json').read_text())
man = json.loads((P / 'evidence/headline-2026-09-20/manifest.json').read_text())
assert prov['sources'] == man
for k, v in man.items():
    assert hashlib.sha256((P / f'evidence/headline-2026-09-20/{k}.json').read_bytes()).hexdigest() == v['sha256'], k
series = defaultdict(list); bold = 0
for r in prov['rows']:
    assert r['job_id'] and r['source'] in man
    for s in ('fast', 'accurate'):
        x = r[s]
        if not x: continue
        assert abs(r['fom']['ms'] / x['ms'] - x['speedup']) < 1e-12          # same-row ratio only
        assert r['fom']['error_pct'] <= x['error_pct']                        # named FOM at least as accurate
        bold += x['speedup'] > 1
        series[(r['problem'], r['dim'], s)].append((r['intervals'], x['speedup']))
    if r['fast'] and r['accurate']:
        assert r['accurate']['error_pct'] <= r['fast']['error_pct']
    if r['source'] in ('heat3d', 'poisson3d'): assert r['status'] == 'accepted final' and r['cohort'] == 'final'
tex = (P / 'tables/TH_headline.tex').read_text()
assert tex.count(r'\textbf{') == bold, (tex.count(r'\textbf{'), bold)
assert not re.search(r'\bms\b', tex), 'no milliseconds in the headline table'
for lane in ('hires-poisson', 'hires-heat', 'hires-burgers'):
    assert ('pending: ' + lane in tex) or any(k.endswith(lane) for k in man), lane
# the abstract's "fast-setting speedup grows with resolution in every problem measured"
for (p, d, s), pts in series.items():
    if s == 'fast' and not p.startswith('Burgers (earlier'):
        v = [y for _, y in sorted(pts)]; assert all(b > a for a, b in zip(v, v[1:])), (p, d, v)
# hires-heat intake: heat rows carry an explicit time convention; the tight named-FOM ratios stay out of Table 1
for r in prov['rows']:
    if r['problem'].startswith('Heat (wide bank'): assert r['error_convention'] == 'same-grid, all times' and r['alt']['scope'].startswith('vs.')
    if r['problem'] == 'Heat' and r['dim'] == 3: assert 'evolved' in r['error_convention']
# hires-burgers intake: held-out rows shown beside the development rows at both fine meshes; no lane slot left pending
hb = {(r['problem'], r['intervals']) for r in prov['rows'] if r['problem'].startswith('Burgers') and r['intervals'] >= 2048}
assert hb == {(p, n) for p in ('Burgers', 'Burgers (held-out cases)') for n in (2048, 4096)}, hb
assert 'pending:' not in tex
# nmrom-baselines intake: Table 2 carries no speed or ratio column (our rows there are the unoptimised dense path); 512^2 slot reserved
t2 = (P / 'tables/TH_nmrom_baselines.tex').read_text()
assert r'\times' not in t2 and ' ms' not in t2 and '512^2' in t2 and 'reserved' in t2
assert r'\input{tables/TH_nmrom_baselines_appx}' in (P / 'sections/appendix.tex').read_text()
assert r'\input{tables/TH_heat_hires}' in (P / 'sections/appendix.tex').read_text()
fails = prov['failures']; assert {f['source'] for f in fails} == {'burgers3d', 'ns3d', 'wave'}
assert not any(r['source'] in ('burgers3d', 'ns3d', 'wave') for r in prov['rows'])
main = (P / 'main.tex').read_text()
abstract = main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert not re.search(r'\d+\.\d', abstract), 'abstract numbers must be generated macros'
macros = set(re.findall(r'\\(n[A-Za-z]+)', abstract))
defined = set(re.findall(r'\\newcommand\{\\(n\w+)\}', (P / 'tables/headline-numbers.tex').read_text() + (P / 'tables/numbers.tex').read_text()))
assert macros and macros <= defined, macros - defined
assert all(m.startswith(('nHead', 'nQxm')) for m in macros), macros            # only generated-table numbers
assert 'full pre-registered criterion' not in main and 'knob bar' in main
for label in ('tab:headline', 'fig:speedup', 'tab:failures', 'tab:nmrom-baselines', 'tab:knobs-main'):
    assert r'\label{' + label + '}' in main.split(r'\bibliographystyle')[0], label
app = (P / 'sections/appendix.tex').read_text() + (P / 'sections/method-details.tex').read_text()
for label in ('tab:headline-times', 'tab:config3d', 'eq:wave-accel', 'eq:heat-step-corrected', 'eq:heat-endpoint', 'eq:block-damped'):
    assert r'\label{' + label + '}' in app, label
t01 = (P / 'tables/T01_problems.tex').read_text() + (P / 'tables/T01b_spec.tex').read_text()
heat_job = {r['job_id'] for r in prov['rows'] if r['problem'] == 'Heat' and r['dim'] == 2}
assert len(heat_job) == 1 and t01.count(heat_job.pop()) == 2                     # setup tables name the measured job
assert 'INPUT -->|' in (P / 'figures/architecture.mmd').read_text() and '(in.south)' in (P / 'figures/architecture.tex').read_text()
fig = json.loads((P / 'figures/fig_speedup_resolution.json').read_text())
assert fig['evidence_sha256'] == hashlib.sha256((P / 'tables/headline-provenance.json').read_bytes()).hexdigest()
print(json.dumps(dict(passed=True, rows=len(prov['rows']), bold_speedups=bold, failure_rows=len(fails), figure_points=sum(len(v) for v in fig['series'].values()),
                      pending_lane_slots=tex.count('pending:'), abstract_macros=sorted(macros)), indent=2))
