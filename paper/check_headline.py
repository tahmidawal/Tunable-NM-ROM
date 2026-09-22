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
# hires-burgers intake: held-out rows shown beside the development rows at both fine meshes; no lane slot left pending
hb = {(r['problem'], r['intervals']) for r in prov['rows'] if r['problem'].startswith('Burgers') and r['intervals'] >= 2048}
assert hb == {(p, n) for p in ('Burgers', 'Burgers (held-out cases)') for n in (2048, 4096)}, hb
assert 'pending:' not in tex and 'reserved' not in tex
# 2026-09-21 user decision: exactly three "results incoming" slots (paused lanes), no number in them
_inc = [l for l in tex.splitlines() if 'results incoming' in l]
assert len(_inc) == 1 and all(l.count('&') == 2 and r'\multicolumn{6}{l}{\emph{results incoming}}' in l for l in _inc), _inc
assert {i['lane'] for i in prov['incoming']} == {'burgers-heldout'}
# 2026-09-21 burgers-eqcert intake (audited blobs @176b2a9a): re-derive the confirmed-rule rows, the FOM rule and every label number
_eq = {k: json.loads((P / f'evidence/headline-2026-09-20/{k}.json').read_bytes()) for k in man if k.startswith('eqcert_')}
assert set(_eq) == {'eqcert_summary', 'eqcert_bc256', 'eqcert_bc256b', 'eqcert_bc512', 'eqcert_bc1024', 'eqcert_bc2048b'}
for k in _eq:
    assert man[k]['commit'].startswith('176b2a9a') and man[k]['read'] == 'committed blob'
    if k != 'eqcert_summary': assert _eq['eqcert_summary']['sources'][k[7:]]['sha256'] == man[k]['sha256'] and _eq['eqcert_summary']['sources'][k[7:]]['accepted']
_cr = {r['intervals']: r for r in prov['rows'] if r['problem'] == 'Burgers, confirmed rule'}
assert set(_cr) == {512, 1024}
for n, r in _cr.items():
    d = _eq[f'eqcert_bc{n}']; v = d['verdict']; a_, f_ = d['table'][v['accurate']['arm']], d['table'][v['fast']['arm']]
    assert v['certified_rule_exists'] and a_['confirmation_pass'] and r['accurate']['arm'] == a_['name'] and r['fast']['arm'] == f_['name']
    assert r['accurate']['error_pct'] == a_['worst_evolved_percent'] and r['accurate']['ms'] == a_['median_gpu_ms'] and r['fast']['ms'] == f_['median_gpu_ms']
    fo = d['table'][r['fom']['arm']]; assert r['fom']['ms'] == fo['median_gpu_ms'] and r['fom']['arm'] == a_['fom_by_paper_rule']
    assert abs(r['accurate']['speedup'] - a_['speedup_gpu']) < 1e-12 and r['accurate']['speedup'] < 1   # slower than Newton--BiCGStab (text: through 2048^2)
    assert set(r['fom']['candidates']) == {k for k, t in d['table'].items() if t['family'] == 'fom'}
_hn = (P / 'tables/headline-numbers.tex').read_text()
def _m(name): return re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', _hn).group(1)
_s512 = _eq['eqcert_bc512']['arm_status']['q256_M1088_scaled_g0p001_fast_chol_clip_lamcarry_pred2']
assert (_m('nEqcScaledPassFiveTwelve'), _m('nEqcScaledDrawsFiveTwelve')) == (str(_s512['draws_passed']), str(_s512['draws'])) and not _s512['confirmation_pass']
assert not _eq['eqcert_bc256']['arm_status']['q256_M1088_scaled_g0p001_fast_chol_clip_lamcarry_pred2']['confirmation_pass']
assert not _eq['eqcert_bc256']['verdict']['certified_rule_exists']
_l = _eq['eqcert_bc2048b']['table']['q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2']
assert _m('nEqcLatConfRho') == f"{_l['confirmation_rho_max']:.4f}" and _m('nEqcBar') == '0.116' and 0 < 0.116 - _l['confirmation_rho_max'] < 5e-4
assert _m('nEqcConfRhoTenTwentyFour') == f"{_eq['eqcert_bc1024']['table'][_eq['eqcert_bc1024']['verdict']['accurate']['arm']]['confirmation_rho_max']:.4f}"
_pk = _eq['eqcert_summary']['combined_256']['pick']; _t = _eq['eqcert_bc256b']['table'][_pk]
assert (_m('nEqcFollowErr'), _m('nEqcFollowS')) == (f"{_t['worst_evolved_percent']:.3f}", f"{_t['speedup_gpu']:.3f}") and _m('nEqcFollowDraws') == '12'
_mt = main.split(r'\bibliographystyle')[0] if 'main' in dir() else (P / 'main.tex').read_text()
assert 'passed the\nheld-out bar in its single draw' not in (P / 'main.tex').read_text()   # stale 256^2/512^2 rule wording removed
# rule identity per Burgers panel row (coordinator follow-up): the lane's re-draw result is attached only to the rule it tested
_bp = json.loads((P / 'evidence/headline-2026-09-20/burgers_panel.json').read_bytes())
_pm = {(r['mesh'], r['subject'], r['metric']): r['value'] for r in _bp['rows']}
_bt = (P / 'tables/TH_headline.tex').read_text()
for n, same, mark in ((256, True, '$^{s}$'), (512, False, '$^{x}$')):
    r = [x for x in prov['rows'] if x['problem'] == 'Burgers' and x['intervals'] == n][0]; a_ = r['accurate']; arm = a_['arm']
    assert (a_['rule']['set'], a_['rule']['m'], a_['rule']['file_sha256']) == (_pm[(n, arm, 'rule_set')], _pm[(n, arm, 'rule_m')], _pm[(n, arm, 'rule_file_sha256')])
    lane = [x for x in _eq[f'eqcert_bc{n}']['rules'] if x['q'] == 256 and x['M'] == 1088 and x['rule'] == 'scaled'][0]
    assert lane['refit'] is None and lane['source'][0]['sha256'] == a_['rule']['file_sha256']        # same source file
    assert (lane['m'] == a_['rule']['m'] and a_['rule']['set'] == 'eqtop' and lane['source'][0]['source_mesh'] == n) == same   # identical rule only at 256^2
    assert a_['rule']['redraw']['same_rule_as_lane_scaled'] == same and a_['eq'] == ('not-confirmed' if same else 'single-draw-refit')
    assert f"{e_(a_['error_pct'])}{mark}" in _bt if (e_ := (lambda x: f'{x:.2f}' if x >= 0.1 else f'{x:.3f}')) else False
assert _m('nEqcRowRuleMFiveTwelve') == '2438' and _m('nEqcLaneRuleM') == '2560'
assert 'marginal' not in (P / 'main.tex').read_text().split(r'\bibliographystyle')[0]
# Figure 2: every ratio reproduces from ms; uncertified rungs are marked in the figure record
_tf = json.loads((P / 'figures/fig_tunability_rank.json').read_text())
assert _tf['evidence_sha256'] == hashlib.sha256((P / 'tables/headline-provenance.json').read_bytes()).hexdigest()
for s_ in prov['tunability']:
    for r_ in s_['rungs']:
        assert abs(s_['fom']['ms'] / r_['ms'] - r_['speedup']) < 1e-9 and s_['fom']['err'] <= min(x['err'] for x in s_['rungs'])
    assert any(not r_['certified'] for r_ in s_['rungs']) == s_['series'].startswith('Burgers')
# nmrom-baselines intake: Table 2 carries no speed or ratio column (our rows there are the unoptimised dense path); no reserved slot
t2 = (P / 'tables/TH_nmrom_baselines.tex').read_text()
assert r'\times' not in t2 and ' ms' not in t2 and 'reserved' not in t2
assert r'\input{tables/TH_nmrom_baselines_appx}' in (P / 'sections/appendix.tex').read_text()
assert r'\input{tables/TH_heat_hires}' in (P / 'sections/appendix.tex').read_text()
# 2026-09-21 user decision: no direct/spectral/sparse-direct/coarse-grid solver is featured in the rendered paper
import subprocess
_pdf = subprocess.check_output(['pdftotext', str(P / 'main.pdf'), '-']).decode()
_hits = re.findall(r'(?i)\bDST\b|sine[- ]?transform|SuperLU|sparse[- ]direct|coarse[- ]grid|Swarztrauber|FFT[- ]based|fast transform\b', _pdf)
assert not _hits, _hits
# 2026-09-21 user decision (one FOM rule): every Table 1 FOM is the fastest tested setting of the named solver with
# error <= the row's accurate setting (the single setting where there is no accurate one), from the recorded candidates
for r in prov['rows']:
    ref = (r['accurate'] or r['fast'])['error_pct']; C = r['fom']['candidates']
    ok = {k: v for k, v in C.items() if v['err'] <= ref + 1e-12}
    assert r['fom']['arm'] == min(ok, key=lambda k: ok[k]['ms']), (r['problem'], r['intervals'], r['fom']['arm'])
    assert abs(C[r['fom']['arm']]['ms'] - r['fom']['ms']) < 1e-9
    if len(C) == 1: assert r['fom'].get('selection', '').startswith('record: Fastest'), (r['problem'], r['intervals'])
fails = prov['failures']; assert {f['source'] for f in fails} - {'burgers3d', 'ns3d', 'wave', 'heat3d'} <= {k for k in man if 'hires-heat' in k}
assert not any(r['problem'] == 'Heat' and r['dim'] == 3 for r in prov['rows'])      # the earlier Heat 3D model is in neither table now
# 2026-09-22 heat3d-bank intake: Table 1 'Heat (new bank)' rows re-derived from the pinned panel-A blob (sealed cohort 921099)
assert not any('heat' in f['source'].lower() or f['problem'].startswith('Heat') for f in fails)   # Heat 3D left the failures table
_ha = json.loads((P / 'evidence/headline-2026-09-20/heat3db_panel_a.json').read_bytes())
assert man['heat3db_panel_a']['commit'].startswith('55165375') and _ha['audit_passed'] and _ha['metadata']['backend'] == 'gpu'
_HN = {m['intervals']: {x['method']: x for x in m['rows']} for m in _ha['meshes'] if m['cohort'] == 'sealed_921099_never_opened'}
_h3 = {(r['problem'], r['intervals']): r for r in prov['rows'] if r['dim'] == 3 and r['problem'].startswith('Heat (new bank')}
assert set(_h3) == {(p_, n) for p_ in ('Heat (new bank)', 'Heat (new bank, batched fit)') for n in (32, 64, 128)}
_hl = (P / 'tables/TH_headline.tex').read_text()
for (p_, n), r in _h3.items():
    arms = ('nmrom_q0_field_cn', 'nmrom_q288_field_cn') if p_ == 'Heat (new bank)' else ('nmrom_q0_field_direct_tol1e-4_chol', 'nmrom_q288_field_direct_tol1e-4_chol')
    for s_, a_ in zip(('fast', 'accurate'), arms):
        x = _HN[n][a_]; assert r[s_]['arm'] == a_ and r[s_]['error_pct'] == 100 * x['error_all_times_worst'] and r[s_]['ms'] == x['device_ms_median']
        assert r[s_]['nonstationary'] == x['failures'] and x['cases'] == 64
    assert r['accurate']['error_pct'] <= 1 and r['error_convention'] == 'same-grid, all times' and r['cohort'] == 'final'   # meets the 1 % all-times target
    cand = {k: v for k, v in _HN[n].items() if k.startswith('fom_cncg_') and v['failures'] == 0}
    assert set(r['fom']['candidates']) == set(cand) and r['fom']['arm'] == _HN[n][arms[1]]['fastest_fom_error_le_rom_all_times']['method']
    assert r['fom']['ms'] == cand[r['fom']['arm']]['device_ms_median']
    if r['accurate']['nonstationary']: assert r'---$^{n}$' in _hl                    # a solve that missed its rule does not enter a speedup
# the text's speed claims: slower than CN--CG with CN stepping at every mesh, faster only with the batched fit at 128^3
assert all(r['accurate']['speedup'] < 1 for (p_, n), r in _h3.items() if p_ == 'Heat (new bank)')
assert [n for (p_, n), r in _h3.items() if p_ != 'Heat (new bank)' and r['accurate']['speedup'] > 1] == [128]
# Navier--Stokes follow-up sentence (ns3d-grok diag07, a different model): macros re-derived from the pinned blob
_ng = json.loads((P / 'evidence/headline-2026-09-20/ns3d_grok_diag07.json').read_bytes()); _cs = _ng['coeff']['stats']
assert man['ns3d_grok_diag07']['commit'].startswith('8852b7cd') and _ng['final_cohort_opened'] and not _ng['smoke']
_hn2 = (P / 'tables/headline-numbers.tex').read_text()
def _m2(name): return re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*)\}', _hn2).group(1)
_ff = min((v for v in _ng['fom'].values() if v['stats']['evolved_worst'] <= _cs['evolved_worst']), key=lambda v: v['median_ms'])
assert _m2('nNsGrokWorst') == f"{100 * _cs['evolved_worst']:.2f}" and _m2('nNsGrokOver') == str(_cs['cases_evolved_over_target']) == '0'
assert _m2('nNsGrokCases') == str(_cs['cases']) and abs(float(_m2('nNsGrokS')) - _ff['median_ms'] / _ng['coeff']['median_ms']) < 0.005 and _ff['dt'] == 0.01
_mt = (P / 'main.tex').read_text().split(r'\bibliographystyle')[0]
assert 'four problems' not in _mt and 'nFailHeatThree' not in _mt and 'nHeatThreeInit' not in _mt
assert not any(r['source'] in ('burgers3d', 'ns3d', 'wave') for r in prov['rows'])
main = (P / 'main.tex').read_text()
abstract = main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert not re.search(r'\d+\.\d', abstract), 'abstract numbers must be generated macros'
macros = set(re.findall(r'\\(n[A-Za-z]+)', abstract))
defined = set(re.findall(r'\\newcommand\{\\(n\w+)\}', (P / 'tables/headline-numbers.tex').read_text() + (P / 'tables/numbers.tex').read_text()))
assert macros and macros <= defined, macros - defined
assert all(m.startswith(('nHead', 'nQxm', 'nHires', 'nHeat', 'nBurg', 'nBase')) for m in macros), macros   # only generated-table numbers
assert 'full pre-registered criterion' not in main and re.search(r'pre-registered\s+secondary criterion', main) and 'knob bar' not in main   # 2026-09-21: 'knob bar' jargon replaced
assert r'\label{tab:knobs-main}' in (P / 'sections/appendix.tex').read_text()   # 2026-09-21: knob table moved to the appendix for the page budget
for label in ('tab:headline', 'tab:tunability', 'tab:failures', 'tab:nmrom-baselines'):
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
# 2026-09-21 user decision: the two figures are replaced by tables; the tunability table is re-derived here
assert 'fig_speedup_resolution' not in main and 'fig_tunability_rank' not in main and 'fig:speedup' not in main and 'fig:tunability' not in main
assert r'\input{tables/TH_tunability}' in main.split(r'\bibliographystyle')[0]
import importlib.util as _iu
_sp = _iu.spec_from_file_location('gtt', P / 'gen_tunability_table.py'); _g = _iu.module_from_spec(_sp); _sp.loader.exec_module(_g)
_tt = (P / 'tables/TH_tunability.tex').read_text()
assert _g.render()[0] == _tt, 'TH_tunability.tex is stale or hand-edited'
_rows = [l for l in _tt.splitlines() if l.endswith(r'\\') and ('$q=' in l or 'FOM:' in l)]
_n = 0
for s_ in prov['tunability']:
    fm = s_['fom']
    for r_ in s_['rungs']:
        exp = [_g.e(r_['err']), _g.ms(r_['ms']), _g.sp(fm['ms'] / r_['ms'])]
        hits = [l for l in _rows if l.split(' & ')[0] == _g.setting(r_, s_['series'].split()[0]) and ' & '.join(exp) in l]
        assert hits, (s_['series'], r_['arm']); _n += 1
    assert any('FOM:' in l and ' & '.join([_g.e(fm['err']), _g.ms(fm['ms'])]) in l for l in _rows), s_['series']
assert _n == sum(len(s_['rungs']) for s_ in prov['tunability'])
print(json.dumps(dict(passed=True, rows=len(prov['rows']), bold_speedups=bold, failure_rows=len(fails), figure_points=sum(len(v) for v in fig['series'].values()),
                      pending_lane_slots=tex.count('pending:'), abstract_macros=sorted(macros)), indent=2))
