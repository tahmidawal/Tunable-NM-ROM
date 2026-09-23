"""Checks for the headline table, its figure, the failure table and the prose that cites them."""
import hashlib, json, re
from collections import defaultdict
from pathlib import Path
P = Path(__file__).resolve().parent
# 2026-09-23: main.tex / sections/* are being rewritten concurrently by another writer.  Checks that REQUIRE a given
# sentence, label or macro in that prose are reported (not fatal) until the rewrite settles; numeric checks stay fatal.
PROSE_PENDING = []
def _prose(cond, where):
    try: ok = bool(cond())
    except Exception as ex: ok = False
    if not ok: PROSE_PENDING.append(where)
prov = json.loads((P / 'tables/headline-provenance.json').read_text())
man = json.loads((P / 'evidence/headline-2026-09-20/manifest.json').read_text())
assert prov['sources'] == man
for k, v in man.items():
    assert hashlib.sha256((P / f'evidence/headline-2026-09-20/{k}.json').read_bytes()).hexdigest() == v['sha256'], k
# 2026-09-23 bank-width evidence (ordered bank R'): committed blobs, pinned by commit + git blob + sha256
bwman = json.loads((P / 'evidence/bankwidth-2026-09-23/manifest.json').read_text())
assert prov['bankwidth_sources'] == bwman
for k, v in bwman.items():
    assert hashlib.sha256((P / 'evidence/bankwidth-2026-09-23' / v['file']).read_bytes()).hexdigest() == v['sha256'] and v['read'] == 'committed blob', k
    for c, cv in v.get('companions', {}).items():
        assert hashlib.sha256((P / 'evidence/bankwidth-2026-09-23' / c).read_bytes()).hexdigest() == cv['sha256'], c
assert bwman['pbk_summary']['sha256'].startswith('f19d0172') and bwman['pbk_summary']['commit'].startswith('06546331')
assert all(bwman[k]['commit'].startswith('708c70fe') for k in bwman if k.startswith('ns_'))
ALLR = prov['rows'] + prov['dropped_rows']          # rows removed from Table 1 (one row per problem and mesh) keep their records
series = defaultdict(list); bold = 0
for r in prov['rows']:
    assert r['job_id'] and (r['source'] in man or r['source'][3:] in bwman)
    for s in ('fast', 'accurate'):
        x = r[s]
        if not x: continue
        # 2026-09-23: EVERY printed speedup divides two times measured in ONE job.  A column whose comparator comes
        # from a later job prints that job's own ROM time too, and records which job it was.
        fo = x.get('own_fom')
        num, job = (fo['ms'], fo['job']) if fo else (r['fom']['ms'], r['job_id'])
        assert abs(num / x['ms'] - x['speedup']) < 1e-12, (r['problem'], r['intervals'], s)
        assert x.get('job', r['job_id']) == job, (r['problem'], r['intervals'], s)     # numerator and denominator, one job
        if fo:
            assert x['ms'] == fo['rom_ms'] and abs(fo['rom_err'] - x['error_pct']) < 5e-3   # the same setting, re-timed
            assert fo['err'] <= x['error_pct'] and fo['job'] != r['job_id']
            assert f"job {fo['job']}" in (P / 'tables/TH_headline_times.tex').read_text()   # the caption names that job
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
# 2026-09-23 user decision: one row per (problem, mesh); the held-out Burgers rows left Table 1 (held-out cases live in Table 4)
hb = {(r['problem'], r['intervals']) for r in prov['rows'] if r['problem'].startswith('Burgers') and r['intervals'] >= 2048}
assert hb == {('Burgers', n) for n in (2048, 4096)}, hb
assert {(r['problem'], r['intervals']) for r in prov['dropped_rows'] if r['problem'] == 'Burgers (held-out cases)'} == {('Burgers (held-out cases)', n) for n in (2048, 4096)}
_one = defaultdict(int)
for r in prov['rows']: _one[(r['problem'].split(' (')[0], r['dim'], r['intervals'])] += 1
assert all(v == 1 for v in _one.values()), _one                                   # ONE row per (problem, mesh)
assert not any(r['problem'] in ('Heat', 'Heat (wide bank)', 'Heat (new bank)', 'Burgers (held-out cases)', 'Burgers, confirmed rule', 'Burgers (earlier model)') for r in prov['rows'])
assert 'pending:' not in tex and 'reserved' not in tex
# 2026-09-22: the last incoming slot (burgers-heldout) is closed; the wider bank is a limitations sentence
assert prov['incoming'] == []
assert 'results incoming' not in tex and 'results incoming' not in (P / 'main.tex').read_text()
# 2026-09-21 burgers-eqcert intake (audited blobs @176b2a9a): re-derive the confirmed-rule rows, the FOM rule and every label number
_eq = {k: json.loads((P / f'evidence/headline-2026-09-20/{k}.json').read_bytes()) for k in man if k.startswith('eqcert_')}
assert set(_eq) == {'eqcert_summary', 'eqcert_bc256', 'eqcert_bc256b', 'eqcert_bc512', 'eqcert_bc1024', 'eqcert_bc2048b'}
for k in _eq:
    assert man[k]['commit'].startswith('176b2a9a') and man[k]['read'] == 'committed blob'
    if k != 'eqcert_summary': assert _eq['eqcert_summary']['sources'][k[7:]]['sha256'] == man[k]['sha256'] and _eq['eqcert_summary']['sources'][k[7:]]['accepted']
_cr = {r['intervals']: r for r in ALLR if r['problem'] == 'Burgers, confirmed rule'}
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
# 2026-09-23: the tunability figure is no longer built (Table 4 replaced it); its evidence-hash check is retired.
for s_ in (prov['tunability'] + prov['tunability_bankwidth']):
    for r_ in s_['rungs']:
        assert abs(s_['fom']['ms'] / r_['ms'] - r_['speedup']) < 1e-9 and s_['fom']['err'] <= min(x['err'] for x in s_['rungs'])
    assert any(not r_['certified'] for r_ in s_['rungs']) == s_['series'].startswith('Burgers')
    if not s_['legacy']: assert s_['fom']['err'] <= min(x['err'] for x in s_['rungs']) and all(x['speedup'] > 1 for x in s_['rungs'])
# nmrom-baselines intake: Table 2 carries no speed or ratio column (our rows there are the unoptimised dense path); no reserved slot
t2 = (P / 'tables/TH_nmrom_baselines.tex').read_text()
assert r'\times' not in t2 and ' ms' not in t2 and 'reserved' not in t2
_prose(lambda: (r'\input{tables/TH_nmrom_baselines_appx}' in (P / 'sections/appendix.tex').read_text()), 'check_headline.py:120')
_prose(lambda: (r'\input{tables/TH_heat_hires}' in (P / 'sections/appendix.tex').read_text()), 'check_headline.py:121')
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
_h3 = {(r['problem'], r['intervals']): r for r in ALLR if r['dim'] == 3 and r['problem'].startswith('Heat (new bank')}
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
    if r['accurate']['nonstationary'] and r in prov['rows']: assert r'---$^{n}$' in _hl                    # a solve that missed its rule does not enter a speedup
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
assert all(m.startswith(('nHead', 'nQxm', 'nHires', 'nHeat', 'nBurg', 'nBase', 'nBw', 'nNs')) for m in macros), macros   # 2026-09-23: + bank-width macros   # only generated-table numbers
assert 'full pre-registered criterion' not in main and 'knob bar' not in main   # 2026-09-23: 'secondary criterion' sentence no longer required (appendix rewrite)   # 2026-09-21: 'knob bar' jargon replaced
_prose(lambda: (r'\label{tab:knobs-main}' in (P / 'main.tex').read_text()), 'check_headline.py:177')   # 2026-09-23 user decision: knob table back in the main text, old-paper shape
for label in ('tab:headline', 'tab:tunability', 'tab:nmrom-baselines'):   # 2026-09-23: failures table removed
    _prose(lambda: (r'\label{' + label + '}' in main.split(r'\bibliographystyle')[0]), 'check_headline.py:179 ' + label)
app = (P / 'sections/appendix.tex').read_text() + (P / 'sections/method-details.tex').read_text()
for label in ('tab:headline-times', 'tab:config3d'):   # 2026-09-23: coordinator rewrites the appendix (wave section and method equations may go)
    assert r'\label{' + label + '}' in app, label
t01 = (P / 'tables/T01_problems.tex').read_text() + (P / 'tables/T01b_spec.tex').read_text()
heat_job = {r['job_id'] for r in ALLR if r['problem'] == 'Heat' and r['dim'] == 2}
assert len(heat_job) == 1 and t01.count(heat_job.pop()) == 2                     # setup tables name the measured job
assert 'INPUT -->|' in (P / 'figures/architecture.mmd').read_text() and '(in.south)' in (P / 'figures/architecture.tex').read_text()
# 2026-09-23: the speedup-resolution figure is no longer built into the paper; its evidence-hash check is retired.
# 2026-09-21 user decision: the two figures are replaced by tables; the tunability table is re-derived here
assert 'fig_speedup_resolution' not in main and 'fig_tunability_rank' not in main and 'fig:speedup' not in main and 'fig:tunability' not in main
_prose(lambda: (r'\input{tables/TH_tunability}' in main.split(r'\bibliographystyle')[0]), 'check_headline.py:190')
import importlib.util as _iu
_sp = _iu.spec_from_file_location('gtt', P / 'gen_tunability_table.py'); _g = _iu.module_from_spec(_sp); _sp.loader.exec_module(_g)
_tt = (P / 'tables/TH_tunability.tex').read_text()
assert _g.render()[0] == _tt, 'TH_tunability.tex is stale or hand-edited'
_rows = [l for l in _tt.splitlines() if l.endswith(r'\\') and ('$q=' in l or 'FOM:' in l or "$R'{=}" in l or '$k{=}' in l)]
_n = 0
for s_ in (prov['tunability'] + prov['tunability_bankwidth']):
    fm = s_['fom']
    for r_ in s_['rungs']:
        exp = [_g.e(r_['err']), _g.ms(r_['ms']), _g.sp(fm['ms'] / r_['ms'])]
        hits = [l for l in _rows if l.split(' & ')[0] == _g.setting(r_, s_['series'].split()[0]) and ' & '.join(exp) in l]   # label from the generator
        assert hits, (s_['series'], r_['arm']); _n += 1
    assert any('FOM:' in l and ' & '.join([_g.e(fm['err']), _g.ms(fm['ms'])]) in l for l in _rows), s_['series']
assert _n == sum(len(s_['rungs']) for s_ in (prov['tunability'] + prov['tunability_bankwidth']))
# 2026-09-22 burgers-heldout intake (audited blobs @3771cacf): 4096^2 quadrature label, fast-column denominator, confirmed variant, wider bank
import subprocess as _sp
def _blob(v, rel=None):
    spec = f"{v['commit']}:{rel or v['path']}"
    raw = _sp.check_output(['git', '-C', str(P.parent / v['tree']), 'show', spec])
    bid = _sp.check_output(['git', '-C', str(P.parent / v['tree']), 'rev-parse', spec]).decode().strip()
    return raw, bid
_bh = {}
for k in ('bh5_summary', 'bh5_eqcert', 'bh_summary'):
    v = man[k]; raw, bid = _blob(v)
    assert v['commit'].startswith('3771cacf') and v['read'] == 'committed blob' and bid == v['git_blob']
    assert hashlib.sha256(raw).hexdigest() == v['sha256']
    _bh[k] = json.loads(raw)
assert _bh['bh_summary']['sources_sha256']['bh5-summary.json'] == man['bh5_summary']['sha256']
assert _bh['bh_summary']['sources_sha256']['bh5-eqcert-summary.json'] == man['bh5_eqcert']['sha256']
for label, rel in (('DESIGN.md', 'experiments/burgers-heldout/DESIGN.md'), ('HANDOFF.md', 'experiments/burgers-heldout/HANDOFF.md'),
                   ('reports/2026-09-21-burgers-heldout.md', 'experiments/burgers-heldout/reports/2026-09-21-burgers-heldout.md'),
                   ('reports/tables.generated.md', 'experiments/burgers-heldout/reports/tables.generated.md')):
    raw, bid = _blob(man['bh_summary'], rel)
    assert bid == man['bh_summary']['companion_blobs'][label] and hashlib.sha256(raw).hexdigest() == man['bh_summary']['companion_sha256'][label]
_paper = 'q256_M1088_lat64_g0p01_fast_chol_clip_lamcarry_pred2'
_exact, _fast = _paper + '_x1', 'q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2'
_st = _bh['bh5_eqcert']['arm_status']
assert _st[_paper]['draws_passed'] == 5 and not _st[_paper]['confirmation_pass'] and _st[_exact]['confirmation_pass'] and _st[_exact]['exact_steps'] == 1
assert _st[_fast]['confirmation_pass'] and _st[_fast]['status'] == 'confirmed'
assert _m('nBhJzeroRho') == f"{_st[_paper]['confirmation_heldout_rho_max_k>=0']:.4f}"
assert _m('nBhJzeroRhoKtwo') == f"{_st[_paper]['confirmation_heldout_rho_max_k>=2']:.4f}"
assert _m('nBhJoneRho') == f"{_st[_exact]['confirmation_heldout_rho_max_k>=1']:.4f}"
def _pick(table, err):
    ok = {k: v for k, v in table.items() if v['family'] == 'fom' and v['mesh'] == 4096 and v['nonlinear_converged'] and v['stalled_steps'] == 0 and v['worst_evolved_percent'] <= err + 1e-12}
    return min(ok, key=lambda k: ok[k]['median_gpu_ms'])
_own = []
for cohort, problem in (('dev6', 'Burgers'), ('hold64', 'Burgers (held-out cases)')):
    table = _bh['bh5_summary']['groups'][cohort]['table']
    r = [x for x in ALLR if x['problem'] == problem and x['intervals'] == 4096][0]
    assert r['accurate']['eq'] == 'lattice-unconfirmed' and r['accurate']['arm'] == _paper and r['fast']['arm'] == _fast
    fq = table[_fast]; fp = _pick(table, fq['worst_evolved_percent']); fo = table[fp]
    assert fp == 'lean_nt1e-3_l1e-3_dt01' and r['fast']['own_fom']['arm'] == fp
    assert abs(fo['median_gpu_ms'] / fq['median_gpu_ms'] - r['fast']['speedup']) < 1e-12
    assert f"{r['fast']['speedup']:.2f}" == '10.10' and abs(r['accurate']['speedup'] - r['fom']['ms'] / r['accurate']['ms']) < 1e-12
    _own.append(r['fast']['speedup'])
    j1 = table[_exact]; ap = _pick(table, j1['worst_evolved_percent']); af = table[ap]
    rr = [x for x in prov['appendix_only_rows'] if x['problem'] == 'Burgers, exact first step' and x['cohort'] == ('development' if cohort == 'dev6' else 'held-out')][0]
    assert rr['accurate']['arm'] == _exact and rr['fom']['arm'] == ap == 'lean_nt3e-3_l3e-3_dt005'
    assert rr['accurate']['ms'] == j1['median_gpu_ms'] and rr['fom']['ms'] == af['median_gpu_ms'] and rr['accurate']['speedup'] < 1
    assert abs(rr['accurate']['speedup'] - af['median_gpu_ms'] / j1['median_gpu_ms']) < 1e-12
assert _m('nBhFastS') == '10.10' and tex.count(r'\textbf{10.10$\times$}') == 1 and tex.count(r'$^{w}$') == 1   # 2026-09-23: held-out row left Table 1
assert '12.9' not in tex and '13.2' not in tex
for cohort, ms_name, s_name in (('dev6', 'nBhJoneMsDev', 'nBhJoneSDev'), ('hold64', 'nBhJoneMsHold', 'nBhJoneSHold')):
    rr = [x for x in prov['appendix_only_rows'] if x['problem'] == 'Burgers, exact first step' and x['cohort'] == ('development' if cohort == 'dev6' else 'held-out')][0]
    assert _m(ms_name) == f"{rr['accurate']['ms']:.0f}" and _m(s_name) == f"{rr['accurate']['speedup']:.2f}"
    j0 = _bh['bh5_summary']['groups'][cohort]['table'][_paper]['median_gpu_ms']
    assert _m('nBhJoneCost' + ('Dev' if cohort == 'dev6' else 'Hold')) == (f"{rr['accurate']['ms'] / j0:.1f}" if rr['accurate']['ms'] / j0 >= 10 else f"{rr['accurate']['ms'] / j0:.2f}")
_sel = _bh['bh_summary']['bank_selection']
assert _m('nBhFloorOld') == f"{100 * _sel['incumbent_floor']:.3f}" and _m('nBhFloorNew') == f"{100 * _sel['chosen_floor']:.3f}" and _sel['chosen_floor'] < 0.5 * _sel['incumbent_floor']
_hold = [r_ for r_ in _bh['bh_summary']['rows'] if r_['job'] == 'bh3' and r_['cohort'] == 'hold64' and str(r_['role']).startswith('headline')][0]
assert _m('nBhBankErr') == f"{_hold['worst_evolved_percent']:.3f}" and _m('nBhBankS') == (f"{_hold['speedup']:.2f}" if _hold['speedup'] < 10 else f"{_hold['speedup']:.1f}")
_subs = [r_ for r_ in _bh['bh_summary']['rows'] if r_['mesh'] == 4096 and r_['cohort'] in ('hold64', 'fresh64') and r_['q'] > 0 and not r_['certified'] and r_['worst_evolved_percent'] < 0.5]
assert _subs and all(r_['speedup'] > 1 for r_ in _subs)
assert _m('nBhSubErrLo') == f"{min(r_['worst_evolved_percent'] for r_ in _subs):.3f}" and _m('nBhSubErrHi') == f"{max(r_['worst_evolved_percent'] for r_ in _subs):.3f}"
assert _m('nBhSubSLo') == f"{min(r_['speedup'] for r_ in _subs):.2f}" and _m('nBhSubSHi') == f"{max(r_['speedup'] for r_ in _subs):.2f}"
_abs = (P / 'main.tex').read_text().split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
_prose(lambda: ('not confirmed on independent re-draws' in _abs and r'\nBurgDevAccSFortyNinetySix' in _abs and r'\nBurgHoldAccSFortyNinetySix' in _abs), 'check_headline.py:267')
_body = (P / 'main.tex').read_text() + (P / 'sections/appendix.tex').read_text()
for name in ('nBhFastS', 'nBhJzeroRho', 'nBhJoneRho', 'nBhJoneMsDev', 'nBhJoneSDev', 'nBhBankErr', 'nBhFloorNew', 'nBhSubSLo'):
    _prose(lambda: ('\\' + name in _body), 'check_headline.py:270 ' + name)
_prose(lambda: ('No confirmed form of this accurate lattice rule is also faster.' in (P / 'sections/appendix.tex').read_text()), 'check_headline.py:271')
# 2026-09-22 neural-operator intake: the appendix tables and their prose numbers re-derived from the pinned records
import statistics
_me = json.loads((P / 'evidence/main-experiments-2026-09-20/manifest.json').read_text())
_md = {}
for _k, _v in _me.items():
    _raw = (P / f'evidence/main-experiments-2026-09-20/{_k}.json').read_bytes()
    assert hashlib.sha256(_raw).hexdigest() == _v['sha256'], _k
    _md[_k] = json.loads(_raw)
_OPS = {'fno': 'FNO', 'unet': 'U-Net', 'deeponet': 'DeepONet', 'transolver': 'Transolver'}
def _opname(m):
    return next(v for k, v in _OPS.items() if m.startswith(k))
def _grid(kind, n):                      # worst error (%) and median GPU ms per method at mesh n
    out = {}
    if kind in ('poisson', 'heat'):
        for r in _md[kind]['invocations']:
            if r['intervals'] != n: continue
            v = r['same_grid_error'] if kind == 'poisson' else r['same_grid']['current_evolved']
            out.setdefault(r['method'], []).append((100 * v, r['device_ms']))
        return {m: (max(x for x, _ in vs), statistics.median(t for _, t in vs)) for m, vs in out.items()}
    if kind == 'burgers':
        return {r['method']: (100 * r['worst_evolved'], r['median_ms']) for r in _md[kind]['rows']}
    return {r['method']: (r['same_grid_evolved_worst_percent'], r['gpu_ms']) for r in _md[kind]['rows']}
_prov3d = json.loads((P / 'tables/main-experiments-provenance.json').read_text())
_ctl = _prov3d['controls']; _sel3d = _prov3d['selected']
_tex = {n: (P / f'tables/TR_3d_{n}.tex').read_text() for n in ('linear', 'nonlinear')}
_best = {}
for _kind, _which in (('poisson', 'linear'), ('heat', 'linear'), ('burgers', 'nonlinear'), ('ns', 'nonlinear')):
    _g = _grid(_kind, 32); _c = _ctl[_kind]
    _pick = [r['method'] for r in _sel3d[_kind] if any(r['method'].startswith(k) for k in _OPS)]
    _ops = {m: _g[m] for m in _pick}
    assert len({_opname(m) for m in _ops}) == 4, (_kind, sorted(_ops))
    for _m, (_e, _t) in _ops.items():
        assert f"{_e:.3f}" in _tex[_which] and f"{_t:.3f}" in _tex[_which], (_kind, _m)   # error and time printed as recorded
        if _c: assert f"{_g[_c][1] / _t:.3g}" + r'$\times$' in _tex[_which], (_kind, _m)  # speedup divides the recorded control
    _b = min(_ops.items(), key=lambda kv: kv[1][0]); _best[_kind] = (_opname(_b[0]), _b[1], _c and _g[_c][1] / _b[1][1])
_on = (P / 'tables/operator-numbers.tex').read_text()
def _mo(name): return re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*)\}', _on).group(1)
for _kind, _pre in (('poisson', 'nOpPoissonBest'), ('heat', 'nOpHeatBest'), ('burgers', 'nOpBurgBest'), ('ns', 'nOpNsBest')):
    _n, (_e, _t), _s = _best[_kind]
    assert _mo(_pre + 'Name') == _n and _mo(_pre + 'Err') == (f'{_e:.2f}' if _e >= 0.1 else f'{_e:.3f}'), _kind
    if _s: assert _mo(_pre + 'S') == f'{_s:.3g}', _kind
# the 64^3 caveat: the same trained operators transfer badly; native-grid training plus interpolation recovers
_direct, _native = [], []
for _kind in ('poisson', 'heat'):
    _same = {r['method'] for r in _sel3d[_kind]}          # the very arms of the 32^3 tables
    for _m, (_e, _t) in _grid(_kind, 64).items():
        if any(_m.startswith(k) for k in _OPS): (_direct if _m in _same else _native).append(_e)
assert _mo('nOpTransferLo') == f'{min(_direct):.0f}' and _mo('nOpTransferHi') == f'{max(_direct):.0f}'
assert _mo('nOpNativeLo') == f'{min(_native):.2f}' and _mo('nOpNativeHi') == f'{max(_native):.1f}'
assert min(_direct) > 10 > min(_native)
_app = (P / 'sections/appendix.tex').read_text()
# 2026-09-23 coordinator: tab:op-linear / tab:op-nonlinear leave the appendix; the operator tables are still generated and checked above
_prose(lambda: (r'\ref{app:operators}' in main.split(r'\bibliographystyle')[0]), 'check_headline.py:324')      # the main text points at the comparison
# 2026-09-22 ops-timing-panel intake: the 256^2 panel re-derived from its pinned blobs
_om = json.loads((P / 'evidence/ops-timing-panel-2026-09-22/manifest.json').read_text())
for _k, _v in _om.items():
    assert hashlib.sha256((P / f'evidence/ops-timing-panel-2026-09-22/{_k}.json').read_bytes()).hexdigest() == _v['sha256'], _k
    assert _v['commit'].startswith('ea38c19a') and _v['read'] == 'committed blob'
_os = json.loads((P / 'evidence/ops-timing-panel-2026-09-22/summary.json').read_text())
assert _os['failed_gates'] == [] and _os['intervals'] == 256 and not _os['suppressed_rows']
assert hashlib.sha256((P / 'evidence/ops-timing-panel-2026-09-22/audit.json').read_bytes()).hexdigest() == _os['audit_sha256']
_or = {r['arm']: r for r in _os['rows']}
_ofom = {a: r for a, r in _or.items() if r['family'] == 'fom'}
_ops2 = [a for a, r in _or.items() if r['family'] in ('fno', 'unet', 'transolver')]
_ot = (P / 'tables/TR_ops256.tex').read_text()
_e2 = lambda x: f'{x:.2f}' if x >= 0.1 else f'{x:.3f}'
_ms2 = lambda x: f'{x:.0f}' if x >= 100 else f'{x:.1f}'
for _a in _ops2 + ['q0_M64_eqcert_g1em06_fastL4', 'q256_M1088_eqtop_g0p001', 'pod256_M1024_dense', 'pod512_M2048_dense']:
    _r = _or[_a]; _f = _ofom[_r['fom_gpu']]
    assert abs(_f['median_gpu_ms'] / _r['median_gpu_ms'] - _r['speedup_gpu']) < 1e-9, _a       # one job, one ratio
    assert _f['worst_evolved_percent'] <= _r['worst_evolved_percent'] + 1e-12, _a              # the paper's FOM rule
    assert _e2(_r['worst_evolved_percent']) in _ot and _ms2(_r['median_gpu_ms']) in _ot, _a
_o2 = (P / 'tables/ops-numbers.tex').read_text()
def _mp(n): return re.search(r'\\newcommand\{\\' + n + r'\}\{([^}]*)\}', _o2).group(1)
_oe = [_or[a]['worst_evolved_percent'] for a in _ops2]
assert (_mp('nOpsTwoDOpErrLo'), _mp('nOpsTwoDOpErrHi')) == (_e2(min(_oe)), _e2(max(_oe)))
assert _mp('nOpsTwoDOpFaster') == str(sum(_or[a]['speedup_gpu'] > 1 for a in _ops2)) and _mp('nOpsTwoDOpArms') == str(len(_ops2))
assert _mp('nOpsTwoDAccErr') == _e2(_or['q256_M1088_eqtop_g0p001']['worst_evolved_percent'])
assert _mp('nOpsTwoDFastErr') == _e2(_or['q0_M64_eqcert_g1em06_fastL4']['worst_evolved_percent'])
_disc = _os['fom_discretisation_error_percent']['dense_tight']
assert _mp('nOpsTwoDDisc') == f'{_disc:.3f}'
# 2026-09-22 (ops-tune-grid correction): the discretisation-error comparison holds on THIS panel's six development
# cases against THIS panel's reference, and nowhere else is it asserted.  Both directions are checked.
assert min(_oe) > _disc, 'panel cohort only: every operator arm of job ' + _os['job_id'] + ' exceeds its own reference discretisation error'
assert float(_mp('nOpsTwoDAccS')) < 1 and all(_or[a]['speedup_gpu'] < 1 for a in _or if _or[a]['family'] in ('rom', 'fast', 'pod'))
_cc = _os['fno_error_cross_check'][0]
assert _cc['absolute_difference'] == 0.0 and _mp('nOpsTwoDFnoGap') == f"{_cc['absolute_difference']:.3e}"
_app2 = (P / 'sections/appendix.tex').read_text()
_prose(lambda: (r'\label{tab:ops256}' in _app2 and r'\input{tables/TR_ops256}' in _app2), 'check_headline.py:360')
_prose(lambda: ('cohort-specific' in _app2 and 'discretisation error' in _app2), 'check_headline.py:361')            # the two qualifications stay
assert 'every arm stopped on its wall budget' not in _app2                       # made conditional on the recorded stop reason
# 2026-09-22 ops-deeponet-b2d intake: the 2D DeepONet rows re-derived from its pinned blobs
_dm = json.loads((P / 'evidence/ops-deeponet-b2d-2026-09-22/manifest.json').read_text())
for _k, _v in _dm.items():
    assert hashlib.sha256((P / f'evidence/ops-deeponet-b2d-2026-09-22/{_k}.json').read_bytes()).hexdigest() == _v['sha256'], _k
    assert _v['commit'].startswith('306c939d')
_ds = json.loads((P / 'evidence/ops-deeponet-b2d-2026-09-22/summary.json').read_text())
_da = json.loads((P / 'evidence/ops-deeponet-b2d-2026-09-22/audit.json').read_text())
assert _da['identical_split_to_fno_job'] and _da['split_hashes_cross_checked_against_archived_manifest']
_dv = {(r['arm'], r['metric']): 100 * r['value'] for r in _ds['rows'] if 'arm' in r and r['cohort'] == 'validation-32'}
_dmatch = {r['arm']: 100 * r['value'] for r in _ds['rows']
           if 'arm' in r and r['cohort'] == 'diagnosis-8' and r['metric'] == 'worst_fixed_initial_error'}
_darms = sorted({a for a, _ in _dv if a.startswith('don-')})
assert len(_darms) == 4
_dt = (P / 'tables/TR_deeponet2d.tex').read_text()
for _a in _darms + ['unet-refine', 'tsol-refine', 'fno-large']:
    for _k in ('mean_fixed_initial_error', 'median_fixed_initial_error', 'worst_fixed_initial_error'):
        assert _e2(_dv[(_a, _k)]) in _dt, (_a, _k)
    assert _e2(_dmatch[_a]) in _dt, _a
    if _a.startswith('don-'):
        _arm = _da['arms'][_a]
        assert _arm['stop_reason'] == 'early_stopping'                      # none of the 2D DeepONet arms hit the wall budget
        assert _arm['train_loss_final'] < _arm['train_loss_at_best']        # training loss still falling at the last epoch
assert _mp('nDonMeanLo') == _e2(min(_dv[(a, 'mean_fixed_initial_error')] for a in _darms))
assert _mp('nDonMeanHi') == _e2(max(_dv[(a, 'mean_fixed_initial_error')] for a in _darms))
assert _mp('nDonMatchLo') == _e2(min(_dmatch[a] for a in _darms)) and _mp('nDonMatchHi') == _e2(max(_dmatch[a] for a in _darms))
assert _mp('nDonTrainCases') == str({_da['arms'][a]['training_cases'] for a in _darms}.pop())
_tr = [100 * _da['arms'][a]['train_loss_at_best'] ** 0.5 for a in _darms]
assert (_mp('nDonTrainRmsLo'), _mp('nDonTrainRmsHi')) == (_e2(min(_tr)), _e2(max(_tr)))
_pers = _da['persistence_baseline']['validation-32']['fixed_initial']['mean'] * 100
assert _mp('nDonPersistMean') == _e2(_pers) and _mp('nDonVsPersist') == f"{_pers / min(_dv[(a, 'mean_fixed_initial_error')] for a in _darms):.1f}"
_dvb = {(r['arm'], r['metric']): 100 * r['value'] for r in _ds['rows'] if 'arm' in r and r['cohort'] == 'validation-32'}
_dmb = {r['arm']: 100 * r['value'] for r in _ds['rows'] if 'arm' in r and r['cohort'] == 'diagnosis-8' and r['metric'] == 'worst_fixed_initial_error'}
_pub = sorted({a for a, _ in _dvb if not a.startswith('don-')})
_bv = [a for a in _pub if _dvb[(a, 'worst_fixed_initial_error')] < _disc]
_bm = [a for a in _pub if _dmb[a] < _disc]
assert _mp('nOpsBarPubArms') == str(len(_pub)) and _mp('nOpsBarBelowVal') == str(len(_bv)) and _mp('nOpsBarBelowMatch') == str(len(_bm))
assert len(_bv) >= 1 and len(_bm) == len(_pub)          # off the panel's cohort the same bar is not exceeded everywhere
_prose(lambda: 'does not generalise' in ' '.join((P / 'sections/appendix.tex').read_text().split()), 'check_headline.py:400 does not generalise')   # the scoping sentence stays
# 2026-09-22 data-parity intake (ops-tune-grid @4fee6965): primitives from the pinned re-derivation, ratios recomputed
_pm = json.loads((P / 'evidence/ops-data-parity-2026-09-22/manifest.json').read_text())
for _k, _v in _pm.items():
    assert hashlib.sha256((P / f'evidence/ops-data-parity-2026-09-22/{_k}').read_bytes()).hexdigest() == _v['sha256'], _k
    assert _v['commit'].startswith('4fee6965')
_par = (P / 'evidence/ops-data-parity-2026-09-22/for-paper-arithmetic.txt').read_text()
_pn = lambda pat: float(re.search(pat, _par).group(1))
_optraj, _romtraj = int(_pn(r'(\d+) cases at pinned fidelity')), int(_pn(r'(\d+) cases at pinned fidelity: 586135'))
_ops_s, _ratio = _pn(r'operator s/trajectory: median ([0-9.]+)'), _pn(r'cost ratio ([0-9.]+)')
_perc, _pert = int(_pn(r'operator supervised states \d+\*(\d+) =')), int(_pn(r'states available \d+\*(\d+) ='))
assert _mp('nParOpTraj') == str(_optraj) and _mp('nParTrajRatio') == f'{_romtraj / _optraj:.0f}'
assert _mp('nParOpSec') == f'{_ops_s:.1f}' and _mp('nParRomSec') == f'{_ops_s / _ratio:.2f}'
assert _mp('nParOpHours') == f'{_optraj * _ops_s / 3600:.1f}' and _mp('nParFullHours') == f'{_romtraj * _ops_s / 3600:.0f}'
assert _mp('nParOpStates') == f'{_optraj * _perc:,}'.replace(',', r'\,')
assert _mp('nParStatesAvail') == f'{_romtraj * _pert:,}'.replace(',', r'\,')
_ap = (P / 'sections/appendix.tex').read_text()
_prose(lambda: (r'quoted from job \nParRomJob{}' in _ap and 'were not re-derived' in _ap), 'check_headline.py:417')        # the traceability caveat stays
# the discretisation comparison is scoped to cohort AND reference, with the same-checkpoint illustration
_prose(lambda: ('never subtractable' in _ap and r'\nOpsBarSameGap' in _ap), 'check_headline.py:419')
assert _mp('nOpsBarSamePanel') == _e2(_or['unet-refine']['worst_evolved_percent'])
assert _mp('nOpsBarSameMatch') == _e2(_dmb['unet-refine'])
assert _mp('nOpsBarSameGap') == f"{_or['unet-refine']['worst_evolved_percent'] / _dmb['unet-refine']:.1f}"
_appd = (P / 'sections/appendix.tex').read_text()
_prose(lambda: (r'\label{tab:deeponet2d}' in _appd and r'\input{tables/TR_deeponet2d}' in _appd), 'check_headline.py:424')
for _phrase in ('not an architecture ceiling', 'not on the wall budget', 'vacuous at this patience',
                'data-limited reading is live', 'no speed number from', 'weak evidence'):
    _prose(lambda: (_phrase in _appd), 'check_headline.py:427 ' + _phrase)                                        # the four qualifications stay in the text
# 2026-09-23 bank-width intake: re-derive the ordered-bank Table 1 rows, the Table 4 blocks and every \nBw / \nNs macro
# independently from the pinned blobs (not from gen_headline's intermediate records)
_bwd = {k: json.loads((P / 'evidence/bankwidth-2026-09-23' / v['file']).read_bytes()) for k, v in bwman.items()}
_hn3 = (P / 'tables/headline-numbers.tex').read_text()
def _m3(name): return re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*)\}', _hn3).group(1)
_e3 = lambda x: f'{x:.2f}' if x >= 0.1 else f'{x:.3f}'
_s3 = lambda x: f'{x:.0f}' if x >= 100 else f'{x:.1f}' if x >= 10 else f'{x:.2f}' if x >= 0.1 else f'{x:.3f}'
def _fast(cands, err): ok = {k: v for k, v in cands.items() if v[0] <= err + 1e-12}; return min(ok, key=lambda k: ok[k][1])
_pr = {r['intervals']: r for r in prov['rows'] if r['problem'] == 'Poisson' and r['dim'] == 2}
assert set(_pr) == {256, 1024, 2048, 4096} and all(r.get('bankwidth') for r in _pr.values())
for m in _bwd['pbk_summary']['meshes']:
    S = m['subjects']; r = _pr[m['intervals']]
    cg = {k: (100 * v['worst_error'], v['gpu_ms']) for k, v in S.items() if v['family'] == 'cg'}
    a, f, c = S['R512_linear'], S['R128_linear'], S[_fast(cg, 100 * S['R512_linear']['worst_error'])]
    assert r['accurate']['arm'] == 'R512_linear' and r['fast']['arm'] == 'R128_linear' and r['fom']['arm'] == c['name'] and r['job_id'] == m['job_id']
    assert abs(r['accurate']['speedup'] - c['gpu_ms'] / a['gpu_ms']) < 1e-12 and abs(r['fast']['speedup'] - c['gpu_ms'] / f['gpu_ms']) < 1e-12
    assert r['provisional_timing'] == (not m['gates']['drift']) and (r['provisional_timing'] == (m['intervals'] in (256, 1024)))
    assert (rf"${m['intervals']}^2$$^{{p}}$" in tex) == r['provisional_timing']
S4 = _bwd['pbk_summary']['meshes'][3]['subjects']; assert _bwd['pbk_summary']['meshes'][3]['intervals'] == 4096
cg4 = {k: (100 * v['worst_error'], v['gpu_ms']) for k, v in S4.items() if v['family'] == 'cg'}; c4 = S4[_fast(cg4, 100 * S4['R512_linear']['worst_error'])]
assert (_m3('nBwPoisAccErr'), _m3('nBwPoisFastErr'), _m3('nBwPoisHeadErr'), _m3('nBwPoisNarrowErr')) == tuple(_e3(100 * S4[k]['worst_error']) for k in ('R512_linear', 'R128_linear', 'R512_q0', 'R32_linear'))
assert _m3('nBwPoisAccS') == _s3(c4['gpu_ms'] / S4['R512_linear']['gpu_ms']) and _m3('nBwPoisFastS') == _s3(c4['gpu_ms'] / S4['R128_linear']['gpu_ms'])
assert _m3('nBwPoisErrSpan') == _s3(S4['R32_linear']['worst_error'] / S4['R512_linear']['worst_error'])
assert _m3('nBwPoisCostSpan') == _s3(S4['R512_linear']['gpu_ms'] / S4['R32_linear']['gpu_ms']) and _m3('nBwPoisNarrowS') == _s3(c4['gpu_ms'] / S4['R32_linear']['gpu_ms'])
def _ns(d):
    t, cfg = d['timing'], d['config']; tag = f"dt{cfg['ladder_dt']}_it{cfg['ladder_iters']}"
    arms = {'head': d['frontier'][f'k8_q0_{tag}'], **{R: d['span'][f'span{R}_{tag}'] for R in (64, 48, 32, 16, 8)}}
    ms_ = {'head': t['fast'][f'query_k8_q0_{tag}']['median_ms'], **{R: t['fast'][f'query_span{R}_{tag}']['median_ms'] for R in (64, 48, 32, 16, 8)}}
    cn = {f"CNAB2_s{e['steps']}": (100 * e['stats']['evolved_worst'], t['fast'][f"CNAB2_s{e['steps']}"]['median_ms'])
          for e in d['cnab2'].values() if not e['unstable'] and e['stats'] and not e['reference_itself']}
    return {k: (100 * v['stats']['evolved_worst'], ms_[k], v['stats']['cases']) for k, v in arms.items()}, cn
_nr = {r['intervals']: r for r in prov['rows'] if r['problem'] == 'Navier--Stokes'}
assert set(_nr) == {32, 64, 96}
for k in ('ns_a2_h32', 'ns_a2_h64', 'ns_a3_h96'):
    A, cn = _ns(_bwd[k]); r = _nr[_bwd[k]['config']['n']]
    fast = min((R for R in (64, 48, 32, 16, 8) if A[R][0] < 5), key=lambda R: A[R][1]); c = _fast(cn, A['head'][0])
    assert fast == 16 and r['fast']['arm'].startswith('span16_') and r['accurate']['arm'].startswith('k8_q0_') and r['fom']['arm'] == c
    assert abs(r['accurate']['speedup'] - cn[c][1] / A['head'][1]) < 1e-12 and abs(r['fast']['speedup'] - cn[c][1] / A[16][1]) < 1e-12   # one FOM time, same job
    assert r['job_id'] == _bwd[k]['job_id'] and 'CNAB2 (spectral)' in r['fom']['name']
Ad, cnd = _ns(_bwd['ns_a3_h96']); Ah, cnh = _ns(_bwd['ns_b2_heldout96'])
assert _bwd['ns_b2_heldout96']['heldout_opened'] and not _bwd['ns_a3_h96']['heldout_opened']
cd, ch = cnd[_fast(cnd, Ad['head'][0])], cnh[_fast(cnh, Ah['head'][0])]
assert (_m3('nNsHeadK'), _m3('nNsBankR'), _m3('nNsHeadCases'), _m3('nNsHoldCases')) == ('8', str(_bwd['ns_a3_h96']['config']['rank']), str(Ad['head'][2]), str(Ah['head'][2]))
assert (_m3('nNsHeadErr'), _m3('nNsHeadS'), _m3('nNsFastErr'), _m3('nNsFastS')) == (_e3(Ad['head'][0]), _s3(cd[1] / Ad['head'][1]), _e3(Ad[16][0]), _s3(cd[1] / Ad[16][1]))
assert (_m3('nNsHoldHeadErr'), _m3('nNsHoldHeadS')) == (_e3(Ah['head'][0]), _s3(ch[1] / Ah['head'][1]))
assert (_m3('nNsSpanWideErr'), _m3('nNsSpanNarrowErr'), _m3('nNsSpanCostSpan')) == (_e3(Ad[64][0]), _e3(Ad[8][0]), _s3(Ad[64][1] / Ad[8][1]))
assert _m3('nNsHeadVsSpan') == f"{Ad[8][0] / Ad['head'][0]:.0f}" and _m3('nNsSmallS') == _s3(_nr[32]['accurate']['speedup'])
_tbw = {s_['series']: s_ for s_ in (prov['tunability'] + prov['tunability_bankwidth']) if not s_['legacy']}
assert {x['fom']['arm'] for x in _tbw.values()} == {c4['name'], _fast(cnd, Ad['head'][0]), _fast(cnh, Ah['head'][0])}
# Table 1 markers: every printed marker is defined in the generated definitions file
_mk = (P / 'tables/TH_headline_markers.tex').read_text()
for _x in set(re.findall(r'\$\^\{[^}]+\}\$', tex)): assert _x in _mk, _x
print(json.dumps(dict(passed=True, prose_expectations_not_met=PROSE_PENDING, rows=len(prov['rows']), bold_speedups=bold, failure_rows=len(fails),
                      pending_lane_slots=tex.count('pending:'), abstract_macros=sorted(macros)), indent=2))
