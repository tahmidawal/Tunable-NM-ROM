"""Generate the headline error/speedup table, its timing companion, the
"where the method currently fails" table and the compact 3D configuration table.

Rule: every number comes from a hash-pinned JSON snapshot under
``evidence/headline-2026-09-20/``.  Normal builds never touch a worktree.

    gen_headline.py            render from the frozen snapshots
    gen_headline.py --refresh  re-copy the sources named in SOURCES / the intake
                               file, re-hash them and rewrite the manifest

One row = one frozen model evaluated at two deployment settings:
  fast      the uncorrected setting q = 0 (for Burgers: its fastest stored residual
            evaluation that meets the stopping rule)
  accurate  the largest stored correction rank at the model's standard time step
            (for Burgers: its lowest-error residual evaluation that meets the rule)
and ONE named full-order solver (FOM) from the same allocation, at least as
accurate as the accurate setting.  Both speedups divide that one FOM time,
except the 4096^2 Burgers fast column, whose denominator is the fastest
tested setting at least as accurate as the fast setting (bh5 grid).

New rows from the running lanes enter through ``headline-intake.json``
(see INTAKE_SCHEMA below); until a path is listed there the slot prints dashes.
"""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
E = HERE / 'evidence/headline-2026-09-20'
WT = 'worktrees/'

# key -> (repo-relative worktree or '.', path inside it, commit to read or None = copy an existing paper snapshot, status)
SOURCES = {
    'paired_cg': ('.', 'paper/evidence/paired-cg-2026-09-20/results.json', None, 'development; paired CG snapshot'),
    'burgers_iter': ('.', 'paper/evidence/burgers-iterative-2026-09-11/results.json', None, 'development; earlier model'),
    'burgers3d': ('.', 'paper/evidence/main-experiments-2026-09-20/burgers.json', None, 'accepted final'),
    'lshape': (WT + '2026-09-17-lshape', 'experiments/lshape/reports/summary.json', 'd80fed7a', 'development'),
    'burgers_panel': (WT + '2026-09-17-b-panel', 'experiments/b-panel/reports/summary.json', '25434a27', 'development'),
    'eqcert_summary': (WT + '2026-09-21-burgers-eqcert', 'experiments/burgers-eqcert/reports/summary.json', '176b2a9a5e4642a0a1338d824e7e3ab94e298ca8', 'audited; burgers-eqcert lane (closed)'),
    'eqcert_bc256': (WT + '2026-09-21-burgers-eqcert', 'experiments/burgers-eqcert/checks/bc256-summary.json', '176b2a9a5e4642a0a1338d824e7e3ab94e298ca8', 'audited; burgers-eqcert lane (closed)'),
    'eqcert_bc256b': (WT + '2026-09-21-burgers-eqcert', 'experiments/burgers-eqcert/checks/bc256b-summary.json', '176b2a9a5e4642a0a1338d824e7e3ab94e298ca8', 'audited; burgers-eqcert lane (closed)'),
    'eqcert_bc512': (WT + '2026-09-21-burgers-eqcert', 'experiments/burgers-eqcert/checks/bc512-summary.json', '176b2a9a5e4642a0a1338d824e7e3ab94e298ca8', 'audited; burgers-eqcert lane (closed)'),
    'eqcert_bc1024': (WT + '2026-09-21-burgers-eqcert', 'experiments/burgers-eqcert/checks/bc1024-summary.json', '176b2a9a5e4642a0a1338d824e7e3ab94e298ca8', 'audited; burgers-eqcert lane (closed)'),
    'eqcert_bc2048b': (WT + '2026-09-21-burgers-eqcert', 'experiments/burgers-eqcert/checks/bc2048b-summary.json', '176b2a9a5e4642a0a1338d824e7e3ab94e298ca8', 'audited; burgers-eqcert lane (closed)'),
    'heat3db_panel_a': (WT + '2026-09-21-heat3d-bank', 'experiments/heat3d-bank/runs/final01/final01/summary.json', '55165375fc9e1193aed1ff2a729439a893e80396', 'audited final; heat3d-bank lane (closed)'),
    'heat3db_panel_b': (WT + '2026-09-21-heat3d-bank', 'experiments/heat3d-bank/runs/final01/final01b/summary.json', '55165375fc9e1193aed1ff2a729439a893e80396', 'audited final; heat3d-bank lane (closed)'),
    'heat3db_final_table': (WT + '2026-09-21-heat3d-bank', 'experiments/heat3d-bank/runs/final01/final_table.json', '55165375fc9e1193aed1ff2a729439a893e80396', 'audited final; heat3d-bank lane (closed)'),
    'heat3db_selection': (WT + '2026-09-21-heat3d-bank', 'experiments/heat3d-bank/selection.json', '55165375fc9e1193aed1ff2a729439a893e80396', 'audited final; heat3d-bank lane (closed)'),
    'heat3db_selection_add1': (WT + '2026-09-21-heat3d-bank', 'experiments/heat3d-bank/selection_addendum1.json', '55165375fc9e1193aed1ff2a729439a893e80396', 'audited final; heat3d-bank lane (closed)'),
    'heat3db_selection_add2': (WT + '2026-09-21-heat3d-bank', 'experiments/heat3d-bank/selection_addendum2.json', '55165375fc9e1193aed1ff2a729439a893e80396', 'audited final; heat3d-bank lane (closed)'),
    'ns3d_grok_diag07': (WT + '2026-09-21-ns3d-grok', 'experiments/ns3d-grok/runs/diag07/output/summary.json', '8852b7cd6365de7a8fdfd747bedee587f1cf9f29', 'audited diagnostic; ns3d-grok lane (a different model: shift-tracking linear POD ROM)'),
    'bh5_summary': (WT + '2026-09-21-burgers-heldout', 'experiments/burgers-heldout/checks/bh5-summary.json', '3771cacfe84730eda2466f02b19f8c2290649b26', 'audited; burgers-heldout lane (closed); incumbent 4096^2 re-timing'),
    'bh5_eqcert': (WT + '2026-09-21-burgers-heldout', 'experiments/burgers-heldout/checks/bh5-eqcert-summary.json', '3771cacfe84730eda2466f02b19f8c2290649b26', 'audited; burgers-heldout lane (closed); quadrature re-draws at 4096^2'),
    'bh_summary': (WT + '2026-09-21-burgers-heldout', 'experiments/burgers-heldout/reports/summary.json', '3771cacfe84730eda2466f02b19f8c2290649b26', 'audited; burgers-heldout lane (closed); wider-bank result'),
    'wave': (WT + '2026-09-17-w-ladder', 'experiments/w-ladder/reports/summary.json', '9b84d0556888ebc052b52bd165a61dcd56b2b53d', 'development'),
    'poisson3d': (WT + '2026-09-20-paper-p3d', 'experiments/paper-p3d/runs/final08/paper-comparisons.json', '26c73030b89adfa321ede751db2798abf9bcd1b2', 'accepted final'),
    'heat3d': (WT + '2026-09-20-paper-h3d', 'experiments/paper-h3d/runs/final08/paper-tables.json', 'f15c7232ab21df20cd0fa279134c06114b5ca721', 'accepted final'),
    'ns3d': (WT + '2026-09-20-paper-ns3d', 'experiments/ns3d/runs/final07/paper_summary.json', '3f4e12d6134da21cf82e5259c2990c8ab625b4c6', 'accepted final'),
}
# companion markdown, pinned by git blob and sha256, not loaded as JSON
COMPANIONS = {
    'heat3db_panel_a': [
        ('experiments/heat3d-bank/DESIGN.md', 'DESIGN.md'),
        ('experiments/heat3d-bank/HANDOFF.md', 'HANDOFF.md'),
        ('experiments/heat3d-bank/runs/final01/FINAL-TABLE.md', 'runs/final01/FINAL-TABLE.md'),
    ],
    'bh_summary': [
        ('experiments/burgers-heldout/DESIGN.md', 'DESIGN.md'),
        ('experiments/burgers-heldout/HANDOFF.md', 'HANDOFF.md'),
        ('experiments/burgers-heldout/reports/2026-09-21-burgers-heldout.md', 'reports/2026-09-21-burgers-heldout.md'),
        ('experiments/burgers-heldout/reports/tables.generated.md', 'reports/tables.generated.md'),
    ],
}
INTAKE = HERE / 'headline-intake.json'
INTAKE_SCHEMA = 'nmrom-headline-rows-v1'
# Slots reserved for the running lanes; printed as dashes until intake rows cover them.
PENDING = [
    dict(problem='Poisson', dim=2, meshes=[2048, 4096], lane='hires-poisson'),
    dict(problem='Heat', dim=2, meshes=[2048, 4096], lane='hires-heat'),
    dict(problem='Burgers', dim=2, meshes=[2048, 4096], lane='hires-burgers'),
    dict(problem='Poisson', dim=3, meshes=[128], lane='hires-poisson'),
    dict(problem='Heat', dim=3, meshes=[128], lane='hires-heat'),
    dict(problem='Burgers', dim=3, meshes=[128], lane='hires-burgers'),
]
INCOMING = [  # paused-lane slots; empty once every listed lane has been ingested
    # burgers-eqcert filled 2026-09-21 (rows 'Burgers, confirmed rule')
    # heat3d-bank filled 2026-09-22 (rows 'Heat (new bank)')
    # burgers-heldout filled 2026-09-22: the wider bank did not improve held-out accuracy, so it is a
    # limitations sentence, not a Table 1 row.  The 4096^2 quadrature re-timing is a relabel plus appendix rows.
]
ORDER = ['Poisson', 'Poisson (dev. sources)', 'Poisson, L-shape', 'Heat', 'Heat (wide bank)', 'Heat (wide bank, batched fit)', 'Burgers', 'Burgers (held-out cases)', 'Burgers, exact first step', 'Burgers, confirmed rule', 'Burgers (earlier model)', 'Heat (new bank)', 'Heat (new bank, batched fit)']


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def refresh():
    E.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, (tree, path, commit, status) in SOURCES.items():
        if commit is None:
            raw = (REPO / path).read_bytes(); full = None
        else:
            full = subprocess.check_output(['git', '-C', str(REPO / tree), 'rev-parse', commit]).decode().strip()
            raw = subprocess.check_output(['git', '-C', str(REPO / tree), 'show', f'{full}:{path}'])
        (E / f'{key}.json').write_bytes(raw)
        manifest[key] = dict(tree=tree, path=path, commit=full, status=status, sha256=digest(raw),
                             read='committed blob' if commit else 'existing hash-pinned paper snapshot')
        if full:
            manifest[key]['git_blob'] = subprocess.check_output(['git', '-C', str(REPO / tree), 'rev-parse', f'{full}:{path}']).decode().strip()
        if key in COMPANIONS:
            blobs, shas = {}, {}
            for rel, label in COMPANIONS[key]:
                raw_c = subprocess.check_output(['git', '-C', str(REPO / tree), 'show', f'{full}:{rel}'])
                blobs[label] = subprocess.check_output(['git', '-C', str(REPO / tree), 'rev-parse', f'{full}:{rel}']).decode().strip()
                shas[label] = digest(raw_c)
            manifest[key]['companion_blobs'] = blobs
            manifest[key]['companion_sha256'] = shas
    if INTAKE.exists():
        for i, item in enumerate(json.loads(INTAKE.read_text())['sources']):
            if item.get('blob_path'):       # preferred: the lane's committed blob, never its working tree
                full = subprocess.check_output(['git', '-C', str(REPO / item['tree']), 'rev-parse', item['commit']]).decode().strip()
                raw = subprocess.check_output(['git', '-C', str(REPO / item['tree']), 'show', f"{full}:{item['blob_path']}"]); path = item['blob_path']
            else:
                raw = Path(item['path']).read_bytes(); full = item.get('commit'); path = item['path']
            if item.get('sha256'): assert digest(raw) == item['sha256'], path
            key = f"intake_{i:02d}_{item['lane']}"
            (E / f'{key}.json').write_bytes(raw)
            manifest[key] = dict(tree=item.get('tree'), path=path, commit=full, status=item['status'], adapter=item.get('adapter'),
                                 sha256=digest(raw), read='coordinator-supplied audited summary', lane=item['lane'])
            if item.get('role'): manifest[key]['role'] = item['role']
    (E / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if '--refresh' in sys.argv:
    refresh()
MAN = json.loads((E / 'manifest.json').read_text())
D = {}
for k, v in MAN.items():
    b = (E / f'{k}.json').read_bytes()
    assert digest(b) == v['sha256'], k
    D[k] = json.loads(b)

ROWS = []  # headline rows
APPX = []  # measured rows shown only in the appendix timing table (re-measures, duplicates of a final-cohort mesh)


def rule_pick(cands, acc_err):
    """User decision 2026-09-21 (one FOM rule for all rows): the fastest tested setting of the named solver, in the same
    allocation, whose error is at most the accurate NM-ROM setting's.  cands: name -> dict(err, ms)."""
    ok = {k: v for k, v in cands.items() if v['err'] <= acc_err + 1e-12}
    return min(ok, key=lambda k: ok[k]['ms'])


def tol_tex(x):
    m_, ex = f'{float(x):.0e}'.split('e'); ex = int(ex)
    return f'10^{{{ex}}}' if m_ == '1' else f'{m_}{{\\times}}10^{{{ex}}}'


def setting(label, err, ms, **kw):
    return dict(label=label, error_pct=err, ms=ms, **kw)


def row(problem, dim, n, fast, acc, fom, source, job, cohort, status, error, timing, note='', alt=None, appendix_only=None):
    for s in (fast, acc):
        if s: assert fom['error_pct'] <= s['error_pct'] + 1e-12, (problem, n)
    (APPX if appendix_only else ROWS).append(dict(problem=problem, dim=dim, intervals=n, fast=fast, accurate=acc, fom=fom, source=source,
                     job_id=str(job), cohort=cohort, status=status, error_convention=error, timing_scope=timing, note=note,
                     alt=alt, appendix_only=appendix_only))


# ---- Poisson 2D and Heat 2D: paired CG snapshot -------------------------------------------
cg = defaultdict(dict)
for r in D['paired_cg']['rows']:
    assert abs(r['cg_ms'] / r['method_ms'] - r['speedup']) < 1e-10
    cg[(r['problem'], r['intervals'])][r['method']] = r
for n in (256, 1024):
    g = cg[('Poisson2D', n)]; f, a = g['q0_m4@new_K32'], g['q256_m4@new_K32']
    assert f['cg'] == a['cg'] and f['cg_ms'] == a['cg_ms'] and f['job_id'] == a['job_id']
    row('Poisson', 2, n, setting('$q=0$', f['error_pct'], f['method_ms']), setting('$q=256$', a['error_pct'], a['method_ms']),
        dict(name='CG, rtol $10^{-2}$', error_pct=a['cg_error_pct'], ms=a['cg_ms'], arm=a['cg'], candidates={a['cg']: dict(err=a['cg_error_pct'], ms=a['cg_ms'])},
             selection='record: ' + D['paired_cg']['selection']), 'paired_cg', a['job_id'],
        'development', 'development', 'same-grid', 'GPU query')
for n in (64, 256, 1024):
    h = cg[('Heat2D', n)]['nmrom_cholesky']
    row('Heat', 2, n, setting('single', h['error_pct'], h['method_ms']), None,
        dict(name='CN--CG, rtol $10^{-2}$', error_pct=h['cg_error_pct'], ms=h['cg_ms'], arm=h['cg'], candidates={h['cg']: dict(err=h['cg_error_pct'], ms=h['cg_ms'])},
             selection='record: ' + D['paired_cg']['selection']), 'paired_cg', h['job_id'],
        'development', 'development; earlier checkpoint, one setting', 'refined reference', 'GPU query')

# ---- L-shaped Poisson ------------------------------------------------------------------------
V = defaultdict(dict); ljob = {}
for r in D['lshape']:
    if r.get('test_modes') == 257 and r['metric'] in ('worst_same_grid', 'median_total_ms'):
        V[(r['mesh'], r['subject'])][r['metric']] = r['value']; ljob[r['mesh']] = r['job_id']
for n in (256, 512):
    f, a = V[(n, 'neural_q0@head_sdf_R512_K16')], V[(n, 'neural_q64@head_sdf_R512_K16')]
    cands = {s: dict(err=100 * V[(n, s)]['worst_same_grid'], ms=V[(n, s)]['median_total_ms']) for (m, s) in V if m == n and s.startswith('fom_cg_gpu_r')}
    ref = rule_pick(cands, 100 * a['worst_same_grid']); c = V[(n, ref)]
    row('Poisson, L-shape', 2, n, setting('$q=0$', 100 * f['worst_same_grid'], f['median_total_ms']),
        setting('$q=64$', 100 * a['worst_same_grid'], a['median_total_ms']),
        dict(name='CG, rtol $' + tol_tex(ref.split('_r')[1]) + '$', error_pct=100 * c['worst_same_grid'], ms=c['median_total_ms'], arm=ref, candidates=cands), 'lshape', ljob[n],
        'development', 'development', 'same-grid', 'complete query')

# ---- Burgers 2D: one frozen model, three meshes -------------------------------------------------
P = defaultdict(dict); pjob = defaultdict(set)
for r in D['burgers_panel']['rows']:
    P[(r['mesh'], r['subject'])][r['metric']] = r['value']; pjob[(r['mesh'], r['subject'])].add(r['job_id'])
for n in (256, 512, 1024):
    roms = {s: v for (m, s), v in P.items() if m == n and s.startswith('q') and v.get('admissible') and v.get('converged_design5')}
    foms = {s: v for (m, s), v in P.items() if m == n and s.startswith(('nt', 'fft')) and v.get('admissible')}
    fs = min(roms, key=lambda s: roms[s]['median_gpu_ms']); as_ = min(roms, key=lambda s: roms[s]['worst_evolved_percent'])
    assert fs.startswith('q0_') and as_.startswith('q256_'), (fs, as_)   # fast is q = 0, accurate the largest stored rank
    ok = {s: v for s, v in foms.items() if v['worst_evolved_percent'] <= roms[as_]['worst_evolved_percent']}
    cs = min(ok, key=lambda s: ok[s]['median_gpu_ms'])
    assert pjob[(n, fs)] == pjob[(n, as_)] == pjob[(n, cs)] and len(pjob[(n, cs)]) == 1
    def lab(s):
        q = s.split('_')[0][1:]; M = s.split('_')[1][1:]
        if 'dense' in s: return f'$q={q}$, $M={M}$, dense'
        return f"$q={q}$, $M={M}$, EQ $m={P[(n, s)]['rule_m']}$"
    assert cs.startswith('nt1e-') and cs.endswith('_dt005')
    row('Burgers', 2, n, setting(lab(fs), roms[fs]['worst_evolved_percent'], roms[fs]['median_gpu_ms'], arm=fs),
        setting(lab(as_), roms[as_]['worst_evolved_percent'], roms[as_]['median_gpu_ms'], arm=as_,
                eq='dense' if 'dense' in as_ else {'certified in one draw': 'single-draw', 'confirmed (3/3)': 'confirmed'}[roms[as_]['rule_status']],
                rule=None if 'dense' in as_ else dict(set=roms[as_]['rule_set'], m=roms[as_]['rule_m'], file_sha256=roms[as_]['rule_file_sha256'],
                                                      source_job=roms[as_]['rule_source_job'], status=roms[as_]['rule_status'])),
        dict(name='Newton--BiCGStab, tol $10^{-%s}$' % cs[5], error_pct=ok[cs]['worst_evolved_percent'], ms=ok[cs]['median_gpu_ms'], arm=cs,
             candidates={k: dict(err=v['worst_evolved_percent'], ms=v['median_gpu_ms']) for k, v in foms.items()}),
        'burgers_panel', next(iter(pjob[(n, cs)])), 'development', 'development', 'same-grid, evolved', 'GPU query')

# ---- Burgers 2D, burgers-eqcert lane (audited, closed 2026-09-21): quadrature rules re-drawn on independent held-out populations
# Pre-registered selection per mesh (DESIGN 5 + A1.3): cheapest q=256 arm confirmed on draws 1-5 that also passes a sixth
# (confirmation) draw.  512^2 and 1024^2 have such a rule; 256^2 does not (its pick failed confirmation).  Same model and dev6 cohort
# as the b-panel rows; one FOM per row by the paper rule, from the same allocation.
EQC = {}
_eqs = D['eqcert_summary']
for _k in ('bc256', 'bc256b', 'bc512', 'bc1024', 'bc2048b'):
    assert _eqs['sources'][_k]['accepted'] and not _eqs['sources'][_k]['failed_gates'] and MAN['eqcert_' + _k]['sha256'] == _eqs['sources'][_k]['sha256'], _k
    assert D['eqcert_' + _k]['failed_gates'] == [], _k
_bars = {r_['rho_bar'] for _k in ('bc256', 'bc256b', 'bc512', 'bc1024', 'bc2048b') for r_ in D['eqcert_' + _k]['rules']}
assert len(_bars) == 1; EQC['bar'] = _bars.pop()
for n, k in ((512, 'eqcert_bc512'), (1024, 'eqcert_bc1024')):
    d = D[k]; v = d['verdict']; assert d['intervals'] == n and v['certified_rule_exists'] and v['selected_passes_confirmation'] and v['accepted']
    a_, f_ = d['table'][v['accurate']['arm']], d['table'][v['fast']['arm']]
    assert a_['stalled_exits'] == f_['stalled_exits'] == 0 and a_['status'] == 'confirmed' and a_['confirmation_pass'] and a_['cases'] == f_['cases'] == d['cohort_cases']
    cands = {nm: dict(err=t_['worst_evolved_percent'], ms=t_['median_gpu_ms']) for nm, t_ in d['table'].items()
             if t_['family'] == 'fom' and t_['nonlinear_converged'] and t_['stalled_steps'] == 0 and t_['cases'] == d['cohort_cases']}
    pick = rule_pick(cands, a_['worst_evolved_percent']); assert pick == v['accurate']['fom'] == a_['fom_by_paper_rule'], (n, pick)
    assert abs(d['table'][pick]['median_gpu_ms'] / a_['median_gpu_ms'] - a_['speedup_gpu']) < 1e-12
    nt, lt = pick.split('_nt')[1].split('_')[0], pick.split('_l')[2].split('_')[0] if pick.count('_l') > 1 else None
    lab_ = lambda t_: f"$q={t_['q']}$, $M={t_['M']}$, EQ " + ('lattice ' if t_['rule'] == 'lat64' else '') + f"$m={t_['m']}$" + ({1: ', first step exact'}[t_['exact_steps']] if t_['exact_steps'] else '')
    row('Burgers, confirmed rule', 2, n, setting(lab_(f_), f_['worst_evolved_percent'], f_['median_gpu_ms'], arm=f_['name']),
        setting(lab_(a_), a_['worst_evolved_percent'], a_['median_gpu_ms'], arm=a_['name'], eq='confirmed', exact_steps=a_['exact_steps'],
                confirmation_rho=a_['confirmation_rho_max'], heldout_rho=a_['heldout_rho_max']),
        dict(name='Newton--BiCGStab, tol $' + tol_tex(nt) + '$', error_pct=cands[pick]['err'], ms=cands[pick]['ms'], arm=pick, candidates=cands),
        k, d['job_id'], 'development', 'development; re-drawn quadrature rule', 'same-grid, evolved', 'GPU query')
    EQC[n] = dict(arm=a_['name'], exact_steps=a_['exact_steps'], conf_rho=a_['confirmation_rho_max'])
# the paper's 256^2 / 512^2 stored q=256 rule ('scaled' in the lane: b-eqtop's m=2560 rule) on independent re-draws
_s256 = D['eqcert_bc256']['arm_status']['q256_M1088_scaled_g0p001_fast_chol_clip_lamcarry_pred2']
_s512 = D['eqcert_bc512']['arm_status']['q256_M1088_scaled_g0p001_fast_chol_clip_lamcarry_pred2']
assert D['eqcert_bc256']['verdict']['selected_on_certification_draws'] == 'q256_M1088_scaled_g0p001_fast_chol_clip_lamcarry_pred2'
assert not D['eqcert_bc256']['verdict']['certified_rule_exists'] and not _s256['confirmation_pass'] and not _s256['audited_confirmation_pass']
assert not _s512['confirmation_pass'] and _s512['draws_passed'] < _s512['draws']
EQC['s512'] = (_s512['draws_passed'], _s512['draws'])
# Rule identity (2026-09-21 coordinator follow-up): which re-draw result belongs to which Table 1 row.
#  256^2 panel arm `eqtop`: b-eqtop's stored m=2560 file at its native mesh.  The lane's `scaled` rule at 256^2 is that file
#    with weights x (256/256)^2 and no refit -> the SAME rule, so its failed re-draw applies to the row (marker s).
#  512^2 panel arm `eqtopxfer`: the same node set mapped to 512^2 with weights REFIT by NNLS at 512^2 (b-panel DESIGN 3.2/A3;
#    support 2438 after the refit).  The lane's `scaled` rule at 512^2 keeps the original weights x 4 on all 2560 nodes -> a
#    DIFFERENT rule; the row keeps its own single-draw status (marker x) and the related result is stated as such.
def _lane_rule(k):
    r_ = [x for x in D[k]['rules'] if x['q'] == 256 and x['M'] == 1088 and x['rule'] == 'scaled']; assert len(r_) == 1; return r_[0]
_b256 = [r_ for r_ in ROWS if r_['problem'] == 'Burgers' and r_['intervals'] == 256][0]['accurate']['rule']
_b512 = [r_ for r_ in ROWS if r_['problem'] == 'Burgers' and r_['intervals'] == 512][0]['accurate']['rule']
_l256, _l512 = _lane_rule('eqcert_bc256'), _lane_rule('eqcert_bc512')
assert _b256['set'] == 'eqtop' and _l256['refit'] is None and _l256['source'][0]['source_mesh'] == 256
assert _l256['source'][0]['sha256'] == _b256['file_sha256'] and _l256['m'] == _b256['m']        # same file, same nodes, same weights
assert _b512['set'] == 'eqtopxfer' and _l512['refit'] is None and _l512['source'][0]['sha256'] == _b512['file_sha256']
assert _l512['m'] != _b512['m']                                                                   # refit support differs: not the same rule
EQC['identity'] = {256: dict(row=dict(_b256), lane=dict(m=_l256['m'], sha256=_l256['source'][0]['sha256'], refit=_l256['refit']), same=True),
                   512: dict(row=dict(_b512), lane=dict(m=_l512['m'], sha256=_l512['source'][0]['sha256'], refit=_l512['refit']), same=False)}
for r_ in ROWS:
    if r_['problem'] == 'Burgers' and r_['intervals'] in (256, 512):
        r_['accurate']['eq'] = 'not-confirmed' if EQC['identity'][r_['intervals']]['same'] else 'single-draw-refit'
        _id = EQC['identity'][r_['intervals']]; r_['accurate']['rule']['redraw'] = dict(same_rule_as_lane_scaled=_id['same'], lane_scaled=_id['lane'])
# 256^2 post-hoc follow-up (A2.1, exact first step): passes every draw (5 + confirmation) in both bc256 and bc256b
_pk = _eqs['combined_256']['pick']; _st = [D[k]['arm_status'][_pk] for k in ('eqcert_bc256', 'eqcert_bc256b')]
assert all(x['draws_passed'] == x['draws'] and x['confirmation_pass'] for x in _st)
_row256 = D['eqcert_bc256b']['table'][_pk]; assert _row256['exact_steps'] == 1
EQC['follow'] = dict(draws=sum(x['draws'] + 1 for x in _st), err=_row256['worst_evolved_percent'], sp=_row256['speedup_gpu'])
# the 2048^2 certificate for the lattice rule with no exact step (the rule of the 2048^2/4096^2 rows)
_c = D['eqcert_bc2048b']['table']['q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2']
assert _c['confirmation_pass'] and _c['exact_steps'] == 0 and D['eqcert_bc2048b']['arm_status']['q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2']['audited_confirmation_pass']
assert abs(_c['confirmation_rho_max'] - D['eqcert_bc2048b']['arm_status']['q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2']['confirmation_heldout_rho_max_k>=0']) < 1e-15 and _eqs['certificate_2048']['q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2']['status'] == 'confirmed'
EQC['lat2048'] = _c['confirmation_rho_max']

# ---- Burgers 2D, earlier model, tight and relaxed Newton (one NM-ROM setting) ---------------------
from statistics import median
b = D['burgers_iter']; S = {}
for name in ('nmrom', 'fft_tight', 'fft_loose'):
    calls = [r for r in b['invocations'] if r['name'] == name]; assert len(calls) == 12
    S[name] = (100 * max(r['error']['fixed_initial_max'] for r in calls), 1000 * median(r['gpu_seconds'] for r in calls))
_bc = {k: dict(err=S[k][0], ms=S[k][1]) for k in ('fft_loose', 'fft_tight')}
_bpick = rule_pick(_bc, S['nmrom'][0])
for name, title in (('fft_loose', 'Newton--BiCGStab, relaxed'), ('fft_tight', 'Newton--BiCGStab, tight')):
    row('Burgers (earlier model)', 2, 1024, setting('single', *S['nmrom']), None, dict(name=title, error_pct=S[name][0], ms=S[name][1], arm=name, candidates=_bc),
        'burgers_iter', b['job_id'], 'development', 'development; earlier model, stalled exits permitted', 'refined reference', 'GPU query',
        appendix_only=None if name == _bpick else 'not the rule\'s full-order setting (a faster setting of the same solver is at least as accurate)')

# ---- Poisson 3D accepted final -------------------------------------------------------------------
p3 = D['poisson3d']; assert p3['final_cohort_opened']
pr = {(r['intervals'], r['method']): r for r in p3['rows']}; pc = {(r['intervals'], r['method']): r for r in p3['cg_controls']}
for n in (32, 64):
    f, a = pr[(n, 'nmrom_K16_q0_dense')], pr[(n, 'nmrom_K16_q96_dense')]
    cands = {k[1]: dict(err=100 * v['same_grid_error_worst'], ms=v['device_ms_median']) for k, v in pc.items() if k[0] == n and v['cg_failed_invocations'] == 0}
    c = pc[(n, rule_pick(cands, 100 * a['same_grid_error_worst']))]; assert f['nonstationary_cases'] == a['nonstationary_cases'] == 0
    assert c['method'] == a['matched_cg_method']
    row('Poisson', 3, n, setting('$q=0$', 100 * f['same_grid_error_worst'], f['device_ms_median']),
        setting('$q=96$', 100 * a['same_grid_error_worst'], a['device_ms_median']),
        dict(name='CG, rtol $' + tol_tex(c['method'].split('rtol')[1]) + '$', error_pct=100 * c['same_grid_error_worst'], ms=c['device_ms_median'], arm=c['method'], candidates=cands), 'poisson3d', p3['job_id'],
        'final', 'accepted final', 'same-grid', 'GPU query')

# ---- Heat 3D accepted final ----------------------------------------------------------------------
H3F = []
h3 = D['heat3d']; assert h3['status'] == 'accepted' and h3['archive_commit']   # accepted 2026-09-20 after actual-Git retention
hr = {(r['intervals'], r['method']): r for r in h3['rows'] if r['cohort'] == 'primary' and r['evaluation_cohort'] == 'final'}
for n in (32, 64):
    f, a = hr[(n, 'nmrom_K32_q0_dense')], hr[(n, 'nmrom_K32_q96_dense')]; c = hr[(n, a['matched_cg'])]
    assert c['method'] == 'fom_cn_cg_dt0.05_rtol1e-04' and f['nonstationary_cases'] == a['nonstationary_cases'] == 0
    # 2026-09-21 user decision: Heat 3D misses the all-times target, so it is reported in the failures table, not Table 1.
    # paper-tables.json records evolved-time errors only (worst_rel_l2 = maximum over evolved times), so these rows say so.
    H3F.append(dict(n=n, fast=f, acc=a, fom=c))

# ---- Heat 3D, new bank (heat3d-bank lane, closed 2026-09-22; user decision 2026-09-22: Table 1 from PANEL A) -------------------
# Panel A = the arms chosen by the rule pre-registered in the lane's DESIGN.md (R=320, K=32, q in {0, 288}); sealed cohort
# 921099 (64 draws) opened once in job 4153878.  Plain Crank--Nicolson row first, the batched exact-propagator fit beneath
# (linear autonomous heat only), both from the same frozen model.  One FOM per row by the paper rule on all-times error.
H3N = {}
_ha = D['heat3db_panel_a']; _hm = _ha['metadata']
assert _ha['audit_passed'] and _hm['backend'] == 'gpu' and _hm['x64'] and _hm['precision'] == 'highest' and not _hm['local_smoke']
assert MAN['heat3db_panel_a']['sha256'] == {r_['summary_sha256'] for r_ in D['heat3db_final_table'] if r_['panel'] == 'final01'}.pop()
_sel = D['heat3db_selection']['chosen']; assert (_sel['R'], _sel['K'], _sel['accurate_q'], _sel['fast_q']) == (320, 32, 288, 0)
_HN = {m['intervals']: {r_['method']: r_ for r_ in m['rows']} for m in _ha['meshes'] if m['cohort'] == 'sealed_921099_never_opened'}
assert set(_HN) == {32, 64, 128}
for n in (32, 64, 128):
    R_ = _HN[n]; assert all(r_['cases'] == r_['timed_cases'] == 64 for r_ in R_.values())
    cands = {k: dict(err=100 * v['error_all_times_worst'], ms=v['device_ms_median']) for k, v in R_.items() if k.startswith('fom_cncg_') and v['failures'] == 0}
    for prob, arms in (('Heat (new bank)', ('nmrom_q0_field_cn', 'nmrom_q288_field_cn')),
                       ('Heat (new bank, batched fit)', ('nmrom_q0_field_direct_tol1e-4_chol', 'nmrom_q288_field_direct_tol1e-4_chol'))):
        f_, a_ = R_[arms[0]], R_[arms[1]]
        pick = rule_pick(cands, 100 * a_['error_all_times_worst']); assert pick == a_['fastest_fom_error_le_rom_all_times']['method'], (n, prob, pick)
        def st_(x, lab):
            return setting(lab, 100 * x['error_all_times_worst'], x['device_ms_median'], arm=x['method'], evolved_pct=100 * x['error_evolved_worst'],
                           nonstationary=x['failures'])
        dt_, rt_ = pick.split('_dt')[1].split('_')[0], pick.split('rtol')[1]
        row(prob, 3, n, st_(f_, '$q=0$'), st_(a_, '$q=288$'),
            dict(name='CN--CG, $\\Delta t{=}' + dt_ + '$, rtol $' + tol_tex(rt_) + '$', error_pct=cands[pick]['err'], ms=cands[pick]['ms'], arm=pick, candidates=cands),
            'heat3db_panel_a', _hm['job_id'], 'final', 'accepted final', 'same-grid, all times', 'GPU query')
    lb = [R_[k] for k in R_ if k.startswith('linear_bank_')]; nm = [R_[k] for k in R_ if k.startswith('nmrom_')]
    H3N[n] = dict(lin_err=[100 * min(x['error_all_times_worst'] for x in lb), 100 * max(x['error_all_times_worst'] for x in lb)],
                  lin_ms=[min(x['device_ms_median'] for x in lb), max(x['device_ms_median'] for x in lb)],
                  nm_min_ms=min(x['device_ms_median'] for x in nm), nm_min_err=100 * min(x['error_all_times_worst'] for x in nm))
assert H3N[128]['lin_ms'][1] < H3N[128]['nm_min_ms'] and H3N[128]['lin_err'][1] < H3N[128]['nm_min_err']   # the bank baseline beats every NM-ROM arm
H3N['floor'] = 100 * [c_ for c_ in D['heat3db_selection']['candidates'] if (c_['R'], c_['K']) == (320, 32)][0]['bank_floor']['128']
# panel B (chosen after the speed addenda): one appendix line only
_hb = {m['intervals']: {r_['method']: r_ for r_ in m['rows']} for m in D['heat3db_panel_b']['meshes'] if m['cohort'] == 'sealed_921099_never_opened'}[128]
_c2 = D['heat3db_selection_add1']['chosen']; assert (_c2['R'], _c2['K'], _c2['accurate_q']) == (256, 16, 160)
H3N['b'] = dict(err=100 * _hb['nmrom_q160_field_cn_tol1e-4_chol_s1']['error_all_times_worst'],
                cn=_hb['nmrom_q160_field_cn_tol1e-4_chol_s1']['fastest_fom_error_le_rom_all_times']['speedup'],
                bf=_hb['nmrom_q160_direct_tol1e-4_chol_mom_s1']['fastest_fom_error_le_rom_all_times']['speedup'])
for k_ in ('nmrom_q160_field_cn_tol1e-4_chol_s1', 'nmrom_q160_direct_tol1e-4_chol_mom_s1'):
    x_ = _hb[k_]['fastest_fom_error_le_rom_all_times']; assert abs(_hb[x_['method']]['device_ms_median'] / _hb[k_]['device_ms_median'] - x_['speedup']) < 1e-12
    assert _hb[k_]['failures'] == 0

# ---- Navier--Stokes follow-up diagnosis (ns3d-grok diag07; a DIFFERENT model, not the NM-ROM): one sentence in the failures paragraph
_ng = D['ns3d_grok_diag07']; assert _ng['final_cohort_opened'] and _ng['sealed_seed_opened'] and not _ng['smoke']
_co = _ng['coeff']; _cs = _co['stats']; assert _cs['cases'] == _ng['config']['eval_cases'] and _co['under_target']
_nf = {k: v for k, v in _ng['fom'].items() if v['stats']['evolved_worst'] <= _cs['evolved_worst']}
_nfk = min(_nf, key=lambda k: _nf[k]['median_ms']); assert _ng['fom'][_nfk]['dt'] == _ng['config']['dt']   # the rule's CNAB2 step is the paper's dt
NSG = dict(worst=100 * _cs['evolved_worst'], over=_cs['cases_evolved_over_target'], cases=_cs['cases'], target=100 * _ng['config']['target_relative'],
           speedup=_ng['fom'][_nfk]['median_ms'] / _co['median_ms'])

HP = {}   # controls of the hires-poisson lane, for the Limitations sentence


def hires_poisson(key, d):
    """Adapter for the hires-poisson lane summary (its own schema).

    Scope rule: never mix timing scopes inside one series.  The existing square/cube Poisson series are GPU-query
    times, so those rows use median_device_ms; the existing L-shape series is complete-query, so it uses
    median_total_ms.  The other scope is carried in ``alt`` and printed in the appendix timing table only.
    Arms are the lane's own verdict arms (accurate) and their q = 0 twins (fast); no variant is picked here.
    """
    audited = {(s['attempt']) for s in d['sources'] if s['audit_passed'] and all(s['gates'].values())}
    R_ = {(r['attempt'], r['mesh'], r['subject']): r for r in d['rows']}
    verdict = {(v['attempt'], v['mesh']): v for v in d['verdicts']}
    floors = {(b['attempt'], b['mesh']): b['worst'] for b in d['bank_floor']}
    plan = [  # attempt, lane mesh label, problem, dim, intervals, scope key, appendix-only reason
        ('hp2048', 'square 2048²', 'Poisson', 2, 2048, 'device', None),
        ('hp4096', 'square 4096²', 'Poisson', 2, 4096, 'device', None),
        ('hp4096b', 'square 4096²', 'Poisson', 2, 4096, 'device', 're-measure of the row above'),
        ('hp3d128', 'cube 64³', 'Poisson (dev. sources)', 3, 64, 'device', 'development sources; the held-out final row is in Table 1'),
        ('hp3d128', 'cube 128³', 'Poisson (dev. sources)', 3, 128, 'device', None),
        ('hp3d256', 'cube 128³', 'Poisson (dev. sources)', 3, 128, 'device', 're-measure of the row above'),
        ('hp3d256', 'cube 256³', 'Poisson (dev. sources)', 3, 256, 'device', None),
        ('hpl32', 'L-shape 1024² (32 sources)', 'Poisson, L-shape', 2, 1024, 'total', None),
        ('hpl32', 'L-shape 2048² (32 sources)', 'Poisson, L-shape', 2, 2048, 'total', None),
    ]
    for attempt, lm, problem, dim, n, scope, only in plan:
        assert attempt in audited, attempt
        v = verdict[(attempt, lm)]; assert not v.get('withdrawn_as_bar_verdict')
        acc = R_[(attempt, lm, v['arm'])]; q = acc['q']
        fast = R_[(attempt, lm, v['arm'].replace(f'rom_q{q}_', 'rom_q0_'))]
        _use = 'median_device_ms' if scope == 'device' else 'median_total_ms'
        cands = {k[2]: dict(err=100 * r_['worst_same_grid'], ms=r_[_use]) for k, r_ in R_.items()
                 if k[0] == attempt and k[1] == lm and r_['family'] == 'cg' and r_['cases'] == acc['cases']}
        fom = R_[(attempt, lm, rule_pick(cands, 100 * acc['worst_same_grid']))]
        assert fast['q'] == 0 and acc['family'] == fast['family'] == 'nm-rom' and fom['family'] == 'cg'
        assert q == max(r['q'] for (a, m, _), r in R_.items() if a == attempt and m == lm and r['family'] == 'nm-rom' and not _.count('m4') and not _.count('m8'))
        assert acc['cases'] == fast['cases'] == fom['cases'] and 'development' in acc['status']
        use, other = ('median_device_ms', 'median_total_ms') if scope == 'device' else ('median_total_ms', 'median_device_ms')
        if fom['subject'] == 'cg_0.01':
            sd = acc['speedups']['named_cg_1e-2']; assert abs(sd[scope] - fom[use] / acc[use]) < 1e-9       # lane ratio reproduces where the lane used the same setting
        alt = dict(scope='complete query' if scope == 'device' else 'GPU query', accurate=fom[other] / acc[other], fast=fom[other] / fast[other])
        row(problem, dim, n, setting(f'$q=0$', 100 * fast['worst_same_grid'], fast[use], arm=fast['subject']),
            setting(f'$q={q}$', 100 * acc['worst_same_grid'], acc[use], arm=acc['subject']),
            dict(name='CG, rtol $' + tol_tex(fom['subject'].split('_')[1]) + '$', error_pct=100 * fom['worst_same_grid'], ms=fom[use], arm=fom['subject'], candidates=cands), key, acc['job_id'], 'development',
            'development', 'same-grid', 'GPU query' if scope == 'device' else 'complete query',
            note=f"{acc['cases']} development sources; {acc['gpu']}; attempt {attempt}", alt=alt, appendix_only=only)
        if not only:
            ctl = {c: acc['speedups'][c] for c in ('dst_direct', 'fastest_coarse_matched') if acc['speedups'].get(c)}
            HP[(problem, n)] = dict(controls=ctl, floor_pct=100 * floors[(attempt, lm)], acc_err=100 * acc['worst_same_grid'])


HH = {}         # hires-heat facts for captions / Limitations / failures prose
HEAT_APPX = []  # rows of the appendix heat table (tight-tolerance ratios, evolved errors, batched variant, 3D speed rows)
HEAT_NAMED = 'fom_cncg_dt0.025_rtol1e-6_NAMED'


def hires_heat(parts):
    """Adapter for the hires-heat lane (schema hires-heat-summary-v1, one summary per job; role -> (key, data)).

    Table 1 convention, re-derived here from milliseconds: the FOM of a row is the fastest same-grid CN--CG setting
    with no failed solve whose worst all-times error does not exceed the accurate setting's; both settings of the row
    divide that one FOM time.  The lane's named CN--CG (rtol 1e-6) ratios go to the appendix only.  Fast = q=0 and
    accurate = the declared accurate rank of the same frozen model, both with the same stepping arm.
    """
    S = {}
    for role in ('h2d-final04', 'h2d-wide02b', 'h2d-ladder01', 'h3d-final08'):
        key, d = parts[role]
        md_ = d['metadata']
        assert d['schema'] == 'hires-heat-summary-v1' and d['audit_passed'] is True, role
        assert md_['backend'] == 'gpu' and md_['precision'] == 'highest' and md_['x64'] and not md_['local_smoke'], role
        S[role] = dict(key=key, job=md_['job_id'], gpu=md_['gpu'].split(',')[0], model=d['model_sha256'],
                       R={(m['cohort'], m['intervals']): {r['method']: r for r in m['rows']} for m in d['meshes']})
    tr = parts['wide2d-training'][1]
    # the 128-function bank of the wide rows is the bank recorded in the training file; final04 and wide02b share it and the K=8 head
    assert tr['config']['bank_rank'] == 128 and S['h2d-final04']['model'] == S['h2d-wide02b']['model']
    assert S['h2d-final04']['model']['bank'] == '93b0ec7ad22526f003b1f2d81dfecab2f6bc883ea73ad715f85f7437ede73233'   # inputs/wide2d/SHA256SUMS
    assert S['h2d-final04']['model']['head'] == '49c2ae73b0fde3c5126604af770af14ddcb5bef254f71fc93cf4f53eccbce498'   # head_K8.pkl
    HH['wide_floor'] = 100 * tr['bank']['validation_projection_worst']
    c_ = tr['config']
    HH['training_sentence'] = (
        r'The wide heat bank of Table~\ref{tab:headline} is a random-Fourier-feature coordinate network '
        f"({c_['fourier_features']} features, scale {c_['fourier_scale']:g}, width {c_['bank_width']}, $R={c_['bank_rank']}$) trained for "
        f"{c_['bank_steps']} steps (learning rate {c_['bank_learning_rate']:g}, {c_['bank_batch_states']} states $\\times$ {c_['bank_batch_points']} points per batch, "
        f"gradient clip {c_['bank_gradient_clip']:g}, exact coefficient refit every {c_['refit_every']} steps), then a head of width {c_['head_width']} with $k=8$ for "
        f"{c_['head_steps']} steps (learning rate {c_['head_learning_rate']:g}, {c_['head_batch_states']} states per batch), on {c_['train_count']} training draws "
        f"at ${c_['train_intervals']}^2$ with {c_['validation_count']} validation draws; its validation projection floor is {100 * tr['bank']['validation_projection_worst']:.2f}\\,\\%.")

    def fom_for(R, err):
        c = {k: v for k, v in R.items() if k.startswith('fom_cncg_') and v['failures'] == 0 and v['error_all_times_worst'] <= err}
        return min(c, key=lambda k: c[k]['device_ms_median'])

    def fom_label(k):
        dt, tol = k.split('_dt')[1].split('_rtol'); tol = tol.split('_')[0]
        return r'CN--CG, $\Delta t{=}%s$, rtol $10^{%s}$' % (dt, tol.split('e')[1])

    def pair(role, cohort, n, fast_arm, acc_arm):
        R = S[role]['R'][(cohort, n)]; f, a = R[fast_arm], R[acc_arm]
        assert f['failures'] == a['failures'] == 0 and f['cases'] == a['cases'] == R[HEAT_NAMED]['cases']
        assert a['error_all_times_worst'] <= f['error_all_times_worst']
        c = fom_for(R, a['error_all_times_worst']); fom = R[c]
        lane = a['fastest_fom_error_le_rom_all_times']                       # the lane's own pick must be ours, and its ratio must reproduce
        assert lane['method'] == c and abs(lane['speedup'] - fom['device_ms_median'] / a['device_ms_median']) < 1e-9, (role, n, lane, c)
        for x in (f, a):
            assert abs(x['speedup_vs_named_fom'] - R[HEAT_NAMED]['device_ms_median'] / x['device_ms_median']) < 1e-9
        q = int(acc_arm.split('_q')[1].split('_')[0])
        st = lambda x, lab: setting(lab, 100 * x['error_all_times_worst'], x['device_ms_median'], arm=x['method'],
                                    evolved_pct=100 * x['error_evolved_worst'])
        return dict(R=R, fast=st(f, '$q=0$'), acc=st(a, f'$q={q}$'), cases=a['cases'], fom_arm=c,
                    fom=dict(name=fom_label(c), error_pct=100 * fom['error_all_times_worst'], ms=fom['device_ms_median'], arm=c,
                             candidates={k: dict(err=100 * v['error_all_times_worst'], ms=v['device_ms_median']) for k, v in R.items() if k.startswith('fom_cncg_') and v['failures'] == 0}),
                    tight=dict(scope=r'vs.\ CN--CG rtol $10^{-6}$', accurate=a['speedup_vs_named_fom'], fast=f['speedup_vs_named_fom']))

    CN = ('nmrom_q0_cn', 'nmrom_q32_cn'); BF = ('nmrom_q0_field_direct_tol1e-4_chol', 'nmrom_q32_field_direct_tol1e-4_chol')
    plan = [  # role, cohort, mesh, problem, arms, Table-1 cohort, status
        ('h2d-final04', 'sealed_opened_once', 1024, 'Heat (wide bank)', CN, 'final', 'sealed held-out; opened once'),
        ('h2d-final04', 'sealed_opened_once', 1024, 'Heat (wide bank, batched fit)', BF, 'final', 'sealed held-out; opened once'),
        ('h2d-final04', 'sealed_opened_once', 2048, 'Heat (wide bank)', CN, 'final', 'sealed held-out; opened once'),
        ('h2d-final04', 'sealed_opened_once', 4096, 'Heat (wide bank)', CN, 'final', 'sealed held-out; opened once'),
        ('h2d-final04', 'sealed_opened_once', 2048, 'Heat (wide bank, batched fit)', BF, 'final', 'sealed held-out; opened once'),
        ('h2d-final04', 'sealed_opened_once', 4096, 'Heat (wide bank, batched fit)', BF, 'final', 'sealed held-out; opened once'),
    ]
    for role, cohort, n, problem, arms, coh, status in plan:
        p = pair(role, cohort, n, *arms)
        row(problem, 2, n, p['fast'], p['acc'], p['fom'], S[role]['key'], S[role]['job'], coh, status, 'same-grid, all times', 'GPU query',
            note=f"{p['cases']} cases; {S[role]['gpu']}; arms {arms[0]} / {arms[1]}", alt=p['tight'])
    # appendix heat table: every NM-ROM pair of the lane that the paper cites, same convention, plus the tight-tolerance ratios
    appx = [  # model label, role, cohort, cohort label, dim, mesh, stepping label, arms
        ('Wide bank', 'h2d-wide02b', 'all', 'development', 2, 1024, 'CN', CN),
        ('Wide bank', 'h2d-final04', 'sealed_opened_once', 'sealed', 2, 1024, 'CN', CN),
        ('Wide bank', 'h2d-final04', 'sealed_opened_once', 'sealed', 2, 1024, 'batched fit', BF),
        ('Wide bank', 'h2d-final04', 'sealed_opened_once', 'sealed', 2, 2048, 'CN', CN),
        ('Wide bank', 'h2d-final04', 'sealed_opened_once', 'sealed', 2, 2048, 'batched fit', BF),
        ('Wide bank', 'h2d-final04', 'sealed_opened_once', 'sealed', 2, 4096, 'CN', CN),
        ('Wide bank', 'h2d-final04', 'sealed_opened_once', 'sealed', 2, 4096, 'batched fit', BF),
        ('Earlier ($R{=}32$)', 'h2d-ladder01', 'all', 'development', 2, 2048, 'CN', ('nmrom_q0_cn', 'nmrom_q24_cn')),
        ('Earlier ($R{=}32$)', 'h2d-ladder01', 'all', 'development', 2, 4096, 'CN', ('nmrom_q0_cn', 'nmrom_q24_cn')),
        ('3D, earlier bank', 'h3d-final08', 'paper_h3d_final_cohort', 'final', 3, 128, 'CN', ('nmrom_q0_cn', 'nmrom_q96_cn')),
        ('3D, earlier bank', 'h3d-final08', 'paper_h3d_final_cohort', 'final', 3, 128, 'batched fit', ('nmrom_q0_direct_tol1e-4_chol', 'nmrom_q96_direct_tol1e-4_chol')),
        ('3D, earlier bank', 'h3d-final08', 'paper_h3d_final_cohort', 'final, 1st 16', 3, 256, 'CN', ('nmrom_q0_cn', 'nmrom_q96_cn')),
        ('3D, earlier bank', 'h3d-final08', 'paper_h3d_final_cohort', 'final, 1st 16', 3, 256, 'batched fit', ('nmrom_q0_direct_tol1e-4_chol', 'nmrom_q96_direct_tol1e-4_chol')),
    ]
    for model, role, cohort, clab, dim, n, step, arms in appx:
        p = pair(role, cohort, n, *arms)
        for s in ('fast', 'acc'): p[s]['speedup'] = p['fom']['ms'] / p[s]['ms']
        HEAT_APPX.append(dict(model=model, role=role, source=S[role]['key'], job=S[role]['job'], gpu=S[role]['gpu'], cohort=clab, cases=p['cases'],
                              dim=dim, intervals=n, stepping=step, fast=p['fast'], accurate=p['acc'], fom=p['fom'], tight=p['tight']))
    # Limitations: controls at 4096^2 on the sealed cohort, all faster than every NM-ROM arm measured there
    R = S['h2d-final04']['R'][('sealed_opened_once', 4096)]
    acc_err = R[CN[1]]['error_all_times_worst']
    nm_min = min(v['device_ms_median'] for k, v in R.items() if k.startswith('nmrom_'))
    coarse = {k: v for k, v in R.items() if k.startswith('coarse') and v['failures'] == 0 and v['physical_all_times_worst'] <= acc_err}
    ck = min(coarse, key=lambda k: coarse[k]['device_ms_median'])
    lin = [R[k] for k in ('linear_bank_field_BASELINE', 'linear_bank_moments_BASELINE')]
    ctl = [coarse[ck]['device_ms_median'], R['dst_exact_CONTROL']['device_ms_median']] + [x['device_ms_median'] for x in lin]
    assert max(ctl) < nm_min, (ctl, nm_min)
    HH.update(coarse_arm=ck, coarse_err=100 * coarse[ck]['physical_all_times_worst'], coarse_ms=coarse[ck]['device_ms_median'],
              dst_ms=R['dst_exact_CONTROL']['device_ms_median'], lin_err=100 * max(x['error_all_times_worst'] for x in lin),
              lin_ms=[min(x['device_ms_median'] for x in lin), max(x['device_ms_median'] for x in lin)], nmrom_min_ms=nm_min)
    # Heat 3D all-times convention (same frozen model, same final cohort, 128^3) and the bank's initial-field limit
    R3 = S['h3d-final08']['R']
    HH['h3d_all'] = 100 * R3[('paper_h3d_final_cohort', 128)]['nmrom_q96_cn']['error_all_times_worst']
    HH['h3d_evolved'] = 100 * R3[('paper_h3d_final_cohort', 128)]['nmrom_q96_cn']['error_evolved_worst']
    init = []
    for n in (128, 256):
        lb = [R3[('paper_h3d_final_cohort', n)][k] for k in ('linear_bank_field_BASELINE', 'linear_bank_moments_BASELINE')]
        for x in lb: assert x['error_all_times_worst'] > x['error_evolved_worst']        # the all-times maximum is attained at t = 0
        init.append(100 * min(x['error_all_times_worst'] for x in lb))
    HH['h3d_init'] = [min(init), max(init)]


HB = {}   # hires-burgers facts for captions / Limitations


def hires_burgers(parts):
    """Adapter for the hires-burgers lane (lane summary + per-job NumPy audit summaries; role -> (key, data)).

    Arms come from the lane's pre-registered roles in ``summary['matrix']`` (chosen on dev6 / pre-declared accurate
    rung / fast q=0); every number is then read from the per-job audit table and every ratio recomputed from ms.
    Table 1 FOM = fastest same-grid Newton--BiCGStab arm that converged every step with worst evolved error <= the
    accurate setting's (the lane's ``fastest_at_least_as_accurate`` for that arm, asserted).  Tight-Newton and
    host-inclusive ratios go to the appendix only.
    """
    sm = parts['summary'][1]
    src = {s['attempt']: s for s in sm['sources']}
    T = {}
    for role in ('hb2k02', 'hb2kh64', 'hb4k04', 'hb4kh64'):
        key, d = parts[role]
        assert MAN[key]['sha256'] == src[role]['sha256'] and d['job_id'] == src[role]['job_id'], role   # the audit the summary names
        g = d['gates']
        for name, v in g.items():
            if name != 'restricted_recomputation_tracks_full_grid': assert v['passed'] in (True, None), (role, name)
        assert g['log_says_backend_gpu']['passed'] and g['x64_and_highest']['passed'] and g['complete']['passed']
        assert set(d['failed_gates']) <= {'restricted_recomputation_tracks_full_grid'}
        T[role] = (key, d)
    M = {(m['attempt'], m['role']): m for m in sm['matrix']}

    def fom_label(a):
        def t(x):
            m, ex = f'{x:.0e}'.split('e'); ex = int(ex)
            return f'10^{{{ex}}}' if m == '1' else f'{m}{{\\times}}10^{{{ex}}}'
        s = t(a['ntol']) if a['ntol'] == a['ltol'] else t(a['ntol']) + '/' + t(a['ltol'])
        return r'Newton--BiCGStab, tol $%s$' % s

    def lab(a):
        return f"$q={a['q']}$, $M={a['M']}$, EQ " + (f"lattice $m={a['m']}$" if a['rule'] == 'lat64' else f"$m={a['m']}$")

    plan = [  # attempt, mesh, accurate role, problem, cohort, status
        ('hb2k02', 2048, 'chosen on dev6', 'Burgers', 'development', 'development; 6 cases, arm chosen here'),
        ('hb4k04', 4096, 'chosen on dev6', 'Burgers', 'development', 'development; 6 cases, arm chosen here'),
        ('hb2kh64', 2048, 'accurate rung q256/M1088', 'Burgers (held-out cases)', 'held-out', 'held-out; 64 cases, one timing repetition'),
        ('hb4kh64', 4096, 'chosen on dev6', 'Burgers (held-out cases)', 'held-out', 'held-out; 64 cases, one timing repetition'),
    ]
    for att, n, arole, problem, coh, status in plan:
        key, d = T[att]; t = d['table']; assert d['intervals'] == n
        ma, mf = M[(att, arole)], M[(att, 'fast q=0')]
        a, f = t[ma['arm']], t[mf['arm']]
        assert ma['certified_primary'] and a['stalled_exits'] == f['stalled_exits'] == 0 and a['cases'] == f['cases'] == d['cohort_cases']
        assert a['q'] == 256 and f['q'] == 0 and a['worst_evolved_percent'] <= f['worst_evolved_percent']
        foms = {k: v for k, v in t.items() if v['family'] == 'fom' and v['mesh'] == n and v['nonlinear_converged'] and v['stalled_steps'] == 0}
        ok = {k: v for k, v in foms.items() if v['worst_evolved_percent'] <= a['worst_evolved_percent']}
        c = min(ok, key=lambda k: ok[k]['median_gpu_ms'])
        assert c == ma['fastest_at_least_as_accurate'] and abs(ma['s_fastest_at_least_as_accurate_gpu'] - t[c]['median_gpu_ms'] / a['median_gpu_ms']) < 1e-9
        rl = [r_ for r_ in d['rules'] if a['name'] in r_['deployed']]
        assert len(rl) == 1 and rl[0]['deployed'][a['name']]['certified_primary'] and rl[0]['source'][0]['kind'] == 'lattice' and rl[0]['m'] == 63 * 63
        tight = t[ma['tight']]
        assert abs(ma['s_tight_gpu'] - tight['median_gpu_ms'] / a['median_gpu_ms']) < 1e-9 and abs(ma['s_tight_host'] - tight['median_host_ms'] / a['median_host_ms']) < 1e-9
        assert abs(mf['s_tight_gpu'] - tight['median_gpu_ms'] / f['median_gpu_ms']) < 1e-9
        fom = t[c]
        alt = [dict(scope='complete query', accurate=fom['median_host_ms'] / a['median_host_ms'], fast=fom['median_host_ms'] / f['median_host_ms']),
               dict(scope=r'vs.\ tight Newton', accurate=tight['median_gpu_ms'] / a['median_gpu_ms'], fast=tight['median_gpu_ms'] / f['median_gpu_ms'])]
        row(problem, 2, n, setting(lab(f), f['worst_evolved_percent'], f['median_gpu_ms'], arm=f['name']),
            setting(lab(a), a['worst_evolved_percent'], a['median_gpu_ms'], arm=a['name'], eq='lattice'),
            dict(name=fom_label(fom), error_pct=fom['worst_evolved_percent'], ms=fom['median_gpu_ms'], arm=c,
                 candidates={k: dict(err=v['worst_evolved_percent'], ms=v['median_gpu_ms']) for k, v in foms.items()}), key, d['job_id'], coh, status,
            'same-grid, evolved', 'GPU query', note=f"{d['cohort_cases']} cases, {a['reps']} repetitions; {d['gpu']}", alt=alt)
        HB[(att, 'coarse')] = {k: v for k, v in t.items() if k.startswith('c') and v['family'] == 'fom' and v['mesh'] < n}
        HB[(att, 'acc')] = a
    # the dev6 arm chosen at 2048^2 (M=544) was not run on hold64; its 4096^2 hold64 twin is quoted so no arm is hidden
    m544 = T['hb4kh64'][1]['table']['q256_M544_lat64_g0p001_fast_chol_clip_lamcarry_pred2']
    assert ('hb2kh64', 'chosen on dev6') not in M and m544['cases'] == 64
    HB['m544_hold_4096'] = m544['worst_evolved_percent']
    # coarse-grid Newton (1024^2 interpolated) against the dev6 accurate setting at both meshes
    for att in ('hb2k02', 'hb4k04'):
        cc = HB[(att, 'coarse')]['c1024_nt1e-4_dt005']; acc = HB[(att, 'acc')]
        assert cc['median_gpu_ms'] < acc['median_gpu_ms'] and cc['nonlinear_converged']
        HB[att] = dict(coarse_err=cc['worst_evolved_percent'], coarse_ms=cc['median_gpu_ms'], acc_err=acc['worst_evolved_percent'], acc_ms=acc['median_gpu_ms'])
    # host copy of six f64 fields at 4096^2: host minus GPU time, every arm of the dev6 job
    t4 = T['hb4k04'][1]['table']; gaps = sorted(v['median_host_ms'] - v['median_gpu_ms'] for v in t4.values())
    HB['host_gap_4096'] = gaps[len(gaps) // 2]; HB['host_gap_range'] = (gaps[0], gaps[-1])
    # the failed restricted-recomputation gate: count rows and compare cohort-worst per arm
    for att in ('hb2kh64', 'hb4kh64'):
        rr = [x for x in T[att][1]['restricted_recheck'] if x['job_full_evolved'] > 1e-6]
        bad = [x for x in rr if abs(x['restricted_evolved'] / x['job_full_evolved'] - 1) >= 0.05]
        worst = defaultdict(lambda: [0., 0.])
        for x in rr:
            w = worst[x['name']]; w[0] = max(w[0], x['restricted_evolved']); w[1] = max(w[1], x['job_full_evolved'])
        fg = T[att][1]['full_grid_recheck']; assert all(x['abs_diff'] < 1e-15 for x in fg)
        HB[(att, 'gate')] = dict(bad=len(bad), rows=len(rr), worst_gap=max(abs(w[0] / w[1] - 1) for w in worst.values()))
    bf = parts['bank-floor-summary'][1]['solve_burgers']['inc512_full']; assert bf['R'] == 512 and bf['kind'] == 'rom'
    HB['bank_floor_confirm'] = 100 * bf['worst_evolved']['confirm']


BH5 = {}


def burgers_heldout():
    """4096^2 quadrature re-test and the wider-bank result, from the pinned burgers-heldout blobs.

    Table 1 keeps the hires-burgers accurate rows.  Their lattice rule is relabelled from the re-draws.
    The fast-column speedup is re-derived on bh5's wider Newton grid: fastest converged same-mesh setting
    whose error is at most the fast setting's, same job as that fast setting.  The confirmed form (first
    time step on the exact residual) is an appendix row.  The wider bank did not improve held-out accuracy,
    so it is prose, not a table row.
    """
    sm, bh, eq = D['bh_summary'], D['bh5_summary'], D['bh5_eqcert']
    assert MAN['bh5_summary']['sha256'] == sm['sources_sha256']['bh5-summary.json']
    assert MAN['bh5_eqcert']['sha256'] == sm['sources_sha256']['bh5-eqcert-summary.json']
    assert bh['job_id'] == eq['job_id'] == sm['jobs']['bh5']['job_id'] and bh['intervals'] == eq['intervals'] == 4096
    for d_ in (bh, eq):
        assert set(d_['failed_gates']) <= {'restricted_recomputation_tracks_full_grid'}
        g_ = d_['gates']
        assert g_['log_says_backend_gpu']['passed'] and g_['x64_and_highest']['passed'] and g_['full_grid_errors_recomputed']['passed']
    assert eq['gates']['arm_status_recomputed_matches_job']['passed'] and eq['gates']['rho_recomputed_in_numpy']['passed']
    assert eq['verdict']['certified_rule_exists'] is False and {r_['rho_bar'] for r_ in eq['rules']} == {EQC['bar']}
    paper, exact = 'q256_M1088_lat64_g0p01_fast_chol_clip_lamcarry_pred2', 'q256_M1088_lat64_g0p01_fast_chol_clip_lamcarry_pred2_x1'
    fast_arm = 'q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2'
    st0, st1, stq = eq['arm_status'][paper], eq['arm_status'][exact], eq['arm_status'][fast_arm]
    assert st0['draws_passed'] == st0['draws'] == 5 and st0['exact_steps'] == 0 and not st0['confirmation_pass'] and not st0['audited_confirmation_pass']
    assert st1['draws_passed'] == st1['draws'] == 5 and st1['exact_steps'] == 1 and st1['confirmation_pass'] and st1['audited_confirmation_pass']
    assert stq['confirmation_pass'] and stq['draws_passed'] == stq['draws'] == 5 and stq['status'] == 'confirmed'   # the fast setting's smaller rule does pass
    assert eq['table'][exact]['fom_by_paper_rule'] == 'lean_nt3e-3_l3e-3_dt005'
    BH5['j0_rho'] = st0['confirmation_heldout_rho_max_k>=0']
    BH5['j0_rho_k2'] = st0['confirmation_heldout_rho_max_k>=2']
    BH5['j1_rho'] = st1['confirmation_heldout_rho_max_k>=1']
    assert abs(BH5['j1_rho'] - eq['table'][exact]['confirmation_rho_max']) < 1e-15
    assert BH5['j0_rho'] > EQC['bar'] > BH5['j0_rho_k2'] and BH5['j1_rho'] < EQC['bar']

    def foms_of(table):
        return {k: v for k, v in table.items() if v['family'] == 'fom' and v['mesh'] == 4096 and v['nonlinear_converged'] and v['stalled_steps'] == 0}

    def newton_name(v, with_dt=False):
        s = tol_tex(v['ntol']) if v['ntol'] == v['ltol'] else tol_tex(v['ntol']) + '/' + tol_tex(v['ltol'])
        lab = r'Newton--BiCGStab, tol $%s$' % s
        return lab + (r', $\Delta t{=}%s$' % f"{v['dt']:g}" if with_dt else '')

    fast_sps = []
    for cohort, problem, coh in (('dev6', 'Burgers', 'development'), ('hold64', 'Burgers (held-out cases)', 'held-out')):
        table = bh['groups'][cohort]['table']
        r = [x for x in ROWS if x['problem'] == problem and x['intervals'] == 4096][0]
        assert r['accurate']['arm'] == paper and r['fast']['arm'] == fast_arm and r['accurate']['eq'] == 'lattice'
        j0, j1, fq = table[paper], table[exact], table[fast_arm]
        assert abs(j1['worst_evolved_percent'] - r['accurate']['error_pct']) < 1e-4
        assert abs(fq['worst_evolved_percent'] - r['fast']['error_pct']) < 1e-6
        assert j1['stalled_exits'] == fq['stalled_exits'] == 0
        foms = foms_of(table)
        cands = {k: dict(err=v['worst_evolved_percent'], ms=v['median_gpu_ms']) for k, v in foms.items()}
        fpick = rule_pick(cands, fq['worst_evolved_percent']); assert fpick == 'lean_nt1e-3_l1e-3_dt01'
        fo = foms[fpick]
        r['fast']['own_fom'] = dict(arm=fpick, ms=fo['median_gpu_ms'], err=fo['worst_evolved_percent'], rom_ms=fq['median_gpu_ms'],
                                    rom_err=fq['worst_evolved_percent'], job=bh['job_id'])
        r['fast']['speed_digits'] = 2
        r['accurate']['eq'] = 'lattice-unconfirmed'
        fast_sps.append(fo['median_gpu_ms'] / fq['median_gpu_ms'])
        apick = rule_pick(cands, j1['worst_evolved_percent']); assert apick == 'lean_nt3e-3_l3e-3_dt005'
        af = foms[apick]
        lab = f"$q={j1['q']}$, $M={j1['M']}$, EQ lattice $m={j1['m']}$, first step exact"
        row('Burgers, exact first step', 2, 4096, None,
            setting(lab, j1['worst_evolved_percent'], j1['median_gpu_ms'], arm=exact, eq='confirmed'),
            dict(name=newton_name(af), error_pct=af['worst_evolved_percent'], ms=af['median_gpu_ms'], arm=apick, candidates=cands),
            'bh5_summary', bh['job_id'], coh, 'confirmed rule; first step on the exact residual', 'same-grid, evolved', 'GPU query',
            appendix_only='confirmed form of the 4096^2 lattice rule; slower than Newton--BiCGStab')
        BH5[cohort] = dict(j1_ms=j1['median_gpu_ms'], j0_ms=j0['median_gpu_ms'], sp=af['median_gpu_ms'] / j1['median_gpu_ms'],
                           cost=j1['median_gpu_ms'] / j0['median_gpu_ms'])
    assert f'{fast_sps[0]:.2f}' == f'{fast_sps[1]:.2f}' == '10.10'
    assert f"{BH5['dev6']['sp']:.2f}" == '0.10' and f"{BH5['hold64']['sp']:.2f}" == '0.08'
    BH5['fast_s'] = '10.10'
    BH5['fast_fom'] = newton_name(bh['groups']['dev6']['table']['lean_nt1e-3_l1e-3_dt01'], with_dt=True)
    assert BH5['fast_fom'] == newton_name(bh['groups']['hold64']['table']['lean_nt1e-3_l1e-3_dt01'], with_dt=True)
    sel, floors = sm['bank_selection'], sm['bank_floors_256_sel32']
    assert sel['chosen'] == 'state' and floors['pod_state_R512'] == sel['chosen_floor'] == floors['pod_state_R512']
    assert floors['incumbent_inc512'] == sel['incumbent_floor'] and sel['chosen_floor'] < 0.5 * sel['incumbent_floor']
    hold = [r_ for r_ in sm['rows'] if r_['job'] == 'bh3' and r_['cohort'] == 'hold64' and str(r_['role']).startswith('headline')]
    assert len(hold) == 1 and hold[0]['certified'] and hold[0]['arm'] == paper
    assert abs(hold[0]['fom_ms'] / hold[0]['rom_ms'] - hold[0]['speedup']) < 1e-12
    paper_hold = [x for x in ROWS if x['problem'] == 'Burgers (held-out cases)' and x['intervals'] == 4096][0]
    assert hold[0]['worst_evolved_percent'] > paper_hold['accurate']['error_pct']
    subs = [r_ for r_ in sm['rows'] if r_['mesh'] == 4096 and r_['cohort'] in ('hold64', 'fresh64') and r_['q'] > 0
            and not r_['certified'] and r_['worst_evolved_percent'] < 0.5]
    assert len(subs) >= 2 and all(abs(r_['fom_ms'] / r_['rom_ms'] - r_['speedup']) < 1e-12 and r_['speedup'] > 1 for r_ in subs)
    BH5.update(floor_old=100 * sel['incumbent_floor'], floor_new=100 * sel['chosen_floor'], bank_err=hold[0]['worst_evolved_percent'],
               bank_s=hold[0]['speedup'], sub_err=(min(r_['worst_evolved_percent'] for r_ in subs), max(r_['worst_evolved_percent'] for r_ in subs)),
               sub_s=(min(r_['speedup'] for r_ in subs), max(r_['speedup'] for r_ in subs)))
    assert f"{BH5['floor_old']:.3f}" == '0.513' and f"{BH5['floor_new']:.3f}" == '0.183' and f"{BH5['bank_err']:.3f}" == '1.601'


# ---- coordinator-supplied lane rows ----------------------------------------------------------------
HEAT_PARTS = {}; BURG_PARTS = {}
for k, v in MAN.items():
    if not k.startswith('intake_'): continue
    d = D[k]
    if v.get('adapter') == 'hires-poisson-v1':
        hires_poisson(k, d); continue
    if v.get('adapter') == 'hires-heat-v1':
        HEAT_PARTS[v['role']] = (k, d); continue
    if v.get('adapter') == 'hires-burgers-v1':
        BURG_PARTS[v['role']] = (k, d); continue
    if v.get('adapter') == 'nmrom-baselines-v1':
        continue                                   # Table 2 only; read by the Table 2 block below
    assert d['schema'] == INTAKE_SCHEMA, k
    for r in d['headline_rows']:
        row(r['problem'], r['dim'], r['intervals'], r.get('fast'), r.get('accurate'), r['fom'], k, r['job_id'], r['cohort'],
            r['status'], r['error_convention'], r['timing_scope'], r.get('note', ''))
if HEAT_PARTS:
    hires_heat(HEAT_PARTS)
if BURG_PARTS:
    hires_burgers(BURG_PARTS)
if 'bh5_summary' in D:
    burgers_heldout()

ROWS.sort(key=lambda r: (r['dim'], ORDER.index(r['problem']), r['intervals'], r['fom']['ms']))
APPX.sort(key=lambda r: (r['dim'], ORDER.index(r['problem']), r['intervals']))
for r in ROWS + APPX:
    for s in ('fast', 'accurate'):
        if not r[s]: continue
        fo = r[s].get('own_fom')
        if fo:
            assert fo['err'] <= r[s]['error_pct'] + 1e-9
            r[s]['speedup'] = fo['ms'] / fo['rom_ms']
        else:
            r[s]['speedup'] = r['fom']['ms'] / r[s]['ms']


# ---- rendering -----------------------------------------------------------------------------------------
def e(x): return f'{x:.2f}' if x >= 0.1 else f'{x:.3f}'
def spn(x): return (f'{x:.0f}' if x >= 100 else f'{x:.1f}' if x >= 10 else f'{x:.2f}' if x >= 0.1 else f'{x:.3f}' if x >= 0.01 else f'{x:.4f}')
def sp(x, nd=None):
    t = (f'{x:.{nd}f}' if nd else (f'{x:.0f}' if x >= 100 else f'{x:.1f}' if x >= 10 else f'{x:.2f}' if x >= 0.1 else f'{x:.3f}' if x >= 0.01 else f'{x:.4f}')) + r'$\times$'
    return r'\textbf{' + t + '}' if x > 1 else t
def mesh(r): return f"${r['intervals']}^{r['dim']}$"
MARK = {'refined reference': r'$^{r}$', 'complete query': r'$^{c}$'}
def marks(r):
    m = MARK.get(r['error_convention'], '') + MARK.get(r['timing_scope'], '')
    if 'all times' in r['error_convention']: m += r'$^{t}$'        # every output time including t = 0
    if 'evolved' in r['error_convention']: m += r'$^{e}$'          # evolved output times only (t = 0 excluded)
    if r['status'].startswith('provisional'): m += r'$^{p}$'
    if r['cohort'] == 'final' and not r['status'].startswith('provisional'): m += r'$^{f}$'
    if r['cohort'] == 'held-out': m += r'$^{h}$'            # held-out cases never used for selection, not the sealed final cohort
    return m
EQMARK = {'dense': r'$^{d}$', 'single-draw': r'$^{s}$', 'not-confirmed': r'$^{s}$', 'single-draw-refit': r'$^{x}$', 'confirmed': r'$^{v}$', 'lattice': r'$^{\ell}$', 'lattice-unconfirmed': r'$^{w}$'}
def cells(s): return [e(s['error_pct']) + EQMARK.get(s.get('eq'), ''), r'---$^{n}$' if s.get('nonstationary') else sp(s['speedup'], s.get('speed_digits'))] if s else ['---', '---']   # a solve that missed its stopping rule does not enter a speedup


INGESTED = {v.get('lane') for v in MAN.values() if v.get('lane')}


def covered(p, n):
    return any(r['problem'].startswith(p['problem']) and ',' not in r['problem'] and r['dim'] == p['dim'] and r['intervals'] == n for r in ROWS)


def write(name, lines, mdhead, mdrows):
    (HERE / 'tables' / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    (HERE / 'tables-md' / f'{name}.md').write_text('| ' + ' | '.join(mdhead) + ' |\n| ' + ' | '.join(['---'] * len(mdhead)) + ' |\n' + ''.join('| ' + ' | '.join(x) + ' |\n' for x in mdrows))


def md(x): return x.replace('{=}', '=').replace(r'.\ ', '. ').replace('---', '—').replace(r'$\times$', '×').replace(r'\textbf{', '**').replace('}', '**' if 'textbf' in x else '}').replace('$', '').replace('--', '–')

lines = [r'% GENERATED by paper/gen_headline.py from hash-pinned snapshots -- do not edit.', r'\small\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{@{}llrrrrrl@{}}', r'\toprule',
         r' & & \multicolumn{2}{c}{NM-ROM accurate} & \multicolumn{2}{c}{NM-ROM fast} & \multicolumn{2}{c}{Full-order model (FOM)} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}',
         r'Problem & Mesh & Err.\ (\%) & Speedup & Err.\ (\%) & Speedup & Err.\ (\%) & Solver \\']
mdrows = []
for dim in (2, 3):
    lines += [r'\midrule', r'\multicolumn{8}{@{}l}{\emph{' + ('Two' if dim == 2 else 'Three') + r'-dimensional}} \\']
    seen = None
    for r in [x for x in ROWS if x['dim'] == dim]:
        pname = r['problem'] + marks(r)
        c = [pname if (r['problem'], r['status']) != seen else '', mesh(r)] + cells(r['accurate']) + cells(r['fast']) + [e(r['fom']['error_pct']), r['fom']['name']]
        seen = (r['problem'], r['status']); lines.append(' & '.join(c) + r' \\')
        mdrows.append([f"{r['problem']} {dim}D ({r['status']})", f"{r['intervals']}^{dim}"] + [md(x) for x in c[2:]])
    for p in [x for x in PENDING if x['dim'] == dim]:
        if p['lane'] in INGESTED: continue       # a closed, ingested lane leaves no reserved slot; its unfilled meshes are reported in the appendix
        todo = [n for n in p['meshes'] if not covered(p, n)]
        if not todo: continue
        m = ', '.join(f'${n}^{dim}$' for n in todo)
        lines.append(f"{p['problem']} & {m} & --- & --- & --- & --- & --- & \\emph{{pending: {p['lane']}}} \\\\")
        mdrows.append([f"{p['problem']} {dim}D", ', '.join(f'{n}^{dim}' for n in todo), '—', '—', '—', '—', '—', f"pending: {p['lane']} lane"])
    # 2026-09-21 user decision: paused fix-the-failures lanes; slots marked "results incoming", filled only from an audited summary.json
    for inc in [x for x in INCOMING if x['dim'] == dim]:
        m = ', '.join(f'${n}^{dim}$' for n in inc['meshes'])
        lines.append(f"{inc['problem']} & {m} & \\multicolumn{{6}}{{l}}{{\\emph{{results incoming}}}} \\\\")
        mdrows.append([f"{inc['problem']} {dim}D", ', '.join(f'{n}^{dim}' for n in inc['meshes']), 'results incoming', '', '', '', '', f"lane {inc['lane']}"])
lines += [r'\bottomrule', r'\end{tabular}']
write('TH_headline', lines, ['Problem', 'Mesh', 'Accurate err. (%)', 'Accurate speedup', 'Fast err. (%)', 'Fast speedup', 'FOM err. (%)', 'FOM'], mdrows)

# supporting times (appendix)
tl = [r'% GENERATED by paper/gen_headline.py -- do not edit.', r'\scriptsize', r'\begin{tabular}{@{}llllrrrllll@{}}', r'\toprule',
      r'Problem & Mesh & Accurate & Fast & Accurate ms & Fast ms & FOM setting & FOM ms & Timing & Other scope or FOM: acc.\ / fast & Status \\', r'\midrule']
tmd = []
for r in sorted(ROWS + APPX, key=lambda r: (r['dim'], ORDER.index(r['problem']), r['intervals'], bool(r['appendix_only']), r['fom']['ms'])):
    a, f = r['accurate'], r['fast']
    c = [f"{r['problem']} {r['dim']}D" + (r'$^{\ast}$' if r['appendix_only'] else ''), mesh(r), a['label'] if a else '---', f['label'] if f else '---', f"{a['ms']:.2f}" if a else '---',
         f"{f['ms']:.2f}" if f else '---', r['fom']['name'], f"{r['fom']['ms']:.2f}", r['timing_scope'],
         '; '.join(spn(x['accurate']) + r'$\times$ / ' + spn(x['fast']) + r'$\times$ (' + x['scope'] + ')' for x in (r['alt'] if isinstance(r['alt'], list) else [r['alt']])) if r.get('alt') else '---',
         r['status'].split(';')[0]]
    tl.append(' & '.join(c) + r' \\'); tmd.append([md(re.sub(r'\\texttt\{(\w+)\}', r'\1', x)) for x in c])
tl += [r'\bottomrule', r'\end{tabular}']
write('TH_headline_times', tl, ['Problem', 'Mesh', 'Accurate', 'Fast', 'Accurate ms', 'Fast ms', 'FOM setting', 'FOM ms', 'Timing', 'Other scope or FOM: acc. / fast', 'Status'], tmd)

# appendix heat table (hires-heat lane): both error conventions, the batched variant, the tight named FOM, 3D speed rows
if HEAT_APPX:
    # split into an accuracy table and a timing table so each stays legible (2026-09-21 review)
    def _heat_table(name, cols, head, cells, mdhead):
        hl = [r'% GENERATED by paper/gen_headline.py from the hires-heat lane summaries -- do not edit.', r'\small',
              r'\begin{tabular}{@{}' + cols + r'@{}}', r'\toprule', head, r'\midrule']; hmd = []; seen = None
        for r in HEAT_APPX:
            c = [r['model'] if r['model'] != seen else '', mesh(r), f"{r['cohort']} ({r['cases']})", r['stepping']] + cells(r)
            seen = r['model']; hl.append(' & '.join(c) + r' \\'); hmd.append([md(x) for x in [r['model']] + c[1:]])
        hl += [r'\bottomrule', r'\end{tabular}']
        write(name, hl, mdhead, hmd)
    _heat_table('TH_heat_hires', 'lllllrr',
                r'Model & Mesh & Cohort & Stepping & $q$ acc./fast & Err.\ all times (\%) & Err.\ evolved (\%) \\',
                lambda r: [f"{r['accurate']['label'][3:-1]} / {r['fast']['label'][3:-1]}", f"{e(r['accurate']['error_pct'])} / {e(r['fast']['error_pct'])}",
                           f"{e(r['accurate']['evolved_pct'])} / {e(r['fast']['evolved_pct'])}"],
                ['Model', 'Mesh', 'Cohort (cases)', 'Stepping', 'Acc. / fast', 'Err. all times (%)', 'Err. evolved (%)'])
    _heat_table('TH_heat_hires_times', 'llllrrlrr',
                r'Model & Mesh & Cohort & Stepping & ms acc./fast & FOM ms & CN--CG FOM & Speedup & vs.\ $10^{-6}$ \\',
                lambda r: [f"{r['accurate']['ms']:.1f} / {r['fast']['ms']:.1f}", f"{r['fom']['ms']:.1f}", r['fom']['name'].replace('CN--CG, ', ''),
                           spn(r['accurate']['speedup']) + ' / ' + spn(r['fast']['speedup']) + r'$\times$', spn(r['tight']['accurate']) + ' / ' + spn(r['tight']['fast']) + r'$\times$'],
                ['Model', 'Mesh', 'Cohort (cases)', 'Stepping', 'ms acc. / fast', 'FOM ms', 'FOM (Table 1 rule)', 'Speedup', 'vs CN–CG 1e-6'])

# ---- Table 2: other nonlinear-manifold ROMs (nmrom-baselines lane, audited 256^2 job) ---------------------
NB = {}
NBP = {MAN[k]['role']: D[k] for k in MAN if MAN[k].get('adapter') == 'nmrom-baselines-v1'}
bl = [r'% GENERATED by paper/gen_headline.py from the nmrom-baselines lane (hash-pinned) -- do not edit.', r'\small',
      r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
      r'Method & Unknowns & Worst (\%) & Median (\%) & Memory (MB) \\', r'\midrule']
bmd = []; appx_nb = []
if NBP:
    import re as _re
    gate, ga = NBP['gate05'], NBP['gate05-audit']
    assert gate['gate']['passed'] and not gate['gate']['hr_passed'] and ga['all_agree'] and ga['log_has_gpu']
    assert gate['provenance']['backend'] == 'gpu' and gate['provenance']['matmul'] == 'highest'
    assert abs(ga['recomputed_nm_median'] - gate['gate']['nm_lspg_median']) < 1e-12
    assert gate['gate']['attempt'] == 3
    act = gate['gate']['recipe']['act']                                                              # admissible Kim rows use the gate activation
    T2 = []; NBJ = {}; NBDROP = {}; NBEP = {}
    for n in (256, 512):
        if f'fam{n}' not in NBP: continue
        fam, fa = NBP[f'fam{n}'], NBP[f'fam{n}-audit']; NBJ[n] = fam['job_id']
        assert fam['backend'] == 'gpu' and fam['x64'] and fam['matmul'] == 'highest' and fam['intervals'] == n
        assert all(g['passed'] for g in fam['gates'].values()) and fa['all_agree'] and fa['log_has_gpu']
        assert fam['gate_binding']['gate']['nm_lspg_median'] == gate['gate']['nm_lspg_median']      # bound to the passed gate
        for f in ('kimae.py', 'lspg.py'):                                                                # ... and the same code as the gate
            assert fam['source_sha256']['experiments/nmrom-baselines/' + f] == gate['source_sha256'][f]
        audited = {r['arm']: r for r in fa['rows']}
        A = fam['arms']
        kimk = (lambda K: f'kim_final_K{K}') if n == 256 else (lambda K: f'sig_K{K}_sel')
        plan = [('Kim et al. NM-LSPG', kimk(8)), ('Kim et al. NM-LSPG', kimk(16)), ('Kim et al. NM-LSPG', kimk(32))]
        if n == 256: plan.append(('Kim et al. NM-LSPG, data-matched', 'kim_final_K16_fit576'))
        plan += [('POD-LSPG', 'pod_lspg_zero_K8'), ('POD-LSPG', 'pod_lspg_zero_K16'), ('POD-LSPG', 'pod_lspg_zero_K32'),
                 ('This work, fast ($q=0$, $k=16$)', 'ours_q0'), ('This work, accurate ($q=256$, $k=16$)', 'ours_q256')]
        for label, arm in plan:
            a = A[arm]; assert a['cohort'] == 'validation' and a['finite'] and len(a['per_case_evolved']) == 32
            if a['family'].startswith('kim'): assert a['activation'] == act, arm
            ad = audited[arm]; assert ad['audited'] and ad['agree'] and abs(ad['recomputed'] / a['worst_evolved'] - 1) <= fa['tolerance'], arm
            assert max(a['per_case_evolved']) == a['worst_evolved']
            T2.append(dict(mesh=n, label=label, arm=arm, family=a['family'], k=a.get('solved_dimension', a['K']), worst=100 * a['worst_evolved'],
                           median=100 * a['median_evolved'], mb=a['memory_analysis']['total'] / 1e6, ms=a['timing']['gpu_ms_median']))
        for K in (8, 16, 32):
            hk = kimk(K) + '_hr'; a = A[hk]; assert a['finite'] and a['activation'] == act
            appx_nb.append(dict(mesh=n, label='Kim et al. NM-LSPG-HR (exploratory)', arm=hk, family='kim_hr', k=K, worst=100 * a['worst_evolved'],
                                median=100 * a['median_evolved'], mb=a['memory_analysis']['total'] / 1e6, ms=a['timing']['gpu_ms_median'], path='hyper-reduced, not reproduced'))
        drop = fam['dropped'][0]; assert drop['reason'] == 'exceeds_device_memory_precheck' and drop['M1'] == 2 * fam['n']
        NBDROP[n] = drop
        if n == 512:   # the capped encoder's training was cut by its wall budget
            tr_ = [fam['training'][kimk(K)] for K in (8, 16, 32)]
            assert all(t['stop_reason'] == 'wall_budget' and t['M1'] == 4096 for t in tr_)
            walls = {v['wall'] for v in fam['config']['variants'] if v.get('select', True)}; assert len(walls) == 1
            NBEP = dict(lo=min(t['epochs'] for t in tr_), hi=max(t['epochs'] for t in tr_), wall=walls.pop(), m1=4096)
    ours = [r for r in T2 if r['family'] == 'ours']; kim = [r for r in T2 if r['family'].startswith('kim')]; pod = [r for r in T2 if r['family'] == 'pod_lspg']
    for n in NBJ:                                                                   # the abstract's qualitative clause, per mesh
        assert max(r['worst'] for r in ours if r['mesh'] == n) < min(r['worst'] for r in kim + pod if r['mesh'] == n)
    meshes = sorted(NBJ)
    keys = []
    for r in T2:
        if (r['label'], r['k'], r['family']) not in keys: keys.append((r['label'], r['k'], r['family']))
    cellv = {(r['label'], r['k'], r['mesh']): r for r in T2}
    bl[2] = r'\begin{tabular}{@{}lr' + 'rrr' * len(meshes) + r'@{}}'
    bl[4] = (r' & & ' + ' & '.join(r'\multicolumn{3}{c}{$' + str(n) + r'^2$}' for n in meshes) + r' \\' + '\n'
             + ''.join(r'\cmidrule(lr){' + f'{3 + 3 * i}-{5 + 3 * i}' + '}' for i in range(len(meshes))) + '\n'
             + r'Method & Unknowns & ' + ' & '.join([r'Worst (\%) & Median (\%) & MB'] * len(meshes)) + r' \\')
    seen = None
    for lab, k, fam_ in keys:
        if seen and fam_ != seen: bl.append(r'\addlinespace')
        seen = fam_
        c = [lab, str(k)]
        for n in meshes:
            r = cellv.get((lab, k, n))
            c += [e(r['worst']), e(r['median']), f"{r['mb']:.0f}"] if r else ['---', '---', '---']
        bl.append(' & '.join(c) + r' \\'); bmd.append([md(x) for x in c])
    appx_nb[:0] = [dict(r, path='dense (reference path)' if r['family'] == 'ours' else 'dense') for r in T2]
    appx_nb.sort(key=lambda r: (r['mesh'], r['family'] == 'kim_hr'))
    rng0 = lambda v: f"{min(v):.0f}\\mbox{{--}}{max(v):.0f}"
    rng2 = lambda v: e(min(v)) + r'\mbox{--}' + e(max(v))
    NB.update(nBaseOursFastWorst=rng2([r['worst'] for r in ours if r['arm'] == 'ours_q0']),
              nBaseOursAccWorst=rng2([r['worst'] for r in ours if r['arm'] == 'ours_q256']),
              nBaseKimRange=rng0([r['worst'] for r in kim]), nBasePodRange=rng0([r['worst'] for r in pod]),
              nBaseGateMedian=e(100 * gate['gate']['nm_lspg_median']), nBaseGateAttempt=str(gate['gate']['attempt']),
              nBaseGateBar=_re.search(r'<= ([\d.]+) %', gate['gate']['rule']).group(1),
              nBaseGatePublished=_re.match(r'<(\d+) %', gate['published']['nm_lspg']).group(1),
              nBaseEncoderNeedGB=f"{NBDROP[256]['need_gb']:.0f}", nBaseDeviceGB=f"{NBDROP[256]['device_limit_gb']:.0f}",
              nBaseDataMatched=str(NBP['fam256']['training']['kim_final_K16_fit576']['fit_trajectories']), nBaseGateAct=act)
    if 512 in NBDROP:
        NB.update(nBaseEncoderNeedGBFiveTwelve=f"{NBDROP[512]['need_gb']:.0f}", nBaseDeviceGBFiveTwelve=f"{NBDROP[512]['device_limit_gb']:.0f}",
                  nBaseKimEpochs=f"{NBEP['lo']}\\mbox{{--}}{NBEP['hi']}", nBaseKimWall=str(NBEP['wall']), nBaseKimWidth=str(NBEP['m1']))
    assert 'sigmoid' == act
else:
    for m in ('This work (accurate / fast)', 'Convolutional-autoencoder NM-ROM', 'Shallow-masked-autoencoder NM-ROM'):
        bl.append(f'{m} & --- & --- & --- & --- \\\\')
bl += [r'\bottomrule', r'\end{tabular}']
write('TH_nmrom_baselines', bl, ['Method', 'Unknowns'] + [f'{n}^2 {h}' for n in (sorted(NBJ) if NBP else []) for h in ('worst (%)', 'median (%)', 'MB')], bmd)
if appx_nb:
    al = [r'% GENERATED by paper/gen_headline.py -- do not edit.', r'\small', r'\begin{tabular}{@{}lllrrrrr@{}}', r'\toprule',
          r'Mesh & Method & Residual path & $k$ & Worst (\%) & Median (\%) & GPU ms (same job) & MB \\', r'\midrule']
    amd = []
    for r in appx_nb:
        c = [f"${r['mesh']}^2$", r['label'], r['path'], str(r['k']), e(r['worst']), e(r['median']), f"{r['ms']:.0f}", f"{r['mb']:.0f}"]
        al.append(' & '.join(c) + r' \\'); amd.append([md(x) for x in c])
    al += [r'\bottomrule', r'\end{tabular}']
    write('TH_nmrom_baselines_appx', al, ['Mesh', 'Method', 'Residual path', 'k', 'Worst (%)', 'Median (%)', 'GPU ms (this job)', 'MB'], amd)

# ---- where the method currently fails ------------------------------------------------------------------
F = []
B = {r['method']: r for r in D['burgers3d']['rows']}
assert D['burgers3d']['final_evaluation']['complete']['complete']
bf = B['fom_n33_dt0.01_nt1e-02_lt5e-01']
_bn = D['burgers3d']['final_evaluation']['physical_reference']['protocol']['same_grid']['nodes']
_lv = [l['nodes'] for l in D['burgers3d']['final_evaluation']['physical_reference']['protocol']['physical_reference']['levels']]
assert all((x - 1) % (_bn - 1) == 0 for x in _lv)          # 33 -> 65 -> 129: nodes include both walls, so intervals = nodes - 1
B3N = _bn - 1
for m, lab_ in (('rom_q0', '$q=0$'), ('rom_q192', '$q=192$')):
    F.append(dict(problem=f'Burgers 3D, ${B3N}^3$ (final)', setting=lab_, error_pct=100 * B[m]['worst_evolved'], fom_error_pct=100 * bf['worst_evolved'],
                  speedup=bf['median_ms'] / B[m]['median_ms'], fom='Newton--BiCGStab', metric='same-grid, evolved', source='burgers3d'))
N = {r['method']: r for r in D['ns3d']['rows']}; nf = N['fom_dt0.01']; assert D['ns3d']['evaluation_cohort'] == 'final'
for m, lab_ in (('nmrom_pca64_free_q0', '$q=0$'), ('nmrom_pca64_free_q256', '$q=256$')):
    assert N[m]['development_selected_fom'] == 'fom_dt0.01'
    F.append(dict(problem='Navier--Stokes 3D, $32^3$ (final)', setting=lab_, error_pct=N[m]['same_grid_evolved_worst_percent'], fom_error_pct=nf['same_grid_evolved_worst_percent'],
                  speedup=nf['gpu_ms'] / N[m]['gpu_ms'], fom='CNAB2', metric='same-grid, evolved', source='ns3d',
                  failing=N[m]['physical_target_failing_cases'], cases=N[m]['cases']))
W = defaultdict(dict)
for r in D['wave']:
    if r.get('mesh') == 1024: W[r['subject']][r['metric']] = r['value']
wf = min((k for k, v in W.items() if k.startswith('cg') and v.get('all_state_pass') is True), key=lambda k: W[k]['median_gpu_ms'])
for m, lab_ in (('head_q0', '$q=0$'), ('nested_q32', '$q=32$')):
    assert W[m]['all_state_pass'] is False
    F.append(dict(problem='Wave 2D, $1024^2$ (dev.)', setting=lab_, error_pct=100 * W[m]['worst_energy_state'], fom_error_pct=100 * W[wf]['worst_energy_state'],
                  speedup=W[wf]['median_gpu_ms'] / W[m]['median_gpu_ms'], fom='midpoint--CG', fom_arm=wf, metric='energy-state (displacement and velocity)', source='wave',
                  displacement_pct=100 * W[m]['worst_current_displacement']))
# 2026-09-22 user decision: Heat 3D (new bank) meets its target and is in Table 1; the earlier Heat 3D model leaves the failures table.
h128 = [r for r in HEAT_APPX if r['dim'] == 3 and r['intervals'] == 128 and r['stepping'] == 'CN']
assert len(h128) == 1; h128 = h128[0]
fl = [r'% GENERATED by paper/gen_headline.py -- do not edit.', r'\small', r'\begin{tabular}{@{}llrrrl@{}}', r'\toprule',
      r'Problem & Setting & Err.\ (\%) & FOM err.\ (\%) & Speedup & FOM \\', r'\midrule']
fmd = []; seen = None
for r in F:
    fe = e(r['fom_error_pct']) if r['fom_error_pct'] is not None else '---'
    c = [r['problem'] if r['problem'] != seen else '', r['setting'], e(r['error_pct']), fe, sp(r['speedup']), r['fom']]; seen = r['problem']
    fl.append(' & '.join(c) + r' \\'); fmd.append([r['problem'], r['setting']] + [md(x) for x in c[2:]])
fl += [r'\bottomrule', r'\end{tabular}']
write('TH_failures', fl, ['Problem', 'Setting', 'NM-ROM err. (%)', 'FOM err. (%)', 'Speedup', 'FOM'], fmd)

# prose macros used by abstract / results: only values present in the generated headline table
def best(pred, key):
    c = [r for r in ROWS if pred(r) and r[key] and not r['source'].startswith('intake_')]   # lane rows stay out of abstract/conclusion macros
    return max(c, key=lambda r: r[key]['speedup'])
mac = {}
pa = best(lambda r: r['problem'] == 'Poisson' and r['dim'] == 2, 'accurate')
mac['nHeadPoissonAccErr'] = e(pa['accurate']['error_pct']); mac['nHeadPoissonAccS'] = spn(pa['accurate']['speedup'])
mac['nHeadPoissonFastS'] = spn(pa['fast']['speedup']); mac['nHeadPoissonMesh'] = f"{pa['intervals']}^2"
la = best(lambda r: r['problem'] == 'Poisson, L-shape', 'accurate'); mac['nHeadLshapeAccS'] = spn(la['accurate']['speedup']); mac['nHeadLshapeAccErr'] = e(la['accurate']['error_pct'])
ha = best(lambda r: r['problem'] == 'Heat' and r['dim'] == 2, 'fast'); mac['nHeadHeatFastS'] = spn(ha['fast']['speedup']); mac['nHeadHeatFastErr'] = e(ha['fast']['error_pct'])
bb = [r for r in ROWS if r['problem'] == 'Burgers']
b1024 = [r for r in bb if r['intervals'] == 1024 and not r['source'].startswith('intake_')][0]; b256 = [r for r in bb if r['intervals'] == 256][0]
mac['nHeadBurgersFastS'] = spn(b1024['fast']['speedup']); mac['nHeadBurgersFastErr'] = e(b1024['fast']['error_pct'])
mac['nHeadBurgersAccErr'] = e(b256['accurate']['error_pct']); mac['nHeadBurgersAccS'] = spn(b256['accurate']['speedup'])
p3a = best(lambda r: r['problem'] == 'Poisson' and r['dim'] == 3, 'accurate'); mac['nHeadPoissonThreeAccS'] = spn(p3a['accurate']['speedup']); mac['nHeadPoissonThreeAccErr'] = e(p3a['accurate']['error_pct'])
bq = P[(1024, 'q128_M576_eqxfer_g1em06')]; assert bq['admissible'] and bq['converged_design5']
mac['nHeadBurgersMidErr'] = e(bq['worst_evolved_percent']); mac['nHeadBurgersMidS'] = spn(b1024['fom']['ms'] / bq['median_gpu_ms'])
# Heat 3D new bank (Table 1 rows 'Heat (new bank)', panel A) and the NS follow-up diagnosis
_h3r = {(r['problem'], r['intervals']): r for r in ROWS if r['dim'] == 3 and r['problem'].startswith('Heat (new bank')}
mac['nHeatNewAccErr'] = e(max(r['accurate']['error_pct'] for r in _h3r.values()))
mac['nHeatNewAccEvolved'] = e(max(r['accurate']['evolved_pct'] for (p_, n_), r in _h3r.items() if p_ == 'Heat (new bank)'))
mac['nHeatNewBatchedAccEvolved'] = e(max(r['accurate']['evolved_pct'] for (p_, n_), r in _h3r.items() if p_ != 'Heat (new bank)'))
mac['nHeatNewCnAccS'] = spn(_h3r[('Heat (new bank)', 128)]['accurate']['speedup']); mac['nHeatNewBfAccS'] = spn(_h3r[('Heat (new bank, batched fit)', 128)]['accurate']['speedup'])
mac['nHeatNewNonstat'] = str(sum(r['accurate']['nonstationary'] for r in _h3r.values()))
_hc = {r_['cases'] for m in D['heat3db_panel_a']['meshes'] if m['cohort'].startswith('sealed') for r_ in m['rows']}; assert len(_hc) == 1; mac['nHeatNewCases'] = str(_hc.pop()); mac['nHeatNewK'] = str(_sel['K']); mac['nHeatNewQ'] = str(_sel['accurate_q']); mac['nHeatNewR'] = str(_sel['R']); mac['nHeatNewFloor'] = f"{H3N['floor']:.2f}"
mac['nHeatNewLinErr'] = e(H3N[128]['lin_err'][1]); mac['nHeatNewLinMs'] = f"{H3N[128]['lin_ms'][0]:.1f}\\mbox{{--}}{H3N[128]['lin_ms'][1]:.1f}"
mac['nHeatNewBR'], mac['nHeatNewBK'], mac['nHeatNewBQ'] = (str(_c2[k_]) for k_ in ('R', 'K', 'accurate_q')); mac['nHeatNewBErr'] = e(H3N['b']['err']); mac['nHeatNewBCnS'] = spn(H3N['b']['cn']); mac['nHeatNewBBfS'] = spn(H3N['b']['bf'])
mac['nNsGrokWorst'] = e(NSG['worst']); mac['nNsGrokOver'] = str(NSG['over']); mac['nNsGrokCases'] = str(NSG['cases']); mac['nNsGrokS'] = spn(NSG['speedup'])
assert f"{NSG['target']:g}" == mac.get('nFailNsTarget', f"{NSG['target']:g}")
FB = {(r['source'], r['setting']): r for r in F}
mac['nFailBurgersQzero'] = e(FB[('burgers3d', '$q=0$')]['error_pct']); mac['nFailBurgersAcc'] = e(FB[('burgers3d', '$q=192$')]['error_pct'])
mac['nFailNsQzero'] = e(FB[('ns3d', '$q=0$')]['error_pct']); mac['nFailNsAcc'] = e(FB[('ns3d', '$q=256$')]['error_pct'])
mac['nFailNsTarget'] = f"{D['ns3d']['target_percent']:g}"; mac['nFailNsFailing'] = str(FB[('ns3d', '$q=256$')]['failing']); mac['nFailNsCases'] = str(FB[('ns3d', '$q=256$')]['cases'])
mac['nFailWaveQzero'] = e(FB[('wave', '$q=0$')]['error_pct']); mac['nFailWaveAcc'] = e(FB[('wave', '$q=32$')]['error_pct'])
if HP:
    def rng(vals): return spn(min(vals)) + r'\mbox{--}' + spn(max(vals))       # en dash that survives math mode
    sq = {k: v for k, v in HP.items() if k[0].startswith('Poisson') and ',' not in k[0]}          # square and cube
    mac['nHiresCtlTotal'] = rng([c['total'] for v in sq.values() for c in v['controls'].values()])
    mac['nHiresCtlDevice'] = rng([c['device'] for v in sq.values() for c in v['controls'].values()])
    ls = {k: v for k, v in HP.items() if k[0] == 'Poisson, L-shape'}
    mac['nHiresLshapeCoarse'] = rng([v['controls']['fastest_coarse_matched']['total'] for v in ls.values()])
    # the same controls as "control is X times faster than the NM-ROM" (reciprocals; every value must be < 1 for that wording)
    ctl_t = [c['total'] for v in sq.values() for c in v['controls'].values()]; ctl_d = [c['device'] for v in sq.values() for c in v['controls'].values()]
    ctl_l = [v['controls']['fastest_coarse_matched']['total'] for v in ls.values()]
    assert max(ctl_t + ctl_d + ctl_l) < 1
    mac['nHiresCtlTotalFaster'] = rng([1 / x for x in ctl_t]); mac['nHiresCtlDeviceFaster'] = rng([1 / x for x in ctl_d])
    mac['nHiresLshapeCoarseFaster'] = rng([1 / x for x in ctl_l])
    mac['nHiresLshapeAccErr'] = e(max(v['acc_err'] for v in ls.values())); mac['nHiresLshapeFloor'] = e(max(v['floor_pct'] for v in ls.values()))
    mac['nHiresFloorSquare'] = f"{max(v['floor_pct'] for k, v in sq.items() if k[0] == 'Poisson'):.3f}"
    mac['nHiresFloorCube'] = f"{max(v['floor_pct'] for k, v in sq.items() if k[0] != 'Poisson'):.3f}"
    for r in ROWS:
        if r['source'].startswith('intake_') and r['problem'] == 'Poisson':
            w = {2048: 'TwentyFortyEight', 4096: 'FortyNinetySix'}[r['intervals']]
            mac['nHiresPoissonAccS' + w] = spn(r['accurate']['speedup']); mac['nHiresPoissonAccTotalS' + w] = spn(r['alt']['accurate'])
if HH:
    W = {(r['problem'], r['intervals']): r for r in ROWS if r['problem'].startswith('Heat (wide bank')}
    cn4, bf4 = W[('Heat (wide bank)', 4096)], W[('Heat (wide bank, batched fit)', 4096)]
    assert cn4['accurate']['error_pct'] == max(W[k]['accurate']['error_pct'] for k in W if k[1] > 1024)   # one sealed value at both meshes
    mac['nHeatWideAccErr'] = e(cn4['accurate']['error_pct']); mac['nHeatWideAccEvolved'] = e(cn4['accurate']['evolved_pct'])
    mac['nHeatWideBatchedAccEvolved'] = e(bf4['accurate']['evolved_pct']); mac['nHeatWideAccS'] = spn(cn4['accurate']['speedup'])
    mac['nHeatWideBatchedAccS'] = spn(bf4['accurate']['speedup']); mac['nHeatWideFloor'] = e(HH['wide_floor'])
    mac['nHeatWideTightS'] = spn(cn4['alt']['accurate']); mac['nHeatWideBatchedTightS'] = spn(bf4['alt']['accurate'])
    mac['nHeatCoarseErr'] = e(HH['coarse_err']); mac['nHeatCoarseMs'] = f"{HH['coarse_ms']:.1f}"; mac['nHeatDstMs'] = f"{HH['dst_ms']:.1f}"
    mac['nHeatLinErr'] = e(HH['lin_err']); mac['nHeatLinMs'] = f"{HH['lin_ms'][0]:.1f}\\mbox{{--}}{HH['lin_ms'][1]:.1f}"; mac['nHeatNmromMinMs'] = f"{HH['nmrom_min_ms']:.1f}"
    mac['nHeatThreeAllTimes'] = e(HH['h3d_all']); mac['nHeatThreeEvolvedCheck'] = e(HH['h3d_evolved'])
    hc = [HH['coarse_ms'], HH['dst_ms']] + list(HH['lin_ms'])
    mac['nHeatCtlFaster'] = f"{HH['nmrom_min_ms'] / max(hc):.1f}\\mbox{{--}}{HH['nmrom_min_ms'] / min(hc):.1f}"
    mac['nHeatThreeInit'] = f"{HH['h3d_init'][0]:.1f}\\mbox{{--}}{HH['h3d_init'][1]:.1f}"
if HB:
    mac['nBurgHostGapMs'] = f"{HB['host_gap_4096']:.0f}"
    mac['nBurgMFiveFourFourHold'] = e(HB['m544_hold_4096'])
    for att, w in (('hb2k02', 'TwentyFortyEight'), ('hb4k04', 'FortyNinetySix')):
        mac['nBurgCoarseErr' + w] = e(HB[att]['coarse_err']); mac['nBurgCoarseMs' + w] = f"{HB[att]['coarse_ms']:.0f}"
        mac['nBurgAccMs' + w] = f"{HB[att]['acc_ms']:.0f}"
    mac['nBurgBankFloorConfirm'] = e(HB['bank_floor_confirm'])
    bf_ = [HB[a]['acc_ms'] / HB[a]['coarse_ms'] for a in ('hb2k02', 'hb4k04')]
    mac['nBurgCoarseFaster'] = f"{min(bf_):.1f}\\mbox{{--}}{max(bf_):.1f}"
    for att, w in (('hb2kh64', 'TwentyFortyEight'), ('hb4kh64', 'FortyNinetySix')):
        g = HB[(att, 'gate')]; mac['nBurgGateBad' + w] = str(g['bad']); mac['nBurgGateRows' + w] = str(g['rows'])
    mac['nBurgGateWorstPct'] = f"{100 * max(HB[(a, 'gate')]['worst_gap'] for a in ('hb2kh64', 'hb4kh64')):.1f}"
    H = {(r['problem'], r['intervals']): r for r in ROWS if r['problem'].startswith('Burgers') and r['intervals'] >= 2048}
    mac['nBurgHoldAccErrFortyNinetySix'] = e(H[('Burgers (held-out cases)', 4096)]['accurate']['error_pct'])
    mac['nBurgDevAccErrFortyNinetySix'] = e(H[('Burgers', 4096)]['accurate']['error_pct'])
    mac['nBurgDevAccSFortyNinetySix'] = spn(H[('Burgers', 4096)]['accurate']['speedup'])
    mac['nBurgHoldAccSFortyNinetySix'] = spn(H[('Burgers (held-out cases)', 4096)]['accurate']['speedup'])
if BH5:
    mac['nBhJzeroRho'] = f"{BH5['j0_rho']:.4f}"; mac['nBhJzeroRhoKtwo'] = f"{BH5['j0_rho_k2']:.4f}"; mac['nBhJoneRho'] = f"{BH5['j1_rho']:.4f}"
    mac['nBhFastS'] = BH5['fast_s']; mac['nBhFastFom'] = BH5['fast_fom']
    mac['nBhJoneMsDev'] = f"{BH5['dev6']['j1_ms']:.0f}"; mac['nBhJoneMsHold'] = f"{BH5['hold64']['j1_ms']:.0f}"
    mac['nBhJoneSDev'] = f"{BH5['dev6']['sp']:.2f}"; mac['nBhJoneSHold'] = f"{BH5['hold64']['sp']:.2f}"
    mac['nBhJoneCostDev'] = spn(BH5['dev6']['cost']); mac['nBhJoneCostHold'] = spn(BH5['hold64']['cost'])
    mac['nBhFloorOld'] = f"{BH5['floor_old']:.3f}"; mac['nBhFloorNew'] = f"{BH5['floor_new']:.3f}"
    mac['nBhBankErr'] = f"{BH5['bank_err']:.3f}"; mac['nBhBankS'] = spn(BH5['bank_s'])
    mac['nBhSubErrLo'] = f"{BH5['sub_err'][0]:.3f}"; mac['nBhSubErrHi'] = f"{BH5['sub_err'][1]:.3f}"
    mac['nBhSubSLo'] = spn(BH5['sub_s'][0]); mac['nBhSubSHi'] = spn(BH5['sub_s'][1])
    for cohort, coh in (('dev6', 'development'), ('hold64', 'held-out')):
        rr = [r for r in APPX if r['problem'] == 'Burgers, exact first step' and r['cohort'] == coh][0]
        assert abs(rr['accurate']['speedup'] - BH5[cohort]['sp']) < 1e-12 and rr['accurate']['speedup'] < 1
def _sci1(x):
    m_, ex = f'{x:.0e}'.split('e'); return f'{m_}{{\\times}}10^{{{int(ex)}}}'
mac['nEqcBar'] = f"{EQC['bar']:g}"; mac['nEqcLatConfRho'] = f"{EQC['lat2048']:.4f}"; mac['nEqcLatMargin'] = _sci1(EQC['bar'] - EQC['lat2048'])
assert f"{EQC['bar']:g}" == '0.116'                                                       # the same primary bar as \nEqtopBar
mac['nEqcRowRuleMFiveTwelve'] = str(EQC['identity'][512]['row']['m']); mac['nEqcLaneRuleM'] = str(EQC['identity'][512]['lane']['m'])
mac['nEqcScaledPassFiveTwelve'] = str(EQC['s512'][0]); mac['nEqcScaledDrawsFiveTwelve'] = str(EQC['s512'][1])
mac['nEqcConfRhoTenTwentyFour'] = f"{EQC[1024]['conf_rho']:.4f}"
mac['nEqcFollowDraws'] = str(EQC['follow']['draws']); mac['nEqcFollowErr'] = f"{EQC['follow']['err']:.3f}"; mac['nEqcFollowS'] = f"{EQC['follow']['sp']:.3f}"
mac['nHeadFasterRows'] = str(sum(1 for r in ROWS if any(r[s] and r[s]['speedup'] > 1 for s in ('fast', 'accurate'))))
mac['nHeadRows'] = str(len(ROWS))
mac['nHeadAccFasterSubOne'] = str(sum(1 for r in ROWS if r['accurate'] and r['accurate']['speedup'] > 1 and r['accurate']['error_pct'] < 1))
mac.update(NB)
(HERE / 'tables/headline-numbers.tex').write_text('% GENERATED by paper/gen_headline.py -- do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(mac.items())))

# ---- compact 3D configuration table (values read from the run records) ---------------------------------------
cfg = []
b3 = D['burgers3d']
cfg.append(['Burgers 3D', f'${B3N}^3$', str(B['rom_q192']['cases']) + ' final', '0, 192', 'Newton--BiCGStab, $\\Delta t=0.01$', 'dense'])
cfg.append(['Poisson 3D', '$32^3$, $64^3$', str(pr[(64, 'nmrom_K16_q96_dense')]['cases']) + ' final', '0, 32, 96 ($k=16$)', 'CG, rtol $10^{-2}$, no preconditioner', 'dense'])
cfg.append(['Heat 3D (new bank)', '$32^3$, $64^3$, $128^3$', '64 sealed final (all times)', '0, 288 ($k=32$, $R=320$)', 'CN--CG, setting per row by the rule; named $\\Delta t=0.025$, rtol $10^{-6}$', 'exact (linear)'])
ns = D['ns3d']
cfg.append(['Navier--Stokes 3D', f"${ns['n']}^3$ periodic", f"{ns['cohort_count']} final", ', '.join(str(q) for q in ns['q_values']) + f" ($k={ns['k']}$, $R={ns['r']}$, $M={ns['test_modes']}$)", 'CNAB2, $\\Delta t=0.01$', 'dense'])
cl = [r'% GENERATED by paper/gen_headline.py -- do not edit.', r'\scriptsize', r'\begin{tabular}{@{}lllp{3.3cm}p{3.6cm}l@{}}', r'\toprule',
      r'Problem & Mesh & Cases & Correction ranks & Named FOM & Residual \\', r'\midrule'] + [' & '.join(c) + r' \\' for c in cfg] + [r'\bottomrule', r'\end{tabular}']
write('TH_config3d', cl, ['Problem', 'Mesh', 'Cases', 'Correction ranks', 'Named FOM', 'Residual'], [[md(x) for x in c] for c in cfg])

# allocations, moved out of the tables into one reproducibility paragraph
def _jobs(rows):
    out = defaultdict(list)
    for r in rows:
        k = f"{r['problem']} {r['dim']}D"
        if r['job_id'] not in out[k]: out[k].append(r['job_id'])
    return '; '.join(f"{k} {', '.join(v)}" for k, v in out.items())
_srt = sorted(ROWS + APPX, key=lambda r: (r['dim'], ORDER.index(r['problem']), r['intervals']))
jp = [r'% GENERATED by paper/gen_headline.py -- do not edit.',
      r'\paragraph{Allocations.} Every ratio pairs times from one Slurm allocation. Tables~\ref{tab:headline} and~\ref{tab:headline-times}: ' + _jobs(_srt) + '. '
      + r'Table~\ref{tab:failures}: Burgers 3D ' + str(D['burgers3d']['job_id']) + '; Navier--Stokes 3D ' + str(ns['job_id']) + '; Wave 2D ' + str(next(iter({r['job_id'] for r in D['wave'] if r.get('mesh') == 1024}))) + '. '
      + r'Table~\ref{tab:config3d}: Burgers 3D ' + str(b3['job_id']) + '; Poisson 3D ' + str(p3['job_id']) + '; Heat 3D ' + str(D['heat3db_panel_a']['metadata']['job_id']) + '; Navier--Stokes 3D ' + str(ns['job_id']) + '. '
      + (r'Table~\ref{tab:heat-hires}: ' + ', '.join(sorted({r['job'] for r in HEAT_APPX})) + '. ' if HEAT_APPX else '')
      + (r'Tables~\ref{tab:nmrom-baselines} and~\ref{tab:nmrom-baselines-appx}: ' + ', '.join(f"${n}^2$ {j}" for n, j in NBJ.items()) + r' (reproduction gate: ' + str(NBP['gate05']['provenance']['job_id']) + ').' if NBP else '')]
# train/evaluation disjointness of the Burgers checkpoint (read-only regeneration check, output snapshotted in evidence/)
_ov = HERE / 'evidence/burgers-train-eval-overlap-2026-09-21'
if (_ov / 'result.json').exists():
    ov = json.loads((_ov / 'result.json').read_text())
    assert digest((_ov / 'check.py').read_bytes()) == ov['script_sha256'] and digest((_ov / 'output.txt').read_bytes()) == ov['output_sha256']
    assert ov['verdict'] == 'disjoint' and all(c['exact'] == c['allclose'] == c['shared_scalars'] == 0 for c in ov['cohorts'].values())
    names = {'val32': 'the 32 validation cases of Table~\\ref{tab:nmrom-baselines}', 'hold64': 'the 64 held-out cases',
             'dev6': 'the 6 development cases', 'sealed6': 'the sealed cohort'}
    jp.append(r'\paragraph{Training/evaluation separation (Burgers).} The ' + str(ov['training_trajectories'])
              + r' training trajectories of the Burgers checkpoint were regenerated from their seeds and compared with every evaluation cohort ('
              + ', '.join(names[k] for k in ('val32', 'hold64', 'dev6', 'sealed6'))
              + r'): no cohort shares a parameter vector, or any single parameter value, with the training set (check script and output in \texttt{evidence/burgers-train-eval-overlap-2026-09-21}).')
(HERE / 'tables/TH_jobs.tex').write_text('\n'.join(jp) + '\n')
(HERE / 'tables/TH_training.tex').write_text('% GENERATED by paper/gen_headline.py from the wide bank training record -- do not edit.\n' + HH.get('training_sentence', '') + '\n')

# ---- Figure 2: correction-rank tunability at 4096^2 (one frozen model and one allocation per series) -------------
# Denominator: ONE full-order setting per series, chosen by the paper's rule applied to the series' most accurate
# rung (fastest same-grid setting of the named solver, same cohort, at least as accurate), so each curve is a pure
# ROM-side tradeoff.  Every speedup is FOM ms / rung ms from the same record.
TUNE = []
def _tune_series(name, key, cohort_lab, rungs, foms, style):
    best = min(r['err'] for r in rungs)
    ok = {k: v for k, v in foms.items() if v['err'] <= best}
    c = min(ok, key=lambda k: ok[k]['ms']); fm = foms[c]
    for r in rungs: r['speedup'] = fm['ms'] / r['ms']
    TUNE.append(dict(series=name, source=key, cohort=cohort_lab, fom=dict(arm=c, **fm), rungs=rungs, style=style))
_kb = {v.get('role'): k for k, v in MAN.items() if v.get('adapter') == 'hires-burgers-v1'}
for role, lab, style in (('hb4k04', 'development (6)', 'filled'), ('hb4kh64', 'held-out (64)', 'hollow')):
    d = D[_kb[role]]; t = d['table']; assert d['intervals'] == 4096
    cert = {arm: dep['certified_primary'] for r_ in d['rules'] for arm, dep in r_['deployed'].items()}
    rung_arms = [('q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2', 'rank'), ('q128_M576_lat64_g0p001_fast_chol_clip_lamcarry_pred2', 'rank'),
                 ('q256_M544_lat64_g0p001_fast_chol_clip_lamcarry_pred2', 'rank'), ('q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2', 'rank'),
                 ('q256_M1088_lat64_g0p01_fast_chol_clip_lamcarry_pred2', 'tolerance'), ('q256_M2176_lat64_g0p001_fast_chol_clip_lamcarry_pred2', 'rank')]
    rungs = []
    for arm, kind in rung_arms:
        a = t[arm]; assert a['stalled_exits'] == 0 and a['cases'] == d['cohort_cases']
        rungs.append(dict(arm=arm, q=a['q'], M=a['M'], kind=kind, err=a['worst_evolved_percent'], ms=a['median_gpu_ms'], certified=bool(cert[arm])))
    foms = {k: dict(err=v['worst_evolved_percent'], ms=v['median_gpu_ms']) for k, v in t.items()
            if v['family'] == 'fom' and v['mesh'] == 4096 and v['nonlinear_converged'] and v['stalled_steps'] == 0 and v['cases'] == d['cohort_cases']}
    _tune_series('Burgers 2D, ' + lab, _kb[role], lab, rungs, foms, style)
    TUNE[-1]['fom']['name'] = 'Newton--BiCGStab'
_kh = {v.get('role'): k for k, v in MAN.items() if v.get('adapter') == 'hires-heat-v1'}
_R = {m['intervals']: {r['method']: r for r in m['rows']} for m in D[_kh['h2d-final04']]['meshes'] if m['cohort'] == 'sealed_opened_once'}[4096]
_hr = [dict(arm=a, q=int(a.split('_q')[1].split('_')[0]), kind='rank', err=100 * _R[a]['error_evolved_worst'], ms=_R[a]['device_ms_median'], certified=True)
       for a in ('nmrom_q0_cn', 'nmrom_q32_cn', 'nmrom_q96_cn')]
assert all(_R[r['arm']]['failures'] == 0 for r in _hr)
_tune_series('Heat 2D (wide bank), sealed (16)', _kh['h2d-final04'], 'sealed (16)', _hr,
             {k: dict(err=100 * v['error_evolved_worst'], ms=v['device_ms_median']) for k, v in _R.items() if k.startswith('fom_cncg_') and v['failures'] == 0}, 'filled')
TUNE[-1]['fom']['name'] = 'CN--CG'
_kp = [k for k, v in MAN.items() if v.get('adapter') == 'hires-poisson-v1'][0]
_P = {r['subject']: r for r in D[_kp]['rows'] if r['attempt'] == 'hp4096' and r['mesh'] == 'square 4096²'}
_pr = [dict(arm=f'rom_q{q}_lean64', q=q, kind='rank', err=100 * _P[f'rom_q{q}_lean64']['worst_same_grid'], ms=_P[f'rom_q{q}_lean64']['median_device_ms'], certified=True)
       for q in (0, 64, 128, 256)]
_nc = _P['rom_q256_lean64']['cases']
_tune_series('Poisson 2D, development (12)', _kp, 'development (12)', _pr,
             {k: dict(err=100 * v['worst_same_grid'], ms=v['median_device_ms']) for k, v in _P.items() if v['family'] == 'cg' and not k.startswith('coarse') and v['cases'] == _nc}, 'filled')
TUNE[-1]['fom']['name'] = 'CG'
for tt_ in TUNE:
    mac_key = {'Burgers 2D, development (6)': 'BurgDev', 'Burgers 2D, held-out (64)': 'BurgHold', 'Heat 2D (wide bank), sealed (16)': 'Heat', 'Poisson 2D, development (12)': 'Poisson'}[tt_['series']]
    sp_ = [r['speedup'] for r in tt_['rungs'] if r['kind'] == 'rank' and r['certified']]
    mac[f'nTune{mac_key}SpeedMin'] = spn(min(sp_)); mac[f'nTune{mac_key}SpeedMax'] = spn(max(sp_))
    mac[f'nTune{mac_key}FomTol'] = tt_['fom']['arm']
def _tol(x):
    m_, ex = f'{float(x):.0e}'.split('e'); ex = int(ex)
    return f'10^{{{ex}}}' if m_ == '1' else f'{m_}{{\\times}}10^{{{ex}}}'
for tt_ in TUNE:
    a = tt_['fom']['arm']; k = {'Newton--BiCGStab': 'Burg', 'CN--CG': 'Heat', 'CG': 'Poisson'}[tt_['fom']['name']]
    if a.startswith('lean_nt'): lab = 'tolerance $' + _tol(a.split('_nt')[1].split('_')[0]) + '$'
    elif a.startswith('fom_cncg'): lab = '$\\Delta t=' + a.split('_dt')[1].split('_')[0] + '$, rtol $' + _tol(a.split('rtol')[1].split('_')[0]) + '$'
    else: lab = 'rtol $' + _tol(a.split('_')[1]) + '$'
    mac[f'nTune{k}FomLabel'] = lab
bdev = TUNE[0]; assert all(r['speedup'] > 1 for r in bdev['rungs'])
(HERE / 'tables/tunability-numbers.tex').write_text('% GENERATED by paper/gen_headline.py -- do not edit.\n' + ''.join(
    f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(mac.items()) if k.startswith('nTune') and not k.endswith('FomTol')))
assert all(r['speedup'] > 1 for t_ in TUNE for r in t_['rungs'])      # every rung of every series is faster than its FOM setting

(HERE / 'tables/headline-provenance.json').write_text(json.dumps(dict(
    rule='One frozen model per row. fast = q=0; accurate = largest stored correction rank at the standard time step (Burgers: fastest / lowest-error admissible residual evaluation at that rank). '
         'One named FOM per row from the same allocation, at least as accurate as the accurate setting; speedup = FOM ms / NM-ROM ms. '
         'Exception: the 4096^2 Burgers fast column divides the fastest tested same-allocation setting at least as accurate as that fast setting (bh5 grid, same job). Bold = speedup > 1.',
    sources=MAN, rows=ROWS, appendix_only_rows=APPX, lane_controls={f'{k[0]}|{k[1]}': v for k, v in HP.items()},
    heat_appendix_rows=HEAT_APPX, heat_facts=HH, tunability=TUNE, incoming=INCOMING, nmrom_baselines=appx_nb, burgers_facts={f'{k[0]}|{k[1]}' if isinstance(k, tuple) else k: v for k, v in HB.items() if not (isinstance(k, tuple) and k[1] in ('coarse', 'acc'))}, failures=F, pending=PENDING, macros=mac, intake_schema=INTAKE_SCHEMA), indent=2) + '\n')
print(f'Headline: {len(ROWS)} rows, {mac["nHeadFasterRows"]} with a faster NM-ROM setting; {len(F)} failure rows; all snapshots hash-verified.')
