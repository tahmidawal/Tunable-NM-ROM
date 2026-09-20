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
accurate as both settings.  Both speedups divide that one FOM time.

New rows from the running lanes enter through ``headline-intake.json``
(see INTAKE_SCHEMA below); until a path is listed there the slot prints dashes.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
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
    'wave': (WT + '2026-09-17-w-ladder', 'experiments/w-ladder/reports/summary.json', '9b84d0556888ebc052b52bd165a61dcd56b2b53d', 'development'),
    'poisson3d': (WT + '2026-09-20-paper-p3d', 'experiments/paper-p3d/runs/final08/paper-comparisons.json', '26c73030b89adfa321ede751db2798abf9bcd1b2', 'accepted final'),
    'heat3d': (WT + '2026-09-20-paper-h3d', 'experiments/paper-h3d/runs/final08/paper-tables-pending-retention.json', '6a6c668b02e81d767a0ceb14a078a5745b629c3d', 'PROVISIONAL final: numerical audits pass, archive retention pending'),
    'ns3d': (WT + '2026-09-20-paper-ns3d', 'experiments/ns3d/runs/final07/paper_summary.json', '3f4e12d6134da21cf82e5259c2990c8ab625b4c6', 'accepted final'),
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
ORDER = ['Poisson', 'Poisson, L-shape', 'Heat', 'Burgers', 'Burgers (earlier model)']


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
    if INTAKE.exists():
        for i, item in enumerate(json.loads(INTAKE.read_text())['sources']):
            raw = Path(item['path']).read_bytes()
            if item.get('sha256'): assert digest(raw) == item['sha256'], item['path']
            key = f"intake_{i:02d}_{item['lane']}"
            (E / f'{key}.json').write_bytes(raw)
            manifest[key] = dict(tree=None, path=item['path'], commit=item.get('commit'), status=item['status'],
                                 sha256=digest(raw), read='coordinator-supplied audited summary', lane=item['lane'])
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


def setting(label, err, ms, **kw):
    return dict(label=label, error_pct=err, ms=ms, **kw)


def row(problem, dim, n, fast, acc, fom, source, job, cohort, status, error, timing, note=''):
    for s in (fast, acc):
        if s: assert fom['error_pct'] <= s['error_pct'] + 1e-12, (problem, n)
    ROWS.append(dict(problem=problem, dim=dim, intervals=n, fast=fast, accurate=acc, fom=fom, source=source,
                     job_id=str(job), cohort=cohort, status=status, error_convention=error, timing_scope=timing, note=note))


# ---- Poisson 2D and Heat 2D: paired CG snapshot -------------------------------------------
cg = defaultdict(dict)
for r in D['paired_cg']['rows']:
    assert abs(r['cg_ms'] / r['method_ms'] - r['speedup']) < 1e-10
    cg[(r['problem'], r['intervals'])][r['method']] = r
for n in (256, 1024):
    g = cg[('Poisson2D', n)]; f, a = g['q0_m4@new_K32'], g['q256_m4@new_K32']
    assert f['cg'] == a['cg'] and f['cg_ms'] == a['cg_ms'] and f['job_id'] == a['job_id']
    row('Poisson', 2, n, setting('$q=0$', f['error_pct'], f['method_ms']), setting('$q=256$', a['error_pct'], a['method_ms']),
        dict(name='CG, rtol $10^{-2}$', error_pct=a['cg_error_pct'], ms=a['cg_ms']), 'paired_cg', a['job_id'],
        'development', 'development', 'same-grid', 'GPU query')
for n in (64, 256, 1024):
    h = cg[('Heat2D', n)]['nmrom_cholesky']
    row('Heat', 2, n, setting('single', h['error_pct'], h['method_ms']), None,
        dict(name='CN--CG, rtol $10^{-2}$', error_pct=h['cg_error_pct'], ms=h['cg_ms']), 'paired_cg', h['job_id'],
        'development', 'development; earlier checkpoint, one setting', 'refined reference', 'GPU query')

# ---- L-shaped Poisson ------------------------------------------------------------------------
V = defaultdict(dict); ljob = {}
for r in D['lshape']:
    if r.get('test_modes') == 257 and r['metric'] in ('worst_same_grid', 'median_total_ms'):
        V[(r['mesh'], r['subject'])][r['metric']] = r['value']; ljob[r['mesh']] = r['job_id']
for n in (256, 512):
    cands = [s for (m, s) in V if m == n and s.startswith('fom_cg_gpu_r')]
    ref = min(cands, key=lambda s: V[(n, s)]['median_total_ms']); c = V[(n, ref)]
    f, a = V[(n, 'neural_q0@head_sdf_R512_K16')], V[(n, 'neural_q64@head_sdf_R512_K16')]
    tol = ref.split('_r')[1]; assert tol == '0.01'
    row('Poisson, L-shape', 2, n, setting('$q=0$', 100 * f['worst_same_grid'], f['median_total_ms']),
        setting('$q=64$', 100 * a['worst_same_grid'], a['median_total_ms']),
        dict(name='CG, rtol $10^{-2}$', error_pct=100 * c['worst_same_grid'], ms=c['median_total_ms']), 'lshape', ljob[n],
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
        q = s.split('_')[0][1:]; kind = 'dense' if 'dense' in s else 'EQ'
        return f'$q={q}$, {kind}'
    assert cs.startswith('nt1e-') and cs.endswith('_dt005')
    row('Burgers', 2, n, setting(lab(fs), roms[fs]['worst_evolved_percent'], roms[fs]['median_gpu_ms'], arm=fs),
        setting(lab(as_), roms[as_]['worst_evolved_percent'], roms[as_]['median_gpu_ms'], arm=as_),
        dict(name='Newton--BiCGStab, tol $10^{-%s}$' % cs[5], error_pct=ok[cs]['worst_evolved_percent'], ms=ok[cs]['median_gpu_ms'], arm=cs),
        'burgers_panel', next(iter(pjob[(n, cs)])), 'development', 'development', 'same-grid, evolved', 'GPU query')

# ---- Burgers 2D, earlier model, tight and relaxed Newton (one NM-ROM setting) ---------------------
from statistics import median
b = D['burgers_iter']; S = {}
for name in ('nmrom', 'fft_tight', 'fft_loose'):
    calls = [r for r in b['invocations'] if r['name'] == name]; assert len(calls) == 12
    S[name] = (100 * max(r['error']['fixed_initial_max'] for r in calls), 1000 * median(r['gpu_seconds'] for r in calls))
for name, title in (('fft_loose', 'Newton--BiCGStab, relaxed'), ('fft_tight', 'Newton--BiCGStab, tight')):
    row('Burgers (earlier model)', 2, 1024, setting('single', *S['nmrom']), None, dict(name=title, error_pct=S[name][0], ms=S[name][1]),
        'burgers_iter', b['job_id'], 'development', 'development; earlier model, stalled exits permitted', 'refined reference', 'GPU query')

# ---- Poisson 3D accepted final -------------------------------------------------------------------
p3 = D['poisson3d']; assert p3['final_cohort_opened']
pr = {(r['intervals'], r['method']): r for r in p3['rows']}; pc = {(r['intervals'], r['method']): r for r in p3['cg_controls']}
for n in (32, 64):
    f, a = pr[(n, 'nmrom_K16_q0_dense')], pr[(n, 'nmrom_K16_q96_dense')]
    c = pc[(n, a['matched_cg_method'])]; assert c['cg_failed_invocations'] == 0 and f['nonstationary_cases'] == a['nonstationary_cases'] == 0
    assert c['method'] == 'cg_identity_plain_rtol1e-02'
    row('Poisson', 3, n, setting('$q=0$', 100 * f['same_grid_error_worst'], f['device_ms_median']),
        setting('$q=96$', 100 * a['same_grid_error_worst'], a['device_ms_median']),
        dict(name='CG, rtol $10^{-2}$', error_pct=100 * c['same_grid_error_worst'], ms=c['device_ms_median']), 'poisson3d', p3['job_id'],
        'final', 'accepted final', 'same-grid', 'GPU query')

# ---- Heat 3D provisional final ----------------------------------------------------------------------
h3 = D['heat3d']; assert h3['status'] == 'numerically_audited_pending_complete_git_retention'
hr = {(r['intervals'], r['method']): r for r in h3['rows'] if r['cohort'] == 'primary' and r['evaluation_cohort'] == 'final'}
for n in (32, 64):
    f, a = hr[(n, 'nmrom_K32_q0_dense')], hr[(n, 'nmrom_K32_q96_dense')]; c = hr[(n, a['matched_cg'])]
    assert c['method'] == 'fom_cn_cg_dt0.05_rtol1e-04' and f['nonstationary_cases'] == a['nonstationary_cases'] == 0
    row('Heat', 3, n, setting('$q=0$', 100 * f['worst_rel_l2'], f['device_ms']), setting('$q=96$', 100 * a['worst_rel_l2'], a['device_ms']),
        dict(name='CN--CG, rtol $10^{-4}$', error_pct=100 * c['worst_rel_l2'], ms=c['device_ms']), 'heat3d', a['job_id'],
        'final', 'provisional', 'same-grid, evolved', 'GPU query')

# ---- coordinator-supplied lane rows ----------------------------------------------------------------
for k, v in MAN.items():
    if not k.startswith('intake_'): continue
    d = D[k]; assert d['schema'] == INTAKE_SCHEMA, k
    for r in d['headline_rows']:
        row(r['problem'], r['dim'], r['intervals'], r.get('fast'), r.get('accurate'), r['fom'], k, r['job_id'], r['cohort'],
            r['status'], r['error_convention'], r['timing_scope'], r.get('note', ''))

ROWS.sort(key=lambda r: (r['dim'], ORDER.index(r['problem']), r['intervals'], r['fom']['ms']))
for r in ROWS:
    for s in ('fast', 'accurate'):
        if r[s]: r[s]['speedup'] = r['fom']['ms'] / r[s]['ms']


# ---- rendering -----------------------------------------------------------------------------------------
def e(x): return f'{x:.2f}' if x >= 0.1 else f'{x:.3f}'
def spn(x): return (f'{x:.0f}' if x >= 100 else f'{x:.1f}' if x >= 10 else f'{x:.2f}' if x >= 0.1 else f'{x:.3f}' if x >= 0.01 else f'{x:.4f}')
def sp(x):
    t = (f'{x:.0f}' if x >= 100 else f'{x:.1f}' if x >= 10 else f'{x:.2f}' if x >= 0.1 else f'{x:.3f}' if x >= 0.01 else f'{x:.4f}') + r'$\times$'
    return r'\textbf{' + t + '}' if x > 1 else t
def mesh(r): return f"${r['intervals']}^{r['dim']}$"
MARK = {'refined reference': r'$^{r}$', 'complete query': r'$^{c}$'}
def marks(r):
    m = MARK.get(r['error_convention'], '') + MARK.get(r['timing_scope'], '')
    if r['status'].startswith('provisional'): m += r'$^{p}$'
    if r['cohort'] == 'final' and not r['status'].startswith('provisional'): m += r'$^{f}$'
    return m
def cells(s): return [e(s['error_pct']), sp(s['speedup'])] if s else ['---', '---']


def covered(p, n):
    return any(r['problem'] == p['problem'] and r['dim'] == p['dim'] and r['intervals'] == n for r in ROWS)


def write(name, lines, mdhead, mdrows):
    (HERE / 'tables' / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    (HERE / 'tables-md' / f'{name}.md').write_text('| ' + ' | '.join(mdhead) + ' |\n| ' + ' | '.join(['---'] * len(mdhead)) + ' |\n' + ''.join('| ' + ' | '.join(x) + ' |\n' for x in mdrows))


def md(x): return x.replace('---', '—').replace(r'$\times$', '×').replace(r'\textbf{', '**').replace('}', '**' if 'textbf' in x else '}').replace('$', '').replace('--', '–')

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
        todo = [n for n in p['meshes'] if not covered(p, n)]
        if not todo: continue
        m = ', '.join(f'${n}^{dim}$' for n in todo)
        lines.append(f"{p['problem']} & {m} & --- & --- & --- & --- & --- & \\emph{{pending: {p['lane']}}} \\\\")
        mdrows.append([f"{p['problem']} {dim}D", ', '.join(f'{n}^{dim}' for n in todo), '—', '—', '—', '—', '—', f"pending: {p['lane']} lane"])
lines += [r'\bottomrule', r'\end{tabular}']
write('TH_headline', lines, ['Problem', 'Mesh', 'Accurate err. (%)', 'Accurate speedup', 'Fast err. (%)', 'Fast speedup', 'FOM err. (%)', 'FOM'], mdrows)

# supporting times (appendix)
tl = [r'% GENERATED by paper/gen_headline.py -- do not edit.', r'\scriptsize', r'\begin{tabular}{@{}llllrrrlll@{}}', r'\toprule',
      r'Problem & Mesh & Accurate & Fast & Accurate ms & Fast ms & FOM ms & Timing & Job & Status \\', r'\midrule']
tmd = []
for r in ROWS:
    a, f = r['accurate'], r['fast']
    c = [f"{r['problem']} {r['dim']}D", mesh(r), a['label'] if a else '---', f['label'] if f else '---', f"{a['ms']:.2f}" if a else '---',
         f"{f['ms']:.2f}" if f else '---', f"{r['fom']['ms']:.2f}", r['timing_scope'], r'\texttt{' + r['job_id'] + '}', r['status'].split(';')[0]]
    tl.append(' & '.join(c) + r' \\'); tmd.append([md(x).replace('\\texttt{', '').replace('}', '') for x in c])
tl += [r'\bottomrule', r'\end{tabular}']
write('TH_headline_times', tl, ['Problem', 'Mesh', 'Accurate', 'Fast', 'Accurate ms', 'Fast ms', 'FOM ms', 'Timing', 'Job', 'Status'], tmd)

# second small table slot: other nonlinear-manifold ROMs (nmrom-baselines lane)
bl = [r'% GENERATED by paper/gen_headline.py -- slot for the nmrom-baselines lane; no value is read until an audited summary is supplied.',
      r'\small', r'\begin{tabular}{@{}llrr@{}}', r'\toprule', r'Problem, mesh & Method & Err.\ (\%) & Speedup vs.\ FOM \\', r'\midrule']
base = [r for k in MAN if k.startswith('intake_') for r in D[k].get('baseline_rows', [])]
if base:
    for r in base: bl.append(f"{r['problem']} & {r['method']} & {e(r['error_pct'])} & {sp(r['fom_ms'] / r['ms'])} \\\\")
else:
    for m in ('This work (accurate / fast)', 'Convolutional-autoencoder NM-ROM', 'Shallow-masked-autoencoder NM-ROM'):
        bl.append(f'--- & {m} & --- & --- \\\\')
bl += [r'\bottomrule', r'\end{tabular}']
write('TH_nmrom_baselines', bl, ['Problem, mesh', 'Method', 'Err. (%)', 'Speedup vs FOM'], [['—', 'pending: nmrom-baselines lane', '—', '—']] if not base else [[r['problem'], r['method'], e(r['error_pct']), md(sp(r['fom_ms'] / r['ms']))] for r in base])

# ---- where the method currently fails ------------------------------------------------------------------
F = []
B = {r['method']: r for r in D['burgers3d']['rows']}
assert D['burgers3d']['final_evaluation']['complete']['complete']
bf = B['fom_n33_dt0.01_nt1e-02_lt5e-01']
for m, lab_ in (('rom_q0', '$q=0$'), ('rom_q192', '$q=192$')):
    F.append(dict(problem='Burgers 3D, $33^3$ (final)', setting=lab_, error_pct=100 * B[m]['worst_evolved'], fom_error_pct=100 * bf['worst_evolved'],
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
    c = [r for r in ROWS if pred(r) and r[key]]
    return max(c, key=lambda r: r[key]['speedup'])
mac = {}
pa = best(lambda r: r['problem'] == 'Poisson' and r['dim'] == 2, 'accurate')
mac['nHeadPoissonAccErr'] = e(pa['accurate']['error_pct']); mac['nHeadPoissonAccS'] = spn(pa['accurate']['speedup'])
mac['nHeadPoissonFastS'] = spn(pa['fast']['speedup']); mac['nHeadPoissonMesh'] = f"{pa['intervals']}^2"
la = best(lambda r: r['problem'] == 'Poisson, L-shape', 'accurate'); mac['nHeadLshapeAccS'] = spn(la['accurate']['speedup']); mac['nHeadLshapeAccErr'] = e(la['accurate']['error_pct'])
ha = best(lambda r: r['problem'] == 'Heat' and r['dim'] == 2, 'fast'); mac['nHeadHeatFastS'] = spn(ha['fast']['speedup']); mac['nHeadHeatFastErr'] = e(ha['fast']['error_pct'])
bb = [r for r in ROWS if r['problem'] == 'Burgers']
b1024 = [r for r in bb if r['intervals'] == 1024][0]; b256 = [r for r in bb if r['intervals'] == 256][0]
mac['nHeadBurgersFastS'] = spn(b1024['fast']['speedup']); mac['nHeadBurgersFastErr'] = e(b1024['fast']['error_pct'])
mac['nHeadBurgersAccErr'] = e(b256['accurate']['error_pct']); mac['nHeadBurgersAccS'] = spn(b256['accurate']['speedup'])
p3a = best(lambda r: r['problem'] == 'Poisson' and r['dim'] == 3, 'accurate'); mac['nHeadPoissonThreeAccS'] = spn(p3a['accurate']['speedup']); mac['nHeadPoissonThreeAccErr'] = e(p3a['accurate']['error_pct'])
bq = P[(1024, 'q128_M576_eqxfer_g1em06')]; assert bq['admissible'] and bq['converged_design5']
mac['nHeadBurgersMidErr'] = e(bq['worst_evolved_percent']); mac['nHeadBurgersMidS'] = spn(b1024['fom']['ms'] / bq['median_gpu_ms'])
h3a = best(lambda r: r['problem'] == 'Heat' and r['dim'] == 3, 'accurate'); mac['nHeadHeatThreeAccErr'] = e(h3a['accurate']['error_pct'])
FB = {(r['source'], r['setting']): r for r in F}
mac['nFailBurgersQzero'] = e(FB[('burgers3d', '$q=0$')]['error_pct']); mac['nFailBurgersAcc'] = e(FB[('burgers3d', '$q=192$')]['error_pct'])
mac['nFailNsQzero'] = e(FB[('ns3d', '$q=0$')]['error_pct']); mac['nFailNsAcc'] = e(FB[('ns3d', '$q=256$')]['error_pct'])
mac['nFailNsTarget'] = f"{D['ns3d']['target_percent']:g}"; mac['nFailNsFailing'] = str(FB[('ns3d', '$q=256$')]['failing']); mac['nFailNsCases'] = str(FB[('ns3d', '$q=256$')]['cases'])
mac['nFailWaveQzero'] = e(FB[('wave', '$q=0$')]['error_pct']); mac['nFailWaveAcc'] = e(FB[('wave', '$q=32$')]['error_pct'])
mac['nHeadFasterRows'] = str(sum(1 for r in ROWS if any(r[s] and r[s]['speedup'] > 1 for s in ('fast', 'accurate'))))
mac['nHeadRows'] = str(len(ROWS))
mac['nHeadAccFasterSubOne'] = str(sum(1 for r in ROWS if r['accurate'] and r['accurate']['speedup'] > 1 and r['accurate']['error_pct'] < 1))
(HERE / 'tables/headline-numbers.tex').write_text('% GENERATED by paper/gen_headline.py -- do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(mac.items())))

# ---- compact 3D configuration table (values read from the run records) ---------------------------------------
cfg = []
b3 = D['burgers3d']
cfg.append(['Burgers 3D', '$33^3$ nodes', str(B['rom_q192']['cases']) + ' final', '0, 192', 'Newton--BiCGStab, $\\Delta t=0.01$', 'dense', 'job \\texttt{%s}' % b3['job_id']])
cfg.append(['Poisson 3D', '$32^3$, $64^3$', str(pr[(64, 'nmrom_K16_q96_dense')]['cases']) + ' final', '0, 32, 96 ($k=16$)', 'CG, rtol $10^{-2}$, no preconditioner', 'dense', 'job \\texttt{%s}' % p3['job_id']])
cfg.append(['Heat 3D$^{p}$', '$32^3$, $64^3$', str(hr[(64, 'nmrom_K32_q96_dense')]['cases']) + ' final', '0, 32, 64, 96 ($k=32$)', 'CN--CG, $\\Delta t=0.05$, rtol $10^{-4}$, warm start', 'dense', 'job \\texttt{%s}' % h3['source']['primary']['job_id']])
ns = D['ns3d']
cfg.append(['Navier--Stokes 3D', f"${ns['n']}^3$ periodic", f"{ns['cohort_count']} final", ', '.join(str(q) for q in ns['q_values']) + f" ($k={ns['k']}$, $R={ns['r']}$, $M={ns['test_modes']}$)", 'CNAB2, $\\Delta t=0.01$', 'dense', 'job \\texttt{%s}' % ns['job_id']])
cl = [r'% GENERATED by paper/gen_headline.py -- do not edit.', r'\scriptsize', r'\begin{tabular}{@{}lllp{3.3cm}p{3.6cm}ll@{}}', r'\toprule',
      r'Problem & Mesh & Cases & Correction ranks & Named FOM & Residual & Allocation \\', r'\midrule'] + [' & '.join(c) + r' \\' for c in cfg] + [r'\bottomrule', r'\end{tabular}']
write('TH_config3d', cl, ['Problem', 'Mesh', 'Cases', 'Correction ranks', 'Named FOM', 'Residual', 'Allocation'], [[md(x) for x in c] for c in cfg])

(HERE / 'tables/headline-provenance.json').write_text(json.dumps(dict(
    rule='One frozen model per row. fast = q=0; accurate = largest stored correction rank at the standard time step (Burgers: fastest / lowest-error admissible residual evaluation at that rank). '
         'One named FOM per row from the same allocation, at least as accurate as both settings; speedup = FOM ms / NM-ROM ms. Bold = speedup > 1.',
    sources=MAN, rows=ROWS, failures=F, pending=PENDING, macros=mac, intake_schema=INTAKE_SCHEMA), indent=2) + '\n')
print(f'Headline: {len(ROWS)} rows, {mac["nHeadFasterRows"]} with a faster NM-ROM setting; {len(F)} failure rows; all snapshots hash-verified.')
