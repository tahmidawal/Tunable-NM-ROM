"""Generate reports/summary.json and the report from the collected result.json / audit.json files (and lane-ref/ns3d.json).
No number in the report is typed by hand.

    python make_report.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'reports' / '2026-09-24-spectral-fom-vs-nmrom.md'

# accepted runs: (attempt, output dir) per problem/mesh; anything not listed is not reported
ACCEPTED = {
    'poisson2d': [('spA', 'output0'), ('spA', 'output1'), ('spA', 'output2'), ('spB', 'output0')],
    'poisson3d': [('spA', 'output3'), ('spA', 'output4'), ('spM', 'output0'), ('spM', 'output1')],
    'burgers2d': [('spF', 'output0'), ('spF', 'output1'), ('spF', 'output2'), ('spJ', 'output0'), ('spL', 'output0')],
    'burgers2d_original_ladder': [('spC', 'output0'), ('spC', 'output1'), ('spC', 'output2'), ('spE', 'output0')],
    'burgers2d_L1_2048_without_half_dst': [('spG', 'output0')],
    'heat2d': [('spI', 'output0'), ('spI', 'output1'), ('spI', 'output2')],
    'heat2d_two_variant_record': [('spD', 'output0'), ('spD', 'output1')],
    'heat3d': [('spN', 'output0'), ('spN', 'output1'), ('spP', 'output0'), ('spP', 'output1')],
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(att, od):
    d = HERE / 'runs' / att / 'archive' / od
    if not (d / 'result.json').exists():
        return None
    R = json.loads((d / 'result.json').read_text())
    A = json.loads((d / 'audit.json').read_text()) if (d / 'audit.json').exists() else None
    jobid = (HERE / 'runs' / att / 'JOBID.txt').read_text().strip()
    return dict(R=R, A=A, path=str(d.relative_to(HERE) / 'result.json'), sha256=sha(d / 'result.json'), job=jobid,
                attempt=att)


def pct(x, nd=3):
    return '—' if x is None else f'{100 * x:.{nd}f}%'


def sci(x):
    return '—' if x is None else f'{x:.1e}'


def f2(x):
    return '—' if x is None else f'{x:.3f}'


def ratio(a, b):
    return None if (a is None or b is None) else a / b


def gate_str(g):
    bad = [k for k, v in g.items() if not v and k != 'neighbour_T1']
    if 'neighbour' in bad and g.get('neighbour_T1'):
        bad.remove('neighbour')
        return ('raw neighbour FAIL (case mix); case-normalised PASS (T1)' + ('' if not bad else '; FAIL: ' + ', '.join(bad)))
    return 'all pass' if not bad else 'FAIL: ' + ', '.join(bad)


def t1(att, od, R):
    if 'neighbour_case_normalised' in R['timing_gates']:
        return R['timing_gates']['neighbour_case_normalised']['passed']
    p = HERE / 'runs' / att / 'archive' / od / 'T1-case-normalised-neighbour.json'
    return json.loads(p.read_text())['case_normalised']['passed'] if p.exists() else None


def rows_poisson(problem):
    out = []
    for att, od in ACCEPTED[problem]:
        x = load(att, od)
        if x is None:
            continue
        R, T, E = x['R'], x['R']['timings'], x['R']['errors']
        roms = {v['role']: k for k, v in T.items() if v['role'].startswith('rom')}
        specs = [k for k, v in T.items() if v['role'] == 'spectral']
        best = min(specs, key=lambda k: T[k]['median_ms'])
        ra, rf = roms['rom_accurate'], roms['rom_fast']
        V = R['validation']
        out.append(dict(problem=problem, mesh=R['intervals'], gpu=R['gpu'], job=x['job'], attempt=att, output=od,
                        result=x['path'], result_sha256=x['sha256'], commit=R['commit'],
                        accurate=dict(arm=ra, worst=E[ra]['worst'], ms=T[ra]['median_ms'], a1=T[ra]['romA1_ms'], a2=T[ra]['romA2_ms']),
                        fast=dict(arm=rf, worst=E[rf]['worst'], ms=T[rf]['median_ms'], a1=T[rf]['romA1_ms'], a2=T[rf]['romA2_ms']),
                        spectral={k: dict(worst=E[k]['worst'], ms=T[k]['median_ms']) for k in specs},
                        spectral_fom=best, spectral_ms=T[best]['median_ms'], spectral_worst=E[best]['worst'],
                        ratio_accurate=T[best]['median_ms'] / T[ra]['median_ms'],
                        ratio_fast=T[best]['median_ms'] / T[rf]['median_ms'],
                        validation=dict(spectral_vs_scipy_worst=max(r['relative'] for r in V['spectral_vs_scipy']),
                                        cg=V['cg_convergence'], control=V['control']['relative_vs_truth']),
                        gates=R['gates'], audit=None if x['A'] is None else x['A']['verdict'], neighbour_T1=t1(att, od, R),
                        drift=R['timing_gates']['drift']['rows'],
                        neighbour_worst=max(r.get('ratio', np.inf) for r in R['timing_gates']['neighbour']['rows'])))
    return out


def rows_burgers(key='burgers2d'):
    out = []
    for att, od in ACCEPTED[key]:
        x = load(att, od)
        if x is None:
            continue
        R, T, E = x['R'], x['R']['timings'], x['R']['errors']
        conv = {}
        for inv in R['invocations']:
            if inv['role'] == 'spectral':
                conv[inv['name']] = conv.get(inv['name'], True) and inv.get('converged', True)
        specs = [k for k, v in T.items() if v['role'] == 'spectral' and conv.get(k, False)]
        arms = {}
        for role in ('rom_accurate', 'rom_fast'):
            a = R['selected'][role]
            ok = [k for k in specs if E[k]['worst'] <= E[a]['worst']]
            m = min(ok, key=lambda k: T[k]['median_ms']) if ok else None
            arms[role] = dict(arm=a, worst=E[a]['worst'], median=E[a]['median'], ms=T[a]['median_ms'],
                              a1=T[a]['romA1_ms'], a2=T[a]['romA2_ms'],
                              matched=m, matched_worst=None if m is None else E[m]['worst'],
                              matched_ms=None if m is None else T[m]['median_ms'],
                              ratio_matched=None if m is None else T[m]['median_ms'] / T[a]['median_ms'],
                              ratio_tight=T['pic_dt005_nt1e-6']['median_ms'] / T[a]['median_ms'])
        V = R['validation']
        out.append(dict(problem='burgers2d', mesh=R['intervals'], gpu=R['gpu'], job=x['job'], attempt=att, output=od,
                        result=x['path'], result_sha256=x['sha256'], commit=R['commit'], arms=arms,
                        spectral={k: dict(worst=E[k]['worst'], median=E[k]['median'], ms=T[k]['median_ms'],
                                          converged=conv.get(k)) for k in T if T[k]['role'] == 'spectral'},
                        dst_variant=R['dst_variant_benchmark']['chosen'],
                        validation=dict(agreement=V['agreement_over_u0'], control=V['control']['agreement_over_u0'],
                                        fom_newton=V['paper_fom_tight']['newton_total'],
                                        picard_sweeps=V['spectral_tight']['iterations_total']),
                        lane_parity=R['lane_error_parity'], gates=R['gates'], neighbour_T1=t1(att, od, R),
                        audit=None if x['A'] is None else x['A']['verdict'],
                        drift=R['timing_gates']['drift']['rows'],
                        neighbour_worst=max(r.get('ratio', np.inf) for r in R['timing_gates']['neighbour']['rows'])))
    return out


def rows_heat(problem):
    out = []
    for att, od in ACCEPTED[problem]:
        x = load(att, od)
        if x is None:
            continue
        R, T, E = x['R'], x['R']['timings'], x['R']['errors']
        specs = [k for k, v in T.items() if v['role'] == 'spectral']
        arms = {}
        for role, a in R['config']['rom_arms'].items():
            ok = [k for k in specs if E[k]['worst'] <= E[a]['worst']]
            m = min(ok, key=lambda k: T[k]['median_ms'])
            ex = min([k for k in specs if k.startswith('modal_exp')], key=lambda k: T[k]['median_ms'])
            arms[role] = dict(arm=a, worst=E[a]['worst'], ms=T[a]['median_ms'], a1=T[a]['romA1_ms'], a2=T[a]['romA2_ms'],
                              matched=m, matched_worst=E[m]['worst'], matched_ms=T[m]['median_ms'],
                              ratio_matched=T[m]['median_ms'] / T[a]['median_ms'],
                              exact=ex, exact_ms=T[ex]['median_ms'], ratio_exact=T[ex]['median_ms'] / T[a]['median_ms'])
        V = R['validation']
        out.append(dict(problem=problem, mesh=R['intervals'], gpu=R['gpu'], job=x['job'], attempt=att, output=od,
                        result=x['path'], result_sha256=x['sha256'], commit=R['commit'], cohort=R['cohort']['name'],
                        arms=arms, spectral={k: dict(worst=E[k]['worst'], ms=T[k]['median_ms']) for k in specs},
                        validation=dict(exp_worst=max(r['worst'] for r in V['exp_vs_truth']),
                                        cn_vs_cncg=V['cn_vs_cncg']['worst'], control=V['control']['worst']),
                        gates=R['gates'], audit=None if x['A'] is None else x['A']['verdict'], neighbour_T1=t1(att, od, R),
                        drift=R['timing_gates']['drift']['rows'],
                        neighbour_worst=max(r.get('ratio', np.inf) for r in R['timing_gates']['neighbour']['rows'])))
    return out


def main():
    S = dict(poisson2d=rows_poisson('poisson2d'), poisson3d=rows_poisson('poisson3d'), burgers2d=rows_burgers(),
             burgers2d_original_ladder=rows_burgers('burgers2d_original_ladder'),
             burgers2d_L1_2048_without_half_dst=rows_burgers('burgers2d_L1_2048_without_half_dst'),
             heat2d_two_variant_record=rows_heat('heat2d_two_variant_record'),
             heat2d=rows_heat('heat2d'), heat3d=rows_heat('heat3d'))
    ns = json.loads((HERE / 'lane-ref' / 'ns3d.json').read_text())
    S['ns3d'] = ns
    (HERE / 'reports').mkdir(exist_ok=True)
    (HERE / 'reports' / 'summary.json').write_text(json.dumps(S, indent=1) + '\n')
    L = []
    w = L.append
    w('# Spectral / fast-transform FOMs vs NM-ROM (stored, not in the paper)\n')
    w('**ARCHIVE ONLY.** On 2026-09-23 the user decided that this lane is an archive. It is not used in the paper. '
      'The numbers below are final for the meshes listed.\n')
    w('This report measures the strongest fast-transform full-order solver for each problem in the paper and times it '
      'against our accurate and fast settings. Each comparison runs in the same allocation, on the same discrete system '
      'and against the same reference. The numbers are **stored for a later decision** and are not in `paper/`. '
      'Every number below is generated by `experiments/spectral-fom/make_report.py` from the collected JSON. '
      'Ratio = spectral FOM GPU ms / our GPU ms, so a ratio **below 1 means the spectral solver is faster**. '
      'Rows are final for the meshes listed; missing meshes are listed under open items.\n')
    w('## Findings (generated)\n')

    def span(vals):
        vals = [v for v in vals if v is not None]
        return '—' if not vals else (f'{min(vals):.3f}' if len(vals) == 1 else f'{min(vals):.3f}–{max(vals):.3f}')
    for prob, title in (('poisson2d', 'Poisson 2D square'), ('poisson3d', 'Poisson 3D cube')):
        rows = S[prob]
        if rows:
            w(f"- **{title}** ({', '.join(str(r['mesh']) for r in rows)}): spectral/ours is {span([r['ratio_accurate'] for r in rows])} "
              f"against the accurate setting and {span([r['ratio_fast'] for r in rows])} against the fast setting. The DST "
              f"solve is exact, with worst error ≤ {max(r['spectral_worst'] for r in rows):.1e}. "
              + ('The spectral solver is **faster and more accurate than both settings at every mesh**.'
                 if max(max(r['ratio_accurate'], r['ratio_fast']) for r in rows) < 1 else
                 'Ratios above 1, where the ROM is faster, occur at: ' + ', '.join(str(r['mesh']) for r in rows if max(r['ratio_accurate'], r['ratio_fast']) >= 1) + '.'))
    if S['burgers2d']:
        rows = S['burgers2d']
        w(f"- **Burgers 2D** ({', '.join(str(r['mesh']) for r in rows)}): the matched spectral/ours ratio is "
          f"{span([r['arms']['rom_accurate']['ratio_matched'] for r in rows])} for the accurate setting and "
          f"{span([r['arms']['rom_fast']['ratio_matched'] for r in rows])} for the fast one. Tight Picard, at reference "
          f"accuracy, gives {span([r['arms']['rom_accurate']['ratio_tight'] for r in rows])} and "
          f"{span([r['arms']['rom_fast']['ratio_tight'] for r in rows])}. "
          + ('At every measured mesh, the fastest spectral setting of at least the same accuracy is faster than both '
             'NM-ROM settings.' if all(r['arms'][ro]['ratio_matched'] is not None and r['arms'][ro]['ratio_matched'] < 1
                                       for r in rows for ro in ('rom_accurate', 'rom_fast'))
             else 'The ROM is faster than its matched spectral setting at: ' + ', '.join(
                 f"{r['mesh']} {ro[4:]}" for r in rows for ro in ('rom_accurate', 'rom_fast')
                 if r['arms'][ro]['ratio_matched'] is None or r['arms'][ro]['ratio_matched'] >= 1) + '.'))
    for prob, title in (('heat2d', 'Heat 2D'), ('heat3d', 'Heat 3D')):
        rows = S[prob]
        if rows:
            roles = list(rows[0]['arms'])
            w(f"- **{title}** ({', '.join(str(r['mesh']) for r in rows)}): matched spectral/ours is "
              + '; '.join(f"{span([r['arms'][ro]['ratio_matched'] for r in rows])} ({ro[4:]} `{rows[0]['arms'][ro]['arm']}`)" for ro in roles)
              + '. The exact modal propagation is also the lane\'s truth.')
    nsr = ns['rows']
    w(f"- **NS 3D** (recorded): CNAB2/ours is {span([r['ratio_matched'] for r in nsr if r['role'] == 'rom_accurate'])} "
      f"for the accurate setting and {span([r['ratio_matched'] for r in nsr if r['role'] == 'rom_fast'])} for the fast "
      f"one. The ROM is faster than its matched CNAB2 setting in {sum(r['ratio_matched'] > 1 for r in nsr)} of "
      f"{len(nsr)} rows. It is not faster at: "
      + (', '.join(f"{r['mesh']}³ {r['role'][4:]} ({r['ratio_matched']:.3f})" for r in nsr if r['ratio_matched'] <= 1) or 'none') + '.')
    w('- **L-shape**: there is no spectral arm (see below).\n')
    for prob, title, unit in (('poisson2d', 'Poisson 2D square (Dirichlet, 5-point)', '²'),
                              ('poisson3d', 'Poisson 3D cube (Dirichlet, 7-point); 32³/64³ final cohort, 128³/256³ development cohort (as the lane)', '³')):
        rows = S[prob]
        if not rows:
            continue
        w(f'## {title}\n')
        w('| mesh | GPU | job | accurate arm | err | ms | fast arm | err | ms | `dst_fft` ms | `dst_mm` ms | spectral err | '
          'ratio vs accurate | ratio vs fast | timing gates | audit |')
        w('|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|')
        for r in rows:
            g = dict(drift=r['gates']['drift'], neighbour=r['gates']['neighbour'], deterministic=r['gates']['deterministic'], neighbour_T1=r['neighbour_T1'])
            w(f"| {r['mesh']}{unit} | {r['gpu']} | {r['job']} | `{r['accurate']['arm']}` | {pct(r['accurate']['worst'])} | "
              f"{f2(r['accurate']['ms'])} | `{r['fast']['arm']}` | {pct(r['fast']['worst'])} | {f2(r['fast']['ms'])} | "
              f"{f2(r['spectral']['dst_fft']['ms'])} | {f2(r['spectral']['dst_mm']['ms'])} | {sci(r['spectral_worst'])} | "
              f"**{r['ratio_accurate']:.3f}** | **{r['ratio_fast']:.3f}** | {gate_str(g)} | {r['audit']} |")
        w('')
        w('Validation. The spectral solve was compared with the SciPy DST truth on every case. The paper\'s CG was run '
          'at rtol $10^{-6}$, $10^{-8}$ and $10^{-10}$ on case 0, and its distance to the spectral field is given. The '
          'control (continuum eigenvalues) must exceed $10^{-10}$. "CG certified" gives the paper CG\'s own '
          'true-residual convergence flag at each rtol.\n')
        w('| mesh | spectral vs SciPy (worst) | CG→spectral distance at rtol 1e-6 / 1e-8 / 1e-10 | CG certified | CG true rel. residual at 1e-10 | control |')
        w('|---:|---:|---|---|---:|---:|')
        for r in rows:
            cg = r['validation']['cg']
            w(f"| {r['mesh']}{unit} | {sci(r['validation']['spectral_vs_scipy_worst'])} | "
              + ' / '.join(sci(c['cg_vs_spectral']) for c in cg) + ' | '
              + ' / '.join('yes' if c['converged'] else 'no' for c in cg)
              + f" | {sci(cg[-1]['true_relative_residual'])} | {sci(r['validation']['control'])} |")
        w('')
    if S['burgers2d']:
        w('## Burgers 2D (Dirichlet walls, sign-upwind advection, backward Euler)\n')
        w('The spectral FOM solves each backward-Euler step by the modal-Helmholtz fixed point '
          '$u \\leftarrow u - (I+\\Delta t\\nu A)^{-1} r(u)$ on the paper\'s own residual (Picard), or takes one '
          'IMEX sweep. It is matched to each arm as the fastest ladder setting whose worst error is no worse than the '
          'arm\'s. "Tight" is Picard at ntol $10^{-6}$, $\\Delta t=0.005$, the same tolerance as the reference.\n')
        w('| mesh | GPU | job | role | arm | err (worst) | ms | matched spectral | its err | its ms | ratio (matched) | ratio (tight) | timing gates | audit |')
        w('|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|')
        for r in S['burgers2d']:
            g = dict(drift=r['gates']['drift'], neighbour=r['gates']['neighbour'], deterministic=r['gates']['deterministic'], neighbour_T1=r['neighbour_T1'])
            for role in ('rom_accurate', 'rom_fast'):
                a = r['arms'][role]
                w(f"| {r['mesh']}² | {r['gpu']} | {r['job']} | {role[4:]} | `{a['arm']}` | {pct(a['worst'])} | {f2(a['ms'])} | "
                  f"{a['matched'] or 'none as accurate'} | {pct(a['matched_worst'])} | {f2(a['matched_ms'])} | "
                  f"{'—' if a['ratio_matched'] is None else f'**{a['ratio_matched']:.3f}**'} | {a['ratio_tight']:.3f} | "
                  f"{gate_str(g)} | {r['audit']} |")
        w('')
        w('Spectral ladder (worst / median error, GPU ms):\n')
        names = list(S['burgers2d'][0]['spectral'])
        w('| setting | ' + ' | '.join(f"{r['mesh']}²" for r in S['burgers2d']) + ' |')
        w('|---|' + '---|' * len(S['burgers2d']))
        for nm in names:
            w(f'| `{nm}` | ' + ' | '.join(f"{pct(r['spectral'][nm]['worst'])} / {pct(r['spectral'][nm]['median'])} / {f2(r['spectral'][nm]['ms'])}"
                                        for r in S['burgers2d']) + ' |')
        w('')
    for key, text in (('burgers2d_original_ladder', 'Original pre-registered 11-setting ladder (spC/spE). It is superseded by the L1 ladder above, which contains it:'),
                      ('burgers2d_L1_2048_without_half_dst', 'L1 ladder at 2048² without the half-length DST candidate (spG). It is superseded by spJ, which adds that candidate (amendment L2):')):
        if not S[key]:
            continue
        w(text + '\n')
        w('| mesh | job | role | err | ms | matched spectral | its err | its ms | ratio (matched) | timing gates |')
        w('|---:|---|---|---:|---:|---|---:|---:|---:|---|')
        for r in S[key]:
            g = dict(drift=r['gates']['drift'], neighbour=r['gates']['neighbour'], deterministic=r['gates']['deterministic'], neighbour_T1=r['neighbour_T1'])
            for role in ('rom_accurate', 'rom_fast'):
                a = r['arms'][role]
                w(f"| {r['mesh']}² | {r['job']} | {role[4:]} | {pct(a['worst'])} | {f2(a['ms'])} | {a['matched']} | {pct(a['matched_worst'])} | "
                  f"{f2(a['matched_ms'])} | {'—' if a['ratio_matched'] is None else f'{a['ratio_matched']:.3f}'} | {gate_str(g)} |")
        w('')
    if S['burgers2d']:
        w('| mesh | DST in $H^{-1}$ | paper FOM (1e-10) vs Picard (1e-10), max over t of diff/‖u0‖ | control (1.01ν) | ROM re-run vs lane errors (worst rel. dev.) |')
        w('|---:|---|---:|---:|---|')
        for r in S['burgers2d']:
            w(f"| {r['mesh']}² | {r['dst_variant']} | {sci(r['validation']['agreement'])} | {sci(r['validation']['control'])} | "
              + ', '.join(f"{sci(p['worst_relative_deviation'])}" for p in r['lane_parity']) + ' |')
        w('')
    for prob, title, unit in (('heat2d', 'Heat 2D', '²'), ('heat3d', 'Heat 3D', '³')):
        rows = S[prob]
        if not rows:
            continue
        w(f'## {title} (Dirichlet, modal CN at the paper Δt and exact modal propagation)\n')
        w('| mesh | GPU | job | cohort | role | arm | err | ms | matched spectral | its err | its ms | ratio (matched) | ratio (exact) | timing gates | audit |')
        w('|---:|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|')
        for r in rows:
            g = dict(drift=r['gates']['drift'], neighbour=r['gates']['neighbour'], deterministic=r['gates']['deterministic'], neighbour_T1=r['neighbour_T1'])
            for role, a in r['arms'].items():
                w(f"| {r['mesh']}{unit} | {r['gpu']} | {r['job']} | {r['cohort']} | {role[4:]} | `{a['arm']}` | {pct(a['worst'], 4)} | {f2(a['ms'])} | "
                  f"`{a['matched']}` | {pct(a['matched_worst'], 4)} | {f2(a['matched_ms'])} | **{a['ratio_matched']:.3f}** | {a['ratio_exact']:.3f} | {gate_str(g)} | {r['audit']} |")
        w('')
    if S['heat2d_two_variant_record']:
        w('Heat 2D record without the half-length DST variant (spD, `fft`/`mm` only). It is superseded by the rows above (amendment L2):\n')
        w('| mesh | job | role | arm | err | ms | matched spectral | its ms | ratio (matched) | timing gates |')
        w('|---:|---|---|---|---:|---:|---|---:|---:|---|')
        for r in S['heat2d_two_variant_record']:
            g = dict(drift=r['gates']['drift'], neighbour=r['gates']['neighbour'], deterministic=r['gates']['deterministic'], neighbour_T1=r['neighbour_T1'])
            for role, a in r['arms'].items():
                w(f"| {r['mesh']}² | {r['job']} | {role[4:]} | `{a['arm']}` | {pct(a['worst'], 4)} | {f2(a['ms'])} | `{a['matched']}` | {f2(a['matched_ms'])} | {a['ratio_matched']:.3f} | {gate_str(g)} |")
        w('')
    w('## Navier–Stokes 3D (recorded from `ns3d-shift-head`, not rerun)\n')
    w(ns['note'] + ' The ratio is CNAB2 ms / ROM ms, from the lane\'s own timing block and comparator rule. That '
      'lane used its protocol v2 (a fast block plus after-heavy gates), not this lane\'s A–B–A.\n')
    w('| mesh | cohort | job | role | arm | err (evolved worst) | ROM ms | matched CNAB2 | its err | its ms | ratio (matched) | most accurate CNAB2 | ratio | lane gates |')
    w('|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---|')
    for r in ns['rows']:
        c = r['cnab2_matched']
        w(f"| {r['mesh']}³ | {r['cohort']} | {r['job_id']} | {r['role'][4:]} | {r['arm']} | {pct(r['worst'])} | {f2(r['rom_ms'])} | "
          f"{c['label']} | {pct(c['worst'])} | {f2(c['ms'])} | **{r['ratio_matched']:.3f}** | {r['cnab2_most_accurate']['label']} | "
          f"{r['ratio_most_accurate']:.3f} | {'pass' if r['timing_gates']['passed'] else 'FAIL'} |")
    w('')
    w('## L-shaped Poisson\n')
    w('No spectral arm. The Dirichlet Laplacian on the L-shaped domain is not diagonalised by any separable fast '
      'transform. The domain is not a tensor product, and its re-entrant corner gives a singular solution. The '
      'standard alternatives are domain decomposition with fast solvers on rectangles, a capacitance-matrix method, '
      'or multigrid. Each of these is an iterative or direct method of the kind the paper already excludes, so none '
      'was built.\n')
    w('## Provenance\n')
    w('| problem | mesh | attempt/output | job | GPU | commit | result.json sha256 |')
    w('|---|---:|---|---|---|---|---|')
    for prob in ('poisson2d', 'poisson3d', 'burgers2d', 'burgers2d_original_ladder', 'burgers2d_L1_2048_without_half_dst',
                 'heat2d', 'heat2d_two_variant_record', 'heat3d'):
        for r in S[prob]:
            w(f"| {prob} | {r['mesh']} | {r['attempt']}/{r['output']} | {r['job']} | {r['gpu']} | {str(r['commit'])[:10]} | `{r['result_sha256']}` |")
    for r in ns['rows']:
        if r['role'] == 'rom_accurate':
            w(f"| ns3d (lane) | {r['mesh']} | {r['job']} | {r['job_id']} | {r['gpu']} | {str(r['commit'])[:10]} | `{r['summary_sha256']}` |")
    w('')
    w('## Audit scope and review\n')
    w('- **Poisson (2D, cube).** The audit recomputes every subject\'s error on every case at full resolution, '
      'against a NumPy dense-sine truth.')
    w('- **Burgers and heat.** The audit is **partial**.')
    w('  - Full-grid errors are recomputed only where full fields were kept: case 0, for the ROM arms and one or two '
      'spectral subjects.')
    w('  - Every other (subject, case) is checked on a subsampled grid against the driver\'s saved subsampled truth.')
    w('  - For Burgers, the truth is the paper\'s own FOM, and it is not recomputed independently.')
    w('- **`check_timing.py`.** It independently recomputes, from the raw invocation records:')
    w('  - every median, the drift gate, the raw neighbour gate and the case-normalised (T1) neighbour gate;')
    w('  - the subject × phase × case × repetition inventory, which must have no duplicates or gaps;')
    w('  - Burgers lane parity, on both arms and every case.')
    w('  It includes two controls that must be detected. It passes on every accepted and record run.')
    w('- **Codex review.** An independent read-only codex review (`checks/codex-review1.md`) found no blocker, and no '
      'numerical mismatch in two recomputed rows. Its disposition is in `checks/codex-review1-disposition.md`.')
    w('- **Burgers DST choice.** The DST implementation inside the Burgers $H^{-1}$ was chosen per mesh by a short, '
      'untimed, fixed-order micro-benchmark with no burn-in, so a sub-optimal choice there would only make the '
      'spectral side slower. The chosen variant is listed per mesh.\n')
    w('## Glossary\n')
    for k, v in (
        ('NM-ROM', 'the paper\'s reduced model: a frozen spatial bank G, with a query solving for a few coefficients.'),
        ("R' (bank width)", 'the number of leading columns of the importance-ordered bank a query uses. It is the '
                            'tunability knob.'),
        ("linear rung (`R<R'>_linear`, `lin_R<R'>`)", "a solve in the span of the first R' bank columns. There is no head."),
        ('head, q', 'u = G h(z), a small network h of a latent z. q is the number of correction directions (0 = head only).'),
        ('accurate / fast setting', 'the owning lane\'s pre-registered choices: the most accurate arm, and the cheapest arm no '
                                   'worse than the paper\'s old fast setting.'),
        ('spectral FOM', 'a full-order solver that diagonalises the discrete operator with a fast sine transform (DST-I). '
                         'Here it solves exactly the same discrete system as the paper\'s iterative FOM.'),
        ('`dst_fft` / `dst_mm`', 'two implementations of the same DST-I: a real FFT of the odd extension, and GEMM with the '
                                 'dense orthonormal sine matrix.'),
        ('Picard (`pic_dt<Δt>_nt<ntol>`)', 'the Burgers step solved by u ← u − (I+ΔtνA)⁻¹ r(u) until ‖r‖ ≤ ntol‖u_prev‖, '
                                          'the paper\'s Newton stopping rule.'),
        ('IMEX (`imex_dt<Δt>`)', 'one Picard sweep per step: explicit upwind advection, implicit diffusion.'),
        ('`modal_cn` / `modal_exp`', 'heat solved in DST modal space: Crank–Nicolson at Δt = 0.025 applied exactly '
                                     '(= CN–CG with an exact linear solve), or exact exponential propagation (= the truth).'),
        ('err', 'the paper\'s per-problem error, worst over the cohort. Poisson: relative L2 against the same-grid discrete '
                'solution. Heat: relative L2 per output time, max over times. Burgers: max over t ≥ 0.05 of ‖u−u_ref‖/‖u0‖ '
                'against the paper\'s tight Newton–BiCGStab reference. NS: the lane\'s evolved worst.'),
        ('ms', 'GPU query time: the median of `fused_device_seconds`, from after the synchronised input copy to completion. '
               'For ROM rows, over phases A1 ∪ A2; for spectral rows, over phase B.'),
        ('ratio', 'spectral FOM ms / our ms. Below 1 means the spectral solver is faster.'),
        ('matched spectral', 'the fastest spectral setting whose worst error is ≤ the arm\'s worst error.'),
        ('A–B–A', 'ROM phase, spectral phase, ROM phase in one allocation. Each is randomised, with burn-in before every '
                  'call, and with a sync, cooldown and fixed dummy kernel between phases.'),
        ('drift gate', 'the ROM median in A2 divided by the median in A1 must lie in [1/1.1, 1.1].'),
        ('neighbour gate', 'the median after a long predecessor divided by the median after a short one must be ≤ 1.10, within '
                           'a phase. "Case-normalised (T1)" divides each time by that subject\'s median on that case first.'),
        ('audit', 'an independent NumPy recomputation of every recorded error from saved fields, using a different truth '
                  'method, with two controls that must be detected.'),
        ('CG certified', 'the paper CG\'s own flag that its true relative residual reached rtol.'),
        ('cohort', 'Poisson 2D: 12 opened development sources. Cube: the final cohort, 64 cases. Burgers: dev6. Heat: the '
                   'held-out sealed cohort. NS: the lane\'s development and held-out cohorts.'),
    ):
        w(f'- **{k}**: {v}')
    w('')
    REPORT.write_text('\n'.join(L) + '\n')
    print('wrote', REPORT.relative_to(HERE), 'and reports/summary.json')


if __name__ == '__main__':
    main()
