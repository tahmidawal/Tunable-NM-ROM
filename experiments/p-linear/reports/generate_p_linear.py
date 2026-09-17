"""Generate the p-linear report, summary.json and the linear-case figure from run JSONs.

    python reports/generate_p_linear.py [--attempts plin1024 plin256] [--head plhead1] [--smoke DIR]

Every number in the report comes from `result.json` (and the matching `audit.json`) of
each attempt; nothing is typed by hand. Figures are drawn with matplotlib and the plotted
points are also written to JSON beside them.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
DATE = '2026-09-17'


def load_attempt(name):
    for cand in (LANE / 'artifacts' / name / 'result.json',
                 LANE / 'runs' / name / 'archive' / 'output' / 'result.json'):
        if cand.exists():
            d = json.loads(cand.read_text())
            audit = None
            for ac in (LANE / 'artifacts' / name / 'audit.json', LANE / 'runs' / name / 'audit.json'):
                if ac.exists():
                    audit = json.loads(ac.read_text())
                    break
            return d, audit, cand
    return None, None, None


def pct(x):
    return f'{100 * x:.4f} %'


def ms(x):
    return f'{1e3 * x:.3f}'


def non_dominated(points):
    keep = []
    for i, (e, c) in enumerate(points):
        if not any((e2 <= e and c2 <= c) and (e2 < e or c2 < c)
                   for j, (e2, c2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def rows_of(d):
    """One summary row per subject from the invocation grid."""
    inv = d['invocations']
    names = []
    for s in d['declared_subjects']:
        if not s.get('skipped') and s['name'] not in names:
            names.append(s['name'])
    setup = {a['arm']: a for a in d['arm_setup']}
    out = []
    for name in names:
        sel = [x for x in inv if x['name'] == name]
        cases = sorted({x['case'] for x in sel})
        per_case = [max(x['same_grid_error'] for x in sel if x['case'] == c) for c in cases]
        per_case_p = [max(x['physical_error'] for x in sel if x['case'] == c) for c in cases]
        tot = [x['total_seconds'] for x in sel]
        dev_ = [x.get('fused_device_seconds', x.get('solver_seconds', 0.0) + x.get('projection_init_seconds', 0.0))
                for x in sel]
        r0 = sel[0]
        kind = r0['kind']
        if kind == 'ladder':
            valid = sum(1 for x in sel if x['solver_valid'])
            stat = sum(1 for x in sel if x['stationary'])
            iters = float(np.median([x['jacobians'] for x in sel]))
        elif kind in ('pa', 'linear'):
            stat = sum(1 for x in sel if x['reason'] == 4)
            valid = stat
            iters = float(np.median([x['iterations'] for x in sel]))
        elif kind == 'cg':
            valid = sum(1 for x in sel if x['cg_converged'])
            stat = valid
            iters = float(np.median([x['iterations'] for x in sel]))
        else:
            valid = stat = len(sel)
            iters = None
        a = setup.get(name, {})
        out.append(dict(name=name, kind=kind, family=r0.get('family'), model=r0.get('model'),
                        k=r0.get('k'), q=r0.get('q'), rule=r0.get('rule'), M=r0.get('M'),
                        tolerance=r0.get('tolerance'), worst=max(per_case), median=float(np.median(per_case)),
                        worst_physical=max(per_case_p), total_ms=float(np.median(tot)) * 1e3,
                        device_ms=float(np.median(dev_)) * 1e3, total_ms_reps=[float(t) * 1e3 for t in tot],
                        stationary=stat, valid=valid, count=len(sel), iterations=iters,
                        degenerate=bool(a.get('degenerate_full_bank', False))))
    return out


def criterion(rows, prim, R):
    """The pre-registered clauses. DESIGN A4: the q = R point is the DIRECT rank-R linear
    solve `d_linear_qr_m4`, not the eliminated `q{R}_m4` arm; A8: falsification is reported
    under both the literal wording and the clause's own parenthetical intent."""
    ladder = [r for r in rows if r['kind'] == 'ladder' and r['model'] == prim and r['rule'] == 'm4'
              and '_ccrule' not in r['name'] and r['q'] < R]
    ladder.sort(key=lambda r: r['q'])
    top = next(r for r in rows if r['name'] == f'd_linear_qr_m4@{prim}')
    ladder = ladder + [top]
    costs = [r['total_ms'] for r in ladder]
    errs = [r['worst'] for r in ladder]
    fom = [r for r in rows if r['kind'] in ('fom', 'cg')]
    allp = non_dominated([(r['worst'], r['total_ms']) for r in rows])
    red = [r for r in rows if r['kind'] not in ('fom', 'cg')]
    redp = non_dominated([(r['worst'], r['total_ms']) for r in red])
    nd_all = [rows[i]['name'] for i in allp]
    nd_red = [red[i]['name'] for i in redp]
    d1 = max(costs) / min(costs)
    d1_neural = max(costs[:-1]) / min(costs[:-1]) if len(costs) > 1 else 1.0
    cheap = int(np.argmin(costs))
    buys = [dict(rung=ladder[i]['name'], cost_factor=costs[i] / costs[cheap], worst=errs[i])
            for i in range(len(ladder))
            if costs[i] >= 2 * costs[cheap] and errs[i] < errs[cheap]]
    mono = all(errs[i + 1] <= errs[i] + 1e-15 for i in range(len(errs) - 1))
    return dict(ladder=[r['name'] for r in ladder], costs=costs, errs=errs,
                D1_span=d1, D1=d1 < 2, D1_neural_span_posthoc=d1_neural,
                D2_lowest=top['worst'] <= min(errs) + 1e-15,
                D2_within=top['total_ms'] <= 1.1 * min(costs),
                D2_strict=top['total_ms'] <= min(costs) + 1e-12,
                D3_fom=any(n in [f['name'] for f in fom] for n in nd_all),
                D3_pod=any(n.startswith('e_pod') for n in nd_red),
                nd_all=nd_all, nd_red=nd_red, monotone=mono,
                falsified_literal=bool(d1 >= 2 and mono), falsified_intent=bool(buys),
                buys=buys, cheapest=ladder[cheap]['name'], top=top['name'])


def subject_table(rows):
    lines = ['| subject | family | unknowns | q | rule | M | worst same-grid | median same-grid | worst physical | median total ms | median device ms | valid | iters |',
             '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| `{r['name']}` | {r['family']} | {r['k'] if r['k'] is not None else '—'} | "
                     f"{r['q'] if r['q'] is not None else '—'} | {r['rule'] or '—'} | {r['M'] or '—'} | "
                     f"{pct(r['worst'])} | {pct(r['median'])} | {pct(r['worst_physical'])} | {r['total_ms']:.3f} | "
                     f"{r['device_ms']:.3f} | {r['valid']}/{r['count']} | "
                     f"{'—' if r['iterations'] is None else f'{r['iterations']:.1f}'} |")
    return '\n'.join(lines)


def figure(d, rows, crit, path_png, path_json):
    """The linear-case figure.

    Two honesty rules, because both have a way of misleading a reader here.
    (1) The ladder LINE is the pre-registered ladder of DESIGN A4 — the q < R rungs plus
        the DIRECT rank-R solve — not the eliminated q = R arm, which is drawn separately
        and greyed because its cost is an inert-iteration artefact.
    (2) The exact solvers have zero same-grid error by construction. They are drawn on an
        explicitly drawn and labelled round-off floor, never at a fabricated small value.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = d['intervals']
    fig, ax = plt.subplots(figsize=(9.9, 5.2), dpi=150)
    prim = d['config']['models'][0]['id']
    # The informative range is the REDUCED models'. The exact/near-exact full-order solvers
    # sit up to twelve decades below and would crush it, so the axis is clipped to the
    # reduced range and everything below is drawn on an explicit, labelled off-scale band
    # carrying its true value. Nothing is ever plotted at a fabricated error.
    red_err = [100 * r['worst'] for r in rows if r['kind'] not in ('fom', 'cg') and r['worst'] > 0]
    FLOOR = 0.45 * min(red_err)
    R = max(d['config']['ladder_q'])
    groups = {
        'ladder (m4)': [r for r in rows if r['kind'] == 'ladder' and r['rule'] == 'm4' and r['model'] == prim and '_ccrule' not in r['name'] and r['q'] < R]
                       + [r for r in rows if r['name'] == f'd_linear_qr_m4@{prim}'],
        'q=R eliminated (inert iteration)': [r for r in rows if r['name'] == f'q{R}_m4@{prim}'],
        'ladder (m256)': [r for r in rows if r['kind'] == 'ladder' and r['rule'] == 'm256' and r['model'] == d['config']['models'][0]['id']],
        'incumbent R=128': [r for r in rows if r['model'] == 'incumbent'],
        'POD-LSPG (m4)': [r for r in rows if r['name'].startswith('e_pod') and r['rule'] == 'm4'],
        'POD-LSPG (m256)': [r for r in rows if r['name'].startswith('e_pod') and r['rule'] == 'm256'],
        'head only / free bank / linear QR': [r for r in rows if r['kind'] in ('pa', 'linear') and r['model'] == prim],
        'CG': [r for r in rows if r['kind'] == 'cg'],
        'DST direct': [r for r in rows if r['kind'] == 'fom'],
    }
    palette = {'ladder (m4)': '#1d4ed8', 'ladder (m256)': '#60a5fa', 'incumbent R=128': '#94a3b8',
               'POD-LSPG (m4)': '#b45309', 'POD-LSPG (m256)': '#f59e0b', 'head only / free bank / linear QR': '#7c3aed',
               'CG': '#059669', 'DST direct': '#dc2626', 'q=R eliminated (inert iteration)': '#cbd5e1'}
    markers = {'ladder (m4)': 'o', 'ladder (m256)': 'o', 'incumbent R=128': 's', 'POD-LSPG (m4)': '^',
               'POD-LSPG (m256)': '^', 'head only / free bank / linear QR': 'D', 'CG': 'x', 'DST direct': '*',
               'q=R eliminated (inert iteration)': 'o'}
    def y_of(r):
        e = 100 * r['worst']
        return e if e > FLOOR else FLOOR
    points, offscale = [], []
    for label, grp in groups.items():
        if not grp:
            continue
        xs = [r['total_ms'] for r in grp]
        ys = [y_of(r) for r in grp]
        ax.scatter(xs, ys, label=label, color=palette[label], marker=markers[label],
                   s=48 if label != 'DST direct' else 150, zorder=3, alpha=0.9,
                   edgecolor='white' if markers[label] != 'x' else None,
                   linewidth=0.6 if markers[label] != 'x' else 1.4)
        if label == 'ladder (m4)':
            order = np.argsort([r['q'] for r in grp])
            ax.plot([xs[i] for i in order], [ys[i] for i in order], color=palette[label], lw=1.4, zorder=2)
            for i in order:
                tag = f"q={grp[i]['q']}" + (' (direct)' if grp[i]['kind'] == 'linear' else '')
                ax.annotate(tag, (xs[i], ys[i]), textcoords='offset points', xytext=(5, 4),
                            fontsize=7, color=palette[label])
        if label == 'POD-LSPG (m4)':
            for r, x, y in zip(grp, xs, ys):
                ax.annotate(f"k'={r['k']}", (x, y), textcoords='offset points', xytext=(5, -10),
                            fontsize=7, color=palette[label])
        if label == 'CG':
            for i, (r, x, y) in enumerate(sorted(zip(grp, xs, ys), key=lambda t: t[1])):
                # always above the marker: the off-scale band prints true values below
                ax.annotate(f"tol {r['tolerance']:g}", (x, y), textcoords='offset points',
                            xytext=(0, 7 if i % 2 == 0 else 16), fontsize=7,
                            color=palette[label], ha='center')
        for r, x, y in zip(grp, xs, ys):
            off = 100 * r['worst'] <= FLOOR
            points.append(dict(group=label, name=r['name'], total_ms=x, worst_same_grid_pct=100 * r['worst'],
                               drawn_on_offscale_band=bool(off),
                               non_dominated_all=r['name'] in crit['nd_all'],
                               non_dominated_reduced=r['name'] in crit['nd_red']))
            if off:
                offscale.append((r, x))
    if offscale:
        ax.axhline(FLOOR, color='#64748b', lw=0.9, ls=':', zorder=1)
        for i, (r, x) in enumerate(sorted(offscale, key=lambda t: t[1])):
            e = 100 * r['worst']
            ax.annotate('exact (0)' if e == 0 else f'{e:.0e} %', (x, FLOOR),
                        textcoords='offset points', xytext=(0, -12 if i % 2 == 0 else -21),
                        fontsize=6.5, color='#475569', ha='center')
        ax.annotate('off-scale band — true worst error printed under each marker',
                    (0.015, FLOOR), xycoords=('axes fraction', 'data'),
                    textcoords='offset points', xytext=(0, 6), fontsize=7, color='#475569')
    nd = [p for p in points if p['non_dominated_all']]
    nd.sort(key=lambda p: p['total_ms'])
    ax.plot([p['total_ms'] for p in nd], [max(p['worst_same_grid_pct'], FLOOR) for p in nd],
            color='black', lw=1.0, ls='--', zorder=1, label='non-dominated (all)')
    ax.set_ylim(FLOOR / 1.6, 1.9 * max(100 * r['worst'] for r in rows))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('median complete-query time, ms (same job, same GPU)')
    ax.set_ylabel('worst relative error vs same-grid FOM, %')
    ax.set_title(f'Poisson 2D, {n}² intervals: correction ladder against every comparator\n'
                 f'(one job, one GPU, 12 development sources, 3 timed repetitions)', fontsize=10.5)
    ax.grid(True, which='both', alpha=0.22)
    ax.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1.0), framealpha=1.0,
              borderaxespad=0.)
    ax.set_ylim(bottom=FLOOR / 2.4)
    fig.tight_layout()
    fig.savefig(path_png)
    fig.savefig(path_png.with_suffix('.pdf'))
    Path(path_json).write_text(json.dumps(dict(intervals=n, job_id=d['job_id'], points=points), indent=2) + '\n')
    plt.close(fig)


def head_section(d, audit):
    cfg = d['config']
    bank = d['bank']
    fd = bank['floor_development']['worst']
    lines = [f"## Job 3 — head capacity on the frozen bank (`{cfg['attempt']}`, job `{d['job_id']}`, `{d['gpu']}`)", '',
             f"Bank `{bank['tag']}`, R = {bank['R']}, {bank['sources']} training sources; development bank floor at 255 intervals "
             f"{pct(fd)} worst, {pct(bank['floor_development']['median'])} median. Recipe: `pbh_fit.train_head_phase`, "
             f"{cfg['arms'][0]['updates']} full-batch Adam updates at lr {cfg['head_lr']}, seed {cfg['head_seed']}, "
             f"reconstruction objective only, on the {d['cohorts']['fit_count']}-source fit split.", '',
             '| arm | K | width | layers | updates | head params | fit (stored codes, worst) | best-found val (worst) | best-found dev (worst) | dev best-found / floor | training s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in d['arms']:
        lines.append(f"| `{a['arm']}` | {a['K']} | {a['width']} | {a['layers']} | {a['updates'] or '—'} | {a.get('head_parameter_count', '—')} | "
                     f"{pct(a['head_at_stored_codes_fit']['worst'])} | {pct(a['best_found_validation']['worst'])} | "
                     f"{pct(a['best_found_development']['worst'])} | {a['ratio_dev_best_found_over_floor']:.3f}x | "
                     f"{a.get('training_seconds', 0):.0f} |")
    ctrl = next(a for a in d['arms'] if a['arm'] == 'K32_w128_L2')
    prim = next(a for a in d['arms'] if a['arm'] == 'pbh02_primary_K32')
    h1 = abs(ctrl['best_found_development']['worst'] - prim['best_found_development']['worst']) / prim['best_found_development']['worst']
    rho0 = prim['ratio_dev_best_found_over_floor']
    arms = [a for a in d['arms'] if a['arm'] not in ('pbh02_primary_K32',)]
    best = min(arms, key=lambda a: a['best_found_validation']['worst'])
    drops = {a['arm']: 1 - a['ratio_dev_best_found_over_floor'] / rho0 for a in arms}
    maxdrop = max(drops.values())
    lines += ['', f"**H1 (reproducibility).** The re-run of the pbh02 recipe (`K32_w128_L2`) lands at {pct(ctrl['best_found_development']['worst'])} "
              f"worst development best-found against pbh02's frozen primary {pct(prim['best_found_development']['worst'])} in the same job: "
              f"{100 * h1:.2f} % relative difference against the pre-registered 5 % bar — **{'pass' if h1 <= 0.05 else 'FAIL'}**.", '',
              (('**H1 FAILED, so every H2 conclusion below is CONDITIONAL** (DESIGN §5: a failed H1 '
                'makes the reproduction failure the headline of job 3, and no capacity claim is safe '
                'until the recipe reproduces the incumbent).\n\n') if h1 > 0.05 else '')
              + f"**H2 (verdict{', conditional' if h1 > 0.05 else ''}).** The primary's ratio is {rho0:.3f}x. "
              f"The largest relative reduction of that ratio by any capacity arm is "
              f"{100 * maxdrop:.1f} % (`{max(drops, key=drops.get)}`), against the 20 % bar; the lowest ratio reached is "
              f"{min(a['ratio_dev_best_found_over_floor'] for a in arms):.3f}x against the 2x bar for a better anchor. "
              + (('The reading that follows would hold only if H1 passed: ' if h1 > 0.05 else '')
                 + ('**Function-class-limited within the tested range.**' if maxdrop <= 0.20 else
                    ('**A better anchor was found.**' if min(a['ratio_dev_best_found_over_floor'] for a in arms) < 2 else
                     '**Partial movement only.**'))),
              f" Selection by the internal-validation split picks `{best['arm']}`; the development sources selected nothing.", '',
              f"### Solved at {d['config']['solve_intervals']} intervals through the unchanged head-ablation kernel", '',
              '| subject | K | worst same-grid | median same-grid | worst physical | median total ms | median device ms | stationary | LM iters |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows_of(d):
        lines.append(f"| `{r['name']}` | {r['k'] or '—'} | {pct(r['worst'])} | {pct(r['median'])} | {pct(r['worst_physical'])} | "
                     f"{r['total_ms']:.3f} | {r['device_ms']:.3f} | {r['stationary']}/{r['count']} | "
                     f"{'—' if r['iterations'] is None else f'{r['iterations']:.1f}'} |")
    recs = d['reconstruction']
    lines += ['', '| model | bank floor (worst) | best-found (worst) | best-found / floor |', '|---|---:|---:|---:|']
    for r in recs:
        lines.append(f"| `{r['model']}` | {pct(r['bank_projection']['worst'])} | {pct(r['best_found']['worst'])} | "
                     f"{r['best_found']['worst'] / r['bank_projection']['worst']:.3f}x |")
    lines += ['', 'Gates: ' + '; '.join(f"`{g['ours']}` vs `{g['theirs']}` {g['worst_relative_difference']:.3e} ({'pass' if g['passed'] else 'FAIL'})"
                                        for g in d['gates'])]
    if audit:
        lines.append(f"Independent NumPy audit: {'all checks pass' if audit['all_passed'] else 'FAILURES'} "
                     f"(recomputed errors worst difference {audit['recomputed_errors']['worst_same_grid_difference']:.2e}).")
    summary = [dict(mesh=255, subject=a['arm'], family='head-capacity', q_or_k=a['K'], metric='dev_best_found_worst',
                    value=a['best_found_development']['worst'], job_id=d['job_id']) for a in d['arms']]
    summary += [dict(mesh=255, subject=a['arm'], family='head-capacity', q_or_k=a['K'], metric='dev_best_found_over_floor',
                     value=a['ratio_dev_best_found_over_floor'], job_id=d['job_id']) for a in d['arms']]
    for r in rows_of(d):
        summary += [dict(mesh=d['config']['solve_intervals'], subject=r['name'], family=r['kind'], q_or_k=r['k'],
                         metric=m, value=v, job_id=d['job_id'])
                    for m, v in (('worst_same_grid', r['worst']), ('median_total_ms', r['total_ms']))]
    return '\n'.join(lines), summary, dict(H1_pass=h1 <= 0.05, H1_relative=h1, H2_max_drop=maxdrop,
                                           H2_min_ratio=min(a['ratio_dev_best_found_over_floor'] for a in arms))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attempts', nargs='*', default=['plin1024', 'plin256'])
    ap.add_argument('--head', default='plhead1')
    ap.add_argument('--out', default=str(HERE / f'{DATE}-p-linear.md'))
    a = ap.parse_args()
    md = ['# Poisson 2D — the correction ladder to q = R on the best checkpoint, against POD-LSPG, DST and CG', '',
          'This report covers one pre-registered cell (`experiments/p-linear/DESIGN.md`): the full correction ladder on the '
          'R=512/K=32 Poisson checkpoint with every comparator timed in the same job, at 1024 and 256 intervals, plus a '
          'head-capacity job on the same frozen bank. All numbers come from completed, checksum-collected, independently '
          'audited GPU jobs on the twelve opened development sources; they are development evidence, not sealed-cohort '
          'results. Numbers marked *provisional* below are from an attempt whose audit or archive is not yet complete.', '']
    summary = []
    verdicts = {}
    for att in a.attempts:
        d, audit, src = load_attempt(att)
        if d is None:
            md.append(f'## `{att}` — not available\n')
            continue
        rows = rows_of(d)
        prim = d['config']['models'][0]['id']
        crit = criterion(rows, prim, max(d['config']['ladder_q']))
        n = d['intervals']
        prov = '' if (audit and audit.get('all_passed')) else ' *(provisional: audit not complete)*'
        md += [f"## {n} intervals — `{att}`, job `{d['job_id']}`, `{d['gpu']}`, source `{d['commit'][:12]}`{prov}", '',
               f"`jax_backend={d['backend']}`, float64, matmul precision `{d['matmul_precision']}`, JAX {d['jax_version']}; "
               f"{d['config']['repetitions']} timed repetitions, {len(d['cohort']['parameters'])} sources, "
               f"{len({r['name'] for r in rows})} subjects, randomised order, burn-in {d['config']['burn_seconds']} s; "
               f"elapsed {d['elapsed_seconds'] / 60:.1f} min.", '',
               '### Pre-registered degenerate-curve criterion (m4 ladder of the primary checkpoint)', '',
               '| clause | requirement | value | verdict |', '|---|---|---:|---|',
               f"| D1 cost span | max/min median total ms over the ladder < 2 | {crit['D1_span']:.3f}x | **{'pass' if crit['D1'] else 'FAIL'}** |",
               f"| D2 top rung lowest error | `{crit['top']}` has the lowest worst error of the ladder | {pct(crit['errs'][-1])} vs min {pct(min(crit['errs']))} | **{'pass' if crit['D2_lowest'] else 'FAIL'}** |",
               f"| D2 top rung within 1.1x of the cheapest | cheapest is `{crit['cheapest']}` | {crit['costs'][-1] / min(crit['costs']):.3f}x | **{'pass' if crit['D2_within'] else 'FAIL'}** |",
               f"| D2 strict: top rung IS the cheapest | | | {'yes' if crit['D2_strict'] else 'no'} |",
               f"| D3 full-order solver on the non-dominated set | | {', '.join(f'`{x}`' for x in crit['nd_all'])} | **{'pass' if crit['D3_fom'] else 'FAIL'}** |",
               f"| D3 POD-LSPG on the reduced non-dominated set | | {', '.join(f'`{x}`' for x in crit['nd_red'])} | **{'pass' if crit['D3_pod'] else 'FAIL'}** |",
               '',
               f"Verdict at {n} intervals: **{'DEGENERATE' if (crit['D1'] and crit['D2_lowest'] and crit['D2_within'] and crit['D3_fom'] and crit['D3_pod']) else 'NOT degenerate under the pre-registered clauses as literally written'}**. "
               f"Ladder error monotone non-increasing in q: {'yes' if crit['monotone'] else 'no'}.", '',
               '**Falsification, both readings (DESIGN §A8).** The clause reads: falsified if D1 fails '
               '*with error monotone non-increasing in q* — its own parenthetical gloss being "the ladder '
               'buys accuracy for $\\ge 2\\times$ cost".', '',
               f"- **Literal**: D1 {'fails' if not crit['D1'] else 'holds'} and the error {'is' if crit['monotone'] else 'is not'} monotone, "
               f"so the literal conjunction is **{'MET' if crit['falsified_literal'] else 'not met'}**.",
               f"- **Intent**: a rung must cost $\\ge 2\\times$ the cheapest ladder point *and* be strictly more accurate than it. "
               f"Rungs that do: **{', '.join(f'`{b['rung']}` ({b['cost_factor']:.2f}x)' for b in crit['buys']) if crit['buys'] else 'none'}**, "
               f"so the intended condition is **{'MET' if crit['falsified_intent'] else 'not met'}**.",
               f"- The cheapest ladder point is `{crit['cheapest']}`; the top rung is `{crit['top']}`. "
               f"The D1 span is driven by the top rung being **{max(crit['costs']) / crit['costs'][-1]:.2f}x cheaper** than the dearest rung, "
               f"not by any rung paying more for accuracy. Span over the $q < R$ rungs alone "
               f"(post-hoc, not pre-registered): {crit['D1_neural_span_posthoc']:.3f}x.", '',
               '### The ladder', '',
               '| rung | q | M | worst same-grid | median same-grid | augmented best-found (worst) | bank floor | median total ms | median device ms | valid | LM Jacobians |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        rec = next(r for r in d['reconstruction'] if r['model'] == prim)
        aug = {x['q']: x['best_found']['worst'] for x in rec['augmented']}
        ladder_rows = [r for r in rows if r['kind'] == 'ladder' and r['model'] == prim
                       and '_ccrule' not in r['name']]
        ladder_rows += [r for r in rows if r['kind'] == 'linear' and r['model'] == prim]
        for r in sorted(ladder_rows, key=lambda r: (r['rule'], r['q'])):
            md.append(f"| `{r['name']}` | {r['q']} | {r['M']} | {pct(r['worst'])} | {pct(r['median'])} | "
                      f"{pct(aug[r['q']]) if r['q'] in aug else '—'} | {pct(rec['bank_projection']['worst'])} | "
                      f"{r['total_ms']:.3f} | {r['device_ms']:.3f} | {r['valid']}/{r['count']}{' (q=R, degenerate by construction)' if r['degenerate'] else ''} | "
                      f"{r['iterations']:.1f} |")
        bfd = rec.get('best_found_dense', {}).get('worst')
        md += ['', f"Three layers at q = 0 for `{prim}`: bank floor {pct(rec['bank_projection']['worst'])}, dense best-found "
               f"{pct(bfd) if bfd is not None else '—'}, projected-oracle best-found {pct(aug[0])}, solved (`q0_m256`) "
               f"{pct(next(r['worst'] for r in rows if r['name'] == f'q0_m256@{prim}'))}.", '',
               '### Every subject', '', subject_table(rows), '', '### Fidelity gates and consistency', '',
               '| gate | metric | worst relative difference | tolerance | verdict |', '|---|---|---:|---:|---|']
        for g in d['gates']:
            md.append(f"| `{g['ours']}` vs `{g['theirs']}` (job {g.get('reference_job')}) | {g.get('metric')} | "
                      f"{g.get('worst_relative_difference', float('nan')):.3e} | {g.get('tolerance')} | {'pass' if g['passed'] else 'FAIL'} |")
        md += ['', '| consistency pair | worst field relative difference | tolerance | verdict |', '|---|---:|---:|---|']
        for c in d['consistency']:
            md.append(f"| `{c['a']}` vs `{c['b']}` | {c['worst_field_relative_difference'] if c['worst_field_relative_difference'] is None else f'{c['worst_field_relative_difference']:.3e}'} | {c['tolerance']} | {'pass' if c['passed'] else 'FAIL'} |")
        for dr in d['directions']:
            if 'retained_prefix_exact' in dr and 'extension_orthonormality_error' in dr:
                md.append(f"\nDirections for `{dr['model']}`: retained prefix exact {dr['retained_prefix_exact']}; rebuilt-prefix subspace defect "
                          f"{dr['rebuilt_prefix_subspace_defect']:.2e}; extension orthonormality error {dr['extension_orthonormality_error']:.2e}; "
                          f"residual energy captured at q=32/128/512: {dr['residual_energy_captured'].get('32', 0):.3f}/"
                          f"{dr['residual_energy_captured'].get('128', 0):.3f}/{dr['residual_energy_captured'].get('512', dr['residual_energy_captured'].get('128', 0)):.3f}.")
        if audit:
            md.append(f"\nIndependent NumPy audit (`audit.json`): {'all checks pass' if audit['all_passed'] else 'FAILURES'}; "
                      f"{audit['recomputed_errors']['invocations']} errors recomputed from {audit['recomputed_errors']['distinct_fields']} retained fields, "
                      f"worst same-grid difference {audit['recomputed_errors']['worst_same_grid_difference']:.2e}; criterion re-derived independently: "
                      f"degenerate = {audit.get('criterion', {}).get('degenerate')}.")
        png = HERE / f'{DATE}-p-linear-{n}.png'
        figure(d, rows, crit, png, HERE / f'{DATE}-p-linear-{n}-points.json')
        md += ['', f'![linear-case figure at {n}]({png.name})', '']
        verdicts[n] = crit
        for r in rows:
            for metric, value in (('worst_same_grid', r['worst']), ('median_same_grid', r['median']),
                                  ('worst_physical', r['worst_physical']), ('median_total_ms', r['total_ms']),
                                  ('median_device_ms', r['device_ms']), ('valid_count', r['valid']),
                                  ('median_iterations', r['iterations'])):
                summary.append(dict(mesh=n, subject=r['name'], family=r['family'] or r['kind'],
                                    q_or_k=r['q'] if r['q'] is not None else r['k'], rule=r['rule'], M=r['M'],
                                    metric=metric, value=value, job_id=d['job_id'], source_sha256=hashlib.sha256(src.read_bytes()).hexdigest()[:16],
                                    non_dominated_all=r['name'] in crit['nd_all'], non_dominated_reduced=r['name'] in crit['nd_red']))
        for x in rec['augmented']:
            summary.append(dict(mesh=n, subject=f"augmented_best_found_q{x['q']}@{prim}", family='oracle', q_or_k=x['q'],
                                metric='worst_same_grid', value=x['best_found']['worst'], job_id=d['job_id']))
        summary.append(dict(mesh=n, subject=f'bank_floor@{prim}', family='oracle', q_or_k=0, metric='worst_same_grid',
                            value=rec['bank_projection']['worst'], job_id=d['job_id']))
        for key in ('D1_span',):
            summary.append(dict(mesh=n, subject='criterion', family='criterion', q_or_k=None, metric=key, value=crit[key], job_id=d['job_id']))
    dh, ah, _ = load_attempt(a.head)
    if dh is not None:
        sec, srows, hv = head_section(dh, ah)
        md += [sec, '']
        summary += srows
        verdicts['head'] = hv
    md += ['## Glossary', '',
           '- **rung / q**: the number of linear correction directions added to the neural head\'s output; q = 0 is the head alone, q = R spans the whole bank (a linear least-squares solve).',
           '- **m4 / m256**: the test-count rule: m4 uses M = 4 x (number of unknowns) sine test modes; m256 uses a fixed 256 requested modes (257 retained).',
           '- **M**: the number of retained sine test modes, i.e. the number of equations in the weak least-squares problem.',
           '- **worst / median same-grid**: relative L2 error against the exact FD-DST solution on the same mesh, worst and median over the 12 development sources.',
           '- **worst physical**: the same against the 2048-interval reference restricted to the mesh.',
           '- **median total ms**: median over all timed invocations of the complete query (host source in, dense field out); **device ms** is the fused device portion.',
           '- **valid**: solves that met every solver-validity rule (stationary, linear recovery, rank); CG rows count converged solves; the q = R rung is flagged invalid by construction because its projected Jacobian is zero.',
           '- **augmented best-found**: the smallest error any point of the augmented manifold {G(h(z)+C_q y)} attains for that source (offline multistart oracle); it brackets the solved error from below.',
           '- **bank floor**: the error of the orthogonal projection of the truth onto the bank\'s column span; no coefficient choice can beat it.',
           '- **POD-LSPG k\'**: a classical proper-orthogonal-decomposition basis of rank k\' from the same 3072 training snapshots, solved with the same weak least-squares objective.',
           '- **DST direct**: the exact discrete-sine-transform full-order solve; **CG tol**: unpreconditioned conjugate gradients stopped at that relative residual.',
           '- **non-dominated set**: subjects for which no other subject in the same job has both lower error and lower cost.',
           '- **D1–D3, H1–H2**: the pre-registered clauses of DESIGN.md sections 4 and 5.',
           '- **incumbent**: the earlier R=128/K=16 checkpoint, carried as a cross-job control; **new_K32 / pbh02 primary**: the R=512/K=32 checkpoint this cell is about.',
           '- **ccrule**: correction directions built with the cheap-corrections lane\'s rule, run only to reproduce that lane\'s q = 64 numbers.',
           '- **best-found / floor ratio**: how far the head\'s own image sits above what its bank could represent.']
    Path(a.out).write_text('\n'.join(md) + '\n')
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (HERE / 'verdicts.json').write_text(json.dumps(verdicts, indent=2, default=str) + '\n')
    print(a.out, len(summary), 'summary rows')


if __name__ == '__main__':
    main()
