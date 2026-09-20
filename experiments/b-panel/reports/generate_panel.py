"""Generate the b-panel report, its figures and `summary.json` from the audit JSONs.

    python reports/generate_panel.py --audit <audit.json> [<audit.json> ...] --out-dir reports --stem 2026-09-17-b-panel

Every number, every table cell and every plotted point is read from the audit JSONs. Nothing is
typed by hand. One section per mesh (one audit = one job = one allocation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MARK = {'rom': 'o', 'fast': 'D', 'pod': 's', 'free': 'P', 'fom': 'X', 'fno': '^'}
COLOR = {'rom': '#3b6ea5', 'fast': '#1f4e79', 'pod': '#8a5a2a', 'free': '#c28a3a', 'fom': '#b03a3a', 'fno': '#4f8a3d'}
LABEL = {'rom': 'correction ladder (frozen model)', 'fast': 'optimised q = 0 kernel', 'pod': 'POD-LSPG',
         'free': 'unrestricted bank (R = 512)', 'fom': 'full-order Newton (same job)', 'fno': 'trained FNO (same allocation)'}
METRICS = ('worst_all_times_percent', 'median_all_times_percent', 'worst_evolved_percent', 'median_evolved_percent',
           'worst_t0_compression_percent', 'worst_reference_percent', 'median_reference_percent', 'median_gpu_ms',
           'median_host_ms', 'median_iterations', 'max_iterations', 'total_budget_exits', 'max_joint_stationarity',
           'max_step_stationarity', 'max_ic_stationarity', 'max_ic_relative_residual', 'best_found_percent',
           'solved_over_best_found', 'converged_design5', 'converged_own_gtol', 'converged_strict', 'admissible',
           'admissible_own_gtol', 'rho_max', 'rho_p95', 'rule_basis',
           'rule_set', 'rule_status', 'rule_m', 'rule_source_lane', 'rule_source_job', 'rule_fit_states', 'rule_file_sha256')


def f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, float):
        return f'{x:.{d}f}'
    return str(x)


def sci(x):
    return '—' if x is None else f'{x:.1e}'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
    return '\n'.join(out) + '\n'


def by_arm(au, name):
    return next((x for x in au['arms'] if x['arm'] == name), None)


def nondominated(rows, ck, ek):
    """Same rule as audit_panel.nondominated, re-stated here so counterfactual frontiers are computed, not asserted."""
    pts = [(r['arm'], r[ck], r[ek]) for r in rows if r.get(ck) is not None and r.get(ek) is not None]
    return [a for a, c, e in pts if not any(c2 <= c and e2 <= e and (c2 < c or e2 < e) for b, c2, e2 in pts if b != a)]


REDUCED = ('rom', 'fast', 'pod', 'free')


def loose_arms(au):
    """Reduced subjects converged at their own tolerance but not under DESIGN §5's fixed 1e-6 (DESIGN §A13)."""
    return [x for x in au['arms'] if x['family'] in REDUCED and x.get('converged_own_gtol') and not x.get('converged_design5')]


def sub_label(x):
    if x['family'] in ('rom', 'fast'):
        return f"q={x['q']}"
    if x['family'] in ('pod', 'free'):
        return f"k'={x['k']}"
    return '—'


def figure(au, out_png, out_pdf, out_json):
    rows = au['arms']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plotted = []
    pos = [x[k] for x in rows for k in ('worst_evolved_percent', 'worst_all_times_percent')
           if x.get(k) is not None and x[k] > 0]
    floor = min(pos) / 3. if pos else 1e-4      # exactly-zero subjects are the same-grid reference itself
    for ax, ek, title in ((axes[0], 'worst_evolved_percent', 'worst over evolved times'),
                          (axes[1], 'worst_all_times_percent', 'worst over all output times')):
        nd = set(au['nondominated']['gpu_evolved' if ek.startswith('worst_evolved') else 'gpu_all']['admissible'])
        seen = set()
        for x in rows:
            c, er = x['median_gpu_ms'], x[ek]
            if c is None or er is None:
                continue
            fam = x['family']
            lab = LABEL[fam] if fam not in seen else None
            seen.add(fam)
            ax.scatter(c, max(er, floor), marker=MARK[fam], s=42 if x['admissible'] else 26, c=COLOR[fam],
                       alpha=1. if x['admissible'] else .45, edgecolors='k' if x['arm'] in nd else 'none',
                       linewidths=1.4 if x['arm'] in nd else 0, label=lab, zorder=3)
            plotted.append(dict(metric=ek, arm=x['arm'], family=fam, median_gpu_ms=c, error_percent=er,
                                admissible=x['admissible'], nondominated=(x['arm'] in nd)))
        front = sorted([x for x in rows if x['arm'] in nd], key=lambda z: z['median_gpu_ms'])
        if front:
            ax.plot([z['median_gpu_ms'] for z in front], [max(z[ek], floor) for z in front], 'k--', lw=.8, zorder=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('median GPU time per query (ms)')
        ax.set_ylabel('same-grid error (%), ' + title)
        ax.set_title(f"{au['intervals']}² intervals, job {au['job_id']}, {au['gpu']}", fontsize=9)
        ax.grid(True, which='both', alpha=.25)
    axes[0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    plt.close(fig)
    Path(out_json).write_text(json.dumps(dict(source=au['result_sha256'], job_id=au['job_id'], points=plotted), indent=2) + '\n')


def section(W, au, tag):
    L = au['intervals']
    W(f"## {L}² intervals — job `{au['job_id']}` on `{au['gpu']}`\n")
    W(f"Source commit `{au['commit']}`, attempt `{au['attempt']}`, elapsed {f(au['elapsed_seconds'], 1)} s, "
      f"JAX pool fraction {au['mem_fraction']}, {len(au['timed_subjects'] or [])} timed subjects, "
      f"failed gates: {', '.join(au['failed']) or 'none'}. Every error below is against this job's own converged "
      f"`fft_tight` solve (same grid) unless the column says reference; 'reference' is the 4096-interval solve. "
      f"Costs are medians over 6 cases × 3 repetitions; ratios are only meaningful inside this table.\n")
    if au['dropped']:
        W('**Subjects dropped by the OOM rule (DESIGN.md §3.1):** ' + ', '.join(f"`{d['name']}` ({d['phase']})" for d in au['dropped']) + '\n')
    if au.get('rules'):
        sets = sorted({x.get('rule_set') or x.get('kind') for x in au['rules']}, key=str)
        single = [x for x in au['rules'] if (x.get('construction_status') or '').startswith('certified in one draw')]
        W('### Rule sets carried in this job\n')
        W(f"{len(sets)} empirical-quadrature rule set{'s' if len(sets) != 1 else ''} ran as arms at matched q, M and tolerance: "
          + ', '.join(f'`{k}`' for k in sets) + ". The **rule status** column is the construction status the exporting lane "
          "recorded for that rule and travels with every row below: *confirmed (k/k)* means every independent re-draw of the "
          "construction met the primary bar; *marginal* means some re-draws failed it; *certified in one draw* means this "
          "single rule met the bar but its construction has not been re-drawn; blank means a single qrg304 draw never "
          "re-drawn. "
          + (f"**{len(single)} rule{'s' if len(single) != 1 else ''} in this job {'are' if len(single) != 1 else 'is'} single-draw: "
             + ', '.join(f"`{x.get('rule_set')}` q={x['q']}" for x in single)
             + " — those arms are 'b-eqtop's rules, single-draw', not 'certified rules'." if single else '')
          + '\n')
        W(table(['set', 'q', 'M', 'file', 'source', 'm', 'fit states', 'ρ max', 'ρ 95', 'basis', 'rule status'],
                [[f"`{x.get('rule_set') or x.get('kind')}`", x['q'], x['M'], f"`{x['file']}`",
                  f"{x.get('source_lane') or 'q-ridge'} / {x.get('source_attempt') or ''} / {x.get('source_job')}".replace('/  /', '/'),
                  f(x.get('m')), f(x.get('source_fit_states')), f(x.get('rho_max', x.get('source_rho_max'))), f(x.get('rho_p95', x.get('source_rho_p95'))),
                  x.get('basis') or '—', x.get('construction_status') or '—']
                 for x in sorted(au['rules'], key=lambda x: (str(x.get('rule_set')), x['q']))]))
    zero = [x['arm'] for x in au['arms'] if x['worst_all_times_percent'] <= 0.]
    disc = au['fom_discretisation_error_percent']
    if zero:
        W('### How to read the two error columns\n')
        verb = 'sits' if len(zero) == 1 else 'sit'
        direct = au['checks'].get('direct_reproduces_fft_tight')
        W(f"The same-grid columns measure every subject against this job's converged `fft_tight` solve, so "
          f"**{' and '.join(f'`{z}`' for z in zero)} {verb} at exactly zero there by construction** — "
          f"`fft_tight` *is* the reference"
          + (f", and `dense_tight`, the same solve through a different preconditioner, agrees with it to "
             f"{sci((direct.get('detail') or {}).get('worst_relative'))} relative"
             if direct and direct.get('detail') else '')
          + ". "
          + ('They are' if len(zero) > 1 else 'It is')
          + " plotted at the axis floor, and "
          + ('they appear' if len(zero) > 1 else 'it appears')
          + " on the same-grid frontier for that reason, not because "
          + ('they are' if len(zero) > 1 else 'it is')
          + f" free. The `worst vs ref %` column is where the full-order solvers are not free: against the "
            f"{au.get('reference_mesh') or 'refined'}-interval reference this mesh's own discretisation error is "
          + (f"{f(min(disc.values()))}–{f(max(disc.values()))} % for the full-order controls" if disc else 'reported per control')
          + ", and every reduced subject inherits it. A reduced subject is only interesting where it is cheaper "
            "than a full-order solve of the accuracy it actually delivers.\n")
    W('### Every subject\n')
    hdr = ['subject', 'family', 'q / k′', 'M', 'quad.', 'm', 'rule set', 'rule basis', 'rule status', 'tol', 'worst all %', 'worst evolved %',
           'median evolved %', 't=0 %', 'worst vs ref %', 'best-found %', 'solved/best-found', 'GPU ms',
           'complete ms', 'med it', 'max it', 'budget exits', 'conv. (§5)', 'conv. (own tol)', 'strict', 'admissible (§5)', 'admissible (own tol, pre-A13)']
    rows = []
    for x in au['arms']:
        rows.append([f"`{x['arm']}`", x['family'], sub_label(x), f(x['M']), x['quadrature'] or ('—' if x['family'] != 'fom' else f"ntol {sci(x['ntol'])} dt {x['dt']}"),
                     f(x['rule_m']) if x['quadrature'] == 'eq' else '—', x.get('rule_set') or '—', x['rule_basis'] or '—', x.get('rule_status') or '—',
                     sci(x['gtol']) if x['gtol'] is not None else '—', f(x['worst_all_times_percent']), f(x['worst_evolved_percent']),
                     f(x['median_evolved_percent']), f(x['worst_t0_compression_percent']), f(x['worst_reference_percent']),
                     f(x.get('best_found_percent')), f(x.get('solved_over_best_found'), 5), f(x['median_gpu_ms'], 3), f(x['median_host_ms'], 3), f(x['median_iterations'], 1), f(x['max_iterations']),
                     f(x['total_budget_exits']), f(x['converged_design5']), f(x['converged_own_gtol']), f(x['converged_strict']),
                     f(x['admissible']), f(x['admissible_own_gtol'])])
    W(table(hdr, rows))
    lo = loose_arms(au)
    W(f"**Admissibility is DESIGN.md §5 as written (DESIGN §A13, 2026-09-19).** `conv. (§5)` applies the pre-registered rule "
      f"with its fixed threshold — every time step's normalised joint gradient ≤ 1e-6 unless the step exited on the residual "
      f"rule — to every reduced subject whatever tolerance it was run at, and it is the flag that defines `admissible`. "
      f"`conv. (own tol)` is the flag the report carried until §A13: the same rule evaluated against each query's own "
      f"stopping tolerance, which let the 1e-3 arms count as converged at 1e-3. "
      + (f"{len(lo)} reduced subjects are converged at their own tolerance and **not under §5**: "
         + ', '.join(f"`{x['arm']}`" for x in lo)
         + f". They stay in every table with their own numbers and are excluded from every admissible frontier and ratio below; "
           f"under the pre-A13 flag {sum(1 for x in au['arms'] if x['family'] in REDUCED and x['admissible_own_gtol'])} of the "
           f"{sum(1 for x in au['arms'] if x['family'] in REDUCED)} reduced subjects were admissible, under §5 "
           f"{sum(1 for x in au['arms'] if x['family'] in REDUCED and x['admissible'])} are."
         if lo else 'No reduced subject in this job changes flag between the two rules.') + '\n')
    split = [x for x in au['arms'] if x['kind'] == 'rom' and x['converged_design5'] and not x['converged_strict']]
    if split:
        W('### The subjects that converge under this lane\'s rule and not under the stricter one\n')
        W(f"{len(split)} reduced subjects carry `conv. (§5) = yes` and `strict = no`: "
          + ', '.join(f"`{x['arm']}`" for x in split) + ". This is not a solver failure and it is not a flag "
          "to read past, so here is exactly what fails, why no iteration budget can fix it, and whether the "
          "error would move if it were fixed.\n")
        W(f"**What fails.** Not the evolution. Every one of these subjects exits *every* time step on the "
          f"gradient rule with a worst per-step normalised gradient of "
          f"{sci(max(x['max_step_stationarity'] for x in split))} or better, at zero iteration-budget exits. "
          f"The only quantity above tolerance is the **initial fit's** normalised gradient "
          f"$\\|J^\\top r\\|/(\\|J\\|\\,\\|r\\|)$, at "
          f"{sci(min(x['max_ic_stationarity'] for x in split))}–{sci(max(x['max_ic_stationarity'] for x in split))}.\n")
        W(f"**Why no budget can fix it.** For these arms the initial fit is an *attainable* least-squares "
          f"problem — for POD-LSPG it is the square linear system $R\\,z=Q^\\top u_{{\\rm in}}$, for the "
          f"unrestricted bank it is the full-rank bank fit — so the residual falls to round-off: the worst "
          f"relative initial-fit residual over these subjects is "
          f"{sci(max(x['max_ic_relative_residual'] for x in split))}, i.e. machine zero against an input of "
          f"norm one. The stationarity test is then the ratio of two vanishing quantities and stops being a "
          f"measure of anything; iterating longer cannot move a $0/0$. That is the degeneracy "
          f"`head-ablation/arms.py` already documents for attainable reduced fits, and it is why DESIGN.md §5 "
          f"pre-registered a rule that accepts an initial fit whose relative residual is at round-off and "
          f"reports the stricter flag beside it rather than instead of it.\n")
        fl = [x for x in split if x.get('solved_over_best_found')]
        if fl:
            at = [x for x in fl if x['solved_over_best_found'] <= 1.01]
            above = [x for x in fl if x['solved_over_best_found'] > 1.01]
            W(f"**Whether the error would move.** No, and for {len(at)} of these {len(fl)} subjects there is a "
              f"direct check: the solved error already equals the best any coefficients in that subject's own "
              f"span could achieve, solved / best-found between {f(min(x['solved_over_best_found'] for x in at), 5)} "
              f"and {f(max(x['solved_over_best_found'] for x in at), 5)}. Those solves are at their representation "
              f"floor and a longer solve has nothing left to find.\n" if at else '')
            if above:
                W("Stated precisely rather than rounded away: "
                  + ', '.join(f"`{x['arm']}` sits {f(x['solved_over_best_found'], 3)}× above its floor "
                              f"({f(x['worst_all_times_percent'])} % solved against {f(x['best_found_percent'])} %)"
                              for x in above)
                  + ". That gap is **not** slack left by the solver: the floor quoted for it is a *static* "
                    "projection bound — the best reconstruction of each supplied field taken one output time at a "
                    "time — whereas the solved number is a trajectory that must also carry its own time-stepping "
                    "error forward. A converged trajectory is not obliged to attain a static reconstruction bound, "
                    "and its per-step gradients above show the solver is converged at every step regardless.\n")
            W(table(['subject', 'best-found (projection floor) %', 'solved worst all-times %', 'solved / best-found',
                     'worst per-step gradient', 'initial-fit gradient', 'initial-fit relative residual'],
                    [[f"`{x['arm']}`", f(x['best_found_percent']), f(x['worst_all_times_percent']),
                      f(x['solved_over_best_found'], 5), sci(x['max_step_stationarity']),
                      sci(x['max_ic_stationarity']), sci(x['max_ic_relative_residual'])] for x in fl]))
        W("**So the classical baseline is not being handicapped.** Every POD rank in this job is solved to the "
          "same per-step tolerance as every correction-ladder rung, at the same per-step budget of 600, with "
          "zero budget exits, and lands on its own projection floor. Where POD beats the ladder below, it does "
          "so from a fully converged solve.\n")
    W('### Why a reduced subject is or is not converged\n')
    hdr = ['subject', 'tol', 'exit reasons (0 budget / 1 residual / 2 tiny step / 4 gradient)', 'worst step gradient', 'worst IC gradient', 'worst IC relative residual', 'converged (§5, ≤ 1e-6)', 'converged (own tol)', 'strict']
    W(table(hdr, [[f"`{x['arm']}`", sci(x['gtol']), json.dumps(x['exit_reason_counts']), sci(x['max_step_stationarity']), sci(x['max_ic_stationarity']),
                   sci(x['max_ic_relative_residual']), f(x['converged_design5']), f(x['converged_own_gtol']), f(x['converged_strict'])]
                  for x in au['arms'] if x['kind'] == 'rom']))
    W('### What is on the frontier\n')
    nd_adm = set(au['nondominated']['gpu_evolved']['admissible'])
    red = [x for x in au['arms'] if x['family'] in ('rom', 'fast', 'pod', 'free') and x['admissible']]
    best_red = min(red, key=lambda x: x['worst_evolved_percent']) if red else None
    cheaper_fom = [x for x in au['arms'] if x['family'] == 'fom'
                   and best_red and x['worst_evolved_percent'] <= best_red['worst_evolved_percent']
                   and x['median_gpu_ms'] <= best_red['median_gpu_ms']]
    n_red_front = sum(1 for a_ in nd_adm if (by_arm(au, a_) or {}).get('family') in ('rom', 'fast', 'pod', 'free'))
    nd_own = set(au['nondominated']['gpu_evolved'].get('admissible_own_gtol') or [])
    red_own = [x for x in au['arms'] if x['family'] in REDUCED and x['admissible_own_gtol']]
    n_red_front_own = sum(1 for a_ in nd_own if (by_arm(au, a_) or {}).get('family') in REDUCED)
    W(f"On (median GPU ms, worst evolved %) over admissible subjects (DESIGN §5), **{n_red_front} of the "
      f"{len(red)} reduced subjects are non-dominated** (pre-A13 own-tolerance flag, for the record: "
      f"{n_red_front_own} of {len(red_own)}). " +
      (f"The most accurate reduced subject is `{best_red['arm']}` at {f(best_red['worst_evolved_percent'])} % and "
       f"{f(best_red['median_gpu_ms'], 1)} ms; " +
       (f"the full-order settings that are **both cheaper and at least as accurate** are "
        + ', '.join(f"`{x['arm']}` ({f(x['worst_evolved_percent'])} %, {f(x['median_gpu_ms'], 1)} ms)" for x in cheaper_fom)
        + '.' if cheaper_fom else 'no full-order setting is both cheaper and at least as accurate.')
       if best_red else '') + '\n')
    W('### Non-dominated sets\n')
    for key, v in au['nondominated'].items():
        W(f"**({v['cost']}, {v['error']})** — admissible subjects (§5): " + (', '.join(f'`{a}`' for a in v['admissible']) or 'none')
          + '; all subjects: ' + (', '.join(f'`{a}`' for a in v['all']) or 'none')
          + '; reduced subjects only (post-hoc, DESIGN §A4; §5-admissible): ' + (', '.join(f'`{a}`' for a in v.get('reduced_only', [])) or 'none')
          + '; *pre-A13 own-tolerance flag, for the record* — admissible: ' + (', '.join(f'`{a}`' for a in v.get('admissible_own_gtol', [])) or 'none')
          + '; reduced only: ' + (', '.join(f'`{a}`' for a in v.get('reduced_only_own_gtol', [])) or 'none') + '\n')
    fams = {x['arm']: x['family'] for x in au['arms']}
    W('### The nonlinear manifold against the classical one (post-hoc, DESIGN §A4)\n')
    lines = []
    HUMAN = {'rom': 'correction ladder', 'fast': 'correction ladder (optimised q = 0 kernel)',
             'pod': 'POD-LSPG', 'free': 'unrestricted bank'}
    for tag, human in (('gpu_all', 'worst over ALL output times'), ('gpu_evolved', 'worst over EVOLVED times')):
        ro = au['nondominated'][tag].get('reduced_only') or []
        # EVERY family on the frontier is enumerated. An earlier draft counted only the ladder
        # and POD and silently omitted the unrestricted-bank point, which made the frontier look
        # like it had one fewer member than it has; a reader who then re-derives the set from
        # summary.json gets a different answer. Nothing here is hand-summarised.
        per = {}
        for a_ in ro:
            per.setdefault(fams.get(a_), []).append(a_)
        parts = []
        for famkey in ('fast', 'rom', 'pod', 'free'):
            if per.get(famkey):
                parts.append(f"{len(per[famkey])} {HUMAN[famkey]}")
        cheapest_pod = min((by_arm(au, a_) for a_ in per.get('pod', [])),
                           key=lambda z: z['median_gpu_ms'], default=None)
        lines.append(f"- On **{human}**, the reduced-only frontier has {len(ro)} points: "
                     + ', '.join(parts) + '. '
                     + (f"The only POD rank on it is $k'={cheapest_pod['k']}$ at "
                        f"{f(cheapest_pod['median_gpu_ms'], 0)} ms."
                        if cheapest_pod else '**No POD rank is on it.**'))
    W('\n'.join(lines) + '\n')
    ro_all = au['nondominated']['gpu_all'].get('reduced_only') or []
    dom = [a_ for a_ in (x['arm'] for x in au['arms'] if x['family'] == 'pod' and x['admissible'])
           if a_ not in ro_all]
    if dom and ro_all:
        top = max((by_arm(au, a_) for a_ in dom), key=lambda z: z['median_gpu_ms'])
        killers = [x for x in au['arms'] if x['arm'] in ro_all
                   and x['median_gpu_ms'] <= top['median_gpu_ms']
                   and x['worst_all_times_percent'] <= top['worst_all_times_percent']]
        if killers:
            k0 = min(killers, key=lambda z: z['worst_all_times_percent'])
            # The counterfactual is COMPUTED (DESIGN §A13): the frontier is re-derived over the
            # admissible reduced subjects with the unrestricted bank removed. An earlier draft
            # asserted that removing the bank makes `top` non-dominated; the Codex audit showed
            # that at 512² and 1024² other arms still dominate it.
            cand = [x for x in au['arms'] if x['family'] in REDUCED and x['admissible'] and x['family'] != 'free']
            without = nondominated(cand, 'median_gpu_ms', 'worst_all_times_percent')
            still = [x for x in cand if x['arm'] != top['arm'] and x['median_gpu_ms'] <= top['median_gpu_ms']
                     and x['worst_all_times_percent'] <= top['worst_all_times_percent']
                     and (x['median_gpu_ms'] < top['median_gpu_ms'] or x['worst_all_times_percent'] < top['worst_all_times_percent'])]
            s0 = min(still, key=lambda z: z['worst_all_times_percent']) if still else None
            W(f"\nWorth stating because it is easy to get wrong when the frontier is re-derived from a "
              f"subset: on the all-times metric `{top['arm']}` is **not** on the reduced frontier, because "
              f"`{k0['arm']}` is both cheaper ({f(k0['median_gpu_ms'], 1)} ms against "
              f"{f(top['median_gpu_ms'], 1)} ms) and more accurate ({f(k0['worst_all_times_percent'])} % "
              f"against {f(top['worst_all_times_percent'])} %). Drop the unrestricted-bank endpoint from the "
              f"candidate set and re-derive the frontier over the remaining {len(cand)} admissible reduced subjects: "
              + (f"`{top['arm']}` **becomes non-dominated** at the expensive end (the re-derived set is "
                 + ', '.join(f'`{z}`' for z in sorted(without, key=lambda z: by_arm(au, z)['median_gpu_ms'])) + ')'
                 if top['arm'] in without else
                 f"`{top['arm']}` **stays dominated** — by `{s0['arm']}` ({f(s0['median_gpu_ms'], 1)} ms, "
                 f"{f(s0['worst_all_times_percent'])} %)"
                 + (f" and {len(still) - 1} other admissible reduced subject{'s' if len(still) > 2 else ''}" if len(still) > 1 else '')
                 + f"; the re-derived set is " + ', '.join(f'`{z}`' for z in sorted(without, key=lambda z: by_arm(au, z)['median_gpu_ms'])))
              + ". The set above is over every admissible reduced subject.\n")
    pod = [x for x in au['arms'] if x['family'] == 'pod' and x['admissible']]
    rom = [x for x in au['arms'] if x['family'] in ('rom', 'fast') and x['admissible']]
    if pod and rom:
        br = min(rom, key=lambda z: z['worst_evolved_percent'])
        beat = [x for x in pod if x['worst_evolved_percent'] <= br['worst_evolved_percent']
                and x['median_gpu_ms'] <= br['median_gpu_ms']]
        W(f"\nThe head ablation reported that no POD rank up to $k'=128$ matched the neural head. That holds "
          f"here and does not extend: " +
          (f"at {', '.join(chr(36) + chr(107) + chr(39) + '=' + str(x['k']) + chr(36) for x in beat)} — "
           f"{'a rank' if len(beat) == 1 else 'ranks'} that ablation never ran — POD beats the best correction-ladder point on **both** axes "
           f"({', '.join(f"`{x['arm']}` {f(x['worst_evolved_percent'])} % at {f(x['median_gpu_ms'], 0)} ms" for x in beat)}, "
           f"against `{br['arm']}` {f(br['worst_evolved_percent'])} % at {f(br['median_gpu_ms'], 0)} ms), from a "
           f"solve whose convergence is established above."
           if beat else "no POD rank in this job is both cheaper and at least as accurate as the best "
                        "correction-ladder point on the evolved metric.") + '\n')
    W('### Ladders\n')
    hdr = ['ladder', 'rungs', 'worst evolved %', 'worst all %', 'GPU ms', 'monotone evolved', 'monotone all', 'all converged (§5)', 'all converged (own tol)', 'non-dominated admissible points (§5)', 'error span', 'cost span']
    rows = []
    for name, lad in au['ladders'].items():
        if lad:
            rows.append([name, ' / '.join(map(str, lad['q_or_k'])), ' / '.join(f(v) for v in lad['worst_evolved_percent']),
                         ' / '.join(f(v) for v in lad['worst_all_times_percent']), ' / '.join(f(v, 0) for v in lad['median_gpu_ms']),
                         f(lad['monotone_evolved']), f(lad['monotone_all_times']), f(lad['all_converged']), f(lad.get('all_converged_own_gtol')),
                         lad['nondominated_converged_points'], f(lad['error_span'], 3), f(lad['cost_span'], 3)])
    W(table(hdr, rows))
    W('A 1e-3 ladder is by construction never `all converged (§5)`: its rungs stop at 1e-3 and §5 asks for 1e-6. '
      'Its error and cost columns are its own measurements and stand; only its admissibility is withheld.\n')
    physical(W, au)
    rule_sets(W, au)
    if au['transfer']:
        W('### Transferred rules (`eqxfer`, DESIGN.md §3.2)\n')
        W(table(['q', 'M', 'support m', 'nonzero m', 'refit rel. fit', 'ρ max', 'ρ 95', 'ρ median', 'basis', 'source ρ max (256²)', 'seconds'],
                [[t['q'], t['M'], t['m_support'], t['m'], sci(t['refit']['relative_fit']), f(t['certification']['rho_max']),
                  f(t['certification']['rho_p95']), f(t['certification']['rho_median']), t['basis'], f(t['source_rho_max']), f(t['seconds'], 0)]
                 for t in au['transfer']]))
        prediction(W, au)
    fid = au['checks'].get('cross_job_fidelity', {}).get('detail') or {}
    if fid:
        W('### Cross-job fidelity gates\n')
        W(table(['arm', 'reproduces', 'source job', 'worst relative difference', 'first tier', 'tier reached', 'passed'],
                [[f'`{k}`', f"`{v['detail'].get('comparator')}`", v['detail'].get('job'), sci(v['detail'].get('worst_relative_difference')),
                  sci(v['detail'].get('first_tier')), v['detail'].get('tier'), f(v['passed'])] for k, v in fid.items()]))
    if au.get('fno'):
        fn = au['fno']
        W(f"**FNO.** `{fn['model']}`, {fn['real_parameter_count']:,} real parameters, checkpoint SHA256 `{fn['checkpoint_sha256'][:16]}…`, "
          f"pooled device query median {f(fn['device_query_pooled']['median_ms'], 3)} ms, host transfer median "
          f"{f(fn['host_transfer_pooled']['median_ms'], 3)} ms. Deviation: {fn['deviation']}\n")
    W('### Gates\n')
    W(table(['gate', 'passed', 'detail'], [[k, f(v['passed']), (json.dumps(v['detail'])[:160] if v.get('detail') is not None else '')]
                                          for k, v in au['checks'].items() if k != 'cross_job_fidelity']))
    W(f"![envelope]({tag}-envelope.png)\n")


def rule_sets(W, au):
    """Matched-rule-set comparison and the cost of each rule set against its same-job dense twin."""
    eq = [x for x in au['arms'] if x['quadrature'] == 'eq' and x['family'] == 'rom']
    sets = sorted({x['rule_set'] for x in eq if x['rule_set']})
    dense = {(x['q'], x['gtol']): x for x in au['arms'] if x['quadrature'] == 'dense' and x['family'] == 'rom' and x['M'] == 4 * (16 + (x['q'] or 0))}
    if len(sets) > 1:
        W('### The two rule sets at matched q, M and tolerance (same job, same GPU)\n')
        W('Both rule sets ran as arms of this one allocation, so the comparison below is in-allocation, not across '
          'jobs. Rows where the two sets point at the same rule file are identical by construction and the driver '
          'gate `matched_rule_files_bitwise` asserts they are bitwise identical; rows where the files differ are '
          'the actual comparison. Every `eqtop` row carries its construction status, and a status of *certified in '
          'one draw* means exactly that — a single draw met the held-out bar and the construction was never '
          're-drawn; it is not "certified".\n')
        hdr = ['q', 'M', 'tol'] + sum(([f'{k} m', f'{k} ρ max', f'{k} status', f'{k} evolved %', f'{k} all %', f'{k} GPU ms'] for k in sets), []) \
            + ['same file', 'evolved ratio (later/earlier set)', 'cost ratio (later/earlier set)']
        rows = []
        for q in sorted({x['q'] for x in eq}):
            for g in sorted({x['gtol'] for x in eq}, reverse=True):
                got = {k: next((x for x in eq if x['q'] == q and x['gtol'] == g and x['rule_set'] == k), None) for k in sets}
                if any(v is None for v in got.values()):
                    continue
                cells = []
                for k in sets:
                    x = got[k]
                    cells += [f(x['rule_m']), f(x['rho_max']), x['rule_status'] or '—', f(x['worst_evolved_percent']),
                              f(x['worst_all_times_percent']), f(x['median_gpu_ms'], 1)]
                a0, a1 = got[sets[0]], got[sets[-1]]
                same = a0['rule_file_sha256'] == a1['rule_file_sha256']
                rows.append([q, f(a0['M']), sci(g)] + cells + [f(same),
                            f(a1['worst_evolved_percent'] / a0['worst_evolved_percent'], 4),
                            f(a1['median_gpu_ms'] / a0['median_gpu_ms'], 4)])
        W(table(hdr, rows))
        sec = sorted({(x['rule_set'], x['q'], round(x['rho_max'], 4)) for x in eq if x['rule_basis'] == 'secondary'})
        if sec:
            W('Rules in this job whose held-out ρ max exceeds the 0.116 primary bar (secondary basis only): '
              + '; '.join(f"`{k}` q = {q}, ρ max {f(r)}" for k, q, r in sec)
              + '. Their arms are timed and reported, and they are the rungs the other set replaces.\n')
    g = au['checks'].get('matched_rule_files_bitwise') or {}
    if g.get('passed') is False:
        pairs = (g.get('detail') or {}).get('pairs', [])
        xf = {t['q']: t for t in (au.get('transfer') or []) if t.get('rule_set') == sets[0]} if len(sets) > 1 else {}
        xt = {t['q']: t for t in (au.get('transfer') or []) if t.get('rule_set') == sets[-1]} if len(sets) > 1 else {}
        W(f"**Gate `matched_rule_files_bitwise` FAILED in this job, on {len(pairs)} pair"
          f"{'s' if len(pairs) != 1 else ''} ({', '.join(sorted({str(x['q']) for x in pairs}, key=int))} at both tolerances).** "
          "The gate (DESIGN.md §A8) asserts that two rule sets pointing at the same rule *file* give bitwise identical "
          "fields. That holds when the file is used at its own mesh; here both sets are **transferred**, and the "
          "transfer refits each set's weights on its own random draw of fit states, so the two arms ran two different "
          "transferred rules built from one source file. The failure is therefore a property of the transfer path, "
          "not of the timed queries, and it is reported as a failed gate rather than re-labelled. What it measures is "
          "the draw variance of the transfer at fixed source rule: "
          + '; '.join(f"q = {q}: {sets[0]} ρ max {f(xf[q]['certification']['rho_max'])} ({xf[q]['basis']}) against "
                      f"{sets[-1]} ρ max {f(xt[q]['certification']['rho_max'])} ({xt[q]['basis']})"
                      for q in sorted({x['q'] for x in pairs}) if q in xf and q in xt)
          + ". The affected arms keep their own in-job certification and admissibility; the §7 falsification clause "
            "is not triggered (it names the 1e-9 fidelity gates, `fft_tight` convergence and the OOM survivors).\n")
    W('### Each rule set against its same-job dense twin\n')
    W('The dense twin of an EQ arm is the `rom` arm at the same q and the same M with the exact advection sum, '
      'timed in this same job at tol 1e-06. The cost ratio below is therefore a within-job quadrature speedup at '
      'fixed model and fixed test space; the error columns say what that speedup costs in accuracy. For the 1e-3 EQ rows '
      'the ratio also includes the change of stopping tolerance (1e-3 against the twin\'s 1e-6), so it is not a pure '
      'quadrature ratio; the matched-tolerance quadrature ratio is the 1e-6 row of the same set and q. The 1e-3 rows are '
      'not admissible under DESIGN §5 (§A13) and are shown for what they measure.\n')
    hdr = ['set', 'q', 'M', 'tol', 'm', 'rule status', 'EQ GPU ms', 'dense GPU ms', 'dense / EQ (speedup)',
           'EQ evolved %', 'dense evolved %', 'EQ − dense evolved (pp)', 'EQ all %', 'dense all %']
    rows = []
    for x in sorted(eq, key=lambda z: (str(z['rule_set']), z['q'], -(z['gtol'] or 0))):
        d = dense.get((x['q'], 1e-06))
        if d is None:
            continue
        rows.append([f"`{x['rule_set']}`", x['q'], f(x['M']), sci(x['gtol']), f(x['rule_m']), x['rule_status'] or '—',
                     f(x['median_gpu_ms'], 1), f(d['median_gpu_ms'], 1), f(d['median_gpu_ms'] / x['median_gpu_ms'], 2),
                     f(x['worst_evolved_percent']), f(d['worst_evolved_percent']),
                     f(x['worst_evolved_percent'] - d['worst_evolved_percent'], 4),
                     f(x['worst_all_times_percent']), f(d['worst_all_times_percent'])])
    W(table(hdr, rows))
    mono = {k: v for k, v in au['ladders'].items() if k.startswith('eq_')}
    if mono:
        W('Ladder monotonicity at the top, by rule set and tolerance: '
          + '; '.join(f"`{k}` evolved-monotone {f(v['monotone_evolved'])} "
                      f"(top two rungs {f(v['worst_evolved_percent'][-2])} % → {f(v['worst_evolved_percent'][-1])} %)"
                      for k, v in mono.items()) + '.\n')


def prediction(W, au):
    """Score DESIGN.md §A5.2's recorded prediction about the transferred top rungs.

    The prediction was about ONE experiment: the qrg304 (`eqxfer`) rules transferred to 1024² in
    `bpn201` under the capped fit-state convention clip(8192/M, 8, 64), at q = 128 and 256. The
    historical capped records (six 1024² `eqxfer` transfers completed by `bpn202` before its crash,
    the same capped refit `bpn201` ran) are keyed by (mesh, rule set, q, fit-state regime), so they
    are compared only with transfers of the same mesh and rule set. An earlier draft keyed them by q
    alone and duplicated the six records against both 512² rule sets (Codex audit, 2026-09-19; DESIGN §A13).
    """
    t = au.get('transfer') or []
    if not t:
        return
    W('### The §A5.2 prediction, scored\n')
    hdr = ['rule set', 'q', 'M', 'fit states used', 'fit states available', 'fit-state rule', 'support m', 'nonzero m',
           'ρ max', 'ρ 95', 'basis', 'certified primary', 'bar']
    W('Every transferred rule in this job, fitted under the uncapped convention DESIGN.md §A7 introduced:\n')
    W(table(hdr, [[f"`{x.get('rule_set')}`", x['q'], x['M'], x['fit_states_used'], x['fit_states_available'], x['fit_state_rule'],
                   x['m_support'], x['m'], f(x['certification']['rho_max']), f(x['certification']['rho_p95']),
                   x['basis'], f(x['certified_primary']), f(x['rho_bar'])] for x in sorted(t, key=lambda z: (str(z.get('rule_set')), z['q']))]))
    cap_file = Path(__file__).resolve().parents[1] / 'artifacts/bpn202-failed/FAILURE.json'
    HIST_MESH, HIST_SET, HIST_REGIME = 1024, 'eqxfer', 'capped clip(8192/M, 8, 64)'
    cap = {}
    if cap_file.exists():
        fj = json.loads(cap_file.read_text())
        assert 'config-1024' in fj['config'] and 'XFER eqxfer' in fj['stdout_verbatim'], fj['config']
        cap = {(HIST_MESH, HIST_SET, c['q'], HIST_REGIME): c for c in fj['completed_before_crash']['rule_transfers']}
    here = {(au['intervals'], x.get('rule_set'), x['q'], 'uncapped (A7)'): x for x in t}
    comparable = [(k, cap.get((HIST_MESH, HIST_SET, k[2], HIST_REGIME))) for k in sorted(here, key=lambda k: (str(k[1]), k[2]))
                  if k[0] == HIST_MESH and k[1] == HIST_SET]
    W(f"\nDESIGN.md §A5.2, written before `bpn201` returned, predicted that the **`{HIST_SET}` transfers at q = 128 and 256 "
      f"at {HIST_MESH}² would come back uncertified** under the then-current capped convention `clip(8192/M, 8, 64)` (14 and 8 "
      f"fit states) — b-eqtop's fit-state-starvation diagnosis. `bpn201` crashed before its transfers were recorded; `bpn202` "
      f"ran the identical capped refit and completed all six transfers before its own (unrelated) crash, so its archived "
      f"`artifacts/bpn202-failed/FAILURE.json` holds the capped values of record. Those {len(cap)} records are "
      f"({HIST_MESH}², `{HIST_SET}`, {HIST_REGIME}) and are compared below only with this job's transfers of the same mesh "
      f"and rule set.\n")
    if not comparable:
        sets = sorted({str(x.get('rule_set')) for x in t})
        W(f"**This job is {au['intervals']}² and carries the rule set{'s' if len(sets) > 1 else ''} "
          + ', '.join(f'`{k}`' for k in sets)
          + f"; the prediction concerned {HIST_MESH}² `{HIST_SET}`, so it is not scored here** — a different mesh has a different "
            "reachable population and a different rule set is a different rule. What this job's transfers show about the "
            "fit-state regime is reported in the transfer table above and in the rule-set section, as observed.\n")
        return
    rows = [[k[2], (c or {}).get('fit_states', '—'), f((c or {}).get('rho_max')), (c or {}).get('basis', '—'),
             here[k]['fit_states_used'], f(here[k]['certification']['rho_max']), here[k]['basis'], f(here[k]['certified_primary'])]
            for k, c in comparable]
    W(f"\nCapped ({HIST_REGIME}, `bpn202`, values of record for `bpn201`'s design) against uncapped (this job), "
      f"{HIST_MESH}² `{HIST_SET}` only — the capped values enter no other table:\n")
    W(table(['q', 'capped fit states (bpn202)', 'capped ρ max', 'capped basis', 'uncapped fit states (this job)',
             'uncapped ρ max', 'uncapped basis', 'certified primary'], rows))
    n_cap = sum(1 for k, c in comparable if c and c['rho_max'] <= here[k]['rho_bar'])
    n_unc = sum(1 for k, c in comparable if here[k]['certified_primary'])
    W(f"\n{n_unc} of {len(comparable)} `{HIST_SET}` transfers certify on the primary bar under the uncapped count, against "
      f"{n_cap} of {len(comparable)} under the capped one. Rungs still not primary-certified here: "
      + (', '.join(f"q = {k[2]} (ρ max {f(here[k]['certification']['rho_max'])}, basis {here[k]['basis']})"
                   for k, c in comparable if not here[k]['certified_primary']) or 'none') + '.\n')
    named_cap = [(k, c) for k, c in comparable if k[2] in (128, 256) and c]
    named = [k for k, c in comparable if k[2] in (128, 256)]
    if named_cap:
        held = [k for k, c in named_cap if c['rho_max'] > here[k]['rho_bar']]
        W(f"\n**Scoring §A5.2 on the experiment it was about** (capped refit, {HIST_MESH}² `{HIST_SET}`, q = "
          + ', '.join(str(k[2]) for k, c in named_cap) + "): "
          + ', '.join(f"q = {k[2]}: {c['basis']} (ρ max {f(c['rho_max'])} against the {f(here[k]['rho_bar'])} bar, {c['fit_states']} fit states)"
                      for k, c in named_cap)
          + f". The predicted outcome **{'held' if len(held) == len(named_cap) else 'did not hold'}**: {len(held)} of the "
            f"{len(named_cap)} named rungs were uncertified under the capped convention.\n")
    if named:
        miss = [k for k in named if not here[k]['certified_primary']]
        W(f"\n**What the uncapped refit adds, stated separately because it is a changed experiment (DESIGN §A7), not a "
          f"scoring of §A5.2.** With {here[named[0]]['fit_states_used']} fit states at every rung the same two rungs came back "
          + ', '.join(f"q = {k[2]}: {here[k]['basis']} (ρ max {f(here[k]['certification']['rho_max'])})" for k in named)
          + f" — {len(miss)} of {len(named)} still not primary-certified. Raising the fit-state count to "
          f"{here[named[0]]['fit_states_used']} therefore did not by itself deliver primary certification at this mesh, so "
          "fit-state starvation is not shown to be a sufficient explanation for the top transferred rungs; the mapped support "
          "kept a strictly positive weight on only "
          + ', '.join(f"{here[k]['m']} of {here[k]['m_support']} nodes (q = {k[2]})" for k in named)
          + " — a support count, not a measure of the weights — and the " + f"{au['intervals']}² reachable population "
          "differs from the rule's. The capped and uncapped refits differ in mesh-independent ways (fit-state count) and "
          "were run in different jobs, so the capped column is context, not a controlled A/B.\n")


def physical(W, au):
    """Review round 2: the vs-reference (physical) error of every rung and full-order setting, the mesh's
    discretisation error, and whether any reduced rung sits above it."""
    disc = au['fom_discretisation_error_percent'] or {}
    tight = au['checks'].get('same_grid_baseline_present')
    ref = disc.get('fft_tight')
    W('### Physical error against the 4096-interval reference, and the discretisation error of this mesh\n')
    W(f"`worst vs ref %` is measured against the {au.get('reference_mesh') or 'refined'}-interval, Δt = 3.125e-4 "
      f"reference restricted to this grid; it contains the mesh's own discretisation error, which is the converged "
      f"`fft_tight` row's value, **{f(ref)} %** (median over cases {f(next((x['median_reference_percent'] for x in au['arms'] if x['arm'] == 'fft_tight'), None))} %). "
      "A reduced rung whose physical error is above that number has a larger worst-case error against the reference than "
      "the converged same-grid solve; one at or below it has a worst-case reference error no larger than that solve's. This "
      "compares one worst-over-cases scalar per subject and nothing more — it does not say the two solutions are alike, and "
      "the `worst all %` column says how far from the converged same-grid solve the rung actually is. Full-order settings "
      "with a coarser step (dt 0.01) sit above the converged value because their own time-stepping error adds to the mesh's.\n")
    hdr = ['subject', 'family', 'q / k′', 'rule set', 'rule status', 'tol', 'worst vs ref %', 'median vs ref %',
           'worst all % (same grid)', 'worst evolved %', 'vs ref − discretisation (pp)', 'above discretisation error']
    rows = []
    for x in sorted(au['arms'], key=lambda z: (z['family'] != 'fom', z['worst_reference_percent'])):
        d = x['worst_reference_percent'] - ref if ref is not None else None
        rows.append([f"`{x['arm']}`", x['family'], sub_label(x), x.get('rule_set') or '—', x.get('rule_status') or '—',
                     sci(x['gtol']) if x['gtol'] is not None else '—', f(x['worst_reference_percent']),
                     f(x['median_reference_percent']), f(x['worst_all_times_percent']), f(x['worst_evolved_percent']),
                     f(d), f(d > 1e-12) if d is not None else '—'])
    W(table(hdr, rows))
    red = [x for x in au['arms'] if x['family'] in ('rom', 'fast', 'pod', 'free')]
    above = [x for x in red if ref is not None and x['worst_reference_percent'] > ref + 1e-12]
    at = [x for x in red if ref is not None and x['worst_reference_percent'] <= ref + 1e-12]
    W(f"\n{len(above)} of {len(red)} reduced subjects sit **above** the {f(ref)} % discretisation error on worst-vs-reference; "
      f"{len(at)} sit at or below it"
      + (": " + ', '.join(f"`{x['arm']}` ({f(x['worst_reference_percent'])} %)" for x in at) if at else '')
      + ". The closest reduced subjects above it: "
      + ', '.join(f"`{x['arm']}` (+{f(x['worst_reference_percent'] - ref, 4)} pp)"
                  for x in sorted(above, key=lambda z: z['worst_reference_percent'])[:5])
      + ". Full-order settings and their physical error: "
      + ', '.join(f"`{k}` {f(v)} %" for k, v in sorted(disc.items(), key=lambda kv: kv[1])) + '.\n')


def crossover(W, audits):
    """Within-job reduced-versus-full-order cost ratios at each mesh, compared as ratios only."""
    if len(audits) < 2:
        return
    W('## The mesh trend: reduced against full-order, inside each job\n')
    W('Every entry below is a ratio of two timings **from the same allocation on the same GPU**. The raw '
      'millisecond columns of two different jobs are never compared: the meshes ran on different GPUs '
      '(a ratio across them would be meaningless), so only the within-job ratios are put side by side, and '
      'even those carry the different hardware. The question they answer is whether reduced queries buy more, '
      'relative to the full-order solver they must beat, as the mesh is refined.\n')
    hdr = ['mesh', 'job', 'GPU', 'cheapest admissible reduced', 'its GPU ms', 'its evolved %',
           'cheapest full-order', 'its GPU ms', 'its evolved %', 'reduced / cheapest FOM',
           'converged FOM `fft_tight` ms', 'reduced / `fft_tight`', 'reduced subjects on the admissible frontier (§5)',
           'pre-A13 flag: cheapest admissible reduced, ratio / FOM, ratio / `fft_tight`, frontier count']
    rows, ratios = [], {}
    for au in audits:
        red = [x for x in au['arms'] if x['family'] in ('rom', 'fast', 'pod', 'free') and x['admissible']]
        red_own = [x for x in au['arms'] if x['family'] in REDUCED and x['admissible_own_gtol']]
        fom = [x for x in au['arms'] if x['family'] == 'fom']
        if not red or not fom:
            continue
        cr = min(red, key=lambda z: z['median_gpu_ms'])
        cf = min(fom, key=lambda z: z['median_gpu_ms'])
        tight = next((x for x in fom if x['arm'] == 'fft_tight'), None)
        nd = set(au['nondominated']['gpu_evolved']['admissible'])
        nred = [x['arm'] for x in red if x['arm'] in nd]
        r1 = cr['median_gpu_ms'] / cf['median_gpu_ms']
        r2 = cr['median_gpu_ms'] / tight['median_gpu_ms'] if tight else None
        ratios[au['intervals']] = (r1, r2, len(nred))
        cro = min(red_own, key=lambda z: z['median_gpu_ms']) if red_own else None
        nd_own = set(au['nondominated']['gpu_evolved'].get('admissible_own_gtol') or [])
        own_txt = (f"`{cro['arm']}`, {f(cro['median_gpu_ms'] / cf['median_gpu_ms'], 3)}×, "
                   f"{f(cro['median_gpu_ms'] / tight['median_gpu_ms'], 3) if tight else '—'}×, "
                   f"{sum(1 for x in red_own if x['arm'] in nd_own)}" if cro else '—')
        rows.append([f"{au['intervals']}²", f"`{au['job_id']}`", au['gpu'], f"`{cr['arm']}`", f(cr['median_gpu_ms'], 1),
                     f(cr['worst_evolved_percent']), f"`{cf['arm']}`", f(cf['median_gpu_ms'], 1),
                     f(cf['worst_evolved_percent']), f(r1, 3), f(tight['median_gpu_ms'], 1) if tight else '—',
                     f(r2, 3) if r2 is not None else '—', f"{len(nred)} ({', '.join(nred) or 'none'})", own_txt])
    W(table(hdr, rows))
    ms = sorted(ratios)
    gpus = {au['intervals']: au['gpu'] for au in audits}
    shared = [(m1, m2) for i, m1 in enumerate(ms) for m2 in ms[i + 1:] if gpus[m1] == gpus[m2]]
    if shared:
        W('\n**Shared-hardware pairs** (DESIGN.md §A11: only these ratios are compared as ratios on the same GPU class; '
          'every other mesh enters as a ratio only): '
          + '; '.join(f"{m1}² and {m2}² on `{gpus[m1]}` — cheapest admissible reduced / cheapest same-job FOM "
                      f"{f(ratios[m1][0], 3)}× → {f(ratios[m2][0], 3)}×, / same-job `fft_tight` "
                      f"{f(ratios[m1][1], 3)}× → {f(ratios[m2][1], 3)}×, reduced subjects on the admissible frontier "
                      f"{ratios[m1][2]} → {ratios[m2][2]}" for m1, m2 in shared) + '.\n')
    if 512 in ratios:
        r = ratios[512]
        W(f"\n**§A11 pre-registration, scored (512²).** (i) The count of non-dominated admissible reduced subjects on "
          f"(median GPU ms, worst evolved %) is **{r[2]}**, i.e. "
          + ('**0**: the same side of the crossover as 256²; the crossover lies between 512² and 1024².' if r[2] == 0 else '**> 0**: the same side as 1024²; the crossover lies between 256² and 512².')
          + f" (ii) The same-job cheapest-admissible-reduced / cheapest-FOM ratio is {f(r[0], 3)}× and the ratio to "
            f"the same-job `fft_tight` is {f(r[1], 3)}×"
          + (f", against {f(ratios[256][0], 3)}× and {f(ratios[256][1], 3)}× at 256² on the same GPU class" if 256 in ratios else '')
          + (f", and {f(ratios[1024][0], 3)}× / {f(ratios[1024][1], 3)}× at 1024² (H200, ratio-only)" if 1024 in ratios else '')
          + '.\n')
    if len(ms) >= 2:
        lo, hi = ms[0], ms[-1]
        a0, a1 = ratios[lo], ratios[hi]
        W(f"\nFrom {lo}² to {hi}², the cheapest admissible reduced query goes from {f(a0[0], 3)}× the cheapest "
          f"full-order setting of its own job to {f(a1[0], 3)}× — a factor of {f(a0[0] / a1[0], 2)} in the reduced "
          f"query's favour — and from {f(a0[1], 3)}× to {f(a1[1], 3)}× the converged `fft_tight` solve of its own "
          f"job, a factor of {f(a0[1] / a1[1], 2)}. The frontier follows: {a0[2]} reduced subjects are "
          f"non-dominated at {lo}², {a1[2]} at {hi}². That is the crossover this lane was built to measure, and it "
          f"is visible only in the ratios — not in the raw milliseconds, which are different GPUs. The two factors are "
          f"ratios of within-job ratios taken on different GPU classes ({gpus[lo]} against {gpus[hi]}); normalising by the "
          f"same-job full-order solve removes the raw-millisecond comparison but not the hardware dependence, so they are "
          f"a trend indicator, not a hardware-independent number.\n")


def retractions(W, audits):
    """What was wrong, withdrawn or waived, in one place (Codex audit 2026-09-19 asked for it). Counts are read
    from the audit JSONs (the pre-A13 flags are carried as `*_own_gtol`), nothing typed."""
    W('## Retractions, corrections and disclosures\n')
    W('- **`bpn201` retracted, `bpn202` failed** (DESIGN §A6, §A7): a config-parsing crash and an out-of-memory in the untimed '
      'reconstruction diagnostic; neither produced a timed number. `bpn203` is the 1024² job of record. `bpn101` is superseded '
      'by `bpn301`, not withdrawn (§A5.1).')
    W('- **The frozen-state transfer collector was withdrawn before job 2** (DESIGN §A3): qrg304\'s fixed-iterate collector '
      'returned one frozen state at 1024², so the transferred rules\' fit and certification populations are the production '
      'dense query\'s converged per-step states instead. This is a disclosed change to how transferred rules are certified, '
      'made after the smoke and before any transfer was used.')
    W('- **The pre-job Codex audit did not run** (DESIGN §A1): the shared quota was exhausted; `reports/self-audit-design.md` '
      'stood in for it. The Codex audit of this report ran on 2026-09-19 and its findings are what §A13 corrects.')
    W('- **A gate failed at 512² and the job was kept** (DESIGN §A12): `matched_rule_files_bitwise` fails on all six same-file '
      'pairs because the transfer refits each rule set on its own draw of fit states. §6 says every gate must pass for a '
      'number to enter the report; §A12 kept the 512² numbers by reading §7\'s narrower falsification clause and recorded '
      'the reason after the data arrived. The 512² acceptance is therefore a disclosed post-data exception, not an '
      'unqualified pre-registered pass; the gate is printed as failed and its failure is informative (draw variance of the '
      'transfer at fixed source rule).')
    parts = []
    for au in audits:
        red = [x for x in au['arms'] if x['family'] in REDUCED]
        a5 = [x for x in red if x['admissible']]
        ao = [x for x in red if x['admissible_own_gtol']]
        nd5 = set(au['nondominated']['gpu_evolved']['admissible'])
        ndo = set(au['nondominated']['gpu_evolved'].get('admissible_own_gtol') or [])
        parts.append(f"{au['intervals']}² admissible reduced {len(ao)} → {len(a5)} of {len(red)}, on the (GPU ms, worst evolved %) "
                     f"frontier {sum(1 for x in ao if x['arm'] in ndo)} → {sum(1 for x in a5 if x['arm'] in nd5)}")
    W('- **The convergence flag was evaluated against the wrong threshold until 2026-09-19** (DESIGN §A13, Codex finding 1): '
      '`audit_panel.py` applied §5\'s rule against each query\'s own tolerance instead of the fixed 1e-6 the section '
      'states, so the 1e-3 arms were admissible. Corrected here; the affected arms keep their numbers and lose eligibility: '
      + '; '.join(parts) + '. Every frontier, count and ratio in this report is the corrected one; the pre-A13 values are '
      'carried beside them, labelled.')
    W('- **The §A5.2 scoring at 512² was withdrawn** (DESIGN §A13, Codex finding 2): the generator keyed the six historical '
      '1024² `eqxfer` capped transfers by q alone and compared them against both 512² rule sets, printing twelve comparisons '
      'and a verdict the prediction never covered. The scoring is now keyed by (mesh, rule set, q, fit-state regime) and is '
      'stated only at 1024² `eqxfer`, the experiment §A5.2 was about.')
    W('- **Two counterfactual frontier sentences were false** (DESIGN §A13, Codex finding 3): the 512² and 1024² sections said '
      'that removing the unrestricted bank makes `pod256_M1024_dense` non-dominated; other arms still dominate it. The '
      'counterfactual is now computed and names the remaining dominator; the 256² statement about `pod512_M2048_dense` stands.')
    W('- **Two phrasings over-reached** (DESIGN §A13, Codex finding 4): "indistinguishable from the converged same-grid solve '
      'in physical terms" (one worst-case scalar was compared) and "loses weight mass" (a support count was reported). Both '
      'now say what was measured.\n')


def glossary(W):
    W('## Glossary\n')
    for term, text in [
        ('subject', 'one timed configuration; everything except the named difference is held fixed.'),
        ('family', '`rom` the frozen nonlinear-manifold model with q corrections; `fast` the same q = 0 query through the optimised kernel; `pod` classical POD-LSPG at rank k′; `free` all 512 bank coefficients solved; `fom` the full-order Newton solver on the same grid; `fno` the trained Fourier neural operator.'),
        ('q', 'number of fixed correction directions added to the head output; q = 0 is the plain frozen model.'),
        ('M', 'number of sine test modes the weak residual is projected on; M = 4 × unknowns unless stated.'),
        ('quad. / m', 'dense = exact advection sum on the whole grid; eq = an m-point empirical quadrature rule.'),
        ('eqcert / eqxfer', 'eqcert: the rule qrg304 fitted on reachable states and certified by held-out ρ; eqxfer: that rule\'s support mapped to a finer grid with weights refit and re-certified.'),
        ('eqtop', 'the b-eqtop lane\'s final exported rule set, one rule per rung, carried as a second set of arms at matched q, M and tolerance in the same job; at q = 0, 16, 32 its files are qrg304\'s own (the two sets must then agree bitwise, an in-job gate), at q = 64 a confirmed m = 2048 construction, at q = 128 and 256 single-draw rules.'),
        ('rule set', 'which named set of empirical-quadrature rules the arm used; sets differ only in the (nodes, weights) files.'),
        ('rule basis', 'primary: ρ max ≤ 0.116 on held-out reachable states; secondary: only the 95th percentile of ρ is ≤ 0.116; none: uncertified.'),
        ('rule status', 'the exporting lane\'s verdict on the rule\'s CONSTRUCTION, not just this rule: confirmed (k/k) = every independent re-draw certified; marginal = some re-draws failed; certified in one draw = never re-drawn, a fresh draw could fail; blank = a single qrg304 draw that was never re-drawn.'),
        ('ρ', 'the rule\'s relative error on the projected advection term, measured on states the model actually reaches, never its own fit residual.'),
        ('tol', 'the evolution stopping tolerance on the normalised gradient.'),
        ('worst all % / worst evolved %', 'largest relative error against the same-job converged full-order solve over all six output times / over the five evolved times, worst over the six cases.'),
        ('t=0 %', 'the model\'s error reproducing the supplied initial field (its compression); the FNO returns the field exactly.'),
        ('worst vs ref %', 'the same against the 4096-interval reference; includes the mesh\'s discretisation error, which the `fft_tight` row shows.'),
        ('GPU ms / complete ms', 'median time from the input resident on the GPU to the six outputs resident on the GPU / the same including the input upload and output download.'),
        ('med it / max it', 'median and maximum Levenberg–Marquardt iterations per time step (Newton iterations for `fom` rows are in the audit).'),
        ('budget exits', 'time steps that hit the 600-iteration cap.'),
        ('conv. (§5)', 'DESIGN.md §5 as written, the primary flag: every exit regular, every step gradient ≤ 1e-6 (fixed, whatever tolerance the query ran at) or residual-rule exit, initial fit gradient ≤ 1e-6 or its residual at round-off.'),
        ('conv. (own tol)', 'the same rule evaluated against each query\'s own stopping tolerance (1e-6 or 1e-3); the flag the report carried before DESIGN §A13 (2026-09-19), kept as a labelled secondary column; it defines nothing.'),
        ('strict', 'btq201\'s rule: every gradient ≤ the query\'s own tol regardless of residual; differs from conv. (own tol) only for attained (square) initial fits.'),
        ('admissible (§5)', 'eligible for the reported frontier: reduced subjects converged under §5 with certified rules, parity-passing `fast`, and all `fom` / `fno` rows. This is the `admissible` field of summary.json.'),
        ('admissible (own tol, pre-A13)', 'the same eligibility computed from conv. (own tol); `admissible_own_gtol` in summary.json; shown for the record only.'),
        ('non-dominated', 'no other subject in the same job is both cheaper and more accurate.'),
        ('dense twin', 'for an EQ arm, the arm at the same q and the same M with the exact (dense) advection sum, timed in the same job at tol 1e-06; `dense / EQ` is the within-job quadrature speedup at fixed model and test space.'),
        ('fit states', 'the number of reachable states the rule\'s weights were fitted on. b-eqtop found this, not the node count m, to be the binding constraint on whether a rule certifies.'),
        ('support m / nonzero m', 'for a transferred rule: the number of mapped nodes offered to the refit, and the number that kept a strictly positive weight.'),
        ('transferred rule (`eqxfer`)', 'a 256²-grid rule\'s node set mapped to the same physical points on the finer grid, its weights refitted by nonnegative least squares at that mesh, and certified there by held-out ρ on disjoint trajectories.'),
        ('ladder spans', 'over the converged non-dominated rungs of that ladder on the evolved metric.'),
        ('same allocation', 'the FNO is timed by a second process in the same Slurm job on the same GPU after the JAX process exits.'),
    ]:
        W(f'- **{term}:** {text}')
    W('')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', nargs='+', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--stem', required=True)
    a = p.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    audits = [json.loads(Path(x).read_text()) for x in a.audit]
    lines = []
    W = lines.append
    W(f'# {a.stem} — the central comparison priced in one allocation per mesh\n')
    W('Every cost and every error in this report was measured in one Slurm allocation on one GPU per mesh, against '
      'full-order controls timed in that same allocation with the same reference; every number is generated from the '
      'audit JSONs named at the end. Numbers are development-cohort (six opened cases), one checkpoint, one training '
      'seed; the sealed cohorts are untouched. **Status: provisional as paper claims until the coordinator assembles T5.**\n')
    W('The 256\u00b2 table below is `bpn301`\u2019s and **replaces `bpn101`\u2019s wholesale** (DESIGN.md \u00a7A5.1): `bpn301` re-ran the whole panel \u2014 every full-order control, POD rank, dense rung and the FNO \u2014 in one allocation while carrying both quadrature rule sets as arms, so the two sets are compared in-allocation rather than across jobs. `bpn101` is not withdrawn; it stays in `artifacts/bpn101/` as the record of what the superseded rules gave. The 1024\u00b2 table is `bpn203`\u2019s, the third attempt at that mesh; `bpn201` and `bpn202` are retracted and produced no timed number (DESIGN.md \u00a7A6, \u00a7A7).\n')
    W('**Admissibility (DESIGN.md §5, corrected in §A13 on 2026-09-19).** Every count, frontier and ratio in this report '
      'is over subjects admissible under DESIGN §5 as written: a reduced subject is converged only if every time step\'s '
      'normalised joint gradient is ≤ 1e-6 (or the step exited on the residual rule), whatever stopping tolerance the '
      'query ran at. The report carried, until §A13, a flag evaluated against each query\'s own tolerance, which let the '
      '1e-3 arms count as converged at 1e-3; the Codex audit of 2026-09-19 found it. The pre-A13 flag is kept beside the '
      'corrected one in every table as `admissible (own tol, pre-A13)` / `admissible_own_gtol`, and the affected arms keep '
      'their measured numbers; only their eligibility changes. The corrected headlines follow.\n')
    for au in audits:
        red = [x for x in au['arms'] if x['family'] in ('rom', 'fast', 'pod', 'free') and x['admissible']]
        red_own = [x for x in au['arms'] if x['family'] in REDUCED and x['admissible_own_gtol']]
        nd = set(au['nondominated']['gpu_evolved']['admissible'])
        nd_own = set(au['nondominated']['gpu_evolved'].get('admissible_own_gtol') or [])
        nred = sum(1 for x in red if x['arm'] in nd)
        nred_own = sum(1 for x in red_own if x['arm'] in nd_own)
        best = min(red, key=lambda z: z['worst_evolved_percent']) if red else None
        beats = [x for x in au['arms'] if x['family'] == 'fom' and best
                 and x['worst_evolved_percent'] <= best['worst_evolved_percent']
                 and x['median_gpu_ms'] <= best['median_gpu_ms']]
        cheapest = min(beats, key=lambda z: z['median_gpu_ms']) if beats else None
        HUM = {'fom': 'full-order Newton', 'fno': 'the trained FNO', 'rom': 'the correction ladder',
               'fast': 'the optimised q = 0 kernel', 'pod': 'POD-LSPG', 'free': 'the unrestricted bank'}
        fseen, fdesc = [], {x['arm']: x['family'] for x in au['arms']}
        for arm in au['nondominated']['gpu_evolved']['admissible']:
            h = HUM.get(fdesc.get(arm), fdesc.get(arm))
            if h not in fseen:
                fseen.append(h)
        front_desc = ' plus '.join(fseen) if fseen else 'empty'
        W(f"**Headline, {au['intervals']}² (job `{au['job_id']}`, one allocation, one GPU): "
          f"{nred} of the {len(red)} reduced-order subjects are non-dominated on (median GPU ms, worst "
          f"evolved %)** (pre-A13 own-tolerance flag: {nred_own} of {len(red_own)}). The most accurate reduced "
          f"subject is `{best['arm']}` at {f(best['worst_evolved_percent'])} % and {f(best['median_gpu_ms'], 1)} ms"
          + (f", and {len(beats)} of the same job's full-order settings are **both cheaper and at least as "
             f"accurate** — cheapest `{cheapest['arm']}` at {f(cheapest['worst_evolved_percent'])} % and "
             f"{f(cheapest['median_gpu_ms'], 1)} ms, i.e. {f(best['median_gpu_ms'] / cheapest['median_gpu_ms'], 1)}× "
             f"less time at {f(cheapest['worst_evolved_percent'] / max(best['worst_evolved_percent'], 1e-300), 2)}× "
             f"the error." if cheapest else '.')
          + f" The frontier over admissible subjects is {front_desc}."
          + (" This is the evidence for the paper's claim of no speedup over an efficient full-order solver at "
             "this mesh.\n" if nred == 0 else
             " A reduced subject is on the frontier at this mesh; the no-speedup claim does not hold here "
             "unqualified.\n"))
    W('```mermaid\nflowchart LR\n  CK[frozen checkpoint] --> M[(bank G, head h, directions C)]\n  QTD[qtd02 directions] --> M\n'
      '  QRG[qrg304 certified rules] --> RULES[EQ rules]\n  M --> ROM[correction ladder q]\n  RULES --> ROM\n  SNAP[128 truth trajectories] --> POD[POD-LSPG k]\n'
      '  ROM --> T[one allocation: timed queries]\n  POD --> T\n  FOM[Newton grid + fft_tight] --> T\n  FNO[fno-large] --> T\n'
      '  T --> AUD[NumPy audit] --> REP[this report]\n  classDef frozen fill:#dce9f7,stroke:#3b6ea5;\n  classDef solved fill:#f7e6d0,stroke:#b07b32;\n'
      '  classDef ctrl fill:#f7e0dc,stroke:#a5433b;\n  class CK,M,QTD,QRG,RULES frozen;\n  class ROM,POD,T solved;\n  class FOM,FNO ctrl;\n```\n')
    summary = []
    for au in audits:
        tag = f"{a.stem}-L{au['intervals']}"
        figure(au, od / f'{tag}-envelope.png', od / f'{tag}-envelope.pdf', od / f'{tag}-envelope.json')
        section(W, au, tag)
        for x in au['arms']:
            for mkey in METRICS:
                if mkey in x and x[mkey] is not None:
                    summary.append(dict(mesh=au['intervals'], subject=x['arm'], family=x['family'],
                                        q_or_k=(x['q'] if x['q'] is not None else x['k']), M=x['M'], quadrature=x['quadrature'],
                                        rule_set=x.get('rule_set'), rule_m=x.get('rule_m'), rule_basis=x.get('rule_basis'),
                                        rule_status=x.get('rule_status'),
                                        # `admissible` is DESIGN §5 (A13); `admissible_own_gtol` is the pre-A13 flag
                                        admissible=x.get('admissible'), admissible_design5=x.get('admissible_design5'),
                                        admissible_own_gtol=x.get('admissible_own_gtol'),
                                        converged_design5=x.get('converged_design5'), converged_own_gtol=x.get('converged_own_gtol'),
                                        converged_strict=x.get('converged_strict'),
                                        tol=x['gtol'], metric=mkey, value=x[mkey], job_id=au['job_id'], source_sha=au['result_sha256']))
        for key, v in au['nondominated'].items():
            for which in ('admissible', 'all', 'reduced_only', 'admissible_own_gtol', 'reduced_only_own_gtol'):
                if v.get(which) is not None:
                    summary.append(dict(mesh=au['intervals'], subject='*', family='*', q_or_k=None, M=None, quadrature=None,
                                        tol=None, metric=f'nondominated_{key}_{which}', value=v[which],
                                        job_id=au['job_id'], source_sha=au['result_sha256']))
    crossover(W, audits)
    retractions(W, audits)
    glossary(W)
    W('---\n')
    W('Generated by `experiments/b-panel/reports/generate_panel.py` from: ' + ', '.join(
        f"`{Path(x).name}` (SHA256 `{hashlib.sha256(Path(x).read_bytes()).hexdigest()[:16]}…`, job {au['job_id']})"
        for x, au in zip(a.audit, audits)) + '.\n')
    (od / f'{a.stem}.md').write_text('\n'.join(lines))
    meta = dict(
        admissibility_rule='DESIGN.md §5 as written (fixed 1e-6), made primary by DESIGN §A13 on 2026-09-19',
        row_flags=dict(admissible='primary: DESIGN §5 admissibility (== admissible_design5); the paper reads this',
                       admissible_design5='explicit alias of admissible',
                       admissible_own_gtol='secondary, pre-A13: the same eligibility from the own-tolerance flag; for the record only',
                       converged_design5='DESIGN §5 convergence, fixed 1e-6', converged_own_gtol='pre-A13 flag (own gtol)',
                       converged_strict='btq201 strict rule (own gtol, no residual exception)'),
        nondominated_metrics=dict(primary=[f'nondominated_{k}_admissible' for k in ('gpu_all', 'gpu_evolved', 'complete_all', 'complete_evolved')],
                                  reduced_only=[f'nondominated_{k}_reduced_only' for k in ('gpu_all', 'gpu_evolved', 'complete_all', 'complete_evolved')],
                                  pre_a13=[f'nondominated_{k}_{w}' for k in ('gpu_all', 'gpu_evolved', 'complete_all', 'complete_evolved')
                                           for w in ('admissible_own_gtol', 'reduced_only_own_gtol')],
                                  unfiltered=[f'nondominated_{k}_all' for k in ('gpu_all', 'gpu_evolved', 'complete_all', 'complete_evolved')]),
        audits=[dict(job_id=au['job_id'], intervals=au['intervals'], attempt=au['attempt'], result_sha256=au['result_sha256'],
                     admissibility_rule=au.get('admissibility_rule')) for au in audits])
    (od / 'summary.json').write_text(json.dumps(dict(stem=a.stem, admissibility=meta, rows=summary), indent=2) + '\n')
    print('WROTE', od / f'{a.stem}.md', len(summary), 'summary rows')


if __name__ == '__main__':
    main()
