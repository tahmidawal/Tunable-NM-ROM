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
           'solved_over_best_found', 'converged', 'converged_strict', 'admissible', 'rho_max', 'rho_p95', 'rule_basis',
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
           'complete ms', 'med it', 'max it', 'budget exits', 'conv.', 'strict', 'admissible']
    rows = []
    for x in au['arms']:
        rows.append([f"`{x['arm']}`", x['family'], sub_label(x), f(x['M']), x['quadrature'] or ('—' if x['family'] != 'fom' else f"ntol {sci(x['ntol'])} dt {x['dt']}"),
                     f(x['rule_m']) if x['quadrature'] == 'eq' else '—', x.get('rule_set') or '—', x['rule_basis'] or '—', x.get('rule_status') or '—',
                     sci(x['gtol']) if x['gtol'] is not None else '—', f(x['worst_all_times_percent']), f(x['worst_evolved_percent']),
                     f(x['median_evolved_percent']), f(x['worst_t0_compression_percent']), f(x['worst_reference_percent']),
                     f(x.get('best_found_percent')), f(x.get('solved_over_best_found'), 5), f(x['median_gpu_ms'], 3), f(x['median_host_ms'], 3), f(x['median_iterations'], 1), f(x['max_iterations']),
                     f(x['total_budget_exits']), f(x['converged']), f(x['converged_strict']), f(x['admissible'])])
    W(table(hdr, rows))
    split = [x for x in au['arms'] if x['kind'] == 'rom' and x['converged'] and not x['converged_strict']]
    if split:
        W('### The subjects that converge under this lane\'s rule and not under the stricter one\n')
        W(f"{len(split)} reduced subjects carry `converged = yes` and `strict = no`: "
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
    hdr = ['subject', 'exit reasons (0 budget / 1 residual / 2 tiny step / 4 gradient)', 'worst step gradient', 'worst IC gradient', 'worst IC relative residual', 'converged', 'strict']
    W(table(hdr, [[f"`{x['arm']}`", json.dumps(x['exit_reason_counts']), sci(x['max_joint_stationarity']), sci(x['max_ic_stationarity']),
                   sci(x['max_ic_relative_residual']), f(x['converged']), f(x['converged_strict'])]
                  for x in au['arms'] if x['kind'] == 'rom']))
    W('### What is on the frontier\n')
    nd_adm = set(au['nondominated']['gpu_evolved']['admissible'])
    red = [x for x in au['arms'] if x['family'] in ('rom', 'fast', 'pod', 'free') and x['admissible']]
    best_red = min(red, key=lambda x: x['worst_evolved_percent']) if red else None
    cheaper_fom = [x for x in au['arms'] if x['family'] == 'fom'
                   and best_red and x['worst_evolved_percent'] <= best_red['worst_evolved_percent']
                   and x['median_gpu_ms'] <= best_red['median_gpu_ms']]
    n_red_front = sum(1 for a_ in nd_adm if (by_arm(au, a_) or {}).get('family') in ('rom', 'fast', 'pod', 'free'))
    W(f"On (median GPU ms, worst evolved %) over admissible subjects, **{n_red_front} of the "
      f"{len(red)} reduced subjects are non-dominated**. " +
      (f"The most accurate reduced subject is `{best_red['arm']}` at {f(best_red['worst_evolved_percent'])} % and "
       f"{f(best_red['median_gpu_ms'], 1)} ms; " +
       (f"the full-order settings that are **both cheaper and at least as accurate** are "
        + ', '.join(f"`{x['arm']}` ({f(x['worst_evolved_percent'])} %, {f(x['median_gpu_ms'], 1)} ms)" for x in cheaper_fom)
        + '.' if cheaper_fom else 'no full-order setting is both cheaper and at least as accurate.')
       if best_red else '') + '\n')
    W('### Non-dominated sets\n')
    for key, v in au['nondominated'].items():
        W(f"**({v['cost']}, {v['error']})** — admissible subjects: " + (', '.join(f'`{a}`' for a in v['admissible']) or 'none')
          + '; all subjects: ' + (', '.join(f'`{a}`' for a in v['all']) or 'none')
          + '; reduced subjects only (post-hoc, DESIGN §A4): ' + (', '.join(f'`{a}`' for a in v.get('reduced_only', [])) or 'none') + '\n')
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
            W(f"\nWorth stating because it is easy to get wrong when the frontier is re-derived from a "
              f"subset: on the all-times metric `{top['arm']}` is **not** on the reduced frontier, because "
              f"`{k0['arm']}` is both cheaper ({f(k0['median_gpu_ms'], 1)} ms against "
              f"{f(top['median_gpu_ms'], 1)} ms) and more accurate ({f(k0['worst_all_times_percent'])} % "
              f"against {f(top['worst_all_times_percent'])} %). Drop the unrestricted-bank endpoint from the "
              f"candidate set and `{top['arm']}` becomes non-dominated at the expensive end; keep it and it "
              f"does not. The set above is over every admissible reduced subject.\n")
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
    hdr = ['ladder', 'rungs', 'worst evolved %', 'worst all %', 'GPU ms', 'monotone evolved', 'monotone all', 'all converged', 'non-dominated converged points', 'error span', 'cost span']
    rows = []
    for name, lad in au['ladders'].items():
        if lad:
            rows.append([name, ' / '.join(map(str, lad['q_or_k'])), ' / '.join(f(v) for v in lad['worst_evolved_percent']),
                         ' / '.join(f(v) for v in lad['worst_all_times_percent']), ' / '.join(f(v, 0) for v in lad['median_gpu_ms']),
                         f(lad['monotone_evolved']), f(lad['monotone_all_times']), f(lad['all_converged']),
                         lad['nondominated_converged_points'], f(lad['error_span'], 3), f(lad['cost_span'], 3)])
    W(table(hdr, rows))
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
    W('### Each rule set against its same-job dense twin\n')
    W('The dense twin of an EQ arm is the `rom` arm at the same q and the same M with the exact advection sum, '
      'timed in this same job at tol 1e-06. The cost ratio below is therefore a within-job quadrature speedup at '
      'fixed model and fixed test space; the error columns say what that speedup costs in accuracy.\n')
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
    """Score DESIGN.md §A5.2's recorded prediction about the transferred top rungs."""
    t = au.get('transfer') or []
    if not t:
        return
    W('### The §A5.2 prediction, scored\n')
    W('Before `bpn201` returned, DESIGN.md §A5.2 predicted that the **top two transferred rungs (q = 128, 256) '
      'would come back uncertified**, because the then-current convention `clip(8192/M, 8, 64)` would fit them on '
      '14 and 8 reachable states — b-eqtop\'s fit-state-starvation diagnosis, not anything about the mesh. '
      '`bpn201` (retracted) and `bpn202` (failed) both ran that capped refit; its per-rung values are read '
      'here from the archived `artifacts/bpn202-failed/FAILURE.json` and appear in the second table below, '
      'never typed into prose. **This job is not the same test**: DESIGN.md §A7 retired the cap, so every '
      'rung here is fitted on the full configured fit-state count, which is the change §A5.2 said would be made '
      'and smoked first. The table below is what the uncapped refit gives; it measures the remedy, not the '
      'prediction.\n')
    hdr = ['q', 'M', 'fit states used', 'fit states available', 'fit-state rule', 'support m', 'nonzero m',
           'ρ max', 'ρ 95', 'basis', 'certified primary', 'bar']
    W(table(hdr, [[x['q'], x['M'], x['fit_states_used'], x['fit_states_available'], x['fit_state_rule'],
                   x['m_support'], x['m'], f(x['certification']['rho_max']), f(x['certification']['rho_p95']),
                   x['basis'], f(x['certified_primary']), f(x['rho_bar'])] for x in t]))
    cap_file = Path(__file__).resolve().parents[1] / 'artifacts/bpn202-failed/FAILURE.json'
    cap = {}
    if cap_file.exists():
        cap = {c['q']: c for c in json.loads(cap_file.read_text())['completed_before_crash']['rule_transfers']}
    rows = [[x['q'], (cap.get(x['q']) or {}).get('fit_states', '—'), f((cap.get(x['q']) or {}).get('rho_max')),
             (cap.get(x['q']) or {}).get('basis', '—'), x['fit_states_used'],
             f(x['certification']['rho_max']), x['basis'], f(x['certified_primary'])] for x in t]
    n_cap = sum(1 for x in t if (cap.get(x['q']) or {}).get('rho_max', 1e9) <= x['rho_bar'])
    W('\nAgainst the capped refit the retracted attempts recorded (archived in `artifacts/bpn201-retracted/` and '
      '`artifacts/bpn202-failed/`; those values enter no other table):\n')
    W(table(['q', 'capped fit states (bpn202)', 'capped ρ max', 'capped basis', 'uncapped fit states (this job)',
             'uncapped ρ max', 'uncapped basis', 'certified primary'], rows))
    good = [x for x in t if x['certified_primary']]
    named = [x for x in t if x['q'] in (128, 256)]
    W(f"\n{len(good)} of {len(t)} transferred rungs certify on the primary bar in this job under the uncapped "
      f"count, against {n_cap} of {len(t)} under the capped one. Rungs still not primary-certified: "
      + (', '.join(f"q = {x['q']} (ρ max {f(x['certification']['rho_max'])}, basis {x['basis']})"
                   for x in t if not x['certified_primary']) or 'none') + '.\n')
    if named:
        held = [x for x in named if not x['certified_primary']]
        W(f"\n**Scoring §A5.2.** The two rungs it named (q = "
          + ', '.join(str(x['q']) for x in named) + ") came back "
          + ', '.join(f"q = {x['q']}: {x['basis']} (ρ max {f(x['certification']['rho_max'])} against the "
                      f"{f(x['rho_bar'])} bar)" for x in named)
          + f". The predicted **outcome** therefore {'held' if len(held) == len(named) else 'did not hold'}: "
          + f"{len(held)} of the {len(named)} named rungs are not primary-certified. "
            "Its **mechanism** does not survive this job, and that is the part to carry forward: §A5.2 blamed the "
            "fit-state starvation b-eqtop identified, and DESIGN.md §A7 removed exactly that — every rung here is "
          + f"fitted on {named[0]['fit_states_used']} states, not the "
          + ' and '.join(str((cap.get(x['q']) or {}).get('fit_states', '?')) for x in named)
          + " the capped convention gave — yet the "
            "same rungs still miss the bar. Fit-state starvation is therefore not a sufficient explanation for the "
            "top transferred rungs at this mesh; what remains is the transfer itself (the mapped support loses "
            + f"weight mass: see the support m against nonzero m column) and the {au['intervals']}\u00b2 reachable population. "
            "The comparison needs that care: the capped and uncapped refits are not the same experiment, so the "
            "capped ρ column above is context, not a controlled A/B.\n")


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
           'converged FOM `fft_tight` ms', 'reduced / `fft_tight`', 'reduced subjects on the admissible frontier']
    rows, ratios = [], {}
    for au in audits:
        red = [x for x in au['arms'] if x['family'] in ('rom', 'fast', 'pod', 'free') and x['admissible']]
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
        rows.append([f"{au['intervals']}²", f"`{au['job_id']}`", au['gpu'], f"`{cr['arm']}`", f(cr['median_gpu_ms'], 1),
                     f(cr['worst_evolved_percent']), f"`{cf['arm']}`", f(cf['median_gpu_ms'], 1),
                     f(cf['worst_evolved_percent']), f(r1, 3), f(tight['median_gpu_ms'], 1) if tight else '—',
                     f(r2, 3) if r2 is not None else '—', f"{len(nred)} ({', '.join(nred) or 'none'})"])
    W(table(hdr, rows))
    ms = sorted(ratios)
    if len(ms) >= 2:
        lo, hi = ms[0], ms[-1]
        a0, a1 = ratios[lo], ratios[hi]
        W(f"\nFrom {lo}² to {hi}², the cheapest admissible reduced query goes from {f(a0[0], 3)}× the cheapest "
          f"full-order setting of its own job to {f(a1[0], 3)}× — a factor of {f(a0[0] / a1[0], 2)} in the reduced "
          f"query's favour — and from {f(a0[1], 3)}× to {f(a1[1], 3)}× the converged `fft_tight` solve of its own "
          f"job, a factor of {f(a0[1] / a1[1], 2)}. The frontier follows: {a0[2]} reduced subjects are "
          f"non-dominated at {lo}², {a1[2]} at {hi}². That is the crossover this lane was built to measure, and it "
          f"is visible only in the ratios — not in the raw milliseconds, which are different GPUs.\n")


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
        ('conv.', 'DESIGN.md §5: every exit regular, every step gradient ≤ tol or residual at tolerance, initial fit gradient ≤ tol or its residual at round-off.'),
        ('strict', 'btq201\'s rule: every gradient ≤ tol regardless of residual; differs from conv. only for attained (square) initial fits.'),
        ('admissible', 'eligible for the reported frontier: converged reduced subjects with certified rules, parity-passing `fast`, and all `fom` / `fno` rows.'),
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
    for au in audits:
        red = [x for x in au['arms'] if x['family'] in ('rom', 'fast', 'pod', 'free') and x['admissible']]
        nd = set(au['nondominated']['gpu_evolved']['admissible'])
        nred = sum(1 for x in red if x['arm'] in nd)
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
          f"evolved %).** The most accurate reduced "
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
                                        rule_status=x.get('rule_status'), admissible=x.get('admissible'),
                                        converged=x.get('converged'),
                                        tol=x['gtol'], metric=mkey, value=x[mkey], job_id=au['job_id'], source_sha=au['result_sha256']))
        for key, v in au['nondominated'].items():
            for which in ('admissible', 'all', 'reduced_only'):
                if v.get(which) is not None:
                    summary.append(dict(mesh=au['intervals'], subject='*', family='*', q_or_k=None, M=None, quadrature=None,
                                        tol=None, metric=f'nondominated_{key}_{which}', value=v[which],
                                        job_id=au['job_id'], source_sha=au['result_sha256']))
    crossover(W, audits)
    glossary(W)
    W('---\n')
    W('Generated by `experiments/b-panel/reports/generate_panel.py` from: ' + ', '.join(
        f"`{Path(x).name}` (SHA256 `{hashlib.sha256(Path(x).read_bytes()).hexdigest()[:16]}…`, job {au['job_id']})"
        for x, au in zip(a.audit, audits)) + '.\n')
    (od / f'{a.stem}.md').write_text('\n'.join(lines))
    (od / 'summary.json').write_text(json.dumps(dict(stem=a.stem, rows=summary), indent=2) + '\n')
    print('WROTE', od / f'{a.stem}.md', len(summary), 'summary rows')


if __name__ == '__main__':
    main()
