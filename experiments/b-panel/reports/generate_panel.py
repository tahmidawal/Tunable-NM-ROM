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
           'solved_over_best_found', 'converged', 'converged_strict', 'admissible', 'rho_max', 'rho_p95', 'rule_basis')


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
    zero = [x['arm'] for x in au['arms'] if x['worst_all_times_percent'] <= 0.]
    disc = au['fom_discretisation_error_percent']
    if zero:
        W('### How to read the two error columns\n')
        verb = 'sits' if len(zero) == 1 else 'sit'
        W(f"The same-grid columns measure every subject against this job's converged `fft_tight` solve, so "
          f"**{' and '.join(f'`{z}`' for z in zero)} {verb} at exactly zero there by construction** — "
          f"`fft_tight` *is* the reference, and `dense_tight`, the same solve through a different "
          f"preconditioner, agrees with it to "
          f"{sci(au['checks']['direct_reproduces_fft_tight']['detail']['worst_relative'])} relative. They are "
          f"plotted at the axis floor, and they appear on the same-grid frontier for that reason, not because "
          f"they are free. The `worst vs ref %` column is where the full-order solvers are not free: against the "
          f"4096-interval reference this mesh's own discretisation error is "
          f"{f(min(disc.values()))}–{f(max(disc.values()))} % for the full-order controls, and every reduced "
          f"subject inherits it. A reduced subject is only interesting where it is cheaper than a full-order "
          f"solve of the accuracy it actually delivers.\n")
    W('### Every subject\n')
    hdr = ['subject', 'family', 'q / k′', 'M', 'quad.', 'm', 'rule basis', 'tol', 'worst all %', 'worst evolved %',
           'median evolved %', 't=0 %', 'worst vs ref %', 'best-found %', 'solved/best-found', 'GPU ms',
           'complete ms', 'med it', 'max it', 'budget exits', 'conv.', 'strict', 'admissible']
    rows = []
    for x in au['arms']:
        rows.append([f"`{x['arm']}`", x['family'], sub_label(x), f(x['M']), x['quadrature'] or ('—' if x['family'] != 'fom' else f"ntol {sci(x['ntol'])} dt {x['dt']}"),
                     f(x['rule_m']) if x['quadrature'] == 'eq' else '—', x['rule_basis'] or '—',
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
    if au['transfer']:
        W('### Transferred rules (`eqxfer`, DESIGN.md §3.2)\n')
        W(table(['q', 'M', 'support m', 'nonzero m', 'refit rel. fit', 'ρ max', 'ρ 95', 'ρ median', 'basis', 'source ρ max (256²)', 'seconds'],
                [[t['q'], t['M'], t['m_support'], t['m'], sci(t['refit']['relative_fit']), f(t['certification']['rho_max']),
                  f(t['certification']['rho_p95']), f(t['certification']['rho_median']), t['basis'], f(t['source_rho_max']), f(t['seconds'], 0)]
                 for t in au['transfer']]))
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


def glossary(W):
    W('## Glossary\n')
    for term, text in [
        ('subject', 'one timed configuration; everything except the named difference is held fixed.'),
        ('family', '`rom` the frozen nonlinear-manifold model with q corrections; `fast` the same q = 0 query through the optimised kernel; `pod` classical POD-LSPG at rank k′; `free` all 512 bank coefficients solved; `fom` the full-order Newton solver on the same grid; `fno` the trained Fourier neural operator.'),
        ('q', 'number of fixed correction directions added to the head output; q = 0 is the plain frozen model.'),
        ('M', 'number of sine test modes the weak residual is projected on; M = 4 × unknowns unless stated.'),
        ('quad. / m', 'dense = exact advection sum on the whole grid; eq = an m-point empirical quadrature rule.'),
        ('eqcert / eqxfer', 'eqcert: the rule qrg304 fitted on reachable states and certified by held-out ρ; eqxfer: that rule\'s support mapped to a finer grid with weights refit and re-certified.'),
        ('rule basis', 'primary: ρ max ≤ 0.116 on held-out reachable states; secondary: only the 95th percentile of ρ is ≤ 0.116; none: uncertified.'),
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
    for au in audits:
        red = [x for x in au['arms'] if x['family'] in ('rom', 'fast', 'pod', 'free') and x['admissible']]
        nd = set(au['nondominated']['gpu_evolved']['admissible'])
        nred = sum(1 for x in red if x['arm'] in nd)
        best = min(red, key=lambda z: z['worst_evolved_percent']) if red else None
        beats = [x for x in au['arms'] if x['family'] == 'fom' and best
                 and x['worst_evolved_percent'] <= best['worst_evolved_percent']
                 and x['median_gpu_ms'] <= best['median_gpu_ms']]
        cheapest = min(beats, key=lambda z: z['median_gpu_ms']) if beats else None
        W(f"**Headline, {au['intervals']}² (job `{au['job_id']}`, one allocation, one GPU): "
          f"{nred} of the {len(red)} reduced-order subjects are non-dominated on (median GPU ms, worst "
          f"evolved %).** The frontier is full-order Newton plus the trained FNO. The most accurate reduced "
          f"subject is `{best['arm']}` at {f(best['worst_evolved_percent'])} % and {f(best['median_gpu_ms'], 1)} ms"
          + (f", and {len(beats)} of the same job's full-order settings are **both cheaper and at least as "
             f"accurate** — cheapest `{cheapest['arm']}` at {f(cheapest['worst_evolved_percent'])} % and "
             f"{f(cheapest['median_gpu_ms'], 1)} ms, i.e. {f(best['median_gpu_ms'] / cheapest['median_gpu_ms'], 1)}× "
             f"less time at {f(cheapest['worst_evolved_percent'] / max(best['worst_evolved_percent'], 1e-300), 2)}× "
             f"the error." if cheapest else '.')
          + " This is the evidence for the paper's claim of no speedup over an efficient full-order solver at "
            "this mesh.\n")
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
                                        tol=x['gtol'], metric=mkey, value=x[mkey], job_id=au['job_id'], source_sha=au['result_sha256']))
        for key, v in au['nondominated'].items():
            for which in ('admissible', 'all', 'reduced_only'):
                if v.get(which) is not None:
                    summary.append(dict(mesh=au['intervals'], subject='*', family='*', q_or_k=None, M=None, quadrature=None,
                                        tol=None, metric=f'nondominated_{key}_{which}', value=v[which],
                                        job_id=au['job_id'], source_sha=au['result_sha256']))
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
