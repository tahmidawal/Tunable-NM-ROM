"""tensor-crossover: assemble results/crossover.json, the Figure-3 PNG(s) and RESULTS.md from the pulled result JSONs.
Nothing is typed by hand. python make_figure.py [job ...]   (default: every results/<job>/result.json that is complete)
Crossover rule: PROTOCOL.md §6.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
import os
RES = Path(os.environ.get('TCX_RESULTS', HERE / 'results'))
OUTDIR = Path(os.environ.get('TCX_OUT', HERE))
COL = dict(tensor='#2a78d6', quad_P='#eb6834', quad_S='#1baf7a', quad_FP='#eda100')
MK = dict(tensor='s', quad_P='o', quad_S='^', quad_FP='D')
FAMNAME = {2: dict(P='Fibonacci lattice', S='Gauss tensor'), 3: dict(P='CBC rank-1 lattice', S='Gauss tensor')}


def load(jobs):
    out = []
    for j in jobs:
        p = RES / j / 'result.json'
        d = json.loads(p.read_text())
        if not d.get('complete'):
            print(f'skip incomplete {j}')
            continue
        d['_job'] = j
        out.append(d)
    return out


def crossover(widths, ratio, sep):
    """smallest tested width from which quadrature wins (ratio>1 and separated) at it and every larger width."""
    win = [bool(r is not None and r > 1 and s) for r, s in zip(ratio, sep)]
    for k in range(len(widths)):
        if all(win[k:]):
            if k == 0:
                return dict(width=widths[0], bracket=None, interp=None, verdict='quadrature cheaper throughout the tested range')
            r0, r1 = ratio[k - 1], ratio[k]
            interp = None
            if r0 and r1 and 0 < r0 <= 1 < r1 and np.isfinite(r1):
                t = (0 - np.log(r0)) / (np.log(r1) - np.log(r0))
                interp = float(np.exp(np.log(widths[k - 1]) + t * (np.log(widths[k]) - np.log(widths[k - 1]))))
            return dict(width=widths[k], bracket=[widths[k - 1], widths[k]], interp=interp,
                        verdict=f'crossover in ({widths[k - 1]}, {widths[k]}]')
    return dict(width=None, bracket=None, interp=None, verdict='no crossover in the tested range')


def table_of(d):
    dim = d['dim']
    rows = []
    widths = sorted(int(r) for r in d['widths'])
    for r in widths:
        W = d['widths'][str(r)]
        tq, tk = W['timing_query']['summary'], W['timing_kernel']['summary']
        row = dict(r=r, M=W['M'], rules=W['rules_used'], m=W['m'], oom=W['oom'],
                   how={f: q['how'] for f, q in W['qualification'].items()},
                   tau={f: q['tau'] for f, q in W['qualification'].items()},
                   tensor_formula=W['op_bytes_tensor_formula'], bytes=W['op_bytes'], shared_bytes=W['shared_bytes'], tensor_build_s=W['tensor_build_seconds'],
                   G_tab=W['G_tab'], G_eq={k: dict(field=v['field_max_rel'], coef=v['coef_max_rel'],
                                                   lm_it_diff=v['lm_iteration_diff_total'], passed=v['passed'])
                                           for k, v in W['G_eq'].items()},
                   query={k: {x: v[x] for x in ('median_ms', 'q1_ms', 'q3_ms', 'min_ms', 'max_ms', 'n', 'block_drift')}
                          for k, v in tq.items()},
                   kernel={k: {x: v[x] for x in ('per_eval_us', 'per_eval_q1_us', 'per_eval_q3_us', 'n', 'block_drift')}
                           for k, v in tk.items()},
                   timed_output_max_diff=W['timing_query']['max_output_diff'],
                   memory={k: (v or {}).get('device_bytes') for k, v in W['memory_analysis'].items()},
                   device_peak=W.get('device_after_width'))
        # frozen-rule rows are only timed when they differ from the selected rule
        for k2, k1 in (('quad_FP', 'quad_P'), ('quad_FS', 'quad_S')):
            if k2 not in row['query'] and k1 in row['query']:
                row['query'][k2] = row['query'][k1]
                row['kernel'][k2] = row['kernel'][k1]
        rows.append(row)
    cross = {}
    for arm in ('P', 'S', 'FP'):
        q = f'quad_{arm}'
        tens_b = [x['bytes'].get('tensor_P', x['tensor_formula']) if not x['oom'] else None for x in rows]
        rb = [(tb / x['bytes'][q]) if tb else np.inf for tb, x in zip(tens_b, rows)]
        cross[f'storage_{arm}'] = dict(ratio=rb, **crossover(widths, rb, [True] * len(rows)))
        for meas, key, lo, hi, med in (('kernel', 'kernel', 'per_eval_q1_us', 'per_eval_q3_us', 'per_eval_us'),
                                       ('query', 'query', 'q1_ms', 'q3_ms', 'median_ms')):
            ratio, sep = [], []
            for x in rows:
                T = x[key].get('tensor_P')
                Qd = x[key].get(q)
                if T is None:                       # OOM tensor counts as a quadrature win
                    ratio.append(np.inf); sep.append(True)
                    continue
                ratio.append(T[med] / Qd[med])
                sep.append(bool(Qd[hi] < T[lo]))
            cross[f'{meas}_{arm}'] = dict(ratio=ratio, separated=sep, **crossover(widths, ratio, sep))
    return dict(job=d['_job'], job_id=d['job_id'], gpu=d['gpu'], commit=d['commit'], dim=dim, mesh=d['mesh'],
                elapsed_s=d['elapsed_seconds'], gates=d['gates'], rows=rows, crossover=cross)


def plot(tabs, path, title):
    fig, axs = plt.subplots(len(tabs), 3, figsize=(13.5, 4.1 * len(tabs)), squeeze=False)
    for i, t in enumerate(tabs):
        rows = t['rows']
        w = [x['r'] for x in rows]
        lab = '2D' if t['dim'] == 2 else '3D'
        mesh = f"{t['mesh']}²" if t['dim'] == 2 else f"{t['mesh'] - 1}³"
        fn = FAMNAME[t['dim']]
        series = [('tensor', 'tensor_P', 'precomputed tensor'),
                  ('quad_P', 'quad_P', f"quadrature, {fn['P']} (qualified per width)"),
                  ('quad_S', 'quad_S', f"quadrature, {fn['S']} (qualified per width)"),
                  ('quad_FP', 'quad_FP', "quadrature, frozen sealed-closeout rule at every width (2D fib987, 3D lat4096)")]
        for c, (meas, ylab) in enumerate((('bytes', 'stored advection operator (GB)'),
                                          ('kernel', 'one nonlinear-term evaluation (µs)'),
                                          ('query', 'full reduced query (ms)'))):
            ax = axs[i, c]
            for key, src, name in series:
                xs, ys, lo, hi = [], [], [], []
                for x in rows:
                    if meas == 'bytes':
                        v = x['bytes'].get(src)
                        if v is None:
                            continue
                        xs.append(x['r']); ys.append(v / 1e9)
                    else:
                        v = x[meas].get(src)
                        if v is None:
                            continue
                        f = 'per_eval' if meas == 'kernel' else None
                        m_, a_, b_ = ((v['per_eval_us'], v['per_eval_q1_us'], v['per_eval_q3_us']) if f
                                      else (v['median_ms'], v['q1_ms'], v['q3_ms']))
                        xs.append(x['r']); ys.append(m_); lo.append(m_ - a_); hi.append(b_ - m_)
                ls = '--' if key == 'quad_FP' else '-'
                ax.plot(xs, ys, ls, color=COL[key], marker=MK[key], ms=6, lw=2, label=name)
                if lo:
                    ax.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor=COL[key], elinewidth=1, capsize=2)
            for x in rows:
                if x['oom']:
                    ax.axvline(x['r'], color='#e34948', lw=1, ls=':')
                    ax.text(x['r'], ax.get_ylim()[1], 'tensor OOM', color='#52514e', fontsize=7, rotation=90, va='top')
            cr = t['crossover'][f"{'storage' if meas == 'bytes' else meas}_P"]
            ax.set_xscale('log', base=2); ax.set_yscale('log')
            ax.set_xticks(w); ax.set_xticklabels([str(v) for v in w])
            ax.set_xlabel('bank width r'); ax.set_ylabel(ylab)
            ax.grid(True, which='major', color='#e6e5e1', lw=0.6); ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.set_title(f"{lab} {mesh}: {cr['verdict']}", fontsize=9, color='#0b0b0b')
            if cr.get('interp'):
                ax.axvline(cr['interp'], color='#52514e', lw=0.8, ls='-.')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gb(b):
    return f'{b / 1e9:.3f}' if b is not None else '–'


def results_md(tabs):
    L = ['# tensor-crossover: when off-mesh quadrature of the Burgers advection term saves storage or time', '',
         'Generated by `make_figure.py` from `results/<job>/result.json` (data: `results/crossover.json`); no number is typed by hand. '
         'Pre-registration: `PROTOCOL.md`. Development cases only (2D `dev6`, 3D `dev4`); no test/sealed cohort used. '
         'Status: **final for the jobs listed**.', '']
    L += ['![Figure 3](fig3_crossover.png)', '', '## Crossover widths (primary rule family, PROTOCOL §6)', '',
          '| dim | mesh | job | GPU | storage | nonlinear-term time | full-query time |', '|---|---|---|---|---|---|---|']
    for t in tabs:
        c = t['crossover']
        f = lambda k: c[k]['verdict'] + (f" (log-interp. r≈{c[k]['interp']:.0f})" if c[k].get('interp') else '')
        L.append(f"| {t['dim']}D | {t['mesh']} | {t['job']} ({t['job_id']}) | {t['gpu']} | {f('storage_P')} | {f('kernel_P')} | {f('query_P')} |")
    L += ['', 'Secondary (Gauss family) and sensitivity (frozen rule at every width):', '',
          '| dim | mesh | arm | storage | nonlinear-term time | full-query time |', '|---|---|---|---|---|---|']
    for t in tabs:
        for arm, nm in (('S', 'Gauss qualified'), ('FP', 'frozen primary rule')):
            c = t['crossover']
            L.append(f"| {t['dim']}D | {t['mesh']} | {nm} | {c[f'storage_{arm}']['verdict']} | {c[f'kernel_{arm}']['verdict']} | {c[f'query_{arm}']['verdict']} |")
    for t in tabs:
        L += ['', f"## {t['dim']}D, mesh {t['mesh']} — job {t['job']} ({t['job_id']}, {t['gpu']}, commit `{(t['commit'] or '')[:9]}`)", '',
              f"Gates: rules G-R {t['gates']['G_R']['passed']}; Jacobian equality G-tab all widths {t['gates']['G_tab_all']}; "
              f"trajectory equality G-eq all widths {t['gates']['G_eq_all']}; timed outputs vs accuracy pass, max rel diff "
              f"{t['gates']['timed_outputs']:.1e}.", '',
              '| r | M | primary rule (m, how) | Gauss rule (m, how) | tensor GB | quad GB (primary) | quad GB (Gauss) | ratio | tensor µs/eval | quad µs/eval (primary) | ratio | tensor query ms [IQR] | quad query ms [IQR] | ratio | G-eq field (P / S) | LM-iteration diff |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for x in t['rows']:
            tb = x['bytes'].get('tensor_P')
            qt, kt_ = x['query'].get('tensor_P'), x['kernel'].get('tensor_P')
            qp, kp = x['query']['quad_P'], x['kernel']['quad_P']
            ge = x['G_eq']
            L.append(
                f"| {x['r']} | {x['M']} | `{x['rules']['P']}` ({x['m']['P']}, {list(x['how'].values())[0]}) | "
                f"`{x['rules']['S']}` ({x['m']['S']}, {list(x['how'].values())[1]}) | "
                f"{gb(tb) if not x['oom'] else 'OOM'} | {gb(x['bytes']['quad_P'])} | {gb(x['bytes']['quad_S'])} | "
                f"{(tb / x['bytes']['quad_P']) if tb else float('inf'):.2f} | "
                f"{kt_['per_eval_us']:.1f} | {kp['per_eval_us']:.1f} | {kt_['per_eval_us'] / kp['per_eval_us']:.2f} | "
                f"{qt['median_ms']:.2f} [{qt['q1_ms']:.2f}–{qt['q3_ms']:.2f}] | {qp['median_ms']:.2f} [{qp['q1_ms']:.2f}–{qp['q3_ms']:.2f}] | "
                f"{qt['median_ms'] / qp['median_ms']:.2f} | {ge['P']['field']:.1e} / {ge['S']['field']:.1e} | "
                f"{ge['P']['lm_it_diff']} / {ge['S']['lm_it_diff']} |")
        L += ['', 'Gauss family and frozen-rule timings (median; µs per nonlinear-term evaluation, ms per query), device memory of the compiled query (XLA memory analysis, GB), tensor build time (offline):', '',
              '| r | Gauss quad µs | Gauss quad query ms | frozen-rule quad µs | frozen-rule query ms | tensor query device GB | quad query device GB | tensor build s | process peak GB |',
              '|---|---|---|---|---|---|---|---|---|']
        for x in t['rows']:
            mem = x['memory']
            pk = (x['device_peak'] or {}).get('peak_bytes_in_use')
            L.append(f"| {x['r']} | {x['kernel']['quad_S']['per_eval_us']:.1f} | {x['query']['quad_S']['median_ms']:.2f} | "
                     f"{x['kernel']['quad_FP']['per_eval_us']:.1f} | {x['query']['quad_FP']['median_ms']:.2f} | "
                     f"{gb(mem.get('query_tensor_P'))} | {gb(mem.get('query_quad_P'))} | "
                     f"{x['tensor_build_s'].get('P', float('nan')):.2f} | {gb(pk) if pk and pk > 0 else '–'} |")
    L += ['', '## Reading notes', '',
          '- Both paths evaluate the same quadrature term, so their trajectories agree to round-off (G-eq column) and equal LM iteration counts; the comparison is cost at identical accuracy by construction. The tensor\'s storage and contraction cost depend only on its shape (M×r×r), not on which rule or stencil produced it, so the tensor curves also stand for the incumbent mesh-stencil tensor of the October lanes.',
          '- The full-query ratio can favour quadrature at a smaller width than the kernel ratio: the solver\'s predictor evaluates the nonlinear *value* (not the Jacobian) at three candidates per step; quadrature gets the value in O(mM) work, the tensor needs a full M×r×r contraction.',
          '- At the frozen width the frozen rule is used even where the suffix rule on these dev cases would pick another rung (field `suffix_selection` in the result JSON); see the qualification rows.',
          '- "process peak GB" is the job-wide maximum so far, dominated by the qualification ladder (largest rules), not by either timed path; use the per-executable device GB columns to compare paths.',
          '- No tensor ran out of memory on the 80 GB A100 in the tested range; the bank has no columns beyond 512, so wider banks were not tested.',
          '- Times are for one GPU type (A100 80GB PCIe), float64, highest matmul precision, development cases only; ratios on other hardware will differ (the tensor is memory-bandwidth bound at large r).']
    L += ['', '## Glossary', '',
          '- **r (bank width)**: number of reduced coefficients; the first r columns of the ordered 512-column bank.',
          '- **M**: number of sine test functions (≈4r).', '- **m**: number of quadrature points of the rule.',
          '- **tensor**: precomputed M×r×r coefficient array of the same quadrature term; contracting it with c gives the Jacobian.',
          '- **quad (quadrature)**: the term evaluated from the bank and its derivatives at the m rule points.',
          '- **primary / Gauss rule**: rule of the frozen family (2D Fibonacci lattice, 3D CBC lattice) / Gauss tensor family, '
          'qualified per width on development cases (PROTOCOL §4); "frozen" = the sealed-closeout rule; "ladder-floor" = smallest rung on the ladder qualified, so the true minimum may be smaller.',
          '- **frozen-rule sensitivity**: the frozen primary rule used at every width (fixed m).',
          '- **stored GB**: bytes of the advection operator arrays only (tensor: T; quadrature: B, D, P); shared arrays are excluded.',
          '- **µs/eval**: median time of one Jacobian+value evaluation, from a scan of K chained evaluations, 5 interleaved repeats.',
          '- **query ms [IQR]**: median wall time of one full reduced query (projection, all time steps, decoding) over dev cases × 5 repeats, with the 25th–75th percentile range.',
          '- **ratio**: tensor ÷ quadrature; above 1 means quadrature is cheaper.',
          '- **separated**: quadrature 75th percentile below tensor 25th percentile.',
          '- **G-eq field**: max relative difference of the decoded fields between the tensor and quadrature paths (same rule); registered tolerance 1e-8.',
          '- **device GB**: argument + output + temporary bytes XLA reports for the compiled query; **process peak**: largest device memory in use so far in the job.',
          '- **OOM**: tensor did not fit in device memory.']
    return '\n'.join(L) + '\n'


def main():
    jobs = sys.argv[1:] or sorted(p.parent.name for p in RES.glob('*/result.json'))
    ds = load(jobs)
    tabs = [table_of(d) for d in ds]
    tabs.sort(key=lambda t: (t['dim'], t['mesh']))
    (RES / 'crossover.json').write_text(json.dumps(tabs, indent=1, default=float))
    prim = [t for t in tabs if (t['dim'], t['mesh']) in ((2, 256), (3, 65))]
    sec = [t for t in tabs if t not in prim]
    if prim:
        plot(prim, OUTDIR / 'fig3_crossover.png', 'Burgers off-mesh quadrature vs precomputed tensor: storage and time vs bank width (A100 80GB, float64)')
    if sec:
        plot(sec, OUTDIR / 'fig3_crossover_finer_meshes.png', 'Same comparison on finer meshes')
    (OUTDIR / 'RESULTS.md').write_text(results_md(tabs))
    print('wrote crossover.json, figures, RESULTS.md for', [t['job'] for t in tabs])


if __name__ == '__main__':
    main()
