"""Generate the hires-burgers report and `summary.json` from the audit summaries ONLY.

    python reports/generate_report.py --out reports/2026-09-2x-hires-burgers checks/hb2k01-summary.json [more ...]

No number is typed by hand: every table cell is read from an audit summary written by
`audit_hires.py`, whose SHA256 is recorded in the output `summary.json`.
"""
import argparse
import hashlib
import json
from pathlib import Path


def f(x, d=3):
    return '—' if x is None else f'{x:.{d}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summaries', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--status', default='provisional')
    a = p.parse_args()
    S = [(Path(x), json.loads(Path(x).read_text())) for x in a.summaries]
    S.sort(key=lambda t: t[1]['intervals'])
    L = []
    w = L.append
    w('# hires-burgers — frozen Burgers NM-ROM against Newton–BiCGStab above $1024^2$')
    w('')
    w(f'Same-allocation accuracy and cost of the frozen 2D Burgers NM-ROM (one checkpoint, $K=16$, $R=512$, trained at '
      f'$256^2$) transferred without retraining to the meshes below. **Status: {a.status}.** Development cases only '
      f'(six; the sealed cohort is unopened); one checkpoint, one seed. Every number is generated from the audit JSONs listed at the end.')
    w('')
    w('```mermaid')
    w('flowchart LR')
    w('  classDef frozen fill:#dbeafe,stroke:#1e40af,color:#111;')
    w('  classDef solved fill:#dcfce7,stroke:#166534,color:#111;')
    w('  classDef fom fill:#fee2e2,stroke:#991b1b,color:#111;')
    w('  U0["supplied dense field u0 (L x L)"] --> IC["initial fit: 48x48 Gauss samples, K+q unknowns"]:::solved')
    w('  IC --> LM["50 implicit steps: block-damped LM on M weak tests, advection sampled at m nodes"]:::solved')
    w('  G["bank G (n x 512), head h, directions C"]:::frozen --> LM')
    w('  LM --> DEC["decode six fields: G (h(z) + C y)"]:::frozen')
    w('  U0 --> FOM["Newton-BiCGStab, FFT Helmholtz preconditioner, n unknowns"]:::fom')
    w('  DEC --> ERR["same-grid error vs fft_tight"]')
    w('  FOM --> ERR')
    w('```')
    w('')
    w('Error: $\\epsilon_k = \\lVert u_k - u^{\\mathrm{tight}}_k \\rVert_2 / \\lVert u_0 \\rVert_2$ on the full grid; '
      '*evolved* $=\\max_{k\\ge 1}\\epsilon_k$, *all-times* $=\\max_{k\\ge 0}\\epsilon_k$; worst over the six cases. '
      'Speedup $S = T_{\\mathrm{FOM}} / T_{\\mathrm{ROM}}$ from median GPU times of the same job.')
    out_summary = dict(status=a.status, sources=[], verdicts={}, headline=[])
    for path, s in S:
        Lm = s['intervals']
        out_summary['sources'].append(dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                           attempt=s['attempt'], job_id=s['job_id'], commit=s['commit'], gpu=s['nvidia_smi'],
                                           result_sha256=s['result_sha256'], failed_gates=s['failed_gates']))
        w('')
        w(f"## ${Lm}^2$ — attempt `{s['attempt']}`, job `{s['job_id']}`, {', '.join(s['nvidia_smi']) or s['gpu']}")
        w('')
        w(f"**Cohort: {s.get('cohort') or 'dev6 (six opened development cases)'} — {s.get('cohort_cases', 6)} cases.** Every number in this section is on this cohort only.")
        w('')
        w(f"Source commit `{s['commit']}`; elapsed {f(s.get('elapsed_seconds'), 0)} s; failed audit gates: "
          f"{', '.join(s['failed_gates']) or 'none'}; dropped: {', '.join(d['name'] for d in s['dropped']) or 'none'}.")
        w('')
        w('### Bar verdict')
        w('')
        w('| setting | arm | worst evolved % | worst all-times % | GPU ms | stalled exits / steps | $S$ vs tight | $S$ vs relaxed passing | $S$ vs fastest tested FOM at least as accurate | bar ($\\le$ limit and $S\\ge5$) |')
        w('|---|---|---|---|---|---|---|---|---|---|')
        for label in ('accurate_1_percent', 'stretch_half_percent'):
            v = s['verdict'].get(label, {})
            if 'arm' not in v:
                w(f"| {label} | — | — | — | — | — | — | — | — | **not met**: {v.get('reason', '')} |")
                continue
            sp = v['speedups']
            w(f"| {label} | `{v['arm']}` | {f(v['worst_evolved_percent'], 4)} | {f(v['worst_all_times_percent'], 4)} | {f(v['median_gpu_ms'], 2)} | "
              f"{v['stalled_exits']} / {v['steps']} | {f(sp['vs_tight'], 2)} (`{v['tight_comparator']}`) | "
              f"{f(sp['vs_relaxed_passing'], 2)} (`{v['relaxed_passing_comparator']}`) | "
              f"{f(sp['vs_fastest_tested_fom_at_least_as_accurate'], 2)} (`{v['fastest_tested_fom_at_least_as_accurate']}`) | "
              f"tight: {'MET' if v['met_vs_tight'] else 'not met'}; relaxed: {'MET' if v['met_vs_relaxed_passing'] else 'not met'}; "
              f"matched: {'MET' if v['met_vs_fastest_tested_at_least_as_accurate'] else 'not met'} |")
            out_summary['headline'].append(dict(mesh=Lm, setting=label, **v))
        v = s['verdict'].get('fast_setting')
        if v:
            w(f"| fast ($q=0$) | `{v['arm']}` | {f(v['worst_evolved_percent'], 4)} | — | {f(v['median_gpu_ms'], 2)} | — | {f(v['vs_tight'], 2)} | "
              f"{f(v['vs_relaxed_passing'], 2)} | {f(v['vs_fastest_tested_fom_at_least_as_accurate'], 2)} | reported beside the accurate rung |")
            out_summary['headline'].append(dict(mesh=Lm, setting='fast_setting', **v))
        out_summary['verdicts'][str(Lm)] = s['verdict']
        w('')
        w('### Every arm')
        w('')
        w('| arm | family | q | M | m | rule | tol | worst evolved % | worst all-times % | GPU ms | host ms | it/step | stalled | retries | ρ_max held-out | ρ_max deployed | certified | vs refined ref % |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for n, t in sorted(s['table'].items(), key=lambda kv: (kv[1]['family'], kv[1]['median_gpu_ms'])):
            if t['family'] == 'rom':
                w(f"| `{n}` | rom | {t['q']} | {t['M']} | {t['m']} | {t['rule']}{' (control)' if t.get('control') else ''} | {t['gtol']:g} | "
                  f"{f(t['worst_evolved_percent'], 4)} | {f(t['worst_all_times_percent'], 4)} | {f(t['median_gpu_ms'], 2)} | {f(t['median_host_ms'], 2)} | "
                  f"{f(t['median_iterations_per_step'], 1)} | {t['stalled_exits']} / {t['steps']} | {t['damping_retries']} | "
                  f"{f(t.get('rho_max_heldout'), 4)} | {f(t.get('rho_max_deployed'), 4)} | {t.get('certified_primary')} | {f(t.get('worst_reference_evolved_percent'), 3)} |")
            else:
                w(f"| `{n}` | fom {t.get('impl') or ''} mesh {t['mesh']} | — | — | — | ntol {t['ntol']:g}, ltol {t['ltol']:g}, dt {t['dt']:g} | — | "
                  f"{f(t['worst_evolved_percent'], 4)} | {f(t['worst_all_times_percent'], 4)} | {f(t['median_gpu_ms'], 2)} | {f(t['median_host_ms'], 2)} | — | "
                  f"{t['stalled_steps']} / {t['steps']} | — | — | — | — | {f(t.get('worst_reference_evolved_percent'), 3)} |")
        w('')
        w(f"Same-grid truth (`fft_tight`) against the refined reference, worst evolved: {f(s.get('truth_vs_refined_reference_evolved_percent'), 3)} % "
          '(the discretisation error of this mesh; no reduced arm can be more physical than this).')
        w('')
        w('### Dense truth (exact advection, same solver, tolerance 1e-6) and the deployed arms against it')
        w('')
        w('| rung | case | worst evolved % | iterations | seconds | deployed arm − dense (relative, per arm) |')
        w('|---|---|---|---|---|---|')
        for d in s['dense_truth']:
            dv = '; '.join(f"`{k}` {v:.2e}" for k, v in sorted(d['deployed_vs_dense'].items()))
            w(f"| `{d['name']}` | {d['case']} | {f(100 * d['same_grid_evolved'], 4)} | {d['total_iterations']} | {f(d['seconds'], 1)} | {dv} |")
        w('')
        w('### Quadrature rules (certified only by held-out ρ; the NNLS fit residual is never a certificate)')
        w('')
        w('| q | M | rule | m | NNLS fit (not a certificate) | ρ_max fit states | ρ_max held-out | ρ_95 held-out | argmax state | primary (≤ bar) | control |')
        w('|---|---|---|---|---|---|---|---|---|---|---|')
        for r in s['rules']:
            h = r['heldout_population']
            w(f"| {r['q']} | {r['M']} | {r['rule']} | {r['m']} | {f((r['refit'] or {}).get('relative_fit'), 6)} | {f(r['fit_population']['rho_max'], 4)} | "
              f"{f(h['rho_max'], 4)} | {f(h['rho_p95'], 4)} | {h['argmax']} | {h['certified_primary']} | {r.get('control', False)} |")
        w('')
        w('### Parity of the optimised kernel against the audited path (same rule, same tolerance)')
        w('')
        w('| fast arm | audited twin | worst relative field difference | integers identical | passed (≤ 1e-9 and integers) |')
        w('|---|---|---|---|---|')
        for x in s['parity']:
            if x.get('covered'):
                w(f"| `{x['fast']}` | `{x['base']}` | {x['worst_relative']:.3e} | {x['integers_identical']} | {x['passed']} |")
            else:
                w(f"| `{x['fast']}` | none in this job | — | — | not covered |")
        w('')
        w('### Profile of the accurate rung (case 0, medians of 7; micro-kernels medians of 30)')
        w('')
        w('| arm | whole query ms | initial fit ms | evolve ms | decode ms | LM iterations | retries | ms per iteration | (r, J) ms | residual ms | Gram ms | Gram + solve ms |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for x in s['profile']:
            w(f"| `{x['arm']}` | {f(x['whole_query_ms'], 2)} | {f(x['initialize_ms'], 2)} | {f(x['evolve_ms'], 2)} | {f(x['decode_ms'], 2)} | "
              f"{x['iterations']} | {x['retries']} | {f(x['evolve_ms_per_iteration'], 3)} | {f(x['evalJ_ms'], 3)} | {f(x['residual_ms'], 3)} | "
              f"{f(x['gram_ms'], 3)} | {f(x['gram_plus_solve_ms'], 3)} |")
    w('')
    w('## Glossary')
    w('')
    for k, v in [
        ('FOM', 'full-order model: backward-Euler upwind finite differences on the full grid, Newton iterations with FFT-preconditioned BiCGStab. `fft_tight` is its converged setting and the truth every same-grid error is measured against.'),
        ('audited / lean FOM', 'two implementations of the same solver; the lean one omits a per-Newton diagnostic and is the FOM at its best. Speedups use the faster of the two.'),
        ('relaxed passing', 'the cheapest tested FOM setting that still converges every step and stays within 0.1 % of `fft_tight`.'),
        ('q', 'number of linear correction directions added to the neural head; q = 0 is the fast setting.'),
        ('M / m', 'number of sine test functions in the weak residual / number of quadrature nodes at which the advection term is sampled.'),
        ('rule', '`scaled`: b-eqtop 256² nodes at the same physical points with weights × (L/256)²; `xfer`: same nodes, weights refit by non-negative least squares; `lat64`: uniform 63×63 lattice, equal weights; `bad0`: control rule expected to fail.'),
        ('ρ', 'relative error of the sampled weak advection term on one state; ρ_max over held-out reachable states is the only certificate (bar 0.116). "Deployed" = on the states that arm itself visits on held-out trajectories.'),
        ('tol', 'stationarity tolerance of the per-step Levenberg–Marquardt solve.'),
        ('stalled', 'ROM: steps that ended on budget, tiny step or rejection instead of the stationarity/residual test. FOM: time steps whose Newton residual missed its tolerance.'),
        ('retries', 'rejected LM trial steps (damping increases).'),
        ('evolved / all-times', 'worst error over output times after t = 0 / including t = 0, where the ROM returns its own compression of the supplied field.'),
        ('vs refined ref', 'error against an 8192² solve at Δt/16 restricted to this grid: the physical error.'),
        ('parity', 'agreement of the optimised kernel with the audited one on output fields and on the integer iteration and exit vectors.'),
    ]:
        w(f'- **{k}** — {v}')
    w('')
    w('## Sources')
    w('')
    for x in out_summary['sources']:
        w(f"- `{x['file']}` SHA256 `{x['sha256']}` (attempt `{x['attempt']}`, job `{x['job_id']}`, commit `{x['commit']}`, result.json `{x['result_sha256']}`)")
    out = Path(a.out)
    out.with_suffix('.md').write_text('\n'.join(L) + '\n')
    out_summary['report_sha256'] = hashlib.sha256(out.with_suffix('.md').read_bytes()).hexdigest()
    (out.parent / 'summary.json').write_text(json.dumps(out_summary, indent=1) + '\n')
    print(out.with_suffix('.md'))


if __name__ == '__main__':
    main()
