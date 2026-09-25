"""Generate reports/summary.json and reports/2026-09-25-ns3d-test.md from the pulled job JSONs.

Usage: make_report.py [--run 32=runs/t32a --run 64=runs/t64a] [--out reports]
Every number in the report comes from a JSON read here; sha256 of every input is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACC, FA, FB = 'nmrom_accurate_head_k8', 'nmrom_fast_span16_dt0.04_it2', 'nmrom_fast_span16_dt0.02_it3'
DEV_KEYS = {ACC: ('frontier', 'k8_q0_dt0.02_it3'), FA: ('span', 'span16_dt0.04_it2'),
            FB: ('span', 'span16_dt0.02_it3')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(x):
    return f'{100 * x:.3g}' if x < 0.1 else f'{100 * x:.3g}'


def sx(x):
    return f'{x:.3g}×'


def paper_dev(n):
    """The paper's development row, recomputed from the a2_h{n} JSON (fast-block timing, case 0)."""
    path = ROOT / f'experiments/ns3d-shift-head/runs/a2_h{n}/output/summary.json'
    s = json.loads(path.read_text())
    f = s['timing']['fast']
    acc = s['frontier']['k8_q0_dt0.02_it3']['stats']['evolved_worst']
    cn = {st: (s['cnab2'][st]['stats']['evolved_worst'], f[f'CNAB2_s{st}']['median_ms'])
          for st in s['cnab2'] if s['cnab2'][st]['stats'] is not None}
    cands = [st for st, (e, _) in cn.items() if e <= acc and e <= 1.0]
    fom = min(cands, key=lambda st: cn[st][1])
    row = dict(source=str(path.relative_to(ROOT)), sha256=sha(path), job=s['job_id'], gpu=s['gpu'].split(',')[0],
               cases=len(s['frontier']['k8_q0_dt0.02_it3']['errors']), timing='fast block, case 0, 7 repetitions',
               fom=f'cnab2_s{fom}', fom_worst=cn[fom][0], fom_ms=cn[fom][1], arms={})
    for arm, (sec, key) in DEV_KEYS.items():
        ms = f[f'query_{key}']['median_ms']
        st = s[sec][key]['stats']
        row['arms'][arm] = dict(worst=st['evolved_worst'], median=st['evolved_median'], ms=ms,
                                speedup=cn[fom][1] / ms, over5=st['cases_evolved_over_target'])
    return row


def job_rows(res, tim, fom):
    rows = {}
    for nm, r in res.items():
        if not r['finite']:
            rows[nm] = dict(finite=False)
            continue
        st = r['stats']
        rows[nm] = dict(finite=True, worst=st['evolved_worst'], median=st['evolved_median'],
                        over5=st['cases_evolved_over_target'], ms=tim[nm]['median_ms'],
                        speedup=r.get('speedup_vs_fom'), matched=r.get('matched_cnab2'),
                        speedup_matched=r.get('speedup_vs_matched_cnab2'), spec=r.get('spec'))
    return dict(fom=fom, fom_worst=res[fom]['stats']['evolved_worst'], fom_ms=tim[fom]['median_ms'], arms=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', default=None)
    ap.add_argument('--out', type=Path, default=HERE / 'reports')
    args = ap.parse_args()
    runs = dict(r.split('=') for r in (args.run or ['32=runs/t32a', '64=runs/t64a']))
    summary = dict(schema='ns3d-test-report-v1', meshes={})
    for n_s, rdir in sorted(runs.items(), key=lambda kv: int(kv[0])):
        n = int(n_s)
        rdir = (HERE / rdir) if not Path(rdir).is_absolute() else Path(rdir)
        sp, ap_ = rdir / 'output' / 'summary.json', rdir / 'output' / 'audit.json'
        s = json.loads(sp.read_text())
        audit = json.loads(ap_.read_text())
        test = job_rows(s['results'], s['timing']['arms'], s['fom_rule']['chosen'])
        dev = job_rows(s['dev']['results'], s['dev']['timing']['arms'], s['dev']['fom_rule']['chosen'])
        summary['meshes'][n_s] = dict(
            job=s['job_id'], gpu=s['gpu'].split(',')[0], commit=s['source_commit'], status=s['status'],
            smoke=s['smoke'], config_sha256=s['config_sha256'], summary_path=(str(sp.relative_to(HERE)) if sp.is_relative_to(HERE) else str(sp)),
            summary_sha256=sha(sp), audit_sha256=sha(ap_), audit_passed=audit['all_passed'],
            audit_perturbed_control_rejected=audit['perturbed_control_rejected'],
            audit_sampled_max_gap=max(a['sampled_max_relative_gap'] for a in audit['arms'].values()),
            test_seed=s['config']['test_seed'], test_cases=s['test_opened']['cases'], dev_cases=s['config']['dev_cases'],
            test_overlap_rows=s['test_overlap_rows'], bank_rebuild_gap=s['bank_rebuild_gap'],
            reproduction_max_gap=max(v['max_relative_gap'] for v in s['reproduction_gate']['arms'].values()),
            gates=s['gates'], timing_gates={nm: dict(drift=s['timing']['arms'][nm]['drift_ratio'],
                                                     order=s['timing']['arms'][nm]['order_ratio'])
                                            for nm in s['gates']['gated_arms']},
            timed_output_gap_max=max(r.get('timed_output_gap', 0.0) for r in s['results'].values() if r['finite']),
            test=test, dev_same_job=dev, dev_paper=paper_dev(n), elapsed_seconds=s['elapsed_seconds'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    write_md(summary, args.out / '2026-09-25-ns3d-test.md')
    print(args.out / 'summary.json', sha(args.out / 'summary.json'))


def write_md(S, path):
    L = []
    M = S['meshes']
    status = ', '.join(f"{n}³ {m['status']}" for n, m in M.items())
    L.append('# Navier–Stokes 3D Table 1 rows at 32³ and 64³ on the held-out test cases')
    L.append('')
    L.append(f'The frozen development models of the paper\'s NS 3D rows, evaluated once on the 32 held-out test '
             f'cases (seed {next(iter(M.values()))["test_seed"]}) at 32³ and 64³, beside the development values. '
             f'Status of the numbers: {status} (generated by `experiments/ns3d-test/make_report.py` from the job '
             f'JSONs; nothing hand-typed). Pre-registration: `experiments/ns3d-test/DESIGN.md`.')
    L.append('')
    L.append('## Table 1 format')
    L.append('')
    L.append('Worst = evolved worst relative error in %; × = speedup against the Table-1 FOM of the same row '
             '(fastest stable CNAB2 at least as accurate as the accurate setting, timed in the same job). '
             'Fast A = span R\'=16, Δt 0.04, 2 sweeps (the development Table 1 fast setting); '
             'fast B = span R\'=16, Δt 0.02, 3 sweeps (the setting used at 96³ on the test cases).')
    L.append('')
    L.append('| mesh | cohort (source) | accurate worst % | accurate × | fast A worst % | fast A × | fast B worst % | '
             'fast B × | FOM worst % | FOM setting | FOM ms |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for n, m in M.items():
        for label, row in (('**test**, %d cases (job %s)' % (m['test_cases'], m['job']), m['test']),
                           ('development, %d cases, same job, same protocol' % m['dev_cases'], m['dev_same_job']),
                           ('development, 16 cases, paper value (job %s)' % m['dev_paper']['job'], m['dev_paper'])):
            a = row['arms']
            L.append(f"| {n}³ | {label} | {pct(a[ACC]['worst'])} | {sx(a[ACC]['speedup'])} | "
                     f"{pct(a[FA]['worst'])} | {sx(a[FA]['speedup'])} | {pct(a[FB]['worst'])} | {sx(a[FB]['speedup'])} | "
                     f"{pct(row['fom_worst'])} | {row['fom']} | {row['fom_ms']:.2f} |")
    L.append('')
    L.append('The paper-value rows are recomputed here from the `a2_h{n}` JSONs with the same rule (their timing '
             'is the fast-block median of case 0, 7 repetitions); the other two rows use the A–B–A protocol over '
             'every case of their cohort. Compare test with the same-job development row for speedups.')
    L.append('')
    L.append('## Milliseconds, medians and outliers (test cohort)')
    L.append('')
    L.append('| mesh | arm | worst % | median-case % | cases > 5 % | median ms | × Table-1 FOM | matched CNAB2 | × matched |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for n, m in M.items():
        t = m['test']
        for nm in (ACC, FA, FB, t['fom']):
            a = t['arms'][nm]
            L.append(f"| {n}³ | `{nm}` | {pct(a['worst'])} | {pct(a['median'])} | {a['over5']} | {a['ms']:.3f} | "
                     f"{sx(a['speedup'])} | {a['matched'] or '—'} | "
                     f"{sx(a['speedup_matched']) if a['speedup_matched'] else '—'} |")
    L.append('')
    L.append('## Span ladders on the test cohort')
    L.append('')
    L.append('| mesh | Δt / sweeps | R\' | worst % | median-case % | cases > 5 % | median ms | × Table-1 FOM |')
    L.append('|---|---|---|---|---|---|---|---|')
    for n, m in M.items():
        t = m['test']['arms']
        spans = [(nm, a) for nm, a in t.items() if a.get('finite') and a['spec'] and a['spec'].get('kind') == 'span']
        spans.sort(key=lambda kv: (-kv[1]['spec']['dt'], -kv[1]['spec']['rank']))
        for nm, a in spans:
            L.append(f"| {n}³ | {a['spec']['dt']} / {a['spec']['iters']} | {a['spec']['rank']} | {pct(a['worst'])} | "
                     f"{pct(a['median'])} | {a['over5']} | {a['ms']:.3f} | {sx(a['speedup'])} |")
        bad = [nm for nm, a in t.items() if not a.get('finite')]
        if bad:
            L.append(f'| {n}³ | non-finite arms | {", ".join(bad)} | | | | | |')
    L.append('')
    L.append('## CNAB2 step ladder (test cohort)')
    L.append('')
    L.append('| mesh | steps | worst % | median ms |')
    L.append('|---|---|---|---|')
    for n, m in M.items():
        t = m['test']['arms']
        for nm, a in sorted(((k, v) for k, v in t.items() if k.startswith('cnab2_')),
                            key=lambda kv: -int(kv[0].split('_s')[1])):
            if a.get('finite'):
                L.append(f"| {n}³ | {nm.split('_s')[1]} | {pct(a['worst'])} | {a['ms']:.3f} |")
            else:
                L.append(f"| {n}³ | {nm.split('_s')[1]} | non-finite | — |")
    L.append('')
    L.append('## Gates and provenance')
    L.append('')
    L.append('| mesh | job | GPU | status | reproduction max gap | bank gap | drift/order (gated arms) | timed-output gap | '
             'audit | perturbed control rejected | test overlap rows |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for n, m in M.items():
        dr = '; '.join(f"{k.replace('nmrom_', '')}: {v['drift']:.3f}/{v['order']:.3f}" for k, v in m['timing_gates'].items())
        L.append(f"| {n}³ | {m['job']} | {m['gpu']} | {m['status']} | {m['reproduction_max_gap']:.2e} | "
                 f"{m['bank_rebuild_gap']:.1e} | {dr} | {m['timed_output_gap_max']:.1e} | "
                 f"{'pass' if m['audit_passed'] else 'FAIL'} | {m['audit_perturbed_control_rejected']} | "
                 f"{sum(m['test_overlap_rows'].values())} |")
    L.append('')
    for n, m in M.items():
        L.append(f"- {n}³: summary `{m['summary_path']}` sha256 `{m['summary_sha256']}`; audit sha256 "
                 f"`{m['audit_sha256']}`; config sha256 `{m['config_sha256']}`; commit `{m['commit']}`; "
                 f"paper-development source `{m['dev_paper']['source']}` sha256 `{m['dev_paper']['sha256']}` "
                 f"({m['dev_paper']['gpu']}); elapsed {m['elapsed_seconds']:.0f} s.")
    L.append('')
    L.append('## Caveats')
    L.append('')
    for n, m in M.items():
        t, d = m['test'], m['dev_same_job']
        if t['fom'] != d['fom']:
            ta, da = t['arms'], d['arms']
            L.append(f"- **{n}³: the Table-1 FOM changes between cohorts.** On the test cases `{t['fom']}` "
                     f"({pct(ta[t['fom']]['worst'])} %) is at least as accurate as the accurate setting "
                     f"({pct(ta[ACC]['worst'])} %), so it becomes the comparator; on the development cases it was "
                     f"not ({pct(da[t['fom']]['worst'])} % vs {pct(da[ACC]['worst'])} %, ratio "
                     f"{da[t['fom']]['worst'] / da[ACC]['worst']:.4f}), and `{d['fom']}` was used. Against "
                     f"`{d['fom']}` on the test cases the accurate setting would be "
                     f"{sx(ta[d['fom']]['ms'] / ta[ACC]['ms'])}; the rule gives {sx(ta[ACC]['speedup'])}.")
        if t['arms'][ACC]['speedup'] < 1:
            L.append(f"- {n}³: on the test cases the accurate setting is slower than the rule FOM "
                     f"({sx(t['arms'][ACC]['speedup'])}).")
    L.append('- The test cohort (seed 202609221) is the one already evaluated four times at 96³; its parameters are '
             'mesh-independent. It was never generated at 32³/64³ and no setting at these meshes was chosen on it '
             '(DESIGN.md §4). The fast A setting was picked on 2026-09-24 from development records, after the 96³ '
             'test results existed.')
    L.append('- Speedups are same-job ratios. The paper-value development rows use a different timing protocol '
             '(fast block, case 0) on a different GPU; the same-job development row is the like-for-like reference.')
    L.append('- CNAB2 is pseudo-spectral and does no linear solve; it is the Table-1 FOM for this problem, not an '
             'iterative solver.')
    L.append('- The audit is restricted: exact on saved full-field cases, sampled estimates (diagnostic) elsewhere; '
             'the truth solver is the same CNAB2 code and is not independently re-solved.')
    L.append('')
    L.append('## Glossary')
    L.append('')
    L.append('- **accurate setting** — the head: a small network mapping k=8 unknowns (plus 3 frame-translation '
             'unknowns) to the 64 bank coefficients, solved by 3 Gauss–Newton sweeps per step of Δt 0.02.')
    L.append('- **fast A / fast B** — the linear span of the first R\'=16 importance-ordered bank columns, stepped '
             'with Δt 0.04 and 2 sweeps (A) or Δt 0.02 and 3 sweeps (B).')
    L.append('- **span R\'** — the same linear solve with the first R\' ordered bank columns.')
    L.append('- **bank** — the rank-64 POD basis of energy-centred training snapshots, used in a frame that moves '
             'with the flow structure.')
    L.append('- **CNAB2 / FOM** — the full-order pseudo-spectral solver; "steps" over T=0.2. The Table-1 FOM is '
             'its fastest stable setting at least as accurate as the accurate setting, in the same job.')
    L.append('- **worst %** — per case the largest relative L2 error (relative to the initial field\'s norm) over '
             'the five output times after t=0, then the largest over cases. **median-case %** — the median over '
             'cases of that per-case value. **cases > 5 %** — how many cases exceed 5 %.')
    L.append('- **×** — FOM median time divided by the arm\'s median time, both from the same job and cohort. '
             '**matched CNAB2** — the fastest CNAB2 setting at least as accurate as that arm.')
    L.append('- **test / development cohort** — 32 held-out cases never used for a choice at these meshes / 16 '
             'cases on which k=8 and the fast settings were chosen.')
    L.append('- **A–B–A timing** — interleaved rounds, then arm-major rounds, then interleaved rounds; '
             '**drift** = median(A2)/median(A1), **order** = median(B)/median(A1∪A2), both gated to within 10 %.')
    L.append('- **reproduction gap** — largest relative difference between this job\'s development error rows and '
             'those of the original development job (gate 1e-6). **bank gap** — rebuilt bank vs stored probe.')
    L.append('- **timed-output gap** — difference between outputs produced during timing and the accuracy pass.')
    L.append('- **perturbed control** — the audit must fail on a copy with one field changed by 1e-6.')
    path.write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
