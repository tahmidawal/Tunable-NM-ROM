"""Generate the lab-log entry for this lane from the audited JSONs.

    python make_lab_entry.py --r1 checks/qrg101-audit.json --r2 checks/qrg201-audit.json \
        --report reports/2026-09-16-q-ridge.md --out checks/2026-09-16-lab-log-entry.md

Every number in the entry comes from the audits, the comparator table or the artifact
manifests. The entry is then appended verbatim to the canonical `LAB-LOG.md` on `main`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CELL = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def fmt(x, d=4):
    return '—' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out) + '\n'


def ladder_block(aud, kind):
    lines = []
    for quad in ('dense', 'eq'):
        lines.append(f'\n*{quad} quadrature:*\n')
        rows = []
        for label, e in sorted(aud['ladders'][quad].items(),
                               key=lambda kv: (kv[1]['lam_rel'], kv[1]['rule'])):
            head = (f'$\\lambda_{{rel}}={label}$' if kind == 'lam'
                    else f"$M={ {'m4': 4, 'm8': 8, 'm16': 16}[label] }(K+q)$")
            rows.append([head,
                         ' / '.join(fmt(v) for v in e['worst_evolved_percent']),
                         ' / '.join(fmt(v) for v in e['worst_all_times_percent']),
                         ' / '.join(fmt(v, 0) for v in e['median_gpu_ms']),
                         yn(e['monotone_evolved']), yn(e['all_converged']),
                         yn(e['cost_within_1p5x']), yn(e['all_times_not_raised']),
                         yn(e['passes'])])
        lines.append(table(
            ['setting', f"worst evolved % at q = {', '.join(map(str, aud['ladders'][quad][list(aud['ladders'][quad])[0]]['q']))}",
             'worst all-times %', 'median GPU ms', 'monotone', 'converged', 'cost ≤1.5x',
             'all-times not raised', 'passes'], rows))
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r1', required=True)
    p.add_argument('--r2', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r1 = json.loads(Path(a.r1).read_text())
    r2 = json.loads(Path(a.r2).read_text())
    W = []

    fp = {}
    for aud in (r1, r2):
        for quad, v in ((aud.get('r3') or {}).get('falsification_pairs') or {}).items():
            if v:
                fp[quad] = v
    falsified = bool(fp) and not any(v['held_out_higher_at_high_q'] for v in fp.values())
    passing = {f"{aud['mode'].upper()}/{quad}": aud['verdict'][quad]['passing']
               for aud in (r1, r2) for quad in ('dense', 'eq')}
    any_pass = any(v for k, v in passing.items() if k.endswith('/eq'))

    headline = ('the held-out weak residual is NOT higher at $q=16$, so test-space '
                'overfitting is FALSIFIED as the cause of the evolved-times regression'
                if falsified else
                'the held-out weak residual IS higher at $q=16$, so test-space overfitting '
                'survives its own control')
    W.append(f'### q-ridge — {headline}; the regression lives in the empirical quadrature, '
             f"and {'some' if any_pass else 'no'} remedy removes it\n")

    W.append(
        'The coordinator asked whether the Burgers $256^2$ correction ladder\'s '
        'non-monotonicity on the worst-over-evolved-times metric ($q=16$ worse than $q=0$) is '
        'the extra unknowns **overfitting the $M$ weak test equations**. Three probes, each '
        'isolated, on the frozen checkpoint `18f0266ae6f0…`, the same six opened development '
        'cases, the **old** direction rule, and `b-ladder-top`\'s budget-600 block-damped '
        'variable-projection contract: **R1** a field-metric ridge $\\lambda\\|y\\|_W^2$ on the '
        'correction block with $\\lambda=\\lambda_{rel}\\sigma_q^2$ dimensionless; **R2** more '
        'tests at fixed $q$, $M\\in\\{4,8,16\\}(K+q)$; **R3** the control, the exactly '
        'integrated weak residual on held-out test modes. Predeclared protocol and every '
        'amendment: `experiments/q-ridge/DESIGN.md`. Nothing was merged or pushed.\n')

    W.append(
        f"Worktree `worktrees/2026-09-16-q-ridge`, branch `exp/2026-09-16-q-ridge`, forked from "
        f"`exp/2026-09-16-b-ladder-top` at `b8efd5b4`. Namespace "
        f"`/cluster/tufts/paralab/tawal01/q_ridge_20260916/`. R1 job `{r1['job_id']}` "
        f"(`{r1['attempt']}`) on `{r1['gpu']}`, source `{r1['commit']}`, elapsed "
        f"{r1['elapsed_seconds']:.1f} s. R2 job `{r2['job_id']}` (`{r2['attempt']}`) on "
        f"`{r2['gpu']}`, source `{r2['commit']}`, elapsed {r2['elapsed_seconds']:.1f} s. Both "
        'printed `jax_backend=gpu`, ran float64 at highest matmul precision, and were '
        'checksum-collected, independently NumPy-audited and archived before their exact '
        'remote attempt directories were removed.\n')

    for aud in (r1, r2):
        W.append(f"**{aud['mode'].upper()} ({aud['attempt']}) gates.** " + '; '.join(
            f"`{k}` {yn(v.get('passed'))}" for k, v in sorted(aud['checks'].items())
            if isinstance(v, dict) and 'passed' in v) + '.\n')

    W.append('**Cross-job fidelity.**\n')
    rows = []
    for aud in (r1, r2):
        for arm, v in sorted((aud['checks'].get('cross_job_fidelity') or {}).get('detail', {}).items()):
            det = v['detail']
            rows.append([aud['attempt'], f'`{arm}`', det.get('source'),
                         f"`{det.get('comparator')}`",
                         f"{det.get('declared_tolerance'):.0e}" if det.get('declared_tolerance') else '—',
                         f"{(det.get('worst_all_times_percent') or {}).get('relative_difference'):.1e}"
                         if (det.get('worst_all_times_percent') or {}).get('relative_difference') is not None else '—',
                         f"{(det.get('worst_evolved_percent') or {}).get('relative_difference'):.1e}"
                         if (det.get('worst_evolved_percent') or {}).get('relative_difference') is not None else '—',
                         yn(v['passed'])])
    W.append(table(['job', 'arm', 'source', 'comparator', 'tolerance',
                    'rel. diff (all-times)', 'rel. diff (evolved)', 'passed'], rows))

    W.append('\n**R1 — the ridge.**\n')
    W.append(ladder_block(r1, 'lam'))
    W.append('\n**R2 — more tests at fixed $q$.**\n')
    W.append(ladder_block(r2, 'rule'))

    W.append('\n**R3 — the held-out weak residual** (worst over cases and steps, per-mode RMS '
             'normalised by the per-node RMS of the previous state; the common held-out block '
             'is the 512 modes ranked 1537-2048, beyond every arm\'s $M$).\n')
    rows = []
    for aud in (r1, r2):
        for arm, v in sorted((aud.get('r3') or {}).get('arms', {}).items()):
            row = next(x for x in aud['arms'] if x['arm'] == arm)
            rows.append([aud['attempt'], f'`{arm}`', row['q'], row['quadrature'],
                         '0' if not row['lam_rel'] else f"{row['lam_rel']:.0e}", row['M'],
                         f"{v['in_space']['worst_normalised_per_mode_rms']:.3e}",
                         f"{v['held_out_common']['worst_normalised_per_mode_rms']:.3e}",
                         f"{v['held_over_in_common']:.3f}"])
    W.append(table(['job', 'arm', '$q$', 'quadrature', '$\\lambda_{rel}$', '$M$',
                    'in-space', 'held-out (common)', 'held/in'], rows))
    W.append('\n' + table(
        ['quadrature', '$q=0$ arm', '$q=16$ arm', 'held-out at $q=0$', 'held-out at $q=16$',
         'higher at $q=16$?'],
        [[q, f"`{v['low']}`", f"`{v['high']}`", f"{v['low_held']:.3e}", f"{v['high_held']:.3e}",
          yn(v['held_out_higher_at_high_q'])] for q, v in sorted(fp.items())]))

    rep = Path(a.report)
    W.append(f'\nSource-generated report: `experiments/q-ridge/{rep.name}` (SHA256 '
             f'`{sha(rep)}`) with its LaTeX twin, its figure and its generator beside it.\n')
    for att in (r1['attempt'], r2['attempt']):
        man = CELL / f'artifacts/{att}/archive.json'
        if man.exists():
            m = json.loads(man.read_text())
            W.append(f"Raw archive `{att}` Git-tracked as bounded chunks: whole SHA256 "
                     f"`{m['sha256']}` ({len(m['chunks'])} chunks). `output/bank_G.npz` is "
                     'excluded by design and its SHA256 recorded beside them.\n')

    Path(a.out).write_text('\n'.join(W) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
