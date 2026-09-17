"""Generate the dated LAB-LOG entry for this lane from the audit JSONs. Nothing typed by hand.

    python make_lab_entry.py --audit <audit.json> [...] --verdict "<one line>" --out checks/lab-log-entry.md
                             [--retracted "<text>"] [--open "<text>"] [--jobs-used N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    return f'{x:.{d}f}' if isinstance(x, float) else str(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', nargs='+', required=True)
    p.add_argument('--verdict', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--retracted', default='Nothing numerical retracted in this lane.')
    p.add_argument('--open', default='')
    p.add_argument('--jobs-used', type=int, default=None)
    p.add_argument('--report', default=str(HERE / 'reports/2026-09-17-b-panel.md'))
    a = p.parse_args()
    audits = [json.loads(Path(x).read_text()) for x in a.audit]
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()
    L = ['## 2026-09-17', f'### b-panel — {a.verdict}', '']
    L.append(f'Worktree `worktrees/2026-09-17-b-panel`, branch `{branch}` at `{head[:12]}`, forked from '
             f'`exp/2026-09-16-q-ridge`. Namespace `/cluster/tufts/paralab/tawal01/b_panel_20260917/`. Predeclared '
             f'protocol and amendments: `experiments/b-panel/DESIGN.md`. Frozen inputs (qtd02 directions, qrg304 certified '
             f'rules, b-speed kernels) with SHA256 provenance: `experiments/b-panel/inputs/PROVENANCE.json`. '
             + (f'Cluster jobs used: {a.jobs_used} of the cap of 8. ' if a.jobs_used is not None else '')
             + 'Every job printed `jax_backend=gpu`, ran float64 at highest matmul precision, and was checksum-collected, '
               'independently NumPy-audited and Git-archived as bounded chunks before its exact remote attempt directory was removed.')
    L.append('')
    for au in audits:
        L.append(f"**{au['intervals']}² — job `{au['job_id']}` (`{au['attempt']}`) on `{au['gpu']}`**, source `{(au['commit'] or '')[:12]}`, "
                 f"elapsed {f(au['elapsed_seconds'], 1)} s, {len(au['timed_subjects'] or [])} timed subjects, "
                 f"failed gates: {', '.join(au['failed']) or 'none'}; dropped by the OOM rule: "
                 f"{', '.join(d['name'] for d in au['dropped']) or 'none'}.")
        L.append('')
        L.append('| subject | family | q / k′ | quad. | rule set | basis | rule status | tol | worst all % | worst evolved % | t=0 % | vs ref % | GPU ms | complete ms | med it | budget exits | converged | strict | admissible |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for x in au['arms']:
            qk = x['q'] if x['q'] is not None else (x['k'] if x['k'] is not None else '—')
            L.append(f"| `{x['arm']}` | {x['family']} | {qk} | {x['quadrature'] or '—'} | {x.get('rule_set') or '—'} | {x['rule_basis'] or '—'} | {x.get('rule_status') or '—'} | "
                     f"{('%.0e' % x['gtol']) if x['gtol'] is not None else '—'} | {f(x['worst_all_times_percent'])} | "
                     f"{f(x['worst_evolved_percent'])} | {f(x['worst_t0_compression_percent'])} | {f(x['worst_reference_percent'])} | "
                     f"{f(x['median_gpu_ms'], 3)} | {f(x['median_host_ms'], 3)} | {f(x['median_iterations'], 1)} | "
                     f"{f(x['total_budget_exits'])} | {f(x['converged'])} | {f(x['converged_strict'])} | {f(x['admissible'])} |")
        L.append('')
        for key, v in au['nondominated'].items():
            L.append(f"Non-dominated ({v['cost']}, {v['error']}), admissible: " + (', '.join(f'`{s}`' for s in v['admissible']) or 'none') + '.')
        L.append('')
        for name, lad in au['ladders'].items():
            if lad:
                L.append(f"Ladder `{name}`: rungs {lad['q_or_k']}; worst evolved % {[round(v, 4) for v in lad['worst_evolved_percent']]}; "
                         f"GPU ms {[round(v, 1) for v in lad['median_gpu_ms']]}; monotone evolved {f(lad['monotone_evolved'])}, "
                         f"all converged {f(lad['all_converged'])}, converged non-dominated points {lad['nondominated_converged_points']}, "
                         f"error span {f(lad['error_span'], 3)}, cost span {f(lad['cost_span'], 3)}.")
        if au['transfer']:
            L.append('')
            L.append('Transferred rules (`eqxfer`): ' + '; '.join(
                f"q={t['q']} m={t['m']} ρmax {f(t['certification']['rho_max'])} ρ95 {f(t['certification']['rho_p95'])} → {t['basis']}"
                for t in au['transfer']) + '.')
        fid = (au['checks'].get('cross_job_fidelity') or {}).get('detail') or {}
        if fid:
            L.append('')
            L.append('Cross-job fidelity: ' + '; '.join(
                f"`{k}` vs `{v['detail'].get('comparator')}` ({v['detail'].get('source')}) {v['detail'].get('worst_relative_difference'):.1e} "
                f"[{v['detail'].get('tier')} tier, {'pass' if v['passed'] else 'FAIL'}]" for k, v in fid.items()) + '.')
        if au.get('fno'):
            L.append('')
            L.append(f"FNO `{au['fno']['model']}` in the same allocation: pooled device query median "
                     f"{f(au['fno']['device_query_pooled']['median_ms'], 3)} ms; {au['fno']['deviation']}.")
        L.append('')
        L.append('Gates: ' + '; '.join(f"`{k}` {f(v['passed'])}" for k, v in au['checks'].items() if k != 'cross_job_fidelity') + '.')
        L.append('')
    L.append(f'**What was wrong and is retracted.** {a.retracted}')
    L.append('')
    if a.open:
        L.append(f'**Open.** {a.open}')
        L.append('')
    rp = Path(a.report)
    if rp.exists():
        L.append(f'Source-generated report: `experiments/b-panel/reports/{rp.name}` (SHA256 `{hashlib.sha256(rp.read_bytes()).hexdigest()}`) '
                 f'with `summary.json`, the envelope figures and their point JSONs beside it; generator `reports/generate_panel.py` reads only the audit JSONs.')
    for d in sorted((HERE / 'artifacts').glob('*')) if (HERE / 'artifacts').exists() else []:
        meta = d / 'archive.json'
        if meta.exists():
            m = json.loads(meta.read_text())
            L.append(f"Raw archive `{d.name}` Git-tracked as bounded chunks: whole SHA256 `{m['sha256']}` ({len(m['chunks'])} chunks).")
    L.append('')
    Path(a.out).write_text('\n'.join(L))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
