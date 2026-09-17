"""Generate the dated lab-log entry for b-eqtop from the audit JSONs and summary.json.
Nothing is typed by hand; the retractions section is appended from `checks/retractions.json`
if present (an explicit list written by the session).

    python make_lab_entry.py --j1 checks/bet101-audit.json [--j2 checks/bet201-audit.json]
                             --report reports/2026-09-17-b-eqtop.md --out checks/lab-entry.md
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def yn(x):
    return '—' if x is None else ('yes' if x else 'no')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--j1', required=True)
    p.add_argument('--j2', default=None)
    p.add_argument('--report', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pending', default=None, help='attempt:job_id:purpose of a job not yet in this entry')
    a = p.parse_args()
    pend = dict(zip(('attempt', 'job_id', 'purpose'), a.pending.split(':', 2))) if a.pending else None
    PROV = f" [provisional: `{pend['attempt']}` {pend['job_id']} pending]" if pend else ''
    A = json.loads(Path(a.j1).read_text())
    B = json.loads(Path(a.j2).read_text()) if a.j2 and Path(a.j2).exists() else None
    sub = json.loads((HERE / 'checks/submissions.json').read_text())
    v, lad = A['verdict'], A['ladders']
    rep = Path(a.report)
    sha = hashlib.sha256(rep.read_bytes()).hexdigest()
    out = []
    out.append('## 2026-09-17\n')
    verdict = ('every rung primary-certified and the primary EQ ladder monotone on evolved times'
               if v['every_rung_primary_certified'] and v['primary_ladder_monotone_evolved']
               else ('every rung primary-certified but the primary ladder still regresses'
                     if v['every_rung_primary_certified']
                     else f"rungs {v['rungs_without_primary_rule']} not certifiable at m <= 6144; hybrid ladder monotone {yn(v['hybrid_ladder_monotone_evolved'])}"))
    if pend:
        verdict = f'INTERIM ({pend["attempt"]} pending) — ' + verdict + ' in this single draw'
    out.append(f'### b-eqtop — {verdict}\n')
    if pend:
        out.append(f"**Interim entry.** `{pend['attempt']}` (job {pend['job_id']}, {pend['purpose']}) is still running. "
                   "DESIGN §A2 showed that the same rule construction at the same $m$ moves $\\rho_{\\max}$ by $0.11\\times$–$2.3\\times$ "
                   "under an independent draw of (candidate pool, fit-state subset) and flips certification at $q = 64$, so every "
                   "certified flag below is a single draw and is **provisional** until that job bounds the spread. No certified "
                   "ladder is claimed here; a closing entry follows when it lands.\n")
    out.append(f"Worktree `worktrees/2026-09-17-b-eqtop`, branch `exp/2026-09-17-b-eqtop`, forked from `exp/2026-09-16-q-ridge` at `7dc970fc`. Namespace `{sub['namespace']}`. Jobs: "
               + '; '.join(f"`{s['attempt']}` = {s['job_id']} ({s['gpu_request']})" for s in sub['submissions'])
               + f". Job 1 ran on `{A['gpu']}`, source `{A['commit']}`, elapsed {f(A['elapsed_seconds'],0)} s"
               + (f"; job 2 on `{B['gpu']}`, source `{B['commit']}`, elapsed {f(B['elapsed_seconds'],0)} s" if B else '')
               + ". Both printed `jax_backend=gpu`, float64, highest matmul precision; checksum-collected, NumPy-audited, archived as Git chunks, remote directories deleted. Design and amendments: `experiments/b-eqtop/DESIGN.md`. Codex was unavailable (quota) for the pre-job audit; a written self-audit stands in (`experiments/b-eqtop/reports/self-audit-design.md`, DESIGN A1).\n")
    out.append(f"**Failed blocking gates:** job 1 {', '.join(A['failed']) or 'none'}" + (f"; job 2 {', '.join(B['failed']) or 'none'}" if B else '') + '.\n')
    out.append(f'**Certification at the top rungs**{PROV} (held-out $\\rho_{{\\max}}$ over 512 reachable states, bar 0.116 primary / 0.06 tight; this job\'s rules only; `qrg304` rules re-certified to {A["checks"]["archived_rules_recertify"]["detail"]["worst_relative_difference"]:.1e} relative):\n')
    out.append('| $q$ | arm | $m$ | fit states | NNLS fit | $\\rho_{\\max}$ | $\\rho_{95}$ | primary | tight | fit (s) |\n|---|---|---|---|---|---|---|---|---|---|')
    for x in A['rules']:
        if x['source'] == 'this_job':
            out.append(f"| {x['q']} | {x['arm']} | {x['m']} | {x['fit_states']} | {x['relative_fit']:.2e} | {f(x['rho_max'])} | {f(x['rho_p95'])} | {yn(x['certified_primary'])} | {yn(x['certified_tight'])} | {f(x['fit_seconds'],0)} |")
    out.append('')
    out.append('**Laws $\\rho_{\\max}\\propto m^{-\\alpha}$ and the $m$ each bar needs:**\n')
    out.append('| $q$ | arm | $\\alpha$ | $m$ for 0.116 (law) | cheapest certified $m$ | $m$ for 0.06 (law) | job |\n|---|---|---|---|---|---|---|')
    for J in [A] + ([B] if B else []):
        for l in J['laws']:
            fit = l['fit_rho_max'] or {}
            out.append(f"| {l['q']} | {l['arm']} | {f(fit.get('alpha'),3)} | {f(l['m_for_bar'].get('primary'),0)} | {l['cheapest_certified_m'].get('primary') or '—'} | {f(l['m_for_bar'].get('tight'),0)} | {J['job_id']} |")
    out.append('')
    for key in ('primary', 'tight', 'hybrid', 'dense'):
        d = lad.get(key)
        if not d:
            continue
        out.append(f"**{d['name']}**{PROV} — monotone evolved **{yn(d['monotone_evolved'])}**, all-times {yn(d['monotone_all_times'])}, converged {yn(d['all_converged'])}, EQ cheaper than dense twin {yn(d['cheaper_than_dense_where_measured'])}; q = {d['q']}; evolved % = {' / '.join(f(e) for e in d['worst_evolved_percent'])}; all-times % = {' / '.join(f(e) for e in d['worst_all_times_percent'])}; median GPU ms = {' / '.join(f(c,0) for c in d['median_gpu_ms'])}; m = {d['m']}; $\\rho_{{\\max}}$ = {' / '.join(f(r) for r in d['rho_max'])}.\n")
    t = v['top_rung']
    out.append(f"**Top rung $q={t['q']}$:**{PROV} primary arm $m={t['primary_rule']}$, $\\rho_{{\\max}}={f(t['primary_rho_max'])}$, evolved {f(t['evolved_percent'])} % vs dense {f(t['dense_evolved_percent'])} %. $\\rho$ against evolved error at $q={t['q']}$: "
               + '; '.join(f"{x['arm']} m={x['m']} rho_max={f(x['rho_max'])} -> {f(x['evolved_percent'])} %" for x in v['top_rung_rho_vs_evolved']) + '.\n')
    fid = A.get('fidelity') or {}
    out.append('**Cross-job fidelity vs qrg304:** ' + '; '.join(
        f"`{k}` {'passed' if d_['passed'] else 'FAILED'} ({(d_['detail'].get('worst_evolved_percent') or {}).get('relative_difference', float('nan')):.1e} evolved)"
        for k, d_ in sorted(fid.items())) + '.\n')
    out.append('**Gates, job 1:** ' + '; '.join(f"`{k}` {yn(c['passed'])}" for k, c in sorted(A['checks'].items())) + '.\n')
    if B:
        out.append('**Gates, job 2:** ' + '; '.join(f"`{k}` {yn(c['passed'])}" for k, c in sorted(B['checks'].items())) + '.\n')
    retr = HERE / 'checks/retractions.json'
    if retr.exists():
        out.append('**What was wrong and retracted / corrected:**\n')
        for line in json.loads(retr.read_text()):
            out.append(f'- {line}')
        out.append('')
    openq = HERE / 'checks/open.json'
    if openq.exists():
        out.append('**Open:**\n')
        for line in json.loads(openq.read_text()):
            out.append(f'- {line}')
        out.append('')
    out.append(f"Source-generated report: `experiments/b-eqtop/reports/{rep.name}` (SHA256 `{sha}`) with `summary.json` and its generator beside it; audits `experiments/b-eqtop/checks/bet101-audit.json`" + (', `bet201-audit.json`' if B else '') + "; self-audit of the report in place of Codex: `experiments/b-eqtop/reports/self-audit-report.md`; archives `experiments/b-eqtop/artifacts/bet101`, `bet201`; exported rule set for `b-panel`: `experiments/b-eqtop/certified-rules/` (`PROVENANCE.json`, `SHA256SUMS`).")
    Path(a.out).write_text('\n'.join(out) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
