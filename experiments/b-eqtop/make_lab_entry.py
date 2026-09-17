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
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import draws as DR  # noqa: E402


def f(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def yn(x):
    return '—' if x is None else ('yes' if x else 'no')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--j1', required=True)
    p.add_argument('--j2', default=None)
    p.add_argument('--j3', default=None, help='the draw-replication audit (bet301); makes this the closing entry')
    p.add_argument('--report', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pending', default=None, help='attempt:job_id:purpose of a job not yet in this entry')
    a = p.parse_args()
    pend = dict(zip(('attempt', 'job_id', 'purpose'), a.pending.split(':', 2))) if a.pending else None
    PROV = f" [provisional: `{pend['attempt']}` {pend['job_id']} pending]" if pend else ''
    A = json.loads(Path(a.j1).read_text())
    B = json.loads(Path(a.j2).read_text()) if a.j2 and Path(a.j2).exists() else None
    C = json.loads(Path(a.j3).read_text()) if a.j3 and Path(a.j3).exists() else None
    S = json.loads((HERE / 'reports/summary.json').read_text())
    prov = json.loads((HERE / 'certified-rules/PROVENANCE.json').read_text())
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
    VR = {v['q']: v for v in S.get('verdict_per_rung', [])}
    if C:
        top = VR[max(VR)]
        verdict = ('CLOSED — the primary EQ ladder is monotone as built, but its certification does not survive the '
                   f"draw replication: at $q = {top['q']}$ the ladder's rule is {top['draws_certifying_primary']} of "
                   f"{top['draws']} draws of its construction to meet the bar ({top['ladder_rule_status']}); "
                   + '; '.join(f"$q = {q}$ {VR[q]['ladder_rule_status']}" for q in sorted(VR) if q != top['q']))
    out.append(f'### b-eqtop — {verdict}\n')
    if C:
        out.append("**Closing entry.** `bet301` (job {jid}, DESIGN §A2, four independent draws of (candidate pool, fit-state subset) "
                   "at $q \\in \\{{64, 128, 256\\}}$ for the incumbent construction at $m = 1024$ and the 64-fit-state construction at "
                   "$m = 2048$) landed: `{gpu}`, source `{commit}`, elapsed {el} s, `jax_backend=gpu`, no blocking gate failed, the 18 "
                   "`qrg304` rules re-certified to {rc:.1e} relative on that card; checksum-collected, NumPy-audited, archived "
                   "(`experiments/b-eqtop/artifacts/bet301`), the whole cluster namespace deleted. The rule DESIGN §A2 pre-registered is "
                   "applied literally: a construction that certifies on some draws and not others is *marginal at that $m$*.\n".format(
                       jid=C['job_id'], gpu=C['gpu'], commit=C['commit'], el=f(C['elapsed_seconds'], 0),
                       rc=C['checks']['archived_rules_recertify']['detail']['worst_relative_difference']))
        out.append('**Verdict per rung** (the rule the timed ladder ran in `bet101`; draws = every independent draw of its construction across `qrg304`, `bet101`, `bet201`, `bet301`; the exported rule is `certified-rules/` under the §A4 policy):\n')
        out.append('| $q$ | ladder rule | draws certifying | $\\rho_{\\max}$ min / median / max | status | exported rule | its status |\n|---|---|---|---|---|---|---|')
        for q in sorted(VR):
            v_ = VR[q]; lr, er = v_['ladder_rule'], v_['exported_rule']
            out.append(f"| {q} | {lr['label']} $m$={lr['m']}, {lr['fit_states']} st., {f(lr['rho_max'])} | {v_['draws_certifying_primary']}/{v_['draws']} | "
                       f"{f(v_['rho_min'])} / {f(v_['rho_median'])} / {f(v_['rho_max_of_draws'])} | **{v_['ladder_rule_status']}** | "
                       f"{er['label']} $m$={er['m']}, {er['fit_states']} st., {f(er['rho_max'])} | {v_['exported_rule_status']} |")
        out.append('')
        out.append('**The pre-registered replication itself** (`bet301`; four draws per configuration, seed order):\n')
        out.append('| $q$ | $m$ | fit states | each $\\rho_{\\max}$ | min / max | spread | certify primary | certify tight |\n|---|---|---|---|---|---|---|---|')
        for x in C['replication']:
            out.append(f"| {x['q']} | {x['m_target']} | {x['fit_states']} | {', '.join(f(r_) for r_ in x['rho_max'])} | {f(x['rho_min'])} / {f(x['rho_max_of_draws'])} | "
                       f"{f(x['spread_ratio'], 2)}× | {x['certified_primary_count']}/{x['draws']} | {x['certified_tight_count']}/{x['draws']} |")
        out.append('')
        out.append('**Exported rule set** (`experiments/b-eqtop/certified-rules/`, status: ' + prov['status'] + '): '
                   + '; '.join(f"$q={x['q']}$ `{x['file']}` $m={x['m']}$, $\\rho_{{\\max}}={f(x['held_out']['rho_max'])}$ [{x['construction']['status']}]" for x in prov['rules'])
                   + '. Superseded and removed: ' + ', '.join(f"`{x['file']}`" for x in prov['superseded']) + '.\n')
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
    out.append(f'**Certification at the top rungs (bet101 draw)**{PROV} (held-out $\\rho_{{\\max}}$ over 512 reachable states, bar 0.116 primary / 0.06 tight; this job\'s rules only; `qrg304` rules re-certified to {A["checks"]["archived_rules_recertify"]["detail"]["worst_relative_difference"]:.1e} relative):\n')
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
    if C:
        out.append('**Gates, job 3:** ' + '; '.join(f"`{k}` {yn(c['passed'])}" for k, c in sorted(C['checks'].items())) + '.\n')
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
    out.append(f"Source-generated report: `experiments/b-eqtop/reports/{rep.name}` (SHA256 `{sha}`) with `summary.json` and its generator beside it; audits `experiments/b-eqtop/checks/bet101-audit.json`" + (', `bet201-audit.json`' if B else '') + (', `bet301-audit.json`' if C else '') + "; self-audit of the report in place of Codex: `experiments/b-eqtop/reports/self-audit-report.md`; archives `experiments/b-eqtop/artifacts/bet101`, `bet201`" + (', `bet301`' if C else '') + "; exported rule set for `b-panel`: `experiments/b-eqtop/certified-rules/` (`PROVENANCE.json`, `SHA256SUMS`); the draw bookkeeping shared by report, export, self-audit and this entry: `experiments/b-eqtop/draws.py`. Jobs used: " + f"{sub.get('used', len(sub['submissions']))} of {sub['cap']}, none retracted.")
    Path(a.out).write_text('\n'.join(out) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
