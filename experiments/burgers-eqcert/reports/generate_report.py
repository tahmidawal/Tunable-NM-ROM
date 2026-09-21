"""Generate reports/2026-09-21-burgers-eqcert.md and reports/summary.json from the audited checks/bc*-summary.json.
Every number in the report comes from those files; nothing is typed by hand.

    python reports/generate_report.py
"""
import hashlib
import json
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
ATTEMPTS = ['bc256', 'bc256b', 'bc512', 'bc1024', 'bc2048']
MAIN = ['bc256', 'bc256b', 'bc512', 'bc1024']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def main():
    S = {}
    for a in ATTEMPTS:
        p = LANE / 'checks' / f'{a}-summary.json'
        if p.exists():
            S[a] = json.loads(p.read_text())
    rows, lines = [], []
    # DESIGN A2.1: combined 12-draw rule at 256^2 (bc256 AND bc256b)
    combined256 = None
    if 'bc256' in S and 'bc256b' in S:
        A_, B_ = S['bc256'], S['bc256b']
        ok = []
        for n, st in B_['arm_status'].items():
            sa = A_['arm_status'].get(n)
            if not sa or st.get('q') != 256 or st.get('control') or st.get('audited_status') == 'exact residual':
                continue
            both = all(x.get('audited_status') == 'confirmed' and x.get('audited_confirmation_pass') for x in (st, sa))
            acc = all(Z['table'][n]['worst_evolved_percent'] <= 1 and Z['table'][n]['stalled_exits'] == 0 for Z in (A_, B_))
            if both and acc:
                ok.append(n)
        pick = min(ok, key=lambda n: B_['table'][n]['median_gpu_ms']) if ok else None
        combined256 = dict(rule='DESIGN A2.1: confirmed 5/5 + confirmation in BOTH bc256 and bc256b; cheapest by bc256b time',
                           eligible=ok, pick=pick)
        if pick:
            t = B_['table'][pick]
            fo = B_['table'][t['fom_by_paper_rule']]
            st = B_['arm_status'][pick]
            rows.append(dict(mesh=256, attempt='bc256+bc256b', job_id=f"{A_['job_id']}+{B_['job_id']}", gpu=B_['gpu'],
                             role='CERTIFIED (A2.1 combined, row from bc256b)', arm=pick, status='confirmed in both',
                             exact_steps=st.get('exact_steps'), m=t['m'],
                             heldout_rho_max=max(st['heldout_rho_max'], A_['arm_status'][pick]['heldout_rho_max']),
                             deployed_rho_max=max(st['deployed_rho_max'], A_['arm_status'][pick]['deployed_rho_max']),
                             confirmation_pass=True, worst_evolved_percent=t['worst_evolved_percent'],
                             worst_all_times_percent=t['worst_all_times_percent'], rom_gpu_ms=t['median_gpu_ms'],
                             rom_host_ms=t['median_host_ms'], stalled_exits=t['stalled_exits'], fom=t['fom_by_paper_rule'],
                             fom_gpu_ms=fo['median_gpu_ms'], fom_worst_evolved_percent=fo['worst_evolved_percent'],
                             speedup_gpu=t['speedup_gpu'], speedup_host=t['speedup_host']))
    for a, s in S.items():
        if a == 'bc2048':
            continue
        v = s['verdict']
        L = s['intervals']
        for key, role in (('accurate', 'accurate row'), ('selected_failed_confirmation', 'selected, failed confirmation'),
                          ('fast', 'fast q=0')):
            r = v.get(key)
            if not r:
                continue
            rows.append(dict(mesh=L, attempt=a, job_id=s['job_id'], gpu=s['gpu'], role=role, **r))
        # every q=256 arm that is confirmed 5/5 AND passes confirmation (reported, not selected unless it was the pick)
        for n, st in s['arm_status'].items():
            t = s['table'].get(n, {})
            if st.get('q') == 256 and st.get('audited_status') == 'confirmed' and st.get('audited_confirmation_pass') \
                    and n != (v.get('accurate') or {}).get('arm'):
                rows.append(dict(mesh=L, attempt=a, job_id=s['job_id'], gpu=s['gpu'], role='passes all six draws, not selected',
                                 arm=n, status='confirmed', exact_steps=st.get('exact_steps'), m=t.get('m'),
                                 heldout_rho_max=st.get('heldout_rho_max'), deployed_rho_max=st.get('deployed_rho_max'),
                                 worst_evolved_percent=t['worst_evolved_percent'], rom_gpu_ms=t['median_gpu_ms'],
                                 fom=t['fom_by_paper_rule'], fom_gpu_ms=S[a]['table'][t['fom_by_paper_rule']]['median_gpu_ms'],
                                 fom_worst_evolved_percent=S[a]['table'][t['fom_by_paper_rule']]['worst_evolved_percent'],
                                 speedup_gpu=t['speedup_gpu'], confirmation_pass=True, stalled_exits=t['stalled_exits']))
    out = dict(status='final' if len(S) == len(ATTEMPTS) else 'partial', attempts=list(S), combined_256=combined256,
               sources={a: dict(summary=f'checks/{a}-summary.json', sha256=sha(LANE / 'checks' / f'{a}-summary.json'),
                                job_id=s['job_id'], job_commit=s['commit'], result_json_sha256=s['sources']['result_json_sha256'],
                                accepted=s['verdict']['accepted'], failed_gates=s['failed_gates']) for a, s in S.items()},
               verdict_per_mesh={a_: dict(mesh=s['intervals'], certified_rule_exists=s['verdict']['certified_rule_exists'],
                                                      selected_on_certification_draws=s['verdict']['selected_on_certification_draws'],
                                                      selected_passes_confirmation=s['verdict']['selected_passes_confirmation'],
                                                      control=s['verdict'].get('control'), accepted=s['verdict']['accepted'])
                                 for a_, s in S.items() if a_ in MAIN},
               certificate_2048=({n: dict(status=st['audited_status'], confirmation=st.get('audited_confirmation_pass'),
                                          heldout_rho_max=st.get('heldout_rho_max'), deployed_rho_max=st.get('deployed_rho_max'),
                                          heldout_rho_max_all_k=st.get('heldout_rho_max_k>=0'),
                                          worst_evolved_percent=S['bc2048']['table'][n]['worst_evolved_percent'])
                                  for n, st in S['bc2048']['arm_status'].items()} if 'bc2048' in S else None),
               rows=rows)
    (LANE / 'reports' / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')

    md = ['# Burgers 2D, $q=256$: certifying a quadrature rule at $256^2$, $512^2$, $1024^2$',
          '',
          f"Status: **{out['status']}** ({', '.join(S)} audited). Generated by `reports/generate_report.py` from "
          '`checks/*-summary.json` (independent NumPy audits); no number is typed by hand. Design and pre-registered '
          'rules: `DESIGN.md` (+ Addendum A1).',
          '',
          '## Verdict per mesh (pre-registered selection, DESIGN §5 + A1.3)', '',
          '| mesh | certified rule | selected on draws 1–5 | selected passes confirmation | control `bad0` | audit accepted |',
          '|---|---|---|---|---|---|']
    for a_, v in out['verdict_per_mesh'].items():
        L = f"{S[a_]['intervals']}^2$ ({a_})"
        c = v['control'][0] if v['control'] else {}
        md.append(f"| ${L} | {'yes' if v['certified_rule_exists'] else '**no**'} | `{v['selected_on_certification_draws']}` | "
                  f"{v['selected_passes_confirmation']} | {c.get('status')}, {f(c.get('worst_evolved_percent'), 3)} % "
                  f"(exact {f(c.get('exact_worst_evolved_percent'), 3)} %) | {v['accepted']} |")
    if combined256:
        md += ['', f"**$256^2$, combined rule (A2.1):** eligible {combined256['eligible']}; pick `{combined256['pick']}`."]
    if out['certificate_2048']:
        md += ['', '**$2048^2$ certificate only (A2.2, bc2048):**', '', '| arm | status | confirmation | $\\rho_{\\max}$ held-out | all $k$ | worst evolved % |', '|---|---|---|---|---|---|']
        for n, c in out['certificate_2048'].items():
            md.append(f"| `{n}` | {c['status']} | {c['confirmation']} | {f(c['heldout_rho_max'])} | {f(c['heldout_rho_max_all_k'])} | {f(c['worst_evolved_percent'], 3)} |")
    md += ['', '## Rows', '',
           '| mesh | role | arm | exact steps | $m$ | $\\rho_{\\max}$ held-out / deployed (draws 1–5) | confirmation | '
           'worst evolved % | ROM ms | FOM (paper rule) | FOM ms | FOM % | speedup |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| ${r['mesh']}^2$ | {r['role']} | `{r['arm']}` | {r.get('exact_steps')} | {r.get('m')} | "
                  f"{f(r.get('heldout_rho_max'))} / {f(r.get('deployed_rho_max'))} | {r.get('confirmation_pass')} | "
                  f"{f(r['worst_evolved_percent'], 3)} | {f(r['rom_gpu_ms'], 1)} | `{r['fom']}` | {f(r['fom_gpu_ms'], 1)} | "
                  f"{f(r['fom_worst_evolved_percent'], 3)} | {f(r['speedup_gpu'], 3)}× |")
    md += ['', '## Every arm', '']
    for a, s in S.items():
        md += [f"### ${s['intervals']}^2$ — `{a}`, job {s['job_id']}, {s['gpu']}, commit `{s['commit'][:8]}`", '',
               '| arm | status | $\\rho_{\\max}$ held-out ($k\\ge j$) | $\\rho_{\\max}$ all $k$ | confirmation | worst evolved % | GPU ms | speedup |',
               '|---|---|---|---|---|---|---|---|']
        for n, st in s['arm_status'].items():
            t = s['table'][n]
            md.append(f"| `{n}` | {st['audited_status']} | {f(st.get('heldout_rho_max'))} | {f(st.get('heldout_rho_max_k>=0'))} | "
                      f"{st.get('audited_confirmation_pass')} | {f(t['worst_evolved_percent'], 3)} | {f(t['median_gpu_ms'], 1)} | "
                      f"{f(t['speedup_gpu'], 3)}× |")
        md += ['', '| FOM | dt | Newton tol | lin. tol | worst evolved % | GPU ms | converged |', '|---|---|---|---|---|---|---|']
        for n, t in s['table'].items():
            if t['family'] == 'fom':
                md.append(f"| `{n}` | {t['dt']} | {t['ntol']} | {t['ltol']} | {f(t['worst_evolved_percent'], 4)} | "
                          f"{f(t['median_gpu_ms'], 1)} | {t['nonlinear_converged']} |")
        md.append('')
    md += ['## What this does and does not show', '',
           '- Certification is empirical coverage on $40$ (+16 confirmation) held-out trajectories of a fresh seed; zero '
           'failures in $n$ trajectories bounds the per-trajectory failure probability by about $3/n$ at 95 %.',
           '- Certificates sample the per-step states $w_k$ only (for an arm with $j$ exact steps, $k\\ge j$), never LM '
           'iterates or predictor candidates.',
           '- Arms with $j\\ge1$ are a different online method (the first $j$ steps use the exact residual); their time '
           'includes those steps.',
           '- Errors are on six opened development cases (dev6); they are development evidence, not held-out accuracy.',
           '- Speedup = median GPU time of the FOM chosen by the paper rule (fastest converged Newton–BiCGStab setting '
           'with worst error ≤ the ROM\'s) over the ROM\'s, same allocation, 5 repetitions × 6 cases.',
           '', '## Glossary', '',
           '- **ρ** — relative error of the quadrature approximation of the weak advection term on one state.',
           '- **ρ_max held-out / deployed** — worst ρ over the five certification draws, on states of the audited dense '
           'query / on states visited by that arm itself.',
           '- **confirmed** — ρ_max ≤ 0.116 on all five draws in both populations; **marginal (n/5)** — some draws pass.',
           '- **confirmation** — a sixth draw of 16 trajectories, used only after the selection.',
           '- **exact steps j** — number of initial time steps solved with the exact (all-node) residual.',
           '- **scaled** — b-eqtop\'s stored rule at the same physical nodes; **lat64** — 63×63 equal-weight lattice; '
           '**lathalf** — every second node; **exact** — no quadrature; **bad0** — control rule that must fail.',
           '- **m** — number of quadrature nodes; **worst evolved %** — worst over cases and output times t>0 of '
           '‖u−u_tight‖/‖u₀‖.',
           '- **FOM** — backward-Euler upwind finite differences, Newton with FFT-preconditioned BiCGStab.']
    (LANE / 'reports' / '2026-09-21-burgers-eqcert.md').write_text('\n'.join(md) + '\n')
    print('written', out['status'])


if __name__ == '__main__':
    main()
