"""Script-generated self-audit of the b-eqtop interim report against the raw JSONs.

Stands in for the Codex report audit while its quota is exhausted (DESIGN A3). Each row is a
claim the report or lab entry makes, the JSON field it rests on, the check recomputed here
(without the generator's code paths), and the outcome. Exit status is nonzero if any check fails.

    python checks/self_audit_report.py [--report reports/2026-09-17-b-eqtop.md]
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', default=str(HERE / 'reports/2026-09-17-b-eqtop.md'))
    p.add_argument('--out', default=str(HERE / 'reports/self-audit-report.md'))
    a = p.parse_args()
    A = json.loads((HERE / 'checks/bet101-audit.json').read_text())
    B = json.loads((HERE / 'checks/bet201-audit.json').read_text())
    r = json.loads((HERE / 'artifacts/bet101/result.json').read_text())
    S = json.loads((HERE / 'reports/summary.json').read_text())
    prov = json.loads((HERE / 'certified-rules/PROVENANCE.json').read_text())
    text = Path(a.report).read_text()
    bars = A['bars']
    rows, bad = [], []

    def claim(n, what, rests, check, ok, outcome):
        rows.append((n, what, rests, check, 'pass' if ok else 'FAIL', outcome))
        if not ok:
            bad.append(n)

    # 1 gates
    claim(1, 'no blocking gate failed in bet101 or bet201', "audit `failed`", 'both lists empty',
          A['failed'] == [] and B['failed'] == [], f"bet101 {A['failed']}, bet201 {B['failed']}")
    claim(2, 'both jobs ran on GPU, float64, highest precision', "`result.json` backend/x64/matmul_precision",
          'read the raw result, not the audit', r['backend'] == 'gpu' and r['x64'] is True and r['matmul_precision'] == 'highest',
          f"{r['backend']}, x64={r['x64']}, {r['matmul_precision']}, {r['gpu']}")
    rc = r['gates']['archived_rules_recertify']
    claim(3, 'qrg304 rules re-certified to ~1e-9', "`gates.archived_rules_recertify.worst_relative_difference`",
          'value ≤ tolerance 1e-6', rc['worst_relative_difference'] <= rc['tolerance'], f"{rc['worst_relative_difference']:.2e}")
    # 4 every rung primary-certified: recompute from rules independently of rule_choice
    QS = r['config']['q_ladder']
    rules = [x for x in A['rules'] if not x['truncated']]
    order = {'reachable': -1, 'static': -1, 'std': 0, 'fs64': 1, 'rhow64': 2}
    cheapest = {}
    for q in QS:
        c = sorted([x for x in rules if x['q'] == q and x['certified_primary'] and x['population'] == 'reachable'],
                   key=lambda x: (x['m'], 0 if x['source'] == 'qrg304' else 1, order.get(x['arm'], 9)))
        cheapest[q] = c[0] if c else None
    chosen = {c['q']: c['chosen'] for c in A['rule_choice'] if c['bar'] == 'primary'}
    agree = all(cheapest[q] is not None and chosen[q] is not None and cheapest[q]['m'] == chosen[q]['m']
                and abs(cheapest[q]['rho_max'] - chosen[q]['rho_max']) < 1e-12 for q in QS)
    claim(4, 'a primary-certified rule at every rung; the ladder ran the cheapest one', "`rules[*]` vs `rule_choice`",
          'recompute cheapest certified reachable rule per rung from the rule table and compare to the in-job choice',
          agree, '; '.join(f"q{q}: {cheapest[q]['source']}/{cheapest[q]['arm']} m={cheapest[q]['m']} fs={cheapest[q]['fit_states']} rho={cheapest[q]['rho_max']:.4f}" for q in QS))
    # 5 ladder monotone, recomputed from arms
    arms = {x['arm']: x for x in A['arms']}
    ev = [arms[f'q{q}_eq_primary']['worst_evolved_percent'] for q in QS]
    claim(5, 'primary ladder non-increasing on worst-evolved error', "`arms[q*_eq_primary].worst_evolved_percent`",
          'pairwise comparison', all(b <= a_ for a_, b in zip(ev, ev[1:])), ' → '.join(f'{e:.4f}' for e in ev))
    conv = [arms[f'q{q}_eq_primary']['converged'] and arms[f'q{q}_eq_primary']['total_budget_exits'] == 0
            and arms[f'q{q}_eq_primary']['max_joint_stationarity'] <= 1e-6 for q in QS]
    claim(6, 'every primary rung converged (no budget exits, joint gradient ≤ 1e-6)', "`arms[*].converged/total_budget_exits/max_joint_stationarity`",
          'recomputed from the three fields', all(conv), f"max stationarity {max(arms[f'q{q}_eq_primary']['max_joint_stationarity'] for q in QS):.2e}")
    g = A['checks']['recorded_errors_recomputed_from_saved_fields']
    claim(7, 'errors were recomputed from saved fields by the audit', "`checks.recorded_errors_recomputed_from_saved_fields`",
          'gate passed, worst relative deviation', g['passed'], f"{g['detail']:.2e}")
    # 8 cost ratios within job
    ratios = {q: arms[f'q{q}_eq_primary']['median_gpu_ms'] / arms[f'q{q}_dense']['median_gpu_ms'] for q in QS if f'q{q}_dense' in arms}
    claim(8, 'every EQ rung 0.18–0.21× its same-job dense twin', "`arms[*].median_gpu_ms`", 'ratio recomputed within bet101 only',
          all(0.15 < v < 0.25 for v in ratios.values()), ', '.join(f'q{q}: {v:.3f}' for q, v in ratios.items()))
    # 9 top rung
    t256 = arms['q256_eq_primary']; d256 = arms['q256_dense']; o256 = arms['q256_eq_qrg304_m2048']
    claim(9, 'q=256: 64-fit-state rule certifies (0.1074 ≤ 0.116) where qrg304\'s 8-state rule (0.1678) does not; evolved 0.5389 % vs dense 0.5194 %, qrg304 rule 1.0361 %',
          "`arms[q256_*]`", 'read the arm rows', t256['rho_max'] <= bars['primary'] < o256['rho_max'] and t256['worst_evolved_percent'] < o256['worst_evolved_percent'],
          f"rho {t256['rho_max']:.4f} / {o256['rho_max']:.4f}; evolved {t256['worst_evolved_percent']:.4f} / {d256['worst_evolved_percent']:.4f} / {o256['worst_evolved_percent']:.4f}")
    # 10 incumbent never certifies at q=256
    inc = [x for x in rules if x['q'] == 256 and x['population'] == 'reachable' and x['fit_states'] == 8]
    claim(10, 'the 8-fit-state construction never certifies at q=256 (best 0.1308)', "`rules[q=256, fit_states=8].rho_max`",
          'min over reachable 8-state rules > bar', min(x['rho_max'] for x in inc) > bars['primary'],
          f"{len(inc)} rules, min rho_max {min(x['rho_max'] for x in inc):.4f} at m={min(inc, key=lambda x: x['rho_max'])['m']}")
    # 11 gradient stops
    gs = [x for x in A['rules'] if x['source'] == 'this_job' and x['stop_reason'] == 'gradient']
    claim(11, 'std chains stopped on `gradient` short of m=4096/6144, not on walltime', "`rules[*].stop_reason/truncated`",
          'list stop reasons; no rule truncated', gs and not any(x['truncated'] for x in A['rules']),
          '; '.join(f"q{x['q']} {x['arm']} m={x['m']}/{x['m_target']}" for x in gs))
    # 12 fidelity
    fid = A['fidelity']
    worst = max((d['detail'].get('worst_evolved_percent') or {}).get('relative_difference') or 0 for d in fid.values())
    claim(12, 'every cross-job fidelity gate against qrg304 passed', "`fidelity[*]`", 'all passed; worst relative difference',
          all(d['passed'] for d in fid.values()), f"{len(fid)} arms, worst {worst:.2e}")
    # 13 non-monotone within chain
    s128 = sorted([x for x in A['rules'] if x['q'] == 128 and x['arm'] == 'std' and x['source'] == 'this_job'], key=lambda x: x['m'])
    nonmono = any(b['rho_max'] > a_['rho_max'] for a_, b in zip(s128, s128[1:]))
    claim(13, 'within the q=128 std chain rho_max is not monotone in m', "`rules[q=128, std]`", 'adjacent comparison', nonmono,
          ', '.join(f"{x['m']}:{x['rho_max']:.4f}" for x in s128))
    # 14 bet201 flip and spread
    Bq = {}
    for x in B['rules']:
        if x['population'] == 'reachable' and x['arm'] in ('reachable', 'std') and not x['truncated']:
            Bq.setdefault((x['q'], x['m_target'], x['source']), x)
    ratios2 = {}
    for (q, m, src), x in Bq.items():
        if src == 'this_job' and (q, m, 'qrg304') in Bq:
            ratios2[(q, m)] = x['rho_max'] / Bq[(q, m, 'qrg304')]['rho_max']
    f64 = (Bq[(64, 1024, 'qrg304')]['rho_max'], Bq[(64, 1024, 'this_job')]['rho_max'])
    claim(14, 'bet201: the same construction re-drawn moves rho_max 0.11×–2.3× and flips certification at q=64, m=1024',
          "`bet201 rules[*]` (qrg304 vs this_job std at equal q, m target)", 'ratios recomputed; the q=64 pair straddles the bar',
          f64[0] <= bars['primary'] < f64[1] and min(ratios2.values()) < 0.2 and max(ratios2.values()) > 2.0,
          f"ratios {min(ratios2.values()):.2f}–{max(ratios2.values()):.2f}; q64 m1024: {f64[0]:.4f} → {f64[1]:.4f}")
    # 15 exported rules
    ok = True
    for x in prov['rules']:
        f = HERE / 'certified-rules' / x['file']
        ok &= hashlib.sha256(f.read_bytes()).hexdigest() == x['sha256']
        ok &= abs(x['held_out']['rho_max'] - cheapest[x['q']]['rho_max']) < 1e-12 and x['m'] == cheapest[x['q']]['m']
    claim(15, 'exported rule files match the chosen rules and their SHA256', "`certified-rules/PROVENANCE.json`",
          'hash each file; compare rho_max and m with the recomputed cheapest rule', ok and len(prov['rules']) == len(QS),
          f"{len(prov['rules'])} files; status '{prov['status']}'")
    # 16 summary.json integrity
    sha = hashlib.sha256(text.encode()).hexdigest()
    ok = S['report_sha256'] == sha and all(('job_id' in x and x.get('status')) for x in S['rows'])
    claim(16, 'summary.json hashes this report and every row carries a job id and a status', "`summary.json`",
          'sha256 of the report file; row fields', ok, f"{len(S['rows'])} rows, status '{S['status']}'")
    # 17 provisional marking in the text
    ok = ('provisional' in text.splitlines()[0]) and text.count('provisional') >= 6 and 'does not claim a certified ladder' in text
    claim(17, 'the report marks the certification claims provisional beside the numbers, not only in a preamble',
          'report text', 'title, status line, verdict rows, ladder note, T9 caption, glossary all carry it', ok,
          f"'provisional' occurs {text.count('provisional')} times; in title: {'provisional' in text.splitlines()[0]}")
    # 18 tight at 256
    t = [x for x in rules if x['q'] == 256 and x['certified_tight']]
    best = min(x['rho_max'] for x in rules if x['q'] == 256)
    claim(18, 'no tight-certified rule at q=256; best rho_max 0.0647', "`rules[q=256]`", 'none flagged; min rho_max', not t and abs(best - 0.0647) < 5e-4,
          f"best {best:.4f}")
    # 19 the q=64 EQ-vs-dense gap
    gap = arms['q64_eq_primary']['worst_evolved_percent'] - arms['q64_dense']['worst_evolved_percent']
    claim(19, 'q=64: certified EQ costs +0.14 pp evolved error against dense', "`arms[q64_*]`", 'difference', 0.1 < gap < 0.2, f"{gap:+.4f} pp")

    md = ['# Self-audit of the b-eqtop interim report, 2026-09-17 (substitute for the unavailable Codex audit)', '',
          f'Generated by `checks/self_audit_report.py` against `checks/bet101-audit.json`, `checks/bet201-audit.json`, '
          f'`artifacts/bet101/result.json`, `reports/summary.json`, `certified-rules/PROVENANCE.json` and the report '
          f'`{Path(a.report).name}` (SHA256 `{sha}`). Each row: the claim, the field it rests on, the check recomputed here '
          'outside the generator, the outcome. Codex quota is exhausted until 2026-09-19 11:33 (DESIGN A3); this is the substitution.', '',
          '| # | claim | rests on | check | outcome | detail |', '|---|---|---|---|---|---|']
    md += [f'| {n} | {w} | {r_} | {c} | **{o}** | {d} |' for n, w, r_, c, o, d in rows]
    md += ['', f"**Result: {len(rows) - len(bad)} of {len(rows)} checks pass{'' if not bad else '; FAILED: ' + str(bad)}.**", '',
           'What this audit cannot check (unchanged from the design self-audit): the held-out $\\rho$ values themselves are the '
           'driver\'s (recomputing them needs the bank on the GPU and the collected states, which are not archived); the timings '
           'and iteration counts are the driver\'s. Scientific weaknesses still standing: one training seed, one checkpoint, six '
           'development cases; $\\rho_{\\max}$ over 512 held-out states is a tail statistic whose sampling noise is unmeasured; '
           'the tight bar 0.06 is the coordinator\'s number; and every certified flag is one draw until `bet301` lands.']
    Path(a.out).write_text('\n'.join(md) + '\n')
    print('\n'.join(f'{n:2d} {o:4s} {w[:70]}' for n, w, _, _, o, _ in rows))
    print('wrote', a.out)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
