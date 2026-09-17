"""Script-generated self-audit of the b-eqtop FINAL report against the raw JSONs.

Stands in for the Codex report audit while its quota is exhausted (DESIGN A3, A4). Each row is
a claim the report or lab entry makes, the JSON field it rests on, the check recomputed here
(without the generator's or `draws.py`'s code paths), and the outcome. Exit status is nonzero
if any check fails.

    python checks/self_audit_report.py [--report reports/2026-09-17-b-eqtop.md]
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', default=str(HERE / 'reports/2026-09-17-b-eqtop.md'))
    p.add_argument('--out', default=str(HERE / 'reports/self-audit-report.md'))
    a = p.parse_args()
    A = json.loads((HERE / 'checks/bet101-audit.json').read_text())
    B = json.loads((HERE / 'checks/bet201-audit.json').read_text())
    C = json.loads((HERE / 'checks/bet301-audit.json').read_text())
    r = json.loads((HERE / 'artifacts/bet101/result.json').read_text())
    r3 = json.loads((HERE / 'artifacts/bet301/result.json').read_text())
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
    claim(1, 'no blocking gate failed in bet101, bet201 or bet301', "audit `failed`", 'all three lists empty',
          A['failed'] == [] and B['failed'] == [] and C['failed'] == [], f"bet101 {A['failed']}, bet201 {B['failed']}, bet301 {C['failed']}")
    claim(2, 'all three jobs ran on GPU, float64, highest precision', "`result.json` backend/x64/matmul_precision (bet101, bet301 read raw)",
          'read the raw results, not the audits',
          all(z['backend'] == 'gpu' and z['x64'] is True and z['matmul_precision'] == 'highest' for z in (r, r3)),
          f"bet101 {r['gpu']}; bet301 {r3['gpu']}")
    rc, rc3 = r['gates']['archived_rules_recertify'], r3['gates']['archived_rules_recertify']
    claim(3, 'qrg304 rules re-certified to ~1e-9 in bet101 and in bet301 (a different GPU model)',
          "`gates.archived_rules_recertify.worst_relative_difference`", 'value ≤ tolerance 1e-6 in both',
          rc['worst_relative_difference'] <= rc['tolerance'] and rc3['worst_relative_difference'] <= rc3['tolerance'],
          f"bet101 {rc['worst_relative_difference']:.2e}; bet301 {rc3['worst_relative_difference']:.2e}")
    # 4 the ladder ran the cheapest certified rule of ITS draw
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
    claim(4, 'in bet101 a primary-certified rule existed at every rung and the ladder ran the cheapest one (of that draw)',
          "`rules[*]` vs `rule_choice`", 'recompute the cheapest certified reachable rule per rung and compare to the in-job choice',
          agree, '; '.join(f"q{q}: {cheapest[q]['source']}/{cheapest[q]['arm']} m={cheapest[q]['m']} rho={cheapest[q]['rho_max']:.4f}" for q in QS))
    arms = {x['arm']: x for x in A['arms']}
    ev = [arms[f'q{q}_eq_primary']['worst_evolved_percent'] for q in QS]
    claim(5, 'primary ladder non-increasing on worst-evolved error (as built)', "`arms[q*_eq_primary].worst_evolved_percent`",
          'pairwise comparison', all(b <= a_ for a_, b in zip(ev, ev[1:])), ' → '.join(f'{e:.4f}' for e in ev))
    conv = [arms[f'q{q}_eq_primary']['converged'] and arms[f'q{q}_eq_primary']['total_budget_exits'] == 0
            and arms[f'q{q}_eq_primary']['max_joint_stationarity'] <= 1e-6 for q in QS]
    claim(6, 'every primary rung converged', "`arms[*].converged/total_budget_exits/max_joint_stationarity`",
          'recomputed from the three fields', all(conv), f"max stationarity {max(arms[f'q{q}_eq_primary']['max_joint_stationarity'] for q in QS):.2e}")
    g = A['checks']['recorded_errors_recomputed_from_saved_fields']
    claim(7, 'errors were recomputed from saved fields by the audit', "`checks.recorded_errors_recomputed_from_saved_fields`",
          'gate passed, worst relative deviation', g['passed'], f"{g['detail']:.2e}")
    ratios = {q: arms[f'q{q}_eq_primary']['median_gpu_ms'] / arms[f'q{q}_dense']['median_gpu_ms'] for q in QS if f'q{q}_dense' in arms}
    claim(8, 'every EQ rung 0.18–0.21× its same-job dense twin', "`arms[*].median_gpu_ms`", 'ratio recomputed within bet101 only',
          all(0.15 < v < 0.25 for v in ratios.values()), ', '.join(f'q{q}: {v:.3f}' for q, v in ratios.items()))

    # ---- the replication, recomputed from bet301's raw result.json (not the audit, not draws.py)
    rep = {}
    for x in r3['rules']:
        if x.get('failed'):
            continue
        d = next(dd for dd in r3['designs'] if dd['key'] == x['key'])
        if d.get('seed_offset') is None:
            continue
        k = (x['q'], x['m_target'], d['fit_states'])
        rep.setdefault(k, []).append((x['certification']['rho_max'], x['fit']['truncated']))
    counts = {k: (sum(1 for v, t in vs if v <= bars['primary'] and not t), len(vs)) for k, vs in rep.items()}
    spread = {k: (min(v for v, _ in vs), statistics.median(v for v, _ in vs), max(v for v, _ in vs)) for k, vs in rep.items()}
    claim(9, 'bet301: 24 untruncated replication fits, 4 per configuration, 6 configurations', "`bet301 result.rules/designs`",
          'count from the raw result', len(rep) == 6 and all(n == 4 for _, n in counts.values()) and not any(t for vs in rep.values() for _, t in vs),
          '; '.join(f"q{q} m{m} fs{fs}: {k}/{n}" for (q, m, fs), (k, n) in sorted(counts.items())))
    claim(10, 'fs64 at q=256, m=2048 certifies in 0 of 4 draws (rho_max 0.130–0.242); bet101\'s 0.1074 was the only passing draw of five',
          "`bet301 rules[q256 reprow64m2048s*]`, `bet101 rules[q256 fs64 m2048]`", 'recount; min/max',
          counts[(256, 2048, 64)] == (0, 4) and spread[(256, 2048, 64)][0] > bars['primary']
          and abs(chosen[256]['rho_max'] - 0.1074) < 5e-4 and chosen[256]['rho_max'] <= bars['primary'],
          f"draws min/median/max {spread[(256, 2048, 64)][0]:.4f}/{spread[(256, 2048, 64)][1]:.4f}/{spread[(256, 2048, 64)][2]:.4f}; bet101 {chosen[256]['rho_max']:.4f}")
    claim(11, 'fs64 at q=128, m=2048 certifies in 3 of 4 draws (4 of 5 with bet101\'s)', "`bet301 rules[q128 reprow64m2048s*]`", 'recount',
          counts[(128, 2048, 64)] == (3, 4) and chosen[128]['rho_max'] <= bars['primary'],
          f"draws {spread[(128, 2048, 64)][0]:.4f}/{spread[(128, 2048, 64)][1]:.4f}/{spread[(128, 2048, 64)][2]:.4f}; bet101 {chosen[128]['rho_max']:.4f}")
    b64 = next(x for x in B['rules'] if x['q'] == 64 and x['arm'] == 'fs64' and x['m_target'] == 2048 and x['source'] == 'this_job')
    claim(12, 'fs64 at q=64, m=2048 certifies in 4 of 4 replication draws but bet201\'s earlier draw of it failed (0.1248): 4 of 5',
          "`bet301 rules[q64 reprow64m2048s*]`, `bet201 rules[q64 fs64 m2048]`", 'recount + the bet201 value',
          counts[(64, 2048, 64)] == (4, 4) and b64['rho_max'] > bars['primary'],
          f"replication {counts[(64, 2048, 64)][0]}/4; bet201 {b64['rho_max']:.4f}")
    q64 = next(x for x in B['rules'] if x['q'] == 64 and x['arm'] == 'std' and x['m_target'] == 1024 and x['source'] == 'this_job')
    a64 = next(x for x in A['rules'] if x['q'] == 64 and x['source'] == 'qrg304' and x['arm'] == 'reachable' and x['m_target'] == 1024)
    claim(13, 'the incumbent construction at q=64, m=1024 (the rule the ladder ran, 0.0531) certifies in 1 of 4 replication draws; 2 of 6 over all draws',
          "`bet301 rules[q64 reprowincumbentm1024s*]`, `bet201 q64 std m1024`, `qrg304 q64 reachable m1024`", 'recount',
          counts[(64, 1024, 25)] == (1, 4) and q64['rho_max'] > bars['primary'] >= a64['rho_max'],
          f"replication {counts[(64, 1024, 25)][0]}/4; bet201 {q64['rho_max']:.4f}; qrg304 {a64['rho_max']:.4f}")
    claim(14, 'the incumbent construction never certifies at q=128 (0/4 at m=1024) or q=256 (0/4 at m=1024)', "`bet301 rules[reprowincumbent*]`", 'recount',
          counts[(128, 1024, 14)] == (0, 4) and counts[(256, 1024, 8)] == (0, 4),
          f"q128 {spread[(128, 1024, 14)][0]:.4f}–{spread[(128, 1024, 14)][2]:.4f}; q256 {spread[(256, 1024, 8)][0]:.4f}–{spread[(256, 1024, 8)][2]:.4f}")
    # 15 summary verdict rows agree with this recount
    V = {v['q']: v for v in S['verdict_per_rung']}
    ok = (V[256]['draws_certifying_primary'] == 1 and V[256]['draws'] == 5 and 'marginal' in V[256]['ladder_rule_status']
          and V[128]['draws_certifying_primary'] == 4 and V[128]['draws'] == 5 and 'marginal' in V[128]['ladder_rule_status']
          and V[64]['draws_certifying_primary'] == 2 and V[64]['draws'] == 6 and 'marginal' in V[64]['ladder_rule_status']
          and all('confirmed' in V[q]['ladder_rule_status'] for q in (0, 16, 32)))
    claim(15, 'summary.json per-rung verdict: marginal at q=64 (2/6), 128 (4/5), 256 (1/5); confirmed at q<=32', "`summary.verdict_per_rung`",
          'compare the counts with the recount from the raw results', ok,
          '; '.join(f"q{q}: {V[q]['ladder_rule_status']}" for q in sorted(V)))
    # 16 exported rules: files hash, each rule certified in its own draw, policy outcomes
    ok = True
    for x in prov['rules']:
        f = HERE / 'certified-rules' / x['file']
        ok &= hashlib.sha256(f.read_bytes()).hexdigest() == x['sha256']
        ok &= x['held_out']['rho_max'] <= bars['primary'] and not x['truncated']
    exp = {x['q']: x for x in prov['rules']}
    ok &= exp[256]['m'] == 2560 and exp[256]['arm'] == 'fs64' and abs(exp[256]['held_out']['rho_max'] - 0.0647) < 5e-4
    ok &= exp[128]['m'] > 2048 and exp[64]['m'] == 2048 and all(exp[q]['m'] == 1024 for q in (0, 16, 32))
    ok &= all('confirmed' in exp[q]['construction']['status'] for q in (0, 16, 32, 64))
    ok &= all('one draw' in exp[q]['construction']['status'] for q in (128, 256))
    ok &= {s_['file'] for s_ in prov['superseded']} == {'rule_q64_m1024_qrg304_reachable.npz', 'rule_q128_m2048_bet101_fs64.npz', 'rule_q256_m2048_bet101_fs64.npz'}
    claim(16, 'exported set: 6 files hash-verified, each certified in its own draw; q<=64 from confirmed constructions, q=128/256 single-draw above m=2048; the three superseded files are listed',
          "`certified-rules/PROVENANCE.json`, files", 'hash each file; check m, arm, status, superseded list',
          ok and len(prov['rules']) == len(QS), '; '.join(f"q{q}: {exp[q]['file']} [{exp[q]['construction']['status']}]" for q in QS))
    # 17 summary.json integrity
    sha = hashlib.sha256(text.encode()).hexdigest()
    ok = S['report_sha256'] == sha and all(('job_id' in x and x.get('status')) for x in S['rows']) and S['status'] == 'final' and S['pending'] == []
    claim(17, 'summary.json hashes this report, status final, nothing pending, every row carries a job id and a status', "`summary.json`",
          'sha256 of the report file; row fields', ok, f"{len(S['rows'])} rows, status '{S['status']}', pending {S['pending']}")
    # 18 the text says marginal where it must
    first = text.splitlines()[0]
    ok = ('does not survive the draw replication' in first and text.count('marginal') >= 10
          and 'marginal at m=2048 (1/5)' in text and 'marginal at m=2048 (4/5)' in text and 'marginal at m=1024 (2/6)' in text
          and 'provisional' not in first)
    claim(18, 'the report title and body say the certification is marginal at q>=64 with the draw counts; nothing is still marked provisional in the title',
          'report text', 'title; occurrences of the status labels', ok,
          f"'marginal' occurs {text.count('marginal')} times; title: {first[:120]}…")
    # 19 tight at 256
    t = [x for x in rules if x['q'] == 256 and x['certified_tight']]
    best = min(x['rho_max'] for x in rules if x['q'] == 256)
    claim(19, 'no tight-certified rule at q=256; best rho_max 0.0647 (one draw)', "`rules[q=256]`", 'none flagged; min rho_max', not t and abs(best - 0.0647) < 5e-4,
          f"best {best:.4f}")
    # 20 spread ratio claims
    sr = {k: spread[k][2] / spread[k][0] for k in spread}
    claim(20, 'within-replication spread of rho_max is 1.4×–9.6× across the six configurations', "`bet301 rules[*].certification.rho_max`",
          'max/min per configuration', 1.3 < min(sr.values()) < 1.5 and 9 < max(sr.values()) < 10,
          '; '.join(f"q{q} m{m} fs{fs}: {v:.2f}×" for (q, m, fs), v in sorted(sr.items())))
    # 21 the 40 GB card and the allocator warnings did not change a number
    err = (HERE / 'runs/bet301/archive/logs' / f"{C['job_id']}.err").read_text()
    n_oom = err.count('ran out of memory')
    claim(21, 'bet301 ran on a 40 GB A100 with 4 BFC-allocator warnings; every fit certified, the archive rules re-certified to 1e-9, ALL-DONE',
          "`.err`, `gates.archived_rules_recertify`, `.out`", 'count warnings; gate; the log tail',
          n_oom == 4 and rc3['worst_relative_difference'] < 1e-6 and 'A100-PCIE-40GB' in r3['gpu']
          and (HERE / 'runs/bet301/archive/logs' / f"{C['job_id']}.out").read_text().rstrip().endswith('ALL-DONE'),
          f"{n_oom} warnings; {r3['gpu']}; recertify {rc3['worst_relative_difference']:.2e}")

    md = ['# Self-audit of the b-eqtop final report, 2026-09-17 (substitute for the unavailable Codex audit)', '',
          f'Generated by `checks/self_audit_report.py` against `checks/bet101-audit.json`, `checks/bet201-audit.json`, '
          f'`checks/bet301-audit.json`, `artifacts/bet101/result.json`, `artifacts/bet301/result.json`, `reports/summary.json`, '
          f'`certified-rules/PROVENANCE.json` and the report `{Path(a.report).name}` (SHA256 `{sha}`). Each row: the claim, '
          'the field it rests on, the check recomputed here outside the generator and outside `draws.py` (the replication '
          'counts are recounted from the raw `result.json`), the outcome. Codex quota is exhausted until 2026-09-19 11:33 '
          '(DESIGN A3, A4); this is the substitution.', '',
          '| # | claim | rests on | check | outcome | detail |', '|---|---|---|---|---|---|']
    md += [f'| {n} | {w} | {r_} | {c} | **{o}** | {d} |' for n, w, r_, c, o, d in rows]
    md += ['', f"**Result: {len(rows) - len(bad)} of {len(rows)} checks pass{'' if not bad else '; FAILED: ' + str(bad)}.**", '',
           'What this audit cannot check: the held-out $\\rho$ values themselves are the driver\'s (recomputing them needs the '
           'bank on the GPU and the collected states, which are not archived); the timings and iteration counts are the '
           'driver\'s. Scientific weaknesses still standing: one training seed, one checkpoint, six development cases; '
           '$\\rho_{\\max}$ over 512 held-out states is a tail statistic whose sampling noise over a second held-out draw is '
           'unmeasured; the tight bar 0.06 is the coordinator\'s number; the exported rules at $q = 128$ and $256$ are single '
           'draws whose re-draw has not been measured; and the timed ladder was run only with the draws that passed, so '
           'whether a failing draw ($\\rho_{\\max} \\approx 0.2$ at $q = 256$) also gives a monotone ladder is unknown.']
    Path(a.out).write_text('\n'.join(md) + '\n')
    print('\n'.join(f'{n:2d} {o:4s} {w[:80]}' for n, w, _, _, o, _ in rows))
    print('wrote', a.out)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
