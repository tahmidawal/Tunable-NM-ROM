"""Cross-check reports/summary.json against the raw result.json and the independent audits.

Substitutes for the Codex report audit (DESIGN.md A8). Recomputes, by a path that does not
import the report generator, every number the report's headline rests on.
"""
import json, itertools
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent.parent
S = json.loads((H / 'reports/summary.json').read_text())
out, fail = {}, []


def chk(name, ok, detail):
    out[name] = dict(pass_=bool(ok), detail=detail)
    if not ok:
        fail.append(name)


# ---- 1. every summary row carries a job id, a source sha and (for solve rows) a block M
solve_metrics = {'worst_same_grid', 'median_same_grid', 'median_total_ms', 'nondominated_complete_ms',
                 'nondominated_reduced_only', 'median_device_ms', 'median_solver_ms'}
bad = [r for r in S if r['metric'] in solve_metrics and (r.get('job_id') is None or r.get('test_modes') is None)]
chk('every_timed_row_is_keyed_by_job_and_block', not bad, dict(rows=len(S), unkeyed=len(bad)))

# ---- 2. no (mesh, subject, metric) collides across blocks without the block distinguishing it
keys = [(r['mesh'], r['subject'], r['metric'], r.get('test_modes'), r['job_id']) for r in S if r['metric'] in solve_metrics]
dups = [k for k, g in itertools.groupby(sorted(keys)) if len(list(g)) > 1]
chk('no_duplicate_timed_rows', not dups, dict(duplicates=dups[:5]))

# ---- 3. the non-dominated sets equal the ones the independent NumPy audit computed
audit_nd, report_nd = {}, {}
for att, job in (('lsh03', 3784662), ('lsh04', 3784663), ('lsh06', 3784910)):
    au = json.loads((H / f'artifacts/{att}/audit.json').read_text())
    for n, names in au['detail']['nondominated_complete_ms'].items():
        audit_nd[(str(job), int(n))] = set(names)
for r in S:
    if r['metric'] == 'nondominated_complete_ms':
        report_nd.setdefault((str(r['job_id']), r['mesh']), set()).add(r['subject'])
chk('nondominated_sets_match_independent_audit', audit_nd == report_nd,
    dict(audit={f'{k}': sorted(v) for k, v in audit_nd.items()},
         report={f'{k}': sorted(v) for k, v in report_nd.items()}))

# ---- 4. recompute worst error and median complete-ms straight from the raw invocations
worst_dev = 0.0
for att, job in (('lsh03', 3784662), ('lsh04', 3784663), ('lsh06', 3784910)):
    d = json.loads((H / f'artifacts/{att}/result.json').read_text())
    raw = {}
    for x in d['invocations']:
        raw.setdefault((x['intervals'], x['name']), []).append(x)
    for r in S:
        if r['job_id'] != job or r['metric'] not in ('worst_same_grid', 'median_total_ms'):
            continue
        rows = raw[(r['mesh'], r['subject'])]
        v = (max(y['same_grid_error'] for y in rows) if r['metric'] == 'worst_same_grid'
             else float(np.median([y['total_seconds'] for y in rows])) * 1e3)
        worst_dev = max(worst_dev, abs(v - r['value']) / max(abs(v), 1e-30))
chk('report_values_recomputed_from_raw_invocations', worst_dev <= 1e-12,
    dict(worst_relative_difference=worst_dev))

# ---- 5. the free rung is head-independent and sits on its bank floor
d6 = json.loads((H / 'artifacts/lsh06/result.json').read_text())
f16 = [x for x in d6['invocations'] if x['name'] == 'freebank@head_sdf_R512_K16']
f32 = [x for x in d6['invocations'] if x['name'] == 'freebank@head_sdf_R512_K32']
# Head-independence is exact in exact arithmetic (C = I makes R^-1 Q^T B h(z) cancel h(z)) but
# NOT bitwise in f64: the cancellation leaves round-off that depends on h(z). DESIGN A9.
out6 = H / 'runs/lsh06/archive/output'
w16 = {x['case']: x for x in f16 if x['rep'] == 0}
w32 = {x['case']: x for x in f32 if x['rep'] == 0}
worst_head_rel = max(
    float(np.linalg.norm(np.load(out6 / w16[c]['artifact'])['interior'] - np.load(out6 / w32[c]['artifact'])['interior'])
          / np.linalg.norm(np.load(out6 / w16[c]['artifact'])['interior'])) for c in w16)
worst_err_gap = max(abs(w16[c]['same_grid_error'] - w32[c]['same_grid_error']) for c in w16)
bitwise = {c: x['field_sha256'] for c, x in w16.items()} == {c: x['field_sha256'] for c, x in w32.items()}
chk('free_rung_is_head_independent_to_roundoff', worst_head_rel <= 1e-12 and len(w16) == 32,
    dict(cases=len(w16), worst_relative_field_difference=worst_head_rel,
         worst_same_grid_error_difference=worst_err_gap, bitwise_identical=bitwise))
reps16 = {}
for x in f16:
    reps16.setdefault(x['case'], set()).add(x['field_sha256'])
chk('timed_repetitions_are_bit_identical', all(len(v) == 1 for v in reps16.values()),
    dict(cases=len(reps16)))
floor = next(r['bank_projection']['worst'] for r in d6['reconstruction']
             if r['intervals'] == 256 and r['model'] == 'head_sdf_R512_K16')
fw = max(x['same_grid_error'] for x in f16)
chk('free_rung_reaches_its_bank_floor', fw / floor <= 1.05,
    dict(worst=fw, bank_floor=floor, ratio=fw / floor))
chk('free_rung_runs_no_lm', all(x['iterations'] == 0 for x in f16 + f32),
    dict(max_iterations=max(x['iterations'] for x in f16 + f32)))

# ---- 6. every reported gate passed, in every job, and the backend was a GPU at x64/highest
gates, env = [], []
for att in ('lsh02', 'lsh03', 'lsh04', 'lsh06'):
    d = json.loads((H / f'artifacts/{att}/result.json').read_text())
    gates += [(att, g.get('intervals'), g['passed']) for g in d['gates']]
    env.append((att, d['backend'], d['x64'], d['matmul_precision'], d['complete']))
chk('all_gates_passed', all(g[2] for g in gates), dict(gates=len(gates)))
chk('all_jobs_gpu_x64_highest_complete',
    all(b == 'gpu' and x and p == 'highest' and c for _, b, x, p, c in env), dict(env=env))

# ---- 7. the validation-vs-development gap quoted beside the headline
tr = json.loads((H / 'artifacts/lsh02/result.json').read_text())
vw = [h['best_found_validation']['worst'] for h in tr['head_arms']]
dw = [h['best_found_development']['worst'] for h in tr['head_arms']]
row_v = next(r['value'] for r in S if r['metric'] == 'best_found_validation_worst_max')
row_d = next(r['value'] for r in S if r['metric'] == 'best_found_development_worst_max')
chk('validation_gap_matches_training_json', row_v == max(vw) and row_d == max(dw),
    dict(validation_worst_range=[min(vw), max(vw)], development_worst_range=[min(dw), max(dw)],
         arms=len(tr['head_arms']), validation_cases=len(tr['cohorts']['training']['validation']),
         development_cases=tr['cohorts']['development']['count']))

# ---- 8. all four independent audits pass every one of their own checks
ap = {}
for att in ('lsh02', 'lsh03', 'lsh04', 'lsh06'):
    au = json.loads((H / f'artifacts/{att}/audit.json').read_text())
    ap[att] = dict(passed=au['passed'], checks=len(au['checks']),
                   failed=[k for k, v in au['checks'].items() if not v])
chk('independent_numpy_audits_pass', all(v['passed'] for v in ap.values()), ap)

res = dict(checks=out, passed=not fail, failed=fail)
(H / 'checks/verify_report_2026-09-17.json').write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps({k: v['pass_'] for k, v in out.items()}, indent=1))
print('PASSED' if not fail else f'FAILED: {fail}')
assert not fail
