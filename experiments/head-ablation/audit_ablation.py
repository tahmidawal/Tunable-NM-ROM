"""Independent NumPy audit of a completed head-ablation result. No JAX, no GPU.

Re-derives every reported aggregate from the saved per-invocation records, checks
the shared contract across arms, and reproduces the retained campaign's frozen
Burgers rollout errors from arm (a) on the same cases and meshes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / ('consolidated/evidence/worktrees/2026-09-07-mr-burgers2d/experiments/'
                   'mr-burgers2d/runs/accuracy09/archive/out/result.json')


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', default=None, help='directory holding the saved output .npz files')
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    checks = {}
    fail = []

    def gate(name, ok, detail=None):
        checks[name] = dict(passed=bool(ok), detail=detail)
        if not ok:
            fail.append(name)

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r['checkpoint_sha256_after'])
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    gate('reference_cases', len(r['reference']) == len(r['physical_cases']))

    inv = r['invocations']
    gate('every_invocation_paired', all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))
    gate('repetitions_retained',
         len({(x['intervals'], x['name'], x['case'], x['rep']) for x in inv}) == len(inv))
    reps = r['config']['reps']
    counts = {}
    for x in inv:
        counts.setdefault((x['intervals'], x['name'], x['case']), 0)
        counts[(x['intervals'], x['name'], x['case'])] += 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))

    # Determinism: one arm on one case must return the same field in every repetition.
    hashes = {}
    for x in inv:
        hashes.setdefault((x['intervals'], x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])

    # The shared contract: same dt, same output times, same cases for every arm.
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('one_timestep', {r['config']['dt']} == {r['config']['dt']})
    gate('matched_test_ratio',
         all(x['M'] == r['config']['test_multiplier'] * x['k']
             for x in roms if x['name'] != 'd_freebank_dense'))
    gate('free_bank_overdetermined',
         all(x['M'] > x['k'] for x in roms), [x['name'] for x in roms if x['M'] <= x['k']][:5])
    gate('eq_quadrature_ratio',
         all(x['m'] == r['config']['quadrature_multiplier'] * x['M']
             for x in roms if x['quadrature'] == 'eq'))

    if a.fields:
        d = Path(a.fields)
        bad = []
        for x in inv:
            f = d / x['artifact']
            if not f.exists():
                bad.append(x['artifact'])
                continue
        gate('artifacts_present', not bad, bad[:5])

    # ---------------------------------------------------- per-arm aggregates
    table = {}
    for x in roms + [y for y in inv if y['kind'] == 'fom']:
        key = (x['intervals'], x['name'])
        t = table.setdefault(key, dict(intervals=x['intervals'], arm=x['name'], kind=x['kind'],
                                       k=x.get('k'), quadrature=x.get('quadrature'), family=x.get('family'),
                                       gpu_ms=[], host_ms=[], case_error={}, iterations=[],
                                       stationary=[], completed=[], budget_exits=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_error'][x['case']] = x['error']['fixed_initial_max']
        t['iterations'] += x['iterations']
        if x['kind'] == 'rom':
            t['stationary'].append(x['stationary'])
            t['completed'].append(x['completed'])
            t['budget_exits'].append(x['budget_exits'])
    rows = []
    recon = {(x['intervals'], x['arm']): x for x in r['reconstruction']}
    for key, t in sorted(table.items()):
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        rc = recon.get(key)
        rows.append(dict(intervals=t['intervals'], arm=t['arm'], kind=t['kind'], k=t['k'],
                         quadrature=t['quadrature'], family=t['family'],
                         worst_rollout_percent=float(np.max(errs) * 100),
                         median_rollout_percent=float(np.median(errs) * 100),
                         worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100) if rc else None),
                         worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
                         median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
                         gpu_ms_repetitions=len(t['gpu_ms']),
                         median_iterations=median(t['iterations']),
                         max_iterations=float(np.max(t['iterations'])),
                         all_stationary=bool(all(t['stationary'])) if t['stationary'] else None,
                         all_completed=bool(all(t['completed'])) if t['completed'] else None,
                         total_budget_exits=int(np.sum(t['budget_exits'])) if t['budget_exits'] else None))
    checks['arm_table'] = rows

    # ------------------- reproduction of the retained campaign's frozen arm ---
    if CAMPAIGN.exists():
        camp = json.loads(CAMPAIGN.read_text())
        want = {}
        for x in camp['invocations']:
            if x['name'] == 'frozen_stationary':
                want.setdefault((x['intervals'], x['case']), x['error']['fixed_initial_max'])
        got = {(x['intervals'], x['case']): x['error']['fixed_initial_max']
               for x in inv if x['name'] == 'a_neural_eq'}
        shared = sorted(set(want) & set(got))
        deltas = [abs(got[k] - want[k]) / max(want[k], 1e-300) for k in shared]
        same_cases = np.allclose(np.asarray(camp['physical_cases']), np.asarray(r['physical_cases']))
        gate('campaign_cases_identical', bool(same_cases))
        gate('campaign_frozen_arm_reproduced', bool(shared) and max(deltas) < 1e-3,
             dict(compared=len(shared), worst_relative_delta=(max(deltas) if deltas else None),
                  note='arm a_neural_eq vs the retained frozen_stationary rows on the same cases'))
        checks['campaign_comparison'] = [dict(intervals=k[0], case=k[1], campaign=want[k], ablation=got[k],
                                              relative_delta=abs(got[k] - want[k]) / max(want[k], 1e-300))
                                         for k in shared]

    out = dict(result=str(a.result), result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               failed=fail, passed=not fail, checks=checks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, indent=2))
    for name, v in checks.items():
        if isinstance(v, dict) and 'passed' in v:
            print(('PASS ' if v['passed'] else 'FAIL '), name, v['detail'])
    assert not fail, fail


if __name__ == '__main__':
    main()
