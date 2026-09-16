"""Independent NumPy audit of the Burgers head-refinement job. No JAX, no GPU.

Recomputes every reported error from the retained output fields, measures each arm
against the converged same-mesh full-order solve, re-decodes the saved refined
weights at a fixed node sample, and checks n = 0 against the head-ablation job's
arm (a) on the same cases.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_common import head_np, median

HERE = Path(__file__).resolve().parent
GATE = HERE / 'gate-reference-burgers.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', default=None, help='directory holding the saved .npz outputs')
    a = p.parse_args()
    r = json.loads(Path(a.result).read_text())
    checks, fail = {}, []

    def gate(name, ok, detail=None):
        checks[name] = dict(passed=bool(ok), detail=detail)
        if not ok:
            fail.append(name)

    gate('complete', r.get('complete') is True)
    gate('backend_gpu', r['backend'] == 'gpu', r['backend'])
    gate('x64', r['x64'] is True)
    gate('precision_highest', r['matmul_precision'] == 'highest')
    gate('spatial_bank_frozen', r['spatial_bank_frozen'] is True)
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r['checkpoint_sha256_after'])
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    gate('calibration_case_not_evaluation',
         r['calibration_case']['sha256'] not in
         {hashlib.sha256(np.ascontiguousarray(np.asarray(c)).tobytes()).hexdigest()
          for c in r['physical_cases']})
    gate('one_step_size_everywhere',
         len({x['alpha'] for x in r['invocations'] if x['kind'] == 'rom'}) == 1,
         sorted({x['alpha'] for x in r['invocations'] if x['kind'] == 'rom'}))
    gate('declared_anchor_weights_only',
         {x['mu'] for x in r['invocations'] if x['kind'] == 'rom' and x['n'] > 0}
         == set(r['config']['mu'].values()))

    inv = r['invocations']
    reps = r['config']['reps']
    counts, hashes = {}, {}
    for x in inv:
        counts[(x['name'], x['case'])] = counts.get((x['name'], x['case']), 0) + 1
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])
    gate('every_invocation_paired',
         all('error' in x and 'gpu_seconds' in x and x['gpu_seconds'] > 0 and x['finite'] for x in inv))
    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms))
    gate('n_zero_is_one_arm', len({x['name'] for x in roms if x['n'] == 0}) == 2,
         sorted({x['name'] for x in roms if x['n'] == 0}))
    gate('n_zero_has_no_drift',
         all(max(x['drift']) == 0.0 for x in roms if x['n'] == 0))
    gate('refined_arms_moved', all(max(x['drift']) > 0.0 for x in roms if x['n'] > 0),
         sorted({x['name'] for x in roms if x['n'] > 0 and max(x['drift']) == 0.0}))

    per_time = {}
    redecode = None
    if a.fields:
        d = Path(a.fields)
        bad = [x['artifact'] for x in inv if not (d / x['artifact']).exists()]
        gate('artifacts_present', not bad, bad[:5])
        if not bad:
            refs = {e['case']: np.load(d / e['artifact'])['fields'] for e in r['reference']}
            base = {x['case']: x['artifact'] for x in inv if x['name'] == 'fft_tight'}
            cache = {}

            def fields_of(name):
                if name not in cache:
                    cache[name] = np.load(d / name)['fields']
                return cache[name]

            worst = 0.
            for x in inv:
                truth = refs[x['case']]
                f = fields_of(x['artifact'])
                n0 = np.linalg.norm(truth[0])
                err = np.linalg.norm((f - truth).reshape(len(f), -1), axis=1) / n0
                worst = max(worst, float(np.max(np.abs(err - np.asarray(
                    x['error']['fixed_initial_per_time']))))
                    / max(x['error']['fixed_initial_max'], 1e-300))
                g = fields_of(base[x['case']])
                sg = np.linalg.norm((f - g).reshape(len(f), -1), axis=1) / n0
                x['same_grid_per_time'] = sg.tolist()
                x['same_grid_max'] = float(np.max(sg))
            gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

            # ---- gate (v): re-decode the saved refined weights in pure NumPy ----
            rd = np.load(d / r['redecode']['artifact'])
            nodes, Gsub, theta0 = rd['nodes'], rd['bank'], rd['theta0']
            shapes = r['theta_layout']['shapes']
            gate('theta_layout_total', int(np.sum([int(np.prod(s)) for s in shapes])) == len(theta0))
            seen, worst_rd = {}, 0.
            for entry in r['reconstruction']:
                art = entry.get('weights_artifact')
                W = np.load(d / art) if art else None
                for row in entry['cases']:
                    case = row['case']
                    inv_row = next(x for x in inv if x['name'] == entry['arm'] and x['case'] == case)
                    npz = np.load(d / inv_row['artifact'])
                    z = np.asarray(npz['latents'])[-1]
                    theta = np.asarray(W[f'case{case}']) if W is not None else theta0
                    got = Gsub @ head_np(theta, z, shapes)
                    want = np.asarray(npz['fields'])[-1][1:-1, 1:-1].ravel()[nodes]
                    rel = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300))
                    worst_rd = max(worst_rd, rel)
                    seen[(entry['arm'], case)] = rel
            redecode = dict(compared=len(seen), worst_relative=worst_rd, tolerance=1e-10,
                            nodes=int(len(nodes)))
            gate('refined_weights_redecode', worst_rd < 1e-10, redecode)

    # -------------------------------------------------------- per-arm aggregates
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(
            arm=x['name'], kind=x['kind'], variant=x.get('variant'), n=x.get('n'),
            mu=x.get('mu'), mu_key=x.get('mu_key'), quadrature=x.get('quadrature'),
            M=x.get('M'), m=x.get('m'), gpu_ms=[], host_ms=[], case_error={}, case_same_grid={},
            case_same_grid_per_time={}, iterations=[], resolve_iterations=[], refine_iterations=[],
            stationarity=[], stationary=[], completed=[], budget_exits=[], rejected_exits=[],
            drift=[], latent_solves=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_error'][x['case']] = x['error']['fixed_initial_max']
        if 'same_grid_max' in x:
            t['case_same_grid'][x['case']] = x['same_grid_max']
            t['case_same_grid_per_time'][x['case']] = x['same_grid_per_time']
        t['iterations'] += x['iterations']
        if x['kind'] == 'rom':
            t['resolve_iterations'] += x['resolve_iterations']
            t['refine_iterations'] += x['refine_iterations']
            t['stationarity'].append(x['step_stationarity'] + [x['ic_stationarity']]
                                     + x['resolve_stationarity'] + x['refine_stationarity'])
            t['stationary'].append(x['stationary'])
            t['completed'].append(x['completed'])
            t['budget_exits'].append(x['budget_exits'])
            t['rejected_exits'].append(x['rejected_exits'])
            t['drift'].append(max(x['drift']))
            t['latent_solves'].append(x['latent_solves'])
    recon = {e['arm']: e for e in r['reconstruction']}
    rows = []
    for name, t in table.items():
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        sg = (np.array([t['case_same_grid'][c] for c in sorted(t['case_same_grid'])])
              if t['case_same_grid'] else None)
        sgt = None
        if t['case_same_grid_per_time']:
            arr = np.array([t['case_same_grid_per_time'][c]
                            for c in sorted(t['case_same_grid_per_time'])])
            sgt = np.max(arr, axis=0).tolist()
        rc = recon.get(name)
        st = [v for row in t['stationarity'] for v in row]
        rows.append(dict(
            arm=name, kind=t['kind'], variant=t['variant'], n=t['n'], mu=t['mu'],
            mu_key=t['mu_key'], quadrature=t['quadrature'], M=t['M'], m=t['m'],
            worst_reference_percent=float(np.max(errs) * 100),
            median_reference_percent=float(np.median(errs) * 100),
            worst_same_grid_percent=(float(np.max(sg) * 100) if sg is not None else None),
            median_same_grid_percent=(float(np.median(sg) * 100) if sg is not None else None),
            worst_same_grid_per_time_percent=([v * 100 for v in sgt] if sgt else None),
            worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100) if rc else None),
            worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
            worst_drift=(float(np.max(t['drift'])) if t['drift'] else None),
            median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
            gpu_ms_repetitions=len(t['gpu_ms']),
            median_iterations=median(t['iterations']),
            median_resolve_iterations=(median(t['resolve_iterations'])
                                       if t['resolve_iterations'] else None),
            median_refine_iterations=(median(t['refine_iterations'])
                                      if t['refine_iterations'] else None),
            latent_solves=(int(np.max(t['latent_solves'])) if t['latent_solves'] else None),
            max_stationarity=(float(np.max(st)) if st else None),
            all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
            all_completed=(bool(all(t['completed'])) if t['completed'] else None),
            total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
            total_rejected_exits=(int(np.sum(t['rejected_exits'])) if t['rejected_exits'] else None)))
    rows.sort(key=lambda x: (x['kind'] != 'rom', str(x['variant']), x['n'] or 0,
                             x['mu'] or 0, x['arm']))
    checks['arm_table'] = rows
    checks['redecode'] = redecode

    # ------------------------------ gate (ii): n0 against the head-ablation job
    if GATE.exists():
        g = json.loads(GATE.read_text())
        want = {x['case']: x for x in g['cases']}
        got = {x['case']: x for x in inv if x['name'] == 'n0'}
        shared = sorted(set(want) & set(got))
        deltas = [abs(got[c]['error']['fixed_initial_max'] - want[c]['fixed_initial_max'])
                  / max(want[c]['fixed_initial_max'], 1e-300) for c in shared]
        gate('n0_reproduces_head_ablation_arm_a', bool(shared) and max(deltas) <= 1e-9,
             dict(compared=len(shared), worst_relative_delta=(max(deltas) if deltas else None),
                  bitwise_identical_fields=int(sum(1 for c in shared if
                                                   want[c]['field_sha256'] == got[c]['field_sha256'])),
                  tolerance=1e-9, reference_job=g['job_id'], reference_gpu=g['gpu'],
                  this_gpu=r.get('gpu')))
        gate('cases_identical', bool(np.allclose(np.asarray(g['physical_cases']),
                                                 np.asarray(r['physical_cases']))))
        gate('in_job_gate_agrees',
             abs((r['gate_in_job']['worst_relative_delta'] or 0) - (max(deltas) if deltas else 0))
             <= 1e-12 * max(1.0, max(deltas) if deltas else 1.0))

    out = dict(result=str(a.result),
               result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               selected_alpha=r['calibration']['selected_alpha'],
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
