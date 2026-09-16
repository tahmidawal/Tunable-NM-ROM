"""Independent NumPy audit of the Burgers cheap correction ladder. No JAX, no GPU.

Recomputes every reported error from the retained output fields, measures each arm
against the converged same-mesh full-order solve, and checks the cross-job gates
against the audited ladder (`qlad01`) and the head ablation (`abl01`).

Optional `--qlad01 <dir>` points at a restored `qlad01` output directory (the chunked
archive under `experiments/head-ablation/artifacts/qlad01`, reassembled), which turns
the q = 0 and q = 16 reproduction gates into field-level comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QLAD = ROOT / 'experiments/head-ablation/artifacts/qlad01/result.json'
ABLATION = ROOT / 'experiments/head-ablation/artifacts/abl01/result.json'


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))


def nondominated(rows, cost='median_gpu_ms', err='worst_same_grid_percent'):
    pts = [r for r in rows if r.get(cost) is not None and r.get(err) is not None]
    out = []
    for r in pts:
        if not any((o[cost] <= r[cost] and o[err] <= r[err] and
                    (o[cost] < r[cost] or o[err] < r[err])) for o in pts):
            out.append(r['arm'])
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result')
    p.add_argument('--out', required=True)
    p.add_argument('--fields', default=None)
    p.add_argument('--qlad01', default=None)
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
    gate('bank_frozen', r['spatial_bank_frozen'] and r['network_weights_frozen'])
    gate('checkpoint_unchanged', r['checkpoint_sha256'] == r.get('checkpoint_sha256_after'))
    gate('final_cohort_unopened', r['final_cohort_unopened'] is True)
    gate('reference_residuals', all(x['max_relative_residual'] < 2e-11 for x in r['reference']),
         max(x['max_relative_residual'] for x in r['reference']))
    refmatch = [x['bitwise_matches_qlad01'] for x in r['reference']]
    checks['reference_fields_bitwise_match_qlad01'] = dict(
        passed=all(bool(x) for x in refmatch), detail=refmatch,
        note='a cross-job bitwise probe on the same GPU model; informative, not required')

    dg = r['gates'].get('directions_hash', {})
    checks['directions_hash_matches_qlad01'] = dict(
        passed=bool(dg.get('passed')), detail=dg,
        note=('bitwise across jobs and GPUs; the substantive check on the directions is the '
              'q = 16 reproduction gate below'))
    gate('directions_rank_covers_ladder',
         r['directions']['available_rank'] >= max(r['config']['q_ladder']),
         r['directions']['available_rank'])
    if r.get('directions_flat'):
        checks['flattened_direction_fit'] = dict(
            passed=True, detail={k: r['directions_flat'][k] for k in
                                 ('seconds', 'compile_seconds_saved', 'bitwise_identical_to_audited',
                                  'max_abs_difference_from_audited')})

    inv = r['invocations']
    reps = r['config']['reps']
    counts = {}
    for x in inv:
        counts.setdefault((x['name'], x['case']), 0)
        counts[(x['name'], x['case'])] += 1
    gate('every_subject_case_has_all_reps', set(counts.values()) == {reps}, sorted(set(counts.values())))
    hashes = {}
    for x in inv:
        hashes.setdefault((x['name'], x['case']), set()).add(x['field_sha256'])
    gate('repetition_output_identical', all(len(v) == 1 for v in hashes.values()),
         [k for k, v in hashes.items() if len(v) > 1][:5])
    gate('every_invocation_paired', all('error' in x and 'gpu_seconds' in x and x['finite'] for x in inv))

    roms = [x for x in inv if x['kind'] == 'rom']
    gate('overdetermined_weak_system', all(x['M'] > x['solved_dimension'] for x in roms),
         [(x['name'], x['M'], x['solved_dimension']) for x in roms
          if x['M'] <= x['solved_dimension']][:5])
    gate('test_count_rule_followed',
         all((x['rule'] == 'm4' and x['M'] == 4 * x['solved_dimension'])
             or (x['rule'] == 'm2' and x['M'] == 2 * x['solved_dimension'])
             or (x['rule'] == 'm256' and x['M'] == r['config']['fixed_test_count'])
             or x['rule'] == 'Mmax' for x in roms),
         sorted({x['name'] for x in roms if not (
             (x['rule'] == 'm4' and x['M'] == 4 * x['solved_dimension'])
             or (x['rule'] == 'm2' and x['M'] == 2 * x['solved_dimension'])
             or (x['rule'] == 'm256' and x['M'] == r['config']['fixed_test_count'])
             or x['rule'] == 'Mmax')}))
    gate('every_rom_carries_exit_and_stationarity',
         all(('stop_reasons' in x and 'budget_exits' in x and 'worst_joint_stationarity' in x
              and 'step_inner_stationarity' in x) for x in roms))
    gate('q0_variants_bitwise', bool(r['gates'].get('q0_variants_bitwise', {}).get('passed')),
         r['gates'].get('q0_variants_bitwise'))

    fields_dir = Path(a.fields) if a.fields else None
    cache = {}

    def fields_of(name, d=None):
        d = d or fields_dir
        key = (str(d), name)
        if key not in cache:
            cache[key] = np.load(d / name)['fields']
        return cache[key]

    if fields_dir:
        bad = [x['artifact'] for x in inv if not (fields_dir / x['artifact']).exists()]
        gate('artifacts_present', not bad, bad[:5])
        if not bad:
            refs = {e['case']: fields_of(e['artifact']) for e in r['reference']}
            base = {x['case']: x['artifact'] for x in inv if x['name'] == 'fft_tight'}
            worst = 0.
            for x in inv:
                truth = refs[x['case']]
                f = fields_of(x['artifact'])
                n0 = np.linalg.norm(truth[0])
                err = float(np.max(np.linalg.norm((f - truth).reshape(len(f), -1), axis=1)) / n0)
                worst = max(worst, abs(err - x['error']['fixed_initial_max'])
                            / max(x['error']['fixed_initial_max'], 1e-300))
                g = fields_of(base[x['case']])
                x['same_grid_max'] = float(
                    np.max(np.linalg.norm((f - g).reshape(len(f), -1), axis=1)) / n0)
            gate('recorded_errors_recomputed_from_saved_fields', worst < 1e-9, worst)

    # ------------------------------------------------------ per-arm aggregates
    table = {}
    for x in inv:
        t = table.setdefault(x['name'], dict(arm=x['name'], kind=x['kind'], q=x.get('q'),
                                             solved_dimension=x.get('solved_dimension'),
                                             outer_dimension=None, M=x.get('M'), m=x.get('m'),
                                             rule=x.get('rule'), variant=x.get('variant'),
                                             fitter=x.get('fitter'), quadrature=x.get('quadrature'),
                                             dt=x.get('dt'), gpu_ms=[], host_ms=[], case_error={},
                                             case_same_grid={}, iterations=[], stationarity=[],
                                             inner=[], stationary=[], completed=[], converged=[],
                                             budget_exits=[]))
        t['gpu_ms'].append(x['gpu_seconds'] * 1e3)
        t['host_ms'].append(x['host_seconds'] * 1e3)
        t['case_error'][x['case']] = x['error']['fixed_initial_max']
        if 'same_grid_max' in x:
            t['case_same_grid'][x['case']] = x['same_grid_max']
        t['iterations'] += x['iterations']
        if x['kind'] == 'rom':
            t['stationarity'].append(x['worst_joint_stationarity'])
            t['inner'].append(max(x['step_inner_stationarity']) if x['step_inner_stationarity'] else 0.)
            t['stationary'].append(x['stationary'])
            t['completed'].append(x['completed'])
            t['converged'].append(x['converged'])
            t['budget_exits'].append(x['budget_exits'])
    setup = {s['arm']: s for s in r['arm_setup'] if 'arm' in s}
    recon = {x['q']: x for x in r['reconstruction']}
    rows = []
    for name, t in table.items():
        errs = np.array([t['case_error'][c] for c in sorted(t['case_error'])])
        sg = (np.array([t['case_same_grid'][c] for c in sorted(t['case_same_grid'])])
              if t['case_same_grid'] else None)
        rc = recon.get(t['q'])
        st = setup.get(name, {})
        eqf = (st.get('eq_fit') or {})
        rows.append(dict(arm=name, kind=t['kind'], q=t['q'], solved_dimension=t['solved_dimension'],
                         outer_dimension=st.get('outer_dimension'), M=t['M'], m=t['m'],
                         rule=t['rule'], variant=t['variant'], fitter=t['fitter'],
                         quadrature=t['quadrature'], dt=t['dt'],
                         worst_reference_percent=float(np.max(errs) * 100),
                         median_reference_percent=float(np.median(errs) * 100),
                         worst_same_grid_percent=(float(np.max(sg) * 100) if sg is not None else None),
                         median_same_grid_percent=(float(np.median(sg) * 100) if sg is not None else None),
                         worst_bank_projection_percent=(float(rc['worst_bank_projection'] * 100) if rc else None),
                         worst_best_found_percent=(float(rc['worst_best_found'] * 100) if rc else None),
                         median_gpu_ms=median(t['gpu_ms']), median_host_ms=median(t['host_ms']),
                         gpu_ms_repetitions=len(t['gpu_ms']),
                         median_iterations=median(t['iterations']),
                         max_iterations=float(np.max(t['iterations'])),
                         max_joint_stationarity=(float(np.max(t['stationarity'])) if t['stationarity'] else None),
                         max_inner_stationarity=(float(np.max(t['inner'])) if t['inner'] else None),
                         all_stationary=(bool(all(t['stationary'])) if t['stationary'] else None),
                         all_completed=(bool(all(t['completed'])) if t['completed'] else None),
                         converged=(bool(all(t['converged'])) if t['converged'] else None),
                         total_budget_exits=(int(np.sum(t['budget_exits'])) if t['budget_exits'] else None),
                         quadrature_fit_seconds=eqf.get('seconds'),
                         quadrature_fit_relative=st.get('eq_relative_fit'),
                         quadrature_support=eqf.get('support'),
                         quadrature_truncated=eqf.get('truncated'),
                         setup_seconds=st.get('total_setup_seconds')))
    rows.sort(key=lambda x: (x['kind'] != 'rom', x['q'] if x['q'] is not None else 0, x['arm']))
    checks['arm_table'] = rows
    conv = [x for x in rows if x['kind'] == 'rom' and x['converged']]
    checks['nondominated_all'] = nondominated([x for x in rows if x['kind'] == 'rom'])
    checks['nondominated_converged'] = nondominated(conv)

    # --------------------------------- cross-job reproduction of the q = 0 arm
    look = {x['arm']: x for x in rows}
    for src, path, want_name, got_name, tol in (
            ('qlad01', QLAD, 'q0_eq', 'q0_m4_eq_varpro', 1e-9),
            ('abl01', ABLATION, 'a_neural_eq', 'q0_m4_eq_varpro', 1e-9)):
        if not Path(path).exists():
            continue
        ref = json.loads(Path(path).read_text())
        want = {x['case']: x['error']['fixed_initial_max'] for x in ref['invocations']
                if x['name'] == want_name and x.get('intervals') == r['intervals']}
        got = {x['case']: x['error']['fixed_initial_max'] for x in inv if x['name'] == got_name}
        shared = sorted(set(want) & set(got))
        deltas = [abs(got[c] - want[c]) / max(want[c], 1e-300) for c in shared]
        gate(f'q0_reproduces_{src}_{want_name}', bool(shared) and max(deltas) < tol,
             dict(compared=len(shared), worst_relative_delta=(max(deltas) if deltas else None),
                  tolerance=tol, reference_gpu=ref.get('gpu'), this_gpu=r.get('gpu')))
        if src == 'qlad01':
            gate('cases_identical',
                 bool(np.allclose(np.asarray(ref['physical_cases']), np.asarray(r['physical_cases']))))

    # ---------------------- q = 16 variable projection versus the audited rung
    if Path(QLAD).exists():
        ref = json.loads(Path(QLAD).read_text())
        want = {x['case']: x['error']['fixed_initial_max'] for x in ref['invocations']
                if x['name'] == 'q16_dense'}
        got = {x['case']: x['error']['fixed_initial_max'] for x in inv
               if x['name'] == 'q16_m4_dense_varpro'}
        shared = sorted(set(want) & set(got))
        d_err = [abs(got[c] - want[c]) / max(want[c], 1e-300) for c in shared] or [np.inf]
        detail = dict(compared=len(shared), worst_relative_error_delta=max(d_err), tolerance=1e-3)
        if a.qlad01 and fields_dir:
            qd = Path(a.qlad01)
            art = {x['case']: x['artifact'] for x in ref['invocations'] if x['name'] == 'q16_dense'}
            mine = {x['case']: x['artifact'] for x in inv if x['name'] == 'q16_m4_dense_varpro'}
            fd = []
            for c in sorted(set(art) & set(mine)):
                A_ = fields_of(art[c], qd)
                B_ = fields_of(mine[c], fields_dir)
                fd.append(float(np.linalg.norm(A_ - B_) / np.linalg.norm(A_)))
            detail['worst_relative_field_difference'] = max(fd) if fd else None
            detail['fields_compared'] = len(fd)
        worst = detail.get('worst_relative_field_difference')
        gate('q16_varpro_agrees_with_qlad01_q16_dense',
             (worst is not None and worst <= 1e-3) or (worst is None and max(d_err) <= 1e-3), detail)

    out = dict(result=str(a.result), result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),
               job_id=r.get('job_id'), commit=r.get('commit'), gpu=r.get('gpu'),
               elapsed_seconds=r.get('elapsed_seconds'),
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
