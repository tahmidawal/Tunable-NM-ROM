"""Independent NumPy audit of one collected mesh-ladder attempt.

Nothing here imports the drivers or JAX. Errors are recomputed from the archived
observation-grid fields and the archived reference, timing identities are checked
against the raw recorded seconds, and the invocation grid is checked for
completeness so a partially written panel cannot pass unnoticed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CELL = Path(__file__).resolve().parent


def sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def restrict(field, intervals):
    source = field.shape[-1] - 1
    assert source % intervals == 0, (source, intervals)
    stride = source // intervals
    return np.ascontiguousarray(field[..., ::stride, ::stride], dtype=np.float64)


def burgers_errors(field, reference):
    """The driver's declared metric, recomputed from scratch."""
    difference = np.linalg.norm((field - reference).reshape(len(reference), -1), axis=1)
    return float(np.max(difference / np.linalg.norm(reference[0])))


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = CELL / 'runs' / args.attempt / 'out'
    result = json.loads((run / 'result.json').read_text())
    pde = result['pde']
    obs = result['restriction']['common_observation_intervals']
    findings = dict(attempt=args.attempt, pde=pde, checks={}, failures=[])

    def require(name, condition, detail=None):
        findings['checks'][name] = dict(passed=bool(condition), detail=detail)
        if not condition:
            findings['failures'].append(name)

    require('driver_reported_complete', result.get('complete') is True)
    require('jax_backend_gpu', result.get('jax_backend') == 'gpu', result.get('jax_backend'))
    require('float64', result.get('x64') is True)
    require('matmul_precision_highest', result.get('matmul_precision') == 'highest')
    require('checkpoint_unchanged',
            result.get('checkpoint_sha256') == result.get('checkpoint_sha256_after'))

    # ---- the invocation grid must be a complete Cartesian product ----------
    seen = defaultdict(set)
    for row in result['invocations']:
        seen[row['intervals']].add((row['name'], row['case'], row['rep']))
    meshes = sorted(seen)
    sizes = {m: len(seen[m]) for m in meshes}
    expected = {}
    for mesh in meshes:
        names = {n for n, _, _ in seen[mesh]}
        cases = {c for _, c, _ in seen[mesh]}
        reps = {r for _, _, r in seen[mesh]}
        expected[mesh] = len(names) * len(cases) * len(reps)
    require('complete_invocation_grid', sizes == expected, dict(observed=sizes, expected=expected))
    require('meshes_are_the_configured_ladder',
            meshes == sorted(result['config']['meshes']), meshes)

    # ---- timing identities and retained repetitions ------------------------
    bad_timing = []
    for row in result['invocations']:
        total = row['input_transfer_seconds'] + row['complete_device_seconds'] + row['output_transfer_seconds']
        if row['complete_device_seconds'] <= 0 or abs(row['host_to_host_seconds'] - total) > 1e-9 + 1e-6 * total:
            bad_timing.append((row['intervals'], row['name'], row['case'], row['rep']))
    require('timing_identity_and_positive_device_time', not bad_timing, bad_timing[:8])
    require('cached_reduced_solve_recorded',
            len(result['cached_invocations']) == sum(
                len({c for _, c, _ in seen[m]}) * len({r for _, _, r in seen[m]}) for m in meshes),
            len(result['cached_invocations']))
    require('cached_times_positive',
            all(row['cached_reduced_seconds'] > 0 for row in result['cached_invocations']))

    # ---- errors recomputed from archived fields ----------------------------
    checked, worst_difference, missing = 0, 0.0, 0
    if pde == 'burgers':
        references = {}
        for row in result['reference']:
            if row['role'] != 'primary':
                continue
            data = np.load(run / row['artifact'])['fields']
            references[row['case']] = restrict(data, obs)
            if sha(data) != row['field_sha256'] and sha(restrict(data, obs)) != row['observation_sha256']:
                findings['failures'].append(f'reference digest mismatch case {row["case"]}')
        for row in result['invocations']:
            artifact = run / row.get('observation_artifact', '')
            if not row.get('matches_first_field_sha256') or not artifact.is_file():
                missing += 1
                continue
            field = np.load(artifact)['observation_fields']
            recomputed = burgers_errors(field, references[row['case']])
            recorded = row['physical_error_common_grid']['fixed_initial_max']
            worst_difference = max(worst_difference, abs(recomputed - recorded))
            checked += 1
    else:
        references = {}
        for row in result['references']:
            data = np.load(run / 'references' / f'case{row["case"]}.npz')['fine']
            if sha(data) != row['fine_sha256']:
                findings['failures'].append(f'reference digest mismatch case {row["case"]}')
            references[row['case']] = restrict(data[None], obs)[0]
        for row in result['invocations']:
            artifact = run / 'fields' / f'{row["field_sha256"]}.npz'
            if not artifact.is_file():
                missing += 1
                continue
            field = np.load(artifact)['observation_field']
            recomputed = relative(field, references[row['case']])
            worst_difference = max(worst_difference, abs(recomputed - row['physical_error_common_grid']))
            checked += 1
    require('common_grid_errors_reproduced', checked > 0 and worst_difference <= 1e-12,
            dict(checked=checked, worst_absolute_difference=worst_difference, unarchived=missing))

    # ---- the staged solver must equal the retained one at every mesh -------
    parities = [row.get('staged_vs_retained_relative_difference',
                        row.get('staged_vs_native_relative_difference'))
                for row in (result.get('setup') or result.get('mesh_setup'))]
    require('staged_solver_matches_retained', all(p is not None and p <= 1e-12 for p in parities),
            parities)

    # ---- stopping status is recorded, not assumed --------------------------
    rom = [row for row in result['invocations'] if row['method'] == 'rom']
    graded = [row for row in rom if row.get('stationary') is not None]
    findings['rom_stationary'] = dict(
        stationary=sum(1 for row in graded if row['stationary']), graded=len(graded),
        total=len(rom),
        note=('Poisson grades stationarity in native_rows, outside every timer, because the timed '
              'complete query deliberately excludes the diagnostics' if not graded else
              'graded from the charged gradients returned by the timed invocation itself'))
    if pde == 'burgers':
        # The charged gradient comes back from the timed invocation; the audited one
        # is recomputed afterwards by a separately compiled function. Both divide by
        # a product of norms of order gtol, so an absolute bound on their difference
        # is the wrong test -- it only says how small that denominator is. What has
        # to hold is that the two agree to a small relative tolerance and that their
        # difference is far smaller than the margin between the worst gradient and
        # the stopping threshold, so no stationarity verdict can turn on it.
        gtol = result['config']['strict']['gtol']
        drift, values = [], []
        for row in rom:
            if 'audit_max_stationarity_difference' not in row:
                continue
            drift.append(row['audit_max_stationarity_difference'])
            values.append(max(float(np.max(row['step_normalized_stationarity'])),
                              float(row['ic_normalized_stationarity'])))
        worst_drift = max(drift) if drift else 0.0
        worst_value = max(values) if values else 0.0
        worst_relative = max((d / v if v else 0.0) for d, v in zip(drift, values)) if drift else 0.0
        margin = gtol - worst_value
        require('charged_and_audited_stationarity_agree',
                bool(drift) and worst_relative <= 1e-3 and worst_drift <= 0.5 * margin,
                dict(worst_absolute_difference=worst_drift, worst_relative_difference=worst_relative,
                     worst_gradient=worst_value, stopping_threshold=gtol,
                     margin_to_threshold=margin,
                     difference_as_fraction_of_margin=(worst_drift / margin if margin else None),
                     note=('the solver exits as soon as the gradient reaches the threshold, so every '
                           'recorded gradient sits just below it; the check is that no verdict can '
                           'turn on the difference between the two evaluations')))
    else:
        native = result['native_rows']
        findings['native_solver_valid'] = dict(
            valid=sum(1 for row in native if row['solver_valid']), total=len(native))
        require('native_rank_valid', all(row['projected_jacobian_rank_valid'] for row in native))

    findings['passed'] = not findings['failures']
    text = json.dumps(findings, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)
    if not findings['passed']:
        raise SystemExit('audit failed: ' + ', '.join(map(str, findings['failures'])))


if __name__ == '__main__':
    main()
