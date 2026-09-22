"""Extend the Burgers training split past its pinned 128 cases, on the pinned generator.

`data.py` (vendored byte-identical at commit 5169c095, sha 8d491d5e...) caps `train` at the
128 cases its `protocol.json` declares and refuses any reference setting its calibration gate
did not pass -- at output 256 only the 4096-interval anchor passed, and that anchor costs
~114 s per case, i.e. ~146 GPU-hours for the 4608 cases our own model's head was trained on.

This module generates the same split further, with `data.py`'s own case seeding, its own
solver construction, its own convergence assertions and its own four-key model-facing schema,
at a reference setting declared on the command line and recorded in the index. Cases 0..127
are therefore the SAME physical draws as the pinned cache, which makes the protocol difference
directly measurable: `--pinned-index` compares every regenerated case against its pinned twin,
asserts the supplied input field and the generation descriptors are bitwise identical, and
records the fixed-initial difference of the evolved targets.

See DESIGN.md section 2.3. Nothing here is claimed to be the pinned protocol; the pinned
validation-32 and the pinned 8-case cohort are untouched and remain the grading data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

import data  # vendored, byte-identical to the generator that made the pinned cache

HERE = Path(__file__).resolve().parent
# The exact modules that produced the pinned cache (its index records these hashes).
PINNED_SOURCES = {
    'data.py': '8d491d5ea25b10f67183ddedf469c69ba5fee0ff90061901b664cc096bba5ca7',
    'protocol.json': '212bc8980229b93606449a1df814632da9aa1f403bbcc69e31ce995255239861',
    'engines.py': '820039e5840abaf92dc911f9de8c6a427f97cbd2617f1e7a5169e66c19fa62e4',
    'sep_common.py': 'a74f0279b7e5299c6e4fd7b445f78e29d9567b5614fec060935e73da4fcdf1d4',
}
TIMES = data.TIMES


def check_sources():
    actual = {name: data.sha(HERE / name) for name in PINNED_SOURCES}
    if actual != PINNED_SOURCES:
        raise RuntimeError(f'generator sources differ from the pinned cache: {actual}')
    return actual


def gpu_modules():
    """`data.gpu_modules` with the flat staged layout's import path and a hash pin.

    `data.gpu_modules` asserts `engines.__file__` equals `<root>/experiments/mr-burgers2d/
    engines.py`, which does not exist in the staged `code/` tree. The hash check above is a
    strictly stronger statement about the module that is actually imported.
    """
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
        raise RuntimeError('JAX_DEFAULT_MATMUL_PRECISION=highest is required')
    import jax
    jax.config.update('jax_enable_x64', True)
    backend = jax.default_backend()
    print(f'jax_backend={backend}', flush=True)
    if backend != 'gpu':
        raise SystemExit(42)
    if not jax.config.jax_enable_x64 or str(jax.config.jax_default_matmul_precision) != 'highest':
        raise RuntimeError('f64 and highest matmul precision are required')
    sys.path.insert(0, str(HERE))
    import engines
    if data.sha(Path(engines.__file__).resolve()) != PINNED_SOURCES['engines.py']:
        raise RuntimeError('imported engines.py is not the pinned FOM')
    return jax, engines


def solve(jax, engines, query, dense, output_intervals, physical):
    """`data.solve` with its two-second clock warm-up removed.

    The warm-up exists to make `data.py`'s recorded per-case wall time a meaningful resource
    profile. This lane makes no timing claim of any kind (DESIGN section 7) and 4608 cases
    would spend 2.6 GPU-hours in that loop, so it is dropped. Every line that touches the
    numbers is `data.solve`'s, unchanged, and `--self-check` asserts the two agree bitwise.
    """
    import jax.numpy as jnp
    started = time.perf_counter()
    fields, iterations, residuals = jax.tree_util.tree_map(np.asarray, query(
        jnp.asarray(engines.initial(dense, physical), dtype=jnp.float64),
        float(physical[4]), data.PROTOCOL['newton_tolerance'], data.PROTOCOL['linear_tolerance'],
    ))
    elapsed = time.perf_counter() - started
    if fields.dtype != np.float64 or fields.shape != (len(TIMES), dense + 1, dense + 1):
        raise RuntimeError('Native FOM returned wrong dtype or shape')
    if not np.isfinite(fields).all() or not np.isfinite(residuals).all():
        raise RuntimeError('Native FOM returned nonfinite fields/residuals')
    if float(np.max(residuals)) > data.PROTOCOL['maximum_relative_residual']:
        raise RuntimeError(f'FOM convergence failed: maximum residual {np.max(residuals)}')
    if np.any(iterations > data.PROTOCOL['max_newton']):
        raise RuntimeError('Native FOM exceeded Newton iteration budget')
    if np.any(fields[:, [0, -1], :]) or np.any(fields[:, :, [0, -1]]):
        raise RuntimeError('Dirichlet boundary check failed')
    fields = fields.copy()
    fields[0] = engines.initial(dense, physical)
    return data.restrict(fields, output_intervals), iterations, residuals, elapsed


def write_manifest(out, records):
    lines = []
    for record in records:
        lines.append(f"{record['sha256']}  train/{record['path']}")
        lines.append(f"{record['solver_audit']['sha256']}  train/{record['solver_audit']['path']}")
    index = out / 'train/index.json'
    lines.append(f'{data.sha(index)}  train/index.json')
    (out / 'DATA.sha256').write_text('\n'.join(lines) + '\n')


def publish(out, report, records):
    """Write a self-consistent, `complete` index plus its manifest for the current prefix."""
    report = dict(report, count=len(records), complete=True, records=records)
    data.write_json(out / 'train/index.json', report)
    write_manifest(out, records)
    return report


def load_pinned(pinned_index):
    if not pinned_index:
        return {}
    index = json.loads(Path(pinned_index).read_text())
    return {row['case_index']: dict(row, absolute=Path(pinned_index).parent / row['path'])
            for row in index['records']}


def compare_with_pinned(pinned, index, params, fields):
    """Bitwise on the supplied input and the descriptors; measured on the evolved targets."""
    row = pinned.get(index)
    if row is None:
        return None
    if row['seed'] != data.case_seed('train', index):
        raise RuntimeError(f'case {index}: pinned seed differs from the protocol seed')
    keys = ('cx', 'cy', 'width', 'amplitude', 'nu')
    if [row['generation_descriptors'][k] for k in keys] != list(params):
        raise RuntimeError(f'case {index}: generation descriptors differ from the pinned case')
    with np.load(row['absolute']) as saved:
        target, supplied = np.array(saved['target']), np.array(saved['input'])
    if not np.array_equal(supplied[0], fields[0]):
        raise RuntimeError(f'case {index}: supplied input field differs from the pinned case')
    difference = data.fixed_initial_errors(fields, target[:, 0])
    return dict(case_index=index, per_time=difference['per_time'], maximum=difference['maximum'])


def generate(args):
    sources = check_sources()
    jax, engines = gpu_modules()
    out = Path(args.out).resolve()
    (out / 'train').mkdir(parents=True, exist_ok=False)
    pinned = load_pinned(args.pinned_index)
    report = dict(schema_version=1, pde='burgers', kind='matched-neural-operator-dataset',
                  split='train', count=0, mesh=args.intervals, complete=False,
                  protocol_sha256=data.sha(HERE / 'protocol.json'),
                  provenance=dict(data.provenance(jax), source_sha256=sources,
                                  generator='gen_more.py', lane='experiments/ops-tune-deeponet'),
                  records=[],
                  reference_setting=dict(intervals=None, dt=None,
                                         selected_by='gen_more profile; DESIGN section 2.3'),
                  reference_deviation='the pinned 4096-interval anchor is NOT used; see '
                                      'pinned_comparison and DESIGN section 2.3',
                  calibration=dict(path=args.calibration_note,
                                   interpretation=data.PROTOCOL['reference_interpretation']),
                  descriptors_are_model_inputs=False, final_cohort='sealed')
    started = time.monotonic()
    physical = [engines.params_draw(data.case_seed('train', i), 1)[0] for i in range(args.count)]

    # -- profile ------------------------------------------------------------------
    profile = []
    for intervals, dt in [tuple(map(float, c.split(':'))) for c in args.candidates]:
        intervals = int(intervals)
        query, dense = data.make_solver(engines, intervals, dt, args.intervals)
        seconds = []
        for index in range(2):
            fields, _, residuals, elapsed = solve(jax, engines, query, dense, args.intervals, physical[index])
            seconds.append(elapsed)
            if not np.array_equal(fields[0], engines.initial(args.intervals, physical[index])):
                raise RuntimeError('candidate changed the requested initial field')
        profile.append(dict(intervals=intervals, dt=dt, first_seconds=seconds[0],
                            steady_seconds=seconds[1], max_relative_residual=float(residuals.max()),
                            projected_seconds_for_count=seconds[1] * args.count))
        print('PROFILE ' + json.dumps(profile[-1]), flush=True)
        del query
        jax.clear_caches()
    report['profile'] = profile
    budget = args.wall_budget_seconds - (time.monotonic() - started) - args.reserve_seconds
    affordable = [p for p in profile if p['projected_seconds_for_count'] <= budget]
    chosen = max(affordable, key=lambda p: p['intervals']) if affordable else \
        min(profile, key=lambda p: p['projected_seconds_for_count'])
    report['reference_setting'] = dict(intervals=chosen['intervals'], dt=chosen['dt'],
                                       selected_by='finest profiled setting projected to fit the '
                                                   'generation budget; DESIGN section 2.3',
                                       remaining_budget_seconds=budget,
                                       projected_seconds_for_count=chosen['projected_seconds_for_count'])
    print('CHOSEN ' + json.dumps(report['reference_setting']), flush=True)
    data.write_json(out / 'train/index.json', report)

    # -- bulk ---------------------------------------------------------------------
    query, dense = data.make_solver(engines, chosen['intervals'], chosen['dt'], args.intervals)
    records, comparisons, milestones = [], [], sorted(args.milestones)
    try:
        for index in range(args.count):
            remaining = args.wall_budget_seconds - (time.monotonic() - started) - args.reserve_seconds
            if remaining < 2 * chosen['steady_seconds'] + 30:
                report['stop_reason'] = f'generation budget reached at {len(records)} cases'
                print('BUDGET STOP ' + report['stop_reason'], flush=True)
                break
            record = data.case_record('train', index)
            fields, iterations, residuals, elapsed = solve(jax, engines, query, dense,
                                                           args.intervals, physical[index])
            if not np.array_equal(fields[0], engines.initial(args.intervals, physical[index])):
                raise RuntimeError('Candidate changed the requested initial field')
            comparison = compare_with_pinned(pinned, index, physical[index].tolist(), fields)
            if comparison is not None:
                comparisons.append(comparison)
            generated = data.save_case(out / 'train', record, fields, physical[index], args.intervals,
                                       dict(intervals=chosen['intervals'], dt=chosen['dt'],
                                            max_relative_residual=float(residuals.max()),
                                            wall_seconds=elapsed,
                                            total_newton_iterations=int(iterations.sum()),
                                            role='declared cheaper reference setting; DESIGN section 2.3'))
            audit_name = record['case_id'] + '.solver.npz'
            np.savez(out / 'train' / audit_name, iterations=iterations, residuals=residuals)
            generated['solver_audit'] = dict(path=audit_name, sha256=data.sha(out / 'train' / audit_name))
            records.append(generated)
            report['pinned_comparison'] = summarize(comparisons)
            if len(records) in milestones or index == args.count - 1:
                publish(out, report, records)
                print(f'MILESTONE {len(records)} elapsed={time.monotonic() - started:.0f}', flush=True)
            if index % 32 == 0:
                print(f'GENERATED {record["case_id"]} seconds={elapsed:.2f} '
                      f'elapsed={time.monotonic() - started:.0f}', flush=True)
    finally:
        report['pinned_comparison'] = summarize(comparisons)
        report['generation_seconds'] = time.monotonic() - started
        final = publish(out, report, records)
        data.write_json(out / 'generation-report.json',
                        dict({k: v for k, v in final.items() if k != 'records'},
                             case_count=len(records), milestones=milestones,
                             pinned_comparison_cases=len(comparisons)))
    print('GEN-DONE ' + json.dumps(dict(cases=len(records), setting=report['reference_setting'],
                                        pinned_comparison=report['pinned_comparison'])), flush=True)


def summarize(comparisons):
    if not comparisons:
        return dict(cases=0, note='no pinned twin was compared')
    values = np.array([c['maximum'] for c in comparisons])
    per_time = np.array([c['per_time'] for c in comparisons])
    return dict(cases=len(comparisons), metric='fixed-initial relative difference between the '
                'newly generated target and the pinned 4096-anchor target of the same case',
                maximum=float(values.max()), median=float(np.median(values)), mean=float(values.mean()),
                per_time_mean=per_time.mean(axis=0).tolist(), cases_detail=comparisons)


def self_check(args):
    """Tiny GPU check: `solve` here and `data.solve` agree bitwise on a 16-interval case."""
    check_sources()
    jax, engines = gpu_modules()
    params = engines.params_draw(data.case_seed('train', 0), 1)[0]
    query, dense = data.make_solver(engines, 16, 0.05, 16)
    mine, _, _, _ = solve(jax, engines, query, dense, 16, params)
    theirs, _, _, _ = data.solve(jax, engines, query, dense, 16, params)
    if not np.array_equal(mine, theirs):
        raise RuntimeError('the warm-up-free solve differs from data.solve')
    seeds = [data.case_seed('train', i) for i in range(200)]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError('case seeds collide')
    out = dict(passed=True, bitwise_equal_to_data_solve=True, distinct_seeds=len(set(seeds)),
               sources=check_sources())
    Path(args.output).write_text(json.dumps(out, indent=2) + '\n')
    print('SELF-CHECK PASS ' + json.dumps({k: v for k, v in out.items() if k != 'sources'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('self-check')
    command.add_argument('--output', required=True)
    command.set_defaults(func=self_check)
    command = commands.add_parser('generate')
    command.add_argument('--out', required=True)
    command.add_argument('--count', type=int, default=4608)
    command.add_argument('--intervals', type=int, default=256)
    command.add_argument('--candidates', nargs='+', default=['1024:0.0003125', '512:0.000625'],
                         help='intervals:dt profiled in order; the finest that fits is used')
    command.add_argument('--milestones', type=int, nargs='+', default=[128, 512, 1024, 2048, 3072, 4608])
    command.add_argument('--pinned-index', default=None)
    command.add_argument('--calibration-note', default='pilot-data01/refinement/index.json gate, '
                         'read for the candidate margins only; this data does not use the gated setting')
    command.add_argument('--wall-budget-seconds', type=float, default=21600.)
    command.add_argument('--reserve-seconds', type=float, default=600.)
    command.set_defaults(func=generate)
    args = parser.parse_args()
    args.func(args)
