"""Extend the Burgers operator training bank past the frozen 128-case protocol count.

Every numerical path is the frozen generator's: this module imports
`experiments/neural-operator-burgers/data.py` **byte-identical** (its SHA256 is asserted
against the hash the published dataset recorded) and calls its own `case_seed`,
`case_record`, `make_solver`, `solve` and `save_case`. Nothing about the draw, the solver,
the residual/Newton/Dirichlet assertions or the saved schema is reimplemented here.

Two things this script does that `data.generate` cannot:

* it generates case indices at or beyond `PROTOCOL["counts"]["train"]` (128), which
  `data.allowed_count` forbids and which is the whole point of the data-parity lane; and
* it generates at a **declared** solver setting rather than the one `data.read_calibration`
  selects, because the pinned 4096/1.5625e-4 reference costs 135 s per case and 4608 cases
  of it do not fit any budget this project has (DESIGN §3.2).

Neither bypass is taken on trust. `data.read_calibration`'s hash gate is replaced by a
strictly stronger one: REPRODUCTION, which regenerates a published case at the pinned
setting and asserts its **arrays are bitwise identical** to the published case's. File-byte
identity is recorded but NOT asserted: `np.savez` is deterministic, so the bytes normally do
match, but they are also a function of the numpy version that wrote them, and a numpy upgrade
must not void a numerical gate for a non-numerical reason. And the
cheap setting's cost is measured, not assumed: FIDELITY re-solves every published training
case at the cheap setting and records its fixed-initial error against that case's own pinned
target, so the label noise the lane trades for 36x more data is a measured per-case
distribution rather than an 8-case development estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
# In the worktree this file sits in the lane, two levels under the repository root; when
# staged it sits in `<job>/code/` with the generator's own repository paths beneath it
# (`cluster/stage.py: ROOTCODE`). Accept either layout, and nothing else.
GENERATOR = next(p for p in (HERE / 'experiments/neural-operator-burgers/data.py',
                             HERE.parents[1] / 'experiments/neural-operator-burgers/data.py')
                 if p.is_file())
# The hash the published training index recorded for the generator it ran.
GENERATOR_SHA256 = '8d491d5ea25b10f67183ddedf469c69ba5fee0ff90061901b664cc096bba5ca7'
PINNED = (4096, 0.00015625)     # the reference the published cases were generated at
CHEAP = (1024, 0.0003125)       # DESIGN 3.2; calibration-measured, 4.8 s/case


def load_generator():
    actual = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    if actual != GENERATOR_SHA256:
        raise RuntimeError(f'frozen generator changed: {actual}')
    spec = importlib.util.spec_from_file_location('frozen_burgers_data', GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules['frozen_burgers_data'] = module
    spec.loader.exec_module(module)
    return module


def summarise(rows):
    values = [row['maximum'] for row in rows]
    return dict(count=len(values),
                worst=max(values) if values else None,
                median=float(np.median(values)) if values else None,
                mean=float(np.mean(values)) if values else None)


def squeeze_target(path):
    """The stored model-facing target is (time, 1, n, n); the metric wants (time, n, n)."""
    with np.load(path, allow_pickle=False) as archive:
        return np.array(archive['target'][:, 0], copy=True)


def write_index(out, records, name, setting, extra):
    index = dict(schema_version=1, pde='burgers', kind='ops-tune-grid-extended-train-bank',
                 split='train', count=len(records), mesh=256, complete=True,
                 solver_setting=dict(intervals=setting[0], dt=setting[1]),
                 physical_accuracy_status='training targets only; evaluation uses the pinned '
                 '4096/1.5625e-4 reference unchanged. See DESIGN 3.2 and the measured '
                 'fidelity table in this index.',
                 descriptors_are_model_inputs=False, final_cohort='sealed',
                 records=records, **extra)
    path = out / name
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(index, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--pinned-cache', required=True, type=Path,
                        help='the published pilot-data01/train directory, read only')
    parser.add_argument('--count', type=int, default=4608)
    parser.add_argument('--prefixes', type=int, nargs='+', default=[128, 512, 2048, 4608])
    parser.add_argument('--wall-budget-seconds', type=float, default=23000.)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('generation must run in a cluster allocation')

    data = load_generator()
    jax, engines = data.gpu_modules()
    out = data.make_output(args.out)
    started = time.monotonic()
    report = dict(schema_version=1, kind='ops-tune-grid-generation-report', complete=False,
                  generator_sha256=GENERATOR_SHA256, source_sha256=data.source_hashes(),
                  provenance=data.provenance(jax), pinned_setting=dict(intervals=PINNED[0], dt=PINNED[1]),
                  cheap_setting=dict(intervals=CHEAP[0], dt=CHEAP[1]),
                  requested_count=args.count, reproduction=None, fidelity=[], timings=[],
                  protocol_note='This run loads the frozen data.py under protocol.json, whose '
                  'contents are the REFINED protocol (sha dec2ba41...), which is the '
                  'protocol_sha256 the published training index records. The published bank '
                  'reached it through refine.py, which monkey-patches data.PROTOCOL_PATH to a '
                  'file named protocol-refined.json, so the published source_sha256 keys that '
                  'file under a different NAME with the same CONTENT. The two protocol '
                  'variants differ only in anchor.dt, which this script does not read: it '
                  'declares PINNED and CHEAP explicitly.')
    path = out / 'generation-report.json'

    def save_report():
        temp = path.with_suffix('.partial')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(path)

    save_report()

    # ---------------------------------------------------------------- REPRODUCTION gate
    # Regenerate a published case at the pinned setting and require its ARRAYS to equal the
    # published case's bitwise. This is what licenses every bypass above. The file-byte
    # comparison beside it is recorded, not enforced (see the module docstring).
    record = data.case_record('train', 0)
    physical = engines.params_draw(record['seed'], 1)[0]
    query, dense = data.make_solver(engines, *PINNED, 256)
    fields, _, _, seconds = data.solve(jax, engines, query, dense, 256, physical)
    scratch = out / '_reproduction'
    scratch.mkdir()
    generated = data.save_case(scratch, record, fields, physical, 256,
                               dict(intervals=PINNED[0], dt=PINNED[1], role='reproduction gate'))
    published = args.pinned_cache / f"{record['case_id']}.npz"
    published_sha = data.sha(published)
    identical = generated['sha256'] == published_sha
    arrays_equal = all(np.array_equal(a, b) for a, b in zip(
        (lambda f: [np.array(f[k], copy=True) for k in ('input', 'target', 'parameters', 'times')])(
            np.load(scratch / generated['path'], allow_pickle=False)),
        (lambda f: [np.array(f[k], copy=True) for k in ('input', 'target', 'parameters', 'times')])(
            np.load(published, allow_pickle=False))))
    report['reproduction'] = dict(case_id=record['case_id'], generated_sha256=generated['sha256'],
                                  published_sha256=published_sha, bytes_identical=identical,
                                  arrays_identical=bool(arrays_equal), seconds=seconds)
    save_report()
    print(f"REPRODUCTION bytes_identical={identical} arrays_identical={arrays_equal}", flush=True)
    if not arrays_equal:
        raise RuntimeError('reproduction gate failed: this generator does not reproduce the published case')
    del query
    jax.clear_caches()

    # ------------------------------------------------------------------- bulk generation
    query, dense = data.make_solver(engines, *CHEAP, 256)
    records, seen = [], {}
    prefixes = set(args.prefixes)
    written = {}

    def write_prefix(count):
        name = f'index-{count:05d}.json'
        written[name] = str(write_index(out, records[:count], name, CHEAP, dict(
            fidelity_vs_pinned_reference=summarise(report['fidelity']),
            generation_report='generation-report.json')).name)
        report['indices'] = sorted(written)
        save_report()
        print(f'INDEX {name} over {count} cases', flush=True)

    stop_reason = 'complete'
    for index in range(args.count):
        if time.monotonic() - started + 60. > args.wall_budget_seconds:
            stop_reason = 'wall budget'
            break
        record = data.case_record('train', index)
        physical = engines.params_draw(record['seed'], 1)[0]
        fields, iterations, residuals, elapsed = data.solve(jax, engines, query, dense, 256, physical)
        if not np.array_equal(fields[0], engines.initial(256, physical)):
            raise RuntimeError('solver changed the requested initial field')
        generated = data.save_case(out, record, fields, physical, 256,
                                   dict(intervals=CHEAP[0], dt=CHEAP[1],
                                        max_relative_residual=float(residuals.max()),
                                        total_newton_iterations=int(iterations.sum()),
                                        wall_seconds_including_first_compile=elapsed,
                                        role='training target only; evaluation reference is the '
                                             'pinned 4096/1.5625e-4 cache, unchanged'))
        records.append(generated)
        # FIDELITY: for a case the published bank also holds, measure this setting's
        # fixed-initial error against that case's own pinned target.
        published = args.pinned_cache / f"{record['case_id']}.npz"
        if published.is_file():
            errors = data.fixed_initial_errors(fields, squeeze_target(published))
            report['fidelity'].append(dict(case_id=record['case_id'], maximum=errors['maximum'],
                                           per_time=errors['per_time']))
        report['timings'].append(elapsed)
        # Disjointness inside the bank, checked HERE rather than at training time:
        # dataset.load_index would catch a repeated physical input, but only after the whole
        # bank had been generated, copied and staged.
        key = hashlib.sha256(fields[0].tobytes() + physical[4].tobytes()).hexdigest()
        if key in seen:
            raise RuntimeError(f"duplicate physical input: {seen[key]} / {record['case_id']}")
        seen[key] = record['case_id']
        # Write every requested prefix index the moment it is reachable, so a run that stops
        # early still leaves usable indices instead of none (DESIGN 5.1 item 4).
        if len(records) in prefixes:
            write_prefix(len(records))
        if index % 64 == 0 or index == args.count - 1:
            save_report()
            print(f"GENERATED {record['case_id']} n={len(records)} "
                  f"elapsed={time.monotonic()-started:.0f}s", flush=True)

    # ------------------------------------------------------------------- DISCRETISATION bar
    # The paper's key qualification is that every operator error exceeds the 256-grid's own
    # discretisation error, quoted as 4.03 %. That figure was measured on the timing panel's
    # six development cases. The comparison this lane makes is on validation-32, so the bar
    # is measured HERE, on those same 32 cases, against the same pinned reference the
    # operators are scored against: solve each validation case ON the 256 grid and score it
    # by the identical fixed-initial metric. `dt_converged` isolates the spatial error (the
    # anchor's own time step on the coarse grid); `dt_panel` is the panel's own coarse-grid
    # setting, for continuity with the published figure.
    del query
    jax.clear_caches()
    validation = args.pinned_cache.parent / 'validation'
    report['discretisation'] = {}
    for label, dt in (('dt_converged', PINNED[1]), ('dt_panel', 0.00125), ('dt_nmrom_bank', 0.005)):
        rows = json.loads((validation / 'index.json').read_text())['records']
        query, dense = data.make_solver(engines, 256, dt, 256)
        errors = []
        for row in rows:
            physical = engines.params_draw(row['seed'], 1)[0]
            fields, _, residuals, _ = data.solve(jax, engines, query, dense, 256, physical)
            errors.append(dict(case_id=row['case_id'],
                               **data.fixed_initial_errors(fields, squeeze_target(validation / row['path']))))
        worst = [e['maximum'] for e in errors]
        report['discretisation'][label] = dict(
            intervals=256, dt=dt, cases=len(errors),
            worst=max(worst), median=float(np.median(worst)), mean=float(np.mean(worst)),
            per_case=errors,
            definition='the 256-interval grid solved on its own mesh at this time step, scored '
                       'by the same fixed-initial metric against the same pinned 4096 / '
                       '1.5625e-4 reference the operators are scored against; a perfect '
                       'operator on this mesh could not do better')
        print(f"DISCRETISATION {label} worst={max(worst):.6g} median={np.median(worst):.6g}", flush=True)
        save_report()
        del query
        jax.clear_caches()

    # --------------------------------------------------------------------- prefix indices
    report['fidelity_summary'] = summarise(report['fidelity'])
    report['generated_count'] = len(records)
    report['stop_reason'] = stop_reason
    report['seconds_per_case_median'] = float(np.median(report['timings'])) if report['timings'] else None
    # Re-emit every reachable prefix now that the fidelity summary is complete, and always
    # emit one at the count actually reached so a short run still has a top rung.
    for prefix in sorted(prefixes | {len(records)}):
        if 1 <= prefix <= len(records):
            write_prefix(prefix)
    summary = report['fidelity_summary']
    report['complete'] = True
    save_report()
    print(json.dumps(dict(generated=len(records), stop_reason=stop_reason,
                          fidelity=summary, indices=written)), flush=True)
    print('GENERATION FINISHED', flush=True)


if __name__ == '__main__':
    main()
