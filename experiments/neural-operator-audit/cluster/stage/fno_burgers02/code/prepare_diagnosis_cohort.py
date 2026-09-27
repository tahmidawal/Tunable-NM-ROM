"""Build the matched 8-case cohort the Burgers ROM/FOM diagnosis was graded on.

The ROM diagnosis (`diagnose.py --intervals 256`) graded every method against
the refined 4096-interval anchor restricted to the 256-interval grid, on the
eight calibration cases. Restricting exactly the same anchor files here, into
exactly the model-facing four-key schema, lets the FNO be scored on identical
inputs against identical references.

Accuracy may then be compared across jobs on these cases because accuracy does
not depend on the hardware. **Timing may not**: the ROM and FOM timings were
measured in a different allocation and must never be divided by timings measured
here.

The cohort is held out from FNO training by construction: calibration, train and
validation use different split codes and therefore different generation seeds.
This script re-checks that against the training index rather than assuming it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TIMES = np.array([0., .05, .1, .15, .2, .25], dtype=np.float64)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restrict(field, intervals):
    source = field.shape[-1] - 1
    if source % intervals:
        raise ValueError('Restriction must use nested grid nodes')
    stride = source // intervals
    return np.ascontiguousarray(field[:, ::stride, ::stride], dtype=np.float64)


def main(args):
    reference = json.loads((args.reference_index).read_text())
    assert reference['pde'] == 'burgers' and reference['complete'] is True
    training = json.loads(args.train_index.read_text())
    training_seeds = {row['seed'] for row in training['records']}
    training_inputs = set()
    for row in training['records']:
        with np.load(args.train_index.parent / row['path']) as case:
            training_inputs.add(hashlib.sha256(case['input'].tobytes()).hexdigest())
    args.out.mkdir(parents=True, exist_ok=False)
    records, report = [], []
    for record in reference['records']:
        case = record['case_id']
        anchors = [s for s in reference['solves']
                   if s['case_id'] == case and s['intervals'] == args.reference_intervals
                   and s['dt'] == args.reference_dt]
        if len(anchors) != 1:
            raise ValueError(f'Expected exactly one anchor solve for {case}: {len(anchors)}')
        anchor = anchors[0]
        path = args.reference_index.parent / anchor['path']
        assert sha(path) == anchor['sha256'], f'Anchor checksum mismatch: {case}'
        with np.load(path) as saved:
            fields = saved['fields'].copy()
        assert fields.shape[0] == TIMES.size
        target = restrict(fields, args.intervals)[:, None]
        supplied = target[0].copy()
        nu = float(record['generation_descriptors']['nu'])
        assert nu > 0
        assert np.isfinite(target).all()
        for edge in (target[..., 0, :], target[..., -1, :], target[..., :, 0], target[..., :, -1]):
            assert not np.any(edge), f'Restricted anchor must keep zero Dirichlet boundaries: {case}'
        assert record['seed'] not in training_seeds, f'Cohort case shares a training seed: {case}'
        identity = hashlib.sha256(supplied.tobytes()).hexdigest()
        assert identity not in training_inputs, f'Cohort case shares a training input field: {case}'
        out = args.out / f'{case}.npz'
        np.savez(out, input=supplied, target=target,
                 parameters=np.array([nu], dtype=np.float64), times=TIMES)
        records.append(dict(case_id=case, split='calibration', case_index=record['case_index'],
                            seed=record['seed'], path=out.name, sha256=sha(out), mesh=args.intervals,
                            anchor=dict(path=anchor['path'], sha256=anchor['sha256'],
                                        intervals=anchor['intervals'], dt=anchor['dt'],
                                        output_intervals=anchor.get('output_intervals'))))
        report.append(dict(case_id=case, seed=record['seed'], viscosity=nu,
                           anchor_sha256=anchor['sha256'], restricted_sha256=records[-1]['sha256'],
                           initial_norm=float(np.linalg.norm(supplied[0, 1:-1, 1:-1]))))
    index = dict(schema_version=1, pde='burgers', kind='matched ROM/FOM diagnosis cohort',
                 split='calibration', count=len(records), mesh=args.intervals, complete=True,
                 reference_index_sha256=sha(args.reference_index),
                 train_index_sha256=sha(args.train_index),
                 reference_setting=dict(intervals=args.reference_intervals, dt=args.reference_dt),
                 derivation='restriction of the Burgers lane refined anchor to the training grid; '
                            'identical references and inputs to the ROM/FOM diagnosis at 256 intervals',
                 descriptors_are_model_inputs=False,
                 limitation='Accuracy on these cases is comparable across jobs; timing is not.',
                 records=records)
    (args.out / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    (args.out / 'cohort-provenance.json').write_text(json.dumps(dict(
        cases=report, index_sha256=sha(args.out / 'index.json'),
        disjoint_from_training_by_seed=True, disjoint_from_training_by_input_field=True), indent=2) + '\n')
    print(json.dumps(dict(cases=len(records), index=str(args.out / 'index.json')), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-index', required=True, type=Path)
    parser.add_argument('--train-index', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--intervals', type=int, default=256)
    parser.add_argument('--reference-intervals', type=int, default=4096)
    parser.add_argument('--reference-dt', type=float, default=0.00015625)
    main(parser.parse_args())
