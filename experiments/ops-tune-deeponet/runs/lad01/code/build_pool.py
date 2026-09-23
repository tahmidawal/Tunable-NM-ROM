"""Pack a checksum-verified case index into three arrays, once per job.

At 4608 cases the training split is ~17 GB in 4608 files. `dataset.load_index` verifies every
one of them -- checksum, schema, dtype, boundary, output times, duplicate-input detection --
and that verification is the integrity gate this lane keeps. Doing it once per *arm* would
spend an hour of a job re-reading the same bytes, so it is done once per job here and every
arm reads the packed arrays instead.

The packed arrays carry the index's own sha256; `train.py --pool` refuses a pool whose
recorded index hash differs from the `--train-index` it was handed, so an arm cannot silently
train on a different split from the one its provenance records.

Case order is index order, which is `case_index` order, so `--pool-limit N` is the nested
prefix the ladder in DESIGN.md section 2.3 needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dataset


def main(args):
    index_path = Path(args.index).resolve()
    validation_path = Path(args.validation_index).resolve()
    # The full gate: per-case checksum, schema, dtype, boundary, output times, duplicate inputs
    # within each split, and train/validation disjointness by seed and by input content.
    pde, records, _ = dataset.load_pair(index_path, validation_path)
    if [r['case_index'] for r in records] != list(range(len(records))):
        raise RuntimeError('pool must be a contiguous prefix of the split starting at 0')
    if any(r['split'] != 'train' for r in records):
        raise RuntimeError('pool is built from the train split only')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    first = dataset.read_case(records[0])
    shapes = {name: (len(records),) + first[name].shape for name in ('input', 'target', 'parameters')}
    memmaps = {name: np.lib.format.open_memmap(out / f'{name}.npy', mode='w+',
                                               dtype=np.float64, shape=shapes[name])
               for name in shapes}
    for position, record in enumerate(records):
        case = dataset.read_case(record)
        for name in shapes:
            memmaps[name][position] = case[name]
        if position % 512 == 0:
            print(f'PACKED {position}/{len(records)}', flush=True)
    for handle in memmaps.values():
        handle.flush()
    del memmaps
    manifest = dict(schema_version=1, split='train', count=len(records),
                    mesh=records[0]['mesh'], pde=pde, index_path=str(index_path),
                    index_sha256=dataset.sha256(index_path),
                    validation_index_sha256=dataset.sha256(validation_path),
                    disjointness_verified=True,
                    shapes={name: list(shape) for name, shape in shapes.items()},
                    case_ids=[r['case_id'] for r in records],
                    case_sha256=[r['sha256'] for r in records],
                    array_sha256={name: dataset.sha256(out / f'{name}.npy') for name in shapes},
                    note='case order is case_index order; a prefix of this pool is a prefix of the split')
    (out / 'cases.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(count=len(records), out=str(out), index_sha256=manifest['index_sha256'],
                          shapes=manifest['shapes']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--validation-index', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
