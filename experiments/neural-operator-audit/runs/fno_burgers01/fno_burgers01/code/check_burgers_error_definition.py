"""Prove that this lane's Burgers validation error is the Burgers lane's metric.

`model.relative_errors` divides the whole-field discrepancy by the norm of the
supplied initial field; the Burgers lane's `data.fixed_initial_errors` divides
the interior discrepancy by the interior norm of the reference initial field and
takes the maximum over requested times. On the dataset contract enforced by
`dataset.validate_case` (exact homogeneous Dirichlet boundaries and
``target[0] == input``) these coincide up to floating-point summation order.
This check imports the Burgers lane function by path, records its source hash,
and compares both definitions on contract-valid random fields, requiring
agreement to 1e-14 absolute (observed differences are a few units in the last
place of the reduction, not a definitional difference).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

BURGERS_DATA = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/'
                    '2026-09-14-no-burgers/experiments/neural-operator-burgers/data.py')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as adapter  # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location('burgers_lane_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zero_boundary(array):
    array[..., 0, :] = 0
    array[..., -1, :] = 0
    array[..., :, 0] = 0
    array[..., :, -1] = 0
    return array


def main(output):
    lane = load(BURGERS_DATA)
    times = np.asarray(lane.TIMES, dtype=np.float64)
    generator = np.random.default_rng(20260914)
    differences = []
    for trial in range(8):
        mesh = 32 + 8 * trial
        target = zero_boundary(generator.normal(size=(times.size, mesh + 1, mesh + 1)))
        prediction = target + 1e-2 * zero_boundary(generator.normal(size=target.shape))
        prediction[0] = target[0]
        lane_errors = lane.fixed_initial_errors(prediction, target)
        here = adapter.relative_errors(
            torch.from_numpy(prediction)[None, :, None], torch.from_numpy(target)[None, :, None],
            torch.from_numpy(target[0])[None, None], 'burgers').numpy()[0]
        differences.append(dict(
            trial=trial, mesh=mesh,
            maximum_absolute_per_time_difference=float(np.abs(here - np.asarray(lane_errors['per_time'])).max()),
            case_maximum_difference=float(abs(here.max() - lane_errors['maximum'])),
            case_maximum=float(here.max())))
    worst = max(row['maximum_absolute_per_time_difference'] for row in differences)
    worst_case_maximum = max(row['case_maximum_difference'] for row in differences)
    tolerance = 1e-14
    result = dict(
        passed=bool(worst <= tolerance and worst_case_maximum <= tolerance),
        agreement_tolerance=tolerance,
        burgers_lane_data_py=str(BURGERS_DATA),
        burgers_lane_data_py_sha256=hashlib.sha256(BURGERS_DATA.read_bytes()).hexdigest(),
        times=times.tolist(), trials=differences,
        worst_absolute_per_time_difference=worst,
        worst_case_maximum_difference=worst_case_maximum,
        definition='maximum over requested output times of the l2 field discrepancy divided by '
                   'the l2 norm of the supplied initial field (fixed-initial normalisation)',
        limitation='Exact agreement relies on the dataset contract: zero Dirichlet boundaries and '
                   'target[0] equal to the supplied input. It is not an accuracy result.')
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit('Error definitions disagree')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
