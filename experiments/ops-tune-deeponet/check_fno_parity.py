"""Reproduce an audited parent-lane number through THIS lane's driver files.

Loads the parent lane's archived `fno-large/best.pt` (job 3710846, reconstructed
from the Git-tracked archive parts and hash-verified) with this lane's `model.py`
and `train.py` — the files that will be staged for the U-Net and Transolver jobs —
and evaluates it on the archived 32 validation cases. The per-case, per-time
fixed-initial errors must agree with the independent NumPy audit recorded in the
parent's `field-audit.json` to 1e-9, which proves the family dispatch left the
FNO contract and the evaluation path untouched. The audited numbers are also
recomputed here from the archived prediction fields with NumPy only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import dataset
import model as adapter
import train

PARENT_AUDIT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-no-second/'
                    'experiments/neural-operator-audit/runs/fno_burgers02/field-audit.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_initial_errors(prediction, target):
    denominator = np.linalg.norm(target[0, 0, 1:-1, 1:-1])
    return np.linalg.norm((prediction - target)[:, 0, 1:-1, 1:-1].reshape(len(target), -1), axis=1) / denominator


def main(args):
    environment = adapter.configure()
    audit = json.loads(PARENT_AUDIT.read_text())
    model_audit = audit['models']['fno-large']
    root = args.archive / 'fno_burgers02'
    folder = root / 'out/fno-large'
    assert sha(folder / 'best.pt') == model_audit['best_checkpoint_sha256']
    _, rows = dataset.load_index(root / 'data/validation/index.json', ('validation',))
    rows = rows['validation']
    assert [r['case_id'] for r in rows] == model_audit['validation_case_ids']
    # 1. NumPy-only recomputation of the audited numbers from the archived fields.
    numpy_errors = []
    for row in rows:
        with np.load(row['absolute_path']) as case:
            target = case['target'].copy()
        with np.load(folder / f"{row['case_id']}.prediction.npz") as saved:
            prediction = saved['prediction'].copy()
        numpy_errors.append(fixed_initial_errors(prediction, target))
    numpy_errors = np.asarray(numpy_errors)
    audited = np.asarray(model_audit['per_time_errors'])
    numpy_gap = float(np.abs(numpy_errors - audited).max())
    # 2. This lane's driver files evaluating the archived checkpoint on the GPU.
    checkpoint = torch.load(folder / 'best.pt', map_location='cuda', weights_only=False)
    assert adapter.family_of(checkpoint['config']) == 'fno'
    network = adapter.make_model(checkpoint['pde'], checkpoint['config'])
    network.load_state_dict(checkpoint['model'])
    adapter.check_dtypes(network)
    norm = tuple(value.cuda() for value in checkpoint['normalization'])
    validation = train.arrays(rows)
    metrics = train.evaluate(network, validation, norm, checkpoint['pde'], checkpoint['config']['batch_size'])
    driver_errors = np.asarray(metrics['errors'])
    driver_gap = float(np.abs(driver_errors - audited).max())
    driver_gap_relative = float((np.abs(driver_errors - audited) / np.maximum(audited, 1e-300)).max())
    worst_gap = abs(metrics['worst_case_max'] - model_audit['fixed_initial']['maximum'])
    median_gap = abs(metrics['median_case_max'] - model_audit['fixed_initial']['median'])
    result = dict(environment=environment, checkpoint_sha256=model_audit['best_checkpoint_sha256'],
                  parent_job_id=audit['job_id'], parent_audit_sha256=sha(PARENT_AUDIT),
                  archive_sha256_expected=audit['archive_sha256'], cases=len(rows),
                  numpy_recomputation_max_abs_gap=numpy_gap,
                  driver_max_abs_gap=driver_gap, driver_max_relative_gap=driver_gap_relative,
                  worst_case_max=dict(this_lane=metrics['worst_case_max'], audited=model_audit['fixed_initial']['maximum'], gap=worst_gap),
                  median_case_max=dict(this_lane=metrics['median_case_max'], audited=model_audit['fixed_initial']['median'], gap=median_gap),
                  tolerance=1e-9,
                  passed=bool(numpy_gap <= 1e-15 and driver_gap <= 1e-9 and worst_gap <= 1e-9 and median_gap <= 1e-9),
                  sources={p.name: dataset.sha256(p) for p in Path(__file__).parent.glob('*.py')},
                  note='Different GPU and CUDA build from the parent job; agreement is limited by f64 FFT/matmul '
                       'rounding, which is far below the 1e-9 gate.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('numpy_recomputation_max_abs_gap', 'driver_max_abs_gap',
                                             'driver_max_relative_gap', 'worst_case_max', 'median_case_max', 'passed')}))
    assert result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args())
