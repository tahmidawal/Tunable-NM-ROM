"""Score trained checkpoints on an extra held-out cohort with the lane metric.

Used for the matched ROM/FOM diagnosis cohort. Predictions are saved in full so
the collection audit can recompute every number independently. Normalisation
comes from the training checkpoint and is never refitted on this cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dataset
import model as adapter


def main(args):
    environment = adapter.configure()
    index, splits = dataset.load_index(args.index, (args.split,))
    records = splits[args.split]
    results = {}
    for folder in sorted(args.runs.glob('fno-*')):
        if not ((folder / 'result.json').exists() and (folder / 'best.pt').exists()):
            continue
        checkpoint = torch.load(folder / 'best.pt', map_location='cuda', weights_only=False)
        network = adapter.make_model(checkpoint['pde'], checkpoint['config'])
        network.load_state_dict(checkpoint['model'])
        adapter.check_dtypes(network)
        network.eval()
        norm = tuple(value.cuda() for value in checkpoint['normalization'])
        destination = args.out / folder.name
        destination.mkdir(parents=True, exist_ok=True)
        errors = []
        with torch.no_grad():
            for row in records:
                case = dataset.read_case(row)
                field = torch.from_numpy(case['input'])[None].cuda()
                parameters = torch.from_numpy(case['parameters'])[None].cuda()
                target = torch.from_numpy(case['target'])[None].cuda()
                prediction = adapter.predict(network, field, parameters, *norm, checkpoint['pde'])
                per_time = adapter.relative_errors(prediction, target, field, checkpoint['pde'])[0]
                errors.append(per_time.cpu().tolist())
                np.savez(destination / f"{row['case_id']}.prediction.npz",
                         prediction=prediction.cpu().numpy()[0])
        per_case = np.asarray(errors, dtype=np.float64).max(axis=1)
        results[folder.name] = dict(
            checkpoint_sha256=dataset.sha256(folder / 'best.pt'), best_epoch=checkpoint['epoch'],
            errors=errors, case_ids=[r['case_id'] for r in records],
            mean_case_max=float(per_case.mean()), median_case_max=float(np.median(per_case)),
            worst_case_max=float(per_case.max()),
            above_threshold_counts={str(t): int((per_case > t).sum()) for t in (.01, .02, .05)})
        del network, checkpoint
        torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'cohort-result.json').write_text(json.dumps(dict(
        environment=environment, split=args.split, cohort_index_sha256=dataset.sha256(args.index),
        cohort_kind=index.get('kind'), cases=len(records), models=results,
        metric='maximum over requested output times of the field discrepancy divided by the norm of '
               'the supplied initial field, identical to the Burgers lane metric',
        comparability='accuracy on these cases is comparable with the Burgers lane diagnosis; '
                      'timing measured in another job is not'), indent=2) + '\n')
    print(json.dumps({k: v['worst_case_max'] for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('index', 'runs', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--split', default='calibration')
    main(parser.parse_args())
