"""Complete-query timing for trained FNO checkpoints, inside one allocation.

Protocol, so that a later interleaved ROM/FNO/FOM confirmation job can reproduce
it exactly:

* The timed region begins with the supplied initial field and the viscosity
  already resident on the GPU and ends when the complete requested trajectory
  (all six output times, including the exactly returned supplied state) is
  resident on the GPU. Coordinate-channel construction, input normalisation,
  the network forward pass, boundary masking and trajectory assembly are all
  inside the timed region; nothing is precomputed outside it.
* ``torch.cuda.synchronize()`` brackets every repetition.
* Every timed block is preceded by its own GPU burn-in of ``--burn-in``
  untimed identical queries.
* Host transfer of the finished trajectory is timed as a separate block with
  its own burn-in and is never folded into the device query time.
* Every individual repetition is retained in ``timing.npz``; no repetition is
  discarded and no minimum is reported in place of a median.
* All models are timed inside this one process on this one GPU. Timings from
  different Slurm jobs must never be divided by one another.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import dataset
import model as adapter


def statistics(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    q1, q3 = np.quantile(x, [.25, .75])
    return dict(count=int(x.size), mean_ms=float(x.mean() * 1e3), median_ms=float(np.median(x) * 1e3),
                p05_ms=float(np.quantile(x, .05) * 1e3), p95_ms=float(np.quantile(x, .95) * 1e3),
                minimum_ms=float(x.min() * 1e3), maximum_ms=float(x.max() * 1e3),
                upper_tukey_outliers=int((x > q3 + 1.5 * (q3 - q1)).sum()))


def timed(function, repetitions, burn_in):
    for _ in range(burn_in):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return samples


def main(args):
    environment = adapter.configure()
    _, _, validation_records = dataset.load_pair(args.train_index, args.validation_index)
    models = {}
    for folder in sorted(args.runs.glob(args.pattern)):
        if (folder / 'result.json').exists() and (folder / 'best.pt').exists():
            models[folder.name] = folder
    if not models:
        raise RuntimeError('No completed model to time')
    device_samples, host_samples, summary = {}, {}, {}
    for name, folder in models.items():
        checkpoint = torch.load(folder / 'best.pt', map_location='cuda', weights_only=False)
        network = adapter.make_model(checkpoint['pde'], checkpoint['config'])
        network.load_state_dict(checkpoint['model'])
        adapter.check_dtypes(network)
        network.eval()
        norm = tuple(value.cuda() for value in checkpoint['normalization'])
        pde = checkpoint['pde']
        per_case_device, per_case_host = [], []
        with torch.no_grad():
            for row in validation_records:
                case = dataset.read_case(row)
                field = torch.from_numpy(case['input'])[None].cuda()
                parameters = torch.from_numpy(case['parameters'])[None].cuda()
                torch.cuda.synchronize()

                def query():
                    return adapter.predict(network, field, parameters, *norm, pde)

                per_case_device.append(timed(query, args.repetitions, args.burn_in))
                trajectory = query()
                torch.cuda.synchronize()
                per_case_host.append(timed(lambda: trajectory.cpu(), args.repetitions, args.burn_in))
        device = np.asarray(per_case_device, dtype=np.float64)
        host = np.asarray(per_case_host, dtype=np.float64)
        device_samples[name], host_samples[name] = device, host
        summary[name] = dict(
            checkpoint_sha256=dataset.sha256(folder / 'best.pt'),
            config=checkpoint['config'], best_epoch=checkpoint['epoch'],
            parameter_tensor_elements=sum(p.numel() for p in network.parameters()),
            real_parameter_count=sum(p.numel() * (2 if p.is_complex() else 1) for p in network.parameters()),
            device_query_pooled=statistics(device),
            device_query_median_of_case_medians_ms=float(np.median(np.median(device, axis=1)) * 1e3),
            host_transfer_pooled=statistics(host),
            host_transfer_median_of_case_medians_ms=float(np.median(np.median(host, axis=1)) * 1e3),
            device_plus_host_pooled_median_ms=float(np.median(device + host) * 1e3))
        del network, checkpoint
        torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'timing.npz',
             case_ids=np.array([r['case_id'] for r in validation_records]),
             **{f'device_{k}': v for k, v in device_samples.items()},
             **{f'host_{k}': v for k, v in host_samples.items()})
    (args.out / 'timing.json').write_text(json.dumps(dict(
        environment=environment, models=summary,
        repetitions_per_case=args.repetitions, burn_in_per_block=args.burn_in,
        cases=[r['case_id'] for r in validation_records],
        timed_region='supplied on-device initial field and viscosity to on-device complete trajectory',
        host_transfer='separately timed device-to-host copy of the finished trajectory',
        retained='every individual repetition is stored in timing.npz',
        scientific_status='same-job, same-GPU FNO timing only; no cross-job ratio and no ROM/FOM comparison',
    ), indent=2) + '\n')
    print(json.dumps({k: v['device_query_pooled']['median_ms'] for k, v in summary.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('train-index', 'validation-index', 'runs', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--pattern', default='fno-*')
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--burn-in', type=int, default=20)
    main(parser.parse_args())
