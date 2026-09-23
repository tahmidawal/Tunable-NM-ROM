"""Time one trained operator on this job's six cases, in this job's allocation (burgers-compare-hires Phase O).

Copied from ops-timing-panel lib/fno_panel.py @ 31af60c8. Changes: the scored fields are ALSO written as
`full_<name>_case<c>.npy` (float64) beside every JAX arm's, so audit_cmp.py scores every family with one code
path; the GPU name is recorded for the same-GPU gate; repetitions/burn-in default to 5/20 (the protocol).

Original docstring:

The timing protocol is the `2026-09-14-no-audit` lane's `timing.py`, replicated:

* the timed region begins with the supplied initial field and the viscosity already
  resident on the GPU and ends when the complete requested trajectory (all six output
  times, including the exactly returned supplied state) is resident on the GPU;
  coordinate-channel construction, input normalisation, the forward pass, boundary
  masking and trajectory assembly are all inside it;
* `torch.cuda.synchronize()` brackets every repetition;
* every timed block is preceded by its own burn-in of untimed identical queries;
* host transfer is a separate timed block with its own burn-in;
* every individual repetition is retained; no minimum is reported in place of a median.

**Declared deviation from that protocol.** It says all models are timed inside one
process. This runs as a second process in the same Slurm allocation on the same GPU,
after the JAX phase has exited, because JAX preallocates most of the device and would
starve PyTorch. The property that matters — one allocation, one GPU, no ratio taken
across jobs — is preserved; the single-process wording is not. This is stated in the
report rather than glossed.

Predictions are written in the same `fields` layout the JAX phase writes, so the
independent audit scores the FNO with the same code, against the same same-job
converged full-order solve, on both the all-times and the evolved-times metric.
"""
from __future__ import annotations

import argparse
import hashlib
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--name', default='fno-large')
    p.add_argument('--repetitions', type=int, default=5)
    p.add_argument('--burn-in', type=int, default=20)
    p.add_argument('--fields', required=True)
    p.add_argument('--role', default='')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    environment = adapter.configure()
    _, by_split = dataset.load_index(Path(a.index), ('development',))
    rows = by_split['development']
    checkpoint = torch.load(a.checkpoint, map_location='cuda', weights_only=False)
    network = adapter.make_model(checkpoint['pde'], checkpoint['config'])
    network.load_state_dict(checkpoint['model'])
    adapter.check_dtypes(network)
    network.eval()
    norm = tuple(value.cuda() for value in checkpoint['normalization'])
    pde = checkpoint['pde']
    assert pde == 'burgers', pde

    device_samples, host_samples, cases = [], [], []
    with torch.no_grad():
        for row in rows:
            case = dataset.read_case(row)
            field = torch.from_numpy(case['input'])[None].cuda()
            parameters = torch.from_numpy(case['parameters'])[None].cuda()
            torch.cuda.synchronize()

            def query():
                return adapter.predict(network, field, parameters, *norm, pde)

            device_samples.append(timed(query, a.repetitions, a.burn_in))
            trajectory = query()
            torch.cuda.synchronize()
            host_samples.append(timed(lambda: trajectory.cpu(), a.repetitions, a.burn_in))
            fields = trajectory.cpu().numpy()[0, :, 0]
            assert fields.shape == case['target'].shape[:1] + case['target'].shape[2:], fields.shape
            supplied = case['input'][0]
            index = int(row['case_index'])
            name = f"full_{a.name}_case{index}.npy"
            np.save(Path(a.fields) / f'full_{a.name}_case{index}.npy', np.ascontiguousarray(fields, dtype=np.float64))
            cases.append(dict(case_id=row['case_id'], case_index=index,
                              artifact=name,
                              field_sha256=hashlib.sha256(
                                  np.ascontiguousarray(fields).tobytes()).hexdigest(),
                              t0_returned_exactly=bool(np.array_equal(fields[0], supplied)),
                              device_seconds=device_samples[-1],
                              host_seconds=host_samples[-1]))
            print('FNO', row['case_id'], 'device_ms',
                  round(float(np.median(device_samples[-1])) * 1e3, 3), flush=True)

    device = np.asarray(device_samples, dtype=np.float64)
    hostt = np.asarray(host_samples, dtype=np.float64)
    np.savez(out / f'{a.name}-timing.npz', device=device, host=hostt,
             case_ids=np.array([r['case_id'] for r in rows]))
    (out / f'{a.name}-timing.json').write_text(json.dumps(dict(
        model=a.name, role=a.role, gpu_name=torch.cuda.get_device_name(), environment=environment,
        checkpoint_sha256=dataset.sha256(a.checkpoint), config=checkpoint['config'],
        best_epoch=checkpoint['epoch'],
        parameter_tensor_elements=sum(p_.numel() for p_ in network.parameters()),
        real_parameter_count=sum(p_.numel() * (2 if p_.is_complex() else 1)
                                 for p_ in network.parameters()),
        repetitions_per_case=a.repetitions, burn_in_per_block=a.burn_in,
        device_query_pooled=statistics(device), host_transfer_pooled=statistics(hostt),
        device_query_median_of_case_medians_ms=float(np.median(np.median(device, axis=1)) * 1e3),
        cases=cases,
        timed_region='supplied on-device initial field and viscosity to on-device complete trajectory',
        host_transfer='separately timed device-to-host copy of the finished trajectory',
        retained='every individual repetition is stored in the npz beside this file',
        deviation=('second process in the same Slurm allocation on the same GPU, after the JAX '
                   'phase exited; the source protocol times all models in one process'),
        scientific_status=('same-allocation, same-GPU timing against the JAX subjects of this job; '
                           'no ratio is taken against any other job'),
    ), indent=2) + '\n')
    print('OPERATOR TIMING COMPLETE', a.name, flush=True)


if __name__ == '__main__':
    main()
