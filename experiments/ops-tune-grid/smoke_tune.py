"""Local constructibility and cost smoke for every config this lane will train.

Builds each config at the real 257x257 contract, runs one forward and one backward at a
small batch, and reports real parameter count, peak allocated bytes and seconds per step.
It catches the failure this lane is most exposed to -- an arm in the pre-registered grid
that does not construct, or does not fit -- on this machine in a minute instead of in a
16-hour cluster job. It trains nothing and reports no accuracy.

    jaxrun python smoke_tune.py --configs configs/fno configs/unet configs/transolver
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import model as adapter


def probe(path, batch, mesh, steps):
    config = json.loads(Path(path).read_text())
    torch.manual_seed(config['seed'])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = adapter.make_model('burgers', config)
    adapter.check_dtypes(net)
    n = mesh + 1
    field = torch.randn(batch, 1, n, n, dtype=torch.float64, device='cuda')
    field[..., 0, :] = field[..., -1, :] = field[..., :, 0] = field[..., :, -1] = 0
    parameters = torch.rand(batch, 1, dtype=torch.float64, device='cuda') * .1 + .01
    target = torch.randn(batch, 6, 1, n, n, dtype=torch.float64, device='cuda')
    norm = (torch.zeros(1, 2, 1, 1, dtype=torch.float64, device='cuda'),
            torch.ones(1, 2, 1, 1, dtype=torch.float64, device='cuda'),
            torch.ones((), dtype=torch.float64, device='cuda'))
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    for step in range(steps):
        if step == 1:
            torch.cuda.synchronize()
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = adapter.predict(net, field, parameters, *norm, 'burgers')
        assert prediction.shape == target.shape, (path, prediction.shape)
        assert torch.equal(prediction[:, 0], field), 'supplied state must be returned bitwise'
        for edge in (prediction[..., 0, :], prediction[..., -1, :],
                     prediction[..., :, 0], prediction[..., :, -1]):
            assert not torch.any(edge), 'boundary must be exactly zero'
        loss = adapter.relative_errors(prediction, target, field, 'burgers')[:, 1:].square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        adapter.check_dtypes(net, gradients=True)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / max(steps - 1, 1)
    peak = torch.cuda.max_memory_allocated()
    del net, optimizer, field, target, prediction, loss
    torch.cuda.empty_cache()
    return dict(config=str(path), family=adapter.family_of(config),
                real_parameters=sum(p.numel() * (2 if p.is_complex() else 1)
                                    for p in adapter.make_model('burgers', config).parameters()),
                batch=batch, peak_allocated_bytes=int(peak), seconds_per_step=seconds,
                projected_peak_bytes_at_batch_8=int(peak * 8 / batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--mesh', type=int, default=256)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    adapter.configure()
    paths = sorted(p for root in args.configs
                   for p in ([root] if root.is_file() else sorted(root.glob('*.json'))))
    rows = []
    for path in paths:
        row = probe(path, args.batch, args.mesh, args.steps)
        rows.append(row)
        print(json.dumps(row), flush=True)
    text = json.dumps(dict(passed=True, mesh=args.mesh, rows=rows), indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(f'SMOKE PASS {len(rows)} configs')


if __name__ == '__main__':
    main()
