"""Bounded CUDA contract checks for the U-Net and Transolver families.

For each family and a tiny config: forward, backward and one optimiser step
succeed; the output is float64 with exact zero Dirichlet boundaries; the supplied
Burgers initial state is returned bitwise; every parameter carries the declared
dtype; a saved checkpoint reloads to bitwise-identical predictions; and, for the
float64 declaration, every floating intermediate is float64 (the parent lane's
dispatch audit). Implementation evidence only, not accuracy evidence.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import model as adapter

TINY = {'unet': dict(family='unet', base=8, groups=8),
        'transolver': dict(family='transolver', dim=32, layers=2, heads=4, slices=8, mlp_ratio=2, patch=4, ref=8),
        'deeponet': dict(family='deeponet', width=8, rank=16, trunk_width=16, levels=2, pool_bins=2)}


class DtypeAudit(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for x in tree_leaves(out):
            if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex()):
                if x.dtype not in (torch.float64, torch.complex128):
                    raise RuntimeError(f'Intermediate precision failure: {func} {x.dtype}')
        return out


def main(out):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu'
    print('jax_backend=gpu', flush=True)
    env = adapter.configure()
    results = []
    for family in ('unet', 'transolver', 'deeponet'):
        for dtype in ('float32', 'float64'):
            for pde in ('burgers', 'poisson'):
                torch.manual_seed(20260914)
                config = dict(TINY[family], dtype=dtype)
                model = adapter.make_model(pde, config)
                adapter.check_dtypes(model)
                assert all(p.dtype == getattr(torch, dtype) for p in model.parameters())
                x = torch.rand((2, 1, 65, 65), device='cuda', dtype=torch.float64)
                x[..., (0, -1), :] = 0
                x[..., :, (0, -1)] = 0
                params = torch.full((2, 0 if pde == 'poisson' else 1), .05, device='cuda', dtype=torch.float64)
                mean = torch.zeros((1, 1 + params.shape[1], 1, 1), device='cuda', dtype=torch.float64)
                std = torch.ones_like(mean)
                scale = torch.tensor(1., device='cuda', dtype=torch.float64)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                context = DtypeAudit() if dtype == 'float64' else _Null()
                with context:
                    y = adapter.predict(model, x, params, mean, std, scale, pde)
                    loss = y[:, -1].square().mean()
                    loss.backward()
                    adapter.check_dtypes(model, gradients=True)
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    optimizer.step()
                with torch.no_grad():
                    expected = adapter.predict(model, x, params, mean, std, scale, pde)
                    assert expected.dtype == torch.float64
                    assert expected.shape == ((2, 1, 1, 65, 65) if pde == 'poisson' else (2, 6, 1, 65, 65))
                    assert (expected[..., 0, :] == 0).all() and (expected[..., -1, :] == 0).all()
                    assert (expected[..., :, 0] == 0).all() and (expected[..., :, -1] == 0).all()
                    if pde == 'burgers':
                        assert torch.equal(expected[:, 0], x)
                    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
                        checkpoint = Path(tmp) / 'model.pt'
                        torch.save(model.state_dict(), checkpoint)
                        restored = adapter.make_model(pde, config)
                        restored.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True))
                        actual = adapter.predict(restored, x, params, mean, std, scale, pde)
                        assert torch.equal(expected, actual)
                results.append(dict(family=family, dtype=dtype, pde=pde, shape=list(actual.shape),
                                    checkpoint_bitwise_parity=True, boundary_exact=True,
                                    supplied_state_returned_exactly=(pde == 'burgers'),
                                    intermediates_f64_audited=(dtype == 'float64'),
                                    real_parameters=sum(p.numel() for p in model.parameters()), loss=float(loss.detach())))
                del model, restored, optimizer
                torch.cuda.empty_cache()
    # `model.features` must build coordinate channels that do not depend on the batch
    # element: `families.DeepONet2d` evaluates its trunk on the first element's channels only.
    coords = adapter.features(torch.rand((3, 1, 33, 33), device='cuda', dtype=torch.float64),
                              torch.full((3, 1), .05, device='cuda', dtype=torch.float64),
                              torch.zeros((1, 2, 1, 1), device='cuda', dtype=torch.float64),
                              torch.ones((1, 2, 1, 1), device='cuda', dtype=torch.float64))[:, -2:]
    assert torch.equal(coords, coords[:1].expand_as(coords)), 'coordinate channels are not batch-invariant'
    env['jax'] = jax.__version__
    out.write_text(json.dumps(dict(passed=True, environment=env, cases=results,
                                   role='implementation smoke, not scientific accuracy or timing evidence'), indent=2) + '\n')
    print(out.read_text())


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    main(args.output)
