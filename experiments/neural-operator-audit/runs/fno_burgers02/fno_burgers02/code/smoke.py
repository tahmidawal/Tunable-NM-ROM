"""Bounded CUDA precision, gradient, boundary and checkpoint checks."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
import model as adapter


class DtypeAudit(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for x in tree_leaves(out):
            if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex()):
                if x.dtype not in (torch.float64, torch.complex128):
                    raise RuntimeError(f"Intermediate precision failure: {func} {x.dtype}")
        return out


def main(out):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu'
    print('jax_backend=gpu', flush=True)
    env = adapter.configure()
    torch.manual_seed(20260914)
    results = []
    for pde in ("poisson", "burgers"):
        config = dict(modes=6, width=8, layers=2)
        model = adapter.make_model(pde, config)
        adapter.check_dtypes(model)
        x = torch.rand((2, 1, 17, 17), device="cuda", dtype=torch.float64)
        x[..., (0, -1), :] = 0
        x[..., :, (0, -1)] = 0
        params = torch.full((2, 0 if pde == "poisson" else 1), .05, device="cuda", dtype=torch.float64)
        mean = torch.zeros((1, 1 + params.shape[1], 1, 1), device="cuda", dtype=torch.float64)
        std = torch.ones_like(mean)
        scale = torch.tensor(1., device="cuda", dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        with DtypeAudit():
            y = adapter.predict(model, x, params, mean, std, scale, pde)
            loss = y[:, -1].square().mean()
            loss.backward()
            adapter.check_dtypes(model, gradients=True)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        with torch.no_grad():
            expected = adapter.predict(model, x, params, mean, std, scale, pde)
            assert expected.dtype == torch.float64
            assert (expected[..., 0, :] == 0).all() and (expected[..., -1, :] == 0).all()
            assert (expected[..., :, 0] == 0).all() and (expected[..., :, -1] == 0).all()
            if pde == "burgers":
                assert torch.equal(expected[:, 0], x)
            with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
                checkpoint = Path(tmp) / 'model.pt'
                torch.save(model.state_dict(), checkpoint)
                restored = adapter.make_model(pde, config)
                # This checkpoint was created immediately above by this test.
                # NeuralOperator stores the conv class in its extra state.
                restored.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False))
                actual = adapter.predict(restored, x, params, mean, std, scale, pde)
                assert torch.equal(expected, actual)
            results.append(dict(pde=pde, shape=list(actual.shape), checkpoint_bitwise_parity=True,
                                forward_backward_optimizer_intermediates_f64=True, loss=float(loss.detach())))
        del model, restored, optimizer
        torch.cuda.empty_cache()
    env['jax'] = jax.__version__
    out.write_text(json.dumps(dict(passed=True, environment=env, cases=results,
                                  role='implementation smoke, not scientific accuracy or timing evidence'), indent=2)+'\n')
    print(out.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    main(args.output)
