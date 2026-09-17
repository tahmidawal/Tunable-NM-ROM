"""Local smoke for the resolution ladder (DESIGN §A5), tiny and sub-minute.

Checks, without any trained checkpoint: (1) the NumPy prolongation equals the Burgers lane's
`engines.output_field` to 1e-12 and is exact at nested nodes; (2) restriction followed by
prolongation is the identity at rung 256; (3) all three families actually run a forward pass
at every rung, which is where an FNO with 32 modes on a 33x33 grid would fail if the library
could not truncate; (4) the contract still holds at a coarse rung — float64 output, exact
zero boundary, and the supplied (coarse) state returned bitwise.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import model as adapter
import resolution as R

TINY = {'fno': dict(width=8, modes=32, layers=2),
        'unet': dict(family='unet', base=8, groups=8, dtype='float32'),
        'transolver': dict(family='transolver', dim=32, layers=2, heads=4, slices=8,
                           mlp_ratio=2, patch=4, ref=8, dtype='float32')}


def main(out):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu'
    print('jax_backend=gpu', flush=True)
    env = adapter.configure()
    prolongation = R.check_prolongation()

    rng = np.random.default_rng(20260917)
    fine = np.zeros((6, 257, 257))
    fine[:, 1:-1, 1:-1] = rng.standard_normal((6, 255, 255))
    assert np.array_equal(R.prolong(R.restrict(fine, 256), 256), fine), 'rung 256 must be the identity'
    for rung in R.RUNGS:
        coarse = R.restrict(fine, rung)
        back = R.prolong(coarse, 256)
        stride = 256 // rung
        assert np.allclose(back[:, ::stride, ::stride], coarse, rtol=0, atol=1e-12), 'nested nodes must be exact'
        for edge in (back[:, 0, :], back[:, -1, :], back[:, :, 0], back[:, :, -1]):
            assert not np.any(edge), 'prolongation must preserve the zero boundary'

    cases = []
    torch.manual_seed(20260914)
    for family, config in TINY.items():
        network = adapter.make_model('burgers', config)
        network.eval()
        mean = torch.zeros((1, 2, 1, 1), device='cuda', dtype=torch.float64)
        std = torch.ones_like(mean)
        scale = torch.tensor(1., device='cuda', dtype=torch.float64)
        shapes = {}
        for rung in R.RUNGS:
            n = rung + 1
            x = torch.rand((1, 1, n, n), device='cuda', dtype=torch.float64)
            x[..., (0, -1), :] = 0
            x[..., :, (0, -1)] = 0
            p = torch.full((1, 1), .05, device='cuda', dtype=torch.float64)
            with torch.no_grad():
                y = adapter.predict(network, x, p, mean, std, scale, 'burgers')
            assert y.shape == (1, 6, 1, n, n) and y.dtype == torch.float64
            assert (y[..., 0, :] == 0).all() and (y[..., -1, :] == 0).all()
            assert (y[..., :, 0] == 0).all() and (y[..., :, -1] == 0).all()
            assert torch.equal(y[:, 0], x), 'supplied state must be returned exactly at every rung'
            lifted = R.prolong(y.cpu().numpy()[0][:, 0], 256)
            assert lifted.shape == (6, 257, 257) and np.isfinite(lifted).all()
            shapes[rung] = list(y.shape)
        cases.append(dict(family=family, config=config, shapes=shapes, ran_every_rung=True))
        print(json.dumps(cases[-1]), flush=True)
        del network
        torch.cuda.empty_cache()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(passed=True, environment=env, rungs=list(R.RUNGS),
                                   prolongation_check=prolongation, families=cases,
                                   role='implementation smoke for the resolution ladder; no accuracy evidence'),
                              indent=2) + '\n')
    print('resolution_smoke=passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
