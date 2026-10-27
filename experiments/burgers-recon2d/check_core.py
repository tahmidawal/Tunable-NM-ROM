"""Bounded algebra smoke; no training/reference/cohort inputs."""
import json
import os
import numpy as np
import jax
import jax.numpy as jnp
import reconstruction_core as C


def run():
    backend = jax.default_backend()
    print(f'jax_backend={backend}', flush=True)
    assert backend == 'gpu'
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    assert jax.config.jax_enable_x64
    results = []
    for dim in (2, 3):
        lc, lf = 4, 8
        xc, xf = C.interior_coordinates(lc, dim), C.interior_coordinates(lf, dim)
        modes = np.ones((4, dim), dtype=int)
        modes[1, 0] = 2
        modes[2, 1] = 2
        modes[3, 0] = lc  # exactly invisible on the observation grid
        bf = np.prod(np.sin(np.pi * xf[:, :, None] * modes.T[None, :, :]), axis=1)
        bc = np.prod(np.sin(np.pi * xc[:, :, None] * modes.T[None, :, :]), axis=1)
        y = bc @ np.array([1., -.3, .2, 0.])
        truth = bf @ np.array([1., -.3, .2, 0.])
        ix = C.restriction_indices(lc, lf, dim)
        assert np.max(np.abs(bf[ix] - bc)) < 1e-14
        for plan in C.interpolation_plans(lc, lf, dim):
            field, _ = plan.query(jnp.asarray(y[None]))
            field = C.host(field)[0]
            if plan.metadata['interpolation']:
                assert np.max(np.abs(field[ix] - y)) < 2e-12, plan.name
            if plan.name == 'sine_full':
                assert np.linalg.norm(field-truth) / np.linalg.norm(truth) < 1e-12
        g = C.prepare_basis(bc, bf, lc, lf, dim, f'fixture{dim}')
        assert g.metadata['physical_rank'] == 4
        assert g.metadata['sampled_effective_rank'] == 3
        rng = np.random.default_rng(901 + dim)
        rot, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        transform = rot @ np.diag([2., .3, 4., 1.])
        other = C.prepare_basis(bc @ transform, bf @ transform, lc, lf, dim, 'changed_basis')
        for ridge in (0., 1e-4):
            p, p2 = g.plan(ridge), other.plan(ridge)
            f, c = p.query(jnp.asarray(y[None]))
            f2, _ = p2.query(jnp.asarray(y[None]))
            assert np.max(np.abs(C.host(f-f2))) < 1e-10
            if ridge == 0:
                assert np.max(np.abs(C.host(f)[0]-truth)) < 1e-11
            explicit = C.host(p.arrays['fine_basis'] @ p.arrays['fit_matrix'])
            amp = np.linalg.svd(explicit * (lc/lf)**(dim/2), compute_uv=False)[0]
            assert abs(amp-p.metadata['physical_amplification']) < 1e-10
        cov = g.plan(1e-4, np.array([1., .5, .2, .1]))
        fitter = C.host(cov.arrays['fit_matrix'])
        t = C.host(g.transform)
        h = t.T @ np.diag([1., 2., 5., 10.]) @ t
        h /= np.linalg.eigvalsh(h)[0]
        qc = C.host(g.coarse)
        expected = np.linalg.solve(qc.T @ qc * lc**-dim + 1e-4*h,
                                   qc.T * lc**-dim)
        assert np.max(np.abs(fitter-expected)) < 1e-9
        measured = C.benchmark(g.plan(0.), y[None], repetitions=3, burn_seconds=.02)
        assert measured['repeated_fields_identical']
        assert len(measured['coefficient_repetitions']) == 3
        for f in measured['field_repetitions']:
            assert np.max(np.abs(f[0]-truth)) < 1e-11
        results.append(dict(dim=dim, status='pass', sampled_rank=3,
                            fine_rank=4, eigenbasis_invariance=True,
                            sine_exact=True, covariance_ridge_checked=True))
    print(json.dumps(dict(status='pass', results=results)), flush=True)


if __name__ == '__main__':
    run()
