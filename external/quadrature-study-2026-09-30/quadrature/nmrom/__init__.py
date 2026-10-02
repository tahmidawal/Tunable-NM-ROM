"""Partial-decoding hyper-reduction study for NM-ROMs.

Modules
-------
grid        uniform Dirichlet grid, sine (DST-I) weak tests, eigenvalues
pdes        PDE definitions: nonlinear term on the mesh and in the continuum
fom         full-order Newton--BiCGStab solver (JAX, matrix free)
bank        random-Fourier-feature coordinate-network bank (partial decoding)
head        latent head with linear skip, nested correction directions
quadrature  quadrature rules: dense mesh, NNLS EQ, mesh lattice, tensor Gauss,
            Smolyak sparse grids, QMC (Sobol, Halton, Fibonacci lattice, tent)
rom         reduced residual, Levenberg--Marquardt with block damping, rollout
"""
import jax
jax.config.update("jax_enable_x64", True)
