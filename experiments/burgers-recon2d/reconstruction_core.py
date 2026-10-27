"""Shared frozen linear reconstructions. No PDE, reference or descriptor input.

Point arrays are interior-only, x slowest, physical amplitudes. All preparation
and queries are intended for an f64 GPU batch allocation. Import itself does no
numerical work. Cached basis/factorization preparation must be timed separately.
"""
from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

RIDGES = (0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
SVD_RTOL = 1e-10
GRAM_RTOL = 1e-12


def host(a):
    return np.asarray(jax.device_get(a))


def ready(a):
    return jax.block_until_ready(a)


@jax.jit
def _dense_fit(y, fitter):
    return y @ fitter.T


@jax.jit
def _dense_decode(c, basis):
    return c @ basis.T


def interior_coordinates(L: int, dim: int):
    x = np.arange(1, L, dtype=np.float64) / L
    return np.stack(np.meshgrid(*([x] * dim), indexing='ij'), -1).reshape(-1, dim)


def restriction_indices(Lc: int, Lf: int, dim: int):
    if Lf % Lc:
        raise ValueError('Only nested point-sample restriction is allowed')
    ix = np.arange(1, Lc) * (Lf // Lc) - 1
    return np.ravel_multi_index(np.meshgrid(*([ix] * dim), indexing='ij'),
                               ((Lf - 1,) * dim)).reshape(-1)


@dataclass
class LinearPlan:
    name: str
    fit: Any
    synthesize: Any
    metadata: dict
    # Arrays are retained for independently reconstructing the timed outputs.
    arrays: dict

    def query(self, y):
        """y shape(batch,coarse_interior_points); return (fine_fields, coefficients)."""
        c = self.fit(y)
        return self.synthesize(c), c


@dataclass
class BasisGeometry:
    label: str
    fine: Any
    coarse: Any
    transform: Any
    left: Any
    singular: Any
    right: Any
    wc: float
    metadata: dict
    prior_cache: dict = field(default_factory=dict)

    def plan(self, ridge: float = 0., prior_variances=None):
        """Fine-field-norm ridge, or covariance-weighted POD ridge.

        prior_variances refer to the ORIGINAL basis coefficients and must be
        derived from training only. Its precision is normalized so its smallest
        eigenvalue in the physical-whitened basis is one.
        """
        s, u, v = self.singular, self.left, self.right
        keep = s > SVD_RTOL * s[0]
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError('ridge must be finite and nonnegative')
        use_prior = prior_variances is not None
        penalty_eigen = None
        solve_residual = 0.
        prior_condition = None
        if use_prior and ridge > 0:
            var = jnp.asarray(prior_variances, dtype=jnp.float64)
            if var.ndim != 1 or var.size != self.transform.shape[0]:
                raise ValueError('prior variances must match original prefix')
            if not bool(host(jnp.all(jnp.isfinite(var) & (var > 0)))):
                raise ValueError('prior variances must be finite and positive')
            key = hashlib.sha256(host(var).tobytes()).hexdigest()
            if key not in self.prior_cache:
                precision = var.max() / var
                h = self.transform.T @ (precision[:, None] * self.transform)
                h = (h + h.T) / 2
                if not bool(host(jnp.all(jnp.isfinite(h)))):
                    raise ValueError('nonfinite covariance precision')
                ev, hv = jnp.linalg.eigh(h)
                if not bool(host(jnp.all(jnp.isfinite(ev) & (ev > 0)))):
                    raise ValueError('nonpositive covariance precision')
                h = h / ev[0]
                normalized_ev = ev / ev[0]
                inv_root = (hv / jnp.sqrt(normalized_ev)[None, :]) @ hv.T
                cw = self.coarse * np.sqrt(self.wc)
                pu, ps, pvh = jnp.linalg.svd(cw @ inv_root, full_matrices=False)
                pv = inv_root @ pvh.T
                # This factorization avoids solving normal equations. Small
                # normal-equation residuals are checked only as diagnostics.
                self.prior_cache[key] = (h, normalized_ev, pu, ps, pv,
                                         cw.T @ cw, cw.T @ pu)
            h, normalized_ev, pu, ps, pv, gram0, rhs_reduced = self.prior_cache[key]
            penalty_eigen = host(normalized_ev).tolist()
            prior_condition = float(penalty_eigen[-1])
            pgain = ps / (ps * ps + ridge)
            reduced_fit = pv * pgain[None, :]
            solve_residual = float(host(jnp.linalg.norm(
                (gram0 + ridge * h) @ reduced_fit - rhs_reduced)
                / jnp.maximum(jnp.linalg.norm(rhs_reduced), 1e-300)))
            if not np.isfinite(solve_residual) or solve_residual > 1e-7:
                raise ValueError(f'covariance ridge residual failed: {solve_residual}')
            fitter = reduced_fit @ pu.T * np.sqrt(self.wc)
            fit_gram = reduced_fit @ reduced_fit.T
            amp = float(host(jnp.sqrt(jnp.maximum(
                jnp.linalg.eigvalsh((fit_gram + fit_gram.T) / 2)[-1], 0.))))
        else:
            safe = jnp.where(keep, s, 1.)
            gain = jnp.where(keep, safe / (safe * safe + ridge), 0.)
            fitter = (v.T * gain[None, :]) @ u.T * np.sqrt(self.wc)
            amp = float(host(jnp.max(jnp.abs(gain))))
        ready(fitter)
        if not bool(host(jnp.all(jnp.isfinite(fitter)))) or not np.isfinite(amp):
            raise ValueError('nonfinite fit map or physical amplification')
        # Fine basis is orthonormal in physical L2. This is the actual map norm
        # from the coarse physical L2 input to the fine physical L2 output.
        # The standard ridge factorization already supplies singular gains.
        # For a non-diagonal prior, use the small exact map Gram above.
        # Matrices are explicit traced arguments, never compiled constants.
        fit = lambda y: _dense_fit(y, fitter)
        qf = self.fine
        synthesize = lambda c: _dense_decode(c, qf)
        suffix = 'covridge' if use_prior else 'ridge'
        name = f'{self.label}_{suffix}{ridge:g}'
        meta = dict(self.metadata, name=name, ridge=float(ridge),
                    prior='training_covariance' if use_prior else 'fine_field_norm',
                    prior_precision_eigenvalues=penalty_eigen,
                    prior_condition=prior_condition,
                    ridge_normal_residual=solve_residual,
                    setup_valid=True,
                    physical_amplification=amp)
        return LinearPlan(name, fit, synthesize, meta,
                          dict(fine_basis=qf, fit_matrix=fitter,
                               original_coefficient_transform=self.transform))


def prepare_basis(bc, bf, Lc: int, Lf: int, dim: int, label: str):
    """Freeze a prefix's physically normalized observation/reconstruction map.

    bf contains network/mode evaluations, never fine target values. Numerical
    loss of fine representation directions is explicitly reported.
    """
    bc = jnp.asarray(bc, dtype=jnp.float64)
    bf = jnp.asarray(bf, dtype=jnp.float64)
    if bc.ndim != 2 or bf.ndim != 2 or bc.shape[1] != bf.shape[1]:
        raise ValueError('basis matrix shape mismatch')
    if bc.shape[0] != (Lc - 1) ** dim or bf.shape[0] != (Lf - 1) ** dim:
        raise ValueError('basis rows do not match interior lattice')
    if not bool(host(jnp.all(jnp.isfinite(bc)) & jnp.all(jnp.isfinite(bf)))):
        raise ValueError('nonfinite basis')
    wf, wc = float(Lf) ** -dim, float(Lc) ** -dim
    gram = bf.T @ bf * wf
    vals, vecs = jnp.linalg.eigh((gram + gram.T) / 2)
    vals_h = host(vals)
    if vals_h[-1] <= 0 or vals_h[0] < -1e-10 * vals_h[-1]:
        raise ValueError('invalid physical Gram')
    idx = np.flatnonzero(vals_h > GRAM_RTOL * vals_h[-1])
    transform = vecs[:, idx] / jnp.sqrt(vals[idx])[None, :]
    qf, qc = bf @ transform, bc @ transform
    u, s, vh = jnp.linalg.svd(qc * np.sqrt(wc), full_matrices=False)
    ready((qf, qc, u, s, vh))
    sh = host(s)
    orth = float(host(jnp.linalg.norm(qf.T @ qf * wf - jnp.eye(len(idx)), ord='fro')
                      / np.sqrt(len(idx))))
    if not np.isfinite(orth) or orth > 1e-7:
        raise ValueError(f'physical whitening failed: {orth}')
    meta = dict(label=label, Lc=Lc, Lf=Lf, dim=dim,
                original_rank=int(bf.shape[1]), physical_rank=int(len(idx)),
                removed_fine_directions=int(bf.shape[1] - len(idx)),
                sampled_effective_rank=int(np.sum(sh > SVD_RTOL * sh[0])),
                singular_values=sh.tolist(), fine_gram_eigenvalues=vals_h.tolist(),
                physical_orthogonality_error=orth, svd_rtol=SVD_RTOL,
                gram_rtol=GRAM_RTOL, coarse_weight=wc, fine_weight=wf)
    return BasisGeometry(label, qf, qc, transform, u, s, vh, wc, meta)


def _tensor_apply(a, matrix, dim):
    """One common matrix per physical axis; batch axis is untouched."""
    for axis in range(1, dim + 1):
        a = jnp.moveaxis(a, axis, -1) @ matrix.T
        a = jnp.moveaxis(a, -1, axis)
    return a


def interpolation_plans(Lc: int, Lf: int, dim: int):
    """Geometry-only classical methods. No state/reference-dependent decisions."""
    if dim not in (2, 3) or Lc < 3 or Lf < Lc or Lf % Lc:
        raise ValueError('expected nested 2D/3D grids with at least four coarse nodes')
    nc, nf = (Lc - 1) ** dim, (Lf - 1) ** dim
    xc = np.arange(Lc + 1) / Lc
    xf = np.arange(1, Lf) / Lf
    eye = np.eye(Lc + 1)[:, 1:-1]
    linear = np.stack([np.interp(xf, xc, eye[:, j]) for j in range(Lc - 1)], axis=1)
    cubic = CubicSpline(xc, eye, axis=0, bc_type='not-a-knot')(xf)
    plans = []
    for name, matrix in [('linear', linear), ('cubic_notaknot', cubic)]:
        m = jnp.asarray(matrix, jnp.float64)
        fit = jax.jit(lambda y: y)
        synth = jax.jit(lambda c, m=m: _tensor_apply(c.reshape((-1,) + (Lc-1,) * dim), m, dim)
                        .reshape((-1, nf)))
        # Exact separable operator norm in volume-weighted physical norms.
        amp = float(np.linalg.svd(matrix, compute_uv=False)[0] ** dim
                    * (Lc / Lf) ** (dim / 2))
        meta = dict(name=name, dim=dim, Lc=Lc, Lf=Lf, physical_amplification=amp,
                    interpolation=True, data_prior=False)
        plans.append(LinearPlan(name, fit, synth, meta, dict(axis_matrix=m)))
    k = np.arange(1, Lc)
    cmat = np.sqrt(2 / Lc) * np.sin(np.pi * np.outer(k, k) / Lc)
    emat = np.sqrt(2 / Lc) * np.sin(np.pi * np.outer(xf, k))
    cm, em = jnp.asarray(cmat), jnp.asarray(emat)
    kk = np.meshgrid(*([k / (Lc - 1)] * dim), indexing='ij')
    freq4 = sum(v*v for v in kk) ** 2
    filters = [('sine_full', np.ones((Lc-1,) * dim))]
    for frac in (.5, .75):
        cutoff = max(1, int(np.floor((Lc-1) * frac)))
        mask = np.ones((Lc-1,) * dim, dtype=bool)
        for axis, g in enumerate(np.meshgrid(*([k] * dim), indexing='ij')):
            mask &= g <= cutoff
        filters.append((f'sine_cut{cutoff}', mask.astype(float)))
    for lam in (1e-6, 1e-4, 1e-2, 1.):
        filters.append((f'sine_smooth{lam:g}', 1 / (1 + lam * freq4)))
    for name, filt in filters:
        f = jnp.asarray(filt)
        fit = jax.jit(lambda y, f=f: (_tensor_apply(y.reshape((-1,) + (Lc-1,) * dim), cm, dim)
                                     * f).reshape((-1, nc)))
        synth = jax.jit(lambda c: _tensor_apply(c.reshape((-1,) + (Lc-1,) * dim), em, dim)
                        .reshape((-1, nf)))
        # Nested fine grid resolves every retained sine mode. The physically
        # normalized transform is isometric and the filter is diagonal.
        meta = dict(name=name, dim=dim, Lc=Lc, Lf=Lf,
                    physical_amplification=float(np.max(filt)),
                    interpolation=name == 'sine_full', data_prior=False)
        plans.append(LinearPlan(name, fit, synth, meta,
                                dict(analysis_axis=cm, synthesis_axis=em, spectral_filter=f)))
    return plans


def field_errors(fields, reference, n0):
    """No hidden fit access: scorer is a separate function."""
    fields, reference = np.asarray(fields), np.asarray(reference)
    if fields.shape != reference.shape or n0 <= 0:
        raise ValueError('incompatible scoring arrays/norm')
    return np.linalg.norm(fields - reference, axis=-1) / n0


def benchmark(plan, y_host, repetitions=3, burn_seconds=.1):
    """Warm compiled plan, burn in, then retain synchronized compute+host costs.

    Return every timing plus coefficients and fields from the LAST timed
    invocation. Caller scores that actual field and stores its hash/coefficient.
    Transfer timing is measured separately and end-to-end includes it.
    """
    y_host = np.ascontiguousarray(y_host, dtype=np.float64)
    y = jnp.asarray(y_host)
    t0 = time.perf_counter()
    ready(plan.query(y))
    compile_warm = time.perf_counter() - t0
    deadline = time.perf_counter() + burn_seconds
    while time.perf_counter() < deadline:
        ready(plan.query(y))
    reps = []
    field_repetitions, coefficient_repetitions = [], []
    last_fields = last_coeff = None
    for _ in range(repetitions):
        start = time.perf_counter()
        y = ready(jax.device_put(y_host))
        ti = time.perf_counter()
        coeff = ready(plan.fit(y))
        tf = time.perf_counter()
        fields = ready(plan.synthesize(coeff))
        td = time.perf_counter()
        last_fields = np.array(jax.device_get(fields), copy=True)
        to = time.perf_counter()
        # coefficient transfer/storage is audit overhead, excluded from the
        # common field-output service contract and timed separately.
        last_coeff = np.array(jax.device_get(coeff), copy=True)
        ta = time.perf_counter()
        if not np.isfinite(last_fields).all() or not np.isfinite(last_coeff).all():
            raise ValueError('nonfinite timed reconstruction; retain failed candidate')
        field_hash = hashlib.sha256(last_fields.tobytes(order='C')).hexdigest()
        coefficient_hash = hashlib.sha256(last_coeff.tobytes(order='C')).hexdigest()
        field_repetitions.append(last_fields)
        coefficient_repetitions.append(last_coeff)
        reps.append(dict(input_seconds=ti-start, fit_seconds=tf-ti,
                         decode_seconds=td-tf, output_seconds=to-td,
                         device_query_seconds=td-ti, host_query_seconds=to-start,
                         audit_coefficient_seconds=ta-to,
                         field_sha256=field_hash, coefficient_sha256=coefficient_hash))
    return dict(repetitions=reps, compile_warm_seconds=compile_warm,
                fields=last_fields, coefficients=last_coeff,
                field_repetitions=field_repetitions,
                coefficient_repetitions=coefficient_repetitions,
                repeated_fields_identical=len({r['field_sha256'] for r in reps}) == 1)
