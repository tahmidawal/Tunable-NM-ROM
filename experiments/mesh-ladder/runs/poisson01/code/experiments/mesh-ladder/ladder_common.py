"""Shared mesh-ladder measurement helpers.

One frozen checkpoint per PDE is evaluated across a mesh ladder inside a single
job on a single GPU. Three cost components are measured separately and are never
obtained by subtracting one measurement from another:

* ``cached``   - the reduced (latent) solve alone, with its inputs already built.
* ``complete`` - the complete device query: supplied input already resident on the
  device to the full dense output still resident on the device.
* ``setup``    - offline per-mesh preparation, charged separately.

Host transfers are timed in the same invocation as the complete device query and
are reported as their own columns, never folded into it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import jax


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def host(value):
    return jax.tree_util.tree_map(np.asarray, value)


def dump(path, value):
    """Atomic, NaN-refusing JSON write so a killed job cannot leave half a record."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def preflight():
    """Assert the environment every accepted result in this project requires."""
    backend = jax.default_backend()
    print(f'jax_backend={backend}', flush=True)
    assert backend == 'gpu', backend
    assert jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    return dict(jax_backend=backend, jax_version=jax.__version__,
                gpu=jax.devices()[0].device_kind, x64=bool(jax.config.jax_enable_x64),
                matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'],
                commit=os.environ.get('COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                slurm_node=os.environ.get('SLURMD_NODENAME'))


def timed(callable_, *args):
    """Wall time of one fully completed device computation."""
    start = time.perf_counter()
    value = jax.block_until_ready(callable_(*args))
    return value, time.perf_counter() - start


def restrict(field, intervals):
    """Nested-node injection onto a coarser grid; no averaging, no interpolation.

    ``field`` has shape (time, n+1, n+1) on a grid of ``n`` intervals per axis.
    The operator is exact node selection, so it is only defined when the target
    grid's nodes are a subset of the source grid's nodes.
    """
    field = np.asarray(field)
    source = field.shape[-1] - 1
    assert field.shape[-2] == source + 1, field.shape
    if source % intervals:
        raise ValueError(f'restriction requires nested nodes: {source} -> {intervals}')
    stride = source // intervals
    return np.ascontiguousarray(field[..., ::stride, ::stride], dtype=np.float64)


def validate_restriction(field, ladder):
    """Evidence that the declared restriction is a well-defined nested injection.

    Checks, for one concrete reference field, that (a) restricting in one step and
    restricting through any intermediate rung of the ladder give bitwise identical
    results, (b) the coarse grid's nodes really are source nodes, and (c) the
    boundary rows and columns survive restriction exactly.
    """
    field = np.asarray(field)
    source = field.shape[-1] - 1
    rungs = sorted({int(x) for x in ladder if source % int(x) == 0}, reverse=True)
    coarsest = rungs[-1]
    direct = restrict(field, coarsest)
    chained = field
    for rung in rungs:
        chained = restrict(chained, rung)
    checks = dict(source_intervals=source, ladder=rungs, coarsest=coarsest,
                  operator='nested-node injection (stride selection)',
                  chained_equals_direct=bool(np.array_equal(direct, chained)),
                  direct_sha256=sha_array(direct), chained_sha256=sha_array(chained))
    stride = source // coarsest
    node_positions = np.arange(0, source + 1, stride)
    checks['nodes_are_source_nodes'] = bool(node_positions[-1] == source and len(node_positions) == coarsest + 1)
    checks['boundary_preserved'] = bool(
        np.array_equal(direct[..., 0, :], field[..., 0, ::stride])
        and np.array_equal(direct[..., -1, :], field[..., -1, ::stride])
        and np.array_equal(direct[..., :, 0], field[..., ::stride, 0])
        and np.array_equal(direct[..., :, -1], field[..., ::stride, -1]))
    assert checks['chained_equals_direct'] and checks['nodes_are_source_nodes'] and checks['boundary_preserved']
    return checks


def summarize(values):
    values = [float(x) for x in values]
    array = np.asarray(values)
    quartile1, quartile3 = np.percentile(array, [25, 75])
    fence = quartile3 + 1.5 * (quartile3 - quartile1)
    return dict(count=len(values), median=float(np.median(array)), minimum=float(array.min()),
                maximum=float(array.max()), mean=float(array.mean()),
                upper_tukey_fence=float(fence), upper_outliers=int(np.sum(array > fence)),
                repetitions=values)
