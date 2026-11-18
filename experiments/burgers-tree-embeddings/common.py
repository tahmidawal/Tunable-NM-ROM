"""Shared precision, evidence and timing helpers for the new study only."""
from pathlib import Path
import hashlib, json, os, time
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent

def save(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, allow_nan=False,
                      default=lambda x: np.asarray(x).tolist()) + '\n'
    q = p.with_suffix(p.suffix + '.tmp'); q.write_text(text); q.replace(p)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def meta():
    return dict(job=os.environ.get('SLURM_JOB_ID'), commit=os.environ.get('SOURCE_COMMIT'),
                devices=[str(d) for d in jax.devices()], jax=jax.__version__,
                x64=jax.config.x64_enabled, precision=str(jax.config.jax_default_matmul_precision))

def timer(fn):
    t = time.perf_counter(); a = fn(); jax.block_until_ready(a)
    return a, time.perf_counter() - t

def preflight():
    m = meta(); print(json.dumps(m), flush=True)
    assert jax.default_backend() == 'gpu'
    assert m['x64'] and m['precision'] == 'highest'
    return m

def burn_gpu(seconds=2.):
    """Hardware conditioning only; elapsed allocation is still charged to this job."""
    multiply=jax.jit(lambda a:(a@a)/a.shape[0])
    value=jnp.ones((1024,1024),dtype=jnp.float64)
    value=multiply(value);jax.block_until_ready(value)
    start=time.perf_counter();count=0
    while time.perf_counter()-start<seconds:
        value=multiply(value);jax.block_until_ready(value);count+=1
    return dict(seconds=time.perf_counter()-start,iterations=count)
