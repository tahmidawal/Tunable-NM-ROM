"""Prospective explicit dynamic teacher arguments; no production integration."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DeviceBlocks:
    names: tuple
    ranges: tuple
    arguments: tuple
    logical_bytes: int


def prepare(arrays,block_points,jax_module,byte_limit=2**30):
    """Copy each ordered immutable block once; never capture it inside a JIT."""
    if type(block_points) is not int or block_points<=0 or type(byte_limit) is not int or byte_limit<=0:raise ValueError('positive explicit bounds required')
    if not arrays:raise ValueError('nonempty ordered teacher arguments required')
    names=tuple(arrays);host=tuple(np.asarray(arrays[k]) for k in names);n=len(host[0])
    if n<=0 or any(a.ndim<1 or len(a)!=n or a.dtype!=np.float64 or not np.isfinite(a).all() for a in host):raise ValueError('same finite f64 point axis required')
    size=sum(a.nbytes for a in host)
    if size>byte_limit:raise MemoryError('teacher device payload exceeds explicit bound before allocation')
    if jax_module.default_backend()!='gpu' or not jax_module.config.x64_enabled or str(jax_module.config.jax_default_matmul_precision)!='highest':raise RuntimeError('GPU/f64/highest required')
    ranges=tuple((i,min(n,i+block_points)) for i in range(0,n,block_points))
    blocks=tuple(tuple(jax_module.device_put(np.ascontiguousarray(a[i:j])) for a in host) for i,j in ranges)
    for row in blocks:
        for a in row:a.block_until_ready()
    if sum(a.size*a.dtype.itemsize for row in blocks for a in row)!=size:raise RuntimeError('device payload size mismatch')
    return DeviceBlocks(names,ranges,blocks,size)


def theta_once(theta,jax_module,byte_limit=2**30):
    """One immutable parameter transfer per full evaluation, explicit JIT argument."""
    a=np.asarray(theta)
    if a.ndim!=1 or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('finite flattened f64 parameter vector required')
    if type(byte_limit) is not int or byte_limit<=0 or a.nbytes>byte_limit:raise MemoryError('parameter payload exceeds bound before allocation')
    if jax_module.default_backend()!='gpu' or not jax_module.config.x64_enabled or str(jax_module.config.jax_default_matmul_precision)!='highest':raise RuntimeError('GPU/f64/highest required')
    return jax_module.device_put(a)
