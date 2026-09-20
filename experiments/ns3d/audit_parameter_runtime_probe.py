"""Tiny independent CPU probe of the fixed final seed's float64 transforms.

Imports no production PDE/data code and reads no final fields. Run unchanged
on both CPU architectures to distinguish RNG membership from libm rounding.
"""
import hashlib,json,platform,sys
import numpy as np

rng=np.random.default_rng(202609203);rows=[];exponents=[]
lo=np.log(.002);hi=np.log(.01)
for _ in range(32):
    center=rng.uniform(0,1,3);width=rng.uniform(.12,.24);strength=rng.uniform(.6,1.4)
    exponent=rng.uniform(lo,hi);exponents.append(float(exponent))
    rows.append([*center,width,strength,np.exp(exponent)])
values=np.asarray(rows,dtype=np.float64)
print(json.dumps(dict(seed=202609203,count=32,platform=platform.platform(),machine=platform.machine(),
    python=sys.version,numpy=np.__version__,log_bounds=[float(lo),float(hi)],exponents=exponents,
    parameters=values.tolist(),parameter_sha256=hashlib.sha256(values.tobytes()).hexdigest()),indent=2))
