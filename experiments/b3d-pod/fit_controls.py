"""Tiny inherited NNLS target/early-optimum and serialization controls."""
from pathlib import Path
import json
import numpy as np
import fit as F

G=F.jnp.eye(8,dtype=F.jnp.float64)
b=F.jnp.array([4.,3.,2.,1.,0.,0.,0.,0.])
rows=[]
for target in [4,6]:
    r=F.NN.greedy_nnls(G,b,taus=(),supports=(target,),cap=target,max_drops=3,max_iter=100,max_seconds=10)
    rec=r['records'].get(('s',target))
    if rec is not None:
        ok=bool(rec['size']>=target and np.all(rec['weights']>0) and r['drift_ok'])
    else:
        ok=bool(r['reason']=='kkt' and r['kkt']['ok'] and np.all(r['weights']>0))
    rows.append(dict(target=target,reason=r['reason'],accepted=ok,kkt=r['kkt'],quality=r['drift_ok']))
serial=json.loads(json.dumps({'finite':np.bool_(True),'failure':np.bool_(False)},default=F.conv))
assert type(serial['finite']) is bool and type(serial['failure']) is bool
out=dict(backend=F.jax.default_backend(),x64=F.jax.config.jax_enable_x64,controls=rows,
         numpy_boolean_serialization=True,pass_=all(x['accepted'] for x in rows))
assert out['backend']=='gpu' and out['x64'] and out['pass_']
p=Path(__file__).parent/'checks/fit-controls.json';p.write_text(json.dumps(out,indent=2,default=F.conv)+'\n')
print(p.read_text(),flush=True)
