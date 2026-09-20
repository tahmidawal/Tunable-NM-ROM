"""Declared-capacity/native-grid forward/backward smoke, two updates only."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jnp
import common as C
from operators import training as T
from operators import heat_adapter as H
kind=sys.argv[1];out=Path(sys.argv[2]);root=Path(__file__).resolve().parents[1]
setting=next(s for s in json.loads((root/'extra03.json').read_text())['operators'] if s['model']['kind']==kind)
cfg={**setting['training'],'steps':2,'validation_every':1,'curve_every':1,'wall_seconds':40,'batch_size':1}
rng=np.random.default_rng(920324);initial=rng.uniform(size=(3,31,31,31))
fields=initial[:,None]*np.exp(-np.arange(6)[None,:,None,None,None]*.1)
x,y=H.arrays(fields,1.)
p,info=T.train(x[:2],y[:2],x[2:],y[2:],setting['model'],cfg,out,C.dump,C.checkpoint)
fine=rng.uniform(size=(63,63,63));q=H.engine(p,setting['model'],1.,64);values=np.asarray(q(jnp.asarray(fine)))
assert values.shape==(6,63,63,63) and np.isfinite(values).all() and values.dtype==np.float64
assert np.array_equal(values[0],fine)
C.dump(out/'audit.json',dict(native_grid_backward=True,direct_transfer_shape=list(values.shape),training=info))
print('NATIVE_CAPACITY_PASS',kind,info['parameter_count'],info['seconds'],flush=True)
