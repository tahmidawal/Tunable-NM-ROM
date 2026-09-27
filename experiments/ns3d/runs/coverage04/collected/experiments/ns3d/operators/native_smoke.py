import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import jax
import jax.numpy as jnp
from operators import models3d as M
from operators import heat_adapter as H
rng=np.random.default_rng(20);u=rng.normal(size=(15,15,15));x=np.stack(np.meshgrid(*([np.arange(1,16)/16]*3),indexing='ij'),axis=-1)
for spec in [dict(kind='fno3d',width=2,modes=[2,2,2],depth=1,padding=1),dict(kind='unet3d',width=2,levels=2,periodic=False)]:
 p=M.init_model(jax.random.PRNGKey(2),spec,4,5)
 coarse=np.asarray(H.engine(p,spec,1.,8)(jnp.asarray(u[1::2,1::2,1::2])))
 actual=np.asarray(H.native_engine(p,spec,1.,8,16)(jnp.asarray(u)))
 expected=np.stack([RegularGridInterpolator((np.arange(9)/8,)*3,np.pad(v,1))(x) for v in coarse[1:]])
 err=np.max(np.abs(actual[1:]-expected))
 assert np.array_equal(actual[0],u) and err<1e-12,err
 print(spec['kind'],'native nodal interpolation maxerror',err,flush=True)
