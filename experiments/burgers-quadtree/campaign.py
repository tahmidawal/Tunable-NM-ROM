from pathlib import Path
import sys,json,os,time,hashlib
import numpy as np
H=Path(__file__).resolve().parent
T=H/'vendor/tree/experiments'
for p in [T/'jcp-references',T/'quadrature-study',T/'mr-burgers2d']:
 sys.path.insert(0,str(p))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import engines as E
import refcore as R
import pscore as P

def save(path,obj):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
def meta():
 return dict(job=os.environ.get('SLURM_JOB_ID'),commit=os.environ.get('SOURCE_COMMIT'),device=str(jax.devices()),jax=jax.__version__,x64=jax.config.x64_enabled,precision=str(jax.config.jax_default_matmul_precision))
def draw():
 # Generate full tables ONCE: inherited column-wise RNG is not prefix stable.
 return E.params_draw(2026111701,256),E.params_draw(2026111702,64)
def timer(fn):
 t=time.perf_counter(); v=fn(); jax.block_until_ready(v); return v,time.perf_counter()-t
