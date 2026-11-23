"""Frozen assets and original POD regeneration recipe; no training updates."""
from __future__ import annotations
import gc, hashlib, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'vendor'))
os.environ.setdefault('JAX_ENABLE_X64','true')
import jax
import jax.numpy as jnp
import bankio as BI
import diag_floor as D
import head_rom as H
import ns3d_fom as F
import shift_rom as SR
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_default_matmul_precision','highest')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda:f.read(1<<20),b''):h.update(piece)
    return h.hexdigest()

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')

def load_head(path):
    d=np.load(path)
    return ({k:jnp.asarray(d['p_'+k]) for k in ('W1','b1','W2','b2','W3','b3','Ws')},np.asarray(d['Z']))

def learned_basis(n):
    asset=HERE/'assets/bank_selected.npz';d=np.load(asset)
    basis,info=BI.mesh_bank(BI.load_params(asset),d['T'],n,chunk=32768,abort=.05)
    info.pop('lowdin_factor')
    return basis,info

def pod_basis(n):
    original=json.loads((HERE/'configs/original_dev64.json').read_text())
    train,_,_=D.generate(F.parameters(original['train_seed'],original['train_cases']),n,
                          original['dt_truth'],original['horizon'])
    center=jax.jit(lambda f:SR.shift_field(f,-SR.grid_centroid(f)*n))
    for c in range(len(train)):
        for t in range(train.shape[1]):train[c,t]=np.asarray(center(jnp.asarray(train[c,t])))
    raw,eigenvalues,orth=D.pod_basis(train.reshape(-1,3*n**3),64,original['gram_block'])
    del train;gc.collect()
    basis,_=D.orthonormalize_prefix(raw,64)
    del raw
    probe=np.load(HERE/'assets/pod_bank_probe.npz')
    fields,_,_=D.generate(F.parameters(original['train_seed'],int(probe['cases'])),n,
                          original['dt_truth'],original['horizon'])
    coefficients=np.stack([basis.T@np.asarray(center(jnp.asarray(f))).ravel() for case in fields for f in case])
    signs=np.sign(np.sum(coefficients*probe['coefficients'],axis=0))
    gap=float(np.linalg.norm(coefficients*signs-probe['coefficients'])/np.linalg.norm(probe['coefficients']))
    if gap>1e-8:raise RuntimeError('regenerated POD coefficient probe mismatch')
    return np.ascontiguousarray(basis*signs),{'coefficient_probe_relative':gap,'rank':64,
        'training_seed':original['train_seed'],'training_cases':original['train_cases'],
        'recipe':'original snapshot Gram, eigh, prefix QR and frozen probe sign correction',
        'new_training':False,'pod_orthogonality_before_qr':orth,'spectrum':eigenvalues}

@jax.jit
def target_projection(initial,basis):
    n=initial.shape[-1]
    center=SR.grid_centroid(initial)
    return basis.T@SR.shift_field(initial,-center*n).ravel(),center

@jax.jit
def query_input(initial,geometry):
    """The unchanged initial output projection in F.make_solver, without stepping."""
    return F.ifft(F.project(F.fft(initial),geometry))

@jax.jit
def reconstruct_initial(coefficients,center,basis):
    n=round((basis.shape[0]/3)**(1/3))
    return SR.shift_field((basis@coefficients).reshape(3,n,n,n),center*n)
