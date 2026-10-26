"""Frozen no-HR POD callable for a paired frontier job; no cohort or reference access."""
from pathlib import Path
import hashlib
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'jcp-hr-baselines'
sys.path.insert(0,str(OLD))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import hrcore as H
import rom as R
import pipeline as P

PHI_SHA='7dc39cf9fbb5ea06d77001e4f8ccd69113dcdc8f22838f3cc12d0159b3431b2c'


def setup(phi_path):
    """Return (fused, initialize, evolve, decode, data, schedule, metadata).

    fused(u0,nu,data,schedule) has flat u0 shape(63**3,), the full mesh interior
    at65nodes, and returns fields(6,63**3), W(6,512), C and native solver stats.
    Split call: c0,scale=initialize(u0,data); W,C,stats=evolve(c0,nu,scale,data,schedule);
    fields=decode(W,data). Large arrays remain explicit arguments.
    """
    assert Path(H.__file__).resolve().parent==OLD.resolve(),'hrcore module collision'
    assert Path(R.__file__).resolve().parent==OLD.resolve(),'POD rom module collision'
    h=hashlib.sha256()
    with Path(phi_path).open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    assert h.hexdigest()==PHI_SHA
    z=np.load(phi_path);trust=float(z['trust'][list(z['ranks']).index(512)])
    ph=np.asarray(z['Phi'][:,:512]);assert np.max(np.abs(ph.T@ph-np.eye(512)))<=1e-10
    Phi=jnp.asarray(ph);del ph,z
    cell=P.Cell(3,64);cell.set_basis(Phi,{512:trust});cell.constructible_r={512}
    data=cell.arm_data('none',512,'-')[0]
    fused=cell.query('none',512,'common')
    initialize=jax.jit(lambda u,d:(d['Phi'].T@u,jnp.linalg.norm(d['Phi'].T@u)))
    evolve=jax.jit(R.make_evolve('none',3,64,'common',H.NMAX[3],trust=trust))
    decode=jax.jit(lambda W,d:W@d['Phi'].T)
    sch=R.schedule(3,'BDF2',1/60,'prod')
    meta=dict(name='POD512-noHR-BDF2-common',r=512,L=64,scheme='BDF2',dt=1/60,
              input_shape=[63**3],output_shape=[6,63**3],basis_sha256=PHI_SHA,trust_radius=trust,
              hyper_reduced=False,qualification='requires audited complete validation; not inferred by setup')
    return fused,initialize,evolve,decode,data,sch,meta
