"""Gated one-architecture training library. No production dataset is bundled or opened.

Caller must supply a reviewed FD1-scoped admission and frozen training data hash. Actual
job launcher adds source/config/input hashes and enforces the hard allocation ceiling.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np
from bank import initialize,FrozenBank
from spaces import ordered_transform,positive_weights
from qualification import require_qualification


def array_hash(*arrays):
    h=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value,dtype=np.float64)
        h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def train(points,weights,velocity_corrections,geometry,admission,steps=3000,max_seconds=1800.,evidence_directory=None):
    """Only trained max-r bank; snapshots [case,point,component] in velocity/U units."""
    require_qualification(admission)
    if admission.get("split")!="training":
        raise RuntimeError("FD1 reviewed training-only admission required")
    if not admission.get("review_record") or not admission.get("reference_manifest_sha256"):
        raise RuntimeError("reference manifest and independent review record required")
    x=np.asarray(points,dtype=np.float64)
    y=np.asarray(velocity_corrections,dtype=np.float64)
    w=positive_weights(weights,len(x))
    if y.shape!=(9,len(x),geometry.dimension) or geometry.dimension!=2:
        raise ValueError("first recipe is only the nine-state 2D steady training family")
    expected=[15+1.25*j for j in range(9)]
    if admission.get("Re")!=expected:
        raise ValueError("training Re/order mismatch")
    if not np.isfinite(y).all() or not geometry.fluid_mask(x).all():
        raise ValueError("nonfinite data or nonfluid training point")
    digest=array_hash(x,w,y)
    if admission.get("array_sha256")!=digest:
        raise ValueError("training array provenance mismatch")
    if not 1<=steps<=3000 or not 0<max_seconds<=1800:
        raise ValueError("recipe exceeds fixed pilot step/time cap")
    if evidence_directory is None:raise ValueError("fresh training evidence directory required")
    from training_evidence import TrainingEvidence
    evidence=TrainingEvidence(evidence_directory,{'training_array_sha256':digest,'admission':admission,
        'step_cap':steps,'time_cap_seconds':max_seconds,'architecture':{'rank':8,'width':64,'depth':2},
        'optimizer':'Adam constant learning rate 0.001','precision':'float64','matmul_precision':'highest'})
    return evidence.guarded(lambda:_train_gpu(x,y,w,geometry,admission,steps,max_seconds,evidence,digest))


def _train_gpu(x,y,w,geometry,admission,steps,max_seconds,evidence,digest):
    # Import only after all admission checks. Any invocation must be under jaxrun locally.
    import jax
    jax.config.update("jax_enable_x64",True)
    jax.config.update("jax_default_matmul_precision","highest")
    import jax.numpy as jnp
    import optax
    if jax.default_backend()!="gpu":
        raise RuntimeError("bank fitting requires declared GPU allocation/preflight")
    bank=initialize(geometry,rank=8,width=64,depth=2,seed=27102027)
    offset,scale=geometry.coordinate_scaling()
    # Geometry factors are fixed inputs, never fitted; training has no pressure/validation data.
    envelope=geometry.envelope(x)[0]
    inp=jnp.asarray((x-offset)/scale)
    b=jnp.asarray(envelope)
    sw=jnp.asarray(np.repeat(np.sqrt(w),geometry.dimension))
    target=jnp.asarray(y.reshape(9,-1)*np.asarray(sw))
    params=[(jnp.asarray(ww),jnp.asarray(bb)) for ww,bb in bank.layers]
    optimizer=optax.adam(learning_rate=1e-3)
    optstate=optimizer.init(params)

    def objective(p,points_arg,envelope_arg,weights_arg,target_arg):
        h=points_arg
        for ww,bb in p[:-1]:
            h=jnp.tanh(h@ww+bb)
        h=h@p[-1][0]+p[-1][1]
        vals=(h.reshape(len(points_arg),2,8)*envelope_arg[:,None,None]).reshape(-1,8)
        q,_=jnp.linalg.qr(vals*weights_arg[:,None],mode="reduced")
        coefficients=target_arg@q
        residual=target_arg-coefficients@q.T
        return jnp.sum(residual*residual)/jnp.maximum(jnp.sum(target_arg*target_arg),1e-300)

    @jax.jit
    def update(p,state,points_arg,envelope_arg,weights_arg,target_arg):
        loss,grad=jax.value_and_grad(objective)(p,points_arg,envelope_arg,weights_arg,target_arg)
        changes,state=optimizer.update(grad,state,p)
        return optax.apply_updates(p,changes),state,loss

    start=time.perf_counter();evidence.fit_started=start;evidence.stage='optimization'
    history=[]
    try:
        evidence.checkpoint(params,optstate,jax,time.perf_counter()-start,'initial')
        for i in range(steps):
            if time.perf_counter()-start>=max_seconds:
                break
            params,optstate,loss=update(params,optstate,inp,b,sw,target)
            value=float(loss) # synchronize: the time cap includes device work and compilation
            row={"step":i+1,"pre_update_relative_squared_loss":value if np.isfinite(value) else None,
                 "elapsed_seconds":time.perf_counter()-start,"loss_finite":bool(np.isfinite(value))}
            history.append(row);evidence.step(row)
            if not np.isfinite(value):
                raise RuntimeError("nonfinite training objective; retain job log and stop")
            if i==0 or (i+1)%100==0:
                evidence.checkpoint(params,optstate,jax,time.perf_counter()-start,'periodic')
        evidence.checkpoint(params,optstate,jax,time.perf_counter()-start,'loop_complete')
    except Exception:
        try:evidence.checkpoint(params,optstate,jax,time.perf_counter()-start,'failed')
        except Exception as retention_error:evidence.retention_errors.append(repr(retention_error))
        raise
    evidence.stage='ordering'
    layers=[(np.asarray(ww),np.asarray(bb)) for ww,bb in params]
    raw=FrozenBank(geometry,layers,np.eye(8),{})
    vals=raw.values_blocked(x)
    transform,ordering=ordered_transform(vals,y,w)
    evidence.finish('completed',time.perf_counter()-start)
    metadata={"trained":True,"seed":27102027,"width":64,"depth":2,"rank":8,
              "steps_executed":len(history),"steps_cap":steps,"time_cap_seconds":max_seconds,
              "seconds":time.perf_counter()-start,"training_array_sha256":digest,
              "admission":admission,"ordering":ordering,"loss":"weighted velocity projection QR; no ridge",
              "optimizer":"Adam constant learning rate 0.001","history":history,
              "training_evidence_directory":str(evidence.path),"last_checkpoint":evidence.last_checkpoint}
    return FrozenBank(geometry,layers,transform,metadata)
