"""Mass-weighted POD/ordering, fixed Gram whitening, and explicit numerical rank gates."""
from __future__ import annotations
import numpy as np

RANK_CUTOFF=1e-12


def positive_weights(weights,n):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError("finite strictly positive physical-volume weights required")
    return w


def checked_cholesky(gram):
    g=np.asarray(gram,dtype=np.float64)
    if g.ndim!=2 or g.shape[0]!=g.shape[1] or not np.isfinite(g).all():
        raise ValueError("finite square Gram required")
    if np.linalg.norm(g-g.T)>1e-12*max(np.linalg.norm(g),1e-300):
        raise ValueError("Gram is not symmetric")
    eig=np.linalg.eigvalsh(g)
    if eig[-1]<=0 or eig[0]<=RANK_CUTOFF*eig[-1]:
        raise ValueError("Gram fails fixed 1e-12 eigenvalue rank gate; no hidden ridge")
    return np.linalg.cholesky(g)


def weighted_gram(values,weights):
    a=np.asarray(values,dtype=np.float64)
    w=positive_weights(weights,len(a))
    flat=a.reshape(-1,a.shape[-1])
    repeated=np.repeat(w,int(np.prod(a.shape[1:-1])))
    return flat.T@(repeated[:,None]*flat)


def ordered_transform(bank_values,correction_snapshots,weights):
    """QR+training SVD; snapshots[s,n,d], bank[n,d,r]. No validation access."""
    g=np.asarray(bank_values,dtype=np.float64)
    u=np.asarray(correction_snapshots,dtype=np.float64)
    if u.ndim!=3 or u.shape[1:]!=g.shape[:2] or len(u)<g.shape[-1]:
        raise ValueError("insufficient snapshots or incompatible shape")
    sw=np.repeat(np.sqrt(positive_weights(weights,len(g))),g.shape[1])
    a=g.reshape(-1,g.shape[-1])*sw[:,None]
    singular=np.linalg.svd(a,compute_uv=False)
    if singular[-1]<=RANK_CUTOFF*singular[0]:
        raise ValueError("neural columns rank deficient before ordering")
    q,rf=np.linalg.qr(a,mode="reduced")
    coefficients=(u.reshape(len(u),-1)*sw[None,:])@q
    _,importance,vt=np.linalg.svd(coefficients,full_matrices=False)
    rotation=vt.T
    sign=np.sign(rotation[np.argmax(np.abs(rotation),axis=0),np.arange(rotation.shape[1])])
    rotation*=np.where(sign==0,1,sign)
    transform=np.linalg.solve(rf,rotation)
    return transform,{"basis_singular_values":singular.tolist(),"training_importance":importance.tolist(),
                      "rank_cutoff":RANK_CUTOFF,"training_count":len(u),"mean_centered":False}


def weighted_pod(snapshots,weights,max_rank=None):
    """Snapshots[s,n,...]; no automatic mean or pressure-gauge shift."""
    u=np.asarray(snapshots,dtype=np.float64)
    if u.ndim<2 or not np.isfinite(u).all():
        raise ValueError("finite snapshots required")
    n=u.shape[1]
    components=int(np.prod(u.shape[2:]))
    sw=np.repeat(np.sqrt(positive_weights(weights,n)),components)
    left,sv,vt=np.linalg.svd((u.reshape(len(u),-1)*sw).T,full_matrices=False)
    numerical_rank=int(np.sum(sv>RANK_CUTOFF*sv[0])) if len(sv) and sv[0]>0 else 0
    rank=numerical_rank if max_rank is None else min(int(max_rank),numerical_rank)
    if rank<1:
        raise ValueError("no nonzero POD mode")
    basis=(left[:,:rank]/sw[:,None]).reshape(*u.shape[1:],rank)
    return basis,{"singular_values":sv.tolist(),"original_count":len(sv),"numerical_rank":numerical_rank,
                  "retained_rank":rank,"rank_cutoff":RANK_CUTOFF,"mean_centered":False,
                  "snapshot_combinations":(vt[:rank].T/sv[:rank]).tolist()}


def pressure_basis(training_pressure_corrections,weights,velocity_prefix):
    """Constant first; then mass POD of centered TRAINING pressure corrections.

    This changes basis construction, not the physical pressure convention: reconstructed
    pressure keeps its solved constant coefficient. No gauge constraint is imposed.
    """
    p=np.asarray(training_pressure_corrections,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or velocity_prefix not in (2,4,8):
        raise ValueError("finite training pressure snapshots and declared prefix required")
    w=positive_weights(weights,p.shape[1]);volume=w.sum()
    means=p@w/volume
    centered=p-means[:,None]
    constant=np.ones(p.shape[1])/np.sqrt(volume)
    matrix=(centered*np.sqrt(w)).T
    left,sv,vt=np.linalg.svd(matrix,full_matrices=False)
    reference=max(1.,float(np.linalg.svd((p*np.sqrt(w)).T,compute_uv=False)[0]))
    numerical_rank=int(np.sum(sv>RANK_CUTOFF*reference))
    if numerical_rank==0:
        return constant[:,None],{"pressure_rank":1,"centered_numerical_rank":0,"constant_first":True,
                                 "physical_gauge_shift":False,"snapshot_means":means.tolist(),
                                 "rank_reference":reference,"centered_singular_values":sv.tolist()}
    rank=min(velocity_prefix-1,numerical_rank)
    modes=left[:,:rank]/np.sqrt(w[:,None])
    meta={"numerical_rank":numerical_rank,"retained_rank":rank,"singular_values":sv.tolist(),
          "rank_reference":reference,"rank_reference_definition":"max(1,original weighted pressure snapshot sigma_max)",
          "relative_rank_cutoff":RANK_CUTOFF,"snapshot_combinations":(vt[:rank].T/sv[:rank]).tolist()}
    result=np.c_[constant,modes]
    return result,{"pressure_rank":result.shape[1],"centered_numerical_rank":meta["numerical_rank"],
                   "constant_first":True,"physical_gauge_shift":False,"snapshot_means":means.tolist(),
                   "centered_pod":meta}


def canonicalize_eigenmodes(eigenvalues,eigenvectors,metric_gram,degeneracy_relative=1e-10):
    """Deterministic orientation inside eigenvalue clusters using ordered FE DOF values.

    Input eigenvectors are columns, already orthonormal in the stated FE metric. Largest
    residual DOF-row norm is the pivot; np.argmax fixes ties to smallest FE DOF index.
    Canonical result is invariant under orthogonal rotations of a degenerate block.
    """
    lam=np.asarray(eigenvalues,dtype=np.float64);v=np.asarray(eigenvectors,dtype=np.float64)
    if v.ndim!=2 or v.shape[1]!=len(lam) or np.any(np.diff(lam)<0):
        raise ValueError("sorted eigenpairs required")
    if not np.allclose(metric_gram,np.eye(len(lam)),atol=1e-10,rtol=1e-10):
        raise ValueError("eigenvectors must be metric-orthonormal before canonicalization")
    out=np.empty_like(v);groups=[];start=0
    while start<len(lam):
        end=start+1
        while end<len(lam) and abs(lam[end]-lam[start])<=degeneracy_relative*max(1.,abs(lam[start]),abs(lam[end])):
            end+=1
        rows=v[:,start:end].copy();rotation=[];pivots=[]
        for _ in range(end-start):
            pivot=int(np.argmax(np.sum(rows*rows,axis=1)))
            direction=rows[pivot].copy();norm=np.linalg.norm(direction)
            if norm<=1e-14:
                raise ValueError("degenerate-block orientation lost rank")
            direction/=norm
            rotation.append(direction);pivots.append(pivot)
            rows-=np.outer(rows@direction,direction)
        out[:,start:end]=v[:,start:end]@np.column_stack(rotation)
        groups.append({"start":start,"end":end,"pivot_FE_dofs":pivots})
        start=end
    return out,{"groups":groups,"relative_eigenvalue_cluster_tolerance":degeneracy_relative,
                "tie_rule":"largest residual DOF-row norm, first FE DOF index on exact tie"}


def select_test_columns(values,weights,target_count,gradients=None):
    """Ordered candidates -> fixed full-rule H1 or L2 independent normalized test bank.

    Momentum candidates: all maximum pressure-prefix supremizers, then canonical vector
    H1 eigenmodes. Continuity: constant, pressure modes, then canonical scalar Laplace
    modes. This routine receives that immutable enumeration and never sees state errors.
    """
    v=np.asarray(values,dtype=np.float64);w=positive_weights(weights,len(v))
    flat=v.reshape(-1,v.shape[-1])*np.repeat(np.sqrt(w),int(np.prod(v.shape[1:-1])))[:,None]
    if gradients is not None:
        g=np.asarray(gradients,dtype=np.float64)
        gf=g.reshape(-1,g.shape[-1])*np.repeat(np.sqrt(w),int(np.prod(g.shape[1:-1])))[:,None]
        flat=np.r_[flat,gf]
    selected=[];rejected=[]
    for j in range(flat.shape[1]):
        sv=np.linalg.svd(flat[:,selected+[j]],compute_uv=False)
        if not len(sv) or sv[0]==0 or sv[-1]<=RANK_CUTOFF*sv[0]:
            rejected.append(j);continue
        selected.append(j)
        if len(selected)==target_count:
            break
    if len(selected)!=target_count:
        raise ValueError(f"test enumeration has {len(selected)} independent candidates, needs {target_count}")
    _,r=np.linalg.qr(flat[:,selected],mode="reduced")
    transform=np.linalg.inv(r)
    return v[...,selected]@transform, (None if gradients is None else g[...,selected]@transform), {
        "candidate_indices":selected,"rejected_indices":rejected,"target_count":target_count,
        "transform":transform.tolist(),"rank_cutoff":RANK_CUTOFF,
        "metric":"dimensionless H1" if gradients is not None else "dimensionless L2"}


def projection_floor(basis,snapshots,weights):
    b=np.asarray(basis,dtype=np.float64)
    u=np.asarray(snapshots,dtype=np.float64)
    w=positive_weights(weights,len(b))
    components=int(np.prod(b.shape[1:-1]))
    sw=np.repeat(np.sqrt(w),components)
    a=b.reshape(-1,b.shape[-1])*sw[:,None]
    y=u.reshape(len(u),-1)*sw[None,:]
    coef,_,rank,sv=np.linalg.lstsq(a,y.T,rcond=RANK_CUTOFF)
    residual=y-coef.T@a.T
    return coef.T,{"squared_errors":np.sum(residual**2,axis=1).tolist(),
                   "squared_reference_norms":np.sum(y**2,axis=1).tolist(),
                   "numerical_rank":int(rank),"singular_values":sv.tolist()}


def inf_sup(divergence,velocity_gram,pressure_gram,reference_beta):
    """Standard velocity H1/pressure L2 coupling; reference MUST use those same norms.

    The coordinator corrected draft A1's inf-sup norm before numerical execution.
    Jacobian unknown whitening separately continues to use velocity/pressure L2.
    """
    lv=checked_cholesky(velocity_gram)
    lp=checked_cholesky(pressure_gram)
    b=np.asarray(divergence,dtype=np.float64)
    if b.shape!=(len(lp),len(lv)) or len(lp)>len(lv):
        raise ValueError("pressure divergence coupling shape or dimension invalid")
    scaled=np.linalg.solve(lp,b)@np.linalg.inv(lv.T)
    sv=np.linalg.svd(scaled,compute_uv=False)
    beta=float(sv[-1])
    if not np.isfinite(reference_beta) or reference_beta<=0:
        raise ValueError("positive full-velocity/same-pressure reference required")
    ratio=beta/reference_beta
    return {"beta":beta,"reference_beta":float(reference_beta),"ratio":ratio,
            "singular_values":sv.tolist(),"norms":"dimensionless velocity H1, pressure L2",
            "passed":bool(beta>=1e-3 and ratio>=0.5)}
