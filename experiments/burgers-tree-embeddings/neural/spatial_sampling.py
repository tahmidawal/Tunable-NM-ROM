"""Offline supervised spatial sampling and explicit training-family strata.

Pure NumPy preparation/drawing. No PDE collocation or online refinement policy.
The mixture changes the sampling distribution, not the physical quadrature
objective: use mean((w_i / p_i) * error_i**2) / current_reference_norm**2.
Do not multiply by the grid size or self-normalize these importance weights.
"""
import numpy as np


def mixture_probabilities(weights, parent_residual, alpha=.5):
    """Return one probability vector per residual field, preserving input rank.

    The nonzero physical-weight component guarantees coverage. Scaling each
    residual before squaring avoids overflow without changing its distribution.
    A zero weighted residual falls back to normalized physical quadrature.
    """
    w=np.asarray(weights,dtype=np.float64)
    r=np.asarray(parent_residual,dtype=np.float64)
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or np.any(w<0) or not np.isfinite(w.sum()) or not w.sum()>0:
        raise ValueError('Quadrature weights must be finite, nonnegative and have positive total')
    if r.ndim not in (1,2) or r.shape[-1]!=len(w) or not np.isfinite(r).all():
        raise ValueError('Residuals must be finite point vectors or field-by-point arrays')
    if not np.isfinite(alpha) or not 0<=alpha<1:
        raise ValueError('Require 0 <= alpha < 1 to retain physical-domain coverage')
    single=r.ndim==1;r=np.atleast_2d(r)
    base=w/w.sum()
    weighted_support=np.where(w[None,:]>0,r,0.)
    scale=np.max(np.abs(weighted_support),axis=1,keepdims=True)
    scaled=weighted_support/np.where(scale>0,scale,1.)
    energy=base[None,:]*scaled**2;total=energy.sum(axis=1,keepdims=True)
    concentrated=np.divide(energy,total,out=np.broadcast_to(base,energy.shape).copy(),where=total>0)
    p=(1-alpha)*base[None,:]+alpha*concentrated
    p/=p.sum(axis=1,keepdims=True)
    return p[0] if single else p


def prepare(weights, parent_residual, alpha=.5, cohort='train'):
    """Prepare immutable CPU CDFs once; residual inputs must be training only.

    For large datasets prepare in field chunks. Probability and CDF arrays each
    require 8 * fields * points bytes. No dense snapshot-by-sample alias table is
    built. Reusing a stale training residual remains unbiased, though it can be
    less efficient than refreshing the proposal.
    """
    if cohort!='train':raise ValueError('Residual-guided sampler is training-only')
    w=np.array(weights,dtype=np.float64,copy=True)
    p=np.atleast_2d(mixture_probabilities(w,parent_residual,alpha))
    cdf=np.cumsum(p,axis=1);cdf[:,-1]=1.
    for value in [w,p,cdf]:value.setflags(write=False)
    return dict(weights=w,probabilities=p,cdf=cdf,alpha=float(alpha),cohort=cohort,
                fields=len(p),points=len(w),proposal_bytes=p.nbytes+cdf.nbytes+w.nbytes,
                importance_upper_bound=float(w.sum()/(1-alpha)))


def draw(rng, sampler, field_indices, count):
    """Draw independently for each selected field with replacement.

    CDFs are prepared once; each row uses NumPy's compiled binary search. The
    returned importance weights are the *original* w/p, with no grid-size factor.
    """
    ids=np.asarray(field_indices)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError('Field indices must be a one-dimensional integer array')
    if np.any(ids<0) or np.any(ids>=sampler['fields']):raise ValueError('Field index out of range')
    if not isinstance(count,(int,np.integer)) or count<=0:raise ValueError('Positive sample count required')
    if sampler['cohort']!='train':raise ValueError('Residual-guided sampler is training-only')
    uniforms=rng.random((len(ids),count))
    indices=np.empty((len(ids),count),dtype=np.int64)
    for row,field in enumerate(ids):
        indices[row]=np.searchsorted(sampler['cdf'][field],uniforms[row],side='right')
    probability=sampler['probabilities'][ids[:,None],indices]
    if np.any(probability<=0):raise RuntimeError('CDF selected a zero-probability point')
    importance=sampler['weights'][indices]/probability
    return dict(indices=indices,importance_weights=importance,probabilities=probability)


def stratify(primary_parameters, expanded_families, probabilities=None, narrow_width=.05, cohort='train'):
    """Partition trajectories, not snapshots, into five explicit training strata.

    Primary parameter convention matches data.parameters: smooth flag column0,
    pulse width column3. Expanded rows follow primary rows in the dataset. Default
    probabilities are equal over nonempty strata; this is family-balanced rather
    than uniform over trajectories and must be registered as an objective change.
    """
    if cohort!='train':raise ValueError('Training strata cannot use held-out cohorts')
    p=np.asarray(primary_parameters,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]<4 or not np.isfinite(p).all():raise ValueError('Invalid primary parameters')
    if not np.isfinite(narrow_width) or narrow_width<=0:raise ValueError('Invalid width threshold')
    if not np.isin(p[:,0],[0.,1.]).all():raise ValueError('Unknown primary smooth/pulse flag')
    names=['smooth','narrow_pulse','broad_pulse','ridge','two_pulse']
    groups={name:[] for name in names}
    for i,row in enumerate(p):
        if row[0]==1:name='smooth'
        else:
            if row[3]<=0:raise ValueError('Pulse width must be positive')
            name='narrow_pulse' if row[3]<=narrow_width else 'broad_pulse'
        groups[name].append(i)
    for i,family in enumerate(expanded_families,start=len(p)):
        if family not in ['ridge','two_pulse']:raise ValueError('Unknown expanded family')
        groups[family].append(i)
    active=[name for name in names if groups[name]]
    if not active:raise ValueError('No training trajectories')
    if probabilities is None:probabilities={name:1/len(active) for name in active}
    if set(probabilities)!=set(active):raise ValueError('Probabilities must name exactly the nonempty strata')
    probs=np.asarray([probabilities[name] for name in active],dtype=np.float64)
    if not np.isfinite(probs).all() or np.any(probs<=0) or not np.isclose(probs.sum(),1.,rtol=0,atol=1e-12):
        raise ValueError('Positive stratum probabilities must sum to one')
    return dict(groups={name:np.asarray(groups[name],dtype=np.int64) for name in active},
        counts={name:len(groups[name]) for name in names},
        probabilities={name:float(probabilities[name]) for name in active},
        trajectory_count=sum(map(len,groups.values())),narrow_width=float(narrow_width),cohort=cohort,
        per_trajectory_probability={name:float(probabilities[name]/len(groups[name])) for name in active})


def draw_stratum(rng, strata, batch_size):
    """Choose one registered stratum, then distinct trajectories within it.

    The caller draws a time uniformly per trajectory. Sampling probabilities are
    explicit; no development errors or online solution fields select the stratum.
    """
    if strata['cohort']!='train':raise ValueError('Training-only strata required')
    names=list(strata['probabilities'])
    if not isinstance(batch_size,(int,np.integer)) or not 0<batch_size<=min(map(len,strata['groups'].values())):
        raise ValueError('Batch must fit every selected stratum without replacement')
    name=names[int(rng.choice(len(names),p=[strata['probabilities'][n] for n in names]))]
    group=strata['groups'][name]
    return name,rng.choice(group,size=batch_size,replace=False)
