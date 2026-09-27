"""Exact weak-bank propagation with stationary nonlinear output projection.

This is a changed method with 64 internal linear coordinates.  It is explicitly
not the 32-coordinate latent manifold RK4 equation.  Returned displacement is
on the frozen nonlinear manifold; velocity differentiates its stationary fit.
"""
from functools import partial
import time
import numpy as np
import pilot as base
from pilot import jax,jnp
from fresh_models import head_apply
from fresh_learning import fit_batch


def prepare(bank, metric):
    """Offline mesh-only eigensystem and frozen metric coordinate transform."""
    eigen,vectors=jnp.linalg.eigh(bank['k'])
    if metric=='l2': weight=jnp.eye(64)
    elif metric=='h1':
        # Scalar normalization affects fit magnitudes, not the minimizer.
        weight=jnp.linalg.cholesky(bank['k']).T/jnp.sqrt(jnp.trace(bank['k'])/64)
    else:raise ValueError(metric)
    return dict(eigen=eigen,vectors=vectors,weight=weight,p_weighted=base.transform_head(bank['p'],weight))


@jax.jit
def modal_targets(a0,b0,c,times,eigen,vectors):
    omega=c*jnp.sqrt(eigen)
    aa,bb=vectors.T@a0,vectors.T@b0
    cosine,sine=jnp.cos(times[:,None]*omega),jnp.sin(times[:,None]*omega)
    return ((cosine*aa+sine*bb/omega)@vectors.T,
            (-sine*omega*aa+cosine*bb)@vectors.T)


@partial(jax.jit,static_argnames=('iterations',))
def project_targets(bank,prepared,targets,velocities,scale,*,iterations):
    n=len(targets);k=bank['p']['linear'].shape[1]
    affine=(targets-bank['common_center'])@bank['common_inverse'].T
    starts=jnp.concatenate((affine[:,None],jnp.zeros_like(affine)[:,None],
        jnp.broadcast_to(bank['fixed_codes'],(n,6,k))),axis=1)
    weighted=targets@prepared['weight'].T
    raw=fit_batch(prepared['p_weighted'],bank['frozen'],jnp.repeat(weighted,8,axis=0),
        jnp.full(n*8,scale),starts.reshape(-1,k),kind='mlp',iterations=iterations)
    zz,objective,gradient,counts,damping,stationarity,rank,finite=[x.reshape(n,8,*x.shape[1:]) for x in raw]
    selected=jnp.argmin(jnp.where(finite,objective,jnp.inf),axis=1)
    z=zz[jnp.arange(n),selected]
    a,b,w,minimum_eigen,maximum_eigen,backward=jax.vmap(lambda z,target,velocity:
        implicit_velocity(bank['p'],bank['frozen'],prepared['weight'],z,target,velocity))(z,targets,velocities)
    fits=dict(z_all=zz,selected=selected,objective=objective,gradient=gradient,iterations=counts,
        damping=damping,stationarity=stationarity,rank_ratio=rank,finite=finite,
        hessian_min_eigenvalue=minimum_eigen,hessian_max_eigenvalue=maximum_eigen,
        implicit_velocity_backward_error=backward)
    return a,b,z,w,fits


def implicit_velocity(p,frozen,weight,z,target,velocity):
    fun=lambda value:head_apply(p,frozen,value,'mlp')
    residual=lambda value:weight@(fun(value)-target)
    objective=lambda value:.5*jnp.sum(residual(value)**2)
    a=fun(z);jac=jax.jacfwd(fun)(z)
    hessian=jax.hessian(objective)(z)
    rhs=jac.T@weight.T@weight@velocity
    w=jnp.linalg.solve(hessian,rhs)
    eigen=jnp.linalg.eigvalsh(hessian)
    backward=jnp.linalg.norm(hessian@w-rhs)/jnp.maximum(jnp.linalg.norm(hessian)*jnp.linalg.norm(w)+jnp.linalg.norm(rhs),1e-300)
    return a,jac@w,w,eigen[0],eigen[-1],backward


@partial(jax.jit,static_argnames=('iterations',))
def projected_evolution(bank,prepared,u0,v0,c,times,*,iterations):
    a0=bank['g'].T@(bank['mass']*u0.ravel())
    b0=bank['g'].T@(bank['mass']*v0.ravel())
    targets,velocities=modal_targets(a0,b0,c,times,prepared['eigen'],prepared['vectors'])
    scale=jnp.maximum(jnp.linalg.norm(prepared['weight']@a0),1e-10)
    a,b,z,w,fits=project_targets(bank,prepared,targets,velocities,scale,iterations=iterations)
    return dict(coefficients=a,velocity_coefficients=b,z=z,w=w,fits=fits,
                targets=targets,velocity_targets=velocities,a0=a0,b0=b0,scale=scale)


@jax.jit
def linear_evolution(bank,prepared,u0,v0,c,times):
    a0=bank['g'].T@(bank['mass']*u0.ravel());b0=bank['g'].T@(bank['mass']*v0.ravel())
    a,b=modal_targets(a0,b0,c,times,prepared['eigen'],prepared['vectors'])
    return dict(coefficients=a,velocity_coefficients=b,a0=a0,b0=b0)


@jax.jit
def decode_coefficients(a,b,g):return a@g.T,b@g.T


def device_query(method,bank,supplied,grid,cfg):
    metric=method.removeprefix('projected_') if method.startswith('projected_') else 'l2'
    prepared=bank['prepared_'+metric]
    times=jnp.arange(round(cfg['end_time']/cfg['observation_dt'])+1)*cfg['observation_dt']
    jax.block_until_ready(times)
    t0=time.perf_counter()
    if method=='linear_bank64':aux=linear_evolution(bank,prepared,*supplied,times)
    else:aux=projected_evolution(bank,prepared,*supplied,times,iterations=cfg['fit_iterations'])
    jax.block_until_ready(aux);t1=time.perf_counter()
    u,v=decode_coefficients(aux['coefficients'],aux['velocity_coefficients'],bank['g'])
    u,v=u.reshape(-1,*grid.shape),v.reshape(-1,*grid.shape)
    jax.block_until_ready((u,v));t2=time.perf_counter()
    u,v=np.asarray(u),np.asarray(v);t3=time.perf_counter();aux=jax.tree.map(np.asarray,aux)
    record=dict(method=method,setting=0.,boundary=grid.bx,intervals=grid.n,steps=0,observation_stride=0,
        internal_configuration_dimension=64,internal_phase_dimension=128,
        nonlinear_output_configuration_dimension=32 if method.startswith('projected_') else None,
        method_label='Linear bank evolution with nonlinear output reconstruction' if method.startswith('projected_') else 'Unconstrained linear learned-bank evolution',
        initialization_contract='Internal64 moments are raw bank projections of suppliedu0/v0; returnedt0 fields use the declared output reconstruction and are scored. Baseline manifold-fitted initial moments differ.',
        seconds=dict(initialization_and_parameter_projection=0.,evolution=t1-t0,dense_device_output=t2-t1,complete_device_query=t2-t0),
        timer_components_note='evolution includes source projection, modal propagation, every nonlinear fit and implicit velocity solve',
        post_timer_host_transfer_seconds=t3-t2,device_plus_output_transfer_seconds=t3-t0,
        output_sha256=dict(u=base.array_sha(u),v=base.array_sha(v)),output_bytes=u.nbytes+v.nbytes,
        completed=bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v))))
    if 'fits' in aux:
        fits=aux['fits'];idx=np.arange(len(times));best=fits['selected']
        select=lambda key:fits[key][idx,best]
        stationary=select('finite')&(select('rank_ratio')>1e-8)&(select('gradient')<=1e-7)&((select('stationarity')<=1e-6)|(select('objective')<=1e-20))
        record.update(all_output_fits_stationary=bool(np.all(stationary)),stationary_output_count=int(np.sum(stationary)),output_count=len(times),
            selected_iterations=select('iterations').tolist(),selected_gradients=select('gradient').tolist(),
            selected_stationarity=select('stationarity').tolist(),selected_objectives=select('objective').tolist(),
            minimum_projection_hessian_ratio=float(np.min(fits['hessian_min_eigenvalue']/fits['hessian_max_eigenvalue'])),
            maximum_implicit_velocity_backward_error=float(np.max(fits['implicit_velocity_backward_error'])))
        record['projection_kinematic_eligible']=bool(np.all(stationary) and record['minimum_projection_hessian_ratio']>1e-10 and record['maximum_implicit_velocity_backward_error']<1e-10)
    return u,v,record,dict(projection=aux,coefficients=aux['coefficients'],velocity_coefficients=aux['velocity_coefficients'])


@partial(jax.jit,static_argnames=('iterations',))
def neighboring_projection(bank,prepared,a0,b0,c,times,starts,scale,*,iterations):
    targets,_=modal_targets(a0,b0,c,times,prepared['eigen'],prepared['vectors'])
    raw=fit_batch(prepared['p_weighted'],bank['frozen'],targets@prepared['weight'].T,
        jnp.full(len(times),scale),starts,kind='mlp',iterations=iterations)
    return jax.vmap(lambda z:head_apply(bank['p'],bank['frozen'],z,'mlp'))(raw[0]),raw


def kinematic_check(bank,prepared,aux,c,times,iterations=800):
    """Diagnostic only: nearby converged projections on each selected branch."""
    checks=[];arrays={}
    for delta in (2e-4,1e-4):
        plus,praw=neighboring_projection(bank,prepared,aux['a0'],aux['b0'],c,times+delta,
            aux['z'],aux['scale'],iterations=iterations)
        minus,mraw=neighboring_projection(bank,prepared,aux['a0'],aux['b0'],c,times-delta,
            aux['z'],aux['scale'],iterations=iterations)
        derivative=(np.asarray(plus)-np.asarray(minus))/(2*delta)
        # Both fit sets must actually satisfy the original stopping criterion.
        stationarity=[]
        for raw in (praw,mraw):
            z,obj,grad,count,damp,stat,rank,finite=map(np.asarray,raw)
            stationarity.extend((finite&(rank>1e-8)&(grad<=1e-7)&((stat<=1e-6)|(obj<=1e-20))).tolist())
        discrepancy=np.linalg.norm(derivative-np.asarray(aux['velocity_coefficients']),axis=1)
        scale=float(np.sqrt(np.asarray(aux['b0'])@np.asarray(aux['b0'])+float(c)**2*np.asarray(aux['a0'])@np.asarray(bank['k'])@np.asarray(aux['a0'])))
        checks.append(dict(delta=delta,all_neighbor_fits_stationary=bool(all(stationarity)),
            max_initial_velocity_scaled_difference=float(np.max(discrepancy)/scale),absolute_differences=discrepancy.tolist()))
        arrays[f'plus_{delta}']=np.asarray(plus);arrays[f'minus_{delta}']=np.asarray(minus)
        arrays[f'plus_fit_stationarity_{delta}']=np.asarray(praw[5]);arrays[f'minus_fit_stationarity_{delta}']=np.asarray(mraw[5])
    return checks,arrays
