"""Untimed native scalar readout of an actual retained FOM invocation; never solve."""
from pathlib import Path
import time
import numpy as np
from pilot2d import save_evidence,array_digest


def readout_control(candidate,invocation,construction,output,deadline=None):
    query=candidate.native;ng=query.ng;contract=invocation['output_contract'];started=time.perf_counter()
    retained=ng.GridFunction(query.X);state=invocation['evidence']['state'];retained.vec.FV().NumPy()[:]=state
    u,p=retained.components;U=query.cfg['U'];D=query.cfg['D'];Re=invocation['evidence']['Re']
    field=query.evaluators['full_velocity_pressure_fields'];_,owners,probes,volume,boundary=field.native_owners
    coordinate=ng.CF((ng.x,ng.y));raw={};result={'passed':False,'state_sha256':array_digest({'state':state}),
        'output_contract':contract,'Re':Re,'no_PDE_solve':True,'tolerance':1e-10,'method':'original scalar CF evaluation on retained native transformations/integration rules'}
    def guard(index):
        if index%4096==0 and deadline is not None and time.perf_counter()>=deadline:raise TimeoutError('original calibration budget exhausted in readout control')
    def mapped(region):
        phase=0
        for owner in owners:
            if isinstance(owner,tuple) and len(owner)==3:
                if phase==region:
                    tr,ir,_=owner
                    for ip in ir:yield tr(ip)
            else:phase+=1
    try:
        if contract=='full_velocity_pressure_fields':
            n=len(candidate.points);velocity=np.full((n,2),np.nan);pressure=np.full(n,np.nan);points=np.full((n,2),np.nan)
            raw.update(velocity=velocity,pressure=pressure,points=points)
            if candidate.release['field_rule_mode']=='certified_requested_points':
                with np.load(Path(construction)/'normal_orientation.mapping.batch.npz',allow_pickle=False) as saved:order=saved['native_to_requested'].copy()
                if order.shape!=(n,) or not np.array_equal(np.sort(order),np.arange(n)):raise ValueError('complete requested/native permutation required')
            else:order=np.arange(n)
            count=0
            for index,mip in enumerate(mapped(0)):
                guard(index);target=int(order[index]);points[target]=coordinate(mip)
                velocity[target]=np.asarray(u(mip))/U;pressure[target]=np.asarray(p(mip)).reshape(()).item()/U**2;count+=1
            if count!=n or not np.isfinite(points).all():raise ValueError('scalar full-rule coverage incomplete')
            result['coordinate_max_absolute_gap']=float(np.max(np.abs(points-candidate.points)))
            if result['coordinate_max_absolute_gap']>1e-10*D:raise ArithmeticError('scalar physical requested-point order differs')
            expected={'velocity':velocity,'pressure':pressure}
        elif contract=='fixed_probes':
            expected={'velocity':np.array([u(m) for m in probes])/U,
                      'pressure':np.array([np.asarray(p(m)).reshape(()).item() for m in probes])/U**2}
            raw.update(expected)
        elif contract=='integrated_forces':
            diagnostics=query.diagnostics_from_evidence(invocation['evidence']);n=len(boundary)
            gradient=np.full((n,2,2),np.nan);pressure=np.full(n,np.nan);raw.update(boundary_velocity_gradient=gradient,boundary_pressure=pressure)
            grad=ng.BoundaryFromVolumeCF(ng.Grad(u));count=0
            for index,mip in enumerate(mapped(1)):
                guard(index);gradient[index]=np.asarray(grad(mip)).reshape(2,2)*(D/U)
                pressure[index]=np.asarray(p(mip)).reshape(()).item()/U**2;count+=1
            if count!=n:raise ValueError('scalar boundary coverage incomplete')
            mask=np.asarray(diagnostics['boundary_labels'])=='obstacle';normals=diagnostics['fluid_normals'];weights=diagnostics['boundary_weights']
            traction=pressure[:,None]*normals-np.einsum('nij,nj->ni',gradient+gradient.swapaxes(1,2),normals)/Re
            force=2*np.einsum('n,ni->i',weights[mask],traction[mask]);expected={'force_coefficient':force}
            raw.update(normals_fluid=normals,weights_dimensionless=weights,obstacle_mask=mask,force_coefficient=force)
        else:raise ValueError('unregistered output contract')
        gaps={}
        for key,value in expected.items():
            actual=invocation['outputs'][key]
            if actual.shape!=value.shape:raise ValueError('actual output shape differs')
            gaps[key]=float(np.linalg.norm(actual-value)/max(float(np.linalg.norm(value)),1e-12))
        result.update(relative_gaps=gaps,passed=bool(all(np.isfinite(v) and v<=1e-10 for v in gaps.values())))
    except Exception as exc:result['exception']=repr(exc)
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        receipt=save_evidence(output,'scalar_readout',{'control':result,'actual_scalar_arrays':raw})
    return dict(result,evidence=receipt)
