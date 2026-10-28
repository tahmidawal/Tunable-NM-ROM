"""Source-only batched FOM outputs and offline exact force contraction.

Native equivalence is required before timed use. Mapping, certification and force
basis assembly are offline. Online outputs use cached native batches, two cached
probe points, or sparse boundary-supported linear coefficient contractions.
"""
from pathlib import Path
import time
import json
import numpy as np
from oriented_normals import fluid_outward_normal
from scalars import scalar_value


def _f64(value,shape,label):
    a=np.asarray(value)
    if a.dtype!=np.dtype('float64') or a.size!=int(np.prod(shape)) or not np.isfinite(a).all():
        raise ValueError('native finite float64 '+label+' shape/dtype mismatch')
    return a.reshape(shape)


def make_output_factory(ng,expected_volume,expected_boundary,normal_evidence_path,*,field_rule_mode="native_exact",requested_point_cap=1300000):
    if field_rule_mode not in ("native_exact","certified_requested_points"):raise ValueError("unknown field rule mode")
    if type(requested_point_cap) is not int or not 0<requested_point_cap<=3000000:raise ValueError("invalid requested-point cap")
    def factory(mesh,V,Q,X,cfg):
        if mesh.dim!=2 or cfg['kind']!='dfg' or cfg['outlet']!='gradient':
            raise ValueError('fixed unforced DFG family required')
        offline_started=time.perf_counter()
        D,U=cfg['D'],cfg['U'];order=cfg['diagnostic_order']
        normal,certificate=fluid_outward_normal(mesh,cfg['center'],order=order,evidence_path=Path(normal_evidence_path))
        if certificate.get('passed') is not True:raise ValueError('normal certificate failed')
        coordinate=ng.CF((ng.x,ng.y));owners=[]
        def cache(boundary=False):
            batches=[];points=[];weights=[];labels=[];normals=[]
            for element in mesh.Elements(ng.BND if boundary else ng.VOL):
                tr=mesh.GetTrafo(element);ir=ng.IntegrationRule(element.type,order)
                batch=tr(ir);owners.append((tr,ir,batch));batches.append(batch)
                for ip in ir:
                    mip=tr(ip)
                    points.append(np.array(mip.point,dtype=np.float64,copy=True))
                    weights.append(scalar_value(ip.weight,'quadrature weight')*scalar_value(mip.measure,'mapped measure'))
                    if boundary:
                        labels.append(str(element.mat));normals.append(np.array(normal(mip),dtype=np.float64,copy=True))
            array=np.concatenate(batches)
            points=np.asarray(points,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
            batched_coordinates=_f64(coordinate(array),points.shape,'mapped coordinates')
            if not np.allclose(batched_coordinates,points,rtol=0,atol=1e-12*D):
                raise ValueError('batched and scalar mapped coordinates differ')
            owners.append(array)
            return array,points,weights,np.asarray(labels),np.asarray(normals)
        mapping_evidence={'mode':field_rule_mode,'physical_coordinates_changed':False,'points_dropped':0}
        if field_rule_mode=='native_exact':
            volume,vpoints,vweights,_,_=cache()
            for actual,key in ((vpoints,'points'),(vweights,'weights')):
                expected=np.asarray(expected_volume[key])
                if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=1e-12,atol=1e-14):
                    raise ValueError('full output rule differs from admitted '+key)
            requested_order=None
        else:
            # Coarser eligible FOM meshes still return every fine-reference output
            # coordinate. Inverse mapping and admissibility checks are offline.
            from g2.exact_lookup import ExactPointLocator
            vpoints=np.asarray(expected_volume['points']);vweights=np.asarray(expected_volume['weights'])
            count=len(vweights)
            if count==0 or count>requested_point_cap or vpoints.shape!=(count,2) or vweights.shape!=(count,):
                raise ValueError('requested full rule shape/cap mismatch')
            _f64(vpoints,(count,2),'requested points');_f64(vweights,(count,),'requested weights')
            if np.any(vweights<=0):raise ValueError('requested weights must remain positive')
            locator=ExactPointLocator(mesh,ng,D);groups={};references=np.full((count,3),np.nan);mapped=np.full((count,2),np.nan);element_numbers=np.full(count,-1,dtype=np.int64);fallbacks=[];invalid=[]
            for index,xy in enumerate(vpoints):
                try:
                    mip,point_owners,certificate,fallback=locator.locate(xy)
                    nr=int(certificate['element_number']);element_numbers[index]=nr
                    references[index]=certificate['reference_coordinates'];mapped[index]=np.array(coordinate(mip),copy=True)
                    groups.setdefault(nr,[]).append(index)
                    if fallback is not None:fallbacks.append({'index':index,**fallback})
                except Exception as exc:
                    invalid.append({'index':index,'requested_point':xy.tolist(),'error':repr(exc)})
                    if hasattr(exc,'lookup_evidence'):fallbacks.append({'index':index,**exc.lookup_evidence})
            evidence_path=Path(normal_evidence_path).with_suffix('.mapping.npz')
            np.savez_compressed(evidence_path,requested_points=vpoints,weights=vweights,mapped_points=mapped,reference_coordinates=references,element_numbers=element_numbers)
            mapping_evidence.update(point_count=count,fallbacks=fallbacks,invalid=invalid,weights_unchanged=True,source_mesh_cells=int(mesh.ne))
            evidence_path.with_suffix('.json').write_text(json.dumps(mapping_evidence,indent=2)+'\n')
            if invalid:raise ValueError('requested full output rule has invalid source-mesh points; original diagnostics retained')
            batches=[];order=[];elements={int(el.nr):el for el in mesh.Elements(ng.VOL)}
            for nr,indices in sorted(groups.items()):
                ir=ng.IntegrationRule([tuple(x) for x in references[indices,:2]],[1.]*len(indices));tr=mesh.GetTrafo(elements[nr]);batch=tr(ir)
                owners.append((tr,ir,batch));batches.append(batch);order.extend(indices)
            volume=np.concatenate(batches);native_to_requested=np.asarray(order,dtype=np.int64);requested_order=np.argsort(native_to_requested);owners.append(volume)
            batched_coordinates=_f64(coordinate(volume),(count,2),'requested mapped batch')
            errors=np.linalg.norm(batched_coordinates-vpoints[native_to_requested],axis=1)
            if float(errors.max())>1e-10*D:raise ValueError('requested native batch reconstruction failed')
            mapping_evidence.update(point_count=count,max_reconstruction_error=float(errors.max()),fallbacks=fallbacks,
                source_mesh_cells=int(mesh.ne),weights_unchanged=True)
            np.savez_compressed(evidence_path.with_name(evidence_path.stem+'.batch.npz'),native_to_requested=native_to_requested,reconstruction_errors=errors)
            evidence_path.with_suffix('.json').write_text(json.dumps(mapping_evidence,indent=2)+'\n')
        if len(vpoints)>requested_point_cap:raise MemoryError('field output rule exceeds explicit cap')
        boundary,bpoints,bweights,labels,normals=cache(True)
        if labels.tolist()!=[r['label'] for r in expected_boundary]:raise ValueError('boundary labels differ')
        for actual,key in ((bpoints,'point'),(bweights,'weight'),(normals,'normal')):
            expected=np.asarray([r[key] for r in expected_boundary])
            if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=1e-12,atol=1e-14):
                raise ValueError('boundary output rule differs from admitted '+key)
        # Cache exact pressure poles; no repeated point search inside query timing.
        poles=[(cfg['center'][0]-cfg['radius'],cfg['center'][1]),(cfg['center'][0]+cfg['radius'],cfg['center'][1])]
        probe_points=[]
        for xy in poles:
            mip=mesh(*xy,tol=1e-12)
            if mip.nr<0:raise ValueError('exact pole outside query mesh')
            ref=np.asarray(mip.pnt,dtype=np.float64);bary=np.array([ref[0],ref[1],1-ref[0]-ref[1]])
            if np.min(bary)<-64*np.finfo(float).eps or abs(ref[2])>64*np.finfo(float).eps:
                raise ValueError('exact pole not certified inside element')
            if np.linalg.norm(np.asarray(coordinate(mip))-xy)>1e-10*D:raise ValueError('exact pole reconstruction failed')
            probe_points.append(mip)
        mapping_seconds=time.perf_counter()-offline_started
        force_started=time.perf_counter()
        # Include all volume shape functions adjoining an obstacle edge: even an
        # interior bubble with zero trace can have nonzero boundary gradient.
        edge_elements={}
        for element in mesh.Elements(ng.VOL):
            vertices=tuple(int(v.nr) for v in element.vertices)
            if len(vertices)!=3:raise ValueError('triangle-only force contraction')
            for i in range(3):edge_elements.setdefault(tuple(sorted((vertices[i],vertices[(i+1)%3]))),[]).append(element)
        velocity_ids=set();pressure_ids=set()
        for element in mesh.Elements(ng.BND):
            if str(element.mat)!='obstacle':continue
            edge=tuple(sorted(int(v.nr) for v in element.vertices));adjacent=edge_elements.get(edge,[])
            if len(adjacent)!=1:raise ValueError('obstacle boundary needs one adjacent fluid triangle')
            velocity_ids.update(int(i) for i in V.GetDofNrs(adjacent[0]) if i>=0)
            pressure_ids.update(int(i) for i in Q.GetDofNrs(adjacent[0]) if i>=0)
        velocity_ids=np.array(sorted(velocity_ids),dtype=np.int64);pressure_ids=np.array(sorted(pressure_ids),dtype=np.int64)
        if not len(velocity_ids) or not len(pressure_ids):raise ValueError('empty obstacle force support')
        obstacle=labels=='obstacle';obstacle_points=boundary[obstacle];owners.append(obstacle_points)
        w=bweights[obstacle];n=normals[obstacle];normalization=2/(U**2*D)
        vbasis=ng.GridFunction(V);pbasis=ng.GridFunction(Q)
        va=vbasis.vec.FV().NumPy();pa=pbasis.vec.FV().NumPy();va[:]=0;pa[:]=0
        derivative=ng.BoundaryFromVolumeCF(ng.Grad(vbasis))
        velocity_force=np.empty((2,len(velocity_ids)));pressure_force=np.empty((2,len(pressure_ids)))
        previous=None
        for column,index in enumerate(velocity_ids):
            if previous is not None:va[previous]=0
            va[index]=1;previous=index
            gradient=_f64(derivative(obstacle_points),(len(w),2,2),'boundary gradient')
            velocity_force[:,column]=normalization*np.einsum('n,nij,nj->i',w,gradient+gradient.swapaxes(1,2),-n)
        previous=None
        for column,index in enumerate(pressure_ids):
            if previous is not None:pa[previous]=0
            pa[index]=1;previous=index
            pressure=_f64(pbasis(obstacle_points),(len(w),),'boundary pressure')
            pressure_force[:,column]=normalization*np.einsum('n,n,ni->i',w,pressure,n)
        va[:]=0;pa[:]=0
        force_assembly_seconds=time.perf_counter()-force_started
        def fields(u,p,nu):
            velocity=_f64(u(volume),(len(vpoints),2),'velocity');pressure=_f64(p(volume),(len(vpoints),),'pressure')
            if requested_order is not None:velocity=velocity[requested_order];pressure=pressure[requested_order]
            return {'velocity':velocity/U,'pressure':pressure/U**2}
        def probes(u,p,nu):
            return {'velocity':np.array([np.asarray(u(m)) for m in probe_points],dtype=np.float64)/U,
                    'pressure':np.array([scalar_value(p(m),'pole pressure') for m in probe_points],dtype=np.float64)/U**2}
        def forces(u,p,nu):
            if not np.isfinite(nu) or nu<=0:raise ValueError('actual positive viscosity required')
            return {'force_coefficient':nu*(velocity_force@u.vec.FV().NumPy()[velocity_ids])+
                    pressure_force@p.vec.FV().NumPy()[pressure_ids]}
        def diagnostics(u,p,nu):
            # Untimed scorer receives the exact retained invocation coefficients,
            # bound to this mesh/space; this callback never solves or warm-starts.
            field_values=fields(u,p,nu)
            gradient=_f64(ng.Grad(u)(volume),(len(vpoints),2,2),'diagnostic volume gradient')
            if requested_order is not None:gradient=gradient[requested_order]
            boundary_gradient=_f64(ng.BoundaryFromVolumeCF(ng.Grad(u))(boundary),(len(bpoints),2,2),'diagnostic boundary gradient')
            return {'velocity':field_values['velocity'],'pressure':field_values['pressure'],
                'velocity_gradient':gradient*(D/U),'volume_points':vpoints.copy(),'volume_weights':vweights.copy()/D**2,
                'boundary_velocity':_f64(u(boundary),(len(bpoints),2),'diagnostic boundary velocity')/U,
                'boundary_pressure':_f64(p(boundary),(len(bpoints),),'diagnostic boundary pressure')/U**2,
                'boundary_velocity_gradient':boundary_gradient*(D/U),'boundary_labels':labels.copy(),
                'boundary_points':bpoints.copy(),'boundary_weights':bweights.copy()/D,'fluid_normals':normals.copy(),
                'force_coefficient':forces(u,p,nu)['force_coefficient'],
                'units':{'velocity':'u/U','pressure':'p/U^2','velocity_gradient':'d(u/U)/d(x/D)',
                    'volume_weights':'dA/D^2','boundary_weights':'ds/D','points':'physical coordinates'},
                'diagnostic_scope':'untimed evaluation of same retained physical state; no new solve'}
        fields.diagnostics=diagnostics
        # Closure keeps all transformation, integration-rule and mesh objects alive.
        keep_alive=(mesh,owners,probe_points,volume,boundary)
        for function in (fields,probes,forces,diagnostics):function.native_owners=keep_alive
        forces.offline_evidence={'mapping_certification_seconds':mapping_seconds,'force_assembly_seconds':force_assembly_seconds,
            'offline_total_seconds':time.perf_counter()-offline_started,'timing_claim':False,'field_rule_mode':field_rule_mode,
            'requested_point_cap':requested_point_cap,'requested_point_count':len(vpoints)}
        fields.mapping_evidence=mapping_evidence
        forces.contraction_evidence={'velocity_ids':velocity_ids,'pressure_ids':pressure_ids,
            'velocity_force_per_nu':velocity_force,'pressure_force':pressure_force,
            'normal_definition':'verified fluid outward; force on solid = -sigma n',
            'offline_support':'all velocity/pressure DOFs on volume triangles adjacent to obstacle edges'}
        return {'full_velocity_pressure_fields':fields,'fixed_probes':probes,'integrated_forces':forces}
    return factory
