"""Three predeclared geometry processes. No PDE, normal repair or old-mesh claim."""
from __future__ import annotations
import argparse
import faulthandler
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
from geometry_math import (TET_VERTICES,TET_FACES,TET_NORMALS,LABELS,BINS,TERMS,
    VOLUME_DTYPE,FACET_DTYPE,BOUNDARY_DTYPE,determinant,scaled_determinant,moments,
    cofactor_normal,facet_reference,boundary_reference,projection)
from storage import Budget,Chunks,sha,encode


def copied(value,shape):
    return np.array(value,dtype=np.float64,copy=True).reshape(shape)


def scalar(value):
    result=float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite(result):raise ArithmeticError('nonfinite native scalar')
    return result


def announce(message):print(json.dumps({'phase':message}),flush=True)


def load_mesh(path,ng):
    # The explicitly verified Netgen Python Load binding handles *.vol.bin.
    from netgen.meshing import Mesh
    native=Mesh(dim=3)
    native.Load(str(path))
    return ng.Mesh(native)


def regenerate(cfg,ng):
    from netgen.csg import CSGeometry,Cylinder,Plane,Pnt,Sphere,Vec
    g,m=cfg['geometry'],cfg['mesh']
    pipe=Cylinder(Pnt(0,0,0),Pnt(1,0,0),g['pipe_radius']).bc('wall').maxh(m['wall_maxh'])
    left=Plane(Pnt(g['x_in'],0,0),Vec(-1,0,0)).bc('inlet')
    right=Plane(Pnt(g['x_out'],0,0),Vec(1,0,0)).bc('outlet')
    sphere=Sphere(Pnt(0,0,0),g['sphere_diameter']/2).bc('sphere').maxh(m['sphere_maxh'])
    geo=CSGeometry();geo.Add((pipe*left*right-sphere).mat('fluid'))
    mesh=ng.Mesh(geo.GenerateMesh(maxh=m['maxh']))
    mesh.Curve(m['geometry_order'])
    return mesh


def topology(mesh,ng):
    elements=list(mesh.Elements(ng.VOL));boundaries=list(mesh.Elements(ng.BND))
    if [int(e.nr) for e in elements]!=list(range(mesh.ne)):
        raise ValueError('volume enumeration is not the saved dense ordering')
    if set(mesh.GetBoundaries())!=set(LABELS):raise ValueError('unexpected labels')
    faces={};vertices=[]
    for e in elements:
        v=tuple(int(x.nr) for x in e.vertices);vertices.append(v)
        if e.type!=ng.ET.TET or len(v)!=4:raise ValueError('non-tetrahedron')
        for lf,inds in enumerate(TET_FACES):
            key=tuple(sorted(v[i] for i in inds));fid=int(e.faces[lf].nr)
            row=faces.setdefault(fid,{'vertices':key,'adjacency':[],'boundary':None})
            if tuple(row['vertices'])!=key:raise ValueError('local/native face topology mismatch')
            row['adjacency'].append((int(e.nr),lf))
    by_vertices={tuple(row['vertices']):fid for fid,row in faces.items()}
    boundary_by_id={}
    for e in boundaries:
        if e.type!=ng.ET.TRIG:raise ValueError('non-triangular boundary')
        v=tuple(int(x.nr) for x in e.vertices)
        fid=by_vertices[tuple(sorted(v))];row=faces[fid]
        if len(row['adjacency'])!=1 or row['boundary'] is not None:
            raise ValueError('boundary does not have exactly one adjacent tetrahedron')
        row['boundary']={'element':int(e.nr),'vertices':v,'label':str(e.mat)}
        boundary_by_id[int(e.nr)]=(e,fid)
    for row in faces.values():
        if len(row['adjacency']) not in (1,2):raise ValueError('non-manifold face')
        if (len(row['adjacency'])==1)!=(row['boundary'] is not None):
            raise ValueError('missing exterior boundary')
    record={'elements':int(mesh.ne),'vertices':int(mesh.nv),
            'boundaries':len(boundaries),'curve_order':int(mesh.GetCurveOrder()),
            'volume_vertices':vertices,'faces':faces}
    return elements,boundaries,vertices,faces,boundary_by_id,record


def observed(mesh,ng,t,folder,budget):
    """Same scalar mapped-point API/order used for the retained observations."""
    elements,boundaries,vertices,faces,by_id,record=t
    vr=ng.IntegrationRule(ng.ET.TET,4);br=ng.IntegrationRule(ng.ET.TRIG,4)
    v=np.zeros(len(elements)*len(vr),dtype=VOLUME_DTYPE)
    b=np.zeros(len(boundaries)*len(br),dtype=BOUNDARY_DTYPE)
    normal=ng.specialcf.normal(3)
    i=0
    for e in elements:
        trafo=mesh.GetTrafo(e)
        for pid,ip in enumerate(vr):
            mip=trafo(ip);j=copied(mip.jacobi.NumPy(),(3,3))
            det=float(np.linalg.det(j));scaled,_=scaled_determinant(j,np.asarray(det))
            v[i]['element']=int(e.nr);v[i]['point_id']=pid;v[i]['reference']=ip.point
            v[i]['point']=copied(mip.point,(3,));v[i]['jacobian']=j
            v[i]['reference_weight']=ip.weight;v[i]['determinant']=det
            v[i]['determinant_from_J']=determinant(j);v[i]['scaled_determinant']=scaled
            v[i]['physical_weight']=ip.weight*float(mip.measure)
            v[i]['reference_weight_sign']=np.sign(ip.weight);v[i]['determinant_sign']=np.sign(det)
            i+=1
    budget.arrays(folder/'observed-volume.npz',points=len(v),records=v)
    i=0
    for e in boundaries:
        trafo=mesh.GetTrafo(e)
        for pid,ip in enumerate(br):
            mip=trafo(ip)
            b[i]['element']=int(e.nr);b[i]['point_id']=pid;b[i]['face']=by_id[int(e.nr)][1]
            b[i]['label']=LABELS.index(str(e.mat));b[i]['reference']=ip.point[:2]
            b[i]['point']=copied(mip.point,(3,));b[i]['jacobian']=copied(mip.jacobi.NumPy(),(3,2))
            b[i]['normal']=copied(normal(mip),(3,));b[i]['reference_weight']=ip.weight
            b[i]['measure']=mip.measure;b[i]['physical_weight']=ip.weight*float(mip.measure)
            b[i]['reference_weight_sign']=np.sign(ip.weight);i+=1
    budget.arrays(folder/'observed-boundary.npz',points=len(b),records=b)
    return v,b


def compare_array(actual,expected,p):
    actual=np.asarray(actual);expected=np.asarray(expected)
    if actual.shape!=expected.shape:return {'pass':False,'actual_shape':actual.shape,'expected_shape':expected.shape}
    finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:return {'pass':False,'nonfinite':True}
    error=np.abs(actual-expected);limit=p['parity_atol']+p['parity_rtol']*np.abs(expected)
    return {'pass':bool(np.all(error<=limit)),'max_absolute_error':float(np.max(error,initial=0.)),
            'max_tolerance_ratio':float(np.max(error/limit,initial=0.))}


def parity(phase,v,b,t,root,policy,budget,folder):
    checks={}
    if phase=='A':
        with np.load(policy['old_volume'],allow_pickle=False) as old:
            for field,key in [('point','points'),('physical_weight','weights'),('determinant','determinants')]:
                checks['volume_'+field]=compare_array(v[field],old[key],policy)
        old=json.loads(Path(policy['old_boundary']).read_text())
        for field,key in [('element','element'),('point_id','point_in_element'),('point','point'),
                          ('normal','normal'),('jacobian','boundary_jacobian'),('physical_weight','weight')]:
            checks['boundary_'+field]=compare_array(b[field],[row[key] for row in old],policy)
        old_record=json.loads(Path(policy['old_mesh_record']).read_text())
        checks['mesh_counts_and_curve_order']={'pass':all(t[5][k]==old_record[k] for k in ('elements','vertices','curve_order'))}
    else:
        for domain,rows in [('volume',v),('boundary',b)]:
            with np.load(root/'B'/f'observed-{domain}.npz',allow_pickle=False) as old:
                for field in rows.dtype.names:
                    checks[domain+'_'+field]=compare_array(rows[field],old['records'][field],policy)
        checks['complete_recorded_topology']={'pass':t[5]==json.loads((root/'B'/'topology.json').read_text())}
        # JSON normalizes tuples and numeric dictionary keys. Compare canonical bytes below.
        checks['complete_recorded_topology']['pass']=encode(t[5])==encode(json.loads((root/'B'/'topology.json').read_text()))
    result={'phase':phase,'strict_saved_observation_parity_passed':all(r['pass'] for r in checks.values()),
        'checks':checks,'atol':policy['parity_atol'],'rtol':policy['parity_rtol'],
        'claim':'sampled mapped-observation parity only; no global mapping validity or original-mesh causal claim',
        'binary_format_admitted_for_G2':False,'mesh_changed_or_recurved':False}
    budget.json(folder/'parity.json',result)
    return result


def ordinary(mesh,ng):
    n=ng.specialcf.normal(3);x=ng.CF((ng.x,ng.y,ng.z))
    terms=(ng.CF(1),n[0],n[1],n[2],ng.InnerProduct(x,n))
    result={}
    for q in (4,12,16):
        result[str(q)]={label:{name:scalar(ng.Integrate(cf,mesh,ng.BND,
            definedon=mesh.Boundaries(label),order=q)) for name,cf in zip(TERMS,terms)} for label in LABELS}
    return result


def native_controls(mesh,ng,elements,folder,budget):
    V=ng.L2(mesh,order=0,dgjumps=True);test=V.TestFunction()
    one=ng.GridFunction(V);one.Set(ng.CF(1))
    coefficients=copied(one.vec.FV().NumPy(),(V.ndof,))
    dofs=[];error=0.
    for e in elements:
        ids=V.GetDofNrs(ng.ElementId(ng.VOL,int(e.nr)))
        if len(ids)!=1:raise ValueError('P0 element does not have one degree of freedom')
        dofs.append(int(ids[0]));trafo=mesh.GetTrafo(e)
        point=trafo(ng.IntegrationRule(ng.ET.TET,0)[0])
        error=max(error,abs(scalar(one(point))-1.))
    if V.ndof!=mesh.ne or len(set(dofs))!=mesh.ne or error>1e-12:
        raise ArithmeticError('P0 constant-test or degree-of-freedom check failed')
    dofs=np.asarray(dofs,dtype=np.int32)
    budget.arrays(folder/'constant-test.npz',coefficients=coefficients,dofs=dofs)
    j=ng.specialcf.JacobianMatrix(3)
    det=(j[0,0]*(j[1,1]*j[2,2]-j[1,2]*j[2,1])
         -j[0,1]*(j[1,0]*j[2,2]-j[1,2]*j[2,0])
         +j[0,2]*(j[1,0]*j[2,1]-j[1,1]*j[2,0]))
    denominator=ng.sqrt(sum(j[k,0]*j[k,0] for k in range(3)))*ng.sqrt(sum(j[k,1]*j[k,1] for k in range(3)))*ng.sqrt(sum(j[k,2]*j[k,2] for k in range(3)))
    pos=ng.IfPos(det,1,0);neg=ng.IfPos(-det,1,0);zero=1-pos-neg
    near=1-ng.IfPos(ng.IfPos(det,det,-det)-budget.policy['near_zero_scaled_cutoff']*denominator,1,0)
    indicators=(ng.CF(1),neg,zero,pos,near)
    normal=ng.specialcf.normal(3);coords=ng.CF((ng.x,ng.y,ng.z))
    terms=(ng.CF(1),normal[0],normal[1],normal[2],ng.InnerProduct(coords,normal))
    matrices={};summary={'constant_test_error':error,'paths':{},'native_vector_semantics':'P0 coefficient times assembled vector, indexed by volume element'}
    for path in (*LABELS,'all'):
        data=np.zeros((len(BINS),len(TERMS),mesh.ne))
        for bi,indicator in enumerate(indicators):
            announce('native_'+path+'_'+BINS[bi])
            for ti,term in enumerate(terms):
                measure=(ng.dx(element_boundary=True,bonus_intorder=12) if path=='all'
                         else ng.ds(path,skeleton=True,bonus_intorder=12))
                form=ng.LinearForm(V);form+=indicator*term*test*measure;form.Assemble()
                vector=copied(form.vec.FV().NumPy(),(V.ndof,))
                data[bi,ti]=coefficients[dofs]*vector[dofs]
            budget.arrays(folder/f'native-{path}.npz',element_moments=data)
        matrices[path]=data;summary['paths'][path]=data.sum(axis=-1).tolist()
    summary['positive_volume']=scalar(ng.Integrate(ng.CF(1),mesh,order=12))
    summary['signed_volume']=scalar(ng.Integrate(pos-neg,mesh,order=12))
    summary['normal_sign_correction_applied']=False
    budget.json(folder/'native.json',summary)
    packed=ng.CF((ng.x,ng.y,ng.z,*(j[i,k] for i in range(3) for k in range(3)),det))
    facet_packed=ng.CF((ng.x,ng.y,ng.z,*(j[i,k] for i in range(3) for k in range(3)),det,*normal))
    return matrices,summary,packed,facet_packed


def native_points(ng,trafo,reference,weights,packed,local_face=None):
    ir=ng.IntegrationRule(points=[tuple(row) for row in reference],weights=weights.tolist())
    mapped=trafo(ir)
    # Public structured MeshPoint fields; the binding calls SetFacetNr(facetnr,eb).
    mapped['facetnr']=-1 if local_face is None else local_face
    mapped['eb']=int(ng.VOL if local_face is None else ng.BND)
    return copied(packed(mapped),(len(reference),13 if local_face is None else 16))


def volume_rows(element,reference,weights,values,dtype=VOLUME_DTYPE):
    rows=np.zeros(len(reference),dtype=dtype)
    rows['element']=element;rows['point_id']=np.arange(len(reference));rows['reference']=reference
    rows['point']=values[:,:3];rows['jacobian']=values[:,3:12].reshape(-1,3,3)
    rows['reference_weight']=weights;rows['determinant']=values[:,12]
    rows['determinant_from_J']=determinant(rows['jacobian'])
    rows['scaled_determinant'],denominator=scaled_determinant(rows['jacobian'],rows['determinant'])
    rows['physical_weight']=weights*np.abs(rows['determinant'])
    rows['reference_weight_sign']=np.sign(weights);rows['determinant_sign']=np.sign(rows['determinant'])
    return rows,denominator


def traces(mesh,ng,t,folder,budget,native,packed,facet_packed):
    elements,boundaries,vertices,faces,by_id,record=t
    vr=ng.IntegrationRule(ng.ET.TET,12);fr=ng.IntegrationRule(ng.ET.TRIG,12)
    vref=np.asarray(vr.points)[:,:3];vw=np.asarray(vr.weights)
    fref=np.asarray(fr.points)[:,:2];fw=np.asarray(fr.weights)
    if len(vr)!=343 or len(fr)!=49:raise ValueError('pinned integration-rule count changed')
    budget.arrays(folder/'exact-rules.npz',volume_reference=vref,volume_weights=vw,face_reference=fref,face_weights=fw)
    trace={path:np.zeros_like(values) for path,values in native.items()}
    volume_sum={'signed':0.,'absolute':0.,'negative_reference_weights':0,'negative_determinants':0,
                'min_determinant':None,'max_determinant':None}
    with Chunks(budget,folder/'volume-points',VOLUME_DTYPE) as writer:
        for e in elements:
            trafo=mesh.GetTrafo(e)
            values=native_points(ng,trafo,vref,vw,packed)
            rows,_=volume_rows(int(e.nr),vref,vw,values);writer.append(rows)
            if not np.isfinite(values).all():raise ArithmeticError('nonfinite volume trace retained')
            d=rows['determinant'];volume_sum['signed']+=float(vw@d);volume_sum['absolute']+=float(vw@np.abs(d))
            volume_sum['negative_reference_weights']+=int(np.count_nonzero(vw<0))
            volume_sum['negative_determinants']+=int(np.count_nonzero(d<0))
            volume_sum['min_determinant']=float(d.min()) if volume_sum['min_determinant'] is None else min(volume_sum['min_determinant'],float(d.min()))
            volume_sum['max_determinant']=float(d.max()) if volume_sum['max_determinant'] is None else max(volume_sum['max_determinant'],float(d.max()))
    budget.json(folder/'volume-trace-summary.json',volume_sum)
    normal_cf=ng.specialcf.normal(3)
    pairs=[];face_summary=[]
    with Chunks(budget,folder/'default-facet-points',FACET_DTYPE) as default_writer, \
         Chunks(budget,folder/'sorted-facet-points',FACET_DTYPE) as sorted_writer, \
         Chunks(budget,folder/'ordinary-matched-points',BOUNDARY_DTYPE) as boundary_writer:
        for fid,face in sorted(faces.items()):
            common=[];defaults=[]
            for sorted_mode,writer in ((False,default_writer),(True,sorted_writer)):
                for eid,lf in face['adjacency']:
                    trafo=mesh.GetTrafo(elements[eid])
                    refs,bary=facet_reference(lf,vertices[eid],fref,sorted_mode)
                    values=native_points(ng,trafo,refs,fw,facet_packed,lf)
                    rows,denominator=volume_rows(eid,refs,fw,values,FACET_DTYPE)
                    conormal,measure,cof_normal=cofactor_normal(rows['jacobian'],lf,rows['determinant'])
                    rows['face']=fid;rows['local_face']=lf;rows['triangle_reference']=fref
                    rows['label']=-1 if face['boundary'] is None else LABELS.index(face['boundary']['label'])
                    rows['measure']=measure;rows['normal']=values[:,13:16];rows['cofactor_normal']=conormal
                    rows['physical_weight']=fw*measure
                    rows['native_normal_error']=np.linalg.norm(rows['normal']-cof_normal,axis=1)
                    offset=writer.append(rows)
                    if not np.isfinite(values).all() or not np.isfinite(cof_normal).all():
                        raise ArithmeticError('nonfinite facet trace retained')
                    contribution=moments(rows['point'],rows['normal'],rows['physical_weight'],
                        rows['determinant'],denominator,budget.policy['near_zero_scaled_cutoff'])
                    path='all' if not sorted_mode else (face['boundary']['label'] if face['boundary'] else None)
                    if path is not None:trace[path][:,:,eid]+=contribution
                    summary={'face':fid,'element':eid,'local_face':lf,'sorted_vertices':sorted_mode,
                        'stream_offset':offset,'rows':len(rows),'min_detJ':float(rows['determinant'].min()),
                        'max_detJ':float(rows['determinant'].max()),'negative_detJ_points':int(np.count_nonzero(rows['determinant']<0)),
                        'negative_reference_weights':int(np.count_nonzero(fw<0)),
                        'max_native_vs_cofactor_normal_error':float(rows['native_normal_error'].max()),
                        'moments_by_bin':contribution.tolist()}
                    if sorted_mode:common.append((rows.copy(),summary))
                    else:defaults.append(summary)
                    if face['boundary'] is not None:
                        be,_=by_id[face['boundary']['element']];btrafo=mesh.GetTrafo(be)
                        bref=boundary_reference(bary,vertices[eid],face['boundary']['vertices'])
                        brule=ng.IntegrationRule(points=[tuple(row) for row in bref],weights=fw.tolist())
                        brows=np.zeros(len(fw),dtype=BOUNDARY_DTYPE)
                        for ipid,ip in enumerate(brule):
                            mip=btrafo(ip);r=brows[ipid]
                            r['element']=int(be.nr);r['point_id']=ipid;r['face']=fid;r['label']=rows['label'][ipid]
                            r['reference']=bref[ipid];r['point']=copied(mip.point,(3,));r['jacobian']=copied(mip.jacobi.NumPy(),(3,2))
                            r['normal']=copied(normal_cf(mip),(3,));r['reference_weight']=fw[ipid]
                            r['measure']=mip.measure;r['physical_weight']=fw[ipid]*float(mip.measure)
                            r['reference_weight_sign']=np.sign(fw[ipid])
                        summary['matched_boundary_stream_offset']=boundary_writer.append(brows)
                        summary['boundary_max_point_difference']=float(np.max(np.linalg.norm(rows['point']-brows['point'],axis=1)))
                        summary['boundary_max_measure_difference']=float(np.max(np.abs(rows['measure']-brows['measure'])))
                        summary['boundary_normal_dot_native_range']=[float(x) for x in (np.min(np.sum(rows['normal']*brows['normal'],axis=1)),np.max(np.sum(rows['normal']*brows['normal'],axis=1)))]
                        summary['ordinary_moments_by_volume_detJ_bin']=moments(brows['point'],brows['normal'],brows['physical_weight'],rows['determinant'],denominator,budget.policy['near_zero_scaled_cutoff']).tolist()
                    face_summary.append(summary)
            if len(common)==2:
                a,b=common[0][0],common[1][0]
                pairs.append({'face':fid,'elements':[int(a['element'][0]),int(b['element'][0])],
                    'sorted_stream_offsets':[common[0][1]['stream_offset'],common[1][1]['stream_offset']],
                    'max_common_point_gap':float(np.max(np.linalg.norm(a['point']-b['point'],axis=1))),
                    'max_common_normal_sum':float(np.max(np.linalg.norm(a['normal']+b['normal'],axis=1))),
                    'common_weighted_normal_sum':np.sum(a['physical_weight'][:,None]*a['normal']+b['physical_weight'][:,None]*b['normal'],axis=0).tolist(),
                    'common_position_moment_sum':float(np.sum(a['physical_weight']*np.sum(a['point']*a['normal'],axis=1)+b['physical_weight']*np.sum(b['point']*b['normal'],axis=1))),
                    'default_moments_sum':(np.asarray(defaults[0]['moments_by_bin'])+np.asarray(defaults[1]['moments_by_bin'])).tolist()})
            if fid%100==0:budget.guard()
    budget.json(folder/'face-attribution.json',face_summary)
    budget.json(folder/'interior-pairs.json',pairs)
    comparisons={}
    for path in trace:
        budget.arrays(folder/f'trace-{path}.npz',element_moments=trace[path])
        comparisons[path]=compare_array(trace[path],native[path],budget.policy)
    budget.json(folder/'trace-native-vector-parity.json',comparisons)
    return volume_sum,comparisons


def phase_run(phase,policy,root):
    import ngsolve as ng
    cfg=json.loads(Path(policy['configuration']).read_text())
    ng.SetNumThreads(int(cfg['threads']));np.random.seed(cfg['seed'])
    folder=root/phase;folder.mkdir();budget=Budget(root,policy)
    with ng.TaskManager():
        if phase=='A':mesh=load_mesh(policy['old_mesh'],ng)
        elif phase=='B':mesh=regenerate(cfg,ng)
        else:
            meta=json.loads((root/'B'/'mesh.json').read_text())
            if sha(root/'B'/'mesh.vol.bin')!=meta['binary_sha256']:raise ValueError('B binary input changed')
            mesh=load_mesh(root/'B'/'mesh.vol.bin',ng)
        if mesh.ne>policy['max_elements'] or mesh.GetCurveOrder()!=cfg['mesh']['geometry_order']:
            raise ValueError('loaded/generated mesh violates fixed element/curve-order envelope')
        t=topology(mesh,ng);budget.json(folder/'topology.json',t[5])
        if phase=='B':
            old=json.loads(Path(policy['old_mesh_record']).read_text())
            estimate=projection(mesh.ne,len(t[1]),old['elements'],policy['old_boundary_elements'])
            budget.json(folder/'actual-resource-projection.json',estimate)
            if estimate['combined_stored_point_records']>policy['max_combined_point_records']:
                raise MemoryError('actual topology exceeds combined A/B/C point plan')
            budget.guard(policy['binary_archive_reserve_bytes'])
            mesh.ngmesh.Save(str(folder/'mesh.vol.bin'))
            if (folder/'mesh.vol.bin').stat().st_size>policy['binary_archive_reserve_bytes']:
                raise MemoryError('binary archive exceeds reserved size')
            budget.finish()
            budget.json(folder/'mesh.json',{'mesh_id':'B-regenerated-'+str(uuid.uuid4()),
                'binary_sha256':sha(folder/'mesh.vol.bin'),'same_as_original_mesh':False,
                'binary_status':'candidate_not_roundtrip_validated','recipe_sha256':sha(policy['configuration']),
                'seed':cfg['seed'],'netgen_seed_relationship':'unverified; no determinism claim',
                'ngsolve':ng.__version__,'source_sha256':sha(__file__)})
        v,b=observed(mesh,ng,t,folder,budget)
        if phase!='B':
            outcome=parity(phase,v,b,t,root,policy,budget,folder)
            budget.json(folder/'result.json',outcome)
            return
        # Baseline and attribution share this exact in-memory mesh object.
        budget.json(folder/'ordinary.json',ordinary(mesh,ng))
        native,summary,packed,facet_packed=native_controls(mesh,ng,t[0],folder,budget)
        volume,comparisons=traces(mesh,ng,t,folder,budget,native,packed,facet_packed)
        V=summary['positive_volume'];total=sum(native[label][0,4].sum() for label in LABELS)
        result={'status':'geometry_diagnostic_completed_not_admitted',
            'exterior_divergence_defect':abs(total-3*V)/(3*V),
            'all_element_divergence_defect':abs(native['all'][0,4].sum()-3*V)/(3*V),
            'trace_native_vector_parity_passed':all(row['pass'] for row in comparisons.values()),
            'signed_volume_trace_vs_native':compare_array(volume['signed'],summary['signed_volume'],policy),
            'absolute_volume_trace_vs_native':compare_array(volume['absolute'],V,policy),
            'same_as_original_mesh':False,'original_mesh_cause_inferred':False,
            'PDE_solve_performed':False,'normal_sign_correction_applied':False,'reference_admitted':False}
        budget.json(folder/'result.json',result)


def main():
    faulthandler.enable(all_threads=True)
    parser=argparse.ArgumentParser();parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--phase',choices=('A','B','C'))
    args=parser.parse_args();policy=json.loads(args.policy.read_text());root=args.out.resolve()
    if os.environ.get('JAX_ENABLE_X64')!='true' or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest':
        raise RuntimeError('required precision environment absent')
    for row in policy['immutable_inputs']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('immutable input hash mismatch: '+row['path'])
    if args.phase:
        try:phase_run(args.phase,policy,root)
        except BaseException as exc:
            folder=root/args.phase;folder.mkdir(exist_ok=True)
            (folder/'FAILED.json').write_bytes(encode({'error':repr(exc),'traceback':traceback.format_exc()}))
            raise
        return
    root.mkdir(parents=True,exist_ok=False);budget=Budget(root,policy)
    budget.json(root/'policy.json',policy)
    start=time.monotonic();processes=[]
    # A/B are independently predeclared. A failure never selects or modifies B.
    for phase in ('A','B','C'):
        if phase=='C' and not all((root/'B'/f).exists() for f in ('mesh.json','mesh.vol.bin','observed-volume.npz','observed-boundary.npz','topology.json')):
            processes.append({'phase':'C','status':'not_run_B_archive_or_observations_absent'});continue
        remaining=policy['total_process_seconds']-(time.monotonic()-start)
        limit=min(policy['phase_timeout_seconds'][phase],remaining-10)
        if limit<10:
            processes.append({'phase':phase,'status':'not_run_global_time_guard'});continue
        command=[sys.executable,str(Path(__file__).resolve()),'--policy',str(args.policy.resolve()),'--out',str(root),'--phase',phase]
        try:
            with (root/f'{phase}.out').open('w') as stdout,(root/f'{phase}.err').open('w') as stderr:
                completed=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=limit,check=False)
            status={'phase':phase,'exit_code':completed.returncode,'status':'terminal'}
        except subprocess.TimeoutExpired:
            status={'phase':phase,'status':'timeout_killed_by_controller'}
        processes.append(status);budget.json(root/'processes.json',processes)
    budget.json(root/'result.json',{'status':'three_process_diagnostic_terminal','processes':processes,
        'binary_format_admitted_for_G2':False,'reference_admitted':False,'original_mesh_cause_inferred':False,
        'elapsed_process_seconds':time.monotonic()-start})
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()]
    budget.json(root/'OUTPUT-MANIFEST.json',{'files':files,'manifest_excludes_itself':True})


if __name__=='__main__':main()
