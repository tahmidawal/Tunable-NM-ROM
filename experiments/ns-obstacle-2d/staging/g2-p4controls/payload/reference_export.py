"""Prospective G2 reference evidence; no native operation runs on import.

These checks do not admit a physical reference or create a training manifest.
The layout algorithm matches reviewed ROM adapter commit 97847a479 exactly.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import resource
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
from scalars import scalar_value

ESSENTIAL='inlet|wall.*|obstacle'
CAPS={'volume_points':1300000,'boundary_points':30000,'archive_bytes':2**30}
RELOAD_TOL=1e-10
POLYNOMIAL_CHECK_OVER_U=1e-12
MESH_FILENAME='mesh.vol.bin'
MESH_FORMAT='netgen-binary-archive-v6.2.2608'
RELOAD_POLICY='saved_curved_binary_verified_fresh_process_no_recurve'


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha_file(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def enabled(cfg):
    flag=cfg.get('reference_export_controls',False)
    if type(flag) is not bool:raise TypeError('reference_export_controls must be boolean')
    lift=cfg.get('export_common_stokes_lift',False)
    if type(lift) is not bool or (lift and not flag):raise ValueError('common lift export requires explicit G2 controls')
    if flag and cfg['release'] not in ('G2_steady_control_candidate','G2_steady_training_export'):
        raise ValueError('G2 export controls need an explicitly reviewed G2 configuration')
    return flag


def layout_record(mesh, velocity_space, pressure_space, ng):
    """Canonical JSON digest includes ordered VOL elements and local FE DOF numbers."""
    record = {'schema': 'ns-obstacle-fe-layout-v1', 'dimension': int(mesh.dim),
              'velocity_ndof': int(velocity_space.ndof), 'pressure_ndof': int(pressure_space.ndof),
              'velocity_free': [int(bool(v)) for v in velocity_space.FreeDofs()],
              'pressure_free': [int(bool(v)) for v in pressure_space.FreeDofs()], 'elements': []}
    for el in mesh.Elements(ng.VOL):
        record['elements'].append({'number': int(el.nr), 'type': str(el.type),
            'velocity_dofs': [int(i) for i in velocity_space.GetDofNrs(el)],
            'pressure_dofs': [int(i) for i in pressure_space.GetDofNrs(el)]})
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return record, digest


def geometry_signature(cfg):
    result={k:cfg[k] for k in ('xmin','xmax','H','geometry_order')}
    result['obstacle']=cfg['kind'] in ('dfg','manufactured')
    if result['obstacle']:result.update(center=cfg['center'],radius=cfg['radius'])
    return result


def make_or_reuse_mesh(cfg,out,geo,ng):
    """Reuse only a hash-bound saved curved mesh with actual prior reload evidence."""
    out=Path(out);source=cfg.get('immutable_mesh')
    if source is None:
        mesh=ng.Mesh(geo.GenerateMesh(maxh=cfg['maxh']))
        mesh.Curve(cfg['geometry_order']);mesh.ngmesh.Save(str(out/MESH_FILENAME))
        return mesh,{'mode':'generated','mesh_filename':MESH_FILENAME,'mesh_format':MESH_FORMAT,
                     'mesh_sha256':sha_file(out/MESH_FILENAME)}
    if not enabled(cfg):raise ValueError('immutable mesh reuse requires reviewed G2 controls')
    mesh_path=Path(source['mesh_path']).resolve();identity_path=Path(source['identity_path']).resolve()
    for path,key in ((mesh_path,'mesh_sha256'),(identity_path,'identity_sha256')):
        if path.stat().st_size>CAPS['archive_bytes'] or sha_file(path)!=source[key]:
            raise ValueError(f'immutable mesh source hash/size mismatch: {path}')
    identity=json.loads(identity_path.read_text())
    reload_path=identity_path.with_name('reload_checks.json')
    if sha_file(reload_path)!=identity.get('reload_checks_sha256'):raise ValueError('immutable source reload-evidence hash mismatch')
    reload_record=json.loads(reload_path.read_text())
    if (reload_record.get('passed') is not True or reload_record.get('policy')!=RELOAD_POLICY
        or reload_record.get('source_files',{}).get(MESH_FILENAME)!=source['mesh_sha256']):
        raise ValueError('immutable source reload evidence does not certify these mesh bytes')
    if (identity.get('policy')!=RELOAD_POLICY or identity.get('reload_passed') is not True
        or identity.get('mesh_filename')!=MESH_FILENAME or identity.get('mesh_format')!=MESH_FORMAT
        or identity.get('geometry')!=geometry_signature(cfg) or identity.get('mesh_sha256')!=source['mesh_sha256']
        or identity.get('ngsolve_version')!=ng.__version__
        or identity.get('mesh_construction')!={k:cfg[k] for k in ('maxh','obstacle_h','seed')}):
        raise ValueError('immutable source lacks matching geometry/reload identity')
    shutil.copyfile(mesh_path,out/MESH_FILENAME)
    mesh=ng.Mesh(str(out/MESH_FILENAME))  # Never silently Curve or regenerate a loaded mesh.
    if int(mesh.GetCurveOrder())!=cfg['geometry_order']:raise ValueError('loaded curve order mismatch')
    return mesh,{'mode':'immutable_byte_copy','mesh_filename':MESH_FILENAME,'mesh_format':MESH_FORMAT,
                 'mesh_sha256':sha_file(out/MESH_FILENAME),
                 'source_identity_sha256':source['identity_sha256'],'source_layout_sha256':identity['layout_sha256']}


def obstacle_quadrature(points,weights,cfg):
    """Count analytically inadmissible points; never delete/reweight them silently."""
    points=np.asarray(points);weights=np.asarray(weights)
    if points.ndim!=2 or points.shape[1]!=2 or weights.shape!=(len(points),):raise ValueError('rule shape')
    if not np.isfinite(points).all() or not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('invalid rule')
    if cfg['kind'] not in ('dfg','manufactured'):
        return None,{'obstacle_present':False,'actually_removed_count':0,'actually_removed_weight':0.}
    distance=np.linalg.norm(points-np.asarray(cfg['center']),axis=1)-cfg['radius']
    inside=distance<0
    result={'obstacle_present':True,'sign':'radius from exact center minus exact radius; negative inside solid',
            'minimum_signed_distance':float(distance.min()),'minimum_signed_distance_over_D':float(distance.min()/cfg['D']),
            'candidate_excluded_count':int(inside.sum()),'candidate_excluded_weight':float(weights[inside].sum()),
            'candidate_excluded_weight_fraction':float(weights[inside].sum()/weights.sum()),
            'actually_removed_count':0,'actually_removed_weight':0.,
            'no_obstacle_quadrature':bool(not inside.any()),'classification':'strict signed distance < 0; no clipping or tolerance offset'}
    return distance.astype(np.float64),result


def polynomial_trace_bound(nodes,errors,probe_nodes,probe_errors,degree,U):
    """Maximum estimate plus a conservative Chebyshev coefficient norm bound.

    FE-minus-prescribed-FE trace is degree <=p on each reference edge. Verify the
    fitted polynomial at independent points, then bound each component by the
    sum of absolute Chebyshev coefficients (|T_j|<=1). Add the fixed roundoff
    allowance; this is a floating-point certificate, not interval arithmetic.
    """
    arrays=[np.asarray(v,dtype=np.float64) for v in (nodes,errors,probe_nodes,probe_errors)]
    nodes,errors,probe_nodes,probe_errors=arrays
    if errors.shape!=(degree+1,2) or probe_errors.shape!=(len(probe_nodes),2) or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('polynomial trace input shape/nonfinite mismatch')
    polys=[np.polynomial.Chebyshev.fit(nodes,errors[:,j],degree,domain=[0,1]) for j in range(2)]
    predicted=np.column_stack([p(probe_nodes) for p in polys])
    residual=float(np.max(np.linalg.norm(predicted-probe_errors,axis=1),initial=0))
    allowance=POLYNOMIAL_CHECK_OVER_U*U
    if residual>allowance:raise ArithmeticError('FE edge trace is not verified degree-p polynomial within roundoff allowance')
    squared=polys[0]*polys[0]+polys[1]*polys[1]
    roots=squared.deriv().roots()
    real=[float(z.real) for z in roots if abs(z.imag)<=1e-10 and 0<float(z.real)<1]
    candidates=[0.,1.,*real]
    maximum=math.sqrt(max(0.,max(float(squared(x)) for x in candidates)))
    bound=float(np.linalg.norm([np.abs(p.coef).sum() for p in polys]))+allowance
    if maximum>bound+allowance:raise ArithmeticError('polynomial maximum exceeds coefficient bound')
    return {'degree':degree,'maximum_estimate':maximum,'coefficient_norm_upper_bound_with_roundoff':bound,
            'independent_polynomial_check_max':residual,'roundoff_allowance':allowance,
            'stationary_candidates':candidates,'chebyshev_coefficients':[p.coef.tolist() for p in polys]}


def essential_trace_certificate(mesh,velocity,prescribed_FE_trace,p,U,ng):
    nodes=.5*(1-np.cos(np.pi*np.arange(p+1)/p))
    probes=np.linspace(0.,1.,2*p+3)
    rows=[]
    for el in mesh.Elements(ng.BND):
        label=str(el.mat)
        if label=='outlet':continue
        if label not in ('inlet','wall_bottom','wall_top','obstacle'):raise ValueError('unexpected essential boundary')
        if el.type!=ng.ET.SEGM:raise ValueError('pointwise trace certificate only supports 2D segment boundaries')
        tr=mesh.GetTrafo(el)
        def sample(abscissae):
            ir=ng.IntegrationRule([(float(t),) for t in abscissae],[1.]*len(abscissae))
            values=[]
            for ip in ir:
                mip=tr(ip)
                values.append(np.array(velocity(mip),dtype=float,copy=True)-np.array(prescribed_FE_trace(mip),dtype=float,copy=True))
            return np.asarray(values)
        row=polynomial_trace_bound(nodes,sample(nodes),probes,sample(probes),p,U)
        row.update(element_number=int(el.nr),label=label);rows.append(row)
    if not rows:raise ValueError('no essential edges')
    return {'target':'prescribed FE Dirichlet trace; analytic-data approximation remains a separate diagnostic',
            'method':'degree-p reference-edge polynomial; stationary-point maximum estimate and Chebyshev coefficient upper bound plus verified roundoff allowance',
            'maximum_estimate_over_U':max(row['maximum_estimate'] for row in rows)/U,
            'upper_bound_over_U':max(row['coefficient_norm_upper_bound_with_roundoff'] for row in rows)/U,
            'roundoff_check_over_U':POLYNOMIAL_CHECK_OVER_U,'interval_arithmetic':False,'elements':rows}


def affine_gradient_orientation(mesh,cfg,ng):
    """Non-symmetric affine witness; identity gradients cannot detect transpose."""
    degree=max(cfg['velocity_order'],cfg['geometry_order'])
    space=ng.VectorH1(mesh,order=degree)
    witness=ng.GridFunction(space);witness.Set(ng.CF((ng.y,0.)))
    volume_gradient=ng.Grad(witness)
    boundary_gradient=ng.BoundaryFromVolumeCF(volume_gradient)
    expected=np.array([[0.,1.],[0.,0.]])
    errors={};counts={}
    for label,vb,cf in [('volume',ng.VOL,volume_gradient),('boundary',ng.BND,boundary_gradient)]:
        worst=0.;count=0
        for el in mesh.Elements(vb):
            tr=mesh.GetTrafo(el);ir=ng.IntegrationRule(el.type,2)
            for ip in ir:
                mip=tr(ip);actual=np.array(cf(mip),dtype=float,copy=True).reshape(2,2)
                if not np.isfinite(actual).all():raise ValueError('nonfinite affine witness gradient')
                worst=max(worst,float(np.linalg.norm(actual-expected)));count+=1
        if count==0:raise ValueError('missing affine witness samples')
        errors[label]=worst;counts[label]=count
    return {'velocity':'(y,0)','expected_gradient':expected.tolist(),'axes':'component,coordinate',
            'fixture_velocity_degree':degree,'production_velocity_degree':cfg['velocity_order'],
            'geometry_order':cfg['geometry_order'],'maximum_Frobenius_errors':errors,'sample_counts':counts,
            'threshold':1e-10,'passed':max(errors.values())<=1e-10,
            'interpretation':'checks transpose orientation; no assertion that previous Grad was incorrect'}


def relative_max_record(actual,expected,scale=1.):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('reload shape/nonfinite mismatch')
    numerator=float(np.max(np.abs(a-b),initial=0))
    denominator=max(float(np.max(np.abs(b),initial=0)),scale)
    return {'numerator':numerator,'denominator':denominator,'value':numerator/denominator}


def analytic_normal_certificate(boundary,cfg):
    """Independent orientation plus closure; closure alone cannot detect a sign flip."""
    points=np.asarray(boundary['points']);normals=np.asarray(boundary['normals'])
    weights=np.asarray(boundary['weights']);labels=np.asarray(boundary['labels'])
    if (points.shape!=normals.shape or points.shape!=(len(weights),2) or labels.shape!=weights.shape
        or not all(np.isfinite(a).all() for a in (points,normals,weights)) or np.any(weights<=0)):
        raise ValueError('invalid boundary normal arrays')
    record={'passed':True,'unit_normal_max_error':float(np.max(np.abs(np.linalg.norm(normals,axis=1)-1))),
            'side_max_errors':{},'closed_components':{},'tolerance':RELOAD_TOL,'obstacle_orientation_max_dot_limit':-.99}
    sides={'inlet':[-1.,0.],'outlet':[1.,0.],'wall_top':[0.,1.],'wall_bottom':[0.,-1.]}
    if not set(labels)<=set(sides)|{'obstacle'}:raise ValueError('unexpected boundary labels')
    for name,expected in sides.items():
        mask=labels==name
        if not np.any(mask):raise ValueError('missing outer face')
        record['side_max_errors'][name]=float(np.max(np.linalg.norm(normals[mask]-expected,axis=1)))
    if cfg['kind'] in ('dfg','manufactured'):
        mask=labels=='obstacle'
        if not np.any(mask):raise ValueError('missing obstacle boundary')
        radial=points[mask]-np.asarray(cfg['center']);length=np.linalg.norm(radial,axis=1)
        if np.any(length<=0):raise ValueError('invalid radial normal witness')
        dot=np.sum(normals[mask]*radial/length[:,None],axis=1)
        record['obstacle_normal_dot_radial_range']=[float(dot.min()),float(dot.max())]
        record['passed'] &= bool(dot.max()<=-.99)
    for name,mask in [('outer_union',labels!='obstacle'),('obstacle',labels=='obstacle')]:
        if not np.any(mask):continue
        integral=np.sum(weights[mask,None]*normals[mask],axis=0);measure=float(weights[mask].sum())
        ratio=float(np.linalg.norm(integral)/measure)
        record['closed_components'][name]={'normal_integral':integral.tolist(),'numerator':float(np.linalg.norm(integral)),
            'boundary_measure_denominator':measure,'value':ratio}
        record['passed'] &= ratio<=RELOAD_TOL
    record['passed'] &= max(record['unit_normal_max_error'],*record['side_max_errors'].values())<=RELOAD_TOL
    return record


def _mapped_state(mesh,velocity,pressure,order,boundary,ng,normal=None):
    points=[];weights=[];values=[];gradients=[];pressures=[];jacobians=[];normals=[];labels=[]
    derivative=ng.Grad(velocity)
    if boundary:derivative=ng.BoundaryFromVolumeCF(derivative)
    if boundary and normal is None:raise ValueError('reload boundary export requires verified ordinary-BND normal')
    raw_normal=ng.specialcf.normal(2)
    raw_normals=[]
    cap=CAPS['boundary_points' if boundary else 'volume_points']
    for el in mesh.Elements(ng.BND if boundary else ng.VOL):
        tr=mesh.GetTrafo(el);ir=ng.IntegrationRule(el.type,order)
        for ip in ir:
            if len(points)>=cap:raise MemoryError('reload rule exceeds reviewed producer point cap')
            mip=tr(ip);w=scalar_value(ip.weight,'reload_reference_weight')*scalar_value(mip.measure,'reload_mapped_measure')
            if w<=0:raise ValueError('nonpositive reloaded weight')
            points.append(np.array(mip.point,dtype=float,copy=True));weights.append(w)
            values.append(np.array(velocity(mip),dtype=float,copy=True));gradients.append(np.array(derivative(mip),dtype=float,copy=True).reshape(2,2))
            pressures.append(scalar_value(pressure(mip),'reloaded_pressure'))
            if boundary:
                normals.append(np.array(normal(mip),dtype=float,copy=True));labels.append(str(el.mat))
                raw_normals.append(np.array(raw_normal(mip),dtype=float,copy=True))
            else:
                jac=np.array(mip.jacobi,dtype=float,copy=True)
                if np.linalg.det(jac)<=0:raise ValueError('nonpositive reloaded Jacobian')
                jacobians.append(jac)
    result={name:np.asarray(data,dtype=np.float64) for name,data in
            [('points',points),('weights',weights),('velocity',values),('velocity_gradient',gradients),('pressure',pressures)]}
    if boundary:result.update(normals=np.asarray(normals,dtype=np.float64),raw_normals=np.asarray(raw_normals,dtype=np.float64),labels=labels)
    else:result['jacobians']=np.asarray(jacobians,dtype=np.float64)
    return result


def reload_certificate(out,cfg,original_mesh,original_layout,ng):
    """Exec a fresh interpreter; no original native mesh or FE object reaches it."""
    out=Path(out).resolve()
    request={'producer_pid':os.getpid(),'original_curve_order':int(original_mesh.GetCurveOrder()),
             'mesh_filename':MESH_FILENAME,'mesh_format':MESH_FORMAT,
             'mesh_sha256':sha_file(out/MESH_FILENAME),'config_sha256':sha_file(out/'config.json'),
             'layout_file_sha256':sha_file(out/'fe_layout.json'),
             'layout_sha256':original_layout['sha256']}
    dump(out/'reload_request.json',request)
    command=[sys.executable,str(Path(__file__).resolve()),'--reload-case',str(out)]
    with (out/'reload.stdout.log').open('w') as stdout,(out/'reload.stderr.log').open('w') as stderr:
        completed=subprocess.run(command,stdout=stdout,stderr=stderr,check=False)
    dump(out/'reload_process.json',{'command':command,'producer_pid':os.getpid(),'returncode':completed.returncode})
    if completed.returncode:raise RuntimeError(f'fresh-process reload failed: exit {completed.returncode}; evidence retained')
    report=json.loads((out/'reload_checks.json').read_text())
    if (report.get('passed') is not True or report.get('policy')!=RELOAD_POLICY
        or report['execution']['pid']==os.getpid()):raise ArithmeticError('fresh-process reload certificate missing')
    return report


def _reload_certificate(out,cfg,original_curve_order,original_layout,ng,request):
    """Actually load saved files, reconstruct spaces and compare every mapped sample."""
    out=Path(out);start=time.perf_counter()
    report={'passed':False,'policy':RELOAD_POLICY,'threshold':RELOAD_TOL,'checks':{},'curve_orders':{},'source_files':{},
            'mesh_filename':MESH_FILENAME,'mesh_format':MESH_FORMAT,
            'execution':{'pid':os.getpid(),'parent_pid':os.getppid(),'producer_pid':request['producer_pid'],
                         'executable':sys.executable,'fresh_interpreter':True}}
    try:
        if os.getpid()==request['producer_pid']:raise RuntimeError('reload must use a fresh process')
        for name in (MESH_FILENAME,'solution.sol','stokes.sol','full_rule.npz','boundary_rule.json','fe_layout.json'):
            report['source_files'][name]=sha_file(out/name)
        with zipfile.ZipFile(out/'full_rule.npz') as archive:
            if sum(row.file_size for row in archive.infolist())>CAPS['archive_bytes']:raise MemoryError('reference archive exceeds adapter bound')
        with np.load(out/'full_rule.npz',allow_pickle=False) as archive:
            data={name:archive[name] for name in archive.files}
        if any(a.dtype!=np.dtype('float64') or not np.isfinite(a).all() for a in data.values()):raise TypeError('reference arrays must already be finite float64')
        mesh=ng.Mesh(str(out/MESH_FILENAME))
        report['curve_orders']={'configured':cfg['geometry_order'],'original':original_curve_order,'reloaded':int(mesh.GetCurveOrder())}
        if len(set(report['curve_orders'].values()))!=1:raise ValueError('saved mesh did not preserve curved order')
        V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
        record,digest=layout_record(mesh,V,Q,ng)
        if record!=original_layout['layout'] or digest!=original_layout['sha256']:raise ValueError('reloaded ordered FE layout differs')
        report['layout_sha256']=digest
        solution=ng.GridFunction(X);solution.Load(str(out/'solution.sol'))
        stokes=ng.GridFunction(X);stokes.Load(str(out/'stokes.sol'))
        for name,g in [('solution',solution),('stokes',stokes)]:
            report['checks'][name+'_coefficients']=relative_max_record(g.vec.FV().NumPy().copy(),data[name+'_coefficients'])
        uh,ph=solution.components
        volume=_mapped_state(mesh,uh,ph,cfg['diagnostic_order'],False,ng)
        scales={'points':cfg['D'],'weights':1.,'jacobians':1.,'velocity':cfg['U'],'velocity_gradient':cfg['U']/cfg['D'],'pressure':cfg['U']**2}
        for key,scale in scales.items():report['checks']['volume_'+key]=relative_max_record(volume[key],data[key],scale)
        from oriented_normals import fluid_outward_normal
        normal, normal_certificate=fluid_outward_normal(mesh, cfg['center'] if cfg['kind'] in ('dfg','manufactured') else None,
            order=cfg['diagnostic_order'], evidence_path=out/'reload_normal_orientation.json')
        original_normals=json.loads((out/'normal_orientation.json').read_text())
        if normal_certificate['factors']!=original_normals['factors']:raise ArithmeticError('reloaded boundary orientation factors changed')
        report['normal_orientation']={'passed':normal_certificate['passed'],'factors':normal_certificate['factors'],
                                      'certificate_sha256':sha_file(out/'reload_normal_orientation.json')}
        boundary=_mapped_state(mesh,uh,ph,cfg['diagnostic_order'],True,ng,normal=normal)
        report['analytic_normal_checks']=analytic_normal_certificate(boundary,cfg)
        if not report['analytic_normal_checks']['passed']:raise ArithmeticError('reloaded analytic normal orientation/closure failed')
        saved=json.loads((out/'boundary_rule.json').read_text())
        if [row['label'] for row in saved]!=boundary['labels']:raise ValueError('reloaded boundary enumeration differs')
        for key,saved_key,scale in [('points','point',cfg['D']),('weights','weight',1.),('normals','normal',1.),('raw_normals','raw_normal',1.),('velocity','velocity',cfg['U']),('velocity_gradient','velocity_gradient',cfg['U']/cfg['D']),('pressure','pressure',cfg['U']**2)]:
            report['checks']['boundary_'+key]=relative_max_record(boundary[key],[row[saved_key] for row in saved],scale)
        if max(row['value'] for row in report['checks'].values())>RELOAD_TOL:raise ArithmeticError('saved FE reconstruction exceeds fixed parity threshold')
        report['passed']=True
    except Exception as exc:
        report['failure']=repr(exc)
        raise
    finally:
        report['seconds']=time.perf_counter()-start
        report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        dump(out/'reload_checks.json',report)
    dump(out/'mesh_identity.json',{'schema':'ns-obstacle-immutable-mesh-v1','policy':report['policy'],'reload_passed':True,
         'geometry':geometry_signature(cfg),'mesh_construction':{k:cfg[k] for k in ('maxh','obstacle_h','seed')},
         'mesh_filename':MESH_FILENAME,'mesh_format':MESH_FORMAT,
         'mesh_sha256':sha_file(out/MESH_FILENAME),'layout_sha256':digest,'ngsolve_version':ng.__version__,
         'reload_checks_sha256':sha_file(out/'reload_checks.json')})
    return report


def reload_main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--reload-case',required=True,type=Path)
    args=parser.parse_args();out=args.reload_case.resolve()
    if (out/'reload_checks.json').exists():raise FileExistsError('refusing to overwrite reload evidence')
    request=json.loads((out/'reload_request.json').read_text())
    if request['mesh_filename']!=MESH_FILENAME or request['mesh_format']!=MESH_FORMAT:
        raise ValueError('unrecognized mesh filename/format; no text conversion permitted')
    for name,key in ((MESH_FILENAME,'mesh_sha256'),('config.json','config_sha256'),('fe_layout.json','layout_file_sha256')):
        if sha_file(out/name)!=request[key]:raise ValueError(f'reload request hash mismatch: {name}')
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest' or os.environ.get('JAX_ENABLE_X64')!='true':
        raise RuntimeError('required precision environment missing')
    cfg=json.loads((out/'config.json').read_text());layout=json.loads((out/'fe_layout.json').read_text())
    if layout['sha256']!=request['layout_sha256']:raise ValueError('reload layout identity mismatch')
    import ngsolve as ng
    ng.SetNumThreads(int(cfg['threads']))
    with ng.TaskManager():
        _reload_certificate(out,cfg,request['original_curve_order'],layout,ng,request)


def wake_centerline(mesh,velocity,cfg,ng):
    """Deterministic FE samples only; no wake-length inference or admission gate."""
    if cfg['kind']!='dfg':return None
    start=cfg['center'][0]+cfg['radius'];end=cfg['xmax'];target_dx=cfg['D']/20
    intervals=int(math.ceil((end-start)/target_dx))
    if intervals+1>2001:raise MemoryError('centerline diagnostic point cap')
    xs=np.linspace(start,end,intervals+1);rows=[]
    for x in xs:
        xy=(float(x),float(cfg['center'][1]))
        try:
            point=mesh(*xy)
            if point.nr<0:raise ValueError('exact sample outside FE mesh; no nudge')
            value=np.array(velocity(point),dtype=float,copy=True)
            if value.shape!=(2,) or not np.isfinite(value).all():raise ValueError('invalid FE velocity sample')
            rows.append({'point':list(xy),'velocity':value.tolist(),'valid':True,'error':None})
        except Exception as exc:rows.append({'point':list(xy),'velocity':None,'valid':False,'error':repr(exc)})
    return {'status':'sampling diagnostic only, not an admission gate or inferred reference','x_start':start,'x_end':end,
            'target_spacing':target_dx,'actual_spacing':float(xs[1]-xs[0]),'height':cfg['center'][1],
            'count':len(rows),'valid_count':sum(row['valid'] for row in rows),'samples':rows,'pressure_probes_changed':False}


if __name__=='__main__':reload_main()
