"""Source-only common-rule comparison of binary G2 candidates; no admission.

All requested points must be valid in every mesh. No masking, nearest nodes,
coordinate nudges, pressure shifts or alternative quadrature are permitted.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import sys
import traceback
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import reference_export as rex
from scalars import scalar_value
from exact_lookup import ExactPointLocator


def field_metrics(velocity,pressure,reference_velocity,reference_pressure,weights,U):
    u,p,ur,pr,w=[np.asarray(x,dtype=np.float64) for x in
                 (velocity,pressure,reference_velocity,reference_pressure,weights)]
    n=len(w)
    if (u.shape!=(n,2) or ur.shape!=u.shape or p.shape!=(n,) or pr.shape!=p.shape
        or w.shape!=(n,) or n==0 or np.any(w<=0) or not np.isfinite(U) or U<=0
        or not all(np.isfinite(x).all() for x in (u,p,ur,pr,w))):
        raise ValueError('invalid common-rule fields/weights/scales')
    du=float(np.sqrt(np.sum(w[:,None]*(u-ur)**2)))
    unorm=float(np.sqrt(np.sum(w[:,None]*ur**2)))
    if unorm<=0:raise ValueError('zero reference velocity norm; no arbitrary denominator floor')
    dp=float(np.sqrt(np.sum(w*(p-pr)**2)));volume=float(w.sum());pscale=float(U**2*np.sqrt(volume))
    return {'velocity_relative':{'numerator':du,'denominator':unorm,'value':du/unorm},
            'pressure_scaled':{'numerator':dp,'denominator':pscale,'value':dp/pscale},
            'volume':volume,'point_count':n,'pressure_shift_fitted':False}


def load_candidate(entry):
    folder=Path(entry['directory']).resolve();manifest=folder/'EXPORT-MANIFEST.json'
    if rex.sha_file(manifest)!=entry['export_manifest_sha256']:raise ValueError('candidate manifest hash mismatch')
    export=json.loads(manifest.read_text())
    if (export.get('mesh_filename')!=rex.MESH_FILENAME or export.get('mesh_format')!=rex.MESH_FORMAT
        or export.get('reload_policy')!=rex.RELOAD_POLICY):raise ValueError('explicit binary contract required')
    for name,digest in export['files'].items():
        if Path(name).name!=name or rex.sha_file(folder/name)!=digest:raise ValueError('candidate artifact hash/path mismatch')
    result=json.loads((folder/'result.json').read_text());cfg=result['configuration']
    reload=json.loads((folder/'reload_checks.json').read_text())
    if reload.get('passed') is not True or reload.get('policy')!=rex.RELOAD_POLICY:
        raise ValueError('successful fresh-process native replay required')
    if cfg['release']!='G2_steady_control_candidate':raise ValueError('comparison expects explicit G2 control candidates')
    if cfg['precision']!='float64':raise ValueError('f64 required')
    return folder,cfg,result


def evaluate_case(folder,cfg,points,out,label,ng):
    mesh=ng.Mesh(str(folder/rex.MESH_FILENAME))
    if mesh.dim!=2 or mesh.ne>cfg['max_elements']:raise MemoryError('dimension or element cap')
    V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL)
    Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
    if X.ndof>cfg['max_dofs']:raise MemoryError('DOF cap')
    layout,digest=rex.layout_record(mesh,V,Q,ng)
    saved=json.loads((folder/'fe_layout.json').read_text())
    if saved['sha256']!=digest or saved['layout']!=layout:raise ValueError('comparison FE layout mismatch')
    solution=ng.GridFunction(X);solution.Load(str(folder/'solution.sol'));u,p=solution.components
    coordinate=ng.CF((ng.x,ng.y));values=np.full((len(points),2),np.nan);pressure=np.full(len(points),np.nan)
    mapped=np.full_like(points,np.nan);valid=np.zeros(len(points),dtype=bool);errors=[]
    reference_coordinates=np.full((len(points),3),np.nan);barycentric=np.full((len(points),3),np.nan)
    element_numbers=np.full(len(points),-1,dtype=np.int64)
    element_types={int(el.nr):str(el.type) for el in mesh.Elements(ng.VOL)}
    lookup_tolerance=1e-12
    reference_membership_roundoff=64*np.finfo(np.float64).eps
    locator=ExactPointLocator(mesh,ng,cfg['D']);fallbacks=[]
    for i,xy in enumerate(points):
        try:
            mip,owners,certificate,fallback=locator.locate(xy)
            if fallback is not None:fallbacks.append({'index':i,**fallback})
            element_numbers[i]=int(certificate['element_number'])
            if element_numbers[i]<0:raise ValueError('requested physical point outside candidate mesh')
            if element_types.get(int(element_numbers[i]))!='ET.TRIG':
                raise ValueError('common-rule lookup requires a volume triangle')
            ref=np.asarray(certificate['reference_coordinates'],dtype=np.float64)
            reference_coordinates[i]=ref
            if ref.shape!=(3,) or not np.isfinite(ref).all():raise ValueError('invalid reference coordinates')
            barycentric[i]=[ref[0],ref[1],1-ref[0]-ref[1]]
            if (abs(ref[2])>reference_membership_roundoff or np.min(barycentric[i]) < -reference_membership_roundoff):
                raise ValueError('returned reference point lies outside triangle; no extrapolation')
            mapped[i]=np.array(coordinate(mip),dtype=float,copy=True)
            if np.linalg.norm(mapped[i]-xy)>rex.RELOAD_TOL*cfg['D']:
                raise ValueError('physical point lookup discrepancy; no coordinate nudge')
            values[i]=np.array(u(mip),dtype=float,copy=True)
            pressure[i]=scalar_value(p(mip),'comparison_pressure_point')
            if not np.isfinite(values[i]).all() or not np.isfinite(pressure[i]):raise ValueError('nonfinite evaluated field')
            valid[i]=True
        except Exception as exc:
            errors.append({'index':i,'point':xy.tolist(),'error':repr(exc)})
            if hasattr(exc,'lookup_evidence'):fallbacks.append({'index':i,**exc.lookup_evidence})
    np.savez_compressed(out/(label+'.npz'),requested_points=points,mapped_points=mapped,
                        velocity=values,pressure=pressure,valid=valid,reference_coordinates=reference_coordinates,
                        barycentric_coordinates=barycentric,element_numbers=element_numbers)
    rex.dump(out/(label+'-validity.json'),{'all_valid':bool(valid.all()),'invalid_count':len(errors),'errors':errors,
                                         'point_adjustments':0,'discarded_points':0,'layout_sha256':digest,
                                         'lookup_tolerance':lookup_tolerance,'reference_membership_roundoff':reference_membership_roundoff,
                                         'fallbacks':fallbacks,'fallback_count':len(fallbacks),'element_types':element_types,'membership_rule':'all triangle barycentric coordinates >= -64eps; reference z within64eps; no coordinate alteration'})
    if not valid.all():raise ValueError(f'{label}: invalid common-rule points; no subset comparison performed')
    return values,pressure


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    try:
        if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest' or os.environ.get('JAX_ENABLE_X64')!='true':
            raise RuntimeError('required precision environment missing')
        plan=json.loads(args.plan.read_text());rex.dump(args.out/'plan.json',plan)
        candidates={label:load_candidate(entry) for label,entry in plan['cases'].items()}
        if not 2<=len(candidates)<=16 or any(not label.isidentifier() for label in candidates):
            raise ValueError('bounded named candidate set required')
        folder,cfg,_=candidates[plan['rule_case']]
        with zipfile.ZipFile(folder/'full_rule.npz') as archive:
            if sum(row.file_size for row in archive.infolist())>rex.CAPS['archive_bytes']:raise MemoryError('rule archive cap')
        with np.load(folder/'full_rule.npz',allow_pickle=False) as data:
            points=data['points'].copy();weights=data['weights'].copy()
        if (len(points)>rex.CAPS['volume_points'] or points.shape!=(len(weights),2)
            or points.dtype!=np.dtype('float64') or weights.dtype!=np.dtype('float64')
            or np.any(weights<=0) or not np.isfinite(points).all() or not np.isfinite(weights).all()):
            raise ValueError('invalid or oversized positive full rule')
        for _,other,_ in candidates.values():
            for key in ('kind','U','D','H','center','radius','nu','campaign_sha256','threads'):
                if other[key]!=cfg[key]:raise ValueError(f'candidate physical-family mismatch: {key}')
        if (np.any(np.linalg.norm(points-np.asarray(cfg['center']),axis=1)<cfg['radius'])
            or np.any(points[:,0]<max(c[1]['xmin'] for c in candidates.values()))
            or np.any(points[:,0]>min(c[1]['xmax'] for c in candidates.values()))):
            raise ValueError('full common rule is not inside all exact physical domains')
        np.savez_compressed(args.out/'common_rule.npz',points=points,weights=weights)
        import ngsolve as ng
        ng.SetNumThreads(int(cfg['threads']))
        fields={}
        with ng.TaskManager():
            for label,(case_folder,case_cfg,_) in candidates.items():
                fields[label]=evaluate_case(case_folder,case_cfg,points,args.out,label,ng)
        rows=[]
        for pair in plan['comparisons']:
            a,b=pair['case'],pair['reference'];row={'case':a,'reference':b,'purpose':pair['purpose']}
            row.update(field_metrics(*fields[a],*fields[b],weights,cfg['U']))
            same_domain=all(candidates[a][1][key]==candidates[b][1][key] for key in ('xmin','xmax','outlet'))
            row['unshifted_pressure_field_eligible_for_same_domain_gate']=same_domain
            if not same_domain:
                row['pressure_field_interpretation']='diagnostic only; domain comparison uses unshifted exact common-pole pressure differences'
                probes=[candidates[key][2]['metrics']['forces']['pressure_probes'] for key in (a,b)]
                if not all(x['valid'] for x in probes) or probes[0]['points']!=probes[1]['points']:
                    raise ValueError('domain comparison requires valid identical exact pole probes')
                row['pressure_drop_absolute_difference']=abs(probes[0]['difference']-probes[1]['difference'])
            rows.append(row)
        rex.dump(args.out/'comparison.json',{'status':'common_rule_fields_evaluated_no_physical_admission',
            'physical_admission':False,'rule_case':plan['rule_case'],'full_rule_preserved':True,'comparisons':rows,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'source_sha256':rex.sha_file(Path(__file__)),'plan_sha256':rex.sha_file(args.plan)})
    except Exception as exc:
        rex.dump(args.out/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'physical_admission':False})
        raise


if __name__=='__main__':main()
