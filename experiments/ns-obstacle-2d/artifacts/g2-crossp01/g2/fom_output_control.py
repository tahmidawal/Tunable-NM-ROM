"""Native cached-output equivalence on a retained mesh; no physical admission."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from compare_fields import load_candidate
import reference_export as rex
from fom_outputs import make_output_factory
from oriented_normals import fluid_outward_normal
from scalars import scalar_value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(args.plan.read_text());folder,cfg,result=load_candidate(plan['case'])
    if cfg['kind']!='dfg' or cfg['outlet']!='gradient':raise ValueError('DFG gradient control only')
    import ngsolve as ng
    ng.SetNumThreads(cfg['threads'])
    with ng.TaskManager():
        mesh=ng.Mesh(str(folder/rex.MESH_FILENAME));V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
        layout,digest=rex.layout_record(mesh,V,Q,ng);saved=json.loads((folder/'fe_layout.json').read_text());assert saved=={'layout':layout,'sha256':digest}
        with np.load(folder/'full_rule.npz',allow_pickle=False) as z:volume={k:z[k] for k in ('points','weights')}
        boundary=json.loads((folder/'boundary_rule.json').read_text());start=time.perf_counter()
        evaluators=make_output_factory(ng,volume,boundary,args.out/'factory_normals.json')(mesh,V,Q,X,cfg)
        offline_seconds=time.perf_counter()-start
        normal,certificate=fluid_outward_normal(mesh,cfg['center'],order=cfg['diagnostic_order'],evidence_path=args.out/'oracle_normals.json');assert certificate['passed']
        state=ng.GridFunction(X);state.Load(str(folder/'solution.sol'));saved_coefficients=state.vec.FV().NumPy().copy();u,p=state.components
        rng=np.random.default_rng(cfg['seed']);states=[saved_coefficients,rng.standard_normal(X.ndof)*.01,saved_coefficients+rng.standard_normal(X.ndof)*.001]
        records=[]
        for index,coefficients in enumerate(states):
            state.vec.FV().NumPy()[:]=coefficients;nu=cfg['U']*cfg['D']/[20.,15.,25.][index]
            actual={name:function(u,p,nu) for name,function in evaluators.items()}
            velocity=[];pressure=[]
            for element in mesh.Elements(ng.VOL):
                tr=mesh.GetTrafo(element);ir=ng.IntegrationRule(element.type,cfg['diagnostic_order'])
                for ip in ir:
                    mip=tr(ip);velocity.append(np.array(u(mip),copy=True)/cfg['U']);pressure.append(scalar_value(p(mip),'oracle pressure')/cfg['U']**2)
            pole_u=[];pole_p=[]
            for x in (cfg['center'][0]-cfg['radius'],cfg['center'][0]+cfg['radius']):
                mip=mesh(x,cfg['center'][1],tol=1e-12);assert mip.nr>=0;pole_u.append(np.array(u(mip),copy=True)/cfg['U']);pole_p.append(scalar_value(p(mip),'oracle pole')/cfg['U']**2)
            derivative=ng.BoundaryFromVolumeCF(ng.Grad(u));force=np.zeros(2)
            for element in mesh.Elements(ng.BND):
                if str(element.mat)!='obstacle':continue
                tr=mesh.GetTrafo(element);ir=ng.IntegrationRule(element.type,cfg['diagnostic_order'])
                for ip in ir:
                    mip=tr(ip);gradient=np.array(derivative(mip),copy=True).reshape(2,2);stress=nu*(gradient+gradient.T)-scalar_value(p(mip),'oracle force pressure')*np.eye(2)
                    weight=scalar_value(ip.weight,'reference weight')*scalar_value(mip.measure,'mapped measure');force+=weight*(stress@(-np.array(normal(mip),copy=True)))
            oracle={'full_velocity_pressure_fields':{'velocity':np.asarray(velocity),'pressure':np.asarray(pressure)},'fixed_probes':{'velocity':np.asarray(pole_u),'pressure':np.asarray(pole_p)},'integrated_forces':{'force_coefficient':2*force/(cfg['U']**2*cfg['D'])}}
            arrays={'physical_mixed_coefficients':coefficients};checks={}
            for contract,fields in oracle.items():
                for field,expected in fields.items():
                    got=actual[contract][field];assert expected.shape==got.shape;key=contract+'__'+field;arrays[key+'__oracle']=expected;arrays[key+'__cached']=got
                    error=float(np.max(np.abs(got-expected)));scale=max(1.,float(np.max(np.abs(expected))));checks[key]={'max_absolute':error,'scale':scale,'scaled_max':error/scale,'tolerance':rex.RELOAD_TOL}
            np.savez_compressed(args.out/f'state-{index:02d}.npz',**arrays)
            records.append({'state':index,'nu':nu,'checks':checks,'passed':all(row['scaled_max']<=rex.RELOAD_TOL for row in checks.values())})
        evidence=evaluators['integrated_forces'].contraction_evidence
        np.savez_compressed(args.out/'force_contraction.npz',**{k:v for k,v in evidence.items() if isinstance(v,np.ndarray)})
        record={'passed':all(row['passed'] for row in records),'physical_admission':False,'source_manifest_sha256':plan['case']['export_manifest_sha256'],'mesh_sha256':rex.sha_file(folder/rex.MESH_FILENAME),'layout_sha256':digest,'states':records,'offline_setup_seconds_diagnostic_only':offline_seconds,'offline_components':evaluators['integrated_forces'].offline_evidence,'timing_claim':False,'field_rule_points':len(volume['weights']),'force_velocity_support':len(evidence['velocity_ids']),'force_pressure_support':len(evidence['pressure_ids']),'all_original_rule_points_used':True,'no_new_PDE_solve':True}
        rex.dump(args.out/'control.json',record)
        if not record['passed']:raise ArithmeticError('cached native FOM output equivalence failed')

if __name__=='__main__':main()
