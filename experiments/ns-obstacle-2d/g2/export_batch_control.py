"""Native readout equivalence on retained coefficients; no new nonlinear solve."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from compare_fields import load_candidate
from oriented_normals import fluid_outward_normal
import reference_export as rex


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(args.plan.read_text());folder,cfg,result=load_candidate(plan['case'])
    import ngsolve as ng
    ng.SetNumThreads(cfg['threads']);records=[]
    with ng.TaskManager():
        mesh=ng.Mesh(str(folder/rex.MESH_FILENAME));V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
        layout,digest=rex.layout_record(mesh,V,Q,ng);assert json.loads((folder/'fe_layout.json').read_text())=={'layout':layout,'sha256':digest}
        state=ng.GridFunction(X);state.Load(str(folder/'solution.sol'));u,p=state.components
        normal,certificate=fluid_outward_normal(mesh,cfg['center'],order=cfg['diagnostic_order'],evidence_path=args.out/'normals.json');assert certificate['passed']
        with np.load(folder/'full_rule.npz',allow_pickle=False) as archive:original_volume={k:archive[k] for k in ('points','weights','velocity','velocity_gradient','pressure','jacobians')}
        original_boundary=json.loads((folder/'boundary_rule.json').read_text());keys={'points':'point','weights':'weight','velocity':'velocity','velocity_gradient':'velocity_gradient','pressure':'pressure','normals':'normal','raw_normals':'raw_normal'}
        original_boundary_arrays={k:np.asarray([row[v] for row in original_boundary],dtype=np.float64) for k,v in keys.items()}
        for boundary in (False,True):
            tag='boundary' if boundary else 'volume';tick=time.perf_counter()
            actual=rex._mapped_state(mesh,u,p,cfg['diagnostic_order'],boundary,ng,normal=normal,volume_point_limit=rex.volume_point_cap(cfg),batched=True)
            seconds=time.perf_counter()-tick;expected=original_boundary_arrays if boundary else original_volume;checks={};arrays={}
            for key,original in expected.items():
                arrays[key+'__original']=original;arrays[key+'__batched']=actual[key]
                scale={'points':cfg['D'],'velocity':cfg['U'],'velocity_gradient':cfg['U']/cfg['D'],'pressure':cfg['U']**2}.get(key,1.)
                checks[key]=rex.relative_max_record(actual[key],original,scale)
            if boundary:
                labels=[row['label'] for row in original_boundary]
                if actual['labels']!=labels:raise ValueError('boundary enumeration changed')
                arrays['labels']=np.asarray(labels)
            np.savez_compressed(args.out/(tag+'.npz'),**arrays)
            records.append({'kind':tag,'checks':checks,'seconds_diagnostic_only':seconds,'passed':all(row['value']<=rex.RELOAD_TOL for row in checks.values())})
        record={'passed':all(row['passed'] for row in records),'physical_admission':False,'case_manifest_sha256':plan['case']['export_manifest_sha256'],'layout_sha256':digest,'records':records,'same_original_coefficients':True,'no_PDE_solve':True,'timing_claim':False,'physics_thresholds_and_rule_unchanged':True};rex.dump(args.out/'control.json',record)
        if not record['passed']:raise ArithmeticError('batched producer readout differs from original native evidence')

if __name__=='__main__':main()
