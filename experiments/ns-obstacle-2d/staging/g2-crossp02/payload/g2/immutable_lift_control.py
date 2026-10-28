"""Source-only native immutable mesh/common-lift control; never admits training.

Run only under an exact reviewed package. Uses an actual retained producer case;
no CommonStokesLift admission is fabricated. No new nonlinear reference is solved.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import reference_export as rex
from shared_lift import split_common_lift,export_common_lift
from compare_fields import load_candidate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    plan=json.loads(a.plan.read_text());folder,cfg,result=load_candidate(plan['case'])
    if cfg['kind']!='dfg' or cfg['outlet']!='gradient':
        raise ValueError('common-lift control requires unforced DFG gradient outlet')
    a.out.mkdir(parents=True,exist_ok=False)
    import ngsolve as ng
    ng.SetNumThreads(cfg['threads'])
    cfg=dict(cfg,immutable_mesh={'mesh_path':str(folder/rex.MESH_FILENAME),'mesh_sha256':rex.sha_file(folder/rex.MESH_FILENAME),'identity_path':str(folder/'mesh_identity.json'),'identity_sha256':rex.sha_file(folder/'mesh_identity.json')})
    with ng.TaskManager():
        mesh,origin=rex.make_or_reuse_mesh(cfg,a.out,None,ng)
        V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=rex.ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order']);X=V*Q
        layout,digest=rex.layout_record(mesh,V,Q,ng);saved=json.loads((folder/'fe_layout.json').read_text())
        assert saved=={'layout':layout,'sha256':digest}
        stokes=ng.GridFunction(X);stokes.Load(str(folder/'stokes.sol'));coefficients=stokes.vec.FV().NumPy().copy()
        u,p_per_nu=split_common_lift(coefficients,V.ndof,cfg['nu'])
        (trial,pressure),(test,q)=X.TnT();free=np.array([bool(v) for v in X.FreeDofs()],dtype=bool)
        U,D=cfg['U'],cfg['D'];scale=np.r_[np.full(V.ndof,U**2*D),np.full(Q.ndof,U*D)]
        trace=ng.GridFunction(X);trace.components[0].Set(ng.CF((6*U*ng.y*(cfg['H']-ng.y)/cfg['H']**2,0)),definedon=mesh.Boundaries('inlet'))
        records=[];states=[];residuals=[];forcings=[]
        for Re in (15.,20.,25.):
            nu=U*D/Re;form=ng.BilinearForm(X)
            form+=(nu*ng.InnerProduct(ng.Grad(trial),ng.Grad(test))-pressure*ng.div(test)-q*ng.div(trial))*ng.dx(bonus_intorder=cfg['assembly_bonus_order']);form.Assemble()
            stokes.vec.FV().NumPy()[:]=np.r_[u,nu*p_per_nu]
            residual=stokes.vec.CreateVector();residual.data=form.mat*stokes.vec
            forcing=stokes.vec.CreateVector();forcing.data=form.mat*trace.vec
            raw=residual.FV().NumPy().copy();numerator=float(np.linalg.norm((raw/scale)[free]));denominator=max(float(np.linalg.norm((forcing.FV().NumPy()/scale)[free])),1e-30)
            states.append(stokes.vec.FV().NumPy().copy());residuals.append(raw);forcings.append(forcing.FV().NumPy().copy())
            records.append({'Re':Re,'nu':nu,'numerator':numerator,'denominator':denominator,'relative_residual':numerator/denominator})
        np.savez_compressed(a.out/'lift_scaling_arrays.npz',reference_coefficients=coefficients,states=np.asarray(states),residuals=np.asarray(residuals),free_mask=free,row_scale=scale,forcing_vectors=np.asarray(forcings),trace_coefficients=trace.vec.FV().NumPy().copy())
        record={'passed':all(r['relative_residual']<=cfg['linear_relative_tolerance'] for r in records),'physical_admission':False,'mesh_origin':origin,'layout_sha256':digest,'source_case_manifest_sha256':plan['case']['export_manifest_sha256'],'records':records,'no_new_nonlinear_solves':True,'no_fabricated_admission':True}
        rex.dump(a.out/'control.json',record)
        if not record['passed']:raise ArithmeticError('common lift scaling residual failed')
        export_common_lift(a.out,cfg,rex.sha_file(a.out/rex.MESH_FILENAME),saved,coefficients,result['metrics']['stokes_linear_residual'],result['reference_phase_seconds']['stokes_lift'])

if __name__=='__main__':main()
