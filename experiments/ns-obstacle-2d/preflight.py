"""Cheap pinned NGSolve runtime checks before any actual flow case."""
from __future__ import annotations
import argparse,json,traceback
from pathlib import Path
import numpy as np


def mapped_probe(mesh, center=None, sample_order=2):
    import ngsolve as ng
    record={'volume':None,'boundary':None,'normal_checks':{},'api_status':'unverified'}
    normal=ng.specialcf.normal(2)
    for el in mesh.Elements(ng.VOL):
        ip=ng.IntegrationRule(el.type,sample_order)[0]
        mip=mesh.GetTrafo(el)(ip)
        jac=np.asarray(mip.jacobi)
        det=float(np.linalg.det(jac))
        if det<=0 or float(ip.weight*mip.measure)<=0:
            raise ArithmeticError('mapped volume preflight has invalid Jacobian/weight')
        record['volume']={'point':list(mip.point),'jacobian':jac.tolist(),'determinant':det,
                          'weight':float(ip.weight*mip.measure)}
        break
    directions={'inlet':[-1.,0.],'outlet':[1.,0.],'wall_top':[0.,1.],'wall_bottom':[0.,-1.]}
    for el in mesh.Elements(ng.BND):
        label=el.mat
        for ip in ng.IntegrationRule(el.type,sample_order):
            mip=mesh.GetTrafo(el)(ip)
            nn=np.asarray(normal(mip),dtype=float)
            xy=np.asarray(mip.point,dtype=float)
            if float(ip.weight*mip.measure)<=0:
                raise ArithmeticError('nonpositive mapped boundary weight')
            if record['boundary'] is None:
                record['boundary']={'label':label,'point':xy.tolist(),'normal':nn.tolist(),'weight':float(ip.weight*mip.measure)}
            if label in directions:
                error=float(np.linalg.norm(nn-np.asarray(directions[label])))
                record['normal_checks'][label]=max(record['normal_checks'].get(label,0.),error)
                if error>1e-10:raise ArithmeticError(f'wrong {label} normal: {nn}')
            elif label=='obstacle':
                if center is None:raise ValueError('obstacle center required')
                radial=xy-np.asarray(center)
                cosine=float(nn@radial/np.linalg.norm(radial))
                error=abs(cosine+1.)
                record['normal_checks'][label]=max(record['normal_checks'].get(label,0.),error)
                if cosine>-.99:raise ArithmeticError('wrong obstacle normal sign')
            else:
                raise ValueError(f'unknown boundary label {label}')
    if record['volume'] is None or record['boundary'] is None:
        raise AssertionError('missing volume or boundary mapped point')
    if set(directions)-set(record['normal_checks']):
        raise AssertionError('missing rectangle boundary labels')
    record['api_status']='passed_sampled_mapping_and_normals'
    return record


def run():
    import ngsolve as ng
    from netgen.geom2d import SplineGeometry
    ng.SetNumThreads(16)
    geo=SplineGeometry()
    geo.AddRectangle((0,0),(1,1),bcs=('wall_bottom','outlet','wall_top','inlet'))
    mesh=ng.Mesh(geo.GenerateMesh(maxh=.6));mesh.Curve(3)
    result=mapped_probe(mesh)
    V=ng.H1(mesh,order=2,dirichlet='wall_bottom|wall_top|inlet|outlet')
    u,v=V.TnT();a=ng.BilinearForm(V);a+=(ng.grad(u)*ng.grad(v)+u*v)*ng.dx
    f=ng.LinearForm(V);f+=v*ng.dx
    with ng.TaskManager():
        a.Assemble();f.Assemble()
        if not any(V.FreeDofs()):raise AssertionError('tiny factorization has no free DOFs')
        inv=a.mat.Inverse(V.FreeDofs(),inverse='umfpack')
        solution=f.vec.CreateVector();solution.data=inv*f.vec
        residual=f.vec.CreateVector();residual.data=a.mat*solution-f.vec
    free=np.array([bool(x) for x in V.FreeDofs()])
    n=float(np.linalg.norm(residual.FV().NumPy()[free]));d=float(np.linalg.norm(f.vec.FV().NumPy()[free]))
    result['umfpack']={'free_dofs':int(sum(free)),'residual_numerator':n,'residual_denominator':d,'relative_residual':n/d}
    if n/d>1e-10:raise ArithmeticError('tiny UMFPACK residual failure')
    result['passed']=True
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    target=Path(args.out)
    if target.exists():raise FileExistsError('preflight output exists')
    try:
        result=run()
    except Exception as exc:
        target.write_text(json.dumps({'passed':False,'error':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise
    target.write_text(json.dumps(result,indent=2)+'\n')
