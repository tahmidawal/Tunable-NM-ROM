"""Sub-minute synthetic CPU algebra checks; no NS reference/data gate is claimed."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from bank import initialize,FrozenBank
from geometry import CHANNEL,PIPE
from spaces import ordered_transform,weighted_pod,projection_floor,inf_sup,pressure_basis,canonicalize_eigenmodes,select_test_columns,weighted_gram
from weak import WeakRule,WeakOperator,force_moments
from solver import jacobian_check,solve_steady
from quadrature import fit_positive,output_moments,convection_candidate
from interpolation import CoefficientPCHIP
from metrics import field_errors,representation_admission
from timing import paired_cpu,outputs_sha256


def polynomial_rule():
    gx,gw=np.polynomial.legendre.leggauss(8)
    xx,yy=np.meshgrid((gx+1)/2,(gx+1)/2,indexing="ij")
    wx,wy=np.meshgrid(gw/2,gw/2,indexing="ij")
    x,y=xx.ravel(),yy.ravel()
    w=(wx*wy).ravel();n=len(w)
    lift=np.c_[1+x,y]
    dl=np.broadcast_to(np.eye(2),(n,2,2)).copy()
    g=np.stack((np.c_[x*x,x*y],np.c_[x*y,y*y]),axis=2)
    dg=np.zeros((n,2,2,2))
    dg[:,0,0,0]=2*x;dg[:,1,0,0]=y;dg[:,1,1,0]=x
    dg[:,0,0,1]=y;dg[:,0,1,1]=x;dg[:,1,1,1]=2*y
    v=np.zeros((n,2,6));dv=np.zeros((n,2,2,6))
    v[:,0,0]=1;v[:,1,1]=1;v[:,0,2]=x;v[:,1,3]=y
    dv[:,0,0,2]=1;dv[:,1,1,3]=1
    v[:,:,4]=np.c_[x*y,x*x];v[:,:,5]=np.c_[y*y,x*y]
    dv[:,0,0,4]=y;dv[:,0,1,4]=x;dv[:,1,0,4]=2*x
    dv[:,0,1,5]=2*y;dv[:,1,0,5]=y;dv[:,1,1,5]=x
    return WeakRule(w,lift,dl,x-y,g,dg,(1+x)[:,None],v,dv,np.c_[np.ones(n),x,y]),np.c_[x,y]


def main():
    start=time.perf_counter();report={"scope":"synthetic CPU infrastructure only","physical_admission":False}
    rng=np.random.default_rng(27102027)
    theta=np.linspace(0,2*np.pi,33,endpoint=False)
    boundary2=np.r_[np.c_[np.zeros(33),np.linspace(0,.41,33)],
                    np.c_[np.linspace(0,2.2,33),np.zeros(33)],
                    np.c_[np.linspace(0,2.2,33),np.full(33,.41)],
                    np.c_[.2+.05*np.cos(theta),.2+.05*np.sin(theta)]]
    boundary3=np.r_[np.c_[np.full(33,-5.),np.cos(theta),np.sin(theta)],
                    np.c_[np.linspace(-5,15,33),2.5*np.cos(theta),2.5*np.sin(theta)],
                    np.c_[.5*np.cos(theta),.5*np.sin(theta),np.zeros(33)]]
    for geom,boundary in ((CHANNEL,boundary2),(PIPE,boundary3)):
        model=initialize(geom,rank=3,width=7,depth=2)
        points=rng.uniform(size=(12,geom.dimension))
        if geom.dimension==3:
            points[:,0]=points[:,0]*20-5
        else:
            points*=np.array([2.2,.41])
        values,gradient=model.evaluate(points)
        fd=[]
        for j in range(geom.dimension):
            eps=np.eye(geom.dimension)[j]*1e-6
            fd.append((model.evaluate(points+eps)[0]-model.evaluate(points-eps)[0])/(2e-6))
        fd=np.stack(fd,axis=2)
        error=float(np.linalg.norm(fd-gradient)/max(np.linalg.norm(gradient),1e-12))
        trace=float(np.max(np.abs(model.evaluate(boundary)[0])))
        assert error<1e-8,(geom.kind,error)
        assert trace<1e-14,(geom.kind,trace)
        report[geom.kind]={"gradient_relative_difference":error,"exact_boundary_trace":trace}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/"bank.npz"
            sha=model.save(path);again=FrozenBank.load(path)
            assert np.array_equal(values,again.evaluate(points)[0])
            report[geom.kind]["roundtrip_sha256"]=sha
    rule,points=polynomial_rule();op=rule.tensor(chunk=13)
    state=np.array([.12,-.03,.7]);nu=.05
    direct=np.r_[*rule.direct(state,nu)]
    tensor=op.residual(state,nu,False)
    gap=float(np.max(np.abs(direct-tensor)))
    assert gap<1e-12,gap
    jac=jacobian_check(op,state,nu,np.array([.3,.4,.7]))
    assert jac["passed"],jac
    report["weak_operator"]={"direct_tensor_max_gap":gap,"jacobian":jac,"condition":op.condition(state,nu)}
    snapshots=np.einsum("ndr,sr->snd",rule.velocity,rng.normal(size=(9,2)))
    tr,order=ordered_transform(rule.velocity,snapshots,rule.weights)
    ordered=rule.velocity@tr
    gram=np.einsum("n,ndr,nds->rs",rule.weights,ordered,ordered)
    assert np.max(np.abs(gram-np.eye(2)))<1e-12
    pod,meta=weighted_pod(snapshots,rule.weights,2)
    _,floor=projection_floor(pod,snapshots,rule.weights)
    assert max(floor["squared_errors"])<1e-24
    report["spaces"]={"ordered_gram_gap":float(np.max(np.abs(gram-np.eye(2)))),"pod":meta,"floor":floor}
    assert np.array_equal(weighted_gram(np.ones((4,2,1)),np.ones(4)),[[8.]])
    assert np.array_equal(weighted_gram(np.ones((4,1)),np.ones(4)),[[4.]])
    pressures=np.array([(1+c)+(2-c)*points[:,0] for c in np.linspace(0,1,9)])
    pb,pmeta=pressure_basis(pressures,rule.weights,2)
    assert np.max(np.abs(pb[:,0]-1/np.sqrt(rule.weights.sum())))<1e-12
    assert np.max(np.abs(pb.T@(rule.weights[:,None]*pb)-np.eye(2)))<1e-12
    constantp,constmeta=pressure_basis(np.repeat(np.arange(1.,10.)[:,None],len(rule.weights),axis=1),rule.weights,8)
    assert constantp.shape[-1]==1,constmeta
    q,_=np.linalg.qr(rng.normal(size=(20,3)))
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    canonical,cm=canonicalize_eigenmodes(np.ones(3),q,np.eye(3))
    rotated,_=canonicalize_eigenmodes(np.ones(3),q@rotation,np.eye(3))
    assert np.max(np.abs(canonical-rotated))<1e-12
    selected,derivatives,ts=select_test_columns(rule.tests,rule.weights,6,rule.test_gradient)
    tg=np.einsum("n,niq,nir->qr",rule.weights,selected,selected)+np.einsum("n,nijq,nijr->qr",rule.weights,derivatives,derivatives)
    assert np.max(np.abs(tg-np.eye(6)))<1e-12
    report["test_enumeration"]={"pressure":pmeta,"constant_pressure_only":constmeta,"eigen_orientation":cm,"test_selection":ts}
    beta=inf_sup(np.array([[1.,0.]]),np.eye(2),np.eye(1),1.)
    assert beta["passed"]
    assert not inf_sup(np.array([[1e-8,0.]]),np.eye(2),np.eye(1),1.)["passed"]
    report["inf_sup_fixture"]=beta
    x,w=np.polynomial.legendre.leggauss(16)
    fit=fit_positive(np.array([np.ones_like(x),x,x*x]),w,max_support=4)
    assert fit["passed"],fit
    fit_small=fit_positive(np.array([np.ones_like(x),x,x*x]),w,max_support=1)
    assert not fit_small["passed"]
    try:
        convection_candidate(op,rule,fit_small)
    except ValueError:
        pass
    else:
        raise AssertionError("failed rule was deployed")
    moments=output_moments(rule,np.array([[.2,.1],[-.1,.1]]),points)
    assert moments.shape==(1+2+2*(2+4+6),len(rule.weights))
    report["quadrature"]={"synthetic_polynomial_fit":fit,"failed_fit_retained":fit_small,"moment_shape":list(moments.shape)}
    # Constant test-space and coefficient dimensions permit a known root fixture.
    root=WeakOperator(np.zeros((2,3,3)),np.array([[-1.,1.,0.],[-2.,0.,1.]]),
                      np.array([[1.],[0.]]),np.zeros(2),np.array([[-1.,1.,0.]]),
                      np.eye(2),np.eye(1),np.eye(2),np.eye(1),"mixed_Galerkin")
    result=solve_steady(root,1.,damping=0.)
    assert result["root_converged"] and np.max(np.abs(result["state"]-[1.,2.,0.]))<1e-12
    report["galerkin_root"]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
    re=15+1.25*np.arange(9)
    interp=CoefficientPCHIP(re,np.c_[re,re*re],{"split":"training"})
    assert np.array_equal(interp.evaluate(20.),[20.,400.])
    row=field_errors(rule.lift,np.ones(len(points)),rule.lift,np.ones(len(points)),rule.weights)
    assert representation_admission([row,row])["passed"]
    # Pressure p=x on a circle: force integral is -pi*a^2 in x, zero y.
    theta=np.linspace(0,2*np.pi,128,endpoint=False);radius=.5
    normal=np.c_[np.cos(theta),np.sin(theta)]
    fm=force_moments(np.full(128,2*np.pi*radius/128),normal,np.zeros((128,2,2,1)),
                     (radius*np.cos(theta))[:,None],np.zeros((128,2,2)),np.zeros(128),1.)
    assert np.max(np.abs(fm["pressure"][:,0]-[-2*np.pi*radius**2,0.]))<1e-12
    report["force_pressure_gradient_fixture"]=fm["pressure"].tolist()
    # Deliberately reuse the SAME arrays and nested accounting objects on every call.
    shared={"output_contract":"fixed_probes","outputs":{"velocity":np.zeros((2,2))},
            "work":{"calls":0,"nested":{"history":[]}},"component_seconds":{"solve":0.}}
    def reused_query():
        count=shared["work"]["calls"]+1
        shared["work"]["calls"]=count
        shared["work"]["nested"]["history"].append(count)
        shared["outputs"]["velocity"][:]=count
        shared["component_seconds"]["solve"]=count/1000.
        return shared
    timing=paired_cpu({"first":reused_query,"second":reused_query},"fixed_probes",1,burn_seconds=0.)
    shared["outputs"]["velocity"][:]=-999
    shared["work"]["calls"]=-999
    shared["work"]["nested"]["history"].append(-999)
    shared["component_seconds"]["solve"]=-999.
    for index,(row,actual) in enumerate(zip(timing["repetitions"],timing["actual_invocation_outputs"])):
        expected=index+3 # two discarded warmups precede the retained invocations
        assert np.all(actual["outputs"]["velocity"]==expected)
        assert not actual["outputs"]["velocity"].flags.writeable
        assert outputs_sha256(actual["outputs"])==row["output_sha256"]
        assert row["work"]["calls"]==expected
        assert row["work"]["nested"]["history"]==list(range(1,expected+1))
        assert row["component_seconds"]["solve"]==expected/1000.
    report["timing_reused_buffer_regression"]={"passed":True,"retained_invocations":len(timing["repetitions"]),
              "output_hashes":[row["output_sha256"] for row in timing["repetitions"]],
              "work_counts":[row["work"]["calls"] for row in timing["repetitions"]],
              "owner_mutated_after_collection":True,"stored_arrays_read_only":True,
              "burn_in_disabled_for_regression_only":True}
    report["seconds"]=time.perf_counter()-start
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
