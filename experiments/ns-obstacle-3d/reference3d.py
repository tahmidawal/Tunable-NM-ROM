"""Bounded three-component curved-FE reference/control implementation.

This initial revision is unexecuted and its smoke cannot admit physical references.
Grad(u)[i,j] = d_j u_i. Mixed residual uses gradient diffusion and minus signs for
both pressure couplings, no pressure mass penalty and no extra pressure gauge.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import ngsolve as ng
from netgen.csg import CSGeometry, Cylinder, Plane, Pnt, Sphere, Vec


def dump(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def guard(cfg, start, ndof=None):
    if rss_gib() > cfg["guards"]["rss_gib_max"]:
        raise RuntimeError("Peak RSS guard fired")
    if time.monotonic() - start > cfg["guards"]["case_seconds_max"]:
        raise RuntimeError("Case wall-time guard fired")
    if ndof is not None:
        # Deliberately conservative screening heuristic, not a proved fill bound.
        estimate = 128 * ndof**2 / 1024**3 + 4
        if ndof > cfg["guards"]["max_dofs"] or estimate > cfg["guards"]["factor_estimate_gib_max"]:
            raise RuntimeError(f"Pre-factorization guard: ndof={ndof}, estimated_GiB={estimate}")
        return estimate


def geometry(cfg, obstacle):
    g, m = cfg["geometry"], cfg["mesh"]
    pipe = Cylinder(Pnt(0,0,0), Pnt(1,0,0), g["pipe_radius"]).bc("wall").maxh(m["wall_maxh"])
    left = Plane(Pnt(g["x_in"],0,0), Vec(-1,0,0)).bc("inlet")
    right = Plane(Pnt(g["x_out"],0,0), Vec(1,0,0)).bc("outlet")
    fluid = pipe * left * right
    if obstacle:
        sphere = Sphere(Pnt(0,0,0), g["sphere_diameter"]/2).bc("sphere").maxh(m["sphere_maxh"])
        fluid = fluid - sphere
    geo = CSGeometry()
    geo.Add(fluid.mat("fluid"))
    mesh = ng.Mesh(geo.GenerateMesh(maxh=m["maxh"]))
    if mesh.ne > m["max_elements"]:
        raise RuntimeError(f"Element-count guard: {mesh.ne}")
    mesh.Curve(m["geometry_order"])
    got = set(mesh.GetBoundaries())
    want = {"inlet","outlet","wall"} | ({"sphere"} if obstacle else set())
    if got != want:
        raise RuntimeError(f"Boundary labels differ: {got} != {want}")
    return mesh


def integrate(cf, mesh, order):
    value=float(ng.Integrate(cf, mesh, order=order))
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite volume integral")
    return value


def surface(cf, mesh, boundary, order):
    value=float(ng.Integrate(cf, mesh, ng.BND, definedon=mesh.Boundaries(boundary), order=order))
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite surface integral")
    return value


def geometry_diagnostics(mesh, cfg, obstacle):
    g=cfg["geometry"]
    R, a, L = g["pipe_radius"], g["sphere_diameter"]/2, g["x_out"]-g["x_in"]
    q=cfg["fe"]["diagnostic_quadrature_order"]
    exact_v=math.pi*R*R*L-(4*math.pi*a**3/3 if obstacle else 0)
    rows=[]
    for order in (q,q+4):
        V=integrate(1,mesh,order)
        areas={name:surface(1,mesh,name,order) for name in sorted(set(mesh.GetBoundaries()))}
        exact={"inlet":math.pi*R*R,"outlet":math.pi*R*R,"wall":2*math.pi*R*L}
        if obstacle:
            exact["sphere"]=4*math.pi*a*a
        n=ng.specialcf.normal(3)
        normal_balance=[surface(n[i],mesh,".*",order) for i in range(3)]
        coordinates=ng.CoefficientFunction((ng.x,ng.y,ng.z))
        div_theorem=surface(coordinates*n,mesh,".*",order)
        sphere_distance_rms=None
        if obstacle:
            sphere_distance_rms=math.sqrt(max(0,surface((ng.sqrt(ng.x**2+ng.y**2+ng.z**2)-a)**2,mesh,"sphere",order)/areas["sphere"]))
        wall_distance=math.sqrt(max(0,surface((ng.sqrt(ng.y**2+ng.z**2)-R)**2,mesh,"wall",order)/areas["wall"]))
        rows.append({"order":order,"volume":V,"exact_volume":exact_v,"volume_relative_error":abs(V-exact_v)/exact_v,"areas":areas,"exact_areas":exact,"area_relative_errors":{k:abs(areas[k]-v)/v for k,v in exact.items()},"closed_normal_integrals":normal_balance,"divergence_theorem_x_dot_n_minus_3V":div_theorem-3*V,"sphere_distance_rms":sphere_distance_rms,"wall_distance_rms":wall_distance})
    return {"elements":mesh.ne,"vertices":mesh.nv,"curving_order":cfg["mesh"]["geometry_order"],"boundaries":list(mesh.GetBoundaries()),"integration_ladder":rows,"signed_curved_jacobian_certificate":"not yet implemented; geometry smoke alone cannot pass G2"}


def exact_data(cfg, case):
    """Analytical polynomial controls with manually differentiated derivatives.

    Manufactured u=(Y²+Z,Z²+X,X²+Y), p=X+Y²−Z. Every velocity component is
    nonzero; div(u)=0 and (u·grad)u != 0. Physical exact data are evaluated on
    the computational curved geometry, with exact natural outlet load.
    """
    x,y,z=ng.x,ng.y,ng.z
    R=cfg["geometry"]["pipe_radius"]
    nu=cfg["physics"]["nu"]
    U=cfg["physics"]["mean_speed"]
    if case == "manufactured":
        L=cfg["geometry"]["x_out"]-cfg["geometry"]["x_in"]
        X,Y,Z=x/L,y/R,z/R
        u=ng.CoefficientFunction((Y*Y+Z,Z*Z+X,X*X+Y))
        G=ng.CoefficientFunction((0,2*y/R**2,1/R,1/L,0,2*z/R**2,2*x/L**2,1/R,0), dims=(3,3))
        lap=ng.CoefficientFunction((2/R**2,2/R**2,2/L**2))
        p=X+Y*Y-Z
        gp=ng.CoefficientFunction((1/L,2*y/R**2,-1/R))
        f=G*u+gp-nu*lap
        return u,p,G,f
    u=ng.CoefficientFunction((2*U*(1-(y*y+z*z)/R**2),0,0))
    G=ng.CoefficientFunction((0,-4*U*y/R**2,-4*U*z/R**2,0,0,0,0,0,0),dims=(3,3))
    p=8*nu*U*(cfg["geometry"]["x_out"]-x)/R**2
    return u,p,G,ng.CoefficientFunction((0,0,0))


def constitutive(G,p,nu,kind):
    return nu*(G+G.trans if kind=="symmetric" else G)-p*ng.Id(3)


def residual_scales(X,V,cfg):
    U,D=cfg["physics"]["mean_speed"],cfg["geometry"]["sphere_diameter"]
    # Force/continuity units fixed across Newton, Reynolds and candidate policy.
    units=np.concatenate((np.full(V.ndof,U*U*D**2),np.full(X.ndof-V.ndof,U*D**2)))
    free=np.fromiter((bool(X.FreeDofs()[i]) for i in range(X.ndof)),dtype=bool,count=X.ndof)
    return units,free


def scaled_norm(vec,units,free):
    value=float(np.linalg.norm(vec.FV().NumPy()[free]/units[free]))
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite free-DOF residual")
    return value


def independent_residual(X,gfu,mesh,cfg,f,traction,conv_factor):
    v,q=X.TestFunction()
    u,p=gfu.components
    nu=cfg["physics"]["nu"]
    order=cfg["fe"]["quadrature_order"]
    kind=cfg["physics"]["outlet"]
    visc=nu*ng.InnerProduct(ng.Grad(u)+ng.Grad(u).trans if kind=="symmetric" else ng.Grad(u),ng.Grad(v))
    value=(visc+conv_factor*(ng.Grad(u)*u)*v-p*ng.div(v)-ng.div(u)*q-f*v)*ng.dx(bonus_intorder=order)
    value-=traction*v*ng.ds("outlet",bonus_intorder=order)
    rr=ng.LinearForm(X)
    rr+=value
    rr.Assemble()
    return rr.vec


def solve(mesh,cfg,case,folder,row,start):
    t0=time.monotonic()
    is_mms=case=="manufactured"
    obstacle=case in ("manufactured","sphere")
    bcs="inlet|wall"+("|sphere" if obstacle else "")
    order=cfg["fe"]["velocity_order"]
    if cfg["fe"]["pressure_order"]!=order-1:
        raise RuntimeError("Only Taylor-Hood degree k/k-1 accepted")
    V=ng.VectorH1(mesh,order=order,dirichlet=bcs)
    Q=ng.H1(mesh,order=order-1)
    X=V*Q
    row["dofs"]={"velocity":V.ndof,"pressure":Q.ndof,"total":X.ndof}
    row["factor_memory_screen_gib"]=guard(cfg,start,X.ndof)
    row["factor_memory_screen_is_bound"]=False
    dump(folder/"progress.json",row)
    u,p=X.TrialFunction()
    v,q=X.TestFunction()
    gfu=ng.GridFunction(X)
    bc=ng.GridFunction(X)
    uex,pex,Gex,f=exact_data(cfg,case)
    if is_mms:
        bc.components[0].Set(uex,definedon=mesh.Boundaries(bcs))
    else:
        bc.components[0].Set(uex,definedon=mesh.Boundaries("inlet"))
    gfu.vec.data=bc.vec
    nu=cfg["physics"]["nu"]
    n=ng.specialcf.normal(3)
    kind=cfg["physics"]["outlet"]
    if kind not in ("gradient","symmetric"):
        raise RuntimeError("Unknown outlet/diffusion form")
    if kind=="symmetric" and case=="empty":
        raise RuntimeError("Empty Poiseuille exact control requires gradient outlet")
    traction=constitutive(Gex,pex,nu,kind)*n if is_mms else ng.CoefficientFunction((0,0,0))
    alpha=ng.Parameter(0)
    qi=cfg["fe"]["quadrature_order"]
    visc=nu*ng.InnerProduct(ng.Grad(u)+ng.Grad(u).trans if kind=="symmetric" else ng.Grad(u),ng.Grad(v))
    a=ng.BilinearForm(X)
    a+=(visc+alpha*(ng.Grad(u)*u)*v-p*ng.div(v)-ng.div(u)*q)*ng.dx(bonus_intorder=qi)
    rhs=ng.LinearForm(X)
    rhs+=f*v*ng.dx(bonus_intorder=qi)+traction*v*ng.ds("outlet",bonus_intorder=qi)
    rhs.Assemble()
    units,free=residual_scales(X,V,cfg)
    r=gfu.vec.CreateVector()
    trial=gfu.vec.CreateVector()
    step=gfu.vec.CreateVector()
    linear=r.CreateVector()
    history=[]
    denom=None
    for factor in cfg["solver"]["continuation"]:
        alpha.Set(factor)
        for it in range(cfg["solver"]["max_newton"]+1):
            guard(cfg,start,X.ndof)
            a.Apply(gfu.vec,r)
            r.data-=rhs.vec
            raw=scaled_norm(r,units,free)
            if denom is None:
                denom=max(raw,cfg["solver"]["initial_residual_floor"])
            record={"convection_factor":factor,"iteration":it,"raw_free_residual":raw,"normalization":denom,"relative_residual":raw/denom,"rss_gib":rss_gib()}
            if raw/denom<=cfg["solver"]["relative_free_residual"]:
                history.append(record)
                break
            if it==cfg["solver"]["max_newton"]:
                raise RuntimeError("Newton iteration limit; no accepted state")
            ta=time.monotonic()
            a.AssembleLinearization(gfu.vec)
            guard(cfg,start,X.ndof)
            inverse=a.mat.Inverse(X.FreeDofs(),inverse=cfg["fe"]["inverse"])
            step.data=inverse*r
            del inverse
            linear.data=a.mat*step-r
            lin_raw=scaled_norm(linear,units,free)
            lin_denom=max(raw,1e-30)
            record.update(linear_residual=lin_raw,linear_denominator=lin_denom,linear_relative=lin_raw/lin_denom,jacobian_factor_solve_seconds=time.monotonic()-ta)
            if lin_raw/lin_denom>cfg["solver"]["linear_relative_residual"]:
                raise RuntimeError(f"Linear residual gate failed: {lin_raw/lin_denom}")
            before=gfu.vec.FV().NumPy().copy()
            accepted=False
            for bt in range(cfg["solver"]["max_backtracks"]+1):
                damping=2.0**(-bt)
                trial.FV().NumPy()[:]=before-damping*step.FV().NumPy()
                a.Apply(trial,r)
                r.data-=rhs.vec
                new=scaled_norm(r,units,free)
                if new<=(1-cfg["solver"]["armijo"]*damping)*raw:
                    gfu.vec.data=trial
                    accepted=True
                    record.update(damping=damping,backtracks=bt,accepted_raw_residual=new)
                    break
            history.append(record)
            row["history"]=history
            dump(folder/"progress.json",row)
            if not accepted:
                raise RuntimeError("Damped Newton failed to reduce residual")
        else:
            raise RuntimeError("Unreachable Newton exit")
    if cfg["solver"]["continuation"][-1]!=1.0:
        raise RuntimeError("Last solve did not use complete convection")
    a.Apply(gfu.vec,r)
    r.data-=rhs.vec
    ri=independent_residual(X,gfu,mesh,cfg,f,traction,1.0)
    independent=scaled_norm(ri,units,free)
    gap=r.CreateVector()
    gap.data=r-ri
    row["residual"]={"free_dof_normalization":"Euclidean row-unit-scaled free residual divided by max(initial exact-BC zero-interior residual, configured floor); velocity unit U²D², continuity UD²","denominator":denom,"independent_free_raw":independent,"independent_relative":independent/denom,"assembled_vector_gap":scaled_norm(gap,units,free)/denom,"dirichlet_coefficient_max_gap":float(np.max(np.abs((gfu.vec.FV().NumPy()-bc.vec.FV().NumPy())[~free]),initial=0))}
    if independent/denom>cfg["solver"]["relative_free_residual"] or row["residual"]["assembled_vector_gap"]>1e-12:
        raise RuntimeError("Independent residual check failed")
    row["history"]=history
    row["solve_seconds"]=time.monotonic()-t0
    row["diagnostics"]=diagnostics(mesh,cfg,case,gfu,bc,X,ri,f,uex,pex,Gex,folder)
    gfu.Save(str(folder/"solution.gf"))
    mesh.ngmesh.Save(str(folder/"mesh.vol"))
    np.savez_compressed(folder/"coefficients.npz",coefficients=gfu.vec.FV().NumPy(),free=free,velocity_dofs=V.ndof,pressure_dofs=Q.ndof)
    return gfu


def diagnostics(mesh,cfg,case,gfu,bc,X,ri,f,uex,pex,Gex,folder):
    u,p=gfu.components
    G=ng.Grad(u)
    nu=cfg["physics"]["nu"]
    U=cfg["physics"]["mean_speed"]
    D=cfg["geometry"]["sphere_diameter"]
    n=ng.specialcf.normal(3)
    sigma=constitutive(G,p,nu,"symmetric")
    gradstress=constitutive(G,p,nu,"gradient")
    obstacle=case in ("sphere","manufactured")
    out=[]
    for order in (cfg["fe"]["diagnostic_quadrature_order"],cfg["fe"]["diagnostic_quadrature_order"]+4):
        V=integrate(1,mesh,order)
        flux={name:surface(u*n,mesh,name,order) for name in sorted(set(mesh.GetBoundaries()))}
        div2=integrate(ng.div(u)**2,mesh,order)
        inlet_area=surface(1,mesh,"inlet",order)
        flux_scale=U*inlet_area
        values={"quadrature_order":order,"volume":V,"fluxes_outward":flux,"mass_imbalance":sum(flux.values()),"mass_denominator":flux_scale,"mass_relative":abs(sum(flux.values()))/flux_scale,"divergence_squared_integral":div2,"divergence_normalized":D*math.sqrt(max(div2,0))/(U*math.sqrt(V)),"outlet_backflow_flux":surface(ng.IfPos(-(u*n),-(u*n),0),mesh,"outlet",order)}
        # Exact essential coefficients and trace RMS are distinct diagnostics. A
        # continuum max trace has not been certified by this implementation smoke.
        traces={}
        for boundary in ("inlet","wall")+(("sphere",) if obstacle else ()):
            desired=uex if case=="manufactured" or boundary=="inlet" else ng.CoefficientFunction((0,0,0))
            traces[boundary]=math.sqrt(max(0,surface(ng.InnerProduct(u-desired,u-desired),mesh,boundary,order)/surface(1,mesh,boundary,order)))/U
        values["boundary_trace_rms_over_U"]=traces
        values["boundary_trace_max_certificate"]="not computed; RMS is not maximum"
        momentum_flux=[]; all_traction=[]; body=[]; mom_res=[]; mom_den=[]
        for i in range(3):
            mf={b:surface(u[i]*(u*n),mesh,b,order) for b in sorted(set(mesh.GetBoundaries()))}
            tf={b:surface((sigma*n)[i],mesh,b,order) for b in sorted(set(mesh.GetBoundaries()))}
            bf=integrate(f[i],mesh,order)
            momentum_flux.append(mf); all_traction.append(tf); body.append(bf)
            mom_res.append(sum(mf.values())-sum(tf.values())-bf)
            mom_den.append(max(sum(abs(v) for v in mf.values())+sum(abs(v) for v in tf.values())+abs(bf),1e-12*U*U*D*D))
        eps=(G+G.trans)*0.5
        diss=2*nu*integrate(ng.InnerProduct(eps,eps),mesh,order)
        kinetic={b:0.5*surface((u*u)*(u*n),mesh,b,order) for b in sorted(set(mesh.GetBoundaries()))}
        power={b:surface(u*(sigma*n),mesh,b,order) for b in sorted(set(mesh.GetBoundaries()))}
        fpower=integrate(u*f,mesh,order)
        energy=diss+sum(kinetic.values())-sum(power.values())-fpower
        eden=max(abs(diss)+sum(abs(v) for v in kinetic.values())+sum(abs(v) for v in power.values())+abs(fpower),1e-12*U**3*D*D)
        divkin=0.5*integrate((u*u)*ng.div(u),mesh,order)
        divp=integrate(p*ng.div(u),mesh,order)
        grad_diss=nu*integrate(ng.InnerProduct(G,G),mesh,order)
        gradpower={b:surface(u*(gradstress*n),mesh,b,order) for b in sorted(set(mesh.GetBoundaries()))}
        grad_identity=grad_diss+sum(kinetic.values())-divkin-divp-sum(gradpower.values())-fpower
        grad_energy_den=max(abs(grad_diss)+sum(abs(v) for v in kinetic.values())+abs(divkin)+abs(divp)+sum(abs(v) for v in gradpower.values())+abs(fpower),1e-12*U**3*D*D)
        gradient_momentum=[]
        for i in range(3):
            gradient_tractions={b:surface((gradstress*n)[i],mesh,b,order) for b in sorted(set(mesh.GetBoundaries()))}
            udiv=integrate(u[i]*ng.div(u),mesh,order)
            mismatch=sum(momentum_flux[i].values())-sum(gradient_tractions.values())-udiv-body[i]
            den=max(sum(abs(v) for v in momentum_flux[i].values())+sum(abs(v) for v in gradient_tractions.values())+abs(udiv)+abs(body[i]),1e-12*U*U*D*D)
            gradient_momentum.append({"gradient_tractions":gradient_tractions,"u_divu":udiv,"residual":mismatch,"denominator":den,"normalized":abs(mismatch)/den})
        values["gradient_momentum_with_finite_divergence"]=gradient_momentum
        values["momentum"]={"convective_boundary_flux":momentum_flux,"physical_boundary_traction":all_traction,"body_force_integrals":body,"residual":mom_res,"denominator":mom_den,"normalized": [abs(a)/b for a,b in zip(mom_res,mom_den)],"finite_divergence_u_divu": [integrate(u[i]*ng.div(u),mesh,order) for i in range(3)]}
        values["energy"]={"physical_strain_dissipation":diss,"kinetic_boundary_flux":kinetic,"physical_boundary_power":power,"body_power":fpower,"raw_residual":energy,"denominator":eden,"normalized":abs(energy)/eden,"half_u_squared_divu":divkin,"p_divu":divp,"gradient_dissipation":grad_diss,"gradient_boundary_power":gradpower,"gradient_identity_with_divergence_terms":grad_identity,"gradient_identity_denominator":grad_energy_den,"gradient_identity_normalized":abs(grad_identity)/grad_energy_den}
        if obstacle:
            fsym=[-surface((sigma*n)[i],mesh,"sphere",order) for i in range(3)]
            fgrad=[-surface((gradstress*n)[i],mesh,"sphere",order) for i in range(3)]
            fp=[surface(p*n[i],mesh,"sphere",order) for i in range(3)]
            unit=0.5*U*U*math.pi*D*D/4
            values["force"]={"physical":fsym,"gradient":fgrad,"pressure":fp,"viscous":[a-b for a,b in zip(fsym,fp)],"coefficient_denominator":unit,"physical_coefficients":[a/unit for a in fsym],"gradient_coefficients":[a/unit for a in fgrad]}
        if case=="manufactured":
            exact_stress=constitutive(Gex,pex,nu,"symmetric")
            exact_surface=[-surface((exact_stress*n)[i],mesh,"sphere",order) for i in range(3)]
            R=cfg["geometry"]["pipe_radius"]
            length=cfg["geometry"]["x_out"]-cfg["geometry"]["x_in"]
            solid_volume=4*math.pi*(D/2)**3/3
            exact_sphere_force=[solid_volume*(-1/length+2*nu/R**2),solid_volume*(2*nu/R**2),solid_volume*(1/R+2*nu/length**2)]
            values["manufactured_force_control"]={"exact_polynomial_on_FE_surface":exact_surface,"analytic_exact_sphere_surface_force":exact_sphere_force,"geometry_only_gap":[a-b for a,b in zip(exact_surface,exact_sphere_force)],"FE_solution_force_minus_exact_on_FE_geometry":[a-b for a,b in zip(values["force"]["physical"],exact_surface)]}
        if case in ("empty","manufactured"):
            eu=integrate(ng.InnerProduct(u-uex,u-uex),mesh,order)
            ur=integrate(uex*uex,mesh,order)
            ep=integrate((p-pex)**2,mesh,order)
            values["exact_error"]={"u_error_squared":eu,"u_reference_squared":ur,"u_relative":math.sqrt(max(0,eu)/max(ur,1e-30)),"p_error_squared":ep,"p_characteristic_rms":math.sqrt(max(ep,0))/(U*U*math.sqrt(V))}
        out.append(values)
    reactions=[]
    if obstacle:
        for i in range(3):
            test=ng.GridFunction(X)
            ei=ng.CoefficientFunction(tuple(float(j==i) for j in range(3)))
            test.components[0].Set(ei,definedon=mesh.Boundaries("sphere"))
            reactions.append(-float(np.dot(ri.FV().NumPy(),test.vec.FV().NumPy())))
    probes=[]
    for point in ((-D/2,0,0),(D/2,0,0)) if obstacle else ((0,0,0),(1,0,0)):
        try:
            mp=mesh(*point)
            probes.append({"point":point,"pressure":float(p(mp)),"velocity":[float(v) for v in u(mp)],"status":"evaluated_on_FE_geometry_no_nearest_node_replacement"})
        except Exception as exc:
            probes.append({"point":point,"status":"unresolved_exact_point","error":str(exc)})
    return {"integration_ladder":out,"gradient_weak_reaction_force":reactions,"pressure_probes":probes,"reference_admitted":False,"admission_note":"Requires independent multi-level convergence, trace-maximum and signed-curved-Jacobian gates; this smoke cannot admit references"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    campaign_path=Path(__file__).with_name(cfg["campaign_file"])
    if hashlib.sha256(campaign_path.read_bytes()).hexdigest()!=cfg["campaign_sha256"]:
        raise RuntimeError("Central campaign bytes differ from reviewed configuration")
    campaign=json.loads(campaign_path.read_text())
    c3=campaign["case3d"]
    expected_geometry={"sphere_diameter":2*c3["sphere_radius"],"pipe_radius":c3["pipe_radius"],"x_in":c3["xmin"],"x_out":c3["xmax"]}
    if cfg["geometry"]!=expected_geometry or cfg["physics"]["mean_speed"]!=c3["inlet_mean"] or cfg["physics"]["nu"]!=c3["pilot_nu"]:
        raise RuntimeError("Initial smoke differs from central fixed geometry/physics")
    if args.out.exists():
        raise RuntimeError("Output exists; inspect old attempt, use a new job directory")
    args.out.mkdir(parents=True)
    ng.SetNumThreads(cfg["threads"])
    np.random.seed(cfg["seed"])
    start=time.monotonic()
    report={"schema":"ns-obstacle-3d-control-results-v1","config":cfg,"config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_commit":Path("SOURCE_COMMIT.txt").read_text().strip() if Path("SOURCE_COMMIT.txt").exists() else None,"job_id":os.environ.get("SLURM_JOB_ID"),"python":sys.version,"ngsolve_version":ng.__version__,"numpy_version":np.__version__,"platform":platform.platform(),"cpu":platform.processor(),"thread_env":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},"arithmetic":"real64 FE/NumPy; GPU JAX preflight external; no FE GPU acceleration claim","cases":[],"reference_admitted":False}
    try:
        report["cpu_info"]=subprocess.check_output(["lscpu"],text=True,timeout=10)
        report["gpu_info"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,memory.total","--format=csv,noheader"],text=True,timeout=10)
    except Exception as exc:
        report["hardware_query_error"]=str(exc)
    dump(args.out/"summary.json",report)
    try:
        with ng.TaskManager():
            for name in cfg["cases"]:
                if time.monotonic()-start>cfg["guards"]["total_seconds_max"]:
                    raise RuntimeError("Total code wall-time guard fired")
                folder=args.out/name
                folder.mkdir()
                row={"case":name,"status":"started","seed":cfg["seed"],"mesh_seed_scope":"NumPy seed recorded; Netgen deterministic default seed/control requires runtime verification"}
                report["cases"].append(row)
                case_start=time.monotonic()
                dump(args.out/"summary.json",report)
                mesh=geometry(cfg,name!="empty")
                row["geometry"]=geometry_diagnostics(mesh,cfg,name!="empty")
                row["geometry_seconds"]=time.monotonic()-case_start
                dump(folder/"progress.json",row)
                if name=="geometry_sphere":
                    mesh.ngmesh.Save(str(folder/"mesh.vol"))
                elif name in ("empty","manufactured","sphere"):
                    solve(mesh,cfg,name,folder,row,case_start)
                else:
                    raise RuntimeError(f"Unknown case {name}")
                row.update(status="diagnostic_complete",elapsed_seconds=time.monotonic()-case_start,peak_rss_gib=rss_gib())
                dump(folder/"result.json",row)
                dump(args.out/"summary.json",report)
        report["status"]="smoke_complete_not_reference_admission"
    except Exception as exc:
        report["status"]="failed"
        report["exception"]=str(exc)
        report["traceback"]=traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"]=time.monotonic()-start
        report["peak_rss_gib"]=rss_gib()
        dump(args.out/"summary.json",report)
        files=sorted(p for p in args.out.rglob("*") if p.is_file() and p.name!="OUTPUTS.sha256")
        (args.out/"OUTPUTS.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p.relative_to(args.out))+"\n" for p in files))

if __name__=="__main__":
    main()
