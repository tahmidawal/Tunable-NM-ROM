"""Curved mixed FE steady controls. No time stepping or ROM training.

Run only under the reviewed configuration. NGSolve solve kernels are CPU kernels.
All NumPy arrays are float64; Grad has [velocity component, spatial coordinate].
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
import time
import traceback

import numpy as np
from scalars import scalar_value, scalar_observations
import reference_export as rex


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metric(numerator, denominator, definition):
    numerator, denominator = float(numerator), float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"invalid metric {definition}: {numerator}, {denominator}")
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator, "definition": definition}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(cfg, out):
    import ngsolve as ng
    from netgen.geom2d import SplineGeometry
    # main() sets the requested thread count before entering TaskManager.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'config.json', cfg)
    started = time.perf_counter()
    U, D, nu = float(cfg['U']), float(cfg['D']), float(cfg['nu'])
    H, xmin, xmax = cfg['H'], cfg['xmin'], cfg['xmax']
    L = xmax-xmin
    kind = cfg['kind']
    reference_controls=rex.enabled(cfg)
    volume_point_limit=rex.volume_point_cap(cfg)
    batched_reference_export=rex.batched_export_enabled(cfg)
    if reference_controls:
        campaign_file=cfg.get('campaign_file','campaign-A2.json')
        if campaign_file not in ('campaign-A2.json','g2/campaign-A6.json'):raise ValueError('unregistered campaign source')
        campaign_path=Path(__file__).parent/campaign_file
        if sha(campaign_path)!=cfg['campaign_sha256']:raise ValueError('central gate binding mismatch')
        central=json.loads(campaign_path.read_text())
    obstacle = kind in ('dfg','manufactured')
    if kind not in ('dfg', 'channel', 'manufactured'):
        raise ValueError(kind)
    if cfg['velocity_order'] < 2 or cfg['pressure_order'] != cfg['velocity_order']-1:
        raise ValueError('Taylor-Hood k/(k-1) required')
    if not (1 <= cfg['newton_max'] <= 30):
        raise ValueError('Newton cap must be between 1 and 30')
    if cfg['outlet'] not in ('gradient', 'symmetric'):
        raise ValueError('unknown outlet')
    if cfg['outlet'] == 'symmetric' and kind != 'dfg':
        raise ValueError('only named DFG-domain-sensitivity case uses symmetric outlet')
    geo = SplineGeometry()
    geo.AddRectangle((xmin, 0), (xmax, H), bcs=('wall_bottom', 'outlet', 'wall_top', 'inlet'))
    if obstacle:
        geo.AddCircle(tuple(cfg['center']), cfg['radius'], leftdomain=0,
                      rightdomain=1, bc='obstacle', maxh=cfg['obstacle_h'])
    # Netgen optimize/mesh operations are single-threaded here; recorded seed is
    # retained as provenance, not a claim that Netgen accepts an RNG seed option.
    mesh,mesh_origin=rex.make_or_reuse_mesh(cfg,out,geo,ng)
    if mesh.ne > cfg['max_elements']:
        raise MemoryError(f'pre-space element guard: {mesh.ne}')
    from preflight import mapped_probe
    from oriented_normals import fluid_outward_normal
    normal, normal_certificate = fluid_outward_normal(mesh, cfg['center'] if obstacle else None,
        order=cfg['diagnostic_order'], evidence_path=out/'normal_orientation.json')
    dump(out/'early_geometry_api_checks.json', mapped_probe(mesh,cfg['center'] if obstacle else None, normal=normal))
    V = ng.VectorH1(mesh, order=cfg['velocity_order'], dirichlet='inlet|wall.*|obstacle')
    Q = ng.H1(mesh, order=cfg['pressure_order'])
    X = V*Q
    if X.ndof > cfg['max_dofs']:
        raise MemoryError(f'pre-assembly DOF guard: {X.ndof}')
    if reference_controls:
        layout,digest=rex.layout_record(mesh,V,Q,ng)
        layout_evidence={'layout':layout,'sha256':digest}
        dump(out/'fe_layout.json',layout_evidence)
        if mesh_origin.get('source_layout_sha256',digest)!=digest:
            raise ValueError('immutable mesh reuse changed FE DOF layout')
        orientation=rex.affine_gradient_orientation(mesh,cfg,ng)
        dump(out/'affine_gradient_orientation.json',orientation)
        if not orientation['passed']:raise ArithmeticError('non-symmetric affine gradient orientation failed')
    (u, p), (v, q) = X.TnT()
    g = ng.GridFunction(X)
    uh, ph = g.components
    x, y = ng.x, ng.y
    raw_normal = ng.specialcf.normal(2)  # provenance only; forces/fluxes use verified normal
    I = ng.Id(2)
    grad = ng.Grad
    exact_u = exact_p = force = traction = None
    witness = None
    if kind == 'channel':
        exact_u = ng.CF((6*U*y*(H-y)/H**2, 0))
        exact_p = 12*nu*U*(xmax-x)/H**2
        force = ng.CF((0, 0))
        traction = ng.CF((0, 0))
    elif kind == 'manufactured':
        xx, yy = (x-xmin)/L, y/H
        psi = U*H*(xx**2*yy + xx*yy**2 + 0.25*xx**3*yy**3)
        exact_u = ng.CF((psi.Diff(y), -psi.Diff(x)))
        exact_p = U**2*(xx+yy+xx*yy)
        gu = ng.CF(tuple(exact_u[i].Diff(c) for i in range(2)
                         for c in (x, y)), dims=(2, 2))
        lap = ng.CF(tuple(sum(exact_u[i].Diff(c).Diff(c) for c in (x,y)) for i in range(2)))
        gp = ng.CF((exact_p.Diff(x), exact_p.Diff(y)))
        force = gu*exact_u - nu*lap + gp
        traction = (nu*gu-exact_p*I)*normal
        witness_path = Path(__file__).with_name('manufactured_witness.json')
        witness = json.loads(witness_path.read_text())
        if not np.allclose(witness['obstacle_force']['gradient'],witness['obstacle_force']['physical_symmetric'],rtol=1e-12,atol=1e-15):
            raise AssertionError('independent manufactured closed-surface stress integrals disagree')
        if any(abs(cfg[k]-witness['configuration'][k]) > 1e-14 for k in ('U','nu','H','xmin','xmax')):
            raise ValueError('manufactured witness is tied to its exact configuration')
        checks = []
        for record in witness['samples']:
            mip = mesh(*record['point'])
            values = np.r_[np.asarray(exact_u(mip)), scalar_value(exact_p(mip),'manufactured_pressure_point'),
                            np.asarray(gu(mip)).ravel(), np.asarray(force(mip))]
            expected = np.array(record['u']+[record['p']]+record['gradient']+record['forcing'])
            checks.append(metric(np.linalg.norm(values-expected), max(np.linalg.norm(expected), 1.0),
                                 'CF analytic derivatives versus independent SymPy witness'))
        dump(out/'manufactured_derivative_checks.json', checks)
        if max(a['value'] for a in checks) > 1e-12:
            raise AssertionError('manufactured analytical derivative witness mismatch')
    else:
        force, traction = ng.CF((0,0)), ng.CF((0,0))
    boundary_u = exact_u if exact_u is not None else ng.CF((6*U*y*(H-y)/H**2,0))
    if kind == 'manufactured':
        uh.Set(boundary_u, definedon=mesh.Boundaries('inlet|wall.*|obstacle'))
    else:
        uh.Set(boundary_u, definedon=mesh.Boundaries('inlet'))
    prescribed_FE_trace=None
    if reference_controls:
        prescribed_FE_trace=ng.GridFunction(V)
        prescribed_FE_trace.vec.data=uh.vec
    ds_out = ng.ds('outlet', bonus_intorder=cfg['assembly_bonus_order'])
    dx = ng.dx(bonus_intorder=cfg['assembly_bonus_order'])
    viscous = nu*ng.InnerProduct(grad(u),grad(v))
    if cfg['outlet'] == 'symmetric':
        viscous = 2*nu*ng.InnerProduct(ng.Sym(grad(u)),ng.Sym(grad(v)))
    stokes = ng.BilinearForm(X)
    stokes += (viscous - p*ng.div(v)-q*ng.div(u))*dx
    rhs = ng.LinearForm(X)
    rhs += force*v*dx + traction*v*ds_out
    rhs.Assemble()
    residual_form = ng.BilinearForm(X)
    residual_form += (viscous + (grad(u)*u)*v-p*ng.div(v)-q*ng.div(u))*dx
    free = np.array([bool(b) for b in X.FreeDofs()], dtype=bool)
    row_scale = np.r_[np.full(V.ndof, U**2*D), np.full(Q.ndof, U*D)]
    if len(row_scale) != X.ndof:
        raise AssertionError('mixed block DOF layout mismatch')
    def arr(vec):
        return vec.FV().NumPy()
    def free_norm(vec):
        return float(np.linalg.norm((arr(vec)/row_scale)[free]))
    def true_residual():
        # Independent assembly path: state is a coefficient, not symbolic trial.
        form = ng.LinearForm(X)
        vf = nu*ng.InnerProduct(grad(uh),grad(v))
        if cfg['outlet'] == 'symmetric':
            vf = 2*nu*ng.InnerProduct(ng.Sym(grad(uh)),ng.Sym(grad(v)))
        form += (vf+(grad(uh)*uh)*v-ph*ng.div(v)-q*ng.div(uh)-force*v)*dx
        form += -traction*v*ds_out
        form.Assemble()
        return form.vec
    def residual():
        r = g.vec.CreateVector()
        residual_form.Apply(g.vec,r)
        r.data -= rhs.vec
        return r
    def factor(mat):
        # Conservative declared accounting model, not a solver-memory guarantee.
        nnz = int(mat.nze)
        estimated = nnz*16*cfg['factor_fill_estimate'] + X.ndof*8*32
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        record = {'nnz':nnz,'estimated_factor_bytes':estimated,'current_maxrss_bytes':rss}
        memory_checks.append(record)
        dump(out/'memory_checks.json',memory_checks)
        if nnz > cfg['max_nnz'] or estimated+rss > cfg['memory_stop_gib']*2**30:
            raise MemoryError(f'pre-factorization guard {record}')
        return mat.Inverse(X.FreeDofs(), inverse=cfg['inverse_backend'])
    memory_checks = []
    stokes_started=time.perf_counter()
    stokes.Assemble()
    r = rhs.vec.CreateVector()
    r.data = rhs.vec-stokes.mat*g.vec
    inv = factor(stokes.mat)
    delta = g.vec.CreateVector()
    delta.data = inv*r
    linear = r.CreateVector()
    linear.data = stokes.mat*delta-r
    stokes_linear_metric = metric(free_norm(linear), max(free_norm(r),1e-30), 'scaled linear residual / scaled RHS')
    if stokes_linear_metric['value'] > cfg['linear_relative_tolerance']:
        raise ArithmeticError('Stokes direct solve residual failed')
    g.vec.data += delta
    del inv
    stokes_state = arr(g.vec).copy()
    stokes_completed=time.perf_counter()
    if reference_controls:g.Save(str(out/'stokes.sol'))
    nonlinear_started=time.perf_counter()
    r = residual()
    initial = free_norm(r)
    denominator = max(initial, cfg['residual_floor'])
    histories = []
    converged = False
    for it in range(cfg['newton_max']+1):
        r = residual()
        value = free_norm(r)
        rec = {'iteration':it,'scaled_residual_norm':value,'residual_denominator':denominator,
               'relative_residual':value/denominator}
        histories.append(rec)
        dump(out/'newton_history.json', histories)
        if value/denominator <= cfg['nonlinear_relative_tolerance']:
            converged = True
            break
        if it == cfg['newton_max']:
            break
        residual_form.AssembleLinearization(g.vec)
        inv = factor(residual_form.mat)
        delta.data = inv*r
        linear.data = residual_form.mat*delta-r
        linmet = metric(free_norm(linear),max(value,1e-30),'scaled linear residual / scaled RHS')
        rec['linear_residual'] = linmet
        if linmet['value'] > cfg['linear_relative_tolerance']:
            raise ArithmeticError('Newton direct solve residual failed')
        base = arr(g.vec).copy()
        accepted = False
        for damping in cfg['damping_sequence']:
            arr(g.vec)[:] = base-damping*arr(delta)
            candidate = free_norm(residual())
            if candidate <= (1-cfg['armijo']*damping)*value:
                accepted = True
                rec['damping'] = damping
                rec['step_l2_coefficients'] = float(np.linalg.norm(damping*arr(delta)))
                break
        del inv
        if not accepted:
            arr(g.vec)[:] = base
            raise ArithmeticError('no declared Newton damping accepted')
    dump(out/'newton_history.json', histories)
    dump(out/'memory_checks.json', memory_checks)
    if not converged:
        raise ArithmeticError('Newton iteration limit: no accepted solution')
    nonlinear_completed=time.perf_counter()
    independent = true_residual()
    independent_norm = free_norm(independent)
    independent_difference = metric(np.linalg.norm(((arr(independent)-arr(r))/row_scale)[free]),
                                    denominator,'independent coefficient residual versus operator apply')
    if independent_norm/denominator > cfg['nonlinear_relative_tolerance'] or independent_difference['value'] > 1e-11:
        raise ArithmeticError('independent residual did not certify returned solution')
    # Directional Jacobian check at a deterministic perturbed admissible state.
    rng = np.random.default_rng(cfg['seed'])
    reference = arr(g.vec).copy()
    direction = rng.normal(size=X.ndof)
    direction[~free] = 0
    direction[:V.ndof] *= U
    direction[V.ndof:] *= U**2
    direction /= np.linalg.norm(direction/np.r_[np.full(V.ndof,U),np.full(Q.ndof,U**2)])
    arr(g.vec)[:] += cfg['jacobian_state_perturbation']*direction
    perturbed = arr(g.vec).copy()
    residual_form.AssembleLinearization(g.vec)
    vector = g.vec.CreateVector()
    arr(vector)[:] = direction
    action = vector.CreateVector()
    action.data = residual_form.mat*vector
    jv = (arr(action)/row_scale)[free].copy()
    checks=[]
    plus_coefficients=[]; minus_coefficients=[]; plus_raw=[]; minus_raw=[]
    for h in cfg['jacobian_steps']:
        arr(g.vec)[:] = perturbed+h*direction
        plus_residual = residual()  # Own the native vector while its NumPy view is used.
        plus_coefficients.append(arr(g.vec).copy())
        plus_raw.append(arr(plus_residual).copy())
        plus = (arr(plus_residual)/row_scale)[free].copy()
        arr(g.vec)[:] = perturbed-h*direction
        minus_residual = residual()
        minus_coefficients.append(arr(g.vec).copy())
        minus_raw.append(arr(minus_residual).copy())
        minus = (arr(minus_residual)/row_scale)[free].copy()
        fd=(plus-minus)/(2*h)
        checks.append({'h':h,'agreement':metric(np.linalg.norm(fd-jv),max(np.linalg.norm(jv),1e-30),
                                              'centered difference versus assembled Jacobian action'),
                       'signal_norm':float(np.linalg.norm(plus-minus)),
                       'signal_floor':float(100*np.finfo(float).eps*max(np.linalg.norm(plus),np.linalg.norm(minus),1.0))})
    arr(g.vec)[:] = reference
    # Persist independent replay evidence before any diagnostic acceptance failure.
    np.savez_compressed(out/'jacobian_arrays.npz', reference=reference, perturbed=perturbed,
                        free_mask=free, row_scale=row_scale, direction=direction,
                        action_raw=arr(action).copy(), steps=np.asarray(cfg['jacobian_steps'],dtype=np.float64),
                        plus_coefficients=np.asarray(plus_coefficients), minus_coefficients=np.asarray(minus_coefficients),
                        plus_residual_raw=np.asarray(plus_raw), minus_residual_raw=np.asarray(minus_raw))
    dump(out/'jacobian_checks.json', checks)
    useful=[a for a in checks if a['signal_norm']>a['signal_floor']]
    values=[a['agreement']['value'] for a in useful]
    # At a quadratic residual, centered differences are exact in real arithmetic;
    # a uniformly small roundoff plateau is expected, not an artificial O(h²) fit.
    plateau=len(useful)>=2 and max(values)<=1e-8
    reducing=len(useful)>=2 and any(values[j+1]<=values[j] for j in range(len(values)-1))
    if not useful or min(values)>1e-6 or not (plateau or reducing):
        raise ArithmeticError('Jacobian finite difference mismatch or unresolved signal/plateau')
    diagnostics_started=time.perf_counter()
    # Quadrature-specific metrics and boundary traces are sampled independently of DOFs.
    qo = cfg['diagnostic_order']
    def integ(cf, boundary=None, order=qo):
        value=ng.Integrate(cf,mesh,definedon=mesh.Boundaries(boundary),order=order) if boundary else ng.Integrate(cf,mesh,order=order)
        return scalar_value(value,'boundary_integral' if boundary else 'volume_integral')
    volume=integ(1)
    exact_volume=L*H-(math.pi*cfg['radius']**2 if obstacle else 0)
    boundary_measures={b:integ(1,b) for b in sorted(set(mesh.GetBoundaries()))}
    geometry={'volume':volume,'volume_error':metric(abs(volume-exact_volume),exact_volume,'area error / exact fluid area'),
              'boundary_measures':boundary_measures,'element_count':mesh.ne,'ndof':X.ndof,'velocity_ndof':V.ndof,
              'pressure_ndof':Q.ndof,'geometry_order':cfg['geometry_order'],'coordinate_derivative_convention':'component,coordinate'}
    if obstacle:
        circumference=2*math.pi*cfg['radius']
        geometry['obstacle_perimeter_error']=metric(abs(boundary_measures['obstacle']-circumference),circumference,'perimeter error / exact circumference')
    # Full positive mapped rule exporter; checks mapped Jacobians and FE traces.
    points=[]; weights=[]; vals=[]; grads=[]; pressures=[]; boundaries=[];jacobians=[]
    volume_cache=rex.native_rule_cache(cfg,'volume',mesh,out/rex.MESH_FILENAME,layout_evidence['sha256'],ng,X) if reference_controls else None
    if volume_cache is not None:
        cached_fields=rex._mapped_state(mesh,uh,ph,qo,False,ng,geometry_cache=volume_cache,state=g)
        points,weights,jacobians,vals,grads,pressures=[cached_fields[k] for k in ('points','weights','jacobians','velocity','velocity_gradient','pressure')]
        min_det=float(np.linalg.det(jacobians).min())
        geometry['immutable_rule_cache']={'descriptor':cfg['immutable_rule_cache']['volume'],'certificate':volume_cache.certificate,'offline_seconds':volume_cache.offline_seconds,'state_fields_recomputed':True}
    else:
        min_det=float('inf')
        for el in mesh.Elements(ng.VOL):
            ir=ng.IntegrationRule(el.type,qo)
            tr=mesh.GetTrafo(el)
            if reference_controls and len(points)+len(ir)>volume_point_limit:raise MemoryError('reference export exceeds reviewed producer volume-point cap')
            batch=rex.field_values_batched(uh,ph,grad(uh),tr,ir) if batched_reference_export else None
            for index,ip in enumerate(ir):
                if reference_controls and len(points)>=volume_point_limit:
                    raise MemoryError('reference export exceeds reviewed producer volume-point cap')
                mip=tr(ip)
                jac=np.array(mip.jacobi,dtype=float,copy=True)
                determinant=float(np.linalg.det(jac))
                min_det=min(min_det,determinant)
                w=float(ip.weight*mip.measure)
                if not math.isfinite(w) or not math.isfinite(determinant) or w<=0 or determinant<=0:
                    raise ArithmeticError('nonpositive volume Jacobian or mapped quadrature weight')
                points.append(list(mip.point));weights.append(w)
                jacobians.append(jac.copy())
                # Persist an owning gradient array, never a possible native view.
                if batch is None:
                    vals.append(list(uh(mip)));grads.append(np.array(grad(uh)(mip),dtype=float,copy=True).reshape(2,2));pressures.append(scalar_value(ph(mip),'volume_pressure_point'))
                else:
                    vals.append(batch[0][index].tolist());grads.append(batch[1][index].copy());pressures.append(scalar_value(float(batch[2][index]),'volume_pressure_point'))
    # A VectorH1 volume derivative is not a BND differential operator.
    # Trace only the field derivative; retain normals on the boundary transform.
    gradient_on_boundary=ng.BoundaryFromVolumeCF(grad(uh))
    trace_max=0.0; obstacle_distance_max=0.0; obstacle_normal_dot=[]
    for el in mesh.Elements(ng.BND):
        ir=ng.IntegrationRule(el.type,qo)
        tr=mesh.GetTrafo(el)
        label=el.mat
        for ip in ir:
            if reference_controls and len(boundaries)>=rex.CAPS['boundary_points']:
                raise MemoryError('reference export exceeds reviewed producer boundary-point cap')
            mip=tr(ip)
            uv=np.array(uh(mip),dtype=float,copy=True); xy=np.array(mip.point,dtype=float,copy=True)
            bv=np.asarray(boundary_u(mip)) if (kind=='manufactured' or label=='inlet') else np.zeros(2)
            if label!='outlet':
                trace_max=max(trace_max,float(np.linalg.norm(uv-bv)))
            if float(ip.weight*mip.measure)<=0:
                raise ArithmeticError('nonpositive boundary rule weight')
            if label=='obstacle':
                radial=xy-np.array(cfg['center'])
                obstacle_normal_dot.append(float(np.asarray(normal(mip))@radial/np.linalg.norm(radial)))
                obstacle_distance_max=max(obstacle_distance_max,abs(np.linalg.norm(xy-np.array(cfg['center']))-cfg['radius']))
            boundaries.append({'point':xy.tolist(),'weight':float(ip.weight*mip.measure),'normal':list(normal(mip)),
                               'raw_normal':list(raw_normal(mip)), 'normal_orientation_factor':normal_certificate['factors'][label],
                               'label':label,'velocity':uv.tolist(),'pressure':scalar_value(ph(mip),'boundary_pressure_point'),
                               'velocity_gradient':np.asarray(gradient_on_boundary(mip)).reshape(2,2).tolist()})
    geometry['minimum_sampled_jacobian_determinant']=min_det
    geometry['max_sampled_obstacle_distance']=obstacle_distance_max
    geometry['mapped_weight_sum']=sum(weights)
    distances,obstacle_rule=rex.obstacle_quadrature(points,weights,cfg)
    geometry['analytic_obstacle_quadrature']=obstacle_rule
    if obstacle:
        geometry['sampled_obstacle_normal_dot_radial_range']=[min(obstacle_normal_dot),max(obstacle_normal_dot)]
        if max(obstacle_normal_dot)>-.99:
            raise ArithmeticError('obstacle outward-fluid normal orientation invalid')
    extra_arrays={'jacobians':np.asarray(jacobians,dtype=np.float64)}
    if distances is not None:extra_arrays['signed_obstacle_distance']=distances
    if reference_controls:extra_arrays['prescribed_FE_trace_coefficients']=prescribed_FE_trace.vec.FV().NumPy().copy()
    dump(out/'geometry_checks.json',geometry)
    np.savez_compressed(out/'full_rule.npz',points=np.asarray(points),weights=np.asarray(weights),
                        velocity=np.asarray(vals),velocity_gradient=np.asarray(grads),pressure=np.asarray(pressures),
                        stokes_coefficients=stokes_state,solution_coefficients=reference,**extra_arrays)
    dump(out/'boundary_rule.json',boundaries)
    g.Save(str(out/'solution.sol'))
    # Gradient-form and symmetric-stress diagnostics both keep nonzero boundary work.
    gu=grad(uh); div=ng.div(uh)
    div_boundary=ng.BoundaryFromVolumeCF(div)
    tau_boundary=nu*gradient_on_boundary-ph*I
    sigma_boundary=nu*(gradient_on_boundary+gradient_on_boundary.trans)-ph*I
    qin=-integ(uh*normal,'inlet');qout=integ(uh*normal,'outlet')
    metrics={'mass':metric(abs(qout-qin),max(abs(qin),1e-30),'abs(Qout-Qin)/abs(Qin)'),
             'divergence':metric(D*math.sqrt(max(integ(div*div),0)),U*math.sqrt(volume),'D||div u||/(U sqrt(volume))'),
             'boundary_trace':metric(trace_max,U,'maximum sampled essential FE trace error/U'),
             'Qin':qin,'Qout':qout,
             'mass_scope':('physical inlet/outlet mass' if kind!='manufactured' else 'inlet/outlet only; not a manufactured-control gate'),
             'global_boundary_flux':metric(abs(sum(integ(uh*normal,b) for b in boundary_measures)),
                                         max(sum(integ(ng.sqrt((uh*normal)**2),b) for b in boundary_measures),1e-12*U*D),
                                         'abs(total boundary flux) / total absolute boundary flux'),
             'independent_residual':metric(independent_norm,denominator,'scaled free-DOF residual / max(initial Stokes scaled residual, fixed floor)'),
             'independent_assembly_agreement':independent_difference,
             'pressure_mean':integ(ph)/volume,
             'pressure_gauge':'fixed by natural outlet, no pin and no shift',
             'row_scale_momentum':U**2*D,'row_scale_continuity':U*D,
             'residual_initial_stokes_norm':initial,'residual_floor':cfg['residual_floor'],
             'residual_mode':('absolute in scaled row units' if initial<=cfg['residual_floor'] else 'relative to initial Stokes residual in scaled row units'),
             'stokes_linear_residual':stokes_linear_metric}
    # Raw and finite-divergence-corrected physical energy equation.
    eps=ng.Sym(gu)
    terms={'dissipation':2*nu*integ(ng.InnerProduct(eps,eps)),
           'convection_boundary':sum(.5*integ((uh*uh)*(uh*normal),b) for b in boundary_measures),
           'physical_traction_work':-sum(integ(uh*(sigma_boundary*normal),b) for b in boundary_measures),
           'body_force_work':-integ(force*uh),
           'finite_divergence_convection':-.5*integ((uh*uh)*div),
           'finite_divergence_pressure':-integ(ph*div),
           # Δu formulation differs from div(2ε) by grad(div u).
           'finite_divergence_viscous':(-nu*integ(div*div)+sum(nu*integ(div_boundary*(uh*normal),b) for b in boundary_measures)) if cfg['outlet']=='gradient' else 0.0}
    energy_floor=1e-12*U**3*D
    physical_parts={f'convective_flux_{b}':.5*integ((uh*uh)*(uh*normal),b) for b in boundary_measures}
    physical_parts.update({f'traction_work_{b}':-integ(uh*(sigma_boundary*normal),b) for b in boundary_measures})
    physical_parts.update({k:z for k,z in terms.items() if k not in ('convection_boundary','physical_traction_work','finite_divergence_viscous')})
    if cfg['outlet']=='gradient':
        physical_parts['finite_divergence_viscous_volume']=-nu*integ(div*div)
        physical_parts.update({f'finite_divergence_viscous_{b}':nu*integ(div_boundary*(uh*normal),b) for b in boundary_measures})
    denominator_energy=max(sum(abs(z) for z in physical_parts.values()),energy_floor)
    metrics['energy']={'terms':terms,'individual_contributions':physical_parts,'physical_raw':metric(abs(sum(terms[k] for k in ('dissipation','convection_boundary','physical_traction_work','body_force_work'))),denominator_energy,'raw physical balance / sum absolute terms'),
                       'corrected':metric(abs(sum(terms.values())),denominator_energy,'finite-divergence-corrected physical balance / sum absolute terms')}
    gt={'dissipation':nu*integ(ng.InnerProduct(gu,gu)),
        'convection_boundary':sum(.5*integ((uh*uh)*(uh*normal),b) for b in boundary_measures),
        'gradient_traction_work':-sum(integ(uh*(tau_boundary*normal),b) for b in boundary_measures),
        'body_force_work':-integ(force*uh),
        'finite_divergence_convection':-.5*integ((uh*uh)*div),
        'finite_divergence_pressure':-integ(ph*div)}
    if cfg['outlet']=='symmetric':
        gt['symmetric_pde_adjustment']=nu*integ(div*div)-sum(nu*integ(div_boundary*(uh*normal),b) for b in boundary_measures)
    gradient_parts={f'convective_flux_{b}':.5*integ((uh*uh)*(uh*normal),b) for b in boundary_measures}
    gradient_parts.update({f'traction_work_{b}':-integ(uh*(tau_boundary*normal),b) for b in boundary_measures})
    gradient_parts.update({k:z for k,z in gt.items() if k not in ('convection_boundary','gradient_traction_work','symmetric_pde_adjustment')})
    if cfg['outlet']=='symmetric':
        gradient_parts['symmetric_div_volume']=nu*integ(div*div)
        gradient_parts.update({f'symmetric_div_{b}':-nu*integ(div_boundary*(uh*normal),b) for b in boundary_measures})
    metrics['gradient_energy']={'terms':gt,'individual_contributions':gradient_parts,'corrected':metric(abs(sum(gt.values())),max(sum(abs(z) for z in gradient_parts.values()),energy_floor),'A2 gradient-form energy including body forcing and finite divergence')}
    gm=[]
    for i in range(2):
        terms_m={'convection':sum(integ(uh[i]*(uh*normal),b) for b in boundary_measures),
                 'gradient_traction':-sum(integ((tau_boundary*normal)[i],b) for b in boundary_measures),
                 'body_force':-integ(force[i]),'finite_divergence_convection':-integ(uh[i]*div)}
        if cfg['outlet']=='symmetric':
            terms_m['symmetric_pde_adjustment']=-sum(nu*integ(div_boundary*normal[i],b) for b in boundary_measures)
        pieces={f'convective_flux_{b}':integ(uh[i]*(uh*normal),b) for b in boundary_measures}
        pieces.update({f'gradient_traction_{b}':-integ((tau_boundary*normal)[i],b) for b in boundary_measures})
        pieces.update({k:z for k,z in terms_m.items() if k not in ('convection','gradient_traction','symmetric_pde_adjustment')})
        if cfg['outlet']=='symmetric':
            pieces.update({f'symmetric_div_{b}':-nu*integ(div_boundary*normal[i],b) for b in boundary_measures})
        gm.append({'terms':terms_m,'individual_contributions':pieces,'corrected':metric(abs(sum(terms_m.values())),max(sum(abs(z) for z in pieces.values()),1e-12*U**2*D),'A2 gradient-form momentum including body forcing and finite divergence')})
    metrics['gradient_momentum']=gm
    momentum=[]
    for i in range(2):
        mt={'convection':sum(integ(uh[i]*(uh*normal),b) for b in boundary_measures),
            'physical_traction':-sum(integ((sigma_boundary*normal)[i],b) for b in boundary_measures),
            'body_force':-integ(force[i]),'finite_divergence_convection':-integ(uh[i]*div),
            'finite_divergence_viscous':sum(nu*integ(div_boundary*normal[i],b) for b in boundary_measures) if cfg['outlet']=='gradient' else 0.0}
        pieces={f'convective_flux_{b}':integ(uh[i]*(uh*normal),b) for b in boundary_measures}
        pieces.update({f'physical_traction_{b}':-integ((sigma_boundary*normal)[i],b) for b in boundary_measures})
        pieces.update({k:z for k,z in mt.items() if k not in ('convection','physical_traction','finite_divergence_viscous')})
        if cfg['outlet']=='gradient':
            pieces.update({f'finite_divergence_viscous_{b}':nu*integ(div_boundary*normal[i],b) for b in boundary_measures})
        momentum.append({'terms':mt,'individual_contributions':pieces,'corrected':metric(abs(sum(mt.values())),max(sum(abs(z) for z in pieces.values()),1e-12*U**2*D),'physical momentum corrected for discrete divergence')})
    metrics['momentum']=momentum
    if exact_u is not None:
        eu=uh-exact_u; ep=ph-exact_p
        unorm=math.sqrt(max(integ(exact_u*exact_u),0))
        metrics['velocity_error']=metric(math.sqrt(max(integ(eu*eu),0)),max(unorm,1e-12*U*math.sqrt(volume)),'weighted velocity L2 error / reference L2')
        metrics['pressure_reference_relative_secondary']=metric(math.sqrt(max(integ(ep*ep),0)),max(math.sqrt(max(integ(exact_p*exact_p),0)),1e-12*U**2*math.sqrt(volume)),'secondary pressure L2 / reference pressure L2')
        metrics['pressure_error']=metric(math.sqrt(max(integ(ep*ep),0)),U**2*math.sqrt(volume),'unshifted pressure RMS error / U²')
        exact_grad=ng.CF(tuple(exact_u[i].Diff(c) for i in range(2) for c in (x,y)),dims=(2,2))
        eg=gu-exact_grad
        metrics['velocity_gradient_error']=metric(math.sqrt(max(integ(ng.InnerProduct(eg,eg)),0)),max(math.sqrt(max(integ(ng.InnerProduct(exact_grad,exact_grad)),0)),1e-12*U/D*math.sqrt(volume)),'gradient L2 error / reference gradient L2')
    if obstacle:
        forces={}
        rforce=true_residual()
        reaction=[]
        for i in range(2):
            test=ng.GridFunction(X)
            unit=ng.CF(tuple(1.0 if j==i else 0.0 for j in range(2)))
            test.components[0].Set(unit,definedon=mesh.Boundaries('obstacle'))
            reaction.append(-scalar_value(ng.InnerProduct(rforce,test.vec),'reaction_inner_product'))
        norm=0.5*U**2*D
        forces['normalization']=norm
        forces['reaction_force']=reaction
        forces['reaction_coefficient']=[z/norm for z in reaction]
        for stress_name,stress in [('gradient',tau_boundary),('physical_symmetric',sigma_boundary)]:
            for order in [qo,qo+4]:
                ff=[-integ((stress*normal)[i],'obstacle',order) for i in range(2)]
                forces[f'{stress_name}_order{order}']={'force':ff,'coefficient':[z/norm for z in ff]}
        # Probe points are geometrical spline vertices; evaluate FE interpolation,
        # never nearest-node substitution or a fitted pressure shift.
        cx,cy=cfg['center'];rad=cfg['radius']
        probes=[(cx-rad,cy),(cx+rad,cy)]
        pressures_probe=[];probe_errors=[]
        for pt in probes:
            try:
                mp=mesh(*pt)
                if mp.nr<0:raise ValueError('exact pressure probe is outside FE mesh')
                value=scalar_value(ph(mp),'exact_pressure_probe')
                if not math.isfinite(value):raise ValueError('nonfinite exact pressure probe')
                pressures_probe.append(value);probe_errors.append(None)
            except Exception as exc:
                pressures_probe.append(None);probe_errors.append(repr(exc))
        probes_valid=all(z is not None for z in pressures_probe)
        forces['pressure_probes']={'points':probes,'values':pressures_probe,
            'difference':pressures_probe[0]-pressures_probe[1] if probes_valid else None,
            'valid':probes_valid,'errors':probe_errors,
            'G2_pressure_gate_eligible':probes_valid,
            'evaluation':'exact coordinate FE interpolation; no nearest node, shift or nudge'}
        if kind=='manufactured':
            forces['analytic_force']=witness['obstacle_force']
            for stress_name in ['gradient','physical_symmetric']:
                expected=np.asarray(witness['obstacle_force'][stress_name])
                measured=np.asarray(forces[f'{stress_name}_order{qo+4}']['force'])
                forces[f'{stress_name}_analytic_error']=metric(np.linalg.norm(measured-expected),U**2*D,'force difference norm / U²D; includes source-driven manufactured force')
            forces['reaction_analytic_error']=metric(np.linalg.norm(np.asarray(reaction)-np.asarray(witness['obstacle_force']['gradient'])),U**2*D,'reaction force minus analytic gradient force / U²D')
        metrics['forces']=forces
    if kind=='channel':
        wall_force=[-integ((sigma_boundary*normal)[i],'wall.*') for i in range(2)]
        expected_wall=12*nu*U*L/H
        metrics['wall_shear_sign_control']={'force_on_both_walls':wall_force,'analytic_axial_force':expected_wall,
             'axial_error':metric(abs(wall_force[0]-expected_wall),abs(expected_wall),'force on both walls versus positive analytic Poiseuille drag')}
    wake=rex.wake_centerline(mesh,uh,cfg,ng)
    if wake is not None:dump(out/'wake_centerline.json',wake)
    reload_started=time.perf_counter()
    reload_evidence=None
    if reference_controls:
        trace=rex.essential_trace_certificate(mesh,uh,prescribed_FE_trace,cfg['velocity_order'],U,ng)
        dump(out/'essential_trace_certificate.json',trace)
        metrics['prescribed_FE_essential_trace']=trace
        if trace['upper_bound_over_U']>central['reference_gates']['essential_trace_over_U']:
            raise ArithmeticError('prescribed FE essential trace bound fails central threshold')
        if obstacle and not obstacle_rule['no_obstacle_quadrature']:
            raise ArithmeticError('full mapped rule contains points inside exact obstacle; none were silently removed')
        reload_evidence=rex.reload_certificate(out,cfg,mesh,layout_evidence,ng)
        if cfg.get('export_common_stokes_lift',False):
            from shared_lift import export_common_lift
            export_common_lift(out,cfg,sha(out/rex.MESH_FILENAME),layout_evidence,stokes_state,stokes_linear_metric,
                               stokes_completed-stokes_started)
    storage_evidence={'mode':'expanded_full_fields'}
    if cfg.get('reference_storage_mode')=='coefficient_primary':
        from reference_primary import prepare_primary
        storage_evidence=prepare_primary(out,cfg,layout_evidence,reference,stokes_state,prescribed_FE_trace.vec.FV().NumPy().copy(),reload_evidence)
    completed=time.perf_counter();elapsed=completed-started
    result={'status':'completed_control_not_reference_admission','configuration':cfg,
            'scalar_return_contract':scalar_observations(),
            'geometry':geometry,'metrics':metrics,'elapsed_seconds':elapsed,
            'normal_orientation':{'passed':normal_certificate['passed'],'factors':normal_certificate['factors'],
                                  'certificate_sha256':sha(out/'normal_orientation.json')},
            'mesh_origin':mesh_origin,'reload_checks':reload_evidence,'storage_contract':storage_evidence,
            'timing_use':'reference generation and diagnostics only; not a tuned paired FOM query benchmark',
            'reference_phase_seconds':{'geometry_spaces_fields_forms':stokes_started-started,
                'stokes_lift':stokes_completed-stokes_started,'stokes_checkpoint':nonlinear_started-stokes_completed,
                'newton_query':nonlinear_completed-nonlinear_started,
                'independent_residual_and_Jacobian':diagnostics_started-nonlinear_completed,
                'reference_diagnostics_and_export':reload_started-diagnostics_started,
                'trace_certificate_and_reload':completed-reload_started,'total':elapsed},
            'cpu_work':'NGSolve finite elements and NumPy; GPU allocation is not GPU acceleration',
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'versions':{'ngsolve':ng.__version__,'numpy':np.__version__,'python':platform.python_version()},
            'threading':{'requested':cfg['threads'],'TaskManager':True,'actual_effective_concurrency':'unmeasured; no performance claim'},
            'provenance':{'execution_source_commit':Path('SOURCE_COMMIT.txt').read_text().strip() if Path('SOURCE_COMMIT.txt').exists() else None,'source_sha256':sha(__file__),'config_sha256':sha(out/'config.json'),
                          'job_id':os.environ.get('SLURM_JOB_ID'),'hostname':platform.node()}}
    dump(out/'result.json',result)
    if reference_controls:
        import importlib.metadata
        dump(out/'EXPORT-MANIFEST.json',{'schema':'ns-obstacle-reference-producer-v1','physical_admission':False,
             'mesh_filename':rex.MESH_FILENAME,'mesh_format':rex.MESH_FORMAT,'reload_policy':rex.RELOAD_POLICY,
             'case':{'kind':kind,'Re':U*D/nu,'split':cfg.get('split','control'),'release':cfg['release']},
             'campaign_sha256':cfg['campaign_sha256'],'layout_sha256':layout_evidence['sha256'],'storage_contract':storage_evidence,
             'versions':{name:importlib.metadata.version(name) for name in ('ngsolve','netgen-mesher','netgen-occt','ngsolve-openblas','numpy')},
             'producer_source_sha256':{name:sha(Path(__file__).with_name(name)) for name in (('fom.py','reference_export.py','shared_lift.py','preflight.py','scalars.py','oriented_normals.py')+
                 (('immutable_rule.py',) if cfg.get('immutable_rule_cache') else ())+
                 (('reference_primary.py','reference_scope.py') if storage_evidence['mode']=='coefficient_primary' else (('reference_scope.py',) if cfg['release'] in ('G2_fixed_finite_channel_training_export','G2_fixed_finite_channel_control_candidate') else ())))},
             'files':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*') if storage_evidence['mode']=='coefficient_primary' else out.iterdir()) if p.is_file() and not (storage_evidence['mode']=='coefficient_primary' and p.name=='full_rule.npz')}})
    if storage_evidence['mode']=='coefficient_primary':
        from reference_primary import split_runtime_inventory
        manifest=json.loads((out/'EXPORT-MANIFEST.json').read_text())
        runtime,audit=split_runtime_inventory(manifest['files'])
        manifest.update(runtime_files=runtime,audit_files=audit,runtime_validation_policy='explicit_primary_runtime_v1',audit_retention='All audit files remain mandatory raw evidence in the complete verified archive; runtime loading does not claim to re-audit them or admit the reference.')
        dump(out/'EXPORT-MANIFEST.json',manifest)
        (out/'full_rule.npz').unlink()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest' or os.environ.get('JAX_ENABLE_X64')!='true':
        raise RuntimeError('required precision environment missing')
    target=Path(args.out)
    if target.exists():
        raise FileExistsError('refusing to overwrite an existing attempt directory')
    try:
        import ngsolve as ng
        ng.SetNumThreads(int(cfg['threads']))
        with ng.TaskManager():
            result=run_case(cfg,target)
        print(json.dumps({'status':result['status'],'out':str(target),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except Exception as exc:
        target.mkdir(parents=True,exist_ok=True)
        dump(target/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'source_sha256':sha(__file__),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'job_id':os.environ.get('SLURM_JOB_ID'),'scalar_return_contract':scalar_observations()})
        raise

if __name__=='__main__':
    main()
