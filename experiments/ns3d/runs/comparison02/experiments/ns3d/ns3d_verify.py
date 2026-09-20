"""Prospective operator/reference checks. Save measured values, including failures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
import ns3d_fom as F
import ns3d_independent as I


def rel(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b),1e-300))


def emit(report, name, passed, **values):
    report['gates'][name] = dict(passed=bool(passed), **values)
    print(name, json.dumps(report['gates'][name]), flush=True)


def forcing_table(t, nu, geom, context):
    return context[1][jnp.rint(t/context[0]).astype(jnp.int32)]


def operator_checks(n=12):
    start = time.time()
    out = dict(gates={})
    geom = F.geometry(n)
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(3,n,n,n))
    h = F.project(F.fft(jnp.asarray(raw)), geom)
    u = np.asarray(F.ifft(h))
    idem = rel(F.project(h,geom), h)
    div = float(np.asarray(F.diagnostics(jnp.asarray(u[None]),geom)[2])[0])
    emit(out,'projector',idem<1e-11 and div<1e-11,idempotence=idem,divergence=div)
    adv_ref = I.advective(I.transform(u),I.setup(n))
    adv_actual = F.nonlinear(h,geom)
    parity = rel(adv_actual,adv_ref)
    emit(out,'independent_advection',parity<1e-10,relative=parity)
    energy = abs(float(np.real(np.vdot(h,adv_actual))))/(
        np.linalg.norm(h)*np.linalg.norm(adv_actual))
    emit(out,'energy_advection',energy<1e-11,relative_transfer=float(energy))
    raw_div = float(np.linalg.norm(np.sum(np.asarray(geom[0])*I.transform(raw),axis=0))/
                    np.linalg.norm(I.transform(raw)))
    emit(out,'projection_negative_control',raw_div>1,unprojected_divergence=raw_div)
    wrong_wave=np.asarray(geom[0]).copy();wrong_wave[2]=0
    wrong_geom=(jnp.asarray(wrong_wave),jnp.asarray(np.sum(wrong_wave**2,axis=0)),geom[2])
    missing_axis=rel(F.nonlinear(h,wrong_geom),adv_ref)
    emit(out,'third_axis_negative_control',missing_axis>1e-2,relative_rhs_difference=missing_axis)

    # Analytic mixed 3D solution; forcing computed solely by independent formulas.
    n_mms = 12
    mgeom = F.geometry(n_mms)
    nu, horizon = .01,.08
    mms_errors=[]
    neg=None
    for steps in (20,40,80):
        dt = horizon/steps
        exact0 = I.manufactured(n_mms,0)[0]
        ft=[]
        for t in np.arange(steps+1)*dt:
            uu,ut,lap,adv = I.manufactured(n_mms,float(t))
            ft.append(I.solenoidal(I.transform(ut+adv-nu*lap),I.setup(n_mms)))
        context = (jnp.asarray(dt),jnp.asarray(np.stack(ft)))
        run = F.make_solver(dt,steps,steps,forcing=forcing_table)
        states = np.asarray(run(jnp.asarray(exact0),nu,mgeom,context))
        exact1=I.manufactured(n_mms,horizon)[0]
        mms_errors.append(rel(states[-1],exact1))
        if steps==40:
            wrong=F.make_solver(dt,steps,steps,forcing=forcing_table,adv_sign=-1.)
            neg=rel(np.asarray(wrong(jnp.asarray(exact0),nu,mgeom,context))[-1],exact1)
    orders=np.log2(np.asarray(mms_errors[:-1])/np.asarray(mms_errors[1:]))
    emit(out,'manufactured_temporal_order',np.all((orders>1.7)&(orders<2.3)),
         errors=mms_errors,orders=orders.tolist())
    emit(out,'advection_sign_negative_control',neg>10*mms_errors[1] and neg>.001,
         wrong_sign_error=neg,correct_error=mms_errors[1])

    # Independent RK4 with a different nonlinear form, not merely a copied stepper.
    phys=F.parameters(202609200,1)[0]
    ini=F.initial(n,phys)
    dt=.0005
    indep=I.solve(ini,phys[-1],dt,20,10)
    jaxrk=F.make_solver(dt,20,10,scheme='rk4')
    own=np.asarray(jaxrk(jnp.asarray(ini),phys[-1],geom))
    parity=rel(own,indep)
    emit(out,'independent_trajectory',parity<1e-10,relative=parity)
    diagnostics=tuple(np.asarray(a) for a in F.diagnostics(jnp.asarray(own),geom))
    components=diagnostics[3][0]
    grads=diagnostics[4][0]
    nonlin=float(diagnostics[5][0])
    change=rel(own[-1],own[0])
    emit(out,'genuine_3d',np.min(components)>1e-3 and np.min(grads)>1e-3
         and nonlin>1e-2 and change>1e-3,component_rms=components.tolist(),
         directional_derivative_norms=grads.tolist(),nonlinear_rhs_over_state=nonlin,
         evolved_relative_change=change)
    out['seconds']=time.time()-start
    out['passed']=all(g['passed'] for g in out['gates'].values())
    return out


def reference_checks(n=24,dt=.001,horizon=.2,count=4,outdir=None):
    out=dict(gates={},cases=[],n=n,dt=dt,horizon=horizon,reference_budget=.005)
    params=F.parameters(202609200,count)
    steps=round(horizon/dt)
    assert steps%5==0
    runners={
        'coarse':(n,dt,F.make_solver(dt,steps,steps//5)),
        'time_half':(n,dt/2,F.make_solver(dt/2,steps*2,steps*2//5)),
        'fine':(2*n,dt/2,F.make_solver(dt/2,steps*2,steps*2//5)),
        'fine_time_half':(2*n,dt/4,F.make_solver(dt/4,steps*4,steps*4//5))}
    saved={}
    for index,param in enumerate(params):
        states={}
        for label,(mesh,_,run) in runners.items():
            print(f'reference case={index} arm={label} N={mesh}',flush=True)
            states[label]=np.asarray(run(jnp.asarray(F.initial(mesh,param)),param[-1],F.geometry(mesh)))
        fine=F.restrict_fields(states['fine_time_half'],n)
        coarse=states['coarse']
        scale=np.linalg.norm(fine[0])
        errors={label:float(np.max(np.linalg.norm(delta.reshape(6,-1),axis=1))/scale)
                for label,delta in (
                    ('total',coarse-fine),
                    ('temporal',coarse-states['time_half']),
                    ('spatial',states['time_half']-fine),
                    ('fine_temporal',F.restrict_fields(states['fine']-states['fine_time_half'],n)))}
        diag=tuple(np.asarray(a) for a in F.diagnostics(jnp.asarray(coarse),F.geometry(n)))
        spec=F.geometry(n)
        h0=F.fft(jnp.asarray(coarse[0]))
        linear=np.stack([np.asarray(F.ifft(h0*jnp.exp(-param[-1]*spec[1]*t)))
                         for t in np.linspace(0,horizon,6)])
        removed=F.relative_errors(coarse,linear,coarse[0])
        def rhs_sizes(u):
            uh=F.fft(u)
            return jnp.linalg.norm(F.nonlinear(uh,spec)),jnp.linalg.norm(param[-1]*spec[1]*uh)
        nl_size,vis_size=tuple(np.asarray(a) for a in jax.vmap(rhs_sizes)(jnp.asarray(coarse)))
        case=dict(index=index,parameter=param.tolist(),errors=errors,
                  energies=diag[0].tolist(),divergence=diag[2].tolist(),
                  initial_vorticity=float(diag[1][0]),
                  velocity_rms=np.sqrt(2*diag[0]).tolist(),
                  nonlinear_rhs_norm=nl_size.tolist(),viscous_rhs_norm=vis_size.tolist(),
                  nonlinear_over_viscous=(nl_size/np.maximum(vis_size,1e-300)).tolist(),
                  advection_removed_error=removed.tolist(),
                  max_speed=float(np.max(np.sqrt(np.sum(coarse*coarse,axis=1)))))
        case['cfl']=case['max_speed']*dt*n
        case['finite']=bool(all(np.all(np.isfinite(a)) for a in states.values()))
        out['cases'].append(case)
        saved.update({f'case{index}_{label}':a for label,a in states.items()})
        saved[f'case{index}_advection_removed']=linear
        print('reference_case',json.dumps(case),flush=True)
    maximum=max(c['errors']['total'] for c in out['cases'])
    emit(out,'physical_reference_budget',maximum<=.005 and all(c['finite'] for c in out['cases']),
         worst_total_discrepancy=maximum,budget=.005)
    maxdiv=max(max(c['divergence']) for c in out['cases'])
    max_energy_increase=max(max(np.diff(c['energies'])) for c in out['cases'])
    emit(out,'trajectory_constraints',maxdiv<1e-10 and max_energy_increase<=1e-10,
         max_divergence=maxdiv,max_energy_increment=max_energy_increase)
    out['passed']=all(g['passed'] for g in out['gates'].values())
    if outdir is not None:
        Path(outdir).mkdir(parents=True,exist_ok=True)
        np.savez_compressed(Path(outdir)/'reference_fields.npz',parameters=params,**saved)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--reference',action='store_true')
    args=p.parse_args()
    report=operator_checks()
    if args.reference and report['passed']:
        report['reference']=reference_checks(outdir=Path(args.out).parent)
        report['passed'] &= report['reference']['passed']
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
