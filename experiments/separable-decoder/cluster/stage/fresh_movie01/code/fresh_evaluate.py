"""Unseen-state reconstruction, physical rollouts and fresh linear baselines."""
from functools import partial
import json
import numpy as np
from scipy.linalg import expm
import jax
import jax.numpy as jnp
from fresh_fom import energy, damping_ratio, positive_laplacian
from fresh_models import head_apply, head_geometry
from fresh_rom import rollout


def stats(values, threshold=.10):
    x = np.asarray(values, dtype=float).ravel()
    finite = x[np.isfinite(x)]
    return {"count": int(len(x)), "nonfinite": int(len(x)-len(finite)), "summary_population":"Finite cases only; nonfinite cases count as failures and outliers.", "mean": None if not len(finite) else float(finite.mean()), "median": None if not len(finite) else float(np.median(finite)), "worst": None if not len(finite) else float(finite.max()), "outlier_threshold": threshold, "outliers": int(np.sum(~np.isfinite(x) | (x > threshold)))}


def physical_metrics_finite(metrics):
    """Call before adding masked/undefined phase arrays to the diagnostics."""
    return all(np.all(np.isfinite(value)) for value in metrics.values())


@partial(jax.jit, static_argnames=("kind",))
def batched_geometry(p, frozen, z, btargets, kind):
    def one(zz, target):
        fn = lambda zzz: head_apply(p, frozen, zzz, kind)
        a = fn(zz)
        jac = jax.jacfwd(fn)(zz)
        q, rr = jnp.linalg.qr(jac, mode="reduced")
        w = jax.scipy.linalg.solve_triangular(rr, q.T@target, lower=False)
        singular = jnp.linalg.svd(jac, compute_uv=False)
        return a, jac@w, w, singular[-1]/singular[0]
    return jax.vmap(one)(z, btargets)


def reconstruction_metrics(p, frozen, z, projected, kind, out):
    a = projected["a"].reshape(-1, projected["a"].shape[-1])
    b = projected["b"].reshape(a.shape)
    predicted, tangent, w, rank = map(np.asarray, batched_geometry(p, frozen, jnp.asarray(z), jnp.asarray(b), kind))
    uerr = np.sqrt(np.sum((predicted-a)**2, axis=-1)+projected["u_floor_squared"].ravel())/projected["u_scale"]
    verr = np.sqrt(np.sum((tangent-b)**2, axis=-1)+projected["v_floor_squared"].ravel())/projected["v_scale"]
    np.savez_compressed(out/"reconstruction.npz", fitted_z=z, tangent_w=w, predictions=predicted, projected_truth=a, projected_velocity=b, reconstructed_velocity=tangent, displacement_error=uerr, tangent_error=verr, jacobian_rank_ratio=rank)
    shape = projected["a"].shape[:2]
    return {"reconstruction": stats(uerr), "tangent": stats(verr), "initial_reconstruction": stats(uerr.reshape(shape)[:, 0]), "initial_tangent": stats(verr.reshape(shape)[:, 0]), "minimum_jacobian_ratio": float(np.min(rank)), "rank_failures": int(np.sum(~np.isfinite(rank) | (rank <= 1e-8))),"nonfinite_predictions":int(np.sum(~np.isfinite(predicted))),"nonfinite_tangent":int(np.sum(~np.isfinite(tangent)))}, w.reshape(*shape, -1)


@partial(jax.jit, static_argnames=("grid",))
def field_metrics(a, b, g, truth_u, truth_v, grid, c, initial_energy, uscale, vscale):
    up = (a@g.T).reshape(-1, *grid.shape)
    vp = (b@g.T).reshape(-1, *grid.shape)
    ut, vt = truth_u.reshape(up.shape), truth_v.reshape(vp.shape)
    mass = jnp.asarray(grid.mass())
    ue = jnp.sqrt(jnp.sum(mass*(up-ut)**2, axis=(-2,-1)))/uscale
    ve = jnp.sqrt(jnp.sum(mass*(vp-vt)**2, axis=(-2,-1)))/vscale
    state = jnp.sqrt(jnp.maximum(0., energy(up-ut, vp-vt, grid, c))/initial_energy)
    ep, et = energy(up, vp, grid, c), energy(ut, vt, grid, c)
    powerp = jnp.sum(mass*damping_ratio(grid, c)*vp*vp, axis=(-2,-1))
    powert = jnp.sum(mass*damping_ratio(grid, c)*vt*vt, axis=(-2,-1))
    meanp, meant = jnp.sum(mass*up, axis=(-2,-1)), jnp.sum(mass*ut, axis=(-2,-1))
    xy = jnp.asarray(grid.coordinates())
    modes = jnp.stack([jnp.sin(i*jnp.pi*xy[..., 0])*jnp.sin(j*jnp.pi*xy[..., 1]) for i,j in ((1,1),(1,2),(2,1))])
    modal_up, modal_vp = jnp.einsum("txy,mxy,xy->tm", up, modes, mass), jnp.einsum("txy,mxy,xy->tm", vp, modes, mass)
    modal_ut, modal_vt = jnp.einsum("txy,mxy,xy->tm", ut, modes, mass), jnp.einsum("txy,mxy,xy->tm", vt, modes, mass)
    strip = (xy[...,0] < .1) | (xy[...,0] > .9) | (xy[...,1] < .1) | (xy[...,1] > .9)
    arrival_p = jnp.sum(mass*strip*up*up, axis=(-2,-1))
    arrival_t = jnp.sum(mass*strip*ut*ut, axis=(-2,-1))
    return {"displacement_error": ue, "velocity_error": ve, "energy_state_error": state,
            "rom_energy": ep, "truth_energy": et, "rom_boundary_power": powerp, "truth_boundary_power": powert,
            "rom_mean": meanp, "truth_mean": meant, "rom_modal_u": modal_up, "rom_modal_v": modal_vp,
            "truth_modal_u": modal_ut, "truth_modal_v": modal_vt, "rom_wall_strip": arrival_p, "truth_wall_strip": arrival_t}


def physical_summary(metrics, grid, c, dt_observe):
    if grid.bx=="dirichlet":
        indices=np.array([[1.,1.],[1.,2.],[2.,1.]])
        omega=2*c/grid.h*np.sqrt(np.sum(np.sin(np.pi*indices/(2*grid.n))**2,axis=1))
        interpretation="Semidiscrete Dirichlet standing-mode phase."
    else:
        omega = c*np.pi*np.sqrt(np.array([2., 5., 5.]))
        interpretation="Sine-projection phase diagnostic only; these are not absorbing-system eigenmodes."
    phasep = np.arctan2(-metrics["rom_modal_v"]/omega, metrics["rom_modal_u"])
    phaset = np.arctan2(-metrics["truth_modal_v"]/omega, metrics["truth_modal_u"])
    amplitude = np.sqrt(metrics["truth_modal_u"]**2+(metrics["truth_modal_v"]/omega)**2)
    prediction_amplitude=np.sqrt(metrics["rom_modal_u"]**2+(metrics["rom_modal_v"]/omega)**2)
    threshold=.05*np.max(amplitude)
    reference_valid=amplitude>threshold
    prediction_valid=prediction_amplitude>threshold
    mask = reference_valid & prediction_valid
    phaseerror = abs(np.angle(np.exp(1j*(phasep-phaset))))
    unwrapped=np.full_like(phaseerror,np.nan)
    for mode in range(mask.shape[1]):
        starts=np.flatnonzero(mask[:,mode]&~np.r_[False,mask[:-1,mode]])
        ends=np.flatnonzero(mask[:,mode]&~np.r_[mask[1:,mode],False])+1
        for start,end in zip(starts,ends):
            unwrapped[start:end,mode]=np.unwrap(np.angle(np.exp(1j*(phasep[start:end,mode]-phaset[start:end,mode]))))
    metrics["rom_modal_amplitude"],metrics["truth_modal_amplitude"]=prediction_amplitude,amplitude
    metrics["modal_phase_defined"],metrics["vanished_rom_modes"]=mask,reference_valid&~prediction_valid
    metrics["unwrapped_phase_error_valid_segments"]=unwrapped
    first = int(round(1.2/dt_observe))+1
    ip, it = int(np.argmax(metrics["rom_wall_strip"][:first])), int(np.argmax(metrics["truth_wall_strip"][:first]))
    return {"max_displacement_error": float(np.max(metrics["displacement_error"])), "max_velocity_error": float(np.max(metrics["velocity_error"])), "max_energy_state_error": float(np.max(metrics["energy_state_error"])),
            "max_defined_modal_phase_error": float(np.max(phaseerror[mask])) if np.any(mask) else None,
            "max_unwrapped_valid_segment_phase_error":float(np.max(abs(unwrapped[mask]))) if np.any(mask) else None,"vanished_mode_observations":int(np.sum(reference_valid&~prediction_valid)),"phase_interpretation":interpretation,
            "wall_strip_peak_time_difference": float((ip-it)*dt_observe), "wall_strip_peak_truth_time": it*dt_observe,
            "final_mean_error": float(abs(metrics["rom_mean"][-1]-metrics["truth_mean"][-1])),
            "final_rom_energy_fraction": float(metrics["rom_energy"][-1]/metrics["truth_energy"][0]), "final_truth_energy_fraction": float(metrics["truth_energy"][-1]/metrics["truth_energy"][0])}


def nonlinear_rollouts(config, grid, bank, data, p, frozen, zs, ws, kind, out):
    r = config["rank"]
    cases, arrays = [], {}
    g = jnp.asarray(bank["g"])
    for dt in config["rom_dts"]:
        stride = int(round(config["observation_dt"]/dt))
        if abs(stride*dt-config["observation_dt"]) > 1e-12:
            raise ValueError("ROM steps must divide observation interval")
        steps = int(round(config["end_time"]/dt))
        for i, pars in enumerate(data["parameters"]):
            print("rollout", grid.bx, kind, dt, i, flush=True)
            kr = jnp.asarray(pars[5]**2*bank["stiffness_unit"])
            dr = jnp.asarray(pars[5]*bank["damping_unit"])
            result = {key: np.asarray(value) for key, value in rollout(p, frozen, jnp.asarray(zs[i, 0]), jnp.asarray(ws[i, 0]), kr, dr, dt, kind=kind, steps=steps, stride=stride).items()}
            key = f"dt{dt}_case{i}"
            for field, value in result.items():
                arrays[key+"_"+field] = value
            completed = bool(np.all(result["completed"]))
            fail = np.flatnonzero(~result["completed"])
            row = {"case": i, "dt": dt, "completed": completed, "first_failed_observation": None if not len(fail) else int(fail[0]), "minimum_jacobian_ratio": float(np.min(result["rank_ratio"]))}
            if completed:
                fn = lambda z,w: jax.jvp(lambda zz: head_apply(p, frozen, zz, kind), (z,), (w,))
                a, b = jax.vmap(fn)(jnp.asarray(result["z"]), jnp.asarray(result["w"]))
                arrays[key+"_coefficients"], arrays[key+"_physical_velocity_coefficients"] = np.asarray(a), np.asarray(b)
                mm = {k: np.asarray(v) for k,v in field_metrics(a, b, g, jnp.asarray(data["u"][i]), jnp.asarray(data["v"][i]), grid, pars[5], data["initial_energy"][i], *data["scales"][i]).items()}
                balance = (mm["rom_energy"]+result["outflux"]-mm["rom_energy"][0])/max(mm["rom_energy"][0], 1e-30)
                metrics_finite=physical_metrics_finite(mm) and np.all(np.isfinite(balance))
                if metrics_finite:
                    row.update(physical_summary(mm, grid, pars[5], config["observation_dt"]))
                    row["max_energy_balance_relative"] = float(np.max(abs(balance)))
                else:
                    row["completed"]=False
                    row["failure_reason"]="Nonfinite decoded physical field/energy/diagnostic."
                for field, value in mm.items():
                    arrays[key+"_"+field] = value
                arrays[key+"_energy_balance_relative"] = balance
            cases.append(row)
    refinements=[]
    middle,fine=config["rom_dts"][-2:]
    for i,pars in enumerate(data["parameters"]):
        first=next(x for x in cases if x["case"]==i and x["dt"]==middle)
        second=next(x for x in cases if x["case"]==i and x["dt"]==fine)
        row={"case":i,"coarse_dt":middle,"fine_dt":fine,"both_completed":first["completed"] and second["completed"]}
        if row["both_completed"]:
            ka,kb=f"dt{middle}_case{i}",f"dt{fine}_case{i}"
            da=arrays[ka+"_coefficients"]-arrays[kb+"_coefficients"]
            db=arrays[ka+"_physical_velocity_coefficients"]-arrays[kb+"_physical_velocity_coefficients"]
            du=np.linalg.norm(da,axis=-1)/data["scales"][i,0]
            dv=np.linalg.norm(db,axis=-1)/data["scales"][i,1]
            de=np.sqrt(np.maximum(0.,(np.sum(db*db,axis=-1)+np.einsum("tr,rs,ts->t",da,pars[5]**2*bank["stiffness_unit"],da))/(2*data["initial_energy"][i])))
            row.update(max_displacement_difference=float(du.max()),max_velocity_difference=float(dv.max()),max_energy_state_difference=float(de.max()))
            row["passed"]=bool(max(du.max(),dv.max(),de.max())<=config["rom_refinement_target"])
            arrays[f"refinement_case{i}_displacement"],arrays[f"refinement_case{i}_velocity"],arrays[f"refinement_case{i}_energy_state"]=du,dv,de
        else:
            row["passed"]=False
        refinements.append(row)
    np.savez_compressed(out/"rollouts.npz", **arrays)
    summaries = []
    for dt in config["rom_dts"]:
        selected = [row for row in cases if row["dt"] == dt]
        complete = [row for row in selected if row["completed"]]
        summaries.append({"dt": dt, "completed": len(complete), "failed": len(selected)-len(complete),
                          "displacement": stats([row.get("max_displacement_error", np.nan) for row in selected]),
                          "velocity": stats([row.get("max_velocity_error", np.nan) for row in selected]),
                          "energy_state": stats([row.get("max_energy_state_error", np.nan) for row in selected])})
    return {"cases": cases, "summaries": summaries,"finest_two_refinement":refinements,"refinement_passed":all(x["passed"] for x in refinements), "primary_dt": config["rom_dts"][1], "selection_rule": "Middle predeclared dt is primary; all three are reported; failed solves are failures."}


def randomized_pod(config, grid, train):
    mass = grid.mass().ravel()
    u = train["u"]/train["scales"][:, 0,None,None]
    v = train["v"]/train["scales"][:, 1,None,None]
    x = jnp.asarray(np.concatenate((u.reshape(-1,u.shape[-1]), v.reshape(-1,v.shape[-1])), axis=0))*jnp.sqrt(jnp.asarray(mass))
    size = config["rank"]+24
    omega = jax.random.normal(jax.random.PRNGKey(config["pod_seed"]), (len(x),size), dtype=jnp.float64)
    q, _ = jnp.linalg.qr(x.T@omega, mode="reduced")
    for _ in range(4):
        z, _ = jnp.linalg.qr(x@q, mode="reduced")
        q, _ = jnp.linalg.qr(x.T@z, mode="reduced")
    small = np.asarray(x@q)
    _, singular, vt = np.linalg.svd(small, full_matrices=False)
    weighted_basis = np.asarray(q)@vt[:config["rank"]].T
    return weighted_basis/np.sqrt(mass)[:,None], singular


def linear_baseline(config, grid, g, validation, label, out):
    mass = grid.mass().ravel()
    rank = g.shape[1]
    lg = np.asarray(positive_laplacian(jnp.asarray(g.T.reshape(rank,*grid.shape)), grid)).reshape(rank,-1).T
    k0 = g.T@(mass[:,None]*lg)
    d0 = g.T@((mass*np.asarray(damping_ratio(grid,1.)).ravel())[:,None]*g)
    arrays, rows = {}, []
    for i, pars in enumerate(validation["parameters"]):
        k, d = pars[5]**2*k0, pars[5]*d0
        matrix = np.block([[np.zeros((rank,rank)), np.eye(rank)], [-k,-d]])
        step = expm(config["observation_dt"]*matrix)
        a0 = g.T@(mass*validation["u"][i,0])
        b0 = g.T@(mass*validation["v"][i,0])
        trajectory = [np.r_[a0,b0]]
        for _ in range(int(round(config["end_time"]/config["observation_dt"]))):
            trajectory.append(step@trajectory[-1])
        trajectory = np.asarray(trajectory)
        a,b = trajectory[:,:rank], trajectory[:,rank:]
        mm = {key:np.asarray(value) for key,value in field_metrics(jnp.asarray(a),jnp.asarray(b),jnp.asarray(g),jnp.asarray(validation["u"][i]),jnp.asarray(validation["v"][i]),grid,pars[5],validation["initial_energy"][i],*validation["scales"][i]).items()}
        if not physical_metrics_finite(mm):
            raise RuntimeError("Nonfinite linear-baseline physical metrics")
        rows.append({"case":i,**physical_summary(mm,grid,pars[5],config["observation_dt"])})
        for key,value in mm.items():
            arrays[f"case{i}_"+key]=value
        arrays[f"case{i}_a"],arrays[f"case{i}_b"]=a,b
    np.savez_compressed(out/(label+".npz"),g=g,**arrays)
    return {"label":label,"rank":rank,"integrator":"independent exact matrix exponential of linear Galerkin system","cases":rows,"displacement":stats([x["max_displacement_error"] for x in rows]),"velocity":stats([x["max_velocity_error"] for x in rows]),"energy_state":stats([x["max_energy_state_error"] for x in rows])}
