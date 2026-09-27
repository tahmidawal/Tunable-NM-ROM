"""Matched-dimensional fresh-wave diagnostics within the unchanged learned bank."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import scipy.linalg
import pilot as base
from pilot import jax, jnp, Grid, localized_initial, parameter_rows, restrict, metrics, clean, save_json, array_sha
from fresh_models import head_apply, head_geometry, tree_from_npz
from fresh_learning import fit_batch
from fresh_fom import integrate_balance, energy, damping_ratio, provenance


@jax.jit
def affine_generator(basis, center, stiffness, damping, speed):
    """Actual reduced mass and the affine offset force; speed is query input."""
    dim = basis.shape[1]
    mass = basis.T @ basis
    kk = speed**2 * basis.T @ stiffness @ basis
    dd = speed * basis.T @ damping @ basis
    force = -speed**2 * basis.T @ stiffness @ center
    acceleration = jnp.linalg.solve(mass, jnp.concatenate((-kk, -dd, force[:, None]), axis=1))
    upper = jnp.concatenate((jnp.zeros((dim, dim)), jnp.eye(dim), jnp.zeros((dim, 1))), axis=1)
    return jnp.concatenate((upper, acceleration, jnp.zeros((1, 2*dim+1))), axis=0)


@partial(jax.jit, static_argnames=("observations",))
def affine_evolve(generator, q0, w0, observation_dt, *, observations):
    propagator = jax.scipy.linalg.expm(generator * observation_dt)
    initial = jnp.concatenate((q0, w0, jnp.ones(1)))
    def step(y, _):
        y = propagator @ y
        return y, y
    _, states = jax.lax.scan(step, initial, None, length=observations-1)
    return jnp.concatenate((initial[None], states)), propagator


@jax.jit
def affine_coefficients(states, basis, center):
    dim = basis.shape[1]
    return states[:, :dim] @ basis.T + center, states[:, dim:2*dim] @ basis.T


def linear_query(arm, u0, v0, c, grid, cfg, bank):
    t0 = time.perf_counter()
    u, v, speed = jnp.asarray(u0), jnp.asarray(v0), jnp.asarray(c)
    jax.block_until_ready((u, v, speed))
    t1 = time.perf_counter()
    a = bank["g"].T @ (bank["mass"]*u.ravel())
    b = bank["g"].T @ (bank["mass"]*v.ravel())
    basis, center, inverse = [arm[k] for k in ("basis", "center", "inverse")]
    q0, w0 = inverse @ (a-center), inverse @ b
    generator = affine_generator(basis, center, bank["k"], bank["d"], speed)
    jax.block_until_ready((q0, w0, generator))
    t2 = time.perf_counter()
    states, propagator = affine_evolve(generator, q0, w0, cfg["observation_dt"], observations=49)
    jax.block_until_ready((states, propagator))
    t3 = time.perf_counter()
    coefficients, velocity = affine_coefficients(states, basis, center)
    up = np.asarray(coefficients@bank["g"].T).reshape(49, *grid.shape)
    vp = np.asarray(velocity@bank["g"].T).reshape(49, *grid.shape)
    t4 = time.perf_counter()
    record = dict(method=arm["name"], setting=0., intervals=grid.n, boundary=grid.bx,
        configuration_dimension=basis.shape[1], phase_dimension=2*basis.shape[1],
        completed=bool(np.all(np.isfinite(up)) and np.all(np.isfinite(vp))),
        seconds=dict(input_transfer=t1-t0, initialization_and_parameter_projection=t2-t1,
                     evolution=t3-t2, field_output_and_transfer=t4-t3, complete_query=t4-t0),
        output_sha256=dict(u=array_sha(up), v=array_sha(vp)), output_bytes=up.nbytes+vp.nbytes)
    aux = dict(coefficients=np.asarray(coefficients), velocity_coefficients=np.asarray(velocity),
               states=np.asarray(states), generator=np.asarray(generator), propagator=np.asarray(propagator),
               projected_u=np.asarray(a), projected_v=np.asarray(b))
    return up, vp, record, aux


def regenerate_ladder(inputs, bc, out, cfg):
    """Stream original training truth. Never generate validation/final here."""
    original = json.loads((inputs/"campaign-config.json").read_text())
    expected = json.loads((inputs/"data_manifest.json").read_text())["splits"]["train"]
    grid = Grid(original["n"], bc, bc)
    raw = tree_from_npz(inputs/"bank_parameters.npz")
    with np.load(inputs/"coordinates.npz") as f:
        rr, saved_linear, saved_center = [f[k] for k in ("qr_r", "common_linear", "common_center")]
    g = np.asarray(base.evaluate_bank(raw["p"], raw["frequency"], jnp.asarray(grid.coordinates().reshape(-1, 2)), jnp.asarray(np.linalg.inv(rr)), bc == "dirichlet"))
    gm = grid.mass().ravel()[:, None] * g
    pars = parameter_rows(original["train_seed"], original["train_count"])
    np.testing.assert_array_equal(pars, expected["parameters"])
    stride = int(np.ceil(original["observation_dt"]/(original["fom_cfl"]*grid.h/1.15)))
    dt = original["observation_dt"]/stride
    nobs = int(round(original["end_time"]/original["observation_dt"]))
    aa, bb, audits = [], [], []
    uh, vh = hashlib.sha256(), hashlib.sha256()
    start = time.perf_counter()
    for ci, par in enumerate(pars):
        u0, v0 = localized_initial(grid, par)
        u, v, flux = integrate_balance(u0, v0, par[5], dt, grid=grid, steps=nobs*stride, stride=stride)
        ee = np.asarray(energy(u, v, grid, par[5]))
        u, v, flux = np.asarray(u), np.asarray(v), np.asarray(flux)
        uh.update(u.reshape(49, -1).tobytes()); vh.update(v.reshape(49, -1).tobytes())
        aa.append(u.reshape(49, -1) @ gm); bb.append(v.reshape(49, -1) @ gm)
        balance = float(np.max(abs(ee+flux-ee[0]))/ee[0])
        invariant = np.sum(grid.mass()*(v+np.asarray(damping_ratio(grid, par[5]))*u), axis=(-2,-1)) if bc == "absorbing" else np.zeros(49)
        drift = float(np.max(abs(invariant-invariant[0])))
        if balance >= 1e-5 or drift >= 1e-10 or not np.all(np.isfinite(u)):
            raise RuntimeError("Regenerated training truth gate failed")
        audits.append(dict(case=ci, energy_balance=balance, invariant_drift=drift))
        print("training_coefficients", bc, ci, flush=True)
    aa, bb = np.stack(aa), np.stack(bb)
    flat = aa.reshape(-1, 64)
    center = flat.mean(axis=0)
    _, singular, vt = np.linalg.svd(flat-center, full_matrices=False)
    coordinates = (flat-center) @ vt.T
    scale = np.maximum(coordinates.std(axis=0), 1e-8)
    linear = vt.T * scale
    saved_q = np.linalg.qr(saved_linear)[0]
    defect = float(np.linalg.norm(saved_q@saved_q.T-vt[:16].T@vt[:16]))
    center_defect = float(np.linalg.norm(center-saved_center))
    match = defect < 1e-8 and center_defect < 1e-9
    np.savez_compressed(out/f"training_ladder_{bc}.npz", parameters=pars, a=aa, b=bb, center=center,
        basis=vt.T, singular_values=singular, standardized_linear=linear, scales=scale,
        covariance=(flat-center).T@(flat-center)/len(flat), saved_linear=saved_linear, saved_center=saved_center)
    manifest = dict(boundary=bc, seed=original["train_seed"], count=len(pars), intervals=grid.n, dt=dt,
        construction="Unnormalized training displacement coefficients, centered SVD and population-standard-deviation scaling, original convention.",
        original_u_sha256=expected["u_sha256"], regenerated_u_sha256=uh.hexdigest(),
        original_v_sha256=expected["v_sha256"], regenerated_v_sha256=vh.hexdigest(),
        training_coefficients_sha256=array_sha(aa), covariance_sha256=array_sha((flat-center).T@(flat-center)/len(flat)),
        saved_projector_defect=defect, saved_center_defect=center_defect, saved_initialization_matched=match,
        data_hashes_match=uh.hexdigest()==expected["u_sha256"] and vh.hexdigest()==expected["v_sha256"],
        seconds_including_first_compile=time.perf_counter()-start, truth_audits=audits,
        ladder_label="same original training initialization extended" if match else "separate regenerated training construction; saved affine16 retained")
    return linear, center, manifest


def arms_for(bank, inputs, linear, center):
    with np.load(inputs/"coordinates.npz") as f:
        saved = f["common_linear"]
    transform = jnp.asarray(bank["transform"])
    choices = [("affine16", transform@jnp.asarray(saved), bank["common_center"]),
               ("affine32", transform@jnp.asarray(linear[:, :32]), transform@jnp.asarray(center)),
               ("full64", jnp.eye(64), jnp.zeros(64))]
    return [dict(name=name, basis=basis, center=offset, inverse=jnp.linalg.pinv(basis)) for name,basis,offset in choices]


def fitted_diagnostics(bank, arms, u, v, grid, c, cfg, outpath):
    """Truth-only diagnostics; none of these fitted states initialize a query."""
    idx = np.asarray(cfg["diagnostic_indices"])
    u0norm = float(np.sqrt(np.sum(grid.mass()*u[0]**2)))
    phase_scale = float(np.sqrt(2*base.energy_np(u[0], v[0], grid, c)))
    force_scale = u0norm/cfg["end_time"]**2
    ug, vg = u[idx].reshape(len(idx), -1), v[idx].reshape(len(idx), -1)
    targets = jnp.asarray(ug) @ (bank["mass"][:, None]*bank["g"])
    velocities = jnp.asarray(vg) @ (bank["mass"][:, None]*bank["g"])
    # Last target is the zero field; normalization is fixed from this case.
    targets = jnp.concatenate((targets, jnp.zeros((1, 64))))
    velocities = jnp.concatenate((velocities, jnp.zeros((1, 64))))
    aff = (targets-bank["common_center"])@bank["common_inverse"].T
    starts = jnp.concatenate((aff[:, None], jnp.zeros_like(aff[:, None]),
        jnp.broadcast_to(bank["fixed_codes"], (len(targets), 6, 16))), axis=1)
    fitted, fits = [], []
    arrays = dict(targets=np.asarray(targets), velocities=np.asarray(velocities), starts=np.asarray(starts),
        diagnostic_indices=idx, truth_u=u[idx], truth_v=v[idx])
    for budget in cfg["diagnostic_fit_budgets"]:
        raw = fit_batch(bank["p"], bank["frozen"], jnp.repeat(targets, 8, axis=0),
            jnp.full(len(targets)*8, u0norm), starts.reshape(-1,16), kind="mlp", iterations=budget)
        values = [np.asarray(x).reshape(len(targets), 8, *x.shape[1:]) for x in raw]
        z,obj,grad,count,damp,stat,rank,finite = values
        selected = np.argmin(np.where(finite,obj,np.inf), axis=1)
        best = z[np.arange(len(targets)), selected]
        chosen = (grad[np.arange(len(targets)), selected] <= 1e-7) & ((stat[np.arange(len(targets)), selected]<=1e-6)|(obj[np.arange(len(targets)),selected]<=1e-20)) & (rank[np.arange(len(targets)),selected]>1e-8) & finite[np.arange(len(targets)),selected]
        for name,val in zip(("z","objective","gradient","iterations","damping","stationarity","rank_ratio","finite"),values):
            arrays[f"fit_{budget}_{name}"] = val
        arrays[f"fit_{budget}_selected"] = selected
        fits.append(dict(budget=budget, selected=selected, selected_stationary=chosen,
            selected_objectives=obj[np.arange(len(targets)),selected], selected_gradients=grad[np.arange(len(targets)),selected],
            selected_rank_ratios=rank[np.arange(len(targets)),selected]))
        fitted.append(best)
    best = jnp.asarray(fitted[-1])
    records = []
    stiffness, damping = c*c*bank["k"], c*bank["d"]
    for arm in [dict(name="mlp16")]+arms:
        aa,bb,normal,curve,jr = [],[],[],[],[]
        for i in range(len(targets)):
            if arm["name"] == "mlp16":
                fun = lambda z: head_apply(bank["p"],bank["frozen"],z,"mlp")
                jac = jax.jacfwd(fun)(best[i])
                q, rr = jnp.linalg.qr(jac, mode="reduced")
                w = jax.scipy.linalg.solve_triangular(rr,q.T@velocities[i],lower=False)
                a,b,jac,curvature = head_geometry(bank["p"],bank["frozen"],best[i],w,"mlp")
            else:
                jac = arm["basis"]
                a = arm["center"] + jac@(arm["inverse"]@(targets[i]-arm["center"]))
                b = jac@(arm["inverse"]@velocities[i])
                q = jnp.linalg.qr(jac,mode="reduced")[0]
                curvature = jnp.zeros_like(a)
            force = -stiffness@a-damping@b-curvature
            aa.append(np.asarray(a)); bb.append(np.asarray(b))
            normal.append(np.asarray(force-q@(q.T@force))); curve.append(np.asarray(curvature))
            singular = jnp.linalg.svd(jac,compute_uv=False)
            jr.append(float(singular[-1]/singular[0]))
        aa,bb,normal,curve = map(np.stack,(aa,bb,normal,curve))
        reconstructed_u = np.asarray(jnp.asarray(aa[:-1])@bank["g"].T).reshape(u[idx].shape)
        reconstructed_v = np.asarray(jnp.asarray(bb[:-1])@bank["g"].T).reshape(v[idx].shape)
        physical = metrics(reconstructed_u,reconstructed_v,u[idx],v[idx],grid,c,cfg)
        name = arm["name"]
        arrays.update({name+"_"+key:value for key,value in dict(coefficients=aa,velocity_coefficients=bb,normal_force=normal,curvature=curve,rank_ratio=np.asarray(jr)).items()})
        records.append(dict(arm=name,snapshot_metrics=physical,rank_ratios=jr,
            normal_force_absolute=np.linalg.norm(normal,axis=1), normal_force_fixed_scaled=np.linalg.norm(normal,axis=1)/force_scale,
            zero_field_mass_norm=float(np.linalg.norm(aa[-1])), zero_field_initial_displacement_scaled=float(np.linalg.norm(aa[-1])/u0norm)))
    np.savez_compressed(outpath,**arrays)
    return clean(dict(diagnostic_indices=idx,diagnostic_times=idx*cfg["observation_dt"],fitting=fits,
        fit_budget_objective_max_change=float(np.max(abs(np.asarray(fits[0]["selected_objectives"])-np.asarray(fits[1]["selected_objectives"])))),
        force_scale=force_scale, force_scale_definition="Initial displacement mass-L2 norm divided by horizon squared; fixed per case, units coefficient acceleration.",
        displacement_scale=u0norm,phase_energy_scale=phase_scale,arms=records,
        interpretation="Best recorded local snapshot fits at only declared times. Zero target appended last. Normal force is weak-bank compatibility, not trajectory error or a causal certificate."))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--inputs",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(); cfg=json.loads(args.config.read_text()); out=args.out; out.mkdir(parents=True,exist_ok=False)
    meta=provenance(); print(json.dumps(meta),flush=True)
    assert meta["jax_backend"]=="gpu" and meta["x64"] and meta["matmul_precision"]=="highest" and not cfg["final_test_opened"]
    frozen=json.loads((base.FRESH/"FROZEN-MATH.json").read_text())
    for name,expected in frozen["sha256"].items():
        assert base.sha(base.FRESH/name)==expected
    result=dict(config=cfg,provenance=meta,frozen_mathematics=frozen,training=[],meshes=[],references=[],invocations=[],warmups=[],diagnostics=[],linear_exponential_checks=[],final_test_opened=False)
    pars=parameter_rows(cfg["validation_seed"],max(cfg["validation_indices"])+1)
    for bc in cfg["boundaries"]:
        linear,center,manifest=regenerate_ladder(args.inputs/bc,bc,out,cfg); result["training"].append(manifest); save_json(out/"result.json",result)
        refs={}
        for ci in cfg["validation_indices"]:
            refs[ci], audit=base.build_reference(bc,pars[ci],cfg)
            result["references"].append(dict(boundary=bc,case=ci,parameters=pars[ci],**audit))
            np.savez_compressed(out/f"reference_{bc}_{ci}.npz",u=refs[ci][0],v=refs[ci][1])
        coarse={}
        for n in cfg["meshes"]:
            grid=Grid(n,bc,bc); common=Grid(cfg["comparison_intervals"],bc,bc)
            bank=base.rebuild(args.inputs/bc,grid); arms=arms_for(bank,args.inputs/bc,linear,center)
            result["meshes"].append(dict(boundary=bc,intervals=n,audits=bank["audits"],assembly_seconds_including_first_compile=bank["assembly_seconds_including_first_compile"]))
            arrays=dict(g=np.asarray(bank["g"]),mass=np.asarray(bank["mass"]),stiffness=np.asarray(bank["k"]),damping=np.asarray(bank["d"]),transform=bank["transform"])
            for arm in arms:
                for key in ("basis","center","inverse"):
                    arrays[arm["name"]+"_"+key]=np.asarray(arm[key])
            np.savez_compressed(out/f"mesh_{bc}_{n}.npz",**arrays)
            methods=[("rom",dt) for dt in cfg["rom_dts"]]+[(a["name"],0.) for a in arms]+([("dst",0.)] if bc=="dirichlet" else [("rk4",cf) for cf in cfg["fom_cfls"]])
            byname={a["name"]:a for a in arms}
            for ci in cfg["validation_indices"]:
                par=pars[ci]; u0,v0=map(np.asarray,localized_initial(grid,par))
                same_u,same_v,srec,_=base.query("dst" if bc=="dirichlet" else "rk4",0. if bc=="dirichlet" else cfg["reference_cfl"],u0,v0,par[5],grid,cfg)
                np.savez_compressed(out/f"samegrid_{bc}_{n}_{ci}.npz",u=same_u,v=same_v,u0=u0,v0=v0)
                same_common=tuple(restrict(a,n,common.n,bc) for a in (same_u,same_v))
                refentry=dict(boundary=bc,case=ci,same_grid_intervals=n,same_grid_to_physical_reference=metrics(*same_common,*refs[ci],common,par[5],cfg))
                if n==cfg["meshes"][0]: coarse[ci]=same_common
                elif bc=="absorbing": refentry["adjacent_coarse_to_this_mesh"]=metrics(*coarse[ci],*same_common,common,par[5],cfg)
                result["references"].append(refentry)
                def call(method,setting):
                    if method in byname: return linear_query(byname[method],u0,v0,par[5],grid,cfg,bank)
                    return base.query(method,setting,u0,v0,par[5],grid,cfg,bank)
                for method,setting in methods:
                    print("warmup",bc,n,ci,method,setting,flush=True)
                    _,_,rec,_=call(method,setting); result["warmups"].append(dict(case=ci,**rec))
                for rep in range(cfg["repetitions"]):
                    for order,(method,setting) in enumerate(methods if rep%2==0 else list(reversed(methods))):
                        print("timed",bc,n,ci,rep,method,setting,flush=True); base.burn()
                        u,v,rec,aux=call(method,setting)
                        invocation=f"{bc}_{n}_{ci}_{rep}_{method}_{setting}"
                        rec.update(case=ci,repetition=rep,order=order,parameters=par,invocation_id=invocation,
                            same_grid_discrepancy=metrics(u,v,same_u,same_v,grid,par[5],cfg),
                            physical_reference_error=metrics(*(restrict(a,n,common.n,bc) for a in (u,v)),*refs[ci],common,par[5],cfg))
                        if method=="rom":
                            fits=aux["fits"];selected=int(fits["selected"])
                            rec["cold_fit"]={k:clean(vv) for k,vv in fits.items() if k not in ("projected_u","projected_v")}
                            rec["fit_stationary"]=bool(fits["finite"][selected] and fits["rank_ratio"][selected]>1e-8 and fits["gradient"][selected]<=1e-7 and (fits["stationarity"][selected]<=1e-6 or fits["objective"][selected]<=1e-20))
                            rec["minimum_dynamic_rank_ratio"]=float(np.min(aux["rollout"]["rank_ratio"]))
                        if rep==0:
                            arrays=dict(u=restrict(u,n,common.n,bc),v=restrict(v,n,common.n,bc))
                            if method=="rom":
                                arrays.update({k:aux[k] for k in ("coefficients","velocity_coefficients")})
                                arrays.update({"rollout_"+k:vv for k,vv in aux["rollout"].items()})
                            elif method in byname:
                                arrays.update(aux)
                                # Independent CPU exponential of the actual query generator.
                                independent=np.stack([scipy.linalg.expm(aux["generator"]*(i*cfg["observation_dt"]))@aux["states"][0] for i in range(49)])
                                discrepancy=float(np.max(abs(independent-aux["states"])))
                                result["linear_exponential_checks"].append(dict(invocation_id=invocation,maximum_state_difference=discrepancy))
                                if discrepancy>1e-8: raise RuntimeError("Independent affine exponential mismatch")
                            np.savez_compressed(out/(invocation+".npz"),**arrays)
                        result["invocations"].append(rec);save_json(out/"result.json",result)
                print("diagnostics",bc,n,ci,flush=True)
                diag=fitted_diagnostics(bank,arms,same_u,same_v,grid,par[5],cfg,out/f"diagnostic_{bc}_{n}_{ci}.npz")
                result["diagnostics"].append(dict(boundary=bc,intervals=n,case=ci,**diag));save_json(out/"result.json",result)
            del bank; jax.clear_caches()
    result["complete"]=True
    result["output_sha256"]={p.name:base.sha(p) for p in out.glob("*.npz")}
    save_json(out/"result.json",result); print("wave_dynamics_complete",flush=True)

if __name__=="__main__": main()
