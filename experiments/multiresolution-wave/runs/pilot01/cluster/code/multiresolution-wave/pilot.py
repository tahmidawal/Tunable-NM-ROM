"""Frozen fresh-wave decoder transfer and paired complete-query measurement.

Frozen mathematical sources are imported without modifications. New-grid QR is
an exact change of coefficient coordinates, not training or new model capacity.
All timed solver outputs are scored and hashed in the same invocation record.
"""
from functools import partial
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.fft import dstn, idstn

FRESH = Path(__file__).resolve().parents[1] / "fresh-wave-head"
sys.path.insert(0, str(FRESH))
import jax
import jax.numpy as jnp
from fresh_fom import Grid, localized_initial, integrate, integrate_balance, positive_laplacian, damping_ratio, provenance
from fresh_models import bank_apply, head_apply, tree_from_npz
from fresh_learning import parameter_rows, fit_batch
from fresh_rom import rollout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save_json(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")


def restrict(field, n, target, bc):
    if n % target:
        raise ValueError("Nested meshes required")
    ratio = n // target
    sl = slice(ratio - 1, None, ratio) if bc == "dirichlet" else slice(None, None, ratio)
    return field[..., sl, sl]


def energy_np(u, v, grid, c):
    """Independent edge-energy calculation, without a discrete Laplacian."""
    mass = grid.mass()
    uu = np.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)]) if grid.bx == "dirichlet" else u
    if grid.bx == "dirichlet":
        potential = np.sum(np.diff(uu, axis=-2)**2, axis=(-2, -1)) + np.sum(np.diff(uu, axis=-1)**2, axis=(-2, -1))
    else:
        w = grid.axis_weights(grid.bx) / grid.h
        potential = np.sum(np.diff(uu, axis=-2)**2 * w, axis=(-2, -1)) + np.sum(np.diff(uu, axis=-1)**2 * w[:, None], axis=(-2, -1))
    return .5 * (np.sum(mass * v*v, axis=(-2, -1)) + c*c*potential)


def metrics(u, v, ut, vt, grid, c, cfg):
    mass = grid.mass()
    l2 = lambda x: np.sqrt(np.sum(mass*x*x, axis=(-2, -1)))
    e0 = float(energy_np(ut[0], vt[0], grid, c))
    u0 = float(l2(ut[0]))
    v0 = np.sqrt(2*e0)
    du, dv = l2(u-ut), l2(v-vt)
    de = np.sqrt(np.maximum(0., 2*energy_np(u-ut, v-vt, grid, c)))
    un, vn = l2(ut), l2(vt)
    en = np.sqrt(np.maximum(0., 2*energy_np(ut, vt, grid, c)))
    result = {}
    for name, absolute, current, scale in (("displacement", du, un, u0), ("velocity", dv, vn, v0), ("energy_state", de, en, v0)):
        zero = current <= 1e-14*scale
        vanishing = current <= cfg["vanishing_fraction"]*scale
        relative = np.divide(absolute, current, out=np.full_like(absolute, np.nan), where=~zero)
        normalized = absolute / scale
        result[name] = dict(absolute=absolute, initial_normalized=normalized, current_relative=relative,
                            reference_norm=current, initial_scale=scale, reference_zero=zero, reference_vanishing=vanishing,
                            max_initial_normalized=float(np.max(normalized)),
                            max_absolute=float(np.max(absolute)), max_current_relative=None if np.all(zero) else float(np.nanmax(relative)))
    result["mean_error"] = np.sum(mass*(u-ut), axis=(-2, -1))
    result["energy_fraction"] = energy_np(u, v, grid, c)/e0
    result["truth_energy_fraction"] = energy_np(ut, vt, grid, c)/e0
    # Sine projections are phase diagnostics for the absorber, not its eigenmodes.
    xy = grid.coordinates()
    modes = np.stack([np.sin(i*np.pi*xy[..., 0])*np.sin(j*np.pi*xy[..., 1]) for i,j in ((1,1),(1,2),(2,1))])
    project = lambda a: np.einsum("txy,mxy,xy->tm", a, modes, mass)
    inds = np.array([[1,1],[1,2],[2,1]])
    omega = c*np.pi*np.sqrt(np.sum(inds**2, axis=1))
    pu, pv, tu, tv = project(u), project(v), project(ut), project(vt)
    amp = np.sqrt(tu*tu+(tv/omega)**2)
    pamp = np.sqrt(pu*pu+(pv/omega)**2)
    valid = (amp > .05*np.max(amp)) & (pamp > .05*np.max(amp))
    phase = abs(np.angle(np.exp(1j*(np.arctan2(-pv/omega, pu)-np.arctan2(-tv/omega, tu)))))
    result["phase"] = dict(defined=valid, error=np.where(valid, phase, np.nan),
        interpretation="Sine-projection phase; absorber modes are not eigenmodes.")
    result["finite"] = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
    return clean(result)


def transform_head(p, transform):
    q = copy.copy(p)
    q["linear"] = transform @ p["linear"]
    q["bias"] = transform @ p["bias"]
    q["out"] = {"w": p["out"]["w"] @ transform.T, "b": p["out"]["b"] @ transform.T}
    return q


@partial(jax.jit, static_argnames=("reflective",))
def evaluate_bank(p, frequency, xy, inverse_r, reflective):
    return bank_apply(p, frequency, xy, reflective) @ inverse_r


@partial(jax.jit, static_argnames=("grid",))
def assemble(g, mass, boundary, *, grid):
    r = g.shape[1]
    lg = positive_laplacian(g.T.reshape(r, *grid.shape), grid).reshape(r, -1).T
    return g.T @ (mass[:, None]*lg), g.T @ ((mass*boundary)[:, None]*g)


def rebuild(inputs, grid):
    start = time.perf_counter()
    bank = tree_from_npz(inputs / "bank_parameters.npz")
    head = tree_from_npz(inputs / "head.npz")
    with np.load(inputs / "coordinates.npz") as f:
        rr0, linear0, center0 = [f[k] for k in ("qr_r", "common_linear", "common_center")]
    xy = jnp.asarray(grid.coordinates().reshape(-1, 2))
    mass = jnp.asarray(grid.mass().ravel())
    g0 = evaluate_bank(bank["p"], bank["frequency"], xy, jnp.asarray(np.linalg.inv(rr0)), grid.bx == "dirichlet")
    q, transform = jnp.linalg.qr(jnp.sqrt(mass)[:, None]*g0, mode="reduced")
    g = q / jnp.sqrt(mass)[:, None]
    p = transform_head(head["p"], transform)
    frozen = head["frozen"]
    k, d = assemble(g, mass, damping_ratio(grid, 1.).ravel(), grid=grid)
    jax.block_until_ready((g, p, k, d))
    seconds = time.perf_counter() - start
    z = jnp.linspace(-.1, .1, p["linear"].shape[1])
    w = jnp.cos(z)
    h0 = lambda z: head_apply(head["p"], frozen, z, "mlp")
    hn = lambda z: head_apply(p, frozen, z, "mlp")
    a0, b0 = jax.jvp(h0, (z,), (w,))
    an, bn = jax.jvp(hn, (z,), (w,))
    audits = dict(
        head_values=float(jnp.max(abs(an-transform@a0))),
        head_jacobian=float(jnp.max(abs(jax.jacfwd(hn)(z)-transform@jax.jacfwd(h0)(z)))),
        physical_value=float(jnp.max(abs(g@an-g0@a0))),
        physical_tangent=float(jnp.max(abs(g@bn-g0@b0))),
        orthogonality=float(jnp.linalg.norm(g.T@(mass[:, None]*g)-jnp.eye(g.shape[1]))),
        stiffness_symmetry=float(jnp.max(abs(k-k.T))),
        damping_symmetry=float(jnp.max(abs(d-d.T))),
        stiffness_min_eigenvalue=float(jnp.linalg.eigvalsh(k)[0]),
        damping_min_eigenvalue=float(jnp.linalg.eigvalsh(d)[0]))
    if max(audits[k] for k in ("head_values", "head_jacobian", "physical_value", "physical_tangent", "orthogonality")) > 1e-9:
        raise RuntimeError("Frozen decoder coordinate parity failed")
    common_linear = transform @ jnp.asarray(linear0)
    return dict(g=g, mass=mass, k=k, d=d, p=p, frozen=frozen,
                common_inverse=jnp.linalg.pinv(common_linear), common_center=transform@jnp.asarray(center0),
                fixed_codes=head["codes"][np.linspace(0, len(head["codes"])-1, 6, dtype=int)],
                assembly_seconds_including_first_compile=seconds, audits=audits,
                storage_bytes=sum(x.nbytes for x in (g, mass, k, d)), transform=np.asarray(transform))


def dst1(a, axis):
    a = jnp.moveaxis(a, axis, -1)
    n = a.shape[-1]+1
    zero = jnp.zeros((*a.shape[:-1], 1), dtype=a.dtype)
    odd = jnp.concatenate((zero, a, zero, -a[..., ::-1]), axis=-1)
    result = -jnp.fft.rfft(odd, axis=-1).imag[..., 1:n]/np.sqrt(2*n)
    return jnp.moveaxis(result, -1, axis)


def dst2(a):
    return dst1(dst1(a, -1), -2)


@jax.jit
def spectral_propagate(u0, v0, c, times):
    n = u0.shape[-1]+1
    modes = jnp.arange(1, n, dtype=jnp.float64)
    eigen = 4*n*n*jnp.sin(np.pi*modes/(2*n))**2
    omega = c*jnp.sqrt(eigen[:, None]+eigen[None, :])
    a, b = dst2(u0), dst2(v0)
    def one(t):
        co, si = jnp.cos(omega*t), jnp.sin(omega*t)
        return dst2(co*a+si/omega*b), dst2(-omega*si*a+co*b)
    return jax.lax.map(one, times)


@jax.jit
def burn_kernel(a):
    return jnp.tanh(a@a*.001)


def burn():
    a = jnp.ones((1024, 1024), dtype=jnp.float64)*.01
    a = burn_kernel(a)
    a.block_until_ready()
    start = time.perf_counter()
    while time.perf_counter()-start < .3:
        a = burn_kernel(a)
        a.block_until_ready()


def cold_fit(bank, u0, v0, iterations):
    p, frozen = bank["p"], bank["frozen"]
    a = bank["g"].T @ (bank["mass"]*u0.ravel())
    b = bank["g"].T @ (bank["mass"]*v0.ravel())
    scale = jnp.sqrt(jnp.sum(bank["mass"]*u0.ravel()**2))
    affine = bank["common_inverse"] @ (a-bank["common_center"])
    starts = jnp.concatenate((affine[None], jnp.zeros_like(affine)[None], bank["fixed_codes"]), axis=0)
    zz, objective, grad, counts, damping, stationarity, rank, finite = fit_batch(
        p, frozen, jnp.broadcast_to(a, (8, a.size)), jnp.full(8, scale), starts, kind="mlp", iterations=iterations)
    best = jnp.argmin(jnp.where(finite, objective, jnp.inf))
    z = zz[best]
    jac = jax.jacfwd(lambda x: head_apply(p, frozen, x, "mlp"))(z)
    q, r = jnp.linalg.qr(jac, mode="reduced")
    w = jax.scipy.linalg.solve_triangular(r, q.T@b, lower=False)
    return z, w, dict(objective=objective, gradient=grad, iterations=counts, stationarity=stationarity,
                      rank_ratio=rank, finite=finite, selected=best, projected_u=a, projected_v=b)


@jax.jit
def decode(p, frozen, z, w, g):
    def one(z, w):
        return jax.jvp(lambda x: head_apply(p, frozen, x, "mlp"), (z,), (w,))
    a, b = jax.vmap(one)(z, w)
    return a@g.T, b@g.T, a, b


def query(method, setting, u0, v0, c, grid, cfg, bank=None):
    t0 = time.perf_counter()
    u, v, speed = jnp.asarray(u0), jnp.asarray(v0), jnp.asarray(c)
    jax.block_until_ready((u, v, speed))
    t1 = time.perf_counter()
    aux = {}
    if method == "rom":
        z, w, fits = cold_fit(bank, u, v, cfg["fit_iterations"])
        stiffness, damping = speed*speed*bank["k"], speed*bank["d"]
        jax.block_until_ready((z, w, fits, stiffness, damping))
        t2 = time.perf_counter()
        stride = int(round(cfg["observation_dt"]/setting))
        steps = int(round(cfg["end_time"]/setting))
        rr = rollout(bank["p"], bank["frozen"], z, w, stiffness, damping, setting, kind="mlp", steps=steps, stride=stride)
        jax.block_until_ready(rr)
        t3 = time.perf_counter()
        up, vp, a, b = decode(bank["p"], bank["frozen"], rr["z"], rr["w"], bank["g"])
        # Physical output copy is required; diagnostics are transferred after timer.
        up, vp = np.asarray(up).reshape(-1, *grid.shape), np.asarray(vp).reshape(-1, *grid.shape)
        t4 = time.perf_counter()
        aux = dict(fits=jax.tree.map(np.asarray, fits), rollout=jax.tree.map(np.asarray, rr), coefficients=np.asarray(a), velocity_coefficients=np.asarray(b))
        completed = bool(np.all(aux["rollout"]["completed"]))
    else:
        if method == "dst":
            times = jnp.arange(int(round(cfg["end_time"]/cfg["observation_dt"]))+1)*cfg["observation_dt"]
            t2 = time.perf_counter()
            up, vp = spectral_propagate(u, v, speed, times)
            steps, stride = 0, 0
        else:
            stride = int(np.ceil(cfg["observation_dt"]/(setting*grid.h/c)))
            dt = cfg["observation_dt"]/stride
            steps = int(round(cfg["end_time"]/cfg["observation_dt"]))*stride
            t2 = time.perf_counter()
            up, vp = integrate(u, v, speed, dt, grid=grid, steps=steps, stride=stride)
        jax.block_until_ready((up, vp))
        t3 = time.perf_counter()
        up, vp = np.asarray(up), np.asarray(vp)
        t4 = time.perf_counter()
        completed = True
    completed = completed and bool(np.all(np.isfinite(up)) and np.all(np.isfinite(vp)))
    return up, vp, dict(method=method, setting=setting, intervals=grid.n, boundary=grid.bx,
        completed=completed, steps=steps, observation_stride=stride,
        seconds=dict(input_transfer=t1-t0, initialization_and_parameter_projection=t2-t1,
                     evolution=t3-t2, field_output_and_transfer=t4-t3, complete_query=t4-t0),
        output_sha256=dict(u=array_sha(up), v=array_sha(vp)), output_bytes=up.nbytes+vp.nbytes), aux


def continuum_reference(par, n, cfg):
    grid = Grid(n)
    u0, v0 = map(np.asarray, localized_initial(grid, par))
    a, b = dstn(u0, type=1, norm="ortho"), dstn(v0, type=1, norm="ortho")
    modes = np.arange(1, n)
    omega = par[5]*np.pi*np.sqrt(modes[:, None]**2+modes[None, :]**2)
    us, vs = [], []
    for ti in range(int(round(cfg["end_time"]/cfg["observation_dt"]))+1):
        co, si = np.cos(ti*cfg["observation_dt"]*omega), np.sin(ti*cfg["observation_dt"]*omega)
        us.append(restrict(idstn(co*a+si/omega*b, type=1, norm="ortho"), n, cfg["comparison_intervals"], "dirichlet"))
        vs.append(restrict(idstn(-omega*si*a+co*b, type=1, norm="ortho"), n, cfg["comparison_intervals"], "dirichlet"))
    return np.stack(us), np.stack(vs)


def build_reference(bc, par, cfg):
    start = time.perf_counter()
    if bc == "dirichlet":
        coarse = continuum_reference(par, cfg["spectral_reference_meshes"][0], cfg)
        fine = continuum_reference(par, cfg["spectral_reference_meshes"][1], cfg)
        audit = dict(kind="independent scipy continuum sine transform", fine_intervals=cfg["spectral_reference_meshes"][1])
    else:
        n = cfg["absorbing_reference_n"]
        grid = Grid(n, bc, bc)
        u0, v0 = localized_initial(grid, par)
        stride = int(np.ceil(cfg["observation_dt"]/(cfg["reference_cfl"]*grid.h/par[5])))
        u, v, flux = map(np.asarray, integrate_balance(u0, v0, par[5], cfg["observation_dt"]/stride,
            grid=grid, steps=int(round(cfg["end_time"]/cfg["observation_dt"]))*stride, stride=stride))
        ee = energy_np(u, v, grid, par[5])
        balance = float(np.max(abs(ee+flux-ee[0]))/ee[0])
        invariant = np.sum(grid.mass()*(v+np.asarray(damping_ratio(grid, par[5]))*u), axis=(-2, -1))
        fine = tuple(restrict(x, n, cfg["comparison_intervals"], bc) for x in (u, v))
        # Same-grid temporal check is distinct from the nested spatial check.
        temporal_stride = int(np.ceil(cfg["observation_dt"]/(2*cfg["reference_cfl"]*grid.h/par[5])))
        tu, tv = map(np.asarray, integrate(u0, v0, par[5], cfg["observation_dt"]/temporal_stride,
            grid=grid, steps=int(round(cfg["end_time"]/cfg["observation_dt"]))*temporal_stride, stride=temporal_stride))
        temporal = metrics(*(restrict(x, n, cfg["comparison_intervals"], bc) for x in (tu, tv)), *fine,
                           Grid(cfg["comparison_intervals"], bc, bc), par[5], cfg)
        coarse = None
        audit = dict(kind="fresh edge-energy boundary operator with independently checked balance; conditional spatial refinement", fine_intervals=n,
                     dt=cfg["observation_dt"]/stride, energy_balance=balance, invariant_drift=float(np.max(abs(invariant-invariant[0]))),
                     energy=ee, flux=flux, temporal_refinement=temporal)
        if balance > 1e-5 or audit["invariant_drift"] > 1e-10:
            raise RuntimeError("Reference balance/invariant failed")
    audit["generation_seconds"] = time.perf_counter()-start
    audit["u_sha256"], audit["v_sha256"] = map(array_sha, fine)
    if coarse is not None:
        audit["self_refinement"] = metrics(*coarse, *fine, Grid(cfg["comparison_intervals"], bc, bc), par[5], cfg)
    return fine, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    meta = provenance()
    print(json.dumps(meta), flush=True)
    if meta["jax_backend"] != "gpu" or not meta["x64"] or meta["matmul_precision"] != "highest":
        raise RuntimeError("GPU/f64/highest required")
    frozen = json.loads((FRESH/"FROZEN-MATH.json").read_text())
    for name, expected in frozen["sha256"].items():
        if sha(FRESH/name) != expected:
            raise RuntimeError("Frozen math source changed: "+name)
    if cfg["final_test_opened"] or cfg["weak_tests"] < 4*cfg["latent"]:
        raise RuntimeError("Cohort/test-space gate failed")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    result = dict(config=cfg, provenance=meta, frozen_mathematics=frozen,
                  input_sha256={str(p.relative_to(args.inputs)): sha(p) for p in args.inputs.rglob("*") if p.is_file()},
                  observations=np.arange(int(round(cfg["end_time"]/cfg["observation_dt"]))+1)*cfg["observation_dt"],
                  meshes=[], references=[], invocations=[], warmups=[], final_test_opened=False,
                  caveats=["Two-case development pilot.", "Dense cold start and dense physical output grow with resolution.",
                           "Scalar c rescaling is inside each timed ROM initialization; mass/unit operators are mesh-offline.",
                           "No forcing is present; no uncounted source projection exists.",
                           "No neural weights retrained. QR head change is an exactly audited coefficient-gauge conversion."])
    parameters = parameter_rows(cfg["validation_seed"], max(cfg["validation_indices"])+1)
    save_json(out/"result.json", result)
    for bc in cfg["boundaries"]:
        refs = {}
        coarse_references = {}
        for ci in cfg["validation_indices"]:
            print("reference", bc, ci, flush=True)
            refs[ci], audit = build_reference(bc, parameters[ci], cfg)
            result["references"].append(dict(boundary=bc, case=ci, parameters=parameters[ci], **audit))
            np.savez_compressed(out/f"reference_{bc}_{ci}.npz", **{name: restrict(value, cfg["comparison_intervals"], cfg["saved_intervals"], bc) for name,value in zip(("u", "v"), refs[ci])})
            save_json(out/"result.json", result)
        for n in cfg["meshes"]:
            grid = Grid(n, bc, bc)
            print("rebuild", bc, n, flush=True)
            bank = rebuild(args.inputs/bc, grid)
            result["meshes"].append(dict(boundary=bc, intervals=n, total_nodes=(n+1)**2, active_unknowns=int(np.prod(grid.shape)),
                boundary_unknowns=0 if bc == "dirichlet" else 4*n,
                assembly_seconds_including_first_compile=bank["assembly_seconds_including_first_compile"],
                storage_bytes=bank["storage_bytes"], audits=bank["audits"]))
            np.savez_compressed(out/f"mesh_{bc}_{n}.npz", coordinate_transform=bank["transform"], stiffness=np.asarray(bank["k"]), damping=np.asarray(bank["d"]))
            methods = [("rom", dt) for dt in cfg["rom_dts"]] + ([("dst", 0.)] if bc == "dirichlet" else [("rk4", cfl) for cfl in cfg["fom_cfls"]])
            for ci in cfg["validation_indices"]:
                par = parameters[ci]
                u0, v0 = map(np.asarray, localized_initial(grid, par))
                same_u, same_v, same_record, _ = query("dst" if bc == "dirichlet" else "rk4", 0. if bc == "dirichlet" else cfg["reference_cfl"], u0, v0, par[5], grid, cfg)
                result["warmups"].append(dict(case=ci, purpose="same-grid reference and compile", **same_record))
                common = Grid(cfg["comparison_intervals"], bc, bc)
                same_common = tuple(restrict(x, n, common.n, bc) for x in (same_u, same_v))
                result["references"].append(dict(boundary=bc, case=ci, same_grid_intervals=n,
                    same_grid_to_physical_reference=metrics(*same_common, *refs[ci], common, par[5], cfg)))
                if n == cfg["meshes"][0]:
                    coarse_references[ci] = same_common
                elif bc == "absorbing":
                    adjacent = metrics(*coarse_references[ci], *same_common, common, par[5], cfg)
                    result["references"][-1]["adjacent_coarse_to_this_mesh"] = adjacent
                for method, setting in methods:
                    print("warmup", bc, n, ci, method, setting, flush=True)
                    _, _, record, _ = query(method, setting, u0, v0, par[5], grid, cfg, bank)
                    result["warmups"].append(dict(case=ci, purpose="compile and warmup", **record))
                for rep in range(cfg["repetitions"]):
                    ordered = methods if rep % 2 == 0 else list(reversed(methods))
                    for order, (method, setting) in enumerate(ordered):
                        print("timed", bc, n, ci, rep, method, setting, flush=True)
                        burn()
                        u, v, record, aux = query(method, setting, u0, v0, par[5], grid, cfg, bank)
                        record.update(case=ci, repetition=rep, order=order, parameters=par,
                            invocation_id=f"{bc}_{n}_{ci}_{rep}_{method}_{setting}",
                            same_grid_discrepancy=metrics(u, v, same_u, same_v, grid, par[5], cfg),
                            physical_reference_error=metrics(*(restrict(x, n, common.n, bc) for x in (u, v)), *refs[ci], common, par[5], cfg))
                        if method == "rom":
                            selected = int(aux["fits"]["selected"])
                            fits = aux["fits"]
                            record["cold_fit"] = {k: clean(vv) for k,vv in fits.items() if k not in ("projected_u", "projected_v")}
                            record["fit_stationary"] = bool(fits["finite"][selected] and fits["rank_ratio"][selected] > 1e-8 and fits["gradient"][selected] <= 1e-7 and (fits["stationarity"][selected] <= 1e-6 or fits["objective"][selected] <= 1e-20))
                            record["minimum_dynamic_rank_ratio"] = float(np.min(aux["rollout"]["rank_ratio"]))
                        result["invocations"].append(record)
                        if rep == 0:
                            arrays = {"u": restrict(u, n, cfg["saved_intervals"], bc), "v": restrict(v, n, cfg["saved_intervals"], bc)}
                            if method == "rom":
                                arrays.update({k: aux[k] for k in ("coefficients", "velocity_coefficients")})
                                arrays.update({"rollout_"+k: vv for k,vv in aux["rollout"].items()})
                            np.savez_compressed(out/(record["invocation_id"]+".npz"), **arrays)
                        save_json(out/"result.json", result)
                # A targeted representation floor uses the same frozen bank only.
                au = jnp.asarray(same_u.reshape(len(same_u), -1)) @ (bank["mass"][:, None]*bank["g"])
                av = jnp.asarray(same_v.reshape(len(same_v), -1)) @ (bank["mass"][:, None]*bank["g"])
                pu, pv = np.asarray(au@bank["g"].T).reshape(same_u.shape), np.asarray(av@bank["g"].T).reshape(same_v.shape)
                result.setdefault("bank_projection_diagnostic", []).append(dict(boundary=bc, intervals=n, case=ci,
                    metrics=metrics(pu, pv, same_u, same_v, grid, par[5], cfg), interpretation="Unrestricted mass-L2 projection; not an evolved or equal-latent-dimensional baseline."))
                save_json(out/"result.json", result)
            del bank
            jax.clear_caches()
    result["complete"] = True
    save_json(out/"result.json", result)
    print("multiresolution_wave_complete", flush=True)


if __name__ == "__main__":
    main()
