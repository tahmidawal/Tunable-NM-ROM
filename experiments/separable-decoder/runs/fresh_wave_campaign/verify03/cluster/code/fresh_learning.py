"""Training and latent fitting for the fresh frozen neural-bank experiment."""
from functools import partial
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from fresh_fom import Grid, localized_initial, integrate, positive_laplacian, damping_ratio, energy
from fresh_models import bank_init, bank_apply, head_init, head_apply, tree_to_npz


def parameter_rows(seed, count):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        sx, sy = rng.uniform(.36, .42, 2)
        cx, cy = rng.uniform(sx+.025, 1-sx-.025), rng.uniform(sy+.025, 1-sy-.025)
        amp, c = rng.uniform(.7, 1.3), rng.uniform(.85, 1.15)
        dx, dy = rng.uniform(-.5, .5, 2)
        if i % 4 == 0:
            dx = dy = 0.
        sigx, sigy = rng.uniform(.12, .16, 2)
        rows.append((cx, cy, sx, sy, amp, c, dx, dy, sigx, sigy))
    return np.asarray(rows, dtype=np.float64)


def generate_data(config, bc, out):
    grid = Grid(config["n"], bc, bc)
    arrays = {}
    for split, seed, count in (("train", config["train_seed"], config["train_count"]), ("validation", config["validation_seed"], config["validation_count"])):
        pars = parameter_rows(seed, count)
        displacement, velocity, energies, scales = [], [], [], []
        stride = int(np.ceil(config["observation_dt"]/(config["fom_cfl"]*grid.h/1.15)))
        dt = config["observation_dt"]/stride
        observations = int(round(config["end_time"]/config["observation_dt"]))
        for i, p in enumerate(pars):
            u0, v0 = localized_initial(grid, p)
            u, v = integrate(u0, v0, p[5], dt, grid=grid, steps=observations*stride, stride=stride)
            u, v = np.asarray(u), np.asarray(v)
            if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
                raise RuntimeError("Nonfinite fresh truth")
            e0 = float(energy(u0, v0, grid, p[5]))
            uscale = float(np.sqrt(np.sum(grid.mass()*np.asarray(u0)**2)))
            displacement.append(u.reshape(len(u), -1))
            velocity.append(v.reshape(len(v), -1))
            energies.append(e0)
            scales.append((uscale, np.sqrt(2*e0)))
        arrays[split] = {"parameters": pars, "u": np.stack(displacement), "v": np.stack(velocity), "initial_energy": np.asarray(energies), "scales": np.asarray(scales), "dt": dt}
        print("data", bc, split, len(pars), flush=True)
    # No final-test generator call occurs here. Selected fields allow independent
    # spot checks; all membership and truth checksums are retained separately.
    import hashlib
    manifest = {"bc": bc, "n": grid.n, "final_test_opened": False, "splits": {}}
    for split, data in arrays.items():
        manifest["splits"][split] = {"parameters": data["parameters"].tolist(), "dt": data["dt"], "u_sha256": hashlib.sha256(data["u"].tobytes()).hexdigest(), "v_sha256": hashlib.sha256(data["v"].tobytes()).hexdigest(), "initial_energy": data["initial_energy"].tolist(), "scales": data["scales"].tolist()}
    (out/"data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    val = arrays["validation"]
    saved_times = np.unique(np.linspace(0, val["u"].shape[1]-1, 4, dtype=int))
    np.savez_compressed(out/"truth_spotchecks.npz", parameters=val["parameters"], times=saved_times*config["observation_dt"], u=val["u"][:, saved_times], v=val["v"][:, saved_times])
    return grid, arrays


def train_bank(config, grid, data, out):
    p, frequency = bank_init(jax.random.PRNGKey(config["bank_seed"]), config["rank"], config["bank_width"])
    xy, sqrtmass = jnp.asarray(grid.coordinates().reshape(-1, 2)), jnp.sqrt(jnp.asarray(grid.mass().ravel()))
    u = data["u"]/data["scales"][:, 0, None, None]
    v = data["v"]/data["scales"][:, 1, None, None]
    target = jnp.asarray(np.concatenate((u.reshape(-1, u.shape[-1]), v.reshape(-1, v.shape[-1])), axis=0))*sqrtmass
    del u, v
    steps = config["bank_steps"]
    optimizer = optax.adam(optax.cosine_decay_schedule(config["bank_lr"], steps, alpha=.1))
    state = optimizer.init(p)
    reflective = grid.bx == "dirichlet"
    def objective(params, target_rows, coordinates, masses, frequencies):
        raw = bank_apply(params, frequencies, coordinates, reflective)*masses[:, None]
        q, rr = jnp.linalg.qr(raw, mode="reduced")
        coeff = target_rows@q
        residue = target_rows-coeff@q.T
        return jnp.mean(jnp.sum(residue*residue, axis=-1))
    @jax.jit
    def update(params, state, target_rows, coordinates, masses, frequencies):
        loss, grad = jax.value_and_grad(objective)(params, target_rows, coordinates, masses, frequencies)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss
    rng = np.random.default_rng(config["bank_seed"]+1)
    history = []
    for step in range(steps):
        idx = rng.integers(0, len(target), config["bank_batch"])
        p, state, loss = update(p, state, target[idx], xy, sqrtmass, frequency)
        if step % 200 == 0 or step == steps-1:
            val = float(loss)
            if not np.isfinite(val):
                raise RuntimeError("Nonfinite bank training objective")
            history.append([step, val])
            print("bank", grid.bx, step, val, flush=True)
    raw = np.asarray(bank_apply(p, frequency, xy, reflective))
    weighted = raw*np.asarray(sqrtmass)[:, None]
    q, rr = np.linalg.qr(weighted, mode="reduced")
    singular = np.linalg.svd(weighted, compute_uv=False)
    if not np.all(np.isfinite(singular)) or singular[0] <= 0:
        raise RuntimeError("Nonfinite or zero raw learned bank")
    condition_ratio = float(singular[-1]/singular[0])
    if condition_ratio <= config["bank_rank_ratio_min"]:
        raise RuntimeError(f"Learned raw bank loses rank: ratio={condition_ratio}")
    g = q/np.asarray(sqrtmass)[:, None]
    if np.linalg.norm(g@rr-raw)/np.linalg.norm(raw) > 1e-10:
        raise RuntimeError("QR changed the learned spatial span")
    # Columns are reshaped as (R,nx,ny) before the FOM's last-two-axis operator.
    lg = np.asarray(positive_laplacian(jnp.asarray(g.T.reshape(config["rank"], *grid.shape)), grid)).reshape(config["rank"], -1).T
    mass = grid.mass().ravel()
    stiffness = g.T@(mass[:, None]*lg)
    damp1 = np.asarray(damping_ratio(grid, 1.)).ravel()
    damping = g.T@((mass*damp1)[:, None]*g)
    raw_lg = np.asarray(positive_laplacian(jnp.asarray(raw.T.reshape(config["rank"], *grid.shape)), grid)).reshape(config["rank"], -1).T
    raw_stiffness = raw.T@(mass[:, None]*raw_lg)
    if np.linalg.norm(rr.T@stiffness@rr-raw_stiffness)/np.linalg.norm(raw_stiffness) > 1e-10:
        raise RuntimeError("QR stiffness-coordinate identity failed")
    defect = float(np.linalg.norm(g.T@(mass[:, None]*g)-np.eye(config["rank"])))
    if defect > 1e-10 or np.max(abs(stiffness-stiffness.T)) > 1e-9:
        raise RuntimeError("Mass QR/table symmetry check failed")
    tree_to_npz(out/"bank_parameters.npz", {"p": p, "frequency": frequency})
    np.savez_compressed(out/"bank_tables.npz", g=g, qr_r=rr, mass=mass, stiffness_unit=stiffness, damping_unit=damping, raw_singular_values=singular, history=np.asarray(history))
    return {"g": g, "mass": mass, "stiffness_unit": stiffness, "damping_unit": damping, "raw_rank_ratio": condition_ratio, "orthogonality_defect": defect, "training_history": history}


def project_data(data, bank):
    gmass = bank["mass"][:, None]*bank["g"]
    projected = {}
    for split, d in data.items():
        a, b = d["u"]@gmass, d["v"]@gmass
        unorm2 = np.sum(d["u"]**2*bank["mass"], axis=-1)
        vnorm2 = np.sum(d["v"]**2*bank["mass"], axis=-1)
        ufloor = np.maximum(0., unorm2-np.sum(a*a, axis=-1))
        vfloor = np.maximum(0., vnorm2-np.sum(b*b, axis=-1))
        projected[split] = {"a": a, "b": b, "u_floor_squared": ufloor, "v_floor_squared": vfloor,
                            "u_scale": np.repeat(d["scales"][:, 0], a.shape[1]), "v_scale": np.repeat(d["scales"][:, 1], a.shape[1])}
    return projected


def common_affine(a, k):
    target = a.reshape(-1, a.shape[-1])
    center = target.mean(axis=0)
    _, _, vt = np.linalg.svd(target-center, full_matrices=False)
    coordinates = (target-center)@vt[:k].T
    scale = np.maximum(coordinates.std(axis=0), 1e-8)
    linear = vt[:k].T*scale
    z = coordinates/scale
    output_scale = float(np.sqrt(np.mean(np.sum(target**2, axis=-1))/target.shape[-1]))
    return linear, center, z, output_scale


def train_head(config, projected, common, kind, velocity_weight, seed, out):
    linear, center, codes, output_scale = common
    p, frozen = head_init(jax.random.PRNGKey(seed), linear, center, output_scale, kind, config["head_width"])
    codes = jnp.asarray(codes)
    a = jnp.asarray(projected["a"].reshape(-1, config["rank"]))
    b = jnp.asarray(projected["b"].reshape(-1, config["rank"]))
    us, vs = jnp.asarray(projected["u_scale"]), jnp.asarray(projected["v_scale"])
    steps = config["head_steps"]
    head_optimizer = optax.adam(optax.cosine_decay_schedule(config["head_lr"], steps, alpha=.1))
    code_optimizer = optax.adam(optax.cosine_decay_schedule(config["code_lr"], steps, alpha=.1))
    state_p, state_z = head_optimizer.init(p), code_optimizer.init(codes)
    def objective(params, zall, index, aa, bb, scale_u, scale_v, f):
        z = zall[index]
        predictions = head_apply(params, f, z, kind)
        reconstruction = jnp.mean(jnp.sum((predictions-aa[index])**2, axis=-1)/scale_u[index]**2)
        tangent = jnp.array(0.)
        if velocity_weight:
            jac = jax.vmap(jax.jacfwd(lambda zz: head_apply(params, f, zz, kind)))(z)
            q, _ = jnp.linalg.qr(jac, mode="reduced")
            fit = jnp.einsum("brk,bk->br", q, jnp.einsum("brk,br->bk", q, bb[index]))
            tangent = jnp.mean(jnp.sum((fit-bb[index])**2, axis=-1)/scale_v[index]**2)
        penalty = config["code_penalty"]*jnp.mean(z*z)
        return reconstruction+velocity_weight*tangent+penalty, (reconstruction, tangent)
    @jax.jit
    def update(params, z, sp, sz, idx, aa, bb, scale_u, scale_v, f):
        (loss, terms), (gp, gz) = jax.value_and_grad(objective, argnums=(0,1), has_aux=True)(params, z, idx, aa, bb, scale_u, scale_v, f)
        up, sp = head_optimizer.update(gp, sp, params)
        uz, sz = code_optimizer.update(gz, sz, z)
        return optax.apply_updates(params, up), optax.apply_updates(z, uz), sp, sz, loss, terms
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        idx = jnp.asarray(rng.integers(0, len(a), config["head_batch"]))
        p, codes, state_p, state_z, loss, terms = update(p, codes, state_p, state_z, idx, a, b, us, vs, frozen)
        if step % 200 == 0 or step == steps-1:
            row = [step, float(loss), float(terms[0]), float(terms[1])]
            if not np.all(np.isfinite(row)):
                raise RuntimeError("Nonfinite head training")
            history.append(row)
            print("head", kind, velocity_weight, seed, *row, flush=True)
    tree_to_npz(out/"head.npz", {"p": p, "frozen": frozen, "codes": codes})
    np.savez_compressed(out/"training_history.npz", history=np.asarray(history))
    return p, frozen, np.asarray(codes), history


@partial(jax.jit, static_argnames=("kind", "iterations"))
def fit_batch(p, frozen, targets, scales, starts, *, kind, iterations):
    """Batched trust-region LM; damping is explicit, not a dynamics ridge."""
    def one(target, scale, start):
        def residual(z):
            return (head_apply(p, frozen, z, kind)-target)/scale
        def step(i, state):
            z, lam, done, stop_iteration = state
            r = residual(z)
            jac = jax.jacfwd(residual)(z)
            grad = jac.T@r
            gnorm = jnp.max(abs(grad))/jnp.maximum(1., jnp.linalg.norm(jac)*jnp.linalg.norm(r))
            converged = gnorm <= 1e-7
            gram = jac.T@jac
            delta = jnp.linalg.solve(gram+lam*jnp.diag(jnp.maximum(jnp.diag(gram), 1e-10)), -grad)
            candidate = z+delta
            rr = residual(candidate)
            accepted = (rr@rr < r@r) & jnp.all(jnp.isfinite(candidate))
            active = ~done & ~converged
            z = jnp.where(active & accepted, candidate, z)
            lam = jnp.where(active, jnp.clip(jnp.where(accepted, lam/3, lam*5), 1e-12, 1e12), lam)
            stop_iteration = jnp.where(~done & converged, i, stop_iteration)
            return z, lam, done | converged, stop_iteration
        def condition(state):
            i, payload = state
            return (i < iterations) & ~payload[2]
        def advance(state):
            i, payload = state
            return i+1, step(i, payload)
        _, (z, lam, done, count) = jax.lax.while_loop(condition, advance, (jnp.array(0), (start, jnp.array(1e-3), jnp.array(False), jnp.array(iterations))))
        r = residual(z)
        jac = jax.jacfwd(residual)(z)
        grad = jac.T@r
        gnorm = jnp.max(abs(grad))/jnp.maximum(1., jnp.linalg.norm(jac)*jnp.linalg.norm(r))
        return z, r@r, gnorm, count, lam
    return jax.vmap(one)(targets, scales, starts)


def latent_fits(config, p, frozen, targets, scales, common, kind, out, trained_codes):
    linear, center, _, _ = common
    affine = (targets-center)@np.linalg.pinv(linear).T
    training_indices = np.linspace(0, len(trained_codes)-1, 6, dtype=int)
    fixed = np.broadcast_to(trained_codes[training_indices][None], (len(targets), 6, affine.shape[-1]))
    starts = np.concatenate((affine[:,None], np.zeros_like(affine)[:,None], fixed), axis=1)
    all_results = []
    for si in range(8):
        batches = []
        for begin in range(0, len(targets), config["fit_batch"]):
            sl = slice(begin, begin+config["fit_batch"])
            batches.append(tuple(np.asarray(x) for x in fit_batch(p, frozen, jnp.asarray(targets[sl]), jnp.asarray(scales[sl]), jnp.asarray(starts[sl, si]), kind=kind, iterations=config["fit_iterations"])))
        all_results.append(tuple(np.concatenate([b[j] for b in batches]) for j in range(5)))
    z, objective, grad, count, damping = [np.stack([r[j] for r in all_results], axis=1) for j in range(5)]
    selected = np.argmin(objective, axis=1)
    best = z[np.arange(len(z)), selected]
    doubled = []
    for si in range(8):
        batches = []
        for begin in range(0, len(targets), config["fit_batch"]):
            sl = slice(begin, begin+config["fit_batch"])
            batches.append(tuple(np.asarray(x) for x in fit_batch(p, frozen, jnp.asarray(targets[sl]), jnp.asarray(scales[sl]), jnp.asarray(starts[sl,si]), kind=kind, iterations=config["fit_iterations"]*2)))
        doubled.append(tuple(np.concatenate([b[j] for b in batches]) for j in range(5)))
    dz_all, dobj_all, dgrad_all, dcount_all, ddamp_all = [np.stack([b[j] for b in doubled], axis=1) for j in range(5)]
    dselected = np.argmin(dobj_all, axis=1)
    ii = np.arange(len(targets))
    dz, dobj, dgrad = dz_all[ii,dselected], dobj_all[ii,dselected], dgrad_all[ii,dselected]
    stopped = np.where(np.isfinite(dgrad_all), np.where(dgrad_all <= config["fit_gradient_tolerance"], "stationary", "budget"), "nonfinite")
    np.savez_compressed(out/"latent_fits.npz", initial_starts=starts, fixed_training_indices=training_indices, fitted_z=z, objectives=objective, gradients=grad, iterations=count, damping=damping, selected_start=selected, selected_z=best, doubled_all_z=dz_all, doubled_all_objectives=dobj_all, doubled_all_gradients=dgrad_all, doubled_all_iterations=dcount_all, doubled_all_damping=ddamp_all, doubled_stop_reasons=stopped, doubled_selected_start=dselected, doubled_z=dz, doubled_objective=dobj, doubled_gradient=dgrad)
    # The doubled result is used by a declared common rule, never a best-arm selector.
    return dz, {"nonstationary": int(np.sum(~np.isfinite(dgrad) | (dgrad > 1e-7))), "maximum_gradient": float(np.max(dgrad)), "max_doubled_objective_change": float(np.max(abs(dobj-objective[ii,selected]))), "selected_start_counts": np.bincount(dselected, minlength=8).tolist()}
