"""Validation-only Burgers-3D bank/head diagnostics and frozen-bank refinement.

See B3D-REPAIR-DESIGN.md. Truth is regenerated on the cluster. No test table is
opened. QR is a coordinate change of the learned bank, not a replacement basis.
All large arrays are explicit compiled-function arguments.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import b3d_common as b3
import sep_hfit as hf


def summarize(values):
    x = np.asarray(values, dtype=float)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.quantile(x, .95)), worst=float(x.max()))


def validate_parameter_manifest(generated, archived, expected_hash):
    """Exact random draws; tightly bounded rounding only in derived exp/max values.

    The archived metadata is a provenance reference, never the source of truth
    snapshots. All actual data still use the regenerated table. A changed seed,
    member, raw parameter or materially changed derived quantity fails closed.
    """
    assert archived["sha256"] == expected_hash, 'archived metadata does not belong to checkpoint'
    result = dict(archived_sha256=archived["sha256"], regenerated_sha256=generated["sha256"],
                  derived_rtol=8 * np.finfo(float).eps, fields={})
    for key in ['seed', 'm', 'B', 'c', 'w', 'rho', 'A', 'nu', 's_star']:
        old, new = np.asarray(archived[key]), np.asarray(generated[key])
        assert old.shape == new.shape and old.dtype == new.dtype, (key, 'shape/dtype')
        assert np.all(np.isfinite(old)) and np.all(np.isfinite(new)), (key, 'nonfinite')
        exact = np.array_equal(old, new)
        relative = float(np.max(np.abs(old-new) / np.maximum(np.abs(old), 1e-300)))
        if key in {'nu', 's_star'}:
            assert np.all(old > 0) and np.all(new > 0)
            assert relative <= result['derived_rtol'], (key, 'derived value mismatch', relative)
        else:
            assert exact, (key, 'random-draw/member mismatch')
        result['fields'][key] = dict(exact=bool(exact), max_relative=relative,
                                     different=int(np.sum(old != new)))
    return result


def make_fit(budget=800, gradient_tol=1e-8):
    """Exact coefficient-metric LM with field-normalized stationarity stopping.

    Unlike the legacy fit, the stopping rule tests the gradient rather than
    objective stagnation. The bank-orthogonal error participates in normalization.
    No claim of global optimality is made. Every start is retained by the caller.
    """
    def fit(qp, target, perpendicular2, norm2, z0):
        def residual(z):
            return hf.head_apply(qp, z) - target

        def rj(z):
            return residual(z), jax.jacfwd(residual)(z)

        def stationarity(r, jac):
            return jnp.linalg.norm(jac.T @ r) / (
                jnp.linalg.norm(jac) * jnp.sqrt(jnp.sum(r * r) + perpendicular2) + 1e-300)

        r, jac = rj(z0)
        opt = stationarity(r, jac)
        # reason: 0 budget, 1 stationary, 2 damping limit, 3 nonfinite.
        reason = jnp.where(~jnp.isfinite(opt), 3, jnp.where(opt <= gradient_tol, 1, 0))
        state = (z0, r, jac, jnp.asarray(1e-6), jnp.int32(0),
                 jnp.int32(0), reason.astype(jnp.int32), opt)

        def cond(s):
            return (s[4] < budget) & (s[6] == 0)

        def body(s):
            z, r, jac, damping, attempts, accepted, _, _ = s
            hess = jac.T @ jac
            diagonal = jnp.maximum(jnp.diag(hess), 1e-30)
            step = jnp.linalg.solve(hess + damping * jnp.diag(diagonal), -jac.T @ r)
            finite = jnp.all(jnp.isfinite(step))
            candidate = z + jnp.where(finite, step, 0.)
            new_r = residual(candidate)
            # Algebraic squared-norm difference avoids subtracting two nearly
            # equal objective values at the end of convergence.
            delta = new_r - r
            reduction = -2 * jnp.dot(r, delta) - jnp.dot(delta, delta)
            accept = finite & jnp.all(jnp.isfinite(new_r)) & (reduction > 0)
            z = jnp.where(accept, candidate, z)
            r, jac = jax.lax.cond(accept, lambda: rj(z), lambda: (r, jac))
            opt = stationarity(r, jac)
            damping = jnp.where(accept, jnp.maximum(damping / 3, 1e-12),
                               jnp.minimum(damping * 10, 1e12))
            reason = jnp.where(~jnp.isfinite(opt), 3,
                               jnp.where(opt <= gradient_tol, 1,
                                         jnp.where(damping >= 1e12, 2, 0)))
            return (z, r, jac, damping, attempts + 1,
                    accepted + accept.astype(jnp.int32), reason.astype(jnp.int32), opt)

        z, r, jac, damping, attempts, accepted, reason, opt = jax.lax.while_loop(cond, body, state)
        error = jnp.sqrt((jnp.dot(r, r) + perpendicular2) / norm2)
        return z, error, opt, attempts, accepted, reason

    return jax.jit(jax.vmap(fit, in_axes=(None, 0, 0, 0, 0)))


def fit_states(qp, target, perpendicular2, norm2, starts, budgets=(400, 800), chunk=128):
    """Same eight starts as the original pilot; never select on optimality."""
    count, ns = len(target), len(starts)
    a = np.repeat(target, ns, axis=0)
    f = np.repeat(perpendicular2, ns)
    u = np.repeat(norm2, ns)
    z0 = np.tile(starts, (count, 1))
    stages = []
    for budget in budgets:
        solve = make_fit(budget)
        collected = [[] for _ in range(6)]
        for s in range(0, len(a), chunk):
            args = [jnp.asarray(x[s:s+chunk]) for x in (a, f, u, z0)]
            result = solve(qp, *args)
            for dst, value in zip(collected, result):
                dst.append(np.asarray(value))
        all_results = [np.concatenate(x) for x in collected]
        zs = all_results[0].reshape(count, ns, -1)
        per_start = [x.reshape(count, ns) for x in all_results[1:]]
        best = np.argmin(per_start[0], axis=1)
        rows = np.arange(count)
        selected = [x[rows, best] for x in per_start]
        stages.append(dict(budget=budget, z=zs[rows, best], error=selected[0],
                           optimality=selected[1], attempts=selected[2],
                           accepted=selected[3], reason=selected[4],
                           best_start=best, all_errors=per_start[0],
                           all_optimality=per_start[1], all_attempts=per_start[2],
                           all_reasons=per_start[4]))
    return stages


def extract(params, cfg, table, n, outdir):
    """Regenerate training and validation separately, retain compact coordinates."""
    assert n == 33, "This first diagnostic is the full-interior N=33 pilot only"
    coords = b3.grid_coords_3d(n)
    interior = b3.interior_indices_3d(n)
    g = np.asarray(b3.features(params, jnp.asarray(coords[interior])))
    q, r = np.linalg.qr(g, mode="reduced")
    sv = np.linalg.svd(r, compute_uv=False)
    rank_cutoff = np.finfo(float).eps * max(g.shape) * sv[0]
    rank = int(np.sum(sv > rank_cutoff))
    assert rank == g.shape[1], ("rank-deficient bank; stop before head fitting", rank, g.shape[1])
    b3.log(f"bank rank={rank} condition={sv[0]/sv[-1]:.3e}")
    assert np.linalg.norm(g - q @ r) / np.linalg.norm(g) < 1e-12
    assert np.linalg.norm(q.T @ q - np.eye(rank)) < 1e-11
    ntrain = int(cfg["n_train_traj"])
    assert ntrain == 512 and cfg["val_rows"] == [512, 575]
    times = np.asarray([0, 10, 25, 50])
    num_times = b3.NUM_STEPS + 1
    pick = np.sort(np.random.default_rng(int(cfg["seed"])).choice(
        ntrain * num_times, int(cfg["max_snaps"]), replace=False))
    selected_train = set(pick.tolist())
    truth = b3.make_truth_rollout(n, "mm")
    qj = jnp.asarray(q)

    @jax.jit
    def project(fields, bank_q):
        target = fields @ bank_q
        norm2 = jnp.sum(fields * fields, axis=1)
        # Direct residual, not subtraction of nearly equal squared norms.
        diff = fields - target @ bank_q.T
        perpendicular2 = jnp.sum(diff * diff, axis=1)
        return target, norm2, perpendicular2

    datasets = {}
    worst = 0.
    for split, rows in [("train", np.arange(512)), ("validation", np.arange(512, 576))]:
        arrays = {key: [] for key in ["target", "norm2", "perpendicular2", "sid", "blob_count"]}
        fields_val = []
        for start in range(0, len(rows), 32):
            rr = rows[start:start+32]
            ic = np.stack([b3.blob_ic_3d(n, table, int(j), coords)[interior] for j in rr])
            snaps, res = truth(jnp.asarray(ic), jnp.asarray(table["nu"][rr]))
            assert bool(jnp.all(jnp.isfinite(snaps))) and np.isfinite(float(res))
            assert float(jnp.min(snaps)) >= -1e-9
            worst = max(worst, float(res))
            assert worst <= 1e-8
            ids = (rr[:, None] * num_times + np.arange(num_times)).ravel()
            mask = np.asarray([int(v) in selected_train for v in ids]) if split == "train" else np.isin(ids % num_times, times)
            selected = snaps.reshape(-1, len(interior))[jnp.asarray(np.flatnonzero(mask))]
            a, u2, p2 = project(selected, qj)
            arrays["target"].append(np.asarray(a))
            arrays["norm2"].append(np.asarray(u2))
            arrays["perpendicular2"].append(np.asarray(p2))
            arrays["sid"].append(ids[mask])
            arrays["blob_count"].append(np.asarray(table["B"])[ids[mask] // num_times])
            if split == "validation":
                fields_val.append(np.asarray(selected))
        datasets[split] = {key: np.concatenate(value) for key, value in arrays.items()}
        if split == "validation":
            datasets[split]["fields"] = np.concatenate(fields_val)
        b3.log(f"extracted {split}: {len(datasets[split]['sid'])} states")
    assert np.array_equal(datasets["train"]["sid"], pick)
    return g, q, r, datasets, dict(rank=rank, rank_cutoff=float(rank_cutoff),
                                  condition=float(sv[0]/sv[-1]), singular_values=sv.tolist(),
                                  truth_max_residual=worst)


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


def refine_head(params, codes, r, train, *, steps, lr, seed, batch):
    """Training-only warm refinement; return a drop-in decoder and audit data.

    QR changes the optimizer coordinates, but leaves the global field-MSE
    objective and represented model class unchanged. The orthogonal term is
    constant during this fit and is included in the reported objective.
    """
    assert steps > 0 and lr > 0 and batch > 0
    hp = dict(h=params['h'], h_lin=params['h_lin'])
    qp = hf.to_q(hp, jnp.asarray(r.T))
    a, p2, n2 = [jnp.asarray(train[key]) for key in ('target', 'perpendicular2', 'norm2')]
    before = np.asarray(hf.batched_rel(qp, jnp.asarray(codes), a, p2, n2))
    qp, new_codes, info = hf.fit(
        jax.random.PRNGKey(seed), a, n2, p2, codes.shape[1], r.shape[0],
        steps=steps, lr=lr, hidden=params['h'][0][0].shape[1],
        layers=len(params['h']) - 1, batch=batch, qp0=qp, Z0=codes,
        norm='global', log_every=5000, tag='b3d_frozen_bank')
    after = np.asarray(hf.batched_rel(qp, new_codes, a, p2, n2))
    assert np.all(np.isfinite(after)) and np.all(np.isfinite(new_codes))
    hp_new = hf.to_h(qp, jnp.asarray(np.linalg.inv(r.T)))
    new_params = dict(params, **hp_new)
    for key in params.keys() - {'h', 'h_lin'}:
        for old, new in zip(jax.tree_util.tree_leaves(params[key]),
                            jax.tree_util.tree_leaves(new_params[key])):
            assert np.array_equal(old, new), ('changed frozen bank parameter', key)
    probe = new_codes[:64]
    q_values = np.asarray(hf.head_apply(qp, probe))
    h_values = np.asarray(hf.head_apply(hp_new, probe))
    identity = np.linalg.norm(q_values - h_values @ r.T) / np.linalg.norm(q_values)
    assert identity < 1e-12
    info.update(seed=seed, bank_unchanged=True, conversion_error=float(identity),
                training_membership_unchanged=True, training_states=len(codes),
                reconstruction_before=summarize(before), reconstruction_after=summarize(after),
                global_relative_mse_before=float(np.sum(before**2 * train['norm2']) / np.sum(train['norm2'])),
                global_relative_mse_after=float(np.sum(after**2 * train['norm2']) / np.sum(train['norm2'])))
    return new_params, np.asarray(new_codes), qp, info


def main():
    started = time.time()
    out = Path(os.environ["OUT"])
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = Path(os.environ["CKPT"])
    assert jax.default_backend() == "gpu", "real diagnostics require a GPU"
    assert os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") == "highest"
    params, codes, cfg = b3.load_pkl(ckpt)
    assert (cfg["N"], cfg["k"], cfg["r"], cfg["seed"]) == (33, 32, 128, 0)
    report = dict(complete=False, kind="b3d_repair_diagnostic", config=dict(
        source_config=cfg, source_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        commit=os.environ["COMMIT"], slurm_job=os.environ["SLURM_JOB_ID"],
        node=os.environ.get("SLURMD_NODENAME"), backend=jax.default_backend(),
        gpu=jax.devices()[0].device_kind, jax_version=jax.__version__,
        x64=bool(jax.config.x64_enabled), matmul_precision=os.environ["JAX_DEFAULT_MATMUL_PRECISION"],
        test_table_opened=False))

    def save():
        temporary = out.with_suffix(".writing")
        temporary.write_text(json.dumps(report, indent=1, default=jsonable, allow_nan=False))
        temporary.replace(out)

    save()
    tables = b3.get_tables(os.environ["TABLE_DIR"], 576, 8, 0, 1, with_test=False)
    report["config"]["table_sha256"] = tables["train"]["sha256"]
    archived = b3.load_param_table(os.environ["REFERENCE_TABLE"])
    report['parameter_manifest_check'] = validate_parameter_manifest(
        tables['train'], archived, cfg['table_sha256'])
    save()
    g, q, r, data, bank_info = extract(params, cfg, tables["train"], 33, out.parent)
    report["bank"] = bank_info
    for split, d in data.items():
        report[split + "_bank_error"] = summarize(np.sqrt(d["perpendicular2"] / d["norm2"]))
    hp = dict(h=params["h"], h_lin=params["h_lin"])
    # to_q folds a RIGHT output transform into the last layers. With G=QR,
    # q(z)=R h(z), so its row-vector transform is R.T.
    qp = hf.to_q(hp, jnp.asarray(r.T))
    probe = jnp.asarray(codes[:64])
    qvals = np.asarray(hf.head_apply(qp, probe))
    hvals = np.asarray(b3.head(params, probe))
    transform_error = np.linalg.norm(qvals - hvals @ r.T) / np.linalg.norm(qvals)
    assert transform_error < 1e-12
    report["transform_error"] = float(transform_error)
    train = data["train"]
    recon = np.asarray(hf.batched_rel(qp, jnp.asarray(codes), jnp.asarray(train["target"]),
                                    jnp.asarray(train["perpendicular2"]), jnp.asarray(train["norm2"])))
    report["train_reconstruction"] = summarize(recon)
    save()
    refine_steps = int(os.environ.get('REFINE_STEPS', '0'))
    assert refine_steps >= 0
    report['config']['mode'] = 'frozen_bank_refinement' if refine_steps else 'incumbent_diagnostic'
    if refine_steps:
        params, codes, qp, info = refine_head(
            params, codes, r, train, steps=refine_steps,
            lr=float(os.environ.get('REFINE_LR', '3e-4')),
            seed=int(os.environ.get('REFINE_SEED', '200')),
            batch=int(os.environ.get('REFINE_BATCH', '4096')))
        report['refinement'] = info
        candidate = out.parent / 'refined_checkpoint.pkl'
        candidate_cfg = dict(cfg, refinement=dict(
            **info, source_checkpoint_sha256=report['config']['source_sha256'],
            commit=os.environ['COMMIT'], slurm_job=os.environ['SLURM_JOB_ID'],
            gpu=report['config']['gpu'], table_sha256=tables['train']['sha256']))
        b3.save_pkl(candidate, params, codes, candidate_cfg)
        report['config']['candidate_checkpoint_sha256'] = b3.sha256_file(candidate)
        save()
    val = data["validation"]
    starts = np.concatenate([np.zeros((1, 32)), codes[np.random.default_rng(5).choice(len(codes), 7, replace=False)]])
    stages = fit_states(qp, val["target"], val["perpendicular2"], val["norm2"], starts)
    report["fits"] = []
    for stage in stages:
        z = jnp.asarray(stage["z"])
        field_r = np.asarray(b3.head(params, z)) @ g.T - val["fields"]
        direct_error = np.linalg.norm(field_r, axis=1) / np.sqrt(val["norm2"])
        assert np.max(np.abs(direct_error - stage["error"])) < 1e-11
        # Independently form the field Jacobian on the worst-gradient state.
        idx = int(np.argmax(stage["optimality"]))
        jac = g @ np.asarray(jax.jacfwd(lambda zz: b3.head(params, zz))(z[idx]))
        direct_opt = np.linalg.norm(jac.T @ field_r[idx]) / (np.linalg.norm(jac) * np.linalg.norm(field_r[idx]) + 1e-300)
        assert abs(direct_opt - stage["optimality"][idx]) < 1e-10
        stage["summary"] = summarize(stage["error"])
        stage["nonstationary"] = int(np.sum(stage["optimality"] > 1e-6))
        stage["direct_optimality_check"] = dict(index=idx, value=float(direct_opt))
        stage["groups"] = {}
        for name, mask in [("initial", val["sid"] % 51 == 0), ("later", val["sid"] % 51 > 0)] + [
                (f"blobs_{b}", val["blob_count"] == b) for b in (1, 2, 3)]:
            stage["groups"][name] = summarize(stage["error"][mask])
        report["fits"].append(stage)
        b3.log(f"fit budget={stage['budget']} mean={stage['summary']['mean']:.6e} worst={stage['summary']['worst']:.6e} nonstationary={stage['nonstationary']}")
        save()
    report["validation_states"] = {key: val[key] for key in ["sid", "blob_count", "norm2", "perpendicular2"]}
    report["budget_change_max"] = float(np.max(np.abs(stages[1]["error"]-stages[0]["error"]) / stages[0]["error"]))
    report["seconds"] = time.time() - started
    report["complete"] = True
    save()
    b3.log(f"DONE {out} [{report['seconds']:.1f}s]")


if __name__ == "__main__":
    main()
