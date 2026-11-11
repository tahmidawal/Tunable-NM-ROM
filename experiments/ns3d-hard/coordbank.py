"""ONE coordinate-network bank for the hard family (PROTOCOL.md amendment A5.1).

  train     fit the network on the POD bank's own snapshots (variable projection, random
            sub-cell grid shifts; Adam then L-BFGS), then run `finalise`
  finalise  read the bank on every mesh, orthonormalise, measure development floors against
            the per-mesh POD bank, write G_<n>.npy (plugs into pipeline.load_rom)
  ops       weak operators (pipeline.build_ops, unchanged) for a finalised bank

The network is a periodic vector potential psi : T^3 -> R^{3 x R}:
  z(x)   = Re sum_{max|k_i|<=kmax} c_k exp(2 pi i k.x)        (width channels; first layer)
  h      = SiLU(z); h = SiLU(h W_l + b_l) ...; psi = h W_out
On a uniform mesh the first layer is an inverse FFT of the zero-padded coefficients (the same
function as the direct sum; `selfcheck`). The bank on mesh n is the spectral curl of the
sampled potential, masked to the solver's modes (|k_i| < n/2) and scaled by n^{-3/2}.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline as P  # noqa: E402  (sets x64, paths)
import jax, jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from pipeline import H, D, log, jdump  # noqa: E402

AX = (-3, -2, -1)


# ------------------------------------------------------------------ network --

def envelope(kmax):
    k = np.arange(-kmax, kmax + 1)
    k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    return 1.0 / (1.0 + k2 / (kmax / 2.0) ** 2)


def init_params(seed, rank, width, hidden, kmax):
    rng = np.random.default_rng(seed); K = 2 * kmax + 1
    env = envelope(kmax)
    scale = 1.0 / np.sqrt(0.5 * np.sum(env ** 2) * 2)      # Var z = 1/2 sum a^2 (wr^2 + wi^2)
    p = dict(wr=rng.normal(size=(width, K, K, K)) * scale, wi=rng.normal(size=(width, K, K, K)) * scale,
             layers=[], out=rng.normal(size=(width, 3 * rank)) * np.sqrt(1.0 / width))
    for _ in range(hidden - 1):
        p["layers"].append((rng.normal(size=(width, width)) * np.sqrt(2.0 / width), np.zeros(width)))
    return jax.tree_util.tree_map(jnp.asarray, p)


def first_layer(p, n, delta, kmax, env):
    """z on the mesh-n grid displaced by delta (box units): (width, n, n, n)."""
    k = jnp.arange(-kmax, kmax + 1)
    ph = [jnp.exp(2j * jnp.pi * k * delta[a]) for a in range(3)]
    c = (p["wr"] + 1j * p["wi"]) * env * (ph[0][:, None, None] * ph[1][None, :, None] * ph[2][None, None, :])
    idx = np.arange(-kmax, kmax + 1) % n
    spec = jnp.zeros((c.shape[0], n, n, n), dtype=c.dtype)
    spec = spec.at[:, idx[:, None, None], idx[None, :, None], idx[None, None, :]].set(c)
    return jnp.fft.ifftn(spec, axes=AX, norm="forward").real


def tail(p, z):
    """z (points, width) -> psi (points, 3R)."""
    h = jax.nn.silu(z)
    for w, b in p["layers"]:
        h = jax.nn.silu(h @ w + b)
    return h @ p["out"]


def curl_bank(psi, n, delta):
    """psi (n^3, 3R) sampled on the displaced grid -> bank (3 n^3, R): samples shifted back
    spectrally, spectral curl, mask |k_i| < n/2, scale n^{-3/2}."""
    R = psi.shape[1] // 3
    f = psi.T.reshape(3, R, n, n, n)
    k = jnp.asarray(np.fft.fftfreq(n, d=1.0 / n))
    hat = jnp.fft.fftn(f, axes=AX, norm="forward")
    back = (jnp.exp(-2j * jnp.pi * k * delta[0])[:, None, None] * jnp.exp(-2j * jnp.pi * k * delta[1])[None, :, None]
            * jnp.exp(-2j * jnp.pi * k * delta[2])[None, None, :])
    hat = hat * back * jnp.asarray(H.keep_mask(n))
    kx, ky, kz = k[:, None, None], k[None, :, None], k[None, None, :]
    c = 2j * jnp.pi
    cur = jnp.stack((c * (ky * hat[2] - kz * hat[1]), c * (kz * hat[0] - kx * hat[2]), c * (kx * hat[1] - ky * hat[0])))
    g = jnp.fft.ifftn(cur, axes=AX, norm="forward").real            # (3, R, n, n, n)
    return jnp.transpose(g, (0, 2, 3, 4, 1)).reshape(3 * n ** 3, R) * n ** -1.5


def sample(p, n, delta, kmax, env):
    z = first_layer(p, n, delta, kmax, env)
    return curl_bank(tail(p, z.reshape(z.shape[0], -1).T), n, delta)


def vp_loss(G, Y, ysq, ridge=1e-12):
    M = G.T @ G; B = G.T @ Y
    eps = ridge * jnp.trace(M) / M.shape[0]
    L = jnp.linalg.cholesky(M + eps * jnp.eye(M.shape[0], dtype=M.dtype))
    W = jax.scipy.linalg.solve_triangular(L, B, lower=True)
    return 1.0 - jnp.sum(W * W) / ysq


def sample_big(p, n, kmax, env, cchunk=64, pchunk=131072):
    """Forward-only sample at delta = 0 with bounded device memory (large meshes)."""
    width = p["wr"].shape[0]; zero = jnp.zeros(3)
    fl = jax.jit(lambda q: first_layer(q, n, zero, kmax, env))
    z = np.empty((n ** 3, width))
    for s in range(0, width, cchunk):
        q = dict(wr=p["wr"][s:s + cchunk], wi=p["wi"][s:s + cchunk])
        z[:, s:s + cchunk] = np.asarray(fl(q)).reshape(-1, n ** 3).T
    tl = jax.jit(tail)
    psi = np.empty((n ** 3, p["out"].shape[1]))
    for s in range(0, n ** 3, pchunk):
        psi[s:s + pchunk] = np.asarray(tl(p, jnp.asarray(z[s:s + pchunk])))
    del z; gc.collect()
    R = psi.shape[1] // 3; G = np.empty((3 * n ** 3, R)); cb = jax.jit(lambda x: curl_bank(x, n, zero))
    for s in range(0, R, 16):
        e = min(s + 16, R)
        cols = np.concatenate([psi[:, c * R + s:c * R + e] for c in range(3)], axis=1)
        G[:, s:e] = np.asarray(cb(jnp.asarray(cols)))
    return G


def save_params(path, p, meta):
    arrs = dict(wr=np.asarray(p["wr"]), wi=np.asarray(p["wi"]), out=np.asarray(p["out"]))
    for i, (w, b) in enumerate(p["layers"]):
        arrs[f"w{i}"] = np.asarray(w); arrs[f"b{i}"] = np.asarray(b)
    tmp = Path(str(path) + ".tmp.npz"); np.savez(tmp, meta=json.dumps(meta), **arrs); tmp.replace(path)


def load_params(path):
    z = np.load(path); meta = json.loads(str(z["meta"]))
    n = sum(1 for k in z.files if k.startswith("w") and k[1:].isdigit())
    p = dict(wr=z["wr"], wi=z["wi"], out=z["out"], layers=[(z[f"w{i}"], z[f"b{i}"]) for i in range(n)])
    return jax.tree_util.tree_map(jnp.asarray, p), meta


# --------------------------------------------------------------------- data --

def centred_snapshots(dirs, n, cases=None):
    """Rows as pipeline.cmd_bank: (cases*6, 3 n^3) centred fields; also ||u0|| per case."""
    arrs, par = P.load_truth(dirs)
    total = sum(x.shape[0] for x in arrs) if cases is None else cases
    centre = P.make_centre(n)
    X = np.empty((total * 6, 3 * n ** 3)); c = 0
    for fr in P.iter_cases(arrs):
        if c >= total: break
        y = P.to_mesh(np.asarray(fr), n)
        for t in range(6):
            X[c * 6 + t] = np.asarray(centre(y[t])).ravel()
        c += 1
    return X


def floors(G, X):
    """Projection floor per case and frame, error / ||u0||: (cases, 6)."""
    Gj = jnp.asarray(G); out = np.empty(X.shape[0])
    f = jax.jit(lambda v: jnp.sqrt(jnp.maximum(jnp.sum(v * v) - jnp.sum((Gj.T @ v) ** 2), 0.0)))
    for i in range(X.shape[0]):
        out[i] = float(f(jnp.asarray(X[i])))
    out = out.reshape(-1, 6)
    u0 = np.linalg.norm(X.reshape(-1, 6, X.shape[1])[:, 0], axis=1)
    return out / u0[:, None]


def lowdin(B):
    M = B.T @ B; lam, V = np.linalg.eigh(M)
    return B @ (V * lam ** -0.5) @ V.T, float(np.max(np.abs(M - np.eye(M.shape[0]))))


# ----------------------------------------------------------------- finalise --

def finalise(p, meta, a, out, rep):
    kmax = meta["kmax"]; env = jnp.asarray(envelope(kmax)); nt = meta["train_grid"]
    G64 = sample_big(p, nt, kmax, env)
    Q, Rq = np.linalg.qr(G64)
    np.save(out / "R_train.npy", Rq)
    rep["finalise"] = dict(meshes={}, R_condition=float(np.linalg.cond(Rq)))
    devdirs = a.dev.split(",")
    for n in P.ints(a.meshes):
        t0 = time.time()
        Graw = G64 if n == nt else sample_big(p, n, kmax, env)
        B = np.linalg.solve(Rq.T, Graw.T).T
        G, dev0 = lowdin(B)
        orth = float(np.max(np.abs(G.T @ G - np.eye(G.shape[1]))))
        if orth > 1e-11:
            G, _ = lowdin(G); orth = float(np.max(np.abs(G.T @ G - np.eye(G.shape[1]))))
        np.save(out / f"G_{n}.npy", G)
        X = centred_snapshots(devdirs, n)
        e = floors(G, X)
        m = dict(pre_lowdin_gram_deviation=dev0, orthonormality=orth, floor=P.stats(e))
        if a.pod:
            Gp = np.load(Path(a.pod) / "output" / f"G_{n}.npy")
            if Gp.shape[1] > G.shape[1]: Gp = Gp[:, :G.shape[1]]
            ep = floors(Gp, X); m["pod_floor"] = P.stats(ep)
            m["floor_ratio_worst"] = m["floor"]["evolved_worst"] / m["pod_floor"]["evolved_worst"]
            m["floor_ratio_median"] = m["floor"]["evolved_median"] / m["pod_floor"]["evolved_median"]
            m["pod_rank"] = int(Gp.shape[1]); del Gp
        m["seconds"] = time.time() - t0
        rep["finalise"]["meshes"][str(n)] = m
        log(f"finalise mesh {n}: gram dev {dev0:.3e} floor worst {m['floor']['evolved_worst']:.5f} median "
            f"{m['floor']['evolved_median']:.5f}" + (f" | POD {m['pod_floor']['evolved_worst']:.5f} "
            f"{m['pod_floor']['evolved_median']:.5f} ratio {m['floor_ratio_worst']:.3f}" if a.pod else ""))
        jdump(out / "coordbank.json", rep)
        del G, X, B; gc.collect()


# -------------------------------------------------------------------- train --

def cmd_train(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if jax.default_backend() != "gpu" and not a.smoke:
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")
    t_job = time.time(); nt = a.train_grid; kmax = a.kmax
    meta = dict(rank=a.rank, width=a.width, hidden=a.hidden, kmax=kmax, train_grid=nt, seed=a.seed)
    rep = dict(schema="ns3d-hard-coordbank-v1", args=vars(a), meta=meta, device=str(jax.devices()[0]),
               job_id=os.environ.get("SLURM_JOB_ID"))
    cfile = HERE.parents[1] / "COMMIT.txt"
    rep["source_commit"] = cfile.read_text().strip() if cfile.exists() else None
    X = centred_snapshots(a.inputs.split(","), nt, a.cases or None)
    ysq = float(np.sum(X * X))
    Y = jnp.asarray(np.ascontiguousarray(X.T)); del X; gc.collect()
    log(f"snapshots {Y.shape} energy {ysq:.6e}")
    if a.pod:
        Gp = jnp.asarray(np.load(Path(a.pod) / "output" / f"G_{nt}.npy")[:, :a.rank])
        rep["pod_loss"] = float(1.0 - jnp.sum((Gp.T @ Y) ** 2) / ysq); del Gp
        log(f"POD rank-{a.rank} loss (optimum) {rep['pod_loss']:.6e}")
    env = jnp.asarray(envelope(kmax))
    p = init_params(a.seed, a.rank, a.width, a.hidden, kmax)
    rep["parameters"] = int(sum(x.size for x in jax.tree_util.tree_leaves(p)))
    h = 1.0 / nt

    def loss_at(q, delta, Y):
        return vp_loss(sample(q, nt, delta, kmax, env), Y, ysq)

    fixed = jnp.asarray(np.array([[0.0, 0.0, 0.0], [0.37, 0.71, 0.53]]) * h)

    def loss_fixed(q, Y):
        return 0.5 * (loss_at(q, fixed[0], Y) + loss_at(q, fixed[1], Y))
    loss_fixed_j = jax.jit(loss_fixed); loss0_j = jax.jit(lambda q, Y: loss_at(q, fixed[0], Y))

    def dump():
        rep["elapsed_seconds"] = time.time() - t_job; jdump(out / "coordbank.json", rep)

    def certify(q, label):
        r = dict(label=label, loss_grid=float(loss0_j(q, Y)), loss_two_shift=float(loss_fixed_j(q, Y)))
        if a.pod: r["ratio_to_pod"] = r["loss_grid"] / rep["pod_loss"]
        log(f"[{label}] {r}")
        return r
    rep["stages"] = [certify(p, "init")]; dump()
    every = a.log_every; key0 = jax.random.PRNGKey(a.seed + 1)

    def make_chunk(opt):
        @jax.jit
        def chunk(q, st, start, Y):
            def body(carry, i):
                q, st = carry
                delta = jax.random.uniform(jax.random.fold_in(key0, i), (3,)) * h
                v, g = jax.value_and_grad(loss_at)(q, delta, Y)
                upd, st = opt.update(g, st, q)
                return (optax.apply_updates(q, upd), st), v
            (q, st), vals = jax.lax.scan(body, (q, st), start + jnp.arange(every))
            return q, st, jnp.mean(vals)
        return chunk

    if a.adam_seconds > 0:
        copt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-7))
        cc = make_chunk(copt); q, s, _ = cc(p, copt.init(p), 0, Y); jax.block_until_ready(q)
        t0 = time.time(); q, s, _ = cc(q, s, 0, Y); jax.block_until_ready(q)
        per = (time.time() - t0) / every; del q, s, cc
        steps = max(int(0.97 * a.adam_seconds / per) // every * every, every)
        warm = min(a.warmup, steps // 4)
        log(f"[adam] {per * 1e3:.1f} ms/step -> {steps} steps")
        sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, warm, steps, a.lr * a.final_lr_ratio)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
        st = opt.init(p); chunk = make_chunk(opt); curve = []; t0 = time.time(); t_ck = t0; done = 0
        while done < steps:
            p, st, v = chunk(p, st, done, Y); done += every; v = float(v)
            curve.append((done, v, time.time() - t0))
            if done % (every * 5) == 0: log(f"  [adam] {done}/{steps} loss {v:.6e} ({time.time() - t0:.0f}s)")
            if not np.isfinite(v): raise RuntimeError("nonfinite Adam training")
            if time.time() - t_ck > 900:
                save_params(out / "params_partial.npz", p, meta); t_ck = time.time()
        save_params(out / "params_adam.npz", p, meta)
        r = certify(p, "adam"); r.update(steps=steps, seconds_per_step=per, seconds=time.time() - t0, curve=curve)
        rep["stages"].append(r); dump()

    if a.lbfgs_seconds > 0:
        lopt = optax.lbfgs(memory_size=20, linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=30, initial_guess_strategy="one"))
        per_call = a.lbfgs_per_call

        @jax.jit
        def lchunk(q, state, Y):
            f = lambda x: loss_fixed(x, Y)
            vg = optax.value_and_grad_from_state(f)

            def body(carry, _):
                q, st, bq, bv = carry
                v, g = vg(q, state=st)
                better = jnp.isfinite(v) & (v < bv)
                bq = jax.tree_util.tree_map(lambda x, y: jnp.where(better, x, y), q, bq)
                bv = jnp.where(better, v, bv)
                upd, st = lopt.update(g, st, q, value=v, grad=g, value_fn=f)
                return (optax.apply_updates(q, upd), st, bq, bv), v
            init = (q, state, q, jnp.asarray(jnp.inf, dtype=jnp.float64))
            (q, st, bq, bv), vals = jax.lax.scan(body, init, None, length=per_call)
            return q, st, bq, bv
        ls = lopt.init(p); t0 = time.time(); t_ck = t0; it = 0; lcurve = []; stop = "deadline"
        prev = float(loss_fixed_j(p, Y)); start_value = prev; stall = 0
        while time.time() - t0 < a.lbfgs_seconds:
            try:
                qn, sn, qb, vb = lchunk(p, ls, Y); cur = float(loss_fixed_j(qn, Y))
            except Exception as exc:
                stop = f"exception {type(exc).__name__}: {str(exc)[:300]} (last accepted iterate kept)"; break
            if not (np.isfinite(cur) and cur <= prev):
                vb = float(vb)
                if np.isfinite(vb) and vb < prev:
                    p = qb; it += per_call; lcurve.append((it, vb, time.time() - t0))
                    stop = f"objective rose over the call ({prev:.6e} -> {cur}); best iterate of the call kept ({vb:.6e})"
                else:
                    stop = f"no decrease over the call ({prev:.6e} -> {cur}); previous iterate kept"
                break
            p, ls = qn, sn; it += per_call; lcurve.append((it, cur, time.time() - t0))
            log(f"  [lbfgs] iter {it} loss {cur:.6e} ({time.time() - t0:.0f}s)")
            stall = stall + 1 if (prev - cur) <= 1e-9 * abs(prev) else 0
            prev = cur
            if stall >= 8: stop = "stalled"; break
            if time.time() - t_ck > 900:
                save_params(out / "params_partial.npz", p, meta); t_ck = time.time()
        r = certify(p, "lbfgs"); r.update(iterations=it, stop=stop, start_value=start_value,
                                          seconds=time.time() - t0, curve=lcurve)
        rep["stages"].append(r)
    save_params(out / "params_final.npz", p, meta)
    (out / "params_partial.npz").unlink(missing_ok=True)
    rep["train_seconds"] = time.time() - t_job; dump()
    del Y; gc.collect()
    if a.dev:
        finalise(p, meta, a, out, rep)
    rep["status"] = "final"; dump(); log("DONE")


def cmd_finalise(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p, meta = load_params(a.params)
    rep = dict(schema="ns3d-hard-coordbank-v1", args=vars(a), meta=meta, device=str(jax.devices()[0]),
               job_id=os.environ.get("SLURM_JOB_ID"))
    finalise(p, meta, a, out, rep); rep["status"] = "final"; jdump(out / "coordbank.json", rep); log("DONE")


def cmd_ops(a):
    """Weak operators for a finalised bank; G files are copied so that this directory is a
    complete bank directory for pipeline.load_rom."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rep = dict(bank=a.bank, modes=a.modes, meshes={})
    for n in P.ints(a.meshes):
        t0 = time.time(); G = np.load(Path(a.bank) / "output" / f"G_{n}.npy")
        np.save(out / f"G_{n}.npy", G)
        ops, orep = P.build_ops(G, n, a.modes)
        np.savez(out / f"ops_{n}.npz", A=ops["A"], T=ops["T"], Dd=ops["Dd"], lam=ops["lam"])
        rep["meshes"][str(n)] = dict(operator_checks=orep, seconds=time.time() - t0)
        jdump(out / "ops.json", rep); log(f"ops {n}: {orep} {time.time() - t0:.0f}s"); del G, ops; gc.collect()
    log("DONE")


def cmd_selfcheck(a):
    """FFT first layer == direct sum; shift-back exact for a band-limited potential; curl is
    divergence-free; tiny training run decreases the loss."""
    n, kmax, R, W = 12, 2, 3, 5; env = jnp.asarray(envelope(kmax)); rng = np.random.default_rng(0)
    p = init_params(1, R, W, 2, kmax); delta = jnp.asarray([0.013, 0.031, 0.007])
    z = np.asarray(first_layer(p, n, delta, kmax, env))
    s = np.arange(n) / n; x = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3) + np.asarray(delta)
    k = np.arange(-kmax, kmax + 1); kk = np.stack(np.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3)
    c = (np.asarray(p["wr"]) + 1j * np.asarray(p["wi"])) * np.asarray(env)
    direct = (np.exp(2j * np.pi * x @ kk.T) @ c.reshape(W, -1).T).real
    e1 = float(np.max(np.abs(direct - z.reshape(W, -1).T)))
    lin = lambda d: np.asarray(curl_bank(jnp.asarray(np.asarray(first_layer(p, n, d, kmax, env)).reshape(W, -1).T[:, :3]), n, d))
    e2 = float(np.max(np.abs(lin(delta) - lin(jnp.zeros(3)))))
    G = np.asarray(sample(p, n, delta, kmax, env)); f = G.T.reshape(R, 3, n, n, n)
    kf = np.fft.fftfreq(n, d=1.0 / n); fh = np.fft.fftn(f, axes=AX)
    div = kf[:, None, None] * fh[:, 0] + kf[None, :, None] * fh[:, 1] + kf[None, None, :] * fh[:, 2]
    e3 = float(np.max(np.abs(div)) / np.max(np.abs(fh)))
    e4 = float(np.max(np.abs(sample_big(p, n, kmax, env, cchunk=2, pchunk=500) - np.asarray(sample(p, n, jnp.zeros(3), kmax, env)))))
    print(dict(first_layer=e1, shift_back=e2, divergence=e3, sample_big=e4))
    assert max(e1, e2, e3, e4) < 1e-10
    print("SELFCHECK OK")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    def fin(q):
        q.add_argument("--out", required=True); q.add_argument("--dev", default=None)
        q.add_argument("--pod", default=None); q.add_argument("--meshes", default="64,96,128")
    q = sub.add_parser("train"); fin(q); q.add_argument("--inputs", required=True)
    q.add_argument("--rank", type=int, default=64); q.add_argument("--width", type=int, default=512)
    q.add_argument("--hidden", type=int, default=4); q.add_argument("--kmax", type=int, default=8)
    q.add_argument("--train-grid", type=int, default=64); q.add_argument("--seed", type=int, default=20261111)
    q.add_argument("--adam-seconds", type=float, default=9000); q.add_argument("--lbfgs-seconds", type=float, default=3600)
    q.add_argument("--lr", type=float, default=1e-3); q.add_argument("--final-lr-ratio", type=float, default=1e-3)
    q.add_argument("--warmup", type=int, default=1000); q.add_argument("--log-every", type=int, default=100)
    q.add_argument("--lbfgs-per-call", type=int, default=25); q.add_argument("--cases", type=int, default=0)
    q.add_argument("--smoke", action="store_true")
    q = sub.add_parser("finalise"); fin(q); q.add_argument("--params", required=True)
    q = sub.add_parser("ops"); q.add_argument("--out", required=True); q.add_argument("--bank", required=True)
    q.add_argument("--meshes", default="64,96,128"); q.add_argument("--modes", type=int, default=2336)
    q = sub.add_parser("selfcheck")
    a = ap.parse_args()
    log(f"backend={jax.default_backend()} args={vars(a)}")
    dict(train=cmd_train, finalise=cmd_finalise, ops=cmd_ops, selfcheck=cmd_selfcheck)[a.cmd](a)


if __name__ == "__main__":
    main()
