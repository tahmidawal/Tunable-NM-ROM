"""Main experiment for the chosen family (PROTOCOL.md section 2). Stages, one job each:

  gen       truth trajectories (fine reference solver), stored on the storage mesh
  bank      per-mesh centred POD bank + weak operators (A, T, Dd, lam) for M_max tests
  headdata  centred bank coefficients of the head-training trajectories (on the fly)
  romdev    rotation, heads, floors, validation rollouts, selection -> frozen.json
  fom       free pseudo-spectral FOM ladder (mesh x steps x scheme) on a cohort
  test      frozen arms on the sealed cohort + FOM ladder + interleaved timing + slices

Every job writes only into its own --out directory and reads earlier job directories.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for extra in ("ns3d", "ns2d", "separable-decoder", "ns3d-grok", "ns3d-shift", "ns3d-shift-head"):
    sys.path.insert(0, str(HERE.parent / extra))
sys.path.insert(0, str(HERE))
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import hard_fom as H
from hard_fom import F
import diag_floor as D
import shift_rom as SR
import head_rom as HR

log = D.log
SEEDS = dict(train=2026111111, dev=2026111112, test=2026111113)
HEAD = dict(width=256, steps=100000, lr=1e-3, floor_lr=1e-5, code_reg=1e-6, seed=20260923,
            ic_iters=12, damping=1e-6, val_every=8)


def jdump(path, obj):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); tmp.replace(path)


def stats(errors):
    e = np.asarray(errors, dtype=np.float64)
    if not np.all(np.isfinite(e)):
        return dict(finite=False, evolved_worst=float("inf"), evolved_median=float("inf"))
    w = e[:, 1:].max(axis=1)
    return dict(finite=True, evolved_worst=float(w.max()), evolved_median=float(np.median(w)),
                initial_worst=float(e[:, 0].max()), initial_median=float(np.median(e[:, 0])),
                final_worst=float(e[:, -1].max()), final_median=float(np.median(e[:, -1])),
                per_time_median=[float(x) for x in np.median(e, axis=0)],
                per_time_worst=[float(x) for x in e.max(axis=0)], cases=int(len(e)))


def cohort_params(cohort, count, fam):
    seed = SEEDS["train" if cohort in ("bank", "head") else cohort]
    return H.parameters(seed, count, fam)


@jax.jit
def rel_err(pred, truth):
    num = jnp.sqrt(jnp.sum((pred - truth) ** 2, axis=(1, 2, 3, 4)))
    return num / jnp.sqrt(jnp.sum(truth[0] ** 2))


def make_centre(n):
    return jax.jit(lambda f: SR.shift_field(f, -SR.grid_centroid(f) * n))


def to_mesh(frames, n):
    """(T,3,S,S,S) on the storage mesh -> jnp (T,3,n,n,n) Fourier truncation."""
    frames = jnp.asarray(frames)
    if frames.shape[-1] == n:
        return frames
    return jnp.stack([H.truncate_field(f, n) for f in frames])


def load_truth(dirs):
    """Concatenate gen outputs (memmaps) in the order given."""
    arrs = [np.load(Path(d) / "output" / "truth.npy", mmap_mode="r") for d in dirs]
    pars = np.concatenate([np.load(Path(d) / "output" / "params.npy") for d in dirs])
    return arrs, pars


def iter_cases(arrs):
    for a in arrs:
        for i in range(a.shape[0]):
            yield a[i]


# ------------------------------------------------------------------------ gen --

def cmd_gen(a):
    fam = H.FAMILIES[a.family]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    total = dict(bank=128, dev=64, test=32)[a.cohort] if a.total is None else a.total
    par = cohort_params(a.cohort, total, fam)[a.start:a.start + a.count]
    steps = D.nsteps_for(a.truth_dt, fam["horizon"])
    solver = F.make_solver(a.truth_dt, steps, steps // 5)
    geom = F.geometry(a.truth_n); S = a.store_n
    store = np.lib.format.open_memmap(out / "truth.npy", mode="w+", dtype=np.float64,
                                      shape=(len(par), 6, 3, S, S, S))
    np.save(out / "params.npy", par)
    t0 = time.time()
    for i, p in enumerate(par):
        x = solver(jnp.asarray(H.initial(a.truth_n, p, fam)), float(p[-1]), geom)
        y = np.asarray(to_mesh(x, S))
        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"nonfinite truth case {i}")
        store[i] = y
        if i % 8 == 0:
            log(f"gen {a.cohort} {i + 1}/{len(par)} {time.time() - t0:.0f}s")
    store.flush()
    jdump(out / "gen.json", dict(family=a.family, cohort=a.cohort, start=a.start, count=len(par),
                                 truth_n=a.truth_n, truth_dt=a.truth_dt, store_n=S,
                                 seconds=time.time() - t0, device=str(jax.devices()[0])))
    log("DONE")


def build_ops(G, n, modes, check_modes=8, chunk=8, seed=11):
    """A, lam, Dd as shift_rom.build_operators_fast; the quadratic tensor is built on a
    zero-padded mesh so the product of bank fields (content up to |k| < n/2) is alias-free
    at the test wavenumbers. Validated against densely built tests on the padded mesh."""
    import ns3d_rom as R
    G = np.asarray(G, dtype=np.float64); r = G.shape[1]
    orth = float(np.max(np.abs(G.T @ G - np.eye(r))))
    if orth > 1e-10:
        raise RuntimeError(f"bank is not orthonormal: {orth}")
    ids, lam = SR.test_mode_ids(n, modes)
    kmax = max(max(abs(w) for w in row["wave"]) for row in ids)
    N = n + 2 * (kmax + 2); N += N % 2
    arrays = SR.ids_to_arrays(ids, n)
    A = np.empty((modes, r)); Dd = np.empty((3, modes, r))
    Gp = np.empty((3 * N ** 3, r))
    for start in range(0, r, chunk):
        block = jnp.asarray(G[:, start:start + chunk].T.reshape(-1, 3, n, n, n))
        base, der = SR.extract_tests(block, arrays, n)
        A[:, start:start + chunk] = np.asarray(base).T
        Dd[:, :, start:start + chunk] = np.asarray(der).transpose(0, 2, 1)
        Gp[:, start:start + chunk] = np.asarray(pad_or_truncate(block, N)).reshape(block.shape[0], -1).T
    scale = (n / N) ** 1.5
    T = R.build_tensor(Gp, N, ids) * scale
    rng = np.random.default_rng(seed)
    subset = np.sort(rng.choice(modes, size=min(check_modes, modes), replace=False))
    c = rng.normal(size=r)
    Phi = SR.dense_test_modes(n, [ids[i] for i in subset])
    rep = dict(modes=int(modes), rank=int(r), bank_orthonormality=orth, pad_mesh=int(N), kmax=int(kmax),
               A_condition=float(np.linalg.cond(A)))
    rep["A_relative"] = float(np.linalg.norm(Phi.T @ (G @ c) - A[subset] @ c) / np.linalg.norm(A[subset] @ c))
    geom = F.geometry(n); field = jnp.asarray(G @ c).reshape(1, 3, n, n, n)
    direct = np.asarray(SR.spectral_derivative(field, n, geom).reshape(3, -1)) @ Phi
    rep["derivative_relative"] = float(np.max(np.linalg.norm(direct - Dd[:, subset] @ c, axis=1)
                                              / np.maximum(np.linalg.norm(direct, axis=1), 1e-300)))
    del Phi
    PhiN = SR.dense_test_modes(N, [ids[i] for i in subset])
    gN = F.geometry(N); uh = F.fft(jnp.asarray(Gp @ c).reshape(3, N, N, N))
    adv = np.asarray(F.cross(F.ifft(uh), F.ifft(1j * F.cross(gN[0], uh)))).ravel()
    want = scale * (PhiN.T @ adv)
    got = np.asarray(R.contract(jnp.asarray(T), jnp.asarray(c)))[subset]
    rep["tensor_relative"] = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300))
    # aliased (on-mesh, 2/3-masked) projection for information only
    old = np.asarray(R.dense_projection(SR.dense_test_modes(n, [ids[i] for i in subset]), jnp.asarray(G), jnp.asarray(c), n))
    rep["tensor_vs_masked_mesh_product"] = float(np.linalg.norm(got - old) / max(np.linalg.norm(old), 1e-300))
    if max(rep["A_relative"], rep["derivative_relative"], rep["tensor_relative"]) > 1e-9:
        raise RuntimeError(f"reduced operators failed validation: {rep}")
    return dict(A=A, lam=np.asarray(lam), T=T, Dd=Dd), rep


# ----------------------------------------------------------------------- bank --

def cmd_bank(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arrs, par = load_truth(a.inputs.split(","))
    cases = sum(x.shape[0] for x in arrs)
    rep = dict(cases=cases, rank=a.rank, modes=a.modes, meshes={})
    for n in [int(x) for x in a.meshes.split(",")]:
        t0 = time.time()
        centre = make_centre(n)
        X = np.empty((cases * 6, 3 * n ** 3), dtype=np.float64)
        for c, fr in enumerate(iter_cases(arrs)):
            y = to_mesh(np.asarray(fr), n)
            for t in range(6):
                X[c * 6 + t] = np.asarray(centre(y[t])).ravel()
        log(f"mesh {n}: centred snapshots {X.shape} {time.time() - t0:.0f}s")
        basis, evals, orth = D.pod_basis(X, a.rank, a.gram_block)
        total = float(np.sum(X * X)) if n <= 96 else None
        del X; gc.collect()
        G, _ = D.orthonormalize_prefix(basis, a.rank); del basis; gc.collect()
        np.save(out / f"G_{n}.npy", G)
        ops, orep = build_ops(G, n, a.modes)
        np.savez(out / f"ops_{n}.npz", A=ops["A"], T=ops["T"], Dd=ops["Dd"], lam=ops["lam"])
        rep["meshes"][str(n)] = dict(pod_eigenvalues=[float(x) for x in evals], orth=orth,
                                     snapshot_energy=total, operator_checks=orep,
                                     seconds=time.time() - t0)
        jdump(out / "bank.json", rep)
        log(f"mesh {n}: bank + operators done {time.time() - t0:.0f}s checks {orep}")
        del G, ops; gc.collect()
    log("DONE")


# ------------------------------------------------------------------- headdata --

def make_project(n):
    centre_val = lambda f: SR.shift_field(f, -SR.grid_centroid(f) * n)

    @jax.jit
    def one(f, basis):
        v = centre_val(f).ravel()
        coef = basis.T @ v
        return coef, jnp.sum(v * v) - jnp.sum(coef * coef), jnp.sum(v * v)
    return one


def cmd_headdata(a):
    fam = H.FAMILIES[a.family]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    par = cohort_params("head", a.total, fam)[a.start:a.start + a.count]
    meshes = [int(x) for x in a.meshes.split(",")]
    banks = {n: jnp.asarray(np.load(Path(a.bank) / "output" / f"G_{n}.npy")) for n in meshes}
    proj = {n: make_project(n) for n in meshes}
    steps = D.nsteps_for(a.truth_dt, fam["horizon"])
    solver = F.make_solver(a.truth_dt, steps, steps // 20)
    geom = F.geometry(a.truth_n)
    res = {n: dict(a=[], o=[], e=[]) for n in meshes}
    t0 = time.time()
    for i, p in enumerate(par):
        x = solver(jnp.asarray(H.initial(a.truth_n, p, fam)), float(p[-1]), geom)
        for n in meshes:
            ca, co, ce = [], [], []
            for t in range(x.shape[0]):
                f = x[t] if n == a.truth_n else H.truncate_field(x[t], n)
                c1, o1, e1 = proj[n](f, banks[n])
                ca.append(np.asarray(c1)); co.append(float(o1)); ce.append(float(e1))
            res[n]["a"].append(np.stack(ca)); res[n]["o"].append(co); res[n]["e"].append(ce)
        if i % 8 == 0:
            log(f"headdata {i + 1}/{len(par)} {time.time() - t0:.0f}s")
    for n in meshes:
        np.savez(out / f"coeff_{n}.npz", a=np.stack(res[n]["a"]), outside=np.asarray(res[n]["o"]),
                 energy=np.asarray(res[n]["e"]), params=par, start=a.start)
    log("DONE")


# --------------------------------------------------------------------- romdev --

def load_rom(bank_dir, n):
    G = np.load(Path(bank_dir) / "output" / f"G_{n}.npy")
    ops = dict(np.load(Path(bank_dir) / "output" / f"ops_{n}.npz"))
    return G, ops


def rotate_ops(ops, V):
    return dict(A=ops["A"] @ V, Dd=np.einsum("dmr,rs->dms", ops["Dd"], V),
                T=np.einsum("mjk,ja,kb->mab", ops["T"], V, V, optimize=True), lam=ops["lam"])


class Arms:
    """Runners for span (linear POD-LSPG) and head arms on one mesh."""

    def __init__(self, n, horizon, G, ops, V, heads):
        self.n, self.T, self.R = n, horizon, G.shape[1]
        self.G = jnp.asarray(G)                      # unrotated bank (head arms)
        self.Gr = jnp.asarray(np.ascontiguousarray(G @ V))
        self.ops = {k: jnp.asarray(v) for k, v in ops.items()}
        self.rops = {k: jnp.asarray(v) for k, v in rotate_ops(ops, V).items()}
        self.heads = {}
        for k, hd in heads.items():
            p = {key: jnp.asarray(v) for key, v in hd["p"].items()}
            Z = jnp.asarray(hd["Z"]); Hc = HR.head_apply(p, Z)
            self.heads[int(k)] = (p, Z, Hc, jnp.sum(Hc * Hc, 1))
        self.cache = {}

    def span(self, width, M, steps, iters=3):
        key = ("span", width, steps, iters)
        if key not in self.cache:
            self.cache[key] = SR.make_frozen_run(self.T / steps, steps, steps // 5, self.n, width,
                                                 iters=iters, damping=HEAD["damping"],
                                                 extrapolate=True, diagnose=False)
        run = self.cache[key]; o = self.rops
        args = (self.Gr[:, :width], o["A"][:M, :width], o["T"][:M, :width, :width], o["lam"][:M],
                o["Dd"][:, :M, :width])
        args = tuple(jnp.asarray(x) for x in args)
        return lambda u0, nu: run(u0, nu, *args)

    def head(self, k, M, steps, iters=3):
        key = ("head", k, steps, iters)
        if key not in self.cache:
            self.cache[key] = HR.make_head_run(self.T / steps, steps, steps // 5, self.n, self.R, k, 0,
                                               iters=iters, damping=HEAD["damping"],
                                               ic_iters=HEAD["ic_iters"])
        run = self.cache[key]; o = self.ops; p, Z, Hc, Hn = self.heads[k]
        args = tuple(jnp.asarray(x) for x in (o["A"][:M], o["T"][:M], o["lam"][:M], o["Dd"][:, :M]))
        C = jnp.zeros((self.R, 0))
        return lambda u0, nu: run(u0, nu, self.G, *args, p, C, Hc, Hn, Z)


def run_cohort(call, truth, nus, keep=None):
    """truth: list of jnp/np (6,3,n,n,n). Returns errors (cases, 6) (inf rows if nonfinite)."""
    errs = np.empty((len(truth), 6)); kept = {}
    for c in range(len(truth)):
        tr = jnp.asarray(truth[c])
        out = call(tr[0], float(nus[c]))
        e = np.asarray(rel_err(out, tr))
        errs[c] = e if np.all(np.isfinite(e)) else np.inf
        if keep and c in keep:
            kept[c] = np.asarray(out)
    return errs, kept


def select(rows, cost_key, slack=1.05):
    ok = [r for r in rows if r["stats"]["finite"]]
    if not ok:
        return None
    best = min(r["stats"]["evolved_worst"] for r in ok)
    near = [r for r in ok if r["stats"]["evolved_worst"] <= slack * best]
    return min(near, key=cost_key)


def cmd_romdev(a):
    fam = H.FAMILIES[a.family]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = a.mesh; T = fam["horizon"]; R = a.rank
    G, ops = load_rom(a.bank, n)
    rep = dict(family=a.family, mesh=n, args=vars(a), device=str(jax.devices()[0]))
    # ---- head data
    shards = sorted((np.load(Path(d) / "output" / f"coeff_{n}.npz") for d in a.headdata.split(",")),
                    key=lambda z: int(z["start"]))
    coef = np.concatenate([z["a"] for z in shards]); outside = np.concatenate([z["outside"] for z in shards])
    energy = np.concatenate([z["energy"] for z in shards])
    ncase, nfr, _ = coef.shape
    case_of = np.repeat(np.arange(ncase), nfr)
    u0sq = energy[:, 0]
    coeffs = coef.reshape(-1, R)
    w_all = 1.0 / u0sq[case_of]; w_all = w_all / w_all.mean()
    is_val = (case_of % HEAD["val_every"]) == HEAD["val_every"] - 1
    tr, va = ~is_val, is_val
    out_rel = np.sqrt(np.maximum(outside, 0) / u0sq[:, None])
    rep["head_data"] = dict(cases=int(ncase), frames=int(nfr), states_train=int(tr.sum()),
                            states_val=int(va.sum()), bank_outside_worst=float(out_rel.max()),
                            bank_outside_median=float(np.median(out_rel.max(axis=1))))
    log(f"head data {rep['head_data']}")
    sigma = coeffs[tr].T @ coeffs[tr] / int(tr.sum())
    sv, V = np.linalg.eigh(sigma); order = np.argsort(sv)[::-1]; sv, V = sv[order], V[:, order]
    V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(R)])[None, :]
    np.savez(out / "rotation.npz", V=V, eigenvalues=sv)
    heads = {}
    rep["heads"] = {}
    steps_head = 400 if a.smoke else HEAD["steps"]
    for k in [int(x) for x in a.ks.split(",")]:
        t0 = time.time()
        p, codes, info = HR.init_head(jax.random.PRNGKey(HEAD["seed"] + k), coeffs[tr], w_all[tr], k,
                                      HEAD["width"])
        p, codes, curve = HR.train_head(p, codes, coeffs[tr], w_all[tr], steps_head, HEAD["lr"],
                                        HEAD["floor_lr"], HEAD["code_reg"], log=log,
                                        log_every=200 if a.smoke else 10000)
        Z = HR.best_codes(p, codes, coeffs[tr])
        Zj = jnp.asarray(Z); Hc = HR.head_apply(p, Zj)
        fit = {}
        for label, mask in (("train", tr), ("val", va)):
            err = HR.head_floor_errors(p, Zj, Hc, coeffs[mask][:, None, :], np.maximum(outside.reshape(-1)[mask], 0)[:, None],
                                       u0sq[case_of[mask]], jnp.zeros((R, 0)), HEAD["ic_iters"])[:, 0]
            fit[label] = dict(median=float(np.median(err)), worst=float(np.max(err)))
        heads[k] = dict(p=p, Z=Z)
        np.savez(out / f"head_k{k}.npz", Z=Z, **{f"p_{key}": np.asarray(v) for key, v in p.items()})
        rep["heads"][str(k)] = dict(fit=fit, seconds=time.time() - t0, final_loss=curve[-1][1], info=info)
        log(f"head k={k} {time.time() - t0:.0f}s fit {fit}")
    jdump(out / "romdev.json", rep)
    # ---- dev truth on this mesh
    arrs, par = load_truth(a.dev.split(","))
    ncases = a.cases or sum(x.shape[0] for x in arrs)
    dev = []
    for c, fr in enumerate(iter_cases(arrs)):
        if c >= ncases: break
        dev.append(np.asarray(to_mesh(np.asarray(fr), n)))
    nus = par[:ncases, -1]
    # ---- floors (oracle centring)
    proj = make_project(n); Gj = jnp.asarray(G)
    da = np.empty((ncases, 6, R)); do = np.empty((ncases, 6)); de = np.empty((ncases, 6))
    for c in range(ncases):
        for t in range(6):
            x, o, e = proj(jnp.asarray(dev[c][t]), Gj); da[c, t] = np.asarray(x); do[c, t] = float(o); de[c, t] = float(e)
    do = np.maximum(do, 0); du0 = de[:, 0]
    arot = da @ V
    floors = dict(span={}, head={})
    spans = [int(x) for x in a.spans.split(",")]
    for w in spans:
        floors["span"][str(w)] = stats(np.sqrt((np.sum(arot[..., w:] ** 2, axis=-1) + do) / du0[:, None]))
    for k, hd in heads.items():
        Zj = jnp.asarray(hd["Z"]); Hc = HR.head_apply(hd["p"], Zj)
        floors["head"][str(k)] = stats(HR.head_floor_errors(hd["p"], Zj, Hc, da, do, du0, jnp.zeros((R, 0)),
                                                             HEAD["ic_iters"]))
    rep["floors"] = floors
    log("floors span " + ", ".join(f"{k}:{v['evolved_worst']:.4f}" for k, v in floors["span"].items())
        + " | head " + ", ".join(f"{k}:{v['evolved_worst']:.4f}" for k, v in floors["head"].items()))
    jdump(out / "romdev.json", rep)
    # ---- validation rollouts
    arms = Arms(n, T, G, ops, V, heads)
    Ms = [int(x) for x in a.modes.split(",")]; steps_l = [int(x) for x in a.steps.split(",")]
    rows = []
    for M in Ms:
        for st in steps_l:
            for w in spans:
                if M < w + 16: continue
                e, _ = run_cohort(arms.span(w, M, st), dev, nus)
                rows.append(dict(arm="span", width=w, M=M, steps=st, iters=3, stats=stats(e), errors=e.tolist()))
                log(f"span{w} M{M} steps{st}: worst {rows[-1]['stats']['evolved_worst']:.5f} median {rows[-1]['stats']['evolved_median']:.5f}")
            for k in heads:
                e, _ = run_cohort(arms.head(k, M, st), dev, nus)
                rows.append(dict(arm="head", k=k, M=M, steps=st, iters=3, stats=stats(e), errors=e.tolist()))
                log(f"head k{k} M{M} steps{st}: worst {rows[-1]['stats']['evolved_worst']:.5f} median {rows[-1]['stats']['evolved_median']:.5f}")
            rep["rollouts"] = rows; jdump(out / "romdev.json", rep)
    # ---- selection (validation only)
    frozen = dict(family=a.family, mesh=n, spans={})
    hsel = select([r for r in rows if r["arm"] == "head"], lambda r: (r["steps"], r["M"], r["k"]))
    frozen["head"] = None if hsel is None else {k: hsel[k] for k in ("k", "M", "steps", "iters")} | dict(
        dev_worst=hsel["stats"]["evolved_worst"], dev_median=hsel["stats"]["evolved_median"])
    fsel = select([r for r in rows if r["arm"] == "head"], lambda r: (r["steps"], r["M"], r["k"]), slack=1.25)
    frozen["head_fast"] = None if fsel is None else {k: fsel[k] for k in ("k", "M", "steps", "iters")} | dict(
        dev_worst=fsel["stats"]["evolved_worst"], dev_median=fsel["stats"]["evolved_median"])
    for w in spans:
        s = select([r for r in rows if r["arm"] == "span" and r["width"] == w], lambda r: (r["steps"], r["M"]))
        frozen["spans"][str(w)] = None if s is None else {k: s[k] for k in ("width", "M", "steps", "iters")} | dict(
            dev_worst=s["stats"]["evolved_worst"], dev_median=s["stats"]["evolved_median"])
    jdump(out / "frozen.json", frozen)
    log(f"frozen {json.dumps(frozen)}")
    log("DONE")


# ------------------------------------------------------------------------ fom --

def pad_or_truncate(x, n):
    """Spectral transfer of (T,3,m,m,m) to mesh n (zero-pad or truncate)."""
    m = x.shape[-1]
    if m == n:
        return x
    if m > n:
        return H.truncate_field(x, n)
    hat = F.fft(x) * jnp.asarray(H.keep_mask(m))
    idx = jnp.asarray(H.sub_index(m, n))
    big = jnp.zeros(x.shape[:-3] + (n, n, n), dtype=hat.dtype)
    big = big.at[..., idx[:, None, None], idx[None, :, None], idx[None, None, :]].set(hat)
    return F.ifft(big)


@jax.jit
def _sq(x):
    return jnp.sum(jnp.abs(x) ** 2, axis=(1, 2, 3, 4))


def fom_ladder(arrs, par, ncases, horizon, fom_meshes, cn_steps, rk_steps, eval_meshes, timing_cases=4):
    """Free FOM ladder. Errors against the truth truncated to each evaluation mesh."""
    S = arrs[0].shape[-1]
    table = {}
    solvers = {}
    for c, fr in enumerate(iter_cases(arrs)):
        if c >= ncases: break
        hat = F.fft(jnp.asarray(np.asarray(fr)))
        nu = float(par[c, -1])
        tn = {n: (hat if n == S else H.truncate_hat(hat, n)) for n in eval_meshes}
        tn2 = {n: _sq(tn[n]) for n in eval_meshes}
        for m in fom_meshes:
            geom = F.geometry(m)
            u0 = F.ifft(hat[0] if m == S else H.truncate_hat(hat[0], m))
            tsub = {n: (hat if min(m, n) == S else H.truncate_hat(hat, min(m, n))) for n in eval_meshes}
            for scheme, steps_l in (("cnab2", cn_steps), ("rk4", rk_steps)):
                for s in steps_l:
                    key = f"{scheme}|{m}|{s}"
                    if key not in solvers:
                        solvers[key] = F.make_solver(horizon / s, s, s // 5, scheme=scheme)
                        jax.block_until_ready(solvers[key](u0, nu, geom))
                    row = table.setdefault(key, dict(scheme=scheme, n=m, steps=s, seconds=[],
                                                     errors={str(n): [] for n in eval_meshes}))
                    if c < timing_cases:
                        best = np.inf
                        for _ in range(3):
                            t0 = time.perf_counter()
                            pred = solvers[key](u0, nu, geom); pred.block_until_ready()
                            best = min(best, time.perf_counter() - t0)
                        row["seconds"].append(best)
                    else:
                        pred = solvers[key](u0, nu, geom)
                    ph = F.fft(pred) * jnp.asarray(H.keep_mask(m))
                    for n in eval_meshes:
                        q = min(m, n)
                        pa = ph if q == m else H.truncate_hat(ph, q)
                        e2 = _sq(pa - tsub[n]) + tn2[n] - _sq(tsub[n])
                        e = np.sqrt(np.maximum(np.asarray(e2), 0) / float(tn2[n][0]))
                        row["errors"][str(n)].append([float(x) if np.isfinite(x) else float("inf") for x in e])
        if c % 4 == 0:
            log(f"fom ladder case {c + 1}/{ncases}")
    rows = []
    for key, row in table.items():
        r = dict(scheme=row["scheme"], n=row["n"], steps=row["steps"],
                 seconds_median=float(np.median(row["seconds"])), seconds=row["seconds"], stats={})
        for n in eval_meshes:
            e = np.asarray(row["errors"][str(n)])
            r["stats"][str(n)] = stats(e) | dict(per_case_worst=[float(x) for x in e[:, 1:].max(axis=1)])
        rows.append(r)
    return rows


def matched(rows, mesh, target, stat="evolved_worst"):
    ok = [r for r in rows if r["stats"][str(mesh)]["finite"] and r["stats"][str(mesh)][stat] <= target]
    return min(ok, key=lambda r: r["seconds_median"]) if ok else None


def ints(s):
    return [int(x) for x in s.split(",")]


def cmd_fom(a):
    fam = H.FAMILIES[a.family]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arrs, par = load_truth(a.cohort.split(","))
    ncases = a.cases or sum(x.shape[0] for x in arrs)
    rows = fom_ladder(arrs, par, ncases, fam["horizon"], ints(a.fom_meshes), ints(a.cn_steps),
                      ints(a.rk_steps), ints(a.meshes))
    rep = dict(family=a.family, args=vars(a), device=str(jax.devices()[0]), rows=rows, cheapest={})
    for n in ints(a.meshes):
        for target in (0.002, 0.005, 0.01, 0.02, 0.03, 0.05):
            b = matched(rows, n, target)
            rep["cheapest"][f"{n}|{target}"] = None if b is None else dict(
                scheme=b["scheme"], n=b["n"], steps=b["steps"], ms=b["seconds_median"] * 1e3,
                worst=b["stats"][str(n)]["evolved_worst"])
            log(f"eval {n} target {target}: {rep['cheapest'][f'{n}|{target}']}")
    jdump(out / "fom.json", rep)
    log("DONE")


# ----------------------------------------------------------------------- test --

def time_block(callables, reps, burn, seed):
    rng = np.random.default_rng(seed); names = list(callables)
    for name in names:
        for _ in range(burn):
            jax.block_until_ready(callables[name]())
    samples = {name: [] for name in names}
    for _ in range(reps):
        for name in rng.permutation(names):
            t0 = time.perf_counter(); jax.block_until_ready(callables[name]())
            samples[name].append((time.perf_counter() - t0) * 1e3)
    return {k: dict(median_ms=float(np.median(v)), min_ms=float(np.min(v)), max_ms=float(np.max(v)),
                    q25_ms=float(np.percentile(v, 25)), q75_ms=float(np.percentile(v, 75)), reps=v)
            for k, v in samples.items()}


def cmd_test(a):
    """Frozen arms on a cohort (sealed test by default), full FOM ladder, timing, slices."""
    fam = H.FAMILIES[a.family]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T = fam["horizon"]
    arrs, par = load_truth(a.cohort.split(","))
    ncases = a.cases or sum(x.shape[0] for x in arrs)
    nus = par[:ncases, -1]
    meshes = ints(a.meshes); romdirs = dict(zip(meshes, a.romdev.split(",")))
    rep = dict(family=a.family, args=vars(a), device=str(jax.devices()[0]), meshes={})
    try:
        import subprocess
        rep["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                              "--format=csv,noheader"], text=True).strip()
    except Exception:
        rep["gpu"] = "unavailable"
    # FOM ladder on this cohort (all configurations scored; times on this GPU)
    rows = fom_ladder(arrs, par, ncases, T, ints(a.fom_meshes), ints(a.cn_steps), ints(a.rk_steps), meshes)
    rep["fom_rows"] = rows
    jdump(out / "test.json", rep)
    dev_rows = json.loads((Path(a.fomdev) / "output" / "fom.json").read_text())["rows"] if a.fomdev else None
    time_of = {(r["scheme"], r["n"], r["steps"]): r["seconds_median"] for r in rows}
    S = arrs[0].shape[-1]
    tcases = [c for c in ints(a.timing_cases) if c < ncases]
    for n in meshes:
        rd = Path(romdirs[n]) / "output"
        frozen = json.loads((rd / "frozen.json").read_text())
        G, ops = load_rom(a.bank, n)
        V = np.load(rd / "rotation.npz")["V"]
        heads = {}
        if frozen["head"]:
            ck = np.load(rd / f"head_k{frozen['head']['k']}.npz")
            heads[frozen["head"]["k"]] = dict(p={k[2:]: ck[k] for k in ck.files if k.startswith("p_")}, Z=ck["Z"])
        arms = Arms(n, T, G, ops, V, heads)
        truth = []
        for c, fr in enumerate(iter_cases(arrs)):
            if c >= ncases: break
            truth.append(np.asarray(to_mesh(np.asarray(fr), n)))
        calls = {}
        if frozen["head"]:
            h = frozen["head"]; calls["head"] = (arms.head(h["k"], h["M"], h["steps"], h["iters"]), h)
        for w, s in frozen["spans"].items():
            if s: calls[f"span{w}"] = (arms.span(s["width"], s["M"], s["steps"], s["iters"]), s)
        mrep = dict(frozen=frozen, arms={})
        fields = {}
        for name, (call, cfg) in calls.items():
            e, _ = run_cohort(call, truth, nus)
            st = stats(e)
            entry = dict(config=cfg, stats=st, errors=e.tolist())
            # accuracy-matched free FOM: chosen on validation (dev ladder, this GPU's times) and on this cohort
            for label, src in (("dev_matched", dev_rows), ("cohort_matched", rows)):
                if src is None: continue
                target = cfg["dev_worst"] if label == "dev_matched" else st["evolved_worst"]
                ok = [r for r in src if r["stats"][str(n)]["finite"] and r["stats"][str(n)]["evolved_worst"] <= target
                      and (r["scheme"], r["n"], r["steps"]) in time_of]
                if ok:
                    b = min(ok, key=lambda r: time_of[(r["scheme"], r["n"], r["steps"])])
                    here = next(r for r in rows if (r["scheme"], r["n"], r["steps"]) == (b["scheme"], b["n"], b["steps"]))
                    entry[label] = dict(scheme=b["scheme"], n=b["n"], steps=b["steps"], target=target,
                                        cohort_worst=here["stats"][str(n)]["evolved_worst"],
                                        cohort_median=here["stats"][str(n)]["evolved_median"],
                                        ladder_ms=time_of[(b["scheme"], b["n"], b["steps"])] * 1e3)
                else:
                    entry[label] = None
            mrep["arms"][name] = entry
            log(f"mesh {n} {name}: worst {st['evolved_worst']:.5f} median {st['evolved_median']:.5f} "
                f"dev-matched {entry.get('dev_matched')} cohort-matched {entry.get('cohort_matched')}")
        rep["meshes"][str(n)] = mrep; jdump(out / "test.json", rep)
        # ---- timing: ROM arms and their matched FOMs, interleaved, per timing case
        timing = {}
        fomcfgs = {}
        for name, entry in mrep["arms"].items():
            for label in ("dev_matched", "cohort_matched"):
                b = entry.get(label)
                if b: fomcfgs[(b["scheme"], b["n"], b["steps"])] = None
        for c in tcases:
            u0 = jnp.asarray(truth[c][0]); nu = float(nus[c])
            hat0 = F.fft(jnp.asarray(np.asarray(next(x for i, x in enumerate(iter_cases(arrs)) if i == c)[0])))
            cl = {name: (lambda call=call: call(u0, nu)) for name, (call, _) in calls.items()}
            for (scheme, m, s) in fomcfgs:
                solver = F.make_solver(T / s, s, s // 5, scheme=scheme); geom = F.geometry(m)
                u0m = F.ifft(hat0 if m == S else H.truncate_hat(hat0, m))
                cl[f"fom|{scheme}|{m}|{s}"] = (lambda solver=solver, u0m=u0m, geom=geom: solver(u0m, nu, geom))
                cl[f"fomT|{scheme}|{m}|{s}"] = (lambda solver=solver, u0m=u0m, geom=geom: pad_or_truncate(solver(u0m, nu, geom), n))
            timing[str(c)] = time_block(cl, a.reps, 3, 1000 + c)
        mrep["timing"] = timing
        # pooled medians and ratios
        pooled = {}
        for name in timing[str(tcases[0])]:
            v = np.concatenate([timing[str(c)][name]["reps"] for c in tcases])
            pooled[name] = dict(median_ms=float(np.median(v)), min_ms=float(v.min()), max_ms=float(v.max()),
                                q25_ms=float(np.percentile(v, 25)), q75_ms=float(np.percentile(v, 75)))
        mrep["timing_pooled"] = pooled
        for name, entry in mrep["arms"].items():
            for label in ("dev_matched", "cohort_matched"):
                b = entry.get(label)
                if b:
                    key = f"fom|{b['scheme']}|{b['n']}|{b['steps']}"
                    b["ms"] = pooled[key]["median_ms"]; b["ms_with_transfer"] = pooled["fomT" + key[3:]]["median_ms"]
                    b["speed_ratio"] = b["ms"] / pooled[name]["median_ms"]
                    per = [timing[str(c)][key]["median_ms"] / timing[str(c)][name]["median_ms"] for c in tcases]
                    b["speed_ratio_per_case"] = per
            entry["ms"] = pooled[name]
            log(f"mesh {n} {name}: {pooled[name]['median_ms']:.2f} ms; dev-matched {entry.get('dev_matched')}; cohort-matched {entry.get('cohort_matched')}")
        jdump(out / "test.json", rep)
        # ---- slices for figures: median and worst head case, final time
        ref_arm = "head" if "head" in calls else next(iter(calls))
        w = np.asarray(mrep["arms"][ref_arm]["errors"])[:, 1:].max(axis=1)
        pick = dict(worst=int(np.argmax(w)), median=int(np.argsort(w)[len(w) // 2]))
        sl = {}
        for label, c in pick.items():
            tr = truth[c]; cz = int(np.round(float(SR.grid_centroid(jnp.asarray(tr[-1]))[2]) * n)) % n
            sl[f"{label}_case"] = c; sl[f"{label}_z"] = cz
            sl[f"{label}_truth"] = tr[[0, -1]][:, :, :, :, cz].astype(np.float32)
            for name, (call, _) in calls.items():
                if name in ("head", "span64", "span16", "span8"):
                    o = np.asarray(call(jnp.asarray(tr[0]), float(nus[c])))
                    sl[f"{label}_{name}"] = o[[0, -1]][:, :, :, :, cz].astype(np.float32)
            b = mrep["arms"][ref_arm].get("cohort_matched")
            if b:
                m = b["n"]; solver = F.make_solver(T / b["steps"], b["steps"], b["steps"] // 5, scheme=b["scheme"])
                fr0 = np.asarray(next(x for i, x in enumerate(iter_cases(arrs)) if i == c)[0])
                h0 = F.fft(jnp.asarray(fr0))
                o = pad_or_truncate(solver(F.ifft(h0 if m == S else H.truncate_hat(h0, m)), float(nus[c]), F.geometry(m)), n)
                sl[f"{label}_fom"] = np.asarray(o)[[0, -1]][:, :, :, :, cz].astype(np.float32)
        np.savez_compressed(out / f"slices_{n}.npz", **sl)
        del arms, truth, G, ops; gc.collect()
    log("DONE")


# ------------------------------------------------- interp + compare (amendment A3) --

def spec_resample(u, n_to):
    """Fourier resampling of (..., n, n, n) real fields to mesh n_to (slices and
    concatenation only). Copies |k_i| <= min(n, n_to)/2 - 1; identical to truncate_field /
    keep_mask + zero padding."""
    n = u.shape[-1]
    if n == n_to:
        return u
    m = min(n, n_to) // 2
    spec = jnp.fft.fftn(u, axes=(-3, -2, -1), norm="forward")
    for axis in (-3, -2, -1):
        low = jax.lax.slice_in_dim(spec, 0, m, axis=axis)
        high = jax.lax.slice_in_dim(spec, spec.shape[axis] - m + 1, spec.shape[axis], axis=axis)
        shape = list(spec.shape); shape[axis] = n_to - (2 * m - 1)
        spec = jnp.concatenate((low, jnp.zeros(shape, dtype=spec.dtype), high), axis=axis)
    return jnp.fft.ifftn(spec, axes=(-3, -2, -1), norm="forward").real


def cmd_interp(a):
    """Interpolated-POD control: the source-mesh bank resampled to each other mesh + operators."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = a.source
    G0 = np.load(Path(a.bank) / "output" / f"G_{src}.npy"); r = G0.shape[1]
    rep = dict(source=src, rank=r, meshes={})
    for n in ints(a.meshes):
        if n == src: continue
        t0 = time.time()
        G = np.empty((3 * n ** 3, r))
        for j in range(r):
            G[:, j] = np.asarray(spec_resample(jnp.asarray(G0[:, j].reshape(3, src, src, src)), n)).ravel() * (src / n) ** 1.5
        orth = float(np.max(np.abs(G.T @ G - np.eye(r))))
        log(f"interp {src}->{n}: orthonormality {orth:.3e}")
        if orth > 1e-10:
            G, _ = np.linalg.qr(G)
        np.save(out / f"G_{n}.npy", G)
        ops, orep = build_ops(G, n, a.modes)
        np.savez(out / f"ops_{n}.npz", A=ops["A"], T=ops["T"], Dd=ops["Dd"], lam=ops["lam"])
        rep["meshes"][str(n)] = dict(orthonormality_before_fix=orth, operator_checks=orep, seconds=time.time() - t0)
        jdump(out / "interp.json", rep)
        log(f"interp {n}: done {time.time() - t0:.0f}s {orep}")
        del G, ops; gc.collect()
    log("DONE")


def time_block2(callables, reps, burn, seed):
    rng = np.random.default_rng(seed); names = list(callables)
    for name in names:
        for _ in range(burn):
            jax.block_until_ready(callables[name]())
    samples = {name: [] for name in names}; last = {}
    for _ in range(reps):
        for name in rng.permutation(names):
            t0 = time.perf_counter(); v = callables[name](); jax.block_until_ready(v)
            samples[name].append((time.perf_counter() - t0) * 1e3); last[name] = v
    return samples, last


def tstat(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(median_ms=float(np.median(v)), min_ms=float(v.min()), max_ms=float(v.max()),
                q25_ms=float(np.percentile(v, 25)), q75_ms=float(np.percentile(v, 75)), n=int(v.size))


def load_heads(rd, ks, scale=1.0):
    heads = {}
    for k in ks:
        ck = np.load(Path(rd) / f"head_k{k}.npz")
        p = {key[2:]: ck[key] for key in ck.files if key.startswith("p_")}
        if scale != 1.0:
            p = {key: (v * scale if key in ("W3", "b3", "Ws") else v) for key, v in p.items()}
        heads[k] = dict(p=p, Z=ck["Z"])
    return heads


def frozen_calls(arms, frozen, prefix=""):
    calls = {}
    for name in ("head", "head_fast"):
        h = frozen.get(name)
        if h: calls[prefix + name] = (arms.head(h["k"], h["M"], h["steps"], h["iters"]), h)
    for w, s in frozen["spans"].items():
        if s: calls[f"{prefix}span{w}"] = (arms.span(s["width"], s["M"], s["steps"], s["iters"]), s)
    return calls


def band(worst):
    return "useful" if worst <= 0.03 else ("marginal" if worst <= 0.05 else "not useful")


def cmd_compare(a):
    """Frozen ROM arms + controls against the free solver ladder on one cohort (A3)."""
    if jax.default_backend() != "gpu" and not a.smoke:
        raise RuntimeError(f"need a GPU backend, got {jax.default_backend()}")
    fam = H.FAMILIES[a.family]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T = fam["horizon"]; t_job = time.time()
    arrs, par = load_truth(a.cohort.split(","))
    ncases = a.cases or sum(x.shape[0] for x in arrs)
    nus = par[:ncases, -1]; S = arrs[0].shape[-1]
    meshes = ints(a.meshes); romdirs = dict(zip(meshes, a.romdev.split(",")))
    rep = dict(schema="ns3d-hard-compare-v1", family=a.family, args=vars(a), device=str(jax.devices()[0]),
               job_id=os.environ.get("SLURM_JOB_ID"), cases=int(ncases), meshes={}, failures=[])
    cfile = HERE.parents[1] / "COMMIT.txt"
    rep["source_commit"] = cfile.read_text().strip() if cfile.exists() else None
    try:
        import subprocess
        rep["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                              "--format=csv,noheader"], text=True).strip()
    except Exception:
        rep["gpu"] = "unavailable"

    def dump():
        rep["elapsed_seconds"] = time.time() - t_job; jdump(out / "compare.json", rep)
    dump()
    # ---- solver ladder for THIS cohort (errors; its timings only shortlist candidates)
    if a.ladder:
        table = {}
        for d in a.ladder.split(","):
            fj = json.loads((Path(d) / "output" / "fom.json").read_text())
            if fj["args"]["cohort"] != a.cohort or (fj["args"]["cases"] or sum(x.shape[0] for x in arrs)) != ncases:
                raise RuntimeError(f"ladder {d} is for another cohort")
            for r in fj["rows"]:
                table[(r["scheme"], r["n"], r["steps"])] = r
        rows = list(table.values()); rep["ladder_source"] = a.ladder
    else:
        rows = fom_ladder(arrs, par, ncases, T, ints(a.fom_meshes), ints(a.cn_steps), ints(a.rk_steps), meshes)
        jdump(out / "ladder.json", dict(rows=rows)); rep["ladder_source"] = "in-job"
    rep["ladder_configurations"] = len(rows); dump()
    log(f"ladder: {len(rows)} configurations ({rep['ladder_source']})")
    rowof = {(r["scheme"], r["n"], r["steps"]): r for r in rows}
    devc = json.loads((Path(a.devcompare) / "output" / "compare.json").read_text()) if a.devcompare else None
    tcases = [c for c in ints(a.timing_cases) if c < ncases]
    solvers = {}; e2es = {}; reps = int(a.reps); gate = float(a.gate)
    for n in meshes:
        rd = Path(romdirs[n]) / "output"
        frozen = json.loads((rd / "frozen.json").read_text())
        G, ops = load_rom(a.bank, n)
        V = np.load(rd / "rotation.npz")["V"]
        ks = sorted({frozen[x]["k"] for x in ("head", "head_fast") if frozen.get(x)})
        arms = Arms(n, T, G, ops, V, load_heads(rd, ks))
        calls = frozen_calls(arms, frozen)
        del G, ops
        mrep = dict(frozen=frozen, arms={}); rep["meshes"][str(n)] = mrep
        if a.interp and n != a.interp_source:
            src = a.interp_source; rd0 = Path(a.interp_romdev or romdirs[src]) / "output"
            fz0 = json.loads((rd0 / "frozen.json").read_text())
            Gi, oi = load_rom(a.interp, n)
            k0 = sorted({fz0[x]["k"] for x in ("head", "head_fast") if fz0.get(x)})
            iarms = Arms(n, T, Gi, oi, np.load(rd0 / "rotation.npz")["V"], load_heads(rd0, k0, (n / src) ** 1.5))
            calls.update(frozen_calls(iarms, fz0, prefix="interp_"))
            mrep["interp_frozen"] = fz0
            del Gi, oi
        gc.collect()
        truth = []
        for c, fr in enumerate(iter_cases(arrs)):
            if c >= ncases: break
            truth.append(np.asarray(to_mesh(np.asarray(fr), n)))
        # ---- accuracy pass
        for name, (call, cfg) in calls.items():
            e, _ = run_cohort(call, truth, nus)
            st = stats(e)
            mrep["arms"][name] = dict(config=cfg, stats=st, errors=e.tolist(),
                                      band=band(st["evolved_worst"]) if st["finite"] else "nonfinite")
            log(f"mesh {n} {name}: worst {st['evolved_worst']:.5f} median {st['evolved_median']:.5f}")
            dump()
        # ---- comparator shortlist
        sn = str(n)
        stable = [r for r in rows if r["stats"][sn]["finite"] and r["stats"][sn]["evolved_worst"] <= 1.0]
        timed = {}
        for name, entry in mrep["arms"].items():
            if not entry["stats"]["finite"]: continue
            worst = entry["stats"]["evolved_worst"]
            elig = sorted((r for r in stable if r["stats"][sn]["evolved_worst"] <= worst), key=lambda r: r["seconds_median"])
            same = [r for r in elig if r["n"] == n]
            short = [(r["scheme"], r["n"], r["steps"]) for r in elig[:a.shortlist] + same[:2]]
            entry["eligible"] = len(elig); entry["eligible_same_mesh"] = len(same)
            entry["shortlist"] = [list(x) for x in dict.fromkeys(short)]
            if devc:
                d = devc["meshes"][sn]["arms"].get(name, {}).get("any_mesh_solve")
                if d:
                    key = (d["scheme"], d["n"], d["steps"])
                    entry["dev_chosen_key"] = list(key) if key in rowof else None
                    if key in rowof:
                        short.append(key)
                        if key in [(r["scheme"], r["n"], r["steps"]) for r in elig] and list(key) not in entry["shortlist"]:
                            entry["shortlist"].append(list(key))
            for key in short: timed[key] = None
        log(f"mesh {n}: {len(timed)} solver configurations in the timing block")
        # ---- timing
        attempts = []
        for attempt in range(3):
            tim = dict(cases={}, gate={}); ok = True; agree_rom = 0.0; agree_fom = 0.0
            for c in tcases:
                u0 = jnp.asarray(truth[c][0]); nu = float(nus[c])
                uS = jnp.asarray(np.asarray(next(x for i, x in enumerate(iter_cases(arrs)) if i == c)[0]))
                cl = {name: (lambda call=call: call(u0, nu)) for name, (call, _) in calls.items()
                      if mrep["arms"][name]["stats"]["finite"]}
                for key in timed:
                    scheme, m, s = key
                    if key not in solvers:
                        solvers[key] = F.make_solver(T / s, s, s // 5, scheme=scheme)
                    solver = solvers[key]; geom = F.geometry(m)
                    if (key, n) not in e2es:
                        e2es[(key, n)] = jax.jit(lambda u, v, g, solver=solver, m=m, n=n:
                                                 spec_resample(solver(spec_resample(u, m), v, g), n))
                    e2e = e2es[(key, n)]
                    u0m = jax.block_until_ready(spec_resample(uS, m))
                    tag = f"{scheme}|{m}|{s}"
                    cl["E2E|" + tag] = (lambda e2e=e2e, geom=geom: e2e(uS, nu, geom))
                    cl["SOLVE|" + tag] = (lambda solver=solver, u0m=u0m, geom=geom: solver(u0m, nu, geom))
                sent = {k: cl[k] for k in ("head", "span64") if k in cl} or {k: cl[k] for k in list(cl)[:1]}
                sk = [k for k in cl if k.startswith("SOLVE|")]
                if sk:  # cheapest and dearest shortlisted solver configurations as solver sentinels (A4)
                    order = sorted(sk, key=lambda k: rowof[(k.split("|")[1], int(k.split("|")[2]), int(k.split("|")[3]))]["seconds_median"])
                    sent[order[0]] = cl[order[0]]; sent[order[-1]] = cl[order[-1]]
                warm = jnp.ones((1024, 1024)) * 1e-3; t0 = time.perf_counter()
                while time.perf_counter() - t0 < 2.0:
                    warm = warm @ warm * 1e-3 + 1e-3
                jax.block_until_ready(warm)
                s0, _ = time_block2(sent, reps, 2, 1)
                samples, last = time_block2(cl, reps, 2, 1000 + c)
                s1, _ = time_block2(sent, reps, 1, 5)
                g = {}
                for k in sent:
                    b, i, e = np.median(s0[k]), np.median(samples[k]), np.median(s1[k])
                    g[k] = dict(solo_before_ms=float(b), in_block_ms=float(i), solo_end_ms=float(e),
                                ratio=float(i / b), end_ratio=float(e / b))
                    ok = ok and (1 / gate <= i / b <= a.gate_block) and (1 / gate <= e / b <= gate)
                tim["gate"][str(c)] = g
                tim["cases"][str(c)] = {k: v for k, v in samples.items()}
                tr = jnp.asarray(truth[c])
                for name, v in last.items():
                    if name.startswith("SOLVE|"): continue
                    e = np.asarray(rel_err(v, tr))
                    if name.startswith("E2E|"):
                        scheme, m, s = name.split("|")[1:]
                        ref = rowof[(scheme, int(m), int(s))]["stats"][sn]["per_case_worst"][c]
                        agree_fom = max(agree_fom, abs(float(e[1:].max()) - ref))
                    else:
                        agree_rom = max(agree_rom, float(np.max(np.abs(e - np.asarray(mrep["arms"][name]["errors"][c])))))
                del cl, last; gc.collect()
            tim["gate_passed"] = bool(ok); tim["agreement_rom"] = agree_rom; tim["agreement_solver"] = agree_fom
            attempts.append(tim)
            log(f"mesh {n} timing attempt {attempt + 1}: gate {ok} agreement rom {agree_rom:.2e} solver {agree_fom:.2e}")
            if ok: break
        tim = attempts[-1]
        mrep["timing_attempts"] = len(attempts)
        mrep["timing_failed_attempts"] = [dict(gate=t["gate"]) for t in attempts[:-1]]
        mrep["timing"] = dict(gate=tim["gate"], gate_passed=tim["gate_passed"], agreement_rom=tim["agreement_rom"],
                              agreement_solver=tim["agreement_solver"], cases=tcases, reps=tim["cases"])
        if not tim["gate_passed"]: rep["failures"].append(f"mesh {n}: timing gate failed in 3 attempts")
        if not tim["agreement_rom"] <= 1e-9: rep["failures"].append(f"mesh {n}: timed ROM outputs differ {tim['agreement_rom']:.2e}")
        if not tim["agreement_solver"] <= 1e-7: rep["failures"].append(f"mesh {n}: timed solver outputs differ {tim['agreement_solver']:.2e}")
        pooled = {k: tstat(np.concatenate([tim["cases"][str(c)][k] for c in tcases])) for k in tim["cases"][str(tcases[0])]}
        percase = {k: [float(np.median(tim["cases"][str(c)][k])) for c in tcases] for k in pooled}
        mrep["timing_pooled"] = pooled

        def comp(name, key, kind):
            tag = f"{kind}|{key[0]}|{key[1]}|{key[2]}"; r = rowof[tuple(key)]
            return dict(scheme=key[0], n=key[1], steps=key[2], kind=kind,
                        cohort_worst=r["stats"][sn]["evolved_worst"], cohort_median=r["stats"][sn]["evolved_median"],
                        ms=pooled[tag]["median_ms"], ms_min=pooled[tag]["min_ms"], ms_max=pooled[tag]["max_ms"],
                        ratio=pooled[tag]["median_ms"] / pooled[name]["median_ms"],
                        ratio_per_case=[x / y for x, y in zip(percase[tag], percase[name])])
        for name, entry in mrep["arms"].items():
            if name not in pooled: continue
            entry["ms"] = pooled[name]
            keys = [tuple(x) for x in entry["shortlist"]]
            same = [k for k in keys if k[1] == n]
            for label, cand, kind in (("any_mesh_solve", keys, "SOLVE"), ("any_mesh_e2e", keys, "E2E"),
                                      ("same_mesh_solve", same, "SOLVE"), ("same_mesh_e2e", same, "E2E")):
                if cand:
                    best = min(cand, key=lambda k: pooled[f"{kind}|{k[0]}|{k[1]}|{k[2]}"]["median_ms"])
                    entry[label] = comp(name, best, kind)
                else:
                    entry[label] = None
            if entry.get("dev_chosen_key"):
                entry["dev_chosen_solve"] = comp(name, entry["dev_chosen_key"], "SOLVE")
                entry["dev_chosen_e2e"] = comp(name, entry["dev_chosen_key"], "E2E")
            x = entry["any_mesh_solve"]; y = entry["same_mesh_solve"]
            log(f"mesh {n} {name}: {entry['stats']['evolved_worst']:.5f} {pooled[name]['median_ms']:.2f} ms | any-mesh solve "
                + (f"{x['scheme']} m{x['n']} s{x['steps']} {x['ms']:.2f} ms ratio {x['ratio']:.2f}" if x else "none")
                + " | same-mesh " + (f"{y['scheme']} s{y['steps']} {y['ms']:.2f} ms ratio {y['ratio']:.2f}" if y else "none"))
        dump()
        del arms, truth, calls; gc.collect()
    rep["status"] = "final" if not rep["failures"] else "failed-gates"
    dump()
    log(f"DONE status {rep['status']} failures {rep['failures']}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--family", required=True); p.add_argument("--out", required=True)
        p.add_argument("--truth-n", type=int, default=192); p.add_argument("--truth-dt", type=float, default=2.5e-4)
        p.add_argument("--store-n", type=int, default=128); p.add_argument("--smoke", action="store_true")
        p.add_argument("--meshes", default="64,96,128"); p.add_argument("--cases", type=int, default=0)
        p.add_argument("--fom-meshes", default="32,40,48,56,64,72,80,88,96,104,112,128")
        p.add_argument("--cn-steps", default="5,10,15,20,30,40,50,60,70,80,100,120,160,240")
        p.add_argument("--rk-steps", default="5,10,15,20,25,30,40,60,80")
    p = sub.add_parser("gen"); common(p); p.add_argument("--cohort", required=True)
    p.add_argument("--start", type=int, default=0); p.add_argument("--count", type=int, required=True)
    p.add_argument("--total", type=int, default=None)
    p = sub.add_parser("bank"); common(p); p.add_argument("--inputs", required=True)
    p.add_argument("--rank", type=int, default=64); p.add_argument("--modes", type=int, default=2336)
    p.add_argument("--gram-block", type=int, default=96)
    p = sub.add_parser("headdata"); common(p); p.add_argument("--bank", required=True)
    p.add_argument("--start", type=int, default=0); p.add_argument("--count", type=int, required=True)
    p.add_argument("--total", type=int, default=512)
    p = sub.add_parser("romdev"); common(p); p.add_argument("--bank", required=True)
    p.add_argument("--headdata", required=True); p.add_argument("--dev", required=True)
    p.add_argument("--mesh", type=int, required=True); p.add_argument("--rank", type=int, default=64)
    p.add_argument("--ks", default="8,16,32"); p.add_argument("--spans", default="64,48,32,16,8")
    p.add_argument("--modes", default="292,584,1168,2336"); p.add_argument("--steps", default="10,20,40")
    p = sub.add_parser("fom"); common(p); p.add_argument("--cohort", required=True)
    p = sub.add_parser("interp"); common(p); p.add_argument("--bank", required=True)
    p.add_argument("--source", type=int, default=64); p.add_argument("--modes", type=int, default=2336)
    p = sub.add_parser("compare"); common(p); p.add_argument("--cohort", required=True)
    p.add_argument("--bank", required=True); p.add_argument("--romdev", required=True)
    p.add_argument("--interp", default=None); p.add_argument("--ladder", default=None)
    p.add_argument("--interp-source", type=int, default=64); p.add_argument("--interp-romdev", default=None)
    p.add_argument("--devcompare", default=None); p.add_argument("--reps", type=int, default=9)
    p.add_argument("--timing-cases", default="0,1,2"); p.add_argument("--shortlist", type=int, default=4)
    p.add_argument("--gate", type=float, default=1.10); p.add_argument("--gate-block", type=float, default=1.25)
    p = sub.add_parser("test"); common(p); p.add_argument("--cohort", required=True)
    p.add_argument("--bank", required=True); p.add_argument("--romdev", required=True)
    p.add_argument("--fomdev", default=None); p.add_argument("--reps", type=int, default=9)
    p.add_argument("--timing-cases", default="0,1,2")
    a = ap.parse_args()
    log(f"backend={jax.default_backend()} devices={jax.devices()} args={vars(a)}")
    dict(gen=cmd_gen, bank=cmd_bank, headdata=cmd_headdata, romdev=cmd_romdev, fom=cmd_fom, test=cmd_test,
         interp=cmd_interp, compare=cmd_compare)[a.cmd](a)


if __name__ == "__main__":
    main()
