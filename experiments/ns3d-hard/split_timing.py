"""Split timing of the frozen head arms (lane ns3d-flat-timing, PROTOCOL.md T2).

Stages:
  asm    reduced operators at meshes above the bank meshes (192, 256): the 128^3 bank of each
         source bank Fourier-resampled to mesh n (exactly orthonormal after (128/n)^{3/2}),
         operators assembled with pipeline.build_ops (the lane's padded weak-form builder and
         its dense-test validation), compared with the 128^3 operators of the same bank.
  split  per mesh, every arm's query timed as (i) ENC: centroid + centring shift + projection
         G^T u0 (grid sized), (ii) FIT: nearest candidate + code fit (reduced), (iii) SOLVE:
         parameter-dependent operator preparation + all time steps in reduced coordinates,
         (iv) DEC: head evaluation + G a + shift for the six output frames, frame by frame as
         in the source query (DECB: the same with one batched matmul, reported separately),
         and FULL: the source lane's monolithic query (what Table G1 times). Interleaved with
         the solver configurations matched to each arm in the source fresh_s0b comparison.
"""
from __future__ import annotations
import argparse, functools, gc, json, os, time
from pathlib import Path
import numpy as np
import pipeline as P
import rank as RK
from pipeline import jax, jnp, H, F, SR, HR, log, HEAD, jdump, ints, tstat, spec_resample, time_block2, rel_err

HERE = Path(__file__).resolve().parent
SRC = 128                      # mesh of the banks that are resampled upwards (--src only for smoke tests)


def resample_bank(G0, src, n):
    r = G0.shape[1]
    G = np.empty((3 * n ** 3, r))
    f = jax.jit(lambda u: spec_resample(u, n))
    for j in range(r):
        G[:, j] = np.asarray(f(jnp.asarray(G0[:, j].reshape(3, src, src, src)))).ravel() * (src / n) ** 1.5
    return G


# ------------------------------------------------------------------------ asm --

def cmd_asm(a):
    global SRC
    SRC = a.src
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P.build_tensor_chunked = functools.partial(P.build_tensor_chunked, chunk=a.tchunk)   # memory only
    rep = dict(schema="ns3d-flat-asm-v1", args=vars(a), device=str(jax.devices()[0]),
               job_id=os.environ.get("SLURM_JOB_ID"), banks={})
    for spec in a.banks.split(","):
        tag, bdir = spec.split("=")
        G0 = np.load(Path(bdir) / "output" / f"G_{SRC}.npy")
        ref = dict(np.load(Path(bdir) / "output" / f"ops_{SRC}.npz"))
        rep["banks"][tag] = dict(source=bdir, rank=int(G0.shape[1]), meshes={})
        for n in ints(a.meshes):
            t0 = time.time()
            G = resample_bank(G0, SRC, n)
            orth = float(np.max(np.abs(G.T @ G - np.eye(G.shape[1]))))
            log(f"{tag} {SRC}->{n}: orthonormality {orth:.2e} ({time.time() - t0:.0f}s)")
            if orth > 1e-10:
                raise RuntimeError(f"resampled bank not orthonormal: {orth}")
            ops, orep = P.build_ops(G, n, a.modes)
            inv = {}
            for key in ("A", "T", "Dd", "lam"):
                x, y = np.asarray(ops[key]), np.asarray(ref[key])
                inv[key] = dict(relative_difference=float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300)),
                                norm_ratio=float(np.linalg.norm(x) / max(np.linalg.norm(y), 1e-300)))
            # the resampled bank's coefficients are (n/SRC)^{3/2} times the SRC ones, so the reduced system is
            # invariant iff A, Dd, lam agree and T agrees after multiplication by (n/SRC)^{3/2}
            Ts = np.asarray(ops["T"]) * (n / SRC) ** 1.5
            inv["T_scaled"] = dict(relative_difference=float(np.linalg.norm(Ts - ref["T"]) / np.linalg.norm(ref["T"])))
            del Ts
            np.savez(out / f"ops_{tag}_{n}.npz", A=ops["A"], T=ops["T"], Dd=ops["Dd"], lam=ops["lam"])
            rep["banks"][tag]["meshes"][str(n)] = dict(orthonormality=orth, operator_checks=orep,
                                                       vs_128_operators=inv, seconds=time.time() - t0)
            jdump(out / "asm.json", rep)
            log(f"{tag} mesh {n}: {time.time() - t0:.0f}s checks {orep} vs128 {inv}")
            del G, ops; gc.collect()
        del G0, ref; gc.collect()
    log("DONE")


# ---------------------------------------------------------------------- split --

def make_split_head(dt, nsteps, out_every, n, R, k, iters, damping, ic_iters):
    """head_rom.make_head_run (q = 0, frame free, extrapolation on) cut into stages."""
    prepare, coeff, _, step_solve = HR.make_head_step(dt, n, R, k, 0, iters, damping, True)
    code_fit = HR.make_code_fit(k, ic_iters)
    C = jnp.zeros((R, 0))

    @jax.jit
    def enc(u0, basis):
        c0 = SR.grid_centroid(u0)
        return basis.T @ SR.shift_field(u0, -c0 * n).ravel(), c0

    @jax.jit
    def fit(a0, p, Hcand, Hnorm, Zcand):
        idx = jnp.argmin(Hnorm - 2 * (Hcand @ a0))
        z0, y0 = code_fit(p, Zcand[idx], a0, C)
        return jnp.concatenate((z0, y0, jnp.zeros(3)))

    @jax.jit
    def solve(w0, c0, nu, A, T, lam, Dd, p):
        ops = prepare(nu, A, T, lam, Dd)

        def step(carry, _):
            w, previous, c = carry
            guess = w + (w - previous)
            new, rn = step_solve(guess, coeff(p, w, C), ops, p, C)
            return (new, w, (c + new[k:]) % 1.0), None

        def block(carry, _):
            carry, _ = jax.lax.scan(step, carry, None, length=out_every)
            return carry, (carry[0], carry[2])
        _, (W, Cs) = jax.lax.scan(block, (w0, w0, c0), None, length=nsteps // out_every)
        return W, Cs

    @jax.jit
    def dec(w0, c0, W, Cs, basis, p):           # frame by frame, as the source query
        Wa = jnp.concatenate((w0[None], W)); Ca = jnp.concatenate((c0[None], Cs))
        return jax.lax.map(lambda wc: SR.shift_field((basis @ coeff(p, wc[0], C)).reshape(3, n, n, n), wc[1] * n),
                           (Wa, Ca))

    @jax.jit
    def decb(w0, c0, W, Cs, basis, p):          # one batched matmul for the six frames
        Wa = jnp.concatenate((w0[None], W)); Ca = jnp.concatenate((c0[None], Cs))
        a = jax.vmap(lambda w: coeff(p, w, C))(Wa)
        fields = (basis @ a.T).T.reshape(-1, 3, n, n, n)
        return jax.vmap(lambda f, c: SR.shift_field(f, c * n))(fields, Ca)
    return enc, fit, solve, dec, decb


def to_device(G, chunk=16):
    """Host bank -> device without the 2x staging transient of jnp.asarray (in-place chunk writes)."""
    out = jnp.zeros(G.shape, dtype=G.dtype)
    put = jax.jit(lambda o, c, s: jax.lax.dynamic_update_slice(o, c, (0, s)), donate_argnums=0)
    for s in range(0, G.shape[1], chunk):
        out = put(out, jnp.asarray(np.ascontiguousarray(G[:, s:s + chunk])), s)
    return jax.block_until_ready(out)


def bank_of(rank):
    return "r64" if rank == 64 else ("r128" if rank <= 128 else "r192")


def cmd_split(a):
    global SRC
    SRC = a.src
    if jax.default_backend() != "gpu" and not a.smoke:
        raise RuntimeError("need a GPU backend")
    fam = H.FAMILIES[a.family]; T = fam["horizon"]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t_job = time.time()
    arrs, par = P.load_truth(a.cohort.split(","))
    ncases = sum(x.shape[0] for x in arrs); nus = par[:ncases, -1]
    tcases = ints(a.timing_cases); arm_names = a.arms.split(",")
    banks = dict(x.split("=") for x in a.banks.split(","))
    heads = {t: dict(zip(ints(a.head_meshes), d.split(":"))) for t, d in (x.split("=") for x in a.heads.split(","))}
    refc = json.loads(Path(a.refcompare).read_text())
    rep = dict(schema="ns3d-flat-split-v1", family=a.family, args=vars(a), device=str(jax.devices()[0]),
               job_id=os.environ.get("SLURM_JOB_ID"), meshes={}, failures=[])
    cfile = HERE.parents[1] / "COMMIT.txt"
    rep["source_commit"] = cfile.read_text().strip() if cfile.exists() else None
    try:
        import subprocess
        rep["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                                              "--format=csv,noheader"], text=True).strip()
        rep["host"] = os.uname().nodename
    except Exception:
        rep["gpu"] = "unavailable"
    try:
        limit = jax.devices()[0].memory_stats()["bytes_limit"]
    except Exception:
        limit = 0
    rep["device_bytes_limit"] = int(limit)

    def dump():
        rep["elapsed_seconds"] = time.time() - t_job; jdump(out / "split.json", rep)
    dump()
    asm_dir = Path(a.asm) / "output" if a.asm else out
    if a.asm_meshes:                       # assemble (and validate) in this job first
        import types
        orig = P.build_tensor_chunked
        try:
            cmd_asm(types.SimpleNamespace(out=str(out), banks=a.banks, meshes=a.asm_meshes, modes=a.modes,
                                          tchunk=a.tchunk, src=a.src))
        except Exception as exc:          # e.g. out of memory for the largest bank: recorded, the rest goes on
            log(f"asm failed: {exc!r}"); rep["failures"].append(f"asm: {exc!r}"[:500])
        P.build_tensor_chunked = orig; SRC = a.src; gc.collect()
        rep["asm"] = json.loads((out / "asm.json").read_text()) if (out / "asm.json").exists() else {}
        dump()
    elif a.asm and (asm_dir / "asm.json").exists():   # assembly done by an earlier job of this lane
        rep["asm"] = json.loads((asm_dir / "asm.json").read_text()); rep["asm_from"] = str(asm_dir); dump()
    scaled = ints(a.scaled_meshes) if a.scaled_meshes else []
    if scaled or a.asm_meshes or rep.get("asm"):                             # operators by the exact scaling, only if assembly confirmed it
        worst = max([v["vs_128_operators"][q]["relative_difference"] for b in rep.get("asm", {}).get("banks", {}).values()
                     for v in b["meshes"].values() for q in ("A", "Dd", "lam", "T_scaled")] or [1.0])
        rep["scaled_operator_check_worst"] = worst
        if worst > 1e-10:
            log(f"invariance not confirmed ({worst}): no scaled operators"); scaled = []
            rep["failures"].append(f"scaled operators disabled: assembly check {worst}")
        confirmed = bool(scaled) or worst <= 1e-10
    else:
        confirmed = False
    case_frames = {c: np.asarray(next(x for i, x in enumerate(P.iter_cases(arrs)) if i == c)) for c in tcases}
    solvers, e2es = {}, {}
    for n in ints(a.meshes):
        sn = str(n); native = n <= SRC; hm = n if native else SRC
        use_decb = n <= a.decb_max          # the batched (6 x 3n^3) gemm fails XLA autotuning at 256^3 (memory)
        fz = json.loads((Path(a.frozen) / f"frozen_{hm}.json").read_text())["arms"]
        cfgs = {name: fz[name] for name in arm_names}
        # ---- matched solver configurations from the source comparison (per arm)
        rmesh = refc["meshes"][str(hm)]["arms"]
        fom = {}
        for name in arm_names:
            x, y = rmesh[name]["any_mesh_solve"], rmesh[name]["same_mesh_solve"]
            fom[name] = dict(any=(x["scheme"], x["n"], x["steps"]),
                             same=((y["scheme"], n, y["steps"]) if y else None),
                             ref_any_ms=x["ms"], ref_same_ms=(y["ms"] if y else None),
                             ref_rom_ms=rmesh[name]["ms"]["median_ms"], ref_any_ratio=x["ratio"])
        mrep = dict(native_bank=native, settings_from_mesh=hm, frozen={k: cfgs[k] for k in arm_names},
                    fom_refs={k: dict(any=list(v["any"]), same=(list(v["same"]) if v["same"] else None),
                                      **{q: v[q] for q in ("ref_any_ms", "ref_same_ms", "ref_rom_ms", "ref_any_ratio")})
                              for k, v in fom.items()},
                    groups=[], arms={}, solvers={})
        rep["meshes"][sn] = mrep
        # ---- memory plan: one interleaved block if the bases fit, else one block per bank
        need = {}
        for name in arm_names:
            b = bank_of(cfgs[name]["rank"]); need.setdefault(b, set()).add(cfgs[name]["rank"])
        Rb = {"r64": 64, "r128": 128, "r192": 192}
        cols = {b: Rb[b] + sum(r for r in rs if r < Rb[b]) for b, rs in need.items()}
        total = sum(cols.values()) * 3 * n ** 3 * 8
        groups = [sorted(need)] if (limit == 0 or total <= a.budget * limit) else [[b] for b in sorted(need)]
        mrep["groups"] = groups; mrep["basis_bytes"] = int(total)
        log(f"mesh {n}: basis bytes {total / 1e9:.1f} GB, limit {limit / 1e9:.1f} GB, groups {groups}")
        up = jax.jit(lambda u: spec_resample(u, n))
        truth = {c: (np.asarray(P.to_mesh(case_frames[c], n)) if native else np.asarray(up(jnp.asarray(case_frames[c]))))
                 for c in tcases}
        for grp in groups:
            # ---- build arms of this group
            arms_by_bank = {}
            for b in grp:
                if native:
                    G, ops = P.load_rom(banks[b], n)
                else:
                    G = resample_bank(np.load(Path(banks[b]) / "output" / f"G_{SRC}.npy"), SRC, n)
                    if n in scaled or (confirmed and not (asm_dir / f"ops_{b}_{n}.npz").exists()):
                        ops = dict(np.load(Path(banks[b]) / "output" / f"ops_{SRC}.npz"))
                        ops["T"] = ops["T"] * (SRC / n) ** 1.5
                        mrep.setdefault("operators", {})[b] = f"ops_{SRC} of the bank, T x ({SRC}/{n})^1.5"
                    else:
                        ops = dict(np.load(asm_dir / f"ops_{b}_{n}.npz"))
                        mrep.setdefault("operators", {})[b] = f"assembled at {n} (asm)"
                    orth = float(np.max(np.abs(G.T @ G - np.eye(G.shape[1]))))
                    mrep.setdefault("resampled_orthonormality", {})[b] = orth
                Gd = to_device(G); del G; gc.collect()
                arms_by_bank[b] = RK.RankArms(n, T, Gd, ops); del Gd, ops; gc.collect()
            scale = 1.0 if native else (n / SRC) ** 1.5
            calls, parts = {}, {}
            for name in arm_names:
                cfg = cfgs[name]; b = bank_of(cfg["rank"])
                if b not in grp: continue
                arms = arms_by_bank[b]; r, k = cfg["rank"], cfg["k"]
                tag = (r, k, cfg["width"])
                if tag not in arms.heads:
                    d = Path(heads[b][hm]) / "output"
                    f = d / (f"head_k{k}.npz" if b == "r64" else RK.head_file(*tag))
                    hd = RK.load_head(f)
                    if scale != 1.0:
                        hd["p"] = {q: (v * scale if q in ("W3", "b3", "Ws") else v) for q, v in hd["p"].items()}
                    arms.add_head(tag, r, hd)
                calls[name] = arms.head(tag, cfg["M"], cfg["steps"], cfg["iters"])
                _, _, p, Z, Hc, Hn = arms.heads[tag]
                Gr = arms.prefix(r); A_, T_, lam_, Dd_ = arms.sliced(r, cfg["M"])
                enc, fit, solve, dec, decb = make_split_head(T / cfg["steps"], cfg["steps"], cfg["steps"] // 5, n, r, k,
                                                            cfg["iters"], HEAD["damping"], HEAD["ic_iters"])
                parts[name] = dict(enc=enc, fit=fit, solve=solve, dec=dec, decb=decb, G=Gr, p=p, Z=Z, Hc=Hc, Hn=Hn,
                                   ops=(A_, T_, lam_, Dd_))
            grp_arms = [x for x in arm_names if x in calls]
            fkeys = {}
            for name in grp_arms:
                fkeys[fom[name]["any"]] = True
                if fom[name]["same"]: fkeys[fom[name]["same"]] = True
            fkeys = list(fkeys)
            # ---- accuracy sanity on the timing cases (FULL query) and split = FULL agreement
            for name in grp_arms:
                e_full, agree = [], 0.0
                for c in tcases:
                    u0 = jnp.asarray(truth[c][0]); nu = float(nus[c]); pr = parts[name]
                    full = calls[name](u0, nu)
                    a0, c0 = pr["enc"](u0, pr["G"]); w0 = pr["fit"](a0, pr["p"], pr["Hc"], pr["Hn"], pr["Z"])
                    W, Cs = pr["solve"](w0, c0, nu, *pr["ops"], pr["p"])
                    sp = pr["dec"](w0, c0, W, Cs, pr["G"], pr["p"])
                    sb = pr["decb"](w0, c0, W, Cs, pr["G"], pr["p"]) if use_decb else sp
                    den = float(jnp.linalg.norm(full))
                    agree = max(agree, float(jnp.linalg.norm(sp - full)) / den, float(jnp.linalg.norm(sb - full)) / den)
                    e_full.append(np.asarray(rel_err(full, jnp.asarray(truth[c]))).tolist())
                    del full, sp, sb
                e = np.asarray(e_full)
                ref_e = None
                src_arm = refc["meshes"][str(hm)]["arms"][name]
                ref_e = np.asarray(src_arm["errors"])[tcases]
                mrep["arms"][name] = dict(config=cfgs[name], timing_case_errors=e.tolist(),
                                          timing_case_evolved_worst=float(e[:, 1:].max()),
                                          source_errors_same_cases_mesh=hm, source_evolved_worst_same_cases=float(ref_e[:, 1:].max()),
                                          max_abs_diff_vs_source=float(np.max(np.abs(e - ref_e))),
                                          split_vs_full_relative=agree)
                log(f"mesh {n} {name}: timing-case worst {e[:, 1:].max():.5f} (source mesh {hm}: {ref_e[:, 1:].max():.5f}, "
                    f"max diff {np.max(np.abs(e - ref_e)):.2e}); split vs full {agree:.2e}")
                if agree > 1e-9: rep["failures"].append(f"mesh {n} {name}: split differs from full {agree:.2e}")
                dump()
            for key in fkeys:
                scheme, m, s = key
                if (scheme, s) not in solvers:
                    solvers[(scheme, s)] = F.make_solver(T / s, s, s // 5, scheme=scheme)
            # ---- timing
            attempts = []
            for attempt in range(3):
                tim = dict(cases={}, gate={}); ok = True
                for c in tcases:
                    u0 = jnp.asarray(truth[c][0]); nu = float(nus[c]); uS = jnp.asarray(case_frames[c][0])
                    cl = {}
                    for name in grp_arms:
                        pr = parts[name]; call = calls[name]
                        a0, c0 = jax.block_until_ready(pr["enc"](u0, pr["G"]))
                        w0 = jax.block_until_ready(pr["fit"](a0, pr["p"], pr["Hc"], pr["Hn"], pr["Z"]))
                        W, Cs = jax.block_until_ready(pr["solve"](w0, c0, nu, *pr["ops"], pr["p"]))
                        cl[f"FULL|{name}"] = (lambda call=call: call(u0, nu))
                        cl[f"ENC|{name}"] = (lambda pr=pr: pr["enc"](u0, pr["G"]))
                        cl[f"FIT|{name}"] = (lambda pr=pr, a0=a0: pr["fit"](a0, pr["p"], pr["Hc"], pr["Hn"], pr["Z"]))
                        cl[f"SOLVE|{name}"] = (lambda pr=pr, w0=w0, c0=c0: pr["solve"](w0, c0, nu, *pr["ops"], pr["p"]))
                        cl[f"DEC|{name}"] = (lambda pr=pr, w0=w0, c0=c0, W=W, Cs=Cs: pr["dec"](w0, c0, W, Cs, pr["G"], pr["p"]))
                        if use_decb: cl[f"DECB|{name}"] = (lambda pr=pr, w0=w0, c0=c0, W=W, Cs=Cs: pr["decb"](w0, c0, W, Cs, pr["G"], pr["p"]))
                    for key in fkeys:
                        scheme, m, s = key; solver = solvers[(scheme, s)]; geom = F.geometry(m)
                        u0m = jax.block_until_ready(spec_resample(uS, m)); tg = f"{scheme}|{m}|{s}"
                        cl["FOMSOLVE|" + tg] = (lambda solver=solver, u0m=u0m, geom=geom: solver(u0m, nu, geom))
                        if m != n:
                            if (key, n) not in e2es:
                                e2es[(key, n)] = jax.jit(lambda u, v, g, solver=solver, m=m, n=n:
                                                         spec_resample(solver(spec_resample(u, m), v, g), n))
                            e2e = e2es[(key, n)]
                            cl["FOME2E|" + tg] = (lambda e2e=e2e, geom=geom: e2e(uS, nu, geom))
                    sent = {f"FULL|{x}": cl[f"FULL|{x}"] for x in grp_arms if x in a.sentinels.split(",")}
                    if not sent:
                        sent = {f"FULL|{grp_arms[0]}": cl[f"FULL|{grp_arms[0]}"]}
                    sk = sorted([x for x in cl if x.startswith("FOMSOLVE|")])
                    if sk:
                        t1 = {}
                        for x in sk:      # one warm call each to order them (not part of any statistic)
                            jax.block_until_ready(cl[x]()); t0 = time.perf_counter(); jax.block_until_ready(cl[x]())
                            t1[x] = time.perf_counter() - t0
                        order = sorted(sk, key=t1.get); sent[order[0]] = cl[order[0]]; sent[order[-1]] = cl[order[-1]]
                    warm = jnp.ones((1024, 1024)) * 1e-3; t0 = time.perf_counter()
                    while time.perf_counter() - t0 < 2.0:
                        warm = warm @ warm * 1e-3 + 1e-3
                    jax.block_until_ready(warm)
                    s0, _ = time_block2(sent, a.reps, 2, 1)
                    samples, last = time_block2(cl, a.reps, 2, 1000 + c)
                    s1, _ = time_block2(sent, a.reps, 1, 5)
                    g = {}
                    for x in sent:
                        b_, i_, e_ = np.median(s0[x]), np.median(samples[x]), np.median(s1[x])
                        g[x] = dict(solo_before_ms=float(b_), in_block_ms=float(i_), solo_end_ms=float(e_),
                                    ratio=float(i_ / b_), end_ratio=float(e_ / b_))
                        ok = ok and (1 / a.gate <= i_ / b_ <= a.gate_block) and (1 / a.gate <= e_ / b_ <= a.gate)
                    tim["gate"][str(c)] = g; tim["cases"][str(c)] = samples
                    # FOM sanity: E2E/SOLVE output error on this case (truth at n; SOLVE output transferred)
                    tr = jnp.asarray(truth[c])
                    for x, v in last.items():
                        if x.startswith("FOME2E|") or (x.startswith("FOMSOLVE|") and int(x.split("|")[2]) == n):
                            mrep["solvers"].setdefault(x.split("|", 1)[1], {}).setdefault("timing_case_evolved_worst", {})[str(c)] = \
                                float(np.asarray(rel_err(v, tr))[1:].max())
                    del cl, last; gc.collect()
                tim["gate_passed"] = bool(ok); attempts.append(tim)
                log(f"mesh {n} group {grp} timing attempt {attempt + 1}: gate {ok}")
                if ok: break
            tim = attempts[-1]
            pooled = {x: tstat(np.concatenate([tim["cases"][str(c)][x] for c in tcases])) for x in tim["cases"][str(tcases[0])]}
            percase = {x: [float(np.median(tim["cases"][str(c)][x])) for c in tcases] for x in pooled}
            mrep.setdefault("timing", {})["+".join(grp)] = dict(
                attempts=len(attempts), gate_passed=tim["gate_passed"], gate=tim["gate"],
                failed_attempt_gates=[t["gate"] for t in attempts[:-1]], pooled=pooled, per_case_median=percase,
                samples=tim["cases"])
            if not tim["gate_passed"]: rep["failures"].append(f"mesh {n} group {grp}: timing gate failed in 3 attempts")
            for name in grp_arms:
                st = {s: pooled[f"{s}|{name}"]["median_ms"] for s in ("FULL", "ENC", "FIT", "SOLVE", "DEC", "DECB") if f"{s}|{name}" in pooled}
                st["SUM"] = st["ENC"] + st["FIT"] + st["SOLVE"] + st["DEC"]
                fa = fom[name]["any"]; tga = f"{fa[0]}|{fa[1]}|{fa[2]}"
                st["FOM_any_solve"] = pooled["FOMSOLVE|" + tga]["median_ms"]
                st["FOM_any_e2e"] = pooled["FOME2E|" + tga]["median_ms"] if ("FOME2E|" + tga) in pooled else st["FOM_any_solve"]
                if fom[name]["same"]:
                    fs = fom[name]["same"]; st["FOM_same_solve"] = pooled[f"FOMSOLVE|{fs[0]}|{fs[1]}|{fs[2]}"]["median_ms"]
                mrep["arms"][name]["ms"] = st
                mrep["arms"][name]["ms_detail"] = {s: pooled[f"{s}|{name}"] for s in ("FULL", "ENC", "FIT", "SOLVE", "DEC", "DECB") if f"{s}|{name}" in pooled}
                log(f"mesh {n} {name}: " + " ".join(f"{s} {v:.2f}" for s, v in st.items()))
            dump()
            del calls, parts, arms_by_bank; gc.collect()
        del truth; gc.collect()
    rep["status"] = "final" if not rep["failures"] else "failed-gates-or-checks"
    dump()
    log(f"DONE status {rep['status']} failures {rep['failures']}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("asm"); p.add_argument("--out", required=True); p.add_argument("--banks", required=True)
    p.add_argument("--meshes", default="192,256"); p.add_argument("--modes", type=int, default=2336)
    p.add_argument("--tchunk", type=int, default=8); p.add_argument("--src", type=int, default=128)
    p = sub.add_parser("split"); p.add_argument("--out", required=True); p.add_argument("--family", default="B")
    p.add_argument("--smoke", action="store_true"); p.add_argument("--src", type=int, default=128)
    p.add_argument("--cohort", required=True); p.add_argument("--meshes", default="64,96,128,192")
    p.add_argument("--arms", default="r64_head,r96_head,r128_head,r128_head_fast,r192_head,r192_head_fast")
    p.add_argument("--banks", required=True, help="r64=dir,r128=dir,r192=dir")
    p.add_argument("--heads", required=True, help="r64=d64:d96:d128,r128=...,r192=...")
    p.add_argument("--head-meshes", default="64,96,128")
    p.add_argument("--asm", default=None); p.add_argument("--frozen", default=str(HERE / "frozen_rank"))
    p.add_argument("--refcompare", required=True); p.add_argument("--reps", type=int, default=9)
    p.add_argument("--timing-cases", default="0,16,32,48,64,80,96,112")
    p.add_argument("--gate", type=float, default=1.10); p.add_argument("--gate-block", type=float, default=1.25)
    p.add_argument("--sentinels", default="r64_head,r128_head"); p.add_argument("--budget", type=float, default=0.6)
    p.add_argument("--decb-max", type=int, default=100000)
    p.add_argument("--asm-meshes", default=""); p.add_argument("--scaled-meshes", default="")
    p.add_argument("--modes", type=int, default=2336); p.add_argument("--tchunk", type=int, default=8)
    a = ap.parse_args()
    log(f"backend={jax.default_backend()} devices={jax.devices()} args={vars(a)}")
    dict(asm=cmd_asm, split=cmd_split)[a.cmd](a)


if __name__ == "__main__":
    main()
