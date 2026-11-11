"""Rank lane (PROTOCOL.md amendment A5): larger banks, replicates, fresh cohort.

Stages (gen, bank, headdata are pipeline.py unchanged, with a rank-128 bank / seed overrides):

  rdev     heads on bank prefixes (rank r <= bank rank), dev floors, dev rollouts.
           Without --frozen: the A5 grid, for selection by freeze_rank.py.
           With --frozen: only the frozen arms (replicates; no selection).
  ladder   free solver ladder on a slice of a cohort, in blocks of (meshes : cnab2 : rk4)
  cmp      frozen arms of all ranks against the merged ladder on one cohort (A3.6 / A4 rules,
           pipeline.cmd_compare with this lane's arm builder)
"""
from __future__ import annotations
import argparse, gc, json, os, time
from pathlib import Path
import numpy as np
import pipeline as P
from pipeline import jax, jnp, H, F, D, SR, HR, log, HEAD, jdump, stats, ints

HERE = Path(__file__).resolve().parent


def head_file(r, k, width):
    return f"head_r{r}_k{k}_w{width}.npz"


def load_head(path):
    ck = np.load(path)
    return dict(p={key[2:]: ck[key] for key in ck.files if key.startswith("p_")}, Z=ck["Z"])


class RankArms:
    """Span and head runners on prefixes of one bank (rank-r bank = first r columns)."""

    def __init__(self, n, horizon, G, ops):
        self.n, self.T = n, horizon
        self.G = jnp.asarray(G)
        self.ops = {k: jnp.asarray(v) for k, v in ops.items()}
        self.pref, self.heads, self.cache = {}, {}, {}

    def prefix(self, r):
        if r not in self.pref:
            self.pref[r] = self.G if r == self.G.shape[1] else jnp.asarray(self.G[:, :r])
        return self.pref[r]

    def sliced(self, r, M):
        o = self.ops
        return tuple(jnp.asarray(x) for x in (o["A"][:M, :r], o["T"][:M, :r, :r], o["lam"][:M], o["Dd"][:, :M, :r]))

    def add_head(self, tag, r, hd):
        p = {key: jnp.asarray(v) for key, v in hd["p"].items()}
        Z = jnp.asarray(hd["Z"]); Hc = HR.head_apply(p, Z)
        assert Hc.shape[1] == r
        self.heads[tag] = (r, int(Z.shape[1]), p, Z, Hc, jnp.sum(Hc * Hc, 1))

    def span(self, r, M, steps, iters=3):
        key = ("span", r, steps, iters)
        if key not in self.cache:
            self.cache[key] = SR.make_frozen_run(self.T / steps, steps, steps // 5, self.n, r, iters=iters,
                                                 damping=HEAD["damping"], extrapolate=True, diagnose=False)
        run = self.cache[key]; G = self.prefix(r); args = self.sliced(r, M)
        return lambda u0, nu: run(u0, nu, G, *args)

    def head(self, tag, M, steps, iters=3):
        r, k, p, Z, Hc, Hn = self.heads[tag]
        key = ("head", r, k, steps, iters)
        if key not in self.cache:
            self.cache[key] = HR.make_head_run(self.T / steps, steps, steps // 5, self.n, r, k, 0, iters=iters,
                                               damping=HEAD["damping"], ic_iters=HEAD["ic_iters"])
        run = self.cache[key]; G = self.prefix(r); args = self.sliced(r, M)
        C = jnp.zeros((r, 0))
        return lambda u0, nu: run(u0, nu, G, *args, p, C, Hc, Hn, Z)


# ----------------------------------------------------------------------- rdev --

def cmd_rdev(a):
    fam = H.FAMILIES[a.family]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = a.mesh; T = fam["horizon"]
    G, ops = P.load_rom(a.bank, n); Rmax = G.shape[1]
    rep = dict(schema="ns3d-hard-rank-rdev-v1", family=a.family, mesh=n, args=vars(a), device=str(jax.devices()[0]),
               job_id=os.environ.get("SLURM_JOB_ID"), bank_rank=int(Rmax), ranks={})
    frozen = None
    if a.frozen:
        frozen = json.loads((Path(a.frozen) / f"frozen_{n}.json").read_text())["arms"]
        need = sorted({(c["rank"], c["k"], c["width"]) for c in frozen.values() if c["kind"] == "head"})
        ranks = sorted({c["rank"] for c in frozen.values()})
    else:
        ranks = ints(a.ranks)
        need = [(r, k, a.width) for r in ranks for k in ints(a.ks)]
    if max(ranks) > Rmax:
        raise RuntimeError(f"bank rank {Rmax} < requested {ranks}")
    # ---- head data (coefficients against the full bank; a rank-r prefix is a slice)
    shards = sorted((np.load(Path(d) / "output" / f"coeff_{n}.npz") for d in a.headdata.split(",")),
                    key=lambda z: int(z["start"]))
    coef = np.concatenate([z["a"] for z in shards]); outside = np.concatenate([z["outside"] for z in shards])
    energy = np.concatenate([z["energy"] for z in shards])
    ncase, nfr, rr = coef.shape
    assert rr == Rmax
    case_of = np.repeat(np.arange(ncase), nfr)
    u0sq = energy[:, 0]
    w_all = 1.0 / u0sq[case_of]; w_all = w_all / w_all.mean()
    is_val = (case_of % HEAD["val_every"]) == HEAD["val_every"] - 1
    tr, va = ~is_val, is_val
    steps_head = 400 if a.smoke else a.head_steps
    heads = {}
    for r in ranks:
        o_r = np.maximum(outside + np.sum(coef[..., r:] ** 2, axis=-1), 0)
        out_rel = np.sqrt(o_r / u0sq[:, None])
        rep["ranks"][str(r)] = dict(head_data=dict(
            cases=int(ncase), frames=int(nfr), states_train=int(tr.sum()), states_val=int(va.sum()),
            bank_outside_worst=float(out_rel.max()), bank_outside_median=float(np.median(out_rel.max(axis=1)))),
            heads={}, floors={})
        log(f"rank {r} head data {rep['ranks'][str(r)]['head_data']}")
    for (r, k, width) in need:
        tag = f"k{k}_w{width}"; t0 = time.time()
        coeffs = np.ascontiguousarray(coef[..., :r].reshape(-1, r))
        o_r = np.maximum(outside + np.sum(coef[..., r:] ** 2, axis=-1), 0).reshape(-1)
        src = Path(a.heads_from) / "output" / head_file(r, k, width) if a.heads_from else None
        if src is not None and src.exists():
            hd = load_head(src); p = {key: jnp.asarray(v) for key, v in hd["p"].items()}; Z = hd["Z"]
            info = dict(loaded=str(src)); final = None
        else:
            seed = a.head_seed + k + (0 if width == HEAD["width"] else width)
            p, codes, info = HR.init_head(jax.random.PRNGKey(seed), coeffs[tr], w_all[tr], k, width)
            p, codes, curve = HR.train_head(p, codes, coeffs[tr], w_all[tr], steps_head, HEAD["lr"],
                                            HEAD["floor_lr"], HEAD["code_reg"], log=log,
                                            log_every=200 if a.smoke else 20000)
            Z = HR.best_codes(p, codes, coeffs[tr]); final = curve[-1][1]; info["seed"] = int(seed)
        Zj = jnp.asarray(Z); Hc = HR.head_apply(p, Zj)
        fit = {}
        for label, mask in (("train", tr), ("val", va)):
            err = HR.head_floor_errors(p, Zj, Hc, coeffs[mask][:, None, :], o_r[mask][:, None],
                                       u0sq[case_of[mask]], jnp.zeros((r, 0)), HEAD["ic_iters"])[:, 0]
            bank_only = np.sqrt(o_r[mask] / u0sq[case_of[mask]])
            fit[label] = dict(median=float(np.median(err)), worst=float(np.max(err)),
                              bank_only_median=float(np.median(bank_only)), bank_only_worst=float(np.max(bank_only)))
        heads[(r, k, width)] = dict(p={key: np.asarray(v) for key, v in p.items()}, Z=np.asarray(Z))
        np.savez(out / head_file(r, k, width), Z=np.asarray(Z), **{f"p_{key}": np.asarray(v) for key, v in p.items()})
        rep["ranks"][str(r)]["heads"][tag] = dict(k=k, width=width, steps=steps_head, fit=fit,
                                                  seconds=time.time() - t0, final_loss=final, info=info)
        log(f"rank {r} head {tag} {time.time() - t0:.0f}s fit {fit}")
        jdump(out / "romdev.json", rep)
    # ---- dev truth and floors (oracle centring)
    arrs, par = P.load_truth(a.dev.split(","))
    ncases = a.cases or sum(x.shape[0] for x in arrs)
    dev = []
    for c, fr in enumerate(P.iter_cases(arrs)):
        if c >= ncases: break
        dev.append(np.asarray(P.to_mesh(np.asarray(fr), n)))
    nus = par[:ncases, -1]
    proj = P.make_project(n); Gj = jnp.asarray(G)
    da = np.empty((ncases, 6, Rmax)); do = np.empty((ncases, 6)); de = np.empty((ncases, 6))
    for c in range(ncases):
        for t in range(6):
            x, o, e = proj(jnp.asarray(dev[c][t]), Gj); da[c, t] = np.asarray(x); do[c, t] = float(o); de[c, t] = float(e)
    del Gj
    du0 = de[:, 0]
    for r in ranks:
        o_r = np.maximum(do + np.sum(da[..., r:] ** 2, axis=-1), 0)
        fl = rep["ranks"][str(r)]["floors"]
        fl["span"] = stats(np.sqrt(o_r / du0[:, None]))
        for (r2, k, width), hd in heads.items():
            if r2 != r: continue
            Zj = jnp.asarray(hd["Z"]); pj = {key: jnp.asarray(v) for key, v in hd["p"].items()}
            Hc = HR.head_apply(pj, Zj)
            fl[f"k{k}_w{width}"] = stats(HR.head_floor_errors(pj, Zj, Hc, np.ascontiguousarray(da[..., :r]), o_r, du0,
                                                              jnp.zeros((r, 0)), HEAD["ic_iters"]))
        log(f"rank {r} dev floors " + ", ".join(f"{k}: worst {v['evolved_worst']:.5f} median {v['evolved_median']:.5f}"
                                                for k, v in fl.items()))
    jdump(out / "romdev.json", rep)
    # ---- dev rollouts
    arms = RankArms(n, T, G, ops)
    for (r, k, width), hd in heads.items():
        arms.add_head((r, k, width), r, hd)
    rows = []

    def roll(name, cfg):
        if cfg["kind"] == "head":
            call = arms.head((cfg["rank"], cfg["k"], cfg["width"]), cfg["M"], cfg["steps"], cfg["iters"])
        else:
            call = arms.span(cfg["rank"], cfg["M"], cfg["steps"], cfg["iters"])
        e, _ = P.run_cohort(call, dev, nus)
        rows.append(dict(cfg, name=name, stats=stats(e), errors=e.tolist()))
        log(f"{name} {json.dumps({k: v for k, v in cfg.items() if k not in ('dev_worst', 'dev_median')})}: worst "
            f"{rows[-1]['stats']['evolved_worst']:.5f} median {rows[-1]['stats']['evolved_median']:.5f}")
        rep["rollouts"] = rows; jdump(out / "romdev.json", rep)

    if frozen is not None:
        for name, cfg in frozen.items():
            roll(name, {k: cfg[k] for k in ("kind", "rank", "k", "width", "M", "steps", "iters") if k in cfg})
    else:
        grid = [tuple(int(y) for y in x.split(":")) for x in a.grid.split(",")]
        for r in ranks:
            for M in ints(a.modes):
                if M < r + 16: continue
                for st, it in grid:
                    if not a.no_span:
                        roll(f"r{r}_span", dict(kind="span", rank=r, M=M, steps=st, iters=it))
                    for (r2, k, width) in heads:
                        if r2 == r:
                            roll(f"r{r}_head", dict(kind="head", rank=r, k=k, width=width, M=M, steps=st, iters=it))
    log("DONE")


# --------------------------------------------------------------------- ladder --

def cmd_ladder(a):
    fam = H.FAMILIES[a.family]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arrs, par = P.load_truth(a.cohort.split(","))
    total = sum(x.shape[0] for x in arrs); count = a.count or total - a.start
    sl, off = [], 0
    for x in arrs:
        lo, hi = max(a.start - off, 0), min(a.start + count - off, x.shape[0])
        if hi > lo: sl.append(x[lo:hi])
        off += x.shape[0]
    parS = par[a.start:a.start + count]
    assert sum(x.shape[0] for x in sl) == count == len(parS)
    rep = dict(schema="ns3d-hard-rank-ladder-v1", family=a.family, args=vars(a), device=str(jax.devices()[0]),
               job_id=os.environ.get("SLURM_JOB_ID"), start=a.start, count=count, total=total, rows=[])
    seen = set()
    for block in a.blocks.split(";"):
        ms, cn, rk = block.split(":")
        t0 = time.time()
        rows = P.fom_ladder(sl, parS, count, fam["horizon"], ints(ms), ints(cn) if cn else [], ints(rk) if rk else [],
                            ints(a.meshes))
        for r in rows:
            key = (r["scheme"], r["n"], r["steps"])
            if key in seen: continue
            seen.add(key); rep["rows"].append(r)
        log(f"block {block}: {len(rows)} configurations {time.time() - t0:.0f}s")
        jdump(out / "fom.json", rep)
    log("DONE")


def merge_ladders(dirs, cohort, ncases, meshes):
    """Rows over the whole cohort from whole-cohort files (parent fom stage) and shards."""
    pieces = {}
    for d in dirs:
        fj = json.loads((Path(d) / "output" / "fom.json").read_text())
        if fj["args"]["cohort"] != cohort:
            raise RuntimeError(f"ladder {d} is for another cohort")
        start = int(fj.get("start", 0))
        for r in fj["rows"]:
            pieces.setdefault((r["scheme"], r["n"], r["steps"]), {})[start] = r     # later directories win
    rows, dropped = [], []
    for key, ps in pieces.items():
        pos, ok = 0, True
        for start in sorted(ps):
            if start != pos: ok = False; break
            pos += len(ps[start]["stats"][str(meshes[0])]["per_case_worst"])
        if not ok or pos != ncases:
            dropped.append(list(key)); continue
        row = dict(scheme=key[0], n=key[1], steps=key[2], seconds_median=ps[0]["seconds_median"], stats={})
        for n in meshes:
            w = np.concatenate([np.asarray(ps[s]["stats"][str(n)]["per_case_worst"], dtype=np.float64) for s in sorted(ps)])
            fin = bool(np.all(np.isfinite(w)))
            row["stats"][str(n)] = dict(finite=fin, evolved_worst=float(w.max()) if fin else float("inf"),
                                        evolved_median=float(np.median(w)) if fin else float("inf"),
                                        per_case_worst=[float(x) for x in w], cases=int(len(w)))
        rows.append(row)
    return rows, dropped


# ------------------------------------------------------------------------ cmp --

def cmd_cmp(a):
    meshes = ints(a.meshes)
    arrs, _ = P.load_truth(a.cohort.split(","))
    ncases = a.cases or sum(x.shape[0] for x in arrs)
    del arrs
    rows, dropped = merge_ladders(a.ladders.split(","), a.cohort, ncases, meshes)
    log(f"merged ladder: {len(rows)} configurations, dropped (incomplete coverage) {dropped}")
    if dropped and not a.smoke:
        raise RuntimeError("ladder shards do not cover the cohort")
    headdirs = dict(zip(meshes, a.heads.split(",")))
    pdirs = dict(zip(meshes, a.parent_romdev.split(","))) if a.parent_bank else {}

    def build(n, T):
        fz = json.loads((Path(a.frozen) / f"frozen_{n}.json").read_text())
        G, ops = P.load_rom(a.bank, n)
        arms = RankArms(n, T, G, ops); del G, ops
        calls = {}
        for name, cfg in fz["arms"].items():
            r = cfg["rank"]
            if r == 64 and a.parent_bank: continue
            if cfg["kind"] == "head":
                tag = (r, cfg["k"], cfg["width"])
                if tag not in arms.heads:
                    arms.add_head(tag, r, load_head(Path(headdirs[n]) / "output" / head_file(*tag)))
                calls[name] = (arms.head(tag, cfg["M"], cfg["steps"], cfg["iters"]), cfg)
            else:
                calls[name] = (arms.span(r, cfg["M"], cfg["steps"], cfg["iters"]), cfg)
        if a.parent_bank:      # replicate 0: the parent's frozen rank-64 files, unchanged
            rd = Path(pdirs[n]) / "output"
            pf = json.loads((rd / "frozen.json").read_text())
            pf["spans"] = {"64": pf["spans"]["64"]}
            G64, o64 = P.load_rom(a.parent_bank, n)
            ks = sorted({pf[x]["k"] for x in ("head", "head_fast") if pf.get(x)})
            parms = P.Arms(n, T, G64, o64, np.load(rd / "rotation.npz")["V"], P.load_heads(rd, ks)); del G64, o64
            c64 = P.frozen_calls(parms, pf, prefix="r64_")
            c64 = {("r64_span" if k == "r64_span64" else k): v for k, v in c64.items()}
            for name, (call, cfg) in c64.items():
                if name not in fz["arms"]:
                    continue                      # head_fast identical to head: not duplicated
                want = fz["arms"][name]
                if any(cfg[x] != want[x] for x in ("M", "steps", "iters")) or cfg.get("k") != want.get("k"):
                    raise RuntimeError(f"parent frozen {name} differs from frozen file: {cfg} vs {want}")
                calls[name] = (call, want)
        calls = {k: calls[k] for k in sorted(calls)}
        gc.collect()
        return calls, fz

    a.ladder = a.ladders; a.romdev = ""; a.interp = None
    P.cmd_compare(a, build=build, rows=rows)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--family", required=True); p.add_argument("--out", required=True)
        p.add_argument("--smoke", action="store_true"); p.add_argument("--meshes", default="64,96,128")
        p.add_argument("--cases", type=int, default=0)
    p = sub.add_parser("rdev"); common(p); p.add_argument("--bank", required=True)
    p.add_argument("--headdata", required=True); p.add_argument("--dev", required=True)
    p.add_argument("--mesh", type=int, required=True); p.add_argument("--ranks", default="96,128")
    p.add_argument("--ks", default="8,16"); p.add_argument("--width", type=int, default=HEAD["width"])
    p.add_argument("--head-steps", type=int, default=HEAD["steps"]); p.add_argument("--head-seed", type=int, default=HEAD["seed"])
    p.add_argument("--modes", default="584,1168,2336"); p.add_argument("--grid", default="10:3,20:3,40:3,80:3,20:5,40:5")
    p.add_argument("--no-span", action="store_true"); p.add_argument("--frozen", default=None)
    p.add_argument("--heads-from", default=None)
    p = sub.add_parser("ladder"); common(p); p.add_argument("--cohort", required=True)
    p.add_argument("--start", type=int, default=0); p.add_argument("--count", type=int, default=0)
    p.add_argument("--blocks", required=True)
    p = sub.add_parser("cmp"); common(p); p.add_argument("--cohort", required=True)
    p.add_argument("--bank", required=True); p.add_argument("--heads", required=True)
    p.add_argument("--frozen", default=str(HERE / "frozen_rank")); p.add_argument("--ladders", required=True)
    p.add_argument("--parent-bank", default=None); p.add_argument("--parent-romdev", default=None)
    p.add_argument("--devcompare", default=None); p.add_argument("--reps", type=int, default=9)
    p.add_argument("--timing-cases", default="0,1,2"); p.add_argument("--shortlist", type=int, default=4)
    p.add_argument("--gate", type=float, default=1.10); p.add_argument("--gate-block", type=float, default=1.25)
    p.add_argument("--sentinels", default="r64_head,r128_head")
    a = ap.parse_args()
    log(f"backend={jax.default_backend()} devices={jax.devices()} args={vars(a)}")
    dict(rdev=cmd_rdev, ladder=cmd_ladder, cmp=cmd_cmp)[a.cmd](a)


if __name__ == "__main__":
    main()
