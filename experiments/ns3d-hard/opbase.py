"""Neural-operator baseline on the hard family (PROTOCOL.md amendment A5.2). PyTorch only.

train: FNO (experiments/ns3d-operators/ops3d.py factory, float64) on the 512 head
       trajectories of family B at 64^3; map (u0, log nu) -> five evolved frames.
eval : errors with the ROM's metric on a cohort at 64/96/128 and batch-1 forward time,
       plus the accuracy-matched solver from a saved ladder.

Differences from the operators lane's train_op.py, all forced by this family and declared:
* data are the stored 192^3 CNAB2 truth frames truncated to 64^3 (not regenerated);
* the output projection keeps every mode |k_i| <= n/2 - 1 (divergence-free, zero mean); the
  lane's 2/3 mask would remove content the truncated 192^3 truth has;
* random cyclic grid shifts (one per sample, input and target together) as augmentation.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ns3d-operators"))
import ops3d as O

CONFIGS = {
    "fno-l": dict(family="fno", width=32, modes=16, layers=4, dtype="float64"),
    "fno-xl": dict(family="fno", width=48, modes=20, layers=4, dtype="float64"),
    "fno-smoke": dict(family="fno", width=4, modes=4, layers=2, dtype="float64"),
}
BATCH, LR, WD, FLOOR, SEED, VAL_EVERY = 8, 1e-3, 1e-4, 1e-5, 20260914, 8


def log(*a):
    print(*a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); tmp.replace(path)


def resample(u, n_to):
    """Fourier resampling of (..., n, n, n) to n_to; copies |k_i| <= min(n, n_to)/2 - 1
    (pipeline.spec_resample in PyTorch)."""
    n = u.shape[-1]
    if n == n_to:
        return u
    m = min(n, n_to) // 2
    spec = torch.fft.fftn(u, dim=(-3, -2, -1), norm="forward")
    for axis in (-3, -2, -1):
        size = spec.shape[axis]
        low = spec.narrow(axis, 0, m); high = spec.narrow(axis, size - m + 1, m - 1)
        shape = list(spec.shape); shape[axis] = n_to - (2 * m - 1)
        spec = torch.cat((low, torch.zeros(shape, dtype=spec.dtype, device=spec.device), high), dim=axis)
    return torch.fft.ifftn(spec, dim=(-3, -2, -1), norm="forward").real


class Projector:
    """Divergence-free, zero-mean projection keeping |k_i| <= n/2 - 1."""

    def __init__(self, n, device="cuda"):
        kx = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64, device=device)
        kz = torch.fft.rfftfreq(n, d=1.0 / n, dtype=torch.float64, device=device)
        modes = torch.stack(torch.meshgrid(kx, kx, kz, indexing="ij"))
        self.k = 2 * math.pi * modes
        k2 = (self.k ** 2).sum(0)
        mask = (modes.abs() < n / 2 - 0.5).all(0); mask[0, 0, 0] = False
        self.mask = mask.to(torch.float64)
        self.inv = torch.where(k2 > 0, 1.0 / torch.where(k2 > 0, k2, torch.ones_like(k2)), torch.zeros_like(k2))
        self.n = n

    def __call__(self, u):
        uh = torch.fft.rfftn(u, dim=(-3, -2, -1))
        dot = (self.k * uh).sum(-4, keepdim=True)
        uh = (uh - self.k * (dot * self.inv)) * self.mask
        return torch.fft.irfftn(uh, s=(self.n,) * 3, dim=(-3, -2, -1))


def load_cases(dirs, n, dtype=torch.float32, limit=0):
    """Stored truth of each directory, truncated to n; returns (cases,6,3,n,n,n) on the GPU."""
    out, pars = [], []
    for d in dirs:
        arr = np.load(Path(d) / "output" / "truth.npy", mmap_mode="r")
        pars.append(np.load(Path(d) / "output" / "params.npy"))
        for i in range(arr.shape[0]):
            x = torch.from_numpy(np.ascontiguousarray(arr[i])).cuda()
            out.append(resample(x, n).to(dtype))
            if limit and len(out) >= limit: break
        log(f"loaded {d} ({len(out)} cases)")
        if limit and len(out) >= limit: break
    par = np.concatenate(pars)[:len(out)]
    return torch.stack(out), par


def setup():
    env = O.configure()
    return env


def evaluate(net, norm, proj, U, NU, idx):
    net.eval(); errs = []
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            ids = torch.as_tensor(idx[s:s + BATCH], device="cuda")
            y = U[ids].double(); u0 = y[:, 0]
            errs.append(O.initial_relative(O.predict(net, u0, NU[ids], norm, proj), y, u0).cpu().numpy())
    e = np.concatenate(errs)
    if not np.isfinite(e).all():
        raise RuntimeError("nonfinite validation error")
    per = e[:, 1:].max(1)
    return dict(mean_case_max=float(per.mean()), median_case_max=float(np.median(per)), worst_case_max=float(per.max()))


def cmd_train(a):
    env = setup(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise RuntimeError("refusing: best.pt exists")
    config = dict(CONFIGS[a.config]); n = a.mesh; t_load = time.time()
    U, par = load_cases(a.data.split(","), n, limit=a.cases)
    count = len(par)
    if not a.smoke and count != 512:
        raise RuntimeError(f"expected 512 trajectories, got {count}")
    if len({tuple(r) for r in np.round(par, 9)}) != count:
        raise RuntimeError("duplicate parameter rows")
    for d in (a.exclude.split(",") if a.exclude else []):
        rows = {tuple(r) for r in np.round(par, 9)}
        other = np.round(np.load(Path(d) / "output" / "params.npy"), 9)
        if any(tuple(r) in rows for r in other):
            raise RuntimeError(f"training rows overlap evaluation cohort {d}")
    load_seconds = time.time() - t_load
    NU = torch.as_tensor(par[:, -1], dtype=torch.float64, device="cuda")
    is_val = (np.arange(count) % VAL_EVERY) == VAL_EVERY - 1
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    s0 = sum(float(U[i, 0].double().square().mean()) for i in tr) / len(tr)
    s1 = sum(float(U[i, 1:].double().square().mean()) for i in tr) / len(tr)
    lognu = torch.log(NU[torch.as_tensor(tr, device="cuda")])
    norm = (torch.tensor(math.sqrt(s0), dtype=torch.float64, device="cuda"), lognu.mean(), lognu.std().clamp_min(1e-12),
            torch.tensor(math.sqrt(s1), dtype=torch.float64, device="cuda"))
    proj = Projector(n)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    net = O.make_model(config); O.check_dtypes(net)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(SEED); wall = float(a.wall_seconds)
    meta = dict(environment=env, config=config, arm=a.config, mesh=n, cases=count, train=len(tr), validation=len(va),
                normalization=[float(v) for v in norm], parameters=O.parameter_count(net), wall_budget_seconds=wall,
                load_seconds=load_seconds, job_id=os.environ.get("SLURM_JOB_ID"), data=a.data,
                augmentation="random cyclic grid shift per sample", split="index % 8 == 7 -> validation")
    jdump(out / "provenance.json", meta); log(json.dumps({k: meta[k] for k in ("arm", "parameters", "load_seconds")}))
    best, best_epoch, hist, step, started = float("inf"), -1, [], 0, time.monotonic()
    stop = "epoch_cap"
    for epoch in range(100000):
        net.train(); order = tr[torch.randperm(len(tr), generator=gen).numpy()]; losses = []
        for s in range(0, len(order), BATCH):
            ids = order[s:s + BATCH]
            frac = min(1.0, (time.monotonic() - started) / wall)
            lr = FLOOR + (LR - FLOOR) * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups: g["lr"] = lr
            y = U[torch.as_tensor(ids, device="cuda")].double()
            sh = torch.randint(0, n, (len(ids), 3), generator=gen)
            y = torch.stack([torch.roll(y[b], tuple(int(v) for v in sh[b]), dims=(-3, -2, -1)) for b in range(len(ids))])
            u0 = y[:, 0]
            e = O.initial_relative(O.predict(net, u0, NU[torch.as_tensor(ids, device="cuda")], norm, proj), y, u0)[:, 1:]
            loss = e.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            opt.step(); step += 1; losses.append(float(loss.detach()))
            if time.monotonic() - started >= wall: break
        m = evaluate(net, norm, proj, U, NU, va)
        if m["mean_case_max"] < best:
            best, best_epoch = m["mean_case_max"], epoch
            O.save_checkpoint(out / "best.tmp", net, config, norm, dict(epoch=epoch, step=step, best=best, mesh=n, arm=a.config))
            (out / "best.tmp").replace(out / "best.pt")
        hist.append(dict(epoch=epoch, steps=step, train_loss=float(np.mean(losses)), validation=m, lr=lr,
                         wall_seconds=time.monotonic() - started, best_so_far=best))
        jdump(out / "history.json", hist)
        if epoch % 5 == 0 or time.monotonic() - started >= wall:
            log(json.dumps(dict(epoch=epoch, steps=step, train=hist[-1]["train_loss"], val_mean=m["mean_case_max"],
                                val_worst=m["worst_case_max"], lr=lr, wall=round(hist[-1]["wall_seconds"], 1))))
        if time.monotonic() - started >= wall:
            stop = "wall_budget"; break
        if a.smoke and epoch >= 1: break
    seconds = time.monotonic() - started
    del net, opt; torch.cuda.empty_cache()
    net2, norm2, ck = O.load_checkpoint(out / "best.pt")
    final = evaluate(net2, norm2, proj, U, NU, va)
    jdump(out / "result.json", dict(arm=a.config, config=config, stop_reason=stop, training_seconds=seconds,
                                    load_seconds=load_seconds, epochs=len(hist), steps=step, best_epoch=ck["epoch"],
                                    validation_best_checkpoint=final, reproduced=bool(abs(final["mean_case_max"] - best) <= 1e-6 * best),
                                    parameters=O.parameter_count(net2), job_id=os.environ.get("SLURM_JOB_ID")))
    log("TRAIN DONE", a.config, "val_mean", final["mean_case_max"], "val_worst", final["worst_case_max"])


def stats(e):
    w = e[:, 1:].max(1)
    return dict(finite=bool(np.isfinite(e).all()), evolved_worst=float(w.max()), evolved_median=float(np.median(w)),
                per_time_median=[float(x) for x in np.median(e, 0)], per_time_worst=[float(x) for x in e.max(0)], cases=int(len(e)))


def cmd_eval(a):
    env = setup(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = a.runs.split(",")
    res = {r: json.loads((Path(r) / "output" / "result.json").read_text()) for r in runs}
    chosen = min(runs, key=lambda r: res[r]["validation_best_checkpoint"]["mean_case_max"])
    todo = runs if a.all else [chosen]
    rows = []
    for f in (a.ladder.split(",") if a.ladder else []):
        rows += json.loads(Path(f).read_text())["rows"]
    block = json.loads(Path(a.compare).read_text()) if a.compare else None
    arr = [np.load(Path(d) / "output" / "truth.npy", mmap_mode="r") for d in a.cohort.split(",")]
    par = np.concatenate([np.load(Path(d) / "output" / "params.npy") for d in a.cohort.split(",")])
    cases = [x[i] for x in arr for i in range(x.shape[0])]
    meshes = [int(x) for x in a.meshes.split(",")]
    rep = dict(schema="ns3d-hard-opbase-v1", label=a.label, cohort=a.cohort, cases=len(cases), environment=env,
               job_id=os.environ.get("SLURM_JOB_ID"), selected_by_validation=chosen,
               validation={r: res[r]["validation_best_checkpoint"] for r in runs}, operators={})
    for r in todo:
        net, norm, ck = O.load_checkpoint(Path(r) / "output" / "best.pt")
        n0 = ck["mesh"]; proj = Projector(n0)

        def query(u0n, nu, n):
            u0 = resample(u0n, n0)
            pred = O.predict(net, u0[None], nu[None], norm, proj)[0]
            return resample(pred, n)
        orep = dict(arm=res[r]["arm"], parameters=res[r]["parameters"], training_seconds=res[r]["training_seconds"],
                    native_mesh=n0, meshes={})
        rep["operators"][r] = orep
        with torch.no_grad():
            for n in meshes:
                errs = np.empty((len(cases), 6)); u0s = {}
                for c, fr in enumerate(cases):
                    tr = resample(torch.from_numpy(np.ascontiguousarray(fr)).cuda(), n)
                    nu = torch.tensor(float(par[c, -1]), dtype=torch.float64, device="cuda")
                    pred = query(tr[0], nu, n)
                    num = (pred - tr).square().sum(dim=(1, 2, 3, 4)).sqrt()
                    errs[c] = (num / tr[0].square().sum().sqrt()).cpu().numpy()
                    if c < 3: u0s[c] = (tr[0].clone(), nu)
                st = stats(errs); samples = []
                for c, (u0, nu) in u0s.items():
                    for i in range(2 + a.reps):
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        query(u0, nu, n); torch.cuda.synchronize()
                        if i >= 2: samples.append((time.perf_counter() - t0) * 1e3)
                ms = dict(median_ms=float(np.median(samples)), min_ms=float(min(samples)), max_ms=float(max(samples)), n=len(samples))
                m = dict(stats=st, errors=errs.tolist(), ms=ms,
                         band="useful" if st["evolved_worst"] <= 0.03 else ("marginal" if st["evolved_worst"] <= 0.05 else "not useful"))
                sn = str(n)
                elig = [x for x in rows if x["stats"][sn]["finite"] and x["stats"][sn]["evolved_worst"] <= min(st["evolved_worst"], 1.0)]
                if elig:
                    b = min(elig, key=lambda x: x["seconds_median"])
                    tag = f"SOLVE|{b['scheme']}|{b['n']}|{b['steps']}"
                    pooled = (block or {}).get("meshes", {}).get(sn, {}).get("timing_pooled", {})
                    sms, src = (pooled[tag]["median_ms"], "timing block") if tag in pooled else (1e3 * b["seconds_median"], "ladder solve-only median")
                    m["matched_solver"] = dict(scheme=b["scheme"], n=b["n"], steps=b["steps"], cohort_worst=b["stats"][sn]["evolved_worst"],
                                               ms=sms, time_source=src, ratio=sms / ms["median_ms"], eligible=len(elig))
                else:
                    m["matched_solver"] = None
                orep["meshes"][sn] = m
                log(f"{res[r]['arm']} mesh {n}: worst {st['evolved_worst']:.5f} median {st['evolved_median']:.5f} "
                    f"{ms['median_ms']:.2f} ms matched {m['matched_solver'] and {k: m['matched_solver'][k] for k in ('scheme', 'n', 'steps', 'ms', 'ratio', 'time_source')}}")
                jdump(out / "opbase.json", rep)
        del net; torch.cuda.empty_cache()
    log("DONE")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("--data", required=True); p.add_argument("--config", required=True)
    p.add_argument("--out", required=True); p.add_argument("--mesh", type=int, default=64)
    p.add_argument("--wall-seconds", type=float, default=5400.); p.add_argument("--exclude", default=None)
    p.add_argument("--cases", type=int, default=0); p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("eval"); p.add_argument("--runs", required=True); p.add_argument("--cohort", required=True)
    p.add_argument("--label", required=True); p.add_argument("--out", required=True); p.add_argument("--meshes", default="64,96,128")
    p.add_argument("--ladder", default=None); p.add_argument("--compare", default=None)
    p.add_argument("--reps", type=int, default=9); p.add_argument("--all", action="store_true")
    a = ap.parse_args()
    dict(train=cmd_train, eval=cmd_eval)[a.cmd](a)


if __name__ == "__main__":
    main()
