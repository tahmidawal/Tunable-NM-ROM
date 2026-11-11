"""Held-out 3D head-to-head from saved JSON only. Solver configurations and ROM ranks are the DEV-selected ones
(no re-selection on the held-out cohort). Writes runs/summary3d.json."""
import json, collections, numpy as np
def rows(f, backend):
    g = collections.defaultdict(dict)
    for l in open(f):
        r = json.loads(l)
        if r["backend"] == backend: g[(r["solver"], r["tol"], r["p"], r["N"])][r["case"]] = r
    return g
def stat(d):
    t = [x["total_ms"] for x in d.values()]; e = [x["l2w"] for x in d.values()]
    return dict(n=len(d), l2w_mean=float(np.mean(e)), l2w_max=float(np.max(e)), t_med=float(np.median(t)), t_min=float(np.min(t)), t_max=float(np.max(t)))
out = {"solvers": {}, "rom": {}, "breakeven": {}}
for lab, devf, bk, testf in [("gpu", "runs/t3/agg_dev_gpu.json", "gpu", "runs/t3/test_gpu.jsonl"),
                             ("cpu1", "runs/t3/agg_dev_cpu.json", "cpu_jax_1core_dense", "runs/t3/test_cpu1_dense.jsonl")]:
    best = json.load(open(devf))["best"]; g = rows(testf, bk)
    for tgt in ("0.03", "0.01", "0.003"):
        b = best[f"{tgt}|{bk}"]["overall"]; key = (b["solver"], b["tol"], b["p"], b["N"])
        out["solvers"][f"{lab}|{tgt}"] = dict(solver=b["solver"], tol=b["tol"], p=b["p"], N=b["N"], dofs=b["dofs"],
            dev=dict(l2w_mean=b["l2w_mean"], l2w_max=b["l2w_max"], t_med=b["t_med"], t_min=b["t_min"], t_max=b["t_max"]),
            test=stat(g[key]) if key in g else None)
def rom(f, rank):
    d = json.load(open(f)); v = {r["case"]: r for r in d["rows"] if r["rank"] == rank}
    s = stat(v); s["recon_truthgrid_ms"] = float(np.median([r["recon_truthgrid_ms"] for r in v.values()])); return d, s
sel = [("pod|0.03", "pod_p4N12_2048", 96), ("pod|0.01", "pod_p4N12_2048", 256), ("bank128|0.03", "bankgal_R128", 128), ("bank256|0.03", "bankgal_R256", 128)]
for lab, stem, rank in sel:
    dd, sd = rom(f"runs/rom3/{stem}_dev_quiet.json" if stem != "bankgal_R128" else f"runs/rom3/{stem}_dev.json", rank)
    dt, st = rom(f"runs/rom3/{stem}_test.json", rank)
    off = dt["snapshots"]["seconds"] + dt["basis_seconds"] + dt["galerkin_offline_seconds"] + dt.get("bank_train_seconds", 0.0)
    out["rom"][lab] = dict(rank=rank, dev_l2w_mean=sd["l2w_mean"], dev_l2w_max=sd["l2w_max"], test=st, offline_seconds=off,
        offline_parts=dict(snapshots=dt["snapshots"]["seconds"], basis=dt["basis_seconds"], galerkin=dt["galerkin_offline_seconds"], bank_train=dt.get("bank_train_seconds", 0.0)))
for lab in out["rom"]:
    arm, tgt = lab.split("|")
    for bk in ("gpu", "cpu1"):
        s = out["solvers"][f"{bk}|{tgt}"]["test"]; r = out["rom"][lab]
        save = s["t_med"] - r["test"]["t_med"]
        out["breakeven"][f"{lab}|vs_{bk}"] = dict(solver_ms=s["t_med"], rom_ms=r["test"]["t_med"], ratio=s["t_med"] / r["test"]["t_med"],
            saving_ms=save, queries_to_break_even=(r["offline_seconds"] * 1e3 / save) if save > 0 else None,
            ratio_incl_recon=s["t_med"] / (r["test"]["t_med"] + r["test"]["recon_truthgrid_ms"]))
json.dump(out, open("runs/summary3d.json", "w"), indent=1)
for k, v in out["solvers"].items():
    t = v["test"]; print(f"{k:11s} {v['solver']} p{v['p']} N{v['N']} tol {v['tol']} dofs {v['dofs']}  dev {100*v['dev']['l2w_mean']:.2f}%/{v['dev']['t_med']:.2f} ms | test {100*t['l2w_mean']:.2f}% max {100*t['l2w_max']:.2f}%  {t['t_med']:.2f} ms [{t['t_min']:.2f}, {t['t_max']:.2f}]")
for k, v in out["rom"].items():
    t = v["test"]; print(f"{k:13s} rank {v['rank']} dev {100*v['dev_l2w_mean']:.2f}% | test {100*t['l2w_mean']:.2f}% max {100*t['l2w_max']:.2f}%  online {t['t_med']:.3f} ms [{t['t_min']:.3f}, {t['t_max']:.3f}] recon {t['recon_truthgrid_ms']:.2f} ms  offline {v['offline_seconds']:.0f} s")
for k, v in out["breakeven"].items():
    print(f"{k:22s} solver {v['solver_ms']:.2f} ms rom {v['rom_ms']:.3f} ms ratio {v['ratio']:.1f}x (incl. recon {v['ratio_incl_recon']:.2f}x) saving {v['saving_ms']:.2f} ms  break-even {v['queries_to_break_even']:.3g} queries")
