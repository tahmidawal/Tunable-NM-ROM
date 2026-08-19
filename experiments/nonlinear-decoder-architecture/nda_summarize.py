"""Generate architecture tables directly from run JSONs and saved controls."""
from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")


def read(path):
    with open(path) as f:
        return json.load(f)


def nparams_pkl(path, pde):
    with open(path, "rb") as f:
        d = pickle.load(f)
    params = d["stages"][0]["params"] if pde == "poisson" else d["params"]
    leaves = []

    def visit(x):
        if isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        elif hasattr(x, "shape"):
            leaves.append(x)

    visit(params)
    return int(sum(np.prod(x.shape) for x in leaves))


def pick(rows, scheme):
    matches = [r for r in rows if r["scheme"] == scheme and r["init"] == "nearest"]
    if len(matches) != 1:
        raise ValueError(f"expected one {scheme}/nearest row, got {len(matches)}")
    return matches[0]


def poisson_rows():
    out = []
    base = os.path.join(WT, "experiments", "poisson2d-rom-objective", "runs",
                        "followup", "pk_K16")
    tr = read(os.path.join(base, "autodec_K16_N64_hbc_report.json"))
    rom = read(os.path.join(base, "rom_K16.json"))
    full = pick(rom["rows"], "full")
    eq = pick(rom["rows"], "nnls")
    cp = os.path.join(WT, "experiments", "cost-to-tolerance", "ckpt_poisson",
                      "autodec_K16_N64_hbc_stages.pkl")
    out.append(dict(pde="Poisson", cell="saved-control", architecture="film",
                    hidden=tr["config"]["hidden"], layers=tr["config"]["n_layers"],
                    group_size=1, n_params=nparams_pkl(cp, "poisson"),
                    train=tr["train_mean_rel_l2"],
                    decoder=rom["oracle"]["nearest"],
                    full=full["rom_rel_l2_mean"], eq=eq["rom_rel_l2_mean"],
                    M=64, m=256))
    for cell in sorted(glob.glob(os.path.join(RUNS, "nda_p*"))):
        rp = os.path.join(cell, "out", "autodec_K16_N64_hbc_report.json")
        mp = os.path.join(cell, "out", "rom.json")
        if not os.path.isfile(rp) or not os.path.isfile(mp):
            continue
        tr, rom = read(rp), read(mp)
        cfg = tr["config"]["decoder_config"]
        full, eq = pick(rom["rows"], "full"), pick(rom["rows"], "nnls")
        out.append(dict(pde="Poisson", cell=os.path.basename(cell),
                        architecture=cfg["name"], hidden=cfg["hidden"],
                        layers=cfg["n_layers"], group_size=cfg["group_size"],
                        n_params=tr["config"]["n_params"],
                        train=tr["train_mean_rel_l2"],
                        decoder=rom["oracle"]["nearest"],
                        full=full["rom_rel_l2_mean"], eq=eq["rom_rel_l2_mean"],
                        M=int(full["objective"].split("M")[-1]), m=eq["m"]))
    return out


def burgers_rows():
    out = []
    base = os.path.join(WT, "experiments", "burgers2d-rom-latent-stepping", "runs",
                        "ad_n64_k16")
    tr = read(os.path.join(base, "blat_ad_N64_K16_report.json"))
    rom = read(os.path.join(base, "blat_rom_N64_K16.json"))
    out.append(dict(pde="Burgers", cell="saved-control", architecture="film",
                    hidden=tr["config"]["ad_hidden"], layers=tr["config"]["ad_layers"],
                    group_size=1, n_freq=tr["n_freq"], n_params=tr["config"]["n_params"],
                    train=tr["train_rel_mean"],
                    decoder=rom["oracle_inferred_latent_test"]["traj_rel_mean"],
                    full=rom["rom"]["lspg:full:weak64"]["traj_rel_mean"],
                    eq256=rom["rom"]["lspg:eq256:weak64"]["traj_rel_mean"],
                    eq512=rom["rom"]["lspg:eq512:weak64"]["traj_rel_mean"]))
    for cell in sorted(glob.glob(os.path.join(RUNS, "nda_b*"))):
        rp = os.path.join(cell, "out", "blat_ad_N64_K16_report.json")
        mp = os.path.join(cell, "out", "blat_rom_N64_K16.json")
        if not os.path.isfile(rp) or not os.path.isfile(mp):
            continue
        tr, rom = read(rp), read(mp)
        cfg = tr["config"]["decoder_config"]
        out.append(dict(pde="Burgers", cell=os.path.basename(cell),
                        architecture=cfg["name"], hidden=cfg["hidden"],
                        layers=cfg["n_layers"], group_size=cfg["group_size"],
                        n_freq=tr["n_freq"], n_params=tr["config"]["n_params"],
                        train=tr["train_rel_mean"],
                        decoder=rom["oracle_inferred_latent_test"]["traj_rel_mean"],
                        full=rom["rom"]["lspg:full:weak64"]["traj_rel_mean"],
                        eq256=rom["rom"]["lspg:eq256:weak64"]["traj_rel_mean"],
                        eq512=rom["rom"]["lspg:eq512:weak64"]["traj_rel_mean"]))
    return out


def f(x):
    return f"{x:.3e}"


def main():
    p, b = poisson_rows(), burgers_rows()
    with open(os.path.join(HERE, "summary.json"), "w") as fp:
        json.dump(dict(poisson=p, burgers=b), fp, indent=2)
    md = ["# Pure nonlinear decoder architecture results", "",
          "These tables are generated directly from the run JSONs. Results are provisional until the selected arms have multi-seed and trust-region confirmation.", "",
          "## Poisson-2D, N=64, k=16", "",
          "| cell | architecture | width×layers | group | parameters | train | decoder/oracle | full weak ROM | EQ weak ROM | M,m | EQ/full |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in p:
        md.append(f"| {r['cell']} | {r['architecture']} | {r['hidden']}×{r['layers']} | "
                  f"{r['group_size']} | {r['n_params']:,} | {f(r['train'])} | "
                  f"{f(r['decoder'])} | {f(r['full'])} | {f(r['eq'])} | "
                  f"{r['M']},{r['m']} | {r['eq']/r['full']:.3f} |")
    md += ["", "## Burgers-2D, N=64, k=16", "",
           "| cell | architecture | width×layers | group | frequencies | parameters | train | decoder/oracle | full weak ROM | EQ256 | EQ512 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in b:
        md.append(f"| {r['cell']} | {r['architecture']} | {r['hidden']}×{r['layers']} | "
                  f"{r['group_size']} | {r['n_freq']} | {r['n_params']:,} | "
                  f"{f(r['train'])} | {f(r['decoder'])} | {f(r['full'])} | "
                  f"{f(r['eq256'])} | {f(r['eq512'])} |")
    md += ["", "The per-cell Burgers timing fields are not used here because those inherited runs did not burn in the GPU after host-bound NNLS fitting. Same-GPU benchmark cells with explicit burn-in provide the accepted decoder-speed comparison.", ""]
    with open(os.path.join(HERE, "SUMMARY.md"), "w") as fp:
        fp.write("\n".join(md))
    print(f"wrote {os.path.join(HERE, 'summary.json')} and SUMMARY.md")


if __name__ == "__main__":
    main()
