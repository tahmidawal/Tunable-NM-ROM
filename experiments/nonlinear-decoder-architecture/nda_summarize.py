"""Generate auditable architecture tables directly from pulled run artifacts."""
from __future__ import annotations

import glob
import json
import os
import pickle
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")


def read(path):
    with open(path) as f:
        return json.load(f)


def first(pattern):
    paths = sorted(glob.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one file matching {pattern}, got {paths}")
    return paths[0]


def nparams_pkl(path, pde):
    with open(path, "rb") as f:
        d = pickle.load(f)
    params = d["stages"][0]["params"] if pde == "poisson" else d["params"]
    leaves = []

    def visit(x):
        if isinstance(x, dict):
            for value in x.values():
                visit(value)
        elif isinstance(x, (list, tuple)):
            for value in x:
                visit(value)
        elif hasattr(x, "shape"):
            leaves.append(x)

    visit(params)
    return int(sum(np.prod(x.shape) for x in leaves))


def pick(rows, scheme):
    matches = [r for r in rows if r["scheme"] == scheme and r["init"] == "nearest"]
    if len(matches) != 1:
        raise ValueError(f"expected one {scheme}/nearest row, got {len(matches)}")
    return matches[0]


def seed_of(config):
    return int(config.get("train_seed", config.get("seed", 0)))


def poisson_rows():
    out = []
    # This is the only fair saved-control ROM comparison: the old checkpoint was
    # re-evaluated at exactly M=128,m=512 in the same validation job as H98.
    cell = "nda_pvalid_g98_r8"
    valid = read(os.path.join(RUNS, cell, "out", "control_tr0.json"))
    old_base = os.path.join(WT, "experiments", "poisson2d-rom-objective", "runs",
                            "followup", "pk_K16")
    old_train = read(os.path.join(old_base, "autodec_K16_N64_hbc_report.json"))
    old_ckpt = os.path.join(WT, "experiments", "cost-to-tolerance", "ckpt_poisson",
                            "autodec_K16_N64_hbc_stages.pkl")
    full, eq = pick(valid["rows"], "full"), pick(valid["rows"], "nnls")
    out.append(dict(
        pde="Poisson", cell="saved-control/fair-M128", seed=0,
        architecture="film", hidden=old_train["config"]["hidden"],
        layers=old_train["config"]["n_layers"], group_size=1,
        n_params=nparams_pkl(old_ckpt, "poisson"),
        train=old_train["train_mean_rel_l2"], decoder=valid["oracle"]["nearest"],
        full=full["rom_rel_l2_mean"], full_median=full["rom_rel_l2_med"],
        full_max=full["rom_rel_l2_max"], eq=eq["rom_rel_l2_mean"],
        eq_median=eq["rom_rel_l2_med"], eq_max=eq["rom_rel_l2_max"],
        M=int(full["objective"].split("M")[-1]), m=eq["m"],
    ))
    for path in sorted(glob.glob(os.path.join(RUNS, "nda_p*"))):
        reports = glob.glob(os.path.join(path, "out", "autodec_K16_N64_hbc*_report.json"))
        rom_path = os.path.join(path, "out", "rom.json")
        if len(reports) != 1 or not os.path.isfile(rom_path):
            continue
        train, rom = read(reports[0]), read(rom_path)
        config = train["config"]
        decoder = config.get("decoder_config")
        if decoder is None:
            continue
        full, eq = pick(rom["rows"], "full"), pick(rom["rows"], "nnls")
        out.append(dict(
            pde="Poisson", cell=os.path.basename(path), seed=seed_of(config),
            architecture=decoder["name"], hidden=decoder["hidden"],
            layers=decoder["n_layers"], group_size=decoder.get("group_size", 1),
            n_params=config["n_params"], train=train["train_mean_rel_l2"],
            decoder=rom["oracle"]["nearest"], full=full["rom_rel_l2_mean"],
            full_median=full["rom_rel_l2_med"], full_max=full["rom_rel_l2_max"],
            eq=eq["rom_rel_l2_mean"], eq_median=eq["rom_rel_l2_med"],
            eq_max=eq["rom_rel_l2_max"],
            M=int(full["objective"].split("M")[-1]), m=eq["m"],
        ))
    return out


def burgers_rows():
    out = []
    base = os.path.join(WT, "experiments", "burgers2d-rom-latent-stepping", "runs",
                        "ad_n64_k16")
    train = read(os.path.join(base, "blat_ad_N64_K16_report.json"))
    rom = read(os.path.join(base, "blat_rom_N64_K16.json"))
    out.append(dict(
        pde="Burgers", cell="saved-control", seed=0, architecture="film",
        hidden=train["config"]["ad_hidden"], layers=train["config"]["ad_layers"],
        group_size=1, n_freq=train["n_freq"], n_params=train["config"]["n_params"],
        train=train["train_rel_mean"],
        decoder=rom["oracle_inferred_latent_test"]["traj_rel_mean"],
        full=rom["rom"]["lspg:full:weak64"]["traj_rel_mean"],
        eq256=rom["rom"]["lspg:eq256:weak64"]["traj_rel_mean"],
        eq512=rom["rom"]["lspg:eq512:weak64"]["traj_rel_mean"],
    ))
    for path in sorted(glob.glob(os.path.join(RUNS, "nda_b*"))):
        report_path = os.path.join(path, "out", "blat_ad_N64_K16_report.json")
        rom_path = os.path.join(path, "out", "blat_rom_N64_K16.json")
        if not os.path.isfile(report_path) or not os.path.isfile(rom_path):
            continue
        train, rom = read(report_path), read(rom_path)
        config = train["config"]
        decoder = config.get("decoder_config")
        if decoder is None or "lspg:full:weak64" not in rom["rom"]:
            continue
        out.append(dict(
            pde="Burgers", cell=os.path.basename(path), seed=seed_of(config),
            architecture=decoder["name"], hidden=decoder["hidden"],
            layers=decoder["n_layers"], group_size=decoder.get("group_size", 1),
            n_freq=train["n_freq"], n_params=config["n_params"],
            train=train["train_rel_mean"],
            decoder=rom["oracle_inferred_latent_test"]["traj_rel_mean"],
            full=rom["rom"]["lspg:full:weak64"]["traj_rel_mean"],
            eq256=rom["rom"]["lspg:eq256:weak64"]["traj_rel_mean"],
            eq512=rom["rom"]["lspg:eq512:weak64"]["traj_rel_mean"],
        ))
    return out


def burgers_objective_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS, "nda_bobj*")) +
                       glob.glob(os.path.join(RUNS, "nda_bg160l4g2f31_s*_r11")) +
                       glob.glob(os.path.join(RUNS, "nda_btrust*"))):
        report_paths = glob.glob(os.path.join(path, "out", "**", "blat_rom_N64_K16.json"),
                                 recursive=True)
        for report_path in sorted(report_paths):
            report = read(report_path)
            config = report["config"]
            seed = seed_of(config.get("ad_config", config))
            subdir = os.path.relpath(os.path.dirname(report_path), os.path.join(path, "out"))
            cell = os.path.basename(path) if subdir == "." else f"{os.path.basename(path)}/{subdir}"
            for M, m in ((96, 384), (128, 512)):
                full_key, eq_key = f"lspg:full:weak{M}", f"lspg:eq{m}:weak{M}"
                if full_key not in report["rom"] or eq_key not in report["rom"]:
                    continue
                full, eq = report["rom"][full_key], report["rom"][eq_key]
                rows.append(dict(
                    cell=cell, seed=seed,
                    trust_factor=float(config.get("tr_factor", 0.0)), M=M, m=m,
                    decoder=report["oracle_inferred_latent_test"]["traj_rel_mean"],
                    full=full["traj_rel_mean"], full_median=full["traj_rel_median"],
                    full_max=full["traj_rel_max"], full_blowups=full["n_blowup"],
                    eq=eq["traj_rel_mean"], eq_median=eq["traj_rel_median"],
                    eq_max=eq["traj_rel_max"], eq_blowups=eq["n_blowup"],
                ))
    return rows


def aggregate(rows, keys, value_keys, name):
    groups = {}
    for row in rows:
        group = tuple(row[k] for k in keys)
        groups.setdefault(group, []).append(row)
    out = []
    for group, members in sorted(groups.items()):
        seeds = sorted({r["seed"] for r in members})
        if seeds != [0, 1, 2]:
            continue
        result = {"name": name, "seeds": seeds, **dict(zip(keys, group))}
        for key in value_keys:
            values = np.asarray([r[key] for r in members], dtype=float)
            result[key] = {
                "values": values.tolist(), "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)), "max": float(values.max()),
            }
        out.append(result)
    return out


def benchmark(cell):
    path = os.path.join(RUNS, cell, "out", "benchmark.json")
    if not os.path.isfile(path):
        return None
    data = read(path)
    rows = []
    for key, control in data["timings"]["control"].items():
        variant = data["timings"]["variant"][key]
        cached = data["timings"].get("variant_cached", {}).get(key)
        match = re.search(r"_p(\d+)$", key)
        rows.append(dict(
            kernel=key.split("_p")[0], points=int(match.group(1)),
            control_ms=1e3 * control["median_s"],
            variant_ms=1e3 * variant["median_s"],
            speedup=float(control["median_s"] / variant["median_s"]),
            cached_variant_ms=None if cached is None else 1e3 * cached["median_s"],
            cached_speedup=None if cached is None else float(control["median_s"] / cached["median_s"]),
            outliers_control=control["outliers_gt_1p5_median"],
            outliers_variant=variant["outliers_gt_1p5_median"],
            outliers_cached=None if cached is None else cached["outliers_gt_1p5_median"],
        ))
    return dict(cell=cell, pde=data["pde"], backend=data["backend"],
                device=data.get("device"), precision=data["precision"], reps=data["reps"],
                warm=data["warm"], burn_seconds=data["burn_seconds"],
                models=data["models"], checks=data["checks"], rows=rows)


def e2e(cell):
    out = []
    for arm in ("control", "variant"):
        path = os.path.join(RUNS, cell, "out", f"{arm}.json")
        if not os.path.isfile(path):
            continue
        report = read(path)
        for row in report["rows"]:
            out.append(dict(
                cell=cell, arm=arm, M=row["M"], m=row["m"], tau=row["tau"],
                time_ms=row["time_ms"], solve_ms=row["time_ms_solve"],
                error=row["err_rel_l2"], error_median=row["err_rel_l2_median"],
                error_max=row["err_rel_l2_max"], jac_evals=row["jac_evals"],
                censored_frac=row["censored_frac"], gpu=row["gpu"],
                reps=report["config"]["time_reps"],
            ))
    return out


def sci(value):
    return f"{value:.3e}"


def markdown_table(headers, align, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    out.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return out


def main():
    poisson = poisson_rows()
    burgers = burgers_rows()
    burgers_objective = burgers_objective_rows()
    poisson_seed = aggregate(
        [r for r in poisson if r["cell"].startswith(("nda_pg98", "nda_pg100"))],
        ["architecture", "hidden", "layers", "group_size", "n_params", "M", "m"],
        ["decoder", "full", "eq"], "Poisson",
    )
    burgers_seed = aggregate(
        [r for r in burgers_objective if r["trust_factor"] == 0.0],
        ["M", "m"], ["decoder", "full", "eq"], "Burgers",
    )
    benchmarks = [x for x in (
        benchmark("nda_pbench_g98b_r8"), benchmark("nda_bbench_g160_r12")) if x]
    e2e_rows = e2e("nda_pe2e_g98_r11") + e2e("nda_be2e_g160_r14")
    result = dict(
        poisson=poisson, poisson_three_seed=poisson_seed,
        burgers=burgers, burgers_objectives=burgers_objective,
        burgers_three_seed=burgers_seed, benchmarks=benchmarks, e2e=e2e_rows,
        rejected=[
            dict(cell="nda_pbench_g98_r8",
                 reason="timed before post-burn exact-kernel warmups; retained but excluded"),
            dict(cell="nda_be2e_g160_r12",
                 reason="failed before the compact arm because the driver dropped checkpoint decoder metadata; retained and superseded by r14"),
        ],
    )
    with open(os.path.join(HERE, "summary.json"), "w") as fp:
        json.dump(result, fp, indent=2)

    md = [
        "# Pure nonlinear decoder architecture results", "",
        "Generated directly from pulled run JSONs. A row appears only when its required artifacts are present; pending cluster cells are therefore omitted rather than guessed.", "",
        "## Poisson-2D architecture screen (N=64, k=16)", "",
    ]
    md += markdown_table(
        ["cell", "seed", "architecture", "width×layers", "group", "parameters", "train", "decoder", "full", "EQ", "M,m", "EQ/full"],
        ["---", "---:", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        [[r["cell"], r["seed"], r["architecture"], f'{r["hidden"]}×{r["layers"]}',
          r["group_size"], f'{r["n_params"]:,}', sci(r["train"]), sci(r["decoder"]),
          sci(r["full"]), sci(r["eq"]), f'{r["M"]},{r["m"]}', f'{r["eq"]/r["full"]:.3f}']
         for r in poisson]
    )
    md += ["", "## Poisson three-seed confirmation", ""]
    md += markdown_table(
        ["width×layers", "parameters", "metric", "mean", "sample std", "max"],
        ["---:", "---:", "---", "---:", "---:", "---:"],
        [[f'{r["hidden"]}×{r["layers"]}', f'{r["n_params"]:,}', metric,
          sci(r[metric]["mean"]), sci(r[metric]["sample_std"]), sci(r[metric]["max"])]
         for r in poisson_seed for metric in ("decoder", "full", "eq")]
    )
    md += ["", "## Burgers-2D initial architecture screen (N=64, k=16, M=64)", ""]
    md += markdown_table(
        ["cell", "seed", "architecture", "width×layers", "group", "frequencies", "parameters", "train", "decoder", "full", "EQ256", "EQ512"],
        ["---", "---:", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        [[r["cell"], r["seed"], r["architecture"], f'{r["hidden"]}×{r["layers"]}',
          r["group_size"], r["n_freq"], f'{r["n_params"]:,}', sci(r["train"]),
          sci(r["decoder"]), sci(r["full"]), sci(r["eq256"]), sci(r["eq512"])]
         for r in burgers]
    )
    md += ["", "## Burgers weak-objective refinement", ""]
    md += markdown_table(
        ["cell", "seed", "trust", "M,m", "decoder", "full mean", "full med", "full max", "EQ mean", "EQ med", "EQ max", "EQ/full", "blowups"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        [[r["cell"], r["seed"], f'{r["trust_factor"]:.3g}', f'{r["M"]},{r["m"]}',
          sci(r["decoder"]), sci(r["full"]), sci(r["full_median"]), sci(r["full_max"]),
          sci(r["eq"]), sci(r["eq_median"]), sci(r["eq_max"]), f'{r["eq"]/r["full"]:.3f}',
          f'{r["full_blowups"]}/{r["eq_blowups"]}'] for r in burgers_objective]
    )
    md += ["", "## Accepted same-GPU decoder benchmarks", ""]
    for bench in benchmarks:
        control = bench["models"]["control"]["n_params"]
        variant = bench["models"]["variant"]["n_params"]
        md += [f'### {bench["pde"].title()}: {bench["cell"]}', "",
               f'f64/highest, {bench["reps"]} persisted repetitions, {bench["warm"]} exact-kernel warmups after a {bench["burn_seconds"]:.1f} s burn. Parameters: {control:,} → {variant:,}.', ""]
        md += markdown_table(
            ["kernel", "points", "control ms", "variant ms", "raw speedup", "cached variant ms", "cached speedup", "outliers c/v/cache"],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            [[r["kernel"], r["points"], f'{r["control_ms"]:.4f}', f'{r["variant_ms"]:.4f}',
              f'{r["speedup"]:.3f}', "—" if r["cached_variant_ms"] is None else f'{r["cached_variant_ms"]:.4f}',
             "—" if r["cached_speedup"] is None else f'{r["cached_speedup"]:.3f}',
             f'{r["outliers_control"]}/{r["outliers_variant"]}/{r["outliers_cached"]}']
             for r in bench["rows"]]
        )
        md.append("")
    md += ["## Same-GPU end-to-end rows", ""]
    md += markdown_table(
        ["cell", "arm", "M,m", "tau", "e2e ms", "solve ms", "error mean", "error med", "error max", "Jac evals", "censored"],
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        [[r["cell"], r["arm"], f'{r["M"]},{r["m"]}', sci(r["tau"]),
          f'{r["time_ms"]:.3f}', f'{r["solve_ms"]:.3f}', sci(r["error"]),
          sci(r["error_median"]), sci(r["error_max"]), f'{r["jac_evals"]:.2f}',
          f'{100*r["censored_frac"]:.1f}%'] for r in e2e_rows]
    )
    md += ["", "## Excluded measurements", "",
           "- `nda_pbench_g98_r8`: rejected because the exact kernels were warmed before, rather than after, the GPU burn. Its artifacts are retained for audit; `nda_pbench_g98b_r8` is the accepted rerun.",
           "- `nda_be2e_g160_r12`: failed after the control arm because the cost driver dropped the compact checkpoint's decoder metadata. The failure is retained as `nda_be2e_g160_r12_failed`; `nda_be2e_g160_r14` is the corrected rerun.", ""]
    with open(os.path.join(HERE, "SUMMARY.md"), "w") as fp:
        fp.write("\n".join(md))
    print(f"wrote {os.path.join(HERE, 'summary.json')} and SUMMARY.md")


if __name__ == "__main__":
    main()
