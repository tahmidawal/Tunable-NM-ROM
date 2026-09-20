"""Export explicit, audited 3D panels as manuscript tables and vector figures.

Consumes coordinator-normalized invocation records; never reads training data,
selects a best run, or compares runtime across allocations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['svg.hashsalt'] = 'nmrom-3d-paper-panels'

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SELECTION = REPORTS / "2026-09-20-3d-paper-panel-selection.json"
RESULTS = REPORTS / "2026-09-20-3d-paper-results.json"
OUT = REPORTS / "2026-09-20-3d-paper-panels"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def percent(value):
    return "—" if value is None else f"{100 * value:.4f}"


def series_style(series):
    if series.startswith("NM-ROM"):
        return "#b63232", "o"
    return {
        "Full learned bank": ("#ce7d18", "s"),
        "POD": ("#387c49", "D"),
        "FOM": ("#454545", "P"),
        "FNO": ("#3369a5", "^"),
        "U-Net": ("#7454a6", "v"),
        "DeepONet": ("#25878a", "<"),
        "Transolver": ("#ab5b88", ">"),
    }[series]


def plot(panel, rows):
    fig, (cost, ladder) = plt.subplots(1, 2, figsize=(10.0, 4.2), layout="constrained")
    used = set()
    zero_errors = []
    for row in rows:
        color, marker = series_style(row["series"])
        value = 100 * row["evolved_worst"]
        # Zero must never be clipped to an invented positive logarithmic value.
        if value == 0:
            zero_errors.append(row)
            continue
        key = row["label"] if row["series"] in {"FNO", "U-Net", "DeepONet", "Transolver"} else row["series"]
        label = key if key not in used else None
        used.add(key)
        cost.scatter(row["gpu_ms_median"], value, facecolors="none" if "direct transfer" in row["label"] else color,
                     edgecolors=color, marker=marker, s=46,
                     label=label, zorder=3)
        if row["nonstationary_cases"] or row["nonfinite_cases"]:
            cost.scatter(row["gpu_ms_median"], value, c="black", marker="x", s=80, zorder=4)
    for series in sorted({r["series"] for r in rows if r["series"].startswith("NM-ROM")}):
        group = sorted((r for r in rows if r["series"] == series), key=lambda r: r["rank"])
        color, _ = series_style(series)
        cost.plot([r["gpu_ms_median"] for r in group], [100*r["evolved_worst"] for r in group],
                  color=color, alpha=.6, lw=1)
        ladder.plot([r["rank"] for r in group], [100*r["evolved_worst"] for r in group],
                    color=color, marker="o", label="Worst case")
        ladder.plot([r["rank"] for r in group], [100*r["evolved_median"] for r in group],
                    color=color, marker="o", ls="--", alpha=.7, label="Median case")
    cost.set(xscale="log", yscale="log", xlabel="Median device query time (ms)",
             ylabel="Worst same-grid error (%)")
    ladder.set(xlabel="Correction rank q", ylabel="Same-grid error (%)", yscale="log")
    for row in zero_errors:
        cost.plot([], [], color="#454545", marker="P", ls="none",
                  label=f"{row['label']}: 0% at {row['gpu_ms_median']:.3f} ms (not plotted)")
    if any(r['nonstationary_cases'] or r['nonfinite_cases'] for r in rows):
        cost.plot([], [], color="black", marker="x", ls="none", label="A numerical check failed")
    for axis in (cost, ladder):
        axis.grid(alpha=.2, which="both")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{panel['pde']} — {panel['mesh_label']} — development", fontsize=11)
    for extension in ("pdf", "svg", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else ({"Date": None} if extension == "svg" else None)
        fig.savefig(OUT / f"{panel['slug']}.{extension}", dpi=180, metadata=metadata)
    plt.close(fig)


def main():
    selection = json.loads(SELECTION.read_text())
    results = json.loads(RESULTS.read_text())
    OUT.mkdir(exist_ok=True)
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in (SELECTION, RESULTS, Path(__file__))}
    exported = []
    report = ["# Three-dimensional comparison panels for the manuscript", "",
              "These generated tables and vector figures are provisional development material. "
              "They show the explicitly selected audited comparisons; independent final evaluation remains pending.", "",
              selection["selection_rule"], ""]
    for panel in selection["panels"]:
        matches = [r for r in results["runs"] if (r["pde"], r["attempt"]) == (panel["pde"], panel["attempt"])]
        assert len(matches) == 1
        run = matches[0]
        assert run["audit"]["passed"]
        available = {r["method"]: r for r in run["rows"] if r.get("mesh") == panel["mesh"]}
        rows = []
        for choice in panel["methods"]:
            row = dict(available[choice["id"]], **choice)
            assert row["invocations"] == len(row["gpu_ms_repetitions"])
            assert row["gpu_ms_median"] == statistics.median(row["gpu_ms_repetitions"])
            row["gpu_ms_p95"] = float(np.quantile(row["gpu_ms_repetitions"], .95))
            rows.append(row)
        exported.append(dict(panel=panel, source=run["source"], job_id=run["job_id"], gpu=run["gpu"], rows=rows))
        cases = sorted({row["cases"] for row in rows})
        assert len(cases) == 1, "A comparison panel must use the same case cohort."
        case_note = f"Each method is evaluated on {cases[0]} distinct cases; all timed repetitions are retained. "
        stem = OUT / panel["slug"]
        with stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["method", "median_error_percent", "worst_error_percent", "all_times_worst_percent",
                             "initial_worst_percent", "physical_worst_percent", "gpu_ms_median", "gpu_ms_p95",
                             "nonfinite_cases", "nonstationary_cases", "timing_outliers", "invocations", "status"])
            for row in rows:
                writer.writerow([row["label"], *[percent(row[key]) for key in
                    ("evolved_median", "evolved_worst", "all_times_worst", "initial_worst", "physical_worst")],
                    row["gpu_ms_median"], row["gpu_ms_p95"], row["nonfinite_cases"], row["nonstationary_cases"],
                    row["timing_outliers"], row["invocations"], row["qualification"]])
        caption = (f"{panel['pde']}, {panel['mesh_label']}; provisional development comparison. "
                   + case_note + "Errors are same-grid relative errors. NF/NS counts nonfinite/nonstationary cases; "
                   "outliers are calls above 1.5 times the method median. " + panel["qualification"])
        latex = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabular}{lrrrrrr}", r"\toprule",
                 r"Method & Median (\%) & Worst (\%) & GPU ms & p95 ms & NF/NS & Outliers \\", r"\midrule"]
        for row in rows:
            latex.append(tex(row["label"]) + " & " + " & ".join([
                percent(row["evolved_median"]), percent(row["evolved_worst"]),
                f"{row['gpu_ms_median']:.3f}", f"{row['gpu_ms_p95']:.3f}",
                f"{row['nonfinite_cases']}/{row['nonstationary_cases']}",
                f"{row['timing_outliers']}/{row['invocations']}"]) + r" \\")
        latex += [r"\bottomrule", r"\end{tabular}", r"\caption{" + tex(caption) + "}",
                  r"\label{tab:" + panel["slug"] + "}", r"\end{table}"]
        stem.with_suffix(".tex").write_text("\n".join(latex) + "\n")
        plot(panel, rows)
        report += [f"## {panel['pde']} — {panel['attempt']} — {panel['mesh_label']}", "",
                   "**Provisional:** " + panel["qualification"], "", case_note, "",
                   f"Source `{run['source']}`; job `{run['job_id']}`; GPU `{run['gpu']}`. "
                   f"[LaTeX table]({panel['slug']}.tex), [CSV]({panel['slug']}.csv), "
                   f"[vector figure]({panel['slug']}.pdf).", "",
                   "| Method | Median error (%) | Worst error (%) | GPU median (ms) | GPU p95 (ms) | NF / NS cases | Outliers / calls |",
                   "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
        for row in rows:
            report.append(f"| {row['label']} | {percent(row['evolved_median'])} | {percent(row['evolved_worst'])} | "
                          f"{row['gpu_ms_median']:.3f} | {row['gpu_ms_p95']:.3f} | "
                          f"{row['nonfinite_cases']} / {row['nonstationary_cases']} | "
                          f"{row['timing_outliers']} / {row['invocations']} |")
        report += ["", panel["omissions"], "", f"![Generated paired comparison]({panel['slug']}.png)", ""]
    report += ["## Glossary", "",
        "- **Development / provisional:** data used during model selection / evidence awaiting final confirmation.",
        "- **NM-ROM, K, q:** neural-manifold reduced model, latent dimension, and added correction rank.",
        "- **Bank / full bank:** learned spatial functions / a model with all their coefficients free.",
        "- **POD / Galerkin / weak:** a linear basis learned from snapshots / projection against basis functions / a residual projected against smooth test functions.",
        "- **FOM / DST:** a full-grid solver / a direct solver using discrete sine transforms.",
        "- **FNO / U-Net / DeepONet / Transolver:** Fourier, convolutional, branch–trunk and transformer solution-map models.",
        "- **Direct transfer / native plus interpolation:** evaluating frozen model weights on a changed grid / evaluating on the training grid and interpolating the prediction, with interpolation cost included.",
        "- **Median / worst error:** median or maximum across cases of each case's largest error over the requested evolved times; Poisson has one stationary output. Norms are defined in the complete audited report. The worst repeated invocation is retained.",
        "- **GPU median / p95:** median / empirical 95th percentile of all retained device query times. The latter is a descriptive tail statistic, not a confidence interval.",
        "- **NF / NS cases:** cases with at least one nonfinite output / failed iterative stopping check. Direct models require no iterative stopping test.",
        "- **Outliers / calls:** timings above one and a half times that method's median / all timed repetitions, including those outliers.",
        "- **Source / job / GPU:** pinned code revision / allocation identifier / graphics processor. Cost comparisons stay within an allocation.",
        "- **All-times / initial / physical errors (CSV):** errors including time zero / initial compression / comparison against a refined numerical reference. A blank cell is unmeasured, not zero.", ""]
    (OUT / "README.md").write_text("\n".join(report))
    (OUT / "panels.json").write_text(json.dumps(dict(status=selection["status"], provenance=provenance, panels=exported), indent=2)+"\n")
    document = [r"\documentclass[10pt]{article}", r"\usepackage[margin=0.65in]{geometry}",
                r"\usepackage{booktabs,graphicx,float}", r"\usepackage[T1]{fontenc}",
                r"\begin{document}", r"\noindent\textbf{3D comparison panels --- provisional development results}",
                r"\par All values are generated from retained, audited invocation records. Final evaluation remains pending."]
    for item in exported:
        slug = item['panel']['slug']
        document += [r"\input{" + slug + ".tex}", r"\begin{figure}[H]\centering",
                     r"\includegraphics[width=\textwidth]{" + slug + ".pdf}", r"\end{figure}", r"\clearpage"]
    document += [r"\end{document}"]
    (OUT / "all-panels.tex").write_text("\n".join(document)+"\n")
    print(f"Exported {len(exported)} explicit panels, {sum(len(p['rows']) for p in exported)} rows.")


if __name__ == "__main__":
    main()
