"""Rebuild source-linked development and frozen fresh confirmation figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 180, "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "jcp-cost-confirm"})
COLORS = {"Head": "#1864ab", "Span": "#862e9c", "POD": "#d97706",
          "Coarse classical": "#00866a", "Same-grid classical": "#5c6770"}
MARKERS = {"Head": "*", "Span": "D", "POD": "o", "Coarse classical": "^", "Same-grid classical": "s"}
TARGET_LABELS = {"learned_head_m292_dt0.02": "Head: M=292, Δt=0.02",
    "learned_head_m584_dt0.02": "Head: M=584, Δt=0.02",
    "learned_head": "Head: M=584, Δt=0.01", "learned_span64": "Span: R=64, M=584, Δt=0.01"}
GROUP_LABELS = {"pod": "POD", "same_mesh_classical": "Same-grid classical", "any_mesh_classical": "Coarse classical"}


def save(fig, out, stem):
    for ext in ["png", "svg", "pdf"]:
        kwargs = {"metadata": {"Date": None}} if ext == "svg" else ({"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {})
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def group(name, row):
    if name.startswith("learned_head"):
        return "Head"
    if name.startswith("learned_span"):
        return "Span"
    if row["family"] == "pod":
        return "POD"
    return "Same-grid classical" if row["mesh"] == 64 else "Coarse classical"


def frontier(points, metric):
    """A marginal observed envelope; never used as accuracy-eligibility logic."""
    best = float("inf")
    chosen = []
    for p in sorted(points, key=lambda p: (p["median_case_ms"], p[metric])):
        if p[metric] < best:
            chosen.append(p)
            best = p[metric]
    return chosen


def development(selection, confirmed, out, data):
    points = [{"name": name, "group": group(name, r), **r} for name, r in selection.items() if r["complete"]]
    excluded = [{"name": name, "reason": "incomplete selection invocations", **r} for name, r in selection.items() if not r["complete"]]
    data["development_selection"] = {"points": points, "incomplete": excluded,
        "source": "evidence/cost-p04-selection.json", "repetitions_per_pair": 3,
        "zoom_max_percent": 10, "scope": "selection-only; exposed development cohort"}
    for full in [False, True]:
        fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.25), constrained_layout=True)
        for ax, metric, title in zip(axs, ["median_case_evolved_max_error", "worst_evolved_error"], ["Cohort median error", "Worst case error"]):
            offscale = 0
            for g, color in COLORS.items():
                pp = [p for p in points if p["group"] == g]
                x, y = np.array([p["median_case_ms"] for p in pp]), 100 * np.array([p[metric] for p in pp])
                if not full:
                    offscale += int(np.sum(y > 10))
                shown = np.ones(len(y), dtype=bool) if full else y <= 10
                ax.scatter(x[shown], y[shown], color=color, marker=MARKERS[g],
                    s=82 if g == "Head" else 32, alpha=.95 if g in ["Head", "Span"] else .5, label=g, zorder=4)
                env = frontier(pp, metric)
                xx, yy = np.array([p["median_case_ms"] for p in env]), 100*np.array([p[metric] for p in env])
                mask = np.ones(len(yy), dtype=bool) if full else yy <= 10
                if len(xx):
                    ax.plot(xx[mask], yy[mask], color=color, lw=1, alpha=.7, zorder=2)
                if not full and np.any(~shown):
                    ax.scatter(x[~shown], np.full(np.sum(~shown), 9.2), marker="^", s=32,
                        facecolors="none", edgecolors=color, zorder=5)
            ax.set_yscale("log")
            ax.set_xlabel("Complete query (ms; median of case medians)")
            ax.set_ylabel("Maximum evolved field error (%)")
            ax.set_title(title)
            ax.grid(axis="both", color="#d9e0e6", alpha=.65, lw=.6)
            if not full:
                ax.set_ylim(.005, 10)
                ax.text(.98, .97, f"{offscale} finite settings above 10%\nshown at upper edge", ha="right", va="top", transform=ax.transAxes, fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=9)
        fig.suptitle("Development selection only: observed accuracy–cost choices" + (" — full finite range" if full else ""), fontsize=12)
        save(fig, out, "accuracy-cost-selection-full-range" if full else "accuracy-cost-selection")
    ratios = {}
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 7.1), sharex=True, constrained_layout=True)
    for ax, (target, groups) in zip(axs.flat, confirmed["comparisons"].items()):
        ratios[target] = {}
        for offset, (key, comp) in enumerate(groups.items()):
            g, name = GROUP_LABELS[key], comp["comparator"]
            a, b = confirmed["confirmation"][target]["cases"], confirmed["confirmation"][name]["cases"]
            ids = sorted(set(a) & set(b), key=int)
            yy = [b[i]["median_ms"] / a[i]["median_ms"] for i in ids]
            ratios[target][key] = {"comparator": name, "case_ids": list(map(int, ids)), "ratios": yy,
                                  "median_paired_ratio": float(np.median(yy)), "qualified": comp["qualified"]}
            ax.scatter(np.arange(len(ids)) + (offset-1)*.11, yy, s=20, color=COLORS[g],
                marker=MARKERS[g], label=f"{g}: paired median {np.median(yy):.3f}")
        ax.axhline(1, color="#444444", ls="--", lw=1)
        ax.set_title(TARGET_LABELS[target])
        ax.set_ylabel("Comparator / ROM query time")
        ax.grid(axis="y", color="#e0e5e9", lw=.6)
        ax.set_xlabel("Development case index")
        ax.set_xticks([0, 5, 10, 15])
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + .27 * (hi - lo))
        ax.legend(fontsize=7.7, loc="upper right")
    fig.suptitle("Separate timing confirmation on development cases · above 1 favors the ROM", fontsize=12)
    save(fig, out, "accuracy-cost-confirmed-case-ratios")
    data["development_confirmation_ratios"] = ratios
    rep_data = []
    labels = []
    for name, row in confirmed["confirmation"].items():
        rep_data.append([100*t/c["median_ms"] for c in row["cases"].values() for t in c["repetitions_ms"]])
        labels.append(name.replace("learned_", "").replace("pod_head_", "POD ").replace("fom_", "FOM "))
    fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
    ax.boxplot(rep_data, tick_labels=labels, showfliers=True, flierprops={"markersize": 2}, medianprops={"color":"#1864ab"})
    ax.axhline(100, color="#555", ls="--", lw=.8)
    ax.set_ylabel("Repetition time / its case median (%)")
    ax.set_title("Timing repetition spread · five repetitions per case, sixteen development cases")
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.grid(axis="y", color="#e0e5e9", lw=.6)
    save(fig, out, "accuracy-cost-confirmation-repetition-spread")
    data["development_repetition_percent_of_case_median"] = dict(zip(confirmed["confirmation"], rep_data))
    return points, excluded


def fresh(result, cfg, out, data):
    targets = list(cfg["frozen_comparators"])
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    data["fresh_confirmation"] = result
    for row, target in enumerate(targets):
        ax, rat = axs[row]
        names = [target] + list(dict.fromkeys(cfg["frozen_comparators"][target].values()))
        threshold = cfg["frozen_thresholds"][target]
        for name in names:
            r = result["confirmation"][name]
            g = group(name, r)
            label = "Head" if name == target else g
            ax.scatter(r["median_case_ms"], 100*r["median_case_evolved_max_error"], c=COLORS[g], marker=MARKERS[g], s=72, label=label+" median")
            ax.scatter(r["median_case_ms"], 100*r["worst_evolved_error"], facecolors="none", edgecolors=COLORS[g], marker=MARKERS[g], s=72)
            ax.plot([r["median_case_ms"]]*2, 100*np.array([r["median_case_evolved_max_error"],r["worst_evolved_error"]]), color=COLORS[g], alpha=.5)
        ax.axhline(100*threshold["median_case_evolved_max_error"], color="#777", ls="--", label="Frozen median limit")
        ax.axhline(100*threshold["worst_evolved_error"], color="#777", ls=":", label="Frozen worst limit")
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Fresh complete query (ms; median of case medians)")
        ax.set_ylabel("Max evolved field error (%)\nfilled: median; hollow: worst")
        ax.legend(fontsize=7.4, ncol=2)
        for offset, (key, comparison) in enumerate(result["comparisons"][target].items()):
            g=GROUP_LABELS[key]
            qualification = "fixed " + ("pass" if comparison["qualified_fixed_threshold"] else "fail") + ", match " + ("pass" if comparison["qualified_fresh_head_match"] else "fail")
            rat.scatter(np.array(comparison["paired_case_ids"])+(offset-1)*.13, comparison["paired_case_ratios"], s=24,
                color=COLORS[g], marker=MARKERS[g], label=f"{g}: {np.median(comparison['paired_case_ratios']):.3f} ({qualification})")
        rat.axhline(1, color="#444", ls="--", lw=1)
        rat.set_xlabel("Fresh case index")
        rat.set_ylabel("Comparator / head query time")
        lo, hi = rat.get_ylim()
        rat.set_ylim(lo, hi + .32 * (hi - lo))
        rat.legend(fontsize=7.2, loc="upper right")
        for a in [ax,rat]:
            a.grid(axis="y", color="#e0e5e9", lw=.6)
    fig.suptitle("Fresh physical confirmation · frozen models, comparators and accuracy limits", fontsize=12)
    save(fig,out,"accuracy-cost-fresh-confirmation")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fresh",type=Path)
    parser.add_argument("--out",type=Path,default=HERE/"plots")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    paths=[HERE/"evidence/cost-p04-selection.json",HERE/"evidence/cost-p04.json",HERE/"configs/fresh16.json"]
    sel,conf,cfg=[json.loads(p.read_text()) for p in paths]
    data={"schema":"jcp-cost-plotted-data-v1","error_contract":"max over five evolved outputs, initial-field norm; initial separate", "ratio_direction":"comparator/head; above1 favors head","sources":{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    points,excluded=development(sel,conf,args.out,data)
    if args.fresh:
        res=json.loads(args.fresh.read_text());data['sources'][str(args.fresh)]=hashlib.sha256(args.fresh.read_bytes()).hexdigest()
        fresh(res,cfg,args.out,data)
    (args.out/"plotted-data.json").write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    captions=f'''# Accuracy–cost figure captions

Final saved development measurements are replotted here; a fresh panel is included only when its result file is supplied. Every displayed value is generated from the linked JSON sources.

**Selection frontier.** Complete-query time versus cohort median and worst casewise maximum evolved field error for {len(sel)} tested configurations on {len(next(iter(sel.values()))['case_ids'])} exposed development cases. Only the three candidate-selection repetitions are used. Lines join each family's marginal observed time/error envelope; these are not a joint median-and-worst eligibility criterion or an interpolated predictor. Filled stars mark learned heads. The primary view shows errors up to ten percent; finite settings above that limit are shown at the upper edge and retain their exact values in the full-range companion and plotted-data file. The {len(excluded)} incomplete configurations are listed below instead of plotted as successful points. No candidate is omitted from the evidence.

**Confirmed case ratios.** The separately measured five-repetition confirmation on the same development cases, after comparator selection. Each point is the ratio of comparator to ROM case-median complete-query time. The dashed line at one is a timing tie. Legend statistics are medians of paired case ratios, not ratios of cohort-median times. Eligibility uses both cohort median and worst error; it does not guarantee pairwise error matching. The near POD tie and coarse classical losses remain visible.

**Repetition spread.** All confirmation timing repetitions normalized by their own method–case median. This displays measurement variation; it is neither case-generalization uncertainty nor trained-model seed uncertainty. Tukey outliers remain displayed and retained.

**Fresh confirmation.** When present, the new physical cohort is plotted separately, using only its own same-allocation timing measurements. Filled/hollow markers show median/worst errors, with unchanged development limits as dashed/dotted lines. Every paired ratio is shown even if the fixed accuracy or numerical/timing qualification fails; the data explicitly contain fixed-limit and fresh-head-matching decisions separately. This is frozen-choice validation, not a new accuracy–cost selection frontier.

## Incomplete selection configurations

| Configuration | Failed invocations | Total invocations |
|---|---:|---:|
'''
    for r in excluded:captions+=f"| `{r['name']}` | {r['failure_count']} | {r['invocation_count']} |\n"
    captions+='''
## Glossary

**Complete query:** all work from the declared host initial field to returned host fields, including native initialization and diagnostics conversion. **Head:** nonlinear coefficient map inside a frozen bank. **Span:** arbitrary linear combinations of the bank. **POD:** a snapshot-derived spatial basis; POD-head settings use a frozen coefficient map. **Coarse/same-grid:** a classical solver using fewer/the same grid points as the returned field. **Median/worst error:** middle/largest casewise maximum evolved error. **Selection:** the development comparison used to choose settings. **Confirmation:** repetitions or fresh cases evaluated after choices were fixed, identified separately. **Paired ratio:** comparator time divided by ROM time for the same case. **Marginal envelope:** observed best time–error choices for one plotted error statistic. **Tukey outlier:** a sample more than one and a half interquartile ranges beyond the lower or upper quartile.
'''
    (args.out/"captions.md").write_text(captions)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}
    (args.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(args.out),'figures':[p.name for p in args.out.glob('*.png')],'complete_selection_configs':len(points),'incomplete_selection_configs':len(excluded)}))


if __name__=='__main__':main()
