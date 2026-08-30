"""Figures for the 2D tensor-ROM results (2026-08-30).  Reads the figure dumps
written by `sep_b2d_tensor.py` with FIGS_DUMP (runs/b2dtensor/figs/figdata_n*.npz:
decoded fields of the selected test trajectories at t = 0/10/25/50, strided to
<= 256^2, pointwise error maps, cross-sections through the blob centre and
per-step rel-L2 curves; committed checkpoints, test seed 1, no retraining) and
writes PNGs (150 dpi) + figs.json under runs/b2dtensor/figs/.

    /home/tahmid/Dev/.venv/bin/python runs/b2dtensor/make_figs.py

Per N and trajectory: (1) fields truth | tensor | full-grid oracle | NNLS-256 at
the four times, shared colour scale per time (viridis); (2) |ROM - truth| for
tensor and NNLS-256 on a shared single-hue scale per time, plus |tensor - oracle|
on its own scale; (3) per-step rel-L2 curves of all arms; (4) cross-sections
through the blob centre at t = 25 and 50.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
ARM_LABEL = {"full": "full-grid oracle ROM", "tensor": "tensor ROM",
             "ex": "NNLS-256 ROM", "ex_learned": "learned-64 ROM"}
ARM_COLOR = {"full": "#000000", "tensor": "#D55E00", "ex": "#0072B2", "ex_learned": "#009E73"}


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def main():
    out = {}
    for p in sorted(glob.glob(os.path.join(FIGS, "figdata_n*.npz"))):
        N = int(re.search(r"figdata_n(\d+)\.npz$", p).group(1))
        d = np.load(p, allow_pickle=False)
        times = [int(t) for t in d["times"]]
        sel = [int(i) for i in d["sel"]]
        arms = [str(a) for a in d["arms"]]
        stride = int(d["stride"])
        x = np.arange(0, N, stride) / (N - 1)
        xline = np.arange(N) / (N - 1)
        for i in sel:
            nu = float(d[f"nu_{i}"])
            cx, cy, w, a = [float(v) for v in d[f"center_{i}"]]
            jc = int(d[f"line_j_{i}"])
            truth = d[f"truth_{i}"]
            curves = {arm: d[f"{arm}_curve_{i}"] for arm in arms}
            err_mean = {arm: float(np.mean(curves[arm])) for arm in arms}
            tag = f"n{N}_traj{i}"
            title = (f"N={N}, test traj {i}, nu={nu:.4f}, blob (cx,cy,w,a)=({cx:.2f},{cy:.2f},{w:.3f},{a:.2f}); "
                     f"mean rel-L2: tensor {err_mean['tensor']:.3e}, oracle {err_mean['full']:.3e}, "
                     f"NNLS-256 {err_mean['ex']:.3e}")
            rec = dict(N=N, traj=i, nu=nu, center=[cx, cy], width=w, amplitude=a, times=times,
                       stride=stride, err_mean=err_mean, per_time={}, figures={})

            # (1) fields ----------------------------------------------------
            cols = ["truth", "tensor", "full", "ex"]
            fig, axes = plt.subplots(len(times), len(cols), figsize=(3.1 * len(cols), 3.0 * len(times)),
                                     constrained_layout=True)
            for r_, t in enumerate(times):
                F = {"truth": truth[r_]}
                F.update({arm: d[f"{arm}_{i}"][r_] for arm in arms})
                vmax = float(max(np.max(np.abs(F[c])) for c in cols))
                for c_, c in enumerate(cols):
                    ax = axes[r_, c_]
                    im = ax.imshow(F[c].T, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax,
                                   extent=[0, 1, 0, 1])
                    e_ = "" if c == "truth" else f"  rel-L2 {rel(d[f'{c}_{i}'][r_], truth[r_]):.2e}"
                    ax.set_title(f"t={t}: {'FOM truth' if c == 'truth' else ARM_LABEL[c]}{e_}", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=axes[r_, :].tolist(), shrink=0.8, pad=0.01)
                rec["per_time"][str(t)] = {c: rel(d[f"{c}_{i}"][r_], truth[r_]) for c in arms}
                rec["per_time"][str(t)]["per_step_curve_tensor"] = float(curves["tensor"][t])
                rec["per_time"][str(t)]["per_step_curve_full"] = float(curves["full"][t])
                rec["per_time"][str(t)]["per_step_curve_ex"] = float(curves["ex"][t])
                rec["per_time"][str(t)]["max_abs_err_tensor"] = float(d[f"tensor_errmax_{i}"][r_])
                rec["per_time"][str(t)]["max_abs_err_ex"] = float(d[f"ex_errmax_{i}"][r_])
                rec["per_time"][str(t)]["max_abs_tensor_minus_full"] = float(d[f"tensor_minus_full_max_{i}"][r_])
            fig.suptitle("Fields — " + title.replace("; mean", ";\nmean"), fontsize=9)
            f1 = os.path.join(FIGS, f"fields_{tag}.png")
            fig.savefig(f1, dpi=150); plt.close(fig)
            rec["figures"]["fields"] = os.path.relpath(f1, HERE)

            # (2) error maps -------------------------------------------------
            fig, axes = plt.subplots(len(times), 3, figsize=(3.4 * 3, 3.0 * len(times)),
                                     constrained_layout=True)
            for r_, t in enumerate(times):
                et, ee = d[f"tensor_err_{i}"][r_], d[f"ex_err_{i}"][r_]
                dtf = d[f"tensor_minus_full_{i}"][r_]
                vmax = float(max(et.max(), ee.max(), 1e-300))
                for c_, (E, lab) in enumerate(((et, "|tensor - truth|"), (ee, "|NNLS-256 - truth|"))):
                    ax = axes[r_, c_]
                    im = ax.imshow(E.T, origin="lower", cmap="Blues", vmin=0.0, vmax=vmax, extent=[0, 1, 0, 1])
                    ax.set_title(f"t={t}: {lab}  max {E.max():.2e}", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=axes[r_, :2].tolist(), shrink=0.8, pad=0.01)
                ax = axes[r_, 2]
                im2 = ax.imshow(dtf.T, origin="lower", cmap="Oranges", vmin=0.0,
                                vmax=float(max(dtf.max(), 1e-12)), extent=[0, 1, 0, 1])
                ax.set_title(f"t={t}: |tensor - oracle|  max {dtf.max():.1e} (own scale)", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im2, ax=[ax], shrink=0.8, pad=0.01)
            fig.suptitle("Pointwise errors — " + title.replace("; mean", ";\nmean"), fontsize=9)
            f2 = os.path.join(FIGS, f"errors_{tag}.png")
            fig.savefig(f2, dpi=150); plt.close(fig)
            rec["figures"]["errors"] = os.path.relpath(f2, HERE)

            # (3) per-step curves ------------------------------------------
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            for arm in ("ex", "full", "tensor"):
                if arm in curves:
                    ax.semilogy(np.arange(len(curves[arm])), curves[arm], color=ARM_COLOR[arm],
                                lw=2.2 if arm == "full" else 1.4, ls="--" if arm == "tensor" else "-",
                                label=f"{ARM_LABEL[arm]} (mean {err_mean[arm]:.3e})")
            dif = np.abs(curves["tensor"] - curves["full"])
            ax.semilogy(np.arange(len(dif)), np.maximum(dif, 1e-12), color="#CC79A7", lw=1.0,
                        label=f"|tensor - oracle| per-step error diff (max {dif.max():.1e}; "
                              f"floored at 1e-12, exactly 0 at t=0: same IC fit)")
            ax.set_ylim(1e-12, 2.0)
            ax.set_xlabel("time step"); ax.set_ylabel("rel-L2 vs FOM truth"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="center left")
            ax.set_title("Per-step error — " + title.replace("; mean", ";\nmean"), fontsize=8)
            f3 = os.path.join(FIGS, f"curves_{tag}.png")
            fig.savefig(f3, dpi=150); plt.close(fig)
            rec["figures"]["curves"] = os.path.relpath(f3, HERE)
            rec["curve_max_abs_diff_tensor_vs_full"] = float(dif.max())

            # (4) cross-sections -------------------------------------------
            sec_t = [t for t in (25, 50) if t in times]
            fig, axes = plt.subplots(1, len(sec_t), figsize=(6 * len(sec_t), 4), constrained_layout=True)
            axes = np.atleast_1d(axes)
            for ax, t in zip(axes, sec_t):
                r_ = times.index(t)
                ax.plot(xline, d[f"truth_line_{i}"][r_], color="k", lw=2.2, label="FOM truth")
                ax.plot(xline, d[f"ex_line_{i}"][r_], color=ARM_COLOR["ex"], lw=1.2, label="NNLS-256 ROM")
                ax.plot(xline, d[f"tensor_line_{i}"][r_], color=ARM_COLOR["tensor"], lw=1.2, ls="--",
                        label="tensor ROM")
                ax.axhline(0.0, color="gray", lw=0.6)
                ax.set_title(f"t={t}: cross-section y={jc/(N-1):.3f} (through blob centre)", fontsize=9)
                ax.set_xlabel("x"); ax.set_ylabel("u"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
                rec["per_time"][str(t)]["section_min_tensor"] = float(d[f"tensor_line_{i}"][r_].min())
                rec["per_time"][str(t)]["section_min_truth"] = float(d[f"truth_line_{i}"][r_].min())
            fig.suptitle("Cross-sections — " + title.replace("; mean", ";\nmean"), fontsize=8)
            f4 = os.path.join(FIGS, f"sections_{tag}.png")
            fig.savefig(f4, dpi=150); plt.close(fig)
            rec["figures"]["sections"] = os.path.relpath(f4, HERE)
            out[tag] = rec
            print(f"{tag}: nu {nu:.4f} tensor {err_mean['tensor']:.3e} oracle {err_mean['full']:.3e} "
                  f"NNLS {err_mean['ex']:.3e}; max|tensor-oracle| per-step {dif.max():.1e}")
    json.dump(out, open(os.path.join(FIGS, "figs.json"), "w"), indent=1)
    print(f"wrote {os.path.join(FIGS, 'figs.json')} ({len(out)} entries)")
    write_md(out)


def write_md(out):
    """FIGURES.md: the notes' Figures section, captions generated from figs.json
    (image paths relative to experiments/separable-decoder/)."""
    L = ["## Figures", "",
         "Generated by `runs/b2dtensor/make_figs.py` from `runs/b2dtensor/figs/figdata_n*.npz` "
         "(dumped by `sep_b2d_tensor.py` with `FIGS_DUMP` from the committed checkpoints, test seed 1, "
         "arms `full` / `ex` (NNLS-256) / `tensor`; N=64 and N=256 rollouts run locally on the GB10 through "
         "`jaxrun`, N=1024 as cluster job 3049945 on an A100-80GB; fields strided to <= 256^2 for the images, "
         "cross-sections and errors at full resolution).  Per N the two trajectories are the median-error one "
         "and the worst one of the 8 test trajectories (by the tensor arm's error in the main jobs).  Numbers in "
         "the captions come from `runs/b2dtensor/figs/figs.json`.", ""]
    for tag, r in out.items():
        N, i = r["N"], r["traj"]
        em = r["err_mean"]
        L.append(f"### N={N}, test trajectory {i} (nu={r['nu']:.4f}; mean rel-L2: tensor {em['tensor']:.3e}, "
                 f"oracle {em['full']:.3e}, NNLS-256 {em['ex']:.3e}; max per-step |tensor - oracle| error "
                 f"difference {r['curve_max_abs_diff_tensor_vs_full']:.1e})")
        L.append("")
        pt = r["per_time"]
        L.append("| t | tensor rel-L2 | oracle rel-L2 | NNLS-256 rel-L2 | max abs err tensor | max abs err NNLS-256 | "
                 "max abs |tensor - oracle| |")
        L.append("|---|---|---|---|---|---|---|")
        for t in r["times"]:
            q = pt[str(t)]
            L.append(f"| {t} | {q['tensor']:.3e} | {q['full']:.3e} | {q['ex']:.3e} | {q['max_abs_err_tensor']:.2e} | "
                     f"{q['max_abs_err_ex']:.2e} | {q['max_abs_tensor_minus_full']:.1e} |")
        L.append("")
        for key, cap in (("fields", "Fields at t = 0, 10, 25, 50: FOM truth | tensor ROM | full-grid oracle ROM | "
                                    "NNLS-256 ROM, shared colour scale per row (viridis)."),
                         ("errors", "Pointwise |ROM - truth| for tensor and NNLS-256 (shared single-hue scale per row) "
                                    "and |tensor - oracle| on its own, much smaller, scale."),
                         ("curves", "Per-step rel-L2 error of every arm over the 50 steps, with the per-step "
                                    "|tensor - oracle| error difference."),
                         ("sections", "Cross-section through the blob centre (constant y) at t = 25 and 50: truth vs "
                                      "tensor vs NNLS-256.")):
            L.append(f"![{key} {tag}]({r['figures'][key]})")
            L.append("")
            L.append(f"*{cap}*")
            L.append("")
    open(os.path.join(FIGS, "FIGURES.md"), "w").write("\n".join(L))
    print(f"wrote {os.path.join(FIGS, 'FIGURES.md')}")


if __name__ == "__main__":
    main()
