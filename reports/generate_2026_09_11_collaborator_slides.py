"""Build the collaborator Beamer deck from audited records, without new solves.

Run with the repository's absolute Python environment.  The generated TeX and
figure folder are portable; the ZIP can be compiled independently with latexmk.
"""

import hashlib
import json
from pathlib import Path
import re
from statistics import median
import subprocess
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
SOURCE = "2026-09-11-iterative-fom-multiresolution"
STEM = "2026-09-11-multiresolution-collaborator-slides"
ASSETS = OUT / (STEM + "-figures")
KEYS = ("poisson", "heat", "burgers", "wave_dirichlet")
INK, TEAL, BLUE, ORANGE = "#172B3A", "#087E8B", "#586B9C", "#D97929"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def render(template, **values):
    for key, value in values.items():
        template = template.replace("@@" + key + "@@", str(value))
    assert not re.search(r"@@\w+@@", template), template
    return template


def lookup(panel, n, method):
    return next(r for r in panel["rows"] if r["intervals"] == n and r["method"] == method)


def qualifies(panel, row):
    if "qualified" in row:
        return row["qualified"]
    return row["converged"] and row["worst_error"] + row.get("error_allowance", 0) <= panel["target"]


def fastest(panel, n):
    rows = [lookup(panel, n, method) for method in panel["iterative_foms"]]
    return min((r for r in rows if qualifies(panel, r)), key=lambda r: r["gpu_ms"], default=None)


def err(row):
    return f'{100 * row["worst_error"]:.4g}'


def frame(title, body):
    return r"\begin{frame}{" + title + "}\n" + body + "\n" + r"\end{frame}" + "\n"


def table(headers, rows, spec=None):
    spec = spec or "l" + "r" * (len(headers) - 1)
    return (r"\par\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + spec + "@{}}\n"
            + r"\toprule" + "\n" + " & ".join(headers) + r" \\ \midrule" + "\n"
            + "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
            + "\n" + r"\bottomrule\end{tabular*}\par" + "\n")


def primary_table(panel):
    rows = []
    for n in panel["intervals"]:
        rom = lookup(panel, n, panel["primary_rom"])
        fom = lookup(panel, n, panel["primary_fom"])
        ratio = fom["gpu_ms"] / rom["gpu_ms"]
        rows.append((n, f'{rom["gpu_ms"]:.2f}', f'{fom["gpu_ms"]:.2f}',
                     f'{ratio:.3f}' if ratio < 1 else f'{ratio:.2f}', err(rom), err(fom)))
    return table([r"$N$", "ROM ms", "FOM ms", "FOM/ROM", r"ROM error \%", r"FOM error \%"], rows)


def runtime_plot(panels):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": INK, "axes.labelcolor": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 5.25), layout="constrained")
    for ax, p in zip(axes.flat, panels):
        ns = p["intervals"]
        ax.plot(ns, [lookup(p, n, p["primary_rom"])["gpu_ms"] for n in ns],
                "o-", color=TEAL, lw=2.3, ms=5)
        ax.plot(ns, [lookup(p, n, p["primary_fom"])["gpu_ms"] for n in ns],
                "s--", color=BLUE, lw=1.8, ms=4)
        passing = [(n, fastest(p, n)) for n in ns]
        passing = [(n, r) for n, r in passing if r is not None]
        ax.plot([n for n, r in passing], [r["gpu_ms"] for n, r in passing],
                "D:", color=ORANGE, lw=2, ms=4)
        ax.set(xscale="log", yscale="log", xticks=ns, xlabel="Intervals per axis", ylabel="GPU query time (ms)")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.grid(True, color="#E5EAF0", lw=.7)
        ax.set_title(p["label"], loc="left", fontweight="bold", fontsize=11)
        if p["key"] == "burgers":
            ax.text(.02, .02, "No passing FOM at the smallest mesh", transform=ax.transAxes,
                    fontsize=8, color=INK)
    handles = [Line2D([], [], color=c, marker=m, linestyle=s, label=l) for c, m, s, l in
               [(TEAL, "o", "-", "Frozen NMROM"), (BLUE, "s", "--", "Tight iterative FOM"),
                (ORANGE, "D", ":", "Fastest tested passing iterative FOM")]]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=10)
    path = ASSETS / "runtime-scaling.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(ASSETS / "runtime-scaling.png", dpi=170)
    plt.close(fig)


PREAMBLE = r"""% Generated from audited JSON by generate_2026_09_11_collaborator_slides.py.
% Build: latexmk -pdf -interaction=nonstopmode -halt-on-error THIS_FILE.tex
\documentclass[10pt,aspectratio=169]{beamer}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,graphicx}
\definecolor{ink}{HTML}{172B3A}
\definecolor{teal}{HTML}{087E8B}
\definecolor{amber}{HTML}{A85317}
\definecolor{muted}{HTML}{5A6C7B}
\definecolor{pale}{HTML}{EEF5F6}
\definecolor{warm}{HTML}{FBF3E9}
\setbeamercolor{normal text}{fg=ink,bg=white}
\setbeamercolor{structure}{fg=teal}
\setbeamercolor{frametitle}{fg=ink}
\setbeamercolor{block title}{fg=teal,bg=pale}
\setbeamercolor{block body}{fg=ink,bg=pale}
\setbeamercolor{block title alerted}{fg=amber,bg=warm}
\setbeamercolor{block body alerted}{fg=ink,bg=warm}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}
\setbeamerfont{title}{size=\LARGE,series=\bfseries}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize items}[circle]
\setbeamersize{text margin left=8mm,text margin right=8mm}
\setbeamertemplate{frametitle}{\vspace{2mm}\insertframetitle\par\vspace{2mm}}
\setbeamertemplate{footline}{\hspace{8mm}\color{muted}\tiny
NM-ROM $\vert$ Audited development results $\vert$ September 2026\hfill
\insertframenumber/\inserttotalframenumber\hspace{8mm}\vspace{3mm}}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.2}
\newcommand{\smallnote}[1]{\par\vspace{1mm}{\scriptsize\color{muted}#1\par}}
\newcommand{\takeaway}[1]{\begin{block}{Takeaway}#1\end{block}}
\newcommand{\caution}[1]{\begin{alertblock}{Interpretation}#1\end{alertblock}}
\title{Separable NM-ROM\\Multiresolution results}
\subtitle{\texorpdfstring{Poisson 2D $\cdot$ Heat 2D $\cdot$ Burgers 2D $\cdot$ Reflective waves}{Poisson 2D; Heat 2D; Burgers 2D; Reflective waves}}
\author{Tunable NM-ROM project}
\date{11 September 2026}
\hypersetup{pdftitle={Separable NM-ROM: multiresolution results},pdfauthor={Tunable NM-ROM project}}
\begin{document}
"""


def main():
    source_path = OUT / (SOURCE + ".json")
    source = json.loads(source_path.read_text())
    manifest = json.loads((OUT / (SOURCE + ".manifest.json")).read_text())
    assert digest(source_path) == manifest["generated_sha256"][str(source_path.relative_to(ROOT))]
    assert source["status"] == "audited development results"
    panels = [next(p for p in source["panels"] if p["key"] == key) for key in KEYS]
    pmap = {p["key"]: p for p in panels}
    ns = panels[0]["intervals"]
    assert all(p["intervals"] == ns for p in panels)
    fine = max(ns)
    sources = {str(source_path.relative_to(ROOT)): digest(source_path)}
    for p in panels:
        for key in ("result_path", "audit_path"):
            path = ROOT / p[key]
            assert digest(path) == source["source_sha256"][p[key]]
            sources[p[key]] = digest(path)
    wave_path = ROOT / pmap["wave_dirichlet"]["result_path"]
    native = json.loads(wave_path.read_text())
    wave_rows = [r for r in native["invocations"] if r["boundary"] == "dirichlet"
                 and r["intervals"] == fine and r["method"] == pmap["wave_dirichlet"]["primary_rom"]]
    assert len(wave_rows) == pmap["wave_dirichlet"]["cases"] * pmap["wave_dirichlet"]["repetitions"]
    assert all(r["completed"] and r["comparison_eligible"] for r in wave_rows)
    for r in wave_rows:
        seconds = r["seconds"]
        assert abs(sum(v for k, v in seconds.items() if k != "complete_device_query") - seconds["complete_device_query"]) < 1e-12
    phases = {k: median(r["seconds"][k] for r in wave_rows) * 1000 for k in wave_rows[0]["seconds"]}
    evolution_share = median(100 * r["seconds"]["evolution"] / r["seconds"]["complete_device_query"] for r in wave_rows)
    ASSETS.mkdir(exist_ok=True)
    runtime_plot(panels)
    slides = []
    slides.append(frame("", r"""\titlepage
\vspace{-4mm}
\begin{center}\small\color{teal}Frozen trained weights across all tested resolutions\end{center}
\smallnote{Paired GPU measurements and independently audited fields. Development cohorts; independent final paper validation remains open.}
"""))

    overview = []
    interpretations = {"poisson": "Accuracy target missed", "heat": "Target and stopping pass",
                       "burgers": r"Accuracy passes; stalls$^*$", "wave_dirichlet": "Full-state target missed"}
    for p in panels:
        rom = lookup(p, fine, p["primary_rom"])
        f = fastest(p, fine)
        overview.append((esc(p["label"]), f'{f["gpu_ms"]/rom["gpu_ms"]:.2f}',
                         err(rom), interpretations[p["key"]]))
    slides.append(frame(f"At {fine} intervals: where the evidence stands",
        r"\small Fastest tested iterative FOM meeting the declared target, divided by ROM time.\par\vspace{3mm}" +
        table(["Problem", "FOM/ROM", r"ROM error \%", "ROM status"], overview, "lrrl") +
        r"\takeaway{Heat has the clearest qualified speed advantage. Burgers has a smaller fine-grid advantage under its existing stopping contract.}" +
        r"\smallnote{Ratio above one favors the ROM. These are named iterative controls, not the fastest possible FOMs. Errors are worst cases under the panel-specific norms.}" +
        r"\smallnote{$^*$Burgers accepts small-step/improvement stalls; stationarity was not measured. Reflective-wave error here is displacement only.}"))

    cohort = [(esc(p["label"]), p["cases"], p["repetitions"],
               {"poisson": "Current solution norm", "heat": "Current field norm",
                "burgers": "Fixed initial field norm", "wave_dirichlet": "Current component norms"}[p["key"]]) for p in panels]
    slides.append(frame("The measurement contract",
        render(r"""\small
\begin{itemize}
\item Meshes: $N=@@meshes@@$ intervals per axis. One frozen checkpoint per PDE/boundary.
\item Online GPU time includes input projection/fitting, the solve or rollout, and requested full fields.
\item Training, mesh assembly and compilation are offline. Host transfers are reported separately.
\item Each ROM/FOM comparison uses one GPU allocation; costs and errors come from the same invocation.
\end{itemize}\vspace{2mm}
@@table@@
\smallnote{Times: medians of all retained repetitions, with equal repeats per case. Errors: maxima over cases, output times and repetitions. All repetitions and outliers are retained.}
\smallnote{The target is @@target@@\%, with panel-specific refinement and solver checks. Wave qualification includes displacement, velocity and energy state.}
""", meshes=", ".join(map(str, ns)), table=table(["Problem", "Cases", "Repeats", "Error normalization"], cohort, "lrrl"), target=f'{100*panels[0]["target"]:g}')))

    slides.append(frame("Shared decoder; explicitly named full solvers", r"""
\vspace{-2mm}\[u(x;z)=\sum_{j=1}^{R} g_j(x)\,h_j(z),\qquad u=G\,h(z).\]
\small The spatial bank is fixed during a query. The nonlinear coefficient map is evaluated as latent coordinates change.
\vspace{2mm}
""" + table(["Problem", "Reduced computation", "Primary iterative FOM"], [
        ("Poisson", "Preassembled weak algebra", r"CG: $10^{-6}$"),
        ("Heat", "Weak CN; Cholesky updates", r"CN--CG: $10^{-6}$"),
        ("Burgers", "Preassembled linear; sampled upwind", r"Newton $10^{-6}$; linear $10^{-8}$"),
        ("Reflective waves", "Nonlinear projected RK4 dynamics", r"Midpoint CG: $10^{-6}$")], "lll") +
        r"\smallnote{CG = conjugate gradient; CN = Crank--Nicolson. Burgers uses FFT-preconditioned Newton--BiCGStab. Weak equations average the residual against smooth tests.}" +
        r"\smallnote{\textbf{Scope:} Burgers uses sampled sign-upwind advection, not the historical polynomial tensor. Wave ROM/FOM use the same step size with different integration formulas.}"))

    slides.append(frame("ROM runtime stays nearly flat over these meshes", render(r"""
\begin{center}\includegraphics[width=.97\textwidth,height=.63\textheight,keepaspectratio]{@@figure@@}\end{center}
\smallnote{Logarithmic axes. Orange: fastest tested iterative FOM satisfying the declared target at that mesh. A faster ROM curve alone does not establish an accuracy-qualified win.}
\smallnote{This is a finite-range observation. Reading dense inputs and producing full fields still depend on mesh size. Compare methods within each panel, not across different GPU allocations.}
""", figure=ASSETS.name + "/runtime-scaling.pdf")))

    p = pmap["poisson"]; rom = lookup(p, fine, p["primary_rom"]); fast = fastest(p, fine)
    slides.append(frame("Poisson: strong runtime scaling; accuracy needs work",
        r"\small Primary FOM: zero-start, unpreconditioned CG at $10^{-6}$.\par\vspace{2mm}" + primary_table(p) +
        render(r"""\vspace{2mm}\takeaway{The decoder transfers across meshes with nearly unchanged error and runtime. Its worst error remains above the @@target@@\% target.}
\small At $N=@@n@@$, loose CG at $10^{-1}$ takes @@fastms@@ ms with @@fasterr@@\% error. The raw FOM/ROM ratio is @@ratio@@, but the ROM fails the target.
\smallnote{Error: current-relative solution $L_2$, against a refined finite-difference reference. Reference allowance is included in qualification. All recorded ROM solves satisfy stationarity and all CG controls converge.}
""", target=f'{100*p["target"]:g}', n=fine, fastms=f'{fast["gpu_ms"]:.2f}', fasterr=err(fast), ratio=f'{fast["gpu_ms"]/rom["gpu_ms"]:.2f}')))

    p = pmap["heat"]; rom = lookup(p, fine, p["primary_rom"]); fast = fastest(p, fine)
    slides.append(frame("Heat: a fine-grid speedup that survives looser CG",
        r"\small Primary FOM: matched Crank--Nicolson steps, CG at $10^{-6}$.\par\vspace{2mm}" + primary_table(p) +
        render(r"""\vspace{2mm}\takeaway{At $N=@@n@@$, the ROM remains @@ratio@@$\times$ faster than the fastest tested passing iterative FOM. Both satisfy the physical target and stopping checks.}
\small Looser CG at $10^{-2}$: @@fastms@@ ms and @@fasterr@@\% error. Including input/output transfers, the FOM/ROM ratio is @@hostratio@@.
\smallnote{Error: current-relative field $L_2$, against a refined continuum sine-series solution. Coarse-grid CG remains faster; the medium-grid advantage depends on tolerance.}
""", n=fine, ratio=f'{fast["gpu_ms"]/rom["gpu_ms"]:.2f}', fastms=f'{fast["gpu_ms"]:.2f}', fasterr=err(fast), hostratio=f'{fast["host_ms"]/rom["host_ms"]:.2f}')))

    p = pmap["burgers"]; rom = lookup(p, fine, p["primary_rom"]); fast = fastest(p, fine)
    slides.append(frame("Burgers: fine-grid gain with stalled solves",
        r"\small Primary FOM: FFT-preconditioned Newton--BiCGStab, $10^{-6}/10^{-8}$.\par\vspace{2mm}" + primary_table(p) +
        render(r"""\vspace{2mm}\takeaway{At $N=@@n@@$: @@ratio@@$\times$ faster than the fastest tested passing iterative FOM. The two larger ROM meshes pass the physical target.}
\small Looser FOM: Newton $10^{-2}$, linear $5\times10^{-1}$; @@fastms@@ ms, @@fasterr@@\% error. With transfers: @@hostratio@@$\times$.
\smallnote{\textbf{Stopping limitation:} all ROM fits/steps stop by a small-step/improvement rule; stationarity is unmeasured. Both ROM and FOM miss the physical target on the smallest mesh.}
\smallnote{Error uses the fixed initial reference norm. Spatial/temporal reference allowances are included in the physical gate.}
""", n=fine, ratio=f'{fast["gpu_ms"]/rom["gpu_ms"]:.2f}', fastms=f'{fast["gpu_ms"]:.2f}', fasterr=err(fast), hostratio=f'{fast["host_ms"]/rom["host_ms"]:.2f}')))

    p = pmap["wave_dirichlet"]; rom = lookup(p, fine, p["primary_rom"]); fom = lookup(p, fine, p["primary_fom"])
    comp_rows = [(name, f'{100*rom["component_errors"][key]["current"]:.3f}',
                  f'{100*fom["component_errors"][key]["current"]:.3f}') for key, name in
                 [("displacement", "Displacement"), ("velocity", "Velocity"), ("energy", "Energy state")]]
    slides.append(frame("Reflective waves: displacement alone is incomplete",
        r"\small Primary FOM: implicit-midpoint CG at $10^{-6}$.\par\vspace{2mm}" + primary_table(p) +
        render(r"""\vspace{1mm}\begin{columns}[T]
\column{.47\textwidth}\small
At $N=@@n@@$: current-relative errors.
@@table@@
\column{.49\textwidth}\small
\caution{The ROM is slower at every mesh. Displacement meets the descriptive target, but velocity and energy state do not.}
\end{columns}
\smallnote{Reference: the exact modal solution of the same spatial discretization. These are not continuum error bounds. Full host-to-host wave timing was not measured.}
""", n=fine, table=table(["Component", r"ROM \%", r"FOM \%"], comp_rows))))

    control_rows = []
    for p in panels:
        rom = lookup(p, fine, p["primary_rom"]); tight = lookup(p, fine, p["primary_fom"]); fast = fastest(p, fine)
        control_rows.append((esc(p["label"]), f'{tight["gpu_ms"]/rom["gpu_ms"]:.2f}',
                             f'{fast["gpu_ms"]/rom["gpu_ms"]:.2f}',
                             f'{fast["host_ms"]/rom["host_ms"]:.2f}' if fast["host_ms"] is not None else "Not measured"))
    direct_rows = []
    for key in ("poisson", "heat", "wave_dirichlet"):
        p = pmap[key]; r = lookup(p, fine, p["diagnostics"][0])
        direct_rows.append((esc(p["label"]), f'{r["gpu_ms"]:.3f}', err(r)))
    slides.append(frame("The FOM algorithm and tolerance change the conclusion",
        render(r"""\small At $N=@@n@@$: all entries below are FOM/ROM runtime ratios.
\vspace{2mm}@@table@@
\vspace{2mm}\begin{columns}[T]
\column{.47\textwidth}\small
Direct sine-transform diagnostics:
@@direct@@
\column{.49\textwidth}\small
\caution{Specialized direct solvers are faster on these separable linear problems. Iterative-FOM wins do not establish a universal FOM advantage.}
\end{columns}
\smallnote{Loose controls are selected on development cases; they are not globally optimal classical solvers. Poisson still misses the ROM target; Burgers retains its stall qualification. Wave direct displacement error is zero against the same-grid modal reference.}
""", n=fine, table=table(["Problem", "Tight iterative", "Fastest passing", "Passing + transfers"], control_rows),
            direct=table(["Problem", "GPU ms", r"Error \%"], direct_rows))))

    phase_rows = [("Initialization and projection", f'{phases["initialization_and_parameter_projection"]:.2f}'),
                  ("Latent evolution", f'{phases["evolution"]:.2f}'),
                  ("Dense GPU output", f'{phases["dense_device_output"]:.2f}')]
    config = native["config"]; steps = wave_rows[0]["steps"]
    slides.append(frame("Reflective-wave cost is in latent evolution",
        render(r"""\begin{columns}[T]
\column{.47\textwidth}\small
At $N=@@n@@$:
@@table@@
\takeaway{@@share@@\% of query time is spent in latent evolution.}
\column{.49\textwidth}\small
$T=@@end@@$, $\Delta t=@@dt@@$:
\[ @@steps@@\text{ steps}\times4 = @@stages@@\text{ RK stages}. \]
At every stage:
\begin{itemize}
\item Decoder Jacobian and directional curvature.
\item QR factorization and triangular solve.
\item SVD check of tangent rank.
\end{itemize}
\end{columns}
\caution{The stage loop is fully reduced, but repeats expensive state-dependent work. Individual derivative, QR, SVD and GPU execution costs have not yet been separated.}
\smallnote{Component medians need not sum exactly to the total median. One additional initial acceleration/rank check precedes the RK stages. Proposed next step: profile stage operations, then test cheaper guarding and fewer stages with accuracy checks.}
""", n=fine, table=table(["Component", "GPU ms"], phase_rows), share=f'{evolution_share:.2f}',
            end=f'{config["end_time"]:g}', dt=f'{config["primary_dt"]:g}', steps=steps, stages=4*steps)))

    slides.append(frame("What we can show now; what needs improvement", r"""
\small
\begin{columns}[T]
\column{.48\textwidth}
\begin{block}{Supported by this development study}
\begin{itemize}
\item Fixed learned weights transfer across the tested meshes.
\item Reduced query time is nearly flat over this range.
\item Heat beats the tested iterative FOM frontier on the finest mesh.
\item Burgers has a smaller fine-grid gain with an explicit stopping limitation.
\end{itemize}
\end{block}
\column{.48\textwidth}
\begin{alertblock}{Next experiments, not current results}
\begin{itemize}
\item Poisson: reduce physical error and diagnose representation versus projection limits.
\item Burgers: measure stationarity and improve coarse-grid accuracy.
\item Reflective waves: reduce stage cost; improve velocity and energy accuracy.
\item Freeze choices, then evaluate independent final cases and training seeds.
\end{itemize}
\end{alertblock}
\end{columns}
\smallnote{A tunability claim requires an additional fixed-checkpoint physical-error versus cost study. Adjustable tolerances alone do not establish that tradeoff.}
"""))

    slides.append(frame("Proposed: tunability from one trained network", r"""
\small
Train once per PDE, then select an inference configuration with unchanged weights.
\vspace{2mm}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lll@{}}
\toprule
Knob & Intended cost change & Current status \\ \midrule
Active latent coordinates $k$ & Smaller derivatives and solves & Subset accuracy unvalidated \\
Active spatial bank size $R$ & Smaller operators and output map & Subset accuracy unvalidated \\
Solver budget / tolerance & Fewer or more corrections & Available with frozen weights \\
Time step $\Delta t$ & Fewer or more evolution steps & Available for time-dependent PDEs \\
\bottomrule
\end{tabular*}
\vspace{2mm}
\[
u_{\ell}(x)=\sum_{j=1}^{R_{\ell}}g_j(x)\,
h_{\ell,j}(z_1,\ldots,z_{k_{\ell}}).
\]
\small Train shared weights with a loss at every supported level $\ell$. At inference,
choose a level and solver budget; physically omit inactive computation.
\smallnote{\textbf{Proposed, not measured:} screen truncation of current checkpoints as an ablation; train shared capacity levels if it is unreliable. Zero masks alone may not save runtime.}
"""))

    slides.append(frame("Proposed: demonstrate a physical accuracy--cost curve", r"""
\small
\begin{columns}[T]
\column{.48\textwidth}
\begin{block}{First: current frozen checkpoints}
\begin{itemize}
\item Sweep solve budgets/tolerances at fixed capacity; time steps separately.
\item Pair physical error and full-query time; retain stopping failures.
\item Find where additional computation stops improving physical accuracy.
\end{itemize}
\end{block}
\column{.48\textwidth}
\begin{block}{Then: shared capacity levels}
\begin{itemize}
\item Train with losses at each $(k_{\ell},R_{\ell})$; preassemble each level's operators.
\item Keep more weak equations than latent coordinates; validate each budget.
\item Freeze choices, then compare with passing FOMs on independent cases.
\end{itemize}
\end{block}
\end{columns}
\caution{Lower weak residual need not mean lower physical error. A tolerance setting does not guarantee a requested accuracy, and iterations cannot remove every representation limit.}
\smallnote{Proposed figure: physical error versus query time, one curve per checkpoint; retain failures and all configurations. Select useful operating points on validation cases. This has not been demonstrated by the mesh-scaling tables.}
"""))

    provenance = [(esc(p["label"]), p["metadata"]["job_id"],
                   "A100 80 GB" if "80" in p["metadata"]["gpu"] else "A100 40 GB",
                   r"\texttt{"+p["source_commit"][:7]+"}") for p in panels]
    outliers = []
    for p in panels:
        rr = [lookup(p, n, p["primary_rom"]) for n in ns]
        outliers.append((esc(p["label"]), "/".join(str(r["gpu_outliers"]) for r in rr),
                         "/".join(str(lookup(p,n,p["primary_fom"])["gpu_outliers"]) for n in ns)))
    slides.append(frame("Audit trail and development limitations",
        r"\small" + table(["Panel", "Job", "GPU", "Scientific source"], provenance, "llll") +
        r"\vspace{2mm}\begin{columns}[T]\column{.47\textwidth}\small " +
        r"GPU outliers, ascending mesh order:\par" + table(["Panel", "ROM", "Tight FOM"], outliers, "lrr") +
        r"\column{.49\textwidth}\footnotesize\begin{itemize}" +
        r"\item GPU, float64 and highest precision verified; burn-in before timing." +
        r"\item Paired full fields and repetition arrays retained; no outlier exclusions." +
        r"\item Source/reference/field audits passed; archives checksum-verified." +
        r"\item Final paper cases remain unopened." +
        r"\end{itemize}\end{columns}" +
        r"\smallnote{Audit success does not imply physical accuracy or stationarity. Exact outlier rules and all controls are in the native audits and accompanying report/JSON.}"))

    slides.append(frame("Glossary: reading the tables and method", r"""
\footnotesize
\begin{columns}[T]
\column{.48\textwidth}
\textbf{NMROM / FOM:} nonlinear-manifold reduced model / solver on the full spatial grid.\par\medskip
\textbf{$N$, mesh, intervals/axis:} spatial cells along one coordinate; stored-node counts depend on boundaries.\par\medskip
\textbf{GPU ms / transfers:} online device time in milliseconds / movement of supplied inputs and requested outputs between CPU and GPU.\par\medskip
\textbf{FOM/ROM:} ratio of median times; above one means the ROM is faster for that named comparator.\par\medskip
\textbf{Worst error:} largest normalized field error across cases, output times and repetitions. Current-relative uses the reference norm at that time; fixed-initial uses its initial norm.\par\medskip
\textbf{Target / refinement:} physical-error threshold with numerical checks / disagreement after refining the reference or time step.
\column{.48\textwidth}
\textbf{Frozen checkpoint / development cases:} unchanged trained weights / already opened inputs used while developing the method.\par\medskip
\textbf{Bank, $G$, $R$, head, $z$:} learned spatial shapes, their grid matrix and count, nonlinear mixing map, and unknown latent coordinates.\par\medskip
\textbf{Weak form / residual:} averaging the PDE mismatch against smooth test functions / the mismatch in the solved equations.\par\medskip
\textbf{CG / BiCGStab / Newton:} iterative linear solvers / successive linearized corrections to nonlinear equations.\par\medskip
\textbf{CN / midpoint / RK4:} Crank--Nicolson / implicit midpoint / fourth-order Runge--Kutta time integration.\par\medskip
\textbf{Stationarity / stall:} small objective gradient / small progress or step, which need not imply stationarity.
\end{columns}
"""))
    slides.append(frame("Glossary: operators, numerical work and evidence", r"""
\footnotesize
\begin{columns}[T]
\column{.48\textwidth}
\textbf{Preassembly / cached algebra:} forming small operators before each online query and reusing them. Dense input/output work is still mesh-dependent.\par\medskip
\textbf{Quadrature / sampled upwind:} weighted spatial samples / evaluating the flow-direction-dependent difference operator at those samples.\par\medskip
\textbf{Tensor:} a multidimensional coefficient array encoding an operator; current Burgers is not the historical polynomial-tensor implementation.\par\medskip
\textbf{FFT / sine transform / modal solution:} transform algorithms that diagonalize these separable linear problems.\par\medskip
\textbf{Preconditioning / Cholesky:} making an iterative system easier to solve / a symmetric positive-definite matrix factorization.
\column{.48\textwidth}
\textbf{Jacobian / curvature:} how decoder output changes with latent coordinates / its directional second derivative.\par\medskip
\textbf{QR / SVD / tangent rank:} matrix factorizations used for least squares and singular values / the number of independent decoder directions.\par\medskip
\textbf{Energy-state error:} wave error measured jointly through velocity and displacement gradients; it is not merely the difference in total scalar energy.\par\medskip
\textbf{Median / repetition / outlier:} middle timing value / rerunning one input / an unusually slow repetition under the recorded rule. Repetitions are not independent physical cases.\par\medskip
\textbf{Tunability / independent final validation:} choosing inference effort at fixed weights / testing frozen choices on previously unopened cases.\par\medskip
\textbf{Nested levels / masks / nondominated points:} shared weights at several capacities / inactive terms / settings unbeaten in both cost and error.
\end{columns}
"""))

    tex = PREAMBLE + "\n".join(slides) + "\n\\end{document}\n"
    tex = tex.replace(r"\par\medskip", r"\par\smallskip")
    assert "absorbing" not in tex.lower()
    tex_path = OUT / (STEM + ".tex")
    tex_path.write_text(tex)
    # Keep only the presented cases in the portable numerical extract.
    extracted = {"status": source["status"], "date": source["date"], "aggregation": source["aggregation"],
                 "panels": [{k: v for k, v in p.items() if k not in ("notes", "reference", "baseline", "coordinator_check")}
                            for p in panels],
                 "reflective_runtime_components_ms": phases,
                 "reflective_median_evolution_fraction_pct": evolution_share,
                 "reflective_steps": steps, "source_sha256": sources}
    (OUT / (STEM + ".json")).write_text(json.dumps(extracted, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="nmrom-slide-build-") as build:
        run = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                              "-outdir=" + build, tex_path.name], cwd=OUT, capture_output=True, text=True)
        if run.returncode:
            print(run.stdout[-10000:] + run.stderr[-3000:])
            raise RuntimeError("LaTeX build failed")
        log = (Path(build) / (STEM + ".log")).read_text(errors="replace")
        boxes = [line for line in log.splitlines() if "Overfull" in line]
        pdf_path = OUT / (STEM + ".pdf")
        pdf_path.write_bytes((Path(build) / (STEM + ".pdf")).read_bytes())
        textfile = Path(build) / "slides.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(textfile)], check=True)
        pdf_text = textfile.read_text()
        assert "absorbing" not in pdf_text.lower()
        assert pdf_text.count("\f") == len(slides), (pdf_text.count("\f"), len(slides))
        assert "stationarity" in pdf_text.lower() and "Fastest" in pdf_text
    readme = f"""# Multiresolution results: collaborator presentation

This LaTeX presentation covers Poisson 2D, heat 2D, Burgers 2D and reflective waves. Its numbers are audited development results; independent final paper confirmation remains open.

- [Compiled presentation]({STEM}.pdf)
- [Editable LaTeX source]({STEM}.tex)
- [Portable source and PDF bundle]({STEM}.zip)
- [Presented numerical records]({STEM}.json)
- [Full underlying results and definitions]({SOURCE}.md)
- [Generator]({Path(__file__).name})

The deck contains {len(slides)} slides, including a compact overview, protocol and decoder context, runtime scaling, individual PDE tables, FOM-tolerance/direct-control comparisons, the reflective runtime breakdown, proposed network/solver tunability studies, provenance and a glossary. Tunability proposals are explicitly separated from measured results. No new numerical experiments were run.

Build the supplied source with its adjacent figure directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error {STEM}.tex
```

To regenerate every measured number from the repository's accepted JSON records:

```bash
/home/tahmid/Dev/.venv/bin/python reports/{Path(__file__).name}
```

All experiment statistics use the accepted pooled-repetition aggregation. Physical failures, Burgers stalls, wave component failures, exact comparator tolerances and development selection are retained. The portable bundle does not include large raw field archives; their locations and content hashes are recorded in the data extract and full report.

## Glossary

- **LaTeX / Beamer / PDF:** the editable typesetting source / its slide format / the compiled presentation.
- **Generator / JSON / SHA-256:** code rebuilding tables and figures / structured numerical records / a content checksum.
- **ROM / FOM:** reduced model / full spatial-grid solver.
- **Frozen weights / resolution:** trained parameters held unchanged / the number of spatial intervals along each axis.
- **Paired GPU timing / pooled median:** results and time from the same device invocation / the central value across all retained timing repetitions.
- **Development selection / final confirmation:** choosing among methods on previously opened cases / evaluating frozen choices on independent unopened cases.
- **Stationarity / stall / target:** a sufficiently small objective gradient / small progress without that guarantee / the declared physical-error threshold and numerical checks.

The presentation's closing glossary defines every method and table term used on the slides.
"""
    (OUT / (STEM + ".md")).write_text(readme)
    files = [OUT / (STEM + suffix) for suffix in (".tex", ".pdf", ".json", ".md")]
    files += list(ASSETS.glob("*"))
    check = {"pages": len(slides), "build_passed": True, "pdf_scope_checked": True,
             "overfull_boxes": boxes, "source_sha256": sources,
             "generator_sha256": digest(Path(__file__)),
             "generated_sha256": {str(p.relative_to(OUT)): digest(p) for p in files}}
    manifest_path = OUT / (STEM + ".manifest.json")
    manifest_path.write_text(json.dumps(check, indent=2) + "\n")
    with zipfile.ZipFile(OUT / (STEM + ".zip"), "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files + [manifest_path, Path(__file__)]:
            z.write(path, path.relative_to(OUT))
    print(json.dumps({"pages": len(slides), "pdf": str(pdf_path), "overfull_boxes": boxes}, indent=2))


if __name__ == "__main__":
    main()
