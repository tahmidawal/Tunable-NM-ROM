"""Render a compact, portable LaTeX table report from audited normalized JSON.

Invoked by generate_iterative_multiresolution.py before its manifest is written.
This module has no scientific execution or training path.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def esc(value):
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
                    "{": r"\{", "}": r"\}", "×": r"$\times$", "–": "--", "—": "---"}
    return "".join(replacements.get(c, c) for c in str(value))


def lookup(panel, n, method):
    return next(r for r in panel["rows"] if r["intervals"] == n and r["method"] == method)


def qualifies(panel, row):
    if "qualified" in row:
        return row["qualified"]
    return row["converged"] and row["worst_error"] + row.get("error_allowance", 0) <= panel["target"]


def fastest(panel, n):
    eligible = [r for m in panel["iterative_foms"] if qualifies(panel, r := lookup(panel, n, m))]
    return min(eligible, key=lambda r: r["gpu_ms"], default=None)


def num(value):
    return "---" if value is None else f"{value:,.3f}"


def pct(value):
    return f"{100 * value:.4g}"


def speed_comparison(fom_ms, rom_ms):
    ratio = fom_ms / rom_ms
    if ratio >= 1:
        return f"{ratio:.3f}" + r"$\times$ faster"
    return f"{1 / ratio:.3f}" + r"$\times$ slower"


def status(panel, row):
    s = "pass" if qualifies(panel, row) else "fail"
    return s + (r"$^{\dagger}$" if row.get("stalled_events") else "")


def table(headers, rows, spec):
    start = (r"\noindent\begin{tabularx}{\linewidth}{@{}" + spec + "@{}}\n") if "L" in spec else (r"\noindent\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + spec + "@{}}\n")
    end = r"\bottomrule\end{tabularx}\par" if "L" in spec else r"\bottomrule\end{tabular*}\par"
    return (start
            + r"\toprule " + " & ".join(headers) + r" \\ \midrule" + "\n"
            + "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
            + "\n" + end + "\n")


def heading(title, subtitle):
    return r"{\LARGE\bfseries\color{ink} " + esc(title) + r"}\\[2mm]" + "\n" + esc(subtitle) + r"\par\vspace{4mm}" + "\n"


PREAMBLE = r"""% Generated from audited JSON; do not edit numerical cells by hand.
% Portable build: latexmk -pdf -interaction=nonstopmode -halt-on-error FILE.tex
\documentclass[10pt,a4paper,landscape]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,booktabs,tabularx,array,xcolor,geometry,fancyhdr,hyperref}
\geometry{left=16mm,right=16mm,top=15mm,bottom=17mm}
\definecolor{ink}{HTML}{172B3A}
\definecolor{teal}{HTML}{087E8B}
\definecolor{muted}{HTML}{536776}
\hypersetup{colorlinks=true,urlcolor=teal,linkcolor=teal,pdftitle={NMROM multiresolution results: tables}}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\footnotesize\color{muted}Separable NMROM | Audited development results | 11 September 2026}
\fancyfoot[R]{\footnotesize\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{2mm}
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.16}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcommand{\note}[1]{{\small\color{muted}#1\par}}
\begin{document}
"""


def primary_page(panels):
    body = heading("Accuracy and speed across resolutions", "Poisson 2D, heat 2D, Burgers 2D and reflective waves. Frozen-network baseline configurations.")
    body += r"Median complete GPU query time; worst error over all cases, times and repetitions. $N$ is intervals per axis. FOM/ROM values above one favor the ROM against that named solver." + "\n\n"
    rows = []
    for p in panels:
        for n in p["intervals"]:
            rom, fom = [lookup(p, n, p[k]) for k in ("primary_rom", "primary_fom")]
            rows.append((esc(p["label"]), n, num(rom["gpu_ms"]), num(fom["gpu_ms"]),
                         num(fom["gpu_ms"]/rom["gpu_ms"]), pct(rom["worst_error"])))
    body += table(["Problem", "$N$", "ROM ms", "FOM ms", "FOM/ROM", r"ROM error \%"], rows, "lrrrrr")
    body += r"\vspace{3mm}" + "\n"
    norms = {
        "poisson": "Current-relative solution",
        "heat": "Current-relative field",
        "burgers": "Fixed-initial-relative field",
        "wave_dirichlet": "Current-relative displacement",
    }
    rows = [(esc(p["label"]), esc(lookup(p, p["intervals"][0], p["primary_fom"])["label"]), norms[p["key"]], f"{p['cases']} / {p['repetitions']}") for p in panels]
    body += table(["Problem", "Primary FOM", "Displayed error norm", "Cases / repeats"], rows, "lLLr")
    body += r"\note{Burgers accepts stall exits under its recorded stopping rule; stationarity is unmeasured. Reflective-wave qualification checks velocity and energy-state errors as well as displacement. Different error norms cannot be ranked across PDEs.}"
    body += r"\note{These are development results with final cases unopened. Training, mesh assembly and compilation are excluded; supplied-input processing, solves and full-field device outputs are included. Pair timings within a job, never across different GPUs.}"
    return body


def tuning_page(tuning):
    body = heading("Poisson: online speed--accuracy tuning", "Same frozen network. Baseline and FOM controls are retimed together in a new Poisson allocation.")
    if tuning is None:
        return body + r"\textbf{Experiment in progress.} The frozen-checkpoint Poisson screen and paired multiresolution confirmation have not yet been accepted. Fastest and most accurate rows will be generated from the audited run records." + "\n"
    # A report adapter supplies selection rows only after source, output and
    # collection checks. Every comparator here belongs to the tuning job.
    assert tuning["audited"]
    fine = max(tuning["configuration"]["intervals"])
    fastest_stationary = next(s for s in tuning["selections"]
                              if s["intervals"] == fine and s["role"] == "Fastest stationary")
    if fastest_stationary["row"] is not None:
        row = fastest_stationary["row"]
        comparator = next(c for c in tuning["comparators"] if c["intervals"] == fine)
        ratio = fastest_stationary["fastest_fom_ms"] / row["gpu_ms"]
        direction = "faster" if ratio >= 1 else "slower"
        factor = ratio if ratio >= 1 else 1 / ratio
        body += (r"\textbf{Fastest stationary NMROM at $N=" + str(fine) + "$: "
                 + f"{factor:.6f}" + r"$\times$ " + direction
                 + r" than the fastest tested passing CG FOM.}\par" + "\n")
        gate = "passes" if row["qualified"] else "does not meet"
        body += (r"\note{ROM " + f"{row['gpu_ms']:.6f}" + " ms versus "
                 + f"{fastest_stationary['fastest_fom_ms']:.6f}" + " ms for "
                 + esc(comparator["fast_label"]) + ". ROM worst error: "
                 + f"{100*row['worst_error']:.6f}" + r"\%; " + gate + " the "
                 + pct(tuning["target"]) + r"\% accuracy gate. This is a measured runtime ratio.}" + "\n")
    body += esc(tuning["scope"]) + r"\par" + "\n"
    rows = []
    for s in tuning["selections"]:
        r = s["row"]
        if r is None:
            if s["role"] == "Fastest passing":
                # Absence is stated explicitly in the finding below the table.
                continue
            rows.append((s["intervals"], esc(s["role"]), "none", "---", "---", "---", "---", "---", "---", "---", "---"))
            continue
        rows.append((s["intervals"], esc(s["role"]), esc(r["short_label"]),
                     num(r["gpu_ms"]), f"{100*r['worst_error']:.6f}", f"{100*r['adjusted_error']:.6f}",
                     f"{r['stationary_invocations']}/{r['invocations']}",
                     r["gpu_outliers"], "pass" if r["qualified"] else "fail",
                     speed_comparison(s["tight_fom_ms"], r["gpu_ms"]),
                     speed_comparison(s["fastest_fom_ms"], r["gpu_ms"])))
    body += table(["$N$", "Selection", "Configuration", "ROM ms", r"Error \%", r"Adjusted \%", "Stationary", "Outliers", pct(tuning['target'])+r"\% gate", r"\shortstack[l]{ROM vs\\tight CG}", r"\shortstack[l]{ROM vs\\passing CG}"], rows, "rlLrrrlrlll")
    body += r"\note{Tight CG uses tolerance " + esc(f"{tuning['configuration']['primary_cg_tolerance']:.0e}") + r"; passing CG is the fastest tested iterative FOM meeting the target, with its setting listed below. ``Times faster'' divides FOM by ROM median time; ``times slower'' reverses that ratio. All timings are paired within this job. ``Fastest tested'' includes nonstationary early exits. Adjusted error includes the empirical reference allowance. GPU timing outliers are retained.}"
    body += esc(tuning["finding"]) + r"\par" + "\n"
    rows = [(r["intervals"], num(r["tight_ms"]), num(r["fast_ms"]), pct(r["fast_error"]),
             esc(r["fast_label"]), num(r["direct_ms"])) for r in tuning["comparators"]]
    body += table(["$N$", "Tight CG ms", "Fast passing CG ms", r"Fast CG error \%", "Fast CG setting", "Direct DST ms"], rows, "rrrrLr")
    body += r"\note{" + esc(tuning["settings_note"]) + "}"
    return body


def controls_page(panels):
    body = heading("How the FOM tolerance changes the comparison", "Fastest tested iterative FOM passing the same development target and original numerical checks.")
    rows = []
    for p in panels:
        for n in p["intervals"]:
            rom, fom = lookup(p, n, p["primary_rom"]), fastest(p, n)
            if fom is None:
                rows.append((esc(p["label"]), n, "none", "---", "---", "---"))
                continue
            host = fom["host_ms"]/rom["host_ms"] if fom.get("host_ms") and rom.get("host_ms") else None
            rows.append((esc(p["label"]), n, esc(fom["label"]), num(fom["gpu_ms"]),
                         num(fom["gpu_ms"]/rom["gpu_ms"]), num(host)))
    body += table(["Problem", "$N$", "Passing FOM", "FOM ms", "GPU FOM/ROM", "Host FOM/ROM"], rows, "lrLrrr")
    body += r"\note{Ratios use the baseline ROM on page 1. A ratio alone is diagnostic when that ROM fails its target. Host timing includes input/output transfers where measured; the wave study measured only output transfer, so its complete host ratio is unavailable. All selections use opened development cases.}"
    body += r"\vspace{1mm}\textbf{Direct-transform controls at the largest mesh}\par" + "\n"
    rows = []
    for p in panels:
        n = max(p["intervals"])
        for method in p["diagnostics"]:
            r = lookup(p, n, method)
            rows.append((esc(p["label"]), n, esc(r["label"]), num(r["gpu_ms"]), pct(r["worst_error"])))
    body += table(["Problem", "$N$", "Control", "GPU ms", r"Error \%"], rows, "lrLrr")
    body += r"\note{A win against iterative CG does not imply a win against a direct sine-transform solver. These controls retain their original measurement contract. Full control rows, retained timing outliers and raw JSON links are in the adjacent Markdown report.}"
    return body


def definitions_page(panels, tuning):
    body = heading("Accuracy checks and provenance", "The tables preserve the existing definitions and stopping limitations.")
    wave = next(p for p in panels if p["key"] == "wave_dirichlet")
    rows = []
    for n in wave["intervals"]:
        r = lookup(wave, n, wave["primary_rom"])
        c = r["component_errors"]
        rows.append((n, *(pct(c[k]["current"]) for k in ("displacement", "velocity", "energy")), status(wave, r)))
    body += table(["Reflective-wave $N$", r"Displacement error \%", r"Velocity error \%", r"Energy-state error \%", "All-state target"], rows, "Lrrrl")
    rows = []
    for p in panels:
        rom_counts = "/".join(str(lookup(p, n, p['primary_rom'])['gpu_outliers']) for n in p['intervals'])
        fom_counts = "/".join(str(lookup(p, n, p['primary_fom'])['gpu_outliers']) for n in p['intervals'])
        rows.append((esc(p["label"]), p["metadata"]["job_id"], esc(p["metadata"]["gpu"]), r"\texttt{"+p["source_commit"][:12]+"}", rom_counts, fom_counts))
    if tuning is not None:
        tight = f"cg_{tuning['configuration']['primary_cg_tolerance']:.0e}"
        fom_counts = "/".join(str(next(r for r in tuning['rows'] if r['phase']=='confirmation' and r['intervals']==n and r['method']==tight)['gpu_outliers']) for n in tuning['configuration']['intervals'])
        rows.append(("Poisson tuning", tuning["job_id"], esc(tuning["gpu"]), r"\texttt{"+tuning["source_commit"][:12]+"}", "see page 2", fom_counts))
    body += r"\vspace{2mm}" + table(["Panel", "Cluster job", "GPU", "Scientific source", "ROM outliers", "Tight FOM outliers"], rows, "llLlll")
    body += r"\note{Outlier counts follow ascending mesh order and include all repetitions. Every panel uses float64 and highest matrix precision, verified GPU execution, burn-in, retained timing arrays, source hashes and checksum-collected full fields. Errors and costs refer to the same solver invocation. Medians pool equal repeats per case; every outlier remains included.}"
    body += r"\textbf{Glossary}\par" + "\n"
    definitions = [
        ("NMROM / ROM / FOM", "Nonlinear-manifold reduced model / reduced model / full-grid numerical solver."),
        ("$N$ / ms / FOM/ROM", "Intervals along each axis / milliseconds / ratio of paired median query times."),
        ("Current / fixed-initial relative error", r"Field error divided by the reference field norm at the same time / by the fixed initial norm. The field norm is a spatial $L^2$ norm."),
        ("Worst / adjusted error / target", "Maximum over every case, output and repeat / includes empirical reference-refinement allowance / physical error plus original solver and refinement checks."),
        ("Stationarity / residual / stall", "Small objective gradient / mismatch in the equation / updates or improvements too small to continue, which need not imply stationarity."),
        ("CG / BiCGStab / Newton / preconditioner", "Iterative linear solver / nonsymmetric linear solver / nonlinear equation iterations / transformation used to help a linear solver."),
        ("Crank--Nicolson / midpoint / DST / FFT", "Implicit time-stepping formulas / midpoint time stepping / discrete sine transform / fast Fourier transform."),
        ("$k$ / $r$ / $M$ / checkpoint / weak modes", "Latent dimension / learned spatial-bank size / test-mode count / frozen trained weights / smooth functions used to test the PDE residual."),
        ("Development / reference / outlier", "Cases already used to select methods / numerical grading solution, with empirical refinement checks / unusually slow retained timing under the native audit's rule."),
    ]
    body += r"{\small\renewcommand{\arraystretch}{1.05}" + table(["Term", "Plain-language meaning"], definitions, "lL") + "}\n"
    return body


def build_tables(payload, output):
    panels = payload["panels"]
    assert {p["key"] for p in panels} == {"poisson", "heat", "burgers", "wave_dirichlet"}
    tuning = payload.get("poisson_tuning")
    pages = [primary_page(panels), tuning_page(tuning), controls_page(panels), definitions_page(panels, tuning)]
    tex = PREAMBLE + "\n\\newpage\n".join(pages) + "\n\\end{document}\n"
    tex_path, pdf_path = output.with_suffix(".tex"), output.with_suffix(".pdf")
    tex_path.write_text(tex)
    with tempfile.TemporaryDirectory(prefix="nmrom-table-pdf-") as temp:
        stage = Path(temp)
        staged_tex = stage / tex_path.name
        staged_tex.write_text(tex)
        process = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", staged_tex.name],
                                 cwd=stage, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(process.stdout[-9000:] + process.stderr[-3000:])
        log = staged_tex.with_suffix(".log").read_text()
        warnings = re.findall(r"Overfull \\[hv]box[^\n]*", log)
        if warnings:
            raise ValueError("Table PDF has layout overflow: " + "; ".join(warnings))
        shutil.copy2(staged_tex.with_suffix(".pdf"), pdf_path)
    return [tex_path, pdf_path]
