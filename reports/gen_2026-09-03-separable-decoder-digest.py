#!/usr/bin/env python3
"""Assemble the separable-decoder results digest from the GENERATED tables of the source reports.

No number in the digest is typed here.  Every table is copied verbatim from a report whose tables
were themselves generated from run JSONs (the source file and its generator are named beside each
table), or converted mechanically from the LaTeX tables that `gen_2026-08-30-cross-pde-cost.py`
emits into `reports/tables/`.  The prose in the template refers to tables by name and carries no
measurements of its own.

    /home/tahmid/Dev/.venv/bin/python reports/gen_2026-09-03-separable-decoder-digest.py

writes `reports/2026-09-03-separable-decoder-results-digest.md`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WAVE = ROOT / "worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/tables"
OUT = HERE / "2026-09-03-separable-decoder-results-digest.md"


# ----------------------------------------------------------------------------- extraction
def section(path: Path, start: str, stop_prefixes=("#",), include_start=True) -> str:
    """Lines from the first line that starts with `start` up to (not including) the next line
    that starts with any of `stop_prefixes`.  The caption line itself is kept."""
    lines = path.read_text().splitlines()
    out, on = [], False
    for ln in lines:
        if not on:
            if ln.startswith(start):
                on = True
                if include_start:
                    out.append(ln)
            continue
        if any(ln.startswith(p) for p in stop_prefixes):
            break
        out.append(ln)
    if not on:
        sys.exit(f"section not found: {path.name}: {start!r}")
    return "\n".join(out).strip("\n")


def table_block(path: Path, header_prefix: str) -> str:
    """A markdown table identified by its header row prefix, up to the next blank line."""
    lines = path.read_text().splitlines()
    out, on = [], False
    for ln in lines:
        if not on:
            if ln.startswith(header_prefix):
                on = True
                out.append(ln)
            continue
        if not ln.strip():
            break
        out.append(ln)
    if not on:
        sys.exit(f"table not found: {path.name}: {header_prefix!r}")
    return "\n".join(out)


_TEX_SUBS = [
    (r"\\footnotesize\s*", ""),
    (r"\\sim", "≈"),
    (r"\\approx", "≈"),
    (r"\s*---\s*", " — "),
    (r"\\textbf\{([^{}]*)\}", r"**\1**"),
    (r"\\mathbf\{([^{}]*)\}", r"**\1**"),
    (r"\\emph\{([^{}]*)\}", r"*\1*"),
    (r"\\textit\{([^{}]*)\}", r"*\1*"),
    (r"\\multicolumn\{\d+\}\{[^{}]*\}\{([^{}]*)\}", r"\1"),
    (r"\\times", "×"),
    (r"---", "—"),
    (r"--", "–"),
    (r"\\mu", "µ"),
    (r"\\ldots", "…"),
    (r"\\,", " "),
    (r"\\quad", " "),
    (r"\\\\", ""),
    (r"\$([^$]*)\$", r"\1"),
    (r"\\&", "&amp;"),
    (r"\\ ", " "),
    (r"\\-", ""),
    (r"~", " "),
]


def tex_to_md(path: Path) -> str:
    """Convert every `tabular` in a generated .tex table file to markdown tables.  Footnote
    paragraphs between tabulars are kept as italic notes.  Mechanical; no values are touched."""
    text = path.read_text()
    out = []
    i = 0
    # the column spec `{@{}rrr@{}}` nests braces, so skip it by counting rather than by regex
    spans = []
    for b in re.finditer(r"\\begin\{tabular\}", text):
        j = b.end()
        depth = 0
        while True:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        e = text.index("\\end{tabular}", j)
        spans.append((b.start(), j, e, e + len("\\end{tabular}")))
    for (s0, body0, body1, s1) in spans:
        pre = text[i:s0]
        note = _tex_note(pre)
        if note:
            out.append(note)
        body = text[body0:body1]
        rows = []
        for ln in body.splitlines():
            s = ln.strip()
            if not s or s.startswith("\\toprule") or s.startswith("\\midrule") or s.startswith("\\bottomrule") \
                    or s.startswith("\\cmidrule"):
                continue
            rows.append(s)
        # a header may span two physical lines (a `&`-terminated line continued on the next)
        joined, buf = [], ""
        for r in rows:
            buf = (buf + " " + r).strip() if buf else r
            if buf.endswith("\\\\"):
                joined.append(buf)
                buf = ""
        if buf:
            joined.append(buf)
        cells = []
        for r in joined:
            for pat, rep in _TEX_SUBS:
                r = re.sub(pat, rep, r)
            r = re.sub(r"(?<![\w*])\*\s+", "*", r)
            cells.append([c.strip() for c in r.split("&")])
        width = max(len(c) for c in cells)
        md = ["| " + " | ".join(cells[0] + [""] * (width - len(cells[0]))) + " |",
              "|" + "---|" * width]
        for c in cells[1:]:
            md.append("| " + " | ".join(c + [""] * (width - len(c))) + " |")
        out.append("\n".join(md))
        i = s1
    note = _tex_note(text[i:])
    if note:
        out.append(note)
    return "\n\n".join(out)


def _tex_note(chunk: str) -> str:
    chunk = re.sub(r"%.*", "", chunk)
    chunk = re.sub(r"\\vspace\{[^}]*\}|\\par", "", chunk)
    for pat, rep in _TEX_SUBS:
        chunk = re.sub(pat, rep, chunk)
    chunk = re.sub(r"\\[a-zA-Z]+", "", chunk)
    chunk = chunk.replace("{", "").replace("}", "")
    chunk = " ".join(chunk.split())
    return f"*{chunk}*" if chunk else ""


# ----------------------------------------------------------------------------- sources
R = HERE
T = {}

# Burgers 2D
T["x1"] = section(R / "2026-08-26-exact-linear-terms-and-gradient-eq.md", "### T-X1.")
T["t8"] = section(R / "2026-08-24-separable-decoder-architecture-and-results.md", "### T8.")
T["t7"] = section(R / "2026-08-24-separable-decoder-architecture-and-results.md", "### T7.")
T["t2"] = section(R / "2026-08-24-separable-decoder-architecture-and-results.md", "### T2.")
T["l1"] = section(R / "2026-08-25-eq-fidelity-ladder.md", "### T-L1.")
T["x4"] = section(R / "2026-08-26-exact-linear-terms-and-gradient-eq.md", "### T-X4.")
T["n1"] = section(R / "2026-08-27-nodes-at-mM-transfer.md", "**T-N1", stop_prefixes=("#", "**T"))
T["n2"] = section(R / "2026-08-27-nodes-at-mM-transfer.md", "**T-N2", stop_prefixes=("#", "**T"))
T["c1"] = section(R / "2026-08-26-codesign-minipilot.md", "### T-C1.")
T["b2t4"] = section(R / "2026-08-30-b2d-tensor-ladder.md", "### T-4")
T["b2t7"] = section(R / "2026-08-30-b2d-tensor-ladder.md", "### T-7")
T["b2t10"] = section(R / "2026-08-30-b2d-tensor-ladder.md", "### T-10")
# Burgers 1D
T["s1"] = section(R / "2026-08-27-b1d-scaling-and-fom-cost.md", "**T-S1", stop_prefixes=("#", "**T"))
T["s3"] = section(R / "2026-08-27-b1d-scaling-and-fom-cost.md", "**T-S3", stop_prefixes=("#", "**T"))
T["o2"] = section(R / "2026-08-28-b1d-rollout-optimization.md", "**T-O2", stop_prefixes=("#", "**T"))
T["b1t2"] = section(R / "2026-08-29-b1d-tensor-sample-free-burgers.md", "### T-2")
T["b1t8"] = section(R / "2026-08-29-b1d-tensor-sample-free-burgers.md", "### T-8")
# Poisson
T["p1"] = section(R / "2026-08-25-poisson-architecture-and-results.md", "### P1.")
T["p3"] = section(R / "2026-08-25-poisson-architecture-and-results.md", "### P3.")
T["pp1"] = section(R / "2026-08-27-b1d-node-screening-and-poisson-qf.md", "**T-P1", stop_prefixes=("#", "**T"))
T["pp2"] = section(R / "2026-08-27-b1d-node-screening-and-poisson-qf.md", "**T-P2", stop_prefixes=("#", "**T"))
T["pp3"] = section(R / "2026-08-27-b1d-node-screening-and-poisson-qf.md", "**T-P3", stop_prefixes=("#", "**T"))
# Stokes and the cross-PDE coverage (LaTeX -> markdown)
T["stokes_res"] = tex_to_md(R / "tables/cross-pde-stokes-results.tex")
T["stokes_cost"] = tex_to_md(R / "tables/cross-pde-cost-stokes.tex")
T["coverage"] = tex_to_md(R / "tables/cross-pde-cost-coverage.tex")
# Waves (POD-bank variant of the same decoder; tables generated by the cell's wav2d_tables.py)
T["w2"] = table_block(WAVE / "wav2d-phase2-gates.md", "| N | BC | head | held-out oracle median")
T["w3"] = table_block(WAVE / "wav2d-phase3-gates.md", "| N | BC | head | G0 (phase 2)")


# ----------------------------------------------------------------------------- template
TEMPLATE = r"""# Separable decoder: results digest

What the separable EQ-decoder ROM has measured on every PDE it has been run on, in one place.
Numbers are **final** for every table unless the table's own status line says otherwise; two
tables are kept only because they are **superseded** and say so. Every table below is copied
verbatim by `gen_2026-09-03-separable-decoder-digest.py` from a source report whose tables were
generated from run JSONs, and the source and its generator are named under each table. The prose
carries no numbers. Current state and the retraction history are in `LAB-LOG.md`.

## The model

$$
u(x; z) \;=\; \mathrm{bc}(x)\,\big\langle g(x),\, h(z) \big\rangle,
\qquad g:\ \mathbb{R}^d \to \mathbb{R}^R,\quad h:\ \mathbb{R}^K \to \mathbb{R}^R .
$$

The spatial track $g$ (random-Fourier lift into a SiLU MLP, linear last layer) carries all
$x$-dependence and never sees $z$; the latent track $h$ (SiLU MLP with a linear skip) carries all
$z$-dependence and all the nonlinearity. Because $g$ is independent of $z$, everything that
depends on space is evaluated once and cached: the weak-form test modes $\Phi$ give a matrix
$A = \Phi^{\mathsf T} G$ so the linear terms of the residual are $A\,h(z)$ exactly, and only a
genuinely nonlinear term needs sampling, or, when it is quadratic, a precomputed tensor. The online
solve is a Levenberg–Marquardt problem in $K$ unknowns whose cost does not depend on the grid.

```mermaid
flowchart LR
  classDef trained fill:#d9e8f5,stroke:#2b5d8c,color:#12314f
  classDef frozen fill:#ece7dc,stroke:#7a6a4a,color:#3d3220
  classDef solved fill:#dcefe0,stroke:#2f7a48,color:#14401f
  x["grid points x"]:::frozen --> g["spatial track g(x)<br/>R features"]:::trained
  g --> G["bank G = g at test modes /<br/>quadrature nodes (cached once)"]:::frozen
  z["latent z (K numbers)"]:::solved --> h["latent track h(z)<br/>SiLU MLP + skip"]:::trained
  G --> r["residual  A h(z) + nonlinear term"]:::solved
  h --> r
  r --> lm["Levenberg–Marquardt in K unknowns"]:::solved
  lm --> z
  h --> u["u = bc(x) ⟨g(x), h(z)⟩"]:::solved
```

Blue = trained offline, sand = frozen and cached, green = solved online.

## Scorecard

| PDE | residual degree | what was found | status |
|---|---|---|---|
| Burgers 2D | 2 | ROM cost flat in $N$; beats the preconditioned Newton FOM at the finest mesh only; exact linear terms and the quadratic tensor remove all quadrature error from the linear and advection terms; head generalisation is the accuracy limiter | final, single seed |
| Burgers 1D | 2 | same accuracy picture, scale-stable to $N = 65\,536$; the tridiagonal FOM is cheaper and far more accurate at every $N$, so no speed story | final, single seed |
| Poisson 2D | 1 | quadrature-free residual equals the full grid to machine precision at the decoder floor; ROM cost flat in $N$; the win is against iterative solvers only | final |
| Stokes 2D | 1 (constrained) | decoder passes its predeclared stop gate over POD-8, but a direct linear solve in the bank span is more accurate and cheaper by orders of magnitude, as predicted | final, not a positive result |
| Wave 2D (POD bank) | 1 (conservative) | reflective walls: the ROM fails the accuracy stop gate on every head and both integrators; absorbing walls: the supervised head passes | final, cause = hypothesis |

## Burgers 2D

### Speed against the classical solver

The paired matched-accuracy protocol: the ROM and the swept Helmholtz-preconditioned Newton
ladder run interleaved AB/BA on one GPU, and the FOM rung is the cheapest one at least as accurate
as the ROM. This is the table that replaced the retracted headline speedups.

{{x1}}

*Source: `2026-08-26-exact-linear-terms-and-gradient-eq.md`, generated by `gen_2026-08-26-exlin.py`.*

The earlier four-resolution table is kept for the record. **Its N=1024 row is superseded**: that
decoder was badly under-trained, so its speedup was bought at an accuracy nobody would accept.

{{t2}}

*Source: `2026-08-24-separable-decoder-architecture-and-results.md`, generated by
`gen_2026-08-24_architecture_study.py`. **Superseded** by T-X1 above.*

### Accuracy: the head-generalisation campaign

{{t8}}

*Source: `2026-08-24-separable-decoder-architecture-and-results.md`, generated by
`gen_2026-08-24_architecture_study.py`.*

### What the nonlinearity buys

{{t7}}

*Source: `2026-08-24-separable-decoder-architecture-and-results.md`, generated by
`gen_2026-08-24_architecture_study.py`.*

### Where the quadrature error sits

The residual the solver minimises was compared with the exact full-grid weak form along real
solver paths. Most of the error was in the linear terms, which is what motivated making them exact.

{{l1}}

*Source: `2026-08-25-eq-fidelity-ladder.md`, generated by `gen_2026-08-25-eq-ladder.py`.*

### Learned quadrature nodes

{{x4}}

*Source: `2026-08-26-exact-linear-terms-and-gradient-eq.md`, generated by `gen_2026-08-26-exlin.py`.*

At the tight budget $m = M$ the win transfers across meshes, but learned nodes never reach the
$m = 4M$ baseline, so there is no cheaper-solve-at-matched-accuracy story.

{{n1}}

{{n2}}

*Source: `2026-08-27-nodes-at-mM-transfer.md`, generated by `gen_2026-08-27-nodes-mM-transfer.py`.*

### Co-designing the decoder with its quadrature

Training the head together with the nodes drives the sampled-gradient error down, but every arm
that touches the head pays reconstruction drift and loses to frozen-decoder node learning on
rollouts. A clean negative at N=64; untested at larger N.

{{c1}}

*Source: `2026-08-26-codesign-minipilot.md`, generated by `gen_2026-08-26-codesign-minipilot.py`.*

### The quadratic tensor: advection projected exactly

Advection is quadratic in $h$, so $\Phi^{\mathsf T}(u \odot D u) = \tfrac12 h^{\mathsf T} Q h$ with $Q$
built once from the frozen bank. Nothing is sampled and nothing is fitted.

{{b2t4}}

{{b2t7}}

{{b2t10}}

*Source: `2026-08-30-b2d-tensor-ladder.md`, generated by `gen_2026-08-30-b2d-tensor.py`. The GPU
differs across rows, so cross-N values are ratios, not exponents.*

## Burgers 1D

A cheap screening testbed. The accuracy findings are scale-stable across a 32× range of $N$, but
the honest tridiagonal Newton FOM is cheaper and more accurate than the ROM at every resolution.

{{s1}}

{{s3}}

*Source: `2026-08-27-b1d-scaling-and-fom-cost.md`, generated by
`gen_2026-08-27-b1d-nodes-and-poisson-qf.py`.*

After the rollout optimisation (broadcast Gauss–Jordan for the $K \times K$ solve, fused
residual and Jacobian, hoisted constants) at bit-level parity:

{{o2}}

*Source: `2026-08-28-b1d-rollout-optimization.md`, generated by
`gen_2026-08-27-b1d-nodes-and-poisson-qf.py`.*

The 1D tensor arm, single GPU, resolutions interleaved in one job:

{{b1t2}}

{{b1t8}}

*Source: `2026-08-29-b1d-tensor-sample-free-burgers.md`, generated by `gen_2026-08-29-b1d-tensor.py`.*

## Poisson 2D

Poisson is the control problem. The decoder floor is a few percent, the ROM cost is flat in $N$,
and the comparison is against **iterative** solvers: an exact spectral solve beats both.

{{p1}}

{{p3}}

*Source: `2026-08-25-poisson-architecture-and-results.md`, generated by
`gen_2026-08-25-poisson-summary.py`. The CG rung is unpreconditioned; the lab log marks the
speedups as soft for that reason.*

### Quadrature-free residual

The whole Poisson residual is $(\Lambda \odot \Phi^{\mathsf T} G)\,h - \Phi^{\mathsf T} f$: no
sampling anywhere, no NNLS fit, identical solutions.

{{pp1}}

{{pp2}}

{{pp3}}

*Source: `2026-08-27-b1d-node-screening-and-poisson-qf.md`, generated by
`gen_2026-08-27-b1d-nodes-and-poisson-qf.py`.*

## Stokes 2D

The Navier–Stokes dress rehearsal: vector-valued, divergence-free, pressure eliminated. The
decoder passed its predeclared stop gate, and the direct reduced solve in the bank span then won
by the margin the design predicted. On a linear PDE the nonlinear head can never buy accuracy.

{{stokes_res}}

{{stokes_cost}}

*Source: `reports/tables/cross-pde-stokes-results.tex` and `cross-pde-cost-stokes.tex`, generated
by `gen_2026-08-30-cross-pde-cost.py`; converted to markdown mechanically here.*

## Wave 2D

Same separable form with the spatial track replaced by the leading $R$ POD modes of the training
snapshots, so the head's distance from its own bank ceiling is measurable. Three heads (`auto`,
`sup` on $(\mu, t)$, `auto+vc`), two ROM integrators (arm A: LSPG–Newmark; arm C: variational
Verlet), reflective and absorbing walls.

{{w2}}

*Phase 2: held-out reconstruction oracle, POD-K and POD-R comparators, and the tangent-space
residual (how much of the true velocity the head's tangent space can represent, relative to POD-K).*

{{w3}}

*Phase 3 decision table. Source: the cell's `tables/wav2d-phase2-gates.md` and
`tables/wav2d-phase3-gates.md`, generated by `wav2d_tables.py` on branch
`exp/2026-09-03-wave2d-mechanism`. The cost ladder was not run because it was gated on a
reflective pass.*

## Coverage

{{coverage}}

*Source: `reports/tables/cross-pde-cost-coverage.tex`, generated by
`gen_2026-08-30-cross-pde-cost.py`. Written before the wave cell ran; the wave row is now
"measured, ROM fails the stop gate on reflective walls, cost ladder withheld".*

## Retracted, so it is not quoted again

- The four-resolution Burgers speedups at N=64 and N=1024 from the first scaling study, and the
  Poisson speedups read as anything but a comparison against an unpreconditioned iterative solver.
- "K is nearly free", "h is capacity-limited", "the codes are unconverged", "more span helps",
  multi-scale Fourier features.
- The tensor arm being "1–9% better" than sampled quadrature (not distinguishable at n=8).
- The 08-16 wave verdict "the ROM fails because it loses energy": the energy-conserving integrator
  fails at the same error, so energy loss is not necessary for the failure.
- Every gate-level retraction of the wave cell (eight, listed in its `WAVE2D-NOTES.md`).

## Glossary

- **ROM / FOM**: reduced-order model (the decoder plus the latent solve) and full-order model (the
  classical grid solver it is compared against).
- **$N$**: grid points per side; $n = N^2$ unknowns in 2D.
- **$K$**: number of latent coordinates, the unknowns of the online solve. **$R$**: width of the
  spatial bank, the number of features $g$ produces. **$M$**: number of weak-form test modes.
  **$m$**: number of quadrature nodes when a term is sampled.
- **head**: the latent track $h$. **bank**: the cached spatial features $G$.
- **paired / AB/BA**: ROM and FOM timed alternately in one job on one GPU so the comparison is fair.
- **matched accuracy**: the FOM rung chosen is the cheapest one at least as accurate as the ROM.
- **e2e**: end to end, including the initial-condition latent fit and the full-grid decode.
- **oracle**: the residual evaluated on the full grid (no quadrature), or, for the wave cell, the
  best reconstruction the head can reach when the latent is fit to the truth.
- **held-out / fresh**: data never seen in training or in fitting the quadrature.
- **rung (b), (c1), (c3)**: the EQ fidelity ladder; (b) is residual error, (c1) gradient error, (c3)
  step error, each relative to the full grid; **cos** is the direction agreement.
- **NNLS**: non-negative least squares, the convex fit that picks quadrature weights.
- **exact-linear / exlin**: linear residual terms computed exactly through $A = \Phi^{\mathsf T} G$.
- **tensor**: the precomputed quadratic form for the advection term.
- **QF**: quadrature-free.
- **trip / tripwire**: a predeclared check that stops a run instead of reporting it.
- **POD-K, POD-R**: linear projection onto the leading K or R proper-orthogonal-decomposition modes;
  the floors a linear model with the same width would reach.
- **G0, D0–D2, W3**: the wave cell's gates; W3 is the accuracy stop gate, G0 the manifold-quality
  gate, its tangent residual the fraction of the true velocity the head's tangent space misses.
- **arm A / arm C**: the wave cell's two integrators, least-squares Petrov–Galerkin with Newmark,
  and variational Störmer–Verlet on the pulled-back Lagrangian.
- **superseded**: replaced by a later measurement; kept only so the history is readable.
"""


def main():
    text = TEMPLATE
    for k, v in T.items():
        text = text.replace("{{" + k + "}}", v)
    missing = re.findall(r"\{\{(\w+)\}\}", text)
    if missing:
        sys.exit(f"unfilled placeholders: {missing}")
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
