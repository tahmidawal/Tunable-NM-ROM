# Authoritative manuscript location

The user explicitly requires all paper edits to occur only in:
`/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/paper`.

This directory is the authoritative source and PDF location. Do not edit the paper
in an experiment worktree or maintain another working manuscript copy. Read the
canonical root `LAB-LOG.md` before work, and append there when finished. Experiment
worktrees supply read-only evidence. Generate numerical tables from retained source
records; rebuild `main.pdf` here after manuscript changes. This explicit user
instruction overrides earlier instructions locating the manuscript in paper-refresh.

Main experiment presentation follows the older paper: readable comparison tables,
selected settings, and meaningful deployment controls in the main section. Use CG
as the main full-order comparator for applicable linear PDEs, including L-shaped
Poisson; keep sparse-direct comparisons out of the main tables unless the user
requests them again. Name the appropriate nonlinear FOM for Burgers and NS rather
than calling it CG. Label development/final cohorts and never fabricate pending
CG timings. Keep linear-bank baseline gains separate from nonlinear NM-ROM gains.

Latest presentation preference (2026-09-20): comparison tables in the PDF
contain NM-ROM and named FOMs only. Keep the appendix compact: method,
configuration and validation needed for main claims. Full historical and
operator/POD comparison records remain in the repository, not as PDF dumps.

User decision (2026-09-21, reversible): the paper compares against the NAMED
iterative full-order solvers only (CG for Poisson/heat/L-shape, Newton–BiCGStab
for Burgers, CNAB2 for Navier–Stokes). Direct/spectral transforms (DST, sine
transform, FFT-based direct), sparse-direct (SuperLU) and coarse-grid FOM
controls are not featured in the PDF; Limitations carries one neutral scoping
sentence instead. Their evidence, macros and provenance stay in the repository;
`check_headline.py` asserts none of them appears in the rendered text. The user
may add them back later.

User decisions (2026-09-21, finishing pass):
- Table 2 reports accuracy and compiled-query memory only; no time column (checked).
- Table 1 keeps its current layout (no one-row-per-problem compaction); GPU-query
  time in Table 1, complete-query in captions/appendix.
- Heat: the wide-bank series uses the sealed cohort at every mesh; plain
  Crank–Nicolson row leads, batched fit labelled beneath it.
- Heat 3D is reported in the failures table (all-times where a record holds it;
  the accepted 32^3/64^3 records hold evolved times only and are labelled so),
  not in Table 1 or Figure 1.
- POD-LSPG rows stay in Table 2: an explicit exception to the "no POD rows"
  preference above.
- The abstract follows the older paper's four-move structure (gap; what we
  present and its controls; "the framework combines"; results + one limits clause).

User decision (2026-09-21, "option A"): ONE full-order selection rule for every row
of Table 1 and Figures 1-2 — the fastest tested setting of the named solver, in the
same allocation, whose error is at most the row's accurate NM-ROM setting's (the
single setting where there is only one). check_headline.py verifies it per row from
the recorded candidate settings; rows whose record retains one setting cite the
record's own identical selection rule.
