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
