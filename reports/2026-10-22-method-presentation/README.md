# Method presentation: architecture, solver, cost/accuracy factors, failures

Source of the slide deck https://claude.ai/artifact/7D1rwb1AthPF4VdGdhtuiU (39 slides, 22 Oct 2026). `make_deck.py` regenerates `project/`; figures are uploaded copies from `../2026-10-17-full-experiment-plan/final/figs/`, `../2026-10-01-offmesh-quadrature-plots/figs/` and `../2026-10-20-spectral-investigation/why-spectral/figs/`.

Numbers are copied from the Codex-audited lane reports named in each slide footer (final report, jcp-wide-bank, jcp-smooth-bank, imp-* lane reports). Exceptions, labelled on the slides: E1 (imp-box) and E2 (imp-geometry) numbers are interim and unaudited; the E2 S range was read from `runs/decide/L|wide|S1/final.json`.

Failure explanations are tagged Measured (a diagnostic in the lane shows it), Derived, Descriptive, or Hypothesis (not tested by ablation).
