# Method deck tutor session: Q&A notes

Running log of questions asked while walking through the method deck (https://claude.ai/artifact/7D1rwb1AthPF4VdGdhtuiU), each with a short answer and the files cited. Labels: Measured / Derived / Hypothesis.

## Slide 26 `kernels`: what does "cost grows with r, but not at the textbook rates" mean?

The "textbook rate" is the exponent p predicted by counting operations for each solver kernel at fixed $(M, m)$: N ~ $r^1$, J ~ $r^1$, $J^\top J$ ~ $r^2$, Cholesky ~ $r^3$. Fitting $t = a + b\,r^p$ to measured kernel times gave 2D 1.17 / 1.08 / 1.92 / 1.24 and 3D 1.01 / 1.03 / 1.33 / 1.03 (Measured). The biggest gap is Cholesky: about $r^{1.0}$–$r^{1.24}$ against the predicted $r^3$. "GPU latency-bound" is the slide's explanation, but it is a Hypothesis: the report labels it a classification, not a cause, and no ablation tested it.

Files: `reports/2026-10-17-full-experiment-plan/final/report.md` §2, §3.1; `worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/report.md:19-26, 264`; `.../DESIGN.md:217-222` (FLOP/byte counts); `.../make_configs.py:84-98` (r grid, fixed M, rules); `.../p1comp.py:189` (`kernel_panel`).
