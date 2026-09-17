# Codex design audit — UNAVAILABLE

Attempted 2026-09-17T05:03Z with `codex exec -s read-only -C <worktree> -o reports/codex-design-audit.md - < prompt` (model gpt-6-astra, reasoning xhigh). The CLI refused with a usage-limit error (shared account, nine concurrent lanes):

```
ERROR: You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at Sep 19th, 2026 11:33 AM.
ERROR: You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at Sep 19th, 2026 11:33 AM.
```

Substitute: an independent fresh-context Claude agent (general-purpose, read-only) was given the identical prompt; its findings are in `codex-design-audit-substitute.md`. This is a weaker independence guarantee than a different model family and is stated as such in the report.

## The prompt

```
You are an independent auditor (read-only). Do NOT modify, create or delete any file. Do NOT run GPU work, do NOT submit any job, do NOT run Python that imports jax. You may read files and run grep/cat/git log.

Audit the pre-registration `experiments/p-linear/DESIGN.md` of a Poisson 2D reduced-order-model experiment before its first cluster job, together with the driver it pre-registers: `experiments/p-linear/plin_solve.py`, `experiments/p-linear/plin_core.py`, `experiments/p-linear/config-1024.json`, `experiments/p-linear/config-256.json`, and the job-3 driver `experiments/p-linear/plin_head.py` with `experiments/p-linear/config-head.json`. The parent lane's context is in `experiments/p-bank-head/DESIGN.md` and `experiments/p-bank-head/reports/2026-09-16-poisson-bank-and-head.md`; the machinery reused unchanged is `experiments/multiresolution-poisson/correction_core.py`, `experiments/head-ablation/poisson_ablation.py`, `experiments/head-ablation/arms.py`, `experiments/p-bank-head/pbh_core.py`, `experiments/p-bank-head/pbh_fit.py`, `experiments/multiresolution-poisson/iterative_core.py`, and `experiments/p-linear/directions.py` (copied from the cheap-corrections lane).

Questions, in priority order:
1. Is the pre-registered degenerate-curve criterion (D1-D3) and its falsification clause well-posed, and could it be gamed by the arm set or the metric choices? Is anything left to post-hoc choice?
2. Correctness of the basis extension `plin_core.extend_basis`: does using the retained 32 columns verbatim plus the SVD of the residual projected off them give nested directions consistent with the retained rule? Any metric mismatch (R_G at the training mesh vs the query mesh)? At q = R, is the elimination path in `correction_core.prepare_correction` well-defined (rank, M > K+q requirement) and is the "q = R equals the free bank" identity check sound given the two arms use the same M?
3. The test-count rules: the `m256` rule is skipped when M <= K+q; the `m4` rule sets M = 4(K+q). Any rung where the ladder solve becomes underdetermined, or where POD-LSPG with identity head and M=4k' is ill-posed? Any subject where the timing contract differs so that a cost comparison across subjects would be unfair (note the ladder path carries diagnostics inside the timed region)?
4. Fidelity gates: are the mapped pairs in `config-1024.json` `gates` actually the same computation as the parent arms (same kernel, same M, same directions, same initializer)? Flag any pair that cannot be expected to agree to 1e-9.
5. Bugs: shape errors, wrong keys, name collisions in flat staging (every module is copied to one directory), config path resolution, anything that would crash on the cluster after hours of compute, memory hazards at 1024 intervals (bank 1023^2 x 512 float64 = 4.3 GB; POD modes of the same size; up to ~34 jitted query kernels resident).
6. Job 3 (`plin_head.py`): is the reproducibility gate H1 and verdict H2 well-posed? Is the shared bank assumption asserted correctly? Any leakage of development sources into training or selection?

Write your findings as a numbered list, each with severity (blocker / should-fix / note), the file and line, and a concrete suggested fix. Be specific; do not pad. End with a one-paragraph overall verdict on whether job 1 may be submitted.
```
