## 2026-09-17
### p-linear — Poisson 2D ladder to q = R on the best checkpoint: at 1024² the pre-registered degenerate-curve criterion PASSES (D1 1.55x, D2 strict, D3); at 256² D1 fails literally (3.17x) only because the top rung is cheaper than the middle; neither mesh meets the falsification intent; the untimed augmented-oracle column for q > 0 is retracted (A10)

INTERIM entry: all three completed jobs are collected, audited and archived; the corrected oracle column (A10) has not been re-run. Worktree `worktrees/2026-09-17-p-linear`, branch `exp/2026-09-17-p-linear`, forked from `exp/2026-09-16-p-bank-head` at `266dea9d`. Namespace `/cluster/tufts/paralab/tawal01/p_linear_20260917/` (empty after collection). Pre-registration with ten dated amendments: `experiments/p-linear/DESIGN.md`. Report generated from the run JSONs: `experiments/p-linear/reports/2026-09-17-p-linear.md`, with `summary.json`, `verdicts.json`, a per-mesh figure and `self-audit-report.md` (Codex unavailable until 2026-09-19 11:33).

**Jobs (4 of 8).** `plin256` = `3780692` (NVIDIA A100-PCIE-40GB, 19.5 min, source `b43a437d7360`); `plin1024` = `3780691` **FAILED, retracted** (below); `plin1024b` = `3783813` (**NVIDIA H200**, 16.2 min, source `3e411b5ac59d`); `plhead1` = `3783883` (NVIDIA A100-PCIE-40GB, 31.9 min, source `2d2cad99709f`). All logged `jax_backend=gpu`, float64, matmul precision `highest`; one job per attempt directory, `squeue` before and after every submit; remote directories deleted after checksum collection. The two meshes ran on different cards and are never compared on cost.

**The question.** Whether the accuracy/cost trade from correction rank q degenerates on a linear PDE: top rung a linear reduced model that is also (about) the cheapest, with POD-LSPG or a direct solver non-dominated. Ladder to q = R = 512 on `pbh02`'s `new_K32` (R = 512, K = 32), every comparator timed in the same job on the same GPU, 12 development sources, 3 timed repetitions.

**1024² (`plin1024b`), the paper's row.** The `m4` ladder with the A4 direct top rung:

| rung | worst same-grid | median total ms |
|---|---:|---:|
| `q0_m4@new_K32` | 3.1495 % | 6.394 |
| `q32_m4@new_K32` | 2.4641 % | 6.592 |
| `q64_m4@new_K32` | 2.0760 % | 6.745 |
| `q128_m4@new_K32` | 1.5459 % | 6.875 |
| `q256_m4@new_K32` | 0.9648 % | 6.849 |
| `d_linear_qr_m4@new_K32` | 0.7421 % | 4.424 |

Clauses: D1 span 1.554x (pass; over the q < R rungs alone, post-hoc, 1.075x); D2 pass, strict yes; D3 pass (non-dominated all: `dst_direct`; reduced: `d_linear_qr_m4@new_K32`, `d_linear_qr_m256@incumbent`, `e_pod512_m4@trainset`); monotone yes; falsified literal not met, intent not met. **Verdict: DEGENERATE under D1 ∧ D2 ∧ D3 as pre-registered.** The top rung reaches the bank floor (0.7421 %) and is the cheapest ladder point; the q < R rungs span only 1.075x in cost. Comparators in the same job: POD-LSPG k'=512 0.1838 % at 6.972 ms; DST direct exact at 3.248 ms; unpreconditioned CG 62.5–121.4 ms (1e-2 to 1e-8). Head alone (`a_neural@new_K32`) 3.1146 % at 5.047 ms. Eliminated `q512_m4` reaches the same floor at 43.220 ms (inert iteration, A4). No speedup over any full-order solver is claimed.

**256² (`plin256`).**

| rung | worst same-grid | median total ms |
|---|---:|---:|
| `q0_m4@new_K32` | 3.1567 % | 9.214 |
| `q32_m4@new_K32` | 2.4699 % | 9.537 |
| `q64_m4@new_K32` | 2.0808 % | 9.819 |
| `q128_m4@new_K32` | 1.5497 % | 10.219 |
| `q256_m4@new_K32` | 0.9689 % | 9.658 |
| `d_linear_qr_m4@new_K32` | 0.7459 % | 3.221 |

Clauses: D1 span 3.173x (FAIL; over the q < R rungs alone, post-hoc, 1.109x); D2 pass, strict yes; D3 pass (non-dominated all: `dst_direct`; reduced: `d_linear_qr_m4@new_K32`, `d_linear_qr_m256@incumbent`, `e_pod512_m4@trainset`); monotone yes; falsified literal MET, intent not met. **Verdict: not degenerate as literally written, because D1 fails — and it fails because the top rung is 3.17x cheaper than the dearest rung, not because any rung buys accuracy for ≥ 2x (A8).** POD-LSPG k'=512 0.1855 % at 10.002 ms; DST 2.526 ms.

**Do the two meshes agree?** On D2 (strict), D3, monotonicity and both falsification readings, yes; on the literal D1 they differ only through the size of the top rung's cost advantage (1.55x at 1024, 3.17x at 256). Under `falsified_intent`, declared in A8 as the deciding clause before the 1024 job ran, both meshes say the same thing: no rung pays ≥ 2x for accuracy.

**Job 3 — head capacity on the frozen R = 512 bank (`plhead1`).** H1: the pbh02 recipe re-run reproduces the primary's development best-found to 0.00 % (pass). H2: the primary's best-found/floor ratio is 4.184x; the largest reduction is 34.2 % (`K64_w256_L3`, 2.755x, dev best-found 2.0553 %), past the 20 % bar but short of the 2x bar for a better anchor: **partial movement**. Width helps (`K32_w256_L2` 3.037x), depth alone does not (`K32_w128_L3` 4.106x), 3x the schedule gives 3.543x. Validation selects `K32_w256_L2`; no arm enters the linear-case table.

**What was wrong and retracted.** (1) `plin1024` (`3780691`) died at 00:14:27 with a GPU RESOURCE_EXHAUSTED in the *untimed* dense best-found oracle on an A100-PCIE-40GB (2.0 GiB jacfwd Jacobians with 32 GiB autotuner variants); the log is complete and the share was at 91% (not disk-full), so this is not the disk-full mode. No timed number and no gate came from it. Fix A7: the dense oracle runs only at ≤ 256 intervals (it is a cross-check of the projected oracle, which agrees with it to 2e-11); resubmitted on an H200 with 240 GB. (2) **A10, found after collection:** the untimed augmented best-found oracle projected with a V whose Gram is ρI, ρ = ((n−1)/254)², not I — measured V^T V = 0.063 I / 1.0079 I / 16.13 I at 64/256/1024. Its q > 0 values are meaningless at 1024 (they rise with q, reading 45.7496 % at q = R where the floor is 0.7421 %) and inflated by ≤ 5e-4 relative at 256; q = 0 is unaffected. The signal was visible in the 64-interval smoke (the column barely moved with q) and I missed it. Retracted in the report and flagged in `summary.json`; fixed by orthonormalising V (verified locally: fixed q = R equals the floor to 0e+00 relative); not re-run. No timed arm, gate or D-clause uses that function. (3) Codex unavailable throughout; design and report audits are written self-audits (A2, A9).

**Audit.** Independent NumPy audits (no driver, no JAX) recompute every reported error from the retained fields: 1024 1440 errors, worst 1.5e-16; bank floor rebuilt by NumPy QR to 6.8e-11; all 23 cross-job gates at 1024 and 18 at 256 pass at 1e-9; the criterion re-derived independently agrees at both meshes.

**Open.** The corrected oracle column needs one untimed re-run (`plorc`, ~5 min per mesh) if the paper wants the bracket; the coordinator decides. Everything is one checkpoint, one seed, development cohorts only; sealed cohorts untouched; nothing merged, nothing pushed.
