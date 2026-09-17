## 2026-09-17

### lshape — the solves are in: on the L-shape a reduced arm IS non-dominated at N=256, but only by being cheaper and less accurate; the direct solve is never beaten on accuracy

**What ran.** Branch `exp/2026-09-17-lshape`, worktree
`worktrees/2026-09-17-lshape`, namespace `/cluster/tufts/paralab/tawal01/lshape_20260917/`,
all A100, `jax_backend=gpu`, x64, matmul `highest`.

| job | attempt | what | state |
|---|---|---|---|
| 3784662 | `lsh03` | solve, N=64 and 128, M=257 | COMPLETED 0:0, 32m49s |
| 3784663 | `lsh04` | solve, N=256, M=257 (headline mesh) | COMPLETED 0:0, 15m07s |
| 3784910 | `lsh06` | solve, N=256, M=1024: the free rung q=R, the ladder, five POD ranks, six full-order comparators | COMPLETED 0:0, 16m20s |
| 3784664 | `lsh05` | solve, N=512, M=257 | **FAILED 1:0 at 1m15s** — see below |
| 3789568 | `lsh07` | resubmit of lsh05, one fix | RUNNING at the time of writing; past the gate that killed lsh05 |

All three completed attempts were checksum-collected (`OUTPUTS.sha256` + `MANIFEST.sha256`
verified remotely before tar and locally after extract), independently re-audited, archived as
bounded Git chunks under `experiments/lshape/artifacts/`, and their remote attempt directories
deleted; paralab is back to 91 %.

**The headline question, which this lane exists for: is any reduced arm non-dominated on
(median complete-query ms, worst same-grid error) with a sparse direct solve and the PCG rungs
in the same job?** The L-shape is the campaign's most favourable domain for a reduced model —
no fast transform applies, so `fom_splu` and IC(0)-PCG are the honest comparators. Answer, per
mesh, with the full-order comparators included:

| block | mesh | non-dominated set | reduced arm on it? |
|---|---:|---|---|
| M=257 | 64 | `fom_splu` alone (0 %, 1.282 ms) | **no** |
| M=257 | 128 | `pod16` (34.628 %, 2.086 ms), `pod32` (19.005 %), `pod64` (7.701 %, 2.294 ms), `fom_splu` (0 %, 2.475 ms) | yes, but cheapness-only |
| M=257 | 256 | `pod16`, `pod32`, `pod64`, `neural_q0@head_smooth_R256_K16` (3.493 %, 2.848 ms), `pod128` (2.457 %, 2.850 ms), `neural_q64@head_sdf_R512_K16` (2.131 %, 3.028 ms), `fom_splu` (0 %, 8.386 ms) | **yes** |
| M=1024 | 256 | `pod16`, `pod32`, `freebank@head_sdf_R512_K32` and `…K16` (0.7791 %, 2.582/2.583 ms), `fom_splu` (0 %, 8.351 ms) | **yes** |

Read plainly: **the reduced models never win on accuracy** — `fom_splu` is exact to round-off at
every mesh — and they are non-dominated only where they are cheaper. At N=64 they are not even
that: the direct solve is both faster and exact, and every reduced arm is dominated. At N=128 the
only non-dominated reduced arms are POD rungs that are 1.08x cheaper at 7.7–34.6 % error, which is
not a usable operating point. At N=256 the story turns: the best neural arm gives 2.131 % worst
error at 2.77x less cost than the direct solve, and in the M=1024 job the free rung gives 0.779 %
at 3.23x less. The margin against `fom_splu` is a *cost* margin of 2.8–3.2x paid for with 0.8–2.1 %
error; there is no accuracy axis on which a reduced arm wins. The crossover is between N=128 and
N=256, and `fom_splu`'s per-query cost grows 1.28 → 2.48 → 8.39 ms across the three meshes while
the reduced arms are nearly flat (2.7 → 2.7 → 3.0 ms), which is what makes the N=512 mesh the
interesting one and why `lsh07` matters.

**The free rung (lsh06, M=1024), reported as its own block and NOT cost-comparable with the
M=257 jobs** (the dense M×n test-mode projection is charged inside every reduced query, so
M=1024 carries ~4x the projection work). It reaches 0.7791 % worst / 0.3170 % median at 2.583 ms,
**1.003x its own bank floor**, with 0 solver iterations — the head is bypassed entirely and every
coefficient comes from one exact least-squares elimination. It is the one place in this cell where
the learned bank beats the classical linear baseline on both axes at once: `pod256` in the same
job is 0.9303 % at 3.987 ms, so the free rung is 1.19x more accurate at 1.54x the speed. It ran at
M=1024 rather than M=513 because at M=R+1 the system is one equation over square and the rung sits
8–13x above its floor (DESIGN §A6 sweep).

**What was wrong, and what is retracted.**

1. **`lsh05` (N=512) died on a gate that no f64 algorithm can pass, not on staging, a config key
   or memory.** It failed `assert resid <= 1e-12` on the sparse-direct reference at
   1.0661e-12. Diagnosed on the local box: the residual *stagnates* (2.486e-12 after the direct
   solve, then 1.064, 1.063, 1.067, 1.069e-12 after one to four refinement steps) because
   evaluating ‖Au−f‖ in f64 has a round-off floor ε‖A‖∞‖u‖/‖f‖ that grows like N², and the
   measured residual sits at a mesh-independent 0.156x of it at **both** N=256 and N=512. The
   1e-12 bound was 5x above that floor at N=256 and 0.13x below it at N=512. **DESIGN §A7**: the
   gate is now `max(1e-12, 1.0·floor)`; it is unchanged at 1e-12 on every mesh already run and
   reachable at N=512 with a 7.4x margin. The reference field, the factorisation and the two
   refinement steps are untouched — the driver's field matches the independent NumPy audit's to
   max absolute difference exactly 0 at N∈{32,64,256}. `lsh03`/`lsh04`/`lsh06` are not re-run and
   nothing they report moves. `lsh07` is the one resubmit; its staged tree differs from `lsh05`'s
   in exactly `lsh_core.py`, `lsh_solve.py`, `config-solve.json` and the inert `stage.py`, with
   all 14 checkpoints and the three shared modules byte-identical.
2. **DESIGN §A6's claim that the free rung gives "byte-identical output" from the K=16 and K=32
   heads is false on the real banks and is retracted (§A9).** With C=I the elimination cancels
   h(z) against R⁻¹QᵀB·h(z) exactly in exact arithmetic, but not in f64: the surviving round-off
   still depends on h(z). Measured across all 32 cases: worst relative field difference
   4.06e-14, worst difference in the reported error 4.86e-17 — the two heads agree on every
   reported digit (0.7790957 % both) but are not bitwise equal. The §A6 smoke ran at N=32, R=32
   where the cancellation happened to be exact. The timed repetitions of one subject *are*
   bitwise identical.
3. **The N=128 "non-dominated reduced model" would have been a misleading headline** and the
   report's generated prose now labels it a cheapness-only membership.

**Also done (coordinator instructions from the previous session, both now implemented).**
The report generator keyed timed subjects by `(mesh, name)` and would have let `lsh06` silently
overwrite `lsh04`'s N=256 numbers. It now keys by `(job_id, mesh, subject)`, groups jobs into
blocks by test-mode count M, **aborts** if two jobs in one block report the same (mesh, subject) —
tested against the real JSONs by passing `lsh04` twice, it names the colliding subject — and
prints M=257 and M=1024 as separate tables with an explicit non-comparability statement
(DESIGN §A10). The validation-versus-development gap now sits immediately below the headline
verdict rather than in a caveat section: worst best-found 3.2–6.8 % on the 32 development sources
against 10.7–16.7 % on the 461 validation sources, a factor ~2.5, so every solved error in the
cell is the optimistic end of a range. No K=64 head was run (the solves had priority).

**Verification.** All three solve jobs pass all 13 checks of the independent NumPy audit
(`lsh_audit_np.py`, imports neither JAX nor any driver), including the non-dominated sets, which
it recomputes and which equal the report's exactly. A 12-check cross-audit of the report against
the raw JSONs (`checks/verify_report_2026-09-17.py`, substituting for Codex, which is quota-locked
until 2026-09-19 11:33 — DESIGN §A8) passes: every summary row is keyed by job and block, every
report value is recomputed from the raw invocations to relative difference 0, every gate passed,
every job was GPU/x64/`highest`/complete.

**Open / next.** (i) `lsh07` (3789568, N=512) was running at hand-off; when it lands, rerun
`reports/generate_lshape.py` with its `result.json` added to `--solve` and re-run the audit and
the cross-check — the N=512 point is where `fom_splu` should be ~35–40 ms against a nearly flat
reduced cost, i.e. where the cost margin is largest. (ii) A 461-source *solve* sweep is the
measurement that would replace the worst-over-32 numbers. (iii) A K=64 head is the remaining
declared lever and one job of the eight is unused. Jobs used: 7 of 8 (lsh01 retracted, lsh02
train, lsh03–lsh07 solve).

**Paths.** `experiments/lshape/DESIGN.md` (§A7 the gate, §A9 the retraction, §A10 the generator),
`experiments/lshape/reports/2026-09-17-lshape.md`, `.../summary.json` (820 rows, each with job id,
test-mode block and source sha), `.../self-audit-2026-09-17-solves.md`,
`.../2026-09-17-lshape-error-cost.png`, `experiments/lshape/checks/verify_report_2026-09-17.{py,json}`,
`experiments/lshape/checks/smoke-a7-reference-gate.json`,
`experiments/lshape/artifacts/lsh0{3,4,6}/` (chunked archives + `result.json` + `audit.json`),
`experiments/lshape/runs/lsh05/archive-failure/` (the failure logs), `experiments/lshape/runs/lsh07/`.
