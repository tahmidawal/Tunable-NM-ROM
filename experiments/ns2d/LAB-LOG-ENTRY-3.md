## 2026-09-17

### ns2d — 2D incompressible NS (INTERIM): the §A4 full-rank rerun ns203 PASSES every Phase-2 gate except H-ORACLE, which fails at the pre-registered bar (POD-16/oracle 1.19 vs 2.0); Phase 3 NOT submitted for K=16; ns204 (K=32) still running

Branch `exp/2026-09-17-ns2d`, worktree `worktrees/2026-09-17-ns2d`, namespace
`/cluster/tufts/paralab/tawal01/ns_20260917/`. Commits `be62a191` (collection + audits),
`8b2bd493` (DESIGN §A6 + report), plus the self-audit addendum. Jobs used: 5 of 8.

**Collected, audited and archived this session.**

- **`ns203` (job 3787319, K=16, R=256, `G_HIDDEN=512`, head 512×3, 100 000 steps; A100
  `pax050`, 4 h 40 m, exit 0, `jax_backend=gpu`, f64, `highest`, `ALL-DONE`, commit
  `beffbb1b`).** The DESIGN §A4 rerun. Independent NumPy audit `artifacts/ns203/audit.json`:
  **67 checks, all match.**
  - **PASS** B-RANKCAP (`g_hidden=512 ≥ R=256`), B-DATA by hash at 64²/128²/256² train+dev,
    **B-ORTH rank 256 = R with κ(R_b)=44.4** at every mesh (was 128 and 2e15 in `ns201`),
    B-FLOOR (bank worst-evolved 0.1408 vs POD-256 0.1303, ratio 1.08; bank floor **median
    0.0122**, i.e. now *below* the POD-256 median 0.0328), H-SOLVED (single-start 0.2231 within
    1.10× of the oracle), H-TRAIN (recon median **0.0503**, was 0.2056 at 30k steps).
  - **FAIL H-ORACLE at every mesh: oracle median 0.2024 vs POD-16 median 0.2402, ratio 1.19,
    bar 2.0** (64²: 0.2034/0.2417; 128²: 0.2026/0.2404). The §A4 whitened-formula
    contamination is gone (`formula_vs_field_worst_rel` ≤ 4.7e-14).
- **`ns202` (job 3783797, K=32, R=512, g_hidden=128)** — completed 17 Sep but never collected;
  collected now and **retracted** as a rank-capped attempt: rank 128 of 512 at every mesh,
  B-FLOOR fails (bank/POD-512 ratio 3.7–3.9), H-ORACLE fails (0.1184 vs POD-32 0.1395, ratio
  1.18). Its audit repeats `ns201`'s six oracle-formula MISMATCH rows (≤1.04e-2), as expected
  for a pre-§A4 run. Kept for the record only.
- Both remote attempt directories deleted after checksum verification; `ns204`'s is the only
  one left on paralab (91 % full).

**The finding, and it is the pre-registered negative (DESIGN §A6).** §A4 stated *before* these
runs that "if H-ORACLE fails again with a full-rank bank and 100k steps, the finding is that
this auto-decoder head cannot beat linear POD-K by 2× on decaying 2D NS at Re ∈ [100,1000], and
the lane reports Phases 1–2 with that negative — it does not lower the bar." That is what
happened, so **Phase 3 was not submitted for the (16,256) head**: its precondition — a neural
manifold closer to held-out data than linear POD-K — does not hold.

**Why it fails (diagnostic, generated from the per-state arrays, not typed).** Not the bank
(floor median 0.012), not the fit (LM median 39 iterations, no budget exits, single-start within
1.10×), not rank, not precision. It is **generalisation**: the head's training reconstruction
improved 4.1× between `ns201` and `ns203` (0.206 → 0.050) while the held-out oracle did not move
at all (0.2028 → 0.2024); held-out / training = **4.0**. Split by output time, the oracle beats
POD-16 by **1.63× at t=0** (0.0258 vs 0.0420) and only **1.19× on evolved times**. The initial
condition is the 12-parameter band-limited field the codes were built for; the evolved states
depend on (12 amplitudes, ν, t) — a 14-dimensional input — and 512 training trajectories give
≈1.6 points per axis. The head memorises the training trajectories and has no neighbours for a
held-out draw.

**Retracted / corrected.** `ns201` and `ns202` are both closed as failed, rank-capped Phase-2
attempts (§A4) and appear in the report only as the record. Nothing from them feeds Phase 3.
No bar was lowered, no re-scoring on training data, sealed cohort still unopened.

**Open.** (1) `ns204` (job 3787320, K=32/R=512, 2R bank, 100k steps) was at 96k/100k steps and
~6 h 30 m of a 14 h allocation at 18:05 EDT; its H-ORACLE verdict is the only remaining Phase-2
question, and its rank-capped predecessor's ratio was 1.18. (2) If K=32 also fails, the lane's
deliverable is Phases 1–2 plus this negative, and whether to spend the remaining 3 jobs on a
*new* pre-registration (a lower-dimensional family, or many more trajectories, to close the 4.0×
gap) is the coordinator's decision, not a rescue of this one. (3) Codex report audit after
2026-09-19 11:33 if the lane is still open.

**Paths.** `experiments/ns2d/DESIGN.md` (§A6), `experiments/ns2d/reports/2026-09-17-ns2d.md`,
`experiments/ns2d/reports/summary.json` (504 rows), `experiments/ns2d/reports/self-audit-phase2.md`
(addendum), `experiments/ns2d/artifacts/{ns101,ns201,ns202,ns203}/`.
