## 2026-09-17

### ns2d — 2D incompressible NS: Phase 2 CLOSED for both heads — ns204 (K=32, full-rank bank) also FAILS H-ORACLE (POD-32/oracle 1.15 vs bar 2.0); Phase 3 NOT submitted; the lane's deliverable is Phases 1–2 plus a gated negative

Branch `exp/2026-09-17-ns2d`, namespace `/cluster/tufts/paralab/tawal01/ns_20260917/` (now
empty). Jobs used: 5 of 8 (ns101, ns201, ns202, ns203, ns204). DESIGN §A7.

**ns204 (job 3787320, K=32, R=512, `G_HIDDEN=1024`, head 512×3, 100k steps; A100 `pax049`,
7 h 30 m, exit 0, `jax_backend=gpu`, `ALL-DONE`, commit `beffbb1b`).** Collected with
checksums, audited (`artifacts/ns204/audit.json`: **67 checks, all match**), chunk-archived,
remote dir deleted.

- **PASS** B-RANKCAP, B-DATA by hash at 64²/128²/256² train+dev (on `pax049`, the node whose
  256² hash mismatched in ns201 — not reproduced, still unexplained), **B-ORTH rank 512 = R,
  κ(R_b)=141**, B-FLOOR (bank worst-evolved 0.0756 vs POD-512 0.0600, ratio 1.26; bank floor
  median **0.0040**, below the POD-512 median 0.0115), H-SOLVED (0.1270, 1.05× oracle),
  H-TRAIN (recon median **0.0376**).
- **FAIL H-ORACLE at every mesh: oracle median 0.1209 vs POD-32 median 0.1395, ratio 1.15**
  (64²: 0.1218/0.1407; 128²: 0.1211/0.1396; bar 2.0).

**The finding.** Both pre-registered heads, with the §A4 full-rank bank and the 100k-step
recipe, fail the H-ORACLE bar: 1.19 (K=16, ns203) and 1.15 (K=32, ns204). Per §A4 that is the
negative finding; the bar was not lowered, nothing was re-scored on training data, the sealed
cohort is unopened, and **no Phase-3 job was submitted**. The lane's deliverable is the
certified FOM (28/28 gates), the hashed dataset, the bank/head floors, and this negative.

**Diagnostic (generated, not typed).** Same mechanism as ns203, sharper: held-out / training
= **3.2** (0.1209 vs 0.0376). Per output time, POD-32 / oracle = **0.82 at t=0** (POD-32 0.0189
beats the oracle 0.0230) and 1.16 on evolved times. At K=32 the 12-parameter initial family is
already inside the span of 32 linear POD modes, so the neural manifold buys nothing even at
t=0; on evolved states the head memorises the 512 training trajectories (recon 0.038) and does
not generalise to held-out draws of a 14-dimensional (12 amplitudes, ν, t) input. The bank
(floor median 0.004) is not the limit.

**Retracted / corrected.** Nothing new; ns201/ns202 remain retracted rank-capped attempts.

**Open.** Whether a lower-dimensional family (e.g. 3 modes / 6 amplitudes) or a much larger
training cohort would close the 3–4× held-out/training gap — a new pre-registration, 2–3 jobs,
coordinator's decision. Codex report audit after 2026-09-19 11:33 if the lane is reopened.

**Paths.** `experiments/ns2d/DESIGN.md` (§A7), `experiments/ns2d/reports/2026-09-17-ns2d.md`,
`experiments/ns2d/reports/summary.json` (631 rows), `experiments/ns2d/reports/self-audit-phase2.md`
(addendum 2), `experiments/ns2d/artifacts/ns204/`.
