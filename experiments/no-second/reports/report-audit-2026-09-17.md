# Independent audit of the no-second report (2026-09-17)

Auditor: an independent Claude subagent with a read-only adversarial brief (Codex remained
unavailable — quota until 2026-09-19; see `codex-design-audit.md`). It recomputed every rendered
cell from the JSONs and re-derived three arms end-to-end from the raw `.npz` prediction and
target fields without touching `audit.json`, and streamed the FNO lane's archives to verify the
split hashes against that job's own `provenance.json`.

**Result: no BLOCKER, and no wrong number** — every figure in every table traceable and correct,
the selection rule and V1 arithmetic right, no forbidden cross-job ratio, gate G5 satisfied.
The 6 MAJOR and 9 MINOR findings were all about what the prose claimed or omitted, not about the
numbers. Actions taken on each are tabulated in `DESIGN.md` §A6; all were accepted, and all were
fixed in the generator rather than by editing prose. Findings verbatim below.

---

Audit complete. I recomputed every rendered cell from the JSONs, and re-derived three arms end-to-end from the raw `.npz` prediction/target fields (bypassing `audit.json` entirely), plus streamed the FNO lane's 1.5 GB archives to check the split hashes against the FNO job's own `provenance.json`.

**No BLOCKER: I found no wrong number.** Every figure in every table is traceable and correct.

---

### Verification performed (zero mismatches)

Recomputed from the per-case arrays and compared against the rendered markdown: all 8 validation rows × 7 columns, all 12 per-time rows × 6 columns, all 19 cohort rows, all 8 timing rows, all 11 capacity rows, all 6 Poisson rows. Independent raw-field recomputes (NumPy only, from `data/validation/*.npz` + `out/<arm>/*.prediction.npz`):

| checked | report | my recompute from raw fields |
|---|---|---|
| `unet-refine` val mean/median/p95/worst | 1.3523 / 1.0806 / 2.1364 / 7.5176 | 1.3523 / 1.0806 / 2.1364 / 7.5176 |
| `tsol-refine` cohort worst/median/mean | 1.5224 / 1.0763 / 1.1331 | 1.5224 / 1.0763 / 1.1331 |
| `tsol-large` cohort worst | 3.0988 | 3.0988 |
| Poisson `unet-medium` disc med/worst, phys mean/med/p95/worst/>5% | 1.6795 / 6.0161 / 2.2183 / 1.6792 / 5.3355 / 6.0196 / 3 | identical |

Also confirmed: `best.pt` SHA matches; `u0` returned bitwise; boundary exactly zero; ROM 1.8671 / FOM 0.9978 / 3.6398 match `refinement02-diagnosis-audit.json`; FNO Poisson rows match `fno_poisson01/field-audit.json` to 4 dp; timing cells match `out/timing/timing.json` re-medians.

**Q2/Q3 (selection rule + V1):** correct and computed in code, not hard-coded. argmin of validation mean case-max over all complete arms incl. refine → `tsol-refine` (1.9493), `unet-refine` (1.3523), FNO → `fno-large` (2.2811 < 2.3122 < 2.3501 < 2.5968) as §A1 finding 4 requires. V1: worst ratios 1.4600 (tsol) / 1.1779 (unet) ≤ 1.5; median ratios 0.7528 / 0.5985 ≤ 1.5 → **pass**, arithmetically right.

---

### Findings

1. **MAJOR — the report references a controls section that does not exist.** Line 67: "the float64 precision **control below** tests whether the network precision matters"; line 8 "one seed except where a seed control is shown"; line 190 "(plus the seed control where present)". `runs/ctrl01/` has no `audit.json`; `checks/submission-ctrl01.json` shows job `3783831` submitted 13:31 UTC, report generated 13:32 UTC. *Fix: drop the forward reference and say the precision/seed control job is in flight, or regenerate after ctrl01 lands.*

2. **MAJOR — line 194 is false about the report's own content.** "Runs ended by the wall budget were still improving or plateauing at that budget; the report says which, per arm, in the 'Ended by' column." The column reads `wall budget` for all eight arms and distinguishes nothing. Applying the generator's own last-5% test (`best_epoch ≥ 0.95·(epochs−1)`) to Burgers: 7 of 8 arms were still improving (only `tsol-refine`, 1542 vs 1545.7, fails) — yet the "lower bound, not a converged result" caveat is given **only** to Poisson. *Fix: run the same test on the Burgers arms and print it per arm, or delete the sentence.*

3. **MAJOR — pre-registered criterion V1-P (§A2) is never evaluated and no Poisson arm is named as selected.** Recomputing it myself: selected U-Net = `unet-medium` (argmin discrete mean 2.2182% vs 2.2383 / 2.9501); vs `fno-large` the worst ratio is 0.2049 and the median ratio 0.8291 → V1-P **passes** comfortably. A pre-registered verdict that would have been favourable is simply missing. *Fix: add the V1-P verdict, computed in `generate_report.py`.*

4. **MAJOR — §A1 finding 17(d) disclosure is absent.** Equal wall in float32 bought the new families far more optimizer steps than the float64 FNO: `unet-refine` 1962 epochs vs `fno-large` 692 (2.84×), `unet-small` 2327 vs `fno-small` 1741. The report gives **no FNO epoch or parameter counts at all** — the capacity table lists only this lane's 8 arms — while §A1 pre-registered that this advantage be "stated in the report". *Fix: add FNO rows (params, epochs, best epoch, stop reason) to the capacity table and state the epoch advantage explicitly.*

5. **MAJOR — the timing table pools two jobs under a "Same-job" heading.** Rows `3780139` (tsol) and `3780138` (unet) sit in one table (lines 177–184); reading `unet-medium` 5.898 ms against `tsol-large` 15.326 ms is exactly the cross-job comparison the caption forbids, and the caption only warns about the FNO. *Fix: one table per job, or add "U-Net and Transolver rows are different allocations; do not compare across them".*

6. **MAJOR — line 217's FNO contrast is untraceable and hand-entered.** "The Poisson FNO, by contrast, early-stopped inside the same cap" is a hard-coded string in `generate_report.py`; `fno_poisson01/field-audit.json` contains no `epochs_completed` and no stop reason (only `best_epoch` 280/376/415). `summary.json` likewise hand-enters `stop_reason='early_stopping'`, `budget_s=7200`, and `FNO_POISSON_PARAMS = {1192801, 5779729, 17876673}`. This contradicts "no number here is typed" — and it is the claim that makes the U-Net's Poisson numbers a *lower bound relative to a converged FNO*. *Fix: source it from the FNO archive or mark it explicitly as an uncited statement from §A2.*

7. **MINOR — apples-to-oranges parenthetical in both verdict bullets.** "worst 1.5224%, median 1.0763% (FNO 2.4829%, ROM 1.8671%, …)" — 2.4829 is the FNO's cohort **worst**, but it is placed after a worst *and* a median. On the 8-case median the FNO (1.0494) beats both new families (1.0763, 1.2193). In the table, not the prose. *Fix: quote "FNO worst 2.4829 / median 1.0494".*

8. **MINOR — the flattering clause leads.** "…**pass**; it is in fact below the FNO on mean, median" is adjacent to a `tsol-refine` worst that is **1.46× the FNO's** (9.3183 vs 6.3825) — the second-worst tail in the whole table — and V1 passes with only a 2.7% margin against the 1.5× bar. *Fix: state the worst ratio numerically in the verdict.*

9. **MINOR — the Transolver sweep is unbracketed and the fairness caveats are missing.** The selected Transolver capacity is the **smallest screened** (`refines_capacity: "small"`, dim 128; capacity scores monotone 2.006 < 3.295 < 3.341), so the optimum lies at or below the edge of the range. §A1 17(a) (3.1–12.3 M vs the FNO's 1.2–17.9 M) and 17(b) (lr/wd/patience never re-tuned per family; the Transolver's own OneCycle/wd-1e-5 recipe unused) are disclosed in DESIGN and **nowhere in the report**. *Fix: add a two-line fairness paragraph.*

10. **MINOR — winner margins are inside plausible seed noise, single seed.** `tsol-refine` beats `tsol-small` by 2.9% of the mean (same capacity, lr only); `unet-refine` beats `unet-medium` by 6.1%. No seed control has returned. *Fix: say the family selection is not separated from seed variation yet.*

11. **MINOR — §A2 says "all 206 train/validation files"; the job verified 196** (`data_verified=196`; `DATA-tv.sha256` = 128 train cases + 32 val + 32 physical-reference + 4 index/stats). One of the two numbers is wrong. *Fix: correct §A2 to 196.*

12. **MINOR — the metric flatters late times and the caveat is absent.** The denominator is ‖u_ref(t₀)‖. Over 8 sampled validation cases the true field decays to mean 0.536× (min 0.279×) of its initial norm by t=0.25, so late-time percentages read ~1.9× (up to 3.6×) smaller than a conventional per-time relative L2. Identical for every method — comparisons are fair — but "1.35%" is not a per-time relative error. Also `max` runs over k=0…5 with k=0 identically zero by construction. *Fix: one sentence under "How accuracy is defined".*

13. **MINOR — the reference is described more strongly than its own record.** The dataset index labels it `physical_accuracy_status: "empirically calibrated on independent development cases; not a per-case continuum certificate"`, `worst_empirical_margin = 9.59e-4`. The report says only "the Burgers lane's refined 4096-interval … solution". *Fix: quote the empirical margin.*

14. **MINOR — the two Poisson metrics are not independent evidence.** The index records `target_relative_to_reference = 6.53e-05`, i.e. the discrete target and the 2048-interval reference agree ~340× better than the model error; the discrete and physical columns duplicate each other to <0.001 pp. Six columns imply more than there is. (Conversely: this positively rules out the known `poisson-analytic-data-inconsistency` failure mode here — the target has discrete residual 7.7e-13.) *Fix: say the physical column is confirmatory, not a second measurement.*

15. **MINOR — three different commits, unflagged.** Burgers `c4f8b045`, Poisson `339c026b`, pending control `cba4e4a4`. The report prints them but never notes the Poisson section's code differs from the Burgers section's. *Fix: one clause.*

16. **NOTE (Q7 — audit integrity: sound, and I could not break it).** `audit.py` re-loads every case and prediction `.npz`, re-hashes each case file against its index row, recomputes the metric in NumPy, and asserts agreement with the driver's declared errors at `rtol=1e-11`; it also asserts `u0` bitwise, boundary exactly zero, finite float64, `best.pt` SHA, arm-set equality with the spec, no signal stop, and the protocol constants. The load-bearing guard is lines 171–175: `history[best_epoch].mean_case_max == batched_selection_score` **and** |that − batch-1 mean from the saved fields| ≤ 1e-5·mean (float32) / 1e-11 (float64) — this is what anchors the saved fields to the *best* checkpoint; without it a final-epoch dump would pass silently. Measured gaps are 0.6e-8–5.7e-8 absolute, i.e. **5.7e-9 to 2.5e-6 relative** — 4 to 1700× inside the bar, so the §A1-finding-1 tolerance was not stretched post hoc. Residual holes, both small: the batch1/batch8 check bounds only the *mean* (a corrupted tail could in principle hide), and Poisson `physical_candidate` is computed with no declared counterpart to disagree with.

17. **NOTE (Q7/Q8 — split hashes really are the FNO's; I verified outside the audit).** Streaming `runs/fno_burgers02/archive-parts`: the FNO job's `out/fno-large/provenance.json` records `train 5333584b… / validation 468b9e70…`; `runs/fno_poisson01`'s records `d20a5994… / 65cc277b…` (with `epochs:500, patience:80`). The no-second jobs' own `data/validation/index.json` hash to `468b9e70…` / `65cc277b…`, and their archived `DATA.sha256` / `DATA-tv.sha256` list `train/index.json` at `5333584b…` / `d20a5994…`. Both splits are genuinely the FNO jobs'. Weakness: `audit.py` asserts these against **literals typed into the file** (lines 25–31) — only the cohort hash is read programmatically from the FNO audit. *Fix: assert the train hash from the archived `DATA.sha256` line instead of a literal.*

18. **NOTE (Q6 — G5 satisfied).** Spec arms {small, medium, large}+refine == audited arms for `unet01`/`tsol01`; {small, medium, large} for `pois02` (no refine, per §A2); `audit.py` asserts set equality and `not stopped_by_signal`. Every arm — including the bad ones (`tsol-medium` 3.2950%, `unet-small` 12.80% Poisson worst) — appears with params, epochs, best epoch, training s, budget and stop reason. Nothing silently dropped except the still-running control (finding 1).

19. **NOTE (Q5 — no forbidden ratio).** Grep confirms the only "×" occurrences are the pre-registered 1.5× V1 bar, kernel/pool sizes, and the "32×30" repetition count. No speed, cost or efficiency claim anywhere; the cohort table carries no time column; the timing table is ratio-free. The only structural risk is finding 5.

20. **NOTE (Q9 — first reviewer attacks, beyond the above).** (a) 128 training cases, one seed, one mesh, one Gaussian family — "U-Net beats FNO" is a weak positive claim at that sample size, though the preamble does mark it provisional. (b) Selecting on the mean cost the tail 90% (`unet-refine` worst 7.5176 vs `unet-medium` 3.9622) — disclosed, and correctly. (c) `tsol-refine` and `tsol-small` are the *same* network at two learning rates, so the Transolver effectively has three distinct capacities, not four.
