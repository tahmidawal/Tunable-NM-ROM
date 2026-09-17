# Round-1 adversarial review of commit 8dd88495 — disposition

Fixed at commit `20c10e8b` unless stated. "Fixed" = the text or generator changed and the
claim now traces to a JSON field; "disputed" = the reviewer's reading is wrong and the field
that shows it is named; "deferred" = not in this draft and said so in Limitations.

| # | finding | disposition |
|---|---|---|
| 1 | page budget | **fixed** — main text ends on p. 9 (reproducibility statement opens p. 10); T1/T4/T7/T11b/T14 moved to Appendix D, method §3.3–3.5 compressed, results and intro cut |
| 2 | placeholders in main text | **fixed** — every `\gen{}` now lives in one "Pending cells" paragraph of Limitations; no result sentence depends on a pending lane; `**[PENDING: …]**` kept in PAPER.md by instruction |
| 3 | double-blind | **fixed** — every reference to the prior submission, reviewer IDs and Appendix D removed from LaTeX and PAPER.md; DISCREPANCY comments stripped from the sources; A.6 rewritten as neutral statements of what the method is; the reviewer map stays private in `REVIEWER-RESPONSE-MAP.md` |
| 4 | heat claimed, not tabulated | **fixed** — T11d generated from the 2026-09-10 heat linear-bank report (job 3511417, A100-PCIE-40GB, commit 73fdaa88); §5.6 and T1 carry it, labelled an earlier cell of the same decoder family |
| 5 | operator premise pre-committed to withdrawal | **fixed, and the clause fired** — job 3787189 makes `fno-large` R-USABLE (rung 128: 2.12× own speed at 1.14× own error; `same_job_speedup_vs_own_256`, `worst_evolved_error_ratio_vs_own_256`), so "one accuracy–cost point per trained operator" is withdrawn from abstract, intro and results; the abstract's first two sentences now read "A learned PDE surrogate is fixed once trained: what it can represent cannot be changed at inference …" — **this changes user-agreed abstract text and needs the user's sign-off**; T14c and the verdicts are generated. **Round-2 update (2026-09-17, review B1):** the rebuild from the old source had restored the old premise sentence in the abstract, intro (twice) and conclusion; all four now read that operators give one point per trained model and that the evaluation grid moves cost but not what the model can represent, which is exactly what T14c supports (fno-large usable, U-Net and Transolver degenerate). Truth over the old wording; recorded in the ledger. |
| 6 | generator keyed on arm name | **fixed** — `meta`/`by` keyed on `(arm, job_id)`; T14 now shows the Burgers jobs and epochs; `\nOpFnoImproving` = 4 of 4, `\nOpLaneImproving` = 7 of 8; every sentence in §5.7 and §6 re-read |
| 7 | wave prose vs table | **fixed** — §5.6 now says the top rung is not strictly the most accurate (`D1_strict` = no) but matches the q=32 rung within the pre-registered tie band (`tie_band_delta` = 0.020 pp at 256²; measured gap 0.017 pp); monotonicity is the generated flag `H_mono_ladder` |
| 8 | cross-job cost inference | **fixed** — §5.7: the FNO dominates the ROM on both axes in the same job (3780638); U-Net and Transolver beat it on accuracy and were not timed against it; no cost statement for them |
| 9 | Contribution 2 asserted | **partly fixed** — (a) the skip is now stated as a design choice with its reasoning and "we do not ablate it" in Contribution 2, §3.1 and Limitations; (b) the block-damped claim now cites T19 (job 3734098: joint LM at q=64 leaves 6 budget exits, block-damped 0, plain variable projection 15× slower; at q=128 varpro 297 budget exits vs 0); (c) the rank claim is reworded to the validity count the elliptic solver records (`valid_count`, the "valid" column of T11b). **Deferred:** a with/without-skip ablation job, if the coordinator wants it run |
| 10 | structural condition confounded | **fixed as reworded, deferred as an experiment** — Contribution 3 now "where the control is not worth having"; §5.6 states the confound with the bank: POD-512 0.1838 % vs bank floor 0.7421 % on Poisson, POD-64 1.50 % vs bank 5.12 % on waves, while on Burgers free bank 0.6027 % vs POD-512 0.6125 % (all-times, panel job) are comparable. The POD-bank ladder on Burgers is not run |
| 11 | PDE specification | **fixed** — T01b (Appendix D) gives every sampled family verbatim from the generator sources: Burgers ν ~ logU(0.01, 0.1), Gaussian IC ranges; Poisson source family; heat κ = 0.02; wave bump family and c ~ U(0.85, 1.15); each row names its source file and function |
| 12 | harder problems not delivered | **deferred** — stated in Limitations ("nothing harder than viscous Burgers is solved"); L-shape solve jobs and NS phases 2–3 pending |
| 13 | no anonymous code | **deferred to the user** — reproducibility statement now promises an anonymised repository with the generator, provenance registry, lane summaries and audit JSONs at submission; the push is the user's decision (`tools/mirror-code-only.sh` exists) |
| 14 | selective fixed-M reporting | **fixed** — T4 shows both ladders; §5.2 states M=256 spans 1.22× / 2.27× and fails, M=1088 passes, and that M=1088 was chosen after the crossed grid by the lane's rule (largest fixed M holding every rung to q=256), not pre-registered; Limitations repeats it |
| 15 | metric-dependent head claim | **fixed** — §5.4 makes the matched-dimension result the claim (head 2.5629 % vs best linear 56.9296 % vs POD-16 61.6503 %); "no POD rank up to 128" now carries both metrics (all-times POD-128 10.1198 %; evolved POD-128 1.9464 % vs q=0 1.8890 %, POD-256 0.7109 %) |
| 16 | prior art on hybrid reduction | **fixed** — new related-work paragraph (Barnett–Farhat–Maday 2023 JCP 492:112420; Fresca–Manzoni 2022 CMAME 388:114181; Peherstorfer–Willcox 2015 SISC 37(4); Carlberg 2015 IJNME 102(5)), all verified against publisher/dblp records; what is new stated narrowly (nested prefix chosen offline, run-time selection without refit, constant-projector elimination at q=R, one-allocation comparison) and what is not |
| 17 | offline cost | **fixed** — T17 from the lane JSONs: head training 387 s (like-for-like) / 1459 s (selected arm), direction fit 1242 s, NNLS per certified rule 72–2920 s, primary ladder total 1.7 h; the incumbent's own training time is not recorded and the table says so |
| 18 | no results figure; Fig. 1 illegible | **fixed** — Fig. 1 is now the family (three panels from b-qxm / b-panel / b-eqtop, filled = converged); the architecture diagram is in Appendix B at full width with the comparator row narrowed |
| 19 | overfull tables | **partly fixed** — T14 reduced to nine columns in `\tiny`; T1 columns narrowed; `main.log` should be re-checked for remaining overfull boxes in the appendix tables (T05, T09c are wide by nature) |
| 20 | two "solved" numbers for one cell | **fixed** — §5.6 states the M=129 solve (3.1495 %) and that the M=257 solve in Table 7 (3.1146 %) differs in the fourth digit |
| 21 | provenance table defects | **fixed (a)** wave row now prints GPU and commit correctly; **partly (b)** the panel row prints "incumbent (gate checkpoint_unchanged; hash in T2, tuning row)" because the panel's own records carry the gate but not the hash — the hash 18f0266a… is in the tuning row; **(c)** the T15 Markdown parse, the typed b-speed job ids and the transcribed T1/T01b constants are declared in WRITING-STATUS and in the table headers |
| 22 | origin of the 0.116 bar | **fixed** — §3.4 states it (the held-out ρ of the incumbent q=0 rule at the state carrying the first-interval penalty, measured in q-diag and adopted in the q-ridge design before any certification job; tight bar declared before b-eqtop) |
| 23 | cross-job spread calibration | **fixed** — §5.1 and Limitations print the anchor spread (q0_M1088: 742.2 vs 848.0 ms, 14 %, jobs 3780175/3780177) from `analysis.json['anchors']`; q256 spread across three jobs is 12 % (macro `\nSpreadQtwoFiftySixPct`) |
| 24 | L-shape floor cohort | **fixed** — §5.6 prints both cohorts (development 0.6650 %, selection 1.6155 % for the selected bank) |
| 25 | completion rule changed between jobs | **fixed** — §5.1 says the rule was pre-registered for the panel and post-dates the head-ablation job |
| 26 | certification claim stronger in related work | **fixed** — both places now say "at the top rungs (q ≥ 128)", and §5.3 records the q=64 counter-example |
| 27 | 36 vs 35 subjects | **fixed** — the count is the macro `\nPanelSubjectCount` (36, from the JSON); the ledger is stale, not the paper |
| 28 | matched-dimension result buried | **fixed** — in Contribution 1 and the abstract ("beats the classical linear reduced model at matched latent dimension" was already there; the numbers are now in Contribution 1) |
| 29 | "0.18–0.21×" undersells | **fixed** — "about 5× cheaper at four-decimal-identical error" in Contribution 1 and §5.1, with the per-rung ratios |
| 30 | certification scatter | **fixed** — Fig. 3 (`figures/gen_fig_eq_certification.py`): NNLS fit residual vs held-out ρ_max over the 97 rules with both bars; 24 rules fit below 1e-3 and still fail the primary bar; labelled provisional |

## Disputed

None of the numerical findings is disputed. One reading is qualified: finding 15's "on the
evolved metric POD-128 = 1.9464 % against the head's 1.8890 %" compares POD-128 with the
q=0 rung at M=64 in the panel job (`q0_M64_dense_g1em06`, `worst_evolved_percent`); at M=256
the q=0 rung is 1.2710 %. The paper quotes the M=64 rung, which is the head-ablation
configuration, and says so.

## Not yet done from the coordinator's list

- Item 13 (skip ablation run) and item 10(ii) (POD-bank ladder on Burgers): not run; both
  are stated as choices/limitations rather than results.
- Remaining overfull boxes in appendix tables (finding 19) to be checked in `main.log`.
