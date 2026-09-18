# Round-2 review disposition (2026-09-17)

Review: `review-round2.md` (score 3/10 at `d131ec11`). One line per finding; status = resolved / partial / open; commit on branch `exp/2026-09-16-paper-refresh`.

| id | finding | status | commit / note |
|---|---|---|---|
| B1 (blocker) | operator premise contradicted by §5.1 in abstract, intro ×2, conclusion | resolved | `d960b7c3` — "at deployment only the evaluation grid can be changed, which moves cost but not what the model can represent"; round-1 disposition row 5 updated |
| R2 | vs-reference error never confronted; 2.44× knob = 1.13× physical | resolved | `60382691` — main-text ladder table with vs-ref column; three sentences; abstract qualifier; limitation (v). Generated ratio: rungs are 1.00–1.13× the discretisation error (slightly above, not below); the prose says so |
| B4 / M2 / R4 | L-shape: linear residual, expensive comparator, POD-128 cheaper | resolved | `e63e9bf7` — no-fast-transform framing, CG named, margins vs cheapest FOM (6.07× at 512²), POD-128 margins printed, "nothing here is a neural-manifold win", main-text four-mesh table |
| B5 | structural condition violated by the L-shape | resolved | `e63e9bf7` — condition restated as a trade *between rungs*; cost advantage separately conditioned on the absence of a fast transform |
| M6 | validation caveat attached to the wrong quantity | resolved | `e63e9bf7` — best-found reconstruction degrades 2.5×; no validation solve sweep |
| M7 | cross-job head trend | resolved | `e63e9bf7` — marked cross-job; head rows printed at every mesh |
| B2 / R5 | abstract splices two ladders; knob bar 2 of 3; incumbent favourable draw | resolved | `9ae3d26f` — ladders separated; 2 of 3 = pre-registered pass; seed 3 convergence failure + spans; incumbent beats every fresh seed at q=16–128, in §6.3 and Limitations |
| B3 | quadrature status labels blank / "certified in one draw" / None | resolved | `76e649e4` — one vocabulary at the generator source across all rule tables; transferred rules carry their 256² verdict; abstract "confirmed on re-draw at q≤32 only" |
| R3 | double-blind: prior source and reviewer map inside paper/; title | partial | `362e74ad` — files moved to private/, release plan stated. **Open**: the title is the user's two-word edit of the prior title (user decision, not changed); the abstract's opening premise was re-worded for B1 but its shape is inherited |
| R1 | one figure, zero tables in the main text | resolved | `284e1d24` — merged ladder table, panel summary, L-shape table, seeds verdict table in the main text; main text ends on p. 9. **Partial**: Figure 2 stays in Appendix B (does not fit) |
| M1 | 4.48× → 1.82× across jobs/GPUs/arm types | resolved | `284e1d24` — two independent same-job facts with GPUs named, no arrow, not a crossover |
| M3 | M=1088 "not pre-registered"; bar text two parts code | resolved | `f7a55667` — M=1088 pre-registered (DESIGN §3/§6); bar quoted as DESIGN states it, generator clauses labelled "as implemented" |
| N1 | three typed bars | resolved | `69a008b7` — `bars.json` with DESIGN sources, read by the generator, sha in provenance |
| N2 | device vs complete-query ms undeclared | resolved | `69a008b7` — every cost column headed "device ms" or "complete-query ms"; §5 defines both; wave ratios labelled device ms |
| N3 | provLshapeJob wrong job | resolved | `69a008b7` — keyed on the printed bank/head rows (3783786) |
| N4 | 1.033× is the device ratio | resolved | `69a008b7` — complete-query ratio 1.52× generated from the printed host rows; device ratio kept as a separate macro |
| N5 | eight word-form numbers | resolved | `69a008b7` / `284e1d24` — macros for the bank/POD ratio, "four times the bar", cohort size; cohort counts replaced by a table reference; q lists as macros |
| N6 | `None` in printed tables | resolved | `76e649e4` |
| N7 | nine uncited bibitems | resolved | `f1ecdf8a` |
| N8 | b-qxm read unpinned | open | tree is clean at b4e38103 and provenance records the sha; a pin can be added on request |
| B6 | 1024² statement qualifications | resolved | `284e1d24` — all-times 0 of 20 stated; provisional until bpn401; ratios as independent facts |
| B7 | "every operator number is a lower bound" false; two FNO errors unexplained | resolved | `284e1d24` — "all but one"; the six-case panel cohort named beside the FNO's 7.42 % |
| B8 / #10 | tab:ns caption stale; upper-bound oracle only in caption | resolved | `f1ecdf8a` / `284e1d24` — caption rewritten for both heads; "oracle values are upper bounds" in §6.5 |
| #2 | tab:sealed pending box | resolved | `f1ecdf8a` (caption stated the job was running) → sealed cohort folded in at lane commit be9415ab: T13 + T13b generated from job 3804465, pending box gone, sealed values are the scheduled ladder's headline |
| #19 | overfull boxes, Table 24 too large | partial | `f1ecdf8a` — T09c split, T01b/T14/T16 wrapped; the remaining overfull boxes are inside T01b's transcribed-formula column and the TikZ figure (≤ 150 pt, inside resizebox) |
| C3 | Contribution 2 asserted, not shown | open | no skip ablation was run; the claim says "design choice, not ablated" |
| 2.5 | parabolic leg thin (legacy heat cell) | open | title kept by user decision; heat stays a supporting appendix cell with its job named |
| 4.4 | readability (abstract length, undefined terms, reading box) | partial | abstract shortened and split into one-claim sentences; no "reading this paper" box (page budget) |
| ns301–ns304 | register | resolved | `f1ecdf8a` |
| M1 / B6 follow-up | 512² panel brackets the crossover | resolved | bpn401 (job 3805065) folded in: 0 of 31 at 512²; 4.48x/2.92x on shared hardware, 1.82x on H200 separate; flip between 512² and 1024² |
| B8 follow-up | NS head-only data-scaling diagnosis (ns301) | resolved | folded in at ns2d f63de724; verdict 'ambiguous' kept; T11f |
| B8 follow-up 3 | NS lower-dimensional family (ns303) | resolved | folded in at ns2d 5ea1cc30; T11e gains a family-dimension-8 row (K=16/R=256, same recipe); one sentence in §6.5 plus the cell-level statement (K=16/32, family dimension 14/8, 128–512 trajectories: never the pre-registered 2x); ns302 kept in the register as the last arm; the family dimensions 14→8 are parsed from the lane's DESIGN §A12, not typed |
| B8 follow-up 2 | NS exploratory q-ladder (ns304) | resolved | folded in at ns2d 46650a1e; T11g/T11h in the extended-results appendix, two sentences in §6.5, all labelled exploratory after a failed phase-2 gate; POD-LSPG more accurate at every rung and cheaper at 5 of 6 (not 'every rung' as the message said — at q=512 both cost about the same); no neural rung non-dominated |
