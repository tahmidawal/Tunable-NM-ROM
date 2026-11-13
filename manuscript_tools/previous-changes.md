# Revision change log (ICLR 2027 submission 49310)

Source: `main.tex` (the submitted text is still in `submitted.tex`, untouched). Build: `latexmk -pdf -interaction=nonstopmode main.tex`. The build has no errors, no undefined references and no overfull boxes. The PDF has 36 pages (the submission had 23).

**Main text: 9 pages**, the same as the submission. The conclusion ends on page 9, and the ethics, reproducibility and AI-use statements start page 10, as in the submission. There is no slack: the conclusion's last line is the last line of page 9.

Every changed or added passage, table and figure caption is set in blue, using `\rev{}` and the `revblock` environment defined at the top of `main.tex`. A footnote in the abstract on page 1 says so, and also says that deletions and passages moved to the appendix are not marked.

Numbering in the revised PDF:
- Table 1 is G1, harder Navier–Stokes.
- Table 2 is G2, cylinder.
- Table 3 is the submission's Table 1, now with matched comparators.
- Table 4 holds the neural-field ROMs and the matched-data operators.

The responses say "Table 1" for the submission's table. In the revision that table is **Table 3**, so the authors may want to add one line to the discussion thread mapping the names.

Responses used: `resp-*.md` as corrected twice by the coordinator on 2026-11-12. The last read was after the second correction.

## Promise → where it is delivered

| Promise (response) | Delivered in |
|---|---|
| Revised claim, win-first (general "claim of the revised paper") | Abstract; introduction ¶2–3 and contributions 1–3; conclusion |
| Remove "speedup grows with mesh", "POD rebuilt per mesh", "up to 27.5×", "empirical quadrature" for the Burgers rows, "more accurate than POD-LSPG" | Abstract and introduction rewritten. §3.3 now names the 63×63 lattice. Old contribution 1 and the old §5.2/§5.3 claims are removed. The POD-rebuilt clause is gone from the introduction, the related work and Appendix B. |
| Table 1 with matched comparators, fast column labelled test-selected, timing spread | Table 3 (§5.3) gives each setting against its own matched rung, with Fast† marked "selected using the test errors". Spread is in Table C.6. |
| Box Poisson and heat as correctness checks | Appendix C.3, Table C.5: matched fast column plus transform-solver ratios |
| Table G1 (harder NS, 3 seeds, per-mesh POD bank) | Table 1, §5.1. Timing spread is in Table C.7; setup is in Appendix E.1. |
| Table G2 (cylinder Re 105), long window, out of band as a limit | Table 2, §5.2. The long window and out-of-band results are in Table E.2; the comparator is described in Appendix E.2. |
| Coordinate bank vs per-mesh POD on the harder family; reused 64³ head 1.05/1.06 %; interpolated POD 1.10 % | §5.1, last paragraph, and Table E.1, which carries the disclosures (fallback rule; cohort seen once before) |
| Neural-field baselines (CROM-style, CoLoRA-style) | Table 4 and §5.4. Model details and a disclosure are in Appendix F.1. |
| Matched-data Burgers operators replace the operator claim | Table 4 and §5.4. All seeds, the 576-trajectory rows and 1024² zero-shot are in Table F.1. The old operator table is now Table B.1, captioned "kept for the record, superseded; heat rows support no claim". |
| FNO on harder NS and cylinder | §5.4. Appendix F.2 and Table F.2 hold the six cylinder operators. |
| Ablations: head vs linear/quadratic closure, quadrature | §5.5 and Table F.3 (also span-128 vs head) |
| POD controls (interpolated POD; POD + ECSW) | §5.5 and Appendix F.3 |
| Nine figures | Appendix G, Figures G.1–G.9, which are F1, F2, F3, F4a, F4b, F4c, F5, F6 and F7 copied to `figs/`. G.1 (error vs cost) is cited in contributions and §5.5. |
| Cost table with offline cost and break-even (iMp8 Q1) | Table C.4, with offline components in Appendix C.2. Figure 1(a) points to it. |
| Bank loss as numbered equations, two forms, per-problem weights (iMp8 Q2) | Appendix C.1: eqs. (21) joint, (22) variable projection, (23) head loss, and Table C.3 with weights per problem |
| A.6 fixed 63×63 lattice | Appendix A.6 correction sentence; §3.3; Table C.10 now says "lattice" instead of "EQ" |
| Fourier frequencies trained (Burgers 2D, Poisson 2D) | §3.4 after eq. (11); Table C.1 note; Table C.3 |
| Interpolated POD transfers as well on periodic boxes | Introduction ¶3; §5.5; Appendix F.3 |
| Original NS family 0.63/1.65/0.94× | Contribution 3; §5.3 |
| Sine-transform faster on box Poisson; transform solvers on Poisson 3D, heat 3D; heat 2D 1.1–1.4×; pseudo-spectral Burgers | §5.3; Table C.5 |
| Test-set sizes 16–128; development tables labelled | §4 Accuracy. Table B.3 is labelled "Development table". Table C.11 (iterations) was already labelled. |
| Limitations: out of band, steep Burgers failure, one NS family and horizon, Galerkin cylinder with POD the same | §5.5 Limitations (i)–(vi); Figure G.8 |
| Figure 1 redrawn with offline and online panels | §2, Figure 1 (TikZ, two framed panels) |
| Mesh independence stated exactly; assembly charged | Introduction ¶3; §3.4; Figure 1(a); Table C.4 |
| Cylinder boundary factor (distance factors) plus lift | §3.1; §5.2; Related Work; Appendix E.2 |
| Coefficients exact after freezing; at query time from the PDE solve | §3.5; Appendix C.1 |
| Linear-skip "guarantees" wording | §3.4 now says "does not guarantee convergence" |
| Seeds (Burgers retrains; settings marked) | Appendix C.4, "Seeds"; § marks in Table D.1 |

## Could not deliver, or delivered with a deviation

1. **Timing spread "in every table".** Spread is given for Table 3, the box table, Table 1 (G1), Table 2 (G2, in the caption) and the coordinate-network and operator tables (min–max where stored). It is **not** given for Table 4 (neural-field and operator times), Table B.2/B.3 or the cylinder operator times, because I did not extract those records. Heat matched-FOM percentiles at 64³–256³ are marked "not extracted".
2. **The nine figures are all in the appendix.** The main text has only Figure 1, because there is no room at 9 pages. Reviewer iMp8 W5 criticised exactly this, so the authors may want to swap one figure in for text, for example F1.
3. **No figure covers the two new headline problems with held-out data.** There is no harder-NS figure, and no cylinder Re 105 figure.

## Figures built from older numbers or other cells

These need an author decision, or regeneration.

- **F4c (Figure G.6, cylinder fields)** was drawn from the *first retrained bank of the earlier campaign*. That bank is less accurate than POD (2.09 % against 1.17 % at the frame shown), which contradicts "a POD basis of equal width performs the same" for the bank of Table 2. I kept it, with a caption that states this. **Recommendation:** regenerate it from the Table 2 bank at Re 105, or drop it. Dropping it leaves eight figures against the "nine" promised.
- **F7 (break-even)** shows 1024² cells: Poisson 6,084/7,049, heat 17,989/43,580, Burgers 201,230/824,486 excluding data generation. The response's Q1 table uses the largest meshes (159 / 1,417 / about 20,000). Both are consistent, but the figure does not show the response's cells. Its caption points to Table C.4.
- **F1 (error vs cost)** uses development cohorts, the original problems only, and 1024² or the original NS family at 96³.
- **F3 (matched speedups)** compares the NS cells against same-mesh CNAB2 only. The caption points to §5.3 for the any-mesh 0.63/1.65/0.94×.
- **L-shape field panels** are missing (no saved arrays; see `F4_missing.md`).

## Inconsistencies between responses and evidence, and how the paper handles them

1. **CROM accuracy ratio.** The general and iMp8 responses say "5 times more accurate in the worst case". The 1dYR response says "4–5.5 times", with the head's median higher (0.67 % against 0.43 %). The evidence gives 4.10 % against 16.9 % and 22.4 %, which is 4.1–5.5×. **The paper uses "4–5.5 times"** and states the higher median. The general and iMp8 responses could be aligned.
2. **Bank loss: the code and the response's Q2 description differ in detail.** These details were read from the training code by a search through the repository's history; evidence paths can be supplied.
   - The L-shape bank also trains its Fourier frequencies. The response names only Burgers 2D and Poisson 2D. The paper (§3.4, Table C.3) says Burgers 2D, Poisson 2D *and the L-shape*.
   - For heat 3D, the "mean" term of the variable-projection objective is a POD-mode-weighted error over the leading 1,536 normalised POD modes, not a per-sample mean. The paper writes the response's form (eq. 22) and does not mention the POD weighting. **The authors should check eq. (22) against `train_vp.py`.**
   - The orthogonality penalty is a mean of squares, which the paper writes as `mean[...]`, consistent with the corrected response.
   - The heat 2D refit in the deployed run used exact QR least squares.
3. **"Two narrower Burgers settings did not requalify on every retrain and are marked."** The seed audit (`timing-spread.md` C2) names R′ = 128, which failed on all 3 retrains, and R′ = 384, which failed on 2 of 3. R′ = 384 is not "narrower". The paper marks those two with § and states the counts. **The authors should confirm which settings are meant.**
4. **CROM test disclosure.** The first CROM/CoLoRA test job was cancelled after printing two CROM test rows, which the lane agent read before resubmitting. The selection was unchanged and the rows reproduced. This is disclosed in Appendix F.1 but not in the responses.
5. **Original NS family: speed against a solver that can coarsen its mesh (0.63/1.65/0.94×).** These are the ratios of the coordinate-network head on that family (0.21 % at 96³) against RK4 at 32³–48³. The POD head gives the same ratios (1.66× and 0.94×). Table 3's NS rows remain the submission's POD head against same-mesh CNAB2.
6. **Heat 2D break-even "about 370,000 against the modal solver".** The source (`break-even.md`, 367,641) uses the R′ = 48 arm timed in the spectral job, not Table 3's R′ = 32. The table row does not say which arm.
7. **Seed-1 span speedup.** The corrected range 0.04–0.56× covers the two gate-passing seeds. The gate-failed seed-1 runs reach 0.63–0.64×. This does not matter for the paper's claims.
8. Minor rounding: width-24 rollout 0.82–0.83 s (0.83 used); G1 64³ width-64 time 16.1 ms and 96³ 17.5 ms (16/17 used, per the corrected response).

## Other edits the authors should know about

- **Deleted or moved without blue marking:**
  - the "Where our method sits" paragraph;
  - the NNLS derivation (moved to Appendix A.6);
  - the Poisson residual equation (moved to A.2);
  - the "Ordered bank" architecture paragraph;
  - the "Where the speedup comes from" paragraph, including the N^p exponents and the 41.8 ms output-cost detail;
  - the old §5.2 operator narrative;
  - the old "Which knob" numbers (108→1136×, 27.5×, 44.5×);
  - old Limitations (i)–(v), replaced by new (i)–(vi);
  - the old conclusion.
- **Bibliography:** added Farhat et al. 2014 (ECSW) and Schäfer & Turek 1996 (DFG benchmark). Please check both entries.
- **New appendix sections:** E (harder NS and cylinder setup), F (new baselines and ablations) and G (figures). C.1–C.4 were added inside Appendix C.
- **AI use statement:** left exactly as submitted. It names tools; the authors should check that it still describes the revision work.
- **Statements describing the new experiments come from lane status files and result JSONs, not from re-run code.** Verified against raw JSON:
  - G1, all errors and speedups, three seeds;
  - the coordinate-network table;
  - cylinder G2 times and errors, long-window and out-of-band targets, and operator targets;
  - Burgers operator tables;
  - neural-field results.

  Taken from the responses or status files only:
  - the NS family-selection criteria and the cylinder offline-cost components;
  - the Burgers ECSW study;
  - the 1024² head and POD-16 ablation times;
  - the steep-Burgers regime numbers, which come from the F6 data file.

## Decisions for the authors

1. F4c: regenerate it, keep it with the current caveat, or drop it.
2. Whether to bring one results figure into the main text, which means cutting text elsewhere; there is no slack at 9 pages.
3. Confirm eq. (22) and Table C.3 against the training code (item 2 above), and the § marks (item 3).
4. Align the "5 times" wording for CROM in the general and iMp8 responses with the paper's "4–5.5 times".
5. Whether to disclose the CROM partial test read in the responses as well.
6. Check the AI use statement and the two new bibliography entries.
